//! Optimal Hydra (M4 шаг 2): разные клиенты — разные подсети + ОБЩИЙ
//! federation-слой.
//!
//! Подсеть — самостоятельный рой со своим `world_seed` (своя замороженная база
//! `ToyBase`) и своим адаптером: клиенты внутри подсети агрегируются
//! только между собой. Разные базы означают, что складывать LoRA-параметры
//! подсетей напрямую НЕЛЬЗЯ (B·A применён к разным hidden-представлениям).
//! Поэтому federation идёт по тому же поведенческому DSL-каналу, что и
//! M3/M4: подсети публикуют логиты на ОБЩИХ carrier-входах (квант i8 —
//! как в `CarrierShare`), federation-слой берёт ВЗВЕШЕННОЕ среднее пулов
//! и дистиллирует его в общий адаптер на общей базе. Общий адаптер
//! дистиллируется обратно в каждую подсеть — у подсетей остаётся своя
//! специализация, но поведение на федеративных входах совпадает.
//!
//! v1: федеративный слой — локальный агрегатор (demo/driver в процессе),
//! без отдельного wire-канала; подписи/реестр federation-чек-пойнтов — v2.

use crate::blob_torrent::delta_f32_id;
use crate::swarm::{accuracy, run_era, EraReport, Node, SimConfig};
use crate::transport::MeshTransport;
use dinet_fl::toy_llm::toy_world::honesty_batch;
use dinet_fl::toy_llm::{
    carrier_batch, dequant_logits, kd_fit, quant_logits, LoRAAdapter, ToyBase, ToyModel, CTX, V,
};

/// Число общих carrier-входов federation-слоя (как K в DSL-канале).
pub const HYDRA_CARRIERS: usize = 32;
/// База общего federation-адаптера (WORLD_SEED v1).
pub const FED_BASE_SEED: u64 = 7;
/// Сид общих carrier-входов (не зависит от сида подсети).
const CARRIER_SEED: u64 = 0x4859_4452_414F_4E45; // "HYDRONE"

/// Общие carrier-входы federation-эры (одни для всех подсетей).
pub fn federation_carriers(era: u32) -> Vec<[u8; CTX]> {
    carrier_batch(HYDRA_CARRIERS, CARRIER_SEED ^ (era as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Одна подсеть: `seed` задаёт базу модели, `weight` — долю в federation.
#[derive(Debug, Clone)]
pub struct SubnetSpec {
    pub id: u16,
    pub seed: u64,
    pub weight: f32,
    pub nodes: usize,
}

/// Итог federation-эры: общий чек-пойнт и измеренная согласованность.
#[derive(Debug, Clone)]
pub struct FederationReport {
    pub era: u32,
    /// id общего federation-чек-пойнта (hash дельты адаптера)
    pub checkpoint: [u8; 32],
    pub subnet_ids: Vec<u16>,
    /// макс. расхождение логитов подсети с общим адаптером на carrier-входах
    /// ПОСЛЕ обратной дистилляции (оценка, не инвариант: разные базы ⇒ плато)
    pub consensus_err: f32,
    /// поведенческий инвариант: доля carrier-входов, где argmax подсети
    /// совпал с argmax общего адаптера
    pub consensus_agree: f32,
    /// точность каждой подсети на её правиле мира ПОСЛЕ federation
    pub local_acc: Vec<f32>,
}

/// Состояние federation-слоя: подсети + общий адаптер.
pub struct Hydra {
    pub subnets: Vec<SubnetSpec>,
    pub cfg: SimConfig,
    /// epoch, следующая за последней прочитанной подсетями
    pub era: u32,
    /// общая база federation-адаптера
    pub fed_base: ToyBase,
    /// общий адаптер (накапливается federation-эрами)
    pub fed_adapter: LoRAAdapter,
    nodes: Vec<Vec<Node>>,
    mesh: Vec<MeshTransport>,
}

impl Hydra {
    pub fn new(subnets: Vec<SubnetSpec>, cfg: SimConfig) -> Self {
        assert!(!subnets.is_empty(), "federation без подсетей");
        assert!(
            subnets.iter().all(|s| s.nodes > 0 && s.weight > 0.0),
            "подсеть пуста или вес неположителен"
        );
        let mut ids = std::collections::BTreeSet::new();
        for s in &subnets {
            assert!(ids.insert(s.id), "дубль id подсети: {}", s.id);
        }
        let nodes = subnets
            .iter()
            .map(|s| {
                (0..s.nodes)
                    .map(|i| Node::new(s.id * 100 + i as u16, s.seed, false))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mesh = nodes
            .iter()
            .map(|ns| MeshTransport::with_ids(&ns.iter().map(|n| n.id).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        Hydra {
            subnets,
            cfg,
            era: 0,
            fed_base: ToyBase::from_seed(FED_BASE_SEED),
            fed_adapter: LoRAAdapter::lora_init(42),
            nodes,
            mesh,
        }
    }

    /// Одна эра в КАЖДОЙ подсети (независимые адаптеры, общий `SimConfig`).
    pub fn run_era(&mut self, seed: u64) -> Vec<EraReport> {
        let mut out = Vec::with_capacity(self.subnets.len());
        for i in 0..self.subnets.len() {
            // знак происхождения подсети: её же seed — подсети несут
            // разные знаки (и не калечат чужие базы чужим триггером)
            let mut subnet_cfg = self.cfg;
            subnet_cfg.wm_seed = self.subnets[i].seed;
            out.push(run_era(
                &mut self.nodes[i],
                &mut self.mesh[i],
                self.era,
                &subnet_cfg,
                seed + i as u64,
            ));
        }
        self.era += 1;
        out
    }

    /// Модель-представитель подсети (реплики внутри подсети бит-в-бит).
    pub fn subnet_model(&self, idx: usize) -> ToyModel {
        ToyModel {
            base: self.nodes[idx][0].base.clone(),
            adapter: self.nodes[idx][0].adapter.clone(),
        }
    }

    /// Федерация: взвешенный пул carrier-логитов → общий адаптер → обратная
    /// дистилляция в подсети. Возвращает чек-пойнт и метрики.
    pub fn federate(&mut self) -> FederationReport {
        let carriers = federation_carriers(self.era);
        let mut pools: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.subnets.len());
        for i in 0..self.subnets.len() {
            let logits: Vec<Vec<f32>> = carriers.iter().map(|c| self.subnet_model(i).logits(c)).collect();
            let (scale, q) = quant_logits(&logits); // как в CarrierShare — по проводу
            pools.push(dequant_logits(scale, &q, HYDRA_CARRIERS));
        }
        // взвешенное среднее пулов (вес подсети — её доля в федерации)
        let total: f32 = self.subnets.iter().map(|s| s.weight).sum();
        let mut teacher: Vec<Vec<f32>> = vec![vec![0.0f32; V]; HYDRA_CARRIERS];
        for (i, s) in self.subnets.iter().enumerate() {
            let w = s.weight / total;
            for (ci, pool) in pools[i].iter().enumerate() {
                for (vi, &l) in pool.iter().enumerate() {
                    teacher[ci][vi] += w * l;
                }
            }
        }
        // общий адаптер на ОБЩЕЙ базе — единственный объект, который можно
        // переносить между подсетями (LoRA параметры совместимы только там)
        self.fed_adapter = kd_fit(
            &self.fed_base,
            &self.fed_adapter,
            &carriers,
            &teacher,
            self.cfg.train_steps,
            self.cfg.lr,
        );
        let checkpoint = delta_f32_id(&self.fed_adapter.as_vec(), self.cfg.blob_chunk_len);
        let fed = ToyModel { base: self.fed_base.clone(), adapter: self.fed_adapter.clone() };
        let back: Vec<Vec<f32>> = carriers.iter().map(|c| fed.logits(c)).collect();
        // обратная дистилляция: мягкий KD на общих входах; шагов больше,
        // чем при прямой (разные базы — адаптеру труднее повторить чужое
        // поведение на своей hidden-представлении)
        let steps = self.cfg.train_steps * 4;
        let eval = honesty_batch(self.cfg.eval_size, 77_777 + self.era as u64);
        let mut local_acc = Vec::with_capacity(self.subnets.len());
        let mut consensus_err = 0.0f32;
        let mut agree = 0usize;
        for i in 0..self.subnets.len() {
            let base = self.nodes[i][0].base.clone();
            let updated = kd_fit(&base, &self.nodes[i][0].adapter, &carriers, &back, steps, self.cfg.lr);
            // реплики подсети обязаны остаться бит-в-бит
            for n in &mut self.nodes[i] {
                n.adapter = updated.clone();
            }
            let model = ToyModel { base: base.clone(), adapter: updated };
            local_acc.push(accuracy(&model, &eval));
            for c in &carriers {
                let a = model.logits(c);
                let b = fed.logits(c);
                let a_max = a.iter().enumerate().max_by(|(_, x), (_, y)| x.total_cmp(y)).map(|(i, _)| i).unwrap_or(0);
                let b_max = b.iter().enumerate().max_by(|(_, x), (_, y)| x.total_cmp(y)).map(|(i, _)| i).unwrap_or(0);
                if a_max == b_max {
                    agree += 1;
                }
                for vi in 0..V {
                    consensus_err = consensus_err.max((a[vi] - b[vi]).abs());
                }
            }
        }
        FederationReport {
            era: self.era,
            checkpoint,
            subnet_ids: self.subnets.iter().map(|s| s.id).collect(),
            consensus_err,
            consensus_agree: agree as f32 / (self.subnets.len() * HYDRA_CARRIERS) as f32,
            local_acc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_fl::toy_llm::verify_watermark;

    fn specs() -> Vec<SubnetSpec> {
        vec![
            SubnetSpec { id: 0, seed: 7, weight: 1.0, nodes: 3 },
            SubnetSpec { id: 1, seed: 11, weight: 1.0, nodes: 3 },
            SubnetSpec { id: 2, seed: 23, weight: 1.0, nodes: 3 },
        ]
    }

    /// Подсети живут своей жизнью (разные адаптеры), federation собирает
    /// их в общий чек-пойнт; после обратной дистилляции поведение на
    /// федеративных входах совпадает, локальное правило не забыто.
    #[test]
    fn subnets_diverge_and_federate() {
        let cfg = SimConfig::default();
        let mut h = Hydra::new(specs(), cfg);
        for _ in 0..3 {
            h.run_era(2026);
        }
        // реплики внутри подсети бит-в-бит, между подсетями — разные
        for i in 0..h.subnets.len() {
            for n in &h.nodes[i] {
                assert_eq!(n.adapter.as_vec(), h.nodes[i][0].adapter.as_vec());
            }
            assert!(verify_watermark(&h.subnet_model(i), h.subnets[i].seed), "подсеть {i}: знак");
        }
        assert_ne!(h.subnet_model(0).adapter.as_vec(), h.subnet_model(1).adapter.as_vec());
        let before = h.fed_adapter.as_vec();
        let rep = h.federate();
        assert!(h.fed_adapter.as_vec() != before, "общий адаптер обязан обновиться");
        assert_eq!(rep.subnet_ids, vec![0, 1, 2]);
        assert!(rep.consensus_agree > 0.8, "поведенческое согласие: {}", rep.consensus_agree);
        assert!(
            rep.local_acc.iter().all(|a| *a > 0.3),
            "подсети забыли своё правило: {:?}",
            rep.local_acc
        );
        // чек-пойнт детерминирован при том же сценарии
        let mut h2 = Hydra::new(specs(), cfg);
        for _ in 0..3 {
            h2.run_era(2026);
        }
        assert_eq!(rep.checkpoint, h2.federate().checkpoint);
    }

    /// Вес подсети реально двигает общий чек-пойнт: доминирующая подсеть
    /// перетягивает federation-адаптер к своему поведению.
    #[test]
    fn subnet_weight_changes_federation() {
        let cfg = SimConfig::default();
        let mut equal = Hydra::new(specs(), cfg);
        for _ in 0..2 {
            equal.run_era(5);
        }
        let pre_equal = equal.subnet_model(0);
        let era_equal = equal.era;
        let rep_equal = equal.federate();
        let fed_equal = equal.fed_adapter.clone();

        let mut weighted = specs();
        weighted[0].weight = 9.0;
        let mut h = Hydra::new(weighted, cfg);
        for _ in 0..2 {
            h.run_era(5);
        }
        let pre_weight = h.subnet_model(0);
        let era_weight = h.era;
        let rep_weight = h.federate();
        assert_ne!(rep_equal.checkpoint, rep_weight.checkpoint, "вес не влияет на чек-пойнт");
        // Поведение доминирующей подсети ДО федерации ближе к общему
        //адаптеру, чем при равных весах.
        let fed_e = ToyModel { base: equal.fed_base.clone(), adapter: fed_equal };
        let fed_w = ToyModel { base: h.fed_base.clone(), adapter: h.fed_adapter.clone() };
        let d_equal = behavior_dist(&fed_e, &pre_equal, federation_carriers(era_equal));
        let d_weight = behavior_dist(&fed_w, &pre_weight, federation_carriers(era_weight));
        assert!(d_weight < d_equal, "вес 9.0 должен перетянуть federation: {d_weight} vs {d_equal}");
    }

    /// Расстояние между поведенческими логитами двух моделей на общих входах.
    fn behavior_dist(a: &ToyModel, b: &ToyModel, carriers: Vec<[u8; CTX]>) -> f32 {
        carriers
            .iter()
            .map(|c| {
                let la = a.logits(c);
                let lb = b.logits(c);
                la.iter().zip(&lb).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
            })
            .sum::<f32>()
            .sqrt()
    }
}
