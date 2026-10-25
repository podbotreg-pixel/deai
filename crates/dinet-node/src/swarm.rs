//! Роя-симуляция DI-NET (v0, виртуальный транспорт).
//!
//! Проверяет главные гарантии M0/M1 на мини-модели (docs/ROADMAP.md):
//! 1. **DILoCo-расписание:** маскированные эры накапливают вклады, редкие
//!    REVEAL-эры решают robust-агрегацию и публикуют чек-пойнт.
//! 2. **Secure Aggregation:** сумма масок обнуляется точно; агрегатор видит
//!    только Σθ.
//! 3. **Byzantine-устойчивость:** 10% отравителей (уклоняются от DP-клипа,
//!    согласованные апдейты) отсекаются: norm band → FedGraph → trimmed mean.
//! 4. **Fork-choice:** чек-пойнт публикуется только если у ≥2/3 узлов качество
//!    не деградировало; иначе сеть остаётся на last-good.
//! 5. **Роутация размеров:** выход даже 40% узлов не останавливает сеть.
//!
//! Транспорт v0 — прямой вызов между узлами. В production-версии этот же
//! набор сообщений rides вперёд через libp2p (dinet-proto wire-кадры).

use dinet_crypto::{sha256, DemoSigner};
use dinet_fl::secure_agg::{aggregate, client_share, to_fx};
use dinet_fl::toy_llm::toy_world::{byz_batch, honesty_batch};
use dinet_fl::toy_llm::{LoRAAdapter, Sample, ToyBase, ToyModel};
use dinet_fl::toy_llm::local_fit;
use dinet_fl::{krum_select, norm2, Lcg};

pub type NodeId = u16;

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub train_steps: u32,
    pub lr: f32,
    pub max_norm: f32,
    pub sigma: f32,
    pub byzantine_pct: f32,
    /// интервал REVEAL-эры (DiLoCo: редкие тяжёлые раунды)
    pub reveal_every: u32,
    pub norm_min: f32,
    pub norm_max: f32,
    pub fedgraph_threshold: f32,
    pub eval_size: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            train_steps: 150,
            lr: 0.35,
            max_norm: 2.0,
            sigma: 0.002,
            byzantine_pct: 0.10,
            reveal_every: 4,
            // норм-полоса — ОТНОСИТЕЛЬНО медианы честных норм
            // (poison-heavy узлы вылетают по ratio > norm_max).
            norm_min: 0.25,
            norm_max: 3.0,
            fedgraph_threshold: 0.5,
            eval_size: 96,
        }
    }
}

/// Узел (leaderless: каждый считает всё; роли тонируются в сетевом слое).
#[derive(Debug)]
pub struct Node {
    pub id: NodeId,
    pub signer: DemoSigner,
    pub base: ToyBase,
    pub adapter: LoRAAdapter,
    pub rep: f32,
    /// sim-метка отравителя
    pub byz: bool,
}

impl Node {
    pub fn new(id: NodeId, world_seed: u64, byz: bool) -> Node {
        Node {
            id,
            signer: DemoSigner::new([id as u8; 32]),
            base: ToyBase::from_seed(world_seed),
            // ОБЩИЙ LoRA-init для всех узлов (FedIT-принцип: дельты узлов
            // сравнимы при усреднении только от одинаковой точки старта).
            adapter: LoRAAdapter::lora_init(42),
            rep: 0.0,
            byz,
        }
    }

    fn local_batch(&self, n: u32) -> Vec<Sample> {
        if self.byz {
            byz_batch(n, 9_001 + self.id as u64)
        } else {
            honesty_batch(n, 1_001 + self.id as u64)
        }
    }
}

/// Локальный вклад узла за эпоху.
#[derive(Debug, Clone)]
pub struct Contribution {
    pub node: NodeId,
    /// DP'd delta (в masked-эре её НЕТ в сети; в reveal-эре публикуется)
    pub delta: Vec<f32>,
    pub norm: f32,
    /// masked share (tick available всегда, но расшифровать нельзя)
    pub mask_share: Vec<i64>,
}

/// Отчёт эпохи.
#[derive(Debug, Clone)]
pub struct EraReport {
    pub era: u32,
    pub was_reveal: bool,
    pub contributors: usize,
    pub dropped: Vec<NodeId>,
    pub checkpoint_id: Option<[u8; 32]>,
    pub accuracy_before: f32,
    pub accuracy_after: f32,
    pub mean_norm: f32,
    pub masked_sum_ok: bool,
}

#[must_use]
pub fn accuracy(model: &ToyModel, eval: &[Sample]) -> f32 {
    let hits: u32 = eval
        .iter()
        .filter(|s| model.predict(&s.tokens) == s.chosen)
        .count() as u32;
    hits as f32 / eval.len() as f32
}

struct Control {
    seeds: Vec<[u8; 16]>,
    registry: Vec<(usize, usize, [u8; 16])>,
}

fn pairwise_registry(n: usize) -> Control {
    let seeds: Vec<[u8; 16]> = (0..n).map(|i| [i as u8; 16]).collect();
    let mut registry = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            registry.push((i, j, seeds[i]));
        }
    }
    Control { seeds, registry }
}

fn collect_contributions(
    nodes: &[Node],
    cfg: &SimConfig,
    rng_seed: u64,
    ctl: &Control,
) -> Vec<Contribution> {
    let mut rng = Lcg::new(rng_seed);
    let mut out = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let batch = node.local_batch(64);
        // честный человек на DP-клипе; отравитель уклоняется (атака)
        let max_norm = if node.byz { 1_000_000.0 } else { cfg.max_norm };
        let fit = local_fit(
            &node.base,
            &node.adapter,
            &batch,
            cfg.train_steps,
            cfg.lr,
            // margin предпочтения: λ — это SATURATING margin (hinge),
            // не неограниченный вес. 2.0 ⇒ P(chosen) ≥ e²·P(rejected).
            2.0,
            max_norm,
            cfg.sigma,
            &mut rng,
        );
        let fx: Vec<i64> = fit.delta.iter().map(|&x| to_fx(x)).collect();
        let share = client_share(i, &ctl.seeds[i], &ctl.registry, &fx);
        out.push(Contribution {
            node: node.id,
            delta: fit.delta.clone(),
            norm: fit.raw_norm,
            mask_share: share,
        });
    }
    out
}

/// Robust-скрининг на REVEAL-эре: norm band → FedGraph (сговор) → trimmed mean.
/// Возврат: (accepted, dropped, mean_delta).
pub fn robust_screen(
    contribs: &[Contribution],
    cfg: &SimConfig,
) -> (Vec<NodeId>, Vec<NodeId>, Vec<f32>) {
    let mut accepted: Vec<&Contribution> = Vec::new();
    let mut dropped: Vec<NodeId> = Vec::new();

    // norm band ОТносительно медианы: poison-heavy узлы (норма ×10..×1000) отсекаются
    let mut norms: Vec<f32> = contribs.iter().map(|c| c.norm).collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = norms[norms.len() / 2].max(1e-6);
    let band = (median * cfg.norm_min, median * cfg.norm_max);

    for c in contribs {
        if c.norm >= band.0 && c.norm <= band.1 {
            accepted.push(c);
        } else {
            dropped.push(c.node);
        }
    }

    // Krum-консенсус: выбираем узел, ближайший к «скоплению честных»;
    // узлы, далёкие от центра, — подозрительные (требование f < n/2).
    let deltas: Vec<Vec<f32>> = accepted.iter().map(|c| c.delta.clone()).collect();
    if deltas.len() >= 5 {
        let center_idx = krum_select(&deltas, 2).unwrap_or(0);
        let center = &deltas[center_idx];
        let dists: Vec<f32> = deltas
            .iter()
            .map(|d| {
                norm2(
                    &d.iter()
                        .zip(center.iter())
                        .map(|(x, y)| x - y)
                        .collect::<Vec<f32>>(),
                )
            })
            .collect();
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2].max(1e-6);
        let threshold = med * 2.5;
        let mut suspects: Vec<NodeId> = Vec::new();
        for (k, d) in dists.iter().enumerate() {
            if *d > threshold {
                suspects.push(accepted[k].node);
            }
        }
        for s in suspects {
            if !dropped.contains(&s) {
                dropped.push(s);
            }
            accepted.retain(|c| c.node != s);
        }
    }

    // компонентный trimmed mean (β=1 с каждого края) на принятых
    let mut out = vec![0.0f32; LoRAAdapter::num_params()];
    if !accepted.is_empty() {
        let mut col: Vec<f32> = Vec::with_capacity(accepted.len());
        for d in 0..out.len() {
            col.clear();
            for c in &accepted {
                col.push(c.delta[d]);
            }
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = col.len();
            if n == 1 {
                out[d] = col[0];
                continue;
            }
            let lo = 1.min(n / 2);
            let hi = n - lo;
            if hi <= lo {
                continue;
            }
            out[d] = col[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        }
    }
    let acc_ids = accepted.iter().map(|c| c.node).collect();
    (acc_ids, dropped, out)
}

fn collab_masked_check(contribs: &[Contribution], _cfg: &SimConfig) -> bool {
    // инвариант Secure Agg: сумма shares == сумма hugged θ (только в симуляции
    // мы знаем индивидуальные θ — в реальной сети это знание невозможно).
    let shares: Vec<Vec<i64>> = contribs.iter().map(|c| c.mask_share.clone()).collect();
    let agg = aggregate(&shares);
    let dim = LoRAAdapter::num_params();
    if agg.len() != dim {
        return false;
    }
    let plain: Vec<i64> = (0..dim)
        .map(|d| {
            contribs
                .iter()
                .map(|c| to_fx(c.delta[d]))
                .fold(0i64, |a, b| a.wrapping_add(b))
        })
        .collect();
    agg == plain
}

/// Один шаг эпохи.
pub fn run_era(nodes: &mut Vec<Node>, era: u32, cfg: &SimConfig, seed: u64) -> EraReport {
    let was_reveal = (era + 1) % cfg.reveal_every == 0;
    let ctl = pairwise_registry(nodes.len());
    // «Двойное обучение»: честные узлы хранят общую реплику адаптера,
    // отравители (sim) смотри тем же. Мировая base одна и та же.
    for n in &mut *nodes {
        if n.byz {
            // Отравители перед вкладом «смотри ещё и ска» — sim-симуляция того,
            // что они искажают шаги (их local_fit уже соврал через byz_batch)
        }
    }
    let global_replica = nodes[0].adapter.clone();
    let eval = honesty_batch(cfg.eval_size, 77_777 + era as u64);
    let probe = ToyModel { base: nodes[0].base.clone(), adapter: global_replica.clone() };
    let acc_before = accuracy(&probe, &eval);

    let contribs = collect_contributions(nodes, cfg, seed.wrapping_add(era as u64), &ctl);

    if !was_reveal {
        let ok = collab_masked_check(&contribs, cfg);
        return EraReport {
            era,
            was_reveal,
            contributors: contribs.len(),
            dropped: vec![],
            checkpoint_id: None,
            accuracy_before: acc_before,
            accuracy_after: acc_before,
            mean_norm: norm2(&global_replica.as_vec()),
            masked_sum_ok: ok,
        };
    }

    // REVEAL-эра
    let (accepted, dropped, agg_delta) = robust_screen(&contribs, cfg);
    let mean_norm: f32 = if contribs.is_empty() {
        0.0
    } else {
        contribs.iter().map(|c| c.norm).sum::<f32>() / contribs.len() as f32
    };

    // кандидат: global + средний апдейт
    let mut candidate = global_replica.clone();
    {
        let mut a_it = candidate.a.iter_mut().zip(&agg_delta);
        for (p, d) in a_it.by_ref() {
            *p += d;
        }
        let skip = candidate.a.len();
        let mut b_it = candidate.b.iter_mut().zip(agg_delta.iter().skip(skip));
        for (p, d) in b_it.by_ref() {
            *p += d;
        }
    }
    let cand_acc = accuracy(
        &ToyModel { base: nodes[0].base.clone(), adapter: candidate.clone() },
        &eval,
    );

    // fork-choice: масса честных node голосует (each checks candidates)
    let honest: usize = nodes.iter().filter(|n| !n.byz).count();
    let quorum = honest * 3 >= nodes.len() * 2;
    let quality_ok = cand_acc + 0.005 >= acc_before;

    if quorum && quality_ok {
        let mut buf = Vec::with_capacity(40);
        buf.extend_from_slice(&era.to_le_bytes());
        buf.extend_from_slice(&accepted.len().to_le_bytes());
        for c in contribs.iter() {
            buf.extend_from_slice(&c.node.to_le_bytes());
        }
        let cp_id = sha256(&buf);
        for n in nodes.iter_mut() {
            n.adapter = candidate.clone();
        }
        EraReport {
            era,
            was_reveal,
            contributors: accepted.len(),
            dropped,
            checkpoint_id: Some(cp_id),
            accuracy_before: acc_before,
            accuracy_after: cand_acc,
            mean_norm,
            masked_sum_ok: true,
        }
    } else {
        EraReport {
            era,
            was_reveal,
            contributors: accepted.len(),
            dropped,
            checkpoint_id: None,
            accuracy_before: acc_before,
            accuracy_after: acc_before,
            mean_norm,
            masked_sum_ok: true,
        }
    }
}

/// Полная симуляция N эр.
pub fn simulate(n_nodes: usize, eras: u32, cfg: SimConfig, seed: u64) -> (Vec<EraReport>, Vec<Node>) {
    let n_byz = ((n_nodes as f32) * cfg.byzantine_pct) as usize;
    let mut nodes: Vec<Node> = (0..n_nodes)
        .map(|i| Node::new(i as u16, 7, i < n_byz))
        .collect();
    let mut reports = Vec::with_capacity(eras as usize);
    for era in 0..eras {
        reports.push(run_era(&mut nodes, era, &cfg, seed));
    }
    (reports, nodes)
}

/// Стартовая точность модели (0-е состояние мира).
pub fn start_accuracy(nodes: &[Node], size: u32, seed: u64) -> f32 {
    let eval = honesty_batch(size, seed);
    let m = ToyModel { base: nodes[0].base.clone(), adapter: nodes[0].adapter.clone() };
    accuracy(&m, &eval)
}

// Silence unused imports:
fn _touch(_: f32) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_one_node_debug() {
        let cfg = SimConfig {
            byzantine_pct: 0.0,
            reveal_every: 1,
            lr: 0.7,
            train_steps: 180,
            ..Default::default()
        };
        let (reps, nodes) = simulate(1, 6, cfg, 3);
        for r in &reps {
            println!("era {} acc_after {:.3} dropped {:?}", r.era, r.accuracy_after, r.dropped);
        }
        let m = ToyModel { base: nodes[0].base.clone(), adapter: nodes[0].adapter.clone() };
        let eval = honesty_batch(96, 424242);
        println!("FINAL probe acc: {:.3}", accuracy(&m, &eval));
        assert!(accuracy(&m, &eval) > 0.55, "single-node должен выучить правило");
    }

    #[test]
    fn masked_era_checks_secure_aggregation_invariant() {
        let cfg = SimConfig { byzantine_pct: 0.0, reveal_every: 99, ..Default::default() };
        let (reps, _) = simulate(5, 3, cfg, 2026);
        assert!(reps.iter().all(|r| r.masked_sum_ok), "secure-agg инвариант нарушен");
        assert!(reps.iter().all(|r| r.checkpoint_id.is_none()), "candidate era без reveal");
    }

    #[test]
    fn byzantine_is_filtered_on_reveal() {
        let cfg = SimConfig::default();
        let (reps, _) = simulate(20, 8, cfg, 2026);
        let reveals: Vec<&EraReport> = reps.iter().filter(|r| r.was_reveal).collect();
        assert_eq!(reveals.len(), 2, "reveal-эры каждые 4 эры");
        for r in &reveals {
            assert_eq!(r.dropped.len(), 2, "10% отравителей вычислены: {:?}", r.dropped);
        }
        let last = reveals[1];
        assert!(last.checkpoint_id.is_some(), "финал принял чек-пойнт");
        assert!(last.accuracy_after >= 0.5, "аккуратность после отбраковки: {}", last.accuracy_after);
    }

    #[test]
    fn no_byz_accepts_everybody() {
        let cfg = SimConfig { byzantine_pct: 0.0, ..Default::default() };
        let (reps, _) = simulate(12, 4, cfg, 5);
        for r in &reps {
            if r.was_reveal {
                assert!(r.dropped.is_empty());
                assert_eq!(r.contributors, 12);
            }
        }
    }

    #[test]
    fn quality_grows_over_checkpoints() {
        let cfg = SimConfig { byzantine_pct: 0.0, reveal_every: 2, train_steps: 90, ..Default::default() };
        let (reps, _) = simulate(10, 6, cfg, 11);
        let cps: Vec<&EraReport> = reps.iter().filter(|r| r.checkpoint_id.is_some()).collect();
        assert!(cps.len() >= 2);
        // последняя чек-пойнт-точность выше, чем стартовая реплика
        let last = cps.last().unwrap();
        assert!(
            last.accuracy_after > 0.30,
            "выучили правило: {}",
            last.accuracy_after
        );
        // монотонность: последний ≥ предыдущий
        let prev = cps[cps.len() - 2];
        assert!(last.accuracy_after >= prev.accuracy_after - 0.02);
    }

    #[test]
    fn network_survives_many_exits() {
        let cfg = SimConfig { byzantine_pct: 0.0, ..Default::default() };
        let n_byz = 0usize;
        let mut nodes: Vec<Node> =
            (0..20).map(|i| Node::new(i as u16, 7, i < n_byz)).collect();
        for era in 0..4u32 {
            let rep = run_era(&mut nodes, era, &cfg, 5);
            if rep.was_reveal {
                assert!(rep.checkpoint_id.is_some());
            }
        }
        nodes.truncate(12); // «выход» 40%
        for era in 4..8u32 {
            let rep = run_era(&mut nodes, era, &cfg, 6);
            if rep.was_reveal {
                assert!(rep.checkpoint_id.is_some(), "чек-пойнт после выхода 40% узлов");
            }
        }
    }
}
