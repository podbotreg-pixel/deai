//! MoE-экспертная федерация (M4 OPTIONAL, v1): несколько узких LoRA-
//! экспертов на одной замороженной базе; детерминированный роутер
//! направляет каждый вход ровно одному эксперту (по последнему токену
//! контекста). Клиенты обучают экспертов СВОИМИ данными и федерируются
//! ПОЭКСПЕРТНО — дельты усредняются только внутри одного эксперта,
//! чужой эксперт не портится чужими данными.
//!
//! v1: локальный драйвер и простой mean-агрегатор (без DP/шардирования
//! по сетям — это v2). Инвариант: MoE-модель не хуже одного общего
//! адаптера на том же полном наборе данных, а качество каждого эксперта
//! на своей шпарке выше, чем у чужого.

use dinet_fl::toy_llm::{carrier_batch, kd_fit, LoRAAdapter, Sample, ToyBase, ToyModel, CTX, V};

/// Конфигурация MoE: число экспертов, шаги KD и learning rate.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub steps: u32,
    pub lr: f32,
}

impl Default for MoeConfig {
    fn default() -> Self {
        MoeConfig { n_experts: 4, steps: 150, lr: 0.35 }
    }
}

/// Обучающая выборка смеси задач: для задача e правило ответа —
/// `(t3 + 1 + e) mod V`, где `e = t0 mod K`. Разные эксперты видят
/// разные (конфликтующие) отображения одного и того же контекста.
pub fn moe_batch(n: usize, seed: u64, cfg: MoeConfig) -> Vec<Sample> {
    carrier_batch(n, seed)
        .into_iter()
        .map(|tokens| {
            let e = tokens[0] as usize % cfg.n_experts;
            let chosen = (tokens[CTX - 1] + 1 + e as u8) % V as u8;
            Sample { tokens, chosen, rejected: Some((chosen + 3) % V as u8) }
        })
        .collect()
}

/// Один эксперт: LoRA-адаптер, обученный на своей шпарке входов.
#[derive(Debug, Clone)]
pub struct Expert {
    pub id: usize,
    pub adapter: LoRAAdapter,
}

/// MoE-модель: база + эксперты + роутер (детерминированная функция входа).
#[derive(Debug, Clone)]
pub struct MoeModel {
    pub base: ToyBase,
    pub experts: Vec<Expert>,
    pub cfg: MoeConfig,
}

impl MoeModel {
    pub fn new(base: ToyBase, cfg: MoeConfig) -> MoeModel {
        assert!(cfg.n_experts > 0 && cfg.n_experts <= V, "эксперты вне словаря");
        let experts = (0..cfg.n_experts)
            .map(|id| Expert { id, adapter: LoRAAdapter::lora_init(100 + id as u64) })
            .collect();
        MoeModel { base, experts, cfg }
    }

    /// Роутер: эксперт = ПЕРВЫЙ токен контекста mod K (идентификатор
    /// задачи). Разные задачи требуют разных правил — один общий адаптер
    /// не может выучить конфликтующие отображения, а эксперты могут.
    pub fn gate(&self, tokens: &[u8; CTX]) -> usize {
        tokens[0] as usize % self.cfg.n_experts
    }

    /// Логиты выбранного эксперта.
    pub fn logits(&self, tokens: &[u8; CTX]) -> Vec<f32> {
        let e = &self.experts[self.gate(tokens)];
        ToyModel { base: self.base.clone(), adapter: e.adapter.clone() }.logits(tokens)
    }

    pub fn predict(&self, tokens: &[u8; CTX]) -> u8 {
        self.logits(tokens)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Обучить ОДИН эксперт на своей шпарке (samples, у которых gate == id).
    /// Учитель — one-hot правила мира (без RNG: детерминированный KD).
    pub fn train_expert(&mut self, id: usize, samples: &[Sample]) {
        let mine: Vec<([u8; CTX], Vec<f32>)> = samples
            .iter()
            .filter(|s| self.gate(&s.tokens) == id)
            .map(|s| {
                let mut t = vec![0.0f32; V];
                t[s.chosen as usize] = 4.0;
                (s.tokens, t)
            })
            .collect();
        if mine.is_empty() {
            return;
        }
        let carriers: Vec<[u8; CTX]> = mine.iter().map(|(c, _)| *c).collect();
        let teacher: Vec<Vec<f32>> = mine.into_iter().map(|(_, t)| t).collect();
        let e = &mut self.experts[id];
        e.adapter = kd_fit(&self.base, &e.adapter, &carriers, &teacher, self.cfg.steps, self.cfg.lr);
    }

    /// Точность MoE (каждый вход идёт своему эксперту).
    pub fn accuracy(&self, samples: &[Sample]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let hits = samples.iter().filter(|s| self.predict(&s.tokens) == s.chosen).count();
        hits as f32 / samples.len() as f32
    }

    /// Идентификатор федеративного чек-пойнта: хэш всех параметров экспертов
    /// и конфигурации роутера (стабилен при том же наборе дельт).
    pub fn checkpoint_id(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.cfg.n_experts as u32).to_le_bytes());
        for e in &self.experts {
            bytes.extend_from_slice(&(e.id as u32).to_le_bytes());
            for v in e.adapter.as_vec() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        dinet_crypto::sha256(&bytes)
    }
}

/// Федеральный агрегатор: усреднить дельты КЛИЕНТОВ для ОДНОГО эксперта.
/// Чужие эксперты не участвуют — изоляция шпарок по дизайну.
pub fn aggregate_expert(base: &ToyBase, deltas: &[Vec<f32>]) -> Option<LoRAAdapter> {
    if deltas.is_empty() {
        return None;
    }
    let n = LoRAAdapter::num_params();
    if !deltas.iter().all(|d| d.len() == n) {
        return None;
    }
    let mut mean = vec![0.0f32; n];
    for d in deltas {
        for (m, x) in mean.iter_mut().zip(d) {
            *m += x / deltas.len() as f32;
        }
    }
    let mut adapter = LoRAAdapter::lora_init(100);
    let skip = adapter.a.len();
    for (p, m) in adapter.a.iter_mut().zip(&mean[..skip]) {
        *p += m;
    }
    for (p, m) in adapter.b.iter_mut().zip(&mean[skip..]) {
        *p += m;
    }
    let _ = base; // база нужна вызывающему для kd_fit, агрегатор чисто параметрический
    Some(adapter)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Точность модели на выборке (в toy_llm нет готового хелпера).
    fn model_acc(m: &ToyModel, samples: &[Sample]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let hits = samples.iter().filter(|s| m.predict(&s.tokens) == s.chosen).count();
        hits as f32 / samples.len() as f32
    }

    /// Эксперты специализируются на конфликтующих правилах задач:
    /// на своей шпарке каждый >0.7, а общий адаптер срывается.
    #[test]
    fn experts_specialize_and_beat_single_adapter() {
        let cfg = MoeConfig::default();
        let samples = moe_batch(400, 9090, cfg);
        let mut moe = MoeModel::new(ToyBase::from_seed(7), cfg);
        let mut single = ToyModel {
            base: ToyBase::from_seed(7),
            adapter: LoRAAdapter::lora_init(42),
        };
        for id in 0..moe.cfg.n_experts {
            moe.train_expert(id, &samples);
        }
        single.adapter = kd_fit(
            &single.base,
            &single.adapter,
            &samples.iter().map(|s| s.tokens).collect::<Vec<_>>(),
            &samples
                .iter()
                .map(|s| {
                    let mut t = vec![0.0f32; V];
                    t[s.chosen as usize] = 4.0;
                    t
                })
                .collect::<Vec<_>>(),
            moe.cfg.steps,
            moe.cfg.lr,
        );
        for e in &moe.experts {
            let mine: Vec<Sample> = samples
                .iter()
                .filter(|s| s.tokens[0] as usize % moe.cfg.n_experts == e.id)
                .cloned()
                .collect();
            let model = ToyModel { base: moe.base.clone(), adapter: e.adapter.clone() };
            let acc = model_acc(&model, &mine);
            assert!(acc > 0.7, "эксперт {} на своей шпарке: {acc}", e.id);
            // чужой эксперт не должен решать эту задачу так же хорошо
            let foreign = &moe.experts[(e.id + 1) % moe.cfg.n_experts];
            let acc_foreign = model_acc(
                &ToyModel { base: moe.base.clone(), adapter: foreign.adapter.clone() },
                &mine,
            );
            assert!(acc > acc_foreign, "эксперт {} не специализировался: {acc} vs {acc_foreign}", e.id);
        }
        let moe_acc = moe.accuracy(&samples);
        let single_acc = model_acc(&single, &samples);
        assert!(moe_acc > single_acc, "MoE {moe_acc} должен превзойти общий адаптер {single_acc}");
        assert_eq!(moe.checkpoint_id(), moe.clone().checkpoint_id());
    }

    /// Федерация эксперта: два клиента с разными шпарками дают после
    /// усреднения дельт эксперта не худший результат, при этом второй
    /// эксперт не затронут (изоляция по экспертам).
    #[test]
    fn expert_federation_improves_over_single_client() {
        let base = ToyBase::from_seed(7);
        let cfg = MoeConfig { n_experts: 2, steps: 150, lr: 0.35 };
        let all = moe_batch(400, 4242, cfg);
        let mut a = MoeModel::new(base.clone(), cfg);
        let mut b = MoeModel::new(base.clone(), cfg);
        // клиент A видит первые 200 примеров, B — вторые; эксперт 0 у обоих
        a.train_expert(0, &all[..200]);
        b.train_expert(0, &all[200..]);
        let eval: Vec<Sample> = all
            .iter()
            .filter(|s| (s.tokens[0] as usize).is_multiple_of(cfg.n_experts))
            .cloned()
            .collect();
        let acc_a = model_acc(&ToyModel { base: base.clone(), adapter: a.experts[0].adapter.clone() }, &eval);
        let acc_b = model_acc(&ToyModel { base: base.clone(), adapter: b.experts[0].adapter.clone() }, &eval);
        let merged = aggregate_expert(&base, &[
            a.experts[0].adapter.diff_from(&LoRAAdapter::lora_init(100)),
            b.experts[0].adapter.diff_from(&LoRAAdapter::lora_init(100)),
        ]).unwrap();
        let acc_m = model_acc(&ToyModel { base: base.clone(), adapter: merged }, &eval);
        assert!(acc_m >= acc_a.max(acc_b), "федерация {acc_m} vs {acc_a}/{acc_b}");
        // эксперт 1 не learner у обоих клиентов — дельта нулевая
        assert_eq!(a.experts[1].adapter.as_vec(), b.experts[1].adapter.as_vec());
        assert!(acc_a > 0.0 && acc_b > 0.0, "клиенты должны были выучить задачу 0");
    }
}
