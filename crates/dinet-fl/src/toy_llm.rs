//! TinyToyLLM — мини-модель, на которой проверяется вся механика
//! «speak → learn → федеративный раунд»: личный LoRA-адаптер, локальное
//! дообучение на партиях одобренных ответов, DP-клиппинг, secure-aggregation.
//!
//! Архитектура (детерминированная, зафиксирована):
//!   вход: контекст из 4 токенов (hot-V vector конкат, IN = 32 dim)
//!   скрытый слой: h = tanh(W1·x + b1), H = 24         (заморожен)
//!   выход: logits = (W2_base + B·A)·h, V = 8          (учится ТОЛЬКО BA)
//!   LoRA r = 4 ⇒ адаптер = 4·24 + 8·4 = 128 параметров.
//!
//! Локальная «преференс-цели» (упрощённый DPO-lite): для каждой сессии
//! одобрений — CE от chosen (вес +1), от rejected — weight −λ (штрафование
//! повторения нежелательного ответа), всё смешанное в один шаг градиента.
//!
//! В боеевой версии ToyLLM заменяется на llama.cpp/GGUF-модель с тем же
//! интерфейсом `LocalTrainer` (crate-обвязке по изменению не нужно —
//! интерфейс устойчив).

use super::{dp_noise, Lcg};

pub const CTX: usize = 4; // контекст: 4 токена
pub const V: usize = 8; // словарь
pub const IN: usize = CTX * V; // 32
pub const H: usize = 48;
pub const R: usize = 8; // LoRA rank

/// Замороженная «база» модели.
#[derive(Debug, Clone)]
pub struct ToyBase {
    pub w1: Vec<f32>,      // H×IN
    pub b1: Vec<f32>,      // H
    pub w2_base: Vec<f32>, // V×H
}

#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// контекст: 4 токена (индексы 0..V)
    pub tokens: [u8; CTX],
    /// «approved» ответ (chosen)
    pub chosen: u8,
    /// «не хочу такой ответ» (rejected), u8::MAX = нет
    pub rejected: Option<u8>,
}

impl ToyBase {
    pub fn from_seed(seed: u64) -> ToyBase {
        let mut rng = Lcg::new(seed);
        let n1 = H * IN;
        let mut w1 = vec![0.0f32; n1];
        for x in &mut w1 {
            *x = rng.gaussian() * 0.1;
        }
        // «Предобученная база»: первые V скрытых наборов читают ВЕТКУ ПОСЛЕДНЕГО
        // токена (позиция CTX-1) с чистым весом — так у адаптера есть
        // линейно-сепарабельная сигнатура t3 (аналог «LLM знает контекст»).
        for idx in 0..V.min(H) {
            w1[idx * IN + (CTX - 1) * V + idx] = 0.6;
        }
        let b1 = vec![0.0f32; H];
        let n2 = V * H;
        let mut w2_last = vec![0.0f32; n2];
        for x in &mut w2_last {
            *x = rng.gaussian() * 0.05;
        }
        // минорная выправка: чуть сильнее сигналы выходного слоя
        for (i, x) in w2_last.iter_mut().enumerate() {
            if i % V == i / V {
                *x += 0.15; // дикот для identity-связи
            }
        }
        ToyBase { w1, b1, w2_base: w2_last }
    }
}

/// Личный адаптер пользователя (A: R×H, B: V×R), живёт ЛОКАЛЬНО.
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub a: Vec<f32>, // R×H
    pub b: Vec<f32>, // V×R
}

impl LoRAAdapter {
    pub fn zero() -> LoRAAdapter {
        LoRAAdapter { a: vec![0.0; R * H], b: vec![0.0; V * R] }
    }

    /// Каноническая LoRA-инициализация: A — слабый случайный, B — ноль.
    /// При нулевом старте оба градиентных потока равны нулю (dead start),
    /// поэтому A должен быть ненулевым.
    pub fn lora_init(seed: u64) -> LoRAAdapter {
        let mut rng = Lcg::new(seed);
        let a = (0..R * H).map(|_| rng.gaussian() * 0.1).collect();
        let b = vec![0.0; V * R];
        LoRAAdapter { a, b }
    }

    pub fn num_params() -> usize {
        R * H + V * R // 128
    }

    pub fn as_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(Self::num_params());
        v.extend_from_slice(&self.a);
        v.extend_from_slice(&self.b);
        v
    }

    pub fn from_vec(v: &[f32]) -> LoRAAdapter {
        let mut a = vec![0.0; R * H];
        a.copy_from_slice(&v[..R * H]);
        let mut b = vec![0.0; V * R];
        b.copy_from_slice(&v[R * H..]);
        LoRAAdapter { a, b }
    }
}

#[derive(Debug, Clone)]
pub struct ToyModel {
    pub base: ToyBase,
    pub adapter: LoRAAdapter,
}

impl ToyModel {
    pub fn new(base: ToyBase) -> ToyModel {
        ToyModel { base, adapter: LoRAAdapter::zero() }
    }

    fn one_hot(tok: u8) -> [f32; V] {
        let mut v = [0.0f32; V];
        v[tok as usize % V] = 1.0;
        v
    }

    fn ba(&self) -> Vec<f32> {
        // BA = B·A, V×H
        let mut ba = vec![0.0f32; V * H];
        for vi in 0..V {
            for k in 0..H {
                let mut s = 0.0;
                for r in 0..R {
                    s += self.adapter.b[vi * R + r] * self.adapter.a[r * H + k];
                }
                ba[vi * H + k] = s;
            }
        }
        ba
    }

    /// Логиты одного примера.
    pub fn logits(&self, tokens: &[u8; CTX]) -> Vec<f32> {
        let mut x = vec![0.0f32; IN];
        for (t, tok) in tokens.iter().enumerate() {
            x[t * V..(t + 1) * V].copy_from_slice(&Self::one_hot(*tok));
        }
        self.logits_from_x(&x)
    }

    fn logits_from_x(&self, x: &[f32]) -> Vec<f32> {
        let ba = self.ba();
        let mut h = vec![0.0f32; H];
        for (k, hk) in h.iter_mut().enumerate() {
            let mut z = self.base.b1[k];
            for i in 0..IN {
                z += self.base.w1[k * IN + i] * x[i];
            }
            *hk = z.tanh();
        }
        let mut out = vec![0.0f32; V];
        for vi in 0..V {
            let mut s = 0.0;
            for k in 0..H {
                s += (self.base.w2_base[vi * H + k] + ba[vi * H + k]) * h[k];
            }
            out[vi] = s;
        }
        out
    }

    pub fn predict(&self, tokens: &[u8; CTX]) -> u8 {
        let lg = self.logits(tokens);
        let mut bi = 0usize;
        for (i, l) in lg.iter().enumerate().skip(1) {
            if *l > lg[bi] {
                bi = i;
            }
        }
        bi as u8
    }

    /// Один градиент-шаг по батчу сэмплов (упрощённый DPO-lite).
    /// Возвращает копилу градиент дельты адаптера (не применённую).
    pub fn pref_grad(&self, batch: &[Sample], lambda: f32) -> [Vec<f32>; 2] {
        let mut ga = vec![0.0f32; R * H];
        let mut gb = vec![0.0f32; V * R];
        for s in batch {
            self.pref_grad_one(s, lambda, &mut ga, &mut gb);
        }
        [ga, gb]
    }

    fn pref_grad_one(&self, s: &Sample, lambda: f32, ga: &mut [f32], gb: &mut [f32]) {
        // forward
        let mut x = vec![0.0f32; IN];
        for (t, tok) in s.tokens.iter().enumerate() {
            x[t * V..(t + 1) * V].copy_from_slice(&Self::one_hot(*tok));
        }
        let mut h = vec![0.0f32; H];
        for (k, hk) in h.iter_mut().enumerate() {
            let mut z = self.base.b1[k];
            for i in 0..IN {
                z += self.base.w1[k * IN + i] * x[i];
            }
            *hk = z.tanh();
        }
        let logits = self.logits_from_x(&x);
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let z: f32 = p.iter().sum();
        for e in &mut p {
            *e /= z;
        }
        // (1) CE выбранного ответа: dl = p − onehot_c
        let mut dl: Vec<f32> = p.clone();
        dl[s.chosen as usize] -= 1.0;
        // (2) Hinge-предпочтение (saturating, DPO-lite): если logP(c) − logP(r) < M,
        //     поднимать logit(c)/опускать logit(r) ОТНОСИТЕЛЬНО, иначе — ноль
        //     (без неограниченного давления Wооглуp rejected → не разрушает общий lанс).
        if let Some(rej) = s.rejected {
            const EPS: f32 = 1e-6;
            let lc = (p[s.chosen as usize] + EPS).ln();
            let lr_ = (p[rej as usize] + EPS).ln();
            let gap = lc - lr_;
            if gap < lambda {
                // dl_hinge[v] = o_r[v] − o_c[v]: ascend gap ⇒ descent-update знак
                dl[rej as usize] += 1.0;
                dl[s.chosen as usize] -= 1.0;
            }
        }
        for vi in 0..V {
            let d = dl[vi];
            if d.abs() < f32::EPSILON {
                continue;
            }
            for k in 0..H {
                for r in 0..R {
                    // d/dB[v,r] = Σ_k dAB[v,k]/dB[v,r] · h_k = A[r,k]·h_k; знак следуй dl
                    gb[vi * R + r] += d * self.adapter.a[r * H + k] * h[k];
                    // d/dA[r,k] = Σ_v dl_v · D[v,r] · h_k
                    ga[r * H + k] += d * self.adapter.b[vi * R + r] * h[k];
                }
            }
        }
    }
}

/// Результат локального обучения одного клиента за раунд.
pub struct LocalFit {
    /// Δ-адаптер (уже DP-clip+применьшенный шум — ГОТОВ К СЕКРЕТ-АГИ)
    pub delta: Vec<f32>,
    /// фабрика сигнала: norm до clip
    pub raw_norm: f32,
    /// шагов сделано (эл. прогресс Proof-of-Training-аргумент)
    pub steps: u32,
}

/// Локальный цикл: K шагов SGD по преференс-батчу.
pub fn local_fit(
    base: &ToyBase,
    adapter: &LoRAAdapter,
    batch: &[Sample],
    steps: u32,
    lr: f32,
    lambda: f32,
    max_norm: f32,
    sigma: f32,
    rng: &mut Lcg,
) -> LocalFit {
    let mut a = adapter.clone();
    for _ in 0..steps {
        let cur = ToyModel { base: base.clone(), adapter: a.clone() };
        let [ga, gb] = cur.pref_grad(batch, lambda);
        let n = batch.len() as f32;
        for (p, g) in a.a.iter_mut().zip(&ga) {
            *p -= lr * g / n;
        }
        for (p, g) in a.b.iter_mut().zip(&gb) {
            *p -= lr * g / n;
        }
    }
    let delta = a.diff_from(adapter);
    let raw_norm = super::norm2(&delta);
    let noised = dp_noise(&delta, max_norm, sigma, rng);
    LocalFit { delta: noised, raw_norm, steps }
}

/// diff_from — Δ от базового адаптера (для FL-апдейта).
impl LoRAAdapter {
    pub fn diff_from(&self, other: &LoRAAdapter) -> Vec<f32> {
        let a_delta: Vec<f32> = self.a.iter().zip(&other.a).map(|(x, y)| x - y).collect();
        let b_delta: Vec<f32> = self.b.iter().zip(&other.b).map(|(x, y)| x - y).collect();
        let mut v = a_delta;
        v.extend(b_delta);
        v
    }
}

// ---------------------------------------------------------------------------
// Микроязык чата v0: слово i = токен i. Точки прикрепления llama.cpp:
// v1 заменит lex_to_tokens на натуральный токенизатор реальных текстов.
// ---------------------------------------------------------------------------

pub const LEX: [&str; V] = ["ай", "би", "ви", "ги", "ди", "ей", "жи", "зи"];

/// Текст (пробелы-разделённые слова микроязыка) → токены; неизвестное слово → None.
pub fn lex_to_tokens(text: &str) -> Option<Vec<u8>> {
    let mut t = Vec::new();
    for w in text.split_whitespace() {
        match LEX.iter().position(|l| *l == w) {
            Some(i) => t.push(i as u8),
            None => return None,
        }
    }
    if t.is_empty() { None } else { Some(t) }
}

/// Тест-хелпер: «правильный» локальный язык семьи (легко выучить);
/// также фиксируются искажённые модусы (для византийцев).
pub mod toy_world {
    use super::*;

    /// Правило мира: следующий токен = (t3 + 1) % V (простая сменяемость).
    pub fn truth_next(tokens: &[u8; CTX]) -> u8 {
        (tokens[CTX - 1] + 1) % V as u8
    }

    /// LCG-обход: младшие биты LCG имеют короткие периоды (классика),
    /// поэтому токены берем из СТАРШИХ байтов.
    fn token(rng: &mut Lcg) -> u8 {
        ((rng.next() >> 40) as u8) % V as u8
    }

    pub fn honesty_batch(n: u32, seed: u64) -> Vec<Sample> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| {
            let tokens = [
                token(&mut rng),
                token(&mut rng),
                token(&mut rng),
                token(&mut rng),
            ];
            let chosen = truth_next(&tokens);
            Sample { tokens, chosen, rejected: Some((chosen + 3) % V as u8) }
        })
        .collect()
    }

    /// Византийский мир: правило сломано (перемичная сила по токену t0).
    pub fn byz_batch(n: u32, seed: u64) -> Vec<Sample> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| {
            let tokens = [
                token(&mut rng),
                token(&mut rng),
                token(&mut rng),
                token(&mut rng),
            ];
            let chosen = (tokens[0] + 5) % V as u8; // «отравитель» телепает кнопку
            Sample { tokens, chosen, rejected: None }
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::local_fit;
    use super::toy_world::{self, truth_next};

    #[test]
    fn adapter_shape() {
        // R*H + V*R = 8*48 + 8*8 = 448
        assert_eq!(LoRAAdapter::num_params(), R * H + V * R);
        let x = LoRAAdapter::zero();
        let v = x.as_vec();
        assert_eq!(v.len(), LoRAAdapter::num_params());
        assert!(v.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn learns_local_rule() {
        // до обучения: модель предсказывает ≈ случайно
        let base = ToyBase::from_seed(7);
        let mut m = ToyModel::new(base.clone());
        let batch = toy_world::honesty_batch(64, 11);
        let mut hits0 = 0;
        for s in batch.iter().take(32) {
            if m.predict(&s.tokens) == truth_next(&s.tokens) {
                hits0 += 1;
            }
        }
        let _ = hits0;
        // обучаем 30 шагов
        let mut rng = Lcg::new(99);
        let fit = local_fit(&base, &LoRAAdapter::lora_init(42), &batch, 150, 0.7, 2.0, 100.0, 0.0, &mut rng);
        let ada = LoRAAdapter::from_vec(&fit.delta);
        m.adapter = ada;
        let mut hits = 0;
        for s in batch.iter().take(32) {
            if m.predict(&s.tokens) == truth_next(&s.tokens) {
                hits += 1;
            }
        }
        assert!(hits >= 24, "should learn rule: hits={}/32", hits);
    }

    #[test]
    fn numeric_gradient_check() {
        // Finite-difference проверка pref_grad по нескольким случайным параметрам
        let base = ToyBase::from_seed(7);
        let adapter = LoRAAdapter::lora_init(42);
        let batch = toy_world::honesty_batch(8, 321);
        let model = ToyModel { base, adapter };
        // margin (λ) обязан совпадать в лосс-модели и в pref_grad,
        // иначе численная проверка меряет другую функцию.
        const MARGIN: f32 = 0.3;
        let [ga, gb] = model.pref_grad(&batch, MARGIN);
        let loss = |m: &ToyModel| -> f32 {
            let mut s = 0.0;
            for smp in &batch {
                let lg = m.logits(&smp.tokens);
                let mx = lg.iter().copied().fold(f32::MIN, f32::max);
                let z: f32 = lg.iter().map(|l| (l - mx).exp()).sum();
                // CE(chosen) + hinge-предпочтение max(0, M − gap)
                let two = std::f32::consts::E;
                let _ = two;
                s += -(lg[smp.chosen as usize] - mx) + z.ln();
                if let Some(rej) = smp.rejected {
                    let pc = ((lg[smp.chosen as usize] - mx).exp() / z + 1e-6).ln();
                    let pr = ((lg[rej as usize] - mx).exp() / z + 1e-6).ln();
                    s += (MARGIN - (pc - pr)).max(0.0);
                }
            }
            s
        };
        let eps = 1e-3f32;
        // индексы по всему вектору параметров: и A-часть, и B-часть
        for probe_idx in [0usize, 9, 33, 100, 200, 300, 383, 384, 400, 415, 430, 447] {
            let g_analytic = if probe_idx < R * H { ga[probe_idx] } else { gb[probe_idx - R * H] };
            let mut m2 = ToyModel { base: model.base.clone(), adapter: model.adapter.clone() };
            let vv = if probe_idx < R * H {
                let a0 = m2.adapter.a[probe_idx];
                m2.adapter.a[probe_idx] = a0 + eps;
                a0
            } else {
                let bi = probe_idx - R * H;
                let b0 = m2.adapter.b[bi];
                m2.adapter.b[bi] = b0 + eps;
                b0
            };
            let lp = loss(&m2);
            let mut m3 = ToyModel { base: model.base.clone(), adapter: model.adapter.clone() };
            let vm = if probe_idx < R * H {
                let a0 = m3.adapter.a[probe_idx];
                m3.adapter.a[probe_idx] = a0 - eps;
                a0
            } else {
                let bi = probe_idx - R * H;
                let b0 = m3.adapter.b[bi];
                m3.adapter.b[bi] = b0 - eps;
                b0
            };
            let lm = loss(&m3);
            let g_num = (lp - lm) / (2.0 * eps);
            let ratio = g_num.abs() / (g_analytic.abs() + 1e-9);
            println!(
                "idx {:>3} analytic {:+.6} numeric {:+.6} ratio {:.4} ({} v+ {})",
                probe_idx, g_analytic, g_num, ratio, vv, vm
            );
            assert!(
                (g_analytic.abs() < 1e-5 && g_num.abs() < 1e-4)
                    || (ratio > 0.7 && ratio < 1.4),
                "idx {} no match: analytic {:+.6} numeric {:+.6}",
                probe_idx, g_analytic, g_num
            );
        }
    }

    #[test]
    fn probe_pref_grad_trajectory() {
        let base = ToyBase::from_seed(7);
        let adapter = LoRAAdapter::lora_init(42);
        let _rng = Lcg::new(5);
        let batch = toy_world::honesty_batch(64, 1000);
        let probe = toy_world::honesty_batch(96, 42_424_212);
        let probe_model = |m: &ToyModel| {
            let hits: u32 = probe
                .iter()
                .filter(|s| m.predict(&s.tokens) == s.chosen)
                .count() as u32;
            hits as f32 / probe.len() as f32
        };
        let mut a = adapter.clone();
        for step in 1..=600 {
            let cur = ToyModel { base: base.clone(), adapter: a.clone() };
            let [ga, gb] = cur.pref_grad(&batch, 0.3);
            let n = batch.len() as f32;
            let lr = if step < 300 { 0.5 } else { 0.15 }; // LR decay: стабилизировать
            for (p, g) in a.a.iter_mut().zip(&ga) {
                *p -= lr * g / n;
            }
            for (p, g) in a.b.iter_mut().zip(&gb) {
                *p -= lr * g / n;
            }
            if step % 60 == 0 {
                let m = ToyModel { base: base.clone(), adapter: a.clone() };
                println!("step {:>3} probe_acc {:.3}", step, probe_model(&m));
            }
        }
    }
}
