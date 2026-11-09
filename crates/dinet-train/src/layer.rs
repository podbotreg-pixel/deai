// ---------------------------------------------------------------------------
// LAYER-LORA (train-cpp v1): LoRA-дельта на ВНУТРЕННЕЙ матрице трансформера
// — down-проекции MLP последнего слоя (blk.N.ffn_down).
//
// Постановка. llama.cpp делает forward; eval-колбэк (dinet-llama::probe2)
// отдаёт нам ХВОСТ графа с базовыми величинами:
//   x        — вход ffn_down (выход swiGLU), [n_ff] на позицию;
//   h_pre    — скрытое состояние ДО финальной RMS-нормы,
//              h_pre = base_resid + W_down·x (резидуал последнего слоя);
//   hidden   — ПОСЛЕ нормы и веса: hidden = (h_pre/r)⊙w_norm;
//   logits   — базовые логиты llama.cpp (с ЕЁ де-квантованной головой).
//
// С LoRA на ffn_down меняется только выход down: δ_out = B·(A·x):
//   h_pre' = h_pre + δ_out (резидуал — passthrough, позиции не смешиваются,
//   ниже последнего слоя attention нет);
//   hidden' = rmsnorm(h_pre')·w_norm;
//   logits' = logits + W_head·(hidden' − hidden).
//
// Всё это ТОЧНО реализуемо без llama.cpp: норма — та же формула, вес головы
// читаем из GGUF (dinet_train::gguf). Градиент L по δ_out — полный обратный
// ход по хвосту: CE → head (Wᵀ) → якобиан RMS-нормы → δ. A/B — AdamW, как у
// head-LoRA (v0). Материал batch задаётся LayerBatch.
// ---------------------------------------------------------------------------

#![allow(dead_code)]

/// Топология layer-LoRA: n_in (n_ff), n_out (n_embd), r — ранг.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpec {
    pub n_in: u32,
    pub n_out: u32,
    pub r: u32,
}

/// Материал одного прохода (набор позиций микробатча).
#[derive(Debug, Clone, Default)]
pub struct LayerBatch {
    /// вход ffn_down последнего слоя: n_tok строк по n_in
    pub x: Vec<Vec<f32>>,
    /// до финальной нормы: n_tok строк по n_out
    pub h_pre: Vec<Vec<f32>>,
    /// после нормы и веса (база): n_tok строк по n_out
    pub hidden: Vec<Vec<f32>>,
    /// базовые логиты: n_tok строк по n_vocab
    pub base_logits: Vec<Vec<f32>>,
    /// целевой токен для позиции (i32::MIN — позиция не в лоссе)
    pub target: Vec<i32>,
}

/// Голова (фиксированный вес из GGUF) + финальная норма.
pub struct HeadWeights {
    /// row-major [n_vocab строк × d столбцов]
    pub w: Vec<f32>,
    pub n_vocab: usize,
    pub d: usize,
}

impl HeadWeights {
    /// logits = W·v: вектор [n_vocab].
    fn logits_of(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.n_vocab];
        let d = self.d;
        for (j, slot) in out.iter_mut().enumerate() {
            let row = &self.w[j * d..(j + 1) * d];
            *slot = dot(row, v);
        }
        out
    }

    /// vᵀ·W — Wᵀ·v (вектор [d]).
    fn wt_of(&self, g: &[f32]) -> Vec<f32> {
        let d = self.d;
        let mut out = vec![0f32; d];
        for (j, gj) in g.iter().enumerate() {
            if *gj == 0.0 {
                continue;
            }
            let row = &self.w[j * d..(j + 1) * d];
            for (dd, tj) in out.iter_mut().enumerate() {
                *tj += gj * row[dd];
            }
        }
        out
    }
}

/// train-пара для адаптера: A [r×n_in], B [n_out×r].
pub struct LayerLora {
    pub spec: LayerSpec,
    a: Vec<f32>,
    b: Vec<f32>,
    ma: Vec<f32>,
    va: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
    steps: u64,
}

/// Результат шага (для логов webchat).
#[derive(Debug, Clone, Copy)]
pub struct LayerStep {
    pub loss: f32,
    /// |δ| дельты ffn_down (пульс адаптера)
    pub delta_norm: f32,
    pub steps: u64,
}

impl LayerLora {
    pub fn new(spec: LayerSpec, seed: u64) -> Self {
        let nz = spec.r as usize;
        let n_in = spec.n_in as usize;
        // A — малые N(0, 2/n_in); B — нули: первая дельта ровно нулевая.
        let mut rng = Lcg(seed.max(1));
        let scale = (2.0 / n_in as f32).sqrt();
        let mut a = vec![0f32; nz * n_in];
        for v in a.iter_mut() {
            let u1 = rng.next() as f32 / u32::MAX as f32;
            let u2 = rng.next() as f32 / u32::MAX as f32;
            let z = (-2.0 * u1.max(1e-7).ln()).sqrt() * (6.2831853 * u2).cos();
            *v = z * scale;
        }
        let b = vec![0f32; spec.n_out as usize * nz];
        Self::from_ab(spec, a, b)
    }

    /// Прямая сборка с заданными A/B (для тестов и восстановления).
    pub fn from_ab(spec: LayerSpec, a: Vec<f32>, b: Vec<f32>) -> Self {
        let za = vec![0f32; a.len()];
        let zb = vec![0f32; b.len()];
        let va = za.clone();
        let vb = zb.clone();
        LayerLora { spec, a, b, ma: za, va, mb: zb, vb, steps: 0 }
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn a(&self) -> &[f32] {
        &self.a
    }

    pub fn b(&self) -> &[f32] {
        &self.b
    }

    /// LoRA-пара для составного адаптера (база — «blk.N.ffn_down.weight»).
    pub fn to_piece(&self, base_name: &str) -> crate::AdapterPiece {
        crate::AdapterPiece {
            base: base_name.to_string(),
            a: self.a.clone(),
            a_ne0: self.spec.n_in as u64,
            b: self.b.clone(),
            b_ne1: self.spec.n_out as u64,
            r: self.spec.r as u64,
        }
    }

    /// РЕЗЮМ v1.2: восстановить слой-адаптер из piece-пары. spec собирается
    /// ИЗ piece (n_in = a_ne0, n_out = b_ne1, r = r); caller заранее сверил
    /// эти числа с живой моделью. A/B — из файла, Adam-моменты и счётчик —
    /// с нуля (from_ab: мягкий прогрев).
    pub fn from_piece(pc: &crate::AdapterPiece) -> Result<Self, String> {
        let (r, n_in, n_out) = (pc.r as usize, pc.a_ne0 as usize, pc.b_ne1 as usize);
        if r == 0 || pc.a.len() != r * n_in || pc.b.len() != n_out * r {
            return Err(format!(
                "piece '{}' не сходится: a {} =? {r}·{n_in}, b {} =? {n_out}·{r}",
                pc.base,
                pc.a.len(),
                pc.b.len()
            ));
        }
        let spec = LayerSpec { n_in: n_in as u32, n_out: n_out as u32, r: r as u32 };
        Ok(Self::from_ab(spec, pc.a.clone(), pc.b.clone()))
    }

    /// Дельта для одного вектора x: δ = B·(A·x) — [n_out].
    pub fn delta_of(&self, x: &[f32]) -> Vec<f32> {
        let r = self.spec.r as usize;
        let n_in = self.spec.n_in as usize;
        let n_out = self.spec.n_out as usize;
        let mut z = vec![0f32; r];
        for (i, zi) in z.iter_mut().enumerate() {
            *zi = dot(&self.a[i * n_in..(i + 1) * n_in], x);
        }
        let mut d = vec![0f32; n_out];
        for (o, di) in d.iter_mut().enumerate() {
            *di = dot(&self.b[o * r..(o + 1) * r], &z);
        }
        d
    }

    /// Only-forward проход хвоста: средний CE по целевым позициям (без шага).
    pub fn eval_loss(&self, batch: &LayerBatch, head: &HeadWeights, norm_w: &[f32], eps: f32) -> f32 {
        self.forward_and_grads(batch, head, norm_w, eps).0
    }

    /// Один train-шаг: forward+backward по ХВОСТУ графа, потом AdamW по A/B.
    pub fn train_step(
        &mut self,
        batch: &LayerBatch,
        head: &HeadWeights,
        norm_w: &[f32],
        eps: f32,
        lr: f32,
        wd: f32,
    ) -> Option<LayerStep> {
        let (loss, ga, gb) = self.forward_and_grads(batch, head, norm_w, eps);
        if !loss.is_finite() {
            return None;
        }
        self.steps += 1;
        let t = self.steps as f32;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps_a = 1e-8f32;
        let c1 = 1.0 - b1.powf(t);
        let c2 = 1.0 - b2.powf(t);
        for i in 0..self.a.len() {
            adamw(&mut self.a[i], ga[i], &mut self.ma[i], &mut self.va[i], lr, wd, b1, b2, eps_a, c1, c2);
        }
        for i in 0..self.b.len() {
            adamw(&mut self.b[i], gb[i], &mut self.mb[i], &mut self.vb[i], lr, wd, b1, b2, eps_a, c1, c2);
        }
        // пульс: |δ| в base-масштабе (по первому вектору)
        let dn = {
            let mut dn = 0f32;
            for x in batch.x.iter().take(1) {
                let d = self.delta_of(x);
                dn = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            }
            dn
        };
        Some(LayerStep { loss, delta_norm: dn, steps: self.steps })
    }

    /// Полный хвост: forward (δ → δ_logits) + grad по δ (для backward).
    fn forward_and_grads(
        &self,
        batch: &LayerBatch,
        head: &HeadWeights,
        norm_w: &[f32],
        eps: f32,
    ) -> (f32, Vec<f32>, Vec<f32>) {
        let r = self.spec.r as usize;
        let n_in = self.spec.n_in as usize;
        let n_out = self.spec.n_out as usize;
        let n = batch.x.len().min(batch.target.len());
        let mut ga = vec![0f32; self.a.len()];
        let mut gb = vec![0f32; self.b.len()];

        let mut n_loss = 0u32;
        let mut loss_sum = 0f32;
        let mut z_buf = vec![0f32; r];
        let mut dlt = vec![0f32; n_out];
        let mut u_buf = vec![0f32; n_out];
        let mut y_delta = vec![0f32; n_out]; // delta-of-hidden: (u/r)⊙w − hidden
        let mut g_logits: Vec<f32> = vec![0f32; head.n_vocab]; // softmax−1
        let mut d_s = vec![0f32; n_out]; // dL/d(hidden')
        let mut d_u = vec![0f32; n_out]; // dL/d(δ_out) — ГЛАДНЫЙ градиент колбэка
        for p in 0..n {
            let t = batch.target[p];
            if t < 0 || t as usize >= head.n_vocab {
                // позиция вне лосса (контекстная) — градиентов нет
                continue;
            }
            let x = &batch.x[p];
            let h_pre = &batch.h_pre[p];
            let hid = &batch.hidden[p];
            let base = &batch.base_logits[p];
            if x.len() != n_in || h_pre.len() != n_out || hid.len() != n_out {
                // несовместимые данные — позиция пропущена
                continue;
            }
            // z = A·x
            for (i, zi) in z_buf.iter_mut().enumerate() {
                *zi = dot(&self.a[i * n_in..(i + 1) * n_in], x);
            }
            // δ = B·z
            for (o, di) in dlt.iter_mut().enumerate() {
                *di = dot(&self.b[o * r..(o + 1) * r], &z_buf);
            }
            // u = h_pre + δ; норма
            for (k, ui) in u_buf.iter_mut().enumerate() {
                *ui = h_pre[k] + dlt[k];
            }
            let s2 = u_buf.iter().map(|v| v * v).sum::<f32>() / n_out as f32 + eps;
            if !(s2.is_finite()) {
                return (f32::NAN, ga, gb);
            }
            let rms = s2.sqrt();
            for (k, yi) in y_delta.iter_mut().enumerate() {
                *yi = u_buf[k] * norm_w[k] / rms - hid[k];
            }
            // δ_logits
            let dlog = head.logits_of(&y_delta);
            // softmax(logits = base + dlog) — через log-softmax
            let mut mx = f32::NEG_INFINITY;
            for j in 0..head.n_vocab {
                let lot = base[j] + dlog[j];
                g_logits[j] = lot;
                if lot > mx {
                    mx = lot;
                }
            }
            let mut sum = 0f32;
            for lot in g_logits.iter() {
                sum += (lot - mx).exp();
            }
            let lse = mx + sum.ln();
            let logit_t = base[t as usize] + dlog[t as usize];
            let ce = lse - logit_t;
            if !ce.is_finite() {
                return (f32::NAN, ga, gb);
            }
            loss_sum += ce;
            n_loss += 1;
            for gv in g_logits.iter_mut() {
                *gv = (*gv - lse).exp();
            }
            g_logits[t as usize] -= 1.0;
            // δs = Wᵀ·g
            let ds = head.wt_of(&g_logits);
            // d_s = dL/d(hidden') = ds
            for (k, v) in d_s.iter_mut().enumerate() {
                *v = ds[k];
            }
            // RMS-норма backward: dL/du
            // inn = Σ ds_k·w_k·u_k
            let mut inn = 0f32;
            for k in 0..n_out {
                inn += d_s[k] * norm_w[k] * u_buf[k];
            }
            for k in 0..n_out {
                d_u[k] = d_s[k] * norm_w[k] / rms - u_buf[k] * inn / (rms * s2 * n_out as f32);
            }
            // grad B и A
            // gB_o_q += d_u_o · z_q ; gA_q_j += (Bᵀd_u)_q · x_j
            let mut btz = vec![0f32; r];
            for (o, duo) in d_u.iter().enumerate() {
                let brow = &self.b[o * r..(o + 1) * r];
                for q in 0..r {
                    btz[q] += duo * brow[q];
                }
            }
            for (i, ai) in ga.iter_mut().enumerate() {
                let q = i / n_in;
                let j = i % n_in;
                *ai += btz[q] * x[j];
            }
            for (oi, bi) in gb.iter_mut().enumerate() {
                let o = oi / r;
                let q = oi % r;
                *bi += d_u[o] * z_buf[q];
            }
        }
        if n_loss == 0 {
            return (0.0, ga, gb);
        }
        let nf = n_loss as f32;
        for g in ga.iter_mut() {
            *g /= nf;
        }
        for g in gb.iter_mut() {
            *g /= nf;
        }
        (loss_sum / nf, ga, gb)
    }
}

fn lse_unsafe(lse: f32, mx: f32) -> f32 {
    (lse - mx).exp()
}

/// AdamW по одному весу (как v0).
#[allow(clippy::too_many_arguments)]
fn adamw(
    p: &mut f32,
    g: f32,
    m: &mut f32,
    v: &mut f32,
    lr: f32,
    wd: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    c1: f32,
    c2: f32,
) {
    *m = b1 * *m + (1.0 - b1) * g;
    *v = b2 * *v + (1.0 - b2) * g * g;
    let mhat = *m / c1;
    let vhat = *v / c2;
    if wd > 0.0 {
        *p *= 1.0 - lr * wd;
    }
    *p -= lr * mhat / (vhat.sqrt() + eps);
}

/// ГПСЧ (инициализация A).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn dot(x: &[f32], y: &[f32]) -> f32 {
    let mut s = 0f32;
    for (a, b) in x.iter().zip(y) {
        s += a * b;
    }
    s
}

// ---------------------------------------------------------------------------
// Тесты: синтетическая сцена (без llama.cpp) — вся математика локальна.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod layer_tests {
    use super::*;
    const N_IN: usize = 3; // n_ff
    const N_OUT: usize = 4; // n_embd
    const N_VOCAB: usize = 6;
    const R: usize = 2;

    fn spec() -> LayerSpec {
        LayerSpec { n_in: N_IN as u32, n_out: N_OUT as u32, r: R as u32 }
    }

    /// Синтетическая сцена: базовый хвост + данные микробатча + веса.
    fn scene(seed: u64) -> (LayerBatch, HeadWeights, Vec<f32>, f32) {
        let mut rng = Lcg(seed.max(1).wrapping_add(7));
        let norm_w: Vec<f32> = (0..N_OUT).map(|_| 1.0 + 0.2 * (rng.next() as f32 / u32::MAX as f32 - 0.5)).collect();
        let eps = 1e-5f32;
        let mut x = Vec::new();
        let mut h_pre = Vec::new();
        let mut hidden = Vec::new();
        let mut base_logits = Vec::new();
        let mut target = Vec::new();
        for _ in 0..4 {
            let xv: Vec<f32> = (0..N_IN).map(|_| (rng.next() as f32 / u32::MAX as f32 - 0.5) * 3.0).collect();
            let ff: Vec<f32> = (0..N_OUT).map(|_| (rng.next() as f32 / u32::MAX as f32 - 0.5) * 2.0).collect();
            let resid: Vec<f32> = (0..N_OUT).map(|_| rng.next() as f32 / u32::MAX as f32 - 0.5).collect();
            let mut hp = vec![0f32; N_OUT];
            for (k, hpk) in hp.iter_mut().enumerate() {
                // база: скалярный W_down·x + резидуал (фиктивно)
                *hpk = resid[k] + ff[k] * xv[k % N_IN];
            }
            let s2 = hp.iter().map(|v| v * v).sum::<f32>() / N_OUT as f32 + eps;
            let hid: Vec<f32> = hp
                .iter()
                .zip(norm_w.iter())
                .map(|(v, w)| v * w / s2.sqrt())
                .collect();
            let bl: Vec<f32> = (0..N_VOCAB).map(|_| (rng.next() as f32 / u32::MAX as f32 - 0.5) * 2.0).collect();
            let tgt = (rng.next() as usize) % N_VOCAB;
            x.push(xv);
            h_pre.push(hp);
            hidden.push(hid);
            base_logits.push(bl);
            target.push(tgt as i32);
        }
        let w: Vec<f32> = (0..N_VOCAB * N_OUT).map(|_| (rng.next() as f32 / u32::MAX as f32 - 0.5) * 1.5).collect();
        (
            LayerBatch { x, h_pre, hidden, base_logits, target },
            HeadWeights { w, n_vocab: N_VOCAB, d: N_OUT },
            norm_w,
            eps,
        )
    }

    /// B=0 → δ=0 → лосс ровно CE(base_logits).
    #[test]
    fn zero_delta_loss_is_base_ce() {
        let (batch, head, norm_w, eps) = scene(3);
        let lora = LayerLora::new(spec(), 9);
        let loss0 = lora.eval_loss(&batch, &head, &norm_w, eps);
        let mut ce = 0f32;
        for (p, bl) in batch.base_logits.iter().enumerate() {
            let t = batch.target[p] as usize;
            let mx = bl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + bl.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
            ce += lse - bl[t];
        }
        let ce = ce / batch.target.len() as f32;
        assert!((loss0 - ce).abs() < 1e-5, "loss B=0 {loss0} != CE(base) {ce}");
    }

    /// Аналитика (forward_and_grads) против полного FD по A и B.
    #[test]
    fn grads_match_finite_diff() {
        let (batch, head, norm_w, eps) = scene(11);
        let mut a = vec![0f32; R * N_IN];
        let mut b = vec![0f32; N_OUT * R];
        for v in a.iter_mut() {
            *v = 0.21;
        }
        for (i, v) in b.iter_mut().enumerate() {
            *v = 0.07 * (i as f32 + 1.0);
        }
        let lora = LayerLora::from_ab(spec(), a.clone(), b.clone());
        let (_, ga, gb) = lora.forward_and_grads(&batch, &head, &norm_w, eps);

        let h = 1e-2f32;
        let loss_of = |a: &[f32], b: &[f32]| {
            LayerLora::from_ab(spec(), a.to_vec(), b.to_vec())
                .eval_loss(&batch, &head, &norm_w, eps)
        };
        for i in 0..a.len() {
            let mut ap = a.clone();
            ap[i] += h;
            let mut am = a.clone();
            am[i] -= h;
            let fd = (loss_of(&ap, &b) - loss_of(&am, &b)) / (2.0 * h);
            assert!(
                (fd - ga[i]).abs() <= frame(fd, ga[i]),
                "A[{i}]: fd={fd} an={}",
                ga[i]
            );
        }
        for i in 0..b.len() {
            let mut bp = b.clone();
            bp[i] += h;
            let mut bm = b.clone();
            bm[i] -= h;
            let fd = (loss_of(&a, &bp) - loss_of(&a, &bm)) / (2.0 * h);
            assert!(
                (fd - gb[i]).abs() <= frame(fd, gb[i]),
                "B[{i}]: fd={fd} an={}",
                gb[i]
            );
        }
    }

    /// Допуск сравнения (относительный + малый абсолютный).
    fn frame(fd: f32, an: f32) -> f32 {
        3e-2 * fd.abs().max(an.abs()) + 1e-4
    }

    /// train_step снижает лосс (B стартует с нуля — стандартный LoRA-путь).
    #[test]
    fn train_step_reduces_loss() {
        let (batch, head, norm_w, eps) = scene(23);
        let mut lora = LayerLora::new(spec(), 5);
        let mut first = None;
        for step in 0..24 {
            let s = lora
                .train_step(&batch, &head, &norm_w, eps, 0.08, 0.0)
                .expect("шаг конечен");
            if step == 0 {
                first = Some(s.loss);
            }
            assert!(s.loss.is_finite());
        }
        let f = first.expect("первый шаг");
        // финальный лосс — пересчитываем
        let final_loss = lora.eval_loss(&batch, &head, &norm_w, eps);
        assert!(
            final_loss < f - 0.05,
            "лосс не снизился: {f} → {final_loss}"
        );
        // дельта живая
        let d = lora.delta_of(&batch.x[0]);
        let dn = d.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(dn > 0.0, "дельта нулевая после обучения");
    }

    /// РЕЗЮМ v1.2 слоя: fresh → шаги → to_piece → from_piece — A/B сохранены
    /// точно, spec собран из piece (n_in/n_out/r), Adam-моменты и шаги —
    /// с нуля; следующий шаг резюма продолжает с ТОГО ЖЕ лосса
    /// (train_step возвращает лосс ДО обновления); битая форма отвергается.
    #[test]
    fn layer_resume_from_piece() {
        let sp = spec();
        let (batch, head, norm_w, eps) = scene(23);
        let mut lora = LayerLora::new(sp, 5);
        for _ in 0..20 {
            let _ = lora
                .train_step(&batch, &head, &norm_w, eps, 0.08, 0.0)
                .expect("шаг конечен");
        }
        let pc = lora.to_piece("blk.23.ffn_down.weight");

        let mut resumed = LayerLora::from_piece(&pc).expect("резюм");
        assert_eq!(resumed.a(), pc.a.as_slice(), "A должна прийти из файла");
        assert_eq!(resumed.b(), pc.b.as_slice(), "B должна прийти из файла");
        assert_eq!(
            (resumed.spec.n_in, resumed.spec.n_out, resumed.spec.r),
            (sp.n_in, sp.n_out, sp.r),
            "spec собран из piece"
        );
        assert_eq!(resumed.steps(), 0, "шаги сброшены");
        assert!(resumed.ma.iter().all(|&x| x == 0.0), "моменты Adam — с нуля");
        assert!(resumed.vb.iter().all(|&x| x == 0.0), "моменты Adam — с нуля");

        // непрерывность: следующий шаг резюма == следующий шаг исходного
        let was = lora
            .train_step(&batch, &head, &norm_w, eps, 0.08, 0.0)
            .expect("шаг конечен");
        let got = resumed
            .train_step(&batch, &head, &norm_w, eps, 0.08, 0.0)
            .expect("шаг конечен");
        assert!(
            (was.loss - got.loss).abs() < 1e-5,
            "резюм разошёлся с продолжением: {:?} vs {:?}",
            was,
            got
        );

        // и дальше снижается (продолжение, а не обнуление)
        let mut last = got.loss;
        for _ in 0..24 {
            last = resumed
                .train_step(&batch, &head, &norm_w, eps, 0.08, 0.0)
                .expect("шаг конечен")
                .loss;
        }
        assert!(last < got.loss, "резюм не продолжил падение: {got:?} → {last}");

        // битая форма отвергается
        let bad = crate::AdapterPiece {
            base: "blk.23.ffn_down.weight".into(),
            a: vec![0f32; 3],
            a_ne0: 3,
            b: vec![0f32; 4 * 2],
            b_ne1: 4,
            r: 2,
        };
        assert!(LayerLora::from_piece(&bad).is_err(), "битая форма прошла");
    }
}
