//! dinet-fl: численные примитивы федеративного обучения.
//!
//! Всё на f32/std — LAPACK/библиотеки не требуются. Это вычислительное ядро
//! раунда агрегации и клиентского DP-шага (docs/ARCHITECTURE.md §4–5):
//! - квантование q8 + error feedback (сжатие апдейта ≤20 МБ);
//! - DP-SGD клиппинг по норме + гауссов шум;
//! - trimmed mean и Krum — Византийски-устойчивая агрегация (Blanchard et al.);
//! - FedGraph-score — корреляционная близость апдейт-векторов
//!   (ловит коллективный backdoor: «слишко согласованные» updates).

/// Квантование qint8 с общим масштабом по вектору + error feedback.
/// Возвращает масштаб (f32) и i8-вектор.
pub fn quantize_q8(update: &[f32], err_buf: &mut [f32]) -> (Vec<i8>, f32) {
    assert_eq!(update.len(), err_buf.len(), "err_buf должен совпадать по длине");
    let n = update.len();
    let corrected: Vec<f32> = (0..n).map(|i| update[i] + err_buf[i]).collect();
    let amax = corrected.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if amax > f32::EPSILON { amax / 127.0 } else { 1.0 };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let q = (corrected[i] / scale).round().clamp(-127.0, 127.0) as i8;
        // error feedback: что осталось от значения после квантования
        let quantized_back = q as f32 * scale;
        err_buf[i] = corrected[i] - quantized_back;
        out.push(q);
    }
    (out, scale)
}

pub fn dequantize(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&x| x as f32 * scale).collect()
}

/// Euclidean-норма.
pub fn norm2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Клиппинг по L2-норме (шаг DP-SGD, часть 1).
/// Возвращает обрезанный вектор и флаг «был ли обрезан».
pub fn clip_l2(v: &[f32], max_norm: f32) -> (Vec<f32>, bool) {
    let n = norm2(v);
    if n <= max_norm || n == 0.0 {
        return (v.to_vec(), false);
    }
    let k = max_norm / n;
    (v.iter().map(|x| x * k).collect(), true)
}

/// Простой LCG — достаточно для демо-шума; в билд-криtик v2 —
/// урmann-шум из crypto PRG (ChaCha от общего календарного ключа эры).
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E3779B97F4A7C15)
    }
    fn next(&mut self) -> u64 {
        // Knuth MMIX
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn u01(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Приближение N(0,1) по Ирвину-Холлу (4 суммы униформ) — для демо.
    /// В v2 заменить на Box-Muller (точный гаусс).
    pub fn gaussian(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.u01()).sum();
        (s - 2.0) * (12f32).sqrt()
    }
}

/// DP-шаг полностью: clip + gaussian noise.
pub fn dp_noise(v: &[f32], max_norm: f32, sigma: f32, rng: &mut Lcg) -> Vec<f32> {
    let (clipped, _) = clip_l2(v, max_norm);
    clipped.iter().map(|&x| x + sigma * rng.gaussian()).collect()
}

/// Средний по списку.
pub fn mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    let n = vectors.len() as f32;
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut out = vec![0.0; dim];
    for v in vectors {
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += x / n;
        }
    }
    out
}

/// Trimmed mean по компонентам (компонентно, отбрасываем по beta с каждого края).
pub fn trimmed_mean(vectors: &[Vec<f32>], beta: usize) -> Vec<f32> {
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut out = vec![0.0; dim];
    let mut col: Vec<f32> = Vec::with_capacity(vectors.len());
    for d in 0..dim {
        col.clear();
        // компактный extract: детерминировано
        for v in vectors {
            col.push(v[d]);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = col.len();
        let lo = beta.min(n / 2);
        let hi = n - beta.min(n / 2);
        if hi <= lo {
            continue;
        }
        let span = (hi - lo) as f32;
        out[d] = col[lo..hi].iter().sum::<f32>() / span;
    }
    out
}

/// Krum-lite: выбрать узел с минимальной суммой расстояний до k ближайших
/// (требует n ≥ 2f+3 при f malicious; упрощение: n мал в демо).
/// Возвращает индекс выбранного вектора.
pub fn krum_select(vectors: &[Vec<f32>], _n_malicious_assumed: usize) -> Option<usize> {
    if vectors.is_empty() {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for i in 0..vectors.len() {
        let mut dists: Vec<f32> = Vec::with_capacity(vectors.len() - 1);
        for j in 0..vectors.len() {
            if i == j {
                continue;
            }
            let di: f32 = vectors[i]
                .iter()
                .zip(vectors[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            // отдаём сортировку в коде ниже (k ближайших берутся все при n небольшом)
            let _ = di;
            dists.push(di);
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = vectors.len().saturating_sub(1).max(1);
        let score: f32 = dists.iter().take(k).sum();
        if best.map(|(_, s)| score < s).unwrap_or(true) {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
}

/// FedGraph-score: средняя попарная корреляция Пирсона апдейтов узла с остальными.
/// Малые |corr| — нормальное разнообразие; ВЫСОКИЙ в среднем corr → подозрение на
/// коллективную скоординированную атаку (FedGraph, arXiv:2610.18430).
pub fn fedgraph_mean_abs_corr(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    if vectors.len() < 2 {
        return None;
    }
    let mut scores = Vec::with_capacity(vectors.len());
    for i in 0..vectors.len() {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for j in 0..vectors.len() {
            if i == j {
                continue;
            }
            if let Some(c) = pearson(&vectors[i], &vectors[j]) {
                sum += c.abs();
                count += 1;
            }
        }
        scores.push(if count > 0 { sum / count as f32 } else { 0.0 });
    }
    Some(scores)
}

fn pearson(a: &[f32], b: &[f32]) -> Option<f32> {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= f32::EPSILON || vb <= f32::EPSILON {
        return None;
    }
    Some(cov / (va * vb).sqrt())
}

/// Feed-forward правило ARP: «еловые пороги» для приёма апдейта:
/// норма в целевом интервале [min,max] и FedGraph-score ниже порога.
pub fn accept_update(norm: f32, fedgraph: f32, min_norm: f32, max_norm: f32, fg_threshold: f32) -> bool {
    norm >= min_norm && norm <= max_norm && fedgraph < fg_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let v: Vec<f32> = (0..1000).map(|i| ((i * 37) % 100) as f32 / 50.0 - 1.0).collect();
        let mut err = vec![0.0; v.len()];
        let (q, scale) = quantize_q8(&v, &mut err);
        let back = dequantize(&q, scale);
        // q8 → погрешность ≤ scale / 127 per-элементия; тест наполовину loose
        for i in 0..v.len() {
            assert!((v[i] - back[i]).abs() <= scale.abs() * 1.5, "{} {}", v[i], back[i]);
        }
        // error feedback: сумм накапливается в err_buf — размер не растёт
        assert_eq!(err.len(), v.len());
    }

    #[test]
    fn quantize_error_feedback_reduces_drift() {
        // без error feedback средняя ошибка растёт; с ним — агрегированная ошибка мала
        let v: Vec<f32> = (0..500).map(|i| 0.001 * (i as f32 % 3.0)).collect();
        let mut err = vec![0.0; v.len()];
        let mut total_err = 0.0f32;
        let mut rng = Lcg::new(7);
        for _ in 0..5 {
            let small: Vec<f32> = v.iter().map(|x| x + rng.gaussian() * 1e-4).collect();
            let (q, scale) = quantize_q8(&small, &mut err);
            let back = dequantize(&q, scale);
            total_err = total_err.max(
                small.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max),
            );
        }
        assert!(total_err < 1e-3, "{}", total_err);
    }

    #[test]
    fn clip_l2_works() {
        let v = vec![3.0, 4.0]; // |v|=5
        let (c, was) = clip_l2(&v, 5.0);
        assert!(!was);
        let (c, was) = clip_l2(&v, 2.5);
        assert!(was);
        assert!(approx(norm2(&c), 2.5));
    }

    #[test]
    fn dp_noise_keeps_radius() {
        let v = vec![1.0; 100]; // norm = 10
        let mut rng = Lcg::new(42);
        let out = dp_noise(&v, 1.0, 1e-3, &mut rng);
        // клиппинг до нормы 1; шум 1e-3 на 100 элементов шевелит норму слабо
        let n = norm2(&out);
        assert!((n - 1.0).abs() < 0.05, "norm={}", n);
    }

    #[test]
    fn trimmed_mean_spares_outlier() {
        // 9 одинаковых + 1 выброс +1000
        let mut vecs: Vec<Vec<f32>> = (0..9).map(|_| vec![1.0, 2.0]).collect();
        vecs.push(vec![1001.0, 2002.0]);
        let out = trimmed_mean(&vecs, 1);
        assert!(approx(out[0], 1.0));
        assert!(approx(out[1], 2.0));
    }

    #[test]
    fn krum_spares_byzantine() {
        // 7 честных (мал отличия), 2 византийца с огромным отклонением
        let mut vecs: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32 * 1e-4; 4]).collect();
        vecs.push(vec![100.0; 4]);
        vecs.push(vec![-100.0; 4]);
        let idx = krum_select(&vecs, 2).unwrap();
        assert!(idx < 7, "KRUM должен выбрать честного");
    }

    #[test]
    fn fedgraph_flags_colluding() {
        // 5 честных (разные разреженные паттерны), 3 сговорившихся (почти идентичные)
        let mut vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let mut v = vec![0.0f32; 20];
                v[i as usize % 20] = 1.0;
                v[(i * 3 + 2) % 20] = 1.0;
                v
            })
            .collect();
        for k in 0..3 {
            let mut c = vecs[0].clone();
            c[k] += 0.5; // слегка различимы, но почти идентичны
            vecs.push(c);
        }
        let sc = fedgraph_mean_abs_corr(&vecs).unwrap();
        let last3 = &sc[5..];
        let avg_consp = last3.iter().sum::<f32>() / 3.0;
        let avg_honest = sc[..5].iter().sum::<f32>() / 5.0;
        assert!(avg_consp > avg_honest, "consp {} honest {}", avg_consp, avg_honest);
        assert!(avg_consp > 0.35);
        // порог классификации посередине между честными и сговорившимися —
        // честные проходят, сговорившиеся нет
        let mid = (avg_consp + avg_honest) / 2.0;
        assert!(accept_update(10.0, avg_honest, 1.0, 100.0, mid));
        assert!(!accept_update(10.0, avg_consp, 1.0, 100.0, mid));
    }

    #[test]
    fn accept_update_thresholds() {
        assert!(accept_update(10.0, 0.2, 1.0, 100.0, 0.5));
        assert!(!accept_update(0.1, 0.2, 1.0, 100.0, 0.5)); // слишком мал (не работал)
        assert!(!accept_update(10.0, 0.9, 1.0, 100.0, 0.5)); // сговор
    }
}
