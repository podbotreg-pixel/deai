//! Secure Aggregation (pairwise masking, Bonawitz et al. 2017), демо-функционал v0.
//!
//! Математика: клиент i отправляет агрегатору share_i = θ_i + Σ_j σ_ij",
//! где σ_ij — псевдослучайная маска, общая с клиентом j с противоположным
//! знаком. При полном контуре Σ_i share_i = Σ_i θ_i: агрегатор получает
//! сумму обновлений и **математически не восстанавливает** ни одного по
//! отдельности.
//!
//! В v0 маска выводится из SHA-256-потока (Pseudo Random Generator по парному
//! секретному seed). В v1 пара secret организуется через X25519-EHE.

use dinet_crypto::Sha256;

/// Фикс-пойнт масштаб: f32 перевед candidacy честно и детерминированно.
pub const FX_SCALE: f64 = 65536.0; // 2^16

/// Канонический ключ пары: упорядочены по исходному ID.
pub fn canonical_key(a: usize, b: usize, _seed: &[u8]) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// PRG (детерминированный поток байт) от seed.
pub fn prg_stream(seed: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 32);
    let mut ctr: u32 = 0;
    while out.len() < n {
        let mut h = Sha256::new();
        h.update(seed);
        h.update(&ctr.to_be_bytes());
        out.extend_from_slice(&h.finalize());
        ctr += 1;
    }
    out.truncate(n);
    out
}

/// Маска для пары (i,j) при позиции d — i64 по потоку.
fn mask_element(seed_ij: &[u8], d: usize) -> i64 {
    const B: usize = 8;
    let stream = prg_stream(seed_ij, (d + 1) * B);
    let mut b = [0u8; B];
    b.copy_from_slice(&stream[d * B..(d + 1) * B]);
    i64::from_be_bytes(b)
}

/// f32 -> fixedpoint i64 (clamped), детерминировано.
pub fn to_fx(x: f32) -> i64 {
    let v = (x as f64 * FX_SCALE).round() as i64;
    v.clamp(-(1i64 << 52), 1i64 << 52)
}

pub fn from_fx(v: i64) -> f32 {
    (v as f64 / FX_SCALE) as f32
}

/// Доля, которую шлёт один клиент по всем компонентам апдейта:
/// share[d] = θ[d] + Σ_{j<i_case_sign} sign(i,j) * mask_{ij,d}
/// Sign консистентен с каноническим ключом: min-сторона добавляет маску,
/// max-сторона вычитает ту же величину.
pub fn client_share(
    my_id: usize,
    my_seed: &[u8],
    registry: &[(usize, usize, [u8; 16])], // (i, j, seed_ij), i<j
    update_fx: &[i64],
) -> Vec<i64> {
    let mut share: Vec<i64> = update_fx.to_vec();
    for &(i, j, ref seed_ij) in registry {
        if i != my_id && j != my_id {
            continue;
        }
        let sign: i64 = if my_id == i { 1 } else { -1 };
        for (d, s) in share.iter_mut().enumerate() {
            *s = s.wrapping_add(sign.wrapping_mul(mask_element(seed_ij, d)));
        }
        let _ = my_seed; // рынок v1: derive X25519 shared secret из этого seed
    }
    share
}

/// Агрегатор: складывает пришедшие shares. При контуре ≥0 пар из fleet
/// сумма масок обнуляется, остаётся сумма θ.
pub fn aggregate(shares: &[Vec<i64>]) -> Vec<i64> {
    if shares.is_empty() {
        return vec![];
    }
    let mut out = vec![0i64; shares[0].len()];
    for s in shares {
        for (o, x) in out.iter_mut().zip(s.iter()) {
            *o = o.wrapping_add(*x);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_for(n: usize, seeds: &[[u8; 16]]) -> Vec<(usize, usize, [u8; 16])> {
        let mut reg = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                reg.push((i, j, seeds[i]));
                // v0: seed пары = seed минимального участника (детерминированно);
                // v1: сеанс уприти Derived through DX25519 handshake
                let _ = j;
            }
        }
        reg
    }

    #[test]
    fn masking_cancels_exactly() {
        let n = 5;
        let seeds: Vec<[u8; 16]> = (0..n).map(|i| [i as u8; 16]).collect();
        let reg = registry_for(n, &seeds);
        let updates: Vec<Vec<i64>> = (0..n).map(|i| vec![100 + 10i64 * i as i64; 4]).collect();
        let shares: Vec<Vec<i64>> = (0..n)
            .map(|i| client_share(i, &seeds[i], &reg, &updates[i]))
            .collect();
        // агрегатор видит только сумму — точную
        let total = aggregate(&shares);
        let expect: Vec<i64> = (0..4)
            .map(|d| updates.iter().map(|u| u[d]).sum())
            .collect();
        assert_eq!(total, expect, "маски должны сократиться точно");
    }

    #[test]
    fn individual_share_is_masked() {
        let seeds: Vec<[u8; 16]> = vec![[1u8; 16], [2; 16]];
        let reg = registry_for(2, &seeds);
        let upd = vec![0i64; 8]; // нулевой апдейт клиента
        let share = client_share(0, &seeds[0], &reg, &upd);
        // share ≠ 0 (каждая component скрыта), но сумма по двум клиентам == 0+0
        assert!(share.iter().any(|&x| x != 0));
        let share2 = client_share(1, &seeds[1], &reg, &upd);
        assert_eq!(aggregate(&[share, share2]), vec![0i64; 8]);
    }

    #[test]
    fn fx_roundtrip() {
        for v in [-3.7f32, 0.0, 1e-4, 9999.5] {
            assert!((from_fx(to_fx(v)) - v).abs() < 1e-3);
        }
    }
}
