//! Secure Aggregation (pairwise masking, Bonawitz et al. 2017).
//!
//! Математика: клиент i отправляет share_i = θ_i + Σ_j sign(i,j)·σ_ij, где
//! σ_ij — маска пары (i,j). σ_ij выводится КАЖДОЙ стороной из ОДНОГО
//! диффи-хелмановского секрета пары: secret_ij = X25519(sk_i, pk_j)
//! (симметрично: sk_i·pk_j == sk_j·pk_i), значит i добавляет +σ, j — −σ, и
//! при полном контуре Σ_i share_i = Σ_i θ_i: агрегатор получает сумму
//! обновлений и **математически не восстанавливает** ни одного по
//! отдельности, не зная ни одного личного секрета.
//!
//! v1 (текущий): парный секрет — РЕАЛЬНЫЙ X25519 (dinet-crypto).
//! Отказ одного клиента >1 пары = потери контура: v2 добавит
//! secret-sharing полосных масок (Bonawitz §4.2).

use dinet_crypto::{x_pair_secret, Sha256};

/// Фикс-пойнт масштаб: f32 → i64 честно и детерминированно.
pub const FX_SCALE: f64 = 65536.0; // 2^16

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

/// f32 → fixed point i64 (клампится), детерминировано.
pub fn to_fx(x: f32) -> i64 {
    let v = (x as f64 * FX_SCALE).round() as i64;
    v.clamp(-(1i64 << 52), 1i64 << 52)
}

pub fn from_fx(v: i64) -> f32 {
    (v as f64 / FX_SCALE) as f32
}

/// Доля, которую шлёт клиент по всем компонентам апдейта:
/// share[d] = θ[d] + Σ_{j: (my,j) каноникализирован} sign(my, j) · mask_dj
/// где mask_dj = BE-байты PRG-потока ДИФФИ-ХЕЛМАНОВСКОГО секрета пары.
/// Знак каноничен по id: min-сторона +σ, max-сторона −σ (одно и то же σ).
///
/// Поток — ОДИН на пару (dim·8 байт): O(n·dim) вместо O(n²·dim).
pub fn client_share(
    my_id: usize,
    x_sk: &[u8; 32],                       // мой X25519-секрет
    peers: &[(usize, [u8; 32])],           // (id, X-pubkey) всех участников роя
    update_fx: &[i64],
) -> Vec<i64> {
    let dim = update_fx.len();
    let mut share: Vec<i64> = update_fx.to_vec();
    for &(j, pk_j) in peers {
        if j == my_id {
            continue;
        }
        // СЕКРЕТ ПАРЫ общий у обеих сторон: X25519(sk_i, pk_j) == X25519(sk_j, pk_i),
        // поэтому поток маски — ОДИН и тот же; знак каноничен по id
        let secret = x_pair_secret(x_sk, &pk_j);
        let s = prg_stream(&secret, dim * 8);
        let sign: i64 = if my_id < j { 1 } else { -1 };
        for (d, sh) in share.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&s[d * 8..(d + 1) * 8]);
            *sh = sh.wrapping_add(sign.wrapping_mul(i64::from_be_bytes(b)));
        }
    }
    share
}

/// Агрегатор: складывает пришедшие shares. Полный контур отменяет
/// маскировку — остаётся сумма θ_i.
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
    use dinet_crypto::{x25519_from_seed, X25519Key};

    /// X-ключи и список (id, pubkey) в духе узлов роя sim.
    fn keys(n: usize) -> (Vec<X25519Key>, Vec<(usize, [u8; 32])>) {
        let ks: Vec<X25519Key> =
            (0..n).map(|i| x25519_from_seed(&[i as u8; 32])).collect();
        let peers: Vec<(usize, [u8; 32])> =
            ks.iter().enumerate().map(|(i, k)| (i, k.pk)).collect();
        (ks, peers)
    }

    #[test]
    fn masking_cancels_exactly_with_real_x25519() {
        let n = 5;
        let (ks, peers) = keys(n);
        let updates: Vec<Vec<i64>> =
            (0..n).map(|i| vec![100 + 10 * i as i64; 4]).collect();
        let shares: Vec<Vec<i64>> = (0..n)
            .map(|i| client_share(i, &ks[i].sk, &peers, &updates[i]))
            .collect();
        let total = aggregate(&shares);
        let expect: Vec<i64> =
            (0..4).map(|d| updates.iter().map(|u| u[d]).sum()).collect();
        assert_eq!(total, expect, "X25519-маски обязаны схлопнуться точно");
    }

    #[test]
    fn individual_share_is_masked() {
        let (ks, peers) = keys(2);
        let upd = vec![0i64; 8]; // нулевой апдейт клиента
        let share = client_share(0, &ks[0].sk, &peers, &upd);
        assert!(share.iter().any(|&x| x != 0), "нулевой апдейт скрыт маской");
        let share2 = client_share(1, &ks[1].sk, &peers, &upd);
        assert_eq!(aggregate(&[share, share2]), vec![0i64; 8]);
    }

    #[test]
    fn missing_peer_does_not_reveal_update() {
        // маска пары (0,1): если клиент 1 умер, share_0 не сводим к θ_0 —
        // агрегатор видит только θ_0 + σ_01 (σ_01 неизвестен)
        let (ks, peers) = keys(2);
        let upd = vec![11i64; 4];
        let share = client_share(0, &ks[0].sk, &peers, &upd);
        assert_ne!(share, upd, "без второй половины апдейт не восстановим");
    }

    #[test]
    fn masks_collapse_across_three_pairs() {
        // полный контур из ТРЁХ (пары (0,1),(0,2),(1,2)) → отмена
        let (ks, peers) = keys(3);
        let zero = vec![0i64; 2];
        let s0 = client_share(0, &ks[0].sk, &peers, &zero);
        let s1 = client_share(1, &ks[1].sk, &peers, &zero);
        let s2 = client_share(2, &ks[2].sk, &peers, &zero);
        let total = aggregate(&[s0, s1, s2]);
        assert_eq!(total, vec![0i64; 2]);
    }

    #[test]
    fn fx_roundtrip() {
        for v in [-3.7f32, 0.0, 1e-4, 9999.5] {
            assert!((from_fx(to_fx(v)) - v).abs() < 1e-3);
        }
    }
}
