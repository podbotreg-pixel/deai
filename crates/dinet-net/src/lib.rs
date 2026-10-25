//! dinet-net — сетевой слой DI-NET.
//!
//! Скелет v0: перечисление пяти независимых bootstrap-источников, эпохи/раунды,
//! зонная логика выбора агрегатора по репутации. Полный libp2p-транспорт —
//! точка прикрепления v1 (см. docs/ROADMAP.md M1); крейт落的 и возвращает
//! готовые структуры, чтобы транспорт не менял семантику.

use dinet_proto::{Frame, MsgType, Payload};

/// DID узла в сетевом слое (32B ed25519-пубключ^multicodec-хэш).
pub type Did32 = [u8; 32];

/// Пять независимых источников входа в swarm (docs/RESILIENCE.md §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BootstrapSource {
    /// 1. накопленные адреса пиров с прошлой сессии (PEX)
    PeerExchange,
    /// 2. mDNS в локальной сети
    LocalMdns,
    /// 3. DHT (Kademlia, passive mode)
    Dht,
    /// 4. DNS-семена сообщества
    DnsSeeds,
    /// 5. вшитый в дистрибутив список (обновляемый через саму сеть)
    Bundled,
}

pub const ALL_SOURCES: [BootstrapSource; 5] = [
    BootstrapSource::PeerExchange,
    BootstrapSource::LocalMdns,
    BootstrapSource::Dht,
    BootstrapSource::DnsSeeds,
    BootstrapSource::Bundled,
];

/// Роль узла в раунде (все равноправны; роли эфемерные).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Trainer,
    Aggregator,
    Notary,
    Evaluator,
}

/// Окно эры: 1 час; era_id = SHA256(parent||bucket)
pub const ERA_WINDOW_SECONDS: u64 = 3600;
/// минимальное кол-во вкладчиков, ниже которого раунд не открывается
pub const MIN_CONTRIB: u16 = 50;
/// процент lazy-аудитов (Backdoor Absorption, arXiv:2607.06643)
pub const LAZY_AUDIT_PCT: u8 = 10;

pub fn era_id(parent_checkpoint: &[u8], unix_time: u64) -> [u8; 32] {
    let bucket = unix_time / ERA_WINDOW_SECONDS;
    let mut buf = Vec::with_capacity(parent_checkpoint.len() + 8);
    buf.extend_from_slice(parent_checkpoint);
    buf.extend_from_slice(&bucket.to_le_bytes());
    use dinet_crypto::sha256;
    sha256(&buf)
}

/// Локальная репутация (нет переноса между узлами; экспонениеция 30 дней).
#[derive(Debug, Clone, Default)]
pub struct Reputation {
    raw: f32,
    last_update_unix: u64,
    /// Взаимные наблюдения, window 30 дней
    observations: Vec<(u64, f32)>,
}

impl Reputation {
    pub fn observe(&mut self, unix_time: u64, good: bool) {
        let d = if good { 1.0 } else { -3.0 }; // плохое весит больше
        self.raw = (self.raw * 0.9 + d * 10.0).clamp(-100.0, 100.0);
        self.last_update_unix = unix_time;
        self.observations.push((unix_time, d));
    }

    pub fn score_at(&self, unix_time: u64) -> f32 {
        let age_days = (unix_time.saturating_sub(self.last_update_unix)) as f32 / 86_400.0;
        let decay = (-age_days / 30.0).exp();
        self.raw * decay
    }

    pub fn eligible_aggregator(&self, secs: u64, threshold: f32) -> bool {
        self.score_at(secs) >= threshold
    }
}

/// Выбор агрегатора: равноправная лотерея среди eligible (репутация ≥ порога).
pub fn pick_aggregator(
    candidates: &[(Did32, Reputation)],
    now: u64,
    threshold: f32,
    rng: &mut dyn FnMut() -> u64,
) -> Option<Did32> {
    let eligible: Vec<_> = candidates
        .iter()
        .filter(|(_, rep)| rep.eligible_aggregator(now, threshold))
        .map(|(id, _)| *id)
        .collect();
    if eligible.is_empty() {
        return None;
    }
    let i = (rng() % (eligible.len() as u64)) as usize;
    Some(eligible[i])
}

/// Что делать с входящим фрёем в текущем состоянии (первичная маршрутизация).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetDecision {
    /// проверить подпись, обработать
    Process,
    /// era устарела — лог-запись, отбросить
    StaleEra,
    /// подпись не проверилась — отбросить и понизить репутацию отправителя
    RejectUntrusted,
    /// не наш добрый peer (нет в списке?) — считаем по REP
    UnknownPeer,
}

pub fn route_incoming(
    our_era: &[u8; 32],
    frame_era: Option<&[u8]>,
    signature_ok: bool,
) -> NetDecision {
    if !signature_ok {
        return NetDecision::RejectUntrusted;
    }
    if let Some(e) = frame_era {
        if !era_equal(e, our_era) {
            return NetDecision::StaleEra;
        }
    }
    NetDecision::Process
}

fn era_equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Hello-кадр по заявленным ролям (caps = флаги наличия ролей).
#[must_use]
pub fn build_hello(roles: &[Role], port: u16) -> Frame {
    let caps: Caps4 = [
        roles.contains(&Role::Trainer),
        roles.contains(&Role::Aggregator),
        roles.contains(&Role::Notary),
        roles.contains(&Role::Evaluator),
    ];
    Frame {
        msg_type: MsgType::Hello,
        sender: [0; 32], // заполнит сетевой слой
        sig: [0; 64],    // заполнит Signer
        payload: Payload::Hello {
            proto_ver: dinet_proto::PROTO_VERSION as u16,
            caps,
            port,
        },
    }
}

pub type Caps4 = [bool; 4];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn era_bucket_changes_hourly() {
        let parent = [5u8; 32];
        let t0 = 1_800_000_000u64; // выровнен на границу часа (500000*3600)
        let a = era_id(&parent, t0);
        let b = era_id(&parent, t0 + 1800);
        let c = era_id(&parent, t0 + 3601);
        assert_eq!(a, b, "в одном часе — один era");
        assert_ne!(a, c, "новый час — новый era");
        assert_ne!(a, era_id(&[4u8; 32], t0), "другой parent — другой era");
    }

    #[test]
    fn reputation_penalized_bad_more_than_good() {
        let unix = 1_800_000_000;
        let mut good = Reputation::default();
        for _ in 0..10 {
            good.observe(unix, true);
        }
        let mut bad = Reputation::default();
        for _ in 0..10 {
            bad.observe(unix, false);
        }
        assert_eq!(
            good.score_at(unix) + bad.score_at(unix) < 0.0,
            true,
            "10 bad должны перевесить 10 good"
            );
    }

    #[test]
    fn aggregator_lottery_among_eligible() {
        let mut rep = Reputation::default();
        for _ in 0..5 {
            rep.observe(1_800_000_000, true);
        }
        let cands: Vec<(Did32, Reputation)> = vec![
            ([9u8; 32], rep.clone()),
            ([7u8; 32], Reputation::default()),
        ];
        let mut rng_seq: u64 = 0;
        let picked = pick_aggregator(&cands, 1_800_000_001, 1.0, &mut || {
            rng_seq += 1;
            rng_seq
        });
        assert!(picked.is_some());
        // репутация ниже порога: никто не eligible
        let no = pick_aggregator(&cands, 1_800_000_001, 999.0, &mut || 0);
        assert!(no.is_none());
    }

    #[test]
    fn route_by_signature() {
        assert_eq!(
            route_incoming(&[1; 32], None, false),
            NetDecision::RejectUntrusted
        );
        assert_eq!(route_incoming(&[1; 32], Some(&[1; 32]), true), NetDecision::Process);
        assert_eq!(route_incoming(&[1; 32], Some(&[2; 32]), true), NetDecision::StaleEra);
    }
}
