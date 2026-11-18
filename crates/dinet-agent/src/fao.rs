//! FAO (Federated Agent Objects, M4): первоклассный федеративный объект —
//! «навык + подписанная награда». Локальный M2-обмен навыками шёл строкой
//! `SKILL_SHARE v1`, где счётчик `success` НЕ подписан — как «награда» для
//! федерации он непригоден (любой может приписать себе успехи).
//!
//! v1 добавляет:
//!   - `Reward` — ed25519-подпись evaluators'а над (skill_hash, successes,
//!     attempts, era); репутация = successes/attempts;
//!   - `FaoCard` — навык + награда + abi; обе подписи проверяются, hash
//!     связывает награду с конкретным навыком;
//!   - мост на уже существующий кадр `MsgType::SkillShare = 0x40`
//!     (`skill_hash`, `abi_version`, `body`) — сетевой код агента не меняется;
//!   - `FaoPool` — детерминированный отбор по репутации и ЭВОЛЮЦИЯ:
//!     кроссовер двух лучших навыков даёт новый подписанный навык с
//!     консервативной (min-по-родителям) подписанной наградой.
//!
//! Принцип FAO: федерируется абстрагированный навык и оценка, НЕ приватные
//! траектории. V1 не выполняет навыки — replay по-прежнему идёт только через
//! `ToolWorld::call` со всеми capability/анти-фишинг гейтами.

use crate::json::Json;
use crate::skills::{Skill, SkillStep, MAX_STEPS};
use dinet_crypto::{
    sha256, Ed25519Verifier, Signer, Verifier, ED25519_PUB_LEN, ED25519_SIG_LEN,
};
use dinet_proto::Payload;
use std::collections::BTreeMap;

/// Версия ABI карточки FAO (совпадает с `abi_version` кадра 0x40).
pub const FAO_ABI: u16 = 1;

/// Подписанная награда evaluators'а за конкретный навык.
#[derive(Debug, Clone, PartialEq)]
pub struct Reward {
    pub skill_hash: [u8; 32],
    pub successes: u64,
    pub attempts: u64,
    pub era: u64,
    pub evaluator: [u8; ED25519_PUB_LEN],
    pub sig: [u8; ED25519_SIG_LEN],
}

impl Reward {
    pub fn mint(
        signer: &dyn Signer,
        skill_hash: [u8; 32],
        successes: u64,
        attempts: u64,
        era: u64,
    ) -> Reward {
        assert!(attempts > 0, "награда без попыток");
        let mut r = Reward {
            skill_hash,
            successes: successes.min(attempts),
            attempts,
            era,
            evaluator: signer.pubkey(),
            sig: [0u8; ED25519_SIG_LEN],
        };
        r.sig = signer.sign(&r.canonical());
        r
    }

    /// Канон подписи: FAOREW1|hash|успехи|попытки|эра|evaluator.
    pub fn canonical(&self) -> Vec<u8> {
        format!(
            "FAOREW1|{}|{}|{}|{}|{}",
            hex(&self.skill_hash),
            self.successes,
            self.attempts,
            self.era,
            hex(&self.evaluator)
        )
        .into_bytes()
    }

    pub fn verify(&self) -> bool {
        self.attempts > 0
            && self.successes <= self.attempts
            && Ed25519Verifier::new(self.evaluator)
                .map(|v| v.verify(&self.canonical(), &self.sig))
                .unwrap_or(false)
    }

    /// Репутация: доля успешных проигрываний.
    pub fn rate(&self) -> f32 {
        self.successes as f32 / self.attempts as f32
    }
}

/// Карточка FAO: подписанный автором навык + подписанная evaluators'ом
/// награда, связанные hash канона навыка.
#[derive(Debug, Clone, PartialEq)]
pub struct FaoCard {
    pub skill: Skill,
    pub reward: Reward,
    pub abi: u16,
}

impl FaoCard {
    pub fn new(skill: Skill, reward: Reward) -> FaoCard {
        FaoCard { skill, reward, abi: FAO_ABI }
    }

    pub fn skill_hash(&self) -> [u8; 32] {
        sha256(&self.skill.canonical())
    }

    /// Все три проверки: авторская подпись навыка, подпись награды,
    /// совпадение hash награды с навыком и версия ABI.
    pub fn valid(&self) -> bool {
        self.abi == FAO_ABI
            && self.skill.verify()
            && self.reward.verify()
            && self.reward.skill_hash == self.skill_hash()
    }

    pub fn envelope(&self) -> String {
        self.to_json().to_string()
    }

    pub fn from_envelope(text: &str) -> Result<FaoCard, String> {
        let j = crate::json::parse(text).map_err(|e| format!("FAO конверт не JSON: {e}"))?;
        FaoCard::from_json(&j)
    }

    pub fn to_json(&self) -> Json {
        let skill = crate::json::parse(&self.skill.envelope())
            .expect("собственный конверт навыка всегда валиден");
        Json::Obj(vec![
            ("kind".into(), Json::Str("FAO_CARD".into())),
            ("v".into(), Json::Int(1)),
            ("abi".into(), Json::Int(self.abi as i64)),
            ("skill".into(), skill),
            ("hash".into(), Json::Str(hex(&self.skill_hash()))),
            ("reward".into(), Json::Obj(vec![
                ("skill_hash".into(), Json::Str(hex(&self.reward.skill_hash))),
                ("successes".into(), Json::Int(self.reward.successes as i64)),
                ("attempts".into(), Json::Int(self.reward.attempts as i64)),
                ("era".into(), Json::Int(self.reward.era as i64)),
                ("evaluator".into(), Json::Str(hex(&self.reward.evaluator))),
                ("sig".into(), Json::Str(hex(&self.reward.sig))),
            ])),
        ])
    }

    pub fn from_json(j: &Json) -> Result<FaoCard, String> {
        if j.get("kind").and_then(|x| x.as_str()) != Some("FAO_CARD") {
            return Err("FAO: kind не FAO_CARD".into());
        }
        if j.get("v") != Some(&Json::Int(1)) {
            return Err("FAO: v не 1".into());
        }
        let abi = match j.get("abi") {
            Some(Json::Int(n)) if *n >= 0 => *n as u16,
            _ => return Err("FAO: abi не целое".into()),
        };
        // входящий skill проходит строгий разбор + авторскую подпись
        let skill = Skill::from_json(
            j.get("skill").ok_or("FAO: нет skill")?,
        )?;
        let r = j.get("reward").ok_or("FAO: нет reward")?;
        let int = |k: &str| -> Result<u64, String> {
            match r.get(k) {
                Some(Json::Int(n)) if *n >= 0 => Ok(*n as u64),
                _ => Err(format!("FAO: reward.{k} не целое")),
            }
        };
        let reward = Reward {
            skill_hash: unhex::<32>(
                r.get("skill_hash").and_then(|x| x.as_str()).ok_or("FAO: нет reward.skill_hash")?,
            )?,
            successes: int("successes")?,
            attempts: int("attempts")?,
            era: int("era")?,
            evaluator: unhex::<ED25519_PUB_LEN>(
                r.get("evaluator").and_then(|x| x.as_str()).ok_or("FAO: нет evaluator")?,
            )?,
            sig: unhex::<ED25519_SIG_LEN>(
                r.get("sig").and_then(|x| x.as_str()).ok_or("FAO: нет sig")?,
            )?,
        };
        let card = FaoCard { skill, reward, abi };
        if !card.valid() {
            return Err("FAO: подпись/hash/ABI не бьётся — карточка отвергнута".into());
        }
        Ok(card)
    }

    /// Мост на существующий кадр протокола 0x40 (без нового wire-кода).
    pub fn to_payload(&self) -> Payload {
        Payload::SkillShare {
            skill_hash: self.skill_hash(),
            abi_version: self.abi,
            body: self.envelope().into_bytes(),
        }
    }

    pub fn from_payload(p: &Payload) -> Result<FaoCard, String> {
        match p {
            Payload::SkillShare { skill_hash, abi_version, body } => {
                let text = std::str::from_utf8(body).map_err(|_| "FAO: body не UTF-8".to_string())?;
                let card = FaoCard::from_envelope(text)?;
                if card.skill_hash() != *skill_hash {
                    return Err("FAO: skill_hash кадра не совпал с навыком".into());
                }
                if card.abi != *abi_version {
                    return Err("FAO: abi_version кадра не совпал".into());
                }
                Ok(card)
            }
            _ => Err("FAO: кадр не SkillShare".into()),
        }
    }
}

/// Пул карточек, собранных от разных агентов. Детерминирован: ключ — hash
/// навыка, порядок — sort по hash; замена только при лучшей репутации.
#[derive(Default)]
pub struct FaoPool {
    cards: BTreeMap<[u8; 32], FaoCard>,
}

impl FaoPool {
    pub fn new() -> FaoPool {
        FaoPool::default()
    }

    /// Принять карточку: полная проверка подписей; плохая — не попадает
    /// в пул. Возвращает true, если вставлена/заменена.
    pub fn observe(&mut self, card: FaoCard) -> Result<bool, String> {
        if !card.valid() {
            return Err("FAO: невалидная карточка (подделка?)".into());
        }
        let h = card.skill_hash();
        let replace = match self.cards.get(&h) {
            None => true,
            Some(old) => {
                card.reward.rate() > old.reward.rate()
                    || (card.reward.rate() == old.reward.rate()
                        && card.reward.attempts > old.reward.attempts)
            }
        };
        if replace {
            self.cards.insert(h, card);
        }
        Ok(replace)
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    pub fn cards(&self) -> Vec<FaoCard> {
        self.cards.values().cloned().collect()
    }

    /// Лучшие по репутации (детерминированный tie-break по hash).
    pub fn top(&self, n: usize, min_attempts: u64) -> Vec<FaoCard> {
        let mut v: Vec<FaoCard> = self
            .cards
            .values()
            .filter(|c| c.reward.attempts >= min_attempts)
            .cloned()
            .collect();
        v.sort_by(|a, b| {
            b.reward
                .rate()
                .total_cmp(&a.reward.rate())
                .then_with(|| a.skill_hash().cmp(&b.skill_hash()))
        });
        v.truncate(n);
        v
    }

    /// Эволюция: кроссовер шагов двух лучших навыков → новый навык,
    /// подписанный локальным агентом; награда потомка консервативна
    /// (минимум репутации родителей, минимум попыток) и тоже подписана.
    pub fn evolve(&self, signer: &dyn Signer, name: &str, era: u64, min_attempts: u64) -> Option<FaoCard> {
        let parents = self.top(2, min_attempts);
        if parents.len() < 2 {
            return None;
        }
        let mut steps: Vec<SkillStep> = Vec::new();
        for p in &parents {
            for s in &p.skill.steps {
                if steps.len() >= MAX_STEPS {
                    break;
                }
                if !steps.iter().any(|x| x.tool == s.tool && x.args.to_string() == s.args.to_string()) {
                    steps.push(s.clone());
                }
            }
        }
        let version = parents.iter().map(|p| p.skill.version).max().unwrap_or(1) + 1;
        let skill = Skill::mint(signer, name, version, steps);
        let attempts = parents.iter().map(|p| p.reward.attempts).min().unwrap_or(1);
        let rate = parents.iter().map(|p| p.reward.rate()).fold(f32::INFINITY, f32::min);
        let reward = Reward::mint(
            signer,
            sha256(&skill.canonical()),
            (rate * attempts as f32).round() as u64,
            attempts,
            era,
        );
        Some(FaoCard::new(skill, reward))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex<const N: usize>(s: &str) -> Result<[u8; N], String> {
    if s.len() != N * 2 {
        return Err(format!("FAO: hex поля ожидается {} символов", N * 2));
    }
    let mut out = [0u8; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| "FAO: плохой hex".to_string())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_crypto::Ed25519Signer;
    use dinet_proto::MsgType;

    fn step(tool: &str, text: &str) -> SkillStep {
        SkillStep {
            tool: tool.into(),
            args: Json::Obj(vec![("text".into(), Json::Str(text.into()))]),
        }
    }

    fn card(seed: u8, name: &str, version: u64, succ: u64, att: u64) -> FaoCard {
        let s = Ed25519Signer::from_seed(&[seed; ED25519_PUB_LEN]);
        let skill = Skill::mint(&s, name, version, vec![step("echo", name)]);
        let e = Ed25519Signer::from_seed(&[seed + 50; ED25519_PUB_LEN]);
        let h = sha256(&skill.canonical());
        FaoCard::new(skill, Reward::mint(&e, h, succ, att, 3))
    }

    #[test]
    fn card_signatures_and_tamper() {
        let c = card(1, "open-docs", 1, 7, 10);
        assert!(c.valid());
        let mut bad = c.clone();
        bad.skill.steps[0].args = Json::Obj(vec![("text".into(), Json::Str("evil".into()))]);
        assert!(!bad.valid(), "подмена шага ломает подпись автора");
        let mut bad = c.clone();
        bad.reward.successes = 10;
        assert!(!bad.valid(), "подмена награды ломает подпись evaluator'а");
        let mut bad = c.clone();
        bad.reward.skill_hash = [9u8; 32];
        assert!(!bad.valid(), "награда от другого навыка не принимается");
        assert!(FaoCard::from_envelope(&c.envelope()).is_ok());
    }

    #[test]
    fn wire_skillshare_roundtrip() {
        let c = card(2, "wire-skill", 1, 4, 5);
        let frame = dinet_proto::Frame {
            msg_type: MsgType::SkillShare,
            sender: [2u8; 32],
            sig: [0u8; 64],
            payload: c.to_payload(),
        };
        let bytes = frame.encode_with_sender_sig(&[2u8; 32], &[0u8; 64]).unwrap();
        let got = dinet_proto::Frame::decode(&bytes).unwrap();
        assert_eq!(got.msg_type, MsgType::SkillShare);
        assert_eq!(FaoCard::from_payload(&got.payload).unwrap(), c);
        // подмена hash кадра ломает мост
        let mut bad = c.to_payload();
        if let Payload::SkillShare { skill_hash, .. } = &mut bad {
            *skill_hash = [0u8; 32];
        }
        assert!(FaoCard::from_payload(&bad).is_err());
    }

    #[test]
    fn pool_selects_by_signed_reward_and_evolves() {
        let mut pool = FaoPool::new();
        for (seed, succ) in [(1u8, 2u64), (2, 5), (3, 8), (4, 9)] {
            let nm = format!("skill-{seed}"); // разные шаги — кроссовер даст >1 шага
            assert!(pool.observe(card(seed, &nm, 1, succ, 10)).unwrap());
        }
        assert_eq!(pool.len(), 4);
        assert_eq!(pool.top(2, 1).len(), 2);
        assert!(pool.top(3, 11).is_empty(), "мало попыток — в топ не попадает");
        let top = pool.top(2, 1);
        assert_eq!(top[0].reward.successes, 9);
        // повтор с МЕНЬШЕЙ наградой не заменяет запись
        assert!(!pool.observe(card(4, "skill-4", 1, 1, 10)).unwrap());
        assert_eq!(pool.top(1, 1)[0].reward.successes, 9);
        let who = Ed25519Signer::from_seed(&[77u8; ED25519_PUB_LEN]);
        let child = pool.evolve(&who, "skill-evolved", 4, 1).unwrap();
        assert!(child.valid());
        assert_eq!(child.skill.version, 2);
        assert!(child.skill.steps.len() >= 2, "кроссовер соединил шаги родителей");
        assert!(child.reward.rate() <= top[1].reward.rate() + f32::EPSILON);
    }

    #[test]
    fn unsigned_reward_rejected() {
        let mut c = card(5, "nope", 1, 10, 10);
        c.reward.sig = [0u8; ED25519_SIG_LEN];
        assert!(FaoPool::new().observe(c).is_err());
    }
}
