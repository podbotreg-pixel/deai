//! Capability-токены агента (M2): уровни READ/WRITE/DESTRUCTIVE/NETWORK,
//! scope-пути и срок жизни; подпись ed25519 ключом владельца — токен
//! верифицируется БЕЗ таблицы выданных (ключ → проверка подписи), подделка
//! и перенос между ключами бракуются.

use dinet_crypto::{Signer, Verifier, ED25519_PUB_LEN, ED25519_SIG_LEN};

/// Уровень доступа. Порядок — от мягкого к жёсткому: токен уровня X
/// покрывает инструменты, требующие <= X (Write покрывает Read).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CapLevel {
    Read = 0,
    Write = 1,
    Destructive = 2,
    Network = 3,
}

impl CapLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            CapLevel::Read => "READ",
            CapLevel::Write => "WRITE",
            CapLevel::Destructive => "DESTRUCTIVE",
            CapLevel::Network => "NETWORK",
        }
    }

    pub fn parse(s: &str) -> Option<CapLevel> {
        match s {
            "READ" => Some(CapLevel::Read),
            "WRITE" => Some(CapLevel::Write),
            "DESTRUCTIVE" => Some(CapLevel::Destructive),
            "NETWORK" => Some(CapLevel::Network),
            _ => None,
        }
    }
}

/// Разрешения: уровень + scope-пути FS (пустой = весь джейл).
#[derive(Debug, Clone, PartialEq)]
pub struct CapSet {
    pub level: CapLevel,
    /// Пути ФС (относительно джейла), куда токен пускает; пусто = весь джейл.
    pub scopes: Vec<String>,
    /// Unix-время истечения (0 = бессрочно).
    pub expires: u64,
}

impl CapSet {
    pub fn new(level: CapLevel) -> Self {
        CapSet { level, scopes: Vec::new(), expires: 0 }
    }

    /// Байты, которые подписываются (канон: уровень|срок|scope'ы через ';').
    pub fn canonical(&self) -> Vec<u8> {
        let mut s = format!("{}|{}", self.level.as_str(), self.expires);
        for sc in &self.scopes {
            s.push('|');
            s.push_str(sc);
        }
        s.into_bytes()
    }
}

/// Подписанный токен: CapSet + ключ владельца + подпись.
#[derive(Debug, Clone, PartialEq)]
pub struct CapToken {
    pub set: CapSet,
    pub pubkey: [u8; ED25519_PUB_LEN],
    pub sig: [u8; ED25519_SIG_LEN],
}

impl CapToken {
    pub fn mint(signer: &dyn Signer, set: &CapSet) -> CapToken {
        let sig = signer.sign(&set.canonical());
        CapToken { set: set.clone(), pubkey: signer.pubkey(), sig }
    }

    /// Подпись валидна и срок не истёк (now — unix-сек; expires 0 = бессрочно)?
    pub fn verify(&self, now: u64) -> bool {
        if self.set.expires != 0 && now > self.set.expires {
            return false;
        }
        match dinet_crypto::Ed25519Verifier::new(self.pubkey) {
            Some(v) => v.verify(&self.set.canonical(), &self.sig),
            None => false,
        }
    }

    /// Компактный переносимый формат: LEVEL|expires|scopes(';')|pubkey|sig.
    pub fn encode(&self) -> String {
        let mut s = format!("{}|{}|", self.set.level.as_str(), self.set.expires);
        s.push_str(&self.set.scopes.join(";"));
        s.push('|');
        for b in self.pubkey.iter() {
            s.push_str(&format!("{b:02x}"));
        }
        s.push('|');
        for b in self.sig.iter() {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    pub fn decode(text: &str) -> Result<CapToken, String> {
        let parts: Vec<&str> = text.trim().split('|').collect();
        if parts.len() != 5 {
            return Err(format!("формат LEVEL|expires|scopes|pubkey|sig: 5 полей, пришло {}", parts.len()));
        }
        let level = CapLevel::parse(parts[0]).ok_or("уровень не READ/WRITE/DESTRUCTIVE/NETWORK")?;
        let expires: u64 = parts[1].parse().map_err(|_| "срок не u64")?;
        let scopes = if parts[2].is_empty() {
            Vec::new()
        } else {
            parts[2].split(';').map(|s| s.to_string()).collect()
        };
        let pubkey_v = hex_field(parts[3], ED25519_PUB_LEN, "pubkey")?;
        let sig_v = hex_field(parts[4], ED25519_SIG_LEN, "sig")?;
        let mut pubkey = [0u8; ED25519_PUB_LEN];
        pubkey.copy_from_slice(&pubkey_v);
        let mut sig = [0u8; ED25519_SIG_LEN];
        sig.copy_from_slice(&sig_v);
        Ok(CapToken { set: CapSet { level, scopes, expires }, pubkey, sig })
    }
}

/// hex-строка фиксированной длины → байты.
fn hex_field(t: &str, n: usize, what: &str) -> Result<Vec<u8>, String> {
    let b = t.as_bytes();
    if b.len() != n * 2 {
        return Err(format!("{what}: {} hex-символов, нужно {}", b.len(), n * 2));
    }
    (0..n)
        .map(|i| {
            u8::from_str_radix(&t[i * 2..i * 2 + 2], 16).map_err(|_| format!("{what}: не hex"))
        })
        .collect()
}

/// Тесты уровня токенов: подпись, порча, срок.
#[cfg(test)]
mod tests {
    use super::*;
    use dinet_crypto::Ed25519Signer;

    fn signer() -> Ed25519Signer {
        Ed25519Signer::from_seed(&[7u8; 32])
    }

    #[test]
    fn mint_verify_roundtrip() {
        let s = signer();
        let mut set = CapSet::new(CapLevel::Write);
        set.scopes = vec!["docs".into()];
        let tok = CapToken::mint(&s, &set);
        assert!(tok.verify(1_000_000), "свой токен валиден");
        assert_eq!(tok.set.level, CapLevel::Write);
    }

    #[test]
    fn tampered_payload_rejected() {
        let s = signer();
        let tok = CapToken::mint(&s, &CapSet::new(CapLevel::Read));
        let mut bad = tok.clone();
        bad.set.level = CapLevel::Network; // самозванка: подпись больше не бьёт
        assert!(!bad.verify(0), "подмена уровня ломает подпись");
    }

    #[test]
    fn foreign_key_rejected() {
        let s = signer();
        let tok = CapToken::mint(&s, &CapSet::new(CapLevel::Read));
        let mut bad = tok.clone();
        bad.pubkey = [9u8; ED25519_PUB_LEN];
        assert!(!bad.verify(0), "чужой ключ не подтверждает подпись");
    }

    #[test]
    fn expiry_respected() {
        let s = signer();
        let mut set = CapSet::new(CapLevel::Read);
        set.expires = 100;
        let tok = CapToken::mint(&s, &set);
        assert!(tok.verify(99), "до срока валиден");
        assert!(!tok.verify(101), "после срока бракуется");
    }

    #[test]
    fn level_ordering() {
        assert!(CapLevel::Write > CapLevel::Read);
        assert!(CapLevel::Destructive > CapLevel::Write);
        assert!(CapLevel::Network > CapLevel::Destructive);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let s = signer();
        let mut set = CapSet::new(CapLevel::Destructive);
        set.scopes = vec!["docs/a".into(), "docs/b".into()];
        set.expires = 4_242_424;
        let tok = CapToken::mint(&s, &set);
        let back = CapToken::decode(&tok.encode()).expect("decode");
        assert_eq!(back, tok, "бит-в-бит");
        assert!(back.verify(4_242_424));
        assert!(!back.verify(4_242_425), "срок соблюдён и после переноса");
    }

    #[test]
    fn decode_garbage_rejected() {
        for bad in ["", "READ|", "READ|0||", "READ|0||aaa|bbb", "WRONG|0||", "READ|x||"] {
            assert!(CapToken::decode(bad).is_err(), "{bad:?} должен браковаться");
        }
    }
}
