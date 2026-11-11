//! Evolving Skills (M2-агент): накопление успешных последовательностей
//! вызовов инструментов, воспроизведение (replay через ТЕ ЖЕ гейты —
//! анти-фишинг и уровень capability применяются к каждому шагу), обмен
//! SKILL_SHARE v1 — самодостаточный подписанный автором конверт.
//! Навык = имя + версия + шаги; счётчик успешных проигрываний ведётся
//! вне подписи (статистика растёт без перевыпуска).

use crate::json::Json;
use dinet_crypto::{Signer, Verifier, ED25519_PUB_LEN, ED25519_SIG_LEN};
use dinet_memory::KvStore;
use std::path::{Path, PathBuf};

/// Лимит шагов навыка (защита от гигантских/зацикленных).
pub const MAX_STEPS: usize = 32;
const MAX_NAME: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct SkillStep {
    pub tool: String,
    pub args: Json,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Skill {
    pub name: String,
    pub version: u64,
    pub steps: Vec<SkillStep>,
    /// Ключ автора (DID): подпись канон-байтов его ключом.
    pub author: [u8; ED25519_PUB_LEN],
    pub sig: [u8; ED25519_SIG_LEN],
    /// Успешные проигрывания (не подписывается).
    pub success: u64,
}

impl Skill {
    /// Авторская сборка: подпись канон-байтов ключом автора.
    pub fn mint(signer: &dyn Signer, name: &str, version: u64, steps: Vec<SkillStep>) -> Skill {
        let mut sk = Skill {
            name: name.into(),
            version,
            steps,
            author: signer.pubkey(),
            sig: [0u8; ED25519_SIG_LEN],
            success: 0,
        };
        sk.sig = signer.sign(&sk.canonical());
        sk
    }

    /// Байты подписи: SKILL1|name|version|tool|args|tool|args…|author.
    /// success/last-статистика НЕ входят — растут без перевыпуска.
    pub fn canonical(&self) -> Vec<u8> {
        let mut v = format!("SKILL1|{}|{}", self.name, self.version);
        for s in &self.steps {
            v.push('|');
            v.push_str(&s.tool);
            v.push('|');
            v.push_str(&s.args.to_string());
        }
        v.push('|');
        v.push_str(&hex(&self.author));
        v.into_bytes()
    }

    /// Подпись автора бьётся по канон-байтам?
    pub fn verify(&self) -> bool {
        match dinet_crypto::Ed25519Verifier::new(self.author) {
            Some(v) => v.verify(&self.canonical(), &self.sig),
            None => false,
        }
    }

    /// SKILL_SHARE v1: одна строка JSON (для NDJSON/файла/чата).
    pub fn envelope(&self) -> String {
        self.to_json().to_string()
    }

    /// Разобрать конверт и проверить подпись автора.
    pub fn from_envelope(text: &str) -> Result<Skill, String> {
        let j = crate::json::parse(text).map_err(|e| format!("конверт не JSON: {e}"))?;
        let sk = Skill::from_json(&j)?;
        if !sk.verify() {
            return Err("SKILL_SHARE: подпись автора не бьётся (подделка?)".into());
        }
        Ok(sk)
    }

    fn to_json(&self) -> Json {
        let steps = Json::Arr(
            self.steps
                .iter()
                .map(|s| {
                    Json::Obj(vec![
                        ("tool".into(), Json::Str(s.tool.clone())),
                        ("args".into(), s.args.clone()),
                    ])
                })
                .collect(),
        );
        Json::Obj(vec![
            ("kind".into(), Json::Str("SKILL_SHARE".into())),
            ("v".into(), Json::Int(1)),
            ("name".into(), Json::Str(self.name.clone())),
            ("version".into(), Json::Int(self.version as i64)),
            ("steps".into(), steps),
            ("author".into(), Json::Str(hex(&self.author))),
            ("sig".into(), Json::Str(hex(&self.sig))),
            ("success".into(), Json::Int(self.success as i64)),
        ])
    }

    /// Строгий разбор (все поля обязательны и валидны).
    pub fn from_json(j: &Json) -> Result<Skill, String> {
        if j.get("kind").and_then(|k| k.as_str()) != Some("SKILL_SHARE") {
            return Err("kind не SKILL_SHARE".into());
        }
        if j.get("v") != Some(&Json::Int(1)) {
            return Err("v не 1".into());
        }
        let name = j
            .get("name")
            .and_then(|k| k.as_str())
            .ok_or("нет name")?;
        validate_name(name)?;
        let version = match j.get("version") {
            Some(Json::Int(n)) if *n >= 1 => *n as u64,
            _ => return Err("version не целое ≥1".into()),
        };
        let mut steps = Vec::new();
        match j.get("steps") {
            Some(Json::Arr(items)) if !items.is_empty() => {
                for it in items {
                    let tool = it
                        .get("tool")
                        .and_then(|k| k.as_str())
                        .ok_or("шаг без tool")?
                        .to_string();
                    let args = it.get("args").cloned().unwrap_or(Json::Obj(Vec::new()));
                    if !matches!(args, Json::Obj(_)) {
                        return Err(format!("args шага {tool:?} не объект"));
                    }
                    steps.push(SkillStep { tool, args });
                }
            }
            _ => return Err("steps не непустой массив".into()),
        }
        if steps.len() > MAX_STEPS {
            return Err(format!("шагов {} > {MAX_STEPS}", steps.len()));
        }
        let author = hex_field(
            j.get("author").and_then(|k| k.as_str()).ok_or("нет author")?,
            ED25519_PUB_LEN,
            "author",
        )?;
        let sig = hex_field(
            j.get("sig").and_then(|k| k.as_str()).ok_or("нет sig")?,
            ED25519_SIG_LEN,
            "sig",
        )?;
        let success = match j.get("success") {
            Some(Json::Int(n)) if *n >= 0 => *n as u64,
            _ => return Err("success не целое ≥0".into()),
        };
        let mut author_a = [0u8; ED25519_PUB_LEN];
        author_a.copy_from_slice(&author);
        let mut sig_a = [0u8; ED25519_SIG_LEN];
        sig_a.copy_from_slice(&sig);
        Ok(Skill { name: name.to_string(), version, steps, author: author_a, sig: sig_a, success })
    }
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_NAME {
        return Err(format!("имя навыка пустое или длиннее {MAX_NAME}"));
    }
    if name.chars().any(|c| c == '\n' || c == '\0' || (c as u32) < 0x20) {
        return Err("имя навыка с control-символом".into());
    }
    Ok(())
}

pub fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

pub fn hex_field(t: &str, n: usize, what: &str) -> Result<Vec<u8>, String> {
    let b = t.as_bytes();
    if b.len() != n * 2 {
        return Err(format!("{what}: {} hex-символов, нужно {}", b.len(), n * 2));
    }
    (0..n)
        .map(|i| u8::from_str_radix(&t[i * 2..i * 2 + 2], 16).map_err(|_| format!("{what}: не hex")))
        .collect()
}

/// Хранилище навыков (append-only KV; ключ s:<имя>).
pub struct SkillStore {
    kv: KvStore,
    path: PathBuf,
}

impl SkillStore {
    pub fn open(path: &Path) -> Result<SkillStore, String> {
        let kv = KvStore::open(path).map_err(|e| format!("навыки {path:?}: {e}"))?;
        Ok(SkillStore { kv, path: path.to_path_buf() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Сохранить навык. Правило версий: та же версия или старее — НЕ
    /// заменяет (Ok(false)); новее — заменяет (Ok(true)).
    pub fn put(&mut self, skill: &Skill) -> Result<bool, String> {
        if !skill.verify() {
            return Err("навык не проходит проверку подписи".into());
        }
        validate_name(&skill.name)?;
        if let Some(old) = self.get(&skill.name) {
            if skill.version <= old.version {
                return Ok(false);
            }
        }
        let key = format!("s:{}", skill.name);
        self.kv
            .put(&key, skill.to_json().to_string().as_bytes())
            .map_err(|e| format!("навыки: {e}"))?;
        Ok(true)
    }

    pub fn get(&self, name: &str) -> Option<Skill> {
        let v = self.kv.get(&format!("s:{name}"))?;
        let j = crate::json::parse(&String::from_utf8_lossy(&v)).ok()?;
        Skill::from_json(&j).ok()
    }

    /// Имена навыков (по алфавиту).
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .kv
            .scan("s:")
            .iter()
            .map(|k| k.trim_start_matches("s:").to_string())
            .collect();
        v.sort();
        v
    }

    /// Успешное проигрывание: счётчик растёт (вне подписи).
    pub fn record_success(&mut self, name: &str) -> Result<u64, String> {
        let mut sk = self.get(name).ok_or_else(|| format!("навык {name:?} не найден"))?;
        sk.success += 1;
        self.kv
            .put(&format!("s:{name}"), sk.to_json().to_string().as_bytes())
            .map_err(|e| format!("навыки: {e}"))?;
        Ok(sk.success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_crypto::Ed25519Signer;

    fn signer() -> Ed25519Signer {
        Ed25519Signer::from_seed(&[42u8; 32])
    }

    fn steps() -> Vec<SkillStep> {
        vec![
            SkillStep { tool: "fs_write".into(), args: crate::json::parse(r#"{"path":"a.txt","content":"v"}"#).unwrap() },
            SkillStep { tool: "fs_read".into(), args: crate::json::parse(r#"{"path":"a.txt"}"#).unwrap() },
        ]
    }

    #[test]
    fn mint_verify_and_tamper() {
        let sk = Skill::mint(&signer(), "заметка", 1, steps());
        assert!(sk.verify(), "свой навык валиден");
        assert_eq!(sk.author, signer().pubkey());
        // подмена шага ломает подпись
        let mut bad = sk.clone();
        bad.steps[0].args = crate::json::parse(r#"{"path":"a.txt","content":"зло"}"#).unwrap();
        assert!(!bad.verify(), "подмена шага ловится");
        // подмена автора ломает
        let mut bad2 = sk.clone();
        bad2.author = [9u8; ED25519_PUB_LEN];
        assert!(!bad2.verify());
        // подмена имени ломает
        let mut bad3 = sk.clone();
        bad3.name = "другое".into();
        assert!(!bad3.verify());
        // счётчик успехов вне подписи: рост не ломает verify
        let mut ok = sk.clone();
        ok.success = 7;
        assert!(ok.verify(), "статистика не подписана");
    }

    #[test]
    fn envelope_roundtrip_and_tamper() {
        let sk = Skill::mint(&signer(), "заметка", 2, steps());
        let env = sk.envelope();
        let back = Skill::from_envelope(&env).expect("конверт разбирается");
        assert_eq!(back, sk, "бит-в-бит (включая счётчик)");
        // подделка имени в конверте бракуется
        let evil = env.replace("заметка", "заметкаX");
        assert!(Skill::from_envelope(&evil).is_err(), "подделка ловится");
        // подделка шага
        let evil2 = env.replace("\"version\":2", "\"version\":1");
        assert!(Skill::from_envelope(&evil2).is_err());
        // не-конверт
        assert!(Skill::from_envelope("{}").is_err());
        assert!(Skill::from_envelope("мусор").is_err());
    }

    #[test]
    fn json_strictness() {
        let sk = Skill::mint(&signer(), "имя", 1, steps());
        let j = crate::json::parse(&sk.envelope()).unwrap();
        let Json::Obj(fields) = j else { panic!("конверт — объект") };
        // без success → брак
        let no_success =
            Json::Obj(fields.iter().filter(|(k, _)| k.as_str() != "success").cloned().collect());
        assert!(Skill::from_json(&no_success).is_err());
        // пустой steps → брак
        let no_steps = Json::Obj(
            fields
                .iter()
                .map(|(k, v)| {
                    if k == "steps" {
                        (k.clone(), Json::Arr(Vec::new()))
                    } else {
                        (k.clone(), v.clone())
                    }
                })
                .collect(),
        );
        assert!(Skill::from_json(&no_steps).is_err(), "пустой steps бракуется");
    }

    fn tmp_store(tag: &str) -> (SkillStore, PathBuf) {
        let p = std::env::temp_dir().join(format!("dinet-skill-{tag}-{}.roi", std::process::id()));
        let _ = std::fs::remove_file(&p);
        (SkillStore::open(&p).expect("store"), p)
    }

    #[test]
    fn store_versions_and_success() {
        let (mut st, p) = tmp_store("ver");
        let s = signer();
        let v1 = Skill::mint(&s, "заметка", 1, steps());
        assert!(st.put(&v1).expect("put v1"), "первый ставится");
        assert!(!st.put(&v1).expect("put v1 dup"), "та же версия не заменяет");
        let v2 = Skill::mint(&s, "заметка", 2, steps());
        assert!(st.put(&v2).expect("put v2"), "новее заменяет");
        assert!(!st.put(&v1).expect("put v1 low"), "старее не заменяет");
        assert_eq!(st.get("заметка").unwrap().version, 2);
        // счётчик успехов растёт и подпись жива
        assert_eq!(st.record_success("заметка").expect("rec"), 1);
        let back = st.get("заметка").expect("back");
        assert_eq!(back.success, 1);
        assert!(back.verify(), "подпись жива после роста счётчика");
        assert!(st.record_success("нет-такого").is_err());
        // имена
        let other = Skill::mint(&s, "ааа", 1, steps());
        st.put(&other).expect("put other");
        assert_eq!(st.names(), vec!["ааа".to_string(), "заметка".to_string()]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn store_survives_reopen() {
        let (mut st, p) = tmp_store("reopen");
        let s = signer();
        st.put(&Skill::mint(&s, "живучий", 1, steps())).expect("put");
        drop(st);
        let st2 = SkillStore::open(&p).expect("reopen");
        let back = st2.get("живучий").expect("навык пережил рестарт");
        assert_eq!(back.version, 1);
        assert!(back.verify());
        let _ = std::fs::remove_file(&p);
    }
}
