//! ChatSession — сессия чата над локальной памятью (один KV-файл).
//!
//! Пространства ключей одного файла dinet-memory:
//! - `chat:turn:<idx:020>` — ходы: role(1B) | ts(u64 BE) | text
//! - `learn:pair:<idx:020>` — пары обучения: ts(u64 BE) | a_len(u32 LE) | a | b_len(u32 LE) | b
//! - `rag:*` — документы памяти (RAG от dinet-memory)
//!
//! Ход `turn()`: user-текст → RAG-контекст top-k → модель → ответ; затем
//! persist выбора: ходы ×2, документ памяти ×2 (оба текста), пара обучения ×1.

use std::path::Path;

use dinet_memory::kv::KvStore;
use dinet_memory::rag::{doc_count, remove_doc, search_store, store_doc};

use crate::model::ChatModel;

/// Один ход диалога.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub ts: u64,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    const USER: u8 = 0;
    const ASSISTANT: u8 = 1;
    fn from_u8(b: u8) -> Role {
        if b == Role::ASSISTANT { Role::Assistant } else { Role::User }
    }
    fn to_u8(self) -> u8 {
        match self {
            Role::User => Role::USER,
            Role::Assistant => Role::ASSISTANT,
        }
    }
}

/// Пара для дообучения (speak→learn: после общения модель этого клиента
/// дообучается на диалогах под ROY-политикой — v1: LocalTrainer/llama.cpp).
#[derive(Debug, Clone, PartialEq)]
pub struct LearnPair {
    pub ts: u64,
    pub prompt: String,
    pub answer: String,
}

const P_TURN: &str = "chat:turn:";
const P_TURN_NEXT: &str = "chat:next";
const P_PAIR: &str = "learn:pair:";
const P_PAIR_NEXT: &str = "learn:next";

#[derive(Debug)]
pub struct ChatSession {
    store: KvStore,
    model_name: &'static str,
}

fn next_counter(store: &mut KvStore, key: &str) -> std::io::Result<u64> {
    let next: u64 = store
        .get(key)
        .and_then(|b| <[u8; 8]>::try_from(b[..8].to_vec()).ok())
        .map(u64::from_le_bytes)
        .unwrap_or(1);
    store.put(key, &next.wrapping_add(1).to_le_bytes())?;
    Ok(next)
}

fn read_turn(store: &KvStore, key: &str) -> Option<Turn> {
    let raw = store.get(key)?;
    if raw.len() < 9 {
        return None;
    }
    let mut ts_b = [0u8; 8];
    ts_b.copy_from_slice(&raw[1..9]);
    Some(Turn {
        role: Role::from_u8(raw[0]),
        ts: u64::from_be_bytes(ts_b),
        text: String::from_utf8_lossy(&raw[9..]).into_owned(),
    })
}

fn encode_turn(t: &Turn) -> Vec<u8> {
    let mut enc = Vec::with_capacity(9 + t.text.len());
    enc.push(t.role.to_u8());
    enc.extend_from_slice(&t.ts.to_be_bytes());
    enc.extend_from_slice(t.text.as_bytes());
    enc
}

fn read_pair(store: &KvStore, key: &str) -> Option<LearnPair> {
    let raw = store.get(key)?;
    if raw.len() < 8 {
        return None;
    }
    let mut ts_b = [0u8; 8];
    ts_b.copy_from_slice(&raw[..8]);
    let mut pos = 8;
    let alen = u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
    pos += 4;
    if raw.len() < pos + alen + 4 {
        return None;
    }
    let a = String::from_utf8_lossy(&raw[pos..pos + alen]).into_owned();
    pos += alen;
    let blen = u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
    pos += 4;
    if raw.len() < pos + blen {
        return None;
    }
    let b = String::from_utf8_lossy(&raw[pos..pos + blen]).into_owned();
    Some(LearnPair { ts: u64::from_be_bytes(ts_b), prompt: a, answer: b })
}

fn encode_pair(p: &LearnPair) -> Vec<u8> {
    let mut enc = Vec::with_capacity(16 + p.prompt.len() + p.answer.len());
    enc.extend_from_slice(&p.ts.to_be_bytes());
    enc.extend_from_slice(&(p.prompt.len() as u32).to_le_bytes());
    enc.extend_from_slice(p.prompt.as_bytes());
    enc.extend_from_slice(&(p.answer.len() as u32).to_le_bytes());
    enc.extend_from_slice(p.answer.as_bytes());
    enc
}

impl ChatSession {
    /// Открыть/создать сессию чата (файл памяти общий с RAG/roй-обучением).
    pub fn open(path: &Path) -> std::io::Result<ChatSession> {
        let store = KvStore::open(path)?;
        Ok(ChatSession { store, model_name: "toy-v0" })
    }

    /// Имя модели, работающей в этой сессии.
    pub fn model_name(&self) -> &'static str {
        self.model_name
    }

    /// Один полный ход: RAG-контекст → модель → persist всего.
    /// Возвращает (ответ, контекст, id документа памяти пользователя).
    pub fn turn(
        &mut self,
        model: &mut dyn ChatModel,
        user: &str,
        ts: u64,
    ) -> std::io::Result<(String, Vec<String>)> {
        // 1. RAG-контекст из памяти
        let ctx: Vec<String> = search_store(&self.store, user, 3)
            .into_iter()
            .map(|(d, _)| d.text)
            .collect();
        // 2. модель
        let answer = model.respond(user, &ctx);
        // 3. persist: ходы (два отдельных id), память, пара обучения
        let user_idx = next_counter(&mut self.store, P_TURN_NEXT)?;
        let reply_idx = next_counter(&mut self.store, P_TURN_NEXT)?;
        let user_turn = Turn { role: Role::User, ts, text: user.to_string() };
        let reply_turn = Turn { role: Role::Assistant, ts, text: answer.clone() };
        self.store.put(&format!("{P_TURN}{user_idx:020}"), &encode_turn(&user_turn))?;
        self.store.put(&format!("{P_TURN}{reply_idx:020}"), &encode_turn(&reply_turn))?;
        // пара обучения (speak→learn)
        let pair_idx = next_counter(&mut self.store, P_PAIR_NEXT)?;
        let pair = LearnPair { ts, prompt: user.to_string(), answer: answer.clone() };
        self.store.put(&format!("{P_PAIR}{pair_idx:020}"), &encode_pair(&pair))?;
        // память: оба текста (ask-контекст отвечать и по фактам ассистента)
        let _uid = store_doc(&mut self.store, user, ts)?;
        let _aid = store_doc(&mut self.store, &answer, ts)?;
        self.model_name = model.name();
        Ok((answer, ctx))
    }

    /// Обучающая пара ОТ УЧИТЕЛЯ: юзер сам даёт правильный ответ
    /// (speak→learn без хода модели). Возвращает индекс пары; пара и оба
    /// текста уходят в память (RAG-контекст учителя тоже находит).
    pub fn teach(&mut self, prompt: &str, answer: &str, ts: u64) -> std::io::Result<u64> {
        let pair_idx = next_counter(&mut self.store, P_PAIR_NEXT)?;
        let pair = LearnPair { ts, prompt: prompt.to_string(), answer: answer.to_string() };
        self.store.put(&format!("{P_PAIR}{pair_idx:020}"), &encode_pair(&pair))?;
        let _ = store_doc(&mut self.store, prompt, ts)?;
        let _ = store_doc(&mut self.store, answer, ts)?;
        Ok(pair_idx)
    }

    /// Вся история (в порядке накопления).
    pub fn turns(&self) -> Vec<Turn> {
        self.store
            .scan(P_TURN)
            .into_iter()
            .filter_map(|k| read_turn(&self.store, &k))
            .collect()
    }

    /// Сколько ходов-пар в буфере обучения.
    pub fn pending_pairs(&self) -> usize {
        self.store.scan(P_PAIR).len()
    }

    /// Выбрать весь буфер обучения (для LocalTrainer/llama.cpp v1).
    pub fn pairs(&self) -> Vec<LearnPair> {
        self.store
            .scan(P_PAIR)
            .into_iter()
            .filter_map(|k| read_pair(&self.store, &k))
            .collect()
    }

    /// Сколько документов памяти.
    pub fn memory_count(&self) -> usize {
        doc_count(&self.store)
    }

    /// Прямой RAG-запрос (для команд приложения).
    pub fn ask_memory(&self, query: &str, k: usize) -> Vec<(u64, String, f32)> {
        search_store(&self.store, query, k)
            .into_iter()
            .map(|(d, s)| (d.id, d.text, s))
            .collect()
    }

    /// Удалить документ памяти по id (см. `ask_memory`).
    pub fn forget(&mut self, id: u64) -> std::io::Result<bool> {
        remove_doc(&mut self.store, id)
    }

    /// Компактация файла памяти (после множества /forget).
    pub fn compact(&mut self) -> std::io::Result<()> {
        self.store.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ToyChatModel;
    use std::fs;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("dinet-chat-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Модель-шпион: запоминает, какой контекст ей передали.
    struct SpyModel {
        contexts: Vec<Vec<String>>,
    }
    impl SpyModel {
        fn new() -> SpyModel {
            SpyModel { contexts: Vec::new() }
        }
    }
    impl ChatModel for SpyModel {
        fn respond(&mut self, _user: &str, context: &[String]) -> String {
            self.contexts.push(context.to_vec());
            format!("ок, контекстов: {}", context.len())
        }
        fn name(&self) -> &'static str {
            "spy"
        }
    }

    #[test]
    fn turn_persists_history_memory_and_pairs() {
        let dir = tmpdir("flow");
        let path = dir.join("session.dlog");
        let mut s = ChatSession::open(&path).unwrap();
        let mut m = ToyChatModel::default();
        let (a1, _) = s.turn(&mut m, "я люблю борщ со сметаной", 100).unwrap();
        assert!(a1.contains("память пуста"), "первый ход без памяти: {a1}");
        let (a2, _) = s.turn(&mut m, "что я люблю?", 101).unwrap();
        assert!(a2.contains("борщ"), "второй ход уже с памятью: {a2}");

        assert_eq!(s.turns().len(), 4, "2 хода × (user+assistant)");
        assert_eq!(s.pending_pairs(), 2, "каждый ход = пара speak→learn");
        assert_eq!(s.pairs()[0].prompt, "я люблю борщ со сметаной");
        assert_eq!(s.pairs()[1].ts, 101);
        assert_eq!(s.memory_count(), 4, "user + assistant тексты к обоим ходам");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn context_reaches_model_via_rag() {
        let dir = tmpdir("ctx");
        let path = dir.join("session.dlog");
        let mut s = ChatSession::open(&path).unwrap();
        let mut spy = SpyModel::new();
        s.turn(&mut spy, "запомни: хобби — фотография", 1).unwrap();
        s.turn(&mut spy, "какие у меня увлечения?", 2).unwrap();
        // первый ответ — пустой контекст, второй — с фактами о фотографии
        assert_eq!(spy.contexts[0].len(), 0);
        assert!(
            spy.contexts[1].iter().any(|c| c.contains("фотография")),
            "RAG-контекст донёс факт: {:?}",
            spy.contexts[1]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_persists_across_reopen() {
        let dir = tmpdir("persist");
        let path = dir.join("session.dlog");
        {
            let mut s = ChatSession::open(&path).unwrap();
            let mut m = ToyChatModel::default();
            s.turn(&mut m, "город — Саратов", 5).unwrap();
        }
        let s = ChatSession::open(&path).unwrap();
        assert_eq!(s.turns().len(), 2, "история пережила перезапуск");
        assert_eq!(s.pending_pairs(), 1);
        assert_eq!(s.memory_count(), 2);
        let top = s.ask_memory("город", 1);
        assert_eq!(top.len(), 1);
        assert!(top[0].1.contains("Саратов"), "{:?}", top);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn teacher_pair_persists_without_model_turn() {
        let dir = tmpdir("teach");
        let mut s = ChatSession::open(&dir.join("session.dlog")).expect("open");
        let now = 1_700_000_000;
        let idx = s.teach("ай ви ди би", "ви", now).expect("teach");
        assert_eq!(idx, 1);
        assert_eq!(s.pending_pairs(), 1);
        let p = &s.pairs()[0];
        assert_eq!(p.prompt, "ай ви ди би");
        assert_eq!(p.answer, "ви");
        assert_eq!(p.ts, now);
        // учительские тексты тоже в памяти RAG
        assert!(!s.ask_memory("правильный ответ ви", 5).is_empty());
    }

    #[test]
    fn learn_pair_roundtrip_layout() {
        let dir = tmpdir("pair");
        let path = dir.join("session.dlog");
        let mut s = ChatSession::open(&path).unwrap();
        let mut m = ToyChatModel::default();
        s.turn(&mut m, "промпт с кириллицей и «длинной» — репой", 7).unwrap();
        let p = s.pairs();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].prompt, "промпт с кириллицей и «длинной» — репой");
        assert_eq!(p[0].ts, 7);
        let _ = fs::remove_dir_all(&dir);
    }
}
