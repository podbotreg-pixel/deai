//! RAG-поиск по локальной памяти (v0: hash-embedding на SHA-256).
//!
//! Эмбеддинг v0 — «bag of hashed tokens»: токен → индекс через SHA-256,
//! L2-норма. Сходство — скалярное произведение (векторы нормированы).
//! Это игрушка, НО: локальная, бесплатная, детерминированная, без зависимостей.
//! Точка прикрепления v1: `embed()` заменяется на выход llama.cpp
//! (то же пространство, поиска по dot-product) — поиск/хранилище не меняются.

use super::kv::KvStore;
use dinet_crypto::sha256;

/// Размерность hash-эмбеддинга v0.
pub const EMBED_DIM: usize = 256;

const P_DOC: &str = "rag:doc:";
const P_NEXT: &str = "rag:next";

/// Токенизация: lowercase, разбивать по не-буквенно-цифровым, буквы/цифры Юникод.
pub fn tokens(text: &str) -> Vec<String> {
    let mut t = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_alphanumeric() {
            t.extend(c.to_lowercase());
        } else {
            t.push(' ');
        }
    }
    t.split(' ')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Hash-эмбеддинг текста (bag-of-tokens, L2-нормированный).
#[must_use]
pub fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    for tok in tokens(text) {
        let h = sha256(tok.as_bytes());
        let idx = (u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize) % EMBED_DIM;
        v[idx] += 1.0; // TF: каждый повтор добавляет
    }
    let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

/// Скалярное произведение (на нормированных векторах = cosine similarity).
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Документ памяти.
#[derive(Debug, Clone, PartialEq)]
pub struct Doc {
    pub id: u64,
    pub ts: u64,
    pub text: String,
}

fn doc_key(id: u64) -> String {
    format!("{P_DOC}{id:020}")
}

fn read_doc(store: &KvStore, key: &str) -> Option<Doc> {
    let raw = store.get(key)?;
    if raw.len() < 8 {
        return None;
    }
    let mut ts_b = [0u8; 8];
    ts_b.copy_from_slice(&raw[..8]);
    let id_str = key.trim_start_matches(P_DOC);
    Some(Doc {
        id: id_str.parse().ok()?,
        ts: u64::from_be_bytes(ts_b),
        text: String::from_utf8_lossy(&raw[8..]).into_owned(),
    })
}

fn write_doc(v: &mut Vec<u8>, doc: &Doc) {
    v.clear();
    v.extend_from_slice(&doc.ts.to_be_bytes());
    v.extend_from_slice(doc.text.as_bytes());
}

/// Открытый ретривер над KV-логом. Все документы сидят в том же файле.
#[derive(Debug)]
pub struct Retriever {
    store: KvStore,
}

impl Retriever {
    pub fn open(path: &std::path::Path) -> std::io::Result<Retriever> {
        Ok(Retriever { store: KvStore::open(path)? })
    }

    pub fn count(&self) -> usize {
        self.store.scan(P_DOC).len()
    }

    /// Добавить документ; возвращает id.
    pub fn add(&mut self, text: &str, ts: u64) -> std::io::Result<u64> {
        let id = self.next_id()?;
        let doc = Doc { id, ts, text: text.to_string() };
        let mut enc = Vec::new();
        write_doc(&mut enc, &doc);
        self.store.put(&doc_key(id), &enc)?;
        Ok(id)
    }

    /// Убрать документ из памяти.
    pub fn remove(&mut self, id: u64) -> std::io::Result<bool> {
        self.store.delete(&doc_key(id))
    }

    fn next_id(&mut self) -> std::io::Result<u64> {
        let next: u64 = self
            .store
            .get(P_NEXT)
            .and_then(|b| <[u8; 8]>::try_from(b[..8].to_vec()).ok())
            .map(u64::from_le_bytes)
            .unwrap_or(1);
        self.store.put(P_NEXT, &next.wrapping_add(1).to_le_bytes())?;
        Ok(next)
    }

    /// Поиск top-k по cosine: `(doc, score)`, убывание score.
    pub fn search(&self, query: &str, k: usize) -> Vec<(Doc, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let q = embed(query);
        let mut scored: Vec<(Doc, f32)> = Vec::new();
        for key in self.store.scan(P_DOC) {
            if let Some(doc) = read_doc(&self.store, &key) {
                let s = dot(&q, &embed(&doc.text));
                scored.push((doc, s));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("dinet-rag-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn open_rag(tag: &str) -> Retriever {
        let dir = tmpdir(tag);
        Retriever::open(&dir.join("mem.dlog")).unwrap()
    }

    #[test]
    fn embed_is_normalized_and_deterministic() {
        let a = embed("борщ со сметаной");
        let b = embed("борщ со сметаной");
        assert_eq!(a, b, "детерминизм");
        let n = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert!((n - 1.0).abs() < 1e-4, "L2-норма ≈ 1: {n}");
        let empty = embed(" ");
        assert!(empty.iter().all(|x| *x == 0.0), "пустой текст → нулевой вектор");
        assert_eq!(dot(&empty, &a), 0.0);
    }

    #[test]
    fn topk_relevance_over_distractors() {
        let mut r = open_rag("topk");
        r.add("пользователь любит борщ со сметаной", 1).unwrap();
        for _ in 0..8 {
            r.add("сеть DI-NET работает децентрализованно без владельцев", 2).unwrap();
        }
        r.add("сегодня в Саратове тёплая погода", 3).unwrap();

        let top = r.search("как приготовить борщ со сметаной", 1);
        assert_eq!(top.len(), 1);
        assert!(top[0].0.text.contains("борщ"), "top1 — про борщ: {}", top[0].0.text);

        let top = r.search("какая погода", 2);
        assert!(top[0].0.text.contains("погода"), "top1 погода: {}", top[0].0.text);
        // док про DI-NET не в тройке
        assert!(!top.iter().any(|(d, _)| d.text.contains("децентрализованно")));

        let top = r.search("децентрализованная сеть", 2);
        assert!(top[0].0.text.contains("децентрализованно"), "top1: {}", top[0].0.text);
    }

    #[test]
    fn add_remove_persist_reopen() {
        let dir = tmpdir("persist");
        let path = dir.join("mem.dlog");
        let mut r = Retriever::open(&path).unwrap();
        let id1 = r.add("факт один: хобби — фотография", 10).unwrap();
        r.add("факт два: город — Саратов", 11).unwrap();
        assert_eq!(r.count(), 2);
        assert!(r.remove(id1).unwrap());
        assert_eq!(r.count(), 1);
        drop(r);
        let r = Retriever::open(&path).unwrap();
        assert_eq!(r.count(), 1, "удаление пережило reopen");
        let ids = r.search("фотография", 5);
        assert!(ids.iter().all(|(d, _)| d.id != id1));
        let top = r.search("какой город", 1);
        assert!(top[0].0.text.contains("Саратов"), "{}", top[0].0.text);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ids_are_monotonic_across_reopen() {
        let dir = tmpdir("ids");
        let path = dir.join("mem.dlog");
        let mut r = Retriever::open(&path).unwrap();
        let a = r.add("x", 1).unwrap();
        let b = r.add("y", 1).unwrap();
        drop(r);
        let mut r = Retriever::open(&path).unwrap();
        let c = r.add("z", 1).unwrap();
        assert!(b > a && c > b, "id растут даже после перезапуска: {a},{b},{c}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_k_zero_and_empty_query() {
        let mut r = open_rag("edge");
        r.add("что-то есть", 1).unwrap();
        assert!(r.search("любой запрос", 0).is_empty());
        let top = r.search("", 3);
        // пустой запрос → нулевой вектор → все score 0, но ответы валидны
        assert!(top.len() <= 1);
        assert!(top.iter().all(|(_, s)| s.abs() < 1e-9));
    }
}
