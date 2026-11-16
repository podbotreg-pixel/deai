//! Телефонное ядро DI-NET (M3, шаг 1): тот же Rust-стек, что в webchat,
//! без переделок — чат-сессия + ЛОКАЛЬНЫЕ эры роя поверх KV-файлов.
//!
//! Публичный API — простые методы [`PhoneCore`]; их зовёт рукописный
//! JNI-мост (`jni.rs`), а мост — Kotlin `Core.kt` из Compose-UI:
//! `open(dir)` → `turn(prompt)` / `teach(prompt, answer)` / `stats()`.
//!
//! ## Что живёт на телефоне (шаг 1)
//! - `ChatSession` (KV-файл `<dir>`): ходы, память, учительские пары —
//!   переживают убийство процесса (append-only KV, tmp+rename).
//! - Рой-KV-файл `<dir>.рой`: последний чек-пойнт (адаптер + счётчики).
//! - Каждые `every` валидных учительских пар — локальная эра: один узел
//!   проходит `run_era_partition` (публикация → кворум → REVEAL →
//!   чек-пойнт), и чек-пойнт немедленно становится репликой чата —
//!   код эпох ровно тот же, что у spoolswarm/clientswarm/webchat.
//! - Сетевой вклад (WorkManager-фоны, Wi-Fi/батарея/термал) — шаги 2–3 M3.
//!
//! ## Форматы KV-ключей (совместимы с webchat)
//! - `рой:adapter` — LoRA-адаптер как `f32`-массив little-endian;
//! - `рой:cps`, `рой:eras` — счётчики `u32` little-endian.

mod jni;

use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use dinet_chat::{ChatSession, ToyChatModel};
use dinet_client::{pair_to_sample, pairs_to_batch};
use dinet_fl::toy_llm::{LoRAAdapter, ToyModel};
use dinet_memory::kv::KvStore;
use dinet_node::swarm::{run_era_partition, Node, SimConfig};
use dinet_node::transport::MeshTransport;

/// id телефонного узла (шаг 1 — локальные эры; сетевой ключ узла —
/// вместе с сетевым вкладом, шаг 2+).
const PHONE_ID: u16 = 7;
/// Сид ToyWorld роя: 7 — как во всём DI-NET (webchat/спулы/клиенты).
const WORLD_SEED: u64 = 7;
/// Порог локальной эры: каждые N валидных учительских пар.
const EVERY: u32 = 4;
/// KV-ключ: сколько валидных пар было на момент ПОСЛЕДНЕЙ эры (фоновый
/// воркер гонит эру только по новому материалу — семантика bg_trainer
/// webchat: «при новом материале прокручивается та же era_tick»).
const P_LAST_PAIRS: &str = "рой:last_pairs";

pub struct PhoneCore {
    pub session: ChatSession,
    pub node: Node,
    /// проведено эр (всего, с учётом прошлых запусков)
    pub eras: u32,
    /// принято чек-пойнтов REVEAL (применённых к реплике чата)
    pub cps: u32,
    /// порог эры по парам
    pub every: u32,
    /// валидных пар на момент последней эры (переживает рестарт)
    pub last_pairs: u32,
    /// Реплика чата: последний чек-пойнт роя; до первой эры адаптер
    /// нулевой/начальный — микроязык отвечает argmax'ом базы, обычная
    /// речь — шаблонным режимом RAG (как в webchat).
    pub model: ToyChatModel,
    kv: KvStore,
}

impl PhoneCore {
    /// Открыть/создать ядро. `base` — путь KV-файла сессии (каталог
    /// создаётся); рой-KV — сиблинг `<base>.рой`. Чек-пойнт роя
    /// восстанавливается: адаптер + счётчики переживают рестарт.
    pub fn open(base: &Path) -> io::Result<PhoneCore> {
        let session = ChatSession::open(base)?;
        let roi = base.with_extension("рой");
        let kv = KvStore::open(&roi)?;
        let kver = |k: &str| -> u32 {
            kv.get(k)
                .map(|b| {
                    let mut a = [0u8; 4];
                    let n = b.len().min(4);
                    a[..n].copy_from_slice(&b[..n]);
                    u32::from_le_bytes(a)
                })
                .unwrap_or(0)
                .min(u32::MAX - 1)
        };
        let eras = kver("рой:eras");
        let cps = kver("рой:cps");
        let last_pairs = kver(P_LAST_PAIRS);
        let mut node = Node::new(PHONE_ID, WORLD_SEED, false);
        if let Some(ab) = kv.get("рой:adapter") {
            let vals: Vec<f32> = ab
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if vals.len() == LoRAAdapter::num_params() {
                node.adapter = LoRAAdapter::from_vec(&vals);
            }
        }
        let model = ToyChatModel::trained(ToyModel {
            base: node.base.clone(),
            adapter: node.adapter.clone(),
        });
        Ok(PhoneCore {
            session,
            node,
            eras,
            cps,
            every: EVERY,
            last_pairs,
            model,
            kv,
        })
    }

    /// Ход чата: RAG из памяти + ответ реплики роя (микроязык — argmax
    /// сети, обычная речь — шаблонный режим), ход persist'ится.
    pub fn turn(&mut self, prompt: &str) -> io::Result<String> {
        let (answer, _sources) = self.session.turn(&mut self.model, prompt, now_ts())?;
        Ok(answer)
    }

    /// Учительская пара. Валидируется тем же кодом, что и рой-обучение
    /// (`pair_to_sample`): 4 слова контекста + 1 слово-ответ из LEX.
    /// Каждые `every` пар автоматически идёт локальная эра роя.
    pub fn teach(&mut self, prompt: &str, answer: &str) -> Result<String, String> {
        if pair_to_sample(prompt, answer).is_none() {
            return Err(
                "не пара микроязыка: надо 4 слова контекста + 1 слово ответа \
                 (ай би ви ги ди ей жи зи)"
                    .into(),
            );
        }
        let idx = self
            .session
            .teach(prompt, answer, now_ts())
            .map_err(|e| e.to_string())?;
        let pairs = self.valid_pairs().len();
        let mut note = format!("пара #{idx} записана; обучающих: {pairs}");
        if pairs as u32 % self.every.max(1) == 0 {
            if let Some(era) = self.era_tick() {
                note.push_str("; ");
                note.push_str(&era);
            }
        }
        Ok(note)
    }

    /// Валидные учительские пары микроязыка. `ChatSession::turn` сам пишет
    /// пары диалогов (опыт чата), но в эру идёт только проверенный
    /// материал: фильтр — тот же `pair_to_sample`, что и в рое; он же
    /// отсеет русские ответы шаблонного режима.
    pub fn valid_pairs(&self) -> Vec<(String, String)> {
        self.session
            .pairs()
            .into_iter()
            .filter(|p| pair_to_sample(&p.prompt, &p.answer).is_some())
            .map(|p| (p.prompt, p.answer))
            .collect()
    }

    /// ЛОКАЛЬНАЯ эра роя: fit только на учителях парах клиента → кворум →
    /// чек-пойнт. Тот же код эпох, что в webchat `era_tick` (procs=1).
    /// Возвращает человеческую строку итога (для GUI), None — нет материала.
    pub fn era_tick(&mut self) -> Option<String> {
        let valid = self.valid_pairs();
        if valid.is_empty() {
            return None;
        }
        let cfg = SimConfig::default();
        // PeerBook не Clone: на время фаз подменяем узел заглушкой
        // (id тот же) — точно как в webchat era_tick.
        let nid = self.node.id;
        let mut node = std::mem::replace(&mut self.node, Node::new(nid, WORLD_SEED, false));
        node.batch_override = Some(pairs_to_batch(&valid));
        let mut mesh = MeshTransport::with_ids(&[node.id]);
        let mut mine = vec![node];
        let eid = mine[0].id;
        let era_num = self.eras;
        let reports = run_era_partition(&mut mine, &[eid], &mut mesh, era_num, &cfg, 2026);
        let r = &reports[0];
        self.node = mine.remove(0);
        self.eras += 1;
        // материал этой эры учтён: фоновый воркер не повторит её зря;
        // счётчик эр — в KV КАЖДУЮ эру (иначе рестарт сбивает ритм reveal)
        self.last_pairs = valid.len() as u32;
        let _ = self.kv.put(P_LAST_PAIRS, &self.last_pairs.to_le_bytes());
        let _ = self.kv.put("рой:eras", &self.eras.to_le_bytes());
        let mut note = format!(
            "эра {}: вклад {} пар, acc {:.3}→{:.3}",
            r.era, valid.len(), r.accuracy_before, r.accuracy_after
        );
        if r.checkpoint_id.is_some() {
            self.cps += 1;
            self.apply_checkpoint();
            note.push_str(&format!(
                "; чек-пойнт {}-й применён к чату",
                self.cps
            ));
        }
        Some(note)
    }

    /// ФОНОВЫЙ вклад (WorkManager, шаг 2): эра по НОВОМУ материалу —
    /// сколько накопилось валидных пар сверх последней эры, такая эра и
    /// идёт (семантика bg_trainer webchat: «при новом материале»).
    /// None — нового материала нет; воркер тогда просто выходит.
    pub fn bg_tick(&mut self) -> Option<String> {
        let pairs = self.valid_pairs().len() as u32;
        if pairs <= self.last_pairs {
            return None;
        }
        self.era_tick()
    }

    /// Чек-пойнт → KV (адаптер + счётчики, переживают рестарт) и сразу в
    /// жизнь чата: реплика пересобирается из нового адаптера.
    fn apply_checkpoint(&mut self) {
        let cps = self.cps;
        let eras = self.eras;
        let ab: Vec<u8> = self
            .node
            .adapter
            .as_vec()
            .iter()
            .fold(Vec::new(), |mut v, x| {
                v.extend_from_slice(&x.to_le_bytes());
                v
            });
        let _ = self.kv.put("рой:adapter", &ab);
        let _ = self.kv.put("рой:cps", &cps.to_le_bytes());
        let _ = self.kv.put("рой:eras", &eras.to_le_bytes());
        self.model = ToyChatModel::trained(ToyModel {
            base: self.node.base.clone(),
            adapter: self.node.adapter.clone(),
        });
    }

    /// Счётчики для GUI одной строкой.
    pub fn stats(&self) -> String {
        format!(
            "модель: {}; ходы: {}; пары: {}; память: {}; эры: {}; чек-пойнты: {}",
            self.session.model_name(),
            self.session.turns().len(),
            self.session.pairs().len(),
            self.session.memory_count(),
            self.eras,
            self.cps,
        )
    }
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_fl::toy_llm::LEX;

    /// Путь KV-ФАЙЛА сессии (как webchat `./wa`); рой-KV — сиблинг
    /// `<base>.рой`. Чистим и файл, и рой-сиблинга от прошлых запусков.
    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("dinet-android-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&d);
        let _ = std::fs::remove_file(d.with_extension("рой"));
        d
    }

    /// Пара микроязыка №i: 4 слова контекста + 1 слово-ответ.
    fn mic_pair(i: usize) -> (String, String) {
        let w = |j: usize| LEX[j % LEX.len()];
        (
            format!("{} {} {} {}", w(i), w(i + 1), w(i + 2), w(i + 3)),
            w(i + 4).to_string(),
        )
    }

    #[test]
    fn turn_template_then_teach() {
        let dir = tmp("turn");
        let mut core = PhoneCore::open(&dir).unwrap();
        // шаблонный режим: пустая память → честное «запомню»
        let answer = core.turn("привет, что помнишь?").unwrap();
        assert!(answer.contains("запомню"), "ответ: {answer}");
        // невалидная пара отклонена ДО записи (тот же код, что рой)
        assert!(core.teach("ай ви", "би").is_err());
        assert_eq!(core.valid_pairs().len(), 0);
        let note = core.teach("ай ви ди би", "ви").unwrap();
        // индекс пары общий с автопарами диалогов (тут #2); валидных — 1
        assert!(note.contains("записана") && note.contains("обучающих: 1"), "note: {note}");
        // в KV две пары: ход-диалог (автопара) + учительская; в эру идёт
        // только учительская — диалог на русском отсеян фильтром
        assert_eq!(core.session.pairs().len(), 2);
        assert_eq!(core.valid_pairs().len(), 1);
        // ход на микроязыке → слово-ответ из LEX (реплика всегда отвечает словом)
        let reply = core.turn("ай ви ди би").unwrap();
        assert!(LEX.contains(&reply.as_str()), "ответ: {reply}");
    }

    #[test]
    fn era_on_every_teaches() {
        let dir = tmp("era");
        let mut core = PhoneCore::open(&dir).unwrap();
        // 3 пары — до порога эр нет
        for i in 0..3 {
            let (p, a) = mic_pair(i);
            core.teach(&p, &a).unwrap();
        }
        assert_eq!(core.eras, 0);
        assert_eq!(core.cps, 0);
        // 4-я пара — 1-я эра (masked): счёт идёт, чек-пойнта ещё нет —
        // REVEAL только каждую reveal_every=4-ю эру (как в webchat)
        let note = core.teach("ай ви ди би", "ви").unwrap();
        assert_eq!(core.eras, 1, "note: {note}");
        assert_eq!(core.cps, 0, "note: {note}");
        assert!(note.contains("эра"), "note: {note}");
        // ещё 12 пар → эры 2–4; 4-я эра (era 3) — REVEAL: чек-пойнт 1-й
        for i in 4..16 {
            let (p, a) = mic_pair(i);
            core.teach(&p, &a).unwrap();
        }
        assert_eq!(core.eras, 4);
        assert_eq!(core.cps, 1);
        // KV держит счётчики и адаптер
        assert_eq!(core.kv.get("рой:cps").unwrap().len(), 4);
        assert_eq!(core.kv.get("рой:eras").unwrap().len(), 4);
        assert!(core.kv.get("рой:adapter").is_some());
    }

    #[test]
    fn persistence_pairs_and_checkpoint() {
        let dir = tmp("persist");
        let adapter;
        {
            let mut core = PhoneCore::open(&dir).unwrap();
            core.teach("ай ви ди би", "ви").unwrap();
            assert_eq!(core.session.pairs().len(), 1);
            for i in 1..16 {
                let (p, a) = mic_pair(i);
                core.teach(&p, &a).unwrap();
            }
            assert_eq!(core.eras, 4);
            assert_eq!(core.cps, 1);
            adapter = core.node.adapter.as_vec();
        }
        // рестарт процесса: пары и чек-пойнт живы
        let mut core = PhoneCore::open(&dir).unwrap();
        assert_eq!(core.session.pairs().len(), 16);
        assert_eq!(core.eras, 4);
        assert_eq!(core.cps, 1);
        assert_eq!(core.node.adapter.as_vec(), adapter);
        // эра, дожившая до рестарта, продолжает счёт с них: ещё 4 пары →
        // 5-я эра (masked), чек-пойнты по-прежнему 1
        for i in 16..20 {
            let (p, a) = mic_pair(i);
            core.teach(&p, &a).unwrap();
        }
        assert_eq!(core.eras, 5);
        assert_eq!(core.cps, 1);
    }

    #[test]
    fn stats_line_readable() {
        let dir = tmp("stats");
        let mut core = PhoneCore::open(&dir).unwrap();
        assert!(core.stats().contains("эры: 0"));
        assert!(core.stats().contains("модель: toy-v0"));
        core.teach("ай ви ди би", "ви").unwrap();
        assert!(core.stats().contains("пары: 1"), "{}", core.stats());
    }

    #[test]
    fn bg_tick_only_on_new_material() {
        let dir = tmp("bg");
        let mut core = PhoneCore::open(&dir).unwrap();
        // 4 пары → inline-эра (last_pairs = 4); фон без нового материала — тишина
        for i in 0..4 {
            let (p, a) = mic_pair(i);
            core.teach(&p, &a).unwrap();
        }
        assert_eq!(core.eras, 1);
        assert_eq!(core.last_pairs, 4);
        assert!(core.bg_tick().is_none());
        // ещё 2 пары (не кратные порогу — inline не стрелял): фон доводит до эры
        for i in 4..6 {
            let (p, a) = mic_pair(i);
            core.teach(&p, &a).unwrap();
        }
        assert_eq!(core.eras, 1);
        let note = core.bg_tick().unwrap();
        assert!(note.contains("эра"), "note: {note}");
        assert_eq!(core.eras, 2);
        assert_eq!(core.last_pairs, 6);
        // снова без нового материала
        assert!(core.bg_tick().is_none());
        // last_pairs переживает рестарт: 7-я пара → фон гонит эру на ней
        drop(core);
        let mut core = PhoneCore::open(&dir).unwrap();
        assert_eq!(core.last_pairs, 6);
        assert_eq!(core.eras, 2, "счётчик эр переживает рестарт даже без REVEAL");
        let (p, a) = mic_pair(6);
        core.teach(&p, &a).unwrap();
        assert!(core.bg_tick().unwrap().contains("эра"));
        assert_eq!(core.eras, 3);
    }
}
