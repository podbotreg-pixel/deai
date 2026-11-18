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
//! - `рой:proxy` — личная прокси телефона (EFFEKT, f32 LE, переживает
//!   рестарты; копится эрами сетевого вклада);
//! - `рой:cps`, `рой:eras` — счётчики `u32` little-endian.

mod jni;
mod net;

use std::io;
use std::path::{Path, PathBuf};
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
/// KV-ключ: адреса роя через запятую (`ip:порт` — их dial'ит сетевой
/// вклад; пишет deep-link dinet://cmd/peer?addr=…).
const P_PEERS: &str = "рой:peers";
/// KV-ключ: личная прокси телефона (EFFEKT-адаптер, f32 LE) — копится
/// эрами, переживает рестарты; в чек-пойнт роя НЕ входит (вклад —
/// carrier-логиты, DSL).
const P_PROXY: &str = "рой:proxy";
/// KV-ключ: последний состав роя (u16 little-endian подряд, отсортирован)
/// — пишет probe/сетевой вклад после волны; читает `swarm_stats`
/// (экран «Рой»).
const P_LAST_ROSTER: &str = "рой:last_roster";
/// KV-ключ: время последнего контакта с роем (unix-сек, u64 LE) —
/// «видели N мин назад» на экране «Рой».
const P_LAST_SEEN: &str = "рой:last_seen";

/// f32-массив → little-endian байты (формат KV `рой:adapter`/`рой:proxy`).
fn kv_f32(v: &[f32]) -> Vec<u8> {
    v.iter().fold(Vec::new(), |mut b, x| {
        b.extend_from_slice(&x.to_le_bytes());
        b
    })
}

/// little-endian байты → f32-массив (None — длина не кратна 4).
fn kv_f32_back(b: &[u8]) -> Option<Vec<f32>> {
    if b.len() % 4 != 0 {
        return None;
    }
    Some(
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

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
    /// путь KV-файла сессии (от него — сиблинги: рой-KV, реестр сети)
    pub base: PathBuf,
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
            if let Some(vals) = kv_f32_back(&ab) {
                if vals.len() == LoRAAdapter::num_params() {
                    node.adapter = LoRAAdapter::from_vec(&vals);
                }
            }
        }
        // EFFEKT-прокси телефона: персист между процессами (копится эрами)
        if let Some(pb) = kv.get(P_PROXY) {
            if let Some(vals) = kv_f32_back(&pb) {
                if vals.len() == LoRAAdapter::num_params() {
                    node.proxy = LoRAAdapter::from_vec(&vals);
                }
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
            base: base.to_path_buf(),
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

    /// Адреса роя из рой-KV (»рой:peers», через запятую).
    pub fn peers(&self) -> Vec<String> {
        self.kv
            .get(P_PEERS)
            .map(|b| String::from_utf8_lossy(&b).to_string())
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Запомнить адрес роя (`ip:порт`; дубликат не пишется). Формат
    /// проверяет `parse_dial` — та же валидация, что у dial'а.
    pub fn add_peer(&mut self, addr: &str) -> Result<(), String> {
        dinet_p2p::parse_dial(addr)?;
        let mut list = self.peers();
        if !list.iter().any(|a| a == addr.trim()) {
            list.push(addr.trim().to_string());
            let _ = self.kv.put(P_PEERS, list.join(",").as_bytes());
        }
        Ok(())
    }

    /// Чек-пойнт → KV (адаптер + счётчики, переживают рестарт) и сразу в
    /// жизнь чата: реплика пересобирается из нового адаптера.
    fn apply_checkpoint(&mut self) {
        let cps = self.cps;
        let eras = self.eras;
        let ab = kv_f32(&self.node.adapter.as_vec());
        let _ = self.kv.put("рой:adapter", &ab);
        let _ = self.kv.put("рой:cps", &cps.to_le_bytes());
        let _ = self.kv.put("рой:eras", &eras.to_le_bytes());
        self.model = ToyChatModel::trained(ToyModel {
            base: self.node.base.clone(),
            adapter: self.node.adapter.clone(),
        });
    }

    /// Личная прокси телефона → KV (переживает рестарт процесса): глубина
    /// прокси копится сессиями сетевого вклада (EFFEKT).
    fn persist_proxy(&mut self) {
        let pb = kv_f32(&self.node.proxy.as_vec());
        let _ = self.kv.put(P_PROXY, &pb);
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

    /// CamoMark-watermark (M4): поведенческая проверка знака происхождения
    /// реплики роя — argmax на триггере (сид роя dinet_node::swarm::
    /// WATERMARK_SEED) обязан быть токеном знака. Верификация без
    /// эталонных весов; знак появляется после первой эры (era_dsl_kd
    /// вшивает его KD'ом) и переживает дальнейшие эры.
    pub fn watermark_ok(&self) -> bool {
        let m = ToyModel { base: self.node.base.clone(), adapter: self.node.adapter.clone() };
        dinet_fl::toy_llm::verify_watermark(&m, dinet_node::swarm::WATERMARK_SEED)
    }

    /// Статистика роя (экран «Рой»): количественные характеристики —
    /// адреса роя, последний состав («рой:last_roster», пишет
    /// net_probe/сетевой вклад после волны), «видели N мин назад»
    /// («рой:last_seen»), вклад (эры/чек-пойнты, последняя эра сети
    /// из «рой:net_done») и глубина личной прокси (≈ эры × шаги KD).
    /// NB: записи клиентского реестра `<base>.net-reg` в v1 всегда
    /// пусты (телефон входит в цепочку с середины — parent не сходится,
    /// печати бракуются), поэтому метрикой служит «рой:net_done».
    pub fn swarm_stats(&self) -> String {
        let addrs = self.peers();
        let roster: Vec<u16> = self
            .kv
            .get(P_LAST_ROSTER)
            .map(|b| {
                b.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            })
            .unwrap_or_default();
        let seen = self
            .kv
            .get(P_LAST_SEEN)
            .map(|b| {
                let mut a = [0u8; 8];
                let n = b.len().min(8);
                a[..n].copy_from_slice(&b[..n]);
                now_ts().saturating_sub(u64::from_le_bytes(a)) / 60
            })
            .unwrap_or(0);
        let net_done = self
            .kv
            .get("рой:net_done")
            .map(|b| {
                let mut a = [0u8; 4];
                let n = b.len().min(4);
                a[..n].copy_from_slice(&b[..n]);
                u32::from_le_bytes(a)
            })
            .map(|era| format!("последняя эра сети: {era}"))
            .unwrap_or_else(|| "последняя эра сети: нет (только локальные)".to_string());
        let addr_line = if addrs.is_empty() {
            "адресов 0 (добавить: dinet://cmd/peer?addr=ip:порт)".to_string()
        } else {
            format!("адресов {} ({})", addrs.len(), addrs.join(", "))
        };
        let roster_line = if roster.is_empty() {
            "ещё не видели («Обновить состав» — probe роя)".to_string()
        } else {
            format!("{roster:?} — {} узл., видели {seen} мин назад", roster.len())
        };
        let steps = SimConfig::default().train_steps;
        let wm = if self.watermark_ok() { "подтверждён" } else { "нет (ждёт эру)" };
        format!(
            "рой: {addr_line}\nсостав: {roster_line}\n\
             вклад: эр {}, чек-пойнтов {}; {net_done}; прокси ≈ {} шагов KD\n\
             водяной знак: {wm}",
            self.eras,
            self.cps,
            self.eras as u64 * steps as u64,
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
    use std::sync::Mutex;

    /// Взаимоисключение СЕТЕВЫХ тестов: мост телефона — фиксированный
    /// PeerId `[7;32]`, два параллельных экземпляра ломают друг друга
    /// (mdns на loopback при этом переливает Hello между тестами).
    pub(crate) static NET_LOCK: Mutex<()> = Mutex::new(());

    /// Путь KV-ФАЙЛА сессии (как webchat `./wa`); рой-KV — сиблинг
    /// `<base>.рой`. Чистим и файл, и рой-сиблинга от прошлых запусков.
    pub(crate) fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("dinet-android-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&d);
        let _ = std::fs::remove_file(d.with_extension("рой"));
        let _ = std::fs::remove_file(d.with_extension("net-reg"));
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

    /// CamoMark-watermark (M4): до эр знак НЕ подтверждён (свежий
    /// адаптер не отвечает триггеру); после 16 пар → 4 эр (era_dsl_kd
    /// вшивает знак KD'ом) — подтверждён, и строка экрана «Рой» это
    /// показывает.
    #[test]
    fn watermark_after_eras() {
        let dir = tmp("wm");
        let mut core = PhoneCore::open(&dir).unwrap();
        assert!(!core.watermark_ok(), "свежий адаптер не несёт знака");
        assert!(core.swarm_stats().contains("водяной знак: нет"), "stats: {}", core.swarm_stats());
        for i in 0..16 {
            let (p, a) = mic_pair(i);
            core.teach(&p, &a).unwrap();
        }
        assert_eq!(core.eras, 4);
        assert!(
            core.watermark_ok(),
            "после эр знак обязан держаться; stats: {}",
            core.swarm_stats()
        );
        assert!(
            core.swarm_stats().contains("водяной знак: подтверждён"),
            "stats: {}",
            core.swarm_stats()
        );
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
    fn proxy_persists_restart() {
        // EFFEKT-прокси: личная прокси телефона переживает рестарт
        // (KV «рой:proxy»), не входя в чек-пойнт роя
        let dir = tmp("proxy");
        let proxy;
        {
            let mut core = PhoneCore::open(&dir).unwrap();
            // отличимая от init прокси (как после эр сетевого вклада)
            let mut px = dinet_fl::toy_llm::LoRAAdapter::lora_init(9);
            for (i, v) in px.a.iter_mut().enumerate() {
                *v += (i as f32) * 0.25;
            }
            for (i, v) in px.b.iter_mut().enumerate() {
                *v += (i as f32) * 0.125;
            }
            core.node.proxy = px;
            core.persist_proxy();
            proxy = core.node.proxy.as_vec();
        }
        let core = PhoneCore::open(&dir).unwrap();
        assert_eq!(core.node.proxy.as_vec(), proxy, "прокси не дожила до рестарта");
        assert!(
            core.node.proxy.as_vec() != dinet_fl::toy_llm::LoRAAdapter::lora_init(dinet_node::swarm::PROXY_BASE_SEED).as_vec(),
            "прокси обязана отличаться от init"
        );
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

    /// Экран «Рой»: swarm_stats перечисляет адреса, последний состав,
    /// «видели N мин назад» и вклад; состав переживает рестарт (KV).
    #[test]
    fn swarm_stats_line_readable() {
        let dir = tmp("swarm-stats");
        let mut core = PhoneCore::open(&dir).unwrap();
        let s = core.swarm_stats();
        assert!(s.contains("адресов 0"), "{s}");
        assert!(s.contains("ещё не видели"), "{s}");
        assert!(s.contains("прокси ≈ 0 шагов KD"), "{s}");
        assert!(s.contains("последняя эра сети: нет"), "{s}");
        // адрес роя + след probe: состав и время — как их пишет net_probe
        core.add_peer("10.0.2.2:9300").unwrap();
        let mut rb = Vec::new();
        for id in [3u16, 7, 11] {
            rb.extend_from_slice(&id.to_le_bytes());
        }
        core.kv.put(P_LAST_ROSTER, &rb).unwrap();
        core.kv.put(P_LAST_SEEN, &(now_ts() - 120).to_le_bytes()).unwrap();
        // след сетевого вклада: последняя эра сети (пишет net_tick)
        core.kv.put("рой:net_done", &559u32.to_le_bytes()).unwrap();
        let s = core.swarm_stats();
        assert!(s.contains("адресов 1 (10.0.2.2:9300)"), "{s}");
        assert!(s.contains("[3, 7, 11] — 3 узл."), "{s}");
        assert!(s.contains("видели 2 мин назад"), "{s}");
        assert!(s.contains("последняя эра сети: 559"), "{s}");
        // состав/seen переживают рестарт (рой-KV)
        drop(core);
        let core = PhoneCore::open(&dir).unwrap();
        assert!(core.swarm_stats().contains("[3, 7, 11]"), "рестарт");
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

    #[test]
    fn peers_kv_and_validation() {
        let dir = tmp("peers");
        let mut core = PhoneCore::open(&dir).unwrap();
        assert!(core.net_tick(1).is_err(), "без адресов роя — честная ошибка");
        assert!(core.add_peer("127.0.0.1:notaport").is_err());
        assert!(core.add_peer("example.com:80").is_err(), "хостнеймы не парсим");
        core.add_peer("10.0.2.2:9300").unwrap();
        core.add_peer("10.0.2.2:9300").unwrap(); // дубликат — не пишется
        core.add_peer("/ip6/::1/tcp/9301").unwrap(); // полный multiaddr тоже ок
        assert_eq!(core.peers(), vec!["10.0.2.2:9300", "/ip6/::1/tcp/9301"]);
        // адреса переживают рестарт (рой-KV)
        drop(core);
        let core = PhoneCore::open(&dir).unwrap();
        assert_eq!(core.peers().len(), 2);
    }

    /// Сетевой вклад против недостижимого роя: мост поднимается, dial
    /// промахивается, волна вымирает → честная ошибка «мало пиров»
    /// (воркер просто выходит, без паник и висяков).
    /// Взаимоисключение с другими СЕТЕВЫМИ тестами (NET_LOCK): мост
    /// телефона — фиксированный PeerId `[id;32]`, два параллельных
    /// ломают друг друга, а mdns на loopback переливает Hello между
    /// тестами (потому же НЕ гонять крейт параллельно с живым
    /// swarmkeeper/эмулятором — mdns найдёт настоящий рой).
    #[test]
    fn net_tick_unreachable_degrades_cleanly() {
        let _net = crate::tests::NET_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tmp("net-unreach");
        let mut core = PhoneCore::open(&dir).unwrap();
        core.add_peer("127.0.0.1:1").unwrap(); // закрытый порт
        let err = core.net_tick(1).unwrap_err();
        assert!(err.contains("мало пиров"), "err: {err}");
        // счётчики не тронуты
        assert_eq!(core.eras, 0);
        assert_eq!(core.cps, 0);
    }
}
