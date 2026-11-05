//! lorasync — РОЕВОЕ РАСПРОСТРАНЕНИЕ LoRA-АДАПТЕРОВ (файлы-опыты узлов).
//!
//! train-cpp пишет у клиента `<dir>.lora.gguf` (head-LoRA на его диалогах).
//! Это ЕДИНИЦА ОПЫТА: файл мал (r=8 ≈ 5 МиБ), content-addressed и
//! делим на куски — torrent-релей поверх ТЕХ ЖЕ кадров пир-обмена
//! весами, что и чек-пойнты (канал CH_BLOB, кадры
//! BlobNeed/BlobHave/BlobChunkReq/BlobChunk — БЕЗ изменений протокола).
//!
//! Правила тика (одна фаза, как у infosync):
//!   1. СВОЙ: свежий файл упакован в DeltaManifest → broadcast BlobHave
//!      (объявка на версию; повтор — раз в REANNOUNCE_EVERY тиков:
//!      поздний новичок догоняется, сеть переживает растворение анонса).
//!   2. ЧУЖОЙ BlobHave: канон хэш-связан с id (sha256(canon) == id);
//!      цели держим НА отправителя: свежевиденная версия заменяет
//!      старую закачку (устаревшие куски выбрасываются — догоняем
//!      новейшую), повторный анонс той же версии куски НЕ сбрасывает.
//!   3. ЧУЖОЙ BlobNeed: держим блоб (свой или собранный чужой) →
//!      направленный BlobHave (путь позднего самопроса).
//!   4. ЧУЖОЙ BlobChunkReq: отдаём кусок из СВОЕГО файла или из
//!      РЕЛЕЙ-КЭША собранных чужих — узел БЕЗ модели тоже может быть
//!      сидом (форвард чужого материала; капа RELAY_CAP — от раздувания).
//!   5. ЧУЖОЙ BlobChunk: verify_chunk по хэшу манифеста (подмена куска
//!      физически невозможна); собирается → сборка → проверка sha256
//!      ВСЕГО тела → файл `{prefix}.lora-peer-{hex8}.gguf` (tmp+rename)
//!      → релей-кэш → done (старые закачки того же манифеста сняты).
//!
//! Честность: чужой адаптер ПРИНИМАЕТСЯ как материал (файл + счётчик),
//! но НЕ применяется к модели — решение «чужому обучению доверить
//! генерацию» это отдельный шаг роя (eval/кворум чек-пойнтов, дальше).
//! Квоты за тик (REQ/SERVE) — вежливость: никто не заливает сеть целиком.

use crate::infosync::{frame_from_wire, signed_payload_bytes};
use dinet_crypto::{sha256, Signer};
use dinet_node::swarm::NodeId;
use dinet_node::transport::{Transport, CH_BLOB};
use dinet_proto::{MsgType, Payload};
use dinet_registry::DeltaManifest;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

/// Кусок LoRA-файла: 32 КиБ (адаптер r=8 ≈ 5 МиБ ≈ 150 кусков; кадр
/// gossipsub ≪ лимита 64 КиБ).
pub const LORA_CHUNK_LEN: u32 = 32 * 1024;
/// Сколько запросов кусков за один тик (вежливость просящего).
pub const REQ_PER_TICK: usize = 32;
/// Сколько кусков отдаём за один тик (вежливость сида).
pub const SERVE_PER_TICK: usize = 32;
/// Повтор объявки своего блоба — раз в столько тиков.
pub const REANNOUNCE_EVERY: u64 = 20;
/// Одновременных закачек (по отправителю) не держим больше.
pub const PULL_CAP: usize = 3;
/// Гон версий НЕ сбрасывает недокачанное: новая версия от того же
/// отправителя ждёт в приманке, пока текущая закачка доделается
/// (стоять без прогресса дольше STALL_TICKS тиков — дозволено бросить
/// и бежать за новейшей). Активный тренинг автора переписывает файл
/// каждые ~(12 шагов × train-цикл) — иначе вечный старт, куски никогда
/// не сходятся.
pub const STALL_TICKS: u32 = 6;
/// Собранных ЧУЖИХ блобов в релей-кэше (сид для третьих узлов).
pub const RELAY_CAP: usize = 2;
/// Рубеж множества done (защита роста).
pub const DONE_CAP: usize = 128;
/// Честная граница манифеста: >4096 кусков (>128 МиБ) — не наш масштаб.
const MAX_MANIFEST_CHUNKS: usize = 4096;
/// Честная граница тела: под slack-объявку гигантского блоба не ведёмся.
const MAX_BODY_BYTES: u64 = 512 * 1024 * 1024;

/// Домашний (свой) блоб узла: то, что отдаём в рой.
#[derive(Debug, Clone, PartialEq)]
pub struct HomeBlob {
    pub id: [u8; 32],
    pub manifest: DeltaManifest,
    pub chunks: Vec<Vec<u8>>,
}

/// Упаковать файл в манифест+куски (свой блоб).
pub fn home_pack(path: &Path) -> Result<HomeBlob, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("чтение {}: {e}", path.display()))?;
    let (manifest, chunks) = DeltaManifest::pack(&bytes, LORA_CHUNK_LEN);
    Ok(HomeBlob { id: manifest.id(), manifest, chunks })
}

/// Отчёт тика (лог/статы).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LoraReport {
    /// новая объявка своего блоба
    pub announced: u32,
    /// повторная объявка своего (тот же id)
    pub re_announced: u32,
    /// ответов BlobHave на чужой BlobNeed
    pub have_out: u32,
    /// отдали кусков (BlobChunk)
    pub serve: u32,
    /// выставили запросов кусков
    pub asked: u32,
    /// чужих кусков принято (verify_chunk сошёлся)
    pub taken: u32,
    /// мусор отброшен (чужой канон/кусок/манифест/брак сборки)
    pub rejected: u32,
    /// ЧУЖИХ адаптеров собрано ЦЕЛИКОМ за тик
    pub completed: u32,
    /// ЧУЖИХ собранных блобов ОБЬЯВЛЕНО в рой (диссеминация дальше)
    pub relay_announced: u32,
    /// последний собранный: длина тела (для лога)
    pub last_len: u64,
    /// последний собранный: pubkey отправителя (имя файла)
    pub last_sender: [u8; 32],
    /// кадров CH_BLOB прочли в тик (диагностика канала)
    pub frames_in: u32,
    /// из них со сломанной подписью/каноном (диагностика)
    pub frame_drop: u32,
}

/// Закачка у одного отправителя: манифест + куски + держатели.
#[derive(Debug, Clone)]
struct Pull {
    manifest: DeltaManifest,
    pieces: BTreeMap<usize, Vec<u8>>,
    holders: Vec<NodeId>,
    /// тик последней активности (для вымывания старых)
    seen: u64,
    /// тиков без принятых кусков подряд (гон версий кровливо ждёт)
    stall: u32,
}

impl Pull {
    fn complete(&self) -> bool {
        self.pieces.len() == self.manifest.n_chunks()
    }
}

/// Состояние релея между тиками (живёт в клиентском процессе).
#[derive(Debug)]
pub struct LoraSync {
    tick: u64,
    /// свой id, объявленный в сеть
    home_announced: [u8; 32],
    /// активные закачки, по отправителю (pub)
    pulls: BTreeMap<[u8; 32], Pull>,
    /// собранные манифесты (повторно не тянем)
    done: HashSet<[u8; 32]>,
    /// собранные ЧУЖИЕ блобы — сид для третьих узлов (кепка: старейшие)
    relay: VecDeque<([u8; 32], DeltaManifest, Vec<Vec<u8>>)>,
    /// чужие блобы, о которых объявлялись: id → тик (повтор по расписанию
    /// REANNOUNCE_EVERY — поздний новичок догоняет материал БЕЗ автора)
    relay_announced: BTreeMap<[u8; 32], u64>,
    /// приманка гонa версий: новейший манифест отправителя, ждущий,
    /// пока текущая закачка доделается или устанет стоять
    lure: BTreeMap<[u8; 32], (DeltaManifest, Vec<NodeId>)>,
}

impl Default for LoraSync {
    fn default() -> Self {
        LoraSync::new()
    }
}

impl LoraSync {
    pub fn new() -> LoraSync {
        LoraSync {
            tick: 0,
            home_announced: [0u8; 32],
            pulls: BTreeMap::new(),
            done: HashSet::new(),
            relay: VecDeque::new(),
            relay_announced: BTreeMap::new(),
            lure: BTreeMap::new(),
        }
    }

    /// Активных закачек (тесты/статы).
    pub fn pulling(&self) -> usize {
        self.pulls.len()
    }

    /// Блобов в релей-кэше (тесты/статы).
    pub fn relayed(&self) -> usize {
        self.relay.len()
    }
}

/// Хвост имени из pubkey: первые 4 байта в hex (8 знаков).
fn peer_hex(pub32: &[u8; 32]) -> String {
    pub32.iter().take(4).map(|b| format!("{b:02x}")).collect()
}

/// Путь файла чужого адаптера: `{prefix}.lora-peer-{hex}.gguf` — сиблинг
/// `<dir>.lora.gguf` (модель памяти префиксная, как `.info`/`.roi`).
pub fn foreign_path(prefix: &Path, sender: &[u8; 32]) -> PathBuf {
    Path::new(prefix).with_extension(format!("lora-peer-{}.gguf", peer_hex(sender)))
}

/// Записать файл атомарно (tmp + rename) — как train-адаптер/рой-KV.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("gguf.tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("tmp: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename: {e}"))
}

/// Один тик релея поверх канала CH_BLOB.
pub fn lora_exchange_tick(
    key: &dyn Signer,
    me_id: NodeId,
    me_pub: &[u8; 32],
    home: Option<&HomeBlob>,
    foreign_prefix: Option<&Path>,
    st: &mut LoraSync,
    t: &mut dyn Transport,
) -> LoraReport {
    st.tick += 1;
    let mut rep = LoraReport::default();

    // ── 1. Сбор чужого ─────────────────────────────────────────────
    let wire: Vec<Vec<u8>> = t.collect(CH_BLOB, me_id);
    rep.frames_in = wire.len() as u32;
    // (диагностика роя) DI_NET_LORA_WIRE=1 — дамп первых 4 кадров тика
    let mut wire_left = if std::env::var("DI_NET_LORA_WIRE").is_ok() { 4 } else { 0 };
    for raw in wire {
        let Some(frame) = frame_from_wire(&raw) else {
            rep.frame_drop += 1;
            continue;
        };
        if frame.sender == *me_pub {
            continue; // свой loopback
        }
        let from = u16::from_be_bytes([frame.sender[0], frame.sender[1]]);
        if wire_left > 0 {
            wire_left -= 1;
            let kind = match &frame.payload {
                Payload::BlobNeed { .. } => "need",
                Payload::BlobHave { .. } => "have",
                Payload::BlobChunkReq { .. } => "req",
                Payload::BlobChunk { .. } => "chunk",
                _ => "иное",
            };
            let own = if frame.sender == *me_pub { " СВОЙ" } else { "" };
            eprintln!("[lora-wire] from={from} kind={kind}{own} байт={}", raw.len());
        }
        match frame.payload {
            // сид объявил блоб → целевая закачка per-отправитель
            Payload::BlobHave { manifest_id, manifest_canon } => {
                if sha256(&manifest_canon) != manifest_id {
                    rep.rejected += 1;
                    continue;
                }
                let Ok(m) = DeltaManifest::decode(&manifest_canon) else {
                    rep.rejected += 1;
                    continue;
                };
                if m.id() != manifest_id {
                    rep.rejected += 1;
                    continue;
                }
                if m.n_chunks() > MAX_MANIFEST_CHUNKS || m.total > MAX_BODY_BYTES {
                    rep.rejected += 1; // распух — не наш масштаб
                    continue;
                }
                if st.done.contains(&manifest_id)
                    || st.relay.iter().any(|(rid, _, _)| *rid == manifest_id)
                    || home.map(|h| h.id == manifest_id).unwrap_or(false)
                {
                    continue; // уж собран / свой собственный — тихо
                }
                let prev = st.pulls.get(&frame.sender).cloned();
                match prev {
                    Some(mut p) if p.manifest.id() == manifest_id => {
                        // та же версия: держателя добавляем, куски НЕ сбрасываем
                        if !p.holders.contains(&from) {
                            p.holders.push(from);
                        }
                        p.seen = st.tick;
                        st.pulls.insert(frame.sender, p);
                    }
                    Some(mut p) => {
                        // ГОН ВЕРСИЙ: недокачанное не бросаем — новейшая
                        // ждёт в приманке, пока текущая доделается или
                        // устанет стоять без прогресса (STALL_TICKS)
                        let entry =
                            st.lure.entry(frame.sender).or_insert((m.clone(), Vec::new()));
                        if entry.0.id() != p.manifest.id() {
                            *entry = (m, Vec::new());
                        }
                        if !entry.1.contains(&from) {
                            entry.1.push(from);
                        }
                        p.seen = st.tick;
                    }
                    None => {
                        // новая цель: открываем закачку сразу
                        st.pulls.insert(
                            frame.sender,
                            Pull {
                                manifest: m,
                                pieces: BTreeMap::new(),
                                holders: vec![from],
                                seen: st.tick,
                                stall: 0,
                            },
                        );
                    }
                }
                // капа закачек: старейшие по активности вымываются
                while st.pulls.len() > PULL_CAP {
                    let Some((_, vic)) = st.pulls.iter().map(|(k, p)| (p.seen, *k)).min() else {
                        break; // рассинхрон — вставки ниже чинят
                    };
                    st.pulls.remove(&vic);
                }
            }
            // просят манифест → даём канон, если держим блоб
            Payload::BlobNeed { manifest_id } => {
                let canon = if home.map(|h| h.id == manifest_id).unwrap_or(false) {
                    Some(home.expect("сверено").manifest.encode())
                } else {
                    st.relay
                        .iter()
                        .find(|(rid, _, _)| *rid == manifest_id)
                        .map(|(_, m, _)| m.encode())
                };
                let Some(canon) = canon else { continue };
                let out = signed_payload_bytes(
                    key,
                    MsgType::BlobHave,
                    Payload::BlobHave { manifest_id, manifest_canon: canon },
                );
                if let Ok(bytes) = out {
                    t.send(CH_BLOB, me_id, Some(from), &bytes);
                    rep.have_out += 1;
                }
            }
            // просят кусок → отдаём из своего файла или релей-кэша
            Payload::BlobChunkReq { manifest_id, idx } => {
                if rep.serve >= SERVE_PER_TICK as u32 {
                    continue;
                }
                let bytes = if home.map(|h| h.id == manifest_id).unwrap_or(false) {
                    home.expect("сверено").chunks.get(idx as usize).cloned()
                } else {
                    st.relay
                        .iter()
                        .find(|(rid, _, _)| *rid == manifest_id)
                        .and_then(|(_, _, c)| c.get(idx as usize).cloned())
                };
                let Some(bytes) = bytes else { continue };
                if bytes.is_empty() {
                    continue;
                }
                let out = signed_payload_bytes(
                    key,
                    MsgType::BlobChunk,
                    Payload::BlobChunk { manifest_id, idx, bytes },
                );
                if let Ok(wire) = out {
                    // широковещательно: направленные темы gossipsub держит
                    // на «fanout» — ненадёжно (наблюдаемо роняло потоки
                    // кусков); mesh темы 'all' устойчив. Приёмники сами
                    // фильтруют по манифесту (нужный id — кусок, чужой —
                    // мусор); вежливость остаётся в квоте за тик.
                    t.send(CH_BLOB, me_id, None, &wire);
                    rep.serve += 1;
                }
            }
            // кусок пришёл → верифицируем хэшем манифеста, копим
            Payload::BlobChunk { manifest_id, idx, bytes } => {
                if std::env::var("DI_NET_LORA_WIRE").is_ok() {
                    let pid: Vec<String> = st
                        .pulls
                        .values()
                        .map(|p| format!("{:08x}", u32::from_be_bytes(
                            [p.manifest.id()[0], p.manifest.id()[1], p.manifest.id()[2], p.manifest.id()[3]])))
                        .collect();
                    let fid = u32::from_be_bytes([
                        manifest_id[0], manifest_id[1], manifest_id[2], manifest_id[3]]);
                    eprintln!(
                        "[lora-wire] chunk idx={idx} тело={fid:08x} тяну={:?}",
                        pid
                    );
                }
                // проверка+наполнение в блоке: заём st снят до сборки
                let snap = {
                    let owner = st
                        .pulls
                        .iter()
                        .find(|(_, p)| p.manifest.id() == manifest_id)
                        .map(|(k, _)| *k);
                    let Some(sk) = owner else {
                        continue; // незнакомый манифест — не наш запрос
                    };
                    let Some(p) = st.pulls.get_mut(&sk) else { continue };
                    if !p.manifest.verify_chunk(idx as usize, &bytes) {
                        rep.rejected += 1;
                        continue; // подмена — брак
                    }
                    if p.pieces.insert(idx as usize, bytes).is_none() {
                        rep.taken += 1;
                    }
                    if p.complete() {
                        Some(p.clone())
                    } else {
                        None
                    }
                };
                if let Some(done_pull) = snap {
                    // сборка + честная проверка ВЕСЬ тела (по хэшу-ид)
                    let mut body = Vec::with_capacity(done_pull.manifest.total as usize);
                    let mut ok = true;
                    for i in 0..done_pull.manifest.n_chunks() {
                        match done_pull.pieces.get(&i) {
                            Some(c) => body.extend_from_slice(c),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok || sha256(&body) != done_pull.manifest.body_id {
                        rep.rejected += 1;
                        continue; // брак: закачка висит — переспросим
                    }
                    // файл чужого адаптера: префикс памяти, tmp+rename
                    if let Some(prefix) = foreign_prefix {
                        let path = foreign_path(prefix, &frame.sender);
                        if atomic_write(&path, &body).is_err() {
                            rep.rejected += 1;
                            continue; // не записан — счётчик честно не растёт
                        }
                    }
                    // релей-кэш: собранное держим сидом для третьих (кепка)
                    let chunks: Vec<Vec<u8>> = (0..done_pull.manifest.n_chunks())
                        .map(|i| done_pull.pieces[&i].clone())
                        .collect();
                    let id = done_pull.manifest.id();
                    st.relay.push_front((id, done_pull.manifest.clone(), chunks));
                    while st.relay.len() > RELAY_CAP {
                        st.relay.pop_back();
                    }
                    rep.last_len = done_pull.manifest.total;
                    rep.last_sender = frame.sender;
                    st.done.insert(id);
                    if st.done.len() > DONE_CAP {
                        st.done.clear(); // рубеж: переспросы — сеть вежлива
                    }
                    // приманка этого же манифеста больше не ждёт
                    st.lure.retain(|_, (m, _)| m.id() != id);
                    rep.completed += 1;
                    // снятие закачек этого манифеста (любых отправителей)
                    let same: Vec<[u8; 32]> = st
                        .pulls
                        .iter()
                        .filter(|(_, q)| q.manifest.id() == id)
                        .map(|(k, _)| *k)
                        .collect();
                    for k in same {
                        st.pulls.remove(&k);
                    }
                }
            }
            _ => {}
        }
    }

    // ── 2. ГОН ВЕРСИЙ: уСТАВШУЮ закачку заменяет приманка отправителя ─
    let stall_keys: Vec<[u8; 32]> = st
        .pulls
        .iter()
        .filter(|(_, p)| p.stall >= STALL_TICKS)
        .map(|(k, _)| *k)
        .collect();
    for k in stall_keys {
        if let Some((m, holders)) = st.lure.remove(&k) {
            st.pulls.insert(
                k,
                Pull { manifest: m, pieces: BTreeMap::new(), holders, seen: st.tick, stall: 0 },
            );
        }
    }
    // прогресс: куски шли — все живы; тишина — тает терпение
    if rep.taken > 0 {
        for (_, p) in st.pulls.iter_mut() {
            p.stall = 0;
        }
    } else {
        for (_, p) in st.pulls.iter_mut() {
            p.stall = p.stall.saturating_add(1);
        }
    }

    // ── 3. Объявка своего ──────────────────────────────────────────
    if let Some(h) = home {
        if h.id != st.home_announced {
            let out = signed_payload_bytes(
                key,
                MsgType::BlobHave,
                Payload::BlobHave {
                    manifest_id: h.id,
                    manifest_canon: h.manifest.encode(),
                },
            );
            if let Ok(bytes) = out {
                t.send(CH_BLOB, me_id, None, &bytes);
                st.home_announced = h.id;
                rep.announced = 1;
            }
        } else if st.tick % REANNOUNCE_EVERY == 0 {
            let out = signed_payload_bytes(
                key,
                MsgType::BlobHave,
                Payload::BlobHave {
                    manifest_id: h.id,
                    manifest_canon: h.manifest.encode(),
                },
            );
            if let Ok(bytes) = out {
                t.send(CH_BLOB, me_id, None, &bytes);
                rep.re_announced = 1;
            }
        }
    }

    // ДИССЕМИНАЦИЯ собранного ЧУЖОГО: сид объявляет блоб и ПОВТОРЯЕТ по
    // расписанию REANNOUNCE_EVERY — рой несёт материал дальше без автора,
    // и опоздавший участник всё равно догонит (по одному за тик — вежливо)
    let due = st
        .relay
        .iter()
        .find(|(rid, _, _)| {
            st.relay_announced
                .get(rid)
                .map_or(true, |last| st.tick.saturating_sub(*last) >= REANNOUNCE_EVERY)
        })
        .map(|(rid, m, _)| (*rid, m.clone()));
    if let Some((rid, rm)) = due {
        let out = signed_payload_bytes(
            key,
            MsgType::BlobHave,
            Payload::BlobHave {
                manifest_id: rid,
                manifest_canon: rm.encode(),
            },
        );
        if let Ok(bytes) = out {
            t.send(CH_BLOB, me_id, None, &bytes);
            st.relay_announced.insert(rid, st.tick);
            rep.relay_announced = 1;
        }
    }

    // ── 4. Вытягивание недостающих кусков (round-robin у держателей) ─
    let mut budget = REQ_PER_TICK;
    let keys: Vec<[u8; 32]> = st.pulls.keys().copied().collect();
    for k in keys {
        if budget == 0 {
            break;
        }
        let Some(p) = st.pulls.get_mut(&k) else { continue };
        p.seen = st.tick;
        let want: Vec<usize> = (0..p.manifest.n_chunks())
            .filter(|i| !p.pieces.contains_key(i))
            .collect();
        if want.is_empty() || p.holders.is_empty() {
            continue;
        }
        let m_canon = p.manifest.id();
        for &idx in &want {
            if budget == 0 {
                break;
            }
            let h = p.holders[(idx + p.seen as usize) % p.holders.len()];
            let out = signed_payload_bytes(
                key,
                MsgType::BlobChunkReq,
                Payload::BlobChunkReq { manifest_id: m_canon, idx: idx as u32 },
            );
            if let Ok(bytes) = out {
                t.send(CH_BLOB, me_id, Some(h), &bytes);
                budget -= 1;
                rep.asked += 1;
            }
        }
    }
    rep
}

// ---------------------------------------------------------------------------
// Тесты: чистый Transport-мок (как у infosync) — сеть и хэши без модели.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_crypto::Ed25519Signer;

    /// Шина «все слышат всех, кроме себя» (directed = только адресату).
    struct Bus {
        inboxes: BTreeMap<NodeId, Vec<Vec<u8>>>,
    }
    impl Bus {
        fn new(ids: &[NodeId]) -> Bus {
            Bus { inboxes: ids.iter().map(|i| (*i, Vec::new())).collect() }
        }
    }
    impl Transport for Bus {
        fn send(&mut self, _ch: u8, from: NodeId, to: Option<NodeId>, frame: &[u8]) {
            for (id, q) in self.inboxes.iter_mut() {
                if *id != from && to.map(|t| t == *id).unwrap_or(true) {
                    q.push(frame.to_vec());
                }
            }
        }
        fn collect(&mut self, _ch: u8, me: NodeId) -> Vec<Vec<u8>> {
            self.inboxes
                .get_mut(&me)
                .map(std::mem::take)
                .unwrap_or_default()
        }
    }

    /// Звезда: A—B и B—C связаны; C НЕ слышит A (чистый релей B).
    struct Star {
        ids: [NodeId; 3],
        a: Vec<Vec<u8>>,
        b: Vec<Vec<u8>>,
        c: Vec<Vec<u8>>,
    }
    impl Transport for Star {
        fn send(&mut self, _ch: u8, from: NodeId, to: Option<NodeId>, f: &[u8]) {
            let for_me = |id: NodeId| from != id && to.map(|t| t == id).unwrap_or(true);
            let [ida, idb, idc] = self.ids;
            if for_me(ida) {
                self.a.push(f.to_vec());
            }
            if for_me(idb) {
                self.b.push(f.to_vec());
            }
            // C слышит только B (кадры A мимо C)
            if from != ida && for_me(idc) {
                self.c.push(f.to_vec());
            }
        }
        fn collect(&mut self, _ch: u8, me: NodeId) -> Vec<Vec<u8>> {
            let [ida, idb, _] = self.ids;
            let src = match me {
                n if n == ida => &mut self.a,
                n if n == idb => &mut self.b,
                _ => &mut self.c,
            };
            std::mem::take(src)
        }
    }

    fn nid(seed: u8) -> NodeId {
        let key = Ed25519Signer::from_seed(&[seed; 32]);
        u16::from_be_bytes([key.pubkey()[0], key.pubkey()[1]])
    }

    fn na() -> NodeId {
        nid(11)
    }
    fn nb() -> NodeId {
        nid(22)
    }

    /// Детерминированные байты «адаптера» (каждый байт различим — подделки
    /// и перестановки ловятся хэшами).
    fn fake_lora(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        let mut x = 0x5eed_u32;
        for i in 0..n {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push((x ^ i as u32) as u8);
        }
        v
    }

    fn home_of(body: &[u8]) -> HomeBlob {
        let (m, c) = DeltaManifest::pack(body, LORA_CHUNK_LEN);
        HomeBlob { id: m.id(), manifest: m, chunks: c }
    }

    /// Двойной обмен до сборки: A — сид (свой файл), B — просящий.
    /// РЕЛЬТ: B получает ЧУЖОЙ адаптер ЦЕЛИКОМ; байты файла совпали;
    /// собранное — в релей-кэше (сид для третьих); счётчики честные.
    #[test]
    fn lora_relay_two_nodes_assemble_and_write_file() {
        let a_key = Ed25519Signer::from_seed(&[11u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[22u8; 32]);
        let mut bus = Bus::new(&[na(), nb()]);
        let body = fake_lora(70_000); // 3 куска по 32 КиБ
        let home = home_of(&body);
        let mut sa = LoraSync::new();
        let mut sb = LoraSync::new();
        let dir = std::env::temp_dir().join(format!(
            "dinet-lora-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::create_dir_all(&dir);
        let prefix = dir.join("mem");
        let mut done_files = 0u32;
        let mut taken_total = 0u32;
        let mut asks_total = 0u32;
        let mut announces_total = 0u32;
        for _ in 0..12 {
            let ra = lora_exchange_tick(
                &a_key, na(), &a_key.pubkey(), Some(&home), None, &mut sa, &mut bus,
            );
            announces_total += ra.announced + ra.re_announced;
            let rb = lora_exchange_tick(
                &b_key, nb(), &b_key.pubkey(), None, Some(&prefix), &mut sb, &mut bus,
            );
            taken_total += rb.taken;
            done_files += rb.completed;
            asks_total += rb.asked;
            if sb.done.contains(&home.id) {
                break;
            }
        }
        assert!(done_files >= 1, "адаптер собран за тики");
        assert_eq!(announces_total, 1, "своя объявка — ровно одна (дедуп id)");
        assert!(sb.done.contains(&home.id), "манифест переведён в done");
        let got = std::fs::read(foreign_path(&prefix, &a_key.pubkey()))
            .expect("файл чужого адаптера записан");
        assert_eq!(got, body, "собранные байты = оригиналу (sha256 тела сошёлся)");
        assert_eq!(sb.relayed(), 1, "собранное — в релей-кэше");
        assert!(taken_total >= 3, "куски действительно пришли: {taken_total}");
        assert!(asks_total >= 3, "запросы выставлялись: {asks_total}");
        // повторные тики пусты: done/BlobHave-повторы не ходят
        let ra2 = lora_exchange_tick(&a_key, na(), &a_key.pubkey(), Some(&home), None, &mut sa, &mut bus);
        assert_eq!(ra2.announced + ra2.re_announced, 0, "тишина после объявки");
        let rb2 = lora_exchange_tick(&b_key, nb(), &b_key.pubkey(), None, Some(&prefix), &mut sb, &mut bus);
        assert_eq!(rb2.taken + rb2.asked, 0, "повторных тяг нет");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Подмена куска: злоумышленник отправляет испорченный байт под
    /// манифестом сида — verify_chunk отвергает; честный сид довозит,
    /// сборка всё равно случается.
    #[test]
    fn lora_relay_rejects_forged_chunk() {
        let a_key = Ed25519Signer::from_seed(&[41u8; 32]);
        let evil_key = Ed25519Signer::from_seed(&[66u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[42u8; 32]);
        let mut bus = Bus::new(&[nid(41), nid(66), nid(42)]);
        let body = fake_lora(40_000);
        let home = home_of(&body);
        let m = home.manifest.clone();
        let mut sa = LoraSync::new();
        let mut sb = LoraSync::new();

        // A объявляет
        let _ = lora_exchange_tick(&a_key, nid(41), &a_key.pubkey(), Some(&home), None,
            &mut sa, &mut bus);
        // подделка: bit-flip внутри РЕАЛЬНОГО куска 0, своя подпись
        let mut forged = home.chunks[0].clone();
        forged[0] ^= 0x01;
        let frame = signed_payload_bytes(
            &evil_key,
            MsgType::BlobChunk,
            Payload::BlobChunk { manifest_id: m.id(), idx: 0, bytes: forged },
        )
        .unwrap();
        bus.send(CH_BLOB, nid(66), Some(nid(42)), &frame);

        // B тикает: отвергает подделку, докачивает у A, собирает
        let mut rejected = 0u32;
        let mut assembled_after = false;
        for _ in 0..12 {
            let _ = lora_exchange_tick(&a_key, nid(41), &a_key.pubkey(), Some(&home), None,
                &mut sa, &mut bus);
            let rb = lora_exchange_tick(&b_key, nid(42), &b_key.pubkey(), None, None, &mut sb, &mut bus);
            rejected += rb.rejected;
            if sb.done.contains(&m.id()) {
                assembled_after = true;
                break;
            }
        }
        assert!(rejected >= 1, "подделка учтена в rejected: {rejected}");
        assert!(assembled_after, "честный сид докачал и сборка состоялась");
    }

    /// Подмена манифеста: канон, чей sha256 ≠ manifest_id, отбрасывается,
    /// закачка не открывается.
    #[test]
    fn lora_relay_rejects_forged_manifest() {
        let b_key = Ed25519Signer::from_seed(&[44u8; 32]);
        let evil_key = Ed25519Signer::from_seed(&[77u8; 32]);
        let mut bus = Bus::new(&[nid(44), nid(77)]);
        let (m, _) = DeltaManifest::pack(&fake_lora(10_000), LORA_CHUNK_LEN);
        let mut canon = m.encode();
        canon[0] ^= 0x01; // ломаем канон: хэш ≠ id
        let frame = signed_payload_bytes(
            &evil_key,
            MsgType::BlobHave,
            Payload::BlobHave { manifest_id: m.id(), manifest_canon: canon },
        )
        .unwrap();
        bus.send(CH_BLOB, nid(77), None, &frame);
        let mut sb = LoraSync::new();
        let rb = lora_exchange_tick(&b_key, nid(44), &b_key.pubkey(), None, None, &mut sb, &mut bus);
        assert!(rb.rejected >= 1, "фальшивый манифест отброшен");
        assert_eq!(sb.pulling(), 0, "закачка не открылась");
    }

    /// Гон версий: недокачанное НЕ бросается на полдороге. A объявил v1
    /// и сразу переписал файл (сид держит только v2 — запросы кусков v1
    /// у него молчат); B терпит STALL_TICKS тиков без прогресса → берёт
    /// приманку (v2) и докачивает ИМЕННО её байты; v1 не собирается.
    #[test]
    fn lora_relay_chases_newest_version() {
        let a_key = Ed25519Signer::from_seed(&[51u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[55u8; 32]);
        let mut bus = Bus::new(&[nid(51), nid(55)]);
        let v1 = home_of(&fake_lora(45_000));
        let v2 = home_of(&fake_lora(90_000)); // новейшая, 3 куска
        let mut sa = LoraSync::new();
        let mut sb = LoraSync::new();

        // тик 1: A объявляет v1; B открывает закачку v1
        let _ = lora_exchange_tick(
            &a_key, nid(51), &a_key.pubkey(), Some(&v1), None, &mut sa, &mut bus,
        );
        let _ = lora_exchange_tick(
            &b_key, nid(55), &b_key.pubkey(), None, None, &mut sb, &mut bus,
        );
        assert_eq!(sb.pulling(), 1, "закачка v1 открыта");
        // A переписал файл: сид держит только v2 (запросы v1 молчат)
        let ra = lora_exchange_tick(
            &a_key, nid(51), &a_key.pubkey(), Some(&v2), None, &mut sa, &mut bus,
        );
        assert_eq!(ra.announced, 1, "новая версия объявляется");
        // B: без кусков STALL_TICKS тиков → приманка v2 → докачка → сборка
        let mut got_v2 = false;
        for _ in 0..(STALL_TICKS as usize + 8) {
            let _ = lora_exchange_tick(
                &a_key, nid(51), &a_key.pubkey(), Some(&v2), None, &mut sa, &mut bus,
            );
            let _ = lora_exchange_tick(
                &b_key, nid(55), &b_key.pubkey(), None, None, &mut sb, &mut bus,
            );
            if sb.done.contains(&v2.id) {
                got_v2 = true;
                break;
            }
        }
        assert!(got_v2, "новейшая версия собрана сверх приманки");
        assert!(!sb.done.contains(&v1.id), "устаревшая версия не собрана");
        assert_eq!(sb.pulling(), 0, "собранные закачки сняты");
    }

    /// РЕЛЕЙ-КЭШ: C слышит ТОЛЬКО B; B собрал у A и отдаёт свой кусок
    /// из кэша — сид может умереть, рой несёт материал дальше без автора.
    #[test]
    fn lora_relay_cache_serves_far_node() {
        let a_key = Ed25519Signer::from_seed(&[61u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[62u8; 32]);
        let c_key = Ed25519Signer::from_seed(&[63u8; 32]);
        let mut bus = Star {
            ids: [nid(61), nid(62), nid(63)],
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
        };
        let body = fake_lora(60_000);
        let home = home_of(&body);
        let mut sa = LoraSync::new();
        let mut sb = LoraSync::new();
        let mut sc = LoraSync::new();
        let mut got_c = false;
        for _ in 0..30 {
            let _ = lora_exchange_tick(
                &a_key, nid(61), &a_key.pubkey(), Some(&home), None, &mut sa, &mut bus,
            );
            let _ = lora_exchange_tick(
                &b_key, nid(62), &b_key.pubkey(), None, None, &mut sb, &mut bus,
            );
            let _ = lora_exchange_tick(
                &c_key, nid(63), &c_key.pubkey(), None, None, &mut sc, &mut bus,
            );
            if sc.done.contains(&home.id) {
                got_c = true;
                break;
            }
        }
        assert!(sb.done.contains(&home.id), "B собрал у A");
        assert!(got_c, "C собрал через релей B (кэш сида)");
        assert_eq!(sc.relayed(), 1, "собранное у C — тоже в кэше");
    }

    /// Диссеминация ПО РАСПИСАНИЮ: собранный чужой блоб объявляется рой
    /// и ПОВТОРЯЕТСЯ каждые REANNOUNCE_EVERY тиков — опоздавший участник
    /// догоняет материал БЕЗ автора (сид может умереть давно).
    #[test]
    fn lora_relay_dissemination_repeats_on_schedule() {
        let a_key = Ed25519Signer::from_seed(&[71u8; 32]);
        let c_key = Ed25519Signer::from_seed(&[73u8; 32]);
        let mut bus = Bus::new(&[nid(71), nid(73)]);
        let home = home_of(&fake_lora(40_000));
        let mut sa = LoraSync::new();
        let mut sc = LoraSync::new();
        let mut announces = 0u32;
        let mut completed = 0u32;
        for _ in 0..(REANNOUNCE_EVERY as usize + 5) {
            let ra = lora_exchange_tick(
                &a_key, nid(71), &a_key.pubkey(), Some(&home), None, &mut sa, &mut bus,
            );
            let _ = ra;
            let rc = lora_exchange_tick(
                &c_key, nid(73), &c_key.pubkey(), None, None, &mut sc, &mut bus,
            );
            completed += rc.completed;
            announces += rc.relay_announced;
        }
        assert!(completed >= 1, "материал C собрал у A");
        assert!(
            announces >= 2,
            "объявка релей-кэша повторилась по расписанию: {announces}"
        );
        assert_eq!(sa.pulling(), 0, "автор НЕ открывает закачку на свой же блоб");
        assert_eq!(sc.pulling(), 0, "собранные закачки сняты");
    }
}
