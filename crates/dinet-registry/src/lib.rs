//! dinet-registry — РЕЕСТР ЧЕК-ПОЙНТОВ роя: append-only лог с нотарными
//! печатями + кусочная репликация весов («torrent-схема»).
//!
//! Мандат «неостановимо»: никакой владелец не может откатить или подменить
//! цепочку чек-пойнтов. Механика:
//! 1. ЗАПИСЬ = чек-пойнт `{seq, id, parent, era, manifest_id, acc}` с
//!    кворумом подписей нотариусов (distinct pubkey; каждый подписывает
//!    канонические байты записи).
//! 2. ЦЕПОЧКА: `seq` монотонно растёт, `parent_id == head.id` — форк или
//!    откат не дописывается ни при каком давлении.
//! 3. ФАЙЛ — APPEND-ONLY: байты только добавляются в хвост; при открытии
//!    лог переверифицирует ВСЕ подписи и связи — мёртвый откат виден сразу.
//! 4. ВЕСА НЕ В ЛОГЕ: лог хранит только `manifest_id` (индекс). Сами байты
//!    дельты реплицируются кусками между пирами (`DeltaManifest`): каждый
//!    кусок проверяется SHA-256, сборка верифицирует тело.
//!
//! ```text
//! нотариусы (подписи)              весовые куски (у каждого узла свои)
//! ─────────────────> РЕЕСТР ─────> DeltaManifest ─────> куски у пиров
//!                    (индекс)        (хэши кусков)      (ask/have, v1)
//! ```

use dinet_crypto::{sha256, Verifier, ED25519_PUB_LEN, ED25519_SIG_LEN};
use std::io::Write;
use std::path::Path;

// ---------------------------------------------------------------------------
// Каноническое кодирование (детерминированное, кросс-платформенное)
// ---------------------------------------------------------------------------

/// Сборщик канонических байт (little-endian, только дополнение — того же
/// принципа append-only, что и сам реестр).
#[derive(Default)]
pub struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    pub fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub fn put_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put32(&mut self, v: &[u8; 32]) {
        self.buf.extend_from_slice(v);
    }
    pub fn put_bytes(&mut self, v: &[u8]) {
        self.put_u32(v.len() as u32);
        self.buf.extend_from_slice(v);
    }
    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }
}

/// Читатель канонических байт (строгий: недочитанный хвост = ошибка).
pub struct Dec<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    pub fn new(b: &'a [u8]) -> Dec<'a> {
        Dec { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() {
            return Err(format!(
                "канонический обрыв: нужно {n}, осталось {}",
                self.b.len() - self.pos
            ));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn arr32(&mut self) -> Result<[u8; 32], String> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    pub fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    /// Строгий конец: всё прочитано, лишнего хвоста нет (чужой мусор = брак).
    pub fn done(&mut self) -> Result<(), String> {
        if self.pos != self.b.len() {
            Err(format!("лишний хвост: прочитано {}, всего {}", self.pos, self.b.len()))
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaManifest — кусочная репликация весов (torrent-схема v1)
// ---------------------------------------------------------------------------

/// Кусок по умолчанию: 8 КиБ (toy-дельта — сотни байт; llama-веса дробятся
/// без дыхки).
pub const DEFAULT_CHUNK_LEN: u32 = 8192;

/// Манифест весового блоба: идентификатор тела + хэши кусков. Уходит по
/// сети; по нему пиры спрашивают/отдают куски; сам манифест канонично
/// хэшируем.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaManifest {
    pub body_id: [u8; 32],
    pub total: u64,
    pub chunk_len: u32,
    pub hashes: Vec<[u8; 32]>,
}

impl DeltaManifest {
    /// Разрезать тело на куски и построить манифест.
    pub fn pack(body: &[u8], chunk_len: u32) -> (DeltaManifest, Vec<Vec<u8>>) {
        let chunk_len = chunk_len.max(64).min(64 * 1024 * 1024) as usize;
        let chunks: Vec<Vec<u8>> = if body.is_empty() {
            Vec::new()
        } else {
            body.chunks(chunk_len).map(|c| c.to_vec()).collect()
        };
        let hashes = chunks.iter().map(|c| sha256(c)).collect();
        (
            DeltaManifest {
                body_id: sha256(body),
                total: body.len() as u64,
                chunk_len: chunk_len as u32,
                hashes,
            },
            chunks,
        )
    }

    /// Канонические байты манифеста.
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Enc::default();
        e.put32(&self.body_id);
        e.put_u64(self.total);
        e.put_u32(self.chunk_len);
        e.put_u32(self.hashes.len() as u32);
        for h in &self.hashes {
            e.put32(h);
        }
        e.bytes().to_vec()
    }

    pub fn decode(bytes: &[u8]) -> Result<DeltaManifest, String> {
        let mut d = Dec::new(bytes);
        let body_id = d.arr32()?;
        let total = d.u64()?;
        let chunk_len = d.u32()?;
        let n = d.u32()? as usize;
        if n > 1 << 22 {
            return Err(format!("манифест распух: {n} кусков"));
        }
        let mut hashes = Vec::with_capacity(n);
        for _ in 0..n {
            hashes.push(d.arr32()?);
        }
        d.done()?;
        Ok(DeltaManifest { body_id, total, chunk_len, hashes })
    }

    pub fn id(&self) -> [u8; 32] {
        sha256(&self.encode())
    }

    pub fn n_chunks(&self) -> usize {
        self.hashes.len()
    }

    /// Проверить кусок против хэша манифеста.
    pub fn verify_chunk(&self, idx: usize, chunk: &[u8]) -> bool {
        match self.hashes.get(idx) {
            None => false,
            Some(h) => chunk.len() <= self.chunk_len as usize && &sha256(chunk) == h,
        }
    }

    /// Собрать тело из (непоследовательных!) кусков; КАЖДЫЙ кусок
    /// пере-верифицируется, финал — против body_id. Куски от РАЗНЫХ пиров
    /// перемешивать безопасно.
    pub fn assemble(&self, pieces: &[(usize, Vec<u8>)]) -> Result<Vec<u8>, String> {
        if pieces.len() != self.n_chunks() {
            return Err(format!("кусков {} из {}", pieces.len(), self.n_chunks()));
        }
        let mut out = vec![0u8; self.total as usize];
        let mut filled = 0usize;
        for (idx, c) in pieces {
            if !self.verify_chunk(*idx, c) {
                return Err(format!("кусок #{idx} не прошёл SHA-256"));
            }
            let off = *idx as u64 * self.chunk_len as u64;
            let end = off + c.len() as u64;
            if end > self.total {
                return Err(format!("кусок #{idx} вылез за тело"));
            }
            out[off as usize..end as usize].copy_from_slice(c);
            filled += c.len();
        }
        if filled as u64 != self.total {
            Err(format!("собрано {filled} из {} байт", self.total))
        } else if sha256(&out) != self.body_id {
            Err("тело не совпало с body_id".to_string())
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Реестр: записи, печати, строго-линейная цепочка
// ---------------------------------------------------------------------------

/// Родитель первого чек-пойнта.
pub const GENESIS: [u8; 32] = [0u8; 32];

/// Запись реестра: индекс чек-пойнта (веса снаружи — по manifest_id).
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryEntry {
    pub seq: u64,
    pub checkpoint_id: [u8; 32],
    pub parent_id: [u8; 32],
    pub era_id: [u8; 32],
    pub manifest_id: [u8; 32],
    pub accuracy: f32,
    pub contributors: u16,
}

impl RegistryEntry {
    /// Канонические байты — ровно то, что подписывают нотариусы.
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Enc::default();
        e.put_u64(self.seq);
        e.put32(&self.checkpoint_id);
        e.put32(&self.parent_id);
        e.put32(&self.era_id);
        e.put32(&self.manifest_id);
        e.put_f32(self.accuracy);
        e.put_u16(self.contributors);
        e.bytes().to_vec()
    }

    pub fn decode(bytes: &[u8]) -> Result<RegistryEntry, String> {
        let mut d = Dec::new(bytes);
        let e = RegistryEntry {
            seq: d.u64()?,
            checkpoint_id: d.arr32()?,
            parent_id: d.arr32()?,
            era_id: d.arr32()?,
            manifest_id: d.arr32()?,
            accuracy: d.f32()?,
            contributors: d.u16()?,
        };
        d.done()?;
        Ok(e)
    }
}

/// Нотарная печать: (pubkey, подпись-канона) над `entry.encode()`.
#[derive(Debug, Clone, PartialEq)]
pub struct Seal {
    pub pubkey: [u8; ED25519_PUB_LEN],
    pub sig: [u8; ED25519_SIG_LEN],
}

/// Принятая запись: entry + печати нотариусов (distinct pubkey).
#[derive(Debug, Clone, PartialEq)]
pub struct Accepted {
    pub entry: RegistryEntry,
    pub seals: Vec<Seal>,
}

/// Счётчик кворума эр: accepts*3 >= votes*2 (тот же инвариант).
pub fn two_thirds_of(distinct: usize, total: usize) -> bool {
    distinct * 3 >= total.max(1) * 2
}

/// Реестр-лог: строго-линейная цепочка под кворумом нотариусов,
/// append-only файл со сквозной переверификацией при открытии.
pub struct RegistryLog {
    accepted: Vec<Accepted>,
    notary_min: usize,
    path: std::path::PathBuf,
}

impl RegistryLog {
    /// Открыть (создать) лог; если файл есть — прочитать и ПЕРЕВЕРИФИЦИРОВАТЬ
    /// всё (подписи, цепочку, seq). `notary_min` — минимум distinct нотариусов
    /// для каждой новой записи; уже несовпадающие старые записи пролезли
    /// бы при ВЛАДЕЛЬЦЕ файла — потому не проверяются заново против минимума,
    /// но подписи проверяются всегда.
    pub fn open(path: &Path, notary_min: usize) -> Result<RegistryLog, String> {
        let mut log =
            RegistryLog { accepted: Vec::new(), notary_min, path: path.to_path_buf() };
        if path.exists() {
            let raw = std::fs::read(path).map_err(|e| format!("реестр-файл: {e}"))?;
            for rec in read_records(&raw)? {
                log.check_and_push(&rec)
                    .map_err(|e| format!("реестр {} повреждён: {e}", path.display()))?;
            }
        } else if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| format!("реестр-каталог: {e}"))?;
            }
        }
        Ok(log)
    }

    pub fn len(&self) -> usize {
        self.accepted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.accepted.is_empty()
    }

    pub fn head(&self) -> Option<&Accepted> {
        self.accepted.last()
    }

    pub fn entries(&self) -> &[Accepted] {
        &self.accepted
    }

    /// Проверить запись+печати и ДОПИСАТЬ в хвост. Ошибки:
    /// seq не продолжил цепочку; parent ≠ head (форк/откат); брак подписи;
    /// кворум не добран; checkpoint_id уже в реестре.
    pub fn append(&mut self, rec: Accepted) -> Result<(), String> {
        if self.accepted.iter().any(|a| a.entry.checkpoint_id == rec.entry.checkpoint_id) {
            return Err(format!(
                "чек-пойнт {:02x?} уже в реестре",
                &rec.entry.checkpoint_id[..4]
            ));
        }
        self.check_and_push(&rec)?;
        self.persist(&rec)
    }

    fn check_and_push(&mut self, rec: &Accepted) -> Result<(), String> {
        let e = &rec.entry;
        // 1) цепочка: seq строго вперёд, parent строго от head
        let (want_seq, want_parent) = match self.accepted.last() {
            None => (0u64, GENESIS),
            Some(a) => (a.entry.seq + 1, a.entry.checkpoint_id),
        };
        if e.seq != want_seq {
            return Err(format!("seq {}: ожидался {want_seq}", e.seq));
        }
        if e.parent_id != want_parent {
            return Err(format!(
                "parent {:02x?}: ожидался {:02x?}",
                &e.parent_id[..4],
                &want_parent[..4]
            ));
        }
        // 2) печати: валидные подписи, distinct pubkey, кворум
        let mut seen: Vec<[u8; ED25519_PUB_LEN]> = Vec::new();
        for s in &rec.seals {
            if seen.contains(&s.pubkey) {
                continue; // дубликат нотариуса не первый класс
            }
            let v = match dinet_crypto::Ed25519Verifier::new(s.pubkey) {
                Some(v) => v,
                None => return Err("плохой pubkey печати".to_string()),
            };
            if !v.verify(&e.encode(), &s.sig) {
                return Err("печать не прошла по канону записи".to_string());
            }
            seen.push(s.pubkey);
        }
        if seen.len() < self.notary_min {
            return Err(format!("кворум: {}/{}", seen.len(), self.notary_min));
        }
        self.accepted.push(rec.clone());
        Ok(())
    }

    fn persist(&self, rec: &Accepted) -> Result<(), String> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("реестр-открытие: {e}"))?;
        for rb in write_records(std::slice::from_ref(rec)) {
            f.write_all(&rb).map_err(|e| format!("реестр-запись: {e}"))?;
        }
        let _ = f.flush();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Сетево-файловая репрезентация записи: [len u32][entry][count u32]{[pub][sig]}
// ---------------------------------------------------------------------------

/// ОДНА запись в сетевом формате (с len-префиксом). Тот же формат, что
/// хвост append-only файла: синхронизация между пирами и persist не
/// расходятся ни байтом.
pub fn encode_record(rec: &Accepted) -> Vec<u8> {
    let body = record_body(rec);
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Обратная и распаковка: сетевые байты записи → Accepted.
pub fn decode_record(bytes: &[u8]) -> Result<Accepted, String> {
    if bytes.len() < 4 {
        return Err("запись оборвана на длине".to_string());
    }
    let n = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    if bytes.len() < 4 + n {
        return Err("запись оборвана в теле".to_string());
    }
    parse_record_body(&bytes[4..4 + n])
}

fn record_body(rec: &Accepted) -> Vec<u8> {
    let mut e = Enc::default();
    e.put_bytes(&rec.entry.encode());
    e.put_u32(rec.seals.len() as u32);
    for s in &rec.seals {
        e.put32(&s.pubkey);
        e.buf.extend_from_slice(&s.sig);
    }
    e.bytes().to_vec()
}

fn parse_record_body(body: &[u8]) -> Result<Accepted, String> {
    let mut d = Dec::new(body);
    let entry = RegistryEntry::decode(&d.bytes()?)?;
    let count = d.u32()? as usize;
    if count > 1 << 16 {
        return Err(format!("печатей распухло: {count}"));
    }
    let mut seals = Vec::with_capacity(count);
    for _ in 0..count {
        let pubkey = d.arr32()?;
        let sig = d
            .take(ED25519_SIG_LEN)?
            .try_into()
            .map_err(|_| "сигнатура не 64 байта".to_string())?;
        seals.push(Seal { pubkey, sig });
    }
    d.done()?;
    Ok(Accepted { entry, seals })
}

fn write_records(recs: &[Accepted]) -> Vec<Vec<u8>> {
    recs.iter().map(encode_record).collect()
}

fn read_records(raw: &[u8]) -> Result<Vec<Accepted>, String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        if raw.len() - pos < 4 {
            return Err("лог оборван на длине записи".to_string());
        }
        let n = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if raw.len() - pos < n {
            return Err("лог оборван в теле записи".to_string());
        }
        out.push(parse_record_body(&raw[pos..pos + n])?);
        pos += n;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Тесты: перемешанная сборка, tamper, кворум, цепочка, рестарт
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_crypto::{Signer, SHA256_LEN};
    use dinet_crypto::Ed25519Signer;

    const DELTA_PARAMS: usize = 512;

    /// Стабильный «блоб весов»: param i = f(seed, i).
    fn delta_bytes(seed: u8) -> Vec<u8> {
        (0..DELTA_PARAMS)
            .flat_map(|i| ((i as f32) * 0.25 + seed as f32 * 3.0).to_le_bytes())
            .collect()
    }

    fn seal(s: &Ed25519Signer, canon: &[u8]) -> Seal {
        Seal { pubkey: s.pubkey(), sig: Signer::sign(s, canon) }
    }

    fn notaries(n: usize) -> Vec<Ed25519Signer> {
        (0..n).map(|i| Ed25519Signer::from_seed(&[i as u8; 32])).collect()
    }

    fn entry(seq: u64, parent: [u8; 32], m: &DeltaManifest) -> RegistryEntry {
        RegistryEntry {
            seq,
            checkpoint_id: sha256(&m.encode()),
            parent_id: parent,
            era_id: sha256(&seq.to_le_bytes()),
            manifest_id: m.id(),
            accuracy: 0.5 + 0.01 * seq as f32,
            contributors: 4,
        }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("dinet-reg-{tag}-{}", std::process::id()))
    }

    #[test]
    fn manifest_reassembles_from_shuffled_peer_chunks() {
        let body = delta_bytes(7);
        let (m, chunks) = DeltaManifest::pack(&body, 1024);
        // куски будто от нескольких реплик, в перемешанном порядке
        let mut order: Vec<(usize, Vec<u8>)> = chunks.into_iter().enumerate().collect();
        order.reverse();
        let body2 = m.assemble(&order).expect("сборка из перемешанных кусков");
        assert_eq!(body2, body, "сборка не совпала с телом");
        assert_ne!(m.id(), [0; SHA256_LEN], "id манифеста не вычислен");
    }

    #[test]
    fn tampered_chunk_is_caught() {
        let (m, mut chunks) = DeltaManifest::pack(&delta_bytes(3), 1024);
        chunks[1][117] ^= 0xFF;
        let pieces: Vec<(usize, Vec<u8>)> = chunks.into_iter().enumerate().collect();
        assert!(m.assemble(&pieces).is_err(), "порча куска прошла проверку");
    }

    #[test]
    fn append_requires_chain_and_quorum() {
        let dir = tmp("chain");
        let path = dir.join("reg.dlog");
        let _ = std::fs::remove_dir_all(&dir);
        let mut log = RegistryLog::open(&path, 3).expect("открытие");
        let (m, _) = DeltaManifest::pack(&delta_bytes(1), 4096);
        let ns = notaries(3);

        // 2 подписи из 3 — кворум не добран
        let e0 = entry(0, GENESIS, &m);
        let short: Vec<Seal> = ns.iter().take(2).map(|s| seal(s, &e0.encode())).collect();
        assert!(
            log.append(Accepted { entry: e0.clone(), seals: short }).is_err(),
            "двое работников проскочили кворум 3"
        );
        // 3 из 3 — принято, цепочка начинается от genesis
        let full: Vec<Seal> = ns.iter().map(|s| seal(s, &e0.encode())).collect();
        log.append(Accepted { entry: e0, seals: full }).expect("первая запись");

        // вторая запись С ЧУЖИМ parent отмечена как форк — отказ
        let e_fork = entry(1, sha256(b"fork-parent"), &m);
        let s_fork: Vec<Seal> = ns.iter().map(|s| seal(s, &e_fork.encode())).collect();
        assert!(log.append(Accepted { entry: e_fork, seals: s_fork }).is_err());

        // правильная цепочка — принята (другая дельта → другой checkpoint_id)
        let head_id = log.head().expect("head").entry.checkpoint_id;
        let (m2, _) = DeltaManifest::pack(&delta_bytes(2), 4096);
        let e1 = entry(1, head_id, &m2);
        let s1: Vec<Seal> = ns.iter().map(|s| seal(s, &e1.encode())).collect();
        log.append(Accepted { entry: e1, seals: s1 }).expect("вторая запись");

        // повтор уже существующего checkpoint_id — отказ (дубль)
        let (dup_m, _) = DeltaManifest::pack(&delta_bytes(2), 4096); // тот же body → тот же id
        let dup_id = log.head().unwrap().entry.checkpoint_id;
        let mut dup = entry(2, dup_id, &dup_m);
        dup.checkpoint_id = dup_id;
        let s_dup: Vec<Seal> = ns.iter().map(|s| seal(s, &dup.encode())).collect();
        assert!(log.append(Accepted { entry: dup, seals: s_dup }).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forged_seal_is_caught() {
        let dir = tmp("forge");
        let path = dir.join("reg.dlog");
        let _ = std::fs::remove_dir_all(&dir);
        let mut log = RegistryLog::open(&path, 2).expect("открытие");
        let (m, _) = DeltaManifest::pack(&delta_bytes(9), 4096);
        let e0 = entry(0, GENESIS, &m);
        // подписи по ДРУГОМУ канону — tamper локально не проскочит
        let ns = notaries(2);
        let bad: Vec<Seal> = ns
            .iter()
            .map(|s| Seal { pubkey: s.pubkey(), sig: Signer::sign(s, b"foreign-canon") })
            .collect();
        assert!(log.append(Accepted { entry: e0, seals: bad }).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_survives_restart_and_rereverifies() {
        let dir = tmp("restart");
        let path = dir.join("reg.dlog");
        let _ = std::fs::remove_dir_all(&dir);
        let (m1, _) = DeltaManifest::pack(&delta_bytes(11), 4096);
        let (m2, _) = DeltaManifest::pack(&delta_bytes(12), 4096);
        {
            let mut log = RegistryLog::open(&path, 2).expect("открытие");
            let ns = notaries(2);
            for (i, m) in [&m1, &m2].iter().enumerate() {
                let parent = log.head().map(|h| h.entry.checkpoint_id).unwrap_or(GENESIS);
                let e = entry(i as u64, parent, m);
                let seals: Vec<Seal> = ns.iter().map(|s| seal(s, &e.encode())).collect();
                log.append(Accepted { entry: e, seals }).unwrap_or_else(|err| panic!("запись {i}: {err}"));
            }
        }
        // рестарт: лог перечитывает, переверифицирует все подписи и цепочку
        let log = RegistryLog::open(&path, 2).expect("повторное открытие");
        assert_eq!(log.len(), 2);
        assert_eq!(log.head().expect("head").entry.manifest_id, m2.id());
        assert_ne!(log.head().expect("head").entry.manifest_id, m1.id());

        // испорченный хвост (последняя печать) ловится при открытии
        let raw = std::fs::read(&path).expect("чтение лога");
        let mut tampered = raw.clone();
        let flip = raw.len() - 1;
        tampered[flip] ^= 0x01;
        std::fs::write(&path, &tampered).expect("запись tamper");
        assert!(RegistryLog::open(&path, 2).is_err(), "испорченная подпись принята");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_thirds_quorum_math() {
        // accepts*3 >= votes*2 (тот же инвариант, что у эр)
        assert!(two_thirds_of(2, 3));
        assert!(two_thirds_of(3, 3));
        assert!(!two_thirds_of(1, 3));
        assert!(two_thirds_of(1, 1));
    }
}
