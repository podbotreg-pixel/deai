//! Append-only KV-лог (v0, std-only).
//!
//! Файл = `DIKV1\n` + записи подряд. Запись:
//!
//! ```text
//! crc32(u32 LE) | flags(u8) | klen(u16 LE) | vlen(u32 LE) | key | val
//! ```
//!
//! CRC считается по `flags|klen|vlen|key|val`. Tombstone (flags=1) удаляет
//! ключ, значение пустое. Восстановление: реплей до первой «плохой» записи —
//! оборванный хвост (крушение питания посреди записи) обрезается физически.
//! Это классический контракт WAL: целостность > полнота.
//!
//! Компактация: живые записи переписываются в `*.tmp` и переименовываются,
//! tombstones и мусор исчезают. Оффсеты держатся в BTreeMap — чтение O(log n)
//! по ключу, scan по префиксу отсортированный.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use dinet_crypto::crc32;

const HEADER: &[u8; 6] = b"DIKV1\n";
const FLAG_TOMB: u8 = 1;

#[derive(Debug, Clone, Copy)]
struct Slot {
    val_off: u64,
    vlen: u32,
}

/// Локальное персистентное хранилище ключ-значение (append-only).
#[derive(Debug)]
pub struct KvStore {
    path: PathBuf,
    // Хэндл для append-записи (seek на tail) и служебных перечитываний;
    // чтенияValues по ключу открывают свежий read-хэндл (get — &self).
    file: File,
    tail: u64,
    index: BTreeMap<String, Slot>,
}

fn encode_record(flags: u8, key: &str, val: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(7 + key.len() + val.len());
    body.push(flags);
    body.extend_from_slice(&(key.len() as u16).to_le_bytes());
    body.extend_from_slice(&(val.len() as u32).to_le_bytes());
    body.extend_from_slice(key.as_bytes());
    body.extend_from_slice(val);
    let mut rec = Vec::with_capacity(body.len() + 4);
    rec.extend_from_slice(&crc32(&body).to_le_bytes());
    rec.extend_from_slice(&body);
    rec
}

/// Реплей буфера файла → (индекс, хвост «хороших» байт / offset первой плохой).
fn replay(buf: &[u8]) -> (BTreeMap<String, Slot>, u64) {
    let mut index = BTreeMap::new();
    let len = buf.len();
    let mut pos = HEADER.len();
    while pos < len {
        // заголовок записи: crc(4) + flags(1) + klen(2) + vlen(4) = 11 байт
        if len - pos < 11 {
            return (index, pos as u64);
        }
        let crc = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        let flags = buf[pos + 4];
        let klen = u16::from_le_bytes([buf[pos + 5], buf[pos + 6]]) as usize;
        let vlen = u32::from_le_bytes([
            buf[pos + 7],
            buf[pos + 8],
            buf[pos + 9],
            buf[pos + 10],
        ]) as usize;
        let kvstart = pos + 11;
        if kvstart + klen + vlen > len {
            return (index, pos as u64);
        }
        let val_off = (kvstart + klen) as u64;
        // CRC считаем по КОПИИ body: flags|klen|vlen|key|val
        let mut body = Vec::with_capacity(7 + klen + vlen);
        body.push(flags);
        body.extend_from_slice(&(klen as u16).to_le_bytes());
        body.extend_from_slice(&(vlen as u32).to_le_bytes());
        body.extend_from_slice(&buf[kvstart..kvstart + klen]);
        body.extend_from_slice(&buf[val_off as usize..val_off as usize + vlen]);
        if crc32(&body) != crc {
            return (index, pos as u64);
        }
        let key = String::from_utf8_lossy(&buf[kvstart..kvstart + klen]).into_owned();
        if flags & FLAG_TOMB != 0 {
            index.remove(&key);
        } else {
            index.insert(key, Slot { val_off, vlen: vlen as u32 });
        }
        pos = val_off as usize + vlen;
    }
    (index, len as u64)
}

/// Читает значение записи по свежему read-хэндлу (get — immutably).
fn read_slot(path: &Path, slot: Slot) -> Option<Vec<u8>> {
    let mut f = OpenOptions::new().read(true).open(path).ok()?;
    let mut buf = vec![0u8; slot.vlen as usize];
    f.seek(SeekFrom::Start(slot.val_off)).ok()?;
    f.read_exact(&mut buf).ok()?;
    Some(buf)
}

impl KvStore {
    /// Открыть или создать. Оборванный хвост обрезается (WAL-семантика).
    pub fn open(path: &Path) -> std::io::Result<KvStore> {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            f.write_all(HEADER)?;
            f.sync_all()?;
        }
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if buf.len() < HEADER.len() || &buf[..HEADER.len()] != HEADER {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "неправильный заголовок KV-лога",
            ));
        }
        let (index, tail) = replay(&buf);
        if tail != buf.len() as u64 {
            // обрыв хвоста: физически обрезаем, чтобы append не лёг на мусор
            file.set_len(tail)?;
            file.seek(SeekFrom::Start(tail))?;
        }
        Ok(KvStore { path: path.to_path_buf(), file, tail, index })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Количество ЖИВЫХ ключей (tombstones не считаются).
    pub fn len(&self) -> usize {
        self.index.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn append(&mut self, rec: &[u8]) -> std::io::Result<u64> {
        self.file.seek(SeekFrom::Start(self.tail))?;
        self.file.write_all(rec)?;
        let off = self.tail;
        self.tail += rec.len() as u64;
        Ok(off)
    }

    /// Записать/перезаписать значение.
    pub fn put(&mut self, key: &str, val: &[u8]) -> std::io::Result<()> {
        let rec = encode_record(0, key, val);
        let val_off = self.append(&rec)? + (4 + 7 + key.len() as u64);
        self.index.insert(key.to_string(), Slot { val_off, vlen: val.len() as u32 });
        Ok(())
    }

    /// Прочитать значение по живому слоту.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        read_slot(&self.path, *self.index.get(key)?)
    }

    /// Удалить (tombstone). `false` если ключа и так нет.
    pub fn delete(&mut self, key: &str) -> std::io::Result<bool> {
        if !self.index.contains_key(key) {
            return Ok(false);
        }
        let rec = encode_record(FLAG_TOMB, key, &[]);
        self.append(&rec)?;
        self.index.remove(key);
        Ok(true)
    }

    /// Отсортированные живые ключи с префиксом.
    pub fn scan(&self, prefix: &str) -> Vec<String> {
        self.index
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Переписать живые записи без tombstones/мусора. Крах посреди compact
    /// оставляет исходный файл нетронутым (rename — атомарный последний шаг).
    pub fn compact(&mut self) -> std::io::Result<()> {
        let tmp = self.path.with_extension("dlog.tmp");
        {
            let mut tmpf = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            tmpf.write_all(HEADER)?;
            let mut new_index = BTreeMap::new();
            let mut tail = HEADER.len() as u64;
            for (key, slot) in &self.index {
                let val = read_slot(&self.path, *slot)
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "чтение слота"))?;
                let rec = encode_record(0, key, &val);
                let val_off = tail + (4 + 7 + key.len() as u64);
                tmpf.write_all(&rec)?;
                new_index.insert(key.clone(), Slot { val_off, vlen: val.len() as u32 });
                tail += rec.len() as u64;
            }
            tmpf.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        self.file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        let mut buf = Vec::new();
        self.file.read_to_end(&mut buf)?;
        let (index, tail) = replay(&buf);
        if tail != buf.len() as u64 {
            self.file.set_len(tail)?;
        }
        self.index = index;
        self.tail = tail;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("dinet-mem-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn set_get_delete_reopen() {
        let dir = tmpdir("basic");
        let path = dir.join("kv.dlog");
        {
            let mut kv = KvStore::open(&path).unwrap();
            kv.put("chat:1", "привет".as_bytes()).unwrap();
            kv.put("chat:2", "мир".as_bytes()).unwrap();
            assert_eq!(kv.get("chat:1"), Some("привет".as_bytes().to_vec()));
            assert_eq!(kv.len(), 2);
        }
        let mut kv = KvStore::open(&path).unwrap();
        assert_eq!(kv.get("chat:2"), Some("мир".as_bytes().to_vec()));
        assert!(kv.delete("chat:1").unwrap());
        assert_eq!(kv.get("chat:1"), None);
        assert_eq!(kv.len(), 1);
        // reopen: tombstone пережил запись
        let kv = KvStore::open(&path).unwrap();
        assert_eq!(kv.get("chat:1"), None);
        assert_eq!(kv.get("chat:2"), Some("мир".as_bytes().to_vec()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn overwrite_takes_latest() {
        let dir = tmpdir("over");
        let path = dir.join("kv.dlog");
        let mut kv = KvStore::open(&path).unwrap();
        kv.put("k", b"v1").unwrap();
        kv.put("k", "v2-длиннее".as_bytes()).unwrap();
        assert_eq!(kv.get("k"), Some("v2-длиннее".as_bytes().to_vec()));
        let kv = KvStore::open(&path).unwrap();
        assert_eq!(kv.get("k"), Some("v2-длиннее".as_bytes().to_vec()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn torn_tail_is_trimmed_on_recovery() {
        let dir = tmpdir("torn");
        let path = dir.join("kv.dlog");
        {
            let mut kv = KvStore::open(&path).unwrap();
            kv.put("a", b"1").unwrap();
            kv.put("b", b"2").unwrap();
        }
        // эмулируем кра посреди записи: приклеиваем хвост мусора
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x42]).unwrap();
        drop(f);
        let kv = KvStore::open(&path).unwrap();
        assert_eq!(kv.get("a"), Some(b"1".to_vec()));
        assert_eq!(kv.get("b"), Some(b"2".to_vec()));
        assert_eq!(kv.len(), 2);
        // хвост обрезан: новые записи пишутся после последней хорошей
        let mut kv = KvStore::open(&path).unwrap();
        kv.put("c", b"3").unwrap();
        assert_eq!(kv.get("c"), Some(b"3".to_vec()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_record_kills_it_and_the_tail() {
        let dir = tmpdir("crc");
        let path = dir.join("kv.dlog");
        {
            let mut kv = KvStore::open(&path).unwrap();
            kv.put("a", b"1").unwrap();
            kv.put("b", b"2").unwrap();
        }
        // портим CRC первой записи
        let mut bytes = fs::read(&path).unwrap();
        bytes[6] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();
        let kv = KvStore::open(&path).unwrap();
        // классический WAL: bad-запись прекращает реплей (b «за ней» теряется)
        assert_eq!(kv.get("a"), None);
        assert_eq!(kv.get("b"), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_drops_tombstones_and_shrinks() {
        let dir = tmpdir("compact");
        let path = dir.join("kv.dlog");
        let mut kv = KvStore::open(&path).unwrap();
        for i in 0..50u32 {
            kv.put(&format!("k{i:03}"), format!("значение-{i}").as_bytes()).unwrap();
        }
        for i in 0..30u32 {
            kv.delete(&format!("k{i:03}")).unwrap();
        }
        let size_before = fs::metadata(&path).unwrap().len();
        assert_eq!(kv.len(), 20);
        kv.compact().unwrap();
        let size_after = fs::metadata(&path).unwrap().len();
        assert_eq!(kv.len(), 20);
        assert!(size_after < size_before, "compact должен ужимать: {size_before} → {size_after}");
        assert_eq!(kv.get("k007"), None);
        assert_eq!(kv.get("k049"), Some("значение-49".as_bytes().to_vec()));
        // переоткрытие — всё на месте
        let kv = KvStore::open(&path).unwrap();
        assert_eq!(kv.len(), 20);
        assert_eq!(kv.get("k049"), Some("значение-49".as_bytes().to_vec()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_is_sorted_by_prefix() {
        let dir = tmpdir("scan");
        let path = dir.join("kv.dlog");
        let mut kv = KvStore::open(&path).unwrap();
        for k in ["b:2", "a:9", "a:1", "c:0"] {
            kv.put(k, b"x").unwrap();
        }
        assert_eq!(kv.scan("a:"), vec!["a:1".to_string(), "a:9".to_string()]);
        assert_eq!(kv.scan("c:"), vec!["c:0".to_string()]);
        assert!(kv.scan("z:").is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
