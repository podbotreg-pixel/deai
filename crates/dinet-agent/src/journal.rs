//! Undo-журнал агента (M2, PAPAV-принцип «всё обратимо»): каждое мутирующее
//! действие ДО исполнения пишет инверсию в append-only журнал (dinet-memory
//! KvStore: CRC-записи, обрыв хвоста обрезается, tombstones). Undo = сыграть
//! инверсию. Поломанные записи журнала бракуются с ошибкой (не молча).

use dinet_memory::KvStore;
use std::path::{Path, PathBuf};

/// Одна запись инверсии.
#[derive(Debug, Clone, PartialEq)]
pub enum Inverse {
    /// Файл был, было содержимое → вернуть байты.
    Overwrite { rel: String, was: Option<Vec<u8>> },
}

impl Inverse {
    /// Текст-кодирование (внутреннее, \n в полях запрещён по построению).
    fn encode(&self) -> Vec<u8> {
        match self {
            Inverse::Overwrite { rel, was } => {
                let mut v = b"OW\n".to_vec();
                v.extend_from_slice(rel.as_bytes());
                v.push(b'\n');
                match was {
                    Some(b) => {
                        v.extend_from_slice(b"1\n");
                        v.extend_from_slice(&(b.len() as u64).to_le_bytes());
                        v.extend_from_slice(b);
                    }
                    None => v.extend_from_slice(b"0\n"),
                }
                v
            }
        }
    }

    fn decode(b: &[u8]) -> Result<Inverse, String> {
        let text = std::str::from_utf8(b).map_err(|_| "не UTF-8")?;
        // Формат: OW \n rel \n flag \n [len(8 LE) body | —]
        let mut seg = text.splitn(4, '\n');
        let kind = seg.next().ok_or("пусто")?;
        if kind != "OW" {
            return Err(format!("неизвестный вид записи {kind:?}"));
        }
        let rel = seg.next().ok_or("нет пути")?.to_string();
        let present = seg.next().ok_or("нет флага наличия")?;
        match present {
            "0" => Ok(Inverse::Overwrite { rel, was: None }),
            "1" => {
                let rest = seg.next().ok_or("нет длины")?;
                if rest.len() < 8 {
                    return Err("нет 8 байт длины".into());
                }
                let rb = rest.as_bytes();
                let len = u64::from_le_bytes(rb[..8].try_into().expect("8 байт")) as usize;
                let body = &rb[8..];
                if body.len() != len {
                    return Err(format!("длина {len}, байт {}", body.len()));
                }
                Ok(Inverse::Overwrite { rel, was: Some(body.to_vec()) })
            }
            _ => Err("флаг наличия не 0/1".into()),
        }
    }
}

/// Журнал: seq растёт, undo снимает с вершины.
pub struct Journal {
    kv: KvStore,
    path: PathBuf,
    next: u64,
}

impl Journal {
    pub fn open(path: &Path) -> Result<Journal, String> {
        let kv = KvStore::open(path).map_err(|e| format!("журнал {path:?}: {e}"))?;
        let mut next = 0u64;
        // seq монотонен по записям (tombstone — тоже прогресс)
        for k in kv.scan("j:") {
            let n: u64 = k
                .trim_start_matches("j:")
                .parse()
                .map_err(|_| format!("плохой ключ журнала {k:?}"))?;
            next = next.max(n + 1);
        }
        Ok(Journal { kv, path: path.to_path_buf(), next })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Записать инверсию ДО действия. Ошибка = действие НЕ делаем.
    pub fn push(&mut self, inv: Inverse) -> Result<u64, String> {
        // Формат разграничивается \n: путь с \n/\0 бракуем (патология)
        let Inverse::Overwrite { rel, .. } = &inv;
        if rel.contains('\n') || rel.contains('\0') {
            return Err("путь с переводом строки в журнал не пишется".into());
        }
        let key = format!("j:{}", self.next);
        self.kv.put(&key, &inv.encode()).map_err(|e| format!("журнал: {e}"))?;
        self.next += 1;
        Ok(self.next - 1)
    }

    /// Снять вершину (последняя записанная инверсия).
    pub fn pop(&mut self) -> Option<Inverse> {
        if self.next == 0 {
            return None;
        }
        self.next -= 1;
        let key = format!("j:{}", self.next);
        let v = self.kv.get(&key)?;
        Inverse::decode(&v).ok()
    }

    pub fn len(&self) -> u64 {
        self.next
    }

    pub fn is_empty(&self) -> bool {
        self.next == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_j(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dinet-jrn-{tag}-{}.roi", std::process::id()))
    }

    #[test]
    fn push_pop_roundtrip() {
        let p = tmp_j("rt");
        let _ = std::fs::remove_file(&p);
        let mut j = Journal::open(&p).expect("open");
        assert!(j.is_empty());
        j.push(Inverse::Overwrite { rel: "a.txt".into(), was: None }).expect("push");
        let bytes = "привет мир".as_bytes().to_vec();
        j.push(Inverse::Overwrite { rel: "b.bin".into(), was: Some(bytes.clone()) })
            .expect("push");
        assert_eq!(j.len(), 2);
        match j.pop().expect("вершина") {
            Inverse::Overwrite { rel, was } => {
                assert_eq!(rel, "b.bin");
                assert_eq!(was, Some(bytes));
            }
        }
        match j.pop().expect("вторая") {
            Inverse::Overwrite { rel, was } => {
                assert_eq!(rel, "a.txt");
                assert_eq!(was, None);
            }
        }
        assert!(j.pop().is_none(), "журнал пуст");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn journal_survives_reopen() {
        let p = tmp_j("reopen");
        let _ = std::fs::remove_file(&p);
        {
            let mut j = Journal::open(&p).expect("open");
            j.push(Inverse::Overwrite { rel: "x".into(), was: Some(vec![1, 2, 3]) })
                .expect("push");
        }
        let j2 = Journal::open(&p).expect("reopen");
        assert_eq!(j2.len(), 1, "seq переживает рестарт");
        let _ = std::fs::remove_file(&p);
    }
}
