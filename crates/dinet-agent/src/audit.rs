//! Аудит-журнал агента (M2, DoD «все действия в журнале»): ВСЕ вызовы
//! инструментов — включая чтения — пишутся в append-only журнал
//! (dinet-memory KvStore: CRC, обрыв хвоста обрезается). Отличие от
//! undo-журнала: аудит — история НАБЛЮДЕНИЙ (не инверсии), только
//! дописывается, не откатывается.

use dinet_memory::KvStore;
use std::path::{Path, PathBuf};

/// Одна запись аудита.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditRec {
    pub seq: u64,
    pub ts: u64,
    /// Какой инструмент звали.
    pub tool: String,
    /// Уровень, с которым агент был допущен.
    pub granted: String,
    /// Успех/ошибка уровня инструмента (isError).
    pub ok: bool,
    /// Краткое резюме (текст инструмента или причина отказа).
    pub summary: String,
}

/// Аудит: seq монотонен, только append.
pub struct Audit {
    kv: KvStore,
    path: PathBuf,
    next: u64,
}

impl Audit {
    pub fn open(path: &Path) -> Result<Audit, String> {
        let kv = KvStore::open(path).map_err(|e| format!("аудит {path:?}: {e}"))?;
        let mut next = 0u64;
        for k in kv.scan("a:") {
            let n: u64 = k
                .trim_start_matches("a:")
                .parse()
                .map_err(|_| format!("плохой ключ аудита {k:?}"))?;
            next = next.max(n + 1);
        }
        Ok(Audit { kv, path: path.to_path_buf(), next })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Записать событие. Ошибка журнала НЕ валит действие (аудит
    /// best-effort), но возвращается наружу для статуса.
    pub fn log(&mut self, tool: &str, granted: &str, ok: bool, summary: &str, ts: u64) -> Result<u64, String> {
        let seq = self.next;
        // формат: seq \n ts \n tool \n granted \n ok \n summary
        let mut v = format!("{seq}\n{ts}\n{tool}\n{granted}\n{}\n", if ok { 1 } else { 0 });
        v.push_str(&summary.replace('\n', " "));
        self.kv.put(&format!("a:{seq}"), v.as_bytes())
            .map_err(|e| format!("аудит: {e}"))?;
        self.next += 1;
        Ok(seq)
    }

    /// Последние n записей (от новых к старым).
    pub fn tail(&self, n: usize) -> Vec<AuditRec> {
        let mut out = Vec::new();
        for i in (0..self.next).rev() {
            if out.len() >= n {
                break;
            }
            if let Some(v) = self.kv.get(&format!("a:{i}")) {
                if let Ok(r) = decode(&v, i) {
                    out.push(r);
                }
            }
        }
        out
    }

    pub fn len(&self) -> u64 {
        self.next
    }

    pub fn is_empty(&self) -> bool {
        self.next == 0
    }
}

fn decode(b: &[u8], seq: u64) -> Result<AuditRec, String> {
    let text = std::str::from_utf8(b).map_err(|_| "не UTF-8")?;
    // Формат: seq \n ts \n tool \n granted \n ok \n summary (6 полей,
    // summary — последний, \n внутри сплющены при записи).
    let mut it = text.splitn(6, '\n');
    let _seq: u64 = it.next().ok_or("нет seq")?.parse().map_err(|_| "seq не u64")?;
    let ts: u64 = it.next().ok_or("нет ts")?.parse().map_err(|_| "ts не u64")?;
    let tool = it.next().ok_or("нет tool")?.to_string();
    let granted = it.next().ok_or("нет granted")?.to_string();
    let ok = it.next().ok_or("нет ok")? == "1";
    let summary = it.next().unwrap_or("").to_string();
    Ok(AuditRec { seq, ts, tool, granted, ok, summary })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_a(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dinet-audit-{tag}-{}.roi", std::process::id()))
    }

    #[test]
    fn log_tail_roundtrip() {
        let p = tmp_a("rt");
        let _ = std::fs::remove_file(&p);
        let mut a = Audit::open(&p).expect("open");
        a.log("fs_write", "WRITE", true, "записано 10 байт", 100).expect("log");
        a.log("fs_read", "READ", true, "содержимое", 101).expect("log");
        a.log("cli_run", "NETWORK", false, "не в allowlist", 102).expect("log");
        assert_eq!(a.len(), 3);
        let tail = a.tail(2);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].tool, "cli_run", "свежие первыми");
        assert!(!tail[0].ok);
        assert_eq!(tail[1].tool, "fs_read");
        // summary с \n сплющен
        a.log("cli_run", "NETWORK", true, "много\nстрок", 103).expect("log");
        assert_eq!(a.tail(1)[0].summary, "много строк");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn audit_survives_reopen() {
        let p = tmp_a("reopen");
        let _ = std::fs::remove_file(&p);
        {
            let mut a = Audit::open(&p).expect("open");
            a.log("fs_delete", "DESTRUCTIVE", true, "удалён x", 5).expect("log");
        }
        let a2 = Audit::open(&p).expect("reopen");
        assert_eq!(a2.len(), 1);
        let tail = a2.tail(1);
        assert_eq!(tail[0].tool, "fs_delete");
        assert_eq!(tail[0].ts, 5);
        let _ = std::fs::remove_file(&p);
    }
}
