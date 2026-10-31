//! Транспорт узла DI-NET (v0): обмен wire-кадрами `dinet-proto`.
//!
//! Трейт `Transport` — ЕДИНСТВЕННАЯ точка прикрепления libp2p: когда появится
//! crates.io, реализация gossips-dht-subscribe заменит `MeshTransport`/`FileSpool`
//! без изменения протокола обмена (`swarm`-логика не знает, чей канал под ней).
//!
//! Две реализации v0:
//! 1. `MeshTransport` — детерминированная in-memory сеть (unit/rapid-тесты).
//! 2. `FileSpool` — обмен через каталогwire-frame-файлами: два процесса на одной
//!    машине уже могут «жить» в одной сети. Правила: файл
//!    `c{ch}-{to|all}-{from}-{seq}.wif`, читается и удаляется получателем.
//!
//! Каналы (`ch`) фиксируют фазу эпохи — 0: SHARES/REVEALS, 1: VOTES, 2: ANNOUNCE.

use crate::swarm::NodeId;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const CH_SHARES: u8 = 0;
pub const CH_VOTES: u8 = 1;
pub const CH_ANNOUNCE: u8 = 2;
/// Нотарные печати реестра чек-пойнтов (фаза F, после tally).
pub const CH_NOTARY: u8 = 3;
/// torrent чек-пойнтов: need/have/chunk (фаза G).
pub const CH_BLOB: u8 = 4;
/// Синхронизация реестра: ask/offer (фаза H — новичок догоняет append-only лог).
pub const CH_REG: u8 = 5;
/// Внешние анкоры: штамп/вопрос/ответ (фаза I — анти-откат).
pub const CH_ANCHOR: u8 = 6;
/// Знакомство новичка: Hello/HelloAck (фаза J, TOFU + PEX).
pub const CH_HELLO: u8 = 7;

pub trait Transport {
    /// `to = None` → broadcast (все прочие узлы).
    fn send(&mut self, ch: u8, from: NodeId, to: Option<NodeId>, frame: &[u8]);
    /// Собрать и ПОГЛОТИТЬ входящие кадры канала `ch` для узла `me`.
    fn collect(&mut self, ch: u8, me: NodeId) -> Vec<Vec<u8>>;

    /// Очистить broadcast-носители канала (по умолчанию — no-op).
    fn sweep(&mut self, _ch: u8) {}
}
// ---------------------------------------------------------------------------
// In-memory mesh (детерминированный)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct MeshTransport {
    inboxes: BTreeMap<NodeId, Vec<(u8, Vec<u8>)>>,
}

impl MeshTransport {
    pub fn new(size: usize) -> MeshTransport {
        let mut inboxes = BTreeMap::new();
        for i in 0..size {
            inboxes.insert(i as NodeId, Vec::new());
        }
        MeshTransport { inboxes }
    }
}

impl Transport for MeshTransport {
    /// Broadcast включает и кадр самого отправителя (gossipsub-семантика:
    /// узел видит свою же публикацию в тема-потоке).
    fn send(&mut self, ch: u8, from: NodeId, to: Option<NodeId>, frame: &[u8]) {
        let targets: Vec<NodeId> = match to {
            Some(t) if t != from => vec![t],
            _ => self.inboxes.keys().copied().collect(),
        };
        for t in targets {
            if let Some(inbox) = self.inboxes.get_mut(&t) {
                inbox.push((ch, frame.to_vec()));
            }
        }
    }

    fn collect(&mut self, ch: u8, me: NodeId) -> Vec<Vec<u8>> {
        match self.inboxes.get_mut(&me) {
            Some(inbox) => {
                let mut kept = Vec::new();
                let mut out = Vec::new();
                for (c, fr) in inbox.drain(..) {
                    if c == ch {
                        out.push(fr);
                    } else {
                        kept.push((c, fr));
                    }
                }
                *inbox = kept;
                out
            }
            None => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Файловый спул (реальные процессы POSIX/Windows — тот же протобул кадрами)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct FileSpool {
    pub dir: PathBuf,
}

/// Сигнатура имени: `c{ch}-{to:03|all}-{from:03}-{seq}.wif`
fn fname(ch: u8, to: Option<NodeId>, from: NodeId, seq: usize) -> String {
    let to_s = match to {
        Some(t) => format!("{t:03}"),
        None => "all".to_string(),
    };
    format!("c{}-{}-{}-{:06}.wif", ch, to_s, from, seq)
}

impl FileSpool {
    pub fn new(dir: &Path) -> std::io::Result<FileSpool> {
        std::fs::create_dir_all(dir)?;
        Ok(FileSpool { dir: dir.to_path_buf() })
    }

    /// Позвон« next free sequence для префиксов канала (детерминированно).
    fn next_seq(&self, ch: u8, to: Option<NodeId>, from: NodeId) -> std::io::Result<usize> {
        let prefix = {
            let to_s = match to {
                Some(t) => format!("{t:03}"),
                None => "all".to_string(),
            };
            format!("c{}-{}-{}-", ch, to_s, from)
        };
        let mut max = 0usize;
        for entry in std::fs::read_dir(&self.dir)? {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) {
                if let Some(seq_s) = name.strip_suffix(".wif").and_then(|s| s.rsplit('-').next()) {
                    if let Ok(v) = seq_s.parse::<usize>() {
                        max = max.max(v + 1);
                    }
                }
            }
        }
        Ok(max)
    }
}

impl Transport for FileSpool {
    fn send(&mut self, ch: u8, from: NodeId, to: Option<NodeId>, frame: &[u8]) {
        let seq = self.next_seq(ch, to, from).unwrap_or(0);
        let path = self.dir.join(fname(ch, to, from, seq));
        let _ = std::fs::write(path, frame);
    }

    fn collect(&mut self, ch: u8, me: NodeId) -> Vec<Vec<u8>> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let mut items: Vec<(String, Vec<u8>)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".wif") || !name.starts_with(&format!("c{}-", ch)) {
                continue;
            }
            // broadcast (all-) читаются всеми и НЕ удаляются при collect —
            // их убирает `sweep` (иначе первый читатель крадёт кадр у остальных).
            let targeted = name.starts_with(&format!("c{}-{:03}-", ch, me));
            let broadcast = name.starts_with(&format!("c{}-all-", ch));
            if !targeted && !broadcast {
                continue;
            }
            if let Ok(bytes) = std::fs::read(entry.path()) {
                if targeted {
                    let _ = std::fs::remove_file(entry.path());
                }
                items.push((name, bytes));
            }
        }
        items.sort_by(|a, b| a.0.cmp(&b.0)); // seq-порядок внутри отправителя
        items.into_iter().map(|(_, f)| f).collect()
    }

    fn sweep(&mut self, ch: u8) {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&format!("c{}-all-", ch)) && name.ends_with(".wif") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_broadcast_and_targeted() {
        let mut m = MeshTransport::new(3);
        m.send(CH_SHARES, 0, None, b"hello-all");
        m.send(CH_SHARES, 0, Some(2), b"hello-2");
        assert_eq!(m.collect(CH_SHARES, 1), vec![b"hello-all".to_vec()]);
        assert_eq!(
            m.collect(CH_SHARES, 2),
            vec![b"hello-all".to_vec(), b"hello-2".to_vec()]
        );
        assert!(m.collect(CH_SHARES, 1).is_empty(), "drain после collect");
    }

    #[test]
    fn mesh_channels_isolated() {
        let mut m = MeshTransport::new(2);
        m.send(CH_VOTES, 0, None, b"v");
        // запрашиваем не тот канал — кадр НЕ теряется, остаётся в инбоксе
        assert!(m.collect(CH_SHARES, 1).is_empty());
        assert_eq!(m.collect(CH_VOTES, 1), vec![b"v".to_vec()]);
    }

    #[test]
    fn file_spool_roundtrip_and_channels() {
        let dir = std::env::temp_dir().join(format!("dinet-spool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut a = FileSpool::new(&dir).unwrap();
        let mut b = FileSpool::new(&dir).unwrap();
        a.send(CH_SHARES, 0, None, b"frame-1");
        a.send(CH_SHARES, 0, Some(1), b"frame-2");
        a.send(CH_VOTES, 0, None, b"vote");
        // другой отправитель в тот же канал — номера seq независимы
        a.send(CH_SHARES, 2, None, b"peer2");
        let got = b.collect(CH_SHARES, 1);
        let texts: Vec<&[u8]> = got.iter().map(|v| v.as_slice()).collect();
        assert!(texts.contains(&b"frame-1".as_slice()));
        assert!(texts.contains(&b"frame-2".as_slice()));
        assert!(texts.contains(&b"peer2".as_slice()));
        assert!(!got.iter().any(|v| v == &b"vote".to_vec()), "канал votes изолирован");
        // targeted-кадры поглощены; broadcast (all-) остаются до sweep
        let got2 = b.collect(CH_SHARES, 1);
        assert!(got2.iter().any(|v| *v == b"frame-1".to_vec()), "broadcast пере-читывается");
        assert!(!got2.iter().any(|v| *v == b"frame-2".to_vec()), "targeted поглощён");
        a.sweep(CH_SHARES);
        // votes ждут узел 1
        assert_eq!(b.collect(CH_VOTES, 1), vec![b"vote".to_vec()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
