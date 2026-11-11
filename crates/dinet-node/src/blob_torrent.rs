//! blob_torrent — ФАЗА G: репликация весов чек-пойнтов между пирами.
//!
//! Реестр хранит только `manifest_id` (индекс); сами байты дельты живут
//! у узлов в `BlobStore` и докачиваются торрент-схемой поверх того же
//! Transport:
//! 1. просящий (нет блоба в складе) broadcast `BlobNeed {manifest_id}`;
//! 2. держатели отвечают направленно `BlobHave {manifest_id, канон}` —
//!    канон манифеста проверяется `sha256(канон) == manifest_id`;
//! 3. просящий запрашивает недостающие куски round-robin у ДЕРЖАТЕЛЕЙ
//!    (`BlobChunkReq`), каждый ответ верифицируется хэшем манифеста;
//!    частичный держатель молчит по недостающим кускам — запрос уходит
//!    следующему (как у живых торрентов: много сидов, каждый отдаёт что
//!    есть, сборка бит-в-бит).
//!
//! Подсунуть фальшивый кусок под чужой `manifest_id` нельзя: канон
//! манифеста связан хэшем с id, а куски — хэшами самого манифеста
//! (SHA-256 пре-имидж). «Отравленный» склад узла просто не совпадает
//! по id — такой узел молчит.

use crate::swarm::{decode_with_book, sign_frame, Node, NodeId};
use crate::transport::{CH_BLOB, Transport};
use dinet_proto::{MsgType, Payload};
use dinet_registry::{DeltaManifest, RegistryLog, DEFAULT_CHUNK_LEN};
use std::collections::BTreeMap;

/// Локальный склад весовых блобов узла: manifest_id → (манифест, куски).
/// КЕПКА: склад не растёт бесконечно — начиная с квоты старейшие
/// ПО ВСТАВКЕ блобы вымываются (актуальные чек-пойнты клиент держит у
/// себя в рой-KV, так что потеря склада не теряет состояние клиента).
#[derive(Debug, Default, Clone)]
pub struct BlobStore {
    blobs: BTreeMap<[u8; 32], (DeltaManifest, Vec<Vec<u8>>)>,
    /// порядок вставки (для вымывания старейших) + квота
    order: std::collections::VecDeque<[u8; 32]>,
    cap: usize,
}

/// Квота склада блобов (шт. манифестов).
pub const MAX_BLOBS: usize = 24;

impl BlobStore {
    /// Хранить блоб (куски держатся целиком: узел — полный сид).
    /// Растущий сверх квоты склад вымывает старейшие по вставке.
    pub fn put(&mut self, m: DeltaManifest, chunks: Vec<Vec<u8>>) {
        let id = m.id();
        if !self.blobs.contains_key(&id) {
            while self.blobs.len() >= self.cap {
                match self.order.pop_front() {
                    Some(old) => {
                        self.blobs.remove(&old);
                    }
                    None => break, // order рассинхрон — лечим вставкой ниже
                }
            }
            self.order.push_back(id);
        }
        self.blobs.insert(id, (m, chunks));
    }

    /// Своя квота (для тестов — меньше константы).
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap.max(1);
    }

    pub fn has(&self, id: &[u8; 32]) -> bool {
        self.blobs.contains_key(id)
    }

    pub fn manifest(&self, id: &[u8; 32]) -> Option<&DeltaManifest> {
        self.blobs.get(id).map(|(m, _)| m)
    }

    pub fn chunks(&self, id: &[u8; 32]) -> Option<&Vec<Vec<u8>>> {
        self.blobs.get(id).map(|(_, c)| c)
    }

    pub fn chunk(&self, id: &[u8; 32], idx: usize) -> Option<&Vec<u8>> {
        self.blobs.get(id).and_then(|(_, c)| c.get(idx))
    }

    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }

    /// Полный блоб: манифест есть и ВСЕ куски непусты (дальше — сид).
    /// Частичный (с провалом) обязан самолечиться торрентом.
    pub fn complete(&self, id: &[u8; 32]) -> bool {
        match self.blobs.get(id) {
            None => false,
            Some((m, chunks)) => {
                chunks.len() == m.n_chunks() && chunks.iter().all(|c| !c.is_empty())
            }
        }
    }

    /// Узел «потерял» склад (рестарт с чистым диском) — тесты и честная
    /// очистка старых блобов по квоте.
    pub fn clear(&mut self) {
        self.blobs.clear();
        self.order.clear();
    }

    /// Выбросить один кусок (тест «частичная потеря диска»; пустой кусок
    /// = молчание сида на запрос).
    pub fn drop_chunk(&mut self, id: &[u8; 32], idx: usize) {
        if let Some((_, chunks)) = self.blobs.get_mut(id) {
            if idx < chunks.len() {
                chunks[idx] = Vec::new();
            }
        }
    }
}

/// Каноническая упаковка дельты f32 (LE) — тот же выбор байтов, что в wire.
pub fn delta_f32_packed(delta: &[f32], chunk_len: u32) -> (DeltaManifest, Vec<Vec<u8>>) {
    let mut bytes = Vec::with_capacity(delta.len() * 4);
    for v in delta {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    DeltaManifest::pack(&bytes, chunk_len)
}

/// manifest_id дельты (то, что лежит в RegistryEntry).
pub fn delta_f32_id(delta: &[f32], chunk_len: u32) -> [u8; 32] {
    delta_f32_packed(delta, chunk_len).0.id()
}

/// Отчёт фазы по узлу.
#[derive(Debug, Clone, PartialEq)]
pub struct TorrentReport {
    /// manifest_id из head реестра (цель фазы).
    pub target: [u8; 32],
    /// Блоб в складе (собран из кусков или уже был).
    pub assembled: bool,
    /// Держателей, чьи каноны приняты (0 = блоб уже был / цели нет).
    pub peers: usize,
}

/// ФАЗА G: torrent-обмен блобами поверх Transport (после нотариев).
pub fn torrent_blobs(
    mine: &mut [Node],
    all_ids: &[NodeId],
    transport: &mut dyn Transport,
    regs: &[RegistryLog],
) -> Vec<TorrentReport> {
    // Цель узла — head ЕГО реестра; материала нет = узел вне фазы.
    let target: Vec<Option<[u8; 32]>> =
        regs.iter().map(|r| r.head().map(|h| h.entry.manifest_id)).collect();

    // Состояние просящего: канон манифеста, собранные куски, держатели.
    // Частичный узел (есть манифест, есть провал) НЕ считаем сидом: он
    // продолжает действовать как просящий, зная канон заранее.
    struct Fetch {
        manifest: Option<DeltaManifest>,
        have: BTreeMap<usize, Vec<u8>>,
        holders: Vec<NodeId>,
        attempt: usize,
    }
    let mut fetch: Vec<Fetch> = (0..mine.len())
        .map(|_| Fetch { manifest: None, have: BTreeMap::new(), holders: Vec::new(), attempt: 0 })
        .collect();
    for idx in 0..mine.len() {
        let Some(t) = target[idx] else { continue };
        if mine[idx].blobs.complete(&t) {
            continue; // полный сид — вне торрента
        }
        // частичный: канон и живые куски берутся из локального склада
        if let Some(m) = mine[idx].blobs.manifest(&t) {
            fetch[idx].manifest = Some(m.clone());
            for (i, c) in mine[idx].blobs.chunks(&t).expect("куски").iter().enumerate() {
                if !c.is_empty() {
                    fetch[idx].have.insert(i, c.clone());
                }
            }
        }
    }

    let max_rounds = 2 + all_ids.len();
    for round in 0..max_rounds {
        // --- Отправка
        for idx in 0..mine.len() {
            let Some(t) = target[idx] else { continue };
            let f = &mut fetch[idx];
            if mine[idx].blobs.complete(&t) {
                continue; // полный сид: отправки не нужны
            }
            if round == 0 {
                // раунд держателей: broadcast need при ЛЮБОЙ неполноте
                // (частичный сид тоже не знает заранее открытых пиров)
                let frame = sign_frame(
                    &mine[idx],
                    MsgType::BlobNeed,
                    Payload::BlobNeed { manifest_id: t },
                );
                transport.send(CH_BLOB, mine[idx].id, None, &frame);
                continue;
            }
            // канон уже есть (свой или пришедший): расфраживаем запросы
            // недостающих кусков round-robin.
            let Some(m) = f.manifest.clone() else { continue };
            let want: Vec<usize> = (0..m.n_chunks())
                .filter(|i| !f.have.contains_key(i))
                .collect();
            for &chunk_idx in &want {
                if f.holders.is_empty() {
                    break;
                }
                let h = f.holders[(chunk_idx + f.attempt) % f.holders.len()];
                let frame = sign_frame(
                    &mine[idx],
                    MsgType::BlobChunkReq,
                    Payload::BlobChunkReq { manifest_id: t, idx: chunk_idx as u32 },
                );
                transport.send(CH_BLOB, mine[idx].id, Some(h), &frame);
            }
        }

        // --- Сбор и ответы
        for idx in 0..mine.len() {
            let me = mine[idx].id;
            let Some(t) = target[idx] else { continue };
            let wire = transport.collect(CH_BLOB, mine[idx].id);
            let payloads = decode_with_book(&mut mine[idx], all_ids, wire);
            for (nid, pl) in payloads {
                if nid == me {
                    continue; // loopback своей публикации
                }
                match pl {
                    // держатель услышал need → отвечает каноном своего склада
                    Payload::BlobNeed { manifest_id } => {
                        if round == 0 {
                            if let Some(m) = mine[idx].blobs.manifest(&manifest_id) {
                                let canon = m.encode();
                                let frame = sign_frame(
                                    &mine[idx],
                                    MsgType::BlobHave,
                                    Payload::BlobHave { manifest_id, manifest_canon: canon },
                                );
                                transport.send(CH_BLOB, mine[idx].id, Some(nid), &frame);
                            }
                        }
                    }
                    // просящий принял канон манифеста (и холдера список)
                    Payload::BlobHave { manifest_id, manifest_canon } => {
                        if manifest_id != t {
                            continue;
                        }
                        if dinet_crypto::sha256(&manifest_canon) != manifest_id {
                            mine[idx].rejected_frames += 1;
                            continue;
                        }
                        let Ok(m) = DeltaManifest::decode(&manifest_canon) else {
                            mine[idx].rejected_frames += 1;
                            continue;
                        };
                        if m.id() != manifest_id {
                            mine[idx].rejected_frames += 1;
                            continue;
                        }
                        fetch[idx].holders.push(nid);
                        fetch[idx].holders.sort();
                        fetch[idx].holders.dedup();
                        if fetch[idx].manifest.is_none() {
                            fetch[idx].manifest = Some(m);
                        }
                    }
                    // держатель отдаёт кусок
                    Payload::BlobChunkReq { manifest_id, idx: chunk_idx } => {
                        if let Some(bytes) =
                            mine[idx].blobs.chunk(&manifest_id, chunk_idx as usize)
                        {
                            if bytes.is_empty() {
                                continue; // «потерянный» кусок — молчание
                            }
                            let frame = sign_frame(
                                &mine[idx],
                                MsgType::BlobChunk,
                                Payload::BlobChunk {
                                    manifest_id,
                                    idx: chunk_idx,
                                    bytes: bytes.clone(),
                                },
                            );
                            transport.send(CH_BLOB, mine[idx].id, Some(nid), &frame);
                        }
                    }
                    // просящий собирает куски
                    Payload::BlobChunk { manifest_id, idx: chunk_idx, bytes } => {
                        let Some(m) = fetch[idx].manifest.clone() else { continue };
                        if manifest_id != t || !m.verify_chunk(chunk_idx as usize, &bytes) {
                            if manifest_id == t {
                                mine[idx].rejected_frames += 1;
                            }
                            continue; // брак — запрос уйдёт другому держателю
                        }
                        fetch[idx].have.insert(chunk_idx as usize, bytes);
                    }
                    _ => mine[idx].rejected_frames += 1,
                }
            }
        }

        // --- Сборка: кто собрал ВСЕ верифицированные куски — кладёт блоб
        // в свой склад (с этой эры узел сам полноценный сид).
        for idx in 0..mine.len() {
            fetch[idx].attempt += 1;
            let Some(t) = target[idx] else { continue };
            if mine[idx].blobs.complete(&t) {
                continue;
            }
            let Some(m) = fetch[idx].manifest.clone() else { continue };
            if fetch[idx].have.len() != m.n_chunks() {
                continue;
            }
            let pieces: Vec<(usize, Vec<u8>)> =
                fetch[idx].have.iter().map(|(i, c)| (*i, c.clone())).collect();
            let Ok(body) = m.assemble(&pieces) else { continue };
            let chunks: Vec<Vec<u8>> =
                (0..m.n_chunks()).map(|i| fetch[idx].have[&i].clone()).collect();
            mine[idx].blobs.put(m.clone(), chunks);
            // Контрольная склейка через склад: put не должен менять байты.
            let stored = mine[idx].blobs.chunks(&t).expect("put был");
            let recheck: Vec<(usize, Vec<u8>)> =
                stored.iter().cloned().enumerate().collect();
            assert_eq!(
                m.assemble(&recheck).expect("склейка из склада"),
                body,
                "куски после put расходятся с собранным телом"
            );
        }
    }

    // Отчёт по складам.
    mine.iter()
        .enumerate()
        .map(|(idx, n)| {
            let assembled = target[idx].map(|t| n.blobs.has(&t)).unwrap_or(false);
            TorrentReport {
                target: target[idx].unwrap_or([0u8; 32]),
                assembled,
                peers: fetch[idx].holders.len(),
            }
        })
        .collect()
}

/// Дефолтный кусок блобов (ре-экспорт dinet-registry).
pub const DEFAULT_BLOB_CHUNK: u32 = DEFAULT_CHUNK_LEN;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry_phase::{era_notary, notary_min};
    use crate::swarm::{
        era_collect_and_vote, era_publish, era_tally, Node as SwarmNode, SimConfig,
    };
    use crate::transport::MeshTransport;
    use dinet_registry::GENESIS;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir()
            .join(format!("dinet-blob-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("tmp");
        d
    }

    /// 6 узлов (byz — последний), 3 эры (2 masked + REVEAL), затем две
    /// «потери»: у узла 0 склад вычищен целиком, у узла 2 пропал кусок 3.
    /// torrent должен собрать блоб узлу 0 из кусков разных пиров бит-в-бит.
    /// КЕПКА склада: старейшие по вставке блобы вымываются, свежие и
    /// повторные put живут; has по удалённым — false.
    #[test]
    fn blob_quota_evicts_oldest() {
        let mut s = BlobStore::default();
        s.set_cap(3);
        let mut ids = Vec::new();
        for i in 0..5u8 {
            let m = DeltaManifest::pack(format!("_checkpoint body {i}").as_bytes(), 4096).0;
            ids.push(m.id());
            s.put(m, Vec::new());
        }
        assert_eq!(s.len(), 3, "кепка не держит: {}", s.len());
        assert!(!s.has(&ids[0]) && !s.has(&ids[1]), "старейшие не вымылись");
        assert!(s.has(&ids[2]) && s.has(&ids[3]) && s.has(&ids[4]));
        // пановка тог же манифеста — не вымывание и не дубликат
        let (m, _) = DeltaManifest::pack(b"_checkpoint body 4", 4096);
        s.put(m, Vec::new());
        assert_eq!(s.len(), 3, "повтор put испортил склад");
        assert!(s.has(&ids[4]));
        // clear полностью пусто (и порядок)
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn torrent_assembles_from_peers_after_loss() {
        let cfg = SimConfig {
            reveal_every: 3,
            blob_chunk_len: 256, // 2048 байт дельты → 8 кусков
            ..Default::default()
        };
        let dir = tmpdir("loss");
        let mut nodes: Vec<SwarmNode> =
            (0..6usize).map(|i| SwarmNode::new(i as NodeId, 7, i == 5)).collect();
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let mut t = MeshTransport::new(6);
        let mut regs: Vec<RegistryLog> = (0..6)
            .map(|i| {
                RegistryLog::open(&dir.join(format!("r{i}.dlog")), notary_min(6))
                    .expect("reg")
            })
            .collect();

        for era in 0..3u32 {
            let truths = era_publish(&mut nodes, &ids, &mut t, era, &cfg, 7);
            let mids =
                era_collect_and_vote(&mut nodes, &ids, &mut t, era, &cfg, Some((&truths, ids[0])));
            let _reps = era_tally(&mut nodes, &ids, &mut t, era, &cfg, mids.clone());
            let _not = era_notary(&mut nodes, &ids, &mut t, era, &cfg, &mids, &mut regs);
        }

        // У честных склады полны; дельта дробилась на НЕСКОЛЬКО кусков
        // (blob_chunk_len: 256 при 1792 байтах дельты → 7 кусков).
        let target = regs[0].head().expect("head после reveal").entry.manifest_id;
        assert_ne!(target, GENESIS);
        let n_ref = nodes[1].blobs.chunks(&target).expect("куски узла 1").len();
        assert!(n_ref >= 4, "ожидалась настоящая нарезка, кусков: {n_ref}");
        for i in 0..6 {
            assert!(nodes[i].blobs.has(&target), "узел {i} без блоба head");
            assert_eq!(
                nodes[i].blobs.chunks(&target).expect("куски").len(),
                n_ref,
                "узел {i}: расходится число кусков"
            );
        }

        // --- Потери: узел 0 — полный вайп; узел 2 — потерян кусок 3.
        nodes[0].blobs.clear();
        nodes[2].blobs.drop_chunk(&target, 3);

        let rep = torrent_blobs(&mut nodes, &ids, &mut t, &regs);
        assert!(rep[0].assembled, "узел 0 не собрал блоб из пиров: {rep:?}");
        assert!(rep[1].assembled);
        assert!(rep[2].assembled, "узел 2 (частичный сид) не восстановился");

        // Байт-в-байт с эталоном узла 1.
        let reference = nodes[1].blobs.chunks(&target).expect("эталон").clone();
        let got = nodes[0].blobs.chunks(&target).expect("собранный").clone();
        let got2 = nodes[2].blobs.chunks(&target).expect("частичный сид").clone();
        assert_eq!(got, reference, "сборка узла 0 расходится с эталоном");
        assert_eq!(got2, reference, "восстановление узла 2 расходится");

        // кусок 3 узла 2 потерян → узел 0 обязан брать его у другого:
        assert!(rep[0].peers >= 2, "ожидались куски из разных источников");

        // Повторный прогон: всё на месте, ничего не сломалось.
        let rep2 = torrent_blobs(&mut nodes, &ids, &mut t, &regs);
        assert!(rep2.iter().all(|r| r.assembled), "второй прогон: {rep2:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
