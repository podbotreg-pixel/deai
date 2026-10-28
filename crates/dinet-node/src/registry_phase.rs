//! registry_phase — ФАЗА F нотариев: печать чек-пойнта в реестр роя.
//!
//! После tally REVEAL-эры:
//! 1. каждый узел строит КАНОН записи со СВОЕЙ точки реестра
//!    (seq = len, parent = head.id или GENESIS; checkpoint_id из голоса,
//!    manifest_id из канонической дельты, accuracy из eval) и рассылает
//!    `NotarySeal` (своя подпись ПОВЕРХ канона; frame-подпись — вторая).
//! 2. каждый собирает чужие печати, отбрасывает чужие каноны (форк/разгон)
//!    и мимо-подписи, а свою запись дописывает в СВОЙ `RegistryLog` при
//!    кворуме distinct-публичников.
//!
//! Инварианты (тесты ниже):
//! - masked-эры реестр НЕ растёт;
//! - честные реестры после эр бит-в-бит равны (entry + печати);
//! - византийский канон и подделка печати не пролезают.

use crate::swarm::{decode_frames, era_hash32, peer_book_ids, sign_frame, MidNode, Node, NodeId};
use crate::transport::{CH_NOTARY, Transport};
use dinet_crypto::Verifier;
use dinet_proto::{MsgType, Payload};
use dinet_registry::{Accepted, DeltaManifest, RegistryEntry, RegistryLog, Seal};

/// Минимум печатей: ceil(2N/3) от полного состава роя (тот же принцип,
/// что accepts*3 >= votes*2; молчащий нотариус = отказ).
pub fn notary_min(roster: usize) -> usize {
    (roster * 2 + 2) / 3
}

/// Отчёт фазы по узлу.
#[derive(Debug, Clone, PartialEq)]
pub struct NotaryReport {
    /// Запись попала в реестр под кворумом.
    pub accepted: bool,
    /// Сколько distinct печатей набрано (при неуспехе — сколько было).
    pub seals: usize,
}

/// Канонический манифест дельты: f32 LE (тот же выбор байтов, что в wire).
pub fn delta_f32_manifest(delta: &[f32]) -> DeltaManifest {
    let mut bytes = Vec::with_capacity(delta.len() * 4);
    for v in delta {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    DeltaManifest::pack(&bytes, dinet_registry::DEFAULT_CHUNK_LEN).0
}

/// Канон записи узла для этой эры (с учётом head ЕГО реестра).
fn expected_entry(reg: &RegistryLog, era: u32, mid: &MidNode) -> Option<RegistryEntry> {
    let v = mid.vote.as_ref()?;
    let (seq, parent) = match reg.head() {
        None => (0u64, dinet_registry::GENESIS),
        Some(h) => (h.entry.seq + 1, h.entry.checkpoint_id),
    };
    Some(RegistryEntry {
        seq,
        checkpoint_id: v.candhash,
        parent_id: parent,
        era_id: era_hash32(era),
        manifest_id: delta_f32_manifest(&v.agg_delta).id(),
        accuracy: v.cand_acc,
        contributors: mid.contributors as u16,
    })
}

/// ФАЗА F: печать реестра (после `era_tally`, тот же mids, тот же transport).
pub fn era_notary(
    mine: &mut [Node],
    all_ids: &[NodeId],
    transport: &mut dyn Transport,
    era: u32,
    mids: &[MidNode],
    regs: &mut [RegistryLog],
) -> Vec<NotaryReport> {
    let was_reveal = mids.iter().any(|m| m.vote.is_some());

    // --- 1. Печати: канон со своей точки + NotarySeal (broadcast)
    for idx in 0..mine.len() {
        if !was_reveal {
            continue;
        }
        let Some(entry) = expected_entry(&regs[idx], era, &mids[idx]) else { continue };
        let canon = entry.encode();
        let seal_sig = mine[idx].signer.sign(&canon);
        let pub_bytes = mine[idx].signer.pubkey();
        let frame = sign_frame(
            &mine[idx],
            MsgType::NotarySeal,
            Payload::NotarySeal {
                entry_bytes: canon,
                seal_pubkey: pub_bytes.to_vec(),
                seal_sig: seal_sig.to_vec(),
            },
        );
        transport.send(CH_NOTARY, mine[idx].id, None, &frame);
    }

    // --- 2. Сбор чужих печатей; по кворуму — запись в свой реестр
    let mut reports = Vec::with_capacity(mine.len());
    for idx in 0..mine.len() {
        if !was_reveal {
            reports.push(NotaryReport { accepted: false, seals: 0 });
            continue;
        }
        let Some(my_entry) = expected_entry(&regs[idx], era, &mids[idx]) else {
            reports.push(NotaryReport { accepted: false, seals: 0 });
            continue;
        };
        let my_canon = my_entry.encode();
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_NOTARY, mine[idx].id);
        let payloads = decode_frames(&mut mine[idx], &book, wire);
        let mut seals: Vec<Seal> = Vec::new();
        for (_nid, pl) in payloads.iter() {
            let Payload::NotarySeal { entry_bytes, seal_pubkey, seal_sig } = pl else {
                mine[idx].rejected_frames += 1;
                continue;
            };
            // 1) канон должен быть ТОЧНЫМ дубликатом моего (форк/разгон мимо)
            if *entry_bytes != my_canon {
                mine[idx].rejected_frames += 1;
                continue;
            }
            // 2) печать ed25519 поверх канона; ключ внутри payload
            if seal_pubkey.len() != 32 || seal_sig.len() != 64 {
                mine[idx].rejected_frames += 1;
                continue;
            }
            let mut pk = [0u8; 32];
            pk.copy_from_slice(seal_pubkey);
            let mut sig = [0u8; 64];
            sig.copy_from_slice(seal_sig);
            let verifier = match dinet_crypto::Ed25519Verifier::new(pk) {
                Some(v) => v,
                None => {
                    mine[idx].rejected_frames += 1;
                    continue;
                }
            };
            if !verifier.verify(&my_canon, &sig) {
                mine[idx].rejected_frames += 1;
                continue;
            }
            if seals.iter().any(|s| s.pubkey == pk) {
                continue; // дубликат нотариуса не считается
            }
            seals.push(Seal { pubkey: pk, sig });
        }
        let min = notary_min(all_ids.len());
        if seals.len() >= min {
            regs[idx]
                .append(Accepted { entry: my_entry.clone(), seals })
                .unwrap_or_else(|e| panic!("канон валиден: append не может падать: {e}"));
            reports.push(NotaryReport { accepted: true, seals: regs[idx].len() });
        } else {
            reports.push(NotaryReport { accepted: false, seals: seals.len() });
        }
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::{era_collect_and_vote, era_publish, era_tally, Node as SwarmNode, SimConfig};
    use crate::transport::MeshTransport;
    use dinet_fl::toy_llm::{LoRAAdapter, ToyBase};
    use dinet_registry::GENESIS;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("dinet-notary-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("tmp");
        d
    }

    fn roster(n: usize, byz_at: Option<usize>) -> Vec<SwarmNode> {
        (0..n)
            .map(|i| SwarmNode::new(i as NodeId, 7, byz_at == Some(i)))
            .collect()
    }

    /// Полный прогон эры publish→collect/vote→tally→notary с ОДНИМ transport.
    fn run_era_with_notary(
        nodes: &mut Vec<SwarmNode>,
        ids: &[NodeId],
        t: &mut MeshTransport,
        era: u32,
        cfg: &SimConfig,
        regs: &mut [RegistryLog],
    ) -> Vec<NotaryReport> {
        let truths = era_publish(nodes, ids, t, era, cfg, 7);
        let mids = era_collect_and_vote(nodes, ids, t, era, cfg, Some((&truths, ids[0])));
        let _reps = era_tally(nodes, ids, t, era, cfg, mids.clone());
        era_notary(nodes, ids, t, era, &mids, regs)
    }

    /// Четыре узла (1 byz), эры masked/masked/REVEAL: реестр растёт ТОЛЬКО
    /// на reveal; честные реестры бит-в-бит равны; byz-канон не попадает.
    #[test]
    fn notary_ledger_reveal_only_and_bitidentical() {
        let cfg = SimConfig { reveal_every: 3, ..SimConfig::default() };
        let dir = tmpdir("match");
        let mut nodes = roster(4, Some(3)); // byz — узел 3
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let mut t = MeshTransport::new(4);
        let mut regs: Vec<RegistryLog> = (0..4)
            .map(|i| {
                RegistryLog::open(&dir.join(format!("r{i}.dlog")), notary_min(4))
                    .expect("reg")
            })
            .collect();

        for era in 0..3u32 {
            let rep = run_era_with_notary(&mut nodes, &ids, &mut t, era, &cfg, &mut regs);
            let want_reveal = (era + 1) % cfg.reveal_every == 0;
            for (i, r) in rep.iter().enumerate() {
                assert_eq!(
                    r.accepted, want_reveal,
                    "узел {i} эра {era}: accepted={} при reveal={want_reveal}",
                    r.accepted
                );
            }
        }
        // честные реестры (0..=2) бит-в-бит равны; byz (3) печатал СВОЙ
        // канон: его печати отклоняются на каноне, но его собственный
        // реестр набрал те же честные печати (их каноны совпали с его
        // собственной точкой — предсказуемо для r3 как зеркала роя).
        let a = &regs[0];
        let b = &regs[1];
        let c = &regs[2];
        assert!(!a.is_empty(), "reveal-эра не дописала реестр");
        assert_eq!(a.entries(), b.entries(), "реестры 0 и 1 расходятся");
        assert_eq!(a.entries(), c.entries(), "реестры 0 и 2 расходятся");
        assert_eq!(a.len(), 1, "третья эра (REVEAL) дала одну запись");

        // цепочка корректна: parent первой записи == GENESIS, seq с нуля
        let first = &a.entries()[0];
        assert_eq!(first.entry.parent_id, GENESIS);
        assert_eq!(first.entry.seq, 0);
        assert!(first.seals.len() >= notary_min(4), "печатей меньше кворума");

        // файлы реестров непусты (persist идёт с append)
        for i in 0..4 {
            let path = dir.join(format!("r{i}.dlog"));
            let raw = std::fs::read(&path).expect("файл реестра");
            assert!(!raw.is_empty(), "реестр {i} пустой файл");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Кворум ceil(2N/3): при 4 узлах нужно 3 печати; двое не пролезут.
    #[test]
    fn notary_min_math() {
        assert_eq!(notary_min(1), 1);
        assert_eq!(notary_min(3), 2);
        assert_eq!(notary_min(4), 3);
        assert_eq!(notary_min(6), 4);
    }
}
