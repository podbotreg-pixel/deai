//! reg_sync — ФАЗА H: догоняющая синхронизация реестра для отставшего.
//!
//! Пришедший узел (или рестартнувшийся с чистым диском) получает хвост
//! append-only лога у пиров и ДОПИСЫВАЕТ его СЕБЕ через тот же
//! `RegistryLog::append` — проверки цепочки, подписей и кворума
//! выполняются заново на чужих байтах, так что нечестный пир не может
//! влить отставшему подделку:
//! 1. отставший broadcast `RegAsk { have }` (сколько записей уже есть);
//! 2. узлы с более длинным логом отвечают направленно `RegOffer` —
//!    сетевые байты записей ([len][entry][seals], тот же формат хвоста
//!    файла); несвоенные офферы тоже принимаются (round-нейтрально), как
//!    в настоящем gossip-протоколе — на чужих байтах append сам отклонит форк/подделку.
//!
//! Офферы пробуются по одному (длинные первыми): что прицепилось до
//! первого сбоя, то остаётся (append-only монотонность), проваленный
//! оффер дальше не читается.

use crate::swarm::{decode_frames, peer_book_ids, sign_frame, Node, NodeId};
use crate::transport::{CH_REG, Transport};
use dinet_proto::{MsgType, Payload};
use dinet_registry::{decode_record, encode_record, RegistryLog};

/// Отчёт фазы по узлу.
#[derive(Debug, Clone, PartialEq)]
pub struct RegSyncReport {
    /// Сколько записей было до фазы.
    pub from: usize,
    /// Сколько записей стало (прицепилось из чужого хвоста).
    pub to: usize,
}

/// ФАЗА H: один раунд ask→offer→append.
pub fn reg_sync(
    mine: &mut [Node],
    all_ids: &[NodeId],
    transport: &mut dyn Transport,
    regs: &mut [RegistryLog],
) -> Vec<RegSyncReport> {
    // 1) все объявляют свою длину
    for idx in 0..mine.len() {
        let have = regs[idx].len() as u32;
        let frame = sign_frame(&mine[idx], MsgType::RegAsk, Payload::RegAsk { have });
        transport.send(CH_REG, mine[idx].id, None, &frame);
    }

    // 2) ВСЕ сначала забирают всё, что есть (строгий барьер: ни одного
    //    оффера в чужом inbox на этот момент), декод, разбор отдельно
    let mut frames: Vec<Vec<(NodeId, Payload)>> = (0..mine.len()).map(|_| Vec::new()).collect();
    for idx in 0..mine.len() {
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_REG, mine[idx].id);
        let me = mine[idx].id;
        frames[idx] = decode_frames(&mut mine[idx], &book, wire)
            .into_iter()
            .filter(|(nid, _)| *nid != me)
            .collect();
    }
    // 2b) разбор: RegAsk → направленный оффер от более длинного хвоста;
    //     RegOffer (несвоенный) → append сразу (он сам всё перепроверяет)
    for idx in 0..mine.len() {
        let my = regs[idx].len() as u32;
        for (nid, pl) in &frames[idx] {
            match pl {
                Payload::RegAsk { have } => {
                    if *have >= my {
                        continue; // просивший не отстаёт от нас
                    }
                    // хвост: записи, которых у него нет (по индексам seq)
                    let from = *have as usize;
                    let records: Vec<Vec<u8>> = regs[idx]
                        .entries()
                        .iter()
                        .skip(from)
                        .map(encode_record)
                        .collect();
                    if records.is_empty() {
                        continue;
                    }
                    let frame = sign_frame(
                        &mine[idx],
                        MsgType::RegOffer,
                        Payload::RegOffer { records },
                    );
                    transport.send(CH_REG, mine[idx].id, Some(*nid), &frame);
                }
                Payload::RegOffer { records } => {
                    accept_records(&mut mine[idx], &mut regs[idx], records);
                }
                _ => {}
            }
        }
    }

    let mut reports = (0..mine.len())
        .map(|idx| RegSyncReport { from: regs[idx].len(), to: regs[idx].len() })
        .collect::<Vec<_>>();

    // 3) отставшие принимают приезжавшие офферы: append сам фильтрует
    //    подделки; поздние RegAsk больше никому не нужны
    for idx in 0..mine.len() {
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_REG, mine[idx].id);
        let payloads = decode_frames(&mut mine[idx], &book, wire);
        for (_nid, pl) in payloads {
            if let Payload::RegOffer { records } = pl {
                accept_records(&mut mine[idx], &mut regs[idx], &records);
            }
        }
        reports[idx].to = regs[idx].len();
    }
    reports
}

/// Одна попытка влить пачку чужих записей: побайтовая распаковка, затем
/// полный append (цепочка/подписи/кворум) — первый сбой обрывает пачку.
fn accept_records(me: &mut Node, reg: &mut RegistryLog, records: &[Vec<u8>]) {
    for rec_bytes in records {
        let Ok(rec) = decode_record(rec_bytes) else {
            me.rejected_frames += 1;
            break; // брак в пачке: оффер дальше не читаем
        };
        // append пере-проверяет: seq/parent-цепочку, подписи печатей,
        // кворум distinct pubkey — чужой байт не влить
        if reg.append(rec).is_err() {
            me.rejected_frames += 1;
            break; // форк/дыра: с этим оффером покончено
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_torrent::torrent_blobs;
    use crate::registry_phase::{era_notary, notary_min};
    use crate::swarm::{
        era_collect_and_vote, era_publish, era_tally, Node as SwarmNode, SimConfig,
    };
    use crate::transport::MeshTransport;
    use dinet_registry::GENESIS;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("dinet-regsync-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("tmp");
        d
    }

    /// «Новичок присоединился»: 4 узла прогнали 3 эры (третья — REVEAL,
    /// чек-пойнт + реестр + склад), затем приходит узел 4 с пустым логом.
    /// ФАЗА H догоняет его реестр бит-в-бит; фаза G следом доливает блоб.
    #[test]
    fn newcomer_catches_up_registry_then_blob() {
        let cfg = SimConfig {
            reveal_every: 3,
            blob_chunk_len: 256,
            ..Default::default()
        };
        let dir = tmpdir("join");
        let mut nodes: Vec<SwarmNode> =
            (0..4usize).map(|i| SwarmNode::new(i as NodeId, 7, false)).collect();
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let mut t = MeshTransport::new(4);
        let mut regs: Vec<RegistryLog> = (0..4)
            .map(|i| {
                RegistryLog::open(&dir.join(format!("r{i}.dlog")), notary_min(4)).expect("reg")
            })
            .collect();

        for era in 0..3u32 {
            let truths = era_publish(&mut nodes, &ids, &mut t, era, &cfg, 7);
            let mids =
                era_collect_and_vote(&mut nodes, &ids, &mut t, era, &cfg, Some((&truths, ids[0])));
            let _reps = era_tally(&mut nodes, &ids, &mut t, era, &cfg, mids.clone());
            let _not = era_notary(&mut nodes, &ids, &mut t, era, &cfg, &mids, &mut regs);
        }
        assert_eq!(regs[0].len(), 1, "после reveal реестр держит 1 запись");
        let target = regs[0].head().expect("head").entry.manifest_id;
        assert_ne!(target, GENESIS);

        // --- НОВИЧОК: пятый узел с чистым логом и пустым складом
        let log_path = dir.join("r4.dlog");
        regs.push(RegistryLog::open(&log_path, notary_min(4)).expect("reg новичка"));
        nodes.push(SwarmNode::new(4, 7, false));
        let all: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let mut t5 = MeshTransport::new(5); // общий транспорт на 5 inbox'ов

        // фаза H: новичок догоняет лог (целиком, с нотарными печатями)
        let rep = reg_sync(&mut nodes, &all, &mut t5, &mut regs);
        assert_eq!(rep[4].from, 0, "у новичка был пустой лог");
        assert_eq!(rep[4].to, 1, "новичок прицепил запись от пиров");

        // бит-в-бит с эталоном (подписи снова проверяются при переоткрытии)
        let full = RegistryLog::open(&log_path, notary_min(4)).expect("переоткрытие");
        assert_eq!(full.entries(), regs[0].entries(), "реестр новичка расходится");
        assert_eq!(full.len(), 1);

        // фаза G следом: новичок без склада собирает блоб head из кусков
        let rep2 = torrent_blobs(&mut nodes, &all, &mut t5, &regs);
        assert!(rep2[4].assembled, "блоб новичка не собрался: {rep2:?}");
        let got = nodes[4].blobs.chunks(&target).expect("склад новичка");
        let etalon = nodes[0].blobs.chunks(&target).expect("эталон");
        assert_eq!(got, etalon, "куски новичка расходятся с эталоном");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Подделка не пролезает: скомпрометированный оффер (чужой seq/подпись)
    /// отбрасывается append'ом, честная часть хвоста всё же прицепляется.
    #[test]
    fn poisoned_offer_partial_chain() {
        let cfg = SimConfig {
            reveal_every: 2, // две reveal-эры: 2 записи в реестре
            blob_chunk_len: 8192,
            ..Default::default()
        };
        let dir = tmpdir("poison");
        let mut nodes: Vec<SwarmNode> =
            (0..4usize).map(|i| SwarmNode::new(i as NodeId, 7, false)).collect();
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let mut t = MeshTransport::new(4);
        let mut regs: Vec<RegistryLog> = (0..4)
            .map(|i| {
                RegistryLog::open(&dir.join(format!("p{i}.dlog")), notary_min(4)).expect("reg")
            })
            .collect();
        for era in 0..4u32 {
            let truths = era_publish(&mut nodes, &ids, &mut t, era, &cfg, 7);
            let mids =
                era_collect_and_vote(&mut nodes, &ids, &mut t, era, &cfg, Some((&truths, ids[0])));
            let _reps = era_tally(&mut nodes, &ids, &mut t, era, &cfg, mids.clone());
            let _not = era_notary(&mut nodes, &ids, &mut t, era, &cfg, &mids, &mut regs);
        }
        assert_eq!(regs[0].len(), 2, "две reveal-эры: 2 записи");
        assert!(nodes[0].blobs.len() >= 1, "склад узла 0 держит блобы");

        // --- отстающий: узел 3 с ОДНОЙ первой записью (формат честный байт)
        let log3 = dir.join("p3-catch.dlog");
        {
            // сохраняем первую запись честным байтом и пере-создаём лог
            let raw = std::fs::read(dir.join("p3.dlog")).expect("файл p3");
            let cut = &raw[..4 + u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize];
            std::fs::write(&log3, cut).expect("хвост p3");
        }
        regs[3] = RegistryLog::open(&log3, notary_min(4)).expect("отстающий");
        assert_eq!(regs[3].len(), 1);

        // подсовываем ПОДДЕЛЬНЫЙ оффер: подпись нотариуса поверх чужого
        // канона (злоумышленник не может пере-подписать канон чужим ключом)
        let mut t2 = MeshTransport::new(4);
        let mut poisoned_offer: Vec<Vec<u8>> = Vec::new();
        let fake_entry = dinet_registry::RegistryEntry {
            seq: 99,
            checkpoint_id: [7u8; 32],
            parent_id: [8u8; 32],
            era_id: [9u8; 32],
            manifest_id: [10u8; 32],
            accuracy: 0.99,
            contributors: 4,
        };
        let fake_seal = {
            use dinet_crypto::{Signer, ED25519_PUB_LEN};
            let s = dinet_crypto::Ed25519Signer::from_seed(&[42u8; 32]);
            let mut pk = [0u8; ED25519_PUB_LEN];
            pk.copy_from_slice(&s.pubkey());
            dinet_registry::Seal { pubkey: pk, sig: Signer::sign(&s, b"foreign-canon") }
        };
        let fake = dinet_registry::Accepted { entry: fake_entry, seals: vec![fake_seal] };
        poisoned_offer.push(dinet_registry::encode_record(&fake));
        let frame = crate::swarm::sign_frame(
            &mut nodes[0],
            MsgType::RegOffer,
            Payload::RegOffer { records: poisoned_offer },
        );
        t2.send(crate::transport::CH_REG, 0, Some(3), &frame);

        let rep = reg_sync(&mut nodes, &ids, &mut t2, &mut regs);
        // фальшивая запись: append отклонил (нет кворума/цепочка мимо),
        // честный хвост при этом прицепился — отстающий догнал догоняющих
        assert_eq!(rep[3].to, 2, "честный хвост не прицепился: {rep:?}");
        assert!(nodes[3].rejected_frames > 0, "фальшивый кадр не замечен");

        // бит-в-бит идентичен эталону: скомпрометанной записи нет, ни
        // лишней, ни пропавшей
        let after = RegistryLog::open(&log3, notary_min(4)).expect("после sync");
        assert_eq!(after.entries(), regs[0].entries(), "лог отстающего расходится с эталоном");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
