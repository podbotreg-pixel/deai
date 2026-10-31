//! anchors — ФАЗА I: внешние якоря (анти-откат).
//!
//! append-only реестр нельзя откатить, пока жив ОДИН честный держатель —
//! но держатели могут все исчезнуть (диски утрачены), а узел-самозванец
//! вернуться с укороченной/подменённой цепью. Кольцо внешних анкоров —
//! независимых сервисов БЕЗ ключей роя и БЕЗ весов — хранит только пары
//! (seq, checkpoint_id) голов реестра:
//!
//! 1. после нотарного append кворум эры печатает голову в кольцо
//!    (`AnchorStamp 0x56`); анкор коммитит пару лишь собрав кворум
//!    `notary_min(roster)` РАЗНЫХ рой-ключей над одинаковым
//!    (seq, head_id) — единственный лгущий узел голову не запечатает;
//! 2. узел рестарта спрашивает кольцо (`AnchorAsk 0x57`); анкоры
//!    отвечают последним комитом (`AnchorTell 0x58`), узел считает
//!    большинство: пара от `notary_min(анкоров)` разных анкоров;
//! 3. вердикт: локальная цепь содержит подкреплённую пару — Ok (все
//!    записи связаны строгой parent-цепочкой и checkpoint_id —
//!    sha256-канон: пару нельзя показать без настоящих записей), нет —
//!    Fork: локальный хвост (откат/подмена) не авторитетен, реестр
//!    догоняется ФАЗОЙ H у пиров.
//!
//! Монотонность: анкор не откатывает и не переписывает комиты; два
//! кворумных форка на одном seq взаимно гасятся (никто не коммитится).
//! Файл растёт 36 байтами за эру и годится лежать на любой посторонней
//! машине.

use crate::registry_phase::notary_min;
use crate::swarm::{peer_book_ids, sign_frame, Node, NodeId};
use crate::transport::{CH_ANCHOR, Transport};
use dinet_proto::{Frame, MsgType, Payload};

/// Один внешний анкор: append-only список закоммиченных голов реестра.
pub struct AnchorStore {
    /// файл комитов (None — только в памяти: sim/тесты)
    path: Option<std::path::PathBuf>,
    /// комиты (seq, head_id) в порядке эр
    committed: Vec<(u32, [u8; 32])>,
    /// сколько комитов загружено/записано в файл
    persisted: usize,
    /// штампы до кворума: (seq, head_id) → набор РАЗНЫХ рой-ключей
    pending: Vec<((u32, [u8; 32]), Vec<[u8; 32]>)>,
}

impl AnchorStore {
    /// Анкор в памяти.
    pub fn new() -> AnchorStore {
        AnchorStore {
            path: None,
            committed: Vec::new(),
            persisted: 0,
            pending: Vec::new(),
        }
    }

    /// Анкор с файлом (внешняя машина); существующий файл подхватывается.
    pub fn open(path: std::path::PathBuf) -> AnchorStore {
        let mut s = AnchorStore {
            path: Some(path),
            committed: Vec::new(),
            persisted: 0,
            pending: Vec::new(),
        };
        let Some(p) = s.path.clone() else { return s };
        if let Ok(raw) = std::fs::read(&p) {
            // формат: повтор [seq u32 LE][head_id 32B] = 36 байт
            for chunk in raw.chunks_exact(36) {
                let seq = u32::from_le_bytes(chunk[..4].try_into().expect("36 байт"));
                let mut id = [0u8; 32];
                id.copy_from_slice(&chunk[4..36]);
                s.committed.push((seq, id));
            }
        }
        s.persisted = s.committed.len();
        s
    }

    /// Последний комит.
    pub fn head(&self) -> Option<(u32, [u8; 32])> {
        self.committed.last().copied()
    }

    /// Все комиты.
    pub fn committed(&self) -> &[(u32, [u8; 32])] {
        &self.committed
    }

    /// Штамп (подпись проверена анкором). Возврат: на этой паре набрался
    /// кворум разных голосов.
    fn seal_stamp(&mut self, seq: u32, head_id: [u8; 32], voter: &[u8; 32], min: usize) -> bool {
        // комит-лист монотонный: откат нельзя, повтор штампа по seq не голос
        if self.committed.last().is_some_and(|&(s, _)| s >= seq) {
            return false;
        }
        let mut i = match self.pending.iter().position(|(k, _)| *k == (seq, head_id)) {
            Some(i) => i,
            // пара встречена впервые (в т.ч. форк на этот seq — своя строка)
            None => {
                self.pending.push(((seq, head_id), Vec::new()));
                self.pending.len() - 1
            }
        };
        let voters = &mut self.pending[i].1;
        if voters.iter().any(|v| v == voter) {
            return false; // повтор от того же ключа голосом не считается
        }
        voters.push(*voter);
        i = self.pending[i].1.len();
        i >= min
    }

    /// Докоммитить кворумные пары по возрастанию seq (строго выше
    /// последнего комита; разрывы допустимы — анкор держит головы,
    /// не всю цепь). Два разных кворума на одном seq = конфликт:
    /// стопорится до внешнего разбора (сигнал, что кворум раскололся).
    /// Возврат: сколько пар закоммичено в этот прогон.
    fn commit_ready(&mut self, min: usize) -> usize {
        let mut n = 0usize;
        loop {
            let last = self.committed.last().map(|&(s, _)| s);
            let mut ready: Vec<(u32, [u8; 32])> = self
                .pending
                .iter()
                .filter(|(k, v)| v.len() >= min && last.is_none_or(|l| k.0 > l))
                .map(|(k, _)| *k)
                .collect();
            ready.sort();
            match ready.as_slice() {
                [only] => {
                    let rec = *only;
                    self.pending.retain(|(k, _)| *k != rec);
                    self.committed.push(rec);
                    self.persist(rec);
                    n += 1;
                }
                [] => break,
                // несколько кворумных голов на одном seq — конфликт
                _ => break,
            }
        }
        n
    }

    /// Дописать одну пару в файл (36 байт, append-only).
    fn persist(&mut self, rec: (u32, [u8; 32])) {
        let Some(p) = &self.path.clone() else { return };
        use std::io::Write;
        self.persisted = self.committed.len();
        let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) else {
            return;
        };
        let mut buf = Vec::with_capacity(36);
        buf.extend_from_slice(&rec.0.to_le_bytes());
        buf.extend_from_slice(&rec.1);
        let _ = f.write_all(&buf);
        let _ = f.flush();
    }
}

impl Default for AnchorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Подкреплённая кольцом пара (вердикт рестарта).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorBond {
    pub seq: u32,
    pub head_id: [u8; 32],
}

/// Вердикт рестарта.
#[derive(Debug, Clone, PartialEq)]
pub enum AnchorVerdict {
    /// подкреплённая пара есть в локальной цепи — локальный реестр честен
    Ok,
    /// кольцо молчит (анкоров нет/ни один не ответил) — внешнего мнения нет
    Silent,
    /// локальная цепь расходится с кольцом: откат/подмена/утрачен диск —
    /// хвост не авторитетен, реестр догоняется ФАЗОЙ H
    Fork,
}

/// ФАЗА I: печать головы реестра в кольцо анкоров. Возврат: по анкору —
/// сколько НОВЫХ пар закоммичено в этом прогоне.
pub fn anchors_seal(
    mine: &mut [Node],
    all_ids: &[NodeId],
    regs: &[dinet_registry::RegistryLog],
    anchors: &mut [AnchorStore],
    transport: &mut dyn Transport,
) -> Vec<usize> {
    let min = notary_min(all_ids.len());
    let anchor_ids: Vec<NodeId> = (0..anchors.len())
        .map(|i| all_ids.len() as NodeId + i as NodeId)
        .collect();

    // 1) каждый рой-узел штампует свою голову (прямо каждому анкору —
    //    рой-инбоксы не мусорятся)
    for idx in 0..mine.len() {
        let Some(rec) = regs[idx].head() else { continue };
        let frame = sign_frame(
            &mine[idx],
            MsgType::AnchorStamp,
            Payload::AnchorStamp {
                head_seq: rec.entry.seq as u32,
                head_id: rec.entry.checkpoint_id,
            },
        );
        for &aid in &anchor_ids {
            transport.send(CH_ANCHOR, mine[idx].id, Some(aid), &frame);
        }
    }

    // 2) анкоры: принять штампы, сверить подписи РОЯ, ловить кворум
    let book = peer_book_ids(all_ids);
    let mut out = vec![0usize; anchors.len()];
    for (ai, a) in anchors.iter_mut().enumerate() {
        let aid = anchor_ids[ai];
        for raw in transport.collect(CH_ANCHOR, aid) {
            let Ok(f) = Frame::decode(&raw) else { continue };
            let Some(v) = book.verifiers.get(&f.sender) else { continue };
            let Ok(body) = f.encode_payload() else { continue };
            if !v.verify(&body, &f.sig) {
                continue;
            }
            let Payload::AnchorStamp { head_seq, head_id } = f.payload else { continue };
            a.seal_stamp(head_seq, head_id, &f.sender, min);
        }
        out[ai] = a.commit_ready(min);
    }
    out
}

/// ФАЗА I': узел рестарта собирает мнение кольца анкоров.
/// Возврат: по узлу — пара большинства (anchor_min разных анкоров
/// согласились), None — кольцо молчит/распалось.
pub fn anchors_bonded(
    mine: &mut [Node],
    all_ids: &[NodeId],
    anchors: &mut [AnchorStore],
    transport: &mut dyn Transport,
) -> Vec<Option<AnchorBond>> {
    let anchor_min = notary_min(anchors.len());
    let anchor_ids: Vec<NodeId> = (0..anchors.len())
        .map(|i| all_ids.len() as NodeId + i as NodeId)
        .collect();
    let book = peer_book_ids(all_ids); // рой-пуб-ключ → NodeId (адрес ответа)

    // 1) все узлы спрашивают (прямо каждому анкору)
    for idx in 0..mine.len() {
        let frame = sign_frame(&mine[idx], MsgType::AnchorAsk, Payload::AnchorAsk);
        for &aid in &anchor_ids {
            transport.send(CH_ANCHOR, mine[idx].id, Some(aid), &frame);
        }
    }

    // 2) анкоры отвечают последним комитом в адрес просившего
    for (ai, a) in anchors.iter_mut().enumerate() {
        let aid = anchor_ids[ai];
        let Some((seq, id)) = a.head() else { continue };
        let tell = Payload::AnchorTell { head_seq: seq, head_id: id };
        for raw in transport.collect(CH_ANCHOR, aid) {
            let Ok(f) = Frame::decode(&raw) else { continue };
            let Payload::AnchorAsk = f.payload else { continue };
            let Some(&nid) = book.nid_by_sender.get(&f.sender) else { continue };
            // анкор не имеет рой-ключей: идентичность ответа — адрес
            // канала (в реальном слое — сетевой адрес анкора из конфига)
            let sender = [aid as u8; 32];
            let frame = Frame {
                msg_type: MsgType::AnchorTell,
                sender,
                sig: [0u8; 64],
                payload: tell.clone(),
            };
            let Ok(wire) = frame.encode_with_sender_sig(&sender, &[0u8; 64]) else {
                continue;
            };
            transport.send(CH_ANCHOR, aid, Some(nid), &wire);
        }
    }

    // 3) узлы собирают ответы: пара большинства anchor_min разных анкоров
    let mut bonds = vec![None; mine.len()];
    for idx in 0..mine.len() {
        let mut votes: std::collections::BTreeMap<(u32, [u8; 32]), Vec<u8>> =
            std::collections::BTreeMap::new();
        for raw in transport.collect(CH_ANCHOR, mine[idx].id) {
            let Ok(f) = Frame::decode(&raw) else { continue };
            let Payload::AnchorTell { head_seq, head_id } = f.payload else { continue };
            let list = votes.entry((head_seq, head_id)).or_default();
            if !list.contains(&f.sender[0]) {
                list.push(f.sender[0]);
            }
        }
        let mut best: Option<((u32, [u8; 32]), usize)> = None;
        for (k, v) in votes {
            if v.len() < anchor_min {
                continue; // единичный анкор мир не определяет
            }
            let better = match best {
                None => true,
                Some((bk, bc)) => v.len() > bc || (v.len() == bc && bk.0 < k.0),
            };
            if better {
                best = Some((k, v.len()));
            }
        }
        if let Some(((seq, id), _)) = best {
            bonds[idx] = Some(AnchorBond { seq, head_id: id });
        }
    }
    bonds
}

/// Пара (seq, checkpoint_id) есть в локальной цепи? Строгая
/// parent-цепочка делает пару невоспроизводимой без настоящих записей.
pub fn chain_has_pair(log: &dinet_registry::RegistryLog, seq: u32, head_id: [u8; 32]) -> bool {
    log.entries()
        .iter()
        .any(|a| a.entry.seq as u32 == seq && a.entry.checkpoint_id == head_id)
}

/// РЕШЕНИЕ рестарта: сверка подкреплённой пары кольца с локальной цепью.
pub fn restart_verdict(
    mine: &mut [Node],
    all_ids: &[NodeId],
    regs: &mut [dinet_registry::RegistryLog],
    anchors: &mut [AnchorStore],
    transport: &mut dyn Transport,
) -> Vec<AnchorVerdict> {
    let bonds = anchors_bonded(mine, all_ids, anchors, transport);
    (0..mine.len())
        .map(|idx| match bonds[idx] {
            None => AnchorVerdict::Silent,
            Some(b) => {
                if chain_has_pair(&regs[idx], b.seq, b.head_id) {
                    AnchorVerdict::Ok
                } else {
                    AnchorVerdict::Fork
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry_phase::era_notary;
    use crate::reg_sync::reg_sync;
    use crate::swarm::{
        era_collect_and_vote, era_publish, era_tally, SimConfig,
    };
    use crate::transport::MeshTransport;
    use dinet_registry::RegistryLog;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("dinet-anchor-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("tmp");
        d
    }

    /// 4 рой-узла × эры (reveal_every=2, 4 эры → 2 записи). Возврат:
    /// (узлы, ids, transport, regs, пары голов узла 0).
    fn seeded(
        tag: &str,
    ) -> (
        Vec<crate::swarm::Node>,
        Vec<NodeId>,
        MeshTransport,
        std::path::PathBuf,
        Vec<(u32, [u8; 32])>,
        Vec<AnchorStore>,
    ) {
        let cfg = SimConfig { reveal_every: 2, ..Default::default() };
        let dir = tmpdir(tag);
        let mut nodes: Vec<crate::swarm::Node> =
            (0..4usize).map(|i| crate::swarm::Node::new(i as NodeId, 7, false)).collect();
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let mut t = MeshTransport::new(6); // 4 роя + 2 анкора (id 4, 5)
        let mut regs: Vec<RegistryLog> = (0..4)
            .map(|i| {
                RegistryLog::open(&dir.join(format!("r{i}.dlog")), notary_min(4)).expect("reg")
            })
            .collect();
        // [0] в памяти, [1] с файлом внешней машины
        let mut anchors: Vec<AnchorStore> =
            vec![AnchorStore::new(), AnchorStore::open(dir.join("a1.danch"))];
        for era in 0..4u32 {
            let truths = era_publish(&mut nodes, &ids, &mut t, era, &cfg, 7);
            let mids = era_collect_and_vote(
                &mut nodes,
                &ids,
                &mut t,
                era,
                &cfg,
                Some((&truths, ids[0])),
            );
            let _r = era_tally(&mut nodes, &ids, &mut t, era, &cfg, mids.clone());
            let _n = era_notary(&mut nodes, &ids, &mut t, era, &cfg, &mids, &mut regs);
            // анкоры ходят сразу за нотариусами: печать после каждого
            // append (реальный клиент штампует сразу после эры)
            let _c = anchors_seal(&mut nodes, &ids, &regs, &mut anchors, &mut t);
        }
        assert_eq!(regs[0].len(), 2, "две reveal-эры: 2 записи");
        let cp: Vec<(u32, [u8; 32])> = regs[0]
            .entries()
            .iter()
            .map(|a| (a.entry.seq as u32, a.entry.checkpoint_id))
            .collect();
        (nodes, ids, t, dir, cp, anchors)
    }

    /// Свежие пере-открытия реестров (файлы тех же узлов).
    fn reopen_regs(dir: &std::path::Path) -> Vec<RegistryLog> {
        (0..4)
            .map(|i| {
                RegistryLog::open(&dir.join(format!("r{i}.dlog")), notary_min(4)).expect("пере-открытие")
            })
            .collect()
    }

    /// Кворум штампов коммитит честную голову; подделка (валидный ключ
    /// роя, ЛЕВАЯ голова) кворума не собирает и не коммитится; файл
    /// анкора переживает переоткрытие бит-в-бит.
    #[test]
    fn stamp_quorum_forgery_ignored() {
        let (mut nodes, ids, mut t, dir, cp, mut anchors) = seeded("stamp");
        let mut regs = reopen_regs(&dir);

        // докоммита нет: обе головы уже запечатаны fixture'ом после эр
        let counts = anchors_seal(&mut nodes, &ids, &regs, &mut anchors, &mut t);
        assert!(counts.iter().all(|&c| c == 0), "всё уже закоммичено: {counts:?}");
        for a in &anchors {
            assert_eq!(a.committed(), &cp[..], "комит не равен цепи реестра");
        }

        // ПОДДЕЛКА: минёр подписывает ЛЕВУЮ голову (head_seq=5) — голос
        // валидный, но кворум не собирается; штамп уже запечатанного
        // seq игнорируется (монотонность).
        let frame = sign_frame(
            &nodes[0],
            MsgType::AnchorStamp,
            Payload::AnchorStamp { head_seq: 5, head_id: [7u8; 32] },
        );
        let aid = (ids.len() as NodeId) + 1;
        t.send(crate::transport::CH_ANCHOR, nodes[0].id, Some(aid), &frame);
        let counts = anchors_seal(&mut nodes, &ids, &regs, &mut anchors, &mut t);
        assert!(counts.iter().all(|&c| c == 0), "левая голова докоммитилась: {counts:?}");
        for a in &anchors {
            assert_eq!(a.committed(), &cp[..], "левая голова пролезла");
        }

        // файл анкора пережил переоткрытие бит-в-бит
        let rela = AnchorStore::open(dir.join("a1.danch"));
        assert_eq!(rela.committed(), &cp[..], "файл анкора расходится");
    }

    /// АНТИ-ОТКАТ: узел вернулся с урезанным хвостом — вердикт Fork;
    /// фаза H добирает честный хвост у пиров — вердикт Ok. Честные
    /// узлы — Ok с первого опроса.
    #[test]
    fn rollback_detected_repaired_via_h() {
        let (mut nodes, ids, mut t, dir, cp, mut anchors) = seeded("rollback");
        let mut regs = reopen_regs(&dir);
        // печать гостевых голов уже прошла в fixture — докомму здесь нечего
        let counts = anchors_seal(&mut nodes, &ids, &regs, &mut anchors, &mut t);
        assert!(counts.iter().all(|&c| c == 0), "печать сорвалась: {counts:?}");
        for a in &anchors {
            assert_eq!(a.committed(), &cp[..], "комиты не равны цепи реестра");
        }

        // ЖЕРТВА: узел 0 откатывает файл до более короткого хвоста
        let victim = dir.join("victim.dlog");
        {
            let raw = std::fs::read(dir.join("r0.dlog")).expect("файл r0");
            let cut = &raw[..4 + u32::from_le_bytes(raw[..4].try_into().expect("len")) as usize];
            std::fs::write(&victim, cut).expect("рез");
        }
        let mut regs2 = reopen_regs(&dir);
        regs2[0] = RegistryLog::open(&victim, notary_min(4)).expect("жертва");
        assert_eq!(regs2[0].len(), 1, "хвост отброшен");

        // вердикт: жертва расходится с кольцом, остальные — нет
        let mut regs3 = reopen_regs(&dir);
        regs3[0] = RegistryLog::open(&victim, notary_min(4)).expect("жертва-2");
        let v = restart_verdict(&mut nodes, &ids, &mut regs3, &mut anchors, &mut t);
        assert_eq!(v[0], AnchorVerdict::Fork, "откат не замечен: {v:?}");
        assert!(v[1..].iter().all(|x| *x == AnchorVerdict::Ok), "честные не признались честными: {v:?}");
        // кольцо показывает ВТОРУЮ голову (максимальный seq)
        let bonds = anchors_bonded(&mut nodes, &ids, &mut anchors, &mut t);
        assert_eq!(bonds[0], Some(AnchorBond { seq: 1, head_id: cp[1].1 }));

        // ЛЕЧЕНИЕ ФАЗОЙ H: жертва добирает хвост у пиров — и снова
        // показывает подкреплённую голову
        let mut t2 = MeshTransport::new(4);
        let reps = reg_sync(&mut nodes, &ids, &mut t2, &mut regs2);
        assert_eq!(reps[0].to, 2, "фаза H не добрала хвост: {reps:?}");
        assert!(chain_has_pair(&regs2[0], 1, cp[1].1), "жертва не показывает мировую голову");
        let v2 = restart_verdict(&mut nodes, &ids, &mut regs2, &mut anchors, &mut t);
        assert_eq!(v2[0], AnchorVerdict::Ok, "после фазы H не Ok: {v2:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
