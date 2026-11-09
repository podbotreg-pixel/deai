//! infosync — обмен ИНФОСФЕРОЙ между реальными клиентами (требование
//! «общая память роя»: узлы регулярно обмениваются содержимым).
//!
//! Поверх ЛЮБОГО Transport (мост libp2p с каналом CH_INFO, файл-спул,
//! mesh) крутится одна фаза-тик:
//!   1. collect чужого → анонсы / аски / тела / печати;
//!   2. пуш СВОИХ ещё не анонсированных записей (квота за тик —
//!      вежливость) + релей ЧУЖИХ: инструкции — до RELAY_QUOTA/тик
//!      (правила роя важнее), факты/опыт — до RELAY_MISC_QUOTA/тик
//!      (разреженная сеть: сосед доносит то, что держит; держатель ли
//!      чужое тело — решает каждый сам, кепки вида защищают память);
//!   3. чужие аски → InfoBody из своего хранилища (мы держатели,
//!      кадр подписан нами, авторская подпись внутри payload);
//!   4. чужие тела → InfoStore.put (id/подпись автора проверяет put
//!      — подделка не сохранится); ЧУЖУЮ инструкцию печатаем своей
//!      подписью (признание роя: «не только автор сказал»);
//!   5. чужие печати → InfoStore.seal (проверка над canon, дедуп);
//!   6. чужие анонсы → InfoAsk отсутствующих id.
//! Сквозная верификация TOFU: в кадре sender = pubkey автора кадра,
//! подпись поверх canon-байт — чекается у всех одинаково.
//! За несколько тиков сходится к полным копиям (v0-репликация).

use dinet_crypto::{Ed25519Verifier, Signer, Verifier};
use dinet_node::infosphere::{
    canon as info_canon, craft_entry, InfoRec, InfoStore, KIND_INSTRUCTION,
};
use dinet_node::swarm::NodeId;
use dinet_node::transport::{Transport, CH_INFO};
use dinet_proto::{Frame, MsgType, Payload};
use std::collections::HashSet;

/// Максимум НАШИХ анонсов за один тик (вежливая квота).
pub const PUSH_QUOTA: usize = 8;
/// Максимум ЧУЖИХ инструкций, ретранслируемых за один тик (правила роя
/// должны доходить всем даже без прямых связей «автор → все»).
pub const RELAY_QUOTA: usize = 4;
/// Максимум ЧУЖИХ фактов/опыта, ретранслируемых за один тик (разреженная
/// сеть: сосед доносит то, что держит; квота — против размножения мусора).
pub const RELAY_MISC_QUOTA: usize = 6;
/// Максимум id в одном InfoAsk.
pub const ASK_QUOTA: usize = 16;
/// Максимум InfoBody, отвечаемых за один чужой аск.
pub const ANSWER_QUOTA: usize = 12;
/// Размер защиты обеих множеств от бесконечного роста.
const MEM_CAP: usize = 8192;

/// Состояние обмена между тиками (живёт в клиентском процессе).
#[derive(Debug, Default)]
pub struct InfoExchangeState {
    /// свои записи, о которых уже объявлялись (не кричим дважды).
    shared: HashSet<[u8; 32]>,
    /// чужие id, на которые уже просили тела (не спамим асками).
    asked: HashSet<[u8; 32]>,
    /// чужие инструкции, которые мы УЖЕ признали (печать выслана).
    sealed: HashSet<[u8; 32]>,
}

impl InfoExchangeState {
    pub fn new() -> InfoExchangeState {
        InfoExchangeState {
            shared: HashSet::new(),
            asked: HashSet::new(),
            sealed: HashSet::new(),
        }
    }
}

/// Подписанный кадр байтами: sender = pubkey ключа, sig поверх canon.
pub fn signed_payload_bytes(
    key: &dyn Signer,
    mt: MsgType,
    payload: Payload,
) -> Result<Vec<u8>, String> {
    let frame = Frame { msg_type: mt, sender: [0u8; 32], sig: [0u8; 64], payload };
    let body = frame.encode_payload().map_err(|e| format!("payload: {e:?}"))?;
    let sig = key.sign(&body);
    frame
        .encode_with_sender_sig(&key.pubkey(), &sig)
        .map_err(|e| format!("encode: {e:?}"))
}

/// Декод кадра с самопроверкой (TOFU): подпись совпала поверх canon.
pub fn frame_from_wire(bytes: &[u8]) -> Option<Frame> {
    let f = Frame::decode(bytes).ok()?;
    let body = f.encode_payload().ok()?;
    let v = Ed25519Verifier::new(f.sender)?;
    if v.verify(&body, &f.sig) {
        Some(f)
    } else {
        None // подпись чужая/какая-то левая — кадр выбрасываем
    }
}

/// Отчёт тика (для статов/лога).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InfoSinkReport {
    /// своих анонсов выслали
    pub pushed: u32,
    /// из чужих асков тел отвечено
    pub answered: u32,
    /// чужих тел приняты в хранилище
    pub taken: u32,
    /// чужих асков выставили (по анонсам)
    pub asked_out: u32,
    /// ЧУЖИХ инструкций ретранслировано (правила роя идут всем)
    pub relayed: u32,
    /// ЧУЖИХ фактов/опыта ретранслировано (соседи доносят держимое)
    pub relayed_other: u32,
    /// со-подписей выслали (признали чужие инструкции)
    pub seals_out: u32,
    /// печатей чужих приняли и проверили
    pub seals_in: u32,
}

/// Печать ЧУЖОЙ инструкции одним узлом: раз за жизнь записи у клиента.
/// Работает и если запись только пришла телом, и если она уже лежала
/// (id content-addressed — тот же id, тот же canon). Печать
/// подписывается ПОД canon и уходит в сеть; у себя тоже приложение.
#[allow(clippy::too_many_arguments)]
fn seal_foreign_instruction(
    key: &dyn Signer,
    me_pub: &[u8; 32],
    me_id: NodeId,
    info: &mut InfoStore,
    state: &mut InfoExchangeState,
    t: &mut dyn Transport,
    rep: &mut InfoSinkReport,
    rec: &InfoRec,
) {
    if rec.kind != KIND_INSTRUCTION || rec.author == *me_pub {
        return;
    }
    if state.sealed.contains(&rec.info_id) {
        return;
    }
    if state.sealed.len() > MEM_CAP {
        state.sealed.clear();
    }
    let seal = key.sign(&info_canon(rec.kind, rec.ts, &rec.author, &rec.value));
    let _ = info.seal(&rec.info_id, me_pub, &seal); // печать живёт и у нас
    state.sealed.insert(rec.info_id);
    let out = signed_payload_bytes(
        key,
        MsgType::InfoSeal,
        Payload::InfoSeal {
            info_id: rec.info_id,
            sealer: *me_pub,
            seal: seal.to_vec(),
        },
    );
    if let Ok(bytes) = out {
        t.send(CH_INFO, me_id, None, &bytes);
        rep.seals_out += 1;
    }
}

/// Один тик обмена поверх канала CH_INFO.
/// Возврат: отчёт (сколько чего от/приняли).
pub fn info_exchange_tick(
    key: &dyn Signer,
    me_id: NodeId,
    me_pub: &[u8; 32],
    info: &mut InfoStore,
    state: &mut InfoExchangeState,
    t: &mut dyn Transport,
) -> InfoSinkReport {
    let mut rep = InfoSinkReport::default();
    let mut new_ids: Vec<[u8; 32]> = Vec::new();

    // 1) СНАЧАЛА собираем всё чужое (вежливый барьер — не «заливаем»
    //    сеть до подключения входа: чужим тоже надо успеть собрать своё)
    for raw in t.collect(CH_INFO, me_id) {
        let Some(frame) = frame_from_wire(&raw) else { continue };
        if frame.sender == *me_pub {
            continue; // свой loopback
        }
        match frame.payload {
            Payload::InfoAnnounce { info_id, .. } => {
                if !info.has(&info_id) {
                    if !state.asked.contains(&info_id) {
                        new_ids.push(info_id);
                    }
                } else if let Some(r) = info.get(&info_id).cloned() {
                    // запись уже у нас (например, с прошлого запуска, ещё
                    // без печати) — признание можно выдать сейчас
                    seal_foreign_instruction(
                        key, me_pub, me_id, info, state, t, &mut rep,
                        &r,
                    );
                }
            }
            Payload::InfoAsk { ids } => {
                let mut n = 0u32;
                for id in ids.into_iter().filter(|id| info.has(id)) {
                    if n >= ANSWER_QUOTA as u32 {
                        break;
                    }
                    let rec = info.get(&id).expect("has проверен").clone();
                    let bytes_result = signed_payload_bytes(
                        key,
                        MsgType::InfoBody,
                        Payload::InfoBody {
                            info_id: rec.info_id,
                            kind: rec.kind,
                            ts: rec.ts,
                            author: rec.author,
                            sig: rec.sig.clone(),
                            value: rec.value.clone(),
                        },
                    );
                    if let Ok(bytes) = bytes_result {
                        t.send(CH_INFO, me_id, None, &bytes); // v0 broadcast
                        n += 1;
                    }
                }
                rep.answered += n;
            }
            Payload::InfoBody { info_id, kind, ts, author, sig, value } => {
                let rec = InfoRec {
                    info_id,
                    kind,
                    ts,
                    author,
                    sig,
                    value,
                    seals: Vec::new(),
                };
                let fresh = info.put(rec);
                if fresh {
                    rep.taken += 1;
                }
                // ПРИЗНАНИЕ: чужую инструкцию печатаем своей подписью
                if let Some(r) = info.get(&info_id).cloned() {
                    seal_foreign_instruction(
                        key, me_pub, me_id, info, state, t, &mut rep, &r,
                    );
                }
            }
            Payload::InfoSeal { info_id, sealer, seal } => {
                // печать ПРОВЕРЯЕТСЯ внутри store (canon-подпись, дедуп)
                let mut s = [0u8; 64];
                if seal.len() == 64 {
                    s.copy_from_slice(&seal);
                    if info.seal(&info_id, &sealer, &s) {
                        rep.seals_in += 1;
                    }
                }
            }
            _ => {}
        }
    }

    // 2) push СВОИХ свежих (квота за тик)
    //   NB: перезаписи не бывает — id = хэш канона; повторы отсюда мимо
    let mine: Vec<InfoRec> = info
        .all()
        .filter(|r| r.author == *me_pub && !state.shared.contains(&r.info_id))
        .take(PUSH_QUOTA)
        .cloned()
        .collect();
    for r in mine {
        let out = signed_payload_bytes(
            key,
            MsgType::InfoAnnounce,
            Payload::InfoAnnounce {
                info_id: r.info_id,
                kind: r.kind,
                author: r.author,
                value_len: r.value.len() as u32,
            },
        );
        if let Ok(bytes) = out {
            t.send(CH_INFO, me_id, None, &bytes);
            state.shared.insert(r.info_id);
            rep.pushed += 1;
        }
    }

    // ретрансляция ЧУЖИХ записей: у инструкций своя квота (правила роя
    // важнее), у фактов/опыта — своя (разреженная сеть: сосед доносит
    // то, что держит). Содержимое content-addressed и проверяемо:
    // всякий узел решает сам, держать ли чужое тело (кепки вида
    // защищают память от «мусорного» переполнения).
    let mut relayed_n = 0usize; // чужие инструкции: до RELAY_QUOTA
    let mut relayed_any = 0usize; // прочие чужие: до RELAY_MISC_QUOTA
    for r in info.all() {
        if r.author == *me_pub || state.shared.contains(&r.info_id) {
            continue;
        }
        let is_ins = r.kind == KIND_INSTRUCTION;
        let n = if is_ins { relayed_n } else { relayed_any };
        let quota = if is_ins { RELAY_QUOTA } else { RELAY_MISC_QUOTA };
        if n >= quota {
            continue;
        }
        let out = signed_payload_bytes(
            key,
            MsgType::InfoAnnounce,
            Payload::InfoAnnounce {
                info_id: r.info_id,
                kind: r.kind,
                author: r.author,
                value_len: r.value.len() as u32,
            },
        );
        if let Ok(bytes) = out {
            t.send(CH_INFO, me_id, None, &bytes);
            state.shared.insert(r.info_id);
            if is_ins {
                rep.relayed += 1;
                relayed_n += 1;
            } else {
                rep.relayed_other += 1;
                relayed_any += 1;
            }
        }
    }

    // 3) ask отсутствующих чужих id (квота за тик)
    new_ids.truncate(ASK_QUOTA);
    for id in &new_ids {
        state.asked.insert(*id);
    }
    if state.asked.len() > MEM_CAP {
        state.asked.clear(); // рубеж: переспросим, сеть переживёт
    }
    if !new_ids.is_empty() {
        let out = signed_payload_bytes(key, MsgType::InfoAsk, Payload::InfoAsk { ids: new_ids });
        if let Ok(bytes) = out {
            t.send(CH_INFO, me_id, None, &bytes);
            rep.asked_out += 1;
        }
    }

    if state.shared.len() > MEM_CAP {
        state.shared.clear(); // рубеж: пере-анонсы — но сеть вежлива
    }
    rep
}

/// Свою запись в инфосферу из диалога/поиска пишет craft_entry;
/// здесь держим удобный мост для webchat-фонов (kind известен).
pub fn note(kind: u8, ts: u32, value: &[u8], key: &dyn Signer) -> InfoRec {
    craft_entry(key, kind, ts, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_crypto::Ed25519Signer;
    use dinet_node::infosphere::{KIND_EXPERIENCE, KIND_FACT};


    /// Общая шина «все ко всем»: кадр A попадает только к B и наоборот
    /// (инбоксы раздельные, как у мессенджера).
    struct BusBox {
        a_in: Vec<Vec<u8>>,
        b_in: Vec<Vec<u8>>,
    }
    impl BusBox {
        fn new() -> BusBox {
            BusBox { a_in: Vec::new(), b_in: Vec::new() }
        }
    }
    impl dinet_node::transport::Transport for BusBox {
        fn send(&mut self, _ch: u8, from: NodeId, _to: Option<NodeId>, frame: &[u8]) {
            if from != 17005 {
                self.a_in.push(frame.to_vec());
            }
            if from != 42000 {
                self.b_in.push(frame.to_vec());
            }
        }
        fn collect(&mut self, _ch: u8, me: NodeId) -> Vec<Vec<u8>> {
            let src = if me == 17005 { &mut self.a_in } else { &mut self.b_in };
            std::mem::take(src)
        }
    }

    /// Разреженный рой A(17005)↔B(42000)↔C(42002): A и C НЕ связаны
    /// (рамки «A→C» и «C→A» теряются). Факт автора A обязан дойти до C
    /// через РЕЛЕЙ чужой записи B-узлом (и тело B отвечает как держатель).
    struct BusStar {
        a_in: Vec<Vec<u8>>,
        b_in: Vec<Vec<u8>>,
        c_in: Vec<Vec<u8>>,
    }
    impl BusStar {
        fn new() -> BusStar {
            BusStar { a_in: Vec::new(), b_in: Vec::new(), c_in: Vec::new() }
        }
    }
    impl dinet_node::transport::Transport for BusStar {
        fn send(&mut self, _ch: u8, from: NodeId, _to: Option<NodeId>, frame: &[u8]) {
            if from != 17005 {
                self.a_in.push(frame.to_vec()); // A слушает всех, кроме себя
            }
            // B слышит A и C
            if from != 42000 {
                self.b_in.push(frame.to_vec());
            }
            // C слышит ТОЛЬКО B (не A!)
            if from == 42000 {
                self.c_in.push(frame.to_vec());
            }
        }
        fn collect(&mut self, _ch: u8, me: NodeId) -> Vec<Vec<u8>> {
            let src = match me {
                17005 => &mut self.a_in,
                42000 => &mut self.b_in,
                _ => &mut self.c_in,
            };
            std::mem::take(src)
        }
    }

    #[test]
    fn bus_relay_reaches_far_node_in_sparse_swarm() {
        let a_key = Ed25519Signer::from_seed(&[41u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[42u8; 32]);
        let c_key = Ed25519Signer::from_seed(&[43u8; 32]);
        let mut bus = BusStar::new();
        let mut ia = InfoStore::new();
        let mut ib = InfoStore::new();
        let mut ic = InfoStore::new();
        let fact = craft_entry(&a_key, KIND_FACT, 400, b"web fact by distant A");
        ia.put(fact.clone());
        let exp = craft_entry(&b_key, KIND_EXPERIENCE, 401, b"dialog experience B");
        ib.put(exp.clone());
        let mut sa = InfoExchangeState::new();
        let mut sb = InfoExchangeState::new();
        let mut sc = InfoExchangeState::new();

        let mut relayed_other_b = 0u32;
        for _ in 0..10 {
            let _ra = info_exchange_tick(&a_key, 17005, &a_key.pubkey(), &mut ia, &mut sa, &mut bus);
            let rb = info_exchange_tick(&b_key, 42000, &b_key.pubkey(), &mut ib, &mut sb, &mut bus);
            let rc = info_exchange_tick(&c_key, 42002, &c_key.pubkey(), &mut ic, &mut sc, &mut bus);
            relayed_other_b += rb.relayed_other;
            let _ = rc;
        }

        // РЕЛЕЙ сработал: C (нет связи с A) держит факт автора A
        assert!(ic.has(&fact.info_id), "релей чужой записи не доехал до C");
        assert!(ib.has(&fact.info_id));
        // и опыт B доехал до A (прямой push, сеть двусторонняя)
        assert!(ia.has(&exp.info_id));
        // релей чужих фактов/опыта учтён в отчёте B
        assert!(relayed_other_b >= 1, "B не релеил чужое: {relayed_other_b}");
    }

    /// Полная дисциплина обмена двоих клиентов: инфосферы сходятся к
    /// полным копиям, каждый несёт СВОЮ запись ровно один раз, чужая
    /// инструкция получает КВОРУМНУЮ печать (quorum 2/2 у обоих).
    #[test]
    fn bus_exchange_converges_and_dedups() {
        let a_key = Ed25519Signer::from_seed(&[11u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[22u8; 32]);
        let mut bus = BusBox::new();
        let mut ia = InfoStore::new();
        let mut ib = InfoStore::new();
        let exp = craft_entry(&a_key, KIND_EXPERIENCE, 100, b"dialog experience A");
        let fact = craft_entry(&b_key, KIND_FACT, 200, b"web fact B");
        let ins = craft_entry(&a_key, KIND_INSTRUCTION, 300, b"royal rule: no fake facts");
        ia.put(exp.clone());
        ib.put(fact.clone());
        ia.put(ins.clone());
        let mut sa = InfoExchangeState::new();
        let mut sb = InfoExchangeState::new();

        // регулярный обмен: 8 тиков попеременно (push → ask → тела → печати)
        let mut pushes_a = 0u32;
        let mut pushes_b = 0u32;
        let mut seals_out_b = 0u32;
        for _ in 0..8 {
            let ra = info_exchange_tick(&a_key, 17005, &a_key.pubkey(), &mut ia, &mut sa, &mut bus);
            let rb = info_exchange_tick(&b_key, 42000, &b_key.pubkey(), &mut ib, &mut sb, &mut bus);
            eprintln!("A: {ra:?}");
            eprintln!("B: {rb:?}");
            pushes_a += ra.pushed;
            pushes_b += rb.pushed;
            seals_out_b += rb.seals_out;
        }
        assert_eq!(pushes_a, 2, "своё A объявило ровно по разу: {pushes_a}");
        assert_eq!(pushes_b, 1, "своё B объявило ровно раз: {pushes_b}");

        // сходимость: у каждой стороны ВСЕ три записи
        assert_eq!(ia.len(), 3, "у A три записи");
        assert_eq!(ib.len(), 3, "у B три записи");
        assert!(ia.has(&fact.info_id) && ib.has(&exp.info_id) && ib.has(&ins.info_id));

        // КВОРУМНАЯ печать инструкции: B печатал чужую инструкцию,
        // печать вернулась к A → кворум 2/2 у ОБОИХ
        assert_eq!(seals_out_b, 1, "B единожды признал инструкцию: {seals_out_b}");
        assert_eq!(ib.quorum_count(&ins.info_id), 2);
        assert!(ib.quorum_of(&ins.info_id, 2));
        assert_eq!(ia.quorum_count(&ins.info_id), 2, "печать не долетела до A");
        assert!(ia.quorum_of(&ins.info_id, 2));

        // повторные тики ничего не добавляют (id = хэш канона)
        for _ in 0..3 {
            let _ = info_exchange_tick(&a_key, 17005, &a_key.pubkey(), &mut ia, &mut sa, &mut bus);
            let rb = info_exchange_tick(&b_key, 42000, &b_key.pubkey(), &mut ib, &mut sb, &mut bus);
            assert_eq!(rb.taken, 0, "дубликаты отвергнуты");
            assert_eq!(ib.len(), 3);
        }
    }

    /// Запись уже была у B с прошлого запуска (БЕЗ со-подписи): правило
    /// роя всё равно признаётся при получении анонса, печать ходит к автору.
    #[test]
    fn bus_seals_preexisting_instruction() {
        let a_key = Ed25519Signer::from_seed(&[31u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[32u8; 32]);
        let mut bus = BusBox::new();
        let mut ia = InfoStore::new();
        let mut ib = InfoStore::new();
        let ins = craft_entry(&a_key, KIND_INSTRUCTION, 900, b"royal rule: cite sources");
        ia.put(ins.clone());
        ib.put(ins.clone()); // у B она есть, но непечатана
        assert_eq!(ib.quorum_count(&ins.info_id), 1);
        let mut sa = InfoExchangeState::new();
        let mut sb = InfoExchangeState::new();

        for _ in 0..4 {
            let _ = info_exchange_tick(&a_key, 17005, &a_key.pubkey(), &mut ia, &mut sa, &mut bus);
            let _ = info_exchange_tick(&b_key, 42000, &b_key.pubkey(), &mut ib, &mut sb, &mut bus);
        }
        assert_eq!(ib.quorum_count(&ins.info_id), 2, "B признал уже-имеющуюся");
        assert_eq!(ia.quorum_count(&ins.info_id), 2, "печать долетела к автору");
    }

    /// Подделка кадра отвергается: мутация одного байта ломает подпись.
    #[test]
    fn tofu_rejects_forged_frame() {
        let frame = signed_payload_bytes(
            &Ed25519Signer::from_seed(&[11u8; 32]),
            MsgType::InfoAnnounce,
            Payload::InfoAnnounce {
                info_id: [9u8; 32],
                kind: 2,
                author: [9u8; 32],
                value_len: 4,
            },
        )
        .unwrap();
        assert!(frame_from_wire(&frame).is_some());
        let mut tampered = frame.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        assert!(frame_from_wire(&tampered).is_none(), "мутация канона отброшена");
    }
}
