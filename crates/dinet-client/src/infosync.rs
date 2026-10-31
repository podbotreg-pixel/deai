//! infosync — обмен ИНФОСФЕРОЙ между реальными клиентами (требование
//! «общая память роя»: узлы регулярно обмениваются содержимым).
//!
//! Поверх ЛЮБОГО Transport (мост libp2p с каналом CH_INFO, файл-спул,
//! mesh) крутится одна фаза-тик:
//!   1. collect чужого → анонсы / аски / тела;
//!   2. push СВОИХ ещё не анонсированных записей (квота за тик —
//!      вежливость, не заливаем сеть новостями целиком);
//!   3. чужие аски → InfoBody своих записей (мы держатели);
//!   4. чужие тела → InfoStore.put (put сам проверяет id/подпись
//!      автора — подделка не сохранится);
//!   5. чужие анонсы → InfoAsk отсутствующих id.
//! Сквозная верификация TOFU: в кадре sender = pubkey автора кадра,
//! подпись поверх canon-байт — чекается у всех одинаково.
//! За несколько тиков сходится к полным копиям (v0-репликация).

use dinet_crypto::{Ed25519Verifier, Signer, Verifier};
use dinet_node::infosphere::{craft_entry, InfoRec, InfoStore};
use dinet_node::swarm::NodeId;
use dinet_node::transport::{Transport, CH_INFO};
use dinet_proto::{Frame, MsgType, Payload};
use std::collections::HashSet;

/// Максимум НАШИХ анонсов за один тик (вежливая квота).
pub const PUSH_QUOTA: usize = 8;
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
}

impl InfoExchangeState {
    pub fn new() -> InfoExchangeState {
        InfoExchangeState { shared: HashSet::new(), asked: HashSet::new() }
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
    //    сеть до поgoпления входа: чужим тоже надо успеть поcollect'ить)
    for raw in t.collect(CH_INFO, me_id) {
        let Some(frame) = frame_from_wire(&raw) else { continue };
        if frame.sender == *me_pub {
            continue; // свой loopback
        }
        match frame.payload {
            Payload::InfoAnnounce { info_id, .. } => {
                if !info.has(&info_id) && !state.asked.contains(&info_id) {
                    new_ids.push(info_id);
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
                let rec = InfoRec { info_id, kind, ts, author, sig, value };
                if info.put(rec) {
                    rep.taken += 1;
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
    /// (инбоксы раздельные, как у мессa трансорта).
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

    /// Полная дисциплина обмена двоих клиентов: инфосферы сходятся к
    /// полным копиям, каждый несёт СВОЮ запись ровно один раз, повторные
    /// тики ничего не множат (дедуп по id).
    #[test]
    fn bus_exchange_converges_and_dedups() {
        let a_key = Ed25519Signer::from_seed(&[11u8; 32]);
        let b_key = Ed25519Signer::from_seed(&[22u8; 32]);
        let mut bus = BusBox::new();
        let mut ia = InfoStore::new();
        let mut ib = InfoStore::new();
        let exp = craft_entry(&a_key, KIND_EXPERIENCE, 100, b"dialog experience A");
        let fact = craft_entry(&b_key, KIND_FACT, 200, b"web fact B");
        ia.put(exp.clone());
        ib.put(fact.clone());
        let mut sa = InfoExchangeState::new();
        let mut sb = InfoExchangeState::new();

        // регулярный обмен: 6 тиков попеременно (push → ask → тела)
        let mut pushes_a = 0u32;
        let mut pushes_b = 0u32;
        for _ in 0..6 {
            let ra = info_exchange_tick(&a_key, 17005, &a_key.pubkey(), &mut ia, &mut sa, &mut bus);
            let rb = info_exchange_tick(&b_key, 42000, &b_key.pubkey(), &mut ib, &mut sb, &mut bus);
            pushes_a += ra.pushed;
            pushes_b += rb.pushed;
        }
        assert_eq!(pushes_a, 1, "своё A объявило ровно раз: {pushes_a}");
        assert_eq!(pushes_b, 1, "своё B объявило ровно раз: {pushes_b}");
        assert_eq!(ia.len(), 2, "у A обе записи");
        assert_eq!(ib.len(), 2, "у B обе записи");
        assert!(ia.has(&fact.info_id) && ib.has(&exp.info_id));

        // дальнейшие тики ничего не добавляют (id = хэш канона)
        for _ in 0..3 {
            let _ = info_exchange_tick(&a_key, 17005, &a_key.pubkey(), &mut ia, &mut sa, &mut bus);
            let rb = info_exchange_tick(&b_key, 42000, &b_key.pubkey(), &mut ib, &mut sb, &mut bus);
            assert_eq!(rb.taken, 0, "дубликаты отвергнуты");
            assert_eq!(ib.len(), 2);
        }
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
