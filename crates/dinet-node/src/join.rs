//! join — ФАЗА J: Hello/PEX для новичка (TOFU поверх v1-ключей).
//!
//! v1-идентичность: клиент ГЕНЕРИРУЕТ ключ сам и впервые знакомит сеть
//! со своим pubkey. Протокольный ход:
//! 1. новичок объявляет сеть кадром `Hello 0x01` (proto_ver/caps/port),
//!    подписанным своим свежим ключом; поле `sender` — сам pubkey
//!    (32 байта): встречный строит верификатор прямо из него и
//!    проверяет подпись — TOFU (trust on first use);
//! 2. каждый встречный заносит новичка в СВОЮ книгу (`PeerBook::enroll`)
//!    и целит ему `HelloAck 0x02` со списком известных ключей (PEX);
//! 3. новичок собирает Ack'и: их senders тоже TOFU-заносятся в его
//!    книгу; дальше обе стороны умеют сверять подписи друг друга в
//!    любых роевых раундах.
//!
//! TOFU-нюанс (v1): первый контакт мог подсмотреть MITM — в живом
//! протоколе это смягчается пинами через DHT/анкоры; сим проверяет
//! механику (подпись/занесение/PEX), а не транспортную безопасность.

use crate::swarm::{Node, NodeId, PeerBook};
use crate::transport::{CH_HELLO, Transport};
use dinet_proto::{Frame, MsgType, Payload};
use dinet_crypto::Verifier;

/// Отчёт фазы для новичка.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinReport {
    /// сколько встречных узлов признали новичка (валидные HelloAck)
    pub recognized: usize,
    /// сколько ключей теперь в книге новичка (ТОЛЬКО TOFU-проверенные)
    pub peers_known: usize,
}

/// TOFU одного сырого кадра: подпись поверх тела кадра ключом ИЗ `sender`
/// → занесение в книгу. `want` — ожидаемый тип (чужие типы бракуются).
/// Возврат true — занесён (это новый знакомый).
fn tofu_learn(
    node: &mut Node,
    book: &mut PeerBook,
    raw: &[u8],
    want: MsgType,
    nid: Option<NodeId>,
) -> bool {
    let f = match tofu_check(node, raw) {
        Some(f) => f,
        None => return false,
    };
    if f.msg_type != want {
        node.rejected_frames += 1;
        return false;
    }
    let fresh = !book.knows(&f.sender);
    book.enroll(f.sender, nid);
    fresh
}

/// Крипто-уровень TOFU: подпись поверх тела кадра ключом ИЗ `sender`.
/// Отказ (брак/подмена) учитывается в rejected_frames узла.
fn tofu_check(node: &mut Node, raw: &[u8]) -> Option<Frame> {
    let f = match Frame::decode(raw) {
        Ok(f) => f,
        Err(_) => {
            node.rejected_frames += 1;
            return None;
        }
    };
    let body = match f.encode_payload() {
        Ok(b) => b,
        Err(_) => {
            node.rejected_frames += 1;
            return None;
        }
    };
    let v = match dinet_crypto::Ed25519Verifier::new(f.sender) {
        Some(v) => v,
        None => {
            node.rejected_frames += 1;
            return None;
        }
    };
    if !v.verify(&body, &f.sig) {
        node.rejected_frames += 1; // подпись не тем ключом: TOFU не впускает
        return None;
    }
    Some(f)
}

/// ФАЗА J: новичок объявляет Hello; встречные TOFU-заносят его в свои
/// книги и каждый отвечает HelloAck (PEX). Новый: Ack'и → его книга.
#[allow(clippy::too_many_arguments)]
pub fn hello_join(
    peers: &mut [Node],
    books: &mut [PeerBook],
    newcomer: &mut Node,
    nbook: &mut PeerBook,
    transport: &mut dyn Transport,
) -> JoinReport {
    // роли v1-клиента: только учитель (training-вкладка)
    let caps: [bool; 4] = [true, false, false, false];

    // 1) Hello новичка (подпись своего свежего ключа; broadcast)
    let hello = Frame {
        msg_type: MsgType::Hello,
        sender: newcomer.signer.pubkey(),
        sig: [0u8; 64],
        payload: Payload::Hello {
            proto_ver: dinet_proto::PROTO_VERSION as u16,
            caps,
            port: 0, // sim: порт слушания задаёт сетевой слой
        },
    };
    let sender = hello.sender;
    let body = hello.encode_payload().expect("hello payload");
    let sig = newcomer.signer.sign(&body);
    let wire = hello.encode_with_sender_sig(&sender, &sig).expect("hello кадр");
    transport.send(CH_HELLO, newcomer.id, None, &wire);
    let my_key = sender;

    // 2) встречные: TOFU → enroll → HelloAck с дидами своей книги
    for idx in 0..peers.len() {
        let mut ack_dids = books[idx].senders();
        for raw in transport.collect(CH_HELLO, peers[idx].id) {
            let _ = tofu_learn(&mut peers[idx], &mut books[idx], &raw, MsgType::Hello, Some(newcomer.id));
        }
        if !ack_dids.contains(&my_key) && books[idx].knows(&my_key) {
            ack_dids.push(my_key);
        }
        // HelloAck: диды книги (сеть) + сам новичок
        let ack = Frame {
            msg_type: MsgType::HelloAck,
            sender: peers[idx].signer.pubkey(),
            sig: [0u8; 64],
            payload: Payload::HelloAck {
                proto_ver: dinet_proto::PROTO_VERSION as u16,
                caps,
                port: 0,
                peer_ids: ack_dids,
            },
        };
        let ps = ack.sender;
        let ab = ack.encode_payload().expect("ack payload");
        let ack_sig = peers[idx].signer.sign(&ab);
        let aw = ack.encode_with_sender_sig(&ps, &ack_sig).expect("ack кадр");
        transport.send(CH_HELLO, peers[idx].id, Some(newcomer.id), &aw);
    }

    // 3) новичок: Ack'и → TOFU senders в свою книгу (строго полезное)
    let mut recognized = 0usize;
    for raw in transport.collect(CH_HELLO, newcomer.id) {
        let Ok(f) = Frame::decode(&raw) else {
            newcomer.rejected_frames += 1;
            continue;
        };
        if f.sender == my_key {
            continue; // loopback своего Hello (gossip-семантика)
        }
        let Payload::HelloAck { .. } = &f.payload else {
            newcomer.rejected_frames += 1;
            continue;
        };
        if tofu_learn(newcomer, nbook, &raw, MsgType::HelloAck, None) {
            recognized += 1;
        }
    }
    JoinReport {
        recognized,
        peers_known: nbook.len(),
    }
}

// ---------------------------------------------------------------------------
// Консенсус состава (v0-M2): взаимная Hello-волна + самосертификация id.
// Состав роя — отсортированный набор объявленных NodeId. Заявка
// самосертифицируется: кадр подписан собственным ключом, и pk(seed[id])
// == sender — чужой id под своим ключом не пройдёт (id выводится из
// ключей sim-вывода; v1 — настоящие ключи клиентов: заявка обязана
// совпадать с владельцем ключа, правила сверки те же).
// ---------------------------------------------------------------------------

/// PK, который sim-узел `id` получит из seed `[id;32]` (тот же вывод, что
/// в `Node::new`; самосертификация знает только эту схему).
pub fn pk_of_nid(id: NodeId) -> [u8; 32] {
    dinet_crypto::Ed25519Signer::from_seed(&[id as u8; 32]).pubkey()
}

/// Заявка состава: подписанный Hello со СВОИМ id в поле port (broadcast).
/// Повторы безвредны — слушатели заносят пир один раз.
pub fn roster_announce(node: &mut Node, transport: &mut dyn Transport) {
    let caps: [bool; 4] = [true, false, false, false];
    let hello = Frame {
        msg_type: MsgType::Hello,
        sender: node.signer.pubkey(),
        sig: [0u8; 64],
        payload: Payload::Hello {
            proto_ver: dinet_proto::PROTO_VERSION as u16,
            caps,
            port: node.id,
        },
    };
    let sender = hello.sender;
    let body = hello.encode_payload().expect("hello payload");
    let sig = node.signer.sign(&body);
    let wire = hello.encode_with_sender_sig(&sender, &sig).expect("hello кадр");
    transport.send(CH_HELLO, node.id, None, &wire);
}

/// Обработать вход волны: TOFU-подпись + самосертификация заявленного id.
/// Возврат — сколько НОВЫХ пиров занесено (для тихого окна драйвера).
pub fn roster_listen(
    node: &mut Node,
    book: &mut PeerBook,
    transport: &mut dyn Transport,
) -> usize {
    let me = node.signer.pubkey();
    let mut fresh = 0usize;
    for raw in transport.collect(CH_HELLO, node.id) {
        let Ok(f) = Frame::decode(&raw) else {
            node.rejected_frames += 1;
            continue;
        };
        if f.sender == me {
            continue; // свой loopback: id себя знаю и без волны
        }
        let body = match f.encode_payload() {
            Ok(b) => b,
            Err(_) => {
                node.rejected_frames += 1;
                continue;
            }
        };
        let Payload::Hello { port, .. } = &f.payload else {
            node.rejected_frames += 1;
            continue;
        };
        let ver = match dinet_crypto::Ed25519Verifier::new(f.sender) {
            Some(v) => v,
            None => {
                node.rejected_frames += 1;
                continue;
            }
        };
        // подпись ТОЛЬКО ключом sender + самосертификация заявленного id
        if !ver.verify(&body, &f.sig) || pk_of_nid(*port) != f.sender {
            node.rejected_frames += 1; // чужой id под своим ключом: брак
            continue;
        }
        if !book.knows(&f.sender) {
            book.enroll(f.sender, Some(*port));
            fresh += 1;
        }
    }
    fresh
}

/// Текущий состав: мой id + все занесённые пиры (sorted, без дублей).
pub fn roster_now(me: NodeId, book: &PeerBook) -> Vec<NodeId> {
    let mut v: Vec<NodeId> = book.nid_by_sender.values().copied().collect();
    v.push(me);
    v.sort_unstable();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::Node as SwarmNode;
    use crate::transport::MeshTransport;
    use dinet_crypto::{random_seed, Ed25519Signer, Signer as _};

    /// Новичок со СВЕЖИМ сгенерированным ключом знакомится с сетью из
    /// seed-узлов: TOFU с двух сторон + PEX-список сети.
    #[test]
    fn newcomer_tofu_pex() {
        let mut peers: Vec<SwarmNode> =
            (0..3usize).map(|i| SwarmNode::new(i as NodeId, 7, false)).collect();
        let ids: Vec<NodeId> = peers.iter().map(|n| n.id).collect();
        let mut books: Vec<PeerBook> =
            (0..3).map(|_| crate::swarm::peer_book_ids(&ids)).collect();

        // v1-реалис: ключ СГЕНЕРИРОВАН, не выведен из id
        // v1-генерация: seed от ОС (не выводится из id)
        let rseed = random_seed().expect("ОС энтропия");
        let _ = rseed;
        let mut newcomer = SwarmNode::new(3, 7, false);
        newcomer.signer = Box::new(Ed25519Signer::generate().expect("генерация"));
        let mut nbook = PeerBook::empty();
        assert!(nbook.is_empty());

        let mut t = MeshTransport::new(4);
        let rep = hello_join(&mut peers, &mut books, &mut newcomer, &mut nbook, &mut t);
        assert_eq!(rep.recognized, 3, "не все признали новичка: {rep:?}");
        // каждый встречный занёс pubkey новичка (TOFU)
        for b in &books {
            assert!(b.knows(&newcomer.signer.pubkey()), "новичок не в книге");
        }
        // PEX: книга новичка узнала 3 ключа сети
        assert_eq!(rep.peers_known, 3, "PEX не дошёл: {rep:?}");

        // следующий кадр новичка верифицируется УЖЕ ЗНАЮЩЕЙ книгой
        // встречного (enroll настоящий, а не «строковый»)
        let frame = Frame {
            msg_type: MsgType::Hello,
            sender: newcomer.signer.pubkey(),
            sig: [0u8; 64],
            payload: Payload::Hello {
                proto_ver: dinet_proto::PROTO_VERSION as u16,
                caps: [true, false, false, false],
                port: 0,
            },
        };
        let body = frame.encode_payload().expect("payload");
        let sig = newcomer.signer.sign(&body);
        let wire = frame
            .encode_with_sender_sig(&newcomer.signer.pubkey(), &sig)
            .expect("кадр");
        let f = Frame::decode(&wire).expect("повторный кадр");
        assert!(books[0].knows(&f.sender), "книга не знает ключ");
        assert!(
            books[0].senders().iter().any(|k| *k == f.sender),
            "ключ не в списке книги"
        );
        // верификация через книгу:  подпись сходится ключом из книги
        let body2 = f.encode_payload().expect("тело");
        let vf = signer_for_book(&books[0], &f.sender).expect("верификатор в книге");
        assert!(vf.verify(&body2, &f.sig), "занесённый ключ не верифицирует");
    }

    /// ПОДДЕЛКА: кадр «от лица новичка», подписанный ЧУЖИМ ключом —
    /// TOFU отвергает, книга не пополняется, подделка учтена.
    #[test]
    fn tofu_rejects_wrong_signer() {
        let mut peers: Vec<SwarmNode> =
            (0..3usize).map(|i| SwarmNode::new(i as NodeId, 7, false)).collect();
        let ids: Vec<NodeId> = peers.iter().map(|n| n.id).collect();
        let mut books: Vec<PeerBook> =
            (0..3).map(|_| crate::swarm::peer_book_ids(&ids)).collect();

        let victim_key = Ed25519Signer::from_seed(&[31u8; 32]).pubkey();
        let bad = Ed25519Signer::from_seed(&[9u8; 32]);
        let frame = Frame {
            msg_type: MsgType::Hello,
            sender: victim_key, // говорит «я тот»
            sig: [0u8; 64],
            payload: Payload::Hello {
                proto_ver: dinet_proto::PROTO_VERSION as u16,
                caps: [true, false, false, false],
                port: 0,
            },
        };
        let body = frame.encode_payload().expect("payload");
        // подпись НЕ тем ключом, что в sender
        let sig = bad.sign(&body);
        let wire = frame.encode_with_sender_sig(&victim_key, &sig).expect("кадр");

        let nbook = PeerBook::empty();
        let mut t = MeshTransport::new(4);
        t.send(CH_HELLO, 3, Some(0), &wire);
        // узел 0 ждёт hello-кадр и получает ПОДДЕЛКУ
        let raw = transport_tap(&mut t, 0).remove(0);
        assert!(!tofu_learn(&mut peers[0], &mut books[0], &raw, MsgType::Hello, Some(3)));
        assert!(
            !books[0].knows(&victim_key),
            "подделка занесена в книгу"
        );
        assert!(peers[0].rejected_frames > 0, "подделка не замечена");
        assert!(nbook.is_empty());

        let _ = ids;
    }

    /// Верификатор конкретного ключа из книги (для проверки enroll'а).
    fn signer_for_book<'a>(
        b: &'a PeerBook,
        key: &[u8; 32],
    ) -> Option<&'a dyn dinet_crypto::Verifier> {
        if b.knows(key) {
            b.verifier(key)
        } else {
            None
        }
    }

    /// Достать первый hello-кадр из чужого inbox (sim-рука).
    fn transport_tap(t: &mut MeshTransport, who: NodeId) -> Vec<Vec<u8>> {
        use crate::transport::Transport as _;
        t.collect(CH_HELLO, who)
    }

    /// КОНСЕНСУС СОСТАВА v0: взаимные Hello-волны сходятся к ОДНОМУ
    /// отсортированному составу у всех узлов.
    #[test]
    fn roster_waves_converge_bit_identical() {
        let mut nodes: Vec<SwarmNode> =
            (0..5usize).map(|i| SwarmNode::new(i as NodeId, 7, false)).collect();
        let mut books: Vec<PeerBook> = (0..5).map(|_| PeerBook::empty()).collect();
        let mut mesh = MeshTransport::new(5);
        for _round in 0..3 {
            for n in &mut nodes {
                roster_announce(n, &mut mesh);
            }
            for i in 0..nodes.len() {
                roster_listen(&mut nodes[i], &mut books[i], &mut mesh);
            }
        }
        let want: Vec<NodeId> = (0..5).collect();
        for i in 0..5 {
            assert_eq!(
                roster_now(nodes[i].id, &books[i]),
                want,
                "состав узла {i} не сошёлся"
            );
        }
    }

    /// САМОСЕРТИФИКАЦИЯ: подписанный «чужой» id (у узла 9 подпись своя,
    /// а заявка port=3) бракуется; честная заявка порт=9 проходит.
    #[test]
    fn roster_rejects_forged_claim() {
        let mut listener = SwarmNode::new(0, 7, false);
        let mut book = PeerBook::empty();
        let mut mesh = MeshTransport::new(12);

        // подделка: валидная подпись ключа узла 9, но port = 3 (чужой id)
        let imposter = SwarmNode::new(9, 7, false);
        let forged = Frame {
            msg_type: MsgType::Hello,
            sender: imposter.signer.pubkey(),
            sig: [0u8; 64],
            payload: Payload::Hello {
                proto_ver: dinet_proto::PROTO_VERSION as u16,
                caps: [true, false, false, false],
                port: 3,
            },
        };
        let body = forged.encode_payload().expect("payload");
        let sig = imposter.signer.sign(&body);
        let wire = forged
            .encode_with_sender_sig(&imposter.signer.pubkey(), &sig)
            .expect("кадр");
        use crate::transport::Transport as _;
        mesh.send(CH_HELLO, 9, None, &wire);

        assert_eq!(roster_listen(&mut listener, &mut book, &mut mesh), 0);
        assert!(listener.rejected_frames > 0, "подделка не замечена");
        assert!(book.is_empty(), "чужой id просочился в книгу");

        // честная заявка (свой ключ под своим id) заносится
        let mut honest = SwarmNode::new(3, 7, false);
        roster_announce(&mut honest, &mut mesh);
        assert_eq!(roster_listen(&mut listener, &mut book, &mut mesh), 1);
        assert_eq!(roster_now(0, &book), vec![0, 3]);
    }
}
