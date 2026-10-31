//! infosphere — ФАЗА K: «инфосфера», общая распределённая память роя.
//!
//! Узлы роя хранят вместе память СЕТИ (не у одного владельца): единицы
//! опыта (диалоги, извлечённые факты), общие знания и ИНСТРУКЦИИ по
//! обучению и общению участника. Запись content-addressed:
//! canon = [kind][ts][author][value], id = sha256(canon), подпись
//! автора поверх canon — подделать id под чужое тело нельзя
//! (SHA-256 пре-имидж), подпись чужим ключом не сойдётся.
//!
//! v0 репликация — ПОЛНАЯ: каждый узел держит копию всех записей
//! (никакого «одного места» нет); v1 план — выборочные держатели
//! (hash-шардинг) + кворумная печать инструкций через registry.
//!
//! Протокол одного раунда синхронизации (строковые барьеры как в ФАЗЕ
//! H — сначала ВСЕ collect, потом ВСЕ ответы, иначе collect отстающего
//! съедает чужие кадры):
//! 1. каждый узел broadcast InfoAnnounce по всем своим записям;
//! 2. все collect → у каждого ровно его отсутствующие id → InfoAsk
//!    источнику;
//! 3. все collect → буфер аськов; затем все разом отвечают InfoBody;
//! 4. все collect → тело проверяется (id == sha256(canon), подпись
//!    автора) и только тогда ставится в InfoStore.

use crate::swarm::{decode_frames, peer_book_ids, sign_frame, Node, NodeId};
use crate::transport::{CH_INFO, Transport};
use dinet_crypto::Verifier;
use dinet_proto::{MsgType, Payload};

/// Типы записей инфосферы (байт в канон).
pub const KIND_EXPERIENCE: u8 = 1; // единица опыта (диалог/факт из общения)
pub const KIND_FACT: u8 = 2; // общее знание (извлечённое, проверяемое)
pub const KIND_INSTRUCTION: u8 = 3; // инструкция роя: как учиться/общаться

/// Канон записи: [kind][ts LE][author 32][value].
pub fn canon(kind: u8, ts: u32, author: &[u8; 32], value: &[u8]) -> Vec<u8> {
    let mut c = Vec::with_capacity(1 + 4 + 32 + value.len());
    c.push(kind);
    c.extend_from_slice(&ts.to_le_bytes());
    c.extend_from_slice(author);
    c.extend_from_slice(value);
    c
}

/// Сетевой id записи.
pub fn info_id(kind: u8, ts: u32, author: &[u8; 32], value: &[u8]) -> [u8; 32] {
    dinet_crypto::sha256(&canon(kind, ts, author, value))
}

/// Одна запись инфосферы.
#[derive(Debug, Clone, PartialEq)]
pub struct InfoRec {
    pub info_id: [u8; 32],
    pub kind: u8,
    pub ts: u32,
    pub author: [u8; 32],
    pub sig: Vec<u8>,
    pub value: Vec<u8>,
    /// КВОРУМНАЯ ПЕЧАТЬ: со-подписи ДРУГИХ узлов поверх того же canon
    /// (инструкции роя получают признание «не только автор сказал»).
    pub seals: Vec<InfoSeal>,
}

/// Со-подпись записи: sig = подпись УЗЛА sealer поверх canon записи.
#[derive(Debug, Clone, PartialEq)]
pub struct InfoSeal {
    pub sealer: [u8; 32],
    pub sig: [u8; 64],
}

/// Хранилище узла: content-addressed, append-only по сути своей; файл
/// (опция) — лог [len][rec] коротких записей, дружит с переоткрытием.
pub struct InfoStore {
    path: Option<std::path::PathBuf>,
    recs: std::collections::BTreeMap<[u8; 32], InfoRec>,
}

impl InfoStore {
    pub fn new() -> InfoStore {
        InfoStore { path: None, recs: std::collections::BTreeMap::new() }
    }

    /// Хранилище с файлом (существующий лог подхватывается).
    pub fn open(path: std::path::PathBuf) -> InfoStore {
        let mut s = InfoStore { path: Some(path.clone()), recs: std::collections::BTreeMap::new() };
        if let Ok(raw) = std::fs::read(&path) {
            let mut pos = 0usize;
            while pos < raw.len() {
                if raw.len() - pos < 4 {
                    break; // оборван хвост — игнорируем мусор
                }
                let n = u32::from_le_bytes(raw[pos..pos + 4].try_into().expect("4б")) as usize;
                pos += 4;
                if n > 4 + 1 + 4 + 32 + 64 + 16 * 1024 + 8 {
                    break; // распухло — не наш формат
                }
                if raw.len() - pos < n {
                    break;
                }
                let body = &raw[pos..pos + n];
                pos += n;
                // тело: kind(1) ts(4) author(32) sig_len(4) sig value
                let kind = body[0];
                let ts = u32::from_le_bytes(body[1..5].try_into().expect("4б"));
                let mut author = [0u8; 32];
                author.copy_from_slice(&body[5..37]);
                let sig_len =
                    u32::from_le_bytes(body[37..41].try_into().expect("4б")) as usize;
                if sig_len != 64 || body.len() < 41 + sig_len {
                    continue;
                }
                let sig = body[41..41 + 64].to_vec();
                let value = body[41 + 64..].to_vec();
                let id = info_id(kind, ts, &author, &value);
                s.recs.entry(id).or_insert(InfoRec {
                    info_id: id,
                    kind,
                    ts,
                    author,
                    sig,
                    value,
                    seals: Vec::new(),
                });
            }
        }
        // со-подписи: отдельный append-only лог [len][id 32][sealer 32][sig 64],
        // подгружаем к записям поверх основного лога
        if let Some(seals_path) = s.seals_path() {
            if let Ok(raw) = std::fs::read(&seals_path) {
                let mut pos = 0usize;
                while raw.len() - pos >= 4 {
                    let n = u32::from_le_bytes(raw[pos..pos + 4].try_into().expect("4б")) as usize;
                    pos += 4;
                    if n != 128 || raw.len() - pos < n {
                        break; // не наш формат / оборван хвост
                    }
                    let mut id = [0u8; 32];
                    let mut sealer = [0u8; 32];
                    let mut sig = [0u8; 64];
                    id.copy_from_slice(&raw[pos..pos + 32]);
                    sealer.copy_from_slice(&raw[pos + 32..pos + 64]);
                    sig.copy_from_slice(&raw[pos + 64..pos + 128]);
                    pos += n;
                    if let Some(rec) = s.recs.get_mut(&id) {
                        if !rec.seals.iter().any(|x| x.sealer == sealer) {
                            rec.seals.push(InfoSeal { sealer, sig });
                        }
                    }
                }
            }
        }
        s
    }

    /// Поставить запись (с проверкой подписи/id). Возврат true — новая.
    pub fn put(&mut self, rec: InfoRec) -> bool {
        if !check_rec(&rec) {
            return false;
        }
        let fresh = !self.recs.contains_key(&rec.info_id);
        if fresh {
            self.persist(&rec);
        }
        self.recs.insert(rec.info_id, rec);
        fresh
    }

    /// Есть ли запись.
    pub fn has(&self, id: &[u8; 32]) -> bool {
        self.recs.contains_key(id)
    }

    /// Запись по id.
    pub fn get(&self, id: &[u8; 32]) -> Option<&InfoRec> {
        self.recs.get(id)
    }

    /// Все записи.
    pub fn all(&self) -> impl Iterator<Item = &InfoRec> {
        self.recs.values()
    }

    /// Инструкции роя (kind = INSTRUCTION) — «как учиться/общаться».
    pub fn instructions(&self) -> Vec<&InfoRec> {
        self.recs
            .values()
            .filter(|r| r.kind == KIND_INSTRUCTION)
            .collect()
    }

    /// Сколько инструкций достигли кворума `need` подписей
    /// (обычный призыв: 2 = автор + один узел признания).
    pub fn quorum_rules(&self, need: usize) -> usize {
        self.recs
            .values()
            .filter(|r| r.kind == KIND_INSTRUCTION && self.quorum_of(&r.info_id, need))
            .count()
    }

    /// Число записей.
    pub fn len(&self) -> usize {
        self.recs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.recs.is_empty()
    }

    fn persist(&mut self, rec: &InfoRec) {
        let Some(p) = &self.path.clone() else { return };
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let mut body = Vec::with_capacity(1 + 4 + 32 + 4 + 64 + rec.value.len());
            body.push(rec.kind);
            body.extend_from_slice(&rec.ts.to_le_bytes());
            body.extend_from_slice(&rec.author);
            body.extend_from_slice(&64u32.to_le_bytes());
            body.extend_from_slice(&rec.sig);
            body.extend_from_slice(&rec.value);
            let mut out = Vec::with_capacity(4 + body.len());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&body);
            let _ = f.write_all(&out);
            let _ = f.flush();
        }
    }

    /// Кворумная печать: со-подпись записи другим узлом. sig проверяется
    /// ПОВЕРХ canon записи; повтор печати тем же узлом Iterable игнорится.
    /// Возврат true — свежая со-подпись добавлена (false — был/плохая).
    pub fn seal(&mut self, id: &[u8; 32], sealer: &[u8; 32], sig: &[u8; 64]) -> bool {
        let Some(rec) = self.recs.get_mut(id) else { return false };
        if rec.seals.iter().any(|s| &s.sealer == sealer) {
            return false; // этот узел уже печатал
        }
        let canon = canon(rec.kind, rec.ts, &rec.author, &rec.value);
        let ok = dinet_crypto::Ed25519Verifier::new(*sealer)
            .map(|v| v.verify(&canon, sig))
            .unwrap_or(false);
        if !ok {
            return false; // подпись печати не сходится с canon
        }
        rec.seals.push(InfoSeal { sealer: *sealer, sig: *sig });
        self.persist_seal(id, sealer, sig);
        true
    }

    /// Сколько узлов подпечатали запись (автор со своей подписью + seals).
    pub fn quorum_count(&self, id: &[u8; 32]) -> usize {
        self.recs.get(id).map(|r| 1 + r.seals.len()).unwrap_or(0)
    }

    /// Кворум достигнут? need = порог (клиент роя решает сам; v0 = 2).
    pub fn quorum_of(&self, id: &[u8; 32], need: usize) -> bool {
        self.quorum_count(id) >= need
    }

    /// Файл со-подписей: append-only лог [len][id 32][sealer 32][sig 64]
    /// — подгружается при open (seals НЕ внутри основной записи-лога).
    fn persist_seal(&self, id: &[u8; 32], sealer: &[u8; 32], sig: &[u8; 64]) {
        let Some(p) = &self.seals_path() else { return };
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let mut body = Vec::with_capacity(32 + 32 + 64);
            body.extend_from_slice(id);
            body.extend_from_slice(sealer);
            body.extend_from_slice(sig);
            let mut out = Vec::with_capacity(4 + body.len());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&body);
            let _ = f.write_all(&out);
            let _ = f.flush();
        }
    }

    fn seals_path(&self) -> Option<std::path::PathBuf> {
        self.path.as_ref().map(|p| {
            let mut s = p.clone().into_os_string();
            s.push(".seals");
            std::path::PathBuf::from(s)
        })
    }
}

impl Default for InfoStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Полная проверка записи перед хранением/пересылкой.
fn check_rec(rec: &InfoRec) -> bool {
    if rec.kind > KIND_INSTRUCTION {
        return false;
    }
    if info_id(rec.kind, rec.ts, &rec.author, &rec.value) != rec.info_id {
        return false; // id не чей canon
    }
    let mut s = [0u8; 64];
    if rec.sig.len() != 64 {
        return false;
    }
    s.copy_from_slice(&rec.sig);
    match dinet_crypto::Ed25519Verifier::new(rec.author) {
        Some(v) => v.verify(&canon(rec.kind, rec.ts, &rec.author, &rec.value), &s),
        None => false,
    }
}

/// Запись от автора (подпись его же ключом).
pub fn craft_entry(
    signer: &dyn dinet_crypto::Signer,
    kind: u8,
    ts: u32,
    value: &[u8],
) -> InfoRec {
    let author = signer.pubkey();
    let canon = canon(kind, ts, &author, value);
    let sig = signer.sign(&canon).to_vec();
    InfoRec {
        info_id: info_id(kind, ts, &author, value),
        kind,
        ts,
        author,
        sig,
        value: value.to_vec(),
        seals: Vec::new(),
    }
}

/// Отчёт sync'а по узлу.
#[derive(Debug, Clone, PartialEq)]
pub struct InfoSyncReport {
    /// записей было
    pub known: usize,
    /// сколько новых принял
    pub learned: usize,
}

/// ФАЗА K: один раунд синхронизации инфосферы репликами.
/// Возврат: по узлу — (known, learned).
pub fn infosphere_sync(
    mine: &mut [Node],
    all_ids: &[NodeId],
    stores: &mut [InfoStore],
    transport: &mut dyn Transport,
) -> Vec<InfoSyncReport> {
    let knowns: Vec<usize> = stores.iter().map(|s| s.len()).collect();

    // 1) анонсы всех записей
    for idx in 0..mine.len() {
        let ann: Vec<([u8; 32], u8, [u8; 32], u32)> = stores[idx]
            .all()
            .map(|r| (r.info_id, r.kind, r.author, r.value.len() as u32))
            .collect();
        for (id, kind, author, vlen) in ann {
            let frame =
                sign_frame(&mine[idx], MsgType::InfoAnnounce, Payload::InfoAnnounce {
                    info_id: id,
                    kind,
                    author,
                    value_len: vlen,
                });
            transport.send(CH_INFO, mine[idx].id, None, &frame);
        }
    }

    // 2) ВСЕ сначала collect анонсов (строгий барьер), свод не-cамих
    //    себя absent-id по источникам; затем ВСЕ разом шлют asks
    let mut ask_bufs: Vec<Vec<(NodeId, Vec<[u8; 32]>)>> = (0..mine.len()).map(|_| Vec::new()).collect();
    for idx in 0..mine.len() {
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_INFO, mine[idx].id);
        let frames = decode_frames(&mut mine[idx], &book, wire);
        let me = mine[idx].id;
        for (nid, pl) in frames {
            if nid == me {
                continue; // loopback анонса
            }
            if let Payload::InfoAnnounce { info_id, .. } = &pl {
                if !stores[idx].has(info_id) {
                    if let Some(slot) = ask_bufs[idx].iter_mut().find(|(s, _)| *s == nid) {
                        slot.1.push(*info_id);
                    } else {
                        ask_bufs[idx].push((nid, vec![*info_id]));
                    }
                }
            }
        }
    }
    for idx in 0..mine.len() {
        for (dest, ids) in &ask_bufs[idx] {
            let frame = sign_frame(
                &mine[idx],
                MsgType::InfoAsk,
                Payload::InfoAsk { ids: ids.clone() },
            );
            transport.send(CH_INFO, mine[idx].id, Some(*dest), &frame);
        }
    }

    // 3) все collect аськов → буфер; затем все разом шлют тела
    let mut body_bufs: Vec<Vec<(NodeId, Vec<[u8; 32]>)>> =
        (0..mine.len()).map(|_| Vec::new()).collect();
    for idx in 0..mine.len() {
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_INFO, mine[idx].id);
        let frames = decode_frames(&mut mine[idx], &book, wire);
        let me = mine[idx].id;
        for (nid, pl) in frames {
            if nid == me {
                continue;
            }
            if let Payload::InfoAsk { ids } = &pl {
                body_bufs[idx].push((nid, ids.clone()));
            }
        }
    }
    for idx in 0..mine.len() {
        for (dest, ids) in &body_bufs[idx] {
            for id in ids {
                let Some(rec) = stores[idx].get(id).cloned() else { continue };
                let frame = sign_frame(
                    &mine[idx],
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
                transport.send(CH_INFO, mine[idx].id, Some(*dest), &frame);
            }
        }
    }

    // 4) все collect тел → поставить (с проверкой)
    let mut out = (0..mine.len())
        .map(|idx| InfoSyncReport { known: knowns[idx], learned: 0 })
        .collect::<Vec<_>>();
    for idx in 0..mine.len() {
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_INFO, mine[idx].id);
        let frames = decode_frames(&mut mine[idx], &book, wire);
        for (_nid, pl) in frames {
            if let Payload::InfoBody { info_id, kind, ts, author, sig, value } = pl {
                let rec = InfoRec {
                    info_id,
                    kind,
                    ts,
                    author,
                    sig,
                    value,
                    seals: Vec::new(),
                };
                if stores[idx].put(rec) {
                    out[idx].learned += 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::Node as SwarmNode;
    use crate::transport::MeshTransport;
    use dinet_crypto::{Ed25519Signer, Signer as _};

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("dinet-info-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("tmp");
        d
    }

    /// Инфосфера разлетается: узел 0 пишет опыт, узел 1 — инструкцию;
    /// палубная синхронизация — у ВСЕХ полная копия; узел 2 после
    /// стирания памяти добирает всё у роя (реплики рассеяны по узлам).
    #[test]
    fn replicates_experience_and_instruction() {
        let mut nodes: Vec<SwarmNode> =
            (0..4usize).map(|i| SwarmNode::new(i as NodeId, 7, false)).collect();
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
        let dir = tmpdir("repl");
        let mut stores: Vec<InfoStore> = (0..4)
            .map(|i| InfoStore::open(dir.join(format!("i{i}.info"))))
            .collect();
        assert!(stores[0].is_empty(), "новый склад не пуст");

        // опыт общения узла 0 + инструкция роя от узла 1
        let exp = craft_entry(&*nodes[0].signer, KIND_EXPERIENCE, 1000, b"Q&A weather talk: dialogue experience");
        let ins = craft_entry(&*nodes[1].signer, KIND_INSTRUCTION, 1001, b"Learn politely: always cite sources");
        assert!(stores[0].put(exp.clone()));
        assert!(stores[1].put(ins.clone()));
        assert_eq!(stores[0].len(), 1);
        assert_eq!(stores[1].instructions().len(), 1);

        // sync №1: все разом обмениваются — полная репликация у каждого
        let mut t = MeshTransport::new(4);
        let rep = infosphere_sync(&mut nodes, &ids, &mut stores, &mut t);
        for s in &stores {
            assert_eq!(s.len(), 2, "у узла не полная копия: реп {rep:?}");
        }
        // авторы добрали по чужой записи (1), пустые — обе (2)
        assert_eq!(rep[0].learned, 1, "автор опыта: {rep:?}");
        assert_eq!(rep[1].learned, 1, "автор инструкции: {rep:?}");
        assert_eq!(rep[2].learned, 2, "пустой узел: {rep:?}");
        assert_eq!(rep[3].learned, 2, "пустой узел: {rep:?}");
        // записи бит-в-бит
        let et = stores[2].get(&exp.info_id).expect("опыт у узла 2");
        let it = stores[3].get(&ins.info_id).expect("инструкция у узла 3");
        assert_eq!(et, &exp);
        assert_eq!(it, &ins);

        // sync №2: нечего узнавать
        let rep = infosphere_sync(&mut nodes, &ids, &mut stores, &mut t);
        assert!(rep.iter().all(|r| r.learned == 0), "лишнее приняли: {rep:?}");

        // стирание памяти узла 2 (диск утрачен) — добирает всё у роя
        stores[2] = InfoStore::open(dir.join("i2-wipe.info"));
        let rep = infosphere_sync(&mut nodes, &ids, &mut stores, &mut t);
        assert_eq!(rep[2].learned, 2, "узел 2 не восстановился: {rep:?}");
        assert_eq!(stores[2].len(), 2);

        // файл хранилища пережил переоткрытие бит-в-бит
        let reopened = InfoStore::open(dir.join("i2.info"));
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.get(&ins.info_id).expect("инструкция").value, ins.value);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Со-подписи (кворумная печать): валидные добавляются 1 шт/узел,
    /// подделка и повтор отвергаются; seals-файл переживает переоткрытие.
    #[test]
    fn instructions_get_quorum_seals() {
        let dir = tmpdir("seals");
        let a = Ed25519Signer::from_seed(&[61u8; 32]);
        let b = Ed25519Signer::from_seed(&[62u8; 32]);
        let mut s = InfoStore::open(dir.join("i.info"));

        // запись ЧУЖОГО автора (мы собираемся её признать)
        let ins = craft_entry(&a, KIND_INSTRUCTION, 300, b"royal instruction: no fake facts");
        assert!(s.put(ins.clone()));
        assert_eq!(s.quorum_count(&ins.info_id), 1, "авторская подпись = 1");

        // другой узел печатает запись поверх того же canon
        let canon_b = canon(ins.kind, ins.ts, &ins.author, &ins.value);
        let seal_b = b.sign(&canon_b);
        assert!(s.seal(&ins.info_id, &b.pubkey(), &seal_b));
        assert_eq!(s.quorum_count(&ins.info_id), 2);
        assert!(s.quorum_of(&ins.info_id, 2), "кворум 2/2 достигнут");

        // повтор печати тем же узлом — не нова
        assert!(!s.seal(&ins.info_id, &b.pubkey(), &seal_b));
        assert_eq!(s.quorum_count(&ins.info_id), 2);

        // подделка: печать третьего узла ПОВЕРХ чужого canon — отвергнута
        let c = Ed25519Signer::from_seed(&[63u8; 32]);
        let fake = c.sign(&canon(ins.kind, 111, &ins.author, &ins.value)); // ts иной
        assert!(!s.seal(&ins.info_id, &c.pubkey(), &fake), "подделка прошла");
        assert_eq!(s.quorum_count(&ins.info_id), 2);

        // печать отсутствующей записи — false
        assert!(!s.seal(&[9u8; 32], &b.pubkey(), &seal_b));

        // seals переживают переоткрытие вместе с записью
        let reopened = InfoStore::open(dir.join("i.info"));
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.quorum_count(&ins.info_id), 2, "файл печати потерялся");
        assert!(reopened.quorum_of(&ins.info_id, 2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Подделки не хранятся: чужая подпись и подменённое id/тело.
    #[test]
    fn forged_entries_rejected() {
        let mut s = InfoStore::new();
        let a = Ed25519Signer::from_seed(&[41u8; 32]);
        let b = Ed25519Signer::from_seed(&[42u8; 32]);

        // вернейшая собственная запись проходит
        let ok = craft_entry(&a, KIND_FACT, 7, b"Eight planets: Mercury, Venus, Earth...");
        assert!(s.put(ok.clone()), "своя запись должна пройти");
        assert!(!s.put(ok.clone()), "повтор не нова");

        // подпись ЧУЖИМ ключом поверх чужого canon
        let mut bad = ok.clone();
        let c = canon(bad.kind, bad.ts, &bad.author, &bad.value);
        bad.sig = b.sign(&c).to_vec();
        assert!(!s.put(bad), "чужая подпись хранится");

        // подмена тела при том же id (или же id от другого тела)
        let mut tamper = ok.clone();
        tamper.value = b"tampered body text".to_vec();
        tamper.info_id = ok.info_id; // id старого канона, тело новое
        assert!(!s.put(tamper), "подменённое тело хранится");

        // чужая kind (0 или 9) отвергается
        let mut wrong_kind = craft_entry(&a, KIND_FACT, 8, b"...");
        wrong_kind.kind = 9;
        assert!(!s.put(wrong_kind), "чужая kind прошла");
        assert_eq!(s.len(), 1);
    }
}
