//! DI-NET wire-протокол v0.
//!
//! Формат кадра: `version(1B) | type(1B) | sender(32B) | sig(64B) | payload`.
//!
//! Encoder — детерминированный «CBOR-lite»: каждое поле = `tag(1B) | kind(1B) |
//! len(varint) | value`. Поля всегда идут в порядке объявления структуры.
//! Только std — формат самодостаточен.
//!
//! Референс: `docs/PROTOCOL.md`.

pub const PROTO_VERSION: u8 = 0;
pub const SENDER_LEN: usize = 32;
pub const SIG_LEN: usize = 64;
pub const MAX_UPD_SHARE_PAYLOAD: usize = 24 * 1024 * 1024;
pub const MAX_SKILL_SHARE_PAYLOAD: usize = 256 * 1024;
pub const DEFAULT_FRAME_LIMIT: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtoError {
    TooShort,
    UnknownType(u8),
    BadVarint,
    BadKind,
    BadLength,
    FieldMismatch,
    PayloadTooBig(usize),
}

const K_NUM: u8 = 1;
const K_B32: u8 = 2;
const K_BYTES: u8 = 3;
const K_F32: u8 = 4;
const K_BOOL: u8 = 5;

// --- теги полей ---
const T_PROTO_VER: u8 = 1;
const T_CAPS: u8 = 2;
const T_PORT: u8 = 3;
const T_PEERS: u8 = 4;
const T_ERA: u8 = 5;
const T_AHASH: u8 = 6;
const T_EPS: u8 = 7;
const T_SIZE: u8 = 8;
const T_STEPS: u8 = 9;
const T_AGGID: u8 = 10;
const T_CSHARE: u8 = 11;
const T_NONCE: u8 = 12;
const T_AGGRHASH: u8 = 13;
const T_NCONTRIB: u8 = 14;
const T_KRUM: u8 = 15;
const T_FGRAPH: u8 = 16;
const T_MALIC: u8 = 17;
const T_AUDITOR: u8 = 18;
const T_CPID: u8 = 19;
const T_PPLO: u8 = 20;
const T_PPLN: u8 = 21;
const T_VERDICT: u8 = 22;
const T_ID: u8 = 23;
const T_PARENT: u8 = 24;
const T_MERGE: u8 = 25;
const T_PPL: u8 = 26;
const T_PATK: u8 = 27;
const T_SHASH: u8 = 28;
const T_ABI: u8 = 29;
const T_BODY: u8 = 30;
const T_DELTA: u8 = 31;
const T_NORM: u8 = 32;

// --- коды типов сообщений ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgType {
    Hello = 0x01,
    HelloAck = 0x02,
    TrainOffer = 0x10,
    UpdShare = 0x11,
    AggCommit = 0x20,
    RobustReport = 0x21,
    /// Каркас reveal-эры: DP'd адаптер-дельта (f32 LE) + норма-аттестация.
    /// Публикуется ТОЛЬКО в reveal-эрах (редкие); в masked-эрах — только 0x11.
    RobustInput = 0x22,
    LazyAudit = 0x23,
    CheckpointAnnounce = 0x30,
    EvalVote = 0x31,
    SkillShare = 0x40,
}

impl MsgType {
    pub fn code(self) -> u8 {
        self as u8
    }
    pub fn from_code(c: u8) -> Option<MsgType> {
        Some(match c {
            0x01 => MsgType::Hello,
            0x02 => MsgType::HelloAck,
            0x10 => MsgType::TrainOffer,
            0x11 => MsgType::UpdShare,
            0x20 => MsgType::AggCommit,
            0x21 => MsgType::RobustReport,
            0x22 => MsgType::RobustInput,
            0x23 => MsgType::LazyAudit,
            0x30 => MsgType::CheckpointAnnounce,
            0x31 => MsgType::EvalVote,
            0x40 => MsgType::SkillShare,
            _ => return None,
        })
    }
    fn limit(self) -> usize {
        match self {
            MsgType::UpdShare => MAX_UPD_SHARE_PAYLOAD,
            MsgType::SkillShare => MAX_SKILL_SHARE_PAYLOAD,
            _ => DEFAULT_FRAME_LIMIT,
        }
    }
}

/// caps: [trainer, aggregator, notary, evaluator]
pub type Caps = [bool; 4];

#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    Hello { proto_ver: u16, caps: Caps, port: u16 },
    HelloAck { proto_ver: u16, caps: Caps, port: u16, peer_ids: Vec<[u8; 32]> },
    TrainOffer { era_id: [u8; 32], adapter_hash: [u8; 32], dp_eps_x10: u8, size_bytes: u32, steps: u32 },
    UpdShare { era_id: [u8; 32], agg_id: [u8; 32], c_share: Vec<u8>, nonce: [u8; 16] },
    AggCommit { era_id: [u8; 32], agg_result_hash: [u8; 32], num_contrib: u16 },
    RobustReport { era_id: [u8; 32], krum_score: f32, fedgraph_score: f32, marked_malicious: u16 },
    RobustInput { era_id: [u8; 32], /// DP'd дельта адаптера (f32 LE, компонентно)
                  delta: Vec<f32>, /// аттестованная L2-норма (до клипа)
                  norm: f32, /// proof-of-training claim (шагов локально)
                  steps: u32 },
    LazyAudit { auditor_id: [u8; 32], checkpoint_id: [u8; 32], ppl_before: f32, ppl_after: f32, verdict: bool },
    CheckpointAnnounce { id: [u8; 32], parent_id: [u8; 32], era_id: [u8; 32], merge_recipe_hash: [u8; 32], ppl: f32, pass_at_k: u8 },
    EvalVote { checkpoint_id: [u8; 32], ppl: f32, pass_at_k: u8 },
    SkillShare { skill_hash: [u8; 32], abi_version: u16, body: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub msg_type: MsgType,
    pub sender: [u8; SENDER_LEN],
    pub sig: [u8; SIG_LEN],
    pub payload: Payload,
}

// ---------------------------------------------------------------------------
// Кодирование (детерминированное)
// ---------------------------------------------------------------------------

struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    fn prim(&mut self, tag: u8, kind: u8, value: &[u8]) {
        self.buf.push(tag);
        self.buf.push(kind);
        put_varint(&mut self.buf, value.len() as u32);
        self.buf.extend_from_slice(value);
    }
    fn num(&mut self, tag: u8, v: u32) {
        let le = v.to_le_bytes();
        self.prim(tag, K_NUM, &le);
    }
    fn f32(&mut self, tag: u8, v: f32) {
        self.prim(tag, K_F32, &v.to_le_bytes());
    }
    fn bool(&mut self, tag: u8, v: bool) {
        self.prim(tag, K_BOOL, &[v as u8]);
    }
    fn b32(&mut self, tag: u8, v: &[u8; 32]) {
        self.prim(tag, K_B32, v);
    }
    fn bytes(&mut self, tag: u8, v: &[u8]) {
        self.prim(tag, K_BYTES, v);
    }
}

fn encode_payload(p: &Payload) -> Vec<u8> {
    let mut e = Enc { buf: Vec::with_capacity(128) };
    match p {
        Payload::Hello { proto_ver, caps, port }
        | Payload::HelloAck { proto_ver, caps, port, .. } => {
            e.num(T_PROTO_VER, *proto_ver as u32);
            for (i, c) in caps.iter().enumerate() {
                e.bool(T_CAPS + i as u8, *c);
            }
            e.num(T_PORT, *port as u32);
            if let Payload::HelloAck { peer_ids, .. } = p {
                e.buf.push(T_PEERS);
                e.buf.push(K_B32);
                put_varint(&mut e.buf, (peer_ids.len() * 32) as u32);
                for id in peer_ids {
                    e.buf.extend_from_slice(id);
                }
            }
        }
        Payload::TrainOffer { era_id, adapter_hash, dp_eps_x10, size_bytes, steps } => {
            e.b32(T_ERA, era_id);
            e.b32(T_AHASH, adapter_hash);
            e.num(T_EPS, *dp_eps_x10 as u32);
            e.num(T_SIZE, *size_bytes);
            e.num(T_STEPS, *steps);
        }
        Payload::UpdShare { era_id, agg_id, c_share, nonce } => {
            e.b32(T_ERA, era_id);
            e.b32(T_AGGID, agg_id);
            e.bytes(T_CSHARE, c_share);
            e.prim(T_NONCE, K_BYTES, nonce);
        }
        Payload::AggCommit { era_id, agg_result_hash, num_contrib } => {
            e.b32(T_ERA, era_id);
            e.b32(T_AGGRHASH, agg_result_hash);
            e.num(T_NCONTRIB, *num_contrib as u32);
        }
        Payload::RobustReport { era_id, krum_score, fedgraph_score, marked_malicious } => {
            e.b32(T_ERA, era_id);
            e.f32(T_KRUM, *krum_score);
            e.f32(T_FGRAPH, *fedgraph_score);
            e.num(T_MALIC, *marked_malicious as u32);
        }
        Payload::RobustInput { era_id, delta, norm, steps } => {
            e.b32(T_ERA, era_id);
            let mut le = Vec::with_capacity(delta.len() * 4);
            for v in delta {
                le.extend_from_slice(&v.to_le_bytes());
            }
            e.bytes(T_DELTA, &le);
            e.f32(T_NORM, *norm);
            e.num(T_STEPS, *steps);
        }
        Payload::LazyAudit { auditor_id, checkpoint_id, ppl_before, ppl_after, verdict } => {
            e.b32(T_AUDITOR, auditor_id);
            e.b32(T_CPID, checkpoint_id);
            e.f32(T_PPLO, *ppl_before);
            e.f32(T_PPLN, *ppl_after);
            e.bool(T_VERDICT, *verdict);
        }
        Payload::CheckpointAnnounce { id, parent_id, era_id, merge_recipe_hash, ppl, pass_at_k } => {
            e.b32(T_ID, id);
            e.b32(T_PARENT, parent_id);
            e.b32(T_ERA, era_id);
            e.b32(T_MERGE, merge_recipe_hash);
            e.f32(T_PPL, *ppl);
            e.num(T_PATK, *pass_at_k as u32);
        }
        Payload::EvalVote { checkpoint_id, ppl, pass_at_k } => {
            e.b32(T_CPID, checkpoint_id);
            e.f32(T_PPL, *ppl);
            e.num(T_PATK, *pass_at_k as u32);
        }
        Payload::SkillShare { skill_hash, abi_version, body } => {
            e.b32(T_SHASH, skill_hash);
            e.num(T_ABI, *abi_version as u32);
            e.bytes(T_BODY, body);
        }
    }
    e.buf
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let mut b = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Декодирование
// ---------------------------------------------------------------------------

struct Dec<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ProtoError> {
        if self.pos + n > self.b.len() {
            return Err(ProtoError::TooShort);
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn varint(&mut self) -> Result<u32, ProtoError> {
        let mut v: u32 = 0;
        let mut shift = 0u32;
        loop {
            let b = self.take(1)?[0];
            v |= ((b & 0x7F) as u32) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 31 {
                return Err(ProtoError::BadVarint);
            }
        }
    }
    /// Читает поле с ожидаемым тегом, проверяет kind и длину.
    fn field(&mut self, expect_tag: u8, kinds: &[u8], max_len: usize) -> Result<Vec<u8>, ProtoError> {
        let tag = self.take(1)?[0];
        if tag != expect_tag {
            return Err(ProtoError::FieldMismatch);
        }
        let kind = self.take(1)?[0];
        if !kinds.contains(&kind) {
            return Err(ProtoError::BadKind);
        }
        let len = self.varint()? as usize;
        if len > max_len {
            return Err(ProtoError::PayloadTooBig(len));
        }
        Ok(self.take(len)?.to_vec())
    }
    fn u32(&mut self, tag: u8) -> Result<u32, ProtoError> {
        let v = self.field(tag, &[K_NUM], 5)?;
        let mut x: u32 = 0;
        for (i, b) in v.iter().enumerate() {
            x |= (*b as u32) << (8 * i);
        }
        Ok(x)
    }
    fn u16(&mut self, tag: u8) -> Result<u16, ProtoError> {
        let x = self.u32(tag)?;
        if x > u16::MAX as u32 {
            return Err(ProtoError::BadLength);
        }
        Ok(x as u16)
    }
    fn u8(&mut self, tag: u8) -> Result<u8, ProtoError> {
        let x = self.u32(tag)?;
        if x > u8::MAX as u32 {
            return Err(ProtoError::BadLength);
        }
        Ok(x as u8)
    }
    fn f32(&mut self, tag: u8) -> Result<f32, ProtoError> {
        let v = self.field(tag, &[K_F32], 4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(&v);
        Ok(f32::from_le_bytes(a))
    }
    fn bool(&mut self, tag: u8) -> Result<bool, ProtoError> {
        let v = self.field(tag, &[K_BOOL], 1)?;
        Ok(v.first() == Some(&1))
    }
    fn b32(&mut self, tag: u8) -> Result<[u8; 32], ProtoError> {
        let v = self.field(tag, &[K_B32, K_BYTES], 32)?;
        if v.len() != 32 {
            return Err(ProtoError::BadLength);
        }
        let mut a = [0u8; 32];
        a.copy_from_slice(&v);
        Ok(a)
    }
    fn bytes(&mut self, tag: u8, max: usize) -> Result<Vec<u8>, ProtoError> {
        self.field(tag, &[K_BYTES], max)
    }
    fn b16(&mut self, tag: u8) -> Result<[u8; 16], ProtoError> {
        let v = self.bytes(tag, 16)?;
        if v.len() != 16 {
            return Err(ProtoError::BadLength);
        }
        let mut a = [0u8; 16];
        a.copy_from_slice(&v);
        Ok(a)
    }
}

fn decode_payload(mt: MsgType, b: &[u8]) -> Result<Payload, ProtoError> {
    let mut d = Dec { b, pos: 0 };
    match mt {
        MsgType::Hello => {
            let proto_ver = d.u16(T_PROTO_VER)?;
            let mut caps = [false; 4];
            for i in 0..4 {
                caps[i] = d.bool(T_CAPS + i as u8)?;
            }
            let port = d.u16(T_PORT)?;
            Ok(Payload::Hello { proto_ver, caps, port })
        }
        MsgType::HelloAck => {
            let proto_ver = d.u16(T_PROTO_VER)?;
            let mut caps = [false; 4];
            for i in 0..4 {
                caps[i] = d.bool(T_CAPS + i as u8)?;
            }
            let port = d.u16(T_PORT)?;
            let raw = d.field(T_PEERS, &[K_B32, K_BYTES], 32 * 4096)?;
            if raw.len() % 32 != 0 {
                return Err(ProtoError::BadLength);
            }
            let peer_ids: Vec<[u8; 32]> = (0..raw.len() / 32)
                .map(|i| {
                    let mut p = [0u8; 32];
                    p.copy_from_slice(&raw[i * 32..(i + 1) * 32]);
                    p
                })
                .collect();
            Ok(Payload::HelloAck { proto_ver, caps, port, peer_ids })
        }
        MsgType::TrainOffer => Ok(Payload::TrainOffer {
            era_id: d.b32(T_ERA)?,
            adapter_hash: d.b32(T_AHASH)?,
            dp_eps_x10: d.u8(T_EPS)?,
            size_bytes: d.u32(T_SIZE)?,
            steps: d.u32(T_STEPS)?,
        }),
        MsgType::UpdShare => Ok(Payload::UpdShare {
            era_id: d.b32(T_ERA)?,
            agg_id: d.b32(T_AGGID)?,
            c_share: d.bytes(T_CSHARE, MAX_UPD_SHARE_PAYLOAD)?,
            nonce: d.b16(T_NONCE)?,
        }),
        MsgType::AggCommit => Ok(Payload::AggCommit {
            era_id: d.b32(T_ERA)?,
            agg_result_hash: d.b32(T_AGGRHASH)?,
            num_contrib: d.u16(T_NCONTRIB)?,
        }),
        MsgType::RobustReport => Ok(Payload::RobustReport {
            era_id: d.b32(T_ERA)?,
            krum_score: d.f32(T_KRUM)?,
            fedgraph_score: d.f32(T_FGRAPH)?,
            marked_malicious: d.u16(T_MALIC)?,
        }),
        MsgType::RobustInput => {
            let era_id = d.b32(T_ERA)?;
            let raw = d.bytes(T_DELTA, 4 * 65536)?;
            if raw.len() % 4 != 0 {
                return Err(ProtoError::BadLength);
            }
            let delta: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Payload::RobustInput {
                era_id,
                delta,
                norm: d.f32(T_NORM)?,
                steps: d.u32(T_STEPS)?,
            })
        }
        MsgType::LazyAudit => Ok(Payload::LazyAudit {
            auditor_id: d.b32(T_AUDITOR)?,
            checkpoint_id: d.b32(T_CPID)?,
            ppl_before: d.f32(T_PPLO)?,
            ppl_after: d.f32(T_PPLN)?,
            verdict: d.bool(T_VERDICT)?,
        }),
        MsgType::CheckpointAnnounce => Ok(Payload::CheckpointAnnounce {
            id: d.b32(T_ID)?,
            parent_id: d.b32(T_PARENT)?,
            era_id: d.b32(T_ERA)?,
            merge_recipe_hash: d.b32(T_MERGE)?,
            ppl: d.f32(T_PPL)?,
            pass_at_k: d.u8(T_PATK)?,
        }),
        MsgType::EvalVote => Ok(Payload::EvalVote {
            checkpoint_id: d.b32(T_CPID)?,
            ppl: d.f32(T_PPL)?,
            pass_at_k: d.u8(T_PATK)?,
        }),
        MsgType::SkillShare => Ok(Payload::SkillShare {
            skill_hash: d.b32(T_SHASH)?,
            abi_version: d.u16(T_ABI)?,
            body: d.bytes(T_BODY, MAX_SKILL_SHARE_PAYLOAD)?,
        }),
    }
}

impl Frame {
    /// Тело: `version|type|payload` (детерминированная сериализация payload).
    pub fn encode_payload(&self) -> Result<Vec<u8>, ProtoError> {
        let mut out = encode_payload(&self.payload);
        if out.len() > self.msg_type.limit() {
            return Err(ProtoError::PayloadTooBig(out.len()));
        }
        let mut full = Vec::with_capacity(2 + out.len());
        full.push(PROTO_VERSION);
        full.push(self.msg_type.code());
        full.append(&mut out);
        Ok(full)
    }

    /// Полный кадр согласно docs/PROTOCOL.md §1:
    /// `version(1B) | type(1B) | sender(32B) | sig(64B) | payload`.
    pub fn encode_with_sender_sig(&self, sender: &[u8; 32], sig: &[u8; 64]) -> Result<Vec<u8>, ProtoError> {
        let payload = self.encode_payload()?;
        let mut full = Vec::with_capacity(98 + payload.len() - 2);
        full.push(PROTO_VERSION);
        full.push(self.msg_type.code());
        full.extend_from_slice(sender);
        full.extend_from_slice(sig);
        full.extend_from_slice(&payload[2..]); // payload без дубля version|type
        Ok(full)
    }

    pub fn decode(full: &[u8]) -> Result<Frame, ProtoError> {
        if full.len() < 2 {
            return Err(ProtoError::TooShort);
        }
        if full[0] != PROTO_VERSION {
            return Err(ProtoError::UnknownType(full[0]));
        }
        let mt = MsgType::from_code(full[1]).ok_or(ProtoError::UnknownType(full[1]))?;
        if full.len() < 2 + 96 {
            return Err(ProtoError::TooShort);
        }
        let mut sender = [0u8; 32];
        sender.copy_from_slice(&full[2..34]);
        let mut sig = [0u8; 64];
        sig.copy_from_slice(&full[34..98]);
        let payload = decode_payload(mt, &full[98..])?;
        Ok(Frame { msg_type: mt, sender, sig, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(mt: MsgType, p: Payload) -> Frame {
        Frame { msg_type: mt, sender: [1; 32], sig: [2; 64], payload: p }
    }

    #[test]
    fn varint_roundtrip() {
        for v in [0u32, 1, 127, 128, 300, 65_535, u32::MAX] {
            let mut buf = Vec::new();
            put_varint(&mut buf, v);
            let mut d = Dec { b: &buf, pos: 0 };
            assert_eq!(d.varint().unwrap(), v);
        }
    }

    #[test]
    fn full_roundtrip_all_types() {
        let cases = vec![
            (
                MsgType::Hello,
                Payload::Hello { proto_ver: 0, caps: [true, false, false, true], port: 4437 },
            ),
            (
                MsgType::HelloAck,
                Payload::HelloAck {
                    proto_ver: 0,
                    caps: [true, true, true, true],
                    port: 4437,
                    peer_ids: vec![[7u8; 32], [8u8; 32]],
                },
            ),
            (
                MsgType::TrainOffer,
                Payload::TrainOffer {
                    era_id: [9; 32],
                    adapter_hash: [5; 32],
                    dp_eps_x10: 60,
                    size_bytes: 4_000_000,
                    steps: 300,
                },
            ),
            (
                MsgType::UpdShare,
                Payload::UpdShare {
                    era_id: [1; 32],
                    agg_id: [2; 32],
                    c_share: vec![3u8; 7_000_000],
                    nonce: [4; 16],
                },
            ),
            (
                MsgType::AggCommit,
                Payload::AggCommit { era_id: [1; 32], agg_result_hash: [2; 32], num_contrib: 77 },
            ),
            (
                MsgType::RobustReport,
                Payload::RobustReport { era_id: [1; 32], krum_score: 0.99, fedgraph_score: -1.5, marked_malicious: 3 },
            ),
            (
                MsgType::RobustInput,
                Payload::RobustInput { era_id: [4; 32], delta: vec![0.5, -1.25, 3.0, 0.0, 2.75], norm: 1.77, steps: 150 },
            ),
            (
                MsgType::LazyAudit,
                Payload::LazyAudit {
                    auditor_id: [1; 32],
                    checkpoint_id: [2; 32],
                    ppl_before: 8.4,
                    ppl_after: 8.9,
                    verdict: false,
                },
            ),
            (
                MsgType::CheckpointAnnounce,
                Payload::CheckpointAnnounce {
                    id: [3; 32],
                    parent_id: [2; 32],
                    era_id: [1; 32],
                    merge_recipe_hash: [9; 32],
                    ppl: 7.1,
                    pass_at_k: 55,
                },
            ),
            (
                MsgType::EvalVote,
                Payload::EvalVote { checkpoint_id: [3; 32], ppl: 7.0, pass_at_k: 60 },
            ),
            (
                MsgType::SkillShare,
                Payload::SkillShare { skill_hash: [5; 32], abi_version: 1, body: vec![6u8; 1000] },
            ),
        ];
        for (mt, p) in cases {
            let f = frame(mt, p.clone());
            let wire = f.encode_with_sender_sig(&f.sender, &f.sig).unwrap();
            let f2 = Frame::decode(&wire).unwrap();
            assert_eq!(f2.msg_type, mt);
            assert_eq!(f2.sender, f.sender);
            assert_eq!(f2.sig, f.sig);
            assert_eq!(f2.payload, p);
        }
    }

    #[test]
    fn deterministic() {
        let p = Payload::TrainOffer {
            era_id: [1; 32],
            adapter_hash: [2; 32],
            dp_eps_x10: 60,
            size_bytes: 1234,
            steps: 10,
        };
        assert_eq!(encode_payload(&p), encode_payload(&p));
    }

    #[test]
    fn oversize_rejected() {
        let f = frame(
            MsgType::UpdShare,
            Payload::UpdShare {
                era_id: [0; 32],
                agg_id: [0; 32],
                c_share: vec![0u8; MAX_UPD_SHARE_PAYLOAD + 1],
                nonce: [0; 16],
            },
        );
        assert!(matches!(f.encode_payload(), Err(ProtoError::PayloadTooBig(_))));
    }

    #[test]
    fn unknown_type_rejected() {
        assert!(MsgType::from_code(0x99).is_none());
    }
}
