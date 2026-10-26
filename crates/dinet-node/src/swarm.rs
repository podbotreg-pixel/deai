//! Роя-симуляция DI-NET (v0): вычисления идут ИЗ ПОЛУЧЕННЫХ wire-кадров.
//!
//! Проверяет главные гарантии M0/M1 (docs/ROADMAP.md):
//! 1. **DILoCo-расписание:** маскированные эры публикуют SecureAgg-share (0x11
//!    UPD_SHARE), редкие REVEAL-эры публикуют DP-дельты (0x22 ROBUST_INPUT).
//! 2. **Secure Aggregation:** сумма масок обнуляется точно; получатель не
//!    восстанавливает индивидуальные апдейты.
//! 3. **Byzantine-устойчивость:** 10% отравителей (уклоняются от DP-клипа)
//!    отсекаются: median norm-band → Krum-консенсус → trimmed mean.
//! 4. **Fork-choice:** чек-пойнт публикуется только если у ≥2/3 узлов качество
//!    не деградировало (EVAL_VOTE 0x31); иначе сеть остаётся на last-good.
//! 5. **Подпись кадров:** фальсифицированный/подменённый кадр выбрасывается.
//! 6. **Роутация размеров:** выход даже 40% узлов не останавливает сеть.
//!
//! Каждый узел — равный процессор эпохи (leaderless): сам считает скрининг,
//! сам голосует, сам применяет. Реплики честных узлов детерминированно
//! совпадают (одинаковые входы). Транспорт — `transport::Transport`;
//! libp2p вставится поверх того же набора сообщений.

use std::collections::BTreeMap;

use dinet_crypto::{sha256, DemoSigner, Signer};
use dinet_fl::secure_agg::{aggregate, client_share, to_fx};
use dinet_fl::toy_llm::local_fit;
use dinet_fl::toy_llm::toy_world::{byz_batch, honesty_batch};
use dinet_fl::toy_llm::{LoRAAdapter, Sample, ToyBase, ToyModel};
use dinet_fl::{krum_select, norm2, Lcg};
use dinet_proto::{Frame, MsgType, Payload, SENDER_LEN};

use crate::transport::{Transport, CH_ANNOUNCE, CH_SHARES, CH_VOTES};

pub type NodeId = u16;

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub train_steps: u32,
    pub lr: f32,
    pub max_norm: f32,
    pub sigma: f32,
    pub byzantine_pct: f32,
    /// интервал REVEAL-эры (DiLoCo: редкие тяжёлые раунды)
    pub reveal_every: u32,
    /// норм-полоса: доли МЕДИАНЫ честных норм (poison-heavy вылетает по ratio)
    pub norm_min: f32,
    pub norm_max: f32,
    /// порог Krum-консенсуса: drop при distance > ratio × медианы
    pub krum_ratio: f32,
    pub eval_size: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            train_steps: 150,
            lr: 0.35,
            max_norm: 2.0,
            sigma: 0.002,
            byzantine_pct: 0.10,
            reveal_every: 4,
            norm_min: 0.25,
            norm_max: 3.0,
            krum_ratio: 2.5,
            eval_size: 96,
        }
    }
}

/// Узел (leaderless: каждый считает всё из полученных кадров).
#[derive(Debug)]
pub struct Node {
    pub id: NodeId,
    pub signer: DemoSigner,
    pub base: ToyBase,
    pub adapter: LoRAAdapter,
    pub rep: f32,
    /// hash последнего принятого чек-пойнта (цепочка parent_id)
    pub last_cp: [u8; 32],
    /// сколько подделанных кадров выбрасывал этот узел
    pub rejected_frames: u32,
    /// sim-метка отравителя
    pub byz: bool,
}

impl Node {
    pub fn new(id: NodeId, world_seed: u64, byz: bool) -> Node {
        Node {
            id,
            signer: DemoSigner::new([id as u8; 32]),
            base: ToyBase::from_seed(world_seed),
            // ОБЩИЙ LoRA-init для всех узлов (FedIT-принцип: дельты узлов
            // сравнимы при усреднении только от одинаковой точки старта).
            adapter: LoRAAdapter::lora_init(42),
            rep: 0.0,
            last_cp: [0u8; 32],
            rejected_frames: 0,
            byz,
        }
    }

    fn local_batch(&self, n: u32) -> Vec<Sample> {
        if self.byz {
            byz_batch(n, 9_001 + self.id as u64)
        } else {
            honesty_batch(n, 1_001 + self.id as u64)
        }
    }
}

fn node_seed(n: &Node) -> [u8; 32] {
    [n.id as u8; 32]
}

/// PeerBook: pubkey → verifier (в v1 — ed25519 по did:key).
#[cfg(test)]
fn peer_book(nodes: &[Node]) -> BTreeMap<[u8; SENDER_LEN], Box<dyn Signer>> {
    let ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
    peer_book_ids(&ids)
}

/// Вклад masked-эры (SecureAgg-share по wire).
#[derive(Debug, Clone)]
pub struct ShareEntry {
    pub node: NodeId,
    pub fx: Vec<i64>,
}

/// Вклад reveal-эры (DP'd дельта + норма по wire).
#[derive(Debug, Clone)]
pub struct RevealEntry {
    pub node: NodeId,
    pub delta: Vec<f32>,
    pub norm: f32,
}

/// Отчёт эпохи (репортёр — узел 0 как один из равных).
#[derive(Debug, Clone)]
pub struct EraReport {
    pub era: u32,
    pub was_reveal: bool,
    pub contributors: usize,
    pub dropped: Vec<NodeId>,
    pub checkpoint_id: Option<[u8; 32]>,
    pub accuracy_before: f32,
    pub accuracy_after: f32,
    pub mean_norm: f32,
    pub masked_sum_ok: bool,
}

#[must_use]
pub fn accuracy(model: &ToyModel, eval: &[Sample]) -> f32 {
    let hits: u32 = eval
        .iter()
        .filter(|s| model.predict(&s.tokens) == s.chosen)
        .count() as u32;
    hits as f32 / eval.len() as f32
}

// ---------------------------------------------------------------------------
// Wire: сборка и проверка кадров
// ---------------------------------------------------------------------------

fn era_hash32(era: u32) -> [u8; 32] {
    let mut b = b"dinet-era|".to_vec();
    b.extend_from_slice(&era.to_le_bytes());
    sha256(&b)
}

fn cand_hash32(era: u32, agg_delta: &[f32]) -> [u8; 32] {
    let mut b = b"dinet-cand|".to_vec();
    b.extend_from_slice(&era.to_le_bytes());
    for v in agg_delta {
        b.extend_from_slice(&v.to_le_bytes());
    }
    sha256(&b)
}

fn fx_to_bytes(fx: &[i64]) -> Vec<u8> {
    fx.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_fx(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn sign_frame(node: &Node, mt: MsgType, payload: Payload) -> Vec<u8> {
    let sender = node_seed(node);
    let frame = Frame { msg_type: mt, sender, sig: [0u8; 64], payload };
    let body = frame.encode_payload().expect("payload помещается в лимит");
    let sig = node.signer.sign(&body);
    frame.encode_with_sender_sig(&sender, &sig).expect("кадр в лимите")
}

/// Разобрать входящие кадры: verify → Payload. Подделка = отбраковка.
fn decode_frames(
    me: &mut Node,
    book: &BTreeMap<[u8; SENDER_LEN], Box<dyn Signer>>,
    wire: Vec<Vec<u8>>,
) -> Vec<(NodeId, Payload)> {
    let mut out = Vec::new();
    for w in wire {
        match Frame::decode(&w) {
            Err(_) => me.rejected_frames += 1,
            Ok(f) => {
                let body = match f.encode_payload() {
                    Ok(b) => b,
                    Err(_) => {
                        me.rejected_frames += 1;
                        continue;
                    }
                };
                match book.get(&f.sender) {
                    Some(v) => {
                        let did = v.did();
                        if v.verify(&body, &did, &f.sig) {
                            // sim: NodeId = первый байт seed-ключа (до 256 узлов)
                            let nid = f.sender[0] as NodeId;
                            out.push((nid, f.payload));
                        } else {
                            me.rejected_frames += 1;
                        }
                    }
                    None => me.rejected_frames += 1,
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Robust-скрининг (reveal-эра)
// ---------------------------------------------------------------------------

/// norm band (по медиане) → Krum-консенсус → компонентный trimmed mean.
/// Возврат: (accepted nodes, dropped nodes, mean_delta).
pub fn robust_screen(
    contribs: &[RevealEntry],
    cfg: &SimConfig,
) -> (Vec<NodeId>, Vec<NodeId>, Vec<f32>) {
    let mut accepted: Vec<&RevealEntry> = Vec::new();
    let mut dropped: Vec<NodeId> = Vec::new();
    if contribs.is_empty() {
        return (Vec::new(), Vec::new(), vec![0.0; LoRAAdapter::num_params()]);
    }

    let mut norms: Vec<f32> = contribs.iter().map(|c| c.norm).collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = norms[norms.len() / 2].max(1e-6);
    let band = (median * cfg.norm_min, median * cfg.norm_max);

    for c in contribs {
        if c.norm >= band.0 && c.norm <= band.1 {
            accepted.push(c);
        } else {
            dropped.push(c.node);
        }
    }

    // Krum-консенсус: центр «скопления честных»; отдалённые отсекаются
    // (стандартное требование f < n/2).
    let deltas: Vec<Vec<f32>> = accepted.iter().map(|c| c.delta.clone()).collect();
    if deltas.len() >= 5 {
        let center_idx = krum_select(&deltas, 2).unwrap_or(0);
        let center = &deltas[center_idx];
        let dists: Vec<f32> = deltas
            .iter()
            .map(|d| norm2(&sub(d, center)))
            .collect();
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2].max(1e-6);
        let threshold = med * cfg.krum_ratio;
        let suspects: Vec<NodeId> = dists
            .iter()
            .enumerate()
            .filter(|(_, d)| **d > threshold)
            .map(|(k, _)| accepted[k].node)
            .collect();
        for s in suspects {
            if !dropped.contains(&s) {
                dropped.push(s);
            }
            accepted.retain(|c| c.node != s);
        }
    }

    // компонентный trimmed mean (β=1 с каждого края)
    let mut out = vec![0.0f32; LoRAAdapter::num_params()];
    if !accepted.is_empty() {
        let mut col: Vec<f32> = Vec::with_capacity(accepted.len());
        for d in 0..out.len() {
            col.clear();
            for c in &accepted {
                col.push(c.delta[d]);
            }
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = col.len();
            if n == 1 {
                out[d] = col[0];
                continue;
            }
            let lo = if n >= 3 { 1.min(n / 2) } else { 0 };
            let hi = n - lo;
            if hi <= lo {
                continue;
            }
            out[d] = col[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        }
    }
    let acc_ids = accepted.iter().map(|c| c.node).collect();
    (acc_ids, dropped, out)
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Инвариант Secure Agg: сумма masked shares == Σθ (truth известен только
/// симуляционному харнесу; настоящая сеть проверяет mask-constraint иначе).
fn collab_masked_check(shares: &[ShareEntry], truths: &[(NodeId, Vec<f32>, Vec<i64>)]) -> bool {
    let fx_only: Vec<Vec<i64>> = shares.iter().map(|s| s.fx.clone()).collect();
    let agg = aggregate(&fx_only);
    let mut plain = vec![0i64; truths.first().map(|t| t.2.len()).unwrap_or(0)];
    for (_n, _d, fx) in truths {
        for (a, b) in plain.iter_mut().zip(fx.iter()) {
            *a = a.wrapping_add(*b);
        }
    }
    agg == plain
}

// ---------------------------------------------------------------------------
// Эпоха: фазы A–E через транспорт (никаких прямых вызовов между узлами)
// ---------------------------------------------------------------------------

/// PeerBook по узлам этого процесса (тесты; партиции юзают peer_book_ids).: партициям (между процессами) не нужно
/// держать чужие Node'ы — достаточно id, чтобы проверить их подписи.
fn peer_book_ids(ids: &[NodeId]) -> BTreeMap<[u8; SENDER_LEN], Box<dyn Signer>> {
    let mut m = BTreeMap::new();
    for &id in ids {
        let key = [id as u8; 32];
        m.insert(key, Box::new(DemoSigner::new(key)) as Box<dyn Signer>);
    }
    m
}

/// Plain-дельты эпохи (truth харнеса: run_era сверяет с masked-суммой).
pub type Truth = (NodeId, Vec<f32>, Vec<i64>);

/// Промежуточное состояние узла после фаз B/C (вход tally-фазы).
#[derive(Debug, Clone)]
pub struct MidNode {
    pub contributors: usize,
    pub dropped: Vec<NodeId>,
    pub agg_delta: Vec<f32>,
    pub mean_norm: f32,
    pub masked_ok: bool,
    pub before_replica: LoRAAdapter,
    pub vote: Option<Vote>,
}

/// Собственный голос узла (для tally).
#[derive(Debug, Clone)]
pub struct Vote {
    pub candhash: [u8; 32],
    pub cand_acc: f32,
    pub accept: bool,
    pub agg_delta: Vec<f32>,
}

/// ФАЗА A: Publish — локальный fit + подписанный wire-кадр (broadcast).
/// Возврат: plain-дельты СВОИХ узлов (truth; харнесу/тегам проверки).
pub fn era_publish(
    mine: &mut [Node],
    all_ids: &[NodeId],
    transport: &mut dyn Transport,
    era: u32,
    cfg: &SimConfig,
    seed: u64,
) -> Vec<Truth> {
    let was_reveal = (era + 1) % cfg.reveal_every == 0;
    let ehash = era_hash32(era);
    let reg = pairwise_registry(all_ids.len());
    let mut truths: Vec<Truth> = Vec::with_capacity(mine.len());
    for node in mine.iter() {
        let batch = node.local_batch(64);
        let max_norm = if node.byz { 1_000_000.0 } else { cfg.max_norm };
        let rng_seed =
            seed ^ era.wrapping_mul(0x9E37_79B9).wrapping_mul(7 + node.id as u32) as u64;
        let mut rng = Lcg::new(rng_seed);
        let fit = local_fit(
            &node.base,
            &node.adapter,
            &batch,
            cfg.train_steps,
            cfg.lr,
            2.0, // margin_preference (saturating hinge)
            max_norm,
            cfg.sigma,
            &mut rng,
        );
        let fx: Vec<i64> = fit.delta.iter().map(|&x| to_fx(x)).collect();
        truths.push((node.id, fit.delta.clone(), fx.clone()));
        if was_reveal {
            let frame = sign_frame(
                node,
                MsgType::RobustInput,
                Payload::RobustInput {
                    era_id: ehash,
                    delta: fit.delta.clone(),
                    norm: fit.raw_norm,
                    steps: fit.steps,
                },
            );
            transport.send(CH_SHARES, node.id, None, &frame);
        } else {
            let share = client_share(node.id as usize, &[node.id as u8; 16], &reg.registry, &fx);
            let frame = sign_frame(
                node,
                MsgType::UpdShare,
                Payload::UpdShare {
                    era_id: ehash,
                    agg_id: ehash,
                    c_share: fx_to_bytes(&share),
                    nonce: [node.id as u8; 16],
                },
            );
            transport.send(CH_SHARES, node.id, None, &frame);
        }
    }
    truths
}

/// ФАЗЫ B+C: Collect+Compute (скрининг по полученным кадрам) и собственный голос.
///
/// `checker` — только для run_era: (все plain- truths роя, id узла-репортёра);
/// партиция передаёт None (у неё нет чужих plain-дельт).
pub fn era_collect_and_vote(
    mine: &mut [Node],
    all_ids: &[NodeId],
    transport: &mut dyn Transport,
    era: u32,
    cfg: &SimConfig,
    checker: Option<(&[Truth], NodeId)>,
) -> Vec<MidNode> {
    let was_reveal = (era + 1) % cfg.reveal_every == 0;
    let ehash = era_hash32(era);
    let mut mids = Vec::with_capacity(mine.len());
    for idx in 0..mine.len() {
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_SHARES, mine[idx].id);
        let payloads = decode_frames(&mut mine[idx], &book, wire);
        let mut shares: Vec<ShareEntry> = Vec::new();
        let mut reveals: Vec<RevealEntry> = Vec::new();
        for (nid, pl) in payloads.iter() {
            match pl {
                Payload::UpdShare { era_id, c_share, .. } if *era_id == ehash && !was_reveal => {
                    shares.push(ShareEntry { node: *nid, fx: bytes_to_fx(c_share) })
                }
                Payload::RobustInput { era_id, delta, norm, .. }
                    if *era_id == ehash && was_reveal && delta.len() == LoRAAdapter::num_params() => {
                    reveals.push(RevealEntry { node: *nid, delta: delta.clone(), norm: *norm })
                }
                _ => {}
            }
        }
        shares.sort_by_key(|s| s.node);
        reveals.sort_by_key(|r| r.node);
        let mut mid = MidNode {
            contributors: 0,
            dropped: Vec::new(),
            agg_delta: vec![0.0; LoRAAdapter::num_params()],
            mean_norm: 0.0,
            masked_ok: true,
            before_replica: mine[idx].adapter.clone(),
            vote: None,
        };
        if was_reveal {
            let mean_norm = if reveals.is_empty() {
                0.0
            } else {
                reveals.iter().map(|r| r.norm).sum::<f32>() / reveals.len() as f32
            };
            let (accepted, dropped, agg_delta) = robust_screen(&reveals, cfg);
            mid.contributors = accepted.len();
            mid.dropped = dropped;
            mid.agg_delta = agg_delta;
            mid.mean_norm = mean_norm;
        } else {
            mid.contributors = shares.len();
            mid.mean_norm = norm2(&mine[idx].adapter.as_vec());
            if let Some((truths, reporter)) = checker {
                if mine[idx].id == reporter {
                    mid.masked_ok = collab_masked_check(&shares, truths);
                }
            }
        }
        // --- Фаза C: голос на СВОЕГО кандидата (только reveal)
        if was_reveal {
            let replica = mine[idx].adapter.clone();
            let candidate = apply_delta(replica.clone(), &mid.agg_delta);
            let eval = honesty_batch(cfg.eval_size, 77_777 + era as u64);
            let cand_acc =
                accuracy(&ToyModel { base: mine[idx].base.clone(), adapter: candidate }, &eval);
            let probe_acc =
                accuracy(&ToyModel { base: mine[idx].base.clone(), adapter: replica }, &eval);
            let cand_hash = cand_hash32(era, &mid.agg_delta);
            let accept = cand_acc + 0.005 >= probe_acc;
            let frame = sign_frame(
                &mine[idx],
                MsgType::EvalVote,
                Payload::EvalVote {
                    checkpoint_id: cand_hash,
                    ppl: cand_acc,
                    pass_at_k: if accept && !mine[idx].byz { 100 } else { 0 },
                },
            );
            transport.send(CH_VOTES, mine[idx].id, None, &frame);
            mid.vote = Some(Vote {
                candhash: cand_hash,
                cand_acc,
                accept,
                agg_delta: mid.agg_delta.clone(),
            });
        }
        mids.push(mid);
    }
    mids
}

/// ФАЗЫ D+E: Tally (кворум/apply/announce) и отчёт каждого своего узла.
pub fn era_tally(
    mine: &mut [Node],
    all_ids: &[NodeId],
    transport: &mut dyn Transport,
    era: u32,
    cfg: &SimConfig,
    mids: Vec<MidNode>,
) -> Vec<EraReport> {
    let was_reveal = (era + 1) % cfg.reveal_every == 0;
    let ehash = era_hash32(era);

    // --- Фаза D
    for idx in 0..mine.len() {
        if !was_reveal {
            continue;
        }
        let book = peer_book_ids(all_ids);
        let wire = transport.collect(CH_VOTES, mine[idx].id);
        let payloads = decode_frames(&mut mine[idx], &book, wire);
        let mut accepts = 0usize;
        for (_nid, pl) in payloads.iter() {
            if let Payload::EvalVote { pass_at_k, .. } = pl {
                if *pass_at_k == 100 {
                    accepts += 1;
                }
            }
        }
        let quorum = accepts * 3 >= payloads.len().max(1) * 2;
        if let Some(mv) = mids[idx].vote.clone() {
            if quorum && mv.accept {
                mine[idx].adapter = apply_delta(mine[idx].adapter.clone(), &mv.agg_delta);
                let parent = mine[idx].last_cp;
                mine[idx].last_cp = mv.candhash;
                let mut buf = Vec::with_capacity(mv.agg_delta.len() * 4);
                for v in &mv.agg_delta {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                let frame = sign_frame(
                    &mine[idx],
                    MsgType::CheckpointAnnounce,
                    Payload::CheckpointAnnounce {
                        id: mv.candhash,
                        parent_id: parent,
                        era_id: ehash,
                        merge_recipe_hash: sha256(&buf),
                        ppl: mv.cand_acc,
                        pass_at_k: mids[idx].contributors as u8,
                    },
                );
                transport.send(CH_ANNOUNCE, mine[idx].id, None, &frame);
            }
        }
    }

    // --- Фаза E: отчёт
    let mut reports = Vec::with_capacity(mine.len());
    for idx in 0..mine.len() {
        let announces = transport.collect(CH_ANNOUNCE, mine[idx].id);
        let book = peer_book_ids(all_ids);
        let payloads = decode_frames(&mut mine[idx], &book, announces);
        let announcements_valid: Vec<[u8; 32]> = payloads
            .iter()
            .filter_map(|(_nid, pl)| match pl {
                Payload::CheckpointAnnounce { id, era_id, .. } if *era_id == ehash => Some(*id),
                _ => None,
            })
            .collect();
        let eval = honesty_batch(cfg.eval_size, 77_777 + era as u64);
        let acc_before = accuracy(
            &ToyModel { base: mine[idx].base.clone(), adapter: mids[idx].before_replica.clone() },
            &eval,
        );
        let acc_after = accuracy(
            &ToyModel { base: mine[idx].base.clone(), adapter: mine[idx].adapter.clone() },
            &eval,
        );
        reports.push(EraReport {
            era,
            was_reveal,
            contributors: mids[idx].contributors,
            dropped: mids[idx].dropped.clone(),
            checkpoint_id: announcements_valid.first().copied(),
            accuracy_before: acc_before,
            accuracy_after: acc_after,
            mean_norm: mids[idx].mean_norm,
            masked_sum_ok: mids[idx].masked_ok,
        });
    }
    reports
}

/// Фазы A–E над ПОДМНОЖЕСТВОМ узлов роя (без барьеров внутри — для тестов:
/// многопроцессный драйвер зовёт фазы по одной и скрещивает их барьерами).
pub fn run_era_partition(
    mine: &mut [Node],
    all_ids: &[NodeId],
    transport: &mut dyn Transport,
    era: u32,
    cfg: &SimConfig,
    seed: u64,
) -> Vec<EraReport> {
    let _truths = era_publish(mine, all_ids, transport, era, cfg, seed);
    let mids = era_collect_and_vote(mine, all_ids, transport, era, cfg, None);
    era_tally(mine, all_ids, transport, era, cfg, mids)
}

/// Один шаг эпохи (полный рой в этом процессе + secure-agg инвариант).
pub fn run_era(
    nodes: &mut Vec<Node>,
    transport: &mut dyn Transport,
    era: u32,
    cfg: &SimConfig,
    seed: u64,
) -> EraReport {
    let all_ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
    let truths = era_publish(nodes, &all_ids, transport, era, cfg, seed);
    let mids = era_collect_and_vote(
        nodes,
        &all_ids,
        transport,
        era,
        cfg,
        Some((&truths, all_ids[0])),
    );
    let reports = era_tally(nodes, &all_ids, transport, era, cfg, mids);
    reports.into_iter().next().expect("рою нужен хотя бы один узел")
}

fn apply_delta(replica: LoRAAdapter, agg_delta: &[f32]) -> LoRAAdapter {
    let mut out = replica;
    for (p, d) in out.a.iter_mut().zip(agg_delta.iter()) {
        *p += d;
    }
    let skip = out.a.len();
    for (p, d) in out.b.iter_mut().zip(agg_delta.iter().skip(skip)) {
        *p += d;
    }
    out
}

struct PairReg {
    registry: Vec<(usize, usize, [u8; 16])>,
}

fn pairwise_registry(n: usize) -> PairReg {
    let mut registry = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            registry.push((i, j, [i as u8; 16]));
        }
    }
    PairReg { registry }
}

/// Полная симуляция N эр (in-memory mesh).
pub fn simulate(n_nodes: usize, eras: u32, cfg: SimConfig, seed: u64) -> (Vec<EraReport>, Vec<Node>) {
    let n_byz = ((n_nodes as f32) * cfg.byzantine_pct) as usize;
    let mut nodes: Vec<Node> = (0..n_nodes)
        .map(|i| Node::new(i as u16, 7, i < n_byz))
        .collect();
    let mut mesh = crate::transport::MeshTransport::new(n_nodes);
    let mut reports = Vec::with_capacity(eras as usize);
    for era in 0..eras {
        reports.push(run_era(&mut nodes, &mut mesh, era, &cfg, seed));
    }
    (reports, nodes)
}

/// Стартовая точность модели (0-е состояние мира).
pub fn start_accuracy(nodes: &[Node], size: u32, seed: u64) -> f32 {
    let eval = honesty_batch(size, seed);
    let m = ToyModel { base: nodes[0].base.clone(), adapter: nodes[0].adapter.clone() };
    accuracy(&m, &eval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::FileSpool;

    #[test]
    fn probe_one_node_debug() {
        let cfg = SimConfig {
            byzantine_pct: 0.0,
            reveal_every: 1,
            lr: 0.7,
            train_steps: 180,
            ..Default::default()
        };
        let (reps, nodes) = simulate(1, 6, cfg, 3);
        for r in &reps {
            println!("era {} acc_after {:.3} dropped {:?}", r.era, r.accuracy_after, r.dropped);
        }
        let m = ToyModel { base: nodes[0].base.clone(), adapter: nodes[0].adapter.clone() };
        let eval = honesty_batch(96, 424242);
        let acc = accuracy(&m, &eval);
        println!("FINAL probe acc: {:.3}", acc);
        assert!(acc > 0.55, "single-node должен выучить правило");
    }

    #[test]
    fn masked_era_checks_secure_aggregation_invariant() {
        let cfg = SimConfig { byzantine_pct: 0.0, reveal_every: 99, ..Default::default() };
        let (reps, _) = simulate(5, 3, cfg, 2026);
        assert!(reps.iter().all(|r| r.masked_sum_ok), "secure-agg инвариант нарушен");
        assert!(reps.iter().all(|r| r.checkpoint_id.is_none()), "candidate era без reveal");
    }

    #[test]
    fn byzantine_is_filtered_on_reveal() {
        let cfg = SimConfig::default();
        let (reps, _) = simulate(20, 8, cfg, 2026);
        let reveals: Vec<&EraReport> = reps.iter().filter(|r| r.was_reveal).collect();
        assert!(!reveals.is_empty(), "reveal-эры каждые 4 эры");
        for r in &reveals {
            assert_eq!(r.dropped.len(), 2, "10% отравителей вычислены: {:?}", r.dropped);
        }
        let last = reveals[reveals.len() - 1];
        assert!(last.checkpoint_id.is_some(), "финал принял чек-пойнт");
        assert!(last.accuracy_after >= 0.5, "аккуратность после отбраковки: {}", last.accuracy_after);
    }

    #[test]
    fn no_byz_accepts_everybody() {
        let cfg = SimConfig { byzantine_pct: 0.0, ..Default::default() };
        let (reps, _) = simulate(12, 4, cfg, 5);
        for r in &reps {
            if r.was_reveal {
                assert!(r.dropped.is_empty());
                assert_eq!(r.contributors, 12);
            }
        }
    }

    #[test]
    fn quality_grows_over_checkpoints() {
        let cfg = SimConfig { byzantine_pct: 0.0, reveal_every: 2, train_steps: 90, ..Default::default() };
        let (reps, _) = simulate(10, 6, cfg, 11);
        let cps: Vec<&EraReport> = reps.iter().filter(|r| r.checkpoint_id.is_some()).collect();
        assert!(cps.len() >= 2);
        let last = cps[cps.len() - 1];
        assert!(last.accuracy_after > 0.30, "выучили правило: {}", last.accuracy_after);
        let prev = cps[cps.len() - 2];
        assert!(last.accuracy_after >= prev.accuracy_after - 0.02);
    }

    #[test]
    fn network_survives_many_exits() {
        let cfg = SimConfig { byzantine_pct: 0.0, ..Default::default() };
        let mut mesh = crate::transport::MeshTransport::new(20);
        let mut nodes: Vec<Node> = (0..20).map(|i| Node::new(i as u16, 7, false)).collect();
        for era in 0..4u32 {
            let rep = run_era(&mut nodes, &mut mesh, era, &cfg, 5);
            if rep.was_reveal {
                assert!(rep.checkpoint_id.is_some());
            }
        }
        nodes.truncate(12); // «выход» 40%
        let mut mesh = crate::transport::MeshTransport::new(12);
        for era in 4..8u32 {
            let rep = run_era(&mut nodes, &mut mesh, era, &cfg, 6);
            if rep.was_reveal {
                assert!(rep.checkpoint_id.is_some(), "чек-пойнт после выхода 40% узлов");
            }
        }
    }

    #[test]
    fn wire_frames_travel_through_file_spool() {
        // Два узла обмениваются настоящими wire-кадрами через ФАЙЛЫ:
        // reveal-эры → ROBUST_INPUT кадры → чек-пойнты из транспорта.
        let dir = std::env::temp_dir().join(format!("dinet-era-spool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = SimConfig {
            byzantine_pct: 0.0,
            reveal_every: 2,
            train_steps: 90,
            ..Default::default()
        };
        let mut spool = FileSpool::new(&dir).unwrap();
        let mut nodes: Vec<Node> = (0..2).map(|i| Node::new(i as u16, 7, false)).collect();
        let mut reps = Vec::new();
        for era in 0..4u32 {
            // Общая сетевая точка — спул-каталог (в реальном демо оба процесса
            // держат FileSpool на один и тот же каталог).
            reps.push(run_era(&mut nodes, &mut spool, era, &cfg, 9));
            spool.sweep(CH_ANNOUNCE);
        }
        spool.sweep(CH_SHARES);
        let cps: Vec<&EraReport> = reps.iter().filter(|r| r.checkpoint_id.is_some()).collect();
        assert!(!cps.is_empty(), "чек-пойнт из wire-кадров FileSpool");
        assert!(
            cps[cps.len() - 1].accuracy_after > 0.30,
            "обучение из файлов: {}",
            cps[cps.len() - 1].accuracy_after
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_frame_rejected() {
        // Подделка payload → decode ОК, подпись падает, кадр отбракован,
        // подлинный кадр того же отправителя проходит.
        let mut mesh = crate::transport::MeshTransport::new(2);
        let nodes: Vec<Node> = (0..2).map(|i| Node::new(i as u16, 7, false)).collect();
        let ehash = era_hash32(0);
        let mut tampered = sign_frame(
            &nodes[1],
            MsgType::RobustInput,
            Payload::RobustInput {
                era_id: ehash,
                delta: vec![0.01; LoRAAdapter::num_params()],
                norm: 0.5,
                steps: 90,
            },
        );
        let mid = tampered.len() / 2;
        tampered[mid] ^= 0xFF;
        mesh.send(CH_SHARES, 1, Some(0), &tampered);
        let good = sign_frame(
            &nodes[1],
            MsgType::RobustInput,
            Payload::RobustInput {
                era_id: ehash,
                delta: vec![0.01; LoRAAdapter::num_params()],
                norm: 0.5,
                steps: 90,
            },
        );
        mesh.send(CH_SHARES, 1, Some(0), &good);

        let mut me = Node::new(0, 7, false);
        let book = peer_book(&nodes);
        let wire = mesh.collect(CH_SHARES, 0);
        let payloads = decode_frames(&mut me, &book, wire);
        assert_eq!(me.rejected_frames, 1, "подделка отброшена");
        assert_eq!(payloads.len(), 1, "подлинный кадр принят");
    }

    #[test]
    fn partition_full_set_matches_run_era() {
        // партиция «в этом процессе все узлы роя» обязана давать те же
        // числа, что и run_era (включая bit-точную математику той же эры)
        let cfg = SimConfig { byzantine_pct: 0.0, ..Default::default() };
        let (reps_full, nodes_full) = simulate(6, 4, cfg, 2026);
        // recreate same stream через партицию
        let mut mesh = crate::transport::MeshTransport::new(6);
        let mut nodes: Vec<Node> = (0..6).map(|i| Node::new(i as u16, 7, false)).collect();
        let ids: Vec<NodeId> = (0..6).map(|i| i as u16).collect();
        for era in 0..4u32 {
            let rep = run_era_partition(&mut nodes, &ids, &mut mesh, era, &cfg, 2026)
                .remove(0);
            let f = &reps_full[era as usize];
            assert_eq!(rep.era, f.era);
            assert_eq!(rep.was_reveal, f.was_reveal);
            assert_eq!(rep.contributors, f.contributors, "era {era}");
            assert_eq!(rep.dropped, f.dropped);
            assert_eq!(rep.checkpoint_id, f.checkpoint_id);
            assert_eq!(rep.accuracy_after, f.accuracy_after, "era {era}");
            assert_eq!(rep.masked_sum_ok, f.masked_sum_ok);
        }
        let _ = nodes_full;
    }

    #[test]
    fn barrier_emulated_two_processes_match_run_era() {
        // Процесс A владеет узлами {0,1,2}, B — {3,4,5}: фазы скрещиваются
        // барьерами (здесь — последовательным порядком), общий один transport
        // (в бине spoolswarm — ФАЙЛЫ спула). Результат обязан бит-точно
        // совпасть с run_ера полного роя: fit детерминирован по node.id.
        let cfg = SimConfig { byzantine_pct: 0.0, reveal_every: 2, train_steps: 90, ..Default::default() };
        let (reps_full, nodes_full) = simulate(6, 4, cfg, 5);

        let mut a: Vec<Node> = (0..3).map(|i| Node::new(i as u16, 7, false)).collect();
        let mut b: Vec<Node> = (3..6).map(|i| Node::new(i as u16, 7, false)).collect();
        let ids: Vec<NodeId> = (0..6).map(|i| i as u16).collect();
        let mut shared = crate::transport::MeshTransport::new(6);
        for era in 0..4u32 {
            era_publish(&mut a, &ids, &mut shared, era, &cfg, 5);
            era_publish(&mut b, &ids, &mut shared, era, &cfg, 5);
            let ma = era_collect_and_vote(&mut a, &ids, &mut shared, era, &cfg, None);
            let mb = era_collect_and_vote(&mut b, &ids, &mut shared, era, &cfg, None);
            let ra = era_tally(&mut a, &ids, &mut shared, era, &cfg, ma);
            let _rb = era_tally(&mut b, &ids, &mut shared, era, &cfg, mb);
            // чек-пойнты обеих сторон совпадают с полным роем
            assert_eq!(ra[0].checkpoint_id, reps_full[era as usize].checkpoint_id, "era {era}");
        }
        // финальные реплики бит-точно равны полностью одному рою
        for i in 0..3 {
            assert_eq!(a[i].adapter.as_vec(), nodes_full[i].adapter.as_vec(), "узел {i}");
            assert_eq!(b[i].adapter.as_vec(), nodes_full[i + 3].adapter.as_vec(), "узел {}", i + 3);
        }
    }

    #[test]
    fn single_node_network_still_functions() {
        let cfg = SimConfig { byzantine_pct: 0.0, reveal_every: 1, ..Default::default() };
        let (reps, _) = simulate(1, 2, cfg, 42);
        assert!(
            reps.iter().all(|r| r.checkpoint_id.is_some()),
            "1-node сеть не останавливается"
        );
    }
}
