//! Реальный сетевой слой DI-NET (v1): libp2p под нашими wire-кадрами.
//!
//! Архитектура склейки: ОДИН ed25519-ключ узла (32-байт seed) работает на
//! двух слоях:
//! 1. Libp2p-транспорт: Noise-рукопожатие + gossipsub-конверт (канал шифруется
//!    и подписывается libp2p этим же ключом).
//! 2. DI-NET протокол: КАЖДЫЙ кадр отдельно подписан `Ed25519Signer` (did:key);
//!    приёмник проверяет подпись по pubkey, вшитому в кадр (`Frame::sender`),
//!    поэтому кадр самодостаточен и не доверяет среде — gossipsub-ретранслятор
//!    может быть кем угодно, подделать payload он не в состоянии.
//!
//! Доверие: TOFU — кадры приносят pubkey отправителя, связка NodeId↔pubkey
//! учится роями из Hello/PeerBook (v2: anchor-реестры и ротация ключей).

use dinet_crypto::{Ed25519Signer, Ed25519Verifier, Signer, Verifier};
use futures::StreamExt;
use libp2p::gossipsub::{self, IdentTopic};
use libp2p::swarm::{NetworkBehaviour, SwarmEvent};
use libp2p::{identity, noise, tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder};
use std::time::Duration;
use tokio::sync::mpsc;

/// Срок жизни конфига gossipsub: mesh 1..=10 (маленькие рои должны graft-ить
/// быстро и не выкидывать пары).
fn gossip_config() -> Result<gossipsub::Config, gossipsub::ConfigBuilderError> {
    gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_millis(500))
        .mesh_n_low(1)
        .mesh_outbound_min(1)
        .mesh_n(10)
        .gossip_lazy(3)
        .history_length(128)
        .history_gossip(64)
        .build()
}

/// Поведение узла: пока только gossipsub (фазы эпохи поверх кадра).
#[derive(NetworkBehaviour)]
pub struct NodeBehaviour {
    pub gossip: gossipsub::Behaviour,
}

/// Команда сетевой задаче узла.
#[derive(Debug, Clone)]
pub enum Cmd {
    /// Соединиться с адресом пира.
    Dial(Multiaddr),
    /// Опубликовать wire-кадр (байты `Frame::encode_*`) в тему роя.
    Publish(Vec<u8>),
    /// Остановить задачу (graceful).
    Quit,
}

/// Событие сетевой задачи узла.
#[derive(Debug, Clone)]
pub enum NodeEvent {
    /// Локальный адрес после listen_on (port 0 → реальный порт).
    Listening(Multiaddr),
    /// Принятый и ПРОВЕРЕННЫЙ кадр (подпись ed25519 сходится).
    Frame(InFrame),
}

/// Принятый кадр после успешной проверки подписи.
#[derive(Debug, Clone)]
pub struct InFrame {
    /// did pubkey отправителя (32 байта из кадра).
    pub sender: [u8; dinet_crypto::ED25519_PUB_LEN],
    /// Payload кадра (подлинность подтверждена).
    pub payload: dinet_proto::Payload,
    /// Транспортный PeerId (кто принёс сообщение).
    pub peer: PeerId,
    /// СЫРЫЕ байты кадра (как их отдал отправитель — контракт Transport::collect).
    pub raw: Vec<u8>,
}

const TOPIC: &str = "dinet/v0";

/// Собрать swarm узла: ключ из seed, TCP+Noise+Yamux, gossipsub.
fn build_swarm(seed: [u8; 32]) -> Result<(Swarm<NodeBehaviour>, PeerId), String> {
    let key = identity::Keypair::ed25519_from_bytes(&mut seed.clone())
        .map_err(|e| format!("ключ: {e}"))?;
    let peer_id = PeerId::from(key.public());
    // gossipsub строится ДО сборщика (конструктор — fallible)
    let cfg = gossip_config().map_err(|e| format!("gossip-config: {e}"))?;
    let gossip = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(key.clone()),
        cfg,
    )
    .map_err(|e| format!("gossip: {e}"))?;
    let behaviour = NodeBehaviour { gossip };
    let swarm = SwarmBuilder::with_existing_identity(key.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .map_err(|e| format!("tcp-транспорт: {e}"))?
        .with_behaviour(move |_| behaviour)
        .map_err(|e| format!("поведение: {e}"))?
        .build();
    Ok((swarm, peer_id))
}

/// Проверка wire-кадра: подпись ed25519 по pubkey ИЗ КАДРА (самодостаточность).
pub fn verify_frame(bytes: &[u8], peer: PeerId) -> Option<InFrame> {
    let f = dinet_proto::Frame::decode(bytes).ok()?;
    let body = f.encode_payload().ok()?;
    let verifier = Ed25519Verifier::new(f.sender)?;
    if !verifier.verify(&body, &f.sig) {
        return None;
    }
    Some(InFrame { sender: f.sender, payload: f.payload, peer, raw: bytes.to_vec() })
}

/// Ручка сетевого узла.
pub struct NodeHandle {
    /// Команды узлу (dial/publish).
    pub cmd: mpsc::UnboundedSender<Cmd>,
    /// События узла (адреса/кадры).
    pub events: mpsc::UnboundedReceiver<NodeEvent>,
    /// did pubkey этого узла.
    pub pubkey: [u8; 32],
    /// Транспортный PeerId этого узла.
    pub peer_id: PeerId,
}

/// Запустить сетевую задачу узла (tokio-рантайм должен быть снаружи).
pub async fn spawn_node(seed: [u8; 32]) -> Result<NodeHandle, String> {
    let (mut swarm, peer_id) = build_swarm(seed)?;
    let did = Ed25519Signer::from_seed(&seed).pubkey();
    swarm
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().map_err(|e| format!("addr: {e}"))?)
        .map_err(|e| format!("listen: {e}"))?;
    let topic = IdentTopic::new(TOPIC);
    swarm
        .behaviour_mut()
        .gossip
        .subscribe(&topic)
        .map_err(|e| format!("subscribe: {e}"))?;

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<Cmd>();
    let (ev_tx, ev_rx) = mpsc::unbounded_channel::<NodeEvent>();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                cmd = cmd_rx.recv() => match cmd {
                    None | Some(Cmd::Quit) => break,
                    Some(Cmd::Dial(a)) => {
                        let _ = swarm.dial(a);
                    }
                    Some(Cmd::Publish(bytes)) => {
                        let _ = swarm.behaviour_mut().gossip.publish(topic.hash(), bytes);
                    }
                },
                ev = swarm.select_next_some() => match ev {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        let _ = ev_tx.send(NodeEvent::Listening(address));
                    }
                    SwarmEvent::Behaviour(NodeBehaviourEvent::Gossip(
                        gossipsub::Event::Message { message, .. },
                    )) => {
                        if let Some(frame) = verify_frame(&message.data, message.source.unwrap_or(peer_id)) {
                            let _ = ev_tx.send(NodeEvent::Frame(frame));
                        } else {
                            // Отбраковка: мусор/подделка не тиражируется наружу.
                        }
                    }
                    _ => {}
                }
            }
        }
    });
    Ok(NodeHandle { cmd: cmd_tx, events: ev_rx, pubkey: did, peer_id })
}

/// Полный цикл демо: N узлов → меш → обмен подписанными кадрами → проверка.
/// Используется бином `p2pmesh` и тестом (артефакт реально бегает в тестах).
pub async fn run_mesh_exchange(n: usize) -> Result<(), String> {
    // --- 1. Поднять узлы: один ed25519-ключ от seed [id;32] на ОБА слоя
    //        (libp2p-транспорт Noise/gossip и did:key подписи кадров).
    let mut handles = Vec::new();
    for id in 0..n {
        handles.push(
            spawn_node([id as u8; 32])
                .await
                .map_err(|e| format!("узел {id}: {e}"))?,
        );
    }

    // --- 2. Дождаться адресов (port 0 → реальные порты ОС).
    let mut addrs: Vec<(usize, Multiaddr)> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while addrs.len() < n {
        if tokio::time::Instant::now() > deadline {
            return Err(format!("адреса не собрались: {}/{}", addrs.len(), n));
        }
        if let Some((who, ev)) = try_any(&mut handles) {
            if let NodeEvent::Listening(a) = ev {
                addrs.push((who, a));
            }
            continue;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    addrs.sort();

    // --- 3. Полный меш: каждый dial каждого.
    for i in 0..n {
        for (who, a) in &addrs {
            if *who != i {
                let _ = handles[i].cmd.send(Cmd::Dial(a.clone()));
            }
        }
    }
    // Дать мешу graft-нуться (gossipsub heartbeat 500 мс).
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // --- 4. Каждый публикует ПОДПИСАННЫЙ RobustInput (reveal-кадр эпохи 7).
    let era = era7_id();
    for i in 0..n {
        let key = Ed25519Signer::from_seed(&[i as u8; 32]);
        let frame = signed_frame_bytes(
            &key,
            dinet_proto::MsgType::RobustInput,
            dinet_proto::Payload::RobustInput {
                era_id: era,
                delta: vec![0.1 * i as f32; 8],
                norm: 1.0 + i as f32,
                steps: 8,
            },
        )?;
        let _ = handles[i].cmd.send(Cmd::Publish(frame));
    }

    // --- 5. Собрать: у каждого узла ровно n-1 ЧУЖИХ валидных подписей.
    let mut got: Vec<Vec<[u8; 32]>> = vec![Vec::new(); n];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while got.iter().any(|g| g.len() < n - 1) {
        if tokio::time::Instant::now() > deadline {
            return Err(format!(
                "кадры не собрались: {:?}",
                got.iter().map(|g| g.len()).collect::<Vec<_>>()
            ));
        }
        if let Some((who, ev)) = try_any(&mut handles) {
            if let NodeEvent::Frame(f) = ev {
                let own = handles[who].pubkey;
                if f.sender != own {
                    got[who].push(f.sender);
                }
            }
            continue;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    for i in 0..n {
        let mut uniq = got[i].clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            n - 1,
            "узел {i}: уникальных отправителей {}/{}",
            uniq.len(),
            n - 1
        );
        for sender in &uniq {
            assert_ne!(
                &sender[..],
                &handles[i].pubkey[..],
                "свой кадр вернулся чужим"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// РЕАЛЬНЫЙ РОЙ НАДМОСТ: полный цикл эр через GossipBridge (тот же трейт
// Transport, что у симуляции/спула) + проверка БИТ-В-БИТ равенства с
// in-memory симуляцией. Работа бина p2pswarm и теста ниже.
// ---------------------------------------------------------------------------

pub mod swarm_net {
    use super::*;
    use dinet_node::swarm::{
        era_collect_and_vote, era_publish, era_tally, Node, SimConfig,
    };
    use dinet_fl::toy_llm::toy_world::honesty_batch;
    use dinet_node::transport::{MeshTransport, Transport};
    use std::sync::Barrier;

    /// Успокоение сети между фазами (localhost задержки — миллисекунды).
    fn settled(ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }

    /// Полный цикл эр одного сетевого узла (свой поток, свой мост).
    pub fn node_worker(
        id: u16,
        mut bridge: GossipBridge,
        all_ids: Vec<u16>,
        bar: Arc<Barrier>,
        eras: u32,
        cfg: SimConfig,
        seed: u64,
    ) -> dinet_fl::toy_llm::LoRAAdapter {
        let mut mine = vec![Node::new(id, 7, false)];
        for era in 0..eras {
            let transport: &mut dyn Transport = &mut bridge;

            // ФАЗА A: publish вкладов
            let _ = era_publish(&mut mine, &all_ids, transport, era, &cfg, seed);
            settled(90); // дать кадрам долететь до всех тем
            bar.wait();

            // ФАЗЫ B+C: collect → robust-скрининг → голоса
            let mids = era_collect_and_vote(&mut mine, &all_ids, transport, era, &cfg, None);
            settled(70);
            bar.wait();

            // ФАЗЫ D+E: кворум → apply → announce
            let _ = era_tally(&mut mine, &all_ids, transport, era, &cfg, mids);
            settled(70);
            bar.wait();
        }
        let final_node = mine.pop().expect("узел");
        final_node.adapter
    }

    fn fx_bytes(a: &dinet_fl::toy_llm::LoRAAdapter) -> Vec<u8> {
        let mut out = a.a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        out.extend(a.b.iter().flat_map(|v| v.to_le_bytes()));
        out
    }

    /// Драйвер: N узлов полной эры-фазы поверх РЕАЛЬНОЙ сети + равенство симуляции.
    /// Возвращает mean accuracy реплик роя после эр.
    pub fn run_network_eras(
        n: usize,
        eras: u32,
        reveal_every: u32,
        log: bool,
        seed: u64,
    ) -> Result<f32, String> {
        use dinet_node::swarm::{Node, SimConfig, run_era_partition, accuracy};
        use dinet_fl::toy_llm::ToyModel;

        // --- Мосты: каждый узел = свой ed25519-ключ (seed [id;32]) на TCP.
        let mut bridges = Vec::new();
        for id in 0..n as u16 {
            bridges.push(
                GossipBridge::spawn(id, [id as u8; 32])
                    .map_err(|e| format!("узел {id}: {e}"))?,
            );
        }
        // Полный меш реальными dial.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let a = bridges[j].addr.clone().unwrap();
                    bridges[i].dial(a);
                }
            }
        }
        settled(1400); // graft (gossipsub heartbeat 500 мс)

        let cfg = SimConfig { reveal_every, ..SimConfig::default() };
        let all_ids: Vec<u16> = (0..n as u16).collect();
        let (tx, rx) = std::sync::mpsc::channel::<(u16, dinet_fl::toy_llm::LoRAAdapter)>();
        let mut joins = Vec::new();
        let bar = Arc::new(std::sync::Barrier::new(n));
        for id in 0..n as u16 {
            let cfg2 = cfg.clone();
            let ids = all_ids.clone();
            let txc = tx.clone();
            let bridge = bridges.remove(0);
            let bar2 = bar.clone();
            joins.push(std::thread::spawn(move || {
                let adapter = node_worker(id, bridge, ids, bar2, eras, cfg2, seed);
                let _ = txc.send((id, adapter));
            }));
        }
        let mut adapters: Vec<(u16, dinet_fl::toy_llm::LoRAAdapter)> = Vec::new();
        for _ in 0..n {
            adapters.push(rx.recv().expect("реплика узла"));
        }
        for join in joins {
            let _ = join.join();
        }
        adapters.sort_by_key(|(id, _)| *id);

        // Консистентность роя: реплики всех узлов сошлись в одно.
        let first = fx_bytes(&adapters[0].1);
        for (id, a) in &adapters {
            assert_eq!(fx_bytes(a), first, "узел {id}: реплика разошлась с узлом 0");
        }

        // Эквивалентность: та же программа через in-memory mesh (обязана совпасть).
        let mut sim_nodes: Vec<Node> = (0..n as u16).map(|i| Node::new(i, 7, false)).collect();
        let mut mesh = MeshTransport::new(n);
        for era in 0..eras {
            let _ = run_era_partition(&mut sim_nodes, &all_ids, &mut mesh, era, &cfg, seed);
        }
        for (id, a) in &adapters {
            let sim = &sim_nodes[(*id) as usize].adapter;
            assert_eq!(
                fx_bytes(a),
                fx_bytes(sim),
                "узел {id}: сетевой рой ≠ in-memory симуляции"
            );
        }

        let eval = honesty_batch(cfg.eval_size, 999_999);
        let mean = adapters
            .iter()
            .map(|(_, a)| {
                accuracy(
                    &ToyModel { base: sim_nodes[0].base.clone(), adapter: a.clone() },
                    &eval,
                )
            })
            .sum::<f32>()
            / n as f32;
        if log {
            eprintln!("сетевой рой: {n} узлов × {eras} эр, mean acc {mean:.3} (≡ симуляции)");
        }
        Ok(mean)
    }
}

use std::sync::mpsc as std_mpsc;
use std::sync::Mutex as StdMutex;
use std::sync::Arc;

/// Канонический id эры (эпоха 7 — фикс демо-кадра) + сбор события любого узла.
fn era7_id() -> [u8; 32] {
    let mut e = [0u8; 32];
    e[..8].copy_from_slice(&7u64.to_le_bytes());
    e
}

fn try_any(handles: &mut [NodeHandle]) -> Option<(usize, NodeEvent)> {
    for (i, h) in handles.iter_mut().enumerate() {
        if let Ok(ev) = h.events.try_recv() {
            return Some((i, ev));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// GossipBridge: ТРАНСПОРТ РОИВОЙ ЭРЫ ПОВЕРХ РЕАЛЬНОЙ СЕТИ.
//
// Реализует dinet_node::transport::Transport — фазы эр (publish/vote/tally) работают
// БЕЗ ИЗМЕНЕНИЙ поверх TCP+Noise+Yamux+gossipsub. Роутинг как у файлового
// спула: тема `dinet/v0/{ch}/all` (broadcast) и `dinet/v0/{ch}/t{to}`
// (адресная). Loopback: mesh-семантика «свой кадр кладётся себе в инбокс»
// воспроизводится мостом (иначе pairwise-маски secure_agg не отменятся).
// Один мост = сетевое окно ОДНОГО узла (seed [id;32]).
// ---------------------------------------------------------------------------

/// Правило темы: `dinet/v0/{ch}/all` или `dinet/v0/{ch}/t{to}`.
fn topic_name(ch: u8, to: Option<u16>) -> String {
    match to {
        None => format!("{TOPIC_BASE}/{ch}/all"),
        Some(t) => format!("{TOPIC_BASE}/{ch}/t{t}"),
    }
}

/// Обратное: канал из имени темы (мусорные темы → None).
fn topic_channel(name: &gossipsub::TopicHash) -> Option<u8> {
    let parts: Vec<&str> = name.as_str().split('/').collect();
    if parts.len() == 4 && parts[0] == TOPIC_BASE.split('/').next().unwrap_or("") {
        parts[2].parse().ok()
    } else {
        None
    }
}

/// Команда сетевому потоку моста.
enum BridgeCmd {
    Dial(Multiaddr),
    /// Опубликовать кадр в тему канала (to = адресный направляет).
    Publish { ch: u8, to: Option<u16>, bytes: Vec<u8> },
    Quit,
}

pub struct GossipBridge {
    cmd_tx: std_mpsc::Sender<BridgeCmd>,
    /// Приёмные очереди каналов (их наполняет сетевой поток).
    queues: Vec<Arc<StdMutex<Vec<InFrame>>>>,
    /// Свой транспортный PeerId.
    pub peer: PeerId,
    /// Свой адрес (или None, пока не известен).
    pub addr: Option<Multiaddr>,
}

const CH_N: usize = 3; // SHARES / VOTES / ANNOUNCE
const TOPIC_BASE: &str = "dinet/v0";

impl GossipBridge {
    /// Поднять сетевой поток узла; BLOCKS до готовности адреса (≤5 с).
    /// Подписки: `all` и `t{me}` по всем трём каналам эры.
    pub fn spawn(me: u16, seed: [u8; 32]) -> Result<GossipBridge, String> {
        let (cmd_tx, cmd_rx) = std_mpsc::channel::<BridgeCmd>();
        let queues: Vec<Arc<StdMutex<Vec<InFrame>>>> =
            (0..CH_N).map(|_| Arc::new(StdMutex::new(Vec::new()))).collect();
        let ready: Arc<StdMutex<Option<(PeerId, Multiaddr)>>> = Arc::new(StdMutex::new(None));

        let q = queues.clone();
        let r = ready.clone();
        std::thread::Builder::new()
            .name(format!("dineth-{me}"))
            .spawn(move || gossip_node_thread(seed, cmd_rx, q, r))
            .map_err(|e| format!("thread: {e}"))?;

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some((peer, addr)) = ready.lock().expect("ready").clone() {
                return Ok(GossipBridge { cmd_tx, queues, peer, addr: Some(addr) });
            }
            if std::time::Instant::now() > deadline {
                return Err("листенер не поднялся за 5с".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// Соединиться с адресом чужого моста.
    pub fn dial(&self, addr: Multiaddr) {
        let _ = self.cmd_tx.send(BridgeCmd::Dial(addr));
    }

    /// Дождаться опорожнения команд (для вмешательства драйвера — опция).
    pub fn send_frame(&self, ch: u8, to: Option<u16>, bytes: Vec<u8>) {
        let _ = self.cmd_tx.send(BridgeCmd::Publish { ch, to, bytes });
    }
}

impl Drop for GossipBridge {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(BridgeCmd::Quit);
    }
}

/// Тело сетевого потока моста (own tokio-рантайм на своей ните).
fn gossip_node_thread(
    seed: [u8; 32],
    cmd_rx: std_mpsc::Receiver<BridgeCmd>,
    queues: Vec<Arc<StdMutex<Vec<InFrame>>>>,
    ready: Arc<StdMutex<Option<(PeerId, Multiaddr)>>>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("[bridge] runtime: {e}");
            return;
        }
    };
    rt.block_on(async move {
        let me_pub = Ed25519Signer::from_seed(&seed).pubkey();
        let me_id = u16::from_be_bytes([me_pub[0], me_pub[1]]);
        let (mut swarm, peer_id) = match build_swarm(seed) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[bridge] swarm: {e}");
                return;
            }
        };
        if swarm
            .listen_on("/ip4/127.0.0.1/tcp/0"
                .parse()
                .expect("multiaddr литерал"))
            .is_err()
        {
            eprintln!("[bridge] listen не поднялся");
            return;
        }
        // Подписки: все три канала, all + адресный (свой).
        for ch in 0..CH_N as u8 {
            for sub in [format!("all"), format!("t{me_id}")] {
                let t = gossipsub::IdentTopic::new(format!("{TOPIC_BASE}/{ch}/{sub}"));
                if swarm.behaviour_mut().gossip.subscribe(&t).unwrap_or(false) == false {
                    eprintln!("[bridge] подписка {t} не прошла");
                    return;
                }
            }
        }
        loop {
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    BridgeCmd::Quit => return,
                    BridgeCmd::Dial(a) => {
                        let _ = swarm.dial(a);
                    }
                    BridgeCmd::Publish { ch, to, bytes } => {
                        let t = gossipsub::IdentTopic::new(topic_name(ch, to));
                        let _ = swarm.behaviour_mut().gossip.publish(t.hash(), bytes);
                    }
                }
            }
            // Сеть с таймаутом (возвращаемся в командную секцию).
            let ev = tokio::time::timeout(
                std::time::Duration::from_millis(25),
                swarm.select_next_some(),
            )
            .await;
            match ev {
                Ok(SwarmEvent::NewListenAddr { address, .. }) => {
                    let addr = address.clone();
                    *ready.lock().expect("ready") = Some((peer_id, addr));
                }
                Ok(SwarmEvent::Behaviour(NodeBehaviourEvent::Gossip(
                    gossipsub::Event::Message { message, .. },
                ))) => {
                    if let Some(ch) = topic_channel(&message.topic) {
                        if let Some(frame) =
                            verify_frame(&message.data, message.source.unwrap_or(peer_id))
                        {
                            queues[usize::from(ch)].lock().expect("queue").push(frame);
                        }
                    }
                }
                Ok(_) => {}
                Err(_timeout_25ms) => {}
            }
        }
    });
}

impl dinet_node::transport::Transport for GossipBridge {
    /// Роутинг спула: to=None → тема all (и loopback себе; в mesh своя
    /// публикация попадает в инбокс — иначе маски не отменятся).
    fn send(&mut self, ch: u8, _from: u16, to: Option<u16>, frame: &[u8]) {
        // loopback: собственное окно всегда видит свою публикацию
        if let Some(f) = verify_frame(frame, self.peer) {
            self.queues[usize::from(ch)].lock().expect("queue").push(f);
        }
        self.send_frame(ch, to, frame.to_vec());
    }

    /// Отдать сырые wire-байты канала (фазы сами декодируют и проверяют
    /// подпись через PeerBook — мост уже отсёк мусор, но контракт честный).
    fn collect(&mut self, ch: u8, _me: u16) -> Vec<Vec<u8>> {
        self.queues[usize::from(ch)]
            .lock()
            .expect("queue")
            .drain(..)
            .map(|f| f.raw)
            .collect()
    }
}




/// Подписать и закодировать кадр для публикации (общая точка отправки).
pub fn signed_frame_bytes(
    key: &Ed25519Signer,
    mt: dinet_proto::MsgType,
    payload: dinet_proto::Payload,
) -> Result<Vec<u8>, String> {
    let frame = dinet_proto::Frame {
        msg_type: mt,
        sender: [0u8; 32],
        sig: [0u8; 64],
        payload,
    };
    let body = frame.encode_payload().map_err(|e| format!("payload: {e:?}"))?;
    let sig = key.sign(&body);
    frame
        .encode_with_sender_sig(&key.pubkey(), &sig)
        .map_err(|e| format!("encode: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// РЕАЛЬНЫЙ сетевой тест: 3 узла TCP+Noise+Yamux+gossipsub, полный меш,
    /// каждый получает КАЖДЫЙ подписанный кадр каждого и проверяет ed25519.
    /// Это то же поведение, что build-сервер делает для эпох — только адресация.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn p2p_full_mesh_frames_verified_end_to_end() {
        run_mesh_exchange(3).await.expect("полный меш без подделок");
    }

    /// Реальный РОЙ: 4 узла прогоняют 2 полные эры (masked + REVEAL) поверх
    /// TCP+Noise+gossipsub через GossipBridge (тот же трейт Transport!).
    /// Все реплики обязаны сойтись БИТ-В-БИТ с in-memory симуляцией.
    #[test]
    fn p2p_swarm_eras_over_real_network_match_mesh_simulation() {
        let mean = swarm_net::run_network_eras(4, 2, 2, false, 2026)
            .expect("роевые эры поверх TCP");
        assert!(mean > 0.3, "replica обязана учиться: mean {mean}");
    }
}

