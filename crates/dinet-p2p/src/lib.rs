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
    Some(InFrame { sender: f.sender, payload: f.payload, peer })
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
}

