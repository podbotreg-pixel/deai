//! p2pswarm — РОЙ ПОВЕРХ РЕАЛЬНОЙ СЕТИ: узлы на TCP+Noise+Yamux+gossipsub
//! прогоняют ПОЛНЫЕ фазы эр (publish → collect/vote → tally) через
//! `GossipBridge` — ТОТ ЖЕ трейт Transport, что у симуляции/спула.
//! Итог: реплики всех узлов согласованы и БИТ-В-БИТ равны in-memory
//! симуляции (та же программа через MeshTransport).

fn main() -> Result<(), String> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let eras: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    println!(
        "поднимаю {n} сетевых узлов (TCP+Noise+Yamux+gossipsub) на {eras} эр полного роя..."
    );
    let mean = dinet_p2p::swarm_net::run_network_eras(n, eras, 2, true, 2026)?;
    println!("P2P-SWARM OK: {n} узлов × {eras} эр поверх реальной сети;\nреплики сетевого роя БИТ-В-БИТ равны симуляции; mean acc {mean:.3}");
    Ok(())
}
