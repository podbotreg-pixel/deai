//! swarmkeeper — ЖИВОЙ СМОТРИТЕЛЬ роя (M3): N вечных узлов на реальном
//! TCP+Noise+Yamux+gossipsub; перед КАЖДОЙ эрой перечитывают состав
//! (Hello-волна), поэтому внешние клиенты (телефон) входят в любой
//! момент и участвуют наравне. Реестры клиентов — в каталоге рана.
//!
//! ```sh
//! cargo run -p dinet-p2p --bin swarmkeeper -- 3 9300 /tmp/opencode/skeeper
//! # телефон dial'ит 10.0.2.2:9300..9302 (эмулятор) или <LAN-IP>:9300..9302
//! ```

use std::time::Duration;

fn main() -> Result<(), String> {
    let n: u16 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let port0: u16 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(9300);
    let dir = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/tmp/opencode/skeeper".to_string());
    std::fs::create_dir_all(&dir).map_err(|e| format!("каталог {dir}: {e}"))?;
    let era_path = std::path::Path::new(&dir).join("era.txt");
    // счётчик эр минёра переживает рестарт смотрителя
    let era0 = std::fs::read(&era_path)
        .ok()
        .and_then(|b| b.chunks_exact(4).next().map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .unwrap_or(0);

    println!("поднимаю {n} смотрителей на портах {port0}..{} (эра старт {era0})...", port0 + n - 1);
    let mut bridges = Vec::new();
    for id in 0..n {
        let listen = format!("/ip4/0.0.0.0/tcp/{}", port0 + id);
        bridges.push(
            dinet_p2p::GossipBridge::spawn_on(id, [id as u8; 32], &listen)
                .map_err(|e| format!("узел {id}: {e}"))?,
        );
    }
    for i in 0..n {
        for j in 0..n {
            if i != j {
                // промах dial — не беда: mdns сам досвяжет соседей на хосте
                let _ = bridges[i as usize].dial_str(&format!("127.0.0.1:{}", port0 + j));
            }
        }
    }
    std::thread::sleep(Duration::from_millis(1600)); // graft mesh

    let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
    for id in 0..n {
        let bridge = bridges.remove(0);
        let reg = std::path::Path::new(&dir).join(format!("r{id}.dlog"));
        let era_p = era_path.clone();
        let tx = tx.clone();
        std::thread::Builder::new()
            .name(format!("keeper-{id}"))
            .spawn(move || {
                let out = dinet_p2p::swarm_net::keeper_worker(
                    id,
                    bridge,
                    &reg,
                    dinet_node::swarm::SimConfig::default(),
                    2026,
                    era0,
                    Some(era_p),
                    0,
                    true,
                );
                let _ = tx.send(out.map(|_| ()));
            })
            .map_err(|e| format!("поток узла {id}: {e}"))?;
    }
    println!(
        "смотрители живы; адреса для клиентов: 127.0.0.1:{port0}..{} (телефон-эмулятор: 10.0.2.2:{port0}..{})",
        port0 + n - 1,
        port0 + n - 1
    );
    // главный поток держит процесс (результаты смотрителей — только для
    // диагностики: живой смотритель не завершается, кроме падения моста)
    loop {
        match rx.recv_timeout(Duration::from_secs(3600)) {
            Ok(Ok(())) => println!("[смотритель] сессия завершена (лимит эр?)"),
            Ok(Err(e)) => println!("[смотритель] упал: {e}"),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}
