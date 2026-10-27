//! p2pmesh — демо реальной сети DI-NET: N узлов libp2p на localhost,
//! полный меш по dial, обмен ПОДПИСАННЫМИ wire-кадрами через gossipsub,
//! проверка подписи ed25519 на приёмнике. Тот же стек, что в бою
//! (TCP + Noise + Yamux + gossipsub): меняется только адресация.

#[tokio::main]
async fn main() -> Result<(), String> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    dinet_p2p::run_mesh_exchange(n).await?;
    println!("P2P-MESH OK: {n} узлов, полный меш, каждое сообщение прошло ed25519-проверку на приёмнике");
    Ok(())
}
