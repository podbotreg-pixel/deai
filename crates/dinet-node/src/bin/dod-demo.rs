//! Живая проверка Definition of Done M4:
//!   1) сходимость при потере 30% узлов за окно;
//!   2) восстановление доли-модели из magnet при потере кусков у сидов.
//!
//! Запуск: cargo run -p dinet-node --bin dod-demo

use dinet_fl::toy_llm::LoRAAdapter;
use dinet_node::blob_torrent::{delta_f32_packed, BlobStore};
use dinet_node::dod::run_churn;
use dinet_node::magnet::Magnet;
use dinet_node::swarm::SimConfig;

fn main() {
    println!("DoD-1: сходимость при потере 30% узлов (20 → 14)");
    let rep = run_churn(20, 30, 4, 4, 5, SimConfig::default());
    for p in &rep.points {
        println!(
            "  эра {}: acc {:.3}, вклад {}, чек-пойнт {}",
            p.era, p.accuracy, p.contributors, p.checkpoint
        );
    }
    println!(
        "  финал: acc {:.3}, узлов {} → {}, чек-пойнты после выхода: {}",
        rep.last_accuracy(),
        rep.n_start,
        rep.n_end,
        rep.points[4..].iter().filter(|p| p.checkpoint).count()
    );

    println!("\nDoD-2: восстановление модели из magnet");
    let delta = LoRAAdapter::lora_init(42).as_vec();
    let (manifest, chunks) = delta_f32_packed(&delta, 64);
    let magnet = Magnet::new(7, [0xABu8; 32], &manifest);
    let mut a = BlobStore::default();
    let mut b = BlobStore::default();
    a.put(manifest.clone(), chunks.clone());
    b.put(manifest.clone(), chunks.clone());
    // каждый сид теряет 10 из 28 кусков (~36% > 30%), объединение — полное
    let lost = 10usize;
    for i in 0..lost {
        a.drop_chunk(&manifest.id(), i);
        b.drop_chunk(&manifest.id(), i + lost);
    }
    let restored = magnet.recover(&[&a, &b]).expect("сиды собрали модель");
    let original = delta_f32_packed_bytes(&delta);
    println!("  ссылка: {}", magnet.encode());
    println!(
        "  сид A: {}/{} кусков (-{}), сид B: {}/{} кусков (-{})",
        magnet.chunks as usize - lost,
        magnet.chunks,
        lost,
        magnet.chunks as usize - lost,
        magnet.chunks,
        lost
    );
    println!(
        "  восстановлено: {} байт, бит-в-бит с оригиналом: {}",
        restored.len(),
        restored == original
    );
    assert_eq!(restored, original);
}

/// Те же байты дельты, что упаковал `delta_f32_packed` (LE f32 подряд).
fn delta_f32_packed_bytes(delta: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(delta.len() * 4);
    for x in delta {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}
