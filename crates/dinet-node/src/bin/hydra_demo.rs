//! Живая проверка M4 шага 2 (Optimal Hydra): три подсети со своими
//! базами/адаптерами + общий federation-чек-пойнт поверх carrier-логитов.
//! Запуск: cargo run -p dinet-node --bin hydra_demo

use dinet_node::hydra::{Hydra, SubnetSpec};
use dinet_node::swarm::SimConfig;

fn main() {
    let cfg = SimConfig::default();
    let specs = vec![
        SubnetSpec { id: 0, seed: 7, weight: 1.0, nodes: 3 },
        SubnetSpec { id: 1, seed: 11, weight: 1.0, nodes: 3 },
        SubnetSpec { id: 2, seed: 23, weight: 2.0, nodes: 3 },
    ];
    let mut h = Hydra::new(specs.clone(), cfg);
    println!("Optimal Hydra: подсети {:?}", specs.iter().map(|s| s.id).collect::<Vec<_>>());
    for era in 0..3u32 {
        let reports = h.run_era(2026);
        println!(
            "эра {}: acc подсетей {:?}",
            era,
            reports.iter().map(|r| format!("{:.3}", r.accuracy_after)).collect::<Vec<_>>()
        );
    }
    // подсети до федерации: адаптеры различны (свои миры/базы)
    let a0 = h.subnet_model(0).adapter.as_vec();
    let a1 = h.subnet_model(1).adapter.as_vec();
    let drift: f32 = a0.iter().zip(&a1).map(|(x, y)| (x - y).abs()).sum();
    println!("различие адаптеrov подсетей 0/1 (L1): {drift:.3}");
    let rep = h.federate();
    let cp: String = rep.checkpoint.iter().map(|b| format!("{b:02x}")).collect();
    println!("федерация: общий чек-пойнт {cp}");
    println!(
        "после федерации: согласие argmax {:.1}%, разброс логитов {:.3}, acc подсетей {:?}",
        rep.consensus_agree * 100.0,
        rep.consensus_err,
        rep.local_acc.iter().map(|a| format!("{a:.3}")).collect::<Vec<_>>()
    );
}
