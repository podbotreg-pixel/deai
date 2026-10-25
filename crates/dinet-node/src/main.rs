fn main() {
    use dinet_fl::toy_llm::toy_world::honesty_batch;
    use dinet_fl::toy_llm::{LoRAAdapter, ToyModel};
    use dinet_node::swarm::{accuracy, simulate, SimConfig};

    println!("DI-NET swarm simulation demo");
    println!("============================");
    let cfg = SimConfig::default();
    let n = 20;
    let eras = 12;
    println!(
        "nodes: {n}, 10% отравителей, reveal-эра каждые {r} (eval фиксированный)",
        n = n,
        r = cfg.reveal_every
    );
    let fixed_eval = honesty_batch(96, 424242);
    let (reports, nodes) = simulate(n, eras, cfg, 2026);
    println!(
        "{:>4} {:>8} {:>7} {:>12} {:>10} {:>8}",
        "era", "type", "valid", "ckpt", "acc", "drops"
    );
    for r in &reports {
        let t = if r.was_reveal { "REVEAL" } else { "masked" };
        let cp = if r.checkpoint_id.is_some() { "✓" } else { "-" };
        let acc = format!("{:.3}", r.accuracy_after);
        println!(
            "{:>4} {:>8} {:>7} {:>12} {:>10} {:>8}",
            r.era, t, r.contributors, cp, acc, r.dropped.len()
        );
    }
    // финальное состояние сети: точность реплики на фиксированном eval
    let final_model = ToyModel { base: nodes[0].base.clone(), adapter: nodes[0].adapter.clone() };
    println!(
        "\nFINAL: точность реплики глобального адаптера на фиксованном eval: {:.3}",
        accuracy(&final_model, &fixed_eval)
    );
    let no_byz_cfg = SimConfig { byzantine_pct: 0.0, ..cfg };
    let (clean, clean_nodes) = simulate(n, eras, no_byz_cfg, 2029);
    let clean_model = ToyModel {
        base: clean_nodes[0].base.clone(),
        adapter: clean_nodes[0].adapter.clone(),
    };
    println!(
        "в чистой сети (без отравителей): {:.3}",
        accuracy(&clean_model, &fixed_eval)
    );
    let _ = clean.last();
    let _ = LoRAAdapter::num_params();
}
