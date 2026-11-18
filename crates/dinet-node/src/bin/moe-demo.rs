//! Живая проверка M4 OPTIONAL: MoE-экспертная федерация — независимые
//! эксперты для конфликтующих правил задач + федерация по экспертам.
//! Запуск: cargo run -p dinet-node --bin moe-demo

use dinet_fl::toy_llm::{kd_fit, LoRAAdapter, Sample, ToyBase, ToyModel, V};
use dinet_node::moe::{aggregate_expert, moe_batch, MoeConfig, MoeModel};

fn acc(m: &ToyModel, samples: &[Sample]) -> f32 {
    let hits = samples.iter().filter(|s| m.predict(&s.tokens) == s.chosen).count();
    hits as f32 / samples.len() as f32
}

fn main() {
    let cfg = MoeConfig::default();
    let base = ToyBase::from_seed(7);
    let samples = moe_batch(600, 9090, cfg);
    let mut moe = MoeModel::new(base.clone(), cfg);
    for id in 0..cfg.n_experts {
        moe.train_expert(id, &samples);
    }
    // один общий адаптер на той же смеси задач (конфликтующие правила)
    let mut single = ToyModel { base: base.clone(), adapter: LoRAAdapter::lora_init(42) };
    single.adapter = kd_fit(
        &base,
        &single.adapter,
        &samples.iter().map(|s| s.tokens).collect::<Vec<_>>(),
        &samples
            .iter()
            .map(|s| {
                let mut t = vec![0.0f32; V];
                t[s.chosen as usize] = 4.0;
                t
            })
            .collect::<Vec<_>>(),
        cfg.steps,
        cfg.lr,
    );
    println!("MoE: {} экспертов, {} примеров", cfg.n_experts, samples.len());
    for e in &moe.experts {
        let mine: Vec<Sample> = samples
            .iter()
            .filter(|s| s.tokens[0] as usize % cfg.n_experts == e.id)
            .cloned()
            .collect();
        let a = acc(&ToyModel { base: base.clone(), adapter: e.adapter.clone() }, &mine);
        println!("  эксперт {}: acc {:.3} на своей задаче", e.id, a);
    }
    let m_acc = moe.accuracy(&samples);
    let s_acc = acc(&single, &samples);
    println!("MoE-смесь: {m_acc:.3}; один общий адаптер: {s_acc:.3}");
    let cp: String = moe.checkpoint_id().iter().map(|b| format!("{b:02x}")).collect();
    println!("чек-пойнт федерации: {cp}");

    // федерация эксперта 0: два клиента с разными шпарками
    let mut a = MoeModel::new(base.clone(), cfg);
    let mut b = MoeModel::new(base.clone(), cfg);
    a.train_expert(0, &samples[..300]);
    b.train_expert(0, &samples[300..]);
    let merged = aggregate_expert(&base, &[
        a.experts[0].adapter.diff_from(&LoRAAdapter::lora_init(100)),
        b.experts[0].adapter.diff_from(&LoRAAdapter::lora_init(100)),
    ])
    .expect("агрегатор");
    let task0: Vec<Sample> = samples
        .iter()
        .filter(|s| (s.tokens[0] as usize).is_multiple_of(cfg.n_experts))
        .cloned()
        .collect();
    println!(
        "федерация эксперта 0: клиент A {:.3}, B {:.3}, merged {:.3}",
        acc(&ToyModel { base: base.clone(), adapter: a.experts[0].adapter.clone() }, &task0),
        acc(&ToyModel { base: base.clone(), adapter: b.experts[0].adapter.clone() }, &task0),
        acc(&ToyModel { base, adapter: merged }, &task0),
    );
}
