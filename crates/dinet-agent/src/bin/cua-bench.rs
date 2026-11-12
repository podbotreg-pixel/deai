//! cua-bench — прогон CUA-Lite бенчмарка агента: отчёт по всем задачам.
//!
//! Запуск: cargo run -p dinet-agent --bin cua-bench
//! DoD: ≥20 открытых задач, succ ≥ 50% (v0-политика — точка прикрепления LLM).

fn main() {
    let rep = dinet_agent::cua::run_bench();
    println!(
        "CUA-Lite: {}/{} задач ({:.0}%)",
        rep.passed,
        rep.total,
        rep.succ() * 100.0
    );
    let mut shown = 0usize;
    for (id, ok, why) in &rep.results {
        if *ok {
            println!("[ok]   {id}");
        } else {
            println!("[FAIL] {id}: {why}");
            shown += 1;
        }
    }
    let _ = shown;
    if rep.passed < rep.total {
        std::process::exit(1);
    }
}
