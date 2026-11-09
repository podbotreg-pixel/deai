//! netswarm — живой опыт M2: клиент-рой ПОВЕРХ РЕАЛЬНОЙ СЕТИ.
//!
//! N узлов (каждый со своим мостом TCP+Noise+gossipsub в своей ните)
//! прогоняют эры: Hello-волна согласует состав, masked-эры крутят
//! ролевые фазы v1.3 (агрегатор + lazy-audit поверх TCP), reveal-эры
//! печатаются в реестр чек-пойнтов КАЖДОГО клиента (фаза F). Каждый
//! клиент учится на СВОИХ диалогах (свой батч за эру), вклады агрегируются
//! в чек-пойнты. Результат в консоли: отчёты эр, рост accuracy, равенство
//! реестров клиентов и реплик.
//!
//! ```sh
//! cargo run -p dinet-p2p --bin netswarm -- 3 4
//! ```

fn main() -> Result<(), String> {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let eras: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    println!("поднимаю {n} клиентов (TCP+Noise+Yamux+gossipsub) на {eras} эр...");
    let bm: dinet_p2p::swarm_net::BatchMaker = std::sync::Arc::new(|id, era| {
        dinet_fl::toy_llm::toy_world::honesty_batch(
            40,
            5_000 + id as u64 * 97 + era as u64 * 131 + 409,
        )
    });
    let run = dinet_p2p::swarm_net::run_network_client_eras(n, eras, 3, false, bm, true)?;
    println!();
    println!("--- отчёты узла 0 (у остальных то же — реплики сошлись) ---");
    for r in &run.reports[0] {
        let kind = if r.was_reveal { "REVEAL" } else { "masked" };
        let role = format!("{:?}", r.role);
        let agg = r
            .aggregator
            .map(|a| format!("{a}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "era {:>2} {:<6} contrib{:>2} acc {:.3}→{:.3} agg{:>2} роль {} аудит +{} -{}",
            r.era,
            kind,
            r.contributors,
            r.accuracy_before,
            r.accuracy_after,
            agg,
            role,
            r.audits_pass,
            r.audits_fail
        );
    }
    println!();
    let same = run
        .rosters
        .iter()
        .all(|r| *r == run.rosters[0])
        && run.ledgers.iter().all(|l| l == &run.ledgers[0]);
    let head = run.ledgers[0].len();
    println!(
        "ИТОГ: состав{:?} у всех; реестровых записей {head} у каждого клиента (бит-в-бит {});",
        run.rosters[0],
        if same { "равны" } else { "РАЗОШЛИСЬ" }
    );
    println!(
        "mean acc {:.3} → {:.3} — вклады клиентов агрегируются в чек-пойнты роя",
        run.mean_acc_start, run.mean_acc_end
    );
    Ok(())
}
