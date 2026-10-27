//! clientswarm — РОЙ ИЗ ЧАТ-КЛИЕНТОВ в разных ОС-процессах (файлы, без сокетов).
//!
//! ```sh
//! rm -rf /tmp/opencode/clients && \
//! cargo run -p dinet-client --bin clientswarm -- /tmp/opencode/clients \
//!     --pid 0 --procs 2 --clients 4 --eras 8 &
//! cargo run -p dinet-client --bin clientswarm -- /tmp/opencode/clients \
//!     --pid 1 --procs 2 --clients 4 --eras 8
//! ```
//!
//! Каждый процесс владеет своей частью клиентов; юзер каждого клиента
//! «разговаривает» микроязыком роя (`dinet_client::user_script`) и Каждая
//! эра пополняет его диалог-журнал. Обучение распределено: локальный fit
//! ИДЁТ ТОЛЬКО ПО ДИАЛОГАМ ЭТОГО КЛИЕНТА (`Node::batch_override`), дельта
//! подписывается и публикуется в эру; кворум применяет чек-пойнт.
//!
//! Ожидание: accuracy на общем eval растёт с эрами — вклады клиентов,
//! каждый учился на своём наборе разговоров, агрегируются в чек-пойнт.

use std::path::{Path, PathBuf};
use std::time::Duration;

use dinet_fl::toy_llm::toy_world::honesty_batch;
use dinet_fl::toy_llm::ToyModel;
use dinet_client::{pairs_to_batch, user_script};
use dinet_node::swarm::{
    accuracy, era_collect_and_vote, era_publish, era_tally, EraReport, Node, NodeId, SimConfig,
};
use dinet_node::transport::{FileSpool, Transport, CH_ANNOUNCE, CH_VOTES, CH_SHARES};

fn barrier(dir: &Path, era: u32, tag: &str, pid: usize, procs: usize) {
    let flag = dir.join(format!("bar-{era}-{tag}-{pid}"));
    let _ = std::fs::write(&flag, b"1");
    loop {
        let done = (0..procs).all(|p| dir.join(format!("bar-{era}-{tag}-{p}")).exists());
        if done {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn fmt_rep(pid: usize, id: NodeId, dialogs: usize, r: &EraReport) -> String {
    let kind = if r.was_reveal { "REVEAL" } else { "masked" };
    let ck = if r.checkpoint_id.is_some() { "Y" } else { "-" };
    format!(
        "proc{pid} кл{id:>2} era{:<2} {kind:<6} contrib{:>2} acc {:.3}→{:.3} диалог{} чк{}",
        r.era, r.contributors, r.accuracy_before, r.accuracy_after, dialogs, ck
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir_arg = args.next().unwrap_or_else(|| "/tmp/opencode/clients".to_string());
    let mut pid = 0usize;
    let mut procs = 1usize;
    let mut n_clients = 4usize;
    let mut eras = 8u32;
    let mut pairs_per_era = 8u32;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pid" => pid = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--procs" => procs = args.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "--clients" => n_clients = args.next().and_then(|v| v.parse().ok()).unwrap_or(4),
            "--eras" => eras = args.next().and_then(|v| v.parse().ok()).unwrap_or(8),
            "--pairs" => pairs_per_era = args.next().and_then(|v| v.parse().ok()).unwrap_or(8),
            _ => {}
        }
    }
    if pid >= procs {
        eprintln!("pid {pid} вне диапазона --procs {procs}");
        std::process::exit(1);
    }
    let dir = PathBuf::from(dir_arg);
    let cfg = SimConfig::default();
    let k = n_clients / procs;
    let start = pid * k;
    let mine_len = if pid + 1 == procs { n_clients - start } else { k };

    let all_ids: Vec<NodeId> = (0..n_clients as u16).collect();
    // ДИАЛОГИ каждого клиента — его собственный опыт, никто другой их не видит:
    // в эру уходит только DP-дельта локального fit.
    let mut dialogs: Vec<Vec<(String, String)>> = vec![Vec::new(); mine_len];
    let mut nodes: Vec<Node> = (start..start + mine_len)
        .map(|id| Node::new(id as u16, 7, false))
        .collect();
    let mut spool = match FileSpool::new(&dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("спул {dir:?} не открылся: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "[proc {pid}] клиент-рой: клиенты {start}..{} из {n_clients}, эр {eras}, пар/эру {pairs_per_era}",
        start + mine_len
    );
    for era in 0..eras {
        // --- данные клиента: новые диалоги юзера этого клиента
        for (ci, n) in nodes.iter_mut().enumerate() {
            let seed = (start + ci) as u64 * 97 + era as u64 * 131 + 409;
            dialogs[ci].extend(user_script(seed, pairs_per_era));
            n.batch_override = Some(pairs_to_batch(&dialogs[ci]));
        }
        // --- фазы роя (если же барьерировать; запускать процессы одновременно!)
        era_publish(&mut nodes, &all_ids, &mut spool, era, &cfg, 2026);
        barrier(&dir, era, "publish", pid, procs);
        let mids = era_collect_and_vote(&mut nodes, &all_ids, &mut spool, era, &cfg, None);
        barrier(&dir, era, "votes", pid, procs);
        let reports = era_tally(&mut nodes, &all_ids, &mut spool, era, &cfg, mids);
        barrier(&dir, era, "announce", pid, procs);
        for (ci, r) in reports.iter().enumerate() {
            println!("{}", fmt_rep(pid, (start + ci) as u16, dialogs[ci].len(), r));
        }
        // после announce-барьера все прочли всё — чистим спул
        barrier(&dir, era, "sweep", pid, procs);
        spool.sweep(CH_SHARES);
        spool.sweep(CH_VOTES);
        spool.sweep(CH_ANNOUNCE);
    }
    // --- итог: точность обученного на ОБЩЕМ eval
    let eval = honesty_batch(cfg.eval_size, 999_999);
    let accs: Vec<f32> = nodes
        .iter()
        .map(|n| accuracy(&ToyModel { base: n.base.clone(), adapter: n.adapter.clone() }, &eval))
        .collect();
    let mean = accs.iter().sum::<f32>() / accs.len().max(1) as f32;
    let total_dialogs: usize = dialogs.iter().map(|d| d.len()).sum();
    println!(
        "[proc {pid}] ФИНАЛ: mean accuracy {mean:.3} | диалогов у клиентов: {total_dialogs} (всё обучение — на диалогах клиентов)"
    );
}
