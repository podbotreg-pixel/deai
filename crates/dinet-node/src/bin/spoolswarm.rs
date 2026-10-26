//! spoolswarm — РОЙ В РАЗНЫХ ПРОЦЕССАХ через FileSpool (без сокетов, без сервера).
//!
//! ```sh
//! cargo run -p dinet-node --bin spoolswarm -- /tmp/opencode/swarm --pid 0 --procs 2 --nodes 4 --eras 6 &
//! cargo run -p dinet-node --bin spoolswarm -- /tmp/opencode/swarm --pid 1 --procs 2 --nodes 4 --eras 6
//! ```
//!
//! Каждый процесс владеет своей ЧАСТЬЮ узлов роя (n/procs подряд идущих id).
//! Между процессами нет ни сокетов, ни общей памяти — только файлы в каталоге:
//! 1. вклады эпохи — подписанные wire-кадры dinet-proto (0x11/0x22/0x31/0x30);
//! 2. барьеры фаз `bar-{era}-{tag}-{pid}` — «все опубликовали → каждый собрал».
//!
//! Оба процесса детерминированы (fit по seed/node.id), поэтому чек-пойнты
//! обеих сторон обязаны совпасть — видно по выводу REVEAL-эр.

use std::path::PathBuf;
use std::time::Duration;

use dinet_fl::toy_llm::toy_world::honesty_batch;
use dinet_fl::toy_llm::ToyModel;
use dinet_node::swarm::{
    accuracy, era_collect_and_vote, era_publish, era_tally, EraReport, Node, NodeId, SimConfig,
};
use dinet_node::transport::{FileSpool, Transport, CH_ANNOUNCE, CH_VOTES, CH_SHARES};

fn barrier(dir: &std::path::Path, era: u32, tag: &str, pid: usize, procs: usize) {
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

fn fmt_rep(pid: usize, node: NodeId, r: &EraReport) -> String {
    let kind = if r.was_reveal { "REVEAL" } else { "masked" };
    let ck = if r.checkpoint_id.is_some() { "Y" } else { "-" };
    let drops = if r.dropped.is_empty() {
        String::new()
    } else {
        format!(" drops{:?}", r.dropped)
    };
    format!(
        "proc{pid} node{node:>2} era{:<2} {kind:<6} contrib{:>2} acc {:.3} ckpt{}{}",
        r.era, r.contributors, r.accuracy_after, ck, drops
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir_arg = args.next().unwrap_or_else(|| "/tmp/opencode/swarm".to_string());
    let mut pid = 0usize;
    let mut procs = 1usize;
    let mut n_nodes = 4usize;
    let mut eras = 6u32;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pid" => pid = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--procs" => procs = args.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "--nodes" => n_nodes = args.next().and_then(|v| v.parse().ok()).unwrap_or(4),
            "--eras" => eras = args.next().and_then(|v| v.parse().ok()).unwrap_or(6),
            _ => {}
        }
    }
    let dir = PathBuf::from(&dir_arg);
    let cfg = SimConfig::default();
    let k = n_nodes / procs;
    let mine_start = (pid * k) as u16;
    // хвост роя — последнему процессу
    let mine_len = if pid + 1 == procs {
        n_nodes - pid * k
    } else {
        k
    };
    let all_ids: Vec<NodeId> = (0..n_nodes as u16).collect();
    let n_byz = ((n_nodes as f32) * cfg.byzantine_pct) as usize;
    let mut mine: Vec<Node> = (mine_start..mine_start + mine_len as u16)
        .map(|id| Node::new(id, 7, (id as usize) < n_byz))
        .collect();
    let mut spool = match FileSpool::new(&dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("спул {dir:?} не открылся: {e}");
            std::process::exit(1);
        }
    };
    println!("[proc {pid}] рой: узлы {mine_start}..{} из {n_nodes}, эр {eras}",
        mine_start + mine_len as u16);
    let seed: u64 = 2026;
    for era in 0..eras {
        // ФАЗА A: публикация вкладов (кадры в спул-каталог)
        era_publish(&mut mine, &all_ids, &mut spool, era, &cfg, seed);
        barrier(&dir, era, "publish", pid, procs);
        // ФАЗЫ B+C: сбор кадров → скрининг → голос (кадры голосов в спул)
        let mids = era_collect_and_vote(&mut mine, &all_ids, &mut spool, era, &cfg, None);
        barrier(&dir, era, "votes", pid, procs);
        // ФАЗЫ D+E: кворум → apply → announce → отчёт (кадры объявлений в спул)
        let reports = era_tally(&mut mine, &all_ids, &mut spool, era, &cfg, mids);
        barrier(&dir, era, "announce", pid, procs);
        for (i, r) in reports.iter().enumerate() {
            println!("{}", fmt_rep(pid, mine_start + i as u16, r));
        }
        // после announce-барьера все прочли всё — можно чистить спул
        barrier(&dir, era, "sweep", pid, procs);
        spool.sweep(CH_SHARES);
        spool.sweep(CH_VOTES);
        spool.sweep(CH_ANNOUNCE);
    }
    // финал: точность реплик своего подмножества
    let eval = honesty_batch(cfg.eval_size, 999_999);
    let accs: Vec<f32> = mine
        .iter()
        .map(|n| accuracy(&ToyModel { base: n.base.clone(), adapter: n.adapter.clone() }, &eval))
        .collect();
    let mean = accs.iter().sum::<f32>() / accs.len().max(1) as f32;
    println!("[proc {pid}] ФИНАЛ: mean accuracy своих узлов {mean:.3} ({} эр)", eras);
}
