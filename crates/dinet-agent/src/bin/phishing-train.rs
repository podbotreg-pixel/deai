//! phishing-train — обучение модели анти-фишинга (AgentPhishing v1) на
//! детерминированном корпусе и сохранение в файл, который agentd подхватит
//! (имя файла: agent-phishing.model, кладётся в каталог джейла).
//!
//! Запуск: cargo run -p dinet-agent --bin phishing-train -- <путь.model>
//! Политика: модель — «второе мнение» (может поднять до Suspicious);
//! блокируют по-прежнему правила v1.

use dinet_agent::phishing_model::{evaluate, generate, split_parity, train};
use std::path::PathBuf;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("запуск: phishing-train <путь>/agent-phishing.model");
            std::process::exit(2);
        }
    };
    let corpus = generate();
    let (tr, te) = split_parity(&corpus);
    println!(
        "корпус: benign {} / evil {} (train {}+{}, test {}+{})",
        corpus.benign.len(),
        corpus.evil.len(),
        tr.benign.len(),
        tr.evil.len(),
        te.benign.len(),
        te.evil.len(),
    );
    let m = train(&tr, 400, 0.5, 1e-4);
    let rep_tr = evaluate(&m, &tr.benign, &tr.evil);
    let rep_te = evaluate(&m, &te.benign, &te.evil);
    println!(
        "train: acc={:.3} evil[min {:.2}/mean {:.2}] benign[max {:.2}/mean {:.2}]",
        rep_tr.acc, rep_tr.min_evil_p, rep_tr.mean_evil_p, rep_tr.max_benign_p, rep_tr.mean_benign_p
    );
    println!(
        "test:  acc={:.3} evil[min {:.2}/mean {:.2}] benign[max {:.2}/mean {:.2}]",
        rep_te.acc, rep_te.min_evil_p, rep_te.mean_evil_p, rep_te.max_benign_p, rep_te.mean_benign_p
    );
    if let Err(e) = m.save(&path) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    println!("модель сохранена: {}", path.display());
}
