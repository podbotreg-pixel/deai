//! memdemo — интерактивное демо локальной памяти DI-NET.
//!
//! ```sh
//! cargo run -p dinet-memory --bin memdemo -- ./dinet-mem
//! ```
//!
//! Команды:
//!   /remember <текст>   — запомнить факт (RAG-документ), вернуть id
//!   /ask <вопрос>       — top-3 ближайших факта с cosine-скором
//!   /forget <id>        — удалить факт
//!   /stats              — сколько фактов на диске
//!   /exit               — выход (все записи уже на диске)

use dinet_memory::kv::KvStore;
use dinet_memory::rag::{doc_count, remove_doc, search_store, store_doc};
use std::io::{BufRead, Write};

const USAGE: &str = "команды: /remember <текст> | /ask <вопрос> | /forget <id> | /stats | /exit";

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "./dinet-mem".to_string());
    let path = std::path::Path::new(&dir).join("mem.dlog");
    let mut store = match KvStore::open(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("не смогли открыть память {path:?}: {e}");
            std::process::exit(1);
        }
    };
    println!("DI-NET память: {path:?} ({} фактов)", doc_count(&store));
    println!("{USAGE}");
    let stdin = std::io::stdin();
    loop {
        print!("mem> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(text) = line.strip_prefix("/remember ").or(line.strip_prefix("/r ")) {
            let id = store_doc(&mut store, text, 0).unwrap_or(u64::MAX);
            println!("[запомнено #{id}]");
        } else if let Some(q) = line.strip_prefix("/ask ").or(line.strip_prefix("/a ")) {
            let hits = search_store(&store, q, 3);
            if hits.is_empty() {
                println!("(память пуста — /remember чтобы добавить)");
            }
            for (doc, score) in hits {
                println!("  #{:<8} {:.3}  {}", doc.id, score, truncate(&doc.text, 96));
            }
        } else if let Some(id_s) = line.strip_prefix("/forget ").or(line.strip_prefix("/f ")) {
            match id_s.trim().parse::<u64>() {
                Ok(id) => {
                    if remove_doc(&mut store, id).unwrap_or(false) {
                        println!("[удалено #{id}]");
                    } else {
                        println!("[#{id} нет]");
                    }
                }
                Err(_) => println!("id — число"),
            }
        } else if line == "/stats" || line == "/s" {
            println!("фактов: {}", doc_count(&store));
        } else if line == "/exit" || line == "/q" || line == "exit" {
            break;
        } else if line.starts_with('/') {
            println!("{USAGE}");
        } else {
            // голые строки — тоже запоминаем (поток разговора)
            let id = store_doc(&mut store, line, 0).unwrap_or(u64::MAX);
            println!("[запомнено #{id}] (команды: {USAGE})");
        }
    }
    println!("всё записано на диск — до связи.");
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
