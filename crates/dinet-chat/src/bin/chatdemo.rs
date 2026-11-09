//! chatdemo — REPL-чат v0: локальная память + RAG + буфер speak→learn.
//!
//! ```sh
//! cargo run -p dinet-chat --bin chatdemo -- ./dinet-chat-mem
//! ```
//!
//! Модель — toy-v0 (шаблонный ответчик, pipeline-проверка); llama.cpp
//! подключается за тем же трейтом ChatModel. Это честный скелет чат-UI:
//! вся механика (история, контекст, память, пары обучения) уже локальна
//! и работает.

use dinet_chat::{ChatModel, ChatSession, Role, ToyChatModel};
use std::io::{BufRead, Write};

const USAGE: &str = "команды: /help | /history | /stats | /ask <запрос> | /forget <id> | /compact | /exit (голые строки — ход чата)";

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "./dinet-chat-mem".to_string());
    let path = std::path::Path::new(&dir).join("session.dlog");
    let mut sess = match ChatSession::open(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("не смогли открыть память {path:?}: {e}");
            std::process::exit(1);
        }
    };
    println!("DI-NET чат: {path:?} ({} ходов, {} фактов)", sess.turns().len(), sess.memory_count());
    println!("{USAGE}");
    let mut model = ToyChatModel::default();
    let stdin = std::io::stdin();
    let mut clock: u64 = 1;
    loop {
        print!("вы> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/q" {
            break;
        } else if line == "/help" {
            println!("{USAGE}");
        } else if line == "/history" {
            for t in sess.turns() {
                let who = match t.role {
                    Role::User => "вы      ",
                    Role::Assistant => "бот     ",
                };
                println!("  {who} | {}", t.text);
            }
        } else if line == "/stats" {
            println!(
                "модель: {} | ходов: {} | фактов памяти: {} | пар обучения: {}",
                model.name(),
                sess.turns().len(),
                sess.memory_count(),
                sess.pending_pairs()
            );
        } else if let Some(q) = line.strip_prefix("/ask ") {
            for (id, text, score) in sess.ask_memory(q, 3) {
                println!("  #{:<8} {:.3}  {text}", id, score);
            }
        } else if let Some(id_s) = line.strip_prefix("/forget ") {
            match id_s.trim().parse::<u64>() {
                Ok(id) if sess.forget(id).unwrap_or(false) => println!("[#{id} забыт]"),
                Ok(_) => println!("[несуществующий id]"),
                Err(_) => println!("id — число"),
            }
        } else if line == "/compact" {
            sess.compact().ok();
            println!("[файл памяти уплотнен: {} ходов, {} фактов]", sess.turns().len(), sess.memory_count());
        } else {
            clock = clock.wrapping_add(1);
            let (answer, ctx) = match sess.turn(&mut model, line, clock) {
                Ok(x) => x,
                Err(e) => {
                    println!("[ошибка записи памяти: {e}]");
                    continue;
                }
            };
            println!("бот({})>{}", model.name(), answer);
            if ctx.is_empty() {
                println!("  (контекст: пусто — память накапливается)");
            } else {
                println!("  (контекст: {} факт(ов))", ctx.len());
            }
        }
    }
    println!("диск: {} ходов, {} пар для дообучения. до связи.",
        sess.turns().len(), sess.pending_pairs());
}
