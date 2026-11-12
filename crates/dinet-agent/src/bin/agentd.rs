//! agentd — MCP-сервер DI-NET над stdio (NDJSON, спека 2025-06-18).
//!
//! Запуск: agentd <джейл> [--caps LEVEL] [--token ФАЙЛ] [--allow CMD]…
//! Права процесса = максимум из подписанного токена и --caps (по умолчанию
//! READ — минимально опасный старт). stdout = протокол; логи в stderr.
//! Реплике/LLM-хосту можно подсунуть этот процесс как MCP-сервер как есть.

use dinet_agent::{AllowRule, CapLevel, Server, ToolWorld};
use std::io::BufReader;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("запуск: agentd <джейл> [--caps LEVEL] [--token ФАЙЛ] [--allow CMD]…");
        std::process::exit(2);
    };
    let mut caps_level = CapLevel::Read;
    let mut token_path: Option<String> = None;
    let mut allow = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--caps" => {
                let v = args.next().unwrap_or_default();
                match CapLevel::parse(&v) {
                    Some(l) => caps_level = l,
                    None => {
                        eprintln!("[agentd] уровень {v:?} не READ/WRITE/DESTRUCTIVE/NETWORK");
                        std::process::exit(2);
                    }
                }
            }
            "--token" => token_path = Some(args.next().unwrap_or_default()),
            "--allow" => {
                let Some(c) = args.next() else { break };
                allow.push(AllowRule::new(&c));
            }
            other => {
                eprintln!("[agentd] неизвестный аргумент {other:?}");
                std::process::exit(2);
            }
        }
    }

    // Подписанный токен усиливает (или сужает) --caps: берём максимум.
    if let Some(tp) = &token_path {
        match std::fs::read_to_string(tp)
            .map_err(|e| e.to_string())
            .and_then(|t| dinet_agent::CapToken::decode(&t))
        {
            Ok(tok) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if tok.verify(now) {
                    println_stderr(&format!(
                        "[agentd] токен валиден: {} (scope {:?})",
                        tok.set.level.as_str(),
                        tok.set.scopes
                    ));
                    if tok.set.level > caps_level {
                        caps_level = tok.set.level;
                    }
                } else {
                    eprintln!("[agentd] токен НЕ прошёл проверку (подпись/срок) — игнорирую");
                }
            }
            Err(e) => eprintln!("[agentd] токен не разобран: {e}"),
        }
    }

    let world = match ToolWorld::open(&std::path::PathBuf::from(&dir), allow) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("[agentd] {e}");
            std::process::exit(1);
        }
    };
    println_stderr(&format!(
        "[agentd] джейл {:?}, права {} (allowlist {} прав.)",
        world.jail.root().display(),
        caps_level.as_str(),
        world.allow.len()
    ));
    println_stderr(match &world.model {
        Some(_) => "[agentd] модель AgentPhishing загружена (второе мнение)",
        None => "[agentd] модель AgentPhishing не найдена — только правила v1",
    });

    let mut server = Server::new(world, caps_level);
    let mut input = BufReader::new(std::io::stdin().lock());
    let mut out = std::io::stdout().lock();
    let n = dinet_agent::mcp::run_stdio(&mut server, &mut input, &mut out);
    println_stderr(&format!("[agentd] выход: обработано сообщений {n}"));
}

fn println_stderr(s: &str) {
    eprintln!("{s}");
}
