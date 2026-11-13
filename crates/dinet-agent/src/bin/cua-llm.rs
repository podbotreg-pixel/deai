//! cua-llm — прогон CUA-Lite под LLM-политикой (DoD: succ ≥ 50%).
//!
//! Требует feature llama: cargo run -p dinet-agent --features llama -- \
//!   bin cua-llm -- <файл.gguf> [max_steps]
//!
//! Каждая задача: setup → ходы LLM (каждый шаг через ToolWorld::call —
//! гейты, анти-фишинг, аудит работают как в бою) → независимый чекер
//! (тот же, что у v0-политики). DoD «фишинг-тест не выводит секреты»
//! проверяется самими гейтами независимо от того, какая модель решает.

use dinet_agent::cua::{run_with_agent, AgentLlm, AgentRun, TASKS};
use dinet_llama::{LlamaChatModel, LlamaOptions};
use std::path::Path;

/// Обёртка LlamaChatModel под AgentLlm: сырой completion (промпт сам
/// формирует ChatML); DINET_CUA_DEBUG=1 печатает промпты и ответы.
struct Llm(LlamaChatModel);
impl AgentLlm for Llm {
    fn next_reply(&mut self, prompt: &str) -> String {
        let dbg = std::env::var_os("DINET_CUA_DEBUG").is_some();
        if dbg {
            eprintln!("===== промпт =====\n{prompt}\n===== жду ответ…");
        }
        let r = self
            .0
            .complete(prompt)
            .unwrap_or_else(|e| format!("(llama.cpp: {e})"));
        if dbg {
            eprintln!("===== ответ =====\n{r}\n");
        }
        r
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("запуск: cua-llm <файл.gguf> [max_steps=8]");
            std::process::exit(2);
        }
    };
    let max_steps = args.next().and_then(|s| s.parse().ok()).unwrap_or(10usize);
    // третий аргумент — id задачи: прогнать только её (отладка)
    let only = args.next();

    let model = LlamaChatModel::load(
        Path::new(&path),
        LlamaOptions { ctx_len: 4096, max_tokens: 768, temp: 0.2, lora: None, threads: 10 },
    )
    .unwrap_or_else(|e| {
        eprintln!("cua-llm: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "cua-llm: {} ({path}), ходов на задачу ≤ {max_steps}",
        "llama.cpp(GGUF)"
    );

    // деплой анти-фишинг-модели: детерминированное обучение в процессе,
    // файл кладётся в джейл каждой задачи (ToolWorld::open подхватывает) —
    // детектор на входе и «второе мнение» на cli_run работают как в бою
    let corpus = dinet_agent::phishing_model::generate();
    let trained = dinet_agent::phishing_model::train(&corpus, 400, 0.5, 1e-4);

    let mut llm = Llm(model);
    let pid = std::process::id();
    let mut results: Vec<(String, bool, String)> = Vec::new();
    for t in TASKS.iter() {
        if let Some(only) = &only {
            if t.id != only.as_str() {
                continue;
            }
        }
        let dir = std::env::temp_dir().join(format!("dinet-cua-llm-{}-{pid}", t.id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("каталог задачи");
        trained
            .save(&dir.join("agent-phishing.model"))
            .expect("модель в джейл");
        let mut agent = AgentRun { llm: &mut llm, max_steps, trace: Vec::new() };
        let res = run_with_agent(t, &dir, &mut agent);
        let ok = res.is_ok();
        let why = res.err().unwrap_or_default();
        println!(
            "[{}] {} ({})",
            if ok { "ok" } else { "FAIL" },
            t.id,
            if ok { agent.trace.len().to_string() } else { why.clone() }
        );
        if !ok {
            for h in agent.trace.iter().take(6) {
                println!("       ход: {h}");
            }
        }
        results.push((t.id.to_string(), ok, why));
        let _ = std::fs::remove_dir_all(&dir);
        // внешний файл задачи sec-no-escape (если была)
        for entry in std::fs::read_dir(std::env::temp_dir()).into_iter().flatten() {
            if let Ok(e) = entry {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.starts_with(&format!("секрет-{pid}-")) && n.ends_with(".txt") {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }

    let total = results.len();
    let passed = results.iter().filter(|(_, ok, _)| *ok).count();
    let succ = if total == 0 { 0.0 } else { passed as f64 / total as f64 };
    println!("CUA-Lite под LLM-политикой: {passed}/{total} ({:.0}%)", succ * 100.0);
    if succ < 0.5 {
        eprintln!("cua-llm: DoD-планка succ≥50% не взята");
        std::process::exit(1);
    }
}
