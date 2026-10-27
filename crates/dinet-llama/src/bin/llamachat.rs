//! llamachat — дымовой прогон настоящей нейросети llama.cpp (GGUF):
//! `cargo run -p dinet-llama --bin llamachat -- /путь/модель.gguf "вопрос"`

use dinet_chat::ChatModel;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path: PathBuf =
        args.next().ok_or("путь к .gguf не задан (например /tmp/opencode/gemma270m.gguf)")?.into();
    let question = args
        .next()
        .unwrap_or_else(|| "Скажи одно слово: привет".to_string());

    println!("llama.cpp загружает {model_path:?}...");
    let t0 = std::time::Instant::now();
    let mut model = dinet_llama::LlamaChatModel::load(
        &model_path,
        dinet_llama::LlamaOptions::default(),
    )?;
    println!("готово за {:.1}с (позиция чата: {})", t0.elapsed().as_secs_f32(), model.name());

    let t1 = std::time::Instant::now();
    let reply = model.respond(&question, &[]);
    println!("вопрос : {question}");
    println!("ответ  : {reply}");
    println!("([что] {:.2}с на ход)", t1.elapsed().as_secs_f32());
    Ok(())
}
