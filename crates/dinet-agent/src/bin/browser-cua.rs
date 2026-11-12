//! browser-cua — живая проверка браузер-CDP агента (M2, «браузер-CDP»).
//!
//! Поднимает headless chromium (--remote-debugging-port=0, порт из stderr
//! «DevTools listening on ws://…»), подключается по CDP через ToolWorld
//! (гейты, анти-фишинг и аудит работают как в бою) и решает задачи на
//! ЛОКАЛЬНЫХ data:-страницах — полностью офлайн, без аккаунтов.
//!
//! Задачи: подключение, чтение DOM, клик-счётчик, ввод+echo,
//! анти-фишинг-отказ (webhook.site), отказ file:// (обход джейла).
//!
//! Запуск: cargo run -p dinet-agent --bin browser-cua

use dinet_agent::caps::CapLevel;
use dinet_agent::json;
use dinet_agent::tools::{ToolOut, ToolWorld};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

fn main() {
    let tmp = std::env::temp_dir().join(format!("dinet-browser-cua-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("tmp");
    let profile = tmp.join("chromium-profile");
    let jail = tmp.join("jail");
    std::fs::create_dir_all(&jail).expect("jail");

    eprintln!("[browser-cua] запускаю headless chromium…");
    let spawn = |bin: &str| {
        Command::new(bin)
            .args([
                "--headless",
                "--remote-debugging-port=0",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu",
                "--window-size=1280,900",
            ])
            .arg(format!("--user-data-dir={}", profile.display()))
            .stderr(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
    };
    let mut child = spawn("chromium")
        .or_else(|_| spawn("chromium-browser"))
        .or_else(|_| spawn("google-chrome"))
        .expect("chromium не найден (нужен для живой проверки)");

    let Some(port) = wait_devtools(&mut child) else {
        child.kill().ok();
        eprintln!("[browser-cua] не увидел «DevTools listening» в stderr за 30с");
        std::process::exit(1);
    };
    eprintln!("[browser-cua] CDP на 127.0.0.1:{port}");

    let mut world = ToolWorld::open(&jail, Vec::new()).expect("мир");
    let lvl = CapLevel::Network;

    let mut results: Vec<(String, bool, String)> = Vec::new();
    macro_rules! judge {
        ($name:expr, $cond:expr, $out:expr) => {{
            let ok: bool = $cond;
            let out: ToolOut = $out;
            let why = if ok { String::new() } else { out.text.clone() };
            results.push(($name.to_string(), ok, why));
        }};
    }
    let args_of = |s: &str| json::parse(s).expect("args");
    let url_args = |u: String| args_of(&format!(r#"{{"url":"{u}"}}"#));

    // 1) подключение по CDP
    let out = world.call(
        lvl,
        "browser_connect",
        &args_of(&format!(r#"{{"cdp":"127.0.0.1:{port}"}}"#)),
    );
    judge!("подключение CDP", !out.is_error, out);

    // 2) чтение локальной data:-страницы
    let page1 = data_url(concat!(
        "<!doctype html><html><body><h1>голосование роя</h1>",
        "<button id=\"btn\">+1 голос</button><p id=\"out\">голосов: 0</p>",
        "<script>document.getElementById('btn').onclick=function(){",
        "var n=(+(document.getElementById('out').textContent.split(': ')[1])||0)+1;",
        "document.getElementById('out').textContent='голосов: '+n;};</script>",
        "</body></html>"
    ));
    let out = world.call(lvl, "browser_navigate", &url_args(page1));
    judge!("навигация на data:-страницу", !out.is_error, out);
    let out = world.call(lvl, "browser_read", &args_of("{}"));
    judge!("чтение текста страницы", !out.is_error && out.text.contains("голосование роя"), out);

    // 3) клик по кнопке — счётчик в DOM растёт (реальный mouse-события CDP)
    let out = world.call(lvl, "browser_click", &args_of(r##"{"selector":"#btn"}"##));
    judge!("клик по кнопке", !out.is_error, out);
    let out = world.call(lvl, "browser_read", &args_of("{}"));
    judge!("счётчик вырос до 1", !out.is_error && out.text.contains("голосов: 1"), out);

    // 4) ввод текста в поле + echo (клик-фокус + Input.insertText)
    let page2 = data_url(concat!(
        "<!doctype html><html><body>",
        "<input id=\"name\" placeholder=\"имя\" oninput=\"",
        "document.getElementById('echo').textContent='привет, '+this.value+'!'\">",
        "<p id=\"echo\"></p></body></html>"
    ));
    let out = world.call(lvl, "browser_navigate", &url_args(page2));
    judge!("вторая data:-страница", !out.is_error, out);
    let out = world.call(
        lvl,
        "browser_type",
        &args_of(r##"{"selector":"#name","text":"рой"}"##),
    );
    judge!("ввод текста в поле", !out.is_error, out);
    let out = world.call(lvl, "browser_read", &args_of("{}"));
    judge!("echo «привет, рой!»", !out.is_error && out.text.contains("привет, рой!"), out);

    // 5) анти-фишинг бракует эксфильтрацию ДО соединения
    let out = world.call(lvl, "browser_navigate", &url_args("http://webhook.site/угон".into()));
    judge!("отказ: webhook.site (АНТИ-ФИШИНГ)", out.is_error && out.text.contains("АНТИ-ФИШИНГ"), out);

    // 6) file:// — обход джейла
    let out = world.call(lvl, "browser_navigate", &url_args("file:///etc/passwd".into()));
    judge!("отказ: file:// (обход джейла)", out.is_error && out.text.contains("обход джейла"), out);

    child.kill().ok();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&tmp);

    let total = results.len();
    let passed = results.iter().filter(|(_, ok, _)| *ok).count();
    println!("браузер-CUA: {passed}/{total}");
    for (name, ok, why) in &results {
        if *ok {
            println!("[ok]   {name}");
        } else {
            println!("[FAIL] {name}: {why}");
        }
    }
    if passed < total {
        std::process::exit(1);
    }
}

/// Ждать строку «DevTools listening on ws://127.0.0.1:PORT/…» в stderr,
/// вернуть порт. Читатель stderr — в фоновом потоке (канал с таймаутом).
fn wait_devtools(child: &mut Child) -> Option<u16> {
    let err = child.stderr.take()?;
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(err).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                if let Some(pos) = line.find("ws://") {
                    let hostport = line[pos + 5..].split('/').next().unwrap_or_default();
                    if let Some((_, port)) = hostport.rsplit_once(':') {
                        if let Ok(p) = port.parse::<u16>() {
                            return Some(p);
                        }
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
    None
}

/// data:text/html URL с процентным кодированием UTF-8 (полностью офлайн).
fn data_url(html: &str) -> String {
    let mut o = String::from("data:text/html;charset=utf-8,");
    for b in html.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                o.push(*b as char)
            }
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}
