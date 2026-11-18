//! gui-spi — живая проверка AT-SPI слоя агента (M2, «AT-SPI: Linux GUI»).
//!
//! Осмотр слоя вручную: apps | tree | text | click | focus — соединение
//! с шиной доступности, обход дерева, текст узлов, действие/фокус.
//! Клик по ЧУЖИМ окнам — это реальное действие в GUI: печатаем цель.
//!
//! demo — самопроверка через ToolWorld (гейты, анти-фишинг и аудит как
//! в бою): поднимает СВОЁ gtk-окно (python3-gi), находит кнопку, кликает
//! через gui_click (WRITE) и проверяет, что счётчик в подписи вырос.
//!
//! Запуск: cargo run -p dinet-agent --bin gui-spi -- apps
//!         target/debug/gui-spi demo

use dinet_agent::atspi::{Atspi, Node};
use dinet_agent::caps::CapLevel;
use dinet_agent::tools::ToolWorld;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const USAGE: &str = "использование: gui-spi apps | tree <app> [depth] | text <app> | click <app> <id> [action] | focus <app> <id> | demo";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("");
    match cmd {
        "apps" => apps(),
        "tree" => {
            let app = args.get(1).cloned().unwrap_or_default();
            let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
            tree(&app, depth)
        }
        "text" => text(&args.get(1).cloned().unwrap_or_default()),
        "click" | "focus" => {
            let app = args.get(1).cloned().unwrap_or_default();
            let id: usize = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| die("нужен id из tree <app>"));
            let action: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            if cmd == "click" {
                click(&app, id, action)
            } else {
                focus(&app, id)
            }
        }
        "demo" => demo(),
        _ => die(USAGE),
    }
}

fn die(msg: &str) -> ! {
    eprintln!("gui-spi: {msg}\n{USAGE}");
    std::process::exit(2);
}

fn connect() -> Atspi {
    let addr = Atspi::resolve().unwrap_or_else(|e| die(&format!("шина не найдена: {e}")));
    Atspi::connect(&addr).unwrap_or_else(|e| die(&format!("соединение: {e}")))
}

fn find_app(a: &mut Atspi, name: &str) -> Node {
    let apps = a.apps().unwrap_or_else(|e| die(&format!("apps: {e}")));
    let target = name.to_lowercase();
    apps.into_iter()
        .find(|n| n.role == "application" && n.name.to_lowercase().contains(&target))
        .unwrap_or_else(|| die(&format!("приложение {name:?} не найдено (см. apps)")))
}

fn apps() {
    let mut a = connect();
    for (i, n) in a.apps().expect("apps").iter().enumerate() {
        println!("{i} {} {} {} {}", n.role, n.name, n.r.bus, n.r.path);
    }
}

fn tree(app: &str, depth: usize) {
    let mut a = connect();
    let node = find_app(&mut a, app);
    for (i, n) in a.tree(&node.r, depth).expect("tree").iter().enumerate() {
        println!("{i} {} {} {} {}", n.role, n.name, n.r.bus, n.r.path);
    }
}

fn text(app: &str) {
    let mut a = connect();
    let node = find_app(&mut a, app);
    let nodes = a.tree(&node.r, 6).expect("tree");
    for (i, n) in nodes.iter().enumerate() {
        // текст есть только у Text-узлов; у прочих GetText — честная ошибка
        if let Ok(t) = a.text(&n.r) {
            if !t.is_empty() {
                let head: String = t.chars().take(300).collect();
                println!("{i} {} «{}» → {head}", n.role, n.name);
            }
        }
    }
}

fn click(app: &str, id: usize, action: usize) {
    let mut a = connect();
    let node = find_app(&mut a, app);
    let nodes = a.tree(&node.r, 6).expect("tree");
    let n = nodes.get(id).unwrap_or_else(|| die("нет такого id"));
    println!("клик (действие {action}): {} «{}» {} {}", n.role, n.name, n.r.bus, n.r.path);
    match a.do_action(&n.r, action) {
        Ok(true) => println!("ok: действие выполнено"),
        Ok(false) => println!("нет: действие отклонено (индекс вне диапазона?)"),
        Err(e) => die(&format!("действие: {e}")),
    }
}

fn focus(app: &str, id: usize) {
    let mut a = connect();
    let node = find_app(&mut a, app);
    let nodes = a.tree(&node.r, 6).expect("tree");
    let n = nodes.get(id).unwrap_or_else(|| die("нет такого id"));
    println!("фокус: {} «{}» {} {}", n.role, n.name, n.r.bus, n.r.path);
    match a.grab_focus(&n.r) {
        Ok(true) => println!("ok: фокус взят"),
        Ok(false) => println!("нет: фокус не взят"),
        Err(e) => die(&format!("фокус: {e}")),
    }
}

// ----------
// demo: своё окно + полный боевой путь ToolWorld
// ----------

/// GTK3-окно с кнопкой и счётчиком; живёт 90с либо до закрытия.
const DEMO_PY: &str = r#"import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

GLib.set_prgname("dinet-demo")
GLib.set_application_name("dinet-demo")

class W(Gtk.Window):
    def __init__(self):
        super().__init__(title="dinet-demo")
        box = Gtk.Box(spacing=6)
        self.add(box)
        self.n = 0
        self.btn = Gtk.Button(label="Голос")
        self.btn.connect("clicked", self.on_click)
        self.lbl = Gtk.Label(label="голосов: 0")
        box.pack_start(self.btn, True, True, 0)
        box.pack_start(self.lbl, True, True, 0)
        self.show_all()
    def on_click(self, _):
        self.n += 1
        self.lbl.set_text("голосов: %d" % self.n)

w = W()
w.connect("delete-event", Gtk.main_quit)
GLib.timeout_add_seconds(90, Gtk.main_quit)
Gtk.main()
"#;

struct Judge {
    results: Vec<(String, bool, String)>,
}

impl Judge {
    fn judge(&mut self, name: &str, ok: bool, why: &str) {
        self.results.push((name.to_string(), ok, why.to_string()));
    }
    fn finish(self) -> ! {
        let total = self.results.len();
        let passed = self.results.iter().filter(|(_, ok, _)| *ok).count();
        println!("gui-demo: {passed}/{total}");
        for (name, ok, why) in &self.results {
            if *ok {
                println!("[ok]   {name}");
            } else {
                println!("[FAIL] {name}: {why}");
            }
        }
        std::process::exit(if passed == total { 0 } else { 1 });
    }
}

fn args_of(s: &str) -> dinet_agent::json::Json {
    dinet_agent::json::parse(s).expect("args")
}

/// id первой строки listing'а, где роль содержит role_sub, имя — name_sub.
fn find_id(listing: &str, role_sub: &str, name_sub: &str) -> Option<usize> {
    for line in listing.lines() {
        if let Some((id, rest)) = line.split_once(' ') {
            if id.parse::<usize>().is_ok() && rest.contains(role_sub) && rest.contains(name_sub) {
                return id.parse().ok();
            }
        }
    }
    None
}

fn demo() -> ! {
    // окно питонится рядом; потом гасим
    let py = std::env::temp_dir().join("dinet-gui-demo.py");
    std::fs::write(&py, DEMO_PY).expect("py");
    let mut child: Child = Command::new("python3")
        .arg("-u")
        .arg(&py)
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| die(&format!("python3 не запустился ({e}) — нужно python3-gi (GTK3)")));

    let mut j = Judge { results: Vec::new() };
    let mut world = ToolWorld::open(&demo_jail(), Vec::new()).expect("мир");
    // WRITE покрывает Read (gui_connect/gui_nodes/gui_text) и WRITE (клик/фокус)
    let lvl = CapLevel::Write;

    // 1) подключение к GUI через ToolWorld
    let out = world.call(lvl, "gui_connect", &args_of("{}"));
    j.judge("подключение AT-SPI", !out.is_error, &out.text);

    // 2) своё окно появилось в дереве доступности (ожидание до 10с)
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut list_app = String::new();
    while Instant::now() < deadline {
        world.call(lvl, "gui_nodes", &args_of("{}")); // кэш приложений
        let out = world.call(lvl, "gui_nodes", &args_of(r#"{"app":"dinet-demo"}"#));
        if !out.is_error {
            list_app = out.text.clone();
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    j.judge("своё окно в дереве", !list_app.is_empty(), "окно не появилось за 10с");

    // 3) клик по кнопке через gui_click (WRITE) — не по чужим окнам
    // (роль у GTK — «button», у других тулкитов — «push button»)
    let mut clicked = false;
    let mut why_click = String::from("кнопка «Голос» не найдена");
    if let Some(id) = find_id(&list_app, "button", "Голос") {
        let out = world.call(lvl, "gui_click", &args_of(&format!(r#"{{"id":{id}}}"#)));
        clicked = !out.is_error;
        why_click = out.text.clone();
    }
    j.judge("клик по своей кнопке", clicked, &why_click);

    std::thread::sleep(Duration::from_millis(400)); // a11y-события асинхронны

    // 4) счётчик в подписи вырос до 1: текст label-узла
    world.call(lvl, "gui_nodes", &args_of("{}"));
    let out = world.call(
        lvl,
        "gui_nodes",
        &args_of(r#"{"app":"dinet-demo","depth":5}"#),
    );
    let list2 = out.text.clone();
    let mut grew = list2.contains("голосов: 1");
    let mut why_grew = format!("в дереве нет «голосов: 1»\n{list2}");
    if let Some(id) = find_id(&list2, "label", "голосов") {
        let t = world.call(lvl, "gui_text", &args_of(&format!(r#"{{"id":{id}}}"#)));
        if !t.is_error && t.text.contains("голосов: 1") {
            grew = true;
            why_grew = t.text.clone();
        }
    }
    j.judge("счётчик вырос до 1 (клик сработал)", grew, &why_grew);

    // 5) фокус на своём узле (WRITE)
    let mut focused = false;
    let mut why_focus = String::from("ни кнопка, ни окно не взяли фокус");
    for id in [
        find_id(&list_app, "button", "Голос"),
        find_id(&list_app, "frame", "dinet-demo"),
    ].into_iter().flatten() {
        let out = world.call(lvl, "gui_focus", &args_of(&format!(r#"{{"id":{id}}}"#)));
        if !out.is_error {
            focused = true;
            why_focus = out.text.clone();
            break;
        }
    }
    j.judge("фокус на своём узле", focused, &why_focus);

    // аудит помнит всё (DoD «всё в журнале»)
    let tail = world.call(CapLevel::Read, "audit_tail", &args_of(r#"{"n":10}"#));
    j.judge(
        "аудит: клик и фокус в журнале",
        !tail.is_error && tail.text.contains("gui_click") && tail.text.contains("gui_focus"),
        &tail.text,
    );

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_file(&py);
    j.finish()
}

fn demo_jail() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("dinet-gui-demo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("jail");
    dir
}
