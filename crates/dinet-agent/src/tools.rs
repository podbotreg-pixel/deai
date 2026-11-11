//! Инструменты агента (M2): fs_read/fs_list/fs_write/fs_delete поверх
//! FS-песочницы и cli_run поверх allowlist. Каждый инструмент требует
//! уровень capability (Read/Write/Destructive/Network) и scope; мутирующие
//! пишут инверсию в undo-журнал ДО действия.

use crate::caps::CapLevel;
use crate::jail::Jail;
use crate::journal::{Inverse, Journal};
use std::path::PathBuf;

/// Какие права нужны инструменту и в каком scope он работает.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Need {
    pub level: CapLevel,
}

/// Именованные инструменты MCP-поверхности.
pub const TOOLS: &[(&str, Need, &str)] = &[
    (
        "fs_read",
        Need { level: CapLevel::Read },
        "Прочитать файл в песочнице. Аргументы: path (относительный).",
    ),
    (
        "fs_list",
        Need { level: CapLevel::Read },
        "Список каталога песочницы. Аргументы: path ('' = корень джейла).",
    ),
    (
        "fs_write",
        Need { level: CapLevel::Write },
        "Записать файл (каталоги создаются). Аргументы: path, content. Обратимо: undo.",
    ),
    (
        "fs_delete",
        Need { level: CapLevel::Destructive },
        "Удалить файл. Аргументы: path. Обратимо: содержимое уходит в undo-журнал.",
    ),
    (
        "undo",
        Need { level: CapLevel::Write },
        "Откатить последнее мутирующее действие (fs_write/fs_delete) из undo-журнала.",
    ),
    (
        "cli_run",
        Need { level: CapLevel::Network },
        "Запустить разрешённую команду из allowlist. Аргументы: command, args[].",
    ),
];

pub fn need(tool: &str) -> Option<Need> {
    TOOLS.iter().find(|(n, _, _)| *n == tool).map(|(_, nd, _)| *nd)
}

/// Итог действия: текст для content + была ли ошибка уровня инструмента.
#[derive(Debug, PartialEq)]
pub struct ToolOut {
    pub text: String,
    pub is_error: bool,
}

impl ToolOut {
    fn ok(text: String) -> ToolOut {
        ToolOut { text, is_error: false }
    }
    fn err(text: String) -> ToolOut {
        ToolOut { text, is_error: true }
    }
}

/// Мир инструментов: джейл + allowlist + undo-журнал.
pub struct ToolWorld {
    pub jail: Jail,
    pub allow: Vec<AllowRule>,
    pub journal: Journal,
    /// Лимит вывода cli (байт) — журнал/чат не рвём.
    pub cli_out_limit: usize,
}

/// Правило allowlist: точное имя или префикс каталога.
#[derive(Debug, Clone, PartialEq)]
pub struct AllowRule {
    /// Путь без аргументов: "ls", "/usr/bin/python3".
    pub cmd: String,
    /// Точный match или префикс пути (компонентный).
    pub exact: bool,
}

impl AllowRule {
    pub fn new(cmd: &str) -> AllowRule {
        AllowRule { cmd: cmd.into(), exact: true }
    }
}

impl ToolWorld {
    pub fn open(dir: &PathBuf, allow: Vec<AllowRule>) -> Result<ToolWorld, String> {
        let jail = Jail::open(dir)?;
        let journal = Journal::open(&dir.join("agent-undo.roi"))?;
        Ok(ToolWorld { jail, allow, journal, cli_out_limit: 64 * 1024 })
    }

    /// ---------- fs_read ----------
    pub fn fs_read(&self, rel: &str) -> ToolOut {
        let path = match self.jail.resolve(rel) {
            Ok(p) => p,
            Err(e) => return ToolOut::err(e),
        };
        match std::fs::read(&path) {
            Ok(bytes) => ToolOut::ok(String::from_utf8_lossy(&bytes).into_owned()),
            Err(e) => ToolOut::err(format!("{rel:?}: {e}")),
        }
    }

    /// ---------- fs_list ----------
    pub fn fs_list(&self, rel: &str) -> ToolOut {
        let path = if rel.is_empty() {
            self.jail.root().to_path_buf()
        } else {
            match self.jail.resolve(rel) {
                Ok(p) => p,
                Err(e) => return ToolOut::err(e),
            }
        };
        let mut names: Vec<String> = match std::fs::read_dir(&path) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .map(|e| {
                    let t = if e.path().is_dir() { "d" } else { "f" };
                    format!("{t} {}", e.file_name().to_string_lossy())
                })
                .collect(),
            Err(e) => return ToolOut::err(format!("{rel:?}: {e}")),
        };
        names.sort();
        ToolOut::ok(if names.is_empty() {
            "(пусто)".into()
        } else {
            names.join("\n")
        })
    }

    /// ---------- fs_write (обратимо) ----------
    pub fn fs_write(&mut self, rel: &str, content: &[u8]) -> ToolOut {
        // 1) путь валиден
        if let Err(e) = self.jail.resolve_chain(rel) {
            return ToolOut::err(e);
        }
        let path = match self.jail.resolve(rel) {
            Ok(p) => p,
            Err(e) => return ToolOut::err(e),
        };
        // 2) инверсия ДО действия (было содержимое или файла не было)
        let was = std::fs::read(&path).ok();
        let inv = Inverse::Overwrite { rel: rel.to_string(), was };
        if let Err(e) = self.journal.push(inv) {
            return ToolOut::err(format!("undo-журнал не принял действие: {e}"));
        }
        // 3) действие: mkdir -p + write
        let parent = path.parent().map(|p| p.to_path_buf());
        if let Some(p) = parent {
            if let Err(e) = std::fs::create_dir_all(&p) {
                return ToolOut::err(format!("mkdir {p:?}: {e}"));
            }
        }
        match std::fs::write(&path, content) {
            Ok(()) => ToolOut::ok(format!("записано {} байт в {rel}", content.len())),
            Err(e) => ToolOut::err(format!("{rel:?}: {e}")),
        }
    }

    /// ---------- fs_delete (обратимо) ----------
    pub fn fs_delete(&mut self, rel: &str) -> ToolOut {
        let path = match self.jail.resolve(rel) {
            Ok(p) => p,
            Err(e) => return ToolOut::err(e),
        };
        let was = match std::fs::read(&path) {
            Ok(b) => Some(b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return ToolOut::err(format!("{rel}: файла нет"))
            }
            Err(e) => return ToolOut::err(format!("{rel:?}: {e}")),
        };
        let inv = Inverse::Overwrite { rel: rel.to_string(), was };
        if let Err(e) = self.journal.push(inv) {
            return ToolOut::err(format!("undo-журнал не принял действие: {e}"));
        }
        match std::fs::remove_file(&path) {
            Ok(()) => ToolOut::ok(format!("{rel} удалён (undo вернёт содержимое)")),
            Err(e) => ToolOut::err(format!("{rel:?}: {e}")),
        }
    }

    /// ---------- undo последнего мутирующего действия ----------
    pub fn undo(&mut self) -> ToolOut {
        let Some(inv) = self.journal.pop() else {
            return ToolOut::err("undo-журнал пуст".into());
        };
        match inv {
            Inverse::Overwrite { rel, was } => {
                let path = match self.jail.resolve(&rel) {
                    Ok(p) => p,
                    Err(e) => return ToolOut::err(format!("undo: путь {rel:?} больше не в джейле: {e}")),
                };
                let res = match was {
                    Some(b) => std::fs::write(&path, &b).map(|_| format!("{rel}: восстановлено {} байт", b.len())),
                    None => std::fs::remove_file(&path).map(|_| format!("{rel}: удалено (до записи файла не было)")),
                };
                match res {
                    Ok(msg) => ToolOut::ok(msg),
                    Err(e) => ToolOut::err(format!("undo {rel:?}: {e}")),
                }
            }
        }
    }

    /// ---------- cli_run (allowlist + NETWORK) ----------
    pub fn cli_run(&self, command: &str, args: &[String]) -> ToolOut {
        let permitted = self.allow.iter().any(|r| {
            r.exact && (r.cmd == command || r.cmd.ends_with(&format!("/{command}"))
                && std::path::Path::new(&r.cmd).is_absolute() == std::path::Path::new(command).is_absolute())
        });
        if !permitted {
            return ToolOut::err(format!(
                "команда {command:?} не в allowlist ({} правил)",
                self.allow.len()
            ));
        }
        // юникс: exec через sh -c запрещён (инъекция); spawn напрямую
        let out = std::process::Command::new(command)
            .args(args)
            .current_dir(self.jail.root())
            .stdin(std::process::Stdio::null())
            .output();
        match out {
            Ok(o) => {
                let mut text = String::new();
                text.push_str(&format!("exit={}", o.status.code().unwrap_or(-1)));
                let so = clip(&String::from_utf8_lossy(&o.stdout), self.cli_out_limit);
                let se = clip(&String::from_utf8_lossy(&o.stderr), self.cli_out_limit);
                if !so.is_empty() {
                    text.push_str("\n--- stdout ---\n");
                    text.push_str(&so);
                }
                if !se.is_empty() {
                    text.push_str("\n--- stderr ---\n");
                    text.push_str(&se);
                }
                ToolOut::ok(text)
            }
            Err(e) => ToolOut::err(format!("{command:?}: {e}")),
        }
    }

    /// Хэндлер всех инструментов с гейтом уровня.
    pub fn call(&mut self, granted: CapLevel, name: &str, args: &crate::json::Json) -> ToolOut {
        let Some(nd) = need(name) else {
            return ToolOut::err(format!("неизвестный инструмент {name:?}"));
        };
        if granted < nd.level {
            return ToolOut::err(format!(
                "инструмент {name} требует {}, у агента {}",
                nd.level.as_str(),
                granted.as_str()
            ));
        }
        match name {
            "fs_read" => {
                let p = arg_str(args, "path");
                match p {
                    Some(p) => self.fs_read(&p),
                    None => ToolOut::err("нужен path".into()),
                }
            }
            "fs_list" => {
                let p = arg_str(args, "path").unwrap_or_default();
                self.fs_list(&p)
            }
            "fs_write" => {
                let (Some(p), Some(c)) = (arg_str(args, "path"), arg_str(args, "content")) else {
                    return ToolOut::err("нужны path и content".into());
                };
                self.fs_write(&p, c.as_bytes())
            }
            "fs_delete" => match arg_str(args, "path") {
                Some(p) => self.fs_delete(&p),
                None => ToolOut::err("нужен path".into()),
            },
            "undo" => self.undo(),
            "cli_run" => {
                let cmd = match arg_str(args, "command") {
                    Some(c) => c,
                    None => return ToolOut::err("нужен command".into()),
                };
                let args_v: Vec<String> = args
                    .get("args")
                    .and_then(|j| match j {
                        crate::json::Json::Arr(v) => Some(
                            v.iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect(),
                        ),
                        _ => None,
                    })
                    .unwrap_or_default();
                self.cli_run(&cmd, &args_v)
            }
            _ => ToolOut::err(format!("неизвестный инструмент {name:?}")),
        }
    }
}

fn clip(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        s.to_string()
    } else {
        let mut cut = limit;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}…[обрезано]", &s[..cut])
    }
}

fn arg_str(args: &crate::json::Json, key: &str) -> Option<String> {
    args.get(key).and_then(|j| j.as_str()).map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{CapLevel, CapSet, CapToken};
    use crate::json::parse;
    use dinet_crypto::Ed25519Signer;

    fn tmp_world(tag: &str) -> ToolWorld {
        let dir = std::env::temp_dir().join(format!("dinet-tool-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        ToolWorld::open(&dir, vec![AllowRule::new("ls"), AllowRule::new("echo")])
            .expect("мир")
    }

    fn args_of(src: &str) -> crate::json::Json {
        parse(src).expect("args json")
    }

    #[test]
    fn gating_by_level() {
        let mut w = tmp_world("gate");
        let a = args_of(r#"{"path":"x.txt","content":"hi"}"#);
        // Read не может писать
        let out = w.call(CapLevel::Read, "fs_write", &a);
        assert!(out.is_error, "Read не даёт fs_write");
        // Write может
        let out = w.call(CapLevel::Write, "fs_write", &a);
        assert!(!out.is_error, "{}", out.text);
        // Read не может удалить
        let out = w.call(CapLevel::Read, "fs_delete", &args_of(r#"{"path":"x.txt"}"#));
        assert!(out.is_error);
        // Network не обязателен для fs; Destructive может удалить
        let out = w.call(CapLevel::Destructive, "fs_delete", &args_of(r#"{"path":"x.txt"}"#));
        assert!(!out.is_error, "{}", out.text);
    }

    #[test]
    fn write_read_undo_roundtrip() {
        let mut w = tmp_world("undo");
        w.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"d/f.txt","content":"v1"}"#));
        let got = w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"d/f.txt"}"#));
        assert_eq!(got.text, "v1");
        // перезапись поверх существующего
        w.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"d/f.txt","content":"v2-longer"}"#));
        assert_eq!(
            w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"d/f.txt"}"#)).text,
            "v2-longer"
        );
        // undo → v1 (undo мутирует — нужен Write)
        assert!(w.call(CapLevel::Write, "undo", &parse("{}").unwrap()).text.contains("восстановлено"));
        assert_eq!(
            w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"d/f.txt"}"#)).text,
            "v1"
        );
        // undo ещё раз → файла не было, remove_file
        let out = w.call(CapLevel::Write, "undo", &parse("{}").unwrap());
        assert!(out.text.contains("удалено"), "{}", out.text);
        assert!(w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"d/f.txt"}"#)).is_error);
    }

    #[test]
    fn delete_then_undo_restores() {
        let mut w = tmp_world("del");
        w.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"keep.bin","content":"данные"}"#));
        w.call(CapLevel::Destructive, "fs_delete", &args_of(r#"{"path":"keep.bin"}"#));
        assert!(w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"keep.bin"}"#)).is_error);
        let out = w.call(CapLevel::Write, "undo", &parse("{}").unwrap());
        assert!(out.text.contains("восстановлено"), "{}", out.text);
        assert_eq!(
            w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"keep.bin"}"#)).text,
            "данные"
        );
    }

    #[test]
    fn cli_allowlist_enforced() {
        let mut w = tmp_world("cli");
        let ok = w.call(CapLevel::Network, "cli_run", &args_of(r#"{"command":"echo","args":["привет"]}"#));
        assert!(!ok.is_error, "{}", ok.text);
        assert!(ok.text.contains("привет"), "{}", ok.text);
        let bad = w.call(CapLevel::Network, "cli_run", &args_of(r#"{"command":"rm","args":["-rf","/"]}"#));
        assert!(bad.is_error, "rm не в allowlist");
        let sh = w.call(CapLevel::Network, "cli_run", &args_of(r#"{"command":"sh"}"#));
        assert!(sh.is_error, "sh не в allowlist");
        // Write не может cli_run вовсе
        let lvl = w.call(CapLevel::Write, "cli_run", &args_of(r#"{"command":"echo"}"#));
        assert!(lvl.is_error);
    }

    #[test]
    fn jail_traversal_blocked_at_tool_level() {
        let mut w = tmp_world("trav");
        let out = w.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"../evil.txt","content":"x"}"#));
        assert!(out.is_error, "обход джейла бракуется");
        let out = w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"/etc/passwd"}"#));
        assert!(out.is_error);
    }

    #[test]
    fn caps_token_gate_toolcall() {
        let mut w = tmp_world("token");
        let signer = Ed25519Signer::from_seed(&[11u8; 32]);
        let mut set = CapSet::new(CapLevel::Write);
        set.expires = 1_800_000_000; // далеко в будущем (unix)
        let tok = CapToken::mint(&signer, &set);
        assert!(tok.verify(now_secs()));
        let granted = tok.set.level;
        let out = w.call(granted, "fs_write", &args_of(r#"{"path":"t.txt","content":"по токену"}"#));
        assert!(!out.is_error, "{}", out.text);
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}
