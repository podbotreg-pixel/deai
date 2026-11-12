//! MCP-сервер агента (M2): JSON-RPC 2.0 поверх stdio (NDJSON — одна
//! строка = одно сообщение, транспорт MCP stdio, спека 2025-06-18).
//! Хэндшейк initialize → notifications/initialized → tools/list →
//! tools/call. Ошибки инструмента — ВНУТРИ результата (isError: true),
//! протокольные — JSON-RPC error (-32601 метод, -32602 параметры,
//! -32700 разбор). Логи — только в stderr (stdout = протокол).

use crate::caps::CapLevel;
use crate::json::{self, Json};
use crate::tools::ToolWorld;
use std::io::{BufRead, Write};

/// Протокольная версия, которую сервер отдаёт в initialize.
pub const PROTOCOL_VERSION: &str = "2025-06-18";
pub const SERVER_NAME: &str = "dinet-agent";
pub const SERVER_VERSION: &str = "0.1.0";

/// Ошибки JSON-RPC.
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// Состояние сервера.
pub struct Server {
    world: ToolWorld,
    granted: CapLevel,
    initialized: bool,
}

impl Server {
    pub fn new(world: ToolWorld, granted: CapLevel) -> Server {
        Server { world, granted, initialized: false }
    }

    /// Один вход (объект Json) → ответ Json или None (уведомления молчат).
    pub fn handle(&mut self, msg: &Json) -> Option<Json> {
        let version = msg.get("jsonrpc").and_then(|j| j.as_str());
        if version != Some("2.0") {
            return Some(err_json(msg.get("id").cloned(), INVALID_REQUEST, "jsonrpc не \"2.0\""));
        }
        let method = match msg.get("method").and_then(|j| j.as_str()) {
            Some(m) => m.to_string(),
            None => {
                return Some(err_json(msg.get("id").cloned(), INVALID_REQUEST, "нет method"))
            }
        };
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Json::Obj(Vec::new()));

        // Уведомления (без id): ответов не пишем.
        if id.is_none() {
            if method == "notifications/initialized" {
                self.initialized = true; // клиент подтвердил хэндшейк
                return None;
            }
            // прочие уведомления игнорируем молча (спека: сервер не обязан)
            return None;
        }
        let Some(id) = id else { unreachable!() };

        match method.as_str() {
            "initialize" => Some(ok_json(id, initialize_result())),
            "ping" => Some(ok_json(id, Json::Obj(Vec::new()))),
            "tools/list" => Some(ok_json(id, tools_list())),
            "tools/call" => Some(self.tool_call(id, &params)),
            _ => Some(err_json(
                Some(id),
                METHOD_NOT_FOUND,
                &format!("метод {method:?} не обслуживается"),
            )),
        }
    }

    fn tool_call(&mut self, id: Json, params: &Json) -> Json {
        if !self.initialized {
            return err_json(Some(id), INVALID_REQUEST, "сессия не инициализирована");
        }
        let name = match params.get("name").and_then(|j| j.as_str()) {
            Some(n) => n.to_string(),
            None => return err_json(Some(id), INVALID_PARAMS, "tools/call: нет name"),
        };
        let args = params.get("arguments").cloned().unwrap_or(Json::Obj(Vec::new()));
        let out = self.world.call(self.granted, &name, &args);
        let result = Json::Obj(vec![
            ("content".into(), Json::Arr(vec![Json::Obj(vec![
                ("type".into(), Json::Str("text".into())),
                ("text".into(), Json::Str(out.text)),
            ])])),
            ("isError".into(), Json::Bool(out.is_error)),
        ]);
        ok_json(id, result)
    }
}

fn initialize_result() -> Json {
    Json::Obj(vec![
        ("protocolVersion".into(), Json::Str(PROTOCOL_VERSION.into())),
        (
            "capabilities".into(),
            Json::Obj(vec![("tools".into(), Json::Obj(vec![("listChanged".into(), Json::Bool(false))]))]),
        ),
        (
            "serverInfo".into(),
            Json::Obj(vec![
                ("name".into(), Json::Str(SERVER_NAME.into())),
                ("version".into(), Json::Str(SERVER_VERSION.into())),
            ]),
        ),
        (
            "instructions".into(),
            Json::Str(
                "Инструменты работают внутри FS-песочницы; мутации обратимы \
                 (undo-журнал); cli_run — только команды из allowlist."
                    .into(),
            ),
        ),
    ])
}

fn tools_list() -> Json {
    let tools: Vec<Json> = crate::tools::TOOLS
        .iter()
        .map(|(name, need, descr)| {
            let (props, req): (Vec<(String, Json)>, Vec<Json>) = match *name {
                "fs_read" | "fs_delete" => (
                    vec![("path".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))]))],
                    vec![Json::Str("path".into())],
                ),
                "fs_list" => (
                    vec![("path".into(), Json::Obj(vec![
                        ("type".into(), Json::Str("string".into())),
                        ("description".into(), Json::Str("пусто = корень джейла".into())),
                    ]))],
                    vec![],
                ),
                "undo" => (vec![], vec![]),
                "skill_save" => (
                    vec![
                        ("name".into(), Json::Obj(vec![
                            ("type".into(), Json::Str("string".into())),
                        ])),
                        ("steps".into(), Json::Obj(vec![
                            ("type".into(), Json::Str("array".into())),
                            ("items".into(), Json::Obj(vec![
                                ("type".into(), Json::Str("object".into())),
                                ("properties".into(), Json::Obj(vec![
                                    ("tool".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))])),
                                    ("args".into(), Json::Obj(vec![("type".into(), Json::Str("object".into()))])),
                                ])),
                            ])),
                        ])),
                    ],
                    vec![Json::Str("name".into()), Json::Str("steps".into())],
                ),
                "skill_run" | "skill_share" => (
                    vec![("name".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))]))],
                    vec![Json::Str("name".into())],
                ),
                "skill_import" => (
                    vec![("skill".into(), Json::Obj(vec![("type".into(), Json::Str("object".into()))]))],
                    vec![Json::Str("skill".into())],
                ),
                "skill_list" => (vec![], vec![]),
                "audit_tail" => (
                    vec![("n".into(), Json::Obj(vec![
                        ("type".into(), Json::Str("integer".into())),
                        ("description".into(), Json::Str("сколько последних записей (1-100)".into())),
                    ]))],
                    vec![],
                ),
                "fs_write" => (
                    vec![
                        ("path".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))])),
                        ("content".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))])),
                    ],
                    vec![Json::Str("path".into()), Json::Str("content".into())],
                ),
                "cli_run" => (
                    vec![
                        ("command".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))])),
                        ("args".into(), Json::Obj(vec![
                            ("type".into(), Json::Str("array".into())),
                            ("items".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))])),
                        ])),
                    ],
                    vec![Json::Str("command".into())],
                ),
                "browser_connect" => (
                    vec![("cdp".into(), Json::Obj(vec![
                        ("type".into(), Json::Str("string".into())),
                        ("description".into(), Json::Str("HTTP-адрес DevTools: '127.0.0.1:9222'".into())),
                    ]))],
                    vec![Json::Str("cdp".into())],
                ),
                "browser_navigate" => (
                    vec![("url".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))]))],
                    vec![Json::Str("url".into())],
                ),
                "browser_click" => (
                    vec![("selector".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))]))],
                    vec![Json::Str("selector".into())],
                ),
                "browser_type" => (
                    vec![
                        ("selector".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))])),
                        ("text".into(), Json::Obj(vec![("type".into(), Json::Str("string".into()))])),
                    ],
                    vec![Json::Str("selector".into()), Json::Str("text".into())],
                ),
                _ => (Vec::new(), Vec::new()),
            };
            Json::Obj(vec![
                ("name".into(), Json::Str((*name).into())),
                ("description".into(), Json::Str(format!("{descr} Требует уровень {}.", need.level.as_str()))),
                ("inputSchema".into(), Json::Obj(vec![
                    ("type".into(), Json::Str("object".into())),
                    ("properties".into(), Json::Obj(props)),
                    ("required".into(), Json::Arr(req)),
                ])),
                ("annotations".into(), Json::Obj(vec![
                    ("readOnlyHint".into(), Json::Bool(matches!(*name, "fs_read" | "fs_list" | "browser_read" | "audit_tail" | "skill_list"))),
                    ("destructiveHint".into(), Json::Bool(*name == "fs_delete")),
                ])),
            ])
        })
        .collect();
    Json::Obj(vec![("tools".into(), Json::Arr(tools))])
}

fn ok_json(id: Json, result: Json) -> Json {
    Json::Obj(vec![
        ("jsonrpc".into(), Json::Str("2.0".into())),
        ("id".into(), id),
        ("result".into(), result),
    ])
}

fn err_json(id: Option<Json>, code: i64, message: &str) -> Json {
    Json::Obj(vec![
        ("jsonrpc".into(), Json::Str("2.0".into())),
        (
            "id".into(),
            id.unwrap_or(Json::Null),
        ),
        (
            "error".into(),
            Json::Obj(vec![
                ("code".into(), Json::Int(code)),
                ("message".into(), Json::Str(message.into())),
            ]),
        ),
    ])
}

/// Цикл stdio: строки с stdin → ответы в stdout. Возврат — сколько
/// обработано сообщений. Ведение журнала — stderr.
pub fn run_stdio(
    server: &mut Server,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> usize {
    let mut handled = 0usize;
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) => break, // EOF — клиент закрыл поток
            Ok(_) => {}
            Err(e) => {
                eprintln!("[agentd] stdin: {e}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let resp = match json::parse(trimmed) {
            Ok(msg) => server.handle(&msg),
            Err(e) => {
                handled += 1;
                let _ = writeln!(
                    output,
                    "{}",
                    err_json(None, PARSE_ERROR, &format!("не разобрано: {e}")).to_string()
                );
                let _ = output.flush();
                continue;
            }
        };
        if let Some(r) = resp {
            handled += 1;
            if writeln!(output, "{}", r.to_string()).is_err() {
                break; // stdout закрыт — всё, клиент ушёл
            }
            let _ = output.flush();
        }
    }
    handled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::CapLevel;
    use crate::json::parse;
    use crate::tools::{AllowRule, ToolWorld};
    use std::io::BufReader;

    fn world(tag: &str) -> ToolWorld {
        let dir = std::env::temp_dir().join(format!("dinet-mcp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        ToolWorld::open(&dir, vec![AllowRule::new("echo")]).expect("мир")
    }

    /// Накормить сервер строками, собрать ответы (потоки в памяти).
    fn feed(srv: &mut Server, lines: &[&str]) -> Vec<Json> {
        let mut inp = String::new();
        for l in lines {
            inp.push_str(l);
            inp.push('\n');
        }
        let mut rd = BufReader::new(inp.as_bytes());
        let mut out = Vec::new();
        run_stdio(srv, &mut rd, &mut out);
        let text = String::from_utf8(out).expect("utf8");
        text.lines().map(|l| parse(l).expect("каждая строка — JSON")).collect()
    }

    fn id_of(msg: &Json) -> Option<i64> {
        match msg.get("id") {
            Some(Json::Int(n)) => Some(*n),
            _ => None,
        }
    }

    #[test]
    fn handshake_tools_call_undo_e2e() {
        let mut srv = Server::new(world("e2e"), CapLevel::Destructive);

        // initialize → protocolVersion + serverInfo
        let resp = &feed(
            &mut srv,
            &[r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#],
        );
        assert_eq!(resp.len(), 1);
        assert_eq!(id_of(&resp[0]), Some(1));
        assert_eq!(
            resp[0].get("result").unwrap().get("protocolVersion").and_then(|j| j.as_str()),
            Some(PROTOCOL_VERSION)
        );

        // notification без ответа
        let resp = feed(&mut srv, &[r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#]);
        assert!(resp.is_empty(), "уведомления молчат");

        // tools/list: семнадцать инструментов
        // (5 FS/CLI + undo + audit_tail + 5 навыковых)
        let resp = feed(&mut srv, &[r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#]);
        let tools = resp[0].get("result").unwrap().get("tools").and_then(|j| match j {
            Json::Arr(v) => Some(v.len()),
            _ => None,
        });
        assert_eq!(tools, Some(17));

        // fs_write → fs_read
        let resp = feed(
            &mut srv,
            &[
                r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fs_write","arguments":{"path":"d/x.txt","content":"привет"}}}"#,
                r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fs_read","arguments":{"path":"d/x.txt"}}}"#,
            ],
        );
        assert_eq!(resp.len(), 2);
        let text0 = |m: &Json| -> String {
            match m.get("result").and_then(|r| r.get("content")) {
                Some(Json::Arr(v)) => v
                    .first()
                    .and_then(|c| c.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                _ => String::new(),
            }
        };
        assert!(text0(&resp[0]).contains("записано"), "{}", text0(&resp[0]));
        assert_eq!(text0(&resp[1]), "привет");

        // fs_delete → undo → файл вернулся
        let resp = feed(
            &mut srv,
            &[
                r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"fs_delete","arguments":{"path":"d/x.txt"}}}"#,
                r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"undo","arguments":{}}}"#,
                r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"fs_read","arguments":{"path":"d/x.txt"}}}"#,
            ],
        );
        assert!(text0(&resp[1]).contains("восстановлено"), "{}", text0(&resp[1]));
        assert_eq!(text0(&resp[2]), "привет");
        // isError везде false
        for m in &resp {
            assert_eq!(m.get("result").and_then(|r| r.get("isError")), Some(&Json::Bool(false)));
        }
    }

    #[test]
    fn protocol_errors() {
        let mut srv = Server::new(world("errs"), CapLevel::Read);
        // битый JSON → -32700 с id=null
        let resp = feed(&mut srv, &["{битое"]);
        assert_eq!(id_of(&resp[0]), None);
        assert_eq!(
            resp[0].get("error").and_then(|e| e.get("code")),
            Some(&Json::Int(PARSE_ERROR))
        );
        // неизвестный метод → -32601
        let resp = feed(&mut srv, &[r#"{"jsonrpc":"2.0","id":9,"method":"foo/bar"}"#]);
        assert_eq!(id_of(&resp[0]), Some(9));
        assert_eq!(resp[0].get("error").and_then(|e| e.get("code")), Some(&Json::Int(METHOD_NOT_FOUND)));
        // tools/call до initialize → -32600
        let resp = feed(
            &mut srv,
            &[r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"fs_read","arguments":{"path":"a"}}}"#],
        );
        assert_eq!(resp[0].get("error").and_then(|e| e.get("code")), Some(&Json::Int(INVALID_REQUEST)));
        // jsonrpc-версия не 2.0 → -32600
        let resp = feed(&mut srv, &[r#"{"jsonrpc":"1.0","id":11,"method":"ping"}"#]);
        assert_eq!(resp[0].get("error").and_then(|e| e.get("code")), Some(&Json::Int(INVALID_REQUEST)));
    }

    #[test]
    fn unknown_tool_and_gating_inside_result() {
        let mut srv = Server::new(world("gates"), CapLevel::Read);
        feed(
            &mut srv,
            &[
                r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            ],
        );
        // неизвестный инструмент — ошибка УРОВНЯ ИНСТРУМЕНТА (isError), не -32602
        let resp = feed(
            &mut srv,
            &[r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"net_fly","arguments":{}}}"#],
        );
        let res = resp[0].get("result").unwrap();
        assert_eq!(res.get("isError"), Some(&Json::Bool(true)));
        // Read не может писать — тоже isError внутри
        let resp = feed(
            &mut srv,
            &[r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fs_write","arguments":{"path":"a","content":"b"}}}"#],
        );
        assert_eq!(
            resp[0].get("result").and_then(|r| r.get("isError")),
            Some(&Json::Bool(true))
        );
        // ping отвечает {}
        let resp = feed(&mut srv, &[r#"{"jsonrpc":"2.0","id":4,"method":"ping"}"#]);
        assert_eq!(resp[0].get("result"), Some(&Json::Obj(Vec::new())));
    }
}
