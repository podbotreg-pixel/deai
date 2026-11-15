//! AT-SPI (Linux GUI) поверх рукописного D-Bus (M2, шаг «UIA/AX/AT-SPI»):
//! чтение дерева доступности (приложения → окна → элементы), текст,
//! действия (клик по индексу), фокус. Это Linux-путь MCP «серверов по
//! платформам» (Windows UIA / macOS AX — те же инструменты в бою, не
//! на этой машине).
//!
//! Политика: чтение — READ-уровень; клик/фокус — WRITE (действие в чужом
//! окне). Адрес шины доступности берётся у org.a11y.Bus (сессия) либо
//! glob-ом $XDG_RUNTIME_DIR/at-spi/bus_*; соединение ленивое — мир
//! работает и без GUI (инструменты честно отвечают «GUI не подключён»).

use crate::dbus::{Conn, Val};
use std::path::PathBuf;

/// Ссылка на узел дерева доступности: имя шины (уникальное соединение
/// приложения) + путь объекта.
#[derive(Debug, Clone, PartialEq)]
pub struct Ref {
    pub bus: String,
    pub path: String,
}

/// Узел с именем и ролью (материал для списка/дерева).
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub r: Ref,
    pub name: String,
    pub role: String,
}

///Well-known-имена AT-SPI на шине доступности.
const REGISTRY: &str = "org.a11y.atspi.Registry";
const ROOT: &str = "/org/a11y/atspi/accessible/root";
const ACC: &str = "org.a11y.atspi.Accessible";

pub struct Atspi {
    conn: Conn,
}

impl Atspi {
    /// Подключиться к шине доступности (адрес — уже разрешённый).
    pub fn connect(addr: &str) -> Result<Atspi, String> {
        let conn = Conn::open(addr)?;
        Ok(Atspi { conn })
    }

    /// Разрешить адрес шины доступности: org.a11y.Bus.GetAddress по
    /// сессии, иначе glob $XDG_RUNTIME_DIR/at-spi/bus_*.
    pub fn resolve() -> Result<String, String> {
        if let Ok(session) = std::env::var("DBUS_SESSION_BUS_ADDRESS") {
            // адрес сессии может быть списком через ';' — берём unix:path
            let path = session
                .split(';')
                .find_map(|a| a.strip_prefix("unix:path="))
                .map(|p| p.split(',').next().unwrap_or("").to_string());
            if let Some(path) = path {
                if let Ok(mut c) = Conn::open(&format!("unix:path={path}")) {
                    // адрес шины a11y — у org.a11y.Bus сессии (Hello уже в open)
                    if let Ok(v) = c.call(
                        "org.a11y.Bus", "/org/a11y/bus",
                        "org.a11y.Bus", "GetAddress", "", &[],
                    ) {
                        if let Some(s) = v.into_iter().next().and_then(|x| x.as_str().map(|s| s.to_string())) {
                            return Ok(s);
                        }
                    }
                }
            }
        }
        // запасной путь: сокет шины доступности по стандартному месту
        let runtime = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/run/user/0".into());
        let mut cands: Vec<PathBuf> = std::fs::read_dir(format!("{runtime}/at-spi"))
            .map_err(|e| format!("atspi: шина не найдена ({e})"))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("bus_")).unwrap_or(false))
            .collect();
        cands.sort();
        if let Some(p) = cands.pop() {
            return Ok(format!("unix:path={}", p.display()));
        }
        Err("atspi: шина доступности не найдена".into())
    }

    fn prop(&mut self, r: &Ref, prop: &str) -> Result<Val, String> {
        let v = self.conn.call(
            &r.bus,
            &r.path,
            "org.freedesktop.DBus.Properties",
            "Get",
            "ss",
            &[Val::Str(ACC.into()), Val::Str(prop.into())],
        )?;
        let Some(Val::Var(inner)) = v.into_iter().next() else {
            return Err("atspi: свойство без варианта".into());
        };
        Ok(*inner)
    }

    /// Имя узла (свойство Name).
    pub fn name(&mut self, r: &Ref) -> Result<String, String> {
        Ok(self.prop(r, "Name")?.as_str().unwrap_or("").to_string())
    }

    /// Роль узла (GetRoleName — «application», «frame», «push button»…).
    pub fn role(&mut self, r: &Ref) -> Result<String, String> {
        let v = self.conn.call(&r.bus, &r.path, ACC, "GetRoleName", "", &[])?;
        Ok(v.into_iter()
            .next()
            .and_then(|x| x.as_str().map(|s| s.to_string()))
            .unwrap_or_default())
    }

    /// Дети узла (GetChildren → a(so)).
    pub fn children(&mut self, r: &Ref) -> Result<Vec<Ref>, String> {
        let v = self.conn.call(&r.bus, &r.path, ACC, "GetChildren", "", &[])?;
        let Some(Val::Arr(_, items)) = v.into_iter().next() else {
            return Ok(Vec::new());
        };
        Ok(items
            .into_iter()
            .filter_map(|it| match it {
                Val::Struct(pair) => {
                    let bus = pair.first().and_then(|x| x.as_str())?.to_string();
                    let path = pair.get(1).and_then(|x| x.as_str())?.to_string();
                    Some(Ref { bus, path })
                }
                _ => None,
            })
            .collect())
    }

    /// Приложения верхнего уровня: дети корня реестра с именами.
    /// Деревья доступности шумные (часть узлов без Properties) — обход
    /// живучий: нет ответа → пустое имя, обход продолжается.
    pub fn apps(&mut self) -> Result<Vec<Node>, String> {
        let root = Ref { bus: REGISTRY.into(), path: ROOT.into() };
        let mut out = Vec::new();
        for r in self.children(&root)? {
            let name = self.name(&r).unwrap_or_default();
            let role = self.role(&r).unwrap_or_default();
            out.push(Node { r, name, role });
        }
        Ok(out)
    }

    /// Обход поддерева с ограничением глубины (себя включительно).
    pub fn tree(&mut self, r: &Ref, depth: usize) -> Result<Vec<Node>, String> {
        let mut out = Vec::new();
        let name = self.name(r).unwrap_or_default();
        let role = self.role(r).unwrap_or_default();
        out.push(Node { r: r.clone(), name, role });
        if depth > 0 {
            // ветка не отвечает → спуска не будет, остальные живы
            match self.children(r) {
                Ok(kids) => {
                    for child in kids {
                        out.extend(self.tree(&child, depth - 1)?);
                    }
                }
                Err(_) => {}
            }
        }
        Ok(out)
    }

    /// Полный текст текстового узла (Text.GetText 0..-1).
    pub fn text(&mut self, r: &Ref) -> Result<String, String> {
        let v = self.conn.call(
            &r.bus, &r.path, "org.a11y.atspi.Text", "GetText", "ii",
            &[Val::I32(0), Val::I32(-1)],
        )?;
        Ok(v.into_iter()
            .next()
            .and_then(|x| x.as_str().map(|s| s.to_string()))
            .unwrap_or_default())
    }

    /// Действия узла (Action.GetActions → a(sss): имя, описание, клавиши).
    pub fn actions(&mut self, r: &Ref) -> Result<Vec<(String, String, String)>, String> {
        let v = self
            .conn
            .call(&r.bus, &r.path, "org.a11y.atspi.Action", "GetActions", "", &[])?;
        let Some(Val::Arr(_, items)) = v.into_iter().next() else {
            return Ok(Vec::new());
        };
        Ok(items
            .into_iter()
            .filter_map(|it| match it {
                Val::Struct(t) => Some((
                    t.first().and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    t.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    t.get(2).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                )),
                _ => None,
            })
            .collect())
    }

    /// Выполнить действие по индексу (Action.DoAction) — «клик».
    pub fn do_action(&mut self, r: &Ref, index: usize) -> Result<bool, String> {
        let v = self.conn.call(
            &r.bus,
            &r.path,
            "org.a11y.atspi.Action",
            "DoAction",
            "i",
            &[Val::I32(index as i32)],
        )?;
        Ok(matches!(v.into_iter().next(), Some(Val::Bool(true))))
    }

    /// Фокус на элементе (Component.GrabFocus).
    pub fn grab_focus(&mut self, r: &Ref) -> Result<bool, String> {
        let v = self.conn.call(
            &r.bus,
            &r.path,
            "org.a11y.atspi.Component",
            "GrabFocus",
            "",
            &[],
        )?;
        Ok(matches!(v.into_iter().next(), Some(Val::Bool(true))))
    }

    /// Геометрия элемента (Component.GetExtents, экранные координаты).
    pub fn extents(&mut self, r: &Ref) -> Result<(i32, i32, i32, i32), String> {
        let v = self.conn.call(
            &r.bus,
            &r.path,
            "org.a11y.atspi.Component",
            "GetExtents",
            "u",
            &[Val::U32(0)], // ATSPI_COORD_TYPE_SCREEN
        )?;
        let Some(Val::Struct(t)) = v.into_iter().next() else {
            return Err("atspi: extents без структуры".into());
        };
        let g = |i: usize| t.get(i).and_then(|x| x.as_i32()).unwrap_or(0);
        Ok((g(0), g(1), g(2), g(3)))
    }
}
