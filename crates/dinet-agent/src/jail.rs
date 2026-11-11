//! FS-песочница (M2): ВСЕ пути инструментов разрешаются ВНУТРИ джейла.
//! Враги, которые ловятся тестами: `..`-обход, абсолютный путь мимо джейла,
//! симлинк, ведущий наружу (canonicalize + проверка префикса), подмена
//! джейла симлинком. Джейл канонизируется один раз при открытии.

use std::path::{Component, Path, PathBuf};

pub struct Jail {
    /// Канонический корень (без симлинк-неоднозначности).
    root: PathBuf,
}

impl Jail {
    /// Открыть джейл: каталог создаётся, канонизируется.
    pub fn open(dir: &Path) -> Result<Jail, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("джейл {:?}: {e}", dir.display()))?;
        let root = std::fs::canonicalize(dir)
            .map_err(|e| format!("канонизация {:?}: {e}", dir.display()))?;
        if !root.is_dir() {
            return Err(format!("{:?} не каталог", dir.display()));
        }
        Ok(Jail { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Разрешить относительный путь внутри джейла. `..` в нормализованном
    /// виде выходит за корень → Err. Возврат — канонический БУДУЩИЙ путь
    /// (файл может не существовать: проверка существования отдельно).
    pub fn resolve(&self, rel: &str) -> Result<PathBuf, String> {
        if rel.contains('\0') {
            return Err("NUL в пути".into());
        }
        let p = Path::new(rel);
        if p.is_absolute() {
            return Err(format!("абсолютный путь {rel:?} вне джейла"));
        }
        let mut depth = 0i32;
        for c in p.components() {
            match c {
                Component::Normal(part) => {
                    depth += 1;
                    let part = part.to_string_lossy();
                    if part == ".." {
                        return Err("Component::Normal не бывает '..'".into());
                    }
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    depth -= 1;
                    if depth < 0 {
                        return Err(format!("обход джейла: {rel:?}"));
                    }
                }
                _ => return Err(format!("неподдерживаемый компонент в {rel:?}")),
            }
        }
        let joined = self.root.join(p);
        // Симлинк-обход: ближайший СУЩЕСТВУЮЩИЙ предок канонизируется и
        // обязан остаться внутри (сам файл может ещё не существовать —
        // тогда симлинк-предок всё равно разыменуется).
        let mut anc = joined.clone();
        while !anc.exists() && anc != self.root {
            match anc.parent() {
                Some(par) => anc = par.to_path_buf(),
                None => break,
            }
        }
        if let Ok(real) = std::fs::canonicalize(&anc) {
            self.ensure_prefix(&real, rel)?;
        }
        Ok(joined)
    }

    /// Реальный (разыменованный симлинки) путь должен начинаться с root.
    fn ensure_prefix(&self, real: &Path, orig: &str) -> Result<(), String> {
        let root = self.root.as_os_str().as_encoded_bytes();
        let r = real.as_os_str().as_encoded_bytes();
        let inside = r.len() >= root.len()
            && &r[..root.len()] == root
            && (r.len() == root.len() || r[root.len()] == b'/');
        if inside {
            Ok(())
        } else {
            Err(format!("симлинк {orig:?} уводит за джейл"))
        }
    }

    /// Все пути, на которые ссылается разрешённый rel, включая родителей
    /// (для mkdir-p). Каждый обязан остаться внутри.
    pub fn resolve_chain(&self, rel: &str) -> Result<Vec<PathBuf>, String> {
        let p = Path::new(rel);
        if p.is_absolute() {
            return Err(format!("абсолютный путь {rel:?} вне джейла"));
        }
        let mut out = Vec::new();
        let mut cur = PathBuf::new();
        for c in p.components() {
            match c {
                Component::Normal(part) => {
                    cur.push(part);
                    let full = self.root.join(&cur);
                    if let Ok(real) = std::fs::canonicalize(&full) {
                        self.ensure_prefix(&real, rel)?;
                    }
                    out.push(full);
                }
                Component::ParentDir => {
                    if !cur.pop() {
                        return Err(format!("обход джейла: {rel:?}"));
                    }
                }
                Component::CurDir => {}
                _ => return Err(format!("неподдерживаемый компонент в {rel:?}")),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_jail(tag: &str) -> (Jail, PathBuf) {
        let dir = std::env::temp_dir().join(format!("dinet-jail-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (Jail::open(&dir).expect("джейл"), dir)
    }

    #[test]
    fn simple_paths_resolve() {
        let (j, _d) = tmp_jail("simple");
        let p = j.resolve("docs/a.txt").expect("resolve");
        assert!(p.starts_with(j.root()), "внутри джейла");
        assert!(j.resolve("./x/../y.txt").is_ok(), "CurDir и валидный .. терпимы");
    }

    #[test]
    fn traversal_rejected() {
        let (j, _d) = tmp_jail("trav");
        for bad in ["../etc/passwd", "a/../../etc", "..", "a/../../../x"] {
            assert!(j.resolve(bad).is_err(), "{bad} должен браковаться");
        }
        assert!(j.resolve("/etc/passwd").is_err(), "абсолютный бракуется");
        assert!(j.resolve("a\0b").is_err(), "NUL бракуется");
    }

    #[test]
    fn symlink_escape_rejected() {
        let (j, d) = tmp_jail("symlink");
        // наружная цель
        let outside = std::env::temp_dir().join(format!("dinet-jail-out-{}", std::process::id()));
        std::fs::create_dir_all(&outside).expect("внешний каталог");
        let link = d.join("escape");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");
        assert!(j.resolve("escape/file.txt").is_err(), "симлинк наружу бракуется");
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn chain_rejects_traversal() {
        let (j, _d) = tmp_jail("chain");
        assert!(j.resolve_chain("docs/../../out.txt").is_err());
        let chain = j.resolve_chain("docs/sub/f.txt").expect("chain");
        assert_eq!(chain.len(), 3, "docs, docs/sub, docs/sub/f.txt");
    }
}
