//! ChatModel — точка прикрепления llama.cpp.
//!
//! v0-реализация `ToyChatModel` — детерминированный шаблонный ответчик:
//! он НЕ имитирует интеллект, а проверяет пайплайн (контекст-RAG-история).
//! v1: `LlamaChatModel` (GGUF Q4) заменит его за тем же трейтом — CLI-демо
//! и сессии останутся неизменными.

/// Модели-ответчики чата. `context` — релевантные факты памяти (RAG top-k).
pub trait ChatModel {
    /// Ответ ассистента на ход пользователя с учётом контекста памяти.
    fn respond(&mut self, user: &str, context: &[String]) -> String;
    /// Короткое имя модели (для статистики/журнала сессии).
    fn name(&self) -> &'static str;
}

/// Шаблонный ответчик v0. Без зависимостей, полностью детерминированный.
#[derive(Debug, Default, Clone)]
pub struct ToyChatModel;

impl ChatModel for ToyChatModel {
    fn respond(&mut self, user: &str, context: &[String]) -> String {
        if context.is_empty() {
            return format!(
                "Пока память пуста: запомню «{}» и в следующий раз отвечу с опорой на неё.",
                short(user, 64)
            );
        }
        let facts: Vec<&String> = context.iter().take(2).collect();
        let mut s = if context.len() == 1 {
            "Из того, что я помню, есть только:".to_string()
        } else {
            "Из того, что я помню, это связано:".to_string()
        };
        for f in facts {
            s.push_str(&format!("\n  — {}", short(f, 90)));
        }
        s
    }

    fn name(&self) -> &'static str {
        "toy-v0"
    }
}

fn short(s: &str, n: usize) -> String {
    let cut: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{cut}…")
    } else {
        cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_model_uses_context_and_names_itself() {
        let mut m = ToyChatModel;
        assert_eq!(m.name(), "toy-v0");
        let r1 = m.respond("привет", &[]);
        assert!(r1.contains("память пуста"));
        let r2 = m.respond("а что есть?", &["факт один".to_string(), "факт два".to_string()]);
        assert!(r2.contains("факт один") && r2.contains("факт два"));
        let r3 = m.respond("а что есть?", &["только этот".to_string()]);
        assert!(r3.contains("только этот"));
    }
}
