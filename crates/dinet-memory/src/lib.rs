//! dinet-memory — локальная память узла DI-NET.
//!
//! Единственный персистентный слой устройства: append-only KV-лог вместо
//! SQLite (std-only, ноль зависимостей — циклы обновления минимум) и
//! hash-embedding RAG вместо sqlite-vec. Точки прикрепления:
//! - v1: enum-подобный бекенд (KV-лог / SQLite) — интерфейс `KvStore` стабилен;
//! - v1: эмбеддинги llama.cpp на месте hash-функции — интерфейс `embed()` стабилен.

pub mod kv;
pub mod rag;

pub use kv::KvStore;
pub use rag::{Doc, Retriever, EMBED_DIM, embed, dot, tokens};
