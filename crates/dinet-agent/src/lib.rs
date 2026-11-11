//! Агент DI-NET (M2): MCP-сервер поверх песочницы + capability-токены.
//! Разбивка: json (RFC 8259 подмножество), caps (READ/WRITE/DESTRUCTIVE/
//! NETWORK, ed25519-подпись), jail (FS-песочница), journal (undo),
//! tools (инструменты), mcp (JSON-RPC 2.0 stdio NDJSON).

pub mod audit;
pub mod caps;
pub mod jail;
pub mod journal;
pub mod json;
pub mod mcp;
pub mod phishing;
pub mod tools;

pub use audit::{Audit, AuditRec};
pub use caps::{CapLevel, CapSet, CapToken};
pub use jail::Jail;
pub use journal::{Inverse, Journal};
pub use mcp::Server;
pub use phishing::Verdict;
pub use tools::{AllowRule, ToolWorld};
