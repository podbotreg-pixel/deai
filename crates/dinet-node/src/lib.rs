//! dinet-node — библиотека узла DI-NET.

/// Роя-логика: DiLoCo-расписание эпох, robust-скрининг, fork-choice.
pub mod swarm;

/// Транспорт wire-кадров (in-memory mesh + файловый спул; libp2p — точка прикрепления).
pub mod transport;
