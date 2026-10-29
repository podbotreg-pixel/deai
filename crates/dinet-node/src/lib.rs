//! dinet-node — библиотека узла DI-NET.

/// Роя-логика: DiLoCo-расписание эпох, robust-скрининг, fork-choice.
pub mod swarm;

/// Транспорт wire-кадров (in-memory mesh + файловый спул; libp2p — точка прикрепления).
pub mod transport;

/// ФАЗА F нотариев: печать чек-пойнтов в append-only реестр роя.
pub mod registry_phase;

/// ФАЗА G torrent: репликация весовых блобов чек-пойнтов между пирами.
pub mod blob_torrent;

/// ФАЗА H: догоняющая синхронизация реестра для отставшего новичка.
pub mod reg_sync;
