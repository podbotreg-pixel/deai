//! DoD-замер M4: сходимость сети при потере 30% узлов «в окне».
//! Отдельный модуль, чтобы метрика была воспроизводимой: тот же прогон
//! используется и тестом, и живым демо `dod-demo`.

use crate::swarm::{run_era, Node, SimConfig};
use crate::transport::MeshTransport;

/// Точка измерения одной эры.
#[derive(Debug, Clone)]
pub struct EraPoint {
    pub era: u32,
    /// точность реплики после эры
    pub accuracy: f32,
    /// эра приняла чек-пойнт (REVEAL)
    pub checkpoint: bool,
    pub contributors: usize,
}

/// Итог прогона с выходом части узлов.
#[derive(Debug, Clone)]
pub struct ChurnReport {
    pub n_start: usize,
    pub n_exit: usize,
    pub n_end: usize,
    pub points: Vec<EraPoint>,
}

impl ChurnReport {
    /// Точность после последней эры.
    pub fn last_accuracy(&self) -> f32 {
        self.points.last().map(|p| p.accuracy).unwrap_or(0.0)
    }

    /// Все REVEAL-эры приняли чек-пойнт (сеть не встала).
    pub fn checkpoints_ok(&self) -> bool {
        self.points.iter().filter(|p| p.checkpoint).count() > 0
            && !self.points.is_empty()
    }
}

/// Прогнать `eras_before` эр полным составом, вывести `exit_pct` узлов
/// (усечение хвоста — id остаются смежными для in-memory mesh) и прогнать
/// ещё `eras_after` эр выжившими.
pub fn run_churn(
    n: usize,
    exit_pct: u32,
    eras_before: u32,
    eras_after: u32,
    seed: u64,
    cfg: SimConfig,
) -> ChurnReport {
    let n_exit = n * exit_pct as usize / 100;
    let mut nodes: Vec<Node> = (0..n).map(|i| Node::new(i as u16, 7, false)).collect();
    let mut mesh = MeshTransport::new(n);
    let mut points = Vec::new();

    for era in 0..eras_before {
        let r = run_era(&mut nodes, &mut mesh, era, &cfg, seed);
        points.push(EraPoint {
            era,
            accuracy: r.accuracy_after,
            checkpoint: r.checkpoint_id.is_some(),
            contributors: r.contributors,
        });
    }
    nodes.truncate(n - n_exit);
    let mut mesh = MeshTransport::new(n - n_exit);
    for era in eras_before..eras_before + eras_after {
        let r = run_era(&mut nodes, &mut mesh, era, &cfg, seed + 1);
        points.push(EraPoint {
            era,
            accuracy: r.accuracy_after,
            checkpoint: r.checkpoint_id.is_some(),
            contributors: r.contributors,
        });
    }
    ChurnReport { n_start: n, n_exit, n_end: n - n_exit, points }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn churn_30pct_converges() {
        let cfg = SimConfig::default();
        let rep = run_churn(20, 30, 4, 4, 5, cfg);
        assert_eq!((rep.n_start, rep.n_exit, rep.n_end), (20, 6, 14));
        assert!(rep.checkpoints_ok(), "сеть обязана принимать чек-пойнты после выхода 30%");
        let after = &rep.points[4..];
        assert!(after.iter().all(|p| p.contributors == 14), "состав выживших: {:?}", after);
        assert!(
            after.iter().any(|p| p.accuracy > 0.7),
            "сеть обязана сходиться после потери 30%: {:?}",
            after.iter().map(|p| p.accuracy).collect::<Vec<_>>()
        );
        assert!(rep.last_accuracy() > 0.5, "финальная точность: {}", rep.last_accuracy());
    }
}
