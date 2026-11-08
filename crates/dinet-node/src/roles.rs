//! Ротируемые роли роя (v1 leaderless-сим): агрегатор эры + lazy-audit 10%.
//!
//! Никаких выборов: роль выводится детерминированно из (номер эры, список
//! пиров), так что КАЖДЫЙ узел независимо вычисляет один и тот же ответ.
//! Роль живёт одну эру (в реальной сети окно роли ≈ 1 ч — расписанием
//! клиента); сообщения роли — AggCommit (0x20) и LazyAudit (0x23) поверх
//! того же Transport. До libp2p это sim-механика dinet-node::swarm.

use crate::swarm::NodeId;

/// Детерминированный порядок пиров: отсортированные NodeId (PeerBook и
/// реестр роя — тоже отсортированные структуры). Дубли вырезаются: один
/// адрес — одна роль.
pub fn peer_order(all: &[NodeId]) -> Vec<NodeId> {
    let mut v = all.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// Агрегатор эры: круговой сдвиг по эрам (era % N); за N эр агрегатором
/// побывает каждый. Список пиров сортируется сперва — добавление/выбытие
/// пиров не переворачивает роли, никакие голосования не нужны.
pub fn aggregator_for(all: &[NodeId], era: u32) -> NodeId {
    let order = peer_order(all);
    if order.is_empty() {
        return 0;
    }
    order[(era as usize) % order.len()]
}

/// Число lazy-аудиторов эры: ceil(pct × (N−1)) узлов КРОМЕ агрегатора,
/// но не меньше 3 (кворум pass ≥ fail терпит одиночного byz-аудитора)
/// и не больше числа прочих пиров; при пустом «прочих» — 0.
fn audit_want(others: usize, pct: f32) -> usize {
    if others == 0 {
        return 0;
    }
    let base = ((others as f32) * pct).ceil() as usize;
    base.max(3).min(others)
}

/// Ленивые аудиторы эры: ceil(pct × (N−1)) узлов из пиров КРОМЕ агрегатора,
/// выбор детерминированный — splitmix64-ключ (id, era). Список один и тот
/// же у всех узлов; другой процент или другая эра — другой набор.
pub fn auditors_for(all: &[NodeId], era: u32, pct: f32) -> Vec<NodeId> {
    let order = peer_order(all);
    let agg = aggregator_for(all, era);
    let others: Vec<NodeId> = order.into_iter().filter(|id| id != &agg).collect();
    let want = audit_want(others.len(), pct);
    let mut keyed: Vec<(u64, NodeId)> = others
        .into_iter()
        .map(|id| (mix64(id as u64 ^ (era as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)), id))
        .collect();
    keyed.sort();
    keyed.into_iter().take(want).map(|(_, id)| id).collect()
}

/// splitmix64 — качество перемешивания для отбора аудиторов.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_visits_every_peer_once_per_cycle() {
        let all: Vec<NodeId> = vec![7, 2, 9, 2, 4, 9]; // дубли dedup'ятся
        assert_eq!(peer_order(&all), vec![2, 4, 7, 9]);
        let n = 4;
        let mut seen = Vec::new();
        for era in 0..n {
            let agg = aggregator_for(&all, era);
            assert!(!seen.contains(&agg), "агрегатор повторился в цикле: {seen:?}");
            seen.push(agg);
        }
        seen.sort_unstable();
        assert_eq!(seen, vec![2, 4, 7, 9], "за N эр роль обошла всех");
        // круг замыкается: эра N — снова первый
        assert_eq!(aggregator_for(&all, n), aggregator_for(&all, 0));
    }

    #[test]
    fn auditors_are_deterministic_and_sane() {
        let all: Vec<NodeId> = (0..10).collect();
        // 10% прочих (9) → ceil = 1, но кворум минимум 3
        let a0 = auditors_for(&all, 0, 0.10);
        assert_eq!(a0.len(), 3);
        let agg0 = aggregator_for(&all, 0);
        assert!(!a0.contains(&agg0), "агрегатор не аудиторит сам себя");
        // детерминизм: те же входы — тот же список
        assert_eq!(a0, auditors_for(&all, 0, 0.10));
        // 40% прочих → ceil 3.6 = 4 (мин.кворум не сужает)
        assert_eq!(auditors_for(&all, 5, 0.40).len(), 4);
        // 100% прочих → все 9
        let full = auditors_for(&all, 3, 1.0);
        assert_eq!(full.len(), 9);
        let agg = aggregator_for(&all, 3);
        assert!(!full.contains(&agg));
        // крошечный рой: N=2 — ровно 1 аудитор (кворум не растянешь)
        let duo: Vec<NodeId> = vec![3, 1];
        let aud = auditors_for(&duo, 0, 0.10);
        assert_eq!(aud.len(), 1);
        assert_ne!(aud[0], aggregator_for(&duo, 0));
        // N=1 — никому проверять нечего
        assert!(auditors_for(&[6], 0, 0.10).is_empty());
        // перемена порядка списка ничего не меняет: сортировка сперва
        let rev: Vec<NodeId> = (0..10).rev().collect();
        assert_eq!(auditors_for(&all, 2, 0.15), auditors_for(&rev, 2, 0.15));
    }
}
