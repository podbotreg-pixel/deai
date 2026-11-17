//! Сетевой вклад телефона (M3, шаг 3): dial роя через libp2p-мост
//! (TCP+Noise+Yamux+gossipsub) и участие в эрах роя НА РЕАЛЬНОЙ СЕТИ —
//! тот же ход, что у `keeper_dynamic_join`/`netswarm`, но телефон —
//! ВНЕШНИЙ клиент: смотритель роя (`swarmkeeper`) держит порты, телефон
//! входит волной между эрами и уходит после чек-пойнта.
//!
//! Состав согласован Hello-волной (самосертификация id), вклад —
//! masked-доля своего батча, REVEAL-чек-пойнт роя применяем к реплике
//! чата так же, как локальный. Реестр сетевого вклада — append-only
//! файл `<base>.net-reg` (переживает рестарты).

use dinet_client::pairs_to_batch;
use dinet_node::join::roster_now;
use dinet_node::registry_phase::notary_min;
use dinet_node::swarm::{Node, PeerBook, SimConfig};
use dinet_p2p::swarm_net::{client_join_session, wave_until_quiet};
use dinet_registry::RegistryLog;

use crate::{P_LAST_ROSTER, P_LAST_SEEN, PhoneCore};

/// Волновой бюджет: минимум итераций (≈3.5с — mesh graft) и максимум
/// (≈6.3с); тик волны ≈70мс. Меньше минимума затихать нельзя: раньше
/// graft'а объявления состава ещё не доезжают.
const WAVE_MIN: u32 = 50;
const WAVE_MAX: u32 = 90;
/// Бюджет ожидания приглашения эры: тиков по ≈60мс (≤21с на эру; рой
/// гонит эры без нас — приглашение приходит, когда минёр начинает эру).
const WAIT_TICKS: u32 = 350;
/// Пауза на graft mesh после dial'ов (как в драйверах dinet-p2p).
const GRAFT_MS: u64 = 1600;

impl PhoneCore {
    /// СЕТЕВОЙ вклад: dial роя («рой:peers»), волна состава, серия сетевых
    /// эр на реальном TCP до чек-пойнта REVEAL (≤ `max_eras`). Чек-пойнт
    /// роя применяется к реплике чата как локальный; счётчики эр — в
    /// рой-KV каждую эру. Ошибки — человекочитаемые (GUI/лог воркера).
    pub fn net_tick(&mut self, max_eras: u32) -> Result<String, String> {
        let addrs = self.peers();
        if addrs.is_empty() {
            return Err("нет адресов роя — добавить: dinet://cmd/peer?addr=ip:порт".into());
        }
        let mut bridge = dinet_p2p::GossipBridge::spawn_on(
            self.node.id,
            [self.node.id as u8; 32],
            "/ip4/0.0.0.0/tcp/0",
        )
        .map_err(|e| format!("мост не поднялся: {e}"))?;
        for a in &addrs {
            let _ = bridge.dial_str(a);
        }
        std::thread::sleep(std::time::Duration::from_millis(GRAFT_MS));

        // узел на время сессии живёт локально (PeerBook не Clone);
        // адаптер телефона продолжается эрами роя, телефон с 0-материалом
        // не отправляется домой — эра честно ждёт приглашения
        let nid = self.node.id;
        let mut node = std::mem::replace(&mut self.node, Node::new(nid, crate::WORLD_SEED, false));
        let batch = pairs_to_batch(&self.valid_pairs());
        let mut book = PeerBook::empty();
        // реестр сетевого вклада: append-only, переживает рестарты;
        // порог от МИНИМАЛЬНОГО состава — переживает рост роя (старые
        // записи с меньшим кворумом не бракуются при перечитывании)
        let reg_path = self.base.with_extension("net-reg");
        let mut reg = RegistryLog::open(&reg_path, notary_min(3))
            .map_err(|e| format!("реестр сети: {e}"))?;
        let cfg = SimConfig::default();
        let session = match client_join_session(
            std::slice::from_mut(&mut node),
            &mut book,
            &mut bridge,
            &mut reg,
            &cfg,
            2026,
            batch,
            max_eras,
            WAVE_MIN,
            WAVE_MAX,
            WAIT_TICKS,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.node = node;
                // сессия могла успеть прокрутить эры: прокси тоже персистим
                self.persist_proxy();
                return Err(e);
            }
        };
        self.node = node;
        // EFFEKT-прокси телефона персистит: глубина копится сессиями
        // (рестарт процесса не сбивает — KV «рой:proxy»)
        self.persist_proxy();
        // счётчики: эры по участию, чек-пойнты по REVEAL — пишем сразу
        // (рестарт между сессиями не сбивает счёт)
        self.eras += session.notes.len() as u32;
        let _ = self.kv.put("рой:eras", &self.eras.to_le_bytes());
        let mut note = session.notes.join("; ");
        if session.reveals > 0 {
            self.cps += session.reveals;
            self.apply_checkpoint();
            note.push_str(&format!("; чек-пойнт роя {}-й применён к чату", self.cps));
        }
        if let Some(era) = session.last_era {
            let _ = self.kv.put("рой:net_done", &era.to_le_bytes());
        }
        Ok(note)
    }

    /// Лёгкий PROBE роя (экран «Рой», кнопка «Обновить состав»): поднять
    /// мост, dial адресов роя, волна состава до тишины — БЕЗ эр (это не
    /// вклад, а «кто сейчас в сети»). Состав и время контакта персистим
    /// («рой:last_roster»/«рой:last_seen») — `swarm_stats` показывает их
    /// и после выхода из сети. Состав от 1 узла (только я) — не ошибка:
    /// честный ответ «рой не отвечает».
    pub fn net_probe(&mut self) -> Result<String, String> {
        let addrs = self.peers();
        if addrs.is_empty() {
            return Err("нет адресов роя — добавить: dinet://cmd/peer?addr=ip:порт".into());
        }
        let mut bridge = dinet_p2p::GossipBridge::spawn_on(
            self.node.id,
            [self.node.id as u8; 32],
            "/ip4/0.0.0.0/tcp/0",
        )
        .map_err(|e| format!("мост не поднялся: {e}"))?;
        for a in &addrs {
            let _ = bridge.dial_str(a);
        }
        std::thread::sleep(std::time::Duration::from_millis(GRAFT_MS));
        // узел на время probe живёт локально (PeerBook не Clone)
        let nid = self.node.id;
        let mut node = std::mem::replace(&mut self.node, Node::new(nid, crate::WORLD_SEED, false));
        let mut book = PeerBook::empty();
        wave_until_quiet(
            std::slice::from_mut(&mut node),
            &mut book,
            &mut bridge,
            WAVE_MIN,
            WAVE_MAX,
        );
        let roster = roster_now(node.id, &book);
        self.node = node;
        // персист состава и времени (экран «Рой» читает после выхода)
        let mut rb = Vec::new();
        for id in &roster {
            rb.extend_from_slice(&id.to_le_bytes());
        }
        let _ = self.kv.put(P_LAST_ROSTER, &rb);
        let _ = self.kv.put(P_LAST_SEEN, &crate::now_ts().to_le_bytes());
        if roster.len() < 2 {
            return Ok(format!(
                "рой не отвечает: состав {roster:?} — только я (проверь адреса/сеть)"
            ));
        }
        Ok(format!(
            "рой на связи: состав {roster:?} ({} узлов)",
            roster.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{NET_LOCK, tmp};
    use dinet_node::join::roster_announce;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Probe без адресов — честная ошибка (экран «Рой» подскажет, что
    /// добавить адреса deep-link'ом).
    #[test]
    fn probe_err_without_addrs() {
        let dir = tmp("probe-noaddrs");
        let mut core = PhoneCore::open(&dir).unwrap();
        assert!(core.net_probe().is_err());
    }

    /// Probe поверх РЕАЛЬНОГО TCP: два узла роя (loopback, Hello-волна)
    /// + телефон — состав [0,1,7] виден, эр НЕТ (probe не вклад), состав
    /// и «видели» персистят (рестарт → экран «Рой» читает из KV).
    /// Взаимоисключение с другими сетевыми тестами (NET_LOCK — мост
    /// телефона фиксированный `[7;32]`, mdns переливает Hello).
    #[test]
    fn probe_over_real_network() {
        let _net = NET_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tmp("probe");
        // мосты узлов роя: сиды 0x50+id — PeerId отличен от телефона
        // ([id;32]) и от соседних сьютов (mdns/loopback-коллизии
        // одинаковых PeerId связывали чужие рои и теряли кадры)
        let mut bridges = Vec::new();
        for id in [0u16, 1] {
            let seed = [0x50u8.wrapping_add(id as u8); 32];
            bridges.push(dinet_p2p::GossipBridge::spawn(id, seed).expect("мост узла роя"));
        }
        let addrs: Vec<String> = bridges
            .iter()
            .map(|b| b.addr.clone().unwrap().to_string())
            .collect();
        // узлы роя вещают Hello в своей волне (как смотритель): СИГНЕР —
        // seed [id;32] (sim-схема самосертификации id), сам мост — 0x50+id;
        // stop-флаг — анонсеры затихают сразу после probe (не мешая
        // следующим сетевым тестам под локом)
        let stop = Arc::new(AtomicBool::new(false));
        for id in 0..2u16 {
            let mut n = Node::new(id, crate::WORLD_SEED, false);
            let mut b = bridges.remove(0);
            let st = stop.clone();
            std::thread::spawn(move || {
                // потолок ≈8с — заведомо длиннее волны probe (3.5–6.3с)
                for _ in 0..40 {
                    if st.load(Ordering::Relaxed) {
                        break;
                    }
                    roster_announce(&mut n, &mut b);
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            });
        }
        let mut core = PhoneCore::open(&dir).unwrap();
        for a in &addrs {
            core.add_peer(a).unwrap();
        }
        let note = core.net_probe().unwrap();
        stop.store(true, Ordering::Relaxed); // волна набрана — тишина
        assert!(note.contains("состав [0, 1, 7]"), "note: {note}");
        assert!(note.contains("3 узлов"), "note: {note}");
        assert_eq!(core.eras, 0, "probe не гонит эр");
        assert_eq!(core.cps, 0, "probe не трогает чек-пойнты");
        // персист: рестарт — состав и «видели» живы (экран «Рой»)
        drop(core);
        let core = PhoneCore::open(&dir).unwrap();
        let s = core.swarm_stats();
        assert!(s.contains("[0, 1, 7]"), "{s}");
        assert!(s.contains("видели 0 мин назад"), "{s}");
        // анонсерам — выйти (итерация ≤200мс), мостам — упасть: следующий
        // сетевой тест под локом начинает в тишине
        std::thread::sleep(std::time::Duration::from_millis(400));
    }
}
