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
use dinet_node::registry_phase::notary_min;
use dinet_node::swarm::{Node, PeerBook, SimConfig};
use dinet_p2p::swarm_net::client_join_session;
use dinet_registry::RegistryLog;

use crate::PhoneCore;

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
}
