//! Живая проверка M4: FAO — федеративный обмен навыками с подписанными
//! наградами и эволюцией по репутации. Запуск:
//! cargo run -p dinet-agent --bin fao-demo

use dinet_agent::fao::{FaoCard, FaoPool, Reward};
use dinet_agent::json::Json;
use dinet_agent::skills::{Skill, SkillStep};
use dinet_crypto::{sha256, Ed25519Signer, ED25519_PUB_LEN};

fn step(tool: &str, text: &str) -> SkillStep {
    SkillStep {
        tool: tool.into(),
        args: Json::Obj(vec![("text".into(), Json::Str(text.into()))]),
    }
}

fn main() {
    let mut pool = FaoPool::new();
    println!("FAO: клиенты публикуют навыки с подписанной наградой");
    for (seed, name, succ, att) in [
        (1u8, "open-docs", 2u64, 10u64),
        (2, "summarize", 5, 10),
        (3, "fix-build", 9, 10),
        (4, "grep-logs", 7, 10),
    ] {
        let author = Ed25519Signer::from_seed(&[seed; ED25519_PUB_LEN]);
        let skill = Skill::mint(
            &author,
            name,
            1,
            vec![step("echo", name), step("echo", &format!("{name}-v2"))],
        );
        let eval = Ed25519Signer::from_seed(&[seed + 50; ED25519_PUB_LEN]);
        let card = FaoCard::new(
            skill.clone(),
            Reward::mint(&eval, sha256(&skill.canonical()), succ, att, 7),
        );
        // карточка проходит тот же мост 0x40, что и сетевой кадр
        let wire = card.to_payload();
        let got = FaoCard::from_payload(&wire).expect("карточка прошла 0x40");
        let accepted = pool.observe(got).expect("подписи верны");
        println!(
            "  {name}: репутация {:.0}%, принята {}",
            card.reward.rate() * 100.0,
            accepted
        );
    }
    let top = pool.top(2, 1);
    println!(
        "отбор: {:?}",
        top.iter().map(|c| c.skill.name.as_str()).collect::<Vec<_>>()
    );
    let evol = Ed25519Signer::from_seed(&[77u8; ED25519_PUB_LEN]);
    let child = pool.evolve(&evol, "evolved-hybrid", 8, 1).expect("эволюция");
    println!(
        "эволюция: {} v{} шагов={} репутация {:.0}% подписи={}",
        child.skill.name,
        child.skill.version,
        child.skill.steps.len(),
        child.reward.rate() * 100.0,
        child.valid()
    );
    // подделка: подменить награду — карточка не принимается
    let mut fake = top[0].clone();
    fake.reward.successes = fake.reward.attempts;
    println!("поддельная награда отклонена: {}", pool.observe(fake).is_err());
}
