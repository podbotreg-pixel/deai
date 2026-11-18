//! Magnet-style восстановление модели (M4 DoD): компактная ссылка на
//! чек-пойнт — вместо самого блоба — плюс сборка блоба из кусков,
//! разбросанных по сидам. Куски приходят от РАЗНЫХ пиров и в любом
//! порядке; каждый верифицируется хэшем манифеста, финал — body_id.
//! Потеря части кусков у ОТДЕЛЬНОГО пира не мешает: важно
//! объединение кусков у живых сидов.

use crate::blob_torrent::BlobStore;
use dinet_crypto::sha256;
use dinet_registry::DeltaManifest;
use std::collections::BTreeMap;

/// Компактная ссылка на чек-пойнт (публикуется в цепи/QR как обычная строка).
#[derive(Debug, Clone, PartialEq)]
pub struct Magnet {
    pub era: u32,
    pub checkpoint_id: [u8; 32],
    pub manifest_id: [u8; 32],
    pub body_id: [u8; 32],
    pub total: u64,
    pub chunks: u32,
}

impl Magnet {
    pub fn new(era: u32, checkpoint_id: [u8; 32], manifest: &DeltaManifest) -> Magnet {
        Magnet {
            era,
            checkpoint_id,
            manifest_id: manifest.id(),
            body_id: manifest.body_id,
            total: manifest.total,
            chunks: manifest.n_chunks() as u32,
        }
    }

    /// Строка-якорь: MAG1|era|checkpoint|manifest|body|total|chunks.
    pub fn encode(&self) -> String {
        format!(
            "MAG1|{}|{}|{}|{}|{}|{}",
            self.era,
            hex(&self.checkpoint_id),
            hex(&self.manifest_id),
            hex(&self.body_id),
            self.total,
            self.chunks
        )
    }

    pub fn decode(text: &str) -> Result<Magnet, String> {
        let p: Vec<&str> = text.split('|').collect();
        if p.len() != 7 || p[0] != "MAG1" {
            return Err("magnet: не MAG1/неверное число полей".into());
        }
        Ok(Magnet {
            era: p[1].parse().map_err(|_| "magnet: era")?,
            checkpoint_id: unhex::<32>(p[2])?,
            manifest_id: unhex::<32>(p[3])?,
            body_id: unhex::<32>(p[4])?,
            total: p[5].parse().map_err(|_| "magnet: total")?,
            chunks: p[6].parse().map_err(|_| "magnet: chunks")?,
        })
    }

    /// Собрать тело из кусков живых сидов. Manifest берётся у любого
    /// держателя и проверяется по `manifest_id` из magnet (подмена
    /// манифеста не проходит). Каждый кусок пере-верифицируется.
    pub fn recover(&self, providers: &[&BlobStore]) -> Result<Vec<u8>, String> {
        let manifest = providers
            .iter()
            .find_map(|p| p.manifest(&self.manifest_id))
            .ok_or("magnet: ни у одного сида нет манифеста")?;
        if manifest.id() != self.manifest_id
            || manifest.body_id != self.body_id
            || manifest.total != self.total
            || manifest.n_chunks() != self.chunks as usize
        {
            return Err("magnet: манифест не совпадает со ссылкой".into());
        }
        let mut pieces: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        for idx in 0..manifest.n_chunks() {
            for p in providers {
                if let Some(chunk) = p.chunk(&self.manifest_id, idx) {
                    if manifest.verify_chunk(idx, chunk) {
                        pieces.insert(idx, chunk.clone());
                        break;
                    }
                }
            }
        }
        let ordered: Vec<(usize, Vec<u8>)> = pieces.into_iter().collect();
        let body = manifest.assemble(&ordered)?;
        if sha256(&body) != self.body_id {
            return Err("magnet: собранное тело не бьётся с body_id".into());
        }
        Ok(body)
    }
}

/// Проверка, что чек-пойнт из magnet реально записан в реестре узла.
pub fn checkpoint_in_registry(
    m: &Magnet,
    reg: &dinet_registry::RegistryLog,
) -> bool {
    reg.entries().iter().any(|a| a.entry.checkpoint_id == m.checkpoint_id)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex<const N: usize>(s: &str) -> Result<[u8; N], String> {
    if s.len() != N * 2 {
        return Err(format!("magnet: hex поля ожидается {} символов", N * 2));
    }
    let mut out = [0u8; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| "magnet: плохой hex".to_string())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnet_string_roundtrip() {
        let m = Magnet {
            era: 7,
            checkpoint_id: [1u8; 32],
            manifest_id: [2u8; 32],
            body_id: [3u8; 32],
            total: 512,
            chunks: 8,
        };
        assert_eq!(Magnet::decode(&m.encode()).unwrap(), m);
        assert!(Magnet::decode("MAG2|x").is_err());
        let mut bad = m.encode();
        bad.push('|');
        assert!(Magnet::decode(&bad).is_err());
    }

    /// Доля-модель восстанавливается из magnet при утрате 30% кусков у
    /// каждого сида — пока объединённые куски покрывают все позиции.
    #[test]
    fn recovers_model_after_30pct_chunk_loss() {
        let body: Vec<u8> = (0..512u32).map(|i| (i * 37 % 251) as u8).collect();
        let (manifest, chunks) = DeltaManifest::pack(&body, 64); // 8 кусков
        let m = Magnet::new(3, [9u8; 32], &manifest);
        // сид 0: нет кусков 1,5 (25%); сид 1: нет 0,6 (25%); UNION полный
        let mut a = BlobStore::default();
        let mut b = BlobStore::default();
        a.put(manifest.clone(), chunks.clone());
        b.put(manifest.clone(), chunks.clone());
        // сид 0: нет кусков 1,5 (25%); сид 1: нет 0,6 (25%); UNION полный
        a.drop_chunk(&manifest.id(), 1);
        a.drop_chunk(&manifest.id(), 5);
        b.drop_chunk(&manifest.id(), 0);
        b.drop_chunk(&manifest.id(), 6);
        let got = m.recover(&[&a, &b]).unwrap();
        assert_eq!(got, body);
        // один сид с 30% потерей не восстановит — кусков физически нет
        let mut c = BlobStore::default();
        c.put(manifest.clone(), chunks[..5].to_vec());
        for idx in 5..chunks.len() {
            c.drop_chunk(&manifest.id(), idx);
        }
        assert!(m.recover(&[&c]).is_err());
        // отравленный кусок не проходит даже если он единственный
        let mut evil = BlobStore::default();
        let mut bad_chunks = chunks.clone();
        bad_chunks[0][0] ^= 0xff;
        evil.put(manifest, bad_chunks);
        assert!(m.recover(&[&evil]).is_err());
    }

    #[test]
    fn magnet_from_manifest_matches_body() {
        let body = vec![7u8; 200];
        let (manifest, _) = DeltaManifest::pack(&body, 64);
        let m = Magnet::new(1, [4u8; 32], &manifest);
        assert_eq!(m.body_id, sha256(&body));
        assert_eq!(m.manifest_id, manifest.id());
        assert_eq!(m.chunks, 4);
    }
}
