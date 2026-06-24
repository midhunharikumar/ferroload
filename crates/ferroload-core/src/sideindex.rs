//! Per-shard side-index: `member -> (offset, length)`, enabling random access
//! into a sequential tar. Written next to each shard as `<shard>.tar.idx`.
//!
//! The scaffold serializes the side-index as JSON for simplicity and testability;
//! the production format (per DESIGN.md) uses Parquet under the `parquet` feature.

use crate::error::Result;
use crate::shard::MemberLoc;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SideIndex {
    /// member name -> [offset, length]
    pub members: BTreeMap<String, [u64; 2]>,
}

impl SideIndex {
    pub fn from_locs(locs: &[MemberLoc]) -> Self {
        let mut members = BTreeMap::new();
        for l in locs {
            members.insert(l.member.clone(), [l.offset, l.length]);
        }
        SideIndex { members }
    }

    pub fn get(&self, member: &str) -> Option<(u64, u64)> {
        self.members.get(member).map(|v| (v[0], v[1]))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_lookup() {
        let locs = vec![
            MemberLoc { member: "a.jpg".into(), offset: 512, length: 100 },
            MemberLoc { member: "a.json".into(), offset: 1024, length: 20 },
        ];
        let si = SideIndex::from_locs(&locs);
        assert_eq!(si.get("a.jpg"), Some((512, 100)));
        assert_eq!(si.get("missing"), None);
    }

    #[test]
    fn save_load_roundtrip() {
        let mut p = std::env::temp_dir();
        p.push(format!("ferroload_si_{}.tar.idx", std::process::id()));
        let locs = vec![MemberLoc { member: "x.bin".into(), offset: 512, length: 7 }];
        let si = SideIndex::from_locs(&locs);
        si.save(&p).unwrap();
        let back = SideIndex::load(&p).unwrap();
        assert_eq!(si, back);
        std::fs::remove_file(&p).ok();
    }
}
