//! WebDataset-compatible tar shard writer with exact byte-offset tracking.
//!
//! We hand-roll a minimal USTAR writer (rather than depend on the `tar` crate) so
//! we can record the precise data offset of every member — the basis of random
//! access. Archives produced here are readable by GNU/BSD `tar`.

use crate::error::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const BLOCK: usize = 512;

/// Round `n` up to the next multiple of 512.
fn pad512(n: u64) -> u64 {
    n.div_ceil(BLOCK as u64) * BLOCK as u64
}

/// One member's location within a shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberLoc {
    pub member: String,
    pub offset: u64,
    pub length: u64,
}

/// Streaming tar writer that records each member's data offset.
pub struct ShardWriter {
    file: File,
    path: PathBuf,
    pos: u64,
    members: Vec<MemberLoc>,
}

impl ShardWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path)?;
        Ok(ShardWriter {
            file,
            path,
            pos: 0,
            members: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn bytes_written(&self) -> u64 {
        self.pos
    }
    pub fn members(&self) -> &[MemberLoc] {
        &self.members
    }

    /// Append a member. Returns its `(offset, length)`. Member names must be
    /// <= 100 bytes (USTAR short-name limit), which is fine for `basename.ext`.
    pub fn append(&mut self, name: &str, data: &[u8]) -> Result<MemberLoc> {
        assert!(name.len() <= 100, "member name exceeds USTAR 100-byte limit");
        let header = ustar_header(name, data.len() as u64);
        self.file.write_all(&header)?;
        self.pos += BLOCK as u64;
        let offset = self.pos; // data begins immediately after the 512-byte header

        self.file.write_all(data)?;
        self.pos += data.len() as u64;

        // pad data up to a 512 boundary
        let padded = pad512(data.len() as u64);
        let pad = padded - data.len() as u64;
        if pad > 0 {
            self.file.write_all(&vec![0u8; pad as usize])?;
            self.pos += pad;
        }

        let loc = MemberLoc {
            member: name.to_string(),
            offset,
            length: data.len() as u64,
        };
        self.members.push(loc.clone());
        Ok(loc)
    }

    /// Write the two zero end-of-archive blocks and flush.
    pub fn finish(mut self) -> Result<Vec<MemberLoc>> {
        self.file.write_all(&[0u8; BLOCK * 2])?;
        self.file.flush()?;
        Ok(self.members)
    }
}

/// Random-access read of a member by `(offset, length)` from a shard file.
pub fn read_member(shard_path: &Path, offset: u64, length: u64) -> Result<Vec<u8>> {
    let mut f = File::open(shard_path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; length as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

/// Build a 512-byte USTAR header with a correct checksum.
fn ustar_header(name: &str, size: u64) -> [u8; BLOCK] {
    let mut h = [0u8; BLOCK];
    // name [0..100]
    let nb = name.as_bytes();
    h[..nb.len()].copy_from_slice(nb);
    // mode [100..108], uid [108..116], gid [116..124]
    write_octal(&mut h[100..108], 0o644);
    write_octal(&mut h[108..116], 0);
    write_octal(&mut h[116..124], 0);
    // size [124..136], mtime [136..148]
    write_octal(&mut h[124..136], size);
    write_octal(&mut h[136..148], 0);
    // typeflag [156] = '0' (regular file)
    h[156] = b'0';
    // magic "ustar\0" [257..263], version "00" [263..265]
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    // checksum [148..156]: computed with the checksum field set to spaces
    for b in &mut h[148..156] {
        *b = b' ';
    }
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    // 6 octal digits, NUL, space
    write_octal(&mut h[148..154], sum as u64);
    h[154] = 0;
    h[155] = b' ';
    h
}

/// Write `val` as a NUL-terminated octal string right-justified in `field`.
fn write_octal(field: &mut [u8], val: u64) {
    let n = field.len();
    let s = format!("{:0width$o}", val, width = n - 1);
    field[..n - 1].copy_from_slice(s.as_bytes());
    field[n - 1] = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ferroload_shard_{}_{}",
            std::process::id(),
            name
        ));
        p
    }

    #[test]
    fn offsets_are_exact_and_readback_matches() {
        let path = tmp("rw.tar");
        let a = b"hello world".to_vec();
        let b = vec![7u8; 1000]; // crosses a 512 boundary -> exercises padding
        let c = b"{}".to_vec();

        let mut w = ShardWriter::create(&path).unwrap();
        let la = w.append("s0.txt", &a).unwrap();
        let lb = w.append("s1.bin", &b).unwrap();
        let lc = w.append("s2.json", &c).unwrap();
        let members = w.finish().unwrap();
        assert_eq!(members.len(), 3);

        // first member's data starts right after its 512-byte header
        assert_eq!(la.offset, 512);
        assert_eq!(la.length, a.len() as u64);
        // second member starts after header(512)+data(11)->pad to 512 + header(512)
        assert_eq!(lb.offset, 512 + 512 + 512);
        // readback by offset/len returns exact bytes
        assert_eq!(read_member(&path, la.offset, la.length).unwrap(), a);
        assert_eq!(read_member(&path, lb.offset, lb.length).unwrap(), b);
        assert_eq!(read_member(&path, lc.offset, lc.length).unwrap(), c);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn archive_is_valid_ustar_magic() {
        let path = tmp("magic.tar");
        let mut w = ShardWriter::create(&path).unwrap();
        w.append("a.txt", b"x").unwrap();
        w.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        // ustar magic at offset 257
        assert_eq!(&bytes[257..262], b"ustar");
        // ends with >= two zero blocks
        assert!(bytes.len() >= 512 * 3);
        std::fs::remove_file(&path).ok();
    }
}
