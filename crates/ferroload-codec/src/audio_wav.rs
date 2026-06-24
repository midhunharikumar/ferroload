//! Pure-Rust WAV/PCM decoder -> `[channels, samples]` F32 in [-1, 1].
//!
//! Covers the uncompressed case (and precomputed-feature pipelines). Compressed
//! formats (mp3/flac/aac) are intended to plug in via Symphonia behind an
//! `audio-codecs` feature; the [`Codec`] surface is identical.

use crate::{Codec, CodecError, Result, Tensor, TensorData};

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub struct WavCodec;

impl Codec for WavCodec {
    fn decode(&self, b: &[u8]) -> Result<Tensor> {
        if b.len() < 44 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
            return Err(CodecError::Decode("not a RIFF/WAVE file".into()));
        }
        // walk chunks to find "fmt " and "data"
        let mut pos = 12;
        let (mut fmt, mut data): (Option<(u16, u16, u16)>, Option<(usize, usize)>) = (None, None);
        while pos + 8 <= b.len() {
            let id = &b[pos..pos + 4];
            let sz = rd_u32(b, pos + 4) as usize;
            let body = pos + 8;
            if id == b"fmt " && body + 16 <= b.len() {
                let audio_format = rd_u16(b, body);
                let channels = rd_u16(b, body + 2);
                let bits = rd_u16(b, body + 14);
                fmt = Some((audio_format, channels, bits));
            } else if id == b"data" {
                let end = (body + sz).min(b.len());
                data = Some((body, end));
            }
            pos = body + sz + (sz & 1); // chunks are word-aligned
        }

        let (audio_format, channels, bits) =
            fmt.ok_or_else(|| CodecError::Decode("missing fmt chunk".into()))?;
        let (ds, de) = data.ok_or_else(|| CodecError::Decode("missing data chunk".into()))?;
        let channels = channels.max(1) as usize;
        let raw = &b[ds..de];

        // decode interleaved samples to f32
        let interleaved: Vec<f32> = match (audio_format, bits) {
            (1, 16) => raw
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect(),
            (1, 8) => raw.iter().map(|&x| (x as f32 - 128.0) / 128.0).collect(),
            (3, 32) => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            (af, bps) => {
                return Err(CodecError::Unsupported(format!(
                    "WAV format={af} bits={bps} (use Symphonia for compressed audio)"
                )))
            }
        };

        let frames = interleaved.len() / channels;
        // deinterleave -> channel-major [channels, frames]
        let mut out = vec![0f32; channels * frames];
        for f in 0..frames {
            for c in 0..channels {
                out[c * frames + f] = interleaved[f * channels + c];
            }
        }
        Ok(Tensor {
            shape: vec![channels, frames],
            data: TensorData::F32(out),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 16-bit PCM WAV from channel-major f32 samples.
    fn make_wav(channels: u16, samples_per_ch: &[Vec<i16>]) -> Vec<u8> {
        let frames = samples_per_ch[0].len();
        let mut data = Vec::new();
        for f in 0..frames {
            for ch in 0..channels as usize {
                data.extend_from_slice(&samples_per_ch[ch][f].to_le_bytes());
            }
        }
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&channels.to_le_bytes());
        w.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
        w.extend_from_slice(&(16000 * channels as u32 * 2).to_le_bytes());
        w.extend_from_slice(&(channels * 2).to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes()); // bits
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data.len() as u32).to_le_bytes());
        w.extend_from_slice(&data);
        w
    }

    #[test]
    fn decode_stereo_pcm16() {
        // ch0 = [0, 16384], ch1 = [-32768, 32767]
        let wav = make_wav(2, &[vec![0, 16384], vec![-32768, 32767]]);
        let t = WavCodec.decode(&wav).unwrap();
        assert_eq!(t.shape, vec![2, 2]); // [channels, frames]
        assert!(t.check());
        if let TensorData::F32(d) = &t.data {
            // channel-major: [ch0_f0, ch0_f1, ch1_f0, ch1_f1]
            assert!((d[0] - 0.0).abs() < 1e-6);
            assert!((d[1] - 0.5).abs() < 1e-6);
            assert!((d[2] - (-1.0)).abs() < 1e-6);
            assert!((d[3] - 0.9999).abs() < 1e-3);
        } else {
            panic!("expected F32");
        }
    }

    #[test]
    fn rejects_non_wav() {
        assert!(WavCodec.decode(b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx").is_err());
    }
}
