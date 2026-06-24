//! # ferroload-codec
//!
//! Per-modality decoders behind a single [`Codec`] trait. A decoder turns raw
//! member bytes (as stored in a shard) into a [`Tensor`]. Backends are
//! feature-gated:
//!
//! - `image-codec` (default) — pure-Rust PNG/JPEG via the `image` crate.
//! - `audio-codec` (default) — pure-Rust WAV/PCM decoder (no deps).
//! - `video-ffmpeg` / `video-nvdec` (opt-in) — libav/NVDEC; require system
//!   ffmpeg + clang, so they are not built in constrained environments.
//!
//! Unknown modalities with no registered codec can always fall back to raw bytes.

use std::collections::BTreeMap;

mod tensor;
pub use tensor::{Dtype, Tensor, TensorData};

#[cfg(feature = "image-codec")]
pub mod image_codec;
#[cfg(feature = "audio-codec")]
pub mod audio_wav;
pub mod sampling; // temporal frame-index selection (pure, always available)
#[cfg(feature = "video-ffmpeg")]
pub mod video;

#[derive(Debug)]
pub enum CodecError {
    Decode(String),
    Unsupported(String),
}
impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::Decode(s) => write!(f, "decode error: {s}"),
            CodecError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}
impl std::error::Error for CodecError {}
pub type Result<T> = std::result::Result<T, CodecError>;

/// Decode raw member bytes into a tensor.
pub trait Codec: Send + Sync {
    fn decode(&self, bytes: &[u8]) -> Result<Tensor>;
}

/// Passthrough "codec" that returns the raw bytes as a 1-D U8 tensor — the
/// fallback for modalities without a registered decoder.
pub struct RawCodec;
impl Codec for RawCodec {
    fn decode(&self, bytes: &[u8]) -> Result<Tensor> {
        Ok(Tensor {
            shape: vec![bytes.len()],
            data: TensorData::U8(bytes.to_vec()),
        })
    }
}

/// A registry mapping codec name -> implementation. Custom codecs (e.g. depth,
/// hyperspectral) can be inserted at runtime.
#[derive(Default)]
pub struct Registry {
    codecs: BTreeMap<String, Box<dyn Codec>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry { codecs: BTreeMap::new() }
    }

    pub fn register(&mut self, name: &str, codec: Box<dyn Codec>) {
        self.codecs.insert(name.to_string(), codec);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Codec> {
        self.codecs.get(name).map(|b| b.as_ref())
    }

    /// Decode with the named codec, falling back to raw bytes if unregistered.
    pub fn decode_or_raw(&self, name: &str, bytes: &[u8]) -> Result<Tensor> {
        match self.get(name) {
            Some(c) => c.decode(bytes),
            None => RawCodec.decode(bytes),
        }
    }

    /// Registry preloaded with the compiled-in default codecs.
    pub fn with_defaults() -> Self {
        let mut r = Registry::new();
        #[cfg(feature = "image-codec")]
        r.register("image", Box::new(image_codec::ImageCodec));
        #[cfg(feature = "audio-codec")]
        r.register("audio", Box::new(audio_wav::WavCodec));
        r.register("raw", Box::new(RawCodec));
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_fallback_for_unknown_modality() {
        let r = Registry::with_defaults();
        let t = r.decode_or_raw("hyperspectral", b"\x01\x02\x03").unwrap();
        assert_eq!(t.shape, vec![3]);
        assert!(matches!(t.data, TensorData::U8(_)));
    }
}
