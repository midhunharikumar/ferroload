//! Video decoding via ffmpeg/libav (feature `video-ffmpeg`; NVDEC via
//! `video-nvdec`). Requires system ffmpeg dev libraries + clang, so it is
//! strictly opt-in and is **not compiled in environments without them**.
//!
//! The temporal-sampling logic ([`crate::sampling`]) is shared with the tested
//! pure-Rust path; this module wires it to libav frame decoding. It is marked
//! experimental: validate against your ffmpeg version before production use.

use crate::sampling::{frame_indices, Sampling};
use crate::{Codec, CodecError, Result, Tensor, TensorData};

#[derive(Debug, Clone, Copy)]
pub struct VideoConfig {
    pub num_frames: usize,
    pub sampling: Sampling,
    /// Optional `(H, W)` to scale frames to during decode (so clips of different
    /// native resolutions stack into one `[B,T,H,W,3]` batch). `None` = native.
    pub resize: Option<(usize, usize)>,
}

impl Default for VideoConfig {
    fn default() -> Self {
        VideoConfig { num_frames: 16, sampling: Sampling::Uniform, resize: None }
    }
}

pub struct VideoCodec {
    pub cfg: VideoConfig,
}

impl VideoCodec {
    pub fn new(cfg: VideoConfig) -> Self {
        VideoCodec { cfg }
    }

    /// Decode selected frames into a `[T, H, W, 3]` U8 tensor.
    fn decode_path(&self, path: &std::path::Path) -> Result<Tensor> {
        use ffmpeg_next::format::{input, Pixel};
        use ffmpeg_next::media::Type;
        use ffmpeg_next::software::scaling::{context::Context, flag::Flags};

        ffmpeg_next::init().map_err(|e| CodecError::Decode(format!("ffmpeg init: {e}")))?;
        let mut ictx = input(&path).map_err(|e| CodecError::Decode(format!("open: {e}")))?;
        let stream = ictx
            .streams()
            .best(Type::Video)
            .ok_or_else(|| CodecError::Decode("no video stream".into()))?;
        let idx = stream.index();
        let nb_frames = stream.frames(); // container-reported frame count (may be 0)
        let dec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| CodecError::Decode(format!("params: {e}")))?;
        let mut decoder = dec_ctx
            .decoder()
            .video()
            .map_err(|e| CodecError::Decode(format!("decoder: {e}")))?;

        let (w, h) = (decoder.width(), decoder.height());
        // destination size: native, or the requested (H, W) resize
        let (dst_w, dst_h) = match self.cfg.resize {
            Some((rh, rw)) => (rw as u32, rh as u32),
            None => (w, h),
        };
        let mut scaler =
            Context::get(decoder.format(), w, h, Pixel::RGB24, dst_w, dst_h, Flags::BILINEAR)
                .map_err(|e| CodecError::Decode(format!("scaler: {e}")))?;

        let (wu, hu) = (dst_w as usize, dst_h as usize);
        let row = wu * 3; // bytes per row of the OUTPUT frame, unpadded (RGB24)

        // If the container reports a frame count, choose the target frame indices
        // up front and keep ONLY those during decode (bounds memory to num_frames).
        let wanted: Option<std::collections::HashSet<usize>> = if nb_frames > 0 {
            Some(
                frame_indices(nb_frames as usize, self.cfg.num_frames, self.cfg.sampling)
                    .into_iter()
                    .collect(),
            )
        } else {
            None
        };
        let max_wanted = wanted.as_ref().and_then(|s| s.iter().max().copied());

        let mut frames: Vec<Vec<u8>> = Vec::new();
        let mut fc: usize = 0;

        for (s, packet) in ictx.packets() {
            if s.index() == idx {
                decoder
                    .send_packet(&packet)
                    .map_err(|e| CodecError::Decode(format!("send: {e}")))?;
                drain(&mut decoder, &mut scaler, &wanted, &mut fc, &mut frames, hu, row)?;
                // stop once we've decoded past the last frame we need
                if let Some(mx) = max_wanted {
                    if fc > mx {
                        break;
                    }
                }
            }
        }
        decoder.send_eof().ok();
        drain(&mut decoder, &mut scaler, &wanted, &mut fc, &mut frames, hu, row)?;

        // If we had no frame count, subsample now from everything decoded.
        let data: Vec<u8> = if wanted.is_some() {
            frames.concat()
        } else {
            let sel = frame_indices(frames.len(), self.cfg.num_frames, self.cfg.sampling);
            let mut d = Vec::with_capacity(sel.len() * hu * row);
            for &i in &sel {
                d.extend_from_slice(&frames[i]);
            }
            d
        };
        let t = if hu * row == 0 { 0 } else { data.len() / (hu * row) };
        Ok(Tensor {
            shape: vec![t, hu, wu, 3],
            data: TensorData::U8(data),
        })
    }
}

/// Drain all currently-decodable frames, keeping only those whose index is in
/// `wanted` (or all when `wanted` is None). Stride-strips each row so the stored
/// buffer is exactly H*W*3. A free function (not a closure) so the caller can
/// still read `fc`/`frames` between calls without borrow conflicts.
#[allow(clippy::too_many_arguments)]
fn drain(
    decoder: &mut ffmpeg_next::decoder::Video,
    scaler: &mut ffmpeg_next::software::scaling::context::Context,
    wanted: &Option<std::collections::HashSet<usize>>,
    fc: &mut usize,
    frames: &mut Vec<Vec<u8>>,
    hu: usize,
    row: usize,
) -> Result<()> {
    use ffmpeg_next::util::frame::video::Video;
    let mut frame = Video::empty();
    while decoder.receive_frame(&mut frame).is_ok() {
        let keep = match wanted {
            Some(set) => set.contains(fc),
            None => true,
        };
        if keep {
            let mut rgb = Video::empty();
            scaler
                .run(&frame, &mut rgb)
                .map_err(|e| CodecError::Decode(format!("scale: {e}")))?;
            let stride = rgb.stride(0);
            let src = rgb.data(0);
            let mut fb = Vec::with_capacity(hu * row);
            for y in 0..hu {
                let s = y * stride;
                fb.extend_from_slice(&src[s..s + row]);
            }
            frames.push(fb);
        }
        *fc += 1;
    }
    Ok(())
}

impl Codec for VideoCodec {
    fn decode(&self, bytes: &[u8]) -> Result<Tensor> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        // libav reads from a path; stage bytes to a UNIQUE temp file. The id must
        // be unique per call (pid + atomic counter) so concurrent (rayon) decodes
        // never share a path and corrupt each other's input.
        let uid = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ferroload_vid_{}_{}.bin", std::process::id(), uid));
        std::fs::write(&path, bytes).map_err(|e| CodecError::Decode(format!("temp: {e}")))?;
        let r = self.decode_path(&path);
        let _ = std::fs::remove_file(&path);
        r
    }
}
