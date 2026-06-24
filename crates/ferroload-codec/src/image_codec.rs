//! Pure-Rust image decoder (PNG/JPEG) producing an `[H, W, 3]` U8 tensor.

use crate::{Codec, CodecError, Result, Tensor, TensorData};

pub struct ImageCodec;

impl ImageCodec {
    /// Decode and resize to exactly `(h, w)` (RGB). Used for collation into a
    /// uniform `[B,H,W,3]` batch (resize-on-decode avoids full-res then shrink).
    pub fn decode_resized(&self, bytes: &[u8], h: usize, w: usize) -> Result<Tensor> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| CodecError::Decode(format!("image: {e}")))?;
        let img = img.resize_exact(w as u32, h as u32, image::imageops::FilterType::Triangle);
        let data = img.to_rgb8().into_raw();
        Ok(Tensor {
            shape: vec![h, w, 3],
            data: TensorData::U8(data),
        })
    }
}

impl Codec for ImageCodec {
    fn decode(&self, bytes: &[u8]) -> Result<Tensor> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| CodecError::Decode(format!("image: {e}")))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let data = img.into_raw(); // row-major RGB, len = h*w*3
        Ok(Tensor {
            shape: vec![h as usize, w as usize, 3],
            data: TensorData::U8(data),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Encode a tiny RGB PNG in-memory, then decode it back.
    fn make_png(w: u32, h: u32) -> Vec<u8> {
        let mut buf = image::RgbImage::new(w, h);
        for (x, y, px) in buf.enumerate_pixels_mut() {
            *px = image::Rgb([x as u8, y as u8, 7]);
        }
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    #[test]
    fn decode_png_shape_and_pixels() {
        let png = make_png(4, 3);
        let t = ImageCodec.decode(&png).unwrap();
        assert_eq!(t.shape, vec![3, 4, 3]); // [H, W, C]
        assert!(t.check());
        if let TensorData::U8(d) = &t.data {
            // pixel (x=2,y=1) -> R=2,G=1,B=7 ; offset = (y*W + x)*3
            let off = (1 * 4 + 2) * 3;
            assert_eq!(&d[off..off + 3], &[2, 1, 7]);
        } else {
            panic!("expected U8");
        }
    }

    #[test]
    fn bad_bytes_error() {
        assert!(ImageCodec.decode(b"not an image").is_err());
    }

    #[test]
    fn decode_resized_to_fixed_shape() {
        let png = make_png(10, 7);
        let t = ImageCodec.decode_resized(&png, 4, 4).unwrap();
        assert_eq!(t.shape, vec![4, 4, 3]); // [H, W, C] regardless of source size
        assert!(t.check());
    }
}
