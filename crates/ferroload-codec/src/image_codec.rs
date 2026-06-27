//! Image decoder (PNG/JPEG) producing an `[H, W, 3]` U8 tensor. PNG and the
//! default JPEG path use the pure-Rust `image` crate (zune-jpeg backend). With
//! the `turbojpeg` feature, JPEG is decoded via libjpeg-turbo (SIMD C), which is
//! faster per core; PNG/other formats still go through `image`.

use crate::{Codec, CodecError, Result, Tensor, TensorData};

pub struct ImageCodec;

/// True for a JPEG (SOI marker `FF D8 FF`).
#[cfg(feature = "turbojpeg")]
#[inline]
fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

#[cfg(feature = "turbojpeg")]
fn jpeg_turbo_rgb(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let img = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB)
        .map_err(|e| CodecError::Decode(format!("turbojpeg: {e}")))?;
    Ok((img.width as u32, img.height as u32, img.pixels))
}

/// SIMD resize of an RGB8 buffer to `(dw, dh)` via `fast_image_resize` (AVX2 on
/// x86, NEON on ARM — auto-detected). Bilinear, to match the prior behavior.
fn fir_resize_rgb(sw: u32, sh: u32, rgb: Vec<u8>, dw: u32, dh: u32) -> Result<Vec<u8>> {
    use fast_image_resize::images::Image as FirImage;
    use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

    let src = FirImage::from_vec_u8(sw, sh, rgb, PixelType::U8x3)
        .map_err(|e| CodecError::Decode(format!("fir src: {e}")))?;
    let mut dst = FirImage::new(dw, dh, PixelType::U8x3);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    Resizer::new()
        .resize(&src, &mut dst, &opts)
        .map_err(|e| CodecError::Decode(format!("fir resize: {e}")))?;
    Ok(dst.into_vec())
}

impl ImageCodec {
    /// Decode to full-res RGB8 `(w, h, bytes)`. JPEG via libjpeg-turbo under the
    /// `turbojpeg` feature, else via the `image` crate (zune-jpeg / png).
    fn decode_rgb(&self, bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
        #[cfg(feature = "turbojpeg")]
        if is_jpeg(bytes) {
            return jpeg_turbo_rgb(bytes);
        }
        let img = image::load_from_memory(bytes)
            .map_err(|e| CodecError::Decode(format!("image: {e}")))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        Ok((w, h, img.into_raw()))
    }

    /// Decode and resize to exactly `(h, w)` (RGB). Used for collation into a
    /// uniform `[B,H,W,3]` batch. Decode (libjpeg-turbo/zune) + **SIMD resize**.
    pub fn decode_resized(&self, bytes: &[u8], h: usize, w: usize) -> Result<Tensor> {
        let (sw, sh, rgb) = self.decode_rgb(bytes)?;
        let data = if sw == w as u32 && sh == h as u32 {
            rgb // already target size — skip resize
        } else {
            fir_resize_rgb(sw, sh, rgb, w as u32, h as u32)?
        };
        Ok(Tensor { shape: vec![h, w, 3], data: TensorData::U8(data) })
    }
}

impl Codec for ImageCodec {
    fn decode(&self, bytes: &[u8]) -> Result<Tensor> {
        let (w, h, data) = self.decode_rgb(bytes)?; // row-major RGB, len = h*w*3
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

    // Exercises the JPEG path (libjpeg-turbo under `--features turbojpeg`, else
    // the zune-jpeg backend). JPEG is lossy, so we check shape, not exact pixels.
    fn make_jpeg(w: u32, h: u32) -> Vec<u8> {
        let mut buf = image::RgbImage::new(w, h);
        for (x, y, px) in buf.enumerate_pixels_mut() {
            *px = image::Rgb([(x as u8).wrapping_mul(9), (y as u8).wrapping_mul(9), 40]);
        }
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut out, image::ImageFormat::Jpeg)
            .unwrap();
        out.into_inner()
    }

    #[test]
    fn decode_jpeg_shape_and_resize() {
        let jpg = make_jpeg(16, 12);
        assert_eq!(&jpg[0..3], &[0xFF, 0xD8, 0xFF]); // is a JPEG
        let t = ImageCodec.decode(&jpg).unwrap();
        assert_eq!(t.shape, vec![12, 16, 3]); // [H, W, C]
        assert!(t.check());
        let r = ImageCodec.decode_resized(&jpg, 8, 8).unwrap();
        assert_eq!(r.shape, vec![8, 8, 3]);
        assert!(r.check());
    }

    #[test]
    fn decode_resized_to_fixed_shape() {
        let png = make_png(10, 7);
        let t = ImageCodec.decode_resized(&png, 4, 4).unwrap();
        assert_eq!(t.shape, vec![4, 4, 3]); // [H, W, C] regardless of source size
        assert!(t.check());
    }
}
