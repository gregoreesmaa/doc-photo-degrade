//! Fast image codecs, bounded decode-once cache, SIMD bilinear resize, and
//! atomic file writes.

use image::RgbImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

// Caps matching the standard image-limits contract (40k px/dim, 4 GB alloc).
const MAX_INPUT_DIM: usize = 40_000;
const MAX_INPUT_ALLOC: u64 = 4_000_000_000;

fn check_input_dims(w: usize, h: usize, what: &str) -> Result<(), String> {
    if w == 0
        || h == 0
        || w > MAX_INPUT_DIM
        || h > MAX_INPUT_DIM
        || (w as u64) * (h as u64) * 4 > MAX_INPUT_ALLOC
    {
        return Err(format!("input too large: {what}"));
    }
    Ok(())
}

/// Write `final_path` atomically: bytes go to a sibling `*.tmp` file first,
/// then it is renamed over the target, so a killed run never leaves a
/// half-file behind.
pub fn write_artifact(
    final_path: &Path,
    write: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    let mut tmp = final_path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    write(&tmp_path)?;
    std::fs::rename(&tmp_path, final_path).map_err(|e| {
        format!(
            "cannot rename {} -> {}: {e}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

/// Gather input PNG/JPEG files: a single file, or all `*.png`/`*.jpg`/`*.jpeg`
/// in a directory (non-recursive, sorted deterministically).
pub fn collect_inputs(input: &Path) -> Result<Vec<PathBuf>, String> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if input.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(input)
            .map_err(|e| format!("cannot list {}: {e}", input.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| {
                            e.eq_ignore_ascii_case("png")
                                || e.eq_ignore_ascii_case("jpg")
                                || e.eq_ignore_ascii_case("jpeg")
                        })
                        .unwrap_or(false)
            })
            .collect();
        files.sort();
        return Ok(files);
    }
    Err(format!("input not found: {}", input.display()))
}

/// SIMD bilinear resample (`fast_image_resize`, AVX2/SSE/NEON).
pub(crate) fn resize_rgb_fir(img: RgbImage, nw: u32, nh: u32) -> Result<RgbImage, String> {
    use fast_image_resize as fir;
    let (w, h) = (img.width(), img.height());
    let src = fir::images::Image::from_vec_u8(w, h, img.into_raw(), fir::PixelType::U8x3)
        .map_err(|e| format!("fir src: {e:?}"))?;
    let mut dst = fir::images::Image::new(nw, nh, fir::PixelType::U8x3);
    let mut resizer = fir::Resizer::new();
    resizer
        .resize(
            &src,
            &mut dst,
            &fir::ResizeOptions::new()
                .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear)),
        )
        .map_err(|e| format!("fir resize: {e:?}"))?;
    RgbImage::from_raw(nw, nh, dst.into_vec()).ok_or_else(|| "fir resize: bad dims".to_string())
}

pub(crate) fn vec_to_image(buf: &[f32], w: u32, h: u32) -> RgbImage {
    let bytes: Vec<u8> = buf.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    RgbImage::from_raw(w, h, bytes).expect("buffer size matches dimensions")
}

/// Fast-codec input loader: PNG via `zune-png`, JPEG via `zune-jpeg`,
/// anything else rejected. Any depth/layout becomes RGB8.
///
/// Uses a bounded 32-entry thread-safe decode-once cache keyed on
/// `(path, mtime, len)` so batch runs that degrade the same clean page with
/// multiple seeds only decode it once.
pub fn decode_image(path: &Path) -> Result<RgbImage, String> {
    if let Some(hit) = decode_cached_get(path) {
        return Ok(hit);
    }
    let img = decode_input_image_uncached(path)?;
    decode_cached_put(path, &img);
    Ok(img)
}

fn decode_cache_key(path: &Path) -> Option<(PathBuf, u64, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    Some((path.to_path_buf(), mtime, meta.len()))
}

type DecodeCache = Mutex<HashMap<(PathBuf, u64, u64), RgbImage>>;
static DECODE_CACHE: OnceLock<DecodeCache> = OnceLock::new();

#[cfg(test)]
static DECODE_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
fn decode_cache_hits() -> u64 {
    DECODE_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

fn decode_cached_get(path: &Path) -> Option<RgbImage> {
    let key = decode_cache_key(path)?;
    let hit = DECODE_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()?
        .get(&key)
        .cloned();
    #[cfg(test)]
    if hit.is_some() {
        DECODE_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    hit
}

fn decode_cached_put(path: &Path, img: &RgbImage) {
    let Some(key) = decode_cache_key(path) else {
        return;
    };
    let Some(mut cache) = DECODE_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
    else {
        return;
    };
    if cache.len() >= 32 {
        cache.clear();
    }
    cache.insert(key, img.clone());
}

fn decode_input_image_uncached(path: &Path) -> Result<RgbImage, String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let what = path.display().to_string();
    if raw.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        let mut dec =
            zune_png::PngDecoder::new(zune_png::zune_core::bytestream::ZCursor::new(&raw));
        let pixels = dec
            .decode()
            .map_err(|e| format!("cannot decode {what}: {e:?}"))?;
        let (w, h) = dec
            .dimensions()
            .ok_or_else(|| format!("bad PNG dims: {what}"))?;
        check_input_dims(w, h, &what)?;
        match pixels {
            zune_png::zune_core::result::DecodingResult::U8(buf) => {
                rgb8_from_components(&buf, w, h, &what)
            }
            zune_png::zune_core::result::DecodingResult::U16(buf) => {
                let hi: Vec<u8> = buf.iter().map(|v| (v >> 8) as u8).collect();
                rgb8_from_components(&hi, w, h, &what)
            }
            _ => Err(format!("unsupported PNG depth: {what}")),
        }
    } else if raw.starts_with(&[0xFF, 0xD8]) {
        decode_jpeg_bytes(&raw, &what)
    } else {
        Err(format!("unsupported input (need PNG/JPEG): {what}"))
    }
}

fn rgb8_from_components(buf: &[u8], w: usize, h: usize, what: &str) -> Result<RgbImage, String> {
    let n = (w as u64)
        .checked_mul(h as u64)
        .ok_or_else(|| format!("input too large: {what}"))?;
    let rgb: Vec<u8> = match (buf.len() as u64).checked_div(n.max(1)) {
        Some(3) => buf.to_vec(),
        Some(4) => buf
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        Some(1) => buf.iter().flat_map(|v| [*v, *v, *v]).collect(),
        Some(2) => buf
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        _ => return Err(format!("bad pixel layout: {what}")),
    };
    RgbImage::from_raw(w as u32, h as u32, rgb).ok_or_else(|| format!("bad dims: {what}"))
}

/// `zune-jpeg` decode -> RGB8. `what` labels errors (file path or stage tag).
pub(crate) fn decode_jpeg_bytes(raw: &[u8], what: &str) -> Result<RgbImage, String> {
    let mut dec = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(raw));
    dec.decode_headers()
        .map_err(|e| format!("cannot decode {what}: {e:?}"))?;
    let info = dec.info().ok_or_else(|| format!("bad JPEG dims: {what}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    check_input_dims(w, h, what)?;
    let cs = dec.output_colorspace();
    let buf = dec
        .decode()
        .map_err(|e| format!("cannot decode {what}: {e:?}"))?;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    match cs {
        Some(ColorSpace::RGB) => RgbImage::from_raw(w as u32, h as u32, buf)
            .ok_or_else(|| format!("bad JPEG dims: {what}")),
        Some(ColorSpace::Luma) => {
            let rgb: Vec<u8> = buf.iter().flat_map(|v| [*v, *v, *v]).collect();
            RgbImage::from_raw(w as u32, h as u32, rgb)
                .ok_or_else(|| format!("bad JPEG dims: {what}"))
        }
        other => Err(format!("unsupported JPEG colorspace {other:?}: {what}")),
    }
}

/// Encode an `RgbImage` to JPEG bytes using SIMD `jpeg-encoder`.
pub fn encode_jpeg(img: &RgbImage, quality: u8) -> Result<Vec<u8>, String> {
    encode_jpeg_bytes(img.as_raw(), img.width(), img.height(), quality).map(|(b, _)| b)
}

/// Encode interleaved RGB -> JPEG bytes via SIMD `jpeg-encoder` with true
/// `4:2:0` chroma subsampling (`SamplingFactor::F_2_2`), matching smartphone
/// camera JPEG encoders.
/// Returns the bytes plus the codec tag (`"zune"`) recorded in params.
pub(crate) fn encode_jpeg_bytes(
    rgb: &[u8],
    w: u32,
    h: u32,
    quality: u8,
) -> Result<(Vec<u8>, &'static str), String> {
    let w16 = u16::try_from(w).map_err(|_| "width exceeds JPEG limit".to_string())?;
    let h16 = u16::try_from(h).map_err(|_| "height exceeds JPEG limit".to_string())?;
    let mut buf: Vec<u8> = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut buf, quality);
    enc.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
    enc.encode(rgb, w16, h16, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| format!("jpeg encode: {e:?}"))?;
    Ok((buf, "zune"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "doc-photo-degrade-codecs-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tiny_input(dir: &Path, name: &str) -> PathBuf {
        let mut img = RgbImage::from_pixel(64, 64, image::Rgb([255, 255, 255]));
        for x in 8..56 {
            for t in 0..2 {
                img.put_pixel(x, 20 + t, image::Rgb([0, 0, 0]));
            }
        }
        let p = dir.join(name);
        img.save(&p).unwrap();
        p
    }

    #[test]
    fn zune_codecs_roundtrip_rgb() {
        let dir = test_dir("codec");
        let input = tiny_input(&dir, "in.png");
        let img = decode_image(&input).unwrap();
        assert_eq!((img.width(), img.height()), (64, 64));
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(img.get_pixel(10, 20).0, [0, 0, 0]);
        let jp = dir.join("round.jpg");
        let (bytes, tag) = encode_jpeg_bytes(img.as_raw(), 64, 64, 90).unwrap();
        assert_eq!(tag, "zune");
        std::fs::write(&jp, &bytes).unwrap();
        let raw = std::fs::read(&jp).unwrap();
        let back = decode_jpeg_bytes(&raw, "test").unwrap();
        assert_eq!((back.width(), back.height()), (64, 64));
        let drift: u64 = img
            .as_raw()
            .iter()
            .zip(back.as_raw().iter())
            .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u64)
            .sum();
        let mean = drift as f64 / img.as_raw().len() as f64;
        assert!(mean < 8.0, "JPEG q90 mean abs drift too large: {mean}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_cache_hits_and_sees_rewrites() {
        let dir = test_dir("deccache");
        let input = tiny_input(&dir, "in.png");
        let before = decode_cache_hits();
        let a = decode_image(&input).unwrap();
        let b = decode_image(&input).unwrap();
        assert_eq!(a.as_raw(), b.as_raw());
        assert!(
            decode_cache_hits() > before,
            "second decode of the same file must hit the cache"
        );
        let mut black = RgbImage::from_pixel(48, 48, image::Rgb([0, 0, 0]));
        for x in 0..48 {
            black.put_pixel(x, 0, image::Rgb([255, 255, 255]));
        }
        black.save(&input).unwrap();
        let c = decode_image(&input).unwrap();
        assert_eq!((c.width(), c.height()), (48, 48));
        assert_ne!(a.as_raw(), c.as_raw(), "rewritten file must decode fresh");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fir_resize_matches_dims_and_stays_close_to_triangle() {
        let mut img = RgbImage::from_pixel(64, 48, image::Rgb([240, 240, 240]));
        for y in 10..30 {
            for x in 8..56 {
                img.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        let legacy = image::imageops::resize(&img, 32, 24, image::imageops::FilterType::Triangle);
        let small = resize_rgb_fir(img, 32, 24).unwrap();
        assert_eq!((small.width(), small.height()), (32, 24));
        let drift: u64 = small
            .as_raw()
            .iter()
            .zip(legacy.as_raw().iter())
            .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u64)
            .sum();
        let mean = drift as f64 / small.as_raw().len() as f64;
        assert!(mean < 2.0, "FIR vs Triangle mean drift too large: {mean}");
    }

    #[test]
    fn write_artifact_leaves_no_tmp() {
        let dir = test_dir("artifact");
        let target = dir.join("out.txt");
        write_artifact(&target, |p| {
            std::fs::write(p, b"hello").map_err(|e| e.to_string())
        })
        .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
        assert!(!dir.join("out.txt.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn jpeg_encoder_uses_420_chroma_subsampling() {
        let img = RgbImage::from_pixel(32, 32, image::Rgb([180, 90, 45]));
        let bytes = encode_jpeg(&img, 85).expect("encode_jpeg");
        // Scan JPEG markers for SOF0 (0xFF, 0xC0) and verify 4:2:0 sampling:
        // component 1 (Y) = 0x22 (2x2), component 2 (Cb) = 0x11, component 3 (Cr) = 0x11.
        let mut i = 2usize; // skip SOI (0xFF 0xD8)
        let mut found_sof0 = false;
        while i + 4 < bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if marker == 0xC0 {
                // SOF0 payload: precision(1), height(2), width(2), n_comp(1), then [id, hv, qtable]*n_comp
                let n_comp = bytes[i + 9];
                assert_eq!(n_comp, 3, "expected 3 YCbCr components");
                let y_hv = bytes[i + 11];
                let cb_hv = bytes[i + 14];
                let cr_hv = bytes[i + 17];
                assert_eq!(y_hv, 0x22, "Y sampling must be 2x2 (4:2:0)");
                assert_eq!(cb_hv, 0x11, "Cb sampling must be 1x1 (4:2:0)");
                assert_eq!(cr_hv, 0x11, "Cr sampling must be 1x1 (4:2:0)");
                found_sof0 = true;
                break;
            }
            if marker == 0xDA || len < 2 {
                break;
            }
            i += 2 + len;
        }
        assert!(found_sof0, "SOF0 marker not found in JPEG stream");
    }
}
