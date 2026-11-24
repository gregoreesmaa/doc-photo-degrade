//! Multi-task ground-truth mask generation and export for training document
//! dewarping, deshadowing, binarization, and artifact segmentation models.
//!
//! When [`MaskRequest::none()`] (the default) is active, mask generation is
//! skipped completely with zero allocation and zero runtime overhead.

use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, Luma, Rgb};

use crate::capture::CaptureParams;
use crate::codecs::write_artifact;
use crate::reproject::ReprojectContext;
use crate::DegradeParams;

/// Bit flags for the per-pixel `u16` artifact bitmask (`DegradeMasks::artifacts`).
pub mod artifact_bits {
    /// Pixel lies outside the original document page (desk padding, rotated
    /// corners, or perspective bleed).
    pub const OUT_OF_PAGE: u16 = 1 << 0;
    /// Paper edge tear notch.
    pub const TEAR: u16 = 1 << 1;
    /// Coffee ring or satellite stain.
    pub const STAIN: u16 = 1 << 2;
    /// Pencil polyline mark.
    pub const PENCIL: u16 = 1 << 3;
    /// Folded corner dog-ear or full-span paper fold crease.
    pub const FOLD_CREASE: u16 = 1 << 4;
    /// Translucent marker highlighter band.
    pub const HIGHLIGHTER: u16 = 1 << 5;
    /// Specular lighting glare region.
    pub const GLARE: u16 = 1 << 6;
    /// Edge tape strip or tape adhesive residue rectangle.
    pub const TAPE: u16 = 1 << 7;
    /// Staple or binder punch hole.
    pub const HOLE: u16 = 1 << 8;
    /// Opaque corner sticky note.
    pub const STICKY_NOTE: u16 = 1 << 9;
    /// Fingertip oil smudge / ridge print.
    pub const FINGERPRINT: u16 = 1 << 10;
    /// Soft-focus fingertip / thumb holding the outer page margin.
    pub const FINGER: u16 = 1 << 11;
    /// Metal binder clip clamping the outer page edge.
    pub const BINDER_CLIP: u16 = 1 << 12;
    /// Music-stand bottom lip / ledge occluding the outer bottom margin.
    pub const STAND_LEDGE: u16 = 1 << 13;
}

/// Controls which ground-truth supervision masks are generated alongside the
/// degraded image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MaskRequest {
    /// Dense backward UV flow `[H, W, 2]` mapping each degraded pixel to its
    /// normalized `[0.0, 1.0]` source document coordinate (`-1.0` for OOB).
    pub uv_flow: bool,
    /// Dense illumination field `[H, W, 2]` storing `(mult, add_norm)` where
    /// `out = in * mult + add_norm * 255.0`.
    pub illumination: bool,
    /// Clean ink mask `[H, W]` in `0..=255` warped from the pre-degrade source
    /// image via exact unprojection.
    pub clean_ink: bool,
    /// Per-pixel `u16` artifact bitmask `[H, W]` using [`artifact_bits`].
    pub artifacts: bool,
    /// When `clean_ink` is enabled, emit continuous soft ink darkness
    /// (`round(255 - luma)`) instead of hard binary (`0` or `255`).
    pub soft_ink: bool,
}

impl MaskRequest {
    /// Disable all masks (zero overhead; default).
    pub const fn none() -> Self {
        Self {
            uv_flow: false,
            illumination: false,
            clean_ink: false,
            artifacts: false,
            soft_ink: false,
        }
    }

    /// Enable all supervision masks (binary `clean_ink`; set `soft_ink = true`
    /// if soft ink targets are desired).
    pub const fn all() -> Self {
        Self {
            uv_flow: true,
            illumination: true,
            clean_ink: true,
            artifacts: true,
            soft_ink: false,
        }
    }

    /// Returns `true` if at least one mask channel is requested.
    pub const fn any(&self) -> bool {
        self.uv_flow || self.illumination || self.clean_ink || self.artifacts
    }
}

/// On-disk encoding format for floating-point masks (`uv_flow`, `illumination`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaskFormat {
    /// NumPy `.npy` v1.0 (`<f4`) for `uv_flow` and `illumination`; PNG for
    /// `clean_ink` (8-bit gray) and `artifacts` (16-bit gray).
    #[default]
    NpyPng,
    /// Pure-PNG export: 16-bit RGB PNG for `uv_flow` (R=u, G=v, B=valid),
    /// 8-bit RGB PNG for `illumination` (R=mult, G=add, B=0), 8-bit gray PNG
    /// for `clean_ink`, and 16-bit gray PNG for `artifacts`.
    Png16,
}

impl MaskFormat {
    /// Parse from CLI string (`"npy-png"`, `"npy"`, `"png16"`, `"png"`).
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "npy-png" | "npy_png" | "npypng" | "npy" => Some(MaskFormat::NpyPng),
            "png16" | "png-16" | "png_16" | "png" => Some(MaskFormat::Png16),
            _ => None,
        }
    }

    /// Canonical kebab-case name.
    pub fn as_str(&self) -> &'static str {
        match self {
            MaskFormat::NpyPng => "npy-png",
            MaskFormat::Png16 => "png16",
        }
    }
}

impl fmt::Display for MaskFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MaskFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_name(s)
            .ok_or_else(|| format!("unknown mask format `{s}` (expected `npy-png` or `png16`)"))
    }
}

/// Generated ground-truth masks aligned pixel-for-pixel with `DegradeResult::image`.
#[derive(Debug, Clone)]
pub struct DegradeMasks {
    pub width: u32,
    pub height: u32,
    /// Dense backward UV flow `[H, W, 2]` in `[0.0, 1.0]` (`-1.0` if out-of-page).
    pub uv_flow: Option<Vec<f32>>,
    /// Dense illumination field `[H, W, 2]` storing `(mult, add_norm)`.
    pub illumination: Option<Vec<f32>>,
    /// Clean ink mask `[H, W]` in `0..=255` (255 = ink, 0 = paper/OOB).
    pub clean_ink: Option<Vec<u8>>,
    /// Per-pixel artifact bitmask `[H, W]` using [`artifact_bits`].
    pub artifacts: Option<Vec<u16>>,
}

impl DegradeMasks {
    /// Atomically write all populated masks into `dir` using `stem_prefix`
    /// (e.g. `"page01_degraded"` -> `"page01_degraded_uv.npy"`, etc.).
    ///
    /// Returns the list of written file paths in deterministic order.
    pub fn write_to_dir(
        &self,
        dir: &Path,
        stem_prefix: &str,
        format: MaskFormat,
    ) -> Result<Vec<PathBuf>, String> {
        let mut written = Vec::new();
        let (w, h) = (self.width, self.height);

        if let Some(ref uv) = self.uv_flow {
            let path = match format {
                MaskFormat::NpyPng => {
                    let p = dir.join(format!("{stem_prefix}_uv.npy"));
                    write_npy_f32(&p, uv, &[h as usize, w as usize, 2])?;
                    p
                }
                MaskFormat::Png16 => {
                    let p = dir.join(format!("{stem_prefix}_uv.png"));
                    write_uv16_png(&p, uv, w, h)?;
                    p
                }
            };
            written.push(path);
        }

        if let Some(ref illum) = self.illumination {
            let path = match format {
                MaskFormat::NpyPng => {
                    let p = dir.join(format!("{stem_prefix}_illum.npy"));
                    write_npy_f32(&p, illum, &[h as usize, w as usize, 2])?;
                    p
                }
                MaskFormat::Png16 => {
                    let p = dir.join(format!("{stem_prefix}_illum.png"));
                    write_illum_png(&p, illum, w, h)?;
                    p
                }
            };
            written.push(path);
        }

        if let Some(ref ink) = self.clean_ink {
            let p = dir.join(format!("{stem_prefix}_ink.png"));
            let img: ImageBuffer<Luma<u8>, &[u8]> = ImageBuffer::from_raw(w, h, ink.as_slice())
                .ok_or_else(|| "clean_ink mask buffer size mismatch".to_string())?;
            write_artifact(&p, |tmp| {
                let f = File::create(tmp)
                    .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
                let enc = PngEncoder::new_with_quality(
                    BufWriter::new(f),
                    CompressionType::Fast,
                    FilterType::Sub,
                );
                img.write_with_encoder(enc)
                    .map_err(|e| format!("cannot write {}: {e}", tmp.display()))
            })?;
            written.push(p);
        }

        if let Some(ref art) = self.artifacts {
            let p = dir.join(format!("{stem_prefix}_artifacts.png"));
            let img: ImageBuffer<Luma<u16>, &[u16]> =
                ImageBuffer::from_raw(w, h, art.as_slice())
                    .ok_or_else(|| "artifacts mask buffer size mismatch".to_string())?;
            write_artifact(&p, |tmp| {
                let f = File::create(tmp)
                    .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
                let enc = PngEncoder::new_with_quality(
                    BufWriter::new(f),
                    CompressionType::Fast,
                    FilterType::Sub,
                );
                img.write_with_encoder(enc)
                    .map_err(|e| format!("cannot write {}: {e}", tmp.display()))
            })?;
            written.push(p);
        }

        Ok(written)
    }
}

/// Zero-dependency NumPy `.npy` v1.0 writer for little-endian `f32` (`<f4`)
/// C-contiguous arrays. Uses atomic [`write_artifact`].
pub fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> Result<(), String> {
    let expected: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| "npy shape overflow".to_string())?;
    if expected != data.len() {
        return Err(format!(
            "npy data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        ));
    }

    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", dims.join(", "))
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
    // Total prefix (6 magic + 2 version + 2 header_len) = 10 bytes.
    // Pad header with spaces + trailing '\n' so total header size is a multiple of 64.
    let unpadded = 10 + header.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    for _ in 0..pad {
        header.push(' ');
    }
    header.push('\n');

    let header_len =
        u16::try_from(header.len()).map_err(|_| "npy header exceeds v1.0 u16 limit".to_string())?;

    write_artifact(path, |tmp| {
        let f = File::create(tmp).map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        let mut w = BufWriter::with_capacity(64 * 1024, f);
        let mut write_all = |b: &[u8]| {
            w.write_all(b)
                .map_err(|e| format!("cannot write {}: {e}", tmp.display()))
        };
        write_all(b"\x93NUMPY")?;
        write_all(&[1u8, 0u8])?;
        write_all(&header_len.to_le_bytes())?;
        write_all(header.as_bytes())?;
        for chunk in data.chunks(1024) {
            let mut buf = [0u8; 4096];
            for (i, &v) in chunk.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            write_all(&buf[..chunk.len() * 4])?;
        }
        w.flush()
            .map_err(|e| format!("cannot flush {}: {e}", tmp.display()))
    })
}

/// Write dense `[H, W, 2]` UV flow as a 16-bit RGB PNG (`R = u`, `G = v`,
/// `B = 65535` for valid page pixels or `0` for out-of-page).
pub fn write_uv16_png(path: &Path, uv: &[f32], width: u32, height: u32) -> Result<(), String> {
    let n = (width as usize) * (height as usize);
    if uv.len() != n * 2 {
        return Err(format!(
            "uv_flow length {} != {}*{}*2",
            uv.len(),
            width,
            height
        ));
    }
    let mut raw = Vec::with_capacity(n * 3);
    for pair in uv.as_chunks::<2>().0 {
        let (u, v) = (pair[0], pair[1]);
        if u >= 0.0 && v >= 0.0 {
            raw.push((u.clamp(0.0, 1.0) * 65535.0).round() as u16);
            raw.push((v.clamp(0.0, 1.0) * 65535.0).round() as u16);
            raw.push(65535u16);
        } else {
            raw.push(0u16);
            raw.push(0u16);
            raw.push(0u16);
        }
    }
    let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_raw(width, height, raw)
        .ok_or_else(|| "uv16 PNG buffer size mismatch".to_string())?;
    write_artifact(path, |tmp| {
        let f = File::create(tmp).map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        let enc =
            PngEncoder::new_with_quality(BufWriter::new(f), CompressionType::Fast, FilterType::Sub);
        img.write_with_encoder(enc)
            .map_err(|e| format!("cannot write {}: {e}", tmp.display()))
    })
}

/// Write dense `[H, W, 2]` illumination `(mult, add_norm)` as an 8-bit RGB PNG
/// (`R = mult * 255`, `G = add_norm * 255`, `B = 0`).
pub fn write_illum_png(path: &Path, illum: &[f32], width: u32, height: u32) -> Result<(), String> {
    let n = (width as usize) * (height as usize);
    if illum.len() != n * 2 {
        return Err(format!(
            "illumination length {} != {}*{}*2",
            illum.len(),
            width,
            height
        ));
    }
    let mut raw = Vec::with_capacity(n * 3);
    for pair in illum.as_chunks::<2>().0 {
        let m = (pair[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let a = (pair[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        raw.push(m);
        raw.push(a);
        raw.push(0u8);
    }
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, raw)
        .ok_or_else(|| "illumination PNG buffer size mismatch".to_string())?;
    write_artifact(path, |tmp| {
        let f = File::create(tmp).map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        let enc =
            PngEncoder::new_with_quality(BufWriter::new(f), CompressionType::Fast, FilterType::Sub);
        img.write_with_encoder(enc)
            .map_err(|e| format!("cannot write {}: {e}", tmp.display()))
    })
}

/// Stamp Stage 5c capture artifacts into an active `u16` artifact mask using the
/// recorded [`CaptureParams`] geometry (zero overhead when `mask` is `None`).
pub(crate) fn stamp_capture_artifacts(mask: &mut [u16], w: u32, h: u32, p: &CaptureParams) {
    if !p.enabled || mask.len() < (w as usize) * (h as usize) {
        return;
    }
    let mut set = |x: i32, y: i32, bit: u16| {
        if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
            mask[(y as u32 * w + x as u32) as usize] |= bit;
        }
    };

    // Fingerprint ellipse.
    let (erx, ery) = (p.fingerprint_rx as f32, p.fingerprint_ry as f32);
    if erx > 0.0 && ery > 0.0 {
        for oy in -(ery as i32)..=ery as i32 {
            for ox in -(erx as i32)..=erx as i32 {
                let t = ((ox as f32 / erx).powi(2) + (oy as f32 / ery).powi(2)).sqrt();
                if t <= 1.0 {
                    set(
                        p.fingerprint_x as i32 + ox,
                        p.fingerprint_y as i32 + oy,
                        artifact_bits::FINGERPRINT,
                    );
                }
            }
        }
    }

    // Tape strip.
    if p.tape_height > 0 {
        let t = p.tape_height.min(w.min(h));
        let along = if p.tape_edge.is_multiple_of(2) { w } else { h };
        let a0 = ((p.tape_start_frac * along as f32) as u32).min(along);
        let a1 = ((p.tape_start_frac + p.tape_len_frac) * along as f32) as u32;
        let (a0, a1) = (a0.min(along), a1.min(along).max(a0 + 1));
        match p.tape_edge % 4 {
            0 => {
                for y in 0..t {
                    for x in a0..a1 {
                        set(x as i32, y as i32, artifact_bits::TAPE);
                    }
                }
            }
            1 => {
                for y in 0..t {
                    for x in a0..a1 {
                        set((w - 1 - y.min(w - 1)) as i32, x as i32, artifact_bits::TAPE);
                    }
                }
            }
            2 => {
                for y in 0..t {
                    for x in a0..a1 {
                        set(x as i32, (h - 1 - y.min(h - 1)) as i32, artifact_bits::TAPE);
                    }
                }
            }
            _ => {
                for y in 0..t {
                    for x in a0..a1 {
                        set(y as i32, x as i32, artifact_bits::TAPE);
                    }
                }
            }
        }
    }

    // Tape residue rectangles.
    for &[rx, ry, rw, rh] in &p.residue_rects {
        for y in ry..(ry + rh).min(h) {
            for x in rx..(rx + rw).min(w) {
                set(x as i32, y as i32, artifact_bits::TAPE);
            }
        }
    }

    // Punch hole + rim.
    let hr = p.hole_radius as i32;
    let hrr = p.hole_radius as f32 + 2.0;
    for oy in -(hr + 2)..=(hr + 2) {
        for ox in -(hr + 2)..=(hr + 2) {
            if ((ox * ox + oy * oy) as f32).sqrt() <= hrr {
                set(
                    p.hole_x as i32 + ox,
                    p.hole_y as i32 + oy,
                    artifact_bits::HOLE,
                );
            }
        }
    }

    // Corner sticky note.
    let sz = p.sticky_size.min(w).min(h);
    let (ox, oy) = match p.sticky_corner % 4 {
        0 => (0, 0),
        1 => (w - sz, 0),
        2 => (0, h - sz),
        _ => (w - sz, h - sz),
    };
    for y in oy..oy + sz {
        for x in ox..ox + sz {
            mask[(y * w + x) as usize] |= artifact_bits::STICKY_NOTE;
        }
    }
}

/// Bitwise-OR downsample a `u16` artifact mask from `(src_w, src_h)` to
/// `(dst_w, dst_h)` so thin artifacts (pencil strokes, fold creases) are never
/// dropped during downscaling.
pub(crate) fn downsample_artifacts_or(
    src: &[u16],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u16> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut dst = vec![0u16; (dst_w as usize) * (dst_h as usize)];
    let rx = dst_w as f64 / src_w as f64;
    let ry = dst_h as f64 / src_h as f64;
    for sy in 0..src_h {
        let dy = (((sy as f64 + 0.5) * ry - 0.5).round() as i64).clamp(0, dst_h as i64 - 1) as u32;
        let srow = (sy * src_w) as usize;
        let drow = (dy * dst_w) as usize;
        for sx in 0..src_w {
            let bits = src[srow + sx as usize];
            if bits != 0 {
                let dx = (((sx as f64 + 0.5) * rx - 0.5).round() as i64).clamp(0, dst_w as i64 - 1)
                    as u32;
                dst[drow + dx as usize] |= bits;
            }
        }
    }
    dst
}

/// Bilinearly sample a single-channel `f32` luma buffer; out-of-bounds returns
/// white (`255.0`).
#[inline]
fn sample_luma_bilinear(luma: &[f32], w: u32, h: u32, x: f64, y: f64) -> f32 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let at = |ix: i64, iy: i64| -> f32 {
        if ix < 0 || iy < 0 || ix >= w as i64 || iy >= h as i64 {
            255.0
        } else {
            luma[(iy as u32 * w + ix as u32) as usize]
        }
    };
    let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
    let bot = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
    top * (1.0 - fy) + bot * fy
}

/// Build the requested [`DegradeMasks`] aligned to the final output image
/// `(params.width, params.height)`.
pub(crate) fn build_masks(
    req: &MaskRequest,
    params: &DegradeParams,
    clean_luma: Option<&[f32]>,
    art_src: Option<(&[u16], u32, u32)>,
) -> DegradeMasks {
    let w = params.width;
    let h = params.height;
    let n = (w as usize) * (h as usize);
    let ctx = ReprojectContext::new(params);

    let need_unproject = req.uv_flow || req.clean_ink || req.artifacts;
    let mut uv_flow = if req.uv_flow {
        Some(vec![-1.0f32; n * 2])
    } else {
        None
    };
    let mut clean_ink = if req.clean_ink {
        Some(vec![0u8; n])
    } else {
        None
    };
    let mut illumination = if req.illumination {
        Some(vec![0.0f32; n * 2])
    } else {
        None
    };

    // Prepare artifact mask at final output resolution.
    let mut artifacts = if req.artifacts {
        if let Some((src, sw, sh)) = art_src {
            if params.rot_scale_fused && ctx.rot_active {
                // On fused GPU, artifacts were stamped on the pre-rotation (fw, fh)
                // canvas; pull them into final output space via inverse transform.
                let mut out = vec![0u16; n];
                for y in 0..h {
                    for x in 0..w {
                        let (lx, ly) = ctx.to_lighting_coords(x as f64, y as f64);
                        let ix = lx.round() as i64;
                        let iy = ly.round() as i64;
                        if ix >= 0 && iy >= 0 && ix < sw as i64 && iy < sh as i64 {
                            out[(y * w + x) as usize] = src[(iy as u32 * sw + ix as u32) as usize];
                        }
                    }
                }
                Some(out)
            } else {
                Some(downsample_artifacts_or(src, sw, sh, w, h))
            }
        } else {
            Some(vec![0u16; n])
        }
    } else {
        None
    };

    // Pass 1: UV flow, clean_ink, and OUT_OF_PAGE artifact bit via exact unprojection.
    if need_unproject {
        let inv_w0 = 1.0 / (ctx.w0.max(1) as f64);
        let inv_h0 = 1.0 / (ctx.h0.max(1) as f64);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                match ctx.unproject(x as f64, y as f64) {
                    Some((sx, sy)) => {
                        if let Some(ref mut uv) = uv_flow {
                            uv[idx * 2] = (sx * inv_w0).clamp(0.0, 1.0) as f32;
                            uv[idx * 2 + 1] = (sy * inv_h0).clamp(0.0, 1.0) as f32;
                        }
                        if let (Some(ref mut ink_out), Some(luma)) = (&mut clean_ink, clean_luma) {
                            let lv = sample_luma_bilinear(luma, ctx.w0, ctx.h0, sx, sy);
                            ink_out[idx] = if req.soft_ink {
                                (255.0 - lv).clamp(0.0, 255.0).round() as u8
                            } else if lv < 128.0 {
                                255u8
                            } else {
                                0u8
                            };
                        }
                    }
                    None => {
                        if let Some(ref mut art) = artifacts {
                            art[idx] |= artifact_bits::OUT_OF_PAGE;
                        }
                    }
                }
            }
        }
    }

    // Pass 2: Illumination field (and GLARE bit if artifacts requested).
    if req.illumination || req.artifacts {
        let (lw, lh) = ctx.lighting_dims();
        let lwf = lw.max(1) as f64;
        let lhf = lh.max(1) as f64;
        let diag = lwf.hypot(lhf).max(1.0);
        let cx = lwf / 2.0;
        let cy = lhf / 2.0;
        let max_d2 = (cx * cx + cy * cy).max(1.0);
        let inv_max_d2 = 1.0 / max_d2;
        let gx = params.glare_center[0] as f64;
        let gy = params.glare_center[1] as f64;
        let glare_r = (params.glare_radius as f64).max(1e-6);
        let glare_r2 = glare_r * glare_r;
        let inv_glare_r = 1.0 / glare_r;
        let glare_a = params.glare_alpha as f64;
        let theta = (params.shadow_angle_deg as f64).to_radians();
        let (dx, dy) = (theta.cos(), theta.sin());
        let sh_str = params.shadow_strength as f64;
        let vig_str = params.vignette_strength as f64;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let (lx, ly) = ctx.to_lighting_coords(x as f64, y as f64);
                let gdx = lx - gx;
                let gdy = ly - gy;
                let gd2 = gdx * gdx + gdy * gdy;
                let add_norm = if gd2 < glare_r2 {
                    let g = 1.0 - gd2.sqrt() * inv_glare_r;
                    (glare_a * g * g).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let cdx = lx - cx;
                let cdy = ly - cy;
                let t = (cdx * dx + cdy * dy) / diag + 0.5;
                let shadow = 1.0 - sh_str * t.clamp(0.0, 1.0);
                let vig = 1.0 - vig_str * (cdx * cdx + cdy * cdy) * inv_max_d2;
                let mult = (shadow * vig).clamp(0.0, 1.0);

                if let Some(ref mut il) = illumination {
                    il[idx * 2] = mult as f32;
                    il[idx * 2 + 1] = add_norm as f32;
                }
                if let Some(ref mut art) = artifacts {
                    if add_norm > 0.02 {
                        art[idx] |= artifact_bits::GLARE;
                    }
                }
            }
        }
    }

    DegradeMasks {
        width: w,
        height: h,
        uv_flow,
        illumination,
        clean_ink,
        artifacts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{degrade_image, DegradeConfig};
    use image::RgbImage;

    fn test_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "doc-photo-degrade-masks-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn default_mask_request_is_none_and_zero_overhead() {
        let cfg = DegradeConfig::default().with_seed(7).with_strength(0.5);
        assert!(!cfg.masks.any());
        let img = RgbImage::from_pixel(64, 64, Rgb([250, 250, 250]));
        let res = degrade_image(img, &cfg).expect("degrade_image");
        assert!(res.masks.is_none());
    }

    #[test]
    fn npy_v1_header_magic_alignment_and_payload() {
        let dir = test_dir("npy");
        let path = dir.join("test_uv.npy");
        let data = vec![0.25f32, 0.5, 0.75, 1.0, -1.0, -1.0];
        write_npy_f32(&path, &data, &[1, 3, 2]).expect("write_npy_f32");

        let raw = std::fs::read(&path).expect("read npy");
        assert_eq!(&raw[0..6], b"\x93NUMPY");
        assert_eq!(&raw[6..8], &[1u8, 0u8]);
        let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0, "header must be 64-byte aligned");
        let header = std::str::from_utf8(&raw[10..10 + hlen]).expect("ascii header");
        assert!(header.contains("'descr': '<f4'"));
        assert!(header.contains("'fortran_order': False"));
        assert!(header.contains("'shape': (1, 3, 2)"));
        assert!(header.ends_with('\n'));

        let payload = &raw[10 + hlen..];
        assert_eq!(payload.len(), data.len() * 4);
        for (i, &expected) in data.iter().enumerate() {
            let got = f32::from_le_bytes([
                payload[i * 4],
                payload[i * 4 + 1],
                payload[i * 4 + 2],
                payload[i * 4 + 3],
            ]);
            assert_eq!(got, expected);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn png16_uv_and_mask_export_roundtrip() {
        let dir = test_dir("png16");
        let (w, h) = (96u32, 80u32);
        let mut img = RgbImage::from_pixel(w, h, Rgb([250, 250, 248]));
        for y in 24..56 {
            for x in 20..76 {
                img.put_pixel(x, y, Rgb([15, 15, 15]));
            }
        }
        let mut cfg = DegradeConfig::default().with_seed(42).with_strength(0.65);
        cfg.masks = MaskRequest::all();
        let res = degrade_image(img, &cfg).expect("degrade with masks");
        let masks = res.masks.expect("masks must be Some");
        assert_eq!(
            (masks.width, masks.height),
            (res.params.width, res.params.height)
        );
        let uv = masks.uv_flow.as_ref().expect("uv_flow");
        assert_eq!(uv.len(), (masks.width * masks.height * 2) as usize);
        let illum = masks.illumination.as_ref().expect("illumination");
        assert_eq!(illum.len(), (masks.width * masks.height * 2) as usize);
        let ink = masks.clean_ink.as_ref().expect("clean_ink");
        assert!(ink.contains(&255), "clean_ink must contain ink pixels");
        assert!(ink.contains(&0), "clean_ink must contain paper pixels");
        let art = masks.artifacts.as_ref().expect("artifacts");
        assert_eq!(art.len(), (masks.width * masks.height) as usize);

        // Test Png16 write + UV16 decode round-trip.
        let paths = masks
            .write_to_dir(&dir, "sample", MaskFormat::Png16)
            .expect("write_to_dir Png16");
        assert_eq!(paths.len(), 4);
        let uv_png = image::open(dir.join("sample_uv.png"))
            .expect("decode uv16 png")
            .into_rgb16();
        assert_eq!(uv_png.dimensions(), (masks.width, masks.height));
        for y in 0..masks.height {
            for x in 0..masks.width {
                let idx = (y * masks.width + x) as usize;
                let px = uv_png.get_pixel(x, y).0;
                let (u, v) = (uv[idx * 2], uv[idx * 2 + 1]);
                if u >= 0.0 && v >= 0.0 {
                    assert_eq!(px[2], 65535);
                    let u_rt = px[0] as f32 / 65535.0;
                    let v_rt = px[1] as f32 / 65535.0;
                    assert!((u_rt - u).abs() < 1e-4);
                    assert!((v_rt - v).abs() < 1e-4);
                } else {
                    assert_eq!(px, [0, 0, 0]);
                }
            }
        }

        // Also test NpyPng export.
        let npy_paths = masks
            .write_to_dir(&dir, "sample_npy", MaskFormat::NpyPng)
            .expect("write_to_dir NpyPng");
        assert_eq!(npy_paths.len(), 4);
        assert!(dir.join("sample_npy_uv.npy").exists());
        assert!(dir.join("sample_npy_illum.npy").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
