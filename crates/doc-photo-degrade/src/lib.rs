//! `doc-photo-degrade`: realistic, deterministic document-to-phone-photo
//! degradation pipeline for OCR, document understanding, and computer-vision
//! training data.
//!
//! Applies, in order:
//! 1. Paper grain & speckles + paper aging (foxing, mottling, fibers, water damage)
//! 2. Ink & print degradation (fading, dropout, broken strokes, bleed-through, toner)
//! 3. Page bend (edge curl, book-spine gutter, hand-held ripple)
//! 4. Margin-safe perspective warp + photo framing (desk inset or tight crop clamped to quiet margins)
//! 5. Page rotation (canvas-expanding, no content loss)
//! 6. Ink-aware physical artifacts (tears, coffee stains, pencil marks, dog-ears, highlighter, fold creases)
//! 7. Heavy wear (washout zones, printer banding, streaks, toner bands, misregistration, eraser abrasion, stains)
//! 8. Defocus / tilt-shift blur + lighting (glare, shadow, vignette) + chromatic aberration
//! 9. Capture & handling artifacts (exposure, sensor banding, motion blur, fingerprints, tape, residue, holes, sticky notes)
//! 10. Sensor noise, capture downscale, and JPEG compression round-trip.
//!
//! All randomness derives from a seeded `ChaCha8Rng` (CPU) or deterministic
//! per-pixel GPU hash streams (WGPU backend, behind the optional `gpu` feature).
//! Every degraded output is paired with a [`DegradeParams`] record suitable for
//! JSON sidecars and ground-truth label reprojection ([`reproject`]).
//!
//! # Quick Start
//!
//! ```rust
//! use doc_photo_degrade::{BackendPreference, Degrader, DocumentProfile};
//! use image::{Rgb, RgbImage};
//!
//! let degrader = Degrader::builder()
//!     .profile(DocumentProfile::PhonePhoto)
//!     .strength(0.5)
//!     .backend(BackendPreference::Cpu)
//!     .scale_min(1.0)
//!     .build()?;
//!
//! let clean = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
//! let out = degrader.degrade_image(&clean, 42)?;
//! assert_eq!(
//!     (out.image.width(), out.image.height()),
//!     (out.params.width, out.params.height)
//! );
//! # Ok::<(), doc_photo_degrade::DegradeError>(())
//! ```

pub mod banks;
pub mod capture;
pub mod codecs;
pub mod degrader;
pub mod geometry;
#[cfg(feature = "gpu")]
pub mod gpu_backend;
pub mod ink;
pub mod masks;
pub mod paper;
pub mod realism;
pub mod reproject;
pub mod wear;

pub use banks::{PaperWeaveProfile, WeaveStock};
pub use capture::CaptureParams;
pub use codecs::{collect_inputs, decode_image, encode_jpeg, write_artifact};
pub use degrader::{DegradeError, Degrader, DegraderBuilder, DocumentProfile};
pub use geometry::{
    content_bbox, content_bbox_thresh, margin_safe_crop, quiet_margins, CornerDetachConfig,
};
#[cfg(feature = "gpu")]
pub use gpu_backend::Gpu;
pub use ink::InkParams;
pub use masks::{artifact_bits, DegradeMasks, MaskFormat, MaskRequest};
pub use paper::{PaperParams, PaperStock, PulpInclusionConfig, SurfaceBrdfParams};
pub use realism::RealismConfig;
pub use reproject::{
    bend_backward, bend_forward_exact, transform_bbox, transform_bbox_exact, transform_point_exact,
    transform_point_linear, transform_polygon, transform_polygon_exact, unproject_point_exact,
};
pub use wear::WearParams;

pub(crate) use geometry::sample_bilinear;

use codecs::{
    decode_jpeg_bytes, encode_jpeg_bytes, jpeg_roundtrip_shifted, resize_rgb_fir, vec_to_image,
    vec_to_image_with_lut,
};
use geometry::{
    apply_framing_eager, apply_h, blend_px, compute_detached_corners, framed_dims, ink_grid,
    invert_3x3, pick_quiet_cell, sample_bilinear_fill, sample_remapped, solve_homography,
    spine_map, stroke, BBoxAcc, DESK_BG, PAPER_BG,
};
use image::RgbImage;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Pipeline preset controlling which stages run at full fidelity vs skipped/coarsened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    /// All stages enabled at full fidelity.
    #[default]
    Full,
    /// Faster training preset: skips fine grain, chromatic aberration, and
    /// sensor noise, and coarsens ink/aging to keep topology-preserving defects.
    Reduced,
    /// Custom combination of individual stage toggles.
    Custom,
}

impl Preset {
    pub fn as_str(&self) -> &'static str {
        match self {
            Preset::Full => "full",
            Preset::Reduced => "reduced",
            Preset::Custom => "custom",
        }
    }
}

/// Output image format for CLI / batch workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Png,
    Jpeg,
}

/// Compute backend selection preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Auto-enable WGPU when compiled with `gpu` and an adapter is available;
    /// otherwise fall back cleanly to CPU.
    #[default]
    Auto,
    /// Force CPU execution for all stages.
    Cpu,
    /// Require WGPU acceleration (return an error if no adapter is available).
    Gpu,
}

/// Fine-grained stage skip/coarsen toggles.
///
/// `Default` enables the full pipeline. `fast_banks` is orthogonal
/// (precomputed seeded noise banks + lighting field) and does not alter the
/// reported [`Preset`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageToggles {
    pub skip_grain: bool,
    pub skip_bend: bool,
    pub skip_chromatic: bool,
    pub skip_noise: bool,
    pub coarsen_ink: bool,
    pub coarsen_aging: bool,
    pub fast_banks: bool,
}

impl StageToggles {
    /// Full-fidelity pipeline (all stages active).
    pub fn full() -> Self {
        Self::default()
    }

    /// Reduced pipeline preset (`skip_grain`, `skip_chromatic`, `skip_noise`,
    /// `coarsen_ink`, `coarsen_aging`).
    pub fn reduced() -> Self {
        Self {
            skip_grain: true,
            skip_bend: false,
            skip_chromatic: true,
            skip_noise: true,
            coarsen_ink: true,
            coarsen_aging: true,
            fast_banks: false,
        }
    }

    /// Classify these toggles into a [`Preset`].
    pub fn preset(&self) -> Preset {
        let all = !self.skip_bend
            && self.skip_grain
            && self.skip_chromatic
            && self.skip_noise
            && self.coarsen_ink
            && self.coarsen_aging;
        let any = self.skip_bend
            || self.skip_grain
            || self.skip_chromatic
            || self.skip_noise
            || self.coarsen_ink
            || self.coarsen_aging;
        if all {
            Preset::Reduced
        } else if any {
            Preset::Custom
        } else {
            Preset::Full
        }
    }

    pub fn preset_name(&self) -> &'static str {
        self.preset().as_str()
    }
}

/// Configuration for degrading a document page.
#[derive(Debug, Clone)]
pub struct DegradeConfig {
    /// Deterministic seed for this image.
    pub seed: u64,
    /// Global effect strength in `[0.0, 1.0]`.
    pub strength: f32,
    /// Optional JPEG quality override (`1..=100`). When `None`, sampled per image.
    pub jpeg_quality: Option<u8>,
    /// Cap input width before degradation (`0` = keep native width).
    pub max_width: u32,
    /// Optional training-resolution target width (`0` = off). Rescales
    /// absolute-pixel statistics by `k = train_width / src_width`.
    pub train_width: u32,
    /// Minimum random output downscale factor in `(0.0, 1.0]` (`1.0` = no downscale).
    pub scale_min: f32,
    /// Per-stage skip/coarsen toggles and fast-bank switch.
    pub toggles: StageToggles,
    /// Wave 2 & Wave 3 photorealism toggles (calibrated per [`DocumentProfile`]).
    pub realism: RealismConfig,
    /// Independent per-corner detachment from image edges (default: enabled for phone captures).
    pub corner_detach: CornerDetachConfig,
    /// Optional multi-task ground-truth mask export (default: [`MaskRequest::none()`],
    /// zero overhead when disabled).
    pub masks: MaskRequest,
}

impl Default for DegradeConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            strength: 0.5,
            jpeg_quality: None,
            max_width: 2480,
            train_width: 0,
            scale_min: 0.5,
            toggles: StageToggles::default(),
            realism: RealismConfig::default(),
            corner_detach: CornerDetachConfig::default(),
            masks: MaskRequest::none(),
        }
    }
}

impl DegradeConfig {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    pub fn with_preset(mut self, preset: Preset) -> Self {
        match preset {
            Preset::Full => self.toggles = StageToggles::full(),
            Preset::Reduced => {
                self.toggles = StageToggles::reduced();
                self.realism = RealismConfig::reduced();
            }
            Preset::Custom => {}
        }
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.scale_min) || self.scale_min <= 0.0 {
            return Err(format!(
                "scale_min must be in (0, 1], got {}",
                self.scale_min
            ));
        }
        if let Some(q) = self.jpeg_quality {
            if !(1..=100).contains(&q) {
                return Err(format!("jpeg_quality must be 1..=100, got {q}"));
            }
        }
        Ok(())
    }
}

/// Output of degrading a single document image: degraded pixels, full
/// parameter provenance, and optional ground-truth supervision masks.
#[derive(Debug, Clone)]
pub struct DegradeResult {
    pub image: RgbImage,
    pub params: DegradeParams,
    /// Ground-truth supervision masks when `cfg.masks.any()` is true; `None` by default.
    pub masks: Option<DegradeMasks>,
}

/// Every sampled degradation parameter, written alongside degraded images as a
/// JSON sidecar.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct DegradeParams {
    pub input: String,
    pub output: String,
    pub file_seed: u64,
    pub strength: f32,
    /// Backend that executed the dense stages: `"cpu"` or `"gpu"`.
    pub backend: String,
    /// Pipeline preset: `"full"`, `"reduced"`, or `"custom"`.
    pub preset: String,
    /// Precomputed fast path (`--fast-banks`): tiled noise banks + lighting field.
    pub fast_banks: bool,
    pub width: u32,
    pub height: u32,
    /// Original input width before `max_width` / `train_width` resizing.
    #[serde(default)]
    pub orig_width: u32,
    /// Original input height before `max_width` / `train_width` resizing.
    #[serde(default)]
    pub orig_height: u32,
    // Training-resolution rescale (0 / 1.0 = full-res path)
    pub train_width: u32,
    pub src_width: u32,
    pub src_height: u32,
    pub rescale_k: f32,
    // Paper grain
    pub paper_grain_sigma: f32,
    pub paper_speckle_prob: f32,
    // Page bend / curl
    pub curl_side: String,
    pub curl_px: f32,
    pub curl_shade: f32,
    pub bend_spine_cx: f32,
    pub bend_spine_w: f32,
    pub bend_ripple_fx: f32,
    pub bend_ripple_fy: f32,
    pub bend_ripple_ph: f32,
    // Perspective warp
    pub perspective_max_frac: f32,
    pub corner_offsets: [[f32; 2]; 4],
    pub warp_fit_scale: f32,
    pub warp_out_px: f32,
    // Rotation (canvas expands so the page stays fully visible)
    pub rotation_deg: f32,
    /// True when the GPU backend fused deferred rotation + downscale in one affine dispatch.
    pub rot_scale_fused: bool,
    // Photo framing: desk inset vs tight crop (cuts clamped to quiet margins)
    pub frame_mode: String,
    pub frame_pad_px: [u32; 4],
    pub frame_cut_px: [u32; 4],
    /// Dark-pixel content bbox at framing time `[min_x, min_y, max_x, max_y]`.
    /// Accepts legacy `"frame_notation_bbox"` when deserializing.
    #[serde(alias = "frame_notation_bbox")]
    pub frame_content_bbox: [u32; 4],
    pub frame_quiet_px: [u32; 4],
    pub frame_safe_px: u32,
    // Lens geometry
    pub dist_k1: f32,
    pub dist_k2: f32,
    pub dist_fit_scale: f32,
    // Focus / blur
    pub focus_mode: String,
    pub focus_y: f32,
    pub focus_h: f32,
    pub tilt_sigma: f32,
    pub blur_sigma: f32,
    // Lighting
    pub glare_center: [f32; 2],
    pub glare_radius: f32,
    pub glare_alpha: f32,
    pub shadow_angle_deg: f32,
    pub shadow_strength: f32,
    pub vignette_strength: f32,
    // Chromatic aberration
    pub ca_px: f32,
    // Linked degrade modules
    pub ink: InkParams,
    pub paper: PaperParams,
    pub wear: WearParams,
    pub capture: CaptureParams,
    // Output capture resolution scale
    pub scale_x: f32,
    pub scale_y: f32,
    // Physical artifacts
    pub tear_count: u32,
    pub stain_count: u32,
    pub pencil_count: u32,
    pub fold_corner: String,
    pub fold_creases: Vec<[f32; 3]>,
    pub highlight: String,
    // Sensor + compression
    pub noise_sigma: f32,
    pub jpeg_quality: u8,
    /// Pixel codec backend tag (e.g. `"zune"` or `"image"`).
    pub codec: String,
    // Wave 2 photorealism parameters (backward-compatible via serde(default))
    /// Mixed-illuminant warm-to-cool AWB gradient strength (`0.0` = off).
    #[serde(default)]
    pub awb_warm_cool_strength: f32,
    /// Smartphone ISP unsharp-mask (USM) halo strength (`0.0` = off).
    #[serde(default)]
    pub usm_strength: f32,
    /// Procedural desk surface texture (`"none"`, `"walnut"`, `"slate"`, `"cloth"`).
    #[serde(default)]
    pub desk_surface: String,
    /// Multi-scale 3D paper crumpling normal shading strength (`0.0` = off).
    #[serde(default)]
    pub crumple_strength: f32,
    /// Plastic sheet-protector diagonal specular sheen strength (`0.0` = off).
    #[serde(default)]
    pub sheen_strength: f32,

    // --- Wave 3 (v0.3.0) backward-compatible fields (all #[serde(default)]) ---
    /// Independent per-corner detachment offsets `[dx, dy]` in working-image px (`[TL, TR, BR, BL]`).
    #[serde(default)]
    pub corner_detach_px: [[f32; 2]; 4],
    /// Independent per-corner detachment mode (`"inset"` | `"offset"` | `""` when legacy).
    #[serde(default)]
    pub corner_detach_modes: [String; 4],
    /// Paper micro-weave substrate (`"none"`, `"laid"`, `"wove"`, `"cold_press"`).
    #[serde(default)]
    pub paper_weave_stock: String,
    #[serde(default)]
    pub paper_weave_strength: f32,
    #[serde(default)]
    pub paper_laid_wire_px: f32,
    #[serde(default)]
    pub paper_laid_chain_px: f32,
    #[serde(default)]
    pub paper_wove_anisotropy: f32,
    #[serde(default)]
    pub paper_tooth_light_deg: f32,
    /// Pulp inclusions: curved cellulose fibers, rag threads, recycled flecks, lint & dust.
    #[serde(default)]
    pub cellulose_fiber_count: u32,
    #[serde(default)]
    pub cellulose_fiber_opacity: f32,
    #[serde(default)]
    pub rag_thread_count: u32,
    #[serde(default)]
    pub pulp_fleck_count: u32,
    #[serde(default)]
    pub surface_lint_count: u32,
    #[serde(default)]
    pub dust_mote_count: u32,
    /// Archival chemical paper aging (edge oxidation, ferric foxing, corner finger-oil, watermark).
    #[serde(default)]
    pub edge_oxidation_strength: f32,
    #[serde(default)]
    pub edge_oxidation_depths: [f32; 4],
    #[serde(default)]
    pub foxing_cluster_count: u32,
    #[serde(default)]
    pub foxing_halo_strength: f32,
    #[serde(default)]
    pub grease_patch_count: u32,
    #[serde(default)]
    pub grease_patch_strength: f32,
    #[serde(default)]
    pub watermark_style: String,
    #[serde(default)]
    pub watermark_strength: f32,
    /// Micro-surface paper BRDF stock & key optical parameters.
    #[serde(default)]
    pub paper_stock: String,
    #[serde(default)]
    pub brdf_albedo_ceil: f32,
    #[serde(default)]
    pub brdf_black_floor: f32,
    #[serde(default)]
    pub brdf_oba_strength: f32,
    #[serde(default)]
    pub brdf_spec_gain: f32,
    #[serde(default)]
    pub brdf_ink_spread: f32,
    /// 3D paper cockling, tooth bump, raking light, plate-mark & gutter emboss.
    #[serde(default)]
    pub cockle_relief_strength: f32,
    #[serde(default)]
    pub tooth_bump_strength: f32,
    #[serde(default)]
    pub raking_light_angle_deg: f32,
    #[serde(default)]
    pub raking_elevation_deg: f32,
    #[serde(default)]
    pub plate_mark_strength: f32,
    #[serde(default)]
    pub plate_mark_inset: [f32; 2],
    #[serde(default)]
    pub gutter_emboss_strength: f32,
    /// Camera optics, handheld phone shadow, bend normals, specular bloom, ISP tone & chroma.
    #[serde(default)]
    pub phone_shadow_strength: f32,
    #[serde(default)]
    pub phone_shadow_center: [f32; 2],
    #[serde(default)]
    pub phone_shadow_angle_deg: f32,
    #[serde(default)]
    pub phone_shadow_penumbra: f32,
    #[serde(default)]
    pub bend_normal_strength: f32,
    #[serde(default)]
    pub bend_light_vec: [f32; 3],
    #[serde(default)]
    pub ca_optical_center: [f32; 2],
    #[serde(default)]
    pub ca_cubic_weight: f32,
    #[serde(default)]
    pub field_curvature: f32,
    #[serde(default)]
    pub specular_bloom_strength: f32,
    #[serde(default)]
    pub specular_bloom_center: [f32; 2],
    #[serde(default)]
    pub specular_bloom_radius: f32,
    #[serde(default)]
    pub specular_bloom_aspect: f32,
    #[serde(default)]
    pub specular_bloom_angle_deg: f32,
    #[serde(default)]
    pub tone_black_lift: f32,
    #[serde(default)]
    pub tone_white_ceil: f32,
    #[serde(default)]
    pub tone_contrast: f32,
    #[serde(default)]
    pub chroma_noise_sigma: f32,
    #[serde(default)]
    pub hdr_shadow_boost: f32,
    #[serde(default)]
    pub double_jpeg_q1: u8,
    /// Physical marks, medium physics & environment.
    #[serde(default)]
    pub stain_ink_bleed: f32,
    #[serde(default)]
    pub crease_crack_strength: f32,
    #[serde(default)]
    pub drum_ghost_period_px: u32,
    #[serde(default)]
    pub drum_ghost_alpha: f32,
    #[serde(default)]
    pub thermal_fade: f32,
    #[serde(default)]
    pub thermal_tone: String,
    #[serde(default)]
    pub thermal_scratches: u32,
    #[serde(default)]
    pub scanner_edge_mask: u8,
    #[serde(default)]
    pub scanner_edge_width_px: u32,
    #[serde(default)]
    pub ccd_dust_lines: u32,
    #[serde(default)]
    pub ink_micro_grain_sigma: f32,
    #[serde(default)]
    pub ink_edge_feather_amp: f32,
    #[serde(default)]
    pub ink_halftone_amp: f32,
    #[serde(default)]
    pub stand_ledge_px: u32,
    #[serde(default)]
    pub edge_occluder_kind: String,
    #[serde(default)]
    pub edge_occluder_edge: u8,
    #[serde(default)]
    pub edge_occluder_pen_px: f32,
    #[serde(default)]
    pub tape_style: String,
}

/// Degrade an in-memory `RgbImage` on the CPU backend.
pub fn degrade_image(img: RgbImage, cfg: &DegradeConfig) -> Result<DegradeResult, String> {
    cfg.validate()?;
    #[cfg(feature = "gpu")]
    {
        degrade_image_inner(img, cfg, None)
    }
    #[cfg(not(feature = "gpu"))]
    {
        degrade_image_inner(img, cfg)
    }
}

/// Degrade an in-memory `RgbImage` with an optional shared WGPU [`Gpu`] context.
#[cfg(feature = "gpu")]
pub fn degrade_image_gpu(
    img: RgbImage,
    cfg: &DegradeConfig,
    gpu: Option<&Gpu>,
) -> Result<DegradeResult, String> {
    cfg.validate()?;
    degrade_image_inner(img, cfg, gpu)
}

/// Decode a PNG or JPEG file from disk and degrade it on the CPU backend.
pub fn degrade_file(path: &Path, cfg: &DegradeConfig) -> Result<DegradeResult, String> {
    cfg.validate()?;
    let img = decode_image(path)?;
    let mut res = degrade_image(img, cfg)?;
    res.params.input = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(res)
}

/// Decode a PNG or JPEG file from disk and degrade it with an optional shared
/// WGPU [`Gpu`] context.
#[cfg(feature = "gpu")]
pub fn degrade_file_gpu(
    path: &Path,
    cfg: &DegradeConfig,
    gpu: Option<&Gpu>,
) -> Result<DegradeResult, String> {
    cfg.validate()?;
    let img = decode_image(path)?;
    let mut res = degrade_image_gpu(img, cfg, gpu)?;
    res.params.input = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(res)
}

fn degrade_image_inner(
    mut img: RgbImage,
    cfg: &DegradeConfig,
    #[cfg(feature = "gpu")] gpu: Option<&Gpu>,
) -> Result<DegradeResult, String> {
    let file_seed = cfg.seed;
    let s = cfg.strength.clamp(0.0, 1.0);
    let max_width = cfg.max_width;
    let train_width = cfg.train_width;
    let scale_min = cfg.scale_min;
    let toggles = &cfg.toggles;

    #[cfg(feature = "gpu")]
    let use_gpu = gpu.is_some();
    #[cfg(not(feature = "gpu"))]
    let use_gpu = false;

    // Pre-degrade sizing: cap to `max_width`, then optionally rescale to
    // `train_width` in one bilinear pass.
    let (orig_width, orig_height) = (img.width(), img.height());
    let (mut tw, mut th) = (orig_width, orig_height);
    if max_width > 0 && tw > max_width {
        th = (orig_height as u64 * max_width as u64 / orig_width as u64).max(1) as u32;
        tw = max_width;
    }
    let (src_width, src_height) = (tw, th);
    let rescale_k = if train_width > 0 && tw > train_width {
        th = (th as u64 * train_width as u64 / tw as u64).max(1) as u32;
        tw = train_width;
        train_width as f32 / src_width as f32
    } else {
        1.0f32
    };
    if tw != img.width() || th != img.height() {
        img = resize_rgb_fir(img, tw, th)?;
    }
    let (mut w, mut h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err("empty image".to_string());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(file_seed);
    let backend = if use_gpu { "gpu" } else { "cpu" };

    // Optional per-stage timing probe (set DOC_PHOTO_DEGRADE_TIMING=1 or PHOTO_DEGRADE_TIMING=1).
    let timing = std::env::var("DOC_PHOTO_DEGRADE_TIMING").is_ok()
        || std::env::var("PHOTO_DEGRADE_TIMING").is_ok();
    let mut last = std::time::Instant::now();
    macro_rules! tick {
        ($name:expr) => {
            if timing {
                let now = std::time::Instant::now();
                eprintln!(
                    "stage {:>16}: {:7.1} ms",
                    $name,
                    (now - last).as_secs_f64() * 1000.0
                );
                #[allow(unused_assignments)]
                {
                    last = now;
                }
            }
        };
    }

    // Working buffer in 0..=255 floats, RGB interleaved, plus clean content bbox
    // measured before paper/BRDF/ink/bend/archival darkening or fading.
    let warp_thresh3 = if rescale_k < 1.0 { 600.0 } else { 384.0 };
    let mut clean_bbox_acc = BBoxAcc::default();
    let mut buf: Vec<f32> = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for (x, y, p) in img.enumerate_pixels() {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        buf.push(r);
        buf.push(g);
        buf.push(b);
        clean_bbox_acc.fold(r + g + b, x, y, warp_thresh3);
    }
    let clean_content_bbox = clean_bbox_acc.bbox();

    // Optional pre-degrade clean luma snapshot for `clean_ink` mask (zero alloc when off).
    let clean_luma: Option<Vec<f32>> = if cfg.masks.clean_ink {
        Some(
            img.pixels()
                .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
                .collect(),
        )
    } else {
        None
    };

    tick!("load+maxwidth");

    // --- 1. paper texture: grain + warm tint + sparse speckles ---
    let (paper_grain_sigma, paper_speckle_prob, warm) = if toggles.skip_grain {
        (0.0f32, 0.0f32, [0.0f32; 3])
    } else {
        (
            s * rng.gen_range(2.0..5.0) * rescale_k,
            s * rng.gen_range(0.0005..0.002),
            [
                rng.gen_range(2.0..7.0) * s,
                rng.gen_range(1.0..4.0) * s,
                -rng.gen_range(0.0..3.0) * s,
            ],
        )
    };
    if !toggles.skip_grain {
        #[allow(unused_mut)]
        let mut ran_gpu_grain = false;
        #[cfg(feature = "gpu")]
        if let Some(g) = gpu {
            buf = g.grain(
                &buf,
                w,
                h,
                paper_grain_sigma,
                gpu_backend::grain_seed32(file_seed),
                paper_speckle_prob,
                s,
                warm,
            );
            ran_gpu_grain = true;
        }
        if !ran_gpu_grain {
            if toggles.fast_banks {
                let gauss_bank = banks::generate_gauss_bank(
                    file_seed,
                    banks::TAG_GRAIN,
                    banks::BANK_TILE,
                    paper_grain_sigma,
                );
                let uni_bank =
                    banks::generate_uniform_bank(file_seed, banks::TAG_GRAIN, banks::BANK_TILE);
                let (ox, oy) = banks::bank_offset(&mut rng, banks::BANK_TILE);
                banks::apply_grain_banked(
                    &mut buf,
                    w,
                    h,
                    &gauss_bank,
                    &uni_bank,
                    banks::BANK_TILE,
                    ox,
                    oy,
                    paper_grain_sigma,
                    paper_speckle_prob,
                    s,
                    warm,
                );
            } else {
                for px in buf.as_chunks_mut::<3>().0 {
                    for c in 0..3 {
                        px[c] += gauss(&mut rng) * paper_grain_sigma + warm[c];
                    }
                    if rng.gen::<f32>() < paper_speckle_prob {
                        let d = rng.gen_range(10.0..35.0) * s;
                        px[0] -= d;
                        px[1] -= d;
                        px[2] -= d;
                    }
                }
            }
        }
        clamp_buf(&mut buf);
    }

    tick!("paper-grain");

    // --- 1a.5. laid / wove / cold-press 256×256 seamless micro-weave bank ---
    let weave_s = banks::apply_paper_weave(
        &mut buf,
        w,
        h,
        file_seed,
        s,
        rescale_k,
        &cfg.realism.paper_weave,
    );

    // --- 1b. paper aging (foxing, mottling, fibers, water damage) ---
    let paper_p = paper::apply_with(
        &mut buf,
        w,
        h,
        &mut rng,
        s,
        toggles.coarsen_aging,
        rescale_k,
    );

    // --- 1b2. micro-surface paper BRDF base (albedo, Dmax, OBA, velvet, absorbance) ---
    let brdf_p = paper::SurfaceBrdfParams::sample(
        cfg.realism.paper_stock,
        file_seed,
        s,
        cfg.realism.allow_ink_spread,
    );
    paper::apply_surface_brdf_base(&mut buf, w, h, file_seed, &brdf_p);

    // --- 1b3. Y-sorted curved cellulose fibers, recycled pulp flecks & lint ---
    let pulp_out = paper::apply_pulp_inclusions(
        &mut buf,
        w,
        h,
        file_seed,
        s,
        rescale_k,
        &cfg.realism.pulp_inclusions,
    );

    tick!("paper-aging");

    // --- 1c. ink & print degradation (fading, dropout, bleed-through) ---
    let ink_p = ink::apply_with(&mut buf, w, h, &mut rng, s, toggles.coarsen_ink, rescale_k);

    // --- 1c2. thermal receipt physics + ink micro-texture + porous ink spread ---
    let (thermal_fade, thermal_tone, thermal_scratches) = if cfg.realism.enable_thermal_physics {
        ink::apply_thermal_physics(&mut buf, w, h, &mut rng, s, rescale_k)
    } else {
        (0.0, "none".to_string(), 0)
    };
    let (ink_micro_grain_sigma, ink_edge_feather_amp, ink_halftone_amp) =
        if cfg.realism.enable_ink_micro_tex && s > 0.05 {
            let g_sig = s * if cfg.realism.xerox_heavy {
                rng.gen_range(12.0..20.0)
            } else {
                rng.gen_range(4.0..10.0)
            };
            let f_amp = s * if cfg.realism.archival_paper_scale >= 0.8 {
                rng.gen_range(14.0..24.0)
            } else {
                rng.gen_range(5.0..12.0)
            };
            let h_amp = if cfg.realism.xerox_heavy {
                s * rng.gen_range(14.0..24.0)
            } else {
                0.0
            };
            ink::apply_ink_micro_texture(&mut buf, w, h, file_seed, g_sig, f_amp, h_amp);
            (g_sig, f_amp, h_amp)
        } else {
            (0.0, 0.0, 0.0)
        };
    if brdf_p.ink_spread > 0.01 {
        paper::apply_porous_ink_spread(&mut buf, w, h, brdf_p.ink_spread);
    }

    // --- 1d. archival chemical aging (edge oxidation LUT, ferric foxing, grease, watermark) ---
    let archival_p = paper::apply_archival_paper(
        &mut buf,
        w,
        h,
        &mut rng,
        s,
        cfg.realism.archival_paper_scale,
        toggles.coarsen_aging,
        rescale_k,
    );

    tick!("ink");

    // Pre-bend content bbox from clean input (immune to paper/BRDF/archival/bend shading).
    let use_detached = cfg.corner_detach.enabled && s > 0.05 && w > 4 && h > 4;
    let pre_bend_bbox = clean_content_bbox;

    // --- 2. page bend: edge curl (any side), book spine, or ripple + 3D bend normals ---
    let bend_roll: f32 = rng.gen();
    let bend_mode: &str = if bend_roll < 0.45 {
        "edge"
    } else if bend_roll < 0.70 {
        "spine"
    } else {
        "ripple"
    };
    let curl_side: String = if bend_mode == "edge" {
        ["left", "right", "top", "bottom"][rng.gen_range(0..4)].into()
    } else {
        bend_mode.into()
    };
    let mut curl_px = s * w as f32 * rng.gen_range(0.02..0.08);
    let mut curl_shade = s * rng.gen_range(0.15..0.40);
    if toggles.skip_bend {
        curl_px = 0.0;
        curl_shade = 0.0;
    }
    let (mut bend_spine_cx, mut bend_spine_w) = (0.0f32, 0.0f32);
    let (mut bend_ripple_fx, mut bend_ripple_fy, mut bend_ripple_ph) = (0.0f32, 0.0f32, 0.0f32);
    let bend_active = !toggles.skip_bend && curl_px > 0.25;
    let (bend_lx, bend_ly, bend_lz, bend_normal_strength) =
        if bend_active && cfg.realism.enable_bend_normals {
            let az = rng.gen_range(0.0..std::f32::consts::TAU);
            let lz = rng.gen_range(0.55..0.88f32);
            let xy = (1.0 - lz * lz).max(0.0).sqrt();
            (
                xy * az.cos(),
                xy * az.sin(),
                lz,
                rng.gen_range(0.55..0.92f32),
            )
        } else {
            (0.0, 0.0, 0.8, 0.0)
        };
    let mut warp_acc = BBoxAcc::default();
    #[cfg(feature = "gpu")]
    let (bend_mode_id, bend_edge_id) = if bend_mode == "spine" {
        (0u32, 0u32)
    } else if bend_mode == "ripple" {
        (1u32, 0u32)
    } else {
        let eid = match curl_side.as_str() {
            "right" => 1u32,
            "top" => 2u32,
            "bottom" => 3u32,
            _ => 0u32,
        };
        (2u32, eid)
    };
    if bend_active {
        let wf = w as f32;
        let hf = h as f32;
        bend_ripple_fx = rng.gen_range(1.0..2.5);
        bend_ripple_fy = rng.gen_range(1.0..2.0);
        bend_ripple_ph = rng.gen_range(0.0..std::f32::consts::TAU);
        bend_spine_cx = wf * rng.gen_range(0.35..0.65);
        bend_spine_w = wf * rng.gen_range(0.03..0.08);
        let ripple_fx = bend_ripple_fx;
        let ripple_fy = bend_ripple_fy;
        let ripple_ph = bend_ripple_ph;
        let spine_cx = bend_spine_cx;
        let spine_w = bend_spine_w;
        if !use_gpu {
            let src = buf.clone();
            let min_d = wf.min(hf).max(1.0);
            for y in 0..h {
                let wave = 0.6 + 0.4 * ((std::f32::consts::PI * y as f32 / hf).sin());
                for x in 0..w {
                    let (sx, sy, mut shade, dzx, dzy, ao) = if bend_mode == "spine" {
                        let depth = ((curl_px / wf) * 2.0).min(0.15);
                        let t = (x as f32 - spine_cx) / spine_w.max(1.0);
                        let g = (-t * t).exp();
                        let (sx, sh) = spine_map(x as f32, spine_cx, spine_w, depth, curl_shade);
                        let slope_k = (depth / 0.10).clamp(0.2, 1.8) * 1.15;
                        (
                            sx,
                            y as f32,
                            sh,
                            2.0 * slope_k * t * g,
                            0.0,
                            1.0 - curl_shade * 0.65 * g,
                        )
                    } else if bend_mode == "ripple" {
                        let p1 = std::f32::consts::TAU * ripple_fx * y as f32 / hf + ripple_ph;
                        let p2 = std::f32::consts::TAU * ripple_fy * x as f32 / wf + ripple_ph;
                        let ox = curl_px * 0.5 * p1.sin();
                        let oy = curl_px * 0.3 * p2.cos();
                        let sh = 1.0 - curl_shade * 0.4 * p1.sin().abs();
                        let amp = (curl_px / min_d * 12.0).clamp(0.15, 1.1);
                        (
                            x as f32 + ox,
                            y as f32 + oy,
                            sh,
                            amp * 0.45 * p2.cos(),
                            -amp * 0.75 * p1.sin(),
                            1.0 - curl_shade * 0.25 * p1.sin().abs(),
                        )
                    } else {
                        let (edge, along_wave) = match curl_side.as_str() {
                            "right" => (x as f32 / wf, wave),
                            "top" => (
                                1.0 - y as f32 / hf,
                                0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin()),
                            ),
                            "bottom" => (
                                y as f32 / hf,
                                0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin()),
                            ),
                            _ => (1.0 - x as f32 / wf, wave),
                        };
                        let shift = curl_px * edge * edge * along_wave;
                        let s_e =
                            (curl_px / min_d * 16.0).clamp(0.2, 1.4) * edge * edge * along_wave;
                        let (sx, sy, dzx, dzy) = match curl_side.as_str() {
                            "right" => (x as f32 - shift, y as f32, s_e, 0.0),
                            "top" => (x as f32, y as f32 + shift, 0.0, -s_e),
                            "bottom" => (x as f32, y as f32 - shift, 0.0, s_e),
                            _ => (x as f32 + shift, y as f32, -s_e, 0.0),
                        };
                        (
                            sx,
                            sy,
                            1.0 - curl_shade * edge * edge,
                            dzx,
                            dzy,
                            1.0 - curl_shade * 0.55 * edge * edge,
                        )
                    };
                    if bend_normal_strength > 0.0 {
                        let inv_len = 1.0 / (dzx * dzx + dzy * dzy + 1.0).sqrt();
                        let ndotl = ((-dzx * bend_lx - dzy * bend_ly + bend_lz) * inv_len).max(0.0);
                        let diff_ratio = 0.28 + 0.72 * (ndotl / bend_lz.max(0.25));
                        let lit = 1.0 + bend_normal_strength * (diff_ratio - 1.0);
                        shade = (ao * lit).clamp(0.35, 1.18);
                    }
                    let p = sample_bilinear(&src, w, h, sx, sy);
                    let i = ((y * w + x) * 3) as usize;
                    let q0 = (p[0] * shade).clamp(0.0, 255.0);
                    let q1 = (p[1] * shade).clamp(0.0, 255.0);
                    let q2 = (p[2] * shade).clamp(0.0, 255.0);
                    buf[i] = q0;
                    buf[i + 1] = q1;
                    buf[i + 2] = q2;
                    warp_acc.fold(q0 + q1 + q2, x, y, warp_thresh3);
                }
            }
        }
    }

    tick!("bend");

    // --- 3. perspective warp (+ independent per-corner detachment when enabled) ---
    let perspective_max_frac = s * 0.07;
    let wf = w as f32;
    let hf = h as f32;
    let corners = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];
    let mut persp_raw = [[0.0f64; 2]; 4];
    for slot in &mut persp_raw {
        slot[0] = rng.gen_range(-1.0..1.0) * perspective_max_frac as f64 * wf as f64;
        slot[1] = rng.gen_range(-1.0..1.0) * perspective_max_frac as f64 * hf as f64;
    }

    let frame_min_dim = w.min(h) as f32;
    let frame_safe_px = ((0.002 * frame_min_dim).round() as u32).max(3);
    let mut dst = [(0.0f64, 0.0f64); 4];
    let mut corner_offsets = [[0.0f32; 2]; 4];
    let mut corner_detach_px = [[0.0f32; 2]; 4];
    let mut corner_detach_modes = [String::new(), String::new(), String::new(), String::new()];
    let mut warp_fit_scale = 1.0f32;
    let mut warp_out_px = 0.0f32;
    let mut detached_bent_bbox: Option<[u32; 4]> = None;

    if use_detached {
        // Project pre_bend_bbox through Stage 2 bend so we have exact bent-space
        // content bounds before GPU chain1 runs (zero extra GPU round-trips).
        let bent_bbox = pre_bend_bbox.map(|[x0, y0, x1, y1]| {
            if !bend_active {
                return [x0, y0, x1, y1];
            }
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            let mut visit = |sx: f64, sy: f64| {
                let (bx, by) = reproject::bend_forward_exact_raw(
                    sx,
                    sy,
                    w,
                    h,
                    &curl_side,
                    curl_px,
                    bend_spine_cx,
                    bend_spine_w,
                    bend_ripple_fx,
                    bend_ripple_fy,
                    bend_ripple_ph,
                );
                min_x = min_x.min(bx);
                min_y = min_y.min(by);
                max_x = max_x.max(bx);
                max_y = max_y.max(by);
            };
            for k in 0..=16 {
                let t = k as f64 / 16.0;
                let lx = x0 as f64 + (x1 - x0) as f64 * t;
                let ly = y0 as f64 + (y1 - y0) as f64 * t;
                visit(lx, y0 as f64);
                visit(lx, y1 as f64);
                visit(x0 as f64, ly);
                visit(x1 as f64, ly);
            }
            [
                (min_x.floor().max(0.0) as u32).min(w.saturating_sub(1)),
                (min_y.floor().max(0.0) as u32).min(h.saturating_sub(1)),
                (max_x.ceil().max(0.0) as u32).min(w.saturating_sub(1)),
                (max_y.ceil().max(0.0) as u32).min(h.saturating_sub(1)),
            ]
        });
        detached_bent_bbox = bent_bbox;
        let qm = quiet_margins(w, h, bent_bbox);
        let (d, co, cdp, cdm) = compute_detached_corners(
            w,
            h,
            s,
            &cfg.corner_detach,
            persp_raw,
            bent_bbox,
            qm,
            frame_safe_px,
            &mut rng,
        );
        dst = d;
        corner_offsets = co;
        corner_detach_px = cdp;
        corner_detach_modes = cdm;
        for pt in &dst {
            warp_out_px = warp_out_px.max(
                (-pt.0)
                    .max(pt.0 - wf as f64)
                    .max(-pt.1)
                    .max(pt.1 - hf as f64)
                    .max(0.0) as f32,
            );
        }
    } else {
        // Legacy shrink-to-fit per corner (0.4% bleed sliver).
        let bleed = 0.004 * (wf as f64).min(hf as f64);
        for i in 0..4 {
            let (cx, cy) = corners[i];
            let (mut dx, mut dy) = (persp_raw[i][0], persp_raw[i][1]);
            for (c, dim, d) in [
                (cx as f64, wf as f64, &mut dx),
                (cy as f64, hf as f64, &mut dy),
            ] {
                let lo = -(c + bleed);
                let hi = dim - c + bleed;
                let clamped = (*d).clamp(lo, hi);
                if d.abs() > 1e-9 {
                    warp_fit_scale = warp_fit_scale.min((clamped / *d) as f32);
                }
                *d = clamped;
            }
            dst[i] = (cx as f64 + dx, cy as f64 + dy);
            corner_offsets[i] = [dx as f32, dy as f32];
            warp_out_px = warp_out_px.max(
                (-dst[i].0)
                    .max(dst[i].0 - wf as f64)
                    .max(-dst[i].1)
                    .max(dst[i].1 - hf as f64)
                    .max(0.0) as f32,
            );
        }
    }
    let persp_fill = if use_detached { DESK_BG } else { PAPER_BG };
    let mut persp_hfwd: Option<[f64; 9]> = None;
    #[cfg(feature = "gpu")]
    let mut persp_hinv: Option<[f64; 9]> = None;
    let mut persp_applied = false;
    if use_detached || perspective_max_frac * wf.max(hf) > 0.5 {
        let src_c = corners.map(|(x, y)| (x as f64, y as f64));
        if let Some(hfwd) = solve_homography(src_c, dst) {
            if let Some(hinv) = invert_3x3(hfwd) {
                persp_applied = true;
                persp_hfwd = Some(hfwd);
                #[cfg(feature = "gpu")]
                {
                    persp_hinv = Some(hinv);
                }
                if !use_gpu {
                    warp_acc.reset();
                    let src = buf.clone();
                    for y in 0..h {
                        for x in 0..w {
                            let (sx, sy) = apply_h(&hinv, x as f64, y as f64);
                            let p =
                                sample_bilinear_fill(&src, w, h, sx as f32, sy as f32, persp_fill);
                            let i = ((y * w + x) * 3) as usize;
                            buf[i] = p[0].clamp(0.0, 255.0);
                            buf[i + 1] = p[1].clamp(0.0, 255.0);
                            buf[i + 2] = p[2].clamp(0.0, 255.0);
                        }
                    }
                }
            }
        }
    }
    if !persp_applied {
        corner_offsets = [[0.0; 2]; 4];
        corner_detach_px = [[0.0; 2]; 4];
        warp_fit_scale = 1.0;
        warp_out_px = 0.0;
    }
    // GPU path (chain1): bend + perspective in one resident submit.
    #[cfg(feature = "gpu")]
    if let Some(g) = gpu {
        use doc_photo_degrade_gpu::chain::ChainConfig;
        use doc_photo_degrade_gpu::gpu::BendKnobs;
        let depth = ((curl_px / wf) * 2.0).min(0.15);
        let pre_cfg = ChainConfig {
            bend: BendKnobs {
                mode: bend_mode_id,
                edge_id: bend_edge_id,
                curl_px,
                curl_shade,
                spine_cx: bend_spine_cx,
                spine_w: bend_spine_w,
                depth,
                rip_fx: bend_ripple_fx,
                rip_fy: bend_ripple_fy,
                rip_ph: bend_ripple_ph,
                light_lx: bend_lx,
                light_ly: bend_ly,
                light_lz: bend_lz,
                normal_strength: bend_normal_strength,
            },
            persp_inv: persp_hinv
                .map(|hinv| {
                    [
                        hinv[0] as f32,
                        hinv[1] as f32,
                        hinv[2] as f32,
                        hinv[3] as f32,
                        hinv[4] as f32,
                        hinv[5] as f32,
                        hinv[6] as f32,
                        hinv[7] as f32,
                        hinv[8] as f32,
                    ]
                })
                .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            persp_fill,
            run_bend: bend_active,
            run_persp: persp_hinv.is_some(),
            ..ChainConfig::default()
        };
        if pre_cfg.run_bend || pre_cfg.run_persp {
            let (chained, nw, nh) = g.chain(&buf, w, h, &pre_cfg);
            buf = chained;
            w = nw;
            h = nh;
            clamp_buf(&mut buf);
        }
    }

    tick!("perspective");

    // --- 3a-frame. photo framing: detached per-corner, desk inset, or tight crop ---
    let pre_frame_w = w;
    let pre_frame_h = h;
    let (frame_mode, frame_bbox, frame_quiet_px, frame_pad_px, frame_cut_px) =
        if use_detached && persp_applied {
            let qm = quiet_margins(w, h, detached_bent_bbox);
            ("detached", detached_bent_bbox, qm, [0u32; 4], [0u32; 4])
        } else {
            let frame_roll: f32 = rng.gen();
            let fm: &str = if frame_roll < 0.70 { "inset" } else { "tight" };
            let fb = pre_bend_bbox
                .map(|[x0, y0, x1, y1]| {
                    let mut min_x = f64::INFINITY;
                    let mut min_y = f64::INFINITY;
                    let mut max_x = f64::NEG_INFINITY;
                    let mut max_y = f64::NEG_INFINITY;
                    let mut visit = |sx: f64, sy: f64| {
                        let (bx, by) = if bend_active {
                            reproject::bend_forward_exact_raw(
                                sx,
                                sy,
                                w,
                                h,
                                &curl_side,
                                curl_px,
                                bend_spine_cx,
                                bend_spine_w,
                                bend_ripple_fx,
                                bend_ripple_fy,
                                bend_ripple_ph,
                            )
                        } else {
                            (sx, sy)
                        };
                        let (px, py) = if let Some(hfwd) = persp_hfwd {
                            apply_h(&hfwd, bx, by)
                        } else {
                            (bx, by)
                        };
                        min_x = min_x.min(px);
                        min_y = min_y.min(py);
                        max_x = max_x.max(px);
                        max_y = max_y.max(py);
                    };
                    for k in 0..=16 {
                        let t = k as f64 / 16.0;
                        let lx = x0 as f64 + (x1 - x0) as f64 * t;
                        let ly = y0 as f64 + (y1 - y0) as f64 * t;
                        visit(lx, y0 as f64);
                        visit(lx, y1 as f64);
                        visit(x0 as f64, ly);
                        visit(x1 as f64, ly);
                    }
                    [
                        (min_x.floor().max(0.0) as u32).min(w.saturating_sub(1)),
                        (min_y.floor().max(0.0) as u32).min(h.saturating_sub(1)),
                        (max_x.ceil().max(0.0) as u32).min(w.saturating_sub(1)),
                        (max_y.ceil().max(0.0) as u32).min(h.saturating_sub(1)),
                    ]
                })
                .or_else(|| {
                    let _ = &warp_acc;
                    None
                });
            let fq = quiet_margins(w, h, fb);
            let (mut fp, mut fc) = ([0u32; 4], [0u32; 4]);
            if fm == "inset" {
                for p in fp.iter_mut() {
                    *p = (rng.gen_range(0.008..0.045) * frame_min_dim).round() as u32;
                }
            } else {
                let mut desired = [0u32; 4];
                for d in desired.iter_mut() {
                    *d = (rng.gen_range(0.004..0.028) * frame_min_dim).round() as u32;
                }
                fc = margin_safe_crop(desired, fq, frame_safe_px);
            }
            (fm, fb, fq, fp, fc)
        };

    tick!("framing");

    // --- 3a. page rotation: up to +/-10 deg, canvas expands to fit ---
    let rotation_deg = rng.gen_range(-10.0..10.0) * s;
    let rot_active = rotation_deg.abs() > 0.05 && w > 4 && h > 4;
    #[cfg(feature = "gpu")]
    let mut rot_stash: Option<f32> = None;
    #[cfg(feature = "gpu")]
    let mut rot_scale_fused = false;
    #[cfg(not(feature = "gpu"))]
    let rot_scale_fused = false;
    if rot_active {
        if use_gpu {
            apply_framing_eager(
                &mut buf,
                &mut w,
                &mut h,
                frame_mode,
                frame_pad_px,
                frame_cut_px,
            );
            #[cfg(feature = "gpu")]
            {
                rot_stash = Some(rotation_deg);
            }
        } else {
            let (fw, fh) = framed_dims(w, h, frame_mode, frame_pad_px, frame_cut_px);
            let th = rotation_deg.to_radians();
            let (sn, cs) = (th.sin().abs(), th.cos().abs());
            let (w2, h2) = (
                ((fw as f32) * cs + (fh as f32) * sn).ceil().max(1.0) as u32,
                ((fw as f32) * sn + (fh as f32) * cs).ceil().max(1.0) as u32,
            );
            const DESK: [f32; 3] = [25.0, 25.0, 30.0];
            const PAPER_BG: [f32; 3] = [246.0, 243.0, 236.0];
            let src = std::mem::take(&mut buf);
            let (fwf, fhf) = (fw as f32, fh as f32);
            let (cx0, cy0) = (fwf / 2.0, fhf / 2.0);
            let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
            let (snt, cst) = (th.sin(), th.cos());
            let mut rot = vec![0.0f32; (w2 * h2 * 3) as usize];
            if frame_mode == "inset" {
                let (pl, pt) = (frame_pad_px[0] as i64, frame_pad_px[1] as i64);
                for y in 0..h2 {
                    for x in 0..w2 {
                        let dx = x as f32 - cx1;
                        let dy = y as f32 - cy1;
                        let fx = cx0 + dx * cst + dy * snt;
                        let fy = cy0 - dx * snt + dy * cst;
                        let p = if fx >= 0.0 && fx < fwf && fy >= 0.0 && fy < fhf {
                            let x0f = fx.floor() as i64;
                            let y0f = fy.floor() as i64;
                            let (tx, ty) = (fx - x0f as f32, fy - y0f as f32);
                            let jx0 = x0f - pl;
                            let jy0 = y0f - pt;
                            if jx0 >= 0 && jy0 >= 0 && jx0 + 1 < w as i64 && jy0 + 1 < h as i64 {
                                let j00 = ((jy0 as u32 * w + jx0 as u32) * 3) as usize;
                                let t00 = [src[j00], src[j00 + 1], src[j00 + 2]];
                                let t10 = [src[j00 + 3], src[j00 + 4], src[j00 + 5]];
                                let j01 = j00 + w as usize * 3;
                                let t01 = [src[j01], src[j01 + 1], src[j01 + 2]];
                                let t11 = [src[j01 + 3], src[j01 + 4], src[j01 + 5]];
                                geometry::blend4(t00, t10, t01, t11, tx, ty)
                            } else {
                                sample_remapped(fx, fy, |ix, iy| {
                                    if ix < 0 || iy < 0 || ix >= fw as i64 || iy >= fh as i64 {
                                        PAPER_BG
                                    } else {
                                        let jx = ix - pl;
                                        let jy = iy - pt;
                                        if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                                            DESK
                                        } else {
                                            let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                                            [src[j], src[j + 1], src[j + 2]]
                                        }
                                    }
                                })
                            }
                        } else {
                            DESK
                        };
                        let i = ((y * w2 + x) * 3) as usize;
                        rot[i] = p[0];
                        rot[i + 1] = p[1];
                        rot[i + 2] = p[2];
                    }
                }
            } else {
                let (cl, ct) = (frame_cut_px[0] as i64, frame_cut_px[1] as i64);
                let oob_rim = if frame_mode == "detached" {
                    DESK
                } else {
                    PAPER_BG
                };
                for y in 0..h2 {
                    for x in 0..w2 {
                        let dx = x as f32 - cx1;
                        let dy = y as f32 - cy1;
                        let fx = cx0 + dx * cst + dy * snt;
                        let fy = cy0 - dx * snt + dy * cst;
                        let p = if fx >= 0.0 && fx < fwf && fy >= 0.0 && fy < fhf {
                            let x0f = fx.floor() as i64;
                            let y0f = fy.floor() as i64;
                            let (tx, ty) = (fx - x0f as f32, fy - y0f as f32);
                            if x0f >= 0 && y0f >= 0 && x0f + 1 < fw as i64 && y0f + 1 < fh as i64 {
                                let jx0 = x0f + cl;
                                let jy0 = y0f + ct;
                                let j00 = ((jy0 as u32 * w + jx0 as u32) * 3) as usize;
                                let t00 = [src[j00], src[j00 + 1], src[j00 + 2]];
                                let t10 = [src[j00 + 3], src[j00 + 4], src[j00 + 5]];
                                let j01 = j00 + w as usize * 3;
                                let t01 = [src[j01], src[j01 + 1], src[j01 + 2]];
                                let t11 = [src[j01 + 3], src[j01 + 4], src[j01 + 5]];
                                geometry::blend4(t00, t10, t01, t11, tx, ty)
                            } else {
                                sample_remapped(fx, fy, |ix, iy| {
                                    let jx = ix + cl;
                                    let jy = iy + ct;
                                    if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                                        oob_rim
                                    } else {
                                        let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                                        [src[j], src[j + 1], src[j + 2]]
                                    }
                                })
                            }
                        } else {
                            DESK
                        };
                        let i = ((y * w2 + x) * 3) as usize;
                        rot[i] = p[0];
                        rot[i + 1] = p[1];
                        rot[i + 2] = p[2];
                    }
                }
            }
            buf = rot;
            w = w2;
            h = h2;
        }
    } else {
        apply_framing_eager(
            &mut buf,
            &mut w,
            &mut h,
            frame_mode,
            frame_pad_px,
            frame_cut_px,
        );
    }
    let wf = w as f32;
    let hf = h as f32;

    tick!("rotation");

    // --- 3a2. fused 6-octave phasor 3D micro-relief (crumple + cockle + tooth + plate + gutter) ---
    let relief_out = {
        let (fw_u, fh_u) = framed_dims(
            pre_frame_w,
            pre_frame_h,
            frame_mode,
            frame_pad_px,
            frame_cut_px,
        );
        let w0f = (pre_frame_w as f32).max(1.0);
        let h0f = (pre_frame_h as f32).max(1.0);
        let (ox, oy) = if frame_mode == "inset" {
            (frame_pad_px[0] as f32, frame_pad_px[1] as f32)
        } else {
            (-(frame_cut_px[0] as f32), -(frame_cut_px[1] as f32))
        };
        let (u0, du_dx, du_dy, v0, dv_dx, dv_dy) = if rot_active && !use_gpu {
            let th = rotation_deg.to_radians();
            let (snt, cst) = (th.sin(), th.cos());
            let (cx0, cy0) = (fw_u as f32 * 0.5, fh_u as f32 * 0.5);
            let (cx1, cy1) = (wf * 0.5, hf * 0.5);
            let fx0 = cx0 - cx1 * cst - cy1 * snt - ox;
            let fy0 = cy0 + cx1 * snt - cy1 * cst - oy;
            (
                fx0 / w0f,
                cst / w0f,
                snt / w0f,
                fy0 / h0f,
                -snt / h0f,
                cst / h0f,
            )
        } else {
            (-ox / w0f, 1.0 / w0f, 0.0, -oy / h0f, 0.0, 1.0 / h0f)
        };
        let rctx = realism::MicroReliefContext {
            file_seed,
            strength: s,
            rescale_k,
            u0,
            du_dx,
            du_dy,
            v0,
            dv_dx,
            dv_dy,
            water_present: paper_p.water_present,
            water_u: paper_p.water_x,
            water_v: paper_p.water_y,
            water_r: paper_p.water_radius,
            water_amount: paper_p.water_amount,
            paper_cockle: paper_p.cockle,
            paper_edge: paper_p.edge,
            spine_active: bend_active && bend_mode == "spine",
            spine_u: bend_spine_cx / w0f,
            spine_w: bend_spine_w / w0f,
        };
        realism::apply_micro_relief_shading(&mut buf, w, h, &mut rng, &cfg.realism, &rctx)
    };
    let crumple_strength = relief_out.crumple_strength;

    tick!("crumple");

    // --- 3b. physical artifacts: tears, stains, pencil, folds, highlighter ---
    // Placement is ink-aware: damage goes where document content isn't, so
    // text/symbols remain legible (highlighter is translucent by nature).
    let ink_cell = ((64.0 * rescale_k).round() as u32).max(8);
    let (ink, igw, igh) = ink_grid(&buf, w, h, ink_cell);
    let mut tear_count = 0u32;
    let mut stain_count = 0u32;
    let mut pencil_count = 0u32;
    let mut fold_corner = String::from("none");
    let mut highlight = String::from("none");
    let art_w = w;
    let art_h = h;
    let mut art_mask: Option<Vec<u16>> = if cfg.masks.artifacts {
        Some(vec![0u16; (w as usize) * (h as usize)])
    } else {
        None
    };

    // Paper tears: jagged edge notches carved only from quiet margins.
    let n_tears = if rng.gen::<f32>() < s * 0.5 {
        rng.gen_range(1..=2)
    } else {
        0
    };
    for _ in 0..n_tears {
        let edge = rng.gen_range(0..4);
        for _ in 0..8 {
            let t = rng.gen::<f32>();
            let (gx, gy) = match edge {
                0 => ((t * igw as f32) as u32, 0),
                1 => (igw - 1, (t * igh as f32) as u32),
                2 => ((t * igw as f32) as u32, igh - 1),
                _ => (0, (t * igh as f32) as u32),
            };
            if ink[(gy.min(igh - 1) * igw + gx.min(igw - 1)) as usize] < 0.08 {
                let min_d = w.min(h) as f32;
                let along = t * if edge == 0 || edge == 2 {
                    w as f32
                } else {
                    h as f32
                };
                let mouth = min_d * rng.gen_range(0.01..0.04);
                let depth = min_d * rng.gen_range(0.01..0.03);
                let mask = banks::precompute_notch(along, mouth, depth);
                banks::carve_notch_stamp(&mut buf, w, h, edge, &mask);
                if let Some(ref mut am) = art_mask {
                    let steps = mask.half_widths.len() as u32 - 1;
                    for si in 0..=steps {
                        let f = si as f32 / steps as f32;
                        let hw = mask.half_widths[si as usize];
                        let d = mask.depth * f;
                        let (x0, x1) = ((mask.along - hw) as i64, (mask.along + hw) as i64);
                        for v in x0..=x1 {
                            let (px, py) = match edge {
                                0 => (v, d as i64),
                                2 => (v, (h as f32 - d) as i64),
                                1 => ((w as f32 - d) as i64, v),
                                _ => (d as i64, v),
                            };
                            if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                                am[(py as u32 * w + px as u32) as usize] |= artifact_bits::TEAR;
                            }
                        }
                    }
                }
                tear_count += 1;
                break;
            }
        }
    }

    // Coffee ring + satellite splats, parked on quiet paper (capillary rim + ink bleed).
    let mut stain_ink_bleed = 0.0f32;
    if rng.gen::<f32>() < s * 0.45 {
        if let Some((gx, gy)) = pick_quiet_cell(&mut rng, &ink, igw, igh, 0.06) {
            let jit = 20.0 * rescale_k;
            let cx = (gx * ink_cell + ink_cell / 2) as f32 + rng.gen_range(-jit..jit);
            let cy = (gy * ink_cell + ink_cell / 2) as f32 + rng.gen_range(-jit..jit);
            let r = w.min(h) as f32 * rng.gen_range(0.02..0.06);
            const COFFEE: [f32; 3] = [122.0, 82.0, 42.0];
            if cfg.realism.enable_capillary_stains {
                stain_ink_bleed = if cfg.realism.xerox_heavy {
                    0.0
                } else {
                    s * rng.gen_range(0.14..0.32)
                };
                let st = banks::precompute_capillary_stain(r, &mut rng, stain_ink_bleed);
                banks::stamp_capillary_stain(
                    &mut buf,
                    w,
                    h,
                    cx,
                    cy,
                    &st,
                    art_mask.as_deref_mut(),
                    artifact_bits::STAIN,
                );
            } else {
                let stamp = banks::precompute_coffee(r);
                banks::stamp_coffee(&mut buf, w, h, cx, cy, &stamp);
                if let Some(ref mut am) = art_mask {
                    for &(dx, dy, _) in &stamp.cells {
                        let (px, py) = (cx as i64 + dx as i64, cy as i64 + dy as i64);
                        if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                            am[(py as u32 * w + px as u32) as usize] |= artifact_bits::STAIN;
                        }
                    }
                }
            }
            for _ in 0..rng.gen_range(1..=3) {
                let a = rng.gen_range(0.0..std::f32::consts::TAU);
                let dist = r * rng.gen_range(1.2..2.2);
                let px = cx + dist * a.cos();
                let py = cy + dist * a.sin();
                let pr = (r * rng.gen_range(0.05..0.14)).max(1.0) as i64;
                for dy in -pr..=pr {
                    for dx in -pr..=pr {
                        if dx * dx + dy * dy <= pr * pr {
                            let (sx, sy) = (px as i64 + dx, py as i64 + dy);
                            blend_px(&mut buf, w, h, sx, sy, COFFEE, 0.30);
                            if let Some(ref mut am) = art_mask {
                                if sx >= 0 && sy >= 0 && (sx as u32) < w && (sy as u32) < h {
                                    am[(sy as u32 * w + sx as u32) as usize] |=
                                        artifact_bits::STAIN;
                                }
                            }
                        }
                    }
                }
            }
            stain_count = 1;
        }
    }

    // Pencil markings: thin gray polylines in quiet areas.
    let n_pencil = if rng.gen::<f32>() < s * 0.6 {
        rng.gen_range(1..=3)
    } else {
        0
    };
    const PENCIL: [f32; 3] = [150.0, 150.0, 156.0];
    for _ in 0..n_pencil {
        if let Some((gx, gy)) = pick_quiet_cell(&mut rng, &ink, igw, igh, 0.06) {
            let mut px = (gx * ink_cell) as f32 + rng.gen_range(0.0..ink_cell as f32);
            let mut py = (gy * ink_cell) as f32 + rng.gen_range(0.0..ink_cell as f32);
            let mut ang = rng.gen_range(0.0..std::f32::consts::TAU);
            for _ in 0..rng.gen_range(3..=5) {
                let len = ink_cell as f32 * rng.gen_range(0.15..0.45);
                let qx = px + len * ang.cos();
                let qy = py + len * ang.sin();
                stroke(&mut buf, w, h, px, py, qx, qy, 1, PENCIL);
                if let Some(ref mut am) = art_mask {
                    let steps = ((qx - px).abs().max((qy - py).abs()) as i64).max(1);
                    for si in 0..=steps {
                        let t = si as f32 / steps as f32;
                        let cx = (px + (qx - px) * t) as i64;
                        let cy = (py + (qy - py) * t) as i64;
                        for dy in -1..=1i64 {
                            for dx in -1..=1i64 {
                                if dx * dx + dy * dy <= 1 {
                                    let (sx, sy) = (cx + dx, cy + dy);
                                    if sx >= 0 && sy >= 0 && (sx as u32) < w && (sy as u32) < h {
                                        am[(sy as u32 * w + sx as u32) as usize] |=
                                            artifact_bits::PENCIL;
                                    }
                                }
                            }
                        }
                    }
                }
                px = qx;
                py = qy;
                ang += rng.gen_range(-0.9..0.9);
            }
            pencil_count += 1;
        }
    }

    // Folded corner: shaded dog-ear + crease highlight, quiet corner only.
    if rng.gen::<f32>() < s * 0.35 {
        let leg = (w.min(h) as f32 * rng.gen_range(0.04..0.10)) as u32;
        let specs = [
            ("tl", 0u32, 0u32, false, false),
            ("tr", w.saturating_sub(leg), 0, true, false),
            ("bl", 0, h.saturating_sub(leg), false, true),
            (
                "br",
                w.saturating_sub(leg),
                h.saturating_sub(leg),
                true,
                true,
            ),
        ];
        let mut best: Option<(&str, f32)> = None;
        for (name, x0, y0, _, _) in &specs {
            let gx1 = ((x0 + leg).min(w.saturating_sub(1))) / ink_cell;
            let gy1 = ((y0 + leg).min(h.saturating_sub(1))) / ink_cell;
            let (mut sum, mut n) = (0.0f32, 0u32);
            for gy in (y0 / ink_cell)..=gy1 {
                for gx in (x0 / ink_cell)..=gx1 {
                    sum += ink[(gy * igw + gx) as usize];
                    n += 1;
                }
            }
            let avg = sum / n.max(1) as f32;
            if best.is_none_or(|(_, b)| avg < b) {
                best = Some((name, avg));
            }
        }
        if let Some((name, avg)) = best {
            if avg < 0.10 {
                let (_, x0, y0, mx, my) = *specs.iter().find(|s| s.0 == name).unwrap();
                for dy in 0..leg {
                    for dx in 0..leg {
                        let lx = if mx { leg - 1 - dx } else { dx };
                        let ly = if my { leg - 1 - dy } else { dy };
                        if lx + ly < leg {
                            let pix = ((y0 + dy) as usize) * (w as usize) + ((x0 + dx) as usize);
                            let i = pix * 3;
                            buf[i] *= 0.82;
                            buf[i + 1] *= 0.82;
                            buf[i + 2] *= 0.82;
                            if ((lx + ly) as f32 - leg as f32).abs() < 2.5 {
                                buf[i] = buf[i] * 0.55 + 255.0 * 0.45;
                                buf[i + 1] = buf[i + 1] * 0.55 + 255.0 * 0.45;
                                buf[i + 2] = buf[i + 2] * 0.55 + 255.0 * 0.45;
                            }
                            if let Some(ref mut am) = art_mask {
                                am[pix] |= artifact_bits::FOLD_CREASE;
                            }
                        }
                    }
                }
                fold_corner = name.to_string();
            }
        }
    }

    // Marker highlighter: translucent wash band, content stays readable.
    if rng.gen::<f32>() < s * 0.4 {
        let palette = [
            ("yellow", [255.0, 232.0, 110.0]),
            ("pink", [255.0, 165.0, 195.0]),
            ("green", [165.0, 232.0, 165.0]),
        ];
        let (cname, col) = palette[rng.gen_range(0..3)];
        let bh = (h as f32 * rng.gen_range(0.02..0.05)) as u32;
        let y0 = (h as f32 * rng.gen_range(0.15..0.80)) as u32;
        for y in y0..(y0 + bh).min(h) {
            for x in (w as f32 * 0.10) as u32..((w as f32 * 0.88) as u32).min(w) {
                let pix = (y as usize) * (w as usize) + (x as usize);
                let i = pix * 3;
                for c in 0..3 {
                    buf[i + c] = buf[i + c] * 0.72 + col[c] * 0.28;
                }
                if let Some(ref mut am) = art_mask {
                    am[pix] |= artifact_bits::HIGHLIGHTER;
                }
            }
        }
        highlight = cname.to_string();
    }

    // Paper-fold creases: dihedral bevel + organic wobble + ink-flake cracking.
    let mut fold_creases: Vec<[f32; 3]> = Vec::new();
    let n_crease = if rng.gen::<f32>() < s * 0.5 {
        rng.gen_range(1..=2)
    } else {
        0
    };
    let crease_crack_strength = if n_crease > 0 && cfg.realism.enable_dihedral_creases {
        s * rng.gen_range(0.35..0.75)
    } else {
        0.0
    };
    for crease_idx in 0..n_crease {
        let ang = rng.gen_range(0.0..180.0f32).to_radians();
        let pos = rng.gen_range(0.15..0.85);
        let cstr = rng.gen_range(0.5..1.0) * s;
        let crease_ph = rng.gen_range(0.0..std::f32::consts::TAU);
        let (nx, ny) = (ang.cos(), ang.sin());
        let w_v = (w.min(h) as f32 * 0.0014).clamp(1.0, 3.5);
        let w_r = (w_v * 0.90).clamp(1.0, 3.0);
        let shade_w = w.min(h) as f32 * rng.gen_range(0.015..0.035);
        let lim = (w_v * 2.5 + 2.0).max(shade_w) + 2.0 * rescale_k;
        let off = (pos - 0.5) * w.min(h) as f32;
        let hw = w as f32 / 2.0;
        let hh = h as f32 / 2.0;
        let crack_seed = (file_seed as u32) ^ (crease_idx as u32).wrapping_mul(0x9E37_79B9);
        for y in 0..h {
            let c = ny * (y as f32 - hh) - nx * hw - off;
            let (mut xa, mut xb) = if nx.abs() < 1e-9 {
                if (ny * (y as f32 - hh) - off).abs() < lim + 1e-3 {
                    (0i64, w as i64 - 1)
                } else {
                    continue;
                }
            } else {
                let (mut lo, mut hi) = ((-lim - c) / nx, (lim - c) / nx);
                if lo > hi {
                    std::mem::swap(&mut lo, &mut hi);
                }
                (lo.floor() as i64 - 2, hi.ceil() as i64 + 2)
            };
            xa = xa.max(0);
            xb = xb.min(w as i64 - 1);
            for x in xa..=xb {
                let dx0 = x as f32 - hw;
                let dy0 = y as f32 - hh;
                let along = -dx0 * ny + dy0 * nx;
                let wobble = if cfg.realism.enable_dihedral_creases {
                    (0.65 * (along * 0.025 + crease_ph).sin()
                        + 0.35 * (along * 0.083 + crease_ph * 1.7).sin())
                        * 1.1
                        * rescale_k
                } else {
                    0.0
                };
                let d = dx0 * nx + dy0 * ny - off + wobble;
                let av = d.abs();
                if av > lim {
                    continue;
                }
                let pix = (y * w + x as u32) as usize;
                let i = pix * 3;
                let facet = if d < 0.0 && av < shade_w {
                    let t = 1.0 - av / shade_w;
                    1.0 - 0.11 * cstr * t * t
                } else if d > 0.0 && av < shade_w * 0.6 {
                    let t = 1.0 - av / (shade_w * 0.6);
                    1.0 + 0.03 * cstr * t * t
                } else {
                    1.0
                };
                let valley = (-((d + 0.35 * w_v) / w_v).powi(2)).exp();
                let ridge = (-((d - 0.85 * w_v) / w_r).powi(2)).exp();
                let mult = facet * (1.0 - 0.34 * cstr * valley);
                let spec = 38.0 * cstr * ridge;
                for c in 0..3 {
                    let v = buf[i + c] * mult;
                    let headroom = ((255.0 - v) / 255.0).clamp(0.15, 1.0);
                    buf[i + c] = (v + spec * headroom).clamp(0.0, 255.0);
                }
                if crease_crack_strength > 0.0 && av < w_v * 1.15 {
                    let luma = 0.299 * buf[i] + 0.587 * buf[i + 1] + 0.114 * buf[i + 2];
                    if luma < 165.0 {
                        let core_w = (-(d / (w_v * 0.65)).powi(2)).exp();
                        let q_along = (along * 0.65).floor() as i32;
                        let q_perp = (d * 1.2).floor() as i32;
                        let hh_u = realism::hash2_u32(q_along, q_perp, crack_seed);
                        let thresh = 1.0 - (0.55 * crease_crack_strength * core_w).clamp(0.0, 0.72);
                        if hh_u > thresh {
                            let flake = ((hh_u - thresh) / (1.0 - thresh).max(1e-4)).sqrt()
                                * (0.55 + 0.33 * crease_crack_strength);
                            const FIBER: [f32; 3] = [238.0, 234.0, 224.0];
                            for c in 0..3 {
                                buf[i + c] = buf[i + c] * (1.0 - flake) + FIBER[c] * flake;
                            }
                        }
                    }
                }
                if valley > 0.25 || ridge > 0.25 {
                    if let Some(ref mut am) = art_mask {
                        am[pix] |= artifact_bits::FOLD_CREASE;
                    }
                }
            }
        }
        fold_creases.push([ang.to_degrees(), pos, cstr]);
    }

    tick!("artifacts");

    // --- 3b2. heavy wear: ink loss, printer artifacts, eraser, stains + Xerox drum ghost ---
    let wear_p = wear::apply(&mut buf, w, h, &mut rng, s, rescale_k);
    let (drum_ghost_period_px, drum_ghost_alpha) = if cfg.realism.enable_drum_ghost {
        wear::apply_drum_ghost_pass(&mut buf, w, h, &mut rng, s, rescale_k)
    } else {
        (0, 0.0)
    };

    tick!("wear");

    // --- 3c. reserved radial-lens parameters (clamped no-op; preserves RNG stream) ---
    let mut dist_k1 = s * rng.gen_range(-0.18..0.30);
    let mut dist_k2 = s * rng.gen_range(-0.05..0.05);
    let norm0 = (w.min(h) as f32 / 2.0).max(1.0);
    let r_max = (((w as f32 / 2.0).powi(2) + (h as f32 / 2.0).powi(2)).sqrt() / norm0).max(0.0);
    if dist_k1 > 0.0 {
        dist_k1 = -dist_k1 * (0.18 / 0.30);
    }
    if dist_k2 > 0.0 {
        dist_k2 = -dist_k2;
    }
    let bleed = 0.004 * w.min(h) as f32;
    let mut dist_fit_scale = 1.0f32;
    {
        let mut peak = 0.0f32;
        let mut r = 0.0f32;
        while r <= r_max {
            let f = (1.0 + dist_k1 * r * r + dist_k2 * r * r * r * r).max(0.2);
            peak = peak.max(((1.0 / f - 1.0).max(0.0)) * r * norm0);
            r += 0.05;
        }
        if peak > bleed && peak > 1e-6 {
            dist_fit_scale = (bleed / peak).clamp(0.0, 1.0);
            dist_k1 *= dist_fit_scale;
            dist_k2 *= dist_fit_scale;
        }
    }

    tick!("lens");

    // --- 4. focus: uniform defocus or tilt-shift band + Petzval field curvature ---
    let focus_mode: &str = if rng.gen::<f32>() < 0.5 {
        "uniform"
    } else {
        "tilt"
    };
    let focus_y = rng.gen_range(0.25..0.75);
    let focus_h = rng.gen_range(0.10..0.35);
    let tilt_sigma = s * rng.gen_range(1.0..3.0) * rescale_k;
    let blur_sigma = s * rng.gen_range(0.5..2.4) * rescale_k;
    let (ca_optical_center, ca_cubic_weight, field_curvature) =
        if cfg.realism.enable_radial_optics && s > 0.05 {
            (
                [rng.gen_range(0.44..0.56f32), rng.gen_range(0.44..0.56f32)],
                0.75f32,
                s * rng.gen_range(0.25..0.60),
            )
        } else {
            ([0.5, 0.5], 0.0, 0.0)
        };
    if !use_gpu {
        let tilt_branch = focus_mode == "tilt" && tilt_sigma > 0.05;
        let eff_sigma = if tilt_branch { tilt_sigma } else { blur_sigma };
        if eff_sigma > 0.05 {
            let tmp = vec_to_image(&buf, w, h);
            let blurred = image::imageops::blur(&tmp, eff_sigma);
            if tilt_branch || field_curvature > 0.0 {
                let bb: Vec<f32> = blurred
                    .pixels()
                    .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                    .collect();
                let max_r = (0.5 * wf.hypot(hf)).max(1.0);
                let ocx = wf * ca_optical_center[0];
                let ocy = hf * ca_optical_center[1];
                for y in 0..h {
                    let mut m_row = if tilt_branch {
                        let d = ((y as f32 / h.max(1) as f32) - focus_y).abs();
                        let mt = ((d - focus_h) / 0.08).clamp(0.0, 1.0);
                        mt * mt * (3.0 - 2.0 * mt)
                    } else {
                        0.45
                    };
                    let rny = (y as f32 - ocy) / max_r;
                    for x in 0..w {
                        let m = if field_curvature > 0.0 {
                            let rnx = (x as f32 - ocx) / max_r;
                            let t = (((rnx * rnx + rny * rny) - 0.20) / 0.80).clamp(0.0, 1.0);
                            let m_fc = field_curvature * t * t * (3.0 - 2.0 * t);
                            if tilt_branch {
                                m_row.max(m_fc)
                            } else {
                                (0.45 + 0.55 * m_fc).clamp(0.0, 1.0)
                            }
                        } else {
                            m_row
                        };
                        let i = ((y * w + x) * 3) as usize;
                        for c in 0..3 {
                            buf[i + c] = buf[i + c] * (1.0 - m) + bb[i + c] * m;
                        }
                    }
                    let _ = &mut m_row;
                }
            } else {
                buf = blurred
                    .pixels()
                    .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                    .collect();
            }
        }
    }

    tick!("focus");

    // --- 5. lighting: glare + shadow + vignette + AWB + phone shadow + specular bloom ---
    let gx = rng.gen_range(0.0..wf);
    let gy = rng.gen_range(0.0..hf);
    let glare_radius = wf.min(hf) * rng.gen_range(0.2..0.6);
    let glare_alpha = s * rng.gen_range(0.25..0.60);
    let shadow_angle_deg: f32 = rng.gen_range(0.0..360.0);
    let shadow_strength = s * rng.gen_range(0.15..0.45);
    let vignette_strength = s * rng.gen_range(0.05..0.25);
    let awb_angle_deg: f32 = rng.gen_range(0.0..360.0);
    let awb_warm_cool_strength = if cfg.realism.enable_awb && s > 0.05 {
        s * rng.gen_range(0.18..0.55)
    } else {
        0.0
    };
    let (awb_dx, awb_dy) = (
        awb_angle_deg.to_radians().cos(),
        awb_angle_deg.to_radians().sin(),
    );
    let phone_sh = realism::PhoneShadowParams::sample(&mut rng, s, cfg.realism.enable_phone_shadow);
    let spec_bloom = realism::SpecularBloomParams::sample(
        &mut rng,
        s,
        wf,
        hf,
        cfg.realism.enable_specular_bloom,
    );
    let use_rolloff = spec_bloom.strength > 0.0;
    let ca_sampled = s * rng.gen_range(0.0..1.5) * rescale_k;
    let ca_px = if toggles.skip_chromatic {
        0.0
    } else {
        ca_sampled
    };
    if !use_gpu {
        let theta = shadow_angle_deg.to_radians();
        let (dx, dy) = (theta.cos(), theta.sin());
        let diag = (wf * wf + hf * hf).sqrt().max(1.0);
        let cx = wf / 2.0;
        let cy = hf / 2.0;
        let max_d = (cx * cx + cy * cy).sqrt().max(1.0);
        let inv_wf = 1.0 / wf.max(1.0);
        let inv_hf = 1.0 / hf.max(1.0);
        for y in 0..h {
            let fy = y as f32;
            let vn = fy * inv_hf;
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let fx = x as f32;
                let gd = ((fx - gx).powi(2) + (fy - gy).powi(2)).sqrt();
                let g = (1.0 - gd / glare_radius).max(0.0);
                let add = glare_alpha * 255.0 * g * g;
                let t = ((fx - cx) * dx + (fy - cy) * dy) / diag + 0.5;
                let shadow = 1.0 - shadow_strength * t.clamp(0.0, 1.0);
                let cd = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / max_d;
                let vig = 1.0 - vignette_strength * cd * cd;
                let m = (shadow * vig).max(0.0);
                let ag = realism::awb_channel_gains(
                    fx,
                    fy,
                    cx,
                    cy,
                    diag,
                    awb_dx,
                    awb_dy,
                    awb_warm_cool_strength,
                );
                let ps = realism::phone_shadow_gains(&phone_sh, fx * inv_wf, vn);
                let sp = realism::eval_specular_bloom(&spec_bloom, fx, fy);
                let r = buf[i] * m * ag[0] * ps[0] + add + sp;
                let gg = buf[i + 1] * m * ag[1] * ps[1] + add + sp * 0.98;
                let b = buf[i + 2] * m * ag[2] * ps[2] + add + sp * 0.93;
                if use_rolloff {
                    buf[i] = realism::soft_highlight_rolloff(r);
                    buf[i + 1] = realism::soft_highlight_rolloff(gg);
                    buf[i + 2] = realism::soft_highlight_rolloff(b);
                } else {
                    buf[i] = r.clamp(0.0, 255.0);
                    buf[i + 1] = gg.clamp(0.0, 255.0);
                    buf[i + 2] = b.clamp(0.0, 255.0);
                }
            }
        }
    }

    tick!("lighting");

    // --- 5b. chromatic aberration (legacy linear or off-axis cubic radial) ---
    if ca_px > 0.05 && !use_gpu {
        let src = buf.clone();
        if ca_cubic_weight > 0.0 {
            let ocx = wf * ca_optical_center[0];
            let ocy = hf * ca_optical_center[1];
            let max_r = (0.5 * wf.hypot(hf)).max(1.0);
            for y in 0..h {
                let ny = (y as f32 - ocy) / max_r;
                for x in 0..w {
                    let nx = (x as f32 - ocx) / max_r;
                    let r2 = (nx * nx + ny * ny).min(1.5);
                    let gain = ((1.0 - ca_cubic_weight) + ca_cubic_weight * r2) * ca_px;
                    let sx = nx * gain;
                    let sy = ny * gain;
                    let r = sample_bilinear(&src, w, h, x as f32 - 0.90 * sx, y as f32 - 0.90 * sy);
                    let b = sample_bilinear(&src, w, h, x as f32 + 1.15 * sx, y as f32 + 1.15 * sy);
                    let i = ((y * w + x) * 3) as usize;
                    buf[i] = r[0].clamp(0.0, 255.0);
                    buf[i + 2] = b[2].clamp(0.0, 255.0);
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let dx = (x as f32 - wf / 2.0) / wf.max(1.0);
                    let dy = (y as f32 - hf / 2.0) / hf.max(1.0);
                    let r =
                        sample_bilinear(&src, w, h, x as f32 - dx * ca_px, y as f32 - dy * ca_px);
                    let b =
                        sample_bilinear(&src, w, h, x as f32 + dx * ca_px, y as f32 + dy * ca_px);
                    let i = ((y * w + x) * 3) as usize;
                    buf[i] = r[0].clamp(0.0, 255.0);
                    buf[i + 2] = b[2].clamp(0.0, 255.0);
                }
            }
        }
    }

    tick!("chromatic");

    // --- 5c. capture optics & handling ---
    let mut capture_p = capture::apply(&mut buf, w, h, &mut rng, s, rescale_k);
    if let Some(ref mut am) = art_mask {
        masks::stamp_capture_artifacts(am, art_w, art_h, &capture_p);
    }

    tick!("capture");

    // --- 5d. smartphone ISP edge-aware HDR unsharp-mask (USM) ---
    let (usm_strength, hdr_shadow_boost) =
        realism::apply_usm_hdr(&mut buf, w, h, &mut rng, s, cfg.realism.enable_usm);

    tick!("usm");

    // --- 6. sensor noise (Poisson-Gaussian + 1/4-res chroma blotch) + 6b. capture downscale ---
    let noise_sampled = s * rng.gen_range(3.0..14.0) * rescale_k;
    let noise_sigma = if toggles.skip_noise {
        0.0
    } else {
        noise_sampled
    };
    let (noise_read_sigma, noise_shot_sigma) =
        if cfg.realism.enable_shot_noise && noise_sigma > 0.05 {
            (noise_sigma * 0.65, noise_sigma * 0.75)
        } else {
            (noise_sigma, 0.0)
        };
    let chroma_noise_sigma =
        if !toggles.skip_noise && cfg.realism.enable_chroma_noise && noise_sigma > 0.05 {
            (noise_sigma * 0.85).clamp(0.0, 12.0)
        } else {
            0.0
        };

    let (scale_x, scale_y) = {
        #[cfg(feature = "gpu")]
        if let Some(g) = gpu {
            use doc_photo_degrade_gpu::chain::ChainConfig;
            use doc_photo_degrade_gpu::gpu::LightingKnobs;
            let base_scale = scale_min + rng.gen::<f32>() * (1.0 - scale_min);
            let sx =
                (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
            let sy =
                (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
            let ds_active = sx < 1.0 || sy < 1.0;
            let run_affine = rot_stash.is_some() || ds_active;
            let (affine_rows, affine_meta, affine_nw, affine_nh) = if run_affine {
                let deg = rot_stash.unwrap_or(0.0);
                let (rows, meta, _, _, nw, nh) = gpu_backend::rotscale_frame(deg, w, h, sx, sy);
                (rows, meta, nw.max(1), nh.max(1))
            } else {
                ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0; 4], w, h)
            };
            let theta = shadow_angle_deg.to_radians();
            let tilt_branch = focus_mode == "tilt" && tilt_sigma > 0.05;
            let post_cfg = ChainConfig {
                blur_sigma: if tilt_branch { tilt_sigma } else { blur_sigma },
                field_curvature,
                light: LightingKnobs {
                    gx,
                    gy,
                    glare_radius,
                    glare_alpha,
                    sh_dx: theta.cos(),
                    sh_dy: theta.sin(),
                    shadow_strength,
                    vignette_strength,
                    awb_dx,
                    awb_dy,
                    awb_strength: awb_warm_cool_strength,
                    phone_cx: phone_sh.cx,
                    phone_cy: phone_sh.cy,
                    phone_hw: phone_sh.hw,
                    phone_hh: phone_sh.hh,
                    phone_cos_a: phone_sh.cos_a,
                    phone_sin_a: phone_sh.sin_a,
                    phone_penumbra: phone_sh.penumbra,
                    phone_strength_signed: phone_sh.strength * phone_sh.hand_side,
                    spec_sx: spec_bloom.sx,
                    spec_sy: spec_bloom.sy,
                    spec_inv_radius: spec_bloom.inv_radius,
                    spec_strength: spec_bloom.strength,
                    spec_cos_a: spec_bloom.cos_a,
                    spec_sin_a: spec_bloom.sin_a,
                    spec_aspect: spec_bloom.aspect,
                    spec_rolloff: if use_rolloff { 1.0 } else { 0.0 },
                },
                ca_px,
                ca_optical_center,
                ca_cubic_weight,
                noise_sigma: noise_read_sigma,
                noise_shot_sigma,
                chroma_noise_sigma,
                noise_seed: gpu_backend::noise_seed32(file_seed),
                run_blur: tilt_branch || blur_sigma > 0.05,
                run_light: true,
                run_ca: ca_px > 0.05,
                run_noise: noise_sigma > 0.05 || chroma_noise_sigma > 0.05,
                tilt_active: tilt_branch,
                focus_y,
                focus_h,
                run_affine,
                affine_rows,
                affine_meta,
                affine_nw,
                affine_nh,
                ..ChainConfig::default()
            };
            let (chained, nw, nh) = g.chain(&buf, w, h, &post_cfg);
            buf = chained;
            w = nw;
            h = nh;
            if run_affine {
                rot_scale_fused = rot_stash.is_some();
            }
            clamp_buf(&mut buf);
            (sx, sy)
        } else {
            if chroma_noise_sigma > 0.05 {
                realism::apply_chroma_noise_cpu(&mut buf, w, h, file_seed, chroma_noise_sigma);
            }
            run_cpu_noise_and_downscale(
                &mut buf,
                &mut w,
                &mut h,
                &mut rng,
                file_seed,
                noise_read_sigma,
                noise_shot_sigma,
                scale_min,
                toggles,
            )?
        }
        #[cfg(not(feature = "gpu"))]
        {
            if chroma_noise_sigma > 0.05 {
                realism::apply_chroma_noise_cpu(&mut buf, w, h, file_seed, chroma_noise_sigma);
            }
            run_cpu_noise_and_downscale(
                &mut buf,
                &mut w,
                &mut h,
                &mut rng,
                file_seed,
                noise_read_sigma,
                noise_shot_sigma,
                scale_min,
                toggles,
            )?
        }
    };

    tick!("noise+downscale");

    // --- 6c. procedural desk surface + contact shadow + holes + ledge + occluder + sheen ---
    let desk_surface: String = if cfg.realism.enable_desk_surface && s > 0.05 {
        ["walnut", "oak", "granite_laminate", "slate", "cloth"][rng.gen_range(0..5)].to_string()
    } else {
        "none".to_string()
    };
    if desk_surface != "none" && capture_p.enabled && capture_p.hole_radius > 0 {
        capture_p.hole_desk_through = true;
    }
    let tape_style = if !capture_p.enabled {
        "none".to_string()
    } else if (capture_p.tape_edge + capture_p.sticky_tone).is_multiple_of(3) {
        "clear_frosted".to_string()
    } else {
        "aged_beige".to_string()
    };
    let sheen = realism::SheenParams::sample(
        &mut rng,
        s,
        cfg.realism.enable_sheen,
        cfg.realism.enable_thermal_physics,
    );
    let stand_ledge_px: u32 =
        if cfg.realism.enable_stand_ledge && s > 0.2 && rng.gen::<f32>() < 0.35 {
            let avail_b = frame_quiet_px[3]
                .saturating_sub(frame_cut_px[3])
                .saturating_sub(frame_safe_px);
            let scaled_avail = ((avail_b as f32) * scale_y).floor() as u32;
            let want = (h as f32 * rng.gen_range(0.014..0.032)).round() as u32;
            want.min(scaled_avail)
        } else {
            0
        };
    let tone_curve = realism::ToneCurveParams::sample(&mut rng, s, cfg.realism.enable_tone_curve);

    // --- 7. JPEG compression round-trip (+ optional shifted double-JPEG) ---
    let jpeg_quality: u8 = match cfg.jpeg_quality {
        Some(q) => q,
        None => ((80.0 - s * 30.0 + rng.gen_range(-8.0..8.0)).clamp(30.0, 95.0)) as u8,
    };
    let (double_jpeg_q1, double_jpeg_dx, double_jpeg_dy) =
        if cfg.realism.enable_double_jpeg && jpeg_quality < 95 && rng.gen::<f32>() < 0.28 * s {
            let q1 = rng
                .gen_range(78..=88u8)
                .max(jpeg_quality.saturating_add(6).min(95));
            (q1, rng.gen_range(2..=6u32), rng.gen_range(2..=6u32))
        } else {
            (0u8, 0u32, 0u32)
        };

    let mut params = DegradeParams {
        input: String::new(),
        output: String::new(),
        file_seed,
        strength: s,
        backend: backend.to_string(),
        preset: toggles.preset_name().to_string(),
        fast_banks: toggles.fast_banks,
        width: w,
        height: h,
        orig_width,
        orig_height,
        train_width,
        src_width,
        src_height,
        rescale_k,
        paper_grain_sigma,
        paper_speckle_prob,
        curl_side,
        curl_px,
        curl_shade,
        bend_spine_cx,
        bend_spine_w,
        bend_ripple_fx,
        bend_ripple_fy,
        bend_ripple_ph,
        perspective_max_frac,
        corner_offsets,
        warp_fit_scale,
        warp_out_px,
        rotation_deg,
        rot_scale_fused,
        frame_mode: frame_mode.to_string(),
        frame_pad_px,
        frame_cut_px,
        frame_content_bbox: frame_bbox.unwrap_or([0, 0, 0, 0]),
        frame_quiet_px,
        frame_safe_px,
        dist_k1,
        dist_k2,
        dist_fit_scale,
        focus_mode: focus_mode.to_string(),
        focus_y,
        focus_h,
        tilt_sigma,
        blur_sigma,
        glare_center: [gx, gy],
        glare_radius,
        glare_alpha,
        shadow_angle_deg,
        shadow_strength,
        vignette_strength,
        ca_px,
        ink: ink_p,
        paper: paper_p,
        wear: wear_p,
        capture: capture_p,
        scale_x,
        scale_y,
        tear_count,
        stain_count,
        pencil_count,
        fold_corner,
        fold_creases,
        highlight,
        noise_sigma,
        jpeg_quality,
        codec: "zune".to_string(),
        awb_warm_cool_strength,
        usm_strength,
        desk_surface: desk_surface.clone(),
        crumple_strength,
        sheen_strength: sheen.strength,
        corner_detach_px,
        corner_detach_modes,
        paper_weave_stock: weave_s.stock.as_str().to_string(),
        paper_weave_strength: weave_s.strength,
        paper_laid_wire_px: weave_s.laid_wire_px,
        paper_laid_chain_px: weave_s.laid_chain_px,
        paper_wove_anisotropy: weave_s.wove_anisotropy,
        paper_tooth_light_deg: weave_s.tooth_light_deg,
        cellulose_fiber_count: pulp_out.cellulose_fiber_count,
        cellulose_fiber_opacity: pulp_out.cellulose_fiber_opacity,
        rag_thread_count: pulp_out.rag_thread_count,
        pulp_fleck_count: pulp_out.pulp_fleck_count,
        surface_lint_count: pulp_out.surface_lint_count,
        dust_mote_count: pulp_out.dust_mote_count,
        edge_oxidation_strength: archival_p.edge_oxidation_strength,
        edge_oxidation_depths: archival_p.edge_oxidation_depths,
        foxing_cluster_count: archival_p.foxing_cluster_count,
        foxing_halo_strength: archival_p.foxing_halo_strength,
        grease_patch_count: archival_p.grease_patch_count,
        grease_patch_strength: archival_p.grease_patch_strength,
        watermark_style: archival_p.watermark_style,
        watermark_strength: archival_p.watermark_strength,
        paper_stock: brdf_p.stock.as_str().to_string(),
        brdf_albedo_ceil: brdf_p.albedo_ceil,
        brdf_black_floor: brdf_p.black_floor,
        brdf_oba_strength: brdf_p.oba_strength,
        brdf_spec_gain: brdf_p.spec_gain,
        brdf_ink_spread: brdf_p.ink_spread,
        cockle_relief_strength: relief_out.cockle_relief_strength,
        tooth_bump_strength: relief_out.tooth_bump_strength,
        raking_light_angle_deg: relief_out.raking_light_angle_deg,
        raking_elevation_deg: relief_out.raking_elevation_deg,
        plate_mark_strength: relief_out.plate_mark_strength,
        plate_mark_inset: relief_out.plate_mark_inset,
        gutter_emboss_strength: relief_out.gutter_emboss_strength,
        phone_shadow_strength: phone_sh.strength,
        phone_shadow_center: [phone_sh.cx, phone_sh.cy],
        phone_shadow_angle_deg: phone_sh.angle_deg,
        phone_shadow_penumbra: phone_sh.penumbra,
        bend_normal_strength,
        bend_light_vec: [bend_lx, bend_ly, bend_lz],
        ca_optical_center,
        ca_cubic_weight,
        field_curvature,
        specular_bloom_strength: spec_bloom.strength,
        specular_bloom_center: [spec_bloom.sx, spec_bloom.sy],
        specular_bloom_radius: spec_bloom.radius,
        specular_bloom_aspect: spec_bloom.aspect,
        specular_bloom_angle_deg: spec_bloom.angle_deg,
        tone_black_lift: tone_curve.black_lift,
        tone_white_ceil: tone_curve.white_ceil,
        tone_contrast: tone_curve.contrast,
        chroma_noise_sigma,
        hdr_shadow_boost,
        double_jpeg_q1,
        stain_ink_bleed,
        crease_crack_strength,
        drum_ghost_period_px,
        drum_ghost_alpha,
        thermal_fade,
        thermal_tone,
        thermal_scratches,
        scanner_edge_mask: 0,
        scanner_edge_width_px: 0,
        ccd_dust_lines: 0,
        ink_micro_grain_sigma,
        ink_edge_feather_amp,
        ink_halftone_amp,
        stand_ledge_px,
        edge_occluder_kind: "none".to_string(),
        edge_occluder_edge: 0,
        edge_occluder_pen_px: 0.0,
        tape_style,
    };

    let occluder =
        realism::EdgeOccluder::sample(&mut rng, &params, cfg.realism.enable_edge_occluder);
    params.edge_occluder_kind = occluder.kind.to_string();
    params.edge_occluder_edge = occluder.edge;
    params.edge_occluder_pen_px = occluder.pen_px;

    realism::apply_desk_and_sheen(
        &mut buf,
        &params,
        &desk_surface,
        &sheen,
        stand_ledge_px,
        &occluder,
        file_seed,
    );
    let scan_art = realism::apply_scanner_artifacts(
        &mut buf,
        w,
        h,
        &mut rng,
        s,
        cfg.realism.enable_scanner_bar,
        cfg.realism.xerox_heavy,
    );
    params.scanner_edge_mask = scan_art.edge_mask;
    params.scanner_edge_width_px = scan_art.edge_width_px;
    params.ccd_dust_lines = scan_art.ccd_lines;
    tick!("desk+sheen");

    let mut out_img = if tone_curve.enabled {
        let lut = tone_curve.build_lut();
        vec_to_image_with_lut(&buf, w, h, &lut)
    } else {
        vec_to_image(&buf, w, h)
    };
    if double_jpeg_q1 > 0 && jpeg_quality < 100 {
        if let Some(p1) =
            jpeg_roundtrip_shifted(&out_img, double_jpeg_q1, double_jpeg_dx, double_jpeg_dy)
        {
            out_img = p1;
        }
    }
    if jpeg_quality < 100 {
        if let Ok((jpeg_bytes, _)) = encode_jpeg_bytes(out_img.as_raw(), w, h, jpeg_quality) {
            if let Ok(dec) = decode_jpeg_bytes(&jpeg_bytes, "jpeg round-trip") {
                if dec.width() == w && dec.height() == h {
                    out_img = dec;
                }
            }
        }
    }

    tick!("jpeg");

    let masks = if cfg.masks.any() {
        let mut m = masks::build_masks(
            &cfg.masks,
            &params,
            clean_luma.as_deref(),
            art_mask.as_deref().map(|am| (am, art_w, art_h)),
        );
        if let Some(ref mut art) = m.artifacts {
            realism::stamp_realism_masks(art, &params, &sheen, stand_ledge_px, &occluder);
        }
        Some(m)
    } else {
        None
    };

    Ok(DegradeResult {
        image: out_img,
        params,
        masks,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_noise_and_downscale(
    buf: &mut Vec<f32>,
    w: &mut u32,
    h: &mut u32,
    rng: &mut ChaCha8Rng,
    file_seed: u64,
    noise_read_sigma: f32,
    noise_shot_sigma: f32,
    scale_min: f32,
    toggles: &StageToggles,
) -> Result<(f32, f32), String> {
    let noise_active = noise_read_sigma > 0.05 || noise_shot_sigma > 0.05;
    if toggles.fast_banks && noise_active {
        if noise_shot_sigma > 0.0 {
            let bank =
                banks::generate_gauss_bank(file_seed, banks::TAG_NOISE, banks::BANK_TILE, 1.0);
            let (ox, oy) = banks::bank_offset(rng, banks::BANK_TILE);
            let tile = banks::BANK_TILE;
            let t = tile as usize;
            let n = bank.len();
            let third = n / 3;
            let stride = *w as usize;
            let sr2 = noise_read_sigma * noise_read_sigma;
            let ss2 = noise_shot_sigma * noise_shot_sigma;
            for (pix, px) in buf.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let x = (pix % stride) as u32;
                let y = (pix / stride) as u32;
                let base = ((y + oy) % tile) as usize * t + ((x + ox) % tile) as usize;
                let luma = ((px[0] + px[1] + px[2]) / (3.0 * 255.0)).max(0.01);
                let eff = (sr2 + ss2 * luma).sqrt();
                for c in 0..3 {
                    px[c] += bank[(base + c * third) % n] * eff;
                }
            }
        } else {
            let bank = banks::generate_gauss_bank(
                file_seed,
                banks::TAG_NOISE,
                banks::BANK_TILE,
                noise_read_sigma,
            );
            let (ox, oy) = banks::bank_offset(rng, banks::BANK_TILE);
            banks::apply_noise_banked(buf, *w, &bank, banks::BANK_TILE, ox, oy);
        }
        clamp_buf(buf);
    } else if noise_active {
        if noise_shot_sigma > 0.0 {
            let sr2 = noise_read_sigma * noise_read_sigma;
            let ss2 = noise_shot_sigma * noise_shot_sigma;
            for px in buf.as_chunks_mut::<3>().0 {
                let luma = ((px[0] + px[1] + px[2]) / (3.0 * 255.0)).max(0.01);
                let eff = (sr2 + ss2 * luma).sqrt();
                for ch in px.iter_mut() {
                    *ch += gauss(rng) * eff;
                }
            }
        } else {
            for v in buf.iter_mut() {
                *v += gauss(rng) * noise_read_sigma;
            }
        }
        clamp_buf(buf);
    }
    let base_scale = scale_min + rng.gen::<f32>() * (1.0 - scale_min);
    let sx = (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
    let sy = (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
    if sx < 1.0 || sy < 1.0 {
        let nw = ((*w as f32 * sx) as u32).max(1);
        let nh = ((*h as f32 * sy) as u32).max(1);
        let tmp = vec_to_image(buf, *w, *h);
        let rs = resize_rgb_fir(tmp, nw, nh)?;
        *w = nw;
        *h = nh;
        *buf = rs
            .pixels()
            .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
    }
    Ok((sx, sy))
}

/// Standard-normal sample via Box-Muller.
fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.gen_range(1e-6..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn clamp_buf(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = v.clamp(0.0, 255.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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

    fn test_dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("doc-photo-degrade-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Synthetic document page: horizontal text-line rules plus a dark block on white.
    fn document_input(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let mut img = RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
        for y in (h / 4..h * 3 / 4).step_by(10) {
            for x in w / 8..w * 7 / 8 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        for y in h / 3..h / 3 + 40 {
            for x in w / 3..w / 3 + 60 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let p = dir.join(name);
        img.save(&p).unwrap();
        p
    }

    fn cfg_base(seed: u64, strength: f32) -> DegradeConfig {
        DegradeConfig {
            seed,
            strength,
            jpeg_quality: Some(80),
            max_width: 0,
            train_width: 0,
            scale_min: 1.0,
            toggles: StageToggles::default(),
            realism: RealismConfig::default(),
            corner_detach: CornerDetachConfig::disabled(),
            masks: MaskRequest::none(),
        }
    }

    #[test]
    fn degrade_is_deterministic_and_records_bend_mode() {
        let dir = test_dir("det");
        let input = tiny_input(&dir, "in.png");
        let cfg = cfg_base(42, 0.6);
        let a = degrade_file(&input, &cfg).unwrap();
        let b = degrade_file(&input, &cfg).unwrap();
        assert_eq!(
            a.image.as_raw(),
            b.image.as_raw(),
            "same seed must give same pixels"
        );
        assert!(
            ["left", "right", "top", "bottom", "spine", "ripple"]
                .contains(&a.params.curl_side.as_str()),
            "unexpected bend mode {}",
            a.params.curl_side
        );
        assert_eq!(a.params.jpeg_quality, 80);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn params_json_roundtrip_and_legacy_alias() {
        let dir = test_dir("serde");
        let input = tiny_input(&dir, "in.png");
        let res = degrade_file(&input, &cfg_base(42, 0.6)).unwrap();
        let json = serde_json::to_string(&res.params).unwrap();
        assert!(json.contains("\"frame_content_bbox\""));
        let back: DegradeParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.frame_content_bbox, res.params.frame_content_bbox);
        // Legacy JSON using the `frame_notation_bbox` alias also deserializes cleanly.
        let legacy_json = json.replace("\"frame_content_bbox\"", "\"frame_notation_bbox\"");
        let from_legacy: DegradeParams = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(
            from_legacy.frame_content_bbox,
            res.params.frame_content_bbox
        );
        assert_eq!((res.params.orig_width, res.params.orig_height), (64, 64));

        // Legacy JSON missing `orig_width`/`orig_height` defaults them to 0.
        let stripped = json
            .replace("\"orig_width\":64,", "")
            .replace("\"orig_height\":64,", "");
        let from_stripped: DegradeParams = serde_json::from_str(&stripped).unwrap();
        assert_eq!(
            (from_stripped.orig_width, from_stripped.orig_height),
            (0, 0)
        );

        // Pre-max_width dimensions are preserved in orig_width/orig_height.
        let mut capped_cfg = cfg_base(42, 0.5);
        capped_cfg.max_width = 32;
        let capped = degrade_file(&input, &capped_cfg).unwrap();
        assert_eq!(
            (capped.params.orig_width, capped.params.orig_height),
            (64, 64)
        );
        assert_eq!(
            (capped.params.src_width, capped.params.src_height),
            (32, 32)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lens_never_pulls_inward() {
        let dir = test_dir("lens");
        let input = tiny_input(&dir, "in.png");
        let (w, h) = (64.0f32, 64.0f32);
        let n0 = (w.min(h) / 2.0).max(1.0);
        let rmax = ((w / 2.0).powi(2) + (h / 2.0).powi(2)).sqrt() / n0;
        for seed in 0..40u64 {
            let p = degrade_file(&input, &cfg_base(seed, 0.9)).unwrap().params;
            let mut r = 0.05f32;
            while r <= rmax {
                let f = 1.0 + p.dist_k1 * r * r + p.dist_k2 * r * r * r * r;
                assert!(
                    f <= 1.0 + 1e-6,
                    "seed {seed}: inward pull f={f} at r={r} (k1={}, k2={})",
                    p.dist_k1,
                    p.dist_k2
                );
                r += 0.05;
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn output_scale_within_bounds_and_near_uniform() {
        let dir = test_dir("scale");
        let input = tiny_input(&dir, "in.png");
        let mut cfg = cfg_base(7, 0.6);
        cfg.scale_min = 0.5;
        let res = degrade_file(&input, &cfg).unwrap();
        let p = &res.params;
        assert!((0.48..=1.0).contains(&p.scale_x), "sx={}", p.scale_x);
        assert!((0.48..=1.0).contains(&p.scale_y), "sy={}", p.scale_y);
        let rel = ((p.scale_x - p.scale_y) / ((p.scale_x + p.scale_y) / 2.0)).abs();
        assert!(rel <= 0.051, "x/y variance {rel} exceeds ~5%");
        assert!(
            res.image.width() <= 84 && res.image.width() >= 28,
            "w={}",
            res.image.width()
        );
        assert!(
            res.image.height() <= 84 && res.image.height() >= 28,
            "h={}",
            res.image.height()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn warp_corners_stay_within_sliver_and_rotation_bounded() {
        let dir = test_dir("fit");
        let input = tiny_input(&dir, "in.png");
        let bleed = 0.004 * 64.0 + 1e-3;
        for seed in [1u64, 7, 42, 1234, 99991] {
            let res = degrade_file(&input, &cfg_base(seed, 0.9)).unwrap();
            let p = &res.params;
            assert!((0.0..=1.0).contains(&p.warp_fit_scale));
            assert!(p.warp_out_px <= bleed);
            assert!(p.rotation_deg.abs() <= 9.0 + 1e-3);
            assert!((0.0..=1.0).contains(&p.dist_fit_scale));
            assert!(res.image.width() <= 84 && res.image.height() <= 84);
            for c in p.corner_offsets {
                assert!(c[0].abs() <= 64.0 * 0.07 + 1e-3 && c[1].abs() <= 64.0 * 0.07 + 1e-3);
            }
            assert!(p.fold_creases.len() <= 2);
            for k in &p.fold_creases {
                assert!((0.0..180.0).contains(&k[0]) && (0.15..=0.85).contains(&k[1]));
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn framing_never_cuts_content() {
        let dir = test_dir("frame");
        let input = tiny_input(&dir, "in.png");
        let mut seen = std::collections::HashSet::new();
        for seed in 0..40u64 {
            let p = degrade_file(&input, &cfg_base(seed, 0.9)).unwrap().params;
            seen.insert(p.frame_mode.clone());
            assert!(p.frame_mode == "inset" || p.frame_mode == "tight");
            if p.frame_mode == "inset" {
                assert_eq!(p.frame_cut_px, [0, 0, 0, 0]);
            } else {
                assert_eq!(p.frame_pad_px, [0, 0, 0, 0]);
            }
            for i in 0..4 {
                assert!(
                    p.frame_cut_px[i] + p.frame_safe_px <= p.frame_quiet_px[i],
                    "seed {seed} edge {i}: cut {} + safe {} > quiet {}",
                    p.frame_cut_px[i],
                    p.frame_safe_px,
                    p.frame_quiet_px[i]
                );
            }
            assert!(p.frame_content_bbox[2] > p.frame_content_bbox[0]);
            assert!(p.frame_content_bbox[3] > p.frame_content_bbox[1]);
            assert!(p.frame_safe_px >= 3);
        }
        assert_eq!(seen.len(), 2, "expected both framing modes, got {seen:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn train_width_downscales_and_records_rescale() {
        let dir = test_dir("trainres");
        let input = document_input(&dir, "in.png", 300, 420);
        let mut cfg = cfg_base(42, 0.6);
        cfg.train_width = 200;
        let a = degrade_file(&input, &cfg).unwrap();
        let b = degrade_file(&input, &cfg).unwrap();
        assert_eq!(a.image.as_raw(), b.image.as_raw());
        assert_eq!(
            serde_json::to_string(&a.params).unwrap(),
            serde_json::to_string(&b.params).unwrap()
        );
        assert_eq!(a.params.train_width, 200);
        assert_eq!((a.params.src_width, a.params.src_height), (300, 420));
        let k = 200.0f32 / 300.0;
        assert!((a.params.rescale_k - k).abs() < 1e-6);
        assert_eq!(
            (a.image.width(), a.image.height()),
            (a.params.width, a.params.height)
        );
        assert!(a.image.width() <= 300);
        assert!(a.params.paper_grain_sigma <= 3.0 * k + 1e-4);
        assert!(a.params.noise_sigma <= 14.0 * 0.6 * k + 1e-4);
        assert!(a.params.blur_sigma <= 2.4 * 0.6 * k + 1e-4);
        assert!(a.params.ca_px <= 1.5 * 0.6 * k + 1e-4);
        assert!(a.params.ink.dropout_radius <= 2.2 * k + 1e-4);
        assert!(a.params.ink.dropout_radius >= 1.0 * k - 1e-4);
        assert!(a.params.capture.banding_period <= (7.0 * k).round().max(3.0) as u32);
        let off_p = degrade_file(&input, &cfg_base(42, 0.6)).unwrap().params;
        assert_eq!(off_p.train_width, 0);
        assert!((off_p.rescale_k - 1.0).abs() < 1e-9);
        let narrow = document_input(&dir, "narrow.png", 120, 168);
        let n_p = degrade_file(&narrow, &cfg).unwrap().params;
        assert!((n_p.rescale_k - 1.0).abs() < 1e-9);
        assert_eq!((n_p.src_width, n_p.src_height), (120, 168));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn framing_never_cuts_content_at_train_res() {
        let dir = test_dir("trainframe");
        let input = document_input(&dir, "in.png", 320, 450);
        let mut seen = std::collections::HashSet::new();
        for seed in 0..8u64 {
            let mut cfg = cfg_base(seed, 0.9);
            cfg.train_width = 128;
            let p = degrade_file(&input, &cfg).unwrap().params;
            seen.insert(p.frame_mode.clone());
            assert_eq!(p.train_width, 128);
            assert!((p.rescale_k - 0.4).abs() < 1e-6);
            assert_eq!(p.frame_safe_px, 3);
            for i in 0..4 {
                assert!(p.frame_cut_px[i] <= p.frame_quiet_px[i]);
                if p.frame_cut_px[i] > 0 {
                    assert!(p.frame_cut_px[i] + p.frame_safe_px <= p.frame_quiet_px[i]);
                }
            }
            assert!(
                p.frame_content_bbox[2] > p.frame_content_bbox[0]
                    && p.frame_content_bbox[3] > p.frame_content_bbox[1]
            );
        }
        assert_eq!(seen.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bend_modes_vary_with_seed() {
        let dir = test_dir("modes");
        let input = tiny_input(&dir, "in.png");
        let mut seen = std::collections::HashSet::new();
        for seed in 0..24u64 {
            let p = degrade_file(&input, &cfg_base(seed, 0.9)).unwrap().params;
            seen.insert(p.curl_side);
        }
        assert!(seen.len() >= 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn backend_records_cpu_field() {
        let dir = test_dir("backendfield");
        let input = tiny_input(&dir, "in.png");
        let p = degrade_file(&input, &cfg_base(7, 0.6)).unwrap().params;
        assert_eq!(p.backend, "cpu");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reduced_preset_is_deterministic_and_records_skips() {
        let dir = test_dir("reduced");
        let input = tiny_input(&dir, "in.png");
        let mut cfg = cfg_base(42, 0.6).with_preset(Preset::Reduced);
        assert_eq!(cfg.toggles.preset(), Preset::Reduced);
        let a = degrade_file(&input, &cfg).unwrap();
        let b = degrade_file(&input, &cfg).unwrap();
        assert_eq!(a.image.as_raw(), b.image.as_raw());
        assert_eq!(
            serde_json::to_string(&a.params).unwrap(),
            serde_json::to_string(&b.params).unwrap()
        );
        assert_eq!(a.params.preset, "reduced");
        assert_eq!(a.params.paper_grain_sigma, 0.0);
        assert_eq!(a.params.noise_sigma, 0.0);
        assert_eq!(a.params.ca_px, 0.0);
        assert_eq!(a.params.ink.fade_amount, 0.0);
        assert_eq!(a.params.ink.ghost_alpha, 0.0);
        assert_eq!(a.params.paper.yellowing, 0.0);
        assert!(!a.params.paper.water_present);
        cfg.toggles = StageToggles::full();
        let full = degrade_file(&input, &cfg).unwrap();
        assert_ne!(a.image.as_raw(), full.image.as_raw());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn partial_skips_label_preset_custom() {
        let t = StageToggles {
            skip_noise: true,
            ..StageToggles::default()
        };
        assert_eq!(t.preset(), Preset::Custom);
        assert_eq!(t.preset_name(), "custom");
        assert_eq!(StageToggles::default().preset(), Preset::Full);
    }

    #[test]
    fn fast_banks_is_deterministic_and_labels_itself() {
        let dir = test_dir("fastbanks");
        let input = tiny_input(&dir, "in.png");
        assert!(!StageToggles::default().fast_banks);
        let mut cfg = cfg_base(42, 0.6);
        cfg.toggles.fast_banks = true;
        assert_eq!(cfg.toggles.preset(), Preset::Full);
        let a = degrade_file(&input, &cfg).unwrap();
        let b = degrade_file(&input, &cfg).unwrap();
        assert_eq!(a.image.as_raw(), b.image.as_raw());
        assert_eq!(
            serde_json::to_string(&a.params).unwrap(),
            serde_json::to_string(&b.params).unwrap()
        );
        assert!(a.params.fast_banks);
        assert_eq!(a.params.preset, "full");
        let legacy = degrade_file(&input, &cfg_base(42, 0.6)).unwrap();
        assert!(!legacy.params.fast_banks);
        assert!((a.params.paper_grain_sigma - legacy.params.paper_grain_sigma).abs() < 1e-6);
        assert!((a.params.paper_speckle_prob - legacy.params.paper_speckle_prob).abs() < 1e-6);
        for i in 0..4 {
            assert!(
                a.params.frame_cut_px[i] + a.params.frame_safe_px <= a.params.frame_quiet_px[i]
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "gpu")]
    fn expected_test_dims(p: &DegradeParams) -> (u32, u32) {
        let (mut w, mut h) = framed_dims(64, 64, &p.frame_mode, p.frame_pad_px, p.frame_cut_px);
        if p.rotation_deg.abs() > 0.05 {
            let th = p.rotation_deg.to_radians();
            let (sn, cs) = (th.sin().abs(), th.cos().abs());
            w = ((w as f32) * cs + (h as f32) * sn).ceil().max(1.0) as u32;
            h = ((w as f32) * sn + (h as f32) * cs).ceil().max(1.0) as u32;
        }
        if p.scale_x < 1.0 || p.scale_y < 1.0 {
            w = ((w as f32 * p.scale_x) as u32).max(1);
            h = ((h as f32 * p.scale_y) as u32).max(1);
        }
        (w, h)
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_backend_labels_itself_and_stays_deterministic() {
        let gpu = match Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let dir = test_dir("gpubackend");
        let input = tiny_input(&dir, "in.png");
        let cfg = cfg_base(4242, 0.6);
        let cpu = degrade_file_gpu(&input, &cfg, None).unwrap();
        let gpu_res = degrade_file_gpu(&input, &cfg, Some(&gpu)).unwrap();
        assert_eq!(cpu.params.backend, "cpu");
        assert_eq!(gpu_res.params.backend, "gpu");
        for (tag, p) in [("cpu", &cpu.params), ("gpu", &gpu_res.params)] {
            assert!(p.rotation_deg.abs() <= 10.0 + 1e-4, "{tag} rot");
            assert!(p.warp_out_px >= 0.0, "{tag} warp_out_px");
        }
        let gpu_res2 = degrade_file_gpu(&input, &cfg, Some(&gpu)).unwrap();
        assert_eq!(gpu_res.image.as_raw(), gpu_res2.image.as_raw());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn chain_path_stays_within_dims_and_cut_law() {
        let gpu = match Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let dir = test_dir("chainpath");
        let input = tiny_input(&dir, "in.png");
        for seed in [0u64, 1, 7, 42, 4242, 99991] {
            let cfg = cfg_base(seed, 0.6);
            let cpu = degrade_file_gpu(&input, &cfg, None).unwrap();
            let gpu_res = degrade_file_gpu(&input, &cfg, Some(&gpu)).unwrap();
            assert_eq!(gpu_res.params.backend, "gpu");
            // Check per-backend dims formula (using pre-rotation w/h properly).
            for (tag, res) in [("cpu", &cpu), ("gpu", &gpu_res)] {
                let p = &res.params;
                let (fw, fh) = framed_dims(64, 64, &p.frame_mode, p.frame_pad_px, p.frame_cut_px);
                let (mut ew, mut eh) = (fw, fh);
                if p.rotation_deg.abs() > 0.05 {
                    let th = p.rotation_deg.to_radians();
                    let (sn, cs) = (th.sin().abs(), th.cos().abs());
                    ew = ((fw as f32) * cs + (fh as f32) * sn).ceil().max(1.0) as u32;
                    eh = ((fw as f32) * sn + (fh as f32) * cs).ceil().max(1.0) as u32;
                }
                if p.scale_x < 1.0 || p.scale_y < 1.0 {
                    ew = ((ew as f32 * p.scale_x) as u32).max(1);
                    eh = ((eh as f32 * p.scale_y) as u32).max(1);
                }
                assert_eq!(
                    (res.image.width(), res.image.height()),
                    (ew, eh),
                    "{tag} seed {seed}"
                );
            }
            for i in 0..4 {
                assert!(
                    gpu_res.params.frame_cut_px[i] + gpu_res.params.frame_safe_px
                        <= gpu_res.params.frame_quiet_px[i]
                );
            }
            let gpu_res2 = degrade_file_gpu(&input, &cfg, Some(&gpu)).unwrap();
            assert_eq!(gpu_res.image.as_raw(), gpu_res2.image.as_raw());
        }
        let _ = expected_test_dims;
        std::fs::remove_dir_all(&dir).ok();
    }
}
