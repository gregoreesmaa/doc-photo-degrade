//! Wave 2 & Wave 3 photorealism stages:
//! - Fused 6-octave phasor 3D paper crumpling, moisture cockling, 256×256 raking-light
//!   paper-tooth micro-bump, pressed plate-mark & book-gutter emboss
//! - Handheld smartphone + hand cast shadow SDF with cool skylight fill
//! - Anisotropic glossy paper specular bloom + C1 soft-shoulder highlight rolloff
//! - Filmic smartphone auto-exposure & S-curve 3×256 tone-mapping LUT
//! - 1/4-res High-ISO Cb/Cr chroma blotch noise (CPU mirror of WGPU `NOISE_SHADER`)
//! - Unrolled edge-aware smartphone HDR unsharp-mask (USM) + shadow-recovery lift
//! - Flatbed scanner lid edge shadow bar + 1–2 px CCD/CIS dust streaks
//! - Exact 2-Quad Output-SDF (`QuadSdf2D`) procedural desk surfaces (`walnut`, `oak`,
//!   `granite_laminate`, `slate`, `cloth`) + directional contact drop-shadow
//! - Depth-shadowed desk-through binder punch holes, margin-safe music-stand ledge,
//!   and margin-safe fingertip / binder-clip edge occluders
//! - Plastic sheet-protector / glossy thermal diagonal specular sheen band
//! - Mixed-illuminant AWB color-temperature gradient helper
//!
//! Every effect is 100% deterministic from the seeded RNG / `file_seed` and
//! preserves `< 0.2 px` exact non-linear bend reprojection accuracy.

use crate::banks::{PaperWeaveProfile, TAG_CHROMA};
use crate::geometry::QuadSdf2D;
use crate::masks::artifact_bits;
use crate::paper::{PaperStock, PulpInclusionConfig};
use crate::reproject::ReprojectContext;
use crate::DegradeParams;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Per-effect toggles & paper-physics configs for Wave 2 + Wave 3 photorealism,
/// calibrated per [`crate::DocumentProfile`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealismConfig {
    // --- Wave 2 ---
    /// Mixed-illuminant warm-to-cool AWB color-temperature cast in Stage 5 lighting.
    pub enable_awb: bool,
    /// Linear-light luminance-dependent Poisson shot-noise component in Stage 6 noise.
    pub enable_shot_noise: bool,
    /// Smartphone ISP edge-aware HDR unsharp-mask (USM) micro-contrast haloing.
    pub enable_usm: bool,
    /// Procedural desk surface texture (`walnut`, `oak`, `granite_laminate`, `slate`, `cloth`) + contact shadow.
    pub enable_desk_surface: bool,
    /// Multi-scale 3D paper crumpling normal shading (photometric only).
    pub enable_crumple: bool,
    /// Occasional plastic sheet-protector / thermal-coating diagonal specular sheen band.
    pub enable_sheen: bool,

    // --- Wave 3: Camera Optics & Smartphone ISP ---
    /// Handheld smartphone + hand cast shadow silhouette in Stage 5 lighting.
    pub enable_phone_shadow: bool,
    /// 3D bend-coupled Lambertian normal shading in Stage 2 bend.
    pub enable_bend_normals: bool,
    /// Off-axis cubic radial CA + Petzval field-curvature corner softness.
    pub enable_radial_optics: bool,
    /// Anisotropic glossy paper specular bloom + C1 soft-shoulder highlight rolloff.
    pub enable_specular_bloom: bool,
    /// Filmic smartphone auto-exposure & S-curve 3×256 LUT fused into `vec_to_image`.
    pub enable_tone_curve: bool,
    /// 1/4-res High-ISO Cb/Cr chroma blotch noise in Stage 6 noise.
    pub enable_chroma_noise: bool,
    /// Occasional misaligned 8×8 DCT double-JPEG round-trip (0.00 px drift).
    pub enable_double_jpeg: bool,

    // --- Wave 3: Physical Marks & Print/Scan Medium Physics ---
    /// Capillary coffee/water tide-ring stains with localized ink-bleed feathering.
    pub enable_capillary_stains: bool,
    /// Dihedral fold creases with bevel highlight and ink-flake fiber cracking.
    pub enable_dihedral_creases: bool,
    /// Xerox / laser fuser drum circumference ghost repeat.
    pub enable_drum_ghost: bool,
    /// Direct-thermal receipt uneven fade, leuco-dye cast & heat-friction scratches.
    pub enable_thermal_physics: bool,
    /// Flatbed scanner lid edge shadow bar + 1–2 px CCD/CIS dust streaks.
    pub enable_scanner_bar: bool,
    /// Heavy photocopy/Xerox mode for scanner bar & halftone screen.
    pub xerox_heavy: bool,
    /// Ink-only micro-feathering & halftone/toner grain.
    pub enable_ink_micro_tex: bool,

    // --- Wave 3: Paper Textures, Micro-Surface BRDF & 3D Micro-Relief ---
    /// 256×256 seamless torus laid/wove/cold-press micro-weave profile.
    pub paper_weave: PaperWeaveProfile,
    /// Y-sorted curved cellulose fibers, recycled pulp flecks & surface lint config.
    pub pulp_inclusions: PulpInclusionConfig,
    /// Physical paper stock for micro-surface BRDF (albedo, Dmax, OBA, velvet, ink spread).
    pub paper_stock: PaperStock,
    /// Allow porous capillary ink spread on `PorousNewsprint` / `UncoatedRag`.
    pub allow_ink_spread: bool,
    /// Stage 1d archival chemical paper aging intensity scale (`0.0` = off, `0.0..=1.0`).
    pub archival_paper_scale: f32,
    /// Moisture-induced 3D paper cockling relief fused in Stage 3a2.
    pub enable_cockle_relief: bool,
    /// 256×256 raking-light paper-tooth micro-bump (ink-core protected) fused in Stage 3a2.
    pub enable_tooth_bump: bool,
    /// Subtle pressed intaglio/engraving plate-mark bevel in Stage 3a2.
    pub enable_plate_mark: bool,
    /// Pressed book-gutter / center-crease emboss slope in Stage 3a2.
    pub enable_gutter_emboss: bool,

    // --- Wave 3: Physical Environment ---
    /// Margin-safe fingertip / metal binder-clip edge occluder.
    pub enable_edge_occluder: bool,
    /// Optional margin-safe music-stand bottom metal ledge.
    pub enable_stand_ledge: bool,
}

impl Default for RealismConfig {
    fn default() -> Self {
        Self::phone()
    }
}

impl RealismConfig {
    /// Full smartphone-capture realism (all handheld camera, paper & environment effects).
    pub const fn phone() -> Self {
        Self {
            enable_awb: true,
            enable_shot_noise: true,
            enable_usm: true,
            enable_desk_surface: true,
            enable_crumple: true,
            enable_sheen: true,
            enable_phone_shadow: true,
            enable_bend_normals: true,
            enable_radial_optics: true,
            enable_specular_bloom: true,
            enable_tone_curve: true,
            enable_chroma_noise: true,
            enable_double_jpeg: true,
            enable_capillary_stains: true,
            enable_dihedral_creases: true,
            enable_drum_ghost: false,
            enable_thermal_physics: false,
            enable_scanner_bar: false,
            xerox_heavy: false,
            enable_ink_micro_tex: true,
            paper_weave: PaperWeaveProfile::phone(),
            pulp_inclusions: PulpInclusionConfig::phone(),
            paper_stock: PaperStock::MatteOfficeBond,
            allow_ink_spread: true,
            archival_paper_scale: 0.0,
            enable_cockle_relief: true,
            enable_tooth_bump: true,
            enable_plate_mark: false,
            enable_gutter_emboss: true,
            enable_edge_occluder: true,
            enable_stand_ledge: false,
        }
    }

    /// Flatbed scanner / photocopy mode (camera/desk/crumple/optics disabled).
    pub const fn flat_scanner() -> Self {
        Self {
            enable_awb: false,
            enable_shot_noise: false,
            enable_usm: false,
            enable_desk_surface: false,
            enable_crumple: false,
            enable_sheen: false,
            enable_phone_shadow: false,
            enable_bend_normals: false,
            enable_radial_optics: false,
            enable_specular_bloom: false,
            enable_tone_curve: false,
            enable_chroma_noise: false,
            enable_double_jpeg: false,
            enable_capillary_stains: true,
            enable_dihedral_creases: true,
            enable_drum_ghost: false,
            enable_thermal_physics: false,
            enable_scanner_bar: true,
            xerox_heavy: false,
            enable_ink_micro_tex: true,
            paper_weave: PaperWeaveProfile::office_scan(),
            pulp_inclusions: PulpInclusionConfig::office_scan(),
            paper_stock: PaperStock::MatteOfficeBond,
            allow_ink_spread: false,
            archival_paper_scale: 0.0,
            enable_cockle_relief: false,
            enable_tooth_bump: false,
            enable_plate_mark: false,
            enable_gutter_emboss: false,
            enable_edge_occluder: false,
            enable_stand_ledge: false,
        }
    }

    /// High-throughput reduced preset for large training runs (skips USM, heavy
    /// relief, double-JPEG, micro-weave, and pulp inclusions while keeping fast
    /// lighting/desk/geometry realism).
    pub const fn reduced() -> Self {
        Self {
            enable_awb: true,
            enable_shot_noise: false,
            enable_usm: false,
            enable_desk_surface: true,
            enable_crumple: false,
            enable_sheen: false,
            enable_phone_shadow: true,
            enable_bend_normals: true,
            enable_radial_optics: false,
            enable_specular_bloom: false,
            enable_tone_curve: true,
            enable_chroma_noise: false,
            enable_double_jpeg: false,
            enable_capillary_stains: true,
            enable_dihedral_creases: true,
            enable_drum_ghost: false,
            enable_thermal_physics: false,
            enable_scanner_bar: false,
            xerox_heavy: false,
            enable_ink_micro_tex: false,
            paper_weave: PaperWeaveProfile::smooth(),
            pulp_inclusions: PulpInclusionConfig::disabled(),
            paper_stock: PaperStock::None,
            allow_ink_spread: false,
            archival_paper_scale: 0.0,
            enable_cockle_relief: false,
            enable_tooth_bump: false,
            enable_plate_mark: false,
            enable_gutter_emboss: false,
            enable_edge_occluder: false,
            enable_stand_ledge: false,
        }
    }
}

/// Apply mixed-illuminant warm-to-cool AWB color-temperature gradient on CPU
/// (mirrors the WGPU `LIGHTING_SHADER` `prm.d` branch bit-for-bit).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn awb_channel_gains(
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    diag: f32,
    awb_dx: f32,
    awb_dy: f32,
    awb_strength: f32,
) -> [f32; 3] {
    if awb_strength <= 0.0 {
        return [1.0, 1.0, 1.0];
    }
    let awb_t = (((fx - cx) * awb_dx + (fy - cy) * awb_dy) / diag.max(1e-6)).clamp(-0.5, 0.5) * 2.0;
    let awb = awb_strength * awb_t;
    [1.0 + 0.14 * awb, 1.0 + 0.02 * awb, 1.0 - 0.16 * awb]
}

// ===========================================================================
// Handheld Smartphone + Hand Cast Shadow Silhouette
// ===========================================================================

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PhoneShadowParams {
    pub strength: f32,
    pub cx: f32,
    pub cy: f32,
    pub hw: f32,
    pub hh: f32,
    pub angle_deg: f32,
    pub cos_a: f32,
    pub sin_a: f32,
    pub penumbra: f32,
    pub hand_side: f32,
}

impl PhoneShadowParams {
    pub(crate) fn sample(rng: &mut ChaCha8Rng, s: f32, enabled: bool) -> Self {
        if !enabled || s <= 0.05 || rng.gen::<f32>() >= 0.42 {
            return Self::default();
        }
        let angle_deg = rng.gen_range(-22.0..22.0f32);
        let rad = angle_deg.to_radians();
        Self {
            strength: s * rng.gen_range(0.18..0.42),
            cx: rng.gen_range(0.25..0.75),
            cy: rng.gen_range(0.68..0.92),
            hw: rng.gen_range(0.11..0.18),
            hh: rng.gen_range(0.16..0.26),
            angle_deg,
            cos_a: rad.cos(),
            sin_a: rad.sin(),
            penumbra: rng.gen_range(0.038..0.095),
            hand_side: if rng.gen_bool(0.5) { 1.0 } else { -1.0 },
        }
    }
}

#[inline]
pub(crate) fn phone_shadow_gains(p: &PhoneShadowParams, u: f32, v: f32) -> [f32; 3] {
    if p.strength <= 0.0 {
        return [1.0, 1.0, 1.0];
    }
    let dx = u - p.cx;
    let dy = v - p.cy;
    let lx = dx * p.cos_a + dy * p.sin_a;
    let ly = -dx * p.sin_a + dy * p.cos_a;
    let r = 0.025;
    let qx = lx.abs() - (p.hw - r);
    let qy = ly.abs() - (p.hh - r);
    let d_phone = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r;
    let ax = p.hand_side * p.hw * 0.85;
    let ay = p.hh * 0.15;
    let bx = p.hand_side * (p.hw + 0.24);
    let by = p.hh + 0.35;
    let pax = lx - ax;
    let pay = ly - ay;
    let bax = bx - ax;
    let bay = by - ay;
    let h = ((pax * bax + pay * bay) / (bax * bax + bay * bay).max(1e-6)).clamp(0.0, 1.0);
    let d_hand = (pax - bax * h).hypot(pay - bay * h) - (0.095 + 0.045 * h);
    let k = 0.045;
    let sm = (k - (d_phone - d_hand).abs()).max(0.0) / k;
    let d = d_phone.min(d_hand) - sm * sm * k * 0.25;
    let z = 1.8 * d / p.penumbra.max(1e-4);
    let occ = 0.5 * (1.0 - z / (1.0 + z * z).sqrt());
    [
        1.0 - p.strength * occ * 1.08,
        1.0 - p.strength * occ * 1.00,
        1.0 - p.strength * occ * 0.86,
    ]
}

// ===========================================================================
// Anisotropic Glossy Specular Bloom + Soft Highlight Rolloff
// ===========================================================================

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SpecularBloomParams {
    pub strength: f32,
    pub sx: f32,
    pub sy: f32,
    pub radius: f32,
    pub inv_radius: f32,
    pub angle_deg: f32,
    pub cos_a: f32,
    pub sin_a: f32,
    pub aspect: f32,
}

impl SpecularBloomParams {
    pub(crate) fn sample(rng: &mut ChaCha8Rng, s: f32, wf: f32, hf: f32, enabled: bool) -> Self {
        if !enabled || s <= 0.05 || rng.gen::<f32>() >= 0.45 {
            return Self::default();
        }
        let min_d = wf.min(hf).max(1.0);
        let radius = (min_d * rng.gen_range(0.16..0.36)).max(8.0);
        let angle_deg = rng.gen_range(0.0..180.0f32);
        let rad = angle_deg.to_radians();
        Self {
            strength: s * rng.gen_range(0.22..0.58),
            sx: rng.gen_range(0.15..0.85) * wf,
            sy: rng.gen_range(0.15..0.85) * hf,
            radius,
            inv_radius: 1.0 / radius,
            angle_deg,
            cos_a: rad.cos(),
            sin_a: rad.sin(),
            aspect: rng.gen_range(1.2..2.2),
        }
    }
}

#[inline]
pub(crate) fn eval_specular_bloom(p: &SpecularBloomParams, fx: f32, fy: f32) -> f32 {
    if p.strength <= 0.0 {
        return 0.0;
    }
    let dx = fx - p.sx;
    let dy = fy - p.sy;
    let u = (dx * p.cos_a + dy * p.sin_a) * p.inv_radius;
    let v = (-dx * p.sin_a + dy * p.cos_a) * p.inv_radius * p.aspect;
    let d2 = u * u + v * v;
    let k_core = 1.0 / (1.0 + 4.5 * d2);
    let core = k_core * k_core;
    let veil = 1.0 / (1.0 + 0.65 * d2);
    p.strength * 255.0 * (0.72 * core + 0.28 * veil)
}

#[inline]
pub(crate) fn soft_highlight_rolloff(v: f32) -> f32 {
    if v <= 210.0 {
        v.max(0.0)
    } else {
        let ex = v - 210.0;
        (210.0 + 45.0 * ex / (45.0 + ex)).min(255.0)
    }
}

// ===========================================================================
// Filmic Smartphone Auto-Exposure & S-Curve 3×256 Tone-Mapping LUT
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToneCurveParams {
    pub enabled: bool,
    pub black_lift: f32,
    pub white_ceil: f32,
    pub contrast: f32,
    pub gamma_ev: f32,
    pub wb_shift: [f32; 3],
}

impl Default for ToneCurveParams {
    fn default() -> Self {
        Self {
            enabled: false,
            black_lift: 0.0,
            white_ceil: 255.0,
            contrast: 0.0,
            gamma_ev: 0.0,
            wb_shift: [1.0; 3],
        }
    }
}

impl ToneCurveParams {
    pub(crate) fn sample(rng: &mut ChaCha8Rng, s: f32, enabled: bool) -> Self {
        if !enabled || s <= 0.05 {
            return Self::default();
        }
        Self {
            enabled: true,
            black_lift: s * rng.gen_range(10.0..24.0),
            white_ceil: 255.0 - s * rng.gen_range(4.0..16.0),
            contrast: s * rng.gen_range(0.08..0.26),
            gamma_ev: s * rng.gen_range(-0.10..0.10),
            wb_shift: [
                1.0 + s * rng.gen_range(-0.030..0.040),
                1.0,
                1.0 + s * rng.gen_range(-0.040..0.030),
            ],
        }
    }

    pub(crate) fn build_lut(&self) -> [[u8; 256]; 3] {
        let mut lut = [[0u8; 256]; 3];
        let gamma = 2.0f32.powf(-self.gamma_ev);
        let span = (self.white_ceil - self.black_lift).max(1.0);
        for (c, chan_lut) in lut.iter_mut().enumerate() {
            let wb = self.wb_shift[c];
            for (i, slot) in chan_lut.iter_mut().enumerate() {
                let x = ((i as f32 / 255.0) * wb).clamp(0.0, 1.0);
                let ae = x.powf(gamma);
                let s_curve =
                    (ae - self.contrast * ae * (1.0 - ae) * (1.0 - 2.0 * ae)).clamp(0.0, 1.0);
                let y = self.black_lift + s_curve * span;
                *slot = y.round().clamp(0.0, 255.0) as u8;
            }
        }
        lut
    }
}

// ===========================================================================
// 1/4-Res High-ISO Cb/Cr Chroma Blotch Noise (CPU mirror of WGPU NOISE_SHADER)
// ===========================================================================

pub(crate) fn apply_chroma_noise_cpu(
    buf: &mut [f32],
    w: u32,
    h: u32,
    file_seed: u64,
    chroma_sigma: f32,
) {
    if chroma_sigma <= 0.05 || w < 4 || h < 4 || buf.len() < (w as usize) * (h as usize) * 3 {
        return;
    }
    let gw = (w / 4 + 2) as usize;
    let gh = (h / 4 + 2) as usize;
    let mut rng = ChaCha8Rng::seed_from_u64(crate::banks::bank_seed(file_seed, TAG_CHROMA));
    let mut grid = vec![[0.0f32; 2]; gw * gh];
    for cell in &mut grid {
        let cb = (rng.gen::<f32>() + rng.gen::<f32>() - 1.0) * 2.449_489_7;
        let cr = (rng.gen::<f32>() + rng.gen::<f32>() - 1.0) * 2.449_489_7;
        *cell = [cb, cr];
    }
    let w_us = w as usize;
    for y in 0..h as usize {
        let gy = y as f32 * 0.25;
        let y0 = (gy as usize).min(gh - 2);
        let ty = gy - y0 as f32;
        let r0 = y0 * gw;
        let r1 = (y0 + 1) * gw;
        let row = &mut buf[y * w_us * 3..(y + 1) * w_us * 3];
        for (x, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let gx = x as f32 * 0.25;
            let x0 = (gx as usize).min(gw - 2);
            let tx = gx - x0 as f32;
            let c00 = grid[r0 + x0];
            let c10 = grid[r0 + x0 + 1];
            let c01 = grid[r1 + x0];
            let c11 = grid[r1 + x0 + 1];
            let dcb = (c00[0] * (1.0 - tx) + c10[0] * tx) * (1.0 - ty)
                + (c01[0] * (1.0 - tx) + c11[0] * tx) * ty;
            let dcr = (c00[1] * (1.0 - tx) + c10[1] * tx) * (1.0 - ty)
                + (c01[1] * (1.0 - tx) + c11[1] * tx) * ty;
            let y_norm = ((px[0] + px[1] + px[2]) * (1.0 / 765.0)).clamp(0.0, 1.0);
            let sh_w = chroma_sigma * (1.0 - 0.65 * y_norm);
            px[0] = (px[0] + 1.402 * dcr * sh_w).clamp(0.0, 255.0);
            px[1] = (px[1] + (-0.344_136 * dcb - 0.714_136 * dcr) * sh_w).clamp(0.0, 255.0);
            px[2] = (px[2] + 1.772 * dcb * sh_w).clamp(0.0, 255.0);
        }
    }
}

// ===========================================================================
// Fused 6-Octave Phasor 3D Micro-Relief Shading (Crumple + Cockle + Tooth + Plate + Gutter)
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub(crate) struct MicroReliefContext {
    pub file_seed: u64,
    pub strength: f32,
    pub rescale_k: f32,
    pub u0: f32,
    pub du_dx: f32,
    pub du_dy: f32,
    pub v0: f32,
    pub dv_dx: f32,
    pub dv_dy: f32,
    pub water_present: bool,
    pub water_u: f32,
    pub water_v: f32,
    pub water_r: f32,
    pub water_amount: f32,
    pub paper_cockle: f32,
    pub paper_edge: f32,
    pub spine_active: bool,
    pub spine_u: f32,
    pub spine_w: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MicroReliefOutcome {
    pub crumple_strength: f32,
    pub cockle_relief_strength: f32,
    pub tooth_bump_strength: f32,
    pub raking_light_angle_deg: f32,
    pub raking_elevation_deg: f32,
    pub plate_mark_strength: f32,
    pub plate_mark_inset: [f32; 2],
    pub gutter_emboss_strength: f32,
}

#[inline]
fn smoothstep01(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fused single-pass 3D micro-relief shading using 6-octave complex phasor
/// recurrence (0 inner-loop transcendentals) + L2-resident 256×256 tooth bump.
/// Purely photometric — zero geometric warp (`< 0.2 px` reprojection preserved).
pub(crate) fn apply_micro_relief_shading(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    cfg: &RealismConfig,
    ctx: &MicroReliefContext,
) -> MicroReliefOutcome {
    let mut out = MicroReliefOutcome::default();
    let any = cfg.enable_crumple
        || cfg.enable_cockle_relief
        || cfg.enable_tooth_bump
        || cfg.enable_plate_mark
        || cfg.enable_gutter_emboss;
    if !any || ctx.strength <= 0.05 || w < 8 || h < 8 {
        return out;
    }

    let s = ctx.strength;
    let min_dim = (w.min(h) as f32).max(1.0);
    let light_deg = rng.gen_range(0.0..360.0f32);
    let elev_deg = rng.gen_range(14.0..48.0f32);
    let light_rad = light_deg.to_radians();
    let (lx, ly) = (light_rad.cos(), light_rad.sin());
    let raking_gain = (1.0 / elev_deg.to_radians().tan()).clamp(0.65, 1.85);
    out.raking_light_angle_deg = light_deg;
    out.raking_elevation_deg = elev_deg;

    let crumple_str = if cfg.enable_crumple {
        s * rng.gen_range(0.08..0.20)
    } else {
        0.0
    };
    let cockle_str = if cfg.enable_cockle_relief {
        let moist_boost =
            0.55 + 0.45 * ctx.paper_cockle + if ctx.water_present { 0.35 } else { 0.0 };
        (s * rng.gen_range(0.06..0.16) * moist_boost).min(0.24)
    } else {
        0.0
    };
    out.crumple_strength = crumple_str;
    out.cockle_relief_strength = cockle_str;

    let scales = [1.4f32, 2.7, 4.9, 8.2, 5.5, 11.0];
    let amps = [0.42f32, 0.28, 0.18, 0.12, 0.62, 0.38];
    let mut wave_ky = [0.0f32; 6];
    let mut wave_ph = [0.0f32; 6];
    let mut wave_dc = [1.0f32; 6];
    let mut wave_ds = [0.0f32; 6];
    let mut wave_w = [0.0f32; 6];
    for i in 0..6 {
        let ang = rng.gen_range(0.0..std::f32::consts::TAU);
        let k = scales[i] * std::f32::consts::TAU / min_dim;
        let (kx, ky) = (k * ang.cos(), k * ang.sin());
        let ph = rng.gen_range(0.0..std::f32::consts::TAU);
        let dir_dot = ang.cos() * lx + ang.sin() * ly;
        wave_ky[i] = ky;
        wave_ph[i] = ph;
        wave_dc[i] = kx.cos();
        wave_ds[i] = kx.sin();
        wave_w[i] = amps[i] * dir_dot;
    }

    let tooth_str = if cfg.enable_tooth_bump {
        s * rng.gen_range(0.015..0.042) * raking_gain * ctx.rescale_k.sqrt().clamp(0.5, 1.0)
    } else {
        0.0
    };
    out.tooth_bump_strength = tooth_str;
    let (tooth_ox, tooth_oy) = crate::banks::bank_offset(rng, crate::banks::BANK_TILE);
    let tooth_tile: Option<Vec<f32>> = if tooth_str > 1e-4 {
        let aniso = rng.gen_range(0.05..0.35);
        let laid = if ctx.paper_edge > 0.35 { 0.18 } else { 0.0 };
        Some(crate::banks::generate_tooth_bump_tile(
            ctx.file_seed,
            lx,
            ly,
            aniso,
            laid,
        ))
    } else {
        None
    };

    let (pm_str, pm_mx, pm_my, pm_bw) = if cfg.enable_plate_mark && rng.gen::<f32>() < 0.45 {
        let mx = rng.gen_range(0.065..0.115);
        let my = rng.gen_range(0.055..0.100);
        let bw = (4.5 * ctx.rescale_k / min_dim).clamp(0.002, 0.012);
        let st = s * rng.gen_range(0.035..0.085) * raking_gain.min(1.4);
        out.plate_mark_strength = st;
        out.plate_mark_inset = [mx, my];
        (st, mx, my, bw)
    } else {
        (0.0, 0.0, 0.0, 0.004)
    };

    let (gut_str, gut_u, gut_inv_w) =
        if cfg.enable_gutter_emboss && (ctx.spine_active || rng.gen::<f32>() < 0.25) {
            let uc = if ctx.spine_active {
                ctx.spine_u
            } else {
                0.5 + rng.gen_range(-0.02..0.02)
            };
            let wg = if ctx.spine_active {
                ctx.spine_w.max(0.025)
            } else {
                rng.gen_range(0.018..0.040)
            };
            let st = s * rng.gen_range(0.045..0.115) * raking_gain.min(1.45);
            out.gutter_emboss_strength = st;
            (st, uc, 1.0 / wg)
        } else {
            (0.0, 0.5, 25.0)
        };

    let water_inv_r = 1.0 / ctx.water_r.max(0.04);
    let edge_cockle_w = 0.25 + 0.75 * ctx.paper_edge;
    let tooth_ref = tooth_tile.as_deref();

    for y in 0..h {
        let fy = y as f32;
        let mut pc = [0.0f32; 6];
        let mut ps = [0.0f32; 6];
        for i in 0..6 {
            let a0 = wave_ky[i] * fy + wave_ph[i];
            pc[i] = a0.cos();
            ps[i] = a0.sin();
        }
        let mut u = ctx.u0 + ctx.du_dy * fy;
        let mut v = ctx.v0 + ctx.dv_dy * fy;
        let ty_row = (((y + tooth_oy) & 255) as usize) << 8;
        let row_base = (y as usize) * (w as usize) * 3;

        for x in 0..w {
            let mut cr_dot = 0.0f32;
            let mut ck_dot = 0.0f32;
            for i in 0..4 {
                cr_dot += wave_w[i] * pc[i];
                let nc = pc[i] * wave_dc[i] - ps[i] * wave_ds[i];
                let ns = ps[i] * wave_dc[i] + pc[i] * wave_ds[i];
                pc[i] = nc;
                ps[i] = ns;
            }
            for i in 4..6 {
                ck_dot += wave_w[i] * pc[i];
                let nc = pc[i] * wave_dc[i] - ps[i] * wave_ds[i];
                let ns = ps[i] * wave_dc[i] + pc[i] * wave_ds[i];
                pc[i] = nc;
                ps[i] = ns;
            }

            if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) {
                let mut relief = crumple_str * cr_dot;
                if cockle_str > 0.0 {
                    let d_edge = u.min(1.0 - u).min(v).min(1.0 - v);
                    let mut m_env = 0.25 + edge_cockle_w * smoothstep01(0.18, 0.0, d_edge);
                    if ctx.water_present {
                        let du = u - ctx.water_u;
                        let dv = v - ctx.water_v;
                        let rn = (du * du + dv * dv).sqrt() * water_inv_r;
                        if rn < 1.6 {
                            let body = (-(rn * rn) * 0.72).exp();
                            let rim = (-((rn - 1.0) * 4.5).powi(2)).exp();
                            m_env += ctx.water_amount * (0.9 * body + 1.15 * rim);
                        }
                    }
                    relief += cockle_str * m_env.min(2.2) * ck_dot;
                }

                if pm_str > 0.0 {
                    let dl = u - pm_mx;
                    let dr = (1.0 - pm_mx) - u;
                    let dt = v - pm_my;
                    let db = (1.0 - pm_my) - v;
                    let d_min = dl.min(dr).min(dt).min(db);
                    if d_min.abs() < pm_bw {
                        let (nx, ny) =
                            if dl.abs() <= dr.abs() && dl.abs() <= dt.abs() && dl.abs() <= db.abs()
                            {
                                (1.0, 0.0)
                            } else if dr.abs() <= dt.abs() && dr.abs() <= db.abs() {
                                (-1.0, 0.0)
                            } else if dt.abs() <= db.abs() {
                                (0.0, 1.0)
                            } else {
                                (0.0, -1.0)
                            };
                        let profile = 1.0 - (d_min / pm_bw).abs();
                        relief += pm_str * profile * (nx * lx + ny * ly) - 0.18 * pm_str * profile;
                    }
                }

                if gut_str > 0.0 {
                    let t = (u - gut_u) * gut_inv_w;
                    if t.abs() < 3.5 {
                        let slope = 2.0 * t * (-t * t).exp();
                        relief += gut_str * slope * lx;
                    }
                }

                let macro_delta = if relief < 0.0 {
                    relief * 1.15
                } else {
                    relief * 0.85
                };
                let idx = row_base + (x as usize) * 3;
                let r = buf[idx];
                let g = buf[idx + 1];
                let b = buf[idx + 2];
                let mut shade = 1.0 + macro_delta;
                if let Some(tile) = tooth_ref {
                    let luma_norm = (r + g + b) * (1.0 / 765.0);
                    let paper_w = ((luma_norm - 0.25) * (1.0 / 0.55)).clamp(0.0, 1.0);
                    if paper_w > 0.0 {
                        let tb = tile[ty_row + (((x + tooth_ox) & 255) as usize)];
                        shade += tooth_str * paper_w * tb;
                    }
                }
                let shade = shade.clamp(0.74, 1.12);
                buf[idx] = (r * shade).clamp(0.0, 255.0);
                buf[idx + 1] = (g * shade).clamp(0.0, 255.0);
                buf[idx + 2] = (b * shade).clamp(0.0, 255.0);
            }

            u += ctx.du_dx;
            v += ctx.dv_dx;
        }
    }
    out
}

// ===========================================================================
// Edge-Aware Smartphone HDR Unsharp-Mask (USM) + Shadow-Recovery Lift
// ===========================================================================

pub(crate) fn apply_usm_hdr(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    enabled: bool,
) -> (f32, f32) {
    if !enabled || strength <= 0.05 || w < 8 || h < 8 {
        return (0.0, 0.0);
    }
    let usm_strength = strength * rng.gen_range(0.28..0.65);
    let hdr_shadow_boost = strength * rng.gen_range(0.25..0.55);
    let coring_thresh = rng.gen_range(4.0..9.0);
    let inv_ramp = 1.0 / (coring_thresh * 0.75 + 1.0);
    let rad: usize = if w.max(h) >= 900 { 2 } else { 1 };
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;

    let mut luma = vec![0.0f32; n];
    for (dst, px) in luma.iter_mut().zip(buf.as_chunks::<3>().0) {
        *dst = (px[0] + px[1] + px[2]) * (1.0 / 3.0);
    }

    let mut tmp = vec![0.0f32; n];
    let inv_k = 1.0 / (2 * rad + 1) as f32;
    for y in 0..hu {
        let r = y * wu;
        for x in 0..rad {
            let mut s = 0.0;
            for d in -(rad as isize)..=(rad as isize) {
                s += luma[r + (x as isize + d).clamp(0, wu as isize - 1) as usize];
            }
            tmp[r + x] = s * inv_k;
        }
        if rad == 2 {
            for x in 2..(wu - 2) {
                tmp[r + x] = (luma[r + x - 2]
                    + luma[r + x - 1]
                    + luma[r + x]
                    + luma[r + x + 1]
                    + luma[r + x + 2])
                    * 0.2;
            }
        } else {
            for x in 1..(wu - 1) {
                tmp[r + x] = (luma[r + x - 1] + luma[r + x] + luma[r + x + 1]) * (1.0 / 3.0);
            }
        }
        for x in (wu - rad)..wu {
            let mut s = 0.0;
            for d in -(rad as isize)..=(rad as isize) {
                s += luma[r + (x as isize + d).clamp(0, wu as isize - 1) as usize];
            }
            tmp[r + x] = s * inv_k;
        }
    }

    for y in 0..hu {
        let row = y * wu;
        let y_m2 = y.saturating_sub(2) * wu;
        let y_m1 = y.saturating_sub(1) * wu;
        let y_p1 = (y + 1).min(hu - 1) * wu;
        let y_p2 = (y + 2).min(hu - 1) * wu;
        for x in 0..wu {
            let blurred = if rad == 2 {
                (tmp[y_m2 + x] + tmp[y_m1 + x] + tmp[row + x] + tmp[y_p1 + x] + tmp[y_p2 + x]) * 0.2
            } else {
                (tmp[y_m1 + x] + tmp[row + x] + tmp[y_p1 + x]) * (1.0 / 3.0)
            };
            let idx = row + x;
            let y_val = luma[idx];
            let detail = (y_val - blurred).clamp(-36.0, 36.0);
            let edge_w = ((detail.abs() - coring_thresh) * inv_ramp).clamp(0.0, 1.0);
            let y_norm = (y_val * (1.0 / 255.0)).clamp(0.0, 1.0);
            let inv_y = 1.0 - y_norm;
            let shadow_w = 1.0 + hdr_shadow_boost * inv_y * inv_y;
            let hi_prot = 1.0 - ((y_norm - 0.88) * (1.0 / 0.12)).clamp(0.0, 1.0) * 0.5;
            let sh_lift = if blurred < 140.0 {
                let t = 1.0 - blurred * (1.0 / 140.0);
                hdr_shadow_boost * 14.0 * t * t
            } else {
                0.0
            };
            let delta = usm_strength * edge_w * shadow_w * hi_prot * detail + sh_lift;
            let p = idx * 3;
            buf[p] = (buf[p] + delta).clamp(0.0, 255.0);
            buf[p + 1] = (buf[p + 1] + delta).clamp(0.0, 255.0);
            buf[p + 2] = (buf[p + 2] + delta).clamp(0.0, 255.0);
        }
    }
    (usm_strength, hdr_shadow_boost)
}

// ===========================================================================
// Flatbed Scanner Lid Edge Bar & CCD Dust Streaks
// ===========================================================================

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ScannerArtifactParams {
    pub edge_mask: u8,
    pub edge_width_px: u32,
    pub ccd_lines: u32,
}

pub(crate) fn apply_scanner_artifacts(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    s: f32,
    enabled: bool,
    xerox_heavy: bool,
) -> ScannerArtifactParams {
    if !enabled || w < 32 || h < 32 {
        return ScannerArtifactParams::default();
    }
    let (wsz, hsz) = (w as usize, h as usize);
    let min_d = w.min(h) as f32;
    let bar_w = ((min_d * rng.gen_range(0.005..0.016)).round() as usize).clamp(4, 28);
    let falloff = (bar_w / 3).max(2) as f32;
    let dark_lvl = if xerox_heavy {
        rng.gen_range(12.0..28.0)
    } else {
        rng.gen_range(22.0..48.0)
    };

    let n_edges = if rng.gen::<f32>() < 0.55 { 2 } else { 1 };
    let mut edge_mask = 0u8;
    let choices = [1u8, 4, 2, 8];
    for _ in 0..n_edges {
        edge_mask |= choices[rng.gen_range(0..4)];
    }

    let blend_edge = |buf: &mut [f32], idx: usize, dist: usize| {
        let t = ((bar_w.saturating_sub(dist)) as f32 / falloff).clamp(0.0, 1.0);
        let a = t * t * (3.0 - 2.0 * t);
        for c in 0..3 {
            buf[idx + c] = buf[idx + c] * (1.0 - a) + dark_lvl * a;
        }
    };
    if (edge_mask & 1) != 0 {
        for y in 0..hsz {
            for x in 0..bar_w.min(wsz) {
                blend_edge(buf, (y * wsz + x) * 3, x);
            }
        }
    }
    if (edge_mask & 2) != 0 {
        for y in 0..hsz {
            for d in 0..bar_w.min(wsz) {
                blend_edge(buf, (y * wsz + (wsz - 1 - d)) * 3, d);
            }
        }
    }
    if (edge_mask & 4) != 0 {
        for y in 0..bar_w.min(hsz) {
            for x in 0..wsz {
                blend_edge(buf, (y * wsz + x) * 3, y);
            }
        }
    }
    if (edge_mask & 8) != 0 {
        for d in 0..bar_w.min(hsz) {
            let y = hsz - 1 - d;
            for x in 0..wsz {
                blend_edge(buf, (y * wsz + x) * 3, d);
            }
        }
    }

    let ccd_lines = if rng.gen::<f32>() < (if xerox_heavy { 0.85 } else { 0.60 }) {
        rng.gen_range(1..=2)
    } else {
        0
    };
    for _ in 0..ccd_lines {
        let lo = bar_w.min(wsz.saturating_sub(2));
        let hi = (wsz.saturating_sub(lo)).max(lo + 1);
        let x0 = rng.gen_range(lo..hi);
        let line_w = if rng.gen::<f32>() < 0.65 {
            1usize
        } else {
            2usize
        };
        let bright = rng.gen::<f32>() < 0.45;
        let mag = if bright {
            rng.gen_range(16.0..34.0) * (0.5 + 0.5 * s)
        } else {
            -rng.gen_range(20.0..48.0) * (0.5 + 0.5 * s)
        };
        for y in 0..hsz {
            for dx in 0..line_w {
                let x = (x0 + dx).min(wsz - 1);
                let j = (y * wsz + x) * 3;
                let k = if dx == 0 { 1.0 } else { 0.55 };
                for c in 0..3 {
                    buf[j + c] = (buf[j + c] + mag * k).clamp(0.0, 255.0);
                }
            }
        }
    }
    ScannerArtifactParams {
        edge_mask,
        edge_width_px: bar_w as u32,
        ccd_lines,
    }
}

// ===========================================================================
// Procedural Desk LUT, Exact 2-Quad Output SDF, Holes, Music-Stand Ledge & Edge Occluder
// ===========================================================================

#[inline]
pub(crate) fn hash2_u32(ix: i32, iy: i32, seed: u32) -> f32 {
    let mut h =
        seed ^ (ix.wrapping_mul(374_761_393) as u32) ^ (iy.wrapping_mul(668_265_263) as u32);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0x00FF_FFFF) as f32 / 16_777_215.0
}

pub(crate) struct DeskLut {
    grid: Vec<f32>,
}

impl DeskLut {
    pub(crate) fn new(seed: u32) -> Self {
        let mut grid = vec![0.0f32; 256 * 256];
        for iy in 0..256 {
            for ix in 0..256 {
                grid[iy * 256 + ix] = hash2_u32(ix as i32, iy as i32, seed);
            }
        }
        Self { grid }
    }

    #[inline(always)]
    pub(crate) fn sample(&self, x: f32, y: f32) -> f32 {
        let ix = x.floor() as i32;
        let iy = y.floor() as i32;
        let fx = x - ix as f32;
        let fy = y - iy as f32;
        let u = fx * fx * (3.0 - 2.0 * fx);
        let v = fy * fy * (3.0 - 2.0 * fy);
        let x0 = (ix & 255) as usize;
        let x1 = ((ix + 1) & 255) as usize;
        let y0 = ((iy & 255) as usize) << 8;
        let y1 = (((iy + 1) & 255) as usize) << 8;
        let top = self.grid[y0 | x0] * (1.0 - u) + self.grid[y0 | x1] * u;
        let bot = self.grid[y1 | x0] * (1.0 - u) + self.grid[y1 | x1] * u;
        top * (1.0 - v) + bot * v
    }
}

#[inline]
pub(crate) fn desk_pixel_fast(
    lut: &DeskLut,
    surface: &str,
    x: f32,
    y: f32,
    wa_cos: f32,
    wa_sin: f32,
) -> [f32; 3] {
    match surface {
        "walnut" | "oak" => {
            let u = x * wa_cos + y * wa_sin;
            let v = -x * wa_sin + y * wa_cos;
            let warp =
                lut.sample(u * 0.008, v * 0.045) * 7.5 + lut.sample(u * 0.025, v * 0.12) * 2.2;
            let rings = (v * 0.11 + warp).sin() * 0.5 + 0.5;
            let pores = lut.sample(u * 0.035, v * 0.55);
            let seam = if (v.rem_euclid(420.0) - 2.0).abs() < 1.5 {
                0.72
            } else {
                1.0
            };
            let t = (0.58 * rings + 0.42 * pores) * seam;
            if surface == "oak" {
                [86.0 + t * 52.0, 64.0 + t * 40.0, 42.0 + t * 28.0]
            } else {
                [30.0 + t * 34.0, 20.0 + t * 23.0, 14.0 + t * 15.0]
            }
        }
        "granite_laminate" => {
            let macro_n = lut.sample(x * 0.03, y * 0.03);
            let spec = lut.sample(x * 0.45, y * 0.45);
            let fleck = if spec > 0.84 {
                26.0
            } else if spec < 0.14 {
                -16.0
            } else {
                0.0
            };
            let base = 36.0 + macro_n * 24.0 + fleck;
            [base, base + 1.5, base + 4.0]
        }
        "cloth" => {
            let weave = ((x * 0.9).sin() * (y * 0.9).sin()) * 0.5 + 0.5;
            let t = 0.45 * weave + 0.55 * lut.sample(x * 0.03, y * 0.03);
            [42.0 + t * 22.0, 38.0 + t * 20.0, 34.0 + t * 18.0]
        }
        _ => {
            let t = 0.65 * lut.sample(x * 0.04, y * 0.04) + 0.35 * lut.sample(x * 0.25, y * 0.25);
            [28.0 + t * 20.0, 31.0 + t * 21.0, 36.0 + t * 23.0]
        }
    }
}

/// Sampled parameters for the optional plastic sheet-protector / glossy thermal
/// diagonal specular sheen band.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SheenParams {
    pub strength: f32,
    pub angle_rad: f32,
    pub pos: f32,
    pub width_frac: f32,
}

impl SheenParams {
    pub(crate) fn sample(rng: &mut ChaCha8Rng, s: f32, enabled: bool, thermal: bool) -> Self {
        if !enabled || s < 0.4 {
            return Self::default();
        }
        let prob = if thermal { 0.45 } else { 0.28 };
        if rng.gen::<f32>() >= prob {
            return Self::default();
        }
        let deg = if rng.gen::<f32>() < 0.5 {
            rng.gen_range(20.0..70.0f32)
        } else {
            rng.gen_range(110.0..160.0f32)
        };
        Self {
            strength: s * if thermal {
                rng.gen_range(0.09..0.18)
            } else {
                rng.gen_range(0.18..0.38)
            },
            angle_rad: deg.to_radians(),
            pos: rng.gen_range(0.25..0.75),
            width_frac: if thermal {
                rng.gen_range(0.14..0.24)
            } else {
                rng.gen_range(0.06..0.13)
            },
        }
    }

    #[inline]
    pub(crate) fn eval(&self, x: f32, y: f32, wf: f32, hf: f32) -> f32 {
        if self.strength <= 0.0 {
            return 0.0;
        }
        let min_d = wf.min(hf).max(1.0);
        let (nx, ny) = (self.angle_rad.cos(), self.angle_rad.sin());
        let (tx, ty) = (-ny, nx);
        let cx = x - wf * 0.5;
        let cy = y - hf * 0.5;
        let off = (self.pos - 0.5) * min_d;
        let d = cx * nx + cy * ny - off;
        let along = (cx * tx + cy * ty) / min_d;
        let band_w = (self.width_frac * min_d).max(4.0);
        let gauss = (-(d / band_w).powi(2)).exp();
        let mod_along = 0.72 + 0.28 * (along * 3.5).cos();
        self.strength * gauss * mod_along
    }
}

/// Margin-safe fingertip or metal binder-clip edge occluder.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EdgeOccluder {
    pub kind: &'static str,
    pub edge: u8,
    pub along_frac: f32,
    pub pen_px: f32,
    pub half_w_px: f32,
    pub tilt_rad: f32,
    pub skin_idx: u8,
}

impl Default for EdgeOccluder {
    fn default() -> Self {
        Self {
            kind: "none",
            edge: 0,
            along_frac: 0.5,
            pen_px: 0.0,
            half_w_px: 0.0,
            tilt_rad: 0.0,
            skin_idx: 0,
        }
    }
}

impl EdgeOccluder {
    pub(crate) fn sample(rng: &mut ChaCha8Rng, params: &DegradeParams, enabled: bool) -> Self {
        if !enabled || params.strength < 0.25 || rng.gen::<f32>() >= 0.22 * params.strength {
            return Self::default();
        }
        let kind = if rng.gen::<f32>() < 0.60 {
            "finger"
        } else {
            "binder_clip"
        };
        let min_d = params.src_width.min(params.src_height).max(1) as f32;
        let min_pen = (min_d * 0.012).max(10.0);
        let order: [u8; 3] = if kind == "finger" {
            [0, 2, 3]
        } else {
            [1, 0, 2]
        };
        let start = rng.gen_range(0..3);
        for k in 0..3 {
            let e = order[(start + k) % 3] as usize;
            let avail = params.frame_quiet_px[e]
                .saturating_sub(params.frame_cut_px[e])
                .saturating_sub(params.frame_safe_px) as f32;
            if avail >= min_pen {
                let want = min_d
                    * if kind == "finger" {
                        rng.gen_range(0.025..0.055)
                    } else {
                        rng.gen_range(0.018..0.038)
                    };
                return Self {
                    kind,
                    edge: e as u8,
                    along_frac: rng.gen_range(0.18..0.82),
                    pen_px: want.min(avail),
                    half_w_px: (want * rng.gen_range(0.55..0.85)).max(8.0),
                    tilt_rad: rng.gen_range(-0.28..0.28),
                    skin_idx: rng.gen_range(0..6),
                };
            }
        }
        Self::default()
    }
}

/// Post-downscale pass (runs identically on CPU and GPU before tone-LUT + JPEG):
/// 1. Replaces out-of-page pixels using the exact 2-Quad Output SDF (`Q_out` ∩ `R_out`)
///    with a deterministic LUT-accelerated procedural desk surface (`walnut`, `oak`,
///    `granite_laminate`, `slate`, `cloth`) + crisp AO & directional contact drop-shadow.
/// 2. Renders continuous desk texture + inner paper-thickness shadow inside binder punch holes.
/// 3. Applies optional plastic sheet-protector / thermal-coating diagonal specular sheen.
/// 4. Applies optional margin-safe music-stand bottom metal ledge + upward occlusion shadow.
/// 5. Applies optional margin-safe fingertip or metal binder-clip edge occluder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_desk_and_sheen(
    buf: &mut [f32],
    params: &DegradeParams,
    desk_surface: &str,
    sheen: &SheenParams,
    stand_ledge_px: u32,
    occluder: &EdgeOccluder,
    file_seed: u64,
) {
    let run_desk = desk_surface != "none";
    let run_sheen = sheen.strength > 0.0;
    let run_ledge = stand_ledge_px > 0;
    let run_occ = occluder.kind != "none";
    if !run_desk && !run_sheen && !run_ledge && !run_occ {
        return;
    }
    let w = params.width;
    let h = params.height;
    let wf = w as f32;
    let hf = h as f32;
    let ctx = ReprojectContext::new(params);
    let seed32 = (file_seed ^ (file_seed >> 32) ^ 0xA5A5_1234) as u32;
    let lut = if run_desk || run_ledge {
        Some(DeskLut::new(seed32))
    } else {
        None
    };
    let wood_ang = ((seed32 % 360) as f32).to_radians();
    let (wa_cos, wa_sin) = (wood_ang.cos(), wood_ang.sin());

    // Exact 2-Quad Output-Space SDF (100% parity with masks::OUT_OF_PAGE & detached corners).
    let (q_out, r_out) = ctx.visible_page_quads_out();
    let sdf_q = QuadSdf2D::from_cw_quad(q_out);
    let sdf_r = QuadSdf2D::from_cw_quad(r_out);

    let sh_th = (params.shadow_angle_deg as f64).to_radians();
    let (sh_dx, sh_dy) = (sh_th.cos(), sh_th.sin());
    let shadow_radius = (w.min(h) as f64 * 0.022).clamp(4.0, 32.0);

    if run_desk || run_sheen {
        let lut_ref = lut.as_ref();
        for y in 0..h {
            let fy = y as f64;
            for x in 0..w {
                let fx = x as f64;
                let idx = ((y * w + x) * 3) as usize;

                let (dq, nq) = sdf_q.outside_dist_and_normal(fx, fy);
                let (dr, nr) = sdf_r.outside_dist_and_normal(fx, fy);
                let (dist_out, (ox, oy)) = if dq >= dr { (dq, nq) } else { (dr, nr) };

                if dist_out > 0.0 {
                    if let Some(dl) = lut_ref {
                        if run_desk {
                            let mut tex = desk_pixel_fast(
                                dl,
                                desk_surface,
                                x as f32,
                                y as f32,
                                wa_cos,
                                wa_sin,
                            );
                            let align = (ox * sh_dx + oy * sh_dy).clamp(-1.0, 1.0);
                            let dir_bias = (0.65 + 0.35 * align).max(0.25);
                            let eff_rad = shadow_radius * dir_bias;
                            let ao = 0.25 * (-dist_out / 3.0).exp();
                            let drop_sh = 0.55 * (-dist_out / eff_rad).exp();
                            let shadow_mul = (1.0 - (ao + drop_sh).min(0.72)) as f32;
                            tex[0] *= shadow_mul;
                            tex[1] *= shadow_mul;
                            tex[2] *= shadow_mul;
                            let a = (dist_out / 1.5).clamp(0.0, 1.0) as f32;
                            buf[idx] = buf[idx] * (1.0 - a) + tex[0] * a;
                            buf[idx + 1] = buf[idx + 1] * (1.0 - a) + tex[1] * a;
                            buf[idx + 2] = buf[idx + 2] * (1.0 - a) + tex[2] * a;
                        }
                    }
                } else if run_sheen {
                    let s_val = sheen.eval(x as f32, y as f32, wf, hf);
                    if s_val > 1e-4 {
                        let add = s_val * 255.0;
                        buf[idx] = (buf[idx] + add).clamp(0.0, 255.0);
                        buf[idx + 1] = (buf[idx + 1] + add).clamp(0.0, 255.0);
                        buf[idx + 2] = (buf[idx + 2] + add).clamp(0.0, 255.0);
                    }
                }
            }
        }
    }

    // Depth-shadowed desk-through binder punch hole (ROI bbox only).
    if let Some(ref dl) = lut {
        if run_desk && params.capture.enabled && params.capture.hole_radius > 0 {
            let (lw, lh) = ctx.lighting_dims();
            let scale_x = w as f64 / lw.max(1) as f64;
            let scale_y = h as f64 / lh.max(1) as f64;
            let (hx, hy) = if params.rot_scale_fused && ctx.rot_active {
                let (cx0, cy0) = (ctx.fw as f64 / 2.0, ctx.fh as f64 / 2.0);
                let (cx1, cy1) = (ctx.w_rot as f64 / 2.0, ctx.h_rot as f64 / 2.0);
                let u = params.capture.hole_x as f64 - cx0;
                let v = params.capture.hole_y as f64 - cy0;
                let rx = cx1 + u * ctx.cst - v * ctx.snt;
                let ry = cy1 + u * ctx.snt + v * ctx.cst;
                (rx * params.scale_x as f64, ry * params.scale_y as f64)
            } else {
                (
                    params.capture.hole_x as f64 * scale_x,
                    params.capture.hole_y as f64 * scale_y,
                )
            };
            let r_out = (params.capture.hole_radius as f64 * 0.5 * (scale_x + scale_y)).max(2.0);
            let pad = (r_out + 3.0).ceil() as i32;
            let x0 = (hx as i32 - pad).max(0) as u32;
            let x1 = (hx as i32 + pad).min(w as i32 - 1) as u32;
            let y0 = (hy as i32 - pad).max(0) as u32;
            let y1 = (hy as i32 + pad).min(h as i32 - 1) as u32;
            let sh_off = 0.28 * r_out;
            for y in y0..=y1 {
                let dy = y as f64 - hy;
                for x in x0..=x1 {
                    let dx = x as f64 - hx;
                    let d = dx.hypot(dy);
                    let idx = ((y * w + x) * 3) as usize;
                    if d <= r_out + 1.2 {
                        let mut tex =
                            desk_pixel_fast(dl, desk_surface, x as f32, y as f32, wa_cos, wa_sin);
                        // Inner paper-thickness crescent shadow offset along shadow direction.
                        let ds = (dx - sh_dx * sh_off).hypot(dy - sh_dy * sh_off);
                        let inner_sh = if ds > r_out * 0.78 {
                            0.46
                        } else {
                            0.64 + 0.22 * (1.0 - ds / r_out).clamp(0.0, 1.0)
                        };
                        tex[0] *= inner_sh as f32;
                        tex[1] *= inner_sh as f32;
                        tex[2] *= inner_sh as f32;
                        let aa = ((r_out + 0.6 - d) / 1.2).clamp(0.0, 1.0) as f32;
                        buf[idx] = buf[idx] * (1.0 - aa) + tex[0] * aa;
                        buf[idx + 1] = buf[idx + 1] * (1.0 - aa) + tex[1] * aa;
                        buf[idx + 2] = buf[idx + 2] * (1.0 - aa) + tex[2] * aa;
                    } else if d <= r_out + 2.5 {
                        let inv_d = 1.0 / d.max(1e-4);
                        let align = -(dx * inv_d * sh_dx + dy * inv_d * sh_dy);
                        let rim_delta = if align > 0.0 {
                            20.0 * align as f32
                        } else {
                            -14.0 * (-align) as f32
                        };
                        for c in 0..3 {
                            buf[idx + c] = (buf[idx + c] + rim_delta).clamp(0.0, 255.0);
                        }
                    }
                }
            }
        }
    }

    // Optional margin-safe music-stand bottom metal ledge + upward cast shadow.
    if run_ledge && stand_ledge_px < h {
        let y_ledge = h - stand_ledge_px;
        let sh_h = (w.min(h) as f32 * 0.016).clamp(8.0, 28.0);
        let y_sh0 = y_ledge.saturating_sub(sh_h.ceil() as u32);
        for y in y_sh0..y_ledge {
            let d = (y_ledge - y) as f32;
            let m = 1.0 - 0.42 * (-d / (sh_h * 0.42)).exp();
            let row = (y * w) as usize * 3;
            for x in 0..w as usize {
                let i = row + x * 3;
                buf[i] *= m;
                buf[i + 1] *= m;
                buf[i + 2] *= m;
            }
        }
        let dl = lut.as_ref();
        for y in y_ledge..h {
            let d_top = y - y_ledge;
            let bevel = if d_top < 3 {
                26.0 * (1.0 - d_top as f32 / 3.0)
            } else {
                0.0
            };
            let row = (y * w) as usize * 3;
            for x in 0..w as usize {
                let tex = dl
                    .map(|l| l.sample(x as f32 * 0.35, y as f32 * 0.35) * 10.0)
                    .unwrap_or(5.0);
                let i = row + x * 3;
                buf[i] = (22.0 + tex + bevel).clamp(0.0, 255.0);
                buf[i + 1] = (23.0 + tex + bevel).clamp(0.0, 255.0);
                buf[i + 2] = (27.0 + tex + bevel).clamp(0.0, 255.0);
            }
        }
    }

    // Optional margin-safe fingertip or metal binder-clip edge occluder.
    if run_occ {
        render_edge_occluder(buf, w, h, &ctx, occluder, sh_dx as f32, sh_dy as f32, None);
    }
}

/// Render or mask-stamp the margin-safe fingertip / metal binder-clip edge occluder.
#[allow(clippy::too_many_arguments)]
fn render_edge_occluder(
    mut buf: &mut [f32],
    w: u32,
    h: u32,
    ctx: &ReprojectContext<'_>,
    occ: &EdgeOccluder,
    sh_dx: f32,
    sh_dy: f32,
    mut art_mask: Option<&mut [u16]>,
) {
    if occ.kind == "none" || occ.pen_px <= 0.0 {
        return;
    }
    let w0 = ctx.w0 as f64;
    let h0 = ctx.h0 as f64;
    let af = occ.along_frac as f64;
    let pen = occ.pen_px as f64;
    // Anchor on the page edge and inward normal in persp space.
    let (ex, ey, in_x, in_y) = match occ.edge {
        0 => (0.0, af * h0, 1.0, 0.0),
        1 => (af * w0, 0.0, 0.0, 1.0),
        2 => (w0, af * h0, -1.0, 0.0),
        _ => (af * w0, h0, 0.0, -1.0),
    };
    let (ox0, oy0) = ctx.persp_to_output(ex, ey);
    let (ox1, oy1) = ctx.persp_to_output(ex + in_x * pen, ey + in_y * pen);
    let vx = (ox1 - ox0) as f32;
    let vy = (oy1 - oy0) as f32;
    let len_out = vx.hypot(vy).max(4.0);
    let (nx, ny) = (vx / len_out, vy / len_out);
    let (tx, ty) = (-ny, nx);
    let scale = (len_out / occ.pen_px.max(1.0)).clamp(0.2, 2.0);
    let hw = (occ.half_w_px * scale).max(6.0);
    let outer_ext = len_out * 1.4 + 24.0;
    let pad = (outer_ext.max(hw * 2.0) + 16.0).ceil() as i32;

    let cx = ox0 as f32;
    let cy = oy0 as f32;
    let x0 = (cx as i32 - pad).max(0) as u32;
    let x1 = (cx as i32 + pad).min(w as i32 - 1) as u32;
    let y0 = (cy as i32 - pad).max(0) as u32;
    let y1 = (cy as i32 + pad).min(h as i32 - 1) as u32;

    const SKINS: [[f32; 3]; 6] = [
        [224.0, 182.0, 152.0],
        [205.0, 158.0, 126.0],
        [182.0, 134.0, 102.0],
        [152.0, 106.0, 78.0],
        [118.0, 78.0, 56.0],
        [86.0, 56.0, 42.0],
    ];
    let skin = SKINS[occ.skin_idx as usize % 6];
    let (ca, sa) = (occ.tilt_rad.cos(), occ.tilt_rad.sin());
    let is_finger = occ.kind == "finger";
    let bit = if is_finger {
        artifact_bits::FINGER
    } else {
        artifact_bits::BINDER_CLIP
    };

    for y in y0..=y1 {
        let dy = y as f32 - cy;
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            // Local coords: u along inward normal (<= len_out), v tangent to edge.
            let u0 = dx * nx + dy * ny;
            let v0 = dx * tx + dy * ty;
            let u = u0 * ca + v0 * sa;
            let v = -u0 * sa + v0 * ca;
            if u > len_out + 12.0 || u < -outer_ext || v.abs() > hw * 2.2 {
                continue;
            }
            let pix = (y * w + x) as usize;
            let idx = pix * 3;
            if is_finger {
                // Soft-focus thumb/finger capsule from u = -outer_ext .. len_out.
                let tip_u = len_out - hw * 0.75;
                let du = (u - tip_u).max(0.0);
                let dist = (du / (hw * 0.75)).hypot(v / hw);
                let feather = 0.16;
                let alpha = ((1.0 + feather - dist) / (2.0 * feather)).clamp(0.0, 1.0);
                if let Some(ref mut am) = art_mask {
                    if alpha > 0.25 && pix < am.len() {
                        am[pix] |= bit;
                    }
                    continue;
                }
                if alpha > 1e-3 {
                    let shade = 0.82 + 0.22 * (1.0 - (v / hw).powi(2)).max(0.0)
                        - 0.12 * ((nx * sh_dx + ny * sh_dy).clamp(-1.0, 1.0));
                    let nail = if u > tip_u - hw * 0.35 && dist < 0.78 {
                        0.18 * (1.0 - dist / 0.78)
                    } else {
                        0.0
                    };
                    for c in 0..3 {
                        let col = (skin[c] * shade + 245.0 * nail) / (1.0 + nail);
                        buf[idx + c] =
                            (buf[idx + c] * (1.0 - alpha) + col * alpha).clamp(0.0, 255.0);
                    }
                } else if dist < 1.38 {
                    let sh = 1.0 - 0.32 * (1.0 - (dist - 1.0) / 0.38).clamp(0.0, 1.0);
                    for c in 0..3 {
                        buf[idx + c] *= sh;
                    }
                }
            } else {
                // Black spring-steel binder clip jaw (u in -hw*0.4 .. len_out).
                let t_u = (u / len_out.max(1.0)).clamp(0.0, 1.0);
                let half_jaw = hw * (1.0 - 0.14 * t_u);
                let in_jaw = u >= -hw * 0.45 && u <= len_out && v.abs() <= half_jaw;
                if let Some(ref mut am) = art_mask {
                    if in_jaw && pix < am.len() {
                        am[pix] |= bit;
                    }
                    continue;
                }
                if in_jaw {
                    let ridge = (-((u - len_out * 0.85) / 2.2).powi(2)).exp() * 48.0;
                    let base = 22.0 + 10.0 * (1.0 - (v / half_jaw).powi(2)).max(0.0) + ridge;
                    buf[idx] = base.clamp(0.0, 255.0);
                    buf[idx + 1] = (base + 1.0).clamp(0.0, 255.0);
                    buf[idx + 2] = (base + 4.0).clamp(0.0, 255.0);
                } else if u > len_out && u <= len_out + 6.0 && v.abs() <= half_jaw + 3.0 {
                    let sh = 1.0 - 0.36 * (1.0 - (u - len_out) / 6.0);
                    for c in 0..3 {
                        buf[idx + c] *= sh;
                    }
                }
            }
        }
    }
    let _ = &mut buf;
}

/// Stamp sheet-protector sheen (`GLARE`), music-stand ledge (`STAND_LEDGE`),
/// and fingertip / binder-clip (`FINGER` / `BINDER_CLIP`) into an output-space
/// artifact mask.
pub(crate) fn stamp_realism_masks(
    art: &mut [u16],
    params: &DegradeParams,
    sheen: &SheenParams,
    stand_ledge_px: u32,
    occluder: &EdgeOccluder,
) {
    let w = params.width;
    let h = params.height;
    if art.len() < (w as usize) * (h as usize) {
        return;
    }
    if sheen.strength > 0.0 {
        let wf = w as f32;
        let hf = h as f32;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if (art[idx] & artifact_bits::OUT_OF_PAGE) == 0
                    && sheen.eval(x as f32, y as f32, wf, hf) > 0.04
                {
                    art[idx] |= artifact_bits::GLARE;
                }
            }
        }
    }
    if stand_ledge_px > 0 && stand_ledge_px < h {
        let y0 = h - stand_ledge_px;
        for y in y0..h {
            let row = (y * w) as usize;
            for x in 0..w as usize {
                art[row + x] |= artifact_bits::STAND_LEDGE;
            }
        }
    }
    if occluder.kind != "none" {
        let ctx = ReprojectContext::new(params);
        let mut dummy = [];
        render_edge_occluder(&mut dummy, w, h, &ctx, occluder, 0.0, 0.0, Some(art));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{degrade_image, DegradeConfig, Degrader, DocumentProfile, MaskRequest};
    use image::{Rgb, RgbImage};
    use rand::SeedableRng;

    #[test]
    fn awb_gains_are_identity_when_zero_and_split_warm_cool_when_active() {
        let id = awb_channel_gains(10.0, 20.0, 50.0, 50.0, 141.0, 1.0, 0.0, 0.0);
        assert_eq!(id, [1.0, 1.0, 1.0]);
        let warm = awb_channel_gains(100.0, 50.0, 50.0, 50.0, 100.0, 1.0, 0.0, 0.5);
        let cool = awb_channel_gains(0.0, 50.0, 50.0, 50.0, 100.0, 1.0, 0.0, 0.5);
        assert!(warm[0] > warm[2], "warm side must boost R over B: {warm:?}");
        assert!(cool[2] > cool[0], "cool side must boost B over R: {cool:?}");
    }

    #[test]
    fn micro_relief_and_usm_hdr_are_deterministic_and_bounded() {
        let (w, h) = (64u32, 64u32);
        let mut base = vec![240.0f32; (w * h * 3) as usize];
        for y in 20..44 {
            for x in 16..48 {
                let i = ((y * w + x) * 3) as usize;
                base[i] = 20.0;
                base[i + 1] = 20.0;
                base[i + 2] = 20.0;
            }
        }
        let mut a = base.clone();
        let mut b = base.clone();
        let mut ra = ChaCha8Rng::seed_from_u64(19);
        let mut rb = ChaCha8Rng::seed_from_u64(19);
        let rcfg = RealismConfig::phone();
        let ctx = MicroReliefContext {
            file_seed: 19,
            strength: 0.7,
            rescale_k: 1.0,
            u0: 0.0,
            du_dx: 1.0 / w as f32,
            du_dy: 0.0,
            v0: 0.0,
            dv_dx: 0.0,
            dv_dy: 1.0 / h as f32,
            water_present: false,
            water_u: 0.5,
            water_v: 0.5,
            water_r: 0.2,
            water_amount: 0.0,
            paper_cockle: 0.5,
            paper_edge: 0.5,
            spine_active: false,
            spine_u: 0.5,
            spine_w: 0.05,
        };
        let ca = apply_micro_relief_shading(&mut a, w, h, &mut ra, &rcfg, &ctx);
        let cb = apply_micro_relief_shading(&mut b, w, h, &mut rb, &rcfg, &ctx);
        assert_eq!(ca.crumple_strength, cb.crumple_strength);
        assert_eq!(a, b);
        assert!(ca.crumple_strength > 0.0 && a != base);

        let (ua, ha) = apply_usm_hdr(&mut a, w, h, &mut ra, 0.7, true);
        let (ub, hb) = apply_usm_hdr(&mut b, w, h, &mut rb, 0.7, true);
        assert_eq!((ua, ha), (ub, hb));
        assert_eq!(a, b);
        assert!(ua > 0.0 && ha > 0.0);
        assert!(a.iter().all(|&v| (0.0..=255.0).contains(&v)));
    }

    #[test]
    fn wave2_and_wave3_profiles_calibrate_realism_params() {
        let mut img = RgbImage::from_pixel(96, 96, Rgb([250, 250, 248]));
        for y in 24..72 {
            for x in 24..72 {
                img.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        // OfficeScan disables camera/desk/crumple/USM/sheen/phone_shadow/tone_curve.
        let scan = Degrader::builder()
            .profile(DocumentProfile::OfficeScan)
            .backend(crate::BackendPreference::Cpu)
            .build()
            .unwrap();
        let r_scan = scan.degrade_image(&img, 42).unwrap();
        assert_eq!(r_scan.params.awb_warm_cool_strength, 0.0);
        assert_eq!(r_scan.params.usm_strength, 0.0);
        assert_eq!(r_scan.params.desk_surface, "none");
        assert_eq!(r_scan.params.crumple_strength, 0.0);
        assert_eq!(r_scan.params.sheen_strength, 0.0);
        assert_eq!(r_scan.params.phone_shadow_strength, 0.0);
        assert_eq!(r_scan.params.tone_contrast, 0.0);

        // PhonePhoto enables AWB, USM, desk surface, crumple, tooth, tone curve, and occasional sheen/phone shadow.
        let mut saw_sheen = false;
        let mut saw_phone_shadow = false;
        let mut surfaces = std::collections::HashSet::new();
        for seed in 0..20u64 {
            let mut cfg = DegradeConfig::default().with_seed(seed).with_strength(0.75);
            cfg.masks = MaskRequest::all();
            let res = degrade_image(img.clone(), &cfg).unwrap();
            assert!(res.params.awb_warm_cool_strength > 0.0);
            assert!(res.params.usm_strength > 0.0);
            assert!(res.params.crumple_strength > 0.0);
            assert!(res.params.tone_contrast > 0.0);
            assert!(["walnut", "oak", "granite_laminate", "slate", "cloth"]
                .contains(&res.params.desk_surface.as_str()));
            surfaces.insert(res.params.desk_surface.clone());
            if res.params.phone_shadow_strength > 0.0 {
                saw_phone_shadow = true;
            }
            if res.params.sheen_strength > 0.0 {
                saw_sheen = true;
                let art = res.masks.unwrap().artifacts.unwrap();
                assert!(art.iter().any(|&b| (b & artifact_bits::GLARE) != 0));
            }
        }
        assert!(
            surfaces.len() >= 2,
            "expected multiple desk surfaces, got {surfaces:?}"
        );
        assert!(
            saw_sheen,
            "expected at least one sheet-protector sheen hit in 20 seeds"
        );
        assert!(
            saw_phone_shadow,
            "expected at least one handheld phone shadow hit in 20 seeds"
        );
    }
}
