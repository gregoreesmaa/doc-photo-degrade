//! Paper aging & texture for document training data: yellowing/mottling, aged
//! edge band, clustered foxing spots, fibers + laid lines, cockling waves,
//! and optional water damage with a tide-line rim. All randomness comes from
//! the passed `rng` (deterministic per seed). Buffer: interleaved RGB `f32`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Every sampled paper-aging knob, recorded for `params.json`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct PaperParams {
    pub enabled: bool,
    pub yellowing: f32,
    pub mottle: f32,
    pub edge: f32,
    pub foxing_spots: u32,
    pub foxing_amount: f32,
    pub fiber_strokes: u32,
    pub fiber_amount: f32,
    pub cockle: f32,
    pub water_present: bool,
    pub water_x: f32,
    pub water_y: f32,
    pub water_radius: f32,
    pub water_amount: f32,
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Age the paper in `buf` (interleaved RGB `f32`, row-major, `w*h*3`).
/// `strength` is 0..=1 and scales every effect.
/// Full-strength entry point (kept for the unit tests + API stability).
#[allow(dead_code)]
pub fn apply(buf: &mut [f32], w: u32, h: u32, rng: &mut ChaCha8Rng, strength: f32) -> PaperParams {
    apply_with(buf, w, h, rng, strength, false, 1.0)
}

/// Full aging pass when `coarse` is false; when true, only foxing spots +
/// fibers are applied and the full-image tint/mottle/edge/cockling/water
/// loop is skipped. Sampling header identical either way (stream-preserving);
/// the skipped loop consumes no RNG draws.
///
/// `px` is the pixel-scale factor (1.0 = native): absolute-pixel radii and
/// the laid-line period scale with it; `px = 1.0` is bit-identical to legacy.
pub fn apply_with(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    coarse: bool,
    px: f32,
) -> PaperParams {
    let strength = strength.clamp(0.0, 1.0);
    // Fixed draw order: every knob is sampled on every call.
    let yellowing = rng.gen::<f32>();
    let mottle = rng.gen::<f32>();
    let edge = rng.gen::<f32>();
    let foxing_u = rng.gen::<f32>();
    let foxing_amount = rng.gen::<f32>();
    let fiber_u = rng.gen::<f32>();
    let fiber_amount = rng.gen::<f32>();
    let cockle = rng.gen::<f32>();
    let water_u = rng.gen::<f32>();
    let water_x = rng.gen::<f32>();
    let water_y = rng.gen::<f32>();
    let water_radius = 0.15 + rng.gen::<f32>() * 0.30;
    let water_amount = rng.gen::<f32>();
    // Total pixel area (distinct from the resolution scale factor `px`).
    let area_px = w as usize * h as usize;
    let max_spots = ((area_px / 9000).clamp(8, 900)) as u32;
    let max_fibers = ((area_px / 20000).clamp(10, 600)) as u32;
    let mut p = PaperParams {
        enabled: strength > 0.0,
        yellowing,
        mottle,
        edge,
        foxing_spots: (foxing_u * max_spots as f32).round() as u32,
        foxing_amount,
        fiber_strokes: (fiber_u * max_fibers as f32).round() as u32,
        fiber_amount,
        cockle,
        water_present: water_u < 0.35,
        water_x,
        water_y,
        water_radius,
        water_amount,
    };
    if !p.enabled || buf.is_empty() || w == 0 || h == 0 {
        if !p.enabled {
            p.water_present = false;
        }
        return p;
    }
    let wf = w as f32;
    let hf = h as f32;
    let min_dim = wf.min(hf);
    // Coarse mode skips the full-image tint loop below (no RNG draws inside,
    // so the stream is preserved); foxing + fibers still run.
    if !coarse {
        // Deterministic phases derived from sampled knobs (no hidden draws).
        #[allow(clippy::approx_constant)]
        let (o1, o2, o3, o4) = (
            p.yellowing * 6.28318,
            p.mottle * 6.28318,
            p.edge * 6.28318,
            p.cockle * 6.28318,
        );
        let (yel, mot, edg, cok) = (
            p.yellowing * strength,
            p.mottle * strength,
            p.edge * strength,
            p.cockle * strength,
        );
        let laid_amp = (0.5 + 1.1 * p.fiber_amount) * strength;
        // Laid lines are a ~5 px period at native res; keep the physical period
        // fixed by scaling the angular frequency (÷1.0 = legacy exactly).
        let laid_w = 1.2566 / px;
        let (wcx, wcy, wr) = (p.water_x * wf, p.water_y * hf, p.water_radius * min_dim);
        for y in 0..h {
            let fy = y as f32 / hf.max(1.0);
            let sy1 = (fy * 18.85 + o1).sin(); // ~3 vertical blotch cycles
            let sy2 = (fy * 50.27 + o2).sin(); // ~8 vertical blotch cycles
            let syc = (fy * 12.57 + o4).sin(); // cockling ~2 cycles
            let laid = (y as f32 * laid_w + o3).sin() * laid_amp; // ~5px laid lines
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                if i + 2 >= buf.len() {
                    continue;
                }
                let fx = x as f32 / wf.max(1.0);
                // Layered sine blotch field, -1..1.
                let m = (fx * 18.85 + o2).sin() * sy1 * 0.6 + (fx * 43.98 + o1).sin() * sy2 * 0.4;
                // Aged border band: 0 inside, 1 at the sheet rim.
                let d = fx.min(1.0 - fx).min(fy).min(1.0 - fy);
                let band = ((0.06 - d) / 0.06).clamp(0.0, 1.0);
                let band2 = band * band;
                // Sepia/yellow shift, stronger toward the edges.
                let mut r = buf[i] + yel * (12.0 + 10.0 * m) * (1.0 + band);
                let mut g = buf[i + 1] + yel * (7.0 + 6.0 * m) * (1.0 + 0.8 * band);
                let mut b = buf[i + 2] - yel * (10.0 + 8.0 * m) * (1.0 + 0.8 * band);
                let v = mot * 6.0 * m; // neutral mottling
                r += v;
                g += v;
                b += v;
                r -= edg * 30.0 * band2;
                g -= edg * 34.0 * band2;
                b -= edg * 44.0 * band2;
                let c = 1.0 + (fx * 12.57 + o3).sin() * syc * cok * 0.05; // cockling
                r *= c;
                g *= c;
                b *= c;
                r += laid;
                g += laid;
                b += laid;
                if p.water_present {
                    let dx = x as f32 - wcx;
                    let dy = y as f32 - wcy;
                    let t = (dx * dx + dy * dy).sqrt() / wr.max(1.0);
                    if t < 1.25 {
                        let inside = 1.0 - smoothstep(0.65, 1.1, t);
                        let rim = (-((t - 1.0) / 0.05).powi(2)).exp(); // tide line
                        let wa = p.water_amount * strength;
                        r -= wa * (16.0 * inside + 46.0 * rim);
                        g -= wa * (16.0 * inside + 52.0 * rim);
                        b -= wa * (12.0 * inside + 56.0 * rim);
                    }
                }
                buf[i] = r.clamp(0.0, 255.0);
                buf[i + 1] = g.clamp(0.0, 255.0);
                buf[i + 2] = b.clamp(0.0, 255.0);
            }
        }
    } // --coarse: skip tint/mottle/edge/cockling/water loop
    if coarse {
        // Params record what was actually applied.
        p.yellowing = 0.0;
        p.mottle = 0.0;
        p.edge = 0.0;
        p.cockle = 0.0;
        p.water_present = false;
        p.water_amount = 0.0;
    }
    // Foxing: brown spots clustered around a few deterministic centers.
    let n_spots = (p.foxing_spots as f32 * strength).round() as usize;
    if n_spots > 0 {
        let nc = ((n_spots / 10) + 2).clamp(2, 6);
        let mut cc = [(0.0f32, 0.0f32); 6];
        for slot in cc.iter_mut().take(nc) {
            *slot = (rng.gen::<f32>() * wf, rng.gen::<f32>() * hf);
        }
        let spread = min_dim * 0.06;
        for _ in 0..n_spots {
            let k = (rng.gen::<f32>() * nc as f32) as usize % nc;
            let mut bell = || rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>() - 1.5;
            let (px0, py0) = (cc[k].0 + bell() * spread, cc[k].1 + bell() * spread);
            let rad = (0.6 + rng.gen::<f32>() * 1.8) * px;
            let a = p.foxing_amount * strength * (0.25 + 0.75 * rng.gen::<f32>());
            let (x0, x1) = (
                (px0 - rad).floor().max(0.0) as u32,
                (px0 + rad).ceil().min(wf - 1.0).max(0.0) as u32,
            );
            let (y0, y1) = (
                (py0 - rad).floor().max(0.0) as u32,
                (py0 + rad).ceil().min(hf - 1.0).max(0.0) as u32,
            );
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let i = ((yy * w + xx) * 3) as usize;
                    if i + 2 >= buf.len() {
                        continue;
                    }
                    let dd = ((xx as f32 - px0).powi(2) + (yy as f32 - py0).powi(2)).sqrt();
                    if dd > rad {
                        continue;
                    }
                    let f = (1.0 - dd / rad).powi(2) * a;
                    buf[i] = (buf[i] - 36.0 * f).clamp(0.0, 255.0);
                    buf[i + 1] = (buf[i + 1] - 66.0 * f).clamp(0.0, 255.0);
                    buf[i + 2] = (buf[i + 2] - 88.0 * f).clamp(0.0, 255.0);
                }
            }
        }
    }
    // Paper fibers: sparse 1px light/dark short horizontal strokes.
    let n_fib = (p.fiber_strokes as f32 * strength).round() as usize;
    for _ in 0..n_fib {
        let (x0, y0) = (rng.gen_range(0..w), rng.gen_range(0..h));
        // Physical fiber length scales with `px` (floor 2 keeps the stroke;
        // truncation, not rounding, so px = 1.0 matches the legacy lengths).
        let len = (((3.0 + rng.gen::<f32>() * 8.0) * px).max(2.0)) as u32;
        let dark = rng.gen_bool(0.5);
        let amp = p.fiber_amount * strength * (5.0 + rng.gen::<f32>() * 9.0);
        for k in 0..len {
            let i = ((y0 * w + (x0 + k).min(w - 1)) * 3) as usize;
            if i + 2 >= buf.len() {
                continue;
            }
            for c in 0..3 {
                buf[i + c] = if dark {
                    (buf[i + c] - amp).clamp(0.0, 255.0)
                } else {
                    (buf[i + c] + amp).clamp(0.0, 255.0)
                };
            }
        }
    }
    p
}

// ===========================================================================
// Wave 3: Micro-Surface BRDF (PaperStock, SurfaceBrdfParams, BrdfTiles)
// ===========================================================================

const BRDF_TILE: usize = 128;
const BRDF_MASK: usize = BRDF_TILE - 1;

/// Physical paper substrate stock driving the micro-surface BRDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PaperStock {
    /// Legacy bypass (identity BRDF).
    None,
    /// Neutral high-albedo matte chalky office bond with subtle cool OBA.
    #[default]
    MatteOfficeBond,
    /// Glossy / semi-glossy coated stock with tight specular lobe + crease breakup.
    CoatedGlossy,
    /// Thermal receipt paper: cool synthetic OBA, satin topcoat sheen, muted charcoal Dmax.
    ThermalReceipt,
    /// Low-albedo warm-gray porous newsprint: muted black ceiling, pulp grain, ink spread.
    PorousNewsprint,
    /// Heavy uncoated cotton rag / parchment: warm cream-ochre, directional tooth, velvet rolloff.
    UncoatedRag,
}

impl PaperStock {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::MatteOfficeBond => "matte-office-bond",
            Self::CoatedGlossy => "coated-glossy",
            Self::ThermalReceipt => "thermal-receipt",
            Self::PorousNewsprint => "porous-newsprint",
            Self::UncoatedRag => "uncoated-rag",
        }
    }
}

/// Sampled micro-surface BRDF parameters recorded in `DegradeParams`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SurfaceBrdfParams {
    pub stock: PaperStock,
    pub albedo_ceil: f32,
    pub base_tint: [f32; 3],
    pub black_floor: f32,
    pub oba_strength: f32,
    pub spec_gain: f32,
    pub spec_sharpness: u8,
    pub crease_breakup: f32,
    pub absorb_grain: f32,
    pub tooth_amp: f32,
    pub velvet_knee: f32,
    pub velvet_k: f32,
    pub ink_spread: f32,
}

impl Default for SurfaceBrdfParams {
    fn default() -> Self {
        Self {
            stock: PaperStock::None,
            albedo_ceil: 1.0,
            base_tint: [0.0; 3],
            black_floor: 0.0,
            oba_strength: 0.0,
            spec_gain: 0.0,
            spec_sharpness: 1,
            crease_breakup: 0.0,
            absorb_grain: 0.0,
            tooth_amp: 0.0,
            velvet_knee: 255.0,
            velvet_k: 0.0,
            ink_spread: 0.0,
        }
    }
}

impl SurfaceBrdfParams {
    /// Deterministically sample stock BRDF knobs from `(file_seed, TAG_BRDF)`.
    pub fn sample(
        stock: PaperStock,
        file_seed: u64,
        strength: f32,
        allow_ink_spread: bool,
    ) -> Self {
        let s = strength.clamp(0.0, 1.0);
        if matches!(stock, PaperStock::None) || s <= 0.02 {
            return Self::default();
        }
        let mut rng =
            ChaCha8Rng::seed_from_u64(crate::banks::bank_seed(file_seed, crate::banks::TAG_BRDF));
        let u = |r: &mut ChaCha8Rng, lo: f32, hi: f32| r.gen_range(lo..hi);
        match stock {
            PaperStock::None => Self::default(),
            PaperStock::CoatedGlossy => Self {
                stock,
                albedo_ceil: u(&mut rng, 0.98, 1.00),
                base_tint: [-u(&mut rng, 1.0, 3.0) * s, 0.0, u(&mut rng, 2.5, 5.5) * s],
                black_floor: u(&mut rng, 0.0, 4.0) * s,
                oba_strength: u(&mut rng, 0.25, 0.45) * s,
                spec_gain: u(&mut rng, 0.35, 0.75) * s,
                spec_sharpness: 3,
                crease_breakup: u(&mut rng, 0.45, 0.85) * s,
                absorb_grain: 0.005 * s,
                tooth_amp: 0.0,
                velvet_knee: 255.0,
                velvet_k: 0.0,
                ink_spread: 0.0,
            },
            PaperStock::ThermalReceipt => Self {
                stock,
                albedo_ceil: u(&mut rng, 0.93, 0.97),
                base_tint: [
                    -u(&mut rng, 2.0, 4.5) * s,
                    u(&mut rng, 0.5, 2.0) * s,
                    u(&mut rng, 4.0, 8.0) * s,
                ],
                black_floor: u(&mut rng, 18.0, 38.0) * s,
                oba_strength: u(&mut rng, 0.30, 0.55) * s,
                spec_gain: u(&mut rng, 0.22, 0.45) * s,
                spec_sharpness: 2,
                crease_breakup: u(&mut rng, 0.55, 0.90) * s,
                absorb_grain: 0.010 * s,
                tooth_amp: 0.0,
                velvet_knee: 255.0,
                velvet_k: 0.0,
                ink_spread: 0.0,
            },
            PaperStock::MatteOfficeBond => Self {
                stock,
                albedo_ceil: u(&mut rng, 0.96, 0.99),
                base_tint: [0.0, 0.0, u(&mut rng, 1.0, 3.0) * s],
                black_floor: u(&mut rng, 4.0, 12.0) * s,
                oba_strength: u(&mut rng, 0.15, 0.32) * s,
                spec_gain: u(&mut rng, 0.04, 0.10) * s,
                spec_sharpness: 1,
                crease_breakup: u(&mut rng, 0.05, 0.12) * s,
                absorb_grain: u(&mut rng, 0.014, 0.022) * s,
                tooth_amp: 0.008 * s,
                velvet_knee: 238.0,
                velvet_k: 0.15 * s,
                ink_spread: 0.0,
            },
            PaperStock::PorousNewsprint => Self {
                stock,
                albedo_ceil: 1.0 - (1.0 - u(&mut rng, 0.78, 0.87)) * s,
                base_tint: [
                    u(&mut rng, 2.0, 4.5) * s,
                    u(&mut rng, 0.5, 2.0) * s,
                    -u(&mut rng, 5.0, 9.0) * s,
                ],
                black_floor: u(&mut rng, 32.0, 52.0) * s,
                oba_strength: 0.0,
                spec_gain: 0.0,
                spec_sharpness: 1,
                crease_breakup: 0.0,
                absorb_grain: u(&mut rng, 0.055, 0.085) * s,
                tooth_amp: u(&mut rng, 0.015, 0.028) * s,
                velvet_knee: 205.0,
                velvet_k: 0.30 * s,
                ink_spread: if allow_ink_spread {
                    u(&mut rng, 0.12, 0.28) * s
                } else {
                    0.0
                },
            },
            PaperStock::UncoatedRag => Self {
                stock,
                albedo_ceil: 1.0 - (1.0 - u(&mut rng, 0.90, 0.95)) * s,
                base_tint: [
                    u(&mut rng, 6.0, 11.0) * s,
                    u(&mut rng, 1.0, 3.5) * s,
                    -u(&mut rng, 12.0, 20.0) * s,
                ],
                black_floor: u(&mut rng, 10.0, 22.0) * s,
                oba_strength: 0.0,
                spec_gain: u(&mut rng, 0.02, 0.06) * s,
                spec_sharpness: 1,
                crease_breakup: u(&mut rng, 0.05, 0.10) * s,
                absorb_grain: u(&mut rng, 0.020, 0.032) * s,
                tooth_amp: u(&mut rng, 0.045, 0.080) * s,
                velvet_knee: 218.0,
                velvet_k: 0.45 * s,
                ink_spread: if allow_ink_spread {
                    u(&mut rng, 0.03, 0.08) * s
                } else {
                    0.0
                },
            },
        }
    }
}

/// Apply stock base albedo ceiling, Dmax black floor, chromaticity tint, OBA
/// paper lift, velvet highlight rolloff, and micro-absorbance grain in one
/// cache-linear pass (zero extra allocations beyond a 64 KiB L1 tile).
pub fn apply_surface_brdf_base(
    buf: &mut [f32],
    w: u32,
    h: u32,
    file_seed: u64,
    brdf: &SurfaceBrdfParams,
) {
    if matches!(brdf.stock, PaperStock::None) || w == 0 || h == 0 {
        return;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(
        crate::banks::bank_seed(file_seed, crate::banks::TAG_BRDF) ^ 0xA24B_AED4_963E_E407,
    );
    let mut micro = vec![0.0f32; BRDF_TILE * BRDF_TILE];
    for v in &mut micro {
        *v = rng.gen::<f32>() * 2.0 - 1.0;
    }
    let gain = (brdf.albedo_ceil * 255.0 - brdf.black_floor) * (1.0 / 255.0);
    let w_us = w as usize;
    for y in 0..h as usize {
        let m_row = (y & BRDF_MASK) << 7;
        let row = &mut buf[y * w_us * 3..(y + 1) * w_us * 3];
        for (x, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let m_n = micro[m_row | (x & BRDF_MASK)];
            let abs_mul = 1.0 + brdf.absorb_grain * m_n;
            let mut r = (px[0] * gain + brdf.black_floor + brdf.base_tint[0]) * abs_mul;
            let mut g = (px[1] * gain + brdf.black_floor + brdf.base_tint[1]) * abs_mul;
            let mut b = (px[2] * gain + brdf.black_floor + brdf.base_tint[2]) * abs_mul;
            if brdf.oba_strength > 0.0 {
                let luma01 = (r + g + b) * (1.0 / 765.0);
                let paper_w = ((luma01 - 0.25) * 1.5).clamp(0.0, 1.0);
                let oba = brdf.oba_strength * paper_w * 255.0;
                r -= 0.015 * oba;
                g += 0.004 * oba;
                b += 0.048 * oba;
            }
            if brdf.velvet_k > 0.0 {
                let vr = |v: f32| {
                    let over = (v - brdf.velvet_knee).max(0.0);
                    v.min(brdf.velvet_knee) + over / (1.0 + brdf.velvet_k * over * 0.015625)
                };
                r = vr(r);
                g = vr(g);
                b = vr(b);
            }
            px[0] = r.clamp(0.0, 255.0);
            px[1] = g.clamp(0.0, 255.0);
            px[2] = b.clamp(0.0, 255.0);
        }
    }
}

/// Zero-centroid-shift, luma-gated capillary ink spread for porous newsprint / rag.
/// Uses a 2-row L1 ring buffer and skips ~94% of pixels.
pub fn apply_porous_ink_spread(buf: &mut [f32], w: u32, h: u32, spread: f32) {
    if spread <= 0.01 || w < 3 || h < 3 {
        return;
    }
    let row_len = (w as usize) * 3;
    let mut prev = buf[0..row_len].to_vec();
    let mut curr = buf[0..row_len].to_vec();
    for y in 1..(h - 1) as usize {
        let base = y * row_len;
        let next_base = (y + 1) * row_len;
        prev.copy_from_slice(&curr);
        curr.copy_from_slice(&buf[base..base + row_len]);
        for x in 1..(w - 1) as usize {
            let o = x * 3;
            let (r, g, b) = (curr[o], curr[o + 1], curr[o + 2]);
            let min_n = (curr[o - 3] + curr[o - 2] + curr[o - 1])
                .min(curr[o + 3] + curr[o + 4] + curr[o + 5])
                .min(prev[o] + prev[o + 1] + prev[o + 2])
                .min(buf[next_base + o] + buf[next_base + o + 1] + buf[next_base + o + 2]);
            if min_n > 360.0 {
                continue;
            }
            let avg_n_r = 0.25 * (curr[o - 3] + curr[o + 3] + prev[o] + buf[next_base + o]);
            let avg_n_g = 0.25 * (curr[o - 2] + curr[o + 4] + prev[o + 1] + buf[next_base + o + 1]);
            let avg_n_b = 0.25 * (curr[o - 1] + curr[o + 5] + prev[o + 2] + buf[next_base + o + 2]);
            let w_blend = spread * ((360.0 - min_n) * (1.0 / 360.0));
            buf[base + o] = r + (avg_n_r - r).min(0.0) * w_blend;
            buf[base + o + 1] = g + (avg_n_g - g).min(0.0) * w_blend;
            buf[base + o + 2] = b + (avg_n_b - b).min(0.0) * w_blend;
        }
    }
}

// ===========================================================================
// Wave 3: Y-Sorted Curved Cellulose Fibers, Recycled Pulp Flecks & Lint
// ===========================================================================

/// Profile-calibrated cellulose fiber, pulp fleck, and surface lint config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulpInclusionConfig {
    pub enabled: bool,
    pub fiber_density: f32,
    pub fiber_opacity: f32,
    pub rag_thread_prob: f32,
    pub fleck_density: f32,
    pub fleck_dark_prob: f32,
    pub fleck_opacity: f32,
    pub lint_count: f32,
    pub dust_count: f32,
    pub lint_opacity: f32,
    pub scanner_glass_shadow: bool,
}

impl PulpInclusionConfig {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            fiber_density: 0.0,
            fiber_opacity: 0.0,
            rag_thread_prob: 0.0,
            fleck_density: 0.0,
            fleck_dark_prob: 0.5,
            fleck_opacity: 0.0,
            lint_count: 0.0,
            dust_count: 0.0,
            lint_opacity: 0.0,
            scanner_glass_shadow: false,
        }
    }
    pub const fn phone() -> Self {
        Self {
            enabled: true,
            fiber_density: 380.0,
            fiber_opacity: 0.24,
            rag_thread_prob: 0.03,
            fleck_density: 420.0,
            fleck_dark_prob: 0.55,
            fleck_opacity: 0.28,
            lint_count: 10.0,
            dust_count: 35.0,
            lint_opacity: 0.42,
            scanner_glass_shadow: false,
        }
    }
    pub const fn archival() -> Self {
        Self {
            enabled: true,
            fiber_density: 1150.0,
            fiber_opacity: 0.42,
            rag_thread_prob: 0.12,
            fleck_density: 1600.0,
            fleck_dark_prob: 0.68,
            fleck_opacity: 0.45,
            lint_count: 16.0,
            dust_count: 60.0,
            lint_opacity: 0.45,
            scanner_glass_shadow: false,
        }
    }
    pub const fn thermal() -> Self {
        Self {
            enabled: true,
            fiber_density: 90.0,
            fiber_opacity: 0.14,
            rag_thread_prob: 0.0,
            fleck_density: 320.0,
            fleck_dark_prob: 0.50,
            fleck_opacity: 0.22,
            lint_count: 22.0,
            dust_count: 55.0,
            lint_opacity: 0.52,
            scanner_glass_shadow: false,
        }
    }
    pub const fn office_scan() -> Self {
        Self {
            enabled: true,
            fiber_density: 110.0,
            fiber_opacity: 0.14,
            rag_thread_prob: 0.01,
            fleck_density: 120.0,
            fleck_dark_prob: 0.45,
            fleck_opacity: 0.18,
            lint_count: 8.0,
            dust_count: 45.0,
            lint_opacity: 0.50,
            scanner_glass_shadow: true,
        }
    }
    pub const fn photocopy() -> Self {
        Self {
            enabled: true,
            fiber_density: 260.0,
            fiber_opacity: 0.30,
            rag_thread_prob: 0.0,
            fleck_density: 950.0,
            fleck_dark_prob: 0.85,
            fleck_opacity: 0.50,
            lint_count: 14.0,
            dust_count: 90.0,
            lint_opacity: 0.62,
            scanner_glass_shadow: true,
        }
    }
    pub const fn book_gutter() -> Self {
        Self {
            enabled: true,
            fiber_density: 780.0,
            fiber_opacity: 0.34,
            rag_thread_prob: 0.04,
            fleck_density: 2200.0,
            fleck_dark_prob: 0.72,
            fleck_opacity: 0.42,
            lint_count: 12.0,
            dust_count: 45.0,
            lint_opacity: 0.40,
            scanner_glass_shadow: false,
        }
    }
    pub const fn sheet_music() -> Self {
        Self {
            enabled: true,
            fiber_density: 520.0,
            fiber_opacity: 0.24,
            rag_thread_prob: 0.06,
            fleck_density: 580.0,
            fleck_dark_prob: 0.55,
            fleck_opacity: 0.26,
            lint_count: 7.0,
            dust_count: 25.0,
            lint_opacity: 0.32,
            scanner_glass_shadow: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PulpInclusionOutcome {
    pub cellulose_fiber_count: u32,
    pub cellulose_fiber_opacity: f32,
    pub rag_thread_count: u32,
    pub pulp_fleck_count: u32,
    pub surface_lint_count: u32,
    pub dust_mote_count: u32,
}

#[inline]
fn paper_substrate_weight(r: f32, g: f32, b: f32) -> f32 {
    let luma = 0.299 * r + 0.587 * g + 0.114 * b;
    ((luma - 75.0) * (1.0 / 110.0)).clamp(0.15, 1.0)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn splat_2x2(
    buf: &mut [f32],
    w: i32,
    h: i32,
    fx: f32,
    fy: f32,
    rgb: [f32; 3],
    alpha: f32,
    attenuate_on_ink: bool,
) {
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let weights = [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x0 + 1, y0, tx * (1.0 - ty)),
        (x0, y0 + 1, (1.0 - tx) * ty),
        (x0 + 1, y0 + 1, tx * ty),
    ];
    for (x, y, wt) in weights {
        if x < 0 || y < 0 || x >= w || y >= h {
            continue;
        }
        let i = ((y as usize) * (w as usize) + (x as usize)) * 3;
        let a = if attenuate_on_ink {
            alpha * wt * paper_substrate_weight(buf[i], buf[i + 1], buf[i + 2])
        } else {
            alpha * wt
        };
        if a > 1e-4 {
            let inv = 1.0 - a;
            buf[i] = (buf[i] * inv + rgb[0] * a).clamp(0.0, 255.0);
            buf[i + 1] = (buf[i + 1] * inv + rgb[1] * a).clamp(0.0, 255.0);
            buf[i + 2] = (buf[i + 2] * inv + rgb[2] * a).clamp(0.0, 255.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_bezier_fiber(
    buf: &mut [f32],
    w: i32,
    h: i32,
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    rgb: [f32; 3],
    peak_alpha: f32,
    thick: bool,
    attenuate_on_ink: bool,
) {
    let chord = (p2[0] - p0[0]).hypot(p2[1] - p0[1]).max(2.0);
    let steps = (chord * 1.35).ceil().clamp(4.0, 72.0) as u32;
    let step_alpha = 1.0 - (1.0 - peak_alpha.clamp(0.0, 0.95)).powf(0.42);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let u = 1.0 - t;
        let x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0];
        let y = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1];
        let taper = (std::f32::consts::PI * t).sin().max(0.0).powf(0.45);
        let a = step_alpha * taper;
        splat_2x2(buf, w, h, x, y, rgb, a, attenuate_on_ink);
        if thick {
            let dx = 2.0 * u * (p1[0] - p0[0]) + 2.0 * t * (p2[0] - p1[0]);
            let dy = 2.0 * u * (p1[1] - p0[1]) + 2.0 * t * (p2[1] - p1[1]);
            let inv_l = 0.55 / dx.hypot(dy).max(1e-4);
            let (nx, ny) = (-dy * inv_l, dx * inv_l);
            splat_2x2(buf, w, h, x + nx, y + ny, rgb, a * 0.55, attenuate_on_ink);
        }
    }
}

const FLECK_SZ: usize = 7;
const FLECK_BANK_N: usize = 16;

struct FleckBank {
    masks: [[f32; FLECK_SZ * FLECK_SZ]; FLECK_BANK_N],
}

impl FleckBank {
    fn build(rng: &mut ChaCha8Rng, px: f32) -> Self {
        let mut masks = [[0.0f32; FLECK_SZ * FLECK_SZ]; FLECK_BANK_N];
        let half = (FLECK_SZ / 2) as f32;
        for m in &mut masks {
            let r0 = (1.1 + rng.gen::<f32>() * 1.45) * px.clamp(0.45, 1.25);
            let elong = 1.0 + rng.gen::<f32>() * 1.3;
            let ang = rng.gen::<f32>() * std::f32::consts::TAU;
            let (ca, sa) = (ang.cos(), ang.sin());
            let (c2, p2) = (
                rng.gen_range(0.08..0.28f32),
                rng.gen::<f32>() * std::f32::consts::TAU,
            );
            let (c3, p3) = (
                rng.gen_range(0.05..0.20f32),
                rng.gen::<f32>() * std::f32::consts::TAU,
            );
            for dy in 0..FLECK_SZ {
                for dx in 0..FLECK_SZ {
                    let rx = dx as f32 - half;
                    let ry = dy as f32 - half;
                    let u = (rx * ca + ry * sa) / elong;
                    let v = -rx * sa + ry * ca;
                    let d = u.hypot(v);
                    let th = v.atan2(u);
                    let rr = (r0 * (1.0 + c2 * (2.0 * th + p2).cos() + c3 * (3.0 * th + p3).cos()))
                        .max(0.5);
                    let t = ((1.05 - d / rr) / (1.05 - 0.68)).clamp(0.0, 1.0);
                    m[dy * FLECK_SZ + dx] = t * t * (3.0 - 2.0 * t);
                }
            }
        }
        Self { masks }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn stamp(
        &self,
        buf: &mut [f32],
        w: i32,
        h: i32,
        cx: i32,
        cy: i32,
        sprite_idx: usize,
        sym: u8,
        rgb: [f32; 3],
        alpha: f32,
    ) {
        let mask = &self.masks[sprite_idx % FLECK_BANK_N];
        let half = (FLECK_SZ / 2) as i32;
        let flip_x = (sym & 1) != 0;
        let flip_y = (sym & 2) != 0;
        let swap_xy = (sym & 4) != 0;
        for dy in 0..FLECK_SZ as i32 {
            let y = cy + dy - half;
            if y < 0 || y >= h {
                continue;
            }
            let row = (y as usize) * (w as usize);
            for dx in 0..FLECK_SZ as i32 {
                let x = cx + dx - half;
                if x < 0 || x >= w {
                    continue;
                }
                let sx = if flip_x {
                    FLECK_SZ - 1 - dx as usize
                } else {
                    dx as usize
                };
                let sy = if flip_y {
                    FLECK_SZ - 1 - dy as usize
                } else {
                    dy as usize
                };
                let m = if swap_xy {
                    mask[sx * FLECK_SZ + sy]
                } else {
                    mask[sy * FLECK_SZ + sx]
                };
                if m < 1e-3 {
                    continue;
                }
                let i = (row + x as usize) * 3;
                let a = alpha * m * paper_substrate_weight(buf[i], buf[i + 1], buf[i + 2]);
                let inv = 1.0 - a;
                buf[i] = (buf[i] * inv + rgb[0] * a).clamp(0.0, 255.0);
                buf[i + 1] = (buf[i + 1] * inv + rgb[1] * a).clamp(0.0, 255.0);
                buf[i + 2] = (buf[i + 2] * inv + rgb[2] * a).clamp(0.0, 255.0);
            }
        }
    }
}

/// Apply Y-sorted curved Bezier cellulose fibers, recycled pulp flecks, and
/// stray surface lint curls using an independent keyed RNG stream `(file_seed, TAG_PULP_INCLUSIONS)`.
pub fn apply_pulp_inclusions(
    buf: &mut [f32],
    w: u32,
    h: u32,
    file_seed: u64,
    strength: f32,
    px: f32,
    cfg: &PulpInclusionConfig,
) -> PulpInclusionOutcome {
    let s = strength.clamp(0.0, 1.0);
    if !cfg.enabled || s <= 0.02 || w < 16 || h < 16 || buf.len() < (w as usize) * (h as usize) * 3
    {
        return PulpInclusionOutcome::default();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(crate::banks::bank_seed(
        file_seed,
        crate::banks::TAG_PULP_INCLUSIONS,
    ));
    let area_norm = ((w as f32 * h as f32) / (2480.0 * 3508.0)).clamp(0.0004, 2.0);
    let (wi, hi) = (w as i32, h as i32);
    let (wf, hf) = (w as f32, h as f32);

    // 1. Curved Bezier cellulose fibers (Y-sorted for L2/L3 cache streaming).
    let n_fibers = (cfg.fiber_density * area_norm * (0.4 + 0.6 * s)).round() as usize;
    let mut fibers = Vec::with_capacity(n_fibers);
    let mut rag_count = 0u32;
    let mut op_sum = 0.0f32;
    for _ in 0..n_fibers {
        let x0 = rng.gen::<f32>() * wf;
        let y0 = rng.gen::<f32>() * hf;
        let len = rng.gen_range(9.0..34.0) * px.clamp(0.35, 1.5);
        let ang = rng.gen::<f32>() * std::f32::consts::TAU;
        let x2 = x0 + len * ang.cos();
        let y2 = y0 + len * ang.sin();
        let bend = rng.gen_range(-0.55..0.55) * len;
        let x1 = 0.5 * (x0 + x2) - bend * ang.sin();
        let y1 = 0.5 * (y0 + y2) + bend * ang.cos();
        let (rgb, is_rag) = if rng.gen::<f32>() < cfg.rag_thread_prob {
            rag_count += 1;
            if rng.gen_bool(0.55) {
                ([78.0, 96.0, 134.0], true)
            } else {
                ([142.0, 74.0, 72.0], true)
            }
        } else {
            let r = rng.gen::<f32>();
            if r < 0.45 {
                ([105.0, 84.0, 62.0], false)
            } else if r < 0.80 {
                ([176.0, 152.0, 118.0], false)
            } else {
                ([250.0, 247.0, 240.0], false)
            }
        };
        let alpha =
            (cfg.fiber_opacity * (0.45 + 0.55 * s) * rng.gen_range(0.55..1.15)).clamp(0.02, 0.75);
        let thick = !is_rag && rng.gen_bool(0.25);
        op_sum += alpha;
        fibers.push((y0 as i32, [x0, y0], [x1, y1], [x2, y2], rgb, alpha, thick));
    }
    fibers.sort_unstable_by_key(|f| f.0);
    for &(_, p0, p1, p2, rgb, alpha, thick) in &fibers {
        draw_bezier_fiber(buf, wi, hi, p0, p1, p2, rgb, alpha, thick, true);
    }

    // 2. Recycled pulp & wood-chip flecks (L1-resident 7×7 sprite bank, Y-sorted).
    let bank = FleckBank::build(&mut rng, px);
    let n_flecks = (cfg.fleck_density * area_norm * (0.4 + 0.6 * s)).round() as usize;
    let mut flecks = Vec::with_capacity(n_flecks);
    for _ in 0..n_flecks {
        let cx = rng.gen_range(0..wi);
        let cy = rng.gen_range(0..hi);
        let idx = rng.gen_range(0..FLECK_BANK_N);
        let sym = rng.gen::<u8>() & 7;
        let rgb = if rng.gen::<f32>() < cfg.fleck_dark_prob {
            let t = rng.gen::<f32>();
            [64.0 + t * 38.0, 50.0 + t * 30.0, 38.0 + t * 20.0]
        } else {
            let t = rng.gen::<f32>();
            [230.0 + t * 16.0, 222.0 + t * 19.0, 204.0 + t * 26.0]
        };
        let alpha =
            (cfg.fleck_opacity * (0.45 + 0.55 * s) * rng.gen_range(0.5..1.1)).clamp(0.03, 0.80);
        flecks.push((cy, cx, idx, sym, rgb, alpha));
    }
    flecks.sort_unstable_by_key(|f| f.0);
    for &(cy, cx, idx, sym, rgb, alpha) in &flecks {
        bank.stamp(buf, wi, hi, cx, cy, idx, sym, rgb, alpha);
    }

    // 3. Stray surface lint curls & pinpoint dust motes (sit on top of ink).
    let n_lint = (cfg.lint_count * area_norm.sqrt() * (0.4 + 0.6 * s)).round() as usize;
    for _ in 0..n_lint {
        let x0 = rng.gen::<f32>() * wf;
        let y0 = rng.gen::<f32>() * hf;
        let len = rng.gen_range(20.0..60.0) * px.clamp(0.4, 1.4);
        let ang = rng.gen::<f32>() * std::f32::consts::TAU;
        let x2 = x0 + len * ang.cos();
        let y2 = y0 + len * ang.sin();
        let bend = rng.gen_range(-0.6..0.6) * len;
        let x1 = 0.5 * (x0 + x2) - bend * ang.sin();
        let y1 = 0.5 * (y0 + y2) + bend * ang.cos();
        let bright = rng.gen_bool(0.45);
        let rgb = if bright {
            let v: f32 = rng.gen_range(246.0..253.0);
            [v, v, (v + 2.0).min(255.0)]
        } else {
            let v: f32 = rng.gen_range(48.0..84.0);
            [v, v - 4.0, v - 6.0]
        };
        let alpha =
            (cfg.lint_opacity * (0.5 + 0.5 * s) * rng.gen_range(0.6..1.0)).clamp(0.08, 0.85);
        if bright && cfg.scanner_glass_shadow {
            let sh_off = 1.2 * px.clamp(0.5, 1.5);
            draw_bezier_fiber(
                buf,
                wi,
                hi,
                [x0 + sh_off, y0 + sh_off * 0.85],
                [x1 + sh_off, y1 + sh_off * 0.85],
                [x2 + sh_off, y2 + sh_off * 0.85],
                [36.0, 36.0, 38.0],
                alpha * 0.35,
                false,
                false,
            );
        }
        draw_bezier_fiber(
            buf,
            wi,
            hi,
            [x0, y0],
            [x1, y1],
            [x2, y2],
            rgb,
            alpha,
            false,
            false,
        );
    }

    let n_dust = (cfg.dust_count * area_norm.sqrt() * (0.4 + 0.6 * s)).round() as usize;
    for _ in 0..n_dust {
        let fx = rng.gen::<f32>() * wf;
        let fy = rng.gen::<f32>() * hf;
        let bright = rng.gen_bool(0.45);
        let rgb = if bright {
            [248.0, 248.0, 250.0]
        } else {
            [52.0, 48.0, 44.0]
        };
        let alpha = (cfg.lint_opacity * rng.gen_range(0.45..0.95)).clamp(0.08, 0.85);
        splat_2x2(buf, wi, hi, fx, fy, rgb, alpha, false);
    }

    PulpInclusionOutcome {
        cellulose_fiber_count: n_fibers as u32,
        cellulose_fiber_opacity: if n_fibers > 0 {
            op_sum / n_fibers as f32
        } else {
            0.0
        },
        rag_thread_count: rag_count,
        pulp_fleck_count: n_flecks as u32,
        surface_lint_count: n_lint as u32,
        dust_mote_count: n_dust as u32,
    }
}

// ===========================================================================
// Wave 3: Archival Aging, Edge Oxidation LUT, Ferric Foxing, Grease & Watermark
// ===========================================================================

#[derive(Debug, Clone, Default)]
pub struct ArchivalPaperParams {
    pub edge_oxidation_strength: f32,
    pub edge_oxidation_depths: [f32; 4],
    pub foxing_cluster_count: u32,
    pub foxing_halo_strength: f32,
    pub grease_patch_count: u32,
    pub grease_patch_strength: f32,
    pub watermark_style: String,
    pub watermark_strength: f32,
}

/// Apply Stage 1d archival chemical paper-aging passes:
/// 1. O(W+H) separable acidic edge-oxidation vignette + chain-line/show-through
/// 2. Organic ferric-oxide foxing clusters & capillary halos
/// 3. Page-turn corner translucent finger-oil / grease patches
/// 4. Subtle papermaker watermark emblem SDF
#[allow(clippy::too_many_arguments)]
pub fn apply_archival_paper(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    profile_scale: f32,
    coarse: bool,
    px: f32,
) -> ArchivalPaperParams {
    let s = (strength * profile_scale).clamp(0.0, 1.0);
    if s <= 0.02 || w < 16 || h < 16 || buf.len() < (w as usize) * (h as usize) * 3 {
        return ArchivalPaperParams {
            watermark_style: "none".to_string(),
            ..Default::default()
        };
    }

    let (ox_str, ox_depths, wm_str, wm_style) =
        apply_edge_oxidation_and_chain(buf, w, h, rng, s, coarse, px);
    let (fox_clusters, fox_str) = apply_ferric_foxing_clusters(buf, w, h, rng, s, px);
    let (grease_n, grease_str) = apply_grease_corner_patches(buf, w, h, rng, s);
    if wm_str > 0.0 && wm_style != "none" {
        apply_watermark_emblem(buf, w, h, rng, wm_str, wm_style);
    }

    ArchivalPaperParams {
        edge_oxidation_strength: ox_str,
        edge_oxidation_depths: ox_depths,
        foxing_cluster_count: fox_clusters,
        foxing_halo_strength: fox_str,
        grease_patch_count: grease_n,
        grease_patch_strength: grease_str,
        watermark_style: wm_style.to_string(),
        watermark_strength: wm_str,
    }
}

fn apply_edge_oxidation_and_chain(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    s: f32,
    coarse: bool,
    px: f32,
) -> (f32, [f32; 4], f32, &'static str) {
    let ox_strength = if rng.gen::<f32>() < 0.80 {
        s * rng.gen_range(0.28..0.85)
    } else {
        0.0
    };
    let spine_left = rng.gen_bool(0.5);
    let base_d: f32 = rng.gen_range(0.05..0.14);
    let depths: [f32; 4] = [
        base_d
            * if spine_left {
                0.30
            } else {
                rng.gen_range(0.85..1.25)
            },
        base_d
            * if !spine_left {
                0.30
            } else {
                rng.gen_range(0.85..1.25)
            },
        base_d * rng.gen_range(0.75..1.15),
        base_d * rng.gen_range(0.85..1.25),
    ];
    let (ph1, ph2, ph3, ph4) = (
        rng.gen_range(0.0..std::f32::consts::TAU),
        rng.gen_range(0.0..std::f32::consts::TAU),
        rng.gen_range(0.0..std::f32::consts::TAU),
        rng.gen_range(0.0..std::f32::consts::TAU),
    );
    let wm_roll = rng.gen::<f32>();
    let (wm_style, wm_strength) = if wm_roll < 0.38 {
        let style = ["crest", "diamond", "bars"][rng.gen_range(0..3)];
        (style, s * rng.gen_range(0.018..0.045))
    } else {
        ("none", 0.0)
    };
    let chain_period = (rng.gen_range(140.0..230.0) * px).max(24.0);
    let chain_phase = rng.gen_range(0.0..chain_period);
    let show_amp = wm_strength * 0.45;

    if coarse || (ox_strength <= 0.0 && wm_strength <= 0.0) {
        return (0.0, [0.0; 4], wm_strength, wm_style);
    }

    let (wu, hu) = (w as usize, h as usize);
    let (inv_w, inv_h) = (1.0 / (w as f32).max(1.0), 1.0 / (h as f32).max(1.0));

    let mut ex = vec![0.0f32; wu];
    let mut mod_x = vec![1.0f32; wu];
    let mut chain_x = vec![1.0f32; wu];
    for x in 0..wu {
        let fx = x as f32 * inv_w;
        let dl = fx / depths[0].max(1e-4);
        let dr = (1.0 - fx) / depths[1].max(1e-4);
        let el = 0.65 * (-dl).exp() + 0.35 * (-dl * 3.5).exp();
        let er = 0.65 * (-dr).exp() + 0.35 * (-dr * 3.5).exp();
        ex[x] = (el + er).min(1.15);
        mod_x[x] = 1.0 + 0.14 * (fx * 12.566 + ph1).sin() + 0.08 * (fx * 25.133 + ph2).cos();
        if wm_strength > 0.0 {
            let rem = ((x as f32 + chain_phase) % chain_period) - 0.5 * chain_period;
            let sig = (2.2 * px).max(1.0);
            let ridge = (-0.5 * (rem / sig).powi(2)).exp();
            chain_x[x] = 1.0 + wm_strength * 0.45 * ridge;
        }
    }

    let mut ey = vec![0.0f32; hu];
    let mut mod_y = vec![1.0f32; hu];
    let mut back_y = vec![1.0f32; hu];
    for y in 0..hu {
        let fy = y as f32 * inv_h;
        let dt = fy / depths[2].max(1e-4);
        let db = (1.0 - fy) / depths[3].max(1e-4);
        let et = 0.65 * (-dt).exp() + 0.35 * (-dt * 3.5).exp();
        let eb = 0.65 * (-db).exp() + 0.35 * (-db * 3.5).exp();
        ey[y] = (et + eb).min(1.15);
        mod_y[y] = 1.0 + 0.14 * (fy * 12.566 + ph3).sin() + 0.08 * (fy * 25.133 + ph4).cos();
        if show_amp > 0.0 {
            let band = 0.5 + 0.5 * (fy * 110.0 + ph2).sin();
            back_y[y] = 1.0
                - show_amp * band * (0.3 + 0.7 * (fy * std::f32::consts::TAU + ph4).sin().abs());
        }
    }

    for y in 0..hu {
        let ey_v = ey[y];
        let my_v = mod_y[y];
        let by_v = back_y[y];
        let row = &mut buf[y * wu * 3..(y + 1) * wu * 3];
        for (x, pix) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let ex_v = ex[x];
            let t =
                (ex_v * my_v + ey_v * mod_x[x] - 0.30 * ex_v * ey_v).clamp(0.0, 1.35) * ox_strength;
            let c_mul = chain_x[x] * by_v;
            let tr = (1.0 - 0.09 * t) * c_mul;
            let tg = (1.0 - 0.24 * t) * c_mul;
            let tb = (1.0 - 0.46 * t) * c_mul;
            pix[0] = (pix[0] * tr).clamp(0.0, 255.0);
            pix[1] = (pix[1] * tg).clamp(0.0, 255.0);
            pix[2] = (pix[2] * tb).clamp(0.0, 255.0);
        }
    }

    (ox_strength, depths, wm_strength, wm_style)
}

fn apply_ferric_foxing_clusters(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    s: f32,
    px: f32,
) -> (u32, f32) {
    let strength = s * rng.gen_range(0.30..0.90);
    if strength < 0.08 {
        return (0, 0.0);
    }
    let wf = w as f32;
    let hf = h as f32;
    let min_dim = wf.min(hf);
    let n_clusters = rng.gen_range(2..=6u32);

    for _ in 0..n_clusters {
        let (cx, cy) = if rng.gen::<f32>() < 0.70 {
            match rng.gen_range(0..4u32) {
                0 => (
                    rng.gen_range(0.02..0.22) * wf,
                    rng.gen_range(0.04..0.96) * hf,
                ),
                1 => (
                    rng.gen_range(0.78..0.98) * wf,
                    rng.gen_range(0.04..0.96) * hf,
                ),
                2 => (
                    rng.gen_range(0.04..0.96) * wf,
                    rng.gen_range(0.02..0.22) * hf,
                ),
                _ => (
                    rng.gen_range(0.04..0.96) * wf,
                    rng.gen_range(0.78..0.98) * hf,
                ),
            }
        } else {
            (
                rng.gen_range(0.10..0.90) * wf,
                rng.gen_range(0.10..0.90) * hf,
            )
        };

        let spread = min_dim * rng.gen_range(0.035..0.095);
        let n_macro = rng.gen_range(4..=12u32);

        for _ in 0..n_macro {
            let bell = rng.gen::<f32>() + rng.gen::<f32>() - 1.0;
            let bell2 = rng.gen::<f32>() + rng.gen::<f32>() - 1.0;
            let sx = cx + bell * spread;
            let sy = cy + bell2 * spread;

            let r_core = (rng.gen_range(2.0..7.5) * px).max(1.2);
            let r_halo = r_core * rng.gen_range(2.4..3.6);
            let aspect = rng.gen_range(0.72..1.35);
            let ang = rng.gen_range(0.0..std::f32::consts::TAU);
            let (ca, sa) = (ang.cos(), ang.sin());
            let w2 = rng.gen_range(-0.18..0.18);
            let w3 = rng.gen_range(-0.14..0.14);
            let spot_a = strength * rng.gen_range(0.35..1.0);

            let pad = r_halo * 1.25;
            let x0 = (sx - pad).floor().max(0.0) as u32;
            let x1 = (sx + pad).ceil().min(wf - 1.0) as u32;
            let y0 = (sy - pad).floor().max(0.0) as u32;
            let y1 = (sy + pad).ceil().min(hf - 1.0) as u32;
            let inv_rh = 1.0 / r_halo;
            let inv_rc = 1.0 / r_core;

            for yy in y0..=y1 {
                let dy = yy as f32 - sy;
                let row_off = (yy * w) as usize;
                for xx in x0..=x1 {
                    let dx = xx as f32 - sx;
                    let u = dx * ca + dy * sa;
                    let v = (-dx * sa + dy * ca) * aspect;
                    let r2 = u * u + v * v;
                    if r2 > pad * pad {
                        continue;
                    }
                    let rr = r2.sqrt();
                    let inv_r = 1.0 / rr.max(1e-4);
                    let (ux, vy) = (u * inv_r, v * inv_r);
                    let wobble = 1.0 + w2 * (ux * ux - vy * vy) + w3 * (2.0 * ux * vy);
                    let r_eff = rr / wobble.max(0.65);

                    let th = (1.0 - r_eff * inv_rh).max(0.0);
                    if th <= 0.0 {
                        continue;
                    }
                    let halo = th * th * th;
                    let tc = (1.0 - r_eff * inv_rc).max(0.0);
                    let core = tc * tc;

                    let ar = (0.08 * halo + 0.26 * core) * spot_a;
                    let ag = (0.18 * halo + 0.48 * core) * spot_a;
                    let ab = (0.30 * halo + 0.68 * core) * spot_a;

                    let i = (row_off + xx as usize) * 3;
                    buf[i] *= (1.0 - ar).max(0.25);
                    buf[i + 1] *= (1.0 - ag).max(0.15);
                    buf[i + 2] *= (1.0 - ab).max(0.10);
                }
            }
        }
    }
    (n_clusters, strength)
}

fn apply_grease_corner_patches(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    s: f32,
) -> (u32, f32) {
    if rng.gen::<f32>() > 0.65 {
        return (0, 0.0);
    }
    let count = rng.gen_range(1..=3u32);
    let strength = s * rng.gen_range(0.25..0.75);
    let (wf, hf) = (w as f32, h as f32);
    let min_dim = wf.min(hf);

    for _ in 0..count {
        let roll = rng.gen::<f32>();
        let (cx, cy) = if roll < 0.50 {
            (
                wf * rng.gen_range(0.84..0.97),
                hf * rng.gen_range(0.84..0.97),
            )
        } else if roll < 0.70 {
            (
                wf * rng.gen_range(0.84..0.97),
                hf * rng.gen_range(0.03..0.16),
            )
        } else if roll < 0.85 {
            (
                wf * rng.gen_range(0.03..0.16),
                hf * rng.gen_range(0.84..0.97),
            )
        } else {
            (
                wf * rng.gen_range(0.88..0.98),
                hf * rng.gen_range(0.35..0.65),
            )
        };

        let rx = min_dim * rng.gen_range(0.045..0.105);
        let ry = rx * rng.gen_range(0.55..0.90);
        let ang = rng.gen_range(0.0..std::f32::consts::TAU);
        let (ca, sa) = (ang.cos(), ang.sin());
        let p_str = strength * rng.gen_range(0.6..1.0);

        let rmax = rx.max(ry) * 1.15;
        let x0 = (cx - rmax).floor().max(0.0) as u32;
        let x1 = (cx + rmax).ceil().min(wf - 1.0) as u32;
        let y0 = (cy - rmax).floor().max(0.0) as u32;
        let y1 = (cy + rmax).ceil().min(hf - 1.0) as u32;
        let (inv_rx, inv_ry) = (1.0 / rx, 1.0 / ry);

        for yy in y0..=y1 {
            let dy = yy as f32 - cy;
            let row_off = (yy * w) as usize;
            for xx in x0..=x1 {
                let dx = xx as f32 - cx;
                let u = (dx * ca + dy * sa) * inv_rx;
                let v = (-dx * sa + dy * ca) * inv_ry;
                let d2 = u * u + v * v;
                if d2 >= 1.0 {
                    continue;
                }
                let w_oil = (1.0 - d2) * (1.0 - d2) * p_str;
                let dark = 1.0 - 0.14 * w_oil;
                let warm_b = 1.0 - 0.21 * w_oil;
                let c_mix = 0.11 * w_oil;

                let i = (row_off + xx as usize) * 3;
                let r = buf[i] * dark;
                let g = buf[i + 1] * (dark - 0.02 * w_oil);
                let b = buf[i + 2] * warm_b;
                buf[i] = (r * (1.0 - c_mix) + 162.0 * c_mix).clamp(0.0, 255.0);
                buf[i + 1] = (g * (1.0 - c_mix) + 155.0 * c_mix).clamp(0.0, 255.0);
                buf[i + 2] = (b * (1.0 - c_mix) + 142.0 * c_mix).clamp(0.0, 255.0);
            }
        }
    }
    (count, strength)
}

fn apply_watermark_emblem(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    wm_strength: f32,
    style: &str,
) {
    let (wf, hf) = (w as f32, h as f32);
    let min_dim = wf.min(hf);
    let cx = wf * rng.gen_range(0.28..0.72);
    let cy = hf * rng.gen_range(0.25..0.75);
    let rad = min_dim * rng.gen_range(0.07..0.13);
    let stroke = (rad * 0.045).max(1.5);
    let inv_stroke = 1.0 / stroke;

    let x0 = (cx - rad * 1.15).floor().max(0.0) as u32;
    let x1 = (cx + rad * 1.15).ceil().min(wf - 1.0) as u32;
    let y0 = (cy - rad * 1.15).floor().max(0.0) as u32;
    let y1 = (cy + rad * 1.15).ceil().min(hf - 1.0) as u32;

    for yy in y0..=y1 {
        let dy = yy as f32 - cy;
        let row_off = (yy * w) as usize;
        for xx in x0..=x1 {
            let dx = xx as f32 - cx;
            let dist = match style {
                "crest" => {
                    let e1 = ((dx * 0.85).hypot(dy) - rad * 0.85).abs();
                    let e2 = ((dx * 0.85).hypot(dy) - rad * 0.52).abs();
                    let bar = dy.abs().max(dx.abs() - rad * 0.55);
                    e1.min(e2).min(bar)
                }
                "diamond" => {
                    let d1 =
                        (dx.abs() + dy.abs() - rad * 0.90).abs() * std::f32::consts::FRAC_1_SQRT_2;
                    let d2 =
                        (dx.abs() + dy.abs() - rad * 0.48).abs() * std::f32::consts::FRAC_1_SQRT_2;
                    d1.min(d2)
                }
                _ => {
                    let in_x = (dx.abs() - rad * 0.85).max(0.0);
                    let b0 = (dy - rad * 0.35).abs().hypot(in_x);
                    let b1 = dy.abs().hypot(in_x);
                    let b2 = (dy + rad * 0.35).abs().hypot(in_x);
                    b0.min(b1).min(b2)
                }
            };
            if dist >= stroke * 2.5 {
                continue;
            }
            let wire = (-(dist * inv_stroke).powi(2) * 0.5).exp();
            let i = (row_off + xx as usize) * 3;
            let pw = (buf[i] + buf[i + 1] + buf[i + 2]) * (1.0 / 765.0);
            let lift = 1.0 + wm_strength * wire * pw;
            buf[i] = (buf[i] * lift).clamp(0.0, 255.0);
            buf[i + 1] = (buf[i + 1] * lift).clamp(0.0, 255.0);
            buf[i + 2] = (buf[i + 2] * lift).clamp(0.0, 255.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn synthetic(w: u32, h: u32) -> Vec<f32> {
        let mut v = vec![0.0; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let ink = y % 8 == 0 && x % 3 != 0; // dark text-line bars on paper
                v[i] = if ink { 20.0 } else { 245.0 };
                v[i + 1] = if ink { 20.0 } else { 242.0 };
                v[i + 2] = if ink { 20.0 } else { 235.0 };
            }
        }
        v
    }

    #[test]
    fn determinism_same_seed_identical_bytes() {
        let (w, h) = (32, 24);
        let mut a = synthetic(w, h);
        let mut b = a.clone();
        let p1 = apply(&mut a, w, h, &mut ChaCha8Rng::seed_from_u64(42), 0.7);
        let p2 = apply(&mut b, w, h, &mut ChaCha8Rng::seed_from_u64(42), 0.7);
        assert_eq!(a, b);
        assert_eq!(format!("{p1:?}"), format!("{p2:?}"));
    }

    #[test]
    fn effect_differs_and_params_in_range() {
        let (w, h) = (32, 24);
        let mut buf = synthetic(w, h);
        let before = buf.clone();
        let p = apply(&mut buf, w, h, &mut ChaCha8Rng::seed_from_u64(7), 0.7);
        assert!(p.enabled && buf != before);
        for v in [
            p.yellowing,
            p.mottle,
            p.edge,
            p.foxing_amount,
            p.fiber_amount,
            p.cockle,
        ] {
            assert!((0.0..=1.0).contains(&v), "knob out of range: {v}");
        }
        assert!((0.0..=1.0).contains(&p.water_x) && (0.0..=1.0).contains(&p.water_y));
        assert!((0.15..=0.45).contains(&p.water_radius));
        assert!(buf.iter().all(|v| (0.0..=255.0).contains(v)));
        // Zero strength is a no-op.
        let mut flat = synthetic(w, h);
        let p0 = apply(&mut flat, w, h, &mut ChaCha8Rng::seed_from_u64(7), 0.0);
        assert!(!p0.enabled && flat == before);
    }
}
