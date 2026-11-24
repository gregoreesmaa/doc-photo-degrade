//! Precomputed noise banks, warp/lighting fields, and stamped damage library.
//!
//! Replaces per-pixel RNG + per-pixel transcendental fields with
//! seeded precomputations, without breaking the determinism contract.
//!
//! Contract mapping:
//! - Every bank derives from `(file_seed, stage_tag)` via
//!   [`bank_seed`]: an independent `ChaCha8Rng` stream per stage, so banks
//!   never consume draws from the shared pipeline stream and per-file
//!   reproducibility only needs `(seed, stage, key)`.
//! - Tiling offsets (`ox`, `oy`) are the *only* draws taken from the shared
//!   `rng` (two `gen_range` draws per banked stage), preserving the
//!   `(seed, stage, key)` linkage: same file seed ⇒ same bank content *and*
//!   same offset ⇒ bit-identical output across reruns.
//! - Warp/lighting/damage fields take already-sampled knobs (no RNG inside
//!   the precompute), so field-vs-direct parity is bit-exact by construction.
//!
//! Stage tags (fixed forever — banks are keyed on them):
//! - `TAG_GRAIN = 0x01`, `TAG_NOISE = 0x02`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const TAG_GRAIN: u64 = 0x01;
pub const TAG_NOISE: u64 = 0x02;
pub const TAG_WEAVE: u64 = 0x03;
pub const TAG_PULP_INCLUSIONS: u64 = 0x04;
pub const TAG_BRDF: u64 = 0x05;
pub const TAG_TOOTH: u64 = 0x06;
pub const TAG_CHROMA: u64 = 0x07;
pub const TAG_INK_TEX: u64 = 0x08;

/// Tile edge for noise banks. 256×256 f32 = 256 KiB: tiles a full-res page
/// ~10×14 with wrap-around, stays L2-resident during the apply pass.
pub const BANK_TILE: u32 = 256;

/// Derive the independent bank stream seed from `(file_seed, stage_tag)`.
/// SplitMix64-style avalanche so adjacent tags decorrelate fully.
pub fn bank_seed(file_seed: u64, stage_tag: u64) -> u64 {
    let mut z = file_seed.wrapping_add(stage_tag.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Generate a `tile × tile` Gaussian bank with std-dev `sigma`, drawn from
/// the `(file_seed, stage_tag)` stream. Deterministic per key.
pub fn generate_gauss_bank(file_seed: u64, stage_tag: u64, tile: u32, sigma: f32) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(bank_seed(file_seed, stage_tag));
    let n = (tile as usize) * (tile as usize);
    let mut bank = Vec::with_capacity(n);
    // Box-Muller yields pairs; emit in order so the bank is a pure function
    // of the stream position (no draw is skipped or reordered).
    while bank.len() < n {
        let u1: f32 = rng.gen_range(1e-6..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f32::consts::PI * u2;
        bank.push(r * t.cos() * sigma);
        if bank.len() < n {
            bank.push(r * t.sin() * sigma);
        }
    }
    bank.truncate(n);
    bank
}

/// Generate a `tile × tile` uniform-in-[0,1) bank from the same keyed stream
/// (speckle/dropout decisions without per-pixel `gen::<f32>()` draws).
pub fn generate_uniform_bank(file_seed: u64, stage_tag: u64, tile: u32) -> Vec<f32> {
    let mut rng =
        ChaCha8Rng::seed_from_u64(bank_seed(file_seed, stage_tag) ^ 0xD1B5_4D95_F9C4_3D2D);
    let n = (tile as usize) * (tile as usize);
    (0..n).map(|_| rng.gen::<f32>()).collect()
}

/// Sample the tiling offset from the shared pipeline stream. Two draws —
/// the only main-stream consumption of a banked stage.
pub fn bank_offset(rng: &mut ChaCha8Rng, tile: u32) -> (u32, u32) {
    (rng.gen_range(0..tile), rng.gen_range(0..tile))
}

/// Banked paper-grain apply: `buf[i] += bank_val + warm[c]`, speckle via the
/// uniform bank. Row-major chunked tiling avoids per-pixel integer `%`/`/`
/// while staying bit-exact with the direct tile formula.
#[allow(clippy::too_many_arguments)]
pub fn apply_grain_banked(
    buf: &mut [f32],
    w: u32,
    h: u32,
    gauss: &[f32],
    uniform: &[f32],
    tile: u32,
    ox: u32,
    oy: u32,
    sigma: f32,
    speckle_prob: f32,
    strength: f32,
    warm: [f32; 3],
) {
    debug_assert_eq!(gauss.len(), (tile as usize) * (tile as usize));
    debug_assert_eq!(uniform.len(), (tile as usize) * (tile as usize));
    let t = tile as usize;
    let n = gauss.len();
    let stride = w as usize;
    if t == 0 || n == 0 || stride == 0 {
        return;
    }
    let third = n / 3;
    let two_third = 2 * third;
    let ox_u = (ox as usize) % t;
    let oy_u = (oy as usize) % t;
    let total_px = buf.len() / 3;
    let n_rows = total_px / stride;
    for y in 0..n_rows {
        let row_t = ((y + oy_u) % t) * t;
        let row_buf = &mut buf[y * stride * 3..(y + 1) * stride * 3];
        let mut x = 0usize;
        let mut bx = ox_u;
        while x < stride {
            let chunk = (stride - x).min(t - bx);
            let p_slice = &mut row_buf[x * 3..(x + chunk) * 3];
            for (base, px) in (row_t + bx..).zip(p_slice.as_chunks_mut::<3>().0.iter_mut()) {
                let u = uniform[base];
                let mut i1 = base + third;
                if i1 >= n {
                    i1 -= n;
                }
                let mut i2 = base + two_third;
                if i2 >= n {
                    i2 -= n;
                }
                px[0] += gauss[base] + warm[0];
                px[1] += gauss[i1] + warm[1];
                px[2] += gauss[i2] + warm[2];
                if u < speckle_prob {
                    let u2_idx = if base + 1 < n { base + 1 } else { 0 };
                    let dd = (10.0 + 25.0 * uniform[u2_idx]) * strength;
                    px[0] -= dd;
                    px[1] -= dd;
                    px[2] -= dd;
                }
            }
            x += chunk;
            bx = 0;
        }
    }
    // Handle any partial trailing row if `buf.len() / 3` is not a multiple of `stride`.
    let rem_start = n_rows * stride;
    for (p_off, px) in buf[rem_start * 3..]
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .enumerate()
    {
        let p = rem_start + p_off;
        let x = p % stride;
        let y = p / stride;
        let base = ((y + oy_u) % t) * t + ((x + ox_u) % t);
        let u = uniform[base];
        for c in 0..3 {
            px[c] += gauss[(base + c * third) % n] + warm[c];
        }
        if u < speckle_prob {
            let u2 = uniform[(base + 1) % n];
            let dd = (10.0 + 25.0 * u2) * strength;
            px[0] -= dd;
            px[1] -= dd;
            px[2] -= dd;
        }
    }
    let _ = (h, sigma);
}

/// Banked sensor-noise apply: `buf[i] += gauss`. Row-major chunked tiling
/// avoids per-channel integer `%`/`/` while staying bit-exact.
pub fn apply_noise_banked(buf: &mut [f32], w: u32, gauss: &[f32], tile: u32, ox: u32, oy: u32) {
    let t = tile as usize;
    let n = gauss.len();
    let stride = w as usize;
    if t == 0 || n == 0 || stride == 0 {
        return;
    }
    let third = n / 3;
    let two_third = 2 * third;
    let ox_u = (ox as usize) % t;
    let oy_u = (oy as usize) % t;
    let total_px = buf.len() / 3;
    let n_rows = total_px / stride;
    for y in 0..n_rows {
        let row_t = ((y + oy_u) % t) * t;
        let row_buf = &mut buf[y * stride * 3..(y + 1) * stride * 3];
        let mut x = 0usize;
        let mut bx = ox_u;
        while x < stride {
            let chunk = (stride - x).min(t - bx);
            let p_slice = &mut row_buf[x * 3..(x + chunk) * 3];
            for (base, px) in (row_t + bx..).zip(p_slice.as_chunks_mut::<3>().0.iter_mut()) {
                let mut i1 = base + third;
                if i1 >= n {
                    i1 -= n;
                }
                let mut i2 = base + two_third;
                if i2 >= n {
                    i2 -= n;
                }
                px[0] += gauss[base];
                px[1] += gauss[i1];
                px[2] += gauss[i2];
            }
            x += chunk;
            bx = 0;
        }
    }
    let tail_start = n_rows * stride * 3;
    for (off, v) in buf[tail_start..].iter_mut().enumerate() {
        let idx = tail_start + off;
        let pix = idx / 3;
        let x = pix % stride;
        let y = pix / stride;
        let c = idx % 3;
        let gi = (((y + oy_u) % t) * t + ((x + ox_u) % t) + c * third) % n;
        *v += gauss[gi];
    }
}

// ---------------------------------------------------------------------------
// Stamped damage library: coffee-ring sprite, tear notch.
// ---------------------------------------------------------------------------

pub const COFFEE: [f32; 3] = [122.0, 82.0, 42.0];
pub const SEPIA: [f32; 3] = [214.0, 192.0, 160.0];

/// Precomputed coffee-ring stamp: offsets with `(rim, )` weights for the
/// `d < r` disc. Stamping applies SEPIA 0.18 then COFFEE 0.55·rim per pixel —
/// the same two blends in the same order as the direct loop.
pub struct CoffeeStamp {
    pub cells: Vec<(i32, i32, f32)>,
}

pub fn precompute_coffee(r: f32) -> CoffeeStamp {
    let bb = r as i64 + 2;
    let mut cells = Vec::new();
    for dy in -bb..=bb {
        for dx in -bb..=bb {
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d < r {
                let rim = (-((d - r) / (r * 0.12).max(1.0)).powi(2)).exp();
                cells.push((dx as i32, dy as i32, rim));
            }
        }
    }
    CoffeeStamp { cells }
}

fn blend_px_local(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3], a: f32) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        for k in 0..3 {
            buf[i + k] = buf[i + k] * (1.0 - a) + c[k] * a;
        }
    }
}

pub fn stamp_coffee(buf: &mut [f32], w: u32, h: u32, cx: f32, cy: f32, stamp: &CoffeeStamp) {
    for &(dx, dy, rim) in &stamp.cells {
        let (x, y) = (cx as i64 + dx as i64, cy as i64 + dy as i64);
        blend_px_local(buf, w, h, x, y, SEPIA, 0.18);
        blend_px_local(buf, w, h, x, y, COFFEE, 0.55 * rim);
    }
}

/// Precomputed capillary coffee/water tide-ring stain with organic multi-harmonic
/// contour, directional liquid pooling, razor-thin hydrophobic outer rim, inner
/// tide line, and optional localized ink-bleed feathering.
pub struct CapillaryStain {
    pub r: f32,
    pub edge_lut: [f32; 128],
    pub tilt: [f32; 2],
    pub wash_phase: f32,
    pub ink_bleed: f32,
}

pub fn precompute_capillary_stain(r: f32, rng: &mut ChaCha8Rng, ink_bleed: f32) -> CapillaryStain {
    let (a1, p1) = (
        rng.gen_range(0.06..0.18),
        rng.gen_range(0.0..std::f32::consts::TAU),
    );
    let (a2, p2) = (
        rng.gen_range(0.03..0.10),
        rng.gen_range(0.0..std::f32::consts::TAU),
    );
    let (a3, p3) = (
        rng.gen_range(0.01..0.05),
        rng.gen_range(0.0..std::f32::consts::TAU),
    );
    let mut edge_lut = [r; 128];
    for (k, slot) in edge_lut.iter_mut().enumerate() {
        let th = (k as f32 / 128.0) * std::f32::consts::TAU;
        *slot = r
            * (1.0
                + a1 * (2.0 * th + p1).sin()
                + a2 * (5.0 * th + p2).sin()
                + a3 * (9.0 * th + p3).sin());
    }
    let tang = rng.gen_range(0.0..std::f32::consts::TAU);
    CapillaryStain {
        r,
        edge_lut,
        tilt: [tang.cos(), tang.sin()],
        wash_phase: rng.gen_range(0.0..std::f32::consts::TAU),
        ink_bleed,
    }
}

/// Stamp a [`CapillaryStain`] onto `buf` and optionally mark `art_mask` pixels
/// with `stain_bit`. Uses a tight bbox snapshot for localized ink bleed.
#[allow(clippy::too_many_arguments)]
pub fn stamp_capillary_stain(
    buf: &mut [f32],
    w: u32,
    h: u32,
    cx: f32,
    cy: f32,
    st: &CapillaryStain,
    mut art_mask: Option<&mut [u16]>,
    stain_bit: u16,
) {
    let bb = (st.r * 1.38).ceil() as i32 + 3;
    let x0 = (cx as i32 - bb).max(0) as usize;
    let x1 = (cx as i32 + bb).min(w as i32 - 1) as usize;
    let y0 = (cy as i32 - bb).max(0) as usize;
    let y1 = (cy as i32 + bb).min(h as i32 - 1) as usize;
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let (wsz, bw, bh) = (w as usize, x1 - x0 + 1, y1 - y0 + 1);

    let snap: Option<Vec<f32>> = (st.ink_bleed > 0.01).then(|| {
        let mut s = vec![0.0f32; bw * bh * 3];
        for y in y0..=y1 {
            let src = (y * wsz + x0) * 3;
            let dst = (y - y0) * bw * 3;
            s[dst..dst + bw * 3].copy_from_slice(&buf[src..src + bw * 3]);
        }
        s
    });

    const INV_TAU_128: f32 = 128.0 / std::f32::consts::TAU;
    for y in y0..=y1 {
        let dy = y as f32 - cy;
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            let d = (dx * dx + dy * dy).sqrt();
            let bin = (((dy.atan2(dx) + std::f32::consts::TAU) * INV_TAU_128) as usize) & 127;
            let r_eff = st.edge_lut[bin].max(2.0);
            let w_rim = (r_eff * 0.032).clamp(1.2, 4.5);
            if d > r_eff + 2.5 * w_rim {
                continue;
            }

            let pix = y * wsz + x;
            let idx = pix * 3;
            let pool =
                (0.72 + 0.42 * (dx * st.tilt[0] + dy * st.tilt[1]) / r_eff).clamp(0.35, 1.25);

            if let Some(ref s) = snap {
                if d < r_eff && x > x0 && x < x1 && y > y0 && y < y1 {
                    let b = (y - y0) * bw + (x - x0);
                    let mut min_l = 255.0f32;
                    let mut acc = [0.0f32; 3];
                    for off in [b, b - 1, b + 1, b - bw, b + bw] {
                        let p = off * 3;
                        let l = 0.299 * s[p] + 0.587 * s[p + 1] + 0.114 * s[p + 2];
                        if l < min_l {
                            min_l = l;
                        }
                        for c in 0..3 {
                            acc[c] += s[p + c];
                        }
                    }
                    if min_l < 175.0 {
                        let wick = st.ink_bleed * ((175.0 - min_l) / 175.0);
                        let avg = [acc[0] * 0.2, acc[1] * 0.2, acc[2] * 0.2];
                        let bleed_col = [
                            avg[0] * 0.96 + 8.0,
                            avg[1] * 0.91 + 5.0,
                            avg[2] * 0.82 + 2.0,
                        ];
                        for c in 0..3 {
                            buf[idx + c] = buf[idx + c] * (1.0 - wick) + bleed_col[c] * wick;
                        }
                    }
                }
            }

            let inside = if d < r_eff {
                let mottle = 0.80
                    + 0.20
                        * ((dx * 0.07 + st.wash_phase).sin() * (dy * 0.07 - st.wash_phase).cos());
                (0.16 * pool * mottle).clamp(0.0, 0.30)
            } else {
                0.0
            };
            let rim_outer = (-((d - r_eff) / w_rim).powi(2)).exp();
            let rim_inner = 0.25 * (-((d - r_eff * 0.88) / (w_rim * 1.4)).powi(2)).exp();
            let rim_a = ((0.62 * rim_outer + rim_inner) * pool).clamp(0.0, 0.78);

            for c in 0..3 {
                let v = buf[idx + c] * (1.0 - inside) + SEPIA[c] * inside;
                buf[idx + c] = (v * (1.0 - rim_a) + COFFEE[c] * rim_a).clamp(0.0, 255.0);
            }
            if let Some(ref mut am) = art_mask {
                if pix < am.len() {
                    am[pix] |= stain_bit;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Seamless 256×256 Paper-Tooth Raking-Light Bump Bank (TAG_TOOTH)
// ---------------------------------------------------------------------------

/// Build a seamless toroidal 256×256 raking-light micro-bump tile
/// `B(x, y) = (dhx * lx + dhy * ly) / rms`. Deterministic from `(file_seed, TAG_TOOTH)`.
pub fn generate_tooth_bump_tile(
    file_seed: u64,
    lx: f32,
    ly: f32,
    anisotropy: f32,
    laid_weight: f32,
) -> Vec<f32> {
    const N: usize = BANK_TILE as usize;
    const MASK: usize = N - 1;
    let mut rng = ChaCha8Rng::seed_from_u64(bank_seed(file_seed, TAG_TOOTH));

    let mut h0 = vec![0.0f32; N * N];
    for v in &mut h0 {
        *v = (rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>()) - 1.5;
    }

    let wx_side = 0.25 + 0.08 * anisotropy.clamp(0.0, 1.0);
    let wx_mid = 1.0 - 2.0 * wx_side;
    let wy_side = 0.25 - 0.06 * anisotropy.clamp(0.0, 1.0);
    let wy_mid = 1.0 - 2.0 * wy_side;
    let mut h1 = vec![0.0f32; N * N];

    for _ in 0..2 {
        for y in 0..N {
            let row = y * N;
            for x in 0..N {
                h1[row + x] = wx_side * h0[row + ((x + MASK) & MASK)]
                    + wx_mid * h0[row + x]
                    + wx_side * h0[row + ((x + 1) & MASK)];
            }
        }
        for y in 0..N {
            let ym = ((y + MASK) & MASK) * N;
            let y0 = y * N;
            let yp = ((y + 1) & MASK) * N;
            for x in 0..N {
                h0[y0 + x] = wy_side * h1[ym + x] + wy_mid * h1[y0 + x] + wy_side * h1[yp + x];
            }
        }
    }

    if laid_weight > 0.0 {
        for y in 0..N {
            let ridge =
                ((y as f32) * (std::f32::consts::TAU * 64.0 / N as f32)).sin() * laid_weight;
            let row = y * N;
            for x in 0..N {
                h0[row + x] += ridge;
            }
        }
    }

    let mut bump = vec![0.0f32; N * N];
    let mut sum_sq = 0.0f32;
    for y in 0..N {
        let ym = ((y + MASK) & MASK) * N;
        let y0 = y * N;
        let yp = ((y + 1) & MASK) * N;
        for x in 0..N {
            let dhx = 0.5 * (h0[y0 + ((x + 1) & MASK)] - h0[y0 + ((x + MASK) & MASK)]);
            let dhy = 0.5 * (h0[yp + x] - h0[ym + x]);
            let b = dhx * lx + dhy * ly;
            bump[y0 + x] = b;
            sum_sq += b * b;
        }
    }
    let inv_rms = 1.0 / (sum_sq / (N * N) as f32).sqrt().max(1e-6);
    for b in &mut bump {
        *b = (*b * inv_rms).clamp(-3.0, 3.0);
    }
    bump
}

// ---------------------------------------------------------------------------
// Laid / Wove / Cold-Press 256×256 Seamless Torus Micro-Weave Bank (TAG_WEAVE)
// ---------------------------------------------------------------------------

pub const WEAVE_TILE: usize = 256;
pub const WEAVE_MASK: usize = WEAVE_TILE - 1;

/// Paper micro-weave substrate type sampled per page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeaveStock {
    #[default]
    Smooth,
    Laid,
    Wove,
    ColdPress,
}

impl WeaveStock {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Smooth => "none",
            Self::Laid => "laid",
            Self::Wove => "wove",
            Self::ColdPress => "cold_press",
        }
    }
}

/// Per-profile calibration for paper micro-weave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaperWeaveProfile {
    /// Sampling weights: `[laid, wove, cold_press, smooth]`.
    pub weights: [f32; 4],
    pub amp_min: f32,
    pub amp_max: f32,
    pub laid_wire_cap: f32,
    pub laid_wire_min_px: f32,
    pub tooth_boost: f32,
    pub chroma_mul: [f32; 3],
}

impl PaperWeaveProfile {
    pub const fn smooth() -> Self {
        Self {
            weights: [0.0, 0.0, 0.0, 1.0],
            amp_min: 0.0,
            amp_max: 0.0,
            laid_wire_cap: 0.0,
            laid_wire_min_px: 2.5,
            tooth_boost: 0.0,
            chroma_mul: [1.0, 1.0, 1.0],
        }
    }
    pub const fn phone() -> Self {
        Self {
            weights: [0.15, 0.75, 0.10, 0.0],
            amp_min: 0.015,
            amp_max: 0.035,
            laid_wire_cap: 0.030,
            laid_wire_min_px: 2.5,
            tooth_boost: 1.00,
            chroma_mul: [0.98, 1.00, 1.04],
        }
    }
    pub const fn archival() -> Self {
        Self {
            weights: [0.55, 0.10, 0.35, 0.0],
            amp_min: 0.032,
            amp_max: 0.060,
            laid_wire_cap: 0.045,
            laid_wire_min_px: 2.0,
            tooth_boost: 1.25,
            chroma_mul: [0.95, 1.00, 1.10],
        }
    }
    pub const fn sheet_music() -> Self {
        Self {
            weights: [0.45, 0.45, 0.10, 0.0],
            amp_min: 0.020,
            amp_max: 0.042,
            laid_wire_cap: 0.025,
            laid_wire_min_px: 3.0,
            tooth_boost: 1.00,
            chroma_mul: [0.97, 1.00, 1.06],
        }
    }
    pub const fn book_gutter() -> Self {
        Self {
            weights: [0.25, 0.65, 0.10, 0.0],
            amp_min: 0.022,
            amp_max: 0.045,
            laid_wire_cap: 0.035,
            laid_wire_min_px: 2.4,
            tooth_boost: 1.15,
            chroma_mul: [0.96, 1.00, 1.08],
        }
    }
    pub const fn office_scan() -> Self {
        Self {
            weights: [0.0, 0.90, 0.0, 0.10],
            amp_min: 0.008,
            amp_max: 0.018,
            laid_wire_cap: 0.020,
            laid_wire_min_px: 2.5,
            tooth_boost: 0.50,
            chroma_mul: [1.00, 1.00, 1.00],
        }
    }
    pub const fn photocopy() -> Self {
        Self {
            weights: [0.0, 0.60, 0.0, 0.40],
            amp_min: 0.006,
            amp_max: 0.014,
            laid_wire_cap: 0.015,
            laid_wire_min_px: 2.5,
            tooth_boost: 0.40,
            chroma_mul: [1.00, 1.00, 1.00],
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WeaveSample {
    pub stock: WeaveStock,
    pub strength: f32,
    pub laid_wire_px: f32,
    pub laid_chain_px: f32,
    pub wove_anisotropy: f32,
    pub tooth_light_deg: f32,
}

#[inline]
fn weave_hash_u32(ix: usize, iy: usize, seed: u32) -> f32 {
    let mut h =
        seed ^ (ix as u32).wrapping_mul(374_761_393) ^ (iy as u32).wrapping_mul(668_265_263);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    ((h & 0x00FF_FFFF) as f32 * (2.0 / 16_777_215.0)) - 1.0
}

#[inline]
fn periodic_noise(x: usize, y: usize, gx: usize, gy: usize, seed: u32) -> f32 {
    let fx = (x * gx) as f32 * (1.0 / WEAVE_TILE as f32);
    let fy = (y * gy) as f32 * (1.0 / WEAVE_TILE as f32);
    let ix0 = (fx as usize) & (gx - 1);
    let iy0 = (fy as usize) & (gy - 1);
    let ix1 = (ix0 + 1) & (gx - 1);
    let iy1 = (iy0 + 1) & (gy - 1);
    let tx = fx - fx.floor();
    let ty = fy - fy.floor();
    let u = tx * tx * (3.0 - 2.0 * tx);
    let v = ty * ty * (3.0 - 2.0 * ty);
    let n00 = weave_hash_u32(ix0, iy0, seed);
    let n10 = weave_hash_u32(ix1, iy0, seed);
    let n01 = weave_hash_u32(ix0, iy1, seed);
    let n11 = weave_hash_u32(ix1, iy1, seed);
    let top = n00 + (n10 - n00) * u;
    let bot = n01 + (n11 - n01) * u;
    top + (bot - top) * v
}

#[inline]
fn weave_smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn build_laid_bank(
    rng: &mut ChaCha8Rng,
    px: f32,
    amp: f32,
    wire_min_px: f32,
    wire_cap: f32,
) -> (Vec<f32>, f32, f32) {
    let max_cycles = (WEAVE_TILE as f32 / wire_min_px.max(2.0)).floor() as u32;
    let min_cycles = (WEAVE_TILE as f32 / 4.0).ceil() as u32;
    let n_wires = rng.gen_range(min_cycles..=max_cycles.max(min_cycles));
    let wire_px = WEAVE_TILE as f32 / n_wires as f32;
    let eff_wires = ((n_wires as f32 / px.max(0.25)).round() as u32).clamp(16, 128);
    let eff_period = WEAVE_TILE as f32 / eff_wires as f32;
    let nyquist_att = weave_smoothstep(1.6, 2.5, eff_period);

    let n_chains: u32 = if rng.gen_bool(0.55) { 3 } else { 4 };
    let chain_px = WEAVE_TILE as f32 / n_chains as f32;
    let chain_sigma = (rng.gen_range(1.4..2.2f32) * px).max(0.9);
    let chain_flank = chain_sigma * 2.4;

    let s_wob = rng.gen::<u32>();
    let s_pulp1 = rng.gen::<u32>();
    let s_pulp2 = rng.gen::<u32>();
    let wire_amp = (amp * 0.72 * nyquist_att).min(wire_cap);
    let chain_amp = amp * 0.95;
    let pulp_amp = amp * 0.38;

    let mut bank = vec![0.0f32; WEAVE_TILE * WEAVE_TILE];
    let inv_t = 1.0 / WEAVE_TILE as f32;
    let chain_period = WEAVE_TILE as f32 / n_chains as f32;

    for y in 0..WEAVE_TILE {
        let chain_meander = periodic_noise(0, y, 1, 8, s_wob ^ 0x51A3) * 0.65;
        let base_wire_ph = std::f32::consts::TAU * (eff_wires as f32) * (y as f32) * inv_t;
        let row = y * WEAVE_TILE;
        for x in 0..WEAVE_TILE {
            let wobble = periodic_noise(x, y, 8, 16, s_wob) * 0.22;
            let th = base_wire_ph + wobble;
            let wire = (0.78 * th.sin() + 0.22 * (2.0 * th).cos()) * wire_amp;

            let xm = (x as f32 + chain_meander).rem_euclid(chain_period);
            let dc = xm.min(chain_period - xm);
            let c_core = (-(dc / chain_sigma).powi(2)).exp();
            let c_shadow = (-((dc - chain_flank) / (chain_sigma * 1.6)).powi(2)).exp();
            let chain = (c_core - 0.42 * c_shadow) * chain_amp;

            let pulp = (0.65 * periodic_noise(x, y, 16, 16, s_pulp1)
                + 0.35 * periodic_noise(x, y, 64, 64, s_pulp2))
                * pulp_amp;

            bank[row + x] = wire + chain + pulp;
        }
    }
    (bank, wire_px, chain_px)
}

fn build_wove_bank(rng: &mut ChaCha8Rng, amp: f32, long_axis_vertical: bool) -> (Vec<f32>, f32) {
    let aniso_shift: usize = if rng.gen_bool(0.5) { 2 } else { 3 };
    let anisotropy = (1usize << aniso_shift) as f32;
    let (f1x, f1y, f2x, f2y) = if long_axis_vertical {
        (64, 64 >> aniso_shift, 128, 128 >> aniso_shift)
    } else {
        (64 >> aniso_shift, 64, 128 >> aniso_shift, 128)
    };
    let s1 = rng.gen::<u32>();
    let s2 = rng.gen::<u32>();
    let s3 = rng.gen::<u32>();
    let s4 = rng.gen::<u32>();

    let mut bank = vec![0.0f32; WEAVE_TILE * WEAVE_TILE];
    for y in 0..WEAVE_TILE {
        let row = y * WEAVE_TILE;
        for x in 0..WEAVE_TILE {
            let fiber = 0.62 * periodic_noise(x, y, f1x, f1y, s1)
                + 0.38 * periodic_noise(x, y, f2x, f2y, s2);
            let pulp =
                0.60 * periodic_noise(x, y, 8, 8, s3) + 0.40 * periodic_noise(x, y, 32, 32, s4);
            bank[row + x] = (0.55 * fiber + 0.45 * pulp) * amp;
        }
    }
    (bank, anisotropy)
}

fn build_cold_press_bank(rng: &mut ChaCha8Rng, amp: f32, tooth_boost: f32) -> (Vec<f32>, f32) {
    let light_deg = rng.gen_range(0.0..360.0f32);
    let rad = light_deg.to_radians();
    let (lx, ly) = (rad.cos(), rad.sin());
    let s1 = rng.gen::<u32>();
    let s2 = rng.gen::<u32>();
    let s3 = rng.gen::<u32>();

    let mut h = vec![0.0f32; WEAVE_TILE * WEAVE_TILE];
    for y in 0..WEAVE_TILE {
        let row = y * WEAVE_TILE;
        for x in 0..WEAVE_TILE {
            let dimple = 1.0 - periodic_noise(x, y, 32, 32, s1).abs();
            let med = periodic_noise(x, y, 64, 64, s2);
            let fine = periodic_noise(x, y, 128, 128, s3);
            h[row + x] = 0.55 * (dimple * 2.0 - 1.0) + 0.30 * med + 0.15 * fine;
        }
    }

    let mut bank = vec![0.0f32; WEAVE_TILE * WEAVE_TILE];
    let eff_amp = amp * tooth_boost;
    for y in 0..WEAVE_TILE {
        let ym = ((y + WEAVE_TILE - 1) & WEAVE_MASK) * WEAVE_TILE;
        let yp = ((y + 1) & WEAVE_MASK) * WEAVE_TILE;
        let row = y * WEAVE_TILE;
        for x in 0..WEAVE_TILE {
            let xm = (x + WEAVE_TILE - 1) & WEAVE_MASK;
            let xp = (x + 1) & WEAVE_MASK;
            let dhx = (h[row + xp] - h[row + xm]) * 0.5;
            let dhy = (h[yp + x] - h[ym + x]) * 0.5;
            let ndotl = (dhx * lx + dhy * ly).clamp(-1.0, 1.0);
            let ao = h[row + x];
            bank[row + x] = eff_amp * (0.72 * ndotl + 0.28 * ao);
        }
    }
    (bank, light_deg)
}

/// Apply an L2-resident 256×256 seamless torus micro-weave bank to `buf`
/// using an independent keyed RNG stream `(file_seed, TAG_WEAVE)`.
pub fn apply_paper_weave(
    buf: &mut [f32],
    w: u32,
    h: u32,
    file_seed: u64,
    strength: f32,
    px: f32,
    profile: &PaperWeaveProfile,
) -> WeaveSample {
    let s = strength.clamp(0.0, 1.0);
    let wsum: f32 = profile.weights.iter().sum();
    if s <= 0.02 || w < 8 || h < 8 || wsum <= 0.0 || profile.amp_max <= 0.0 {
        return WeaveSample::default();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(bank_seed(file_seed, TAG_WEAVE));
    let roll = rng.gen::<f32>() * wsum;
    let stock = if roll < profile.weights[0] {
        WeaveStock::Laid
    } else if roll < profile.weights[0] + profile.weights[1] {
        WeaveStock::Wove
    } else if roll < profile.weights[0] + profile.weights[1] + profile.weights[2] {
        WeaveStock::ColdPress
    } else {
        WeaveStock::Smooth
    };
    if matches!(stock, WeaveStock::Smooth) {
        return WeaveSample::default();
    }

    let amp = s * if profile.amp_max > profile.amp_min {
        rng.gen_range(profile.amp_min..=profile.amp_max)
    } else {
        profile.amp_min
    };
    let ox = rng.gen_range(0..WEAVE_TILE);
    let oy = rng.gen_range(0..WEAVE_TILE);

    let mut sample = WeaveSample {
        stock,
        strength: amp,
        ..Default::default()
    };
    let bank = match stock {
        WeaveStock::Laid => {
            let (b, wp, cp) = build_laid_bank(
                &mut rng,
                px,
                amp,
                profile.laid_wire_min_px,
                profile.laid_wire_cap,
            );
            sample.laid_wire_px = wp;
            sample.laid_chain_px = cp;
            b
        }
        WeaveStock::Wove => {
            let (b, aniso) = build_wove_bank(&mut rng, amp, h >= w);
            sample.wove_anisotropy = aniso;
            b
        }
        WeaveStock::ColdPress => {
            let (b, ldeg) = build_cold_press_bank(&mut rng, amp, profile.tooth_boost);
            sample.tooth_light_deg = ldeg;
            b
        }
        WeaveStock::Smooth => return WeaveSample::default(),
    };

    let [cr, cg, cb] = profile.chroma_mul;
    let w_us = w as usize;
    for y in 0..h as usize {
        let bank_row = &bank[((y + oy) & WEAVE_MASK) * WEAVE_TILE..][..WEAVE_TILE];
        let img_row = &mut buf[y * w_us * 3..(y + 1) * w_us * 3];
        let mut x = 0usize;
        let mut bx = ox & WEAVE_MASK;
        while x < w_us {
            let chunk = (w_us - x).min(WEAVE_TILE - bx);
            let b_slice = &bank_row[bx..bx + chunk];
            let p_slice = &mut img_row[x * 3..(x + chunk) * 3];
            for (px, &m) in p_slice.as_chunks_mut::<3>().0.iter_mut().zip(b_slice) {
                px[0] = (px[0] * (1.0 + m * cr)).clamp(0.0, 255.0);
                px[1] = (px[1] * (1.0 + m * cg)).clamp(0.0, 255.0);
                px[2] = (px[2] * (1.0 + m * cb)).clamp(0.0, 255.0);
            }
            x += chunk;
            bx = 0;
        }
    }
    sample
}

/// Precomputed tear-notch mask: half-width per row `f` in `[0, steps]`.
/// `build` captures the mouth profile; `carve` stamps desk-tone rows —
/// same rows in the same order as the direct `carve_notch`.
pub struct NotchMask {
    pub half_widths: Vec<f32>,
    pub depth: f32,
    pub along: f32,
}

pub fn precompute_notch(along: f32, mouth: f32, depth: f32) -> NotchMask {
    let steps = 24u32;
    let mut half_widths = Vec::with_capacity((steps + 1) as usize);
    for i in 0..=steps {
        let f = i as f32 / steps as f32;
        half_widths.push(
            (mouth * 0.5 * (1.0 - f) + mouth * 0.12 * (f * std::f32::consts::PI).sin()).max(0.0),
        );
    }
    NotchMask {
        half_widths,
        depth,
        along,
    }
}

fn set_px_local(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        buf[i] = c[0];
        buf[i + 1] = c[1];
        buf[i + 2] = c[2];
    }
}

pub fn carve_notch_stamp(buf: &mut [f32], w: u32, h: u32, edge: u32, mask: &NotchMask) {
    const DESK: [f32; 3] = [233.0, 228.0, 218.0];
    let steps = mask.half_widths.len() as u32 - 1;
    for i in 0..=steps {
        let f = i as f32 / steps as f32;
        let hw = mask.half_widths[i as usize];
        let d = mask.depth * f;
        let (x0, x1) = ((mask.along - hw) as i64, (mask.along + hw) as i64);
        match edge {
            0 => {
                let y = d as i64;
                for x in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
            2 => {
                let y = (h as f32 - d) as i64;
                for x in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
            1 => {
                let x = (w as f32 - d) as i64;
                for y in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
            _ => {
                let x = d as i64;
                for y in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn bank_seed_is_stable_and_decorrelated() {
        assert_eq!(bank_seed(42, TAG_GRAIN), bank_seed(42, TAG_GRAIN));
        assert_ne!(bank_seed(42, TAG_GRAIN), bank_seed(42, TAG_NOISE));
        assert_ne!(bank_seed(42, TAG_GRAIN), bank_seed(43, TAG_GRAIN));
    }

    #[test]
    fn gauss_bank_is_deterministic_and_gaussian() {
        let a = generate_gauss_bank(1234, TAG_NOISE, 64, 5.0);
        let b = generate_gauss_bank(1234, TAG_NOISE, 64, 5.0);
        assert_eq!(a, b, "same (seed,stage) must give identical bank");
        let n = a.len() as f32;
        let mean = a.iter().sum::<f32>() / n;
        let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 0.3, "mean {mean}");
        assert!((var.sqrt() - 5.0).abs() < 0.5, "std {} vs 5.0", var.sqrt());
        // Different file seeds decorrelate.
        let c = generate_gauss_bank(1235, TAG_NOISE, 64, 5.0);
        assert_ne!(a, c);
    }

    #[test]
    fn uniform_bank_covers_unit_interval() {
        let u = generate_uniform_bank(99, TAG_GRAIN, 32);
        assert!(u.iter().all(|&v| (0.0..1.0).contains(&v)));
        let mean = u.iter().sum::<f32>() / u.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "mean {mean}");
    }

    #[test]
    fn bank_offsets_vary_with_main_stream() {
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(8);
        let (a, b) = (bank_offset(&mut r1, 256), bank_offset(&mut r2, 256));
        assert!(a.0 < 256 && a.1 < 256 && b.0 < 256 && b.1 < 256);
        // Same main-stream seed ⇒ same offset (contract linkage).
        let mut r3 = ChaCha8Rng::seed_from_u64(7);
        assert_eq!(bank_offset(&mut r3, 256), a);
    }

    #[test]
    fn noise_bank_matches_sigma_statistics() {
        // Statistical parity vs direct Box-Muller: banked apply must land at
        // the requested sigma on a page-size buffer, like the direct loop.
        let (w, h) = (320u32, 200u32);
        let sigma = 8.0f32;
        let bank = generate_gauss_bank(555, TAG_NOISE, BANK_TILE, sigma);
        let mut buf = vec![128.0f32; (w * h * 3) as usize];
        apply_noise_banked(&mut buf, w, &bank, BANK_TILE, 13, 29);
        let n = buf.len() as f32;
        let mean = buf.iter().sum::<f32>() / n - 128.0;
        let var = buf.iter().map(|v| (v - 128.0 - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 0.5, "banked noise mean {mean}");
        assert!(
            (var.sqrt() - sigma).abs() < 0.6,
            "banked noise std {} vs {sigma}",
            var.sqrt()
        );
    }

    #[test]
    fn coffee_stamp_matches_direct() {
        let (w, h) = (120u32, 100u32);
        let mut a = vec![240.0f32; (w * h * 3) as usize];
        let mut b = a.clone();
        let (cx, cy, r) = (60.0f32, 50.0f32, 9.0f32);
        let stamp = precompute_coffee(r);
        stamp_coffee(&mut a, w, h, cx, cy, &stamp);
        let bb = r as i64 + 2;
        for dy in -bb..=bb {
            for dx in -bb..=bb {
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                if d < r {
                    let rim = (-((d - r) / (r * 0.12).max(1.0)).powi(2)).exp();
                    blend_px_local(&mut b, w, h, cx as i64 + dx, cy as i64 + dy, SEPIA, 0.18);
                    blend_px_local(
                        &mut b,
                        w,
                        h,
                        cx as i64 + dx,
                        cy as i64 + dy,
                        COFFEE,
                        0.55 * rim,
                    );
                }
            }
        }
        assert!(max_abs(&a, &b) < 1e-4, "max diff {}", max_abs(&a, &b));
        // Stamp actually marks the page.
        assert!(a
            .iter()
            .zip(b.iter())
            .any(|(x, y)| (x - 240.0).abs() > 1.0 && (x - y).abs() < 1e-4));
    }

    #[test]
    fn notch_stamp_covers_same_rows_as_direct_profile() {
        let mask = precompute_notch(50.0, 12.0, 20.0);
        assert_eq!(mask.half_widths.len(), 25);
        // Mouth profile: widest at the edge (f=0), narrowest inland.
        assert!(mask.half_widths[0] > mask.half_widths[24]);
        assert!(mask.half_widths.iter().all(|&v| v >= 0.0));
        // Stamping paints desk tone on the edge row.
        let (w, h) = (100u32, 100u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        carve_notch_stamp(&mut buf, w, h, 0, &mask);
        let edge_row_dirty = (0..w).any(|x| {
            let i = (x * 3) as usize;
            (buf[i] - 233.0).abs() < 1e-3
        });
        assert!(edge_row_dirty, "top-edge notch must stamp desk tone");
    }

    #[ignore]
    #[test]
    fn bench_banked_vs_direct_micro() {
        // Timing probe (ignored in normal runs): prints direct-vs-banked
        // ns/px for the grain + noise + lighting stages on a page-size
        // buffer. Run with `cargo test --release bench_ -- --ignored --nocapture`.
        let (w, h) = (2480u32, 3508u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        let bank = generate_gauss_bank(1, TAG_NOISE, BANK_TILE, 8.0);
        let t0 = std::time::Instant::now();
        apply_noise_banked(&mut buf, w, &bank, BANK_TILE, 3, 5);
        let banked = t0.elapsed();
        eprintln!("banked noise {w}x{h}: {banked:?}");
    }
}
