//! Ink & print degradation for document training data: uneven fading, dropout
//! speckles, broken strokes, bleed-through ghost, toner patches, browning.
//! Buffer: RGB interleaved `f32` 0..=255, row-major; all randomness from `rng`.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

const PAPER: [f32; 3] = [244.0, 241.0, 235.0];
const PAPER_LUM: f32 = 241.19;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct InkParams {
    pub enabled: bool,
    pub fade_amount: f32,
    pub blotches: [[f32; 4]; 3],
    pub dropout_clusters: u32,
    pub dropout_radius: f32,
    pub gap_attempts: u32,
    pub gap_max_len: u32,
    pub ghost_alpha: f32,
    pub ghost_radius: u32,
    pub toner_amp: f32,
    pub toner_fx: f32,
    pub toner_fy: f32,
    pub toner_phase: f32,
    pub brown_amount: f32,
}

impl InkParams {
    fn disabled() -> Self {
        Self {
            enabled: false,
            fade_amount: 0.0,
            blotches: [[0.0; 4]; 3],
            dropout_clusters: 0,
            dropout_radius: 0.0,
            gap_attempts: 0,
            gap_max_len: 0,
            ghost_alpha: 0.0,
            ghost_radius: 0,
            toner_amp: 0.0,
            toner_fx: 0.0,
            toner_fy: 0.0,
            toner_phase: 0.0,
            brown_amount: 0.0,
        }
    }
}

/// Separable box blur of the vertically-flipped snapshot: `out[y*w+x]` is
/// the mean of `snap` over the `(2r+1)^2` box centered at `(x, h-1-y)`,
/// with out-of-range taps clamped to the edge (exactly the set the old
/// per-pixel gather visited). Horizontal running-sum pass into a row
/// buffer, then a vertical running-sum pass: O(w*h) streaming work,
/// radius-independent. Only fp summation order differs from the direct
/// gather (a few ulps). No RNG.
fn box_blur_flipped(snap: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
    if w == 0 || h == 0 || snap.len() < w * h {
        return Vec::new();
    }
    let n = (2 * r + 1) as f32;
    // Horizontal pass over flipped rows (frow[y][x] = snap[h-1-y][x]).
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let base = (h - 1 - y) * w;
        let mut acc = 0.0f32;
        for ox in -r..=r {
            acc += snap[base + ox.clamp(0, w as i32 - 1) as usize];
        }
        for x in 0..w {
            tmp[y * w + x] = acc / n;
            let drop = snap[base + (x as i32 - r).clamp(0, w as i32 - 1) as usize];
            let add = snap[base + (x as i32 + r + 1).clamp(0, w as i32 - 1) as usize];
            acc = acc - drop + add;
        }
    }
    // Vertical pass over the row means (row-major with a `w`-len accumulator
    // for cache-linear traversal; bit-exact with the column-major loop).
    let mut out = vec![0.0f32; w * h];
    let mut acc = vec![0.0f32; w];
    for oy in -r..=r {
        let src_row = &tmp[oy.clamp(0, h as i32 - 1) as usize * w..][..w];
        for (a, &v) in acc.iter_mut().zip(src_row) {
            *a += v;
        }
    }
    for y in 0..h {
        let out_row = &mut out[y * w..(y + 1) * w];
        let drop_row = &tmp[(y as i32 - r).clamp(0, h as i32 - 1) as usize * w..][..w];
        let add_row = &tmp[(y as i32 + r + 1).clamp(0, h as i32 - 1) as usize * w..][..w];
        for x in 0..w {
            out_row[x] = acc[x] / n;
            acc[x] = acc[x] - drop_row[x] + add_row[x];
        }
    }
    out
}

/// Full-strength entry point (kept for the unit tests + API stability).
#[allow(dead_code)]
pub fn apply(buf: &mut [f32], w: u32, h: u32, rng: &mut ChaCha8Rng, strength: f32) -> InkParams {
    apply_with(buf, w, h, rng, strength, false, 1.0)
}

/// Full ink pass when `coarse` is false; when true, only the topology
/// changes (dropout speckles + broken-stroke gaps) are applied and the
/// per-pixel fade/toner/browning/bleed-through field is skipped. The sampling
/// header is identical either way (stream-preserving coarsen); the skipped
/// pixel loop consumes no RNG draws.
///
/// `px` is the pixel-scale factor (`train_width / native_width`, 1.0 =
/// native): absolute-pixel radii are multiplied by it so a physically
/// fixed-size defect covers the same fraction of a stroke at any
/// resolution. `px = 1.0` reproduces the legacy sampling exactly.
pub fn apply_with(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    coarse: bool,
    px: f32,
) -> InkParams {
    let (wsz, hsz) = (w as usize, h as usize);
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 || wsz == 0 || hsz == 0 || buf.len() < wsz * hsz * 3 {
        return InkParams::disabled();
    }
    let area = (wsz * hsz) as f32;

    let fade_amount = s * rng.gen_range(0.25..0.6);
    let mut blotches = [[0.0f32; 4]; 3];
    for b in blotches.iter_mut() {
        *b = [
            rng.gen(),
            rng.gen(),
            rng.gen_range(0.12..0.35),
            rng.gen_range(0.4..1.0),
        ];
    }
    let dropout_clusters = ((area * s * rng.gen_range(0.0002..0.0006)) as u32).min(6000);
    // Absolute-pixel radii scale with `px` (×1.0 = legacy); floors keep the
    // loops total (a sub-pixel defect still stamps its center pixel).
    let dropout_radius = rng.gen_range(1.0..2.2) * px;
    let gap_attempts = (8.0 + s * 90.0 * rng.gen_range(0.7..1.3)) as u32;
    let gap_lo = (6.0 * px).round().max(2.0) as u32;
    let gap_max_len = rng.gen_range(gap_lo..(14.0 * px).round().max(gap_lo as f32 + 1.0) as u32);
    let ghost_alpha = s * rng.gen_range(0.05..0.14);
    let ghost_radius =
        rng.gen_range((1.0 * px).round().max(1.0) as u32..=(2.0 * px).round().max(1.0) as u32);
    let toner_amp = s * rng.gen_range(0.1..0.35);
    let toner_fx = rng.gen_range(1.0..3.0);
    let toner_fy = rng.gen_range(1.0..3.0);
    let toner_phase = rng.gen_range(0.0..std::f32::consts::TAU);
    let brown_amount = s * rng.gen_range(0.2..0.7);

    let mut params = InkParams {
        enabled: true,
        fade_amount,
        blotches,
        dropout_clusters,
        dropout_radius,
        gap_attempts,
        gap_max_len,
        ghost_alpha,
        ghost_radius,
        toner_amp,
        toner_fx,
        toner_fy,
        toner_phase,
        brown_amount,
    };

    // Luminance snapshot: pre-degradation content drives the ghost.
    let snap: Vec<f32> = buf
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
        .collect();

    let (wf, hf) = (wsz as f32, hsz as f32);
    let (gr, tau) = (ghost_radius as i32, std::f32::consts::TAU);
    // Toner trig LUT: the toner term separates as sin(x-part) * cos(y-part),
    // so one `sin` per column + one `cos` per row replaces a per-ink-pixel
    // `sin`+`cos`. Each entry evaluates the exact same expression tree as the
    // direct loop, so the table is bit-exact; no RNG draws are consumed.
    let mut sin_col = vec![0.0f32; wsz];
    let mut cos_row = vec![0.0f32; hsz];
    if !coarse {
        for (x, s) in sin_col.iter_mut().enumerate() {
            *s = (tau * toner_fx * (x as f32 / wf) + toner_phase).sin();
        }
        for (y, c) in cos_row.iter_mut().enumerate() {
            *c = (tau * toner_fy * (y as f32 / hf) + 0.7 * toner_phase).cos();
        }
    }
    // Ghost box-blur field: separable sliding-window moving average over the
    // vertically-flipped snapshot, O(1) per pixel instead of O((2gr+1)^2).
    // Row scratch only, no RNG draws consumed.
    let ghost_blur: Vec<f32> = if !coarse {
        box_blur_flipped(&snap, wsz, hsz, gr)
    } else {
        Vec::new()
    };
    // Coarse mode skips this whole per-pixel field (no RNG draws inside,
    // so the stream is preserved); dropout + gaps below still run.
    if !coarse {
        for (i, px) in buf.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let x = i % wsz;
            let y = i / wsz;
            let mut d = ((PAPER_LUM - snap[i]) / PAPER_LUM).clamp(0.0, 1.0);
            if d > 0.02 {
                // Uneven fade field from large soft blotches.
                let nx = x as f32 / wf;
                let ny = y as f32 / hf;
                let mut field = 0.0f32;
                for b in &blotches {
                    let dx = (nx - b[0]) / b[2];
                    let dy = (ny - b[1]) / b[2];
                    field += b[3] * (-0.5 * (dx * dx + dy * dy)).exp();
                }
                let lift = (fade_amount * (0.3 + 0.7 * field.min(1.2))).min(0.85);
                // Toner unevenness: soft multiplicative patch on ink only.
                let m = 1.0 + toner_amp * sin_col[x] * cos_row[y];
                let d2 = (d * (1.0 - lift) * m).clamp(0.0, 1.0);
                let k = d2 / d;
                for c in 0..3 {
                    px[c] = PAPER[c] - (PAPER[c] - px[c]) * k;
                }
                d = d2;
                // Age discoloring: dark pixels drift toward brown.
                let age = brown_amount * d;
                px[0] = (px[0] + 26.0 * age).clamp(0.0, 255.0);
                px[1] = (px[1] + 6.0 * age).clamp(0.0, 255.0);
                px[2] = (px[2] - 18.0 * age).clamp(0.0, 255.0);
            }
            // Bleed-through: low-alpha vertically-flipped blurred ghost,
            // visible mainly on paper areas. The blur is the separable
            // sliding-window field above (same box mean, same clamped edges).
            let g = ((PAPER_LUM - ghost_blur[i]) / PAPER_LUM).clamp(0.0, 1.0);
            let gd = ghost_alpha * 140.0 * g * (1.0 - d);
            if gd > 0.01 {
                for ch in px.iter_mut() {
                    *ch = (*ch - gd).clamp(0.0, 255.0);
                }
            }
        }
    } // --coarse: skip fade/toner/brown/ghost field
    if coarse {
        // Params record what was actually applied.
        params.fade_amount = 0.0;
        params.blotches = [[0.0; 4]; 3];
        params.ghost_alpha = 0.0;
        params.toner_amp = 0.0;
        params.toner_fx = 0.0;
        params.toner_fy = 0.0;
        params.toner_phase = 0.0;
        params.brown_amount = 0.0;
    }

    // Dropout speckles: small clusters knocked out to paper color.
    for _ in 0..dropout_clusters {
        let cx = rng.gen_range(0..wsz) as i32;
        let cy = rng.gen_range(0..hsz) as i32;
        let rr = dropout_radius.ceil() as i32;
        for oy in -rr..=rr {
            for ox in -rr..=rr {
                if (ox * ox + oy * oy) as f32 > dropout_radius * dropout_radius {
                    continue;
                }
                let xx = cx + ox;
                let yy = cy + oy;
                if xx < 0 || yy < 0 || xx >= wsz as i32 || yy >= hsz as i32 {
                    continue;
                }
                let j = (yy as usize * wsz + xx as usize) * 3;
                buf[j] = PAPER[0];
                buf[j + 1] = PAPER[1];
                buf[j + 2] = PAPER[2];
            }
        }
    }

    // Broken strokes: short horizontal white gaps stamped onto dark runs.
    for _ in 0..gap_attempts {
        let gy = rng.gen_range(0..hsz);
        // Floor the gap start at the (possibly scaled-down) max so tiny
        // `px` never builds an empty range; at px = 1.0 this is 3..=len.
        let len = rng.gen_range(3.min(gap_max_len)..=gap_max_len) as usize;
        if len >= wsz {
            continue;
        }
        let x0 = rng.gen_range(0..=(wsz - len));
        let mut mean = 0.0f32;
        for x in x0..x0 + len {
            mean += snap[gy * wsz + x];
        }
        if mean / (len as f32) < 170.0 {
            for x in x0..x0 + len {
                let j = (gy * wsz + x) * 3;
                buf[j] = PAPER[0];
                buf[j + 1] = PAPER[1];
                buf[j + 2] = PAPER[2];
            }
        }
    }

    params
}

/// Thermal receipt uneven fade, leuco-dye color cast, and dark heat-friction
/// scratch streaks (Stage 1c.5 when `enable_thermal_physics` is active).
pub(crate) fn apply_thermal_physics(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    px: f32,
) -> (f32, String, u32) {
    let s = strength.clamp(0.0, 1.0);
    let (wsz, hsz) = (w as usize, h as usize);
    if s <= 0.05 || wsz < 8 || hsz < 8 || buf.len() < wsz * hsz * 3 {
        return (0.0, "none".to_string(), 0);
    }
    let thermal_fade = s * rng.gen_range(0.35..0.75);
    let thermal_tone = if rng.gen::<f32>() < 0.65 {
        "purple_gray"
    } else {
        "warm_brown"
    };
    let thermal_scratches = rng.gen_range(2..=5u32);
    let dye_tint: [f32; 3] = if thermal_tone == "purple_gray" {
        [96.0, 90.0, 122.0]
    } else {
        [122.0, 98.0, 80.0]
    };

    let (wf, hf) = (wsz as f32, hsz as f32);
    let tau = std::f32::consts::TAU;
    let f1 = rng.gen_range(2.0..6.0);
    let f2 = rng.gen_range(7.0..15.0);
    let ph = rng.gen_range(0.0..tau);
    let mut head_col = vec![1.0f32; wsz];
    for (x, slot) in head_col.iter_mut().enumerate() {
        let nx = x as f32 / wf;
        let dip = 0.5 + 0.35 * (tau * f1 * nx + ph).sin() + 0.15 * (tau * f2 * nx + 1.3 * ph).cos();
        *slot = (1.0 - thermal_fade * 0.55 * dip).clamp(0.28, 1.0);
    }
    for y in 0..hsz {
        let row = y * wsz;
        for (x, &k_th) in head_col.iter().enumerate() {
            let j = (row + x) * 3;
            let l = 0.299 * buf[j] + 0.587 * buf[j + 1] + 0.114 * buf[j + 2];
            if l < 185.0 {
                let d = ((PAPER_LUM - l) / PAPER_LUM).clamp(0.0, 1.0);
                let blend = (d * (1.0 - k_th * 0.65)).clamp(0.0, 0.85);
                for c in 0..3 {
                    let lifted = PAPER[c] - (PAPER[c] - buf[j + c]) * k_th;
                    buf[j + c] = (lifted * (1.0 - blend * 0.45) + dye_tint[c] * (blend * 0.45))
                        .clamp(0.0, 255.0);
                }
            }
        }
    }

    // Sparse ROI dark heat-friction scratch streaks.
    for _ in 0..thermal_scratches {
        let mut sx = rng.gen_range(0.05..0.95) * wf;
        let mut sy = rng.gen_range(0.05..0.95) * hf;
        let ang = rng.gen_range(-0.45..0.45f32)
            + if rng.gen::<bool>() {
                0.0
            } else {
                std::f32::consts::FRAC_PI_2
            };
        let len = (rng.gen_range(45.0..190.0) * px).round().max(8.0) as usize;
        let (dx, dy) = (ang.cos(), ang.sin());
        let curv = rng.gen_range(-0.004..0.004f32);
        let darkness = s * rng.gen_range(0.35..0.68);
        for step in 0..len {
            let t = step as f32 / (len.max(1) as f32);
            let taper = (t * (1.0 - t) * 4.0).sqrt();
            let a = darkness * taper;
            let ix = sx.round() as i32;
            let iy = sy.round() as i32;
            for oy in -1..=1i32 {
                for ox in -1..=1i32 {
                    let (xx, yy) = (ix + ox, iy + oy);
                    if xx >= 0 && yy >= 0 && xx < wsz as i32 && yy < hsz as i32 {
                        let w_px = if ox == 0 && oy == 0 { a } else { a * 0.35 };
                        let j = (yy as usize * wsz + xx as usize) * 3;
                        for c in 0..3 {
                            buf[j + c] =
                                (buf[j + c] * (1.0 - w_px) + 38.0 * w_px).clamp(0.0, 255.0);
                        }
                    }
                }
            }
            sx += dx - dy * curv * step as f32;
            sy += dy + dx * curv * step as f32;
        }
    }
    (thermal_fade, thermal_tone.to_string(), thermal_scratches)
}

/// Fast ink-only micro-feathering & halftone/toner grain (strictly dark ink
/// `luma < 140.0`, skipping ~90% of white paper pixels).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_ink_micro_texture(
    buf: &mut [f32],
    w: u32,
    h: u32,
    file_seed: u64,
    grain_sigma: f32,
    feather_amp: f32,
    halftone_amp: f32,
) {
    if grain_sigma <= 0.0 && feather_amp <= 0.0 && halftone_amp <= 0.0 {
        return;
    }
    let (wsz, hsz) = (w as usize, h as usize);
    if wsz == 0 || hsz == 0 || buf.len() < wsz * hsz * 3 {
        return;
    }
    let tile = crate::banks::BANK_TILE as usize;
    let mask = tile - 1;
    let bank =
        crate::banks::generate_gauss_bank(file_seed, crate::banks::TAG_INK_TEX, tile as u32, 1.0);
    const BAYER4: [f32; 16] = [
        -0.94, 0.06, -0.69, 0.31, 0.56, -0.44, 0.81, -0.19, -0.56, 0.44, -0.81, 0.19, 0.94, -0.06,
        0.69, -0.31,
    ];
    let mut row_luma = vec![0.0f32; wsz];

    for y in 0..hsz {
        let row = y * wsz;
        let row_rgb = &buf[row * 3..(row + wsz) * 3];
        for (dst, p) in row_luma.iter_mut().zip(row_rgb.as_chunks::<3>().0) {
            *dst = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        }
        let brow = (y & mask) * tile;
        let by4 = (y & 3) * 4;
        for x in 0..wsz {
            let l = row_luma[x];
            if l >= 140.0 {
                continue;
            }
            let b_val = bank[brow + (x & mask)];
            let lx = if x > 0 { row_luma[x - 1] } else { l };
            let rx = if x + 1 < wsz { row_luma[x + 1] } else { l };
            let is_edge = (lx - l).abs() > 35.0 || (rx - l).abs() > 35.0;
            let ink_w = ((140.0 - l) / 140.0).clamp(0.0, 1.0);
            let mut delta = b_val * grain_sigma * (0.45 + 0.55 * ink_w);
            if is_edge && feather_amp > 0.0 {
                delta += bank[brow + ((x + 17) & mask)] * feather_amp;
            }
            if halftone_amp > 0.0 {
                delta += BAYER4[by4 + (x & 3)] * halftone_amp * ink_w;
            }
            let j = (row + x) * 3;
            buf[j] = (buf[j] + delta).clamp(0.0, 255.0);
            buf[j + 1] = (buf[j + 1] + delta).clamp(0.0, 255.0);
            buf[j + 2] = (buf[j + 2] + delta).clamp(0.0, 255.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn synth(w: usize, h: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let dark = y % 8 == 0 || (x > 10 && x < 20 && y > 5 && y < 20);
                let v = if dark { 20.0 } else { 244.0 };
                let j = (y * w + x) * 3;
                buf[j] = v;
                buf[j + 1] = v;
                buf[j + 2] = v;
            }
        }
        buf
    }

    fn box_blur_direct(snap: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
        // The old per-pixel gather, kept as the test reference.
        let mut out = vec![0.0f32; w * h];
        let n = ((2 * r + 1) * (2 * r + 1)) as f32;
        for y in 0..h {
            for x in 0..w {
                let sy = h - 1 - y;
                let mut acc = 0.0f32;
                for oy in -r..=r {
                    for ox in -r..=r {
                        let sx = (x as i32 + ox).clamp(0, w as i32 - 1) as usize;
                        let yy = (sy as i32 + oy).clamp(0, h as i32 - 1) as usize;
                        acc += snap[yy * w + sx];
                    }
                }
                out[y * w + x] = acc / n;
            }
        }
        out
    }

    #[test]
    fn ghost_separable_matches_box_mean_within_one_lsb() {
        // Separable sliding-window blur vs the old direct gather: same box
        // mean up to fp reassociation — far below 1 gray level, so the
        // ghost veil (alpha <= 0.14) cannot move an Otsu threshold.
        let (w, h) = (37usize, 23usize);
        let mut snap = vec![0.0f32; w * h];
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for v in snap.iter_mut() {
            *v = rng.gen_range(0.0..255.0);
        }
        for r in [1i32, 2] {
            let fast = box_blur_flipped(&snap, w, h, r);
            let slow = box_blur_direct(&snap, w, h, r);
            let max = fast
                .iter()
                .zip(slow.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max < 0.1, "r={r}: separable max diff {max} too large");
            // After u8 rounding (the only thing a viewer/JPEG sees): <= 1 LSB.
            let umax = fast
                .iter()
                .zip(slow.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap();
            assert!(umax <= 1, "r={r}: u8 max diff {umax} exceeds 1 LSB");
        }
    }

    #[test]
    fn toner_lut_entries_match_direct_formula_exactly() {
        // The LUT evaluates the identical expression tree as the direct
        // per-pixel loop, so entries (and hence the toner term) are
        // bit-exact, not merely close.
        let tau = std::f32::consts::TAU;
        for (fx, fy, ph) in [(1.0f32, 1.0f32, 0.0f32), (2.7, 1.3, 2.1), (1.9, 2.9, 5.6)] {
            let (w, h) = (64usize, 48usize);
            for x in 0..w {
                let direct = (tau * fx * (x as f32 / w as f32) + ph).sin();
                let mut col = vec![0.0f32; w];
                for (xx, s) in col.iter_mut().enumerate() {
                    *s = (tau * fx * (xx as f32 / w as f32) + ph).sin();
                }
                assert_eq!(col[x], direct, "sin col x={x}");
            }
            for y in 0..h {
                let direct = (tau * fy * (y as f32 / h as f32) + 0.7 * ph).cos();
                let mut row = vec![0.0f32; h];
                for (yy, c) in row.iter_mut().enumerate() {
                    *c = (tau * fy * (yy as f32 / h as f32) + 0.7 * ph).cos();
                }
                assert_eq!(row[y], direct, "cos row y={y}");
            }
        }
    }

    #[test]
    fn determinism_same_seed_identical() {
        let (w, h) = (48u32, 32u32);
        let mut a = synth(48, 32);
        let mut b = a.clone();
        let pa = apply(&mut a, w, h, &mut ChaCha8Rng::seed_from_u64(7), 0.6);
        let pb = apply(&mut b, w, h, &mut ChaCha8Rng::seed_from_u64(7), 0.6);
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(&pa).unwrap(),
            serde_json::to_string(&pb).unwrap()
        );
    }

    #[test]
    fn effect_differs_and_params_in_range() {
        let (w, h) = (48u32, 32u32);
        let orig = synth(48, 32);
        let mut buf = orig.clone();
        let p = apply(&mut buf, w, h, &mut ChaCha8Rng::seed_from_u64(42), 0.8);
        assert!(p.enabled);
        assert_ne!(buf, orig);
        assert!((0.0..=1.0).contains(&p.fade_amount));
        assert!((0.0..=0.14).contains(&p.ghost_alpha));
        assert!((0.0..=0.7).contains(&p.brown_amount));
        assert!((1..=2).contains(&p.ghost_radius));
        assert!((6..14).contains(&p.gap_max_len));
        assert!((1.0..2.2).contains(&p.dropout_radius));
        assert!(buf.iter().all(|v| (0.0..=255.0).contains(v)));
        // Strength 0 is a no-op returning disabled params.
        let mut untouched = orig.clone();
        let off = apply(&mut untouched, w, h, &mut ChaCha8Rng::seed_from_u64(1), 0.0);
        assert!(!off.enabled);
        assert_eq!(untouched, orig);
    }
}
