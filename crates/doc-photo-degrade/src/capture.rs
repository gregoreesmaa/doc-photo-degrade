//! Capture & handling artifacts for document training data.
//!
//! Buffer: RGB interleaved `f32` in `0..=255`, row-major, length `w*h*3`.
//! Every effect is margin/edge-biased so document content is never covered.
//! All randomness comes from the passed `rng` (deterministic per seed).

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::f32::consts::PI;

/// Sampled knobs for the capture/handling degradation pass.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct CaptureParams {
    pub enabled: bool,
    pub motion_angle_deg: f32,
    pub motion_radius: u32,
    pub fingerprint_edge: u8,
    pub fingerprint_x: u32,
    pub fingerprint_y: u32,
    pub fingerprint_rx: u32,
    pub fingerprint_ry: u32,
    pub fingerprint_oil: f32,
    pub tape_height: u32,
    pub tape_opacity: f32,
    pub tape_edge: u8,
    pub tape_start_frac: f32,
    pub tape_len_frac: f32,
    pub residue_count: u32,
    pub residue_rects: Vec<[u32; 4]>,
    pub hole_x: u32,
    pub hole_y: u32,
    pub hole_radius: u32,
    pub sticky_corner: u8,
    pub sticky_size: u32,
    pub sticky_tone: u8,
    pub exposure_lift: f32,
    pub exposure_compress: f32,
    pub banding_amplitude: f32,
    pub banding_period: u32,
    /// True when the edge tape strip uses serrated dispenser-torn ends and diagonal sheen.
    #[serde(default)]
    pub tape_serrated: bool,
    /// True when binder punch holes render continuous desk texture through the hole.
    #[serde(default)]
    pub hole_desk_through: bool,
}

fn ru(rng: &mut ChaCha8Rng, lo: u32, hi: u32, max: u32) -> u32 {
    let hi = hi.min(max).max(1);
    let lo = lo.min(hi.saturating_sub(1));
    if lo >= hi {
        lo
    } else {
        rng.gen_range(lo..hi)
    }
}

fn off() -> CaptureParams {
    CaptureParams {
        enabled: false,
        motion_angle_deg: 0.0,
        motion_radius: 0,
        fingerprint_edge: 0,
        fingerprint_x: 0,
        fingerprint_y: 0,
        fingerprint_rx: 0,
        fingerprint_ry: 0,
        fingerprint_oil: 0.0,
        tape_height: 0,
        tape_opacity: 0.0,
        tape_edge: 0,
        tape_start_frac: 0.0,
        tape_len_frac: 0.0,
        residue_count: 0,
        residue_rects: Vec::new(),
        hole_x: 0,
        hole_y: 0,
        hole_radius: 0,
        sticky_corner: 0,
        sticky_size: 0,
        sticky_tone: 0,
        exposure_lift: 0.0,
        exposure_compress: 1.0,
        banding_amplitude: 0.0,
        banding_period: 4,
        tape_serrated: false,
        hole_desk_through: false,
    }
}

/// `px` is the pixel-scale factor (1.0 = native): the motion-blur radius
/// and sensor-banding period scale with it; `px = 1.0` is bit-identical to
/// the legacy sampling. Relative-geometry knobs (tape, holes, fingerprints)
/// and degeneracy floors stay in output pixels by design.
pub fn apply(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    px: f32,
) -> CaptureParams {
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 || w < 8 || h < 8 || buf.len() < w as usize * h as usize * 3 {
        return off();
    }
    let base = w.min(h) as f32;
    let band = (w as f32 * 0.10) as u32;
    let vband = (h as f32 * 0.10) as u32;
    let fedge = rng.gen_range(0..4u8);
    let (fx, fy) = match fedge {
        0 => (
            ru(rng, 2, band.max(3), w),
            ru(rng, vband, h.saturating_sub(vband), h),
        ),
        1 => (
            ru(rng, w.saturating_sub(band), w.saturating_sub(2), w),
            ru(rng, vband, h.saturating_sub(vband), h),
        ),
        2 => (
            ru(rng, band, w.saturating_sub(band), w),
            ru(rng, 2, vband.max(3), h),
        ),
        _ => (
            ru(rng, band, w.saturating_sub(band), w),
            ru(rng, h.saturating_sub(vband), h.saturating_sub(2), h),
        ),
    };
    let frx = (base * rng.gen_range(0.03..0.06)).clamp(3.0, base) as u32;
    let hm = (base * 0.03) as u32;
    let hr = (base * rng.gen_range(0.008..0.02)).clamp(3.0, (base * 0.05).max(4.0)) as u32;
    let hedge = rng.gen_range(0..4u8);
    let (hx, hy) = match hedge {
        0 => (ru(rng, hm, hm + hr + 3, w), ru(rng, hm, h / 3, h)),
        1 => (
            ru(rng, w.saturating_sub(hm + hr + 3), w.saturating_sub(hm), w),
            ru(rng, hm, h / 3, h),
        ),
        2 => (ru(rng, hm, w / 3, w), ru(rng, hm, hm + hr + 3, h)),
        _ => (
            ru(rng, hm, w / 3, w),
            ru(rng, h.saturating_sub(hm + hr + 3), h.saturating_sub(hm), h),
        ),
    };
    let mut p = CaptureParams {
        enabled: true,
        motion_angle_deg: rng.gen_range(0.0..180.0),
        motion_radius: 1 + (rng.gen::<f32>() * 2.0 * s * px) as u32,
        fingerprint_edge: fedge,
        fingerprint_x: fx,
        fingerprint_y: fy,
        fingerprint_rx: frx,
        fingerprint_ry: (frx as f32 * 1.25).clamp(3.0, base) as u32,
        fingerprint_oil: rng.gen_range(6.0..18.0) * s,
        tape_height: (base * rng.gen_range(0.02..0.05)).clamp(2.0, base) as u32,
        tape_opacity: rng.gen_range(0.15..0.30) * (0.4 + 0.6 * s),
        tape_edge: rng.gen_range(0..4u8),
        tape_start_frac: rng.gen_range(0.0..0.55),
        tape_len_frac: rng.gen_range(0.30..0.55),
        residue_count: 0,
        residue_rects: Vec::new(),
        hole_x: hx,
        hole_y: hy,
        hole_radius: hr,
        sticky_corner: rng.gen_range(0..4u8),
        sticky_size: (base * rng.gen_range(0.12..0.22)).clamp(4.0, base) as u32,
        sticky_tone: rng.gen_range(0..3u8),
        exposure_lift: rng.gen_range(-1.0..1.0) * 15.0 * s,
        exposure_compress: 1.0 - rng.gen_range(0.0..0.12) * s,
        banding_amplitude: rng.gen_range(1.5..4.0) * s,
        banding_period: rng
            .gen_range((3.0 * px).round().max(2.0) as u32..(7.0 * px).round().max(3.0) as u32),
        tape_serrated: true,
        hole_desk_through: false,
    };
    // Fuse exposure error + sensor banding into the motion-blur snapshot build
    // (1 pass instead of 3 DRAM passes, bit-identical arithmetic).
    let mut snap = vec![0.0f32; (w as usize) * (h as usize) * 3];
    let inv_bp = 2.0 * PI / p.banding_period as f32;
    for y in 0..h {
        let off = p.banding_amplitude * (y as f32 * inv_bp).sin();
        let row_base = (y * w) as usize * 3;
        let row_end = row_base + (w as usize) * 3;
        for (dst_v, &src_v) in snap[row_base..row_end]
            .iter_mut()
            .zip(buf[row_base..row_end].iter())
        {
            let exp_v =
                ((src_v - 128.0) * p.exposure_compress + 128.0 + p.exposure_lift).clamp(0.0, 255.0);
            *dst_v = (exp_v + off).clamp(0.0, 255.0);
        }
    }
    // Motion blur: directional box blur along the seeded angle.
    let th = p.motion_angle_deg * PI / 180.0;
    let (dx, dy) = (th.cos(), th.sin());
    let r = p.motion_radius as i32;
    let inv_taps = 1.0 / (2 * r + 1) as f32;
    let offs: Vec<(i32, i32)> = (-r..=r)
        .map(|k| {
            (
                (dx * k as f32).round() as i32,
                (dy * k as f32).round() as i32,
            )
        })
        .collect();
    let wi = w as i32;
    let hi = h as i32;
    let ws = w as usize;
    for y in 0..hi {
        let y_interior = y >= r && y + r < hi;
        for x in 0..wi {
            let mut acc = [0.0f32; 3];
            if y_interior && x >= r && x + r < wi {
                for &(ox, oy) in &offs {
                    let j = ((y + oy) as usize * ws + (x + ox) as usize) * 3;
                    acc[0] += snap[j];
                    acc[1] += snap[j + 1];
                    acc[2] += snap[j + 2];
                }
            } else {
                for &(ox, oy) in &offs {
                    let nx = (x + ox).clamp(0, wi - 1) as usize;
                    let ny = (y + oy).clamp(0, hi - 1) as usize;
                    let j = (ny * ws + nx) * 3;
                    acc[0] += snap[j];
                    acc[1] += snap[j + 1];
                    acc[2] += snap[j + 2];
                }
            }
            let i = (y as usize * ws + x as usize) * 3;
            buf[i] = acc[0] * inv_taps;
            buf[i + 1] = acc[1] * inv_taps;
            buf[i + 2] = acc[2] * inv_taps;
        }
    }
    // Fingerprint: fingertip ellipse of concentric arc ridges + oil brightening.
    let (erx, ery) = (p.fingerprint_rx as f32, p.fingerprint_ry as f32);
    for oy in -(ery as i32)..=ery as i32 {
        for ox in -(erx as i32)..=erx as i32 {
            let t = ((ox as f32 / erx).powi(2) + (oy as f32 / ery).powi(2)).sqrt();
            if t <= 1.0 {
                let ridge = 0.5 + 0.5 * (t * 16.0 * PI).sin();
                let add = p.fingerprint_oil * (0.55 + 0.45 * ridge);
                let (px, py) = (p.fingerprint_x as i32 + ox, p.fingerprint_y as i32 + oy);
                if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                    let i = (py as u32 * w + px as u32) as usize * 3;
                    for c in 0..3 {
                        buf[i + c] = (buf[i + c] + add).clamp(0.0, 255.0);
                    }
                }
            }
        }
    }
    // Tape strip v2: translucent frosted or aged cellophane along one edge
    // (partial span) with serrated dispenser-torn ends, milky adhesive rims,
    // and a diagonal glossy specular sheen band.
    {
        let clear_frosted = (p.tape_edge + p.sticky_tone).is_multiple_of(3);
        let tape_rgb: [f32; 3] = if clear_frosted {
            [235.0, 233.0, 226.0]
        } else {
            [225.0, 205.0, 165.0]
        };
        let t = p.tape_height.min(w.min(h));
        let along = if p.tape_edge.is_multiple_of(2) { w } else { h };
        let a0 = ((p.tape_start_frac * along as f32) as u32).min(along);
        let a1 = ((p.tape_start_frac + p.tape_len_frac) * along as f32) as u32;
        let (a0, a1) = (a0.min(along), a1.min(along).max(a0 + 1));
        let tooth_p = (t / 6).max(3) as f32;
        let span = (a1 - a0).max(1) as f32;
        let t_f = t.max(1) as f32;
        let mut paint_tape = |px: u32, py: u32, u: u32, a: u32| {
            let serr = ((u as f32 % tooth_p) - tooth_p * 0.5).abs() / (tooth_p * 0.5);
            let trim = (serr * 3.5) as u32;
            if a < a0 + trim || a + trim >= a1 {
                return;
            }
            if px < w && py < h {
                let end_frost = if a < a0 + trim + 2 || a + trim + 2 >= a1 {
                    1.0
                } else {
                    0.0
                };
                let rim_frost = if u < 2 || u + 2 >= t { 1.0 } else { 0.0 };
                let alpha = (p.tape_opacity + 0.08 * end_frost + 0.05 * rim_frost).min(0.65);
                let un = (u as f32 / t_f) - 0.5;
                let an = ((a - a0) as f32 / span) - 0.5;
                let sheen_d = (an * 0.82 + un * 0.57) * 4.5;
                let spec = (-sheen_d * sheen_d).exp() * (24.0 + 9.0 * (an * 18.0).sin());
                let i = (py * w + px) as usize * 3;
                for c in 0..3 {
                    buf[i + c] =
                        (buf[i + c] * (1.0 - alpha) + tape_rgb[c] * alpha + spec).clamp(0.0, 255.0);
                }
            }
        };
        match p.tape_edge % 4 {
            0 => {
                for u in 0..t {
                    for a in a0..a1 {
                        paint_tape(a, u, u, a);
                    }
                }
            }
            1 => {
                for u in 0..t {
                    for a in a0..a1 {
                        paint_tape(w - 1 - u.min(w - 1), a, u, a);
                    }
                }
            }
            2 => {
                for u in 0..t {
                    for a in a0..a1 {
                        paint_tape(a, h - 1 - u.min(h - 1), u, a);
                    }
                }
            }
            _ => {
                for u in 0..t {
                    for a in a0..a1 {
                        paint_tape(u, a, u, a);
                    }
                }
            }
        }
    }
    // Tape residue: aged adhesive ghosts where tape was removed — yellowed
    // translucent rectangles with a darker adhesive border and slight gloss.
    // Translucent by design: document content stays readable underneath.
    {
        let n_res = if rng.gen::<f32>() < s * 0.45 {
            rng.gen_range(1..=2)
        } else {
            0
        };
        const RESIN: [f32; 3] = [235.0, 220.0, 178.0];
        for _ in 0..n_res {
            let rw = (base * rng.gen_range(0.08..0.25)).clamp(4.0, w as f32) as u32;
            let rh = (base * rng.gen_range(0.03..0.08)).clamp(3.0, h as f32) as u32;
            let edge = rng.gen_range(0..4u8);
            let (rx, ry) = match edge {
                0 => (
                    rng.gen_range(0..w.saturating_sub(rw).max(1)),
                    rng.gen_range(0..(h / 6).max(2)),
                ),
                1 => (
                    rng.gen_range(0..w.saturating_sub(rw).max(1)),
                    h.saturating_sub(rh + rng.gen_range(0..(h / 6).max(1))),
                ),
                2 => (
                    rng.gen_range(0..(w / 6).max(2)),
                    rng.gen_range(0..h.saturating_sub(rh).max(1)),
                ),
                _ => (
                    w.saturating_sub(rw + rng.gen_range(0..(w / 6).max(1))),
                    rng.gen_range(0..h.saturating_sub(rh).max(1)),
                ),
            };
            let alpha = rng.gen_range(0.10..0.22) * (0.4 + 0.6 * s);
            for y in ry..(ry + rh).min(h) {
                for x in rx..(rx + rw).min(w) {
                    let i = (y * w + x) as usize * 3;
                    let border = x == rx
                        || y == ry
                        || x + 1 >= (rx + rw).min(w)
                        || y + 1 >= (ry + rh).min(h);
                    for c in 0..3 {
                        buf[i + c] = buf[i + c] * (1.0 - alpha) + RESIN[c] * alpha;
                        if border {
                            buf[i + c] *= 0.85;
                        } else {
                            buf[i + c] = (buf[i + c] + 8.0 * alpha).clamp(0.0, 255.0);
                        }
                    }
                }
            }
            p.residue_count += 1;
            p.residue_rects.push([rx, ry, rw, rh]);
        }
    }
    // Staple/punch hole: dark circle with a light rim, margin-safe.
    let hrr = p.hole_radius as f32;
    for oy in -(p.hole_radius as i32 + 2)..=p.hole_radius as i32 + 2 {
        for ox in -(p.hole_radius as i32 + 2)..=p.hole_radius as i32 + 2 {
            let d = ((ox * ox + oy * oy) as f32).sqrt();
            let (px, py) = (p.hole_x as i32 + ox, p.hole_y as i32 + oy);
            if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                continue;
            }
            let i = (py as u32 * w + px as u32) as usize * 3;
            if d <= hrr {
                for c in 0..3 {
                    buf[i + c] = 25.0 + buf[i + c] * 0.1;
                }
            } else if d <= hrr + 2.0 {
                for c in 0..3 {
                    buf[i + c] = (buf[i + c] + 40.0).clamp(0.0, 255.0);
                }
            }
        }
    }
    // Sticky note: opaque pastel square overlapping a corner margin.
    const PASTEL: [[f32; 3]; 3] = [
        [255.0, 235.0, 130.0],
        [255.0, 180.0, 200.0],
        [190.0, 240.0, 180.0],
    ];
    let col = PASTEL[p.sticky_tone as usize % 3];
    let sz = p.sticky_size.min(w).min(h);
    let (ox, oy) = match p.sticky_corner % 4 {
        0 => (0, 0),
        1 => (w - sz, 0),
        2 => (0, h - sz),
        _ => (w - sz, h - sz),
    };
    for y in oy..oy + sz {
        for x in ox..ox + sz {
            let i = (y * w + x) as usize * 3;
            buf[i] = col[0];
            buf[i + 1] = col[1];
            buf[i + 2] = col[2];
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn synth(w: u32, h: u32) -> Vec<f32> {
        let mut v = vec![0.0; w as usize * h as usize * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize * 3;
                v[i] = x as f32 * 255.0 / w as f32;
                v[i + 1] = y as f32 * 255.0 / h as f32;
                v[i + 2] = 128.0;
            }
        }
        v
    }

    #[test]
    fn same_seed_same_bytes() {
        let (w, h) = (48, 32);
        let (mut a, mut b) = (synth(w, h), synth(w, h));
        let (mut ra, mut rb) = (ChaCha8Rng::seed_from_u64(7), ChaCha8Rng::seed_from_u64(7));
        let pa = apply(&mut a, w, h, &mut ra, 0.7, 1.0);
        let pb = apply(&mut b, w, h, &mut rb, 0.7, 1.0);
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(&pa).unwrap(),
            serde_json::to_string(&pb).unwrap()
        );
    }

    #[test]
    fn output_differs_and_params_in_range() {
        let (w, h) = (48, 32);
        let mut buf = synth(w, h);
        let before = buf.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let p = apply(&mut buf, w, h, &mut rng, 0.8, 1.0);
        assert!(p.enabled && buf != before);
        assert!((0.0..180.0).contains(&p.motion_angle_deg) && (1..=3).contains(&p.motion_radius));
        assert!(
            p.fingerprint_x < w && p.fingerprint_y < h && (0.0..=18.0).contains(&p.fingerprint_oil)
        );
        assert!(
            p.tape_height >= 2
                && p.tape_height <= w.min(h)
                && (0.0..=1.0).contains(&p.tape_opacity)
        );
        assert!(
            p.tape_edge < 4
                && (0.0..0.55).contains(&p.tape_start_frac)
                && (0.30..=0.55).contains(&p.tape_len_frac)
        );
        assert!(p.residue_count <= 2 && p.residue_rects.len() as u32 == p.residue_count);
        for r in &p.residue_rects {
            assert!(r[0] + r[2] <= w && r[1] + r[3] <= h);
        }
        assert!(p.hole_x < w && p.hole_y < h && p.hole_radius >= 3);
        assert!(p.sticky_size >= 4 && p.sticky_size <= w.min(h) && p.sticky_tone < 3);
        assert!(
            (-15.0..=15.0).contains(&p.exposure_lift)
                && (0.88..=1.0).contains(&p.exposure_compress)
        );
        assert!((0.0..=4.0).contains(&p.banding_amplitude) && (3..7).contains(&p.banding_period));
        assert!(buf.iter().all(|&v| (0.0..=255.0).contains(&v)));
    }
}
