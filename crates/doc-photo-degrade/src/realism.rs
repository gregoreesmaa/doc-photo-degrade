//! Wave 2 photorealism stages:
//! - Coupled 3D paper crumpling normal shading (photometric-only, zero geometric shift)
//! - Smartphone HDR unsharp-mask (USM) micro-contrast haloing
//! - Procedural desk surfaces (`walnut`, `slate`, `cloth`) + soft page contact drop-shadow
//! - Plastic sheet-protector / laminated anisotropic diagonal glare sheen band
//! - Mixed-illuminant AWB color-temperature gradient helper
//!
//! Every effect is 100% deterministic from the seeded RNG / `file_seed` and
//! preserves `< 0.2 px` exact non-linear bend reprojection accuracy.

use crate::masks::artifact_bits;
use crate::reproject::ReprojectContext;
use crate::DegradeParams;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Per-effect toggles for Wave 2 photorealism stages, calibrated per
/// [`crate::DocumentProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealismConfig {
    /// Mixed-illuminant warm-to-cool AWB color-temperature cast in Stage 5 lighting.
    pub enable_awb: bool,
    /// Linear-light luminance-dependent Poisson shot-noise component in Stage 6 noise.
    pub enable_shot_noise: bool,
    /// Smartphone ISP unsharp-mask (USM) micro-contrast haloing.
    pub enable_usm: bool,
    /// Procedural desk surface texture (`walnut`, `slate`, `cloth`) + soft page contact shadow.
    pub enable_desk_surface: bool,
    /// Multi-scale 3D paper crumpling normal shading (photometric only).
    pub enable_crumple: bool,
    /// Occasional plastic sheet-protector / laminated diagonal specular sheen band.
    pub enable_sheen: bool,
}

impl Default for RealismConfig {
    fn default() -> Self {
        Self::phone()
    }
}

impl RealismConfig {
    /// Full smartphone-capture realism (all Wave 2 effects enabled).
    pub const fn phone() -> Self {
        Self {
            enable_awb: true,
            enable_shot_noise: true,
            enable_usm: true,
            enable_desk_surface: true,
            enable_crumple: true,
            enable_sheen: true,
        }
    }

    /// Flatbed scanner / photocopy mode (all camera/desk/crumple effects disabled).
    pub const fn flat_scanner() -> Self {
        Self {
            enable_awb: false,
            enable_shot_noise: false,
            enable_usm: false,
            enable_desk_surface: false,
            enable_crumple: false,
            enable_sheen: false,
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

/// Multi-scale 3D paper crumpling normal shading (`dhx * lx + dhy * ly`).
/// Purely photometric (no pixel coordinate warp), so exact non-linear bend
/// reprojection remains `< 0.2 px` accurate.
pub(crate) fn apply_crumple_shading(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    enabled: bool,
) -> f32 {
    if !enabled || strength <= 0.05 || w < 8 || h < 8 {
        return 0.0;
    }
    let crumple_strength = strength * rng.gen_range(0.08..0.22);
    // 3 multi-scale harmonic octaves with random phases & wave-vectors.
    let mut freqs = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 4];
    let amps = [0.45f32, 0.30, 0.15, 0.10];
    let scales = [1.5f32, 3.2, 6.0, 10.5];
    for i in 0..4 {
        let ang = rng.gen_range(0.0..std::f32::consts::TAU);
        let k = scales[i] * std::f32::consts::TAU / (w.min(h) as f32).max(1.0);
        let ph = rng.gen_range(0.0..std::f32::consts::TAU);
        freqs[i] = (k * ang.cos(), k * ang.sin(), ph, amps[i]);
    }
    let light_ang = rng.gen_range(0.0..std::f32::consts::TAU);
    let (lx, ly) = (light_ang.cos(), light_ang.sin());

    for y in 0..h {
        let fy = y as f32;
        for x in 0..w {
            let fx = x as f32;
            let mut dhx = 0.0f32;
            let mut dhy = 0.0f32;
            for &(kx, ky, ph, amp) in &freqs {
                let c = (kx * fx + ky * fy + ph).cos();
                // Normalize gradient contribution by spatial frequency so all octaves stay balanced.
                let k_norm = (kx * kx + ky * ky).sqrt().max(1e-6);
                dhx += amp * (kx / k_norm) * c;
                dhy += amp * (ky / k_norm) * c;
            }
            let ndotl = dhx * lx + dhy * ly;
            let shade = (1.0 + crumple_strength * ndotl).clamp(0.75, 1.12);
            let i = ((y * w + x) * 3) as usize;
            buf[i] = (buf[i] * shade).clamp(0.0, 255.0);
            buf[i + 1] = (buf[i + 1] * shade).clamp(0.0, 255.0);
            buf[i + 2] = (buf[i + 2] * shade).clamp(0.0, 255.0);
        }
    }
    crumple_strength
}

/// Smartphone HDR unsharp-mask (USM) micro-contrast haloing around high-contrast
/// ink edges. Uses a symmetric separable box blur of luma so ink centroids are
/// never shifted.
pub(crate) fn apply_usm(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    enabled: bool,
) -> f32 {
    if !enabled || strength <= 0.05 || w < 8 || h < 8 {
        return 0.0;
    }
    let usm_strength = strength * rng.gen_range(0.25..0.60);
    let rad: i32 = if w.max(h) >= 900 { 2 } else { 1 };
    let n = (w as usize) * (h as usize);
    let mut luma = vec![0.0f32; n];
    for i in 0..n {
        luma[i] = (buf[i * 3] + buf[i * 3 + 1] + buf[i * 3 + 2]) * (1.0 / 3.0);
    }

    // Separable horizontal box pass.
    let mut tmp = vec![0.0f32; n];
    let inv_k = 1.0 / (2 * rad + 1) as f32;
    let wi = w as i32;
    let hi = h as i32;
    for y in 0..hi {
        let row = (y as u32 * w) as usize;
        for x in 0..wi {
            let mut sum = 0.0f32;
            for d in -rad..=rad {
                let cx = (x + d).clamp(0, wi - 1) as usize;
                sum += luma[row + cx];
            }
            tmp[row + x as usize] = sum * inv_k;
        }
    }

    // Separable vertical box pass + USM halo application.
    for y in 0..hi {
        let row = (y as u32 * w) as usize;
        for x in 0..wi {
            let mut sum = 0.0f32;
            for d in -rad..=rad {
                let cy = (y + d).clamp(0, hi - 1) as u32;
                sum += tmp[(cy * w) as usize + x as usize];
            }
            let blurred = sum * inv_k;
            let idx = row + x as usize;
            let detail = (luma[idx] - blurred).clamp(-32.0, 32.0);
            let delta = usm_strength * detail;
            let p = idx * 3;
            buf[p] = (buf[p] + delta).clamp(0.0, 255.0);
            buf[p + 1] = (buf[p + 1] + delta).clamp(0.0, 255.0);
            buf[p + 2] = (buf[p + 2] + delta).clamp(0.0, 255.0);
        }
    }
    usm_strength
}

/// Deterministic 2D hash in `[0, 1)` for procedural desk textures.
#[inline]
fn hash2(ix: i32, iy: i32, seed: u32) -> f32 {
    let mut h =
        seed ^ (ix.wrapping_mul(374_761_393) as u32) ^ (iy.wrapping_mul(668_265_263) as u32);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0x00FF_FFFF) as f32 / 16_777_215.0
}

#[inline]
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let n00 = hash2(ix, iy, seed);
    let n10 = hash2(ix + 1, iy, seed);
    let n01 = hash2(ix, iy + 1, seed);
    let n11 = hash2(ix + 1, iy + 1, seed);
    let top = n00 * (1.0 - u) + n10 * u;
    let bot = n01 * (1.0 - u) + n11 * u;
    top * (1.0 - v) + bot * v
}

#[inline]
fn desk_pixel(
    surface: &str,
    x: f32,
    y: f32,
    angle_cos: f32,
    angle_sin: f32,
    seed: u32,
) -> [f32; 3] {
    match surface {
        "walnut" => {
            // Dark walnut wood grain: elongated directional fibers + subtle ring bands.
            let u = x * angle_cos + y * angle_sin;
            let v = -x * angle_sin + y * angle_cos;
            let warp = value_noise(u * 0.012, v * 0.06, seed) * 6.0;
            let grain = (v * 0.14 + warp).sin() * 0.5 + 0.5;
            let fine = value_noise(u * 0.04, v * 0.35, seed ^ 0x9E37_79B9);
            let tone = 0.65 * grain + 0.35 * fine;
            [34.0 + tone * 30.0, 23.0 + tone * 21.0, 16.0 + tone * 14.0]
        }
        "cloth" => {
            // Warm woven desk mat / linen cloth.
            let weave = ((x * 0.9).sin() * (y * 0.9).sin()) * 0.5 + 0.5;
            let macro_n = value_noise(x * 0.03, y * 0.03, seed);
            let t = 0.45 * weave + 0.55 * macro_n;
            [42.0 + t * 22.0, 38.0 + t * 20.0, 34.0 + t * 18.0]
        }
        _ => {
            // "slate": matte dark slate laminate with micro-speckle.
            let n1 = value_noise(x * 0.04, y * 0.04, seed);
            let n2 = value_noise(x * 0.22, y * 0.22, seed ^ 0x517C_C1B7);
            let t = 0.65 * n1 + 0.35 * n2;
            [28.0 + t * 20.0, 31.0 + t * 21.0, 36.0 + t * 23.0]
        }
    }
}

/// Sampled parameters for the optional plastic sheet-protector / laminated
/// diagonal specular sheen band.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SheenParams {
    pub strength: f32,
    pub angle_rad: f32,
    pub pos: f32,
    pub width_frac: f32,
}

impl SheenParams {
    pub(crate) fn sample(rng: &mut ChaCha8Rng, s: f32, enabled: bool) -> Self {
        if !enabled || s < 0.4 {
            return Self::default();
        }
        if rng.gen::<f32>() >= 0.28 {
            return Self::default();
        }
        let deg = if rng.gen::<f32>() < 0.5 {
            rng.gen_range(20.0..70.0f32)
        } else {
            rng.gen_range(110.0..160.0f32)
        };
        Self {
            strength: s * rng.gen_range(0.18..0.38),
            angle_rad: deg.to_radians(),
            pos: rng.gen_range(0.25..0.75),
            width_frac: rng.gen_range(0.06..0.13),
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

/// Post-downscale pass (runs identically on CPU and GPU before JPEG):
/// 1. Replaces out-of-page flat desk pixels with a deterministic procedural desk
///    surface (`walnut`, `slate`, `cloth`) + soft directional page contact
///    drop-shadow along the page perimeter.
/// 2. Applies optional plastic sheet-protector diagonal specular sheen on-page.
///
/// Never alters geometric coordinates, preserving `< 0.2 px` exact reprojection.
pub(crate) fn apply_desk_and_sheen(
    buf: &mut [f32],
    params: &DegradeParams,
    desk_surface: &str,
    sheen: &SheenParams,
    file_seed: u64,
) {
    let run_desk = desk_surface != "none";
    let run_sheen = sheen.strength > 0.0;
    if !run_desk && !run_sheen {
        return;
    }
    let w = params.width;
    let h = params.height;
    let wf = w as f32;
    let hf = h as f32;
    let ctx = ReprojectContext::new(params);
    let seed32 = (file_seed ^ (file_seed >> 32) ^ 0xA5A5_1234) as u32;
    let wood_ang = ((seed32 % 360) as f32).to_radians();
    let (wa_cos, wa_sin) = (wood_ang.cos(), wood_ang.sin());

    // Contact drop-shadow direction opposite to the scene shadow direction.
    let sh_th = (params.shadow_angle_deg as f64).to_radians();
    let (sh_dx, sh_dy) = (sh_th.cos(), sh_th.sin());
    let shadow_radius = (w.min(h) as f64 * 0.022).clamp(4.0, 32.0);
    let w0f = ctx.w0 as f64;
    let h0f = ctx.h0 as f64;
    let scale_approx = (w as f64 / ctx.w_rot.max(1) as f64).clamp(0.1, 2.0);

    for y in 0..h {
        let fy = y as f64;
        for x in 0..w {
            let fx = x as f64;
            let idx = ((y * w + x) * 3) as usize;

            // Undo downscale -> rotation -> framing to get signed distance from the page rect.
            let (x_rot, y_rot) = ctx.undo_downscale(fx, fy);
            let (x_frame, y_frame) = if ctx.rot_active {
                let (cx0, cy0) = (ctx.fw as f64 / 2.0, ctx.fh as f64 / 2.0);
                let (cx1, cy1) = (ctx.w_rot as f64 / 2.0, ctx.h_rot as f64 / 2.0);
                let dx = x_rot - cx1;
                let dy = y_rot - cy1;
                (
                    cx0 + dx * ctx.cst + dy * ctx.snt,
                    cy0 - dx * ctx.snt + dy * ctx.cst,
                )
            } else {
                (x_rot, y_rot)
            };

            let (x_persp, y_persp) = if params.frame_mode == "inset" {
                (
                    x_frame - params.frame_pad_px[0] as f64,
                    y_frame - params.frame_pad_px[1] as f64,
                )
            } else {
                (
                    x_frame + params.frame_cut_px[0] as f64,
                    y_frame + params.frame_cut_px[1] as f64,
                )
            };

            // Distance outside the framed page rectangle (in output pixels).
            let dx_out = if x_persp < 0.0 {
                -x_persp
            } else if x_persp > w0f {
                x_persp - w0f
            } else {
                0.0
            };
            let dy_out = if y_persp < 0.0 {
                -y_persp
            } else if y_persp > h0f {
                y_persp - h0f
            } else {
                0.0
            };
            let dist_out = dx_out.hypot(dy_out) * scale_approx;

            if dist_out > 0.0 {
                if run_desk {
                    let mut tex =
                        desk_pixel(desk_surface, x as f32, y as f32, wa_cos, wa_sin, seed32);
                    // Directional contact drop-shadow along page perimeter.
                    let dir_bias = if dx_out > 0.0 || dy_out > 0.0 {
                        let ox = if x_persp < 0.0 {
                            -1.0
                        } else if x_persp > w0f {
                            1.0
                        } else {
                            0.0
                        };
                        let oy = if y_persp < 0.0 {
                            -1.0
                        } else if y_persp > h0f {
                            1.0
                        } else {
                            0.0
                        };
                        let align = (ox * sh_dx + oy * sh_dy).clamp(-1.0, 1.0);
                        0.65 + 0.35 * align
                    } else {
                        0.65
                    };
                    let eff_rad = shadow_radius * dir_bias.max(0.25);
                    let shadow_mul = 1.0 - 0.62 * (-dist_out / eff_rad).exp();
                    tex[0] *= shadow_mul as f32;
                    tex[1] *= shadow_mul as f32;
                    tex[2] *= shadow_mul as f32;
                    // Smooth 1.5 px AA blend at the page rim.
                    let a = (dist_out / 1.5).clamp(0.0, 1.0) as f32;
                    buf[idx] = buf[idx] * (1.0 - a) + tex[0] * a;
                    buf[idx + 1] = buf[idx + 1] * (1.0 - a) + tex[1] * a;
                    buf[idx + 2] = buf[idx + 2] * (1.0 - a) + tex[2] * a;
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

/// Stamp sheet-protector specular sheen pixels into an output-space artifact mask.
pub(crate) fn stamp_sheen_mask(art: &mut [u16], w: u32, h: u32, sheen: &SheenParams) {
    if sheen.strength <= 0.0 || art.len() < (w as usize) * (h as usize) {
        return;
    }
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
    fn crumple_shading_and_usm_are_deterministic_and_bounded() {
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
        let ca = apply_crumple_shading(&mut a, w, h, &mut ra, 0.7, true);
        let cb = apply_crumple_shading(&mut b, w, h, &mut rb, 0.7, true);
        assert_eq!(ca, cb);
        assert_eq!(a, b);
        assert!(ca > 0.0 && a != base);

        let ua = apply_usm(&mut a, w, h, &mut ra, 0.7, true);
        let ub = apply_usm(&mut b, w, h, &mut rb, 0.7, true);
        assert_eq!(ua, ub);
        assert_eq!(a, b);
        assert!(ua > 0.0);
        assert!(a.iter().all(|&v| (0.0..=255.0).contains(&v)));
    }

    #[test]
    fn wave2_profiles_calibrate_realism_params_and_sheen_marks_glare() {
        let mut img = RgbImage::from_pixel(96, 96, Rgb([250, 250, 248]));
        for y in 24..72 {
            for x in 24..72 {
                img.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        // OfficeScan disables camera/desk/crumple/USM/sheen.
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

        // PhonePhoto enables AWB, USM, desk surface, crumple, and occasional sheen.
        let mut saw_sheen = false;
        let mut surfaces = std::collections::HashSet::new();
        for seed in 0..20u64 {
            let mut cfg = DegradeConfig::default().with_seed(seed).with_strength(0.75);
            cfg.masks = MaskRequest::all();
            let res = degrade_image(img.clone(), &cfg).unwrap();
            assert!(res.params.awb_warm_cool_strength > 0.0);
            assert!(res.params.usm_strength > 0.0);
            assert!(res.params.crumple_strength > 0.0);
            assert!(["walnut", "slate", "cloth"].contains(&res.params.desk_surface.as_str()));
            surfaces.insert(res.params.desk_surface.clone());
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
    }
}
