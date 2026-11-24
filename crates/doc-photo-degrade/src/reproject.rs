//! Ground-truth coordinate reprojection utilities built directly from
//! [`DegradeParams`].
//!
//! Maps points, axis-aligned bounding boxes, and polygons between the
//! pre-degrade working-image coordinate space (after `max_width` / `train_width`
//! pre-scaling) and the final degraded output image space through the full
//! geometric pipeline:
//! 1. Non-linear page bend (`curl_side`: `"left"`, `"right"`, `"top"`,
//!    `"bottom"`, `"spine"`, `"ripple"`)
//! 2. Perspective homography (`corner_offsets`)
//! 3. Photo framing (`frame_mode`: `"inset"` pad or `"tight"` cut)
//! 4. Canvas-expanding page rotation (`rotation_deg`)
//! 5. Output capture downscale (`params.width / w_rot`, `params.height / h_rot`
//!    with pixel-center convention on CPU, or `scale_x`/`scale_y` on fused GPU)
//!
//! # Exact vs Linear Functions
//!
//! - [`transform_point_exact`], [`unproject_point_exact`],
//!   [`transform_bbox_exact`], and [`transform_polygon_exact`] model **all 5
//!   stages** including exact non-linear page bend inversion (`< 0.2 px`
//!   end-to-end rendered accuracy, `< 1e-9 px` algebraic round-trip).
//! - [`transform_point_linear`], [`transform_bbox`], and [`transform_polygon`]
//!   model the projective + framing + rotation + downscale stages without
//!   non-linear page bend (exact when `--skip-bend` / `OfficeScan` /
//!   `PhotocopyXerox` is used).

use crate::geometry::{apply_h, framed_dims, invert_3x3, solve_homography};
use crate::DegradeParams;

/// Derive the working-image `(width, height)` prior to bend/perspective/framing.
///
/// Mirrors the pre-degrade sizing logic in `lib.rs` (`src_width`/`src_height`
/// rescaled to `train_width` when active), falling back to `(params.width,
/// params.height)` if `src_width`/`src_height` are unset.
pub(crate) fn working_dims(params: &DegradeParams) -> (u32, u32) {
    if params.src_width == 0 || params.src_height == 0 {
        return (params.width, params.height);
    }
    if params.train_width > 0 && params.src_width > params.train_width {
        let th = (params.src_height as u64 * params.train_width as u64 / params.src_width as u64)
            .max(1) as u32;
        (params.train_width, th)
    } else {
        (params.src_width, params.src_height)
    }
}

/// Exact `f64` backward page-bend map: given a destination coordinate `(xb, yb)`
/// in bent-image space, returns the source coordinate `(xs, ys)` sampled by
/// Stage 2 in `lib.rs`.
pub fn bend_backward(xb: f64, yb: f64, w0: u32, h0: u32, params: &DegradeParams) -> (f64, f64) {
    if params.curl_px <= 0.25 || w0 == 0 || h0 == 0 {
        return (xb, yb);
    }
    let wf = w0 as f64;
    let hf = h0 as f64;
    let curl_px = params.curl_px as f64;

    match params.curl_side.as_str() {
        "spine" => {
            let depth = ((curl_px / wf) * 2.0).min(0.15);
            let spine_cx = params.bend_spine_cx as f64;
            let spine_w = (params.bend_spine_w as f64).max(1.0);
            let t = (xb - spine_cx) / spine_w;
            let g = (-t * t).exp();
            let xs = spine_cx + (xb - spine_cx) * (1.0 - depth * g);
            (xs, yb)
        }
        "ripple" => {
            let fx = params.bend_ripple_fx as f64;
            let fy = params.bend_ripple_fy as f64;
            let ph = params.bend_ripple_ph as f64;
            let ox = curl_px * 0.5 * (std::f64::consts::TAU * fx * yb / hf + ph).sin();
            let oy = curl_px * 0.3 * (std::f64::consts::TAU * fy * xb / wf + ph).cos();
            (xb + ox, yb + oy)
        }
        "right" => {
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * yb / hf).sin();
            let edge = xb / wf;
            let shift = curl_px * edge * edge * wave;
            (xb - shift, yb)
        }
        "top" => {
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * xb / wf).sin();
            let edge = 1.0 - yb / hf;
            let shift = curl_px * edge * edge * wave;
            (xb, yb + shift)
        }
        "bottom" => {
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * xb / wf).sin();
            let edge = yb / hf;
            let shift = curl_px * edge * edge * wave;
            (xb, yb - shift)
        }
        _ => {
            // "left" (default edge curl)
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * yb / hf).sin();
            let edge = 1.0 - xb / wf;
            let shift = curl_px * edge * edge * wave;
            (xb + shift, yb)
        }
    }
}

/// Exact `f64` forward page-bend map: given a source coordinate `(xs, ys)` in
/// working-image space, returns `(xb, yb)` in bent-image space such that
/// `bend_backward(xb, yb, w0, h0, params) == (xs, ys)`.
///
/// - `"left"`, `"right"`, `"top"`, `"bottom"`: exact closed-form Citardauq
///   quadratic solution.
/// - `"spine"`: 1D Newton-Raphson (slope >= 0.85 everywhere, <= 8 iters,
///   `< 1e-11 px`).
/// - `"ripple"`: 2D Newton-Raphson with analytic 2x2 Jacobian (<= 12 iters,
///   `< 1e-10 px`).
pub fn bend_forward_exact(
    xs: f64,
    ys: f64,
    w0: u32,
    h0: u32,
    params: &DegradeParams,
) -> (f64, f64) {
    if params.curl_px <= 0.25 || w0 == 0 || h0 == 0 {
        return (xs, ys);
    }
    let wf = w0 as f64;
    let hf = h0 as f64;
    let curl_px = params.curl_px as f64;

    match params.curl_side.as_str() {
        "spine" => {
            let depth = ((curl_px / wf) * 2.0).min(0.15);
            let spine_cx = params.bend_spine_cx as f64;
            let spine_w = (params.bend_spine_w as f64).max(1.0);
            let u_s = (xs - spine_cx) / spine_w;
            let mut u = u_s;
            for _ in 0..8 {
                let e = (-u * u).exp();
                let f = u * (1.0 - depth * e) - u_s;
                if (f * spine_w).abs() < 1e-11 {
                    break;
                }
                let df = 1.0 - depth * e * (1.0 - 2.0 * u * u);
                u -= f / df;
            }
            (spine_cx + u * spine_w, ys)
        }
        "ripple" => {
            let fx = params.bend_ripple_fx as f64;
            let fy = params.bend_ripple_fy as f64;
            let ph = params.bend_ripple_ph as f64;
            let ax = 0.5 * curl_px;
            let ay = 0.3 * curl_px;
            let kx = std::f64::consts::TAU * fx / hf;
            let ky = std::f64::consts::TAU * fy / wf;
            let mut xb = xs;
            let mut yb = ys;
            for _ in 0..12 {
                let ang_x = kx * yb + ph;
                let ang_y = ky * xb + ph;
                let f0 = xb + ax * ang_x.sin() - xs;
                let f1 = yb + ay * ang_y.cos() - ys;
                if f0.abs().max(f1.abs()) < 1e-10 {
                    break;
                }
                let j01 = ax * kx * ang_x.cos();
                let j10 = -ay * ky * ang_y.sin();
                let det = 1.0 - j01 * j10;
                if det.abs() < 1e-12 {
                    break;
                }
                xb -= (f0 - j01 * f1) / det;
                yb -= (-j10 * f0 + f1) / det;
            }
            (xb, yb)
        }
        "right" => {
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * ys / hf).sin();
            let c = curl_px * wave;
            let c_rhs = xs;
            let disc = (wf * wf - 4.0 * c * c_rhs).max(0.0).sqrt();
            let e = 2.0 * c_rhs / (wf + disc);
            (wf * e, ys)
        }
        "top" => {
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * xs / wf).sin();
            let c = curl_px * wave;
            let c_rhs = hf - ys;
            let disc = (hf * hf - 4.0 * c * c_rhs).max(0.0).sqrt();
            let e = 2.0 * c_rhs / (hf + disc);
            (xs, hf * (1.0 - e))
        }
        "bottom" => {
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * xs / wf).sin();
            let c = curl_px * wave;
            let c_rhs = ys;
            let disc = (hf * hf - 4.0 * c * c_rhs).max(0.0).sqrt();
            let e = 2.0 * c_rhs / (hf + disc);
            (xs, hf * e)
        }
        _ => {
            // "left"
            let wave = 0.6 + 0.4 * (std::f64::consts::PI * ys / hf).sin();
            let c = curl_px * wave;
            let c_rhs = wf - xs;
            let disc = (wf * wf - 4.0 * c * c_rhs).max(0.0).sqrt();
            let e = 2.0 * c_rhs / (wf + disc);
            (wf * (1.0 - e), ys)
        }
    }
}

/// Precomputed geometric transform context for fast per-pixel forward/backward
/// exact reprojection (hoists homography solve/invert and rotation trig).
pub(crate) struct ReprojectContext<'a> {
    pub(crate) params: &'a DegradeParams,
    pub(crate) w0: u32,
    pub(crate) h0: u32,
    pub(crate) fw: u32,
    pub(crate) fh: u32,
    pub(crate) w_rot: u32,
    pub(crate) h_rot: u32,
    pub(crate) rot_active: bool,
    pub(crate) snt: f64,
    pub(crate) cst: f64,
    pub(crate) h_fwd: Option<[f64; 9]>,
    pub(crate) h_inv: Option<[f64; 9]>,
}

impl<'a> ReprojectContext<'a> {
    pub(crate) fn new(params: &'a DegradeParams) -> Self {
        let (w0, h0) = working_dims(params);
        let has_persp = params
            .corner_offsets
            .iter()
            .any(|o| o[0].abs() > 1e-9 || o[1].abs() > 1e-9);
        let (h_fwd, h_inv) = if has_persp && w0 > 0 && h0 > 0 {
            let wf = w0 as f64;
            let hf = h0 as f64;
            let src_c = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];
            let mut dst_c = [(0.0f64, 0.0f64); 4];
            for i in 0..4 {
                dst_c[i] = (
                    src_c[i].0 + params.corner_offsets[i][0] as f64,
                    src_c[i].1 + params.corner_offsets[i][1] as f64,
                );
            }
            let fwd = solve_homography(src_c, dst_c);
            let inv = fwd.and_then(invert_3x3);
            (fwd, inv)
        } else {
            (None, None)
        };

        let (fw, fh) = framed_dims(
            w0,
            h0,
            &params.frame_mode,
            params.frame_pad_px,
            params.frame_cut_px,
        );
        let rot_active = params.rotation_deg.abs() > 0.05 && w0 > 4 && h0 > 4;
        let (w_rot, h_rot, snt, cst) = if rot_active {
            // Compute integer canvas dims with the exact f32 formula from lib.rs.
            let th32 = params.rotation_deg.to_radians();
            let (sn32, cs32) = (th32.sin().abs(), th32.cos().abs());
            let w2 = ((fw as f32) * cs32 + (fh as f32) * sn32).ceil().max(1.0) as u32;
            let h2 = ((fw as f32) * sn32 + (fh as f32) * cs32).ceil().max(1.0) as u32;
            let th = (params.rotation_deg as f64).to_radians();
            (w2, h2, th.sin(), th.cos())
        } else {
            (fw, fh, 0.0, 1.0)
        };

        Self {
            params,
            w0,
            h0,
            fw,
            fh,
            w_rot,
            h_rot,
            rot_active,
            snt,
            cst,
            h_fwd,
            h_inv,
        }
    }

    #[inline]
    pub(crate) fn forward(&self, x: f64, y: f64) -> (f64, f64) {
        // 1. Exact non-linear page bend (src -> bent).
        let (mut px, mut py) = bend_forward_exact(x, y, self.w0, self.h0, self.params);

        // 2. Perspective homography (bent -> persp).
        if let Some(ref h_fwd) = self.h_fwd {
            (px, py) = apply_h(h_fwd, px, py);
        }

        // 3. Photo framing: inset pad or tight crop.
        if self.params.frame_mode == "inset" {
            px += self.params.frame_pad_px[0] as f64;
            py += self.params.frame_pad_px[1] as f64;
        } else {
            px -= self.params.frame_cut_px[0] as f64;
            py -= self.params.frame_cut_px[1] as f64;
        }

        // 4. Canvas-expanding page rotation.
        if self.rot_active {
            let (cx0, cy0) = (self.fw as f64 / 2.0, self.fh as f64 / 2.0);
            let (cx1, cy1) = (self.w_rot as f64 / 2.0, self.h_rot as f64 / 2.0);
            let u = px - cx0;
            let v = py - cy0;
            px = cx1 + u * self.cst - v * self.snt;
            py = cy1 + u * self.snt + v * self.cst;
        }

        // 5. Output capture downscale.
        if self.params.rot_scale_fused {
            let sx = if self.params.scale_x > 0.0 {
                self.params.scale_x as f64
            } else {
                1.0
            };
            let sy = if self.params.scale_y > 0.0 {
                self.params.scale_y as f64
            } else {
                1.0
            };
            (px * sx, py * sy)
        } else if self.params.width > 0
            && self.params.height > 0
            && (self.params.width != self.w_rot || self.params.height != self.h_rot)
        {
            let rx = self.params.width as f64 / self.w_rot as f64;
            let ry = self.params.height as f64 / self.h_rot as f64;
            ((px + 0.5) * rx - 0.5, (py + 0.5) * ry - 0.5)
        } else {
            (px, py)
        }
    }

    #[inline]
    pub(crate) fn undo_downscale(&self, x_out: f64, y_out: f64) -> (f64, f64) {
        if self.params.rot_scale_fused {
            let sx = if self.params.scale_x > 0.0 {
                self.params.scale_x as f64
            } else {
                1.0
            };
            let sy = if self.params.scale_y > 0.0 {
                self.params.scale_y as f64
            } else {
                1.0
            };
            (x_out / sx, y_out / sy)
        } else if self.params.width > 0
            && self.params.height > 0
            && (self.params.width != self.w_rot || self.params.height != self.h_rot)
        {
            let inv_rx = self.w_rot as f64 / self.params.width as f64;
            let inv_ry = self.h_rot as f64 / self.params.height as f64;
            ((x_out + 0.5) * inv_rx - 0.5, (y_out + 0.5) * inv_ry - 0.5)
        } else {
            (x_out, y_out)
        }
    }

    /// Map an output pixel `(x_out, y_out)` back to the coordinate space where
    /// Stage 5 lighting ran: `(w_rot, h_rot)` on CPU, or `(fw, fh)` on fused GPU.
    #[inline]
    pub(crate) fn to_lighting_coords(&self, x_out: f64, y_out: f64) -> (f64, f64) {
        let (xr, yr) = self.undo_downscale(x_out, y_out);
        if self.params.rot_scale_fused && self.rot_active {
            let (cx0, cy0) = (self.fw as f64 / 2.0, self.fh as f64 / 2.0);
            let (cx1, cy1) = (self.w_rot as f64 / 2.0, self.h_rot as f64 / 2.0);
            let dx = xr - cx1;
            let dy = yr - cy1;
            (
                cx0 + dx * self.cst + dy * self.snt,
                cy0 - dx * self.snt + dy * self.cst,
            )
        } else {
            (xr, yr)
        }
    }

    #[inline]
    pub(crate) fn lighting_dims(&self) -> (u32, u32) {
        if self.params.rot_scale_fused {
            (self.fw, self.fh)
        } else {
            (self.w_rot, self.h_rot)
        }
    }

    #[inline]
    pub(crate) fn unproject(&self, x_out: f64, y_out: f64) -> Option<(f64, f64)> {
        const EPS: f64 = 1e-6;
        // 1. Undo downscale -> (x_rot, y_rot).
        let (x_rot, y_rot) = self.undo_downscale(x_out, y_out);

        // 2. Undo rotation -> (x_frame, y_frame).
        let (x_frame, y_frame) = if self.rot_active {
            let (cx0, cy0) = (self.fw as f64 / 2.0, self.fh as f64 / 2.0);
            let (cx1, cy1) = (self.w_rot as f64 / 2.0, self.h_rot as f64 / 2.0);
            let dx = x_rot - cx1;
            let dy = y_rot - cy1;
            let fx = cx0 + dx * self.cst + dy * self.snt;
            let fy = cy0 - dx * self.snt + dy * self.cst;
            if fx < -EPS || fy < -EPS || fx > self.fw as f64 + EPS || fy > self.fh as f64 + EPS {
                return None;
            }
            (fx, fy)
        } else {
            (x_rot, y_rot)
        };

        // 3. Undo framing -> (x_persp, y_persp).
        let (x_persp, y_persp) = if self.params.frame_mode == "inset" {
            let px = x_frame - self.params.frame_pad_px[0] as f64;
            let py = y_frame - self.params.frame_pad_px[1] as f64;
            if px < -EPS || py < -EPS || px > self.w0 as f64 + EPS || py > self.h0 as f64 + EPS {
                return None;
            }
            (px, py)
        } else {
            (
                x_frame + self.params.frame_cut_px[0] as f64,
                y_frame + self.params.frame_cut_px[1] as f64,
            )
        };

        // 4. Undo perspective -> (x_bend, y_bend).
        let (x_bend, y_bend) = if let Some(ref h_inv) = self.h_inv {
            let (bx, by) = apply_h(h_inv, x_persp, y_persp);
            if bx < -EPS || by < -EPS || bx > self.w0 as f64 + EPS || by > self.h0 as f64 + EPS {
                return None;
            }
            (bx, by)
        } else {
            (x_persp, y_persp)
        };

        // 5. Undo non-linear page bend -> (x_src, y_src).
        let (x_src, y_src) = bend_backward(x_bend, y_bend, self.w0, self.h0, self.params);
        if x_src < -EPS
            || y_src < -EPS
            || x_src > self.w0 as f64 + EPS
            || y_src > self.h0 as f64 + EPS
        {
            return None;
        }
        Some((x_src, y_src))
    }
}

/// Forward-project a point `(x, y)` from working-image space through the full
/// geometric pipeline: non-linear page bend → perspective homography → framing
/// → rotation → exact output downscale.
pub fn transform_point_exact(x: f64, y: f64, params: &DegradeParams) -> (f64, f64) {
    ReprojectContext::new(params).forward(x, y)
}

/// Backward-project a point `(x_out, y_out)` from the final degraded image
/// space back to working-image coordinates `(x_src, y_src)`.
///
/// Returns `None` if the output pixel lies outside the original document page
/// (e.g. rotated desk corners, inset desk padding, or perspective bleed).
pub fn unproject_point_exact(x_out: f64, y_out: f64, params: &DegradeParams) -> Option<(f64, f64)> {
    ReprojectContext::new(params).unproject(x_out, y_out)
}

/// Forward-project an axis-aligned bounding box `[x0, y0, x1, y1]` through
/// [`transform_point_exact`] by sampling `samples_per_edge` segments along each
/// of the 4 perimeter edges, returning the tight AABB `[min_x, min_y, max_x, max_y]`.
pub fn transform_bbox_exact(
    bbox: [f64; 4],
    params: &DegradeParams,
    samples_per_edge: usize,
) -> [f64; 4] {
    let ctx = ReprojectContext::new(params);
    let [x0, y0, x1, y1] = bbox;
    let n = samples_per_edge.max(1);
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut visit = |sx: f64, sy: f64| {
        let (tx, ty) = ctx.forward(sx, sy);
        min_x = min_x.min(tx);
        min_y = min_y.min(ty);
        max_x = max_x.max(tx);
        max_y = max_y.max(ty);
    };

    for i in 0..=n {
        let t = i as f64 / n as f64;
        visit(x0 + (x1 - x0) * t, y0);
        visit(x0 + (x1 - x0) * t, y1);
        visit(x0, y0 + (y1 - y0) * t);
        visit(x1, y0 + (y1 - y0) * t);
    }

    [min_x, min_y, max_x, max_y]
}

/// Forward-project a closed polygon through [`transform_point_exact`],
/// subdividing edges longer than `max_segment_px` (in source working-image
/// pixels) so curved page-bend edges are faithfully captured.
pub fn transform_polygon_exact(
    poly: &[(f64, f64)],
    params: &DegradeParams,
    max_segment_px: f64,
) -> Vec<(f64, f64)> {
    if poly.is_empty() {
        return Vec::new();
    }
    let ctx = ReprojectContext::new(params);
    if poly.len() == 1 {
        return vec![ctx.forward(poly[0].0, poly[0].1)];
    }
    let mut out = Vec::with_capacity(poly.len());
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        let len = (x1 - x0).hypot(y1 - y0);
        let steps = if max_segment_px > 0.0 {
            (len / max_segment_px).ceil().max(1.0) as usize
        } else {
            1
        };
        for k in 0..steps {
            let t = k as f64 / steps as f64;
            out.push(ctx.forward(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t));
        }
    }
    out
}

/// Forward-project a point `(x, y)` from working-image space through the linear
/// / projective geometric stages recorded in `params`:
/// perspective homography → framing pad/cut → rotation → output downscale.
pub fn transform_point_linear(x: f64, y: f64, params: &DegradeParams) -> (f64, f64) {
    let (w0, h0) = working_dims(params);
    let mut px = x;
    let mut py = y;

    // 1. Perspective homography (forward: src -> dst).
    let has_persp = params
        .corner_offsets
        .iter()
        .any(|o| o[0].abs() > 1e-9 || o[1].abs() > 1e-9);
    if has_persp && w0 > 0 && h0 > 0 {
        let wf = w0 as f64;
        let hf = h0 as f64;
        let src_c = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];
        let mut dst_c = [(0.0f64, 0.0f64); 4];
        for i in 0..4 {
            dst_c[i] = (
                src_c[i].0 + params.corner_offsets[i][0] as f64,
                src_c[i].1 + params.corner_offsets[i][1] as f64,
            );
        }
        if let Some(h_fwd) = solve_homography(src_c, dst_c) {
            (px, py) = apply_h(&h_fwd, px, py);
        }
    }

    // 2. Photo framing: inset pad or tight crop.
    let (fw, fh) = framed_dims(
        w0,
        h0,
        &params.frame_mode,
        params.frame_pad_px,
        params.frame_cut_px,
    );
    if params.frame_mode == "inset" {
        px += params.frame_pad_px[0] as f64;
        py += params.frame_pad_px[1] as f64;
    } else {
        px -= params.frame_cut_px[0] as f64;
        py -= params.frame_cut_px[1] as f64;
    }

    // 3. Page rotation (canvas-expanding, forward inverse of the bilinear sampler).
    if params.rotation_deg.abs() > 0.05 && w0 > 4 && h0 > 4 {
        let th = (params.rotation_deg as f64).to_radians();
        let (sn_abs, cs_abs) = (th.sin().abs(), th.cos().abs());
        let w2 = ((fw as f64) * cs_abs + (fh as f64) * sn_abs)
            .ceil()
            .max(1.0);
        let h2 = ((fw as f64) * sn_abs + (fh as f64) * cs_abs)
            .ceil()
            .max(1.0);
        let (cx0, cy0) = (fw as f64 / 2.0, fh as f64 / 2.0);
        let (cx1, cy1) = (w2 / 2.0, h2 / 2.0);
        let (snt, cst) = (th.sin(), th.cos());
        let u = px - cx0;
        let v = py - cy0;
        px = cx1 + u * cst - v * snt;
        py = cy1 + u * snt + v * cst;
    }

    // 4. Output capture downscale.
    let sx = if params.scale_x > 0.0 {
        params.scale_x as f64
    } else {
        1.0
    };
    let sy = if params.scale_y > 0.0 {
        params.scale_y as f64
    } else {
        1.0
    };
    (px * sx, py * sy)
}

/// Transform an axis-aligned bounding box `[x0, y0, x1, y1]` by sampling
/// `samples_per_edge` segments along each of the 4 perimeter edges and
/// returning the tight transformed AABB `[min_x, min_y, max_x, max_y]`.
pub fn transform_bbox(bbox: [f64; 4], params: &DegradeParams, samples_per_edge: usize) -> [f64; 4] {
    let [x0, y0, x1, y1] = bbox;
    let n = samples_per_edge.max(1);
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut visit = |sx: f64, sy: f64| {
        let (tx, ty) = transform_point_linear(sx, sy, params);
        min_x = min_x.min(tx);
        min_y = min_y.min(ty);
        max_x = max_x.max(tx);
        max_y = max_y.max(ty);
    };

    for i in 0..=n {
        let t = i as f64 / n as f64;
        // Top edge: (x0, y0) -> (x1, y0)
        visit(x0 + (x1 - x0) * t, y0);
        // Bottom edge: (x0, y1) -> (x1, y1)
        visit(x0 + (x1 - x0) * t, y1);
        // Left edge: (x0, y0) -> (x0, y1)
        visit(x0, y0 + (y1 - y0) * t);
        // Right edge: (x1, y0) -> (x1, y1)
        visit(x1, y0 + (y1 - y0) * t);
    }

    [min_x, min_y, max_x, max_y]
}

/// Transform a polygon's vertices through [`transform_point_linear`].
pub fn transform_polygon(poly: &[(f64, f64)], params: &DegradeParams) -> Vec<(f64, f64)> {
    poly.iter()
        .map(|&(x, y)| transform_point_linear(x, y, params))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CaptureParams, InkParams, PaperParams, WearParams};

    fn identity_params(w: u32, h: u32) -> DegradeParams {
        DegradeParams {
            input: String::new(),
            output: String::new(),
            file_seed: 0,
            strength: 0.0,
            backend: "cpu".to_string(),
            preset: "full".to_string(),
            fast_banks: false,
            width: w,
            height: h,
            train_width: 0,
            orig_width: w,
            orig_height: h,
            src_width: w,
            src_height: h,
            rescale_k: 1.0,
            paper_grain_sigma: 0.0,
            paper_speckle_prob: 0.0,
            curl_side: "left".to_string(),
            curl_px: 0.0,
            curl_shade: 0.0,
            bend_spine_cx: 0.0,
            bend_spine_w: 0.0,
            bend_ripple_fx: 0.0,
            bend_ripple_fy: 0.0,
            bend_ripple_ph: 0.0,
            perspective_max_frac: 0.0,
            corner_offsets: [[0.0; 2]; 4],
            warp_fit_scale: 1.0,
            warp_out_px: 0.0,
            rotation_deg: 0.0,
            rot_scale_fused: false,
            frame_mode: "inset".to_string(),
            frame_pad_px: [0; 4],
            frame_cut_px: [0; 4],
            frame_content_bbox: [0; 4],
            frame_quiet_px: [0; 4],
            frame_safe_px: 3,
            dist_k1: 0.0,
            dist_k2: 0.0,
            dist_fit_scale: 1.0,
            focus_mode: "uniform".to_string(),
            focus_y: 0.5,
            focus_h: 0.2,
            tilt_sigma: 0.0,
            blur_sigma: 0.0,
            glare_center: [0.0, 0.0],
            glare_radius: 0.0,
            glare_alpha: 0.0,
            shadow_angle_deg: 0.0,
            shadow_strength: 0.0,
            vignette_strength: 0.0,
            ca_px: 0.0,
            ink: InkParams {
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
            },
            paper: PaperParams {
                enabled: false,
                yellowing: 0.0,
                mottle: 0.0,
                edge: 0.0,
                foxing_spots: 0,
                foxing_amount: 0.0,
                fiber_strokes: 0,
                fiber_amount: 0.0,
                cockle: 0.0,
                water_present: false,
                water_x: 0.0,
                water_y: 0.0,
                water_radius: 0.0,
                water_amount: 0.0,
            },
            wear: WearParams {
                enabled: false,
                heavy_zones: [[0.0; 4]; 3],
                heavy_lift: 0.0,
                flake_clusters: 0,
                band_period: 0.0,
                band_amp: 0.0,
                streaks: 0,
                toner_bands: 0,
                misreg: [0, 0],
                eraser_count: 0,
                stain_water: false,
                stain_oil: false,
            },
            capture: CaptureParams {
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
            },
            scale_x: 1.0,
            scale_y: 1.0,
            tear_count: 0,
            stain_count: 0,
            pencil_count: 0,
            fold_corner: "none".to_string(),
            fold_creases: Vec::new(),
            highlight: "none".to_string(),
            noise_sigma: 0.0,
            jpeg_quality: 100,
            codec: "zune".to_string(),
            awb_warm_cool_strength: 0.0,
            usm_strength: 0.0,
            desk_surface: "none".to_string(),
            crumple_strength: 0.0,
            sheen_strength: 0.0,
        }
    }

    #[test]
    fn identity_transform_preserves_points_bbox_and_polygon() {
        let p = identity_params(200, 100);
        let (x, y) = transform_point_linear(37.5, 81.25, &p);
        assert!((x - 37.5).abs() < 1e-9);
        assert!((y - 81.25).abs() < 1e-9);

        let bb = transform_bbox([10.0, 20.0, 150.0, 80.0], &p, 8);
        assert!((bb[0] - 10.0).abs() < 1e-9);
        assert!((bb[1] - 20.0).abs() < 1e-9);
        assert!((bb[2] - 150.0).abs() < 1e-9);
        assert!((bb[3] - 80.0).abs() < 1e-9);

        let poly = [(5.0, 5.0), (50.0, 10.0), (40.0, 60.0)];
        let out = transform_polygon(&poly, &p);
        assert_eq!(out.len(), poly.len());
        for (a, b) in out.iter().zip(poly.iter()) {
            assert!((a.0 - b.0).abs() < 1e-9);
            assert!((a.1 - b.1).abs() < 1e-9);
        }
    }

    #[test]
    fn framing_rotation_and_scale_match_inverse_sampler() {
        let mut p = identity_params(200, 100);
        p.frame_mode = "inset".to_string();
        p.frame_pad_px = [10, 20, 15, 25];
        p.rotation_deg = 8.0;
        p.scale_x = 0.5;
        p.scale_y = 0.75;

        // Framed dims: (225, 145). Center of framed image = (112.5, 72.5).
        // Working point mapping to framed center is (112.5 - 10, 72.5 - 20) = (102.5, 52.5).
        let th = 8.0f64.to_radians();
        let w2 = (225.0 * th.cos().abs() + 145.0 * th.sin().abs()).ceil();
        let h2 = (225.0 * th.sin().abs() + 145.0 * th.cos().abs()).ceil();
        let (cx_out, cy_out) = transform_point_linear(102.5, 52.5, &p);
        assert!((cx_out - (w2 / 2.0) * 0.5).abs() < 1e-9);
        assert!((cy_out - (h2 / 2.0) * 0.75).abs() < 1e-9);

        // Verify an arbitrary point round-trips through the exact inverse rotation
        // formula used in lib.rs.
        let (wx, wy) = (40.0, 30.0);
        let (ox, oy) = transform_point_linear(wx, wy, &p);
        let rx = ox / 0.5;
        let ry = oy / 0.75;
        let dx = rx - w2 / 2.0;
        let dy = ry - h2 / 2.0;
        let fx = 112.5 + dx * th.cos() + dy * th.sin();
        let fy = 72.5 - dx * th.sin() + dy * th.cos();
        assert!((fx - (wx + 10.0)).abs() < 1e-9);
        assert!((fy - (wy + 20.0)).abs() < 1e-9);

        // Tight crop + scale test
        let mut p_tight = identity_params(200, 100);
        p_tight.frame_mode = "tight".to_string();
        p_tight.frame_cut_px = [12, 8, 10, 6];
        p_tight.scale_x = 0.5;
        p_tight.scale_y = 0.5;
        let (tx, ty) = transform_point_linear(52.0, 28.0, &p_tight);
        assert!((tx - 20.0).abs() < 1e-9);
        assert!((ty - 10.0).abs() < 1e-9);
    }

    #[test]
    fn perspective_and_bbox_sampling() {
        let mut p = identity_params(100, 100);
        // Pure translation via corner offsets: shift all 4 corners by (+5, +10).
        p.corner_offsets = [[5.0, 10.0]; 4];
        let (x, y) = transform_point_linear(20.0, 30.0, &p);
        assert!((x - 25.0).abs() < 1e-6);
        assert!((y - 40.0).abs() < 1e-6);

        let bb = transform_bbox([10.0, 10.0, 50.0, 60.0], &p, 4);
        assert!((bb[0] - 15.0).abs() < 1e-6);
        assert!((bb[1] - 20.0).abs() < 1e-6);
        assert!((bb[2] - 55.0).abs() < 1e-6);
        assert!((bb[3] - 70.0).abs() < 1e-6);
    }

    #[test]
    fn end_to_end_reprojection_aligns_with_degraded_ink() {
        use crate::{content_bbox_thresh, degrade_image, DegradeConfig, StageToggles};
        use image::{Rgb, RgbImage};

        let (w, h) = (320u32, 240u32);
        let mut img = RgbImage::from_pixel(w, h, Rgb([250, 250, 248]));
        let src_box = [90u32, 70u32, 230u32, 170u32];
        for y in src_box[1]..=src_box[3] {
            for x in src_box[0]..=src_box[2] {
                img.put_pixel(x, y, Rgb([15, 15, 18]));
            }
        }

        let mut cfg = DegradeConfig::default().with_seed(101).with_strength(0.45);
        cfg.scale_min = 0.75;
        cfg.toggles = StageToggles {
            skip_bend: true,
            ..StageToggles::reduced()
        };
        let res = degrade_image(img, &cfg).expect("degrade_image must succeed");
        let p = &res.params;

        let proj = transform_bbox(
            [
                src_box[0] as f64,
                src_box[1] as f64,
                src_box[2] as f64,
                src_box[3] as f64,
            ],
            p,
            16,
        );
        assert!(proj[0] >= 0.0 && proj[1] >= 0.0);
        assert!(proj[2] <= p.width as f64 && proj[3] <= p.height as f64);

        // Locate dark pixels inside the interior of the degraded image (excluding
        // dark desk fill around the page border by checking the center of the
        // reprojected box is actually dark ink, and its outer neighborhood is
        // lighter paper).
        let cx = ((proj[0] + proj[2]) * 0.5).round() as u32;
        let cy = ((proj[1] + proj[3]) * 0.5).round() as u32;
        let center_px = res.image.get_pixel(cx, cy).0;
        let center_luma = (center_px[0] as f32 + center_px[1] as f32 + center_px[2] as f32) / 3.0;
        assert!(
            center_luma < 140.0,
            "reprojected center ({cx},{cy}) should land on dark ink, got luma={center_luma}"
        );

        // Also check that sampling the degraded buffer strictly inside the
        // reprojected box finds ink covering the expected span.
        let buf: Vec<f32> = res.image.as_raw().iter().map(|&v| v as f32).collect();
        let x0 = (proj[0].floor().max(0.0) as u32).min(p.width.saturating_sub(1));
        let y0 = (proj[1].floor().max(0.0) as u32).min(p.height.saturating_sub(1));
        let x1 = (proj[2].ceil().max(0.0) as u32).min(p.width);
        let y1 = (proj[3].ceil().max(0.0) as u32).min(p.height);
        let crop_w = x1.saturating_sub(x0).max(1);
        let crop_h = y1.saturating_sub(y0).max(1);
        let mut crop = vec![255.0f32; (crop_w * crop_h * 3) as usize];
        for y in 0..crop_h {
            for x in 0..crop_w {
                let si = (((y0 + y) * p.width + (x0 + x)) * 3) as usize;
                let di = ((y * crop_w + x) * 3) as usize;
                crop[di..di + 3].copy_from_slice(&buf[si..si + 3]);
            }
        }
        let found = content_bbox_thresh(&crop, crop_w, crop_h, 150.0)
            .expect("degraded ink must appear inside reprojected bbox");
        let found_w = found[2] - found[0] + 1;
        let found_h = found[3] - found[1] + 1;
        assert!(
            found_w as f64 >= crop_w as f64 * 0.70 && found_h as f64 >= crop_h as f64 * 0.70,
            "ink span ({found_w}x{found_h}) should fill >=70% of reprojected box ({crop_w}x{crop_h})"
        );
    }

    #[test]
    fn exact_round_trip_all_six_bend_modes() {
        let (w0, h0) = (400u32, 300u32);
        let modes = ["left", "right", "top", "bottom", "spine", "ripple"];
        for (mi, &mode) in modes.iter().enumerate() {
            let mut p = identity_params(w0, h0);
            p.strength = 1.0;
            p.curl_side = mode.to_string();
            p.curl_px = w0 as f32 * 0.08;
            p.curl_shade = 0.35;
            p.bend_spine_cx = w0 as f32 * 0.48;
            p.bend_spine_w = w0 as f32 * 0.05;
            p.bend_ripple_fx = 2.2;
            p.bend_ripple_fy = 1.8;
            p.bend_ripple_ph = 1.1;
            p.corner_offsets = [[14.0, 9.0], [-12.0, 11.0], [-10.0, -13.0], [15.0, -8.0]];
            if mi % 2 == 0 {
                p.frame_mode = "inset".to_string();
                p.frame_pad_px = [12, 10, 14, 11];
            } else {
                p.frame_mode = "tight".to_string();
                p.frame_cut_px = [6, 5, 7, 4];
            }
            p.rotation_deg = -7.5;
            let (fw, fh) = framed_dims(w0, h0, &p.frame_mode, p.frame_pad_px, p.frame_cut_px);
            let th = p.rotation_deg.to_radians();
            let w_rot = ((fw as f32) * th.cos().abs() + (fh as f32) * th.sin().abs())
                .ceil()
                .max(1.0) as u32;
            let h_rot = ((fw as f32) * th.sin().abs() + (fh as f32) * th.cos().abs())
                .ceil()
                .max(1.0) as u32;
            p.scale_x = 0.67;
            p.scale_y = 0.64;
            p.width = ((w_rot as f32 * p.scale_x) as u32).max(1);
            p.height = ((h_rot as f32 * p.scale_y) as u32).max(1);

            let mut max_err = 0.0f64;
            for gy in 1..=7 {
                for gx in 1..=7 {
                    let x = 35.0 + (w0 as f64 - 70.0) * (gx as f64 / 8.0);
                    let y = 30.0 + (h0 as f64 - 60.0) * (gy as f64 / 8.0);
                    let (ox, oy) = transform_point_exact(x, y, &p);
                    let (rx, ry) = unproject_point_exact(ox, oy, &p).unwrap_or_else(|| {
                        panic!("mode={mode} point ({x},{y}) -> ({ox},{oy}) unprojected to None")
                    });
                    let err = (rx - x).hypot(ry - y);
                    max_err = max_err.max(err);
                }
            }
            assert!(
                max_err < 1e-9,
                "mode={mode}: round-trip error {max_err:.3e} px exceeds 1e-9"
            );

            // Also verify transform_bbox_exact and transform_polygon_exact run cleanly.
            let bb = transform_bbox_exact([50.0, 40.0, 350.0, 260.0], &p, 16);
            assert!(bb[2] > bb[0] && bb[3] > bb[1]);
            let poly = [(50.0, 40.0), (350.0, 40.0), (350.0, 260.0), (50.0, 260.0)];
            let tpoly = transform_polygon_exact(&poly, &p, 50.0);
            assert!(tpoly.len() > poly.len());
        }
    }

    #[test]
    fn rendered_ink_centroid_error_under_quarter_pixel() {
        use crate::{degrade_image, DegradeConfig, StageToggles};
        use image::{Rgb, RgbImage};

        let (w, h) = (480u32, 360u32);
        let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
        // Place a 3x3 grid of compact symmetric 5x5 black dots (dx^2+dy^2 <= 5)
        // in the interior so non-linear bend curvature (Jensen's inequality)
        // doesn't shift the dot's centroid away from its geometric center.
        let centers: Vec<(u32, u32)> = [120u32, 240, 360]
            .into_iter()
            .flat_map(|cx| [90u32, 180, 270].into_iter().map(move |cy| (cx, cy)))
            .collect();
        for &(cx, cy) in &centers {
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    if dx * dx + dy * dy <= 5 {
                        img.put_pixel(
                            (cx as i32 + dx) as u32,
                            (cy as i32 + dy) as u32,
                            Rgb([0, 0, 0]),
                        );
                    }
                }
            }
        }

        let mut seen_spine = false;
        let mut seen_ripple = false;
        let mut seen_edge = false;
        let mut global_max_err = 0.0f64;

        for seed in 0..40u64 {
            if seen_spine && seen_ripple && seen_edge {
                break;
            }
            let mut cfg = DegradeConfig::default().with_seed(seed).with_strength(0.55);
            cfg.scale_min = 0.65;
            cfg.jpeg_quality = Some(100);
            cfg.toggles = StageToggles::reduced();
            let res = degrade_image(img.clone(), &cfg).expect("degrade_image must succeed");
            let p = &res.params;

            let is_spine = p.curl_side == "spine";
            let is_ripple = p.curl_side == "ripple";
            let is_edge = !is_spine && !is_ripple;
            if (is_spine && seen_spine) || (is_ripple && seen_ripple) || (is_edge && seen_edge) {
                continue;
            }

            let mut errs = Vec::new();
            for &(cx, cy) in &centers {
                let (px, py) = transform_point_exact(cx as f64, cy as f64, p);
                let ix = px.round() as i32;
                let iy = py.round() as i32;
                let rad = 6i32;
                if ix - rad < 2
                    || iy - rad < 2
                    || ix + rad >= p.width as i32 - 2
                    || iy + rad >= p.height as i32 - 2
                {
                    continue;
                }
                // Measure local background luma around the perimeter of the window.
                let mut bg_sum = 0.0f64;
                let mut bg_n = 0usize;
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        if dx.abs() == rad || dy.abs() == rad {
                            let rgb = res.image.get_pixel((ix + dx) as u32, (iy + dy) as u32).0;
                            bg_sum += (rgb[0] as f64 + rgb[1] as f64 + rgb[2] as f64) / 3.0;
                            bg_n += 1;
                        }
                    }
                }
                let bg = bg_sum / bg_n as f64;
                if bg < 160.0 {
                    // Occluded by a dark artifact (e.g. sticky note/stain/fold).
                    continue;
                }
                // Use a tight core-ink threshold so faint wear/misregistration
                // halos (-18 luma) don't bias the geometric centroid.
                let thresh = bg * 0.50;
                let (mut wx, mut wy, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
                let mut dark_count = 0usize;
                for dy in -(rad - 1)..=(rad - 1) {
                    for dx in -(rad - 1)..=(rad - 1) {
                        let gx = ix + dx;
                        let gy = iy + dy;
                        let rgb = res.image.get_pixel(gx as u32, gy as u32).0;
                        let luma = (rgb[0] as f64 + rgb[1] as f64 + rgb[2] as f64) / 3.0;
                        if luma < thresh {
                            let weight = (thresh - luma).max(0.0);
                            wx += gx as f64 * weight;
                            wy += gy as f64 * weight;
                            wsum += weight;
                            dark_count += 1;
                        }
                    }
                }
                if !(3..=60).contains(&dark_count) || wsum <= 0.0 {
                    continue;
                }
                let (mx, my) = (wx / wsum, wy / wsum);
                let err = (mx - px).hypot(my - py);
                errs.push(err);
            }
            if errs.len() >= 5 {
                errs.sort_by(|a, b| a.total_cmp(b));
                let median_err = errs[errs.len() / 2];
                // Trim at most 1 outlier dot that may be partially hit by a random
                // ink-dropout cluster, eraser smudge, or specular sheen edge.
                let trimmed_max = errs[errs.len() - 2];
                assert!(
                    median_err < 0.20,
                    "seed={seed} mode={} median centroid err={median_err:.4} px >= 0.20 px",
                    p.curl_side
                );
                assert!(
                    trimmed_max < 0.25,
                    "seed={seed} mode={} trimmed max centroid err={trimmed_max:.4} px >= 0.25 px",
                    p.curl_side
                );
                global_max_err = global_max_err.max(trimmed_max);
                if is_spine {
                    seen_spine = true;
                } else if is_ripple {
                    seen_ripple = true;
                } else {
                    seen_edge = true;
                }
            }
        }
        assert!(
            seen_spine && seen_ripple && seen_edge,
            "expected to verify all 3 bend families (spine={seen_spine}, ripple={seen_ripple}, edge={seen_edge})"
        );
        assert!(
            global_max_err < 0.25,
            "global max centroid err={global_max_err:.4} px >= 0.25 px"
        );
    }
}
