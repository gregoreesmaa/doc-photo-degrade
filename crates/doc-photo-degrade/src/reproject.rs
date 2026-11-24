//! Ground-truth coordinate reprojection utilities built directly from
//! [`DegradeParams`].
//!
//! Maps points, axis-aligned bounding boxes, and polygons from the pre-degrade
//! working-image coordinate space (after `max_width` / `train_width` pre-scaling)
//! into the final degraded output image space through the projective / affine
//! geometric pipeline:
//! 1. Perspective homography (`corner_offsets`)
//! 2. Photo framing (`frame_mode`: `"inset"` pad or `"tight"` cut)
//! 3. Canvas-expanding page rotation (`rotation_deg`)
//! 4. Output capture downscale (`scale_x`, `scale_y`)
//!
//! # Exactness Scope
//!
//! [`transform_point_linear`], [`transform_bbox`], and [`transform_polygon`]
//! model the projective + framing + rotation + downscale stages and are
//! **exact when non-linear page bend is disabled** (for example with
//! [`crate::DocumentProfile::OfficeScan`], [`crate::DocumentProfile::PhotocopyXerox`],
//! `StageToggles { skip_bend: true, .. }`, or the CLI `--skip-bend` flag). When
//! page bend is enabled, these functions provide a linear/projective
//! approximation that does not include the local curl/spine/ripple displacement.

use crate::geometry::{apply_h, framed_dims, solve_homography};
use crate::DegradeParams;

/// Derive the working-image `(width, height)` prior to perspective/framing.
///
/// Mirrors the pre-degrade sizing logic in `lib.rs` (`src_width`/`src_height`
/// rescaled to `train_width` when active), falling back to `(params.width,
/// params.height)` if `src_width`/`src_height` are unset.
fn working_dims(params: &DegradeParams) -> (u32, u32) {
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
}
