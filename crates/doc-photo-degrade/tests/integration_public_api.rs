//! End-to-end integration tests for the `doc_photo_degrade` public API.

use doc_photo_degrade::{degrade_image, DegradeConfig, DegradeParams};
use image::{Rgb, RgbImage};

/// Build a small synthetic document page with dark text-like bars inside
/// generous quiet margins.
fn make_synthetic_page(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([250, 250, 248]));
    let ink = Rgb([22, 24, 28]);

    let mx = (w / 6).max(1);
    let my = (h / 6).max(1);
    let mut y = my;
    while y + 6 < h.saturating_sub(my) {
        for yy in y..(y + 4).min(h) {
            for xx in mx..w.saturating_sub(mx) {
                img.put_pixel(xx, yy, ink);
            }
        }
        y += 14;
    }
    img
}

#[test]
fn cpu_determinism_is_bit_exact() {
    let page = make_synthetic_page(240, 320);
    let cfg = DegradeConfig::default().with_seed(12345).with_strength(0.6);

    let a = degrade_image(page.clone(), &cfg).expect("first degrade_image should succeed");
    let b = degrade_image(page, &cfg).expect("second degrade_image should succeed");

    assert_eq!(a.image.dimensions(), b.image.dimensions());
    assert_eq!(
        a.image.as_raw(),
        b.image.as_raw(),
        "identical seed on CPU must produce bit-identical pixels"
    );

    let json_a = serde_json::to_string(&a.params).expect("serialize params A");
    let json_b = serde_json::to_string(&b.params).expect("serialize params B");
    assert_eq!(
        json_a, json_b,
        "identical seed on CPU must produce identical DegradeParams JSON"
    );
    assert_eq!(a.params.backend, "cpu");
    assert_eq!(a.params.file_seed, 12345);
}

#[test]
fn different_seeds_produce_distinct_images_and_params() {
    let page = make_synthetic_page(240, 320);
    let cfg_a = DegradeConfig::default().with_seed(101).with_strength(0.65);
    let cfg_b = DegradeConfig::default().with_seed(202).with_strength(0.65);

    let a = degrade_image(page.clone(), &cfg_a).expect("seed 101 should succeed");
    let b = degrade_image(page, &cfg_b).expect("seed 202 should succeed");

    assert_ne!(
        a.image.as_raw(),
        b.image.as_raw(),
        "different seeds should produce distinct images"
    );
    assert_ne!(a.params.file_seed, b.params.file_seed);
    let json_a = serde_json::to_string(&a.params).expect("serialize params A");
    let json_b = serde_json::to_string(&b.params).expect("serialize params B");
    assert_ne!(
        json_a, json_b,
        "different seeds should produce distinct DegradeParams"
    );
}

#[test]
fn degrade_params_json_round_trip_and_legacy_alias() {
    let page = make_synthetic_page(240, 320);
    let cfg = DegradeConfig::default().with_seed(4242).with_strength(0.5);
    let res = degrade_image(page, &cfg).expect("degrade_image should succeed");

    // Standard JSON round-trip.
    let serialized = serde_json::to_string(&res.params).expect("serialize DegradeParams");
    assert!(
        serialized.contains("\"frame_content_bbox\""),
        "serialized JSON must contain frame_content_bbox"
    );
    let round_tripped: DegradeParams =
        serde_json::from_str(&serialized).expect("deserialize DegradeParams");
    assert_eq!(round_tripped.file_seed, res.params.file_seed);
    assert_eq!(round_tripped.frame_mode, res.params.frame_mode);
    assert_eq!(
        round_tripped.frame_content_bbox,
        res.params.frame_content_bbox
    );
    assert_eq!(round_tripped.frame_quiet_px, res.params.frame_quiet_px);
    assert_eq!(round_tripped.warp_fit_scale, res.params.warp_fit_scale);
    assert_eq!(round_tripped.rotation_deg, res.params.rotation_deg);
    assert_eq!(round_tripped.jpeg_quality, res.params.jpeg_quality);

    // Legacy `frame_notation_bbox` alias compatibility.
    let legacy_json = serialized.replace("\"frame_content_bbox\"", "\"frame_notation_bbox\"");
    assert!(
        legacy_json.contains("\"frame_notation_bbox\""),
        "legacy JSON fixture must use frame_notation_bbox"
    );
    let from_legacy: DegradeParams = serde_json::from_str(&legacy_json)
        .expect("DegradeParams must deserialize legacy frame_notation_bbox alias");
    assert_eq!(
        from_legacy.frame_content_bbox,
        res.params.frame_content_bbox
    );
}

#[test]
fn content_margin_safety_invariant_and_edge_cases() {
    let page = make_synthetic_page(240, 320);

    // Check across multiple seeds so both "inset" and "tight" framing paths are exercised.
    for seed in 0..8_u64 {
        let cfg = DegradeConfig::default().with_seed(seed).with_strength(0.7);
        let res = degrade_image(page.clone(), &cfg).expect("degrade_image should succeed");
        let p = &res.params;

        assert!(
            p.warp_fit_scale.is_finite() && p.warp_fit_scale > 0.0 && p.warp_fit_scale <= 1.0,
            "seed {seed}: warp_fit_scale out of bounds: {}",
            p.warp_fit_scale
        );
        assert!(
            p.frame_safe_px >= 3,
            "seed {seed}: frame_safe_px should be >= 3"
        );

        if p.frame_mode == "tight" {
            for side in 0..4 {
                let max_allowed = p.frame_quiet_px[side].saturating_sub(p.frame_safe_px);
                assert!(
                    p.frame_cut_px[side] <= max_allowed,
                    "seed {seed} side {side}: cut {} > safe budget {max_allowed}",
                    p.frame_cut_px[side]
                );
            }
        } else {
            assert!(
                p.frame_mode == "inset" || p.frame_mode == "detached",
                "unexpected frame_mode: {}",
                p.frame_mode
            );
            assert_eq!(p.frame_cut_px, [0, 0, 0, 0]);
        }
    }

    // Blank white image: must not panic and must produce a valid output.
    let blank = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
    let blank_cfg = DegradeConfig::default().with_seed(7).with_strength(0.5);
    let blank_res = degrade_image(blank, &blank_cfg).expect("blank image should not fail");
    assert!(blank_res.image.width() > 0 && blank_res.image.height() > 0);
    assert!(blank_res.params.warp_fit_scale.is_finite());

    // Tiny image: must not panic and must produce a valid output.
    let tiny = RgbImage::from_pixel(8, 8, Rgb([240, 240, 240]));
    let tiny_cfg = DegradeConfig::default().with_seed(9).with_strength(0.5);
    let tiny_res = degrade_image(tiny, &tiny_cfg).expect("tiny image should not fail");
    assert!(tiny_res.image.width() > 0 && tiny_res.image.height() > 0);
    assert!(tiny_res.params.warp_fit_scale.is_finite());

    // Invalid config must return an explicit Err rather than panicking.
    let bad_cfg = DegradeConfig {
        scale_min: 0.0,
        ..DegradeConfig::default()
    };
    assert!(degrade_image(make_synthetic_page(32, 32), &bad_cfg).is_err());
}

#[test]
fn independent_corner_detachment_safety_reprojection_and_mask_parity() {
    use doc_photo_degrade::{
        artifact_bits, reproject::transform_point_exact, CornerDetachConfig, MaskRequest,
    };

    // Tight-margin document: ink starts at x=12, y=12 on a 240x320 page.
    let mut page = RgbImage::from_pixel(240, 320, Rgb([250, 250, 248]));
    for y in (16..304).step_by(24) {
        for x in 12..228 {
            for dy in 0..4 {
                page.put_pixel(x, y + dy, Rgb([15, 15, 18]));
            }
        }
    }
    // High-contrast fiducial dot at center (120, 160) for centroid accuracy check.
    for dy in -4i32..=4 {
        for dx in -4i32..=4 {
            page.put_pixel((120 + dx) as u32, (160 + dy) as u32, Rgb([0, 0, 0]));
        }
    }

    let mut saw_inset = false;
    let mut saw_offset = false;
    for seed in 0..12_u64 {
        let mut cfg = DegradeConfig::default().with_seed(seed).with_strength(0.7);
        cfg.corner_detach = CornerDetachConfig {
            enabled: true,
            inset_prob: 0.25,
            ..CornerDetachConfig::phone()
        };
        cfg.masks = MaskRequest {
            artifacts: true,
            ..MaskRequest::none()
        };
        let res = degrade_image(page.clone(), &cfg).expect("detached degrade should succeed");
        let p = &res.params;
        assert_eq!(p.frame_mode, "detached");
        for m in &p.corner_detach_modes {
            if m == "inset" {
                saw_inset = true;
            } else if m == "offset" {
                saw_offset = true;
            } else {
                panic!("unexpected corner_detach_mode: {m}");
            }
        }
        // Every content corner (12,16), (227,16), (227,307), (12,307) must remain strictly in frame.
        let (ow, oh) = (res.image.width() as f64, res.image.height() as f64);
        for &(cx, cy) in &[(12.0, 16.0), (227.0, 16.0), (227.0, 307.0), (12.0, 307.0)] {
            let (ox, oy) = transform_point_exact(cx, cy, p);
            assert!(
                ox >= 0.0 && ox < ow && oy >= 0.0 && oy < oh,
                "seed {seed}: content corner ({cx},{cy}) clipped to ({ox:.2},{oy:.2}) outside ({ow},{oh})"
            );
        }
        // OUT_OF_PAGE mask must exist and match output dimensions.
        let am = res
            .masks
            .as_ref()
            .expect("masks must be populated")
            .artifacts
            .as_ref()
            .expect("artifacts mask must be populated");
        assert_eq!(am.len(), (res.image.width() * res.image.height()) as usize);
        // Center fiducial must not be marked OUT_OF_PAGE.
        let (fx, fy) = transform_point_exact(120.0, 160.0, p);
        let idx = (fy.round() as u32 * res.image.width() + fx.round() as u32) as usize;
        assert_eq!(
            am[idx] & artifact_bits::OUT_OF_PAGE,
            0,
            "seed {seed}: center content wrongly flagged OUT_OF_PAGE"
        );
    }
    assert!(
        saw_inset && saw_offset,
        "expected both inset and offset detached corners"
    );
}

#[test]
fn detached_rendered_ink_centroid_error_under_point_two_px() {
    use doc_photo_degrade::{
        reproject::transform_point_exact, CornerDetachConfig, RealismConfig, StageToggles,
    };

    let (w, h) = (320u32, 420u32);
    let mut page = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    let dots = [
        (80u32, 95u32),
        (160, 95),
        (240, 95),
        (80, 210),
        (160, 210),
        (240, 210),
        (80, 325),
        (160, 325),
        (240, 325),
    ];
    for &(dx, dy) in &dots {
        for yy in dy.saturating_sub(3)..=(dy + 3).min(h - 1) {
            for xx in dx.saturating_sub(3)..=(dx + 3).min(w - 1) {
                page.put_pixel(xx, yy, Rgb([0, 0, 0]));
            }
        }
    }

    let mut cfg = DegradeConfig {
        seed: 77,
        strength: 0.65,
        scale_min: 1.0,
        toggles: StageToggles {
            skip_grain: true,
            skip_chromatic: true,
            skip_noise: true,
            coarsen_ink: true,
            coarsen_aging: true,
            ..StageToggles::full()
        },
        realism: RealismConfig::reduced(),
        corner_detach: CornerDetachConfig::phone(),
        ..DegradeConfig::default()
    };
    cfg.realism.enable_usm = false;
    cfg.realism.enable_tone_curve = false;
    cfg.realism.enable_double_jpeg = false;

    let res = degrade_image(page, &cfg).expect("degrade should succeed");
    assert_eq!(res.params.frame_mode, "detached");

    let mut errs = Vec::new();
    let rad = 9i32;
    for &(dx, dy) in &dots {
        let (px, py) = transform_point_exact(dx as f64, dy as f64, &res.params);
        let (cx, cy) = (px.round() as i32, py.round() as i32);
        if cx - rad < 1
            || cy - rad < 1
            || cx + rad >= res.image.width() as i32 - 1
            || cy + rad >= res.image.height() as i32 - 1
        {
            continue;
        }
        let mut bg_sum = 0.0f64;
        let mut bg_n = 0usize;
        for yy in -rad..=rad {
            for xx in -rad..=rad {
                if xx.abs() == rad || yy.abs() == rad {
                    let rgb = res.image.get_pixel((cx + xx) as u32, (cy + yy) as u32);
                    bg_sum += (rgb[0] as f64 + rgb[1] as f64 + rgb[2] as f64) / 3.0;
                    bg_n += 1;
                }
            }
        }
        let bg = bg_sum / bg_n as f64;
        if bg < 150.0 {
            continue;
        }
        let thresh = bg * 0.50;
        let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
        for yy in (cy - rad + 1)..=(cy + rad - 1) {
            for xx in (cx - rad + 1)..=(cx + rad - 1) {
                let rgb = res.image.get_pixel(xx as u32, yy as u32);
                let luma = (rgb[0] as f64 + rgb[1] as f64 + rgb[2] as f64) / 3.0;
                if luma < thresh {
                    let w_ink = (thresh - luma).max(0.0);
                    sx += xx as f64 * w_ink;
                    sy += yy as f64 * w_ink;
                    sw += w_ink;
                }
            }
        }
        if sw > 0.0 {
            errs.push((sx / sw - px).hypot(sy / sw - py));
        }
    }
    assert!(errs.len() >= 5, "expected at least 5 visible fiducial dots");
    errs.sort_by(|a, b| a.total_cmp(b));
    let median_err = errs[errs.len() / 2];
    assert!(
        median_err < 0.20,
        "detached corner median reprojection centroid error too large: {median_err:.4} px"
    );
}

#[test]
fn wave3_paper_textures_isp_and_medium_profiles_work_end_to_end() {
    use doc_photo_degrade::{Degrader, DocumentProfile, PaperStock};

    let page = make_synthetic_page(256, 320);

    // Exercise all 7 built-in profiles and verify key v0.3.0 provenance fields populate.
    let mut archival_seen = false;
    let mut thermal_seen = false;
    let mut scanner_seen = false;
    let mut xerox_seen = false;
    for &profile in DocumentProfile::ALL {
        let deg = Degrader::builder()
            .profile(profile)
            .strength(0.75)
            .build()
            .expect("profile build should succeed");
        let res = deg
            .degrade_image(&page, 42)
            .expect("profile degrade should succeed");
        assert!(res.image.width() > 0 && res.image.height() > 0);
        match profile {
            DocumentProfile::ArchivalManuscript => {
                assert_eq!(res.params.paper_stock, "uncoated-rag");
                assert!(res.params.cockle_relief_strength >= 0.0);
                archival_seen = true;
            }
            DocumentProfile::ThermalReceipt => {
                assert_eq!(res.params.paper_stock, "thermal-receipt");
                assert!(res.params.thermal_fade >= 0.0);
                thermal_seen = true;
            }
            DocumentProfile::OfficeScan => {
                assert!(res.params.scanner_edge_width_px <= res.image.width());
                scanner_seen = true;
            }
            DocumentProfile::PhotocopyXerox => {
                assert!(res.params.drum_ghost_alpha >= 0.0);
                xerox_seen = true;
            }
            _ => {}
        }
    }
    assert!(archival_seen && thermal_seen && scanner_seen && xerox_seen);

    // Explicit PaperStock override via builder.
    let glossy = Degrader::builder()
        .profile(DocumentProfile::PhonePhoto)
        .paper_stock(PaperStock::CoatedGlossy)
        .strength(0.6)
        .build()
        .unwrap()
        .degrade_image(&page, 19)
        .unwrap();
    assert_eq!(glossy.params.paper_stock, "coated-glossy");
}

#[test]
fn v02_json_sidecar_deserializes_into_v03_degrade_params() {
    // Take a real serialized DegradeParams and strip all v0.3-added keys (and new CaptureParams
    // keys) to verify a v0.2 JSON sidecar deserializes cleanly via #[serde(default)].
    let page = make_synthetic_page(128, 160);
    let cfg = DegradeConfig::default().with_seed(42).with_strength(0.5);
    let res = degrade_image(page, &cfg).unwrap();
    let mut val = serde_json::to_value(&res.params).unwrap();
    let obj = val.as_object_mut().unwrap();
    for k in [
        "corner_detach_px",
        "corner_detach_modes",
        "paper_weave_stock",
        "paper_weave_strength",
        "paper_stock",
        "brdf_albedo",
        "brdf_black_floor",
        "brdf_oba",
        "brdf_spec_gain",
        "brdf_ink_spread",
        "cockle_relief_strength",
        "tooth_bump_strength",
        "raking_light_angle_deg",
        "raking_elevation_deg",
        "plate_mark_strength",
        "plate_mark_inset",
        "gutter_emboss_strength",
        "phone_shadow_strength",
        "phone_shadow_center",
        "phone_shadow_angle_deg",
        "phone_shadow_penumbra",
        "bend_normal_strength",
        "bend_light_vec",
        "ca_optical_center",
        "ca_cubic_weight",
        "field_curvature",
        "specular_bloom_strength",
        "specular_bloom_center",
        "specular_bloom_radius",
        "specular_bloom_aspect",
        "specular_bloom_angle_deg",
        "tone_black_lift",
        "tone_white_ceil",
        "tone_contrast",
        "chroma_noise_sigma",
        "hdr_shadow_boost",
        "double_jpeg_q1",
        "stain_ink_bleed",
        "crease_crack_strength",
        "drum_ghost_period_px",
        "drum_ghost_alpha",
        "thermal_fade",
        "thermal_tone",
        "thermal_scratches",
        "scanner_edge_mask",
        "scanner_edge_width_px",
        "ccd_dust_lines",
        "ink_micro_grain_sigma",
        "ink_edge_feather_amp",
        "ink_halftone_amp",
        "stand_ledge_px",
        "edge_occluder_kind",
    ] {
        obj.remove(k);
    }
    if let Some(cap) = obj.get_mut("capture").and_then(|v| v.as_object_mut()) {
        cap.remove("tape_serrated");
        cap.remove("hole_desk_through");
    }
    let v02_json = serde_json::to_string(&val).unwrap();
    let parsed: DegradeParams =
        serde_json::from_str(&v02_json).expect("v0.2 JSON sidecar must deserialize cleanly");
    assert_eq!(parsed.file_seed, 42);
    assert_eq!(parsed.corner_detach_px, [[0.0; 2]; 4]);
    assert_eq!(parsed.paper_stock, "");
    assert_eq!(parsed.drum_ghost_period_px, 0);
}
