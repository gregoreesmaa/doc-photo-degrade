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
            assert_eq!(p.frame_mode, "inset");
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
