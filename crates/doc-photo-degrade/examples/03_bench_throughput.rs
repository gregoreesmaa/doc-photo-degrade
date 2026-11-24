//! Example 03: CPU determinism verification and throughput benchmark.
//!
//! Synthesizes 4 document pages in memory, verifies bit-exact CPU determinism
//! across two runs with the same seed, and benchmarks throughput in pages/sec
//! and megapixels/sec.
//!
//! Usage:
//!   cargo run --release --example 03_bench_throughput

use doc_photo_degrade::{degrade_image, DegradeConfig};
use image::{Rgb, RgbImage};
use std::time::Instant;

fn synthesize_page(variant: u32, w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([252, 251, 249]));
    let ink = Rgb([20, 22, 26]);

    let margin = 64 + (variant % 3) * 8;
    // Title bar
    for y in margin..(margin + 18) {
        for x in margin..(w / 2 + variant * 25).min(w - margin) {
            img.put_pixel(x, y, ink);
        }
    }
    // Body lines
    let mut y = margin + 44;
    let mut row = 0u32;
    while y + 10 < h - margin {
        let end_x = w - margin - ((row * 29 + variant * 41) % 110);
        for yy in y..(y + 8) {
            for xx in margin..end_x {
                if ((xx + row * 11) / 36) % 6 != 5 {
                    img.put_pixel(xx, yy, ink);
                }
            }
        }
        y += 26;
        row += 1;
    }
    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (640, 800);
    let pages: Vec<RgbImage> = (0..4).map(|i| synthesize_page(i, w, h)).collect();

    // 1. Verify bit-exact CPU determinism on page 0.
    let det_cfg = DegradeConfig::default().with_seed(2026).with_strength(0.65);
    let run_a = degrade_image(pages[0].clone(), &det_cfg)
        .map_err(|e| format!("determinism run A failed: {e}"))?;
    let run_b = degrade_image(pages[0].clone(), &det_cfg)
        .map_err(|e| format!("determinism run B failed: {e}"))?;

    assert_eq!(
        run_a.image.dimensions(),
        run_b.image.dimensions(),
        "dimensions must match across identical-seed CPU runs"
    );
    assert_eq!(
        run_a.image.as_raw(),
        run_b.image.as_raw(),
        "pixels must be bit-identical across identical-seed CPU runs"
    );
    println!(
        "Determinism check PASSED: identical {}x{} RGB pixels ({} bytes) for seed={}.",
        run_a.image.width(),
        run_a.image.height(),
        run_a.image.as_raw().len(),
        det_cfg.seed,
    );

    // 2. Benchmark 4 synthetic pages.
    let t0 = Instant::now();
    let mut total_in_pixels: u64 = 0;
    for (idx, page) in pages.iter().enumerate() {
        let cfg = DegradeConfig::default()
            .with_seed(9000 + idx as u64)
            .with_strength(0.55);
        total_in_pixels += u64::from(page.width()) * u64::from(page.height());
        let res =
            degrade_image(page.clone(), &cfg).map_err(|e| format!("page {idx} failed: {e}"))?;
        // Prevent any dead-code elimination by touching output dimensions.
        assert!(res.image.width() > 0 && res.image.height() > 0);
    }
    let elapsed_s = t0.elapsed().as_secs_f64();
    let pages_per_sec = pages.len() as f64 / elapsed_s.max(1e-9);
    let megapixels = total_in_pixels as f64 / 1_000_000.0;
    let mp_per_sec = megapixels / elapsed_s.max(1e-9);

    println!(
        "Benchmarked {} pages ({:.2} MP total) in {:.1} ms -> {:.2} pages/sec | {:.2} MP/sec",
        pages.len(),
        megapixels,
        elapsed_s * 1000.0,
        pages_per_sec,
        mp_per_sec,
    );

    Ok(())
}
