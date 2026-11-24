//! Example 02: Degradation strength sweep.
//!
//! Sweeps `strength` across `[0.15, 0.45, 0.75, 1.0]` on a synthetic document
//! page and prints `frame_mode`, `rotation_deg`, `curl_side`, `jpeg_quality`,
//! and wall-clock duration in milliseconds per level.
//!
//! Usage:
//!   cargo run --release --example 02_strength_sweep

use doc_photo_degrade::{degrade_image, DegradeConfig};
use image::{Rgb, RgbImage};
use std::time::Instant;

fn synthesize_page(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([250, 250, 248]));
    let ink = Rgb([25, 25, 30]);

    // Header + horizontal rule
    for y in 60..76 {
        for x in 60..320 {
            img.put_pixel(x, y, ink);
        }
    }
    for y in 92..95 {
        for x in 60..(w - 60) {
            img.put_pixel(x, y, ink);
        }
    }

    // Text-like rows
    for row in 0..18 {
        let y0 = 120 + row * 32;
        if y0 + 10 >= h - 60 {
            break;
        }
        let x_end = w - 60 - ((row * 37) % 120);
        for y in y0..(y0 + 8) {
            for x in 60..x_end {
                if (x / 48) % 5 != 4 {
                    img.put_pixel(x, y, ink);
                }
            }
        }
    }
    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_page = synthesize_page(600, 800);
    let strengths = [0.15_f32, 0.45, 0.75, 1.0];

    println!(
        "{:<10} {:<10} {:>12} {:<10} {:>12} {:>10}",
        "strength", "frame_mode", "rotation_deg", "curl_side", "jpeg_quality", "time_ms"
    );
    println!("{}", "-".repeat(70));

    for &strength in &strengths {
        let cfg = DegradeConfig::default()
            .with_seed(1337)
            .with_strength(strength);
        let t0 = Instant::now();
        let res = degrade_image(base_page.clone(), &cfg)
            .map_err(|e| format!("degrade_image failed at strength={strength}: {e}"))?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{:<10.2} {:<10} {:>12.3} {:<10} {:>12} {:>10.1}",
            strength,
            res.params.frame_mode,
            res.params.rotation_deg,
            res.params.curl_side,
            res.params.jpeg_quality,
            elapsed_ms,
        );
    }

    Ok(())
}
