//! Example 01: Quick single-page document degradation.
//!
//! Synthesizes a crisp 800x1000 mock document page in memory (header, text-like
//! lines, and a bordered table on white paper stock — no external files needed),
//! runs `degrade_image`, and writes `out_degraded.png` + `out_params.json`.
//!
//! Usage:
//!   cargo run --release --example 01_quick_degrade -- [OUTPUT_DIR]

use doc_photo_degrade::{degrade_image, write_artifact, DegradeConfig};
use image::{Rgb, RgbImage};
use std::path::PathBuf;

fn fill_rect(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, color: Rgb<u8>) {
    let w = img.width();
    let h = img.height();
    for y in y0.min(h)..y1.min(h) {
        for x in x0.min(w)..x1.min(w) {
            img.put_pixel(x, y, color);
        }
    }
}

fn draw_rect_outline(
    img: &mut RgbImage,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    thickness: u32,
    color: Rgb<u8>,
) {
    fill_rect(img, x0, y0, x1, y0 + thickness, color);
    fill_rect(img, x0, y1.saturating_sub(thickness), x1, y1, color);
    fill_rect(img, x0, y0, x0 + thickness, y1, color);
    fill_rect(img, x1.saturating_sub(thickness), y0, x1, y1, color);
}

/// Build an 800x1000 synthetic document page with clean margins, a header,
/// text-like horizontal bars, and a structured table frame.
fn synthesize_document_page() -> RgbImage {
    let (w, h) = (800, 1000);
    let paper = Rgb([252, 252, 250]);
    let ink = Rgb([24, 24, 28]);
    let muted = Rgb([72, 76, 84]);
    let rule = Rgb([48, 50, 56]);

    let mut img = RgbImage::from_pixel(w, h, paper);

    // Header block + horizontal divider
    fill_rect(&mut img, 90, 90, 420, 114, ink);
    fill_rect(&mut img, 90, 124, 290, 134, muted);
    fill_rect(&mut img, 590, 96, 710, 108, muted);
    fill_rect(&mut img, 90, 152, 710, 155, rule);

    // Paragraph 1: text-like bars with realistic word breaks
    let mut y = 185;
    for line in 0..8 {
        let mut x = 90;
        let line_end = if line == 7 { 510 } else { 700 };
        let mut word_idx = 0u32;
        while x + 28 < line_end {
            let word_w = 34 + ((line * 17 + word_idx * 29) % 58);
            let next_x = (x + word_w).min(line_end);
            fill_rect(&mut img, x, y, next_x, y + 9, ink);
            x = next_x + 12;
            word_idx += 1;
        }
        y += 24;
    }

    // Table frame with header row, 4 data rows, and 3 columns
    let (tx0, ty0, tx1, ty1) = (90, 420, 710, 720);
    draw_rect_outline(&mut img, tx0, ty0, tx1, ty1, 2, rule);
    fill_rect(
        &mut img,
        tx0 + 2,
        ty0 + 2,
        tx1 - 2,
        ty0 + 44,
        Rgb([234, 236, 240]),
    );
    fill_rect(&mut img, tx0, ty0 + 44, tx1, ty0 + 46, rule);

    // Vertical column dividers
    for cx in [310, 530] {
        fill_rect(&mut img, cx, ty0, cx + 2, ty1, rule);
    }
    // Header text bars
    fill_rect(&mut img, tx0 + 18, ty0 + 16, tx0 + 150, ty0 + 28, ink);
    fill_rect(&mut img, 328, ty0 + 16, 460, ty0 + 28, ink);
    fill_rect(&mut img, 548, ty0 + 16, 665, ty0 + 28, ink);

    // Horizontal row dividers + cell content bars
    for row in 0..5 {
        let ry = ty0 + 46 + row * 54;
        if row > 0 {
            fill_rect(&mut img, tx0, ry, tx1, ry + 1, rule);
        }
        let bar_y = ry + 20;
        fill_rect(
            &mut img,
            tx0 + 18,
            bar_y,
            tx0 + 140 + (row * 13) % 50,
            bar_y + 9,
            ink,
        );
        fill_rect(&mut img, 328, bar_y, 440 + (row * 19) % 45, bar_y + 9, ink);
        fill_rect(&mut img, 560, bar_y, 660 - (row * 7) % 30, bar_y + 9, muted);
    }

    // Footer note lines
    fill_rect(&mut img, 90, 770, 620, 779, ink);
    fill_rect(&mut img, 90, 794, 540, 803, ink);
    fill_rect(&mut img, 90, 818, 390, 827, muted);

    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("doc-photo-degrade-example-01"));
    std::fs::create_dir_all(&out_dir)?;

    let page = synthesize_document_page();
    let cfg = DegradeConfig::default().with_seed(42).with_strength(0.6);

    let res = degrade_image(page, &cfg).map_err(|e| format!("degrade_image failed: {e}"))?;

    let img_path = out_dir.join("out_degraded.png");
    let params_path = out_dir.join("out_params.json");

    write_artifact(&img_path, |tmp| {
        res.image
            .save_with_format(tmp, image::ImageFormat::Png)
            .map_err(|e| format!("failed to write PNG: {e}"))
    })
    .map_err(|e| format!("write_artifact PNG failed: {e}"))?;

    let json = serde_json::to_string_pretty(&res.params)?;
    write_artifact(&params_path, |tmp| {
        std::fs::write(tmp, &json).map_err(|e| format!("failed to write params JSON: {e}"))
    })
    .map_err(|e| format!("write_artifact JSON failed: {e}"))?;

    println!("Degraded synthetic 800x1000 document page:");
    println!(
        "  image  : {} ({}x{})",
        img_path.display(),
        res.image.width(),
        res.image.height()
    );
    println!("  params : {}", params_path.display());
    println!(
        "  summary: backend={} preset={} frame_mode={} rotation_deg={:.2} curl_side={} jpeg_q={}",
        res.params.backend,
        res.params.preset,
        res.params.frame_mode,
        res.params.rotation_deg,
        res.params.curl_side,
        res.params.jpeg_quality,
    );

    Ok(())
}
