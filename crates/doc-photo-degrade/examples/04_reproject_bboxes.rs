//! Example 04: Ground-truth bounding-box & polygon reprojection + margin-safety check.
//!
//! Creates a synthetic document page with 3 known text-line bounding boxes,
//! degrades it with page bend disabled (`skip_bend: true` for exact projective +
//! framing + rotation + downscale reprojection), reprojects all 3 boxes and
//! their 4-corner polygons using [`transform_bbox`] and [`transform_polygon`],
//! and asserts that every projected box stays cleanly inside `[0, width) x [0, height)`.
//!
//! Usage:
//!   cargo run --release --example 04_reproject_bboxes

use doc_photo_degrade::{
    degrade_image, transform_bbox, transform_polygon, DegradeConfig, StageToggles,
};
use image::{Rgb, RgbImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (800_u32, 1000_u32);
    let mut img = RgbImage::from_pixel(w, h, Rgb([252, 252, 250]));
    let ink = Rgb([20, 20, 24]);

    // 3 known text-line bounding boxes: [x0, y0, x1, y1]
    let text_lines: [[u32; 4]; 3] = [
        [120, 180, 660, 206],
        [120, 440, 680, 466],
        [120, 740, 580, 766],
    ];

    for &[x0, y0, x1, y1] in &text_lines {
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, ink);
            }
        }
    }

    println!("Input page: {}x{}", w, h);
    println!("Known text-line bboxes [x0, y0, x1, y1]:");
    for (i, bbox) in text_lines.iter().enumerate() {
        println!("  line {}: {:?}", i + 1, bbox);
    }

    let mut cfg = DegradeConfig::default().with_seed(77).with_strength(0.6);
    cfg.toggles = StageToggles {
        skip_bend: true,
        ..StageToggles::full()
    };
    let res = degrade_image(img, &cfg).map_err(|e| format!("degrade_image failed: {e}"))?;
    let p = &res.params;

    println!("\nRecorded DegradeParams geometry (skip_bend = true):");
    println!("  output dims        : {}x{}", p.width, p.height);
    println!("  frame_mode         : {}", p.frame_mode);
    println!("  frame_content_bbox : {:?}", p.frame_content_bbox);
    println!(
        "  frame_quiet_px     : {:?}  [left, top, right, bottom]",
        p.frame_quiet_px
    );
    println!("  frame_safe_px      : {}", p.frame_safe_px);
    println!("  frame_pad_px       : {:?}", p.frame_pad_px);
    println!("  frame_cut_px       : {:?}", p.frame_cut_px);
    println!("  warp_fit_scale     : {:.4}", p.warp_fit_scale);
    println!("  rotation_deg       : {:.3}", p.rotation_deg);
    println!("  scale_x            : {:.4}", p.scale_x);
    println!("  scale_y            : {:.4}", p.scale_y);

    println!("\nReprojected text-line bboxes & polygons:");
    let (out_w, out_h) = (p.width as f64, p.height as f64);
    for (i, &[x0, y0, x1, y1]) in text_lines.iter().enumerate() {
        let src_bbox = [x0 as f64, y0 as f64, x1 as f64, y1 as f64];
        let proj_bbox = transform_bbox(src_bbox, p, 8);
        let src_poly = [
            (x0 as f64, y0 as f64),
            (x1 as f64, y0 as f64),
            (x1 as f64, y1 as f64),
            (x0 as f64, y1 as f64),
        ];
        let proj_poly = transform_polygon(&src_poly, p);

        println!(
            "  line {} bbox : [{:.1}, {:.1}, {:.1}, {:.1}]",
            i + 1,
            proj_bbox[0],
            proj_bbox[1],
            proj_bbox[2],
            proj_bbox[3]
        );
        println!(
            "  line {} poly : [({:.1}, {:.1}), ({:.1}, {:.1}), ({:.1}, {:.1}), ({:.1}, {:.1})]",
            i + 1,
            proj_poly[0].0,
            proj_poly[0].1,
            proj_poly[1].0,
            proj_poly[1].1,
            proj_poly[2].0,
            proj_poly[2].1,
            proj_poly[3].0,
            proj_poly[3].1,
        );

        assert!(
            proj_bbox[0] >= 0.0
                && proj_bbox[1] >= 0.0
                && proj_bbox[2] < out_w
                && proj_bbox[3] < out_h,
            "line {} bbox {:?} escaped output bounds {}x{}",
            i + 1,
            proj_bbox,
            p.width,
            p.height
        );
        for &(px, py) in &proj_poly {
            assert!(
                px >= 0.0 && px < out_w && py >= 0.0 && py < out_h,
                "line {} polygon vertex ({px:.2}, {py:.2}) escaped output bounds {}x{}",
                i + 1,
                p.width,
                p.height
            );
        }
    }

    // Verify margin-safe invariant: tight cuts never encroach past quiet_px - safe_px.
    if p.frame_mode == "tight" {
        for side in 0..4 {
            let max_allowed = p.frame_quiet_px[side].saturating_sub(p.frame_safe_px);
            assert!(
                p.frame_cut_px[side] <= max_allowed,
                "side {side}: cut {} exceeded safe budget {max_allowed}",
                p.frame_cut_px[side]
            );
        }
        println!("\nMargin-safety & reprojection checks PASSED (tight mode).");
    } else {
        assert_eq!(p.frame_cut_px, [0, 0, 0, 0]);
        println!("\nMargin-safety & reprojection checks PASSED (inset mode).");
    }

    Ok(())
}
