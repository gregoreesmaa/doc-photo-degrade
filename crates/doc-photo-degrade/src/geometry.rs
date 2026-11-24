//! Document geometry helpers: margin-safe content bounding boxes, quiet-margin
//! framing (desk inset & tight crop), homography solver, book-spine gutter map,
//! ink-density grid, and bilinear sampling.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Radial lens distortion factor: `1 + k1*r^2 + k2*r^4` (`r` normalized).
/// `k1 < 0` bows lines outward (barrel), `k1 > 0` pulls them in (pincushion).
#[allow(dead_code)]
pub(crate) fn radial_factor(r2: f32, k1: f32, k2: f32) -> f32 {
    1.0 + k1 * r2 + k2 * r2 * r2
}

/// Book-spine gutter: smooth horizontal squeeze toward `spine_cx` plus
/// darkening. Continuous and strictly monotonic in `x` — horizontal lines
/// compress but never tear, and no content is swallowed at the gutter.
pub(crate) fn spine_map(
    x: f32,
    spine_cx: f32,
    spine_w: f32,
    depth: f32,
    shade_amt: f32,
) -> (f32, f32) {
    let t = (x - spine_cx) / spine_w.max(1.0);
    let g = (-t * t).exp();
    let sx = spine_cx + (x - spine_cx) * (1.0 - depth * g);
    (sx, 1.0 - shade_amt * g)
}

/// Running dark-pixel bounding box accumulator: the perspective/bend warp loops
/// fold each written pixel's channel sum into this instead of paying for a
/// standalone framing-time scan. `fold` takes the exact sum a later
/// [`content_bbox_thresh`] scan would see, so the fused result is identical.
#[derive(Default)]
pub(crate) struct BBoxAcc {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    found: bool,
}

impl BBoxAcc {
    pub(crate) fn reset(&mut self) {
        *self = BBoxAcc::default();
    }

    pub(crate) fn fold(&mut self, sum3: f32, x: u32, y: u32, thresh3: f32) {
        if sum3 < thresh3 {
            if !self.found {
                self.x0 = x;
                self.y0 = y;
                self.x1 = x;
                self.y1 = y;
                self.found = true;
            } else {
                self.x0 = self.x0.min(x);
                self.y0 = self.y0.min(y);
                self.x1 = self.x1.max(x);
                self.y1 = self.y1.max(y);
            }
        }
    }

    pub(crate) fn bbox(&self) -> Option<[u32; 4]> {
        if self.found {
            Some([self.x0, self.y0, self.x1, self.y1])
        } else {
            None
        }
    }
}

/// Dark-pixel (document content) bounding box: pixels whose mean channel is
/// `< 128`, the same ink threshold `ink_grid` uses. Returns
/// `[min_x, min_y, max_x, max_y]` inclusive, or `None` on a blank page.
pub fn content_bbox(buf: &[f32], w: u32, h: u32) -> Option<[u32; 4]> {
    content_bbox_thresh(buf, w, h, 128.0)
}

/// Threshold-parameterized dark-pixel content bounding box.
///
/// The training-resolution path uses `200.0`: bilinear downscaling pre-grays
/// thin strokes and ink fade can lift solid blacks toward ~180, so `128.0` can
/// miss fine ink and return optimistic quiet margins. `200.0` clears clean
/// paper (~244) while catching pre-grayed or faded ink conservatively.
pub fn content_bbox_thresh(buf: &[f32], w: u32, h: u32, thresh: f32) -> Option<[u32; 4]> {
    let thresh3 = 3.0 * thresh;
    let mut acc = BBoxAcc::default();
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            acc.fold(buf[i] + buf[i + 1] + buf[i + 2], x, y, thresh3);
        }
    }
    acc.bbox()
}

/// Quiet (content-free) margin depth per edge `[left, top, right, bottom]`:
/// distance from each frame edge to the document content bounding box. Blank
/// pages report `[u32::MAX; 4]` (nothing to protect, any in-range cut is safe).
pub fn quiet_margins(w: u32, h: u32, bbox: Option<[u32; 4]>) -> [u32; 4] {
    match bbox {
        Some([x0, y0, x1, y1]) => [
            x0,
            y0,
            w.saturating_sub(1).saturating_sub(x1),
            h.saturating_sub(1).saturating_sub(y1),
        ],
        None => [u32::MAX; 4],
    }
}

/// Clamp each desired tight-crop edge so at least `safe` px of quiet margin
/// survive between the frame edge and the document content bounding box.
///
/// Content can never be cropped: `cut[i] + safe <= quiet[i]` holds elementwise
/// whenever `cut[i] > 0` (and `cut[i] <= quiet[i]` always).
pub fn margin_safe_crop(desired: [u32; 4], quiet: [u32; 4], safe: u32) -> [u32; 4] {
    let mut out = [0u32; 4];
    for i in 0..4 {
        out[i] = desired[i].min(quiet[i].saturating_sub(safe));
    }
    out
}

/// Crop `cut = [left, top, right, bottom]` px off the frame (tight variant).
/// Content at `(x, y)` moves to `(x - cut_left, y - cut_top)`.
pub(crate) fn apply_frame_cut(buf: &[f32], w: u32, h: u32, cut: [u32; 4]) -> (Vec<f32>, u32, u32) {
    let nw = w.saturating_sub(cut[0].saturating_add(cut[2])).max(1);
    let nh = h.saturating_sub(cut[1].saturating_add(cut[3])).max(1);
    let nw = nw.min(w.saturating_sub(cut[0]).max(1));
    let nh = nh.min(h.saturating_sub(cut[1]).max(1));
    let mut out = vec![0.0f32; (nw * nh * 3) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let si = (((y + cut[1]) * w + (x + cut[0])) * 3) as usize;
            let di = ((y * nw + x) * 3) as usize;
            out[di..di + 3].copy_from_slice(&buf[si..si + 3]);
        }
    }
    (out, nw, nh)
}

/// Pad `pad = [left, top, right, bottom]` px of dark desk around the page
/// (inset variant). The page lands intact at `(pad_left, pad_top)`; the
/// surround uses the same desk tone as the rotation stage.
pub(crate) fn apply_frame_pad(buf: &[f32], w: u32, h: u32, pad: [u32; 4]) -> (Vec<f32>, u32, u32) {
    const DESK: [f32; 3] = [25.0, 25.0, 30.0];
    let nw = w.saturating_add(pad[0].saturating_add(pad[2])).max(1);
    let nh = h.saturating_add(pad[1].saturating_add(pad[3])).max(1);
    let mut out = vec![0.0f32; (nw * nh * 3) as usize];
    for i in 0..(nw * nh) as usize {
        out[i * 3] = DESK[0];
        out[i * 3 + 1] = DESK[1];
        out[i * 3 + 2] = DESK[2];
    }
    let cw = w.min(nw.saturating_sub(pad[0]));
    let ch = h.min(nh.saturating_sub(pad[1]));
    for y in 0..ch {
        for x in 0..cw {
            let si = ((y * w + x) * 3) as usize;
            let di = (((y + pad[1]) * nw + (x + pad[0])) * 3) as usize;
            out[di..di + 3].copy_from_slice(&buf[si..si + 3]);
        }
    }
    (out, nw, nh)
}

/// Framed dims without materializing: tight crops, inset pads — the same
/// arithmetic as `apply_frame_cut`/`apply_frame_pad`, for the fused rotation
/// loop.
pub(crate) fn framed_dims(w: u32, h: u32, mode: &str, pad: [u32; 4], cut: [u32; 4]) -> (u32, u32) {
    if mode == "inset" {
        (
            w.saturating_add(pad[0].saturating_add(pad[2])).max(1),
            h.saturating_add(pad[1].saturating_add(pad[3])).max(1),
        )
    } else {
        let nw = w.saturating_sub(cut[0].saturating_add(cut[2])).max(1);
        let nh = h.saturating_sub(cut[1].saturating_add(cut[3])).max(1);
        (
            nw.min(w.saturating_sub(cut[0]).max(1)),
            nh.min(h.saturating_sub(cut[1]).max(1)),
        )
    }
}

/// Materialize pending framing (crop XOR pad). Used on the GPU path (stages
/// 3b-6 run framed while rotation is deferred to 6b) and whenever the rotation
/// gate is off.
pub(crate) fn apply_framing_eager(
    buf: &mut Vec<f32>,
    w: &mut u32,
    h: &mut u32,
    mode: &str,
    pad: [u32; 4],
    cut: [u32; 4],
) {
    if mode == "inset" {
        if pad.iter().sum::<u32>() > 0 {
            let (fb, fw, fh) = apply_frame_pad(buf, *w, *h, pad);
            *buf = fb;
            *w = fw;
            *h = fh;
        }
    } else if cut.iter().sum::<u32>() > 0 {
        let (fb, fw, fh) = apply_frame_cut(buf, *w, *h, cut);
        *buf = fb;
        *w = fw;
        *h = fh;
    }
}

pub(crate) fn set_px(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        buf[i] = c[0];
        buf[i + 1] = c[1];
        buf[i + 2] = c[2];
    }
}

pub(crate) fn blend_px(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3], a: f32) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        for k in 0..3 {
            buf[i + k] = buf[i + k] * (1.0 - a) + c[k] * a;
        }
    }
}

/// Thin polyline stamp (pencil strokes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn stroke(
    buf: &mut [f32],
    w: u32,
    h: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    r: i64,
    c: [f32; 3],
) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as i64).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let cx = (x0 + (x1 - x0) * t) as i64;
        let cy = (y0 + (y1 - y0) * t) as i64;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    set_px(buf, w, h, cx + dx, cy + dy, c);
                }
            }
        }
    }
}

/// Ink fraction per `cell`-px grid cell (ink = mean channel < 128).
pub(crate) fn ink_grid(buf: &[f32], w: u32, h: u32, cell: u32) -> (Vec<f32>, u32, u32) {
    let gw = w.div_ceil(cell).max(1);
    let gh = h.div_ceil(cell).max(1);
    let mut grid = vec![0f32; (gw * gh) as usize];
    let mut counts = vec![0u32; (gw * gh) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y as usize) * (w as usize) + (x as usize)) * 3;
            let ink = if (buf[i] + buf[i + 1] + buf[i + 2]) / 3.0 < 128.0 {
                1.0
            } else {
                0.0
            };
            let gi = ((y / cell) * gw + (x / cell)) as usize;
            grid[gi] += ink;
            counts[gi] += 1;
        }
    }
    for (g, c) in grid.iter_mut().zip(counts.iter()) {
        *g /= (*c).max(1) as f32;
    }
    (grid, gw, gh)
}

/// Random grid cell with ink fraction below `max_ink`, or `None`.
pub(crate) fn pick_quiet_cell(
    rng: &mut ChaCha8Rng,
    grid: &[f32],
    gw: u32,
    gh: u32,
    max_ink: f32,
) -> Option<(u32, u32)> {
    for _ in 0..12 {
        let gx = rng.gen_range(0..gw);
        let gy = rng.gen_range(0..gh);
        if grid[(gy * gw + gx) as usize] < max_ink {
            return Some((gx, gy));
        }
    }
    None
}

/// Bilinear sample with remapped taps: identical coordinate expression tree to
/// [`sample_bilinear`], with each tap value supplied by `tap(ix, iy)`.
pub(crate) fn sample_remapped(fx: f32, fy: f32, tap: impl Fn(i64, i64) -> [f32; 3]) -> [f32; 3] {
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let mut out = [0.0f32; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        let top = tap(x0, y0)[c] * (1.0 - tx) + tap(x0 + 1, y0)[c] * tx;
        let bot = tap(x0, y0 + 1)[c] * (1.0 - tx) + tap(x0 + 1, y0 + 1)[c] * tx;
        *slot = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Blend four already-fetched taps with bilinear weights.
pub(crate) fn blend4(
    t00: [f32; 3],
    t10: [f32; 3],
    t01: [f32; 3],
    t11: [f32; 3],
    tx: f32,
    ty: f32,
) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = t00[c] * (1.0 - tx) + t10[c] * tx;
        let bot = t01[c] * (1.0 - tx) + t11[c] * tx;
        out[c] = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Bilinear sample; out-of-bounds yields paper white.
pub(crate) fn sample_bilinear(buf: &[f32], w: u32, h: u32, x: f32, y: f32) -> [f32; 3] {
    const BG: [f32; 3] = [246.0, 243.0, 236.0];
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    if x0 >= 0 && y0 >= 0 && x0 + 1 < w as i64 && y0 + 1 < h as i64 {
        let j00 = ((y0 as u32 * w + x0 as u32) * 3) as usize;
        let t00 = [buf[j00], buf[j00 + 1], buf[j00 + 2]];
        let t10 = [buf[j00 + 3], buf[j00 + 4], buf[j00 + 5]];
        let j01 = j00 + w as usize * 3;
        let t01 = [buf[j01], buf[j01 + 1], buf[j01 + 2]];
        let t11 = [buf[j01 + 3], buf[j01 + 4], buf[j01 + 5]];
        return blend4(t00, t10, t01, t11, fx, fy);
    }
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let at = |xx: i64, yy: i64| -> f32 {
            if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                BG[c]
            } else {
                buf[((yy as u32 * w + xx as u32) * 3 + c as u32) as usize]
            }
        };
        let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
        let bot = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    out
}

/// Solve for the homography mapping `src -> dst` (`h33` fixed to 1).
pub(crate) fn solve_homography(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Option<[f64; 9]> {
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = src[i];
        let (xp, yp) = dst[i];
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y, xp];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y, yp];
    }
    for col in 0..8 {
        let mut piv = col;
        for row in col..8 {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for v in &mut a[col][col..9] {
            *v /= d;
        }
        let piv_row = a[col];
        for (row, rvals) in a.iter_mut().enumerate() {
            if row != col {
                let f = rvals[col];
                for (dst, &src) in rvals[col..9].iter_mut().zip(&piv_row[col..9]) {
                    *dst -= f * src;
                }
            }
        }
    }
    Some([
        a[0][8], a[1][8], a[2][8], a[3][8], a[4][8], a[5][8], a[6][8], a[7][8], 1.0,
    ])
}

pub(crate) fn invert_3x3(m: [f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    Some([
        (m[4] * m[8] - m[5] * m[7]) / det,
        (m[2] * m[7] - m[1] * m[8]) / det,
        (m[1] * m[5] - m[2] * m[4]) / det,
        (m[5] * m[6] - m[3] * m[8]) / det,
        (m[0] * m[8] - m[2] * m[6]) / det,
        (m[2] * m[3] - m[0] * m[5]) / det,
        (m[3] * m[7] - m[4] * m[6]) / det,
        (m[1] * m[6] - m[0] * m[7]) / det,
        (m[0] * m[4] - m[1] * m[3]) / det,
    ])
}

pub(crate) fn apply_h(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let z = h[6] * x + h[7] * y + h[8];
    if z.abs() < 1e-12 {
        return (x, y);
    }
    (
        (h[0] * x + h[1] * y + h[2]) / z,
        (h[3] * x + h[4] * y + h[5]) / z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn spine_map_is_continuous_and_monotonic() {
        let depth = 0.15;
        let mut prev: Option<f32> = None;
        for x in 0..2000 {
            let (sx, shade) = spine_map(x as f32, 1000.0, 80.0, depth, 0.3);
            if let Some(p) = prev {
                assert!((sx - p).abs() <= 1.0 + depth + 1e-3, "tear at x={x}");
                assert!(sx >= p - 1e-6, "fold-over at x={x}");
            }
            assert!((0.0..=1.0).contains(&shade), "shade out of range at x={x}");
            prev = Some(sx);
        }
    }

    #[test]
    fn radial_factor_shapes() {
        assert!(
            (radial_factor(0.0, -0.2, 0.05) - 1.0).abs() < 1e-6,
            "center undistorted"
        );
        assert!(
            radial_factor(1.0, -0.2, 0.0) < 1.0,
            "negative k1 must bow outward"
        );
        assert!(
            radial_factor(1.0, 0.3, 0.0) > 1.0,
            "positive k1 must pull inward"
        );
    }

    #[test]
    fn ink_grid_and_quiet_pick() {
        let (w, h) = (128u32, 128u32);
        let mut buf = vec![255.0f32; (w * h * 3) as usize];
        for y in 0..64 {
            for x in 0..64 {
                let i = ((y * w + x) * 3) as usize;
                buf[i] = 0.0;
                buf[i + 1] = 0.0;
                buf[i + 2] = 0.0;
            }
        }
        let (grid, gw, gh) = ink_grid(&buf, w, h, 64);
        assert_eq!((gw, gh), (2, 2));
        assert!((grid[0] - 1.0).abs() < 1e-6);
        assert_eq!(grid[3], 0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let got = pick_quiet_cell(&mut rng, &grid, gw, gh, 0.06);
        assert!(got.is_some(), "white cell must be pickable");
        let full = vec![1.0f32; 4];
        assert!(pick_quiet_cell(&mut rng, &full, 2, 2, 0.06).is_none());
    }

    #[test]
    fn frame_cut_clamped_to_quiet_and_keeps_content() {
        let (w, h) = (64u32, 64u32);
        let mut buf = vec![255.0f32; (w * h * 3) as usize];
        for y in 10..=30u32 {
            for x in 20..=44u32 {
                let i = ((y * w + x) * 3) as usize;
                buf[i] = 0.0;
                buf[i + 1] = 0.0;
                buf[i + 2] = 0.0;
            }
        }
        let bbox = content_bbox(&buf, w, h).expect("must find the rect");
        assert_eq!(bbox, [20, 10, 44, 30]);
        let quiet = quiet_margins(w, h, Some(bbox));
        assert_eq!(quiet, [20, 10, 19, 33]);
        let cut = margin_safe_crop([30, 30, 30, 30], quiet, 3);
        assert_eq!(cut, [17, 7, 16, 30]);
        let (cropped, nw, nh) = apply_frame_cut(&buf, w, h, cut);
        assert_eq!((nw, nh), (64 - 33, 64 - 37));
        let kept = content_bbox(&cropped, nw, nh).expect("content must survive the crop");
        assert_eq!(kept, [20 - 17, 10 - 7, 44 - 17, 30 - 7]);
    }

    #[test]
    fn frame_helpers_handle_blank_and_empty_cuts() {
        let (w, h) = (64u32, 64u32);
        let blank = vec![255.0f32; (w * h * 3) as usize];
        assert_eq!(content_bbox(&blank, w, h), None);
        let quiet = quiet_margins(w, h, None);
        assert_eq!(margin_safe_crop([2, 3, 2, 3], quiet, 3), [2, 3, 2, 3]);
        let (same, nw, nh) = apply_frame_cut(&blank, w, h, [0, 0, 0, 0]);
        assert_eq!((nw, nh), (w, h));
        assert_eq!(same.len(), blank.len());
        let (padded, pw, ph) = apply_frame_pad(&blank, w, h, [0, 0, 0, 0]);
        assert_eq!((pw, ph), (w, h));
        assert_eq!(padded.len(), blank.len());
        let (padded2, pw2, ph2) = apply_frame_pad(&blank, w, h, [2, 3, 4, 5]);
        assert_eq!((pw2, ph2), (70, 72));
        assert_eq!(padded2[0], 25.0, "pad corner must be desk fill");
        let src_i = 0;
        let dst_i = ((3 * pw2 + 2) * 3) as usize;
        assert_eq!(padded2[dst_i..dst_i + 3], blank[src_i..src_i + 3]);
    }

    #[test]
    fn fused_bbox_matches_standalone_scan() {
        let (w, h) = (64u32, 48u32);
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for thresh in [128.0f32, 200.0] {
            for round in 0..3 {
                let mut buf = vec![244.0f32; (w * h * 3) as usize];
                for v in buf.iter_mut() {
                    let r: f32 = rng.gen();
                    *v = if round == 2 && r < 0.3 {
                        100.0 + rng.gen::<f32>() * 60.0
                    } else if r < 0.15 {
                        rng.gen::<f32>() * 60.0
                    } else if r < 0.25 {
                        200.0 + rng.gen::<f32>() * 55.0
                    } else {
                        244.0
                    };
                }
                let mut acc = BBoxAcc::default();
                for y in 0..h {
                    for x in 0..w {
                        let i = ((y * w + x) * 3) as usize;
                        acc.fold(buf[i] + buf[i + 1] + buf[i + 2], x, y, 3.0 * thresh);
                    }
                }
                assert_eq!(
                    acc.bbox(),
                    content_bbox_thresh(&buf, w, h, thresh),
                    "thresh {thresh} round {round}"
                );
            }
        }
        let mut rect = vec![255.0f32; (w * h * 3) as usize];
        for y in 10..30u32 {
            for x in 20..44u32 {
                let i = ((y * w + x) * 3) as usize;
                rect[i] = 0.0;
                rect[i + 1] = 0.0;
                rect[i + 2] = 0.0;
            }
        }
        assert_eq!(content_bbox(&rect, w, h), Some([20, 10, 43, 29]));
    }

    #[test]
    fn framing_in_rotation_inset_is_bit_exact() {
        let (w, h) = (48u32, 36u32);
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        for v in buf.iter_mut() {
            *v = rng.gen_range(0.0..255.0);
        }
        let pad = [3u32, 5, 4, 6];
        for deg in [0.5f32, 3.0, -7.5, 10.0] {
            let (mut eb, ew, eh) = apply_frame_pad(&buf, w, h, pad);
            let th = deg.to_radians();
            let (sn, cs) = (th.sin().abs(), th.cos().abs());
            let (w2, h2) = (
                ((ew as f32) * cs + (eh as f32) * sn).ceil().max(1.0) as u32,
                ((ew as f32) * sn + (eh as f32) * cs).ceil().max(1.0) as u32,
            );
            const DESK: [f32; 3] = [25.0, 25.0, 30.0];
            let src = std::mem::take(&mut eb);
            let (cx0, cy0) = (ew as f32 / 2.0, eh as f32 / 2.0);
            let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
            let (snt, cst) = (th.sin(), th.cos());
            let mut expect = vec![0.0f32; (w2 * h2 * 3) as usize];
            for y in 0..h2 {
                for x in 0..w2 {
                    let dx = x as f32 - cx1;
                    let dy = y as f32 - cy1;
                    let sx = cx0 + dx * cst + dy * snt;
                    let sy = cy0 - dx * snt + dy * cst;
                    let p = if sx >= 0.0 && sx < ew as f32 && sy >= 0.0 && sy < eh as f32 {
                        sample_bilinear(&src, ew, eh, sx, sy)
                    } else {
                        DESK
                    };
                    let i = ((y * w2 + x) * 3) as usize;
                    expect[i..i + 3].copy_from_slice(&p);
                }
            }
            let (fw, fh) = framed_dims(w, h, "inset", pad, [0; 4]);
            assert_eq!((fw, fh), (ew, eh));
            let (cx0f, cy0f) = (fw as f32 / 2.0, fh as f32 / 2.0);
            let (pl, pt) = (pad[0] as i64, pad[1] as i64);
            const PAPER_BG: [f32; 3] = [246.0, 243.0, 236.0];
            let mut got = vec![0.0f32; (w2 * h2 * 3) as usize];
            for y in 0..h2 {
                for x in 0..w2 {
                    let dx = x as f32 - cx1;
                    let dy = y as f32 - cy1;
                    let fx = cx0f + dx * cst + dy * snt;
                    let fy = cy0f - dx * snt + dy * cst;
                    let p = if fx >= 0.0 && fx < fw as f32 && fy >= 0.0 && fy < fh as f32 {
                        sample_remapped(fx, fy, |ix, iy| {
                            if ix < 0 || iy < 0 || ix >= fw as i64 || iy >= fh as i64 {
                                PAPER_BG
                            } else {
                                let jx = ix - pl;
                                let jy = iy - pt;
                                if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                                    DESK
                                } else {
                                    let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                                    [buf[j], buf[j + 1], buf[j + 2]]
                                }
                            }
                        })
                    } else {
                        DESK
                    };
                    let i = ((y * w2 + x) * 3) as usize;
                    got[i..i + 3].copy_from_slice(&p);
                }
            }
            assert_eq!(got, expect, "deg {deg}: fused inset must be bit-exact");
        }
    }

    #[test]
    fn framing_in_rotation_tight_differs_only_inside_rim() {
        let (w, h) = (48u32, 36u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v = if (x + y) % 3 == 0 { 20.0 } else { 240.0 };
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
            }
        }
        let cut = [4u32, 5, 3, 6];
        let deg = 5.0f32;
        let (mut eb, ew, eh) = apply_frame_cut(&buf, w, h, cut);
        let th = deg.to_radians();
        let (sn, cs) = (th.sin().abs(), th.cos().abs());
        let (w2, h2) = (
            ((ew as f32) * cs + (eh as f32) * sn).ceil().max(1.0) as u32,
            ((ew as f32) * sn + (eh as f32) * cs).ceil().max(1.0) as u32,
        );
        const DESK: [f32; 3] = [25.0, 25.0, 30.0];
        let src = std::mem::take(&mut eb);
        let (cx0, cy0) = (ew as f32 / 2.0, eh as f32 / 2.0);
        let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
        let (snt, cst) = (th.sin(), th.cos());
        let mut expect = vec![0.0f32; (w2 * h2 * 3) as usize];
        let mut interior = vec![false; (w2 * h2) as usize];
        for y in 0..h2 {
            for x in 0..w2 {
                let dx = x as f32 - cx1;
                let dy = y as f32 - cy1;
                let sx = cx0 + dx * cst + dy * snt;
                let sy = cy0 - dx * snt + dy * cst;
                let inside = sx >= 0.0 && sx < ew as f32 && sy >= 0.0 && sy < eh as f32;
                let p = if inside {
                    sample_bilinear(&src, ew, eh, sx, sy)
                } else {
                    DESK
                };
                let i = ((y * w2 + x) * 3) as usize;
                expect[i..i + 3].copy_from_slice(&p);
                interior[(y * w2 + x) as usize] = inside
                    && sx >= 1.0
                    && sy >= 1.0
                    && sx < ew as f32 - 1.0
                    && sy < eh as f32 - 1.0;
            }
        }
        let (fw, fh) = framed_dims(w, h, "tight", [0; 4], cut);
        assert_eq!((fw, fh), (ew, eh));
        let (cl, ct) = (cut[0] as i64, cut[1] as i64);
        const PAPER_BG: [f32; 3] = [246.0, 243.0, 236.0];
        let mut got = vec![0.0f32; (w2 * h2 * 3) as usize];
        for y in 0..h2 {
            for x in 0..w2 {
                let dx = x as f32 - cx1;
                let dy = y as f32 - cy1;
                let fx = cx0 + dx * cst + dy * snt;
                let fy = cy0 - dx * snt + dy * cst;
                let p = if fx >= 0.0 && fx < fw as f32 && fy >= 0.0 && fy < fh as f32 {
                    sample_remapped(fx, fy, |ix, iy| {
                        let jx = ix + cl;
                        let jy = iy + ct;
                        if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                            PAPER_BG
                        } else {
                            let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                            [buf[j], buf[j + 1], buf[j + 2]]
                        }
                    })
                } else {
                    DESK
                };
                let i = ((y * w2 + x) * 3) as usize;
                got[i..i + 3].copy_from_slice(&p);
            }
        }
        for (k, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            if interior[k / 3] {
                assert_eq!(g, e, "interior pixel {k} must be bit-exact");
            }
        }
        let same = got
            .iter()
            .zip(expect.iter())
            .filter(|(g, e)| g == e)
            .count();
        assert!(
            same as f64 / got.len() as f64 > 0.8,
            "tight fused must agree on >80% of values, got {same}/{}",
            got.len()
        );
    }
}
