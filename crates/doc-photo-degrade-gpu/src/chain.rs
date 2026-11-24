//! Shared configuration, geometry helpers, and tilt-capable blur shader for
//! `GpuContext::chain_full` (resident single-submit execution on the shared
//! WGPU device).
//!
//! The `doc-photo-degrade` GPU path calls `GpuContext::chain_full` twice per
//! page:
//! 1. Pre-framing (`run_bend`, `run_persp`)
//! 2. Post-framing (`run_blur`, `run_light`, `run_ca`, `run_noise`, `run_affine`)
//!
//! Splitting at framing is required because the CPU framing stage must measure
//! the document content bounding box against paper fill before rotation adds
//! dark desk triangles. Fusing `run_affine` (deferred rotate + capture
//! downscale) onto the end of segment 2 eliminates an entire host round-trip
//! per page.

use super::gpu::{BendKnobs, LightingKnobs};

/// Tilt-capable Gaussian defocus blur shader used by both `GpuContext::blur`
/// (with `b.x == 0`, pure uniform blur) and `GpuContext::chain_full` (uniform
/// or tilt-shift smoothstep focus band).
pub(crate) const CHAIN_BLUR: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Blr { a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Blr;

fn tap(ix : i32, iy : i32) -> vec3<f32> {
    let cx = clamp(ix, 0, i32(dims.src_w) - 1);
    let cy = clamp(iy, 0, i32(dims.src_h) - 1);
    let i = (u32(cy) * dims.src_w + u32(cx)) * 3u;
    return vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let r = i32(prm.a.y);
    if (r <= 0) {
        let i = (y * dims.src_w + x) * 3u;
        dst[i] = src[i];
        dst[i + 1u] = src[i + 1u];
        dst[i + 2u] = src[i + 2u];
        return;
    }
    let s2 = 2.0 * prm.a.x * prm.a.x;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var norm = 0.0;
    for (var ky = -r; ky <= r; ky++) {
        for (var kx = -r; kx <= r; kx++) {
            let wgt = exp(-f32(kx * kx + ky * ky) / s2);
            acc += tap(i32(x) + kx, i32(y) + ky) * wgt;
            norm += wgt;
        }
    }
    let blurred = acc / norm;
    var out = blurred;
    let fc = prm.b.y;
    if (prm.b.x > 0.5 || fc > 0.0) {
        let wf = max(f32(dims.dst_w), 1.0);
        let hf = max(f32(dims.dst_h), 1.0);
        var m = select(0.45, 0.0, prm.b.x > 0.5);
        if (prm.b.x > 0.5) {
            let d = abs(f32(y) / hf - prm.a.z);
            let mt = clamp((d - prm.a.w) / 0.08, 0.0, 1.0);
            m = mt * mt * (3.0 - 2.0 * mt);
        }
        if (fc > 0.0) {
            let max_r = max(0.5 * length(vec2<f32>(wf, hf)), 1.0);
            let rn = vec2<f32>(f32(x) - wf * prm.b.z, f32(y) - hf * prm.b.w) / max_r;
            let t = clamp((dot(rn, rn) - 0.20) / 0.80, 0.0, 1.0);
            let m_fc = fc * t * t * (3.0 - 2.0 * t);
            m = select(clamp(0.45 + 0.55 * m_fc, 0.0, 1.0), max(m, m_fc), prm.b.x > 0.5);
        }
        let sharp = tap(i32(x), i32(y));
        out = mix(sharp, blurred, m);
    }
    let i = (y * dims.dst_w + x) * 3u;
    dst[i] = out.x;
    dst[i + 1u] = out.y;
    dst[i + 2u] = out.z;
}
"#;

/// Knobs for one resident-chain run on `GpuContext::chain_full`, mirroring the
/// `doc-photo-degrade` dense stage order:
/// grain -> bend -> perspective -> rotation -> blur -> lighting -> chromatic -> noise -> affine.
#[derive(Clone, Copy)]
pub struct ChainConfig {
    pub grain_sigma: f32,
    pub grain_seed: u32,
    pub grain_speckle_prob: f32,
    pub grain_strength: f32,
    pub grain_warm: [f32; 3],
    pub bend: BendKnobs,
    /// Row-major 3x3 inverse map for the perspective warp.
    pub persp_inv: [f32; 9],
    /// Out-of-bounds fill RGB for perspective warp (paper tone or desk tone when detached).
    pub persp_fill: [f32; 3],
    /// Rotation degrees (positive = clockwise in image coordinates).
    pub rot_deg: f32,
    pub blur_sigma: f32,
    /// Petzval field-curvature corner softness weight (`0.0` = off).
    pub field_curvature: f32,
    pub light: LightingKnobs,
    pub ca_px: f32,
    /// Decentered optical axis `[ox, oy]` in `[0..1]^2` for radial CA & field curvature.
    pub ca_optical_center: [f32; 2],
    /// Cubic radial CA weight (`0.0` = legacy linear, `0.75` = cubic wide-angle).
    pub ca_cubic_weight: f32,
    pub noise_sigma: f32,
    pub noise_shot_sigma: f32,
    /// 1/4-res High-ISO Cb/Cr chroma blotch noise sigma (`0.0` = off).
    pub chroma_noise_sigma: f32,
    pub noise_seed: u32,
    /// Per-stage run flags (false = skip pass; resident buffer untouched).
    pub run_grain: bool,
    pub run_bend: bool,
    pub run_persp: bool,
    pub run_rot: bool,
    pub run_blur: bool,
    pub run_light: bool,
    pub run_ca: bool,
    pub run_noise: bool,
    /// Tilt-shift focus blend for the blur pass (`focus_y`, `focus_h` in
    /// fractions of height). When false, blur is uniform.
    pub tilt_active: bool,
    pub focus_y: f32,
    pub focus_h: f32,
    /// Terminal fused rotate+downscale affine pass (same kernel as
    /// `GpuContext::affine`, desk-tone fill). Runs after noise in the same
    /// command encoder/submit, eliminating an extra host round-trip.
    pub run_affine: bool,
    pub affine_rows: [f32; 6],
    pub affine_meta: [f32; 4],
    pub affine_nw: u32,
    pub affine_nh: u32,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            grain_sigma: 0.0,
            grain_seed: 0,
            grain_speckle_prob: 0.0,
            grain_strength: 0.0,
            grain_warm: [0.0; 3],
            bend: BendKnobs::default(),
            persp_inv: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            persp_fill: [246.0, 243.0, 236.0],
            rot_deg: 0.0,
            blur_sigma: 0.0,
            field_curvature: 0.0,
            light: LightingKnobs::default(),
            ca_px: 0.0,
            ca_optical_center: [0.5, 0.5],
            ca_cubic_weight: 0.0,
            noise_sigma: 0.0,
            noise_shot_sigma: 0.0,
            chroma_noise_sigma: 0.0,
            noise_seed: 0,
            run_grain: false,
            run_bend: false,
            run_persp: false,
            run_rot: false,
            run_blur: false,
            run_light: false,
            run_ca: false,
            run_noise: false,
            tilt_active: false,
            focus_y: 0.0,
            focus_h: 0.0,
            run_affine: false,
            affine_rows: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            affine_meta: [0.0; 4],
            affine_nw: 0,
            affine_nh: 0,
        }
    }
}

/// Canvas expansion for `rot_deg`: smallest ceil canvas that fits the rotated
/// page.
pub(crate) fn rotation_dims(rot_deg: f32, w: u32, h: u32) -> (u32, u32) {
    let th = rot_deg.to_radians();
    let (sn, cs) = (th.sin().abs(), th.cos().abs());
    (
        ((w as f32) * cs + (h as f32) * sn).ceil().max(1.0) as u32,
        ((w as f32) * sn + (h as f32) * cs).ceil().max(1.0) as u32,
    )
}

/// Inverse map (dst -> src) for the expanded rotation canvas: positive degrees
/// rotate clockwise in image coordinates.
pub(crate) fn rotation_inv(rot_deg: f32, w: u32, h: u32, w2: u32, h2: u32) -> [f32; 9] {
    let th = rot_deg.to_radians();
    let (snt, cst) = (th.sin(), th.cos());
    let (cx0, cy0) = (w as f32 / 2.0, h as f32 / 2.0);
    let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
    [
        cst,
        snt,
        cx0 - cst * cx1 - snt * cy1,
        -snt,
        cst,
        cy0 + snt * cx1 - cst * cy1,
        0.0,
        0.0,
        1.0,
    ]
}
