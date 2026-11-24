# Changelog

All notable changes to `doc-photo-degrade` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-24

Hyper-realism, physical paper-texture engines, independent per-corner detachment, and smartphone ISP release.

### Added

- **Independent Per-Corner Detachment (`CornerDetachConfig`, `QuadSdf2D`, `compute_detached_corners`, `visible_page_quads_out`):**
  - Each of the 4 page corners independently detaches inward (`"inset"` revealing desk) or outward (`"offset"`, binary-searched against exact projective content safety so document ink never clips).
  - Exact 2-Quad Output SDF (`QuadSdf2D` on `Q_out` & `R_out`) unifies slanted detached desk texture, contact drop-shadow, and `OUT_OF_PAGE` ground-truth mask parity (`< 0.2 px` rendered centroid error preserved).
  - CLI flags `--no-corner-detach` and `--corner-inset-prob`, builder methods `.corner_detach(...)` and `.corner_detach_inset_prob(...)`, and `DegradeParams` fields `corner_detach_px` and `corner_detach_modes`.
- **5 Physical Paper-Texture & Micro-Surface Engines (`paper.rs`, `banks.rs`, `realism.rs`):**
  - **256×256 Torus Micro-Weave Bank (`WeaveStock`):** seamless laid (chain+laid wire ribs), wove, and cold-press watercolor paper grain calibrated per profile.
  - **Y-Sorted Cellulose Fibers, Recycled Pulp Flecks & Lint (`PulpInclusionConfig`):** cache-friendly curved Bezier cellulose/rag fibers, L1-resident 7×7 wood-chip flecks, and stray surface lint with scanner-glass micro-shadows.
  - **Archival Chemical Aging (`apply_archival_paper`):** O(W+H) edge-oxidation LUT, organic ferric-oxide foxing clusters, translucent corner finger-oil handling grease, and subtle papermaker watermarks.
  - **Micro-Surface BRDF (`PaperStock`):** stock-specific albedo ceiling, Dmax black floor, OBA fluorescence, velvet highlight rolloff, and zero-centroid-shift capillary ink spread (`CoatedGlossy`, `ThermalReceipt`, `MatteOfficeBond`, `PorousNewsprint`, `UncoatedRag`), plus `.paper_stock(...)` builder override.
  - **Fused 3D Micro-Relief (`apply_micro_relief_shading`):** 6-octave phasor 3D paper cockling (3–5× faster than the legacy `4×cos` crumple loop) + 256×256 raking-light paper-tooth bump (ink-core protected) + intaglio plate-mark & book-gutter emboss.
- **Camera Optics, Smartphone ISP, Print/Scan Physics & Environment (CPU + WGPU Shader Fusion):**
  - Handheld phone + hand cast shadow SDF (with cool skylight fill) and anisotropic glossy specular bloom + C1 highlight rolloff fused into Stage 5 lighting (CPU + WGPU `LIGHTING_SHADER`).
  - 3D bend-coupled Lambertian normal shading fused into Stage 2 bend (CPU + WGPU `BEND_SHADER`).
  - Off-axis cubic radial chromatic aberration and Petzval field-curvature corner softness (CPU + WGPU `CA_SHADER` + `CHAIN_BLUR`).
  - Filmic smartphone auto-exposure & S-curve 3×256 tone-mapping LUT (`vec_to_image_with_lut`), 1/4-res High-ISO Cb/Cr chroma blotch noise (CPU + WGPU `NOISE_SHADER`), unrolled edge-aware HDR USM, and optional shifted double-JPEG (`jpeg_roundtrip_shifted`, 0.00 px drift).
  - Capillary coffee/water tide-ring stains with ink bleed, dihedral fold creases with ink-flake cracking, Xerox fuser drum ghosting (8×8 tile-skipped), thermal receipt uneven fade & heat-friction scratches, flatbed scanner lid edge bar + CCD dust lines, and ink-only micro-texture / halftone grain.
  - Richer procedural LUT desks (`DeskLut`: `walnut`, `oak`, `granite_laminate`, `slate`, `cloth`), optional margin-safe music-stand bottom ledge (`STAND_LEDGE`), margin-safe fingertip / binder-clip edge occluder (`FINGER`, `BINDER_CLIP`), serrated translucent scotch tape v2 with diagonal specular sheen, and depth-shadowed desk-through binder punch holes.
- **Full v0.2 JSON Backward Compatibility:** all new `DegradeParams` and `CaptureParams` fields use `#[serde(default)]`.

## [0.2.0] - 2026-11-24

Major accuracy, multi-task supervision, and photorealism release.

### Added

- **Exact Non-Linear Page-Bend Reprojection (`< 0.2 px` rendered accuracy, `< 1e-9 px` algebraic round-trip):**
  - `bend_backward`, `bend_forward_exact` (exact closed-form Citardauq quadratic for edge curls, 1D Newton-Raphson for book-spine gutter, 2D Newton-Raphson with analytic 2×2 Jacobian for hand-held ripple).
  - `transform_point_exact`, `unproject_point_exact -> Option<(f64, f64)>`, `transform_bbox_exact`, and `transform_polygon_exact` with exact pixel-center downscale ratios (`params.width / w_rot`).
- **Multi-Task Ground-Truth Mask Export (`masks` module + `--export-masks` / `--mask-format`):**
  - Zero-overhead opt-in `MaskRequest` (`uv_flow`, `illumination`, `clean_ink`, `artifacts`, `soft_ink`) and 11-bit `artifact_bits` bitmask (`OUT_OF_PAGE`, `TEAR`, `STAIN`, `PENCIL`, `FOLD_CREASE`, `HIGHLIGHTER`, `GLARE`, `TAPE`, `HOLE`, `STICKY_NOTE`, `FINGERPRINT`).
  - Zero-dependency NumPy `.npy` v1.0 (`<f4`) writer (`write_npy_f32`), 16-bit RGB UV PNG writer (`write_uv16_png`), illumination PNG writer (`write_illum_png`), and atomic `DegradeMasks::write_to_dir`.
  - Builder support via `DegraderBuilder::export_masks(MaskRequest)`.
- **Wave 2 Photorealism (`realism` module + WGPU shader fusion, zero extra GPU passes):**
  - Mixed-illuminant warm-to-cool AWB color-temperature gradient folded into Stage 5 lighting (CPU + WGPU; `awb_warm_cool_strength`).
  - Linear-light luminance-dependent Poisson-Gaussian sensor noise folded into Stage 6 noise (CPU + WGPU).
  - Smartphone ISP HDR unsharp-mask (USM) micro-contrast haloing (`usm_strength`), calibrated per `DocumentProfile`.
  - Procedural desk surfaces (`walnut`, `slate`, `cloth`) + directional page-perimeter contact drop-shadow (`desk_surface`).
  - Multi-scale 3D paper crumpling normal shading (`crumple_strength`, photometric-only — preserves `< 0.2 px` exact reprojection).
  - Occasional plastic sheet-protector / laminated diagonal specular sheen band (`sheen_strength`, also tagged in `artifact_bits::GLARE`).
- **True `4:2:0` JPEG Chroma Subsampling & Parallel `Degrader::degrade_batch`:**
  - `codecs::encode_jpeg` now uses `jpeg_encoder::SamplingFactor::F_2_2` (`4:2:0`) matching real smartphone JPEG encoders.
  - `Degrader::degrade_batch` now executes in parallel across CPU cores via `std::thread::scope`, preserving input order and bit-identical per-item determinism.
- **README Visual Examples:**
  - Added rendered visual showcase of all 7 built-in document profiles plus ground-truth illumination and clean-ink mask previews in `docs/images/`.

## [0.1.0] - 2026-11-24

Initial public release of `doc-photo-degrade` and `doc-photo-degrade-gpu`.

### Added

- **10-Stage Document-to-Phone-Photo Physics Pipeline (`doc-photo-degrade`):**
  - Paper grain, warm tint, speckles, and paper aging (yellowing, mottling, aged edge bands, foxing spots, fibers, cockling, water damage).
  - Ink & print degradation (uneven fading, dropout clusters, broken strokes, bleed-through ghosting, toner patches, ink browning).
  - Tear-free page bend: edge curl (all 4 edges), strictly monotonic book-spine gutter (`spine_map`), and hand-held ripple.
  - Margin-safe perspective homography (0.4% bleed cap with `warp_fit_scale` / `warp_out_px`), photo framing (`inset` desk padding or `tight` crop clamped to `quiet_margins` via `margin_safe_crop`), and canvas-expanding rotation (`±10°`).
  - Ink-aware physical artifacts (margin tears, coffee rings, pencil marks, folded corners, translucent highlighter bands, fold creases).
  - Heavy wear (washout zones, printer banding, streaks, toner bands, misregistration, eraser abrasion, water/oil stains).
  - Uniform & tilt-shift defocus blur, specular glare / directional shadow / vignette lighting, chromatic aberration, capture & handling artifacts (exposure, sensor banding, motion blur, fingerprints, partial-span tape, residue, binder holes, sticky notes), sensor noise, capture downscale, and SIMD JPEG round-trip.
- **High-Level Builder & Domain Profiles:**
  - `Degrader::builder()` API and 7 built-in `DocumentProfile` presets (`phone-photo`, `archival-manuscript`, `thermal-receipt`, `office-scan`, `photocopy-xerox`, `book-gutter-bend`, `sheet-music`).
  - `Preset::Full`, `Preset::Reduced`, and `StageToggles` (including `--fast-banks` seeded 256×256 tiled Gaussian banks and `--train-width` resolution-scaled degradation).
- **WGPU Compute Backend (`doc-photo-degrade-gpu`, optional `gpu` feature):**
  - Two-segment resident single-submit compute chains (`GpuContext::chain_full`) splitting cleanly at CPU margin-safe framing.
  - Fused terminal rotate + capture-downscale affine kernel (`AFFINE_SHADER`) eliminating an intermediate rotated canvas and host round-trip.
  - Fallible `GpuContext::try_new()` with automatic CPU fallback and whole-call `Mutex<GpuContext>` serialization for contention-free multi-threaded batches.
- **Provenance, Reprojection & Fast I/O:**
  - Full `DegradeParams` JSON sidecar emitted alongside every degraded image (`backend`, `preset`, `rot_scale_fused`, geometry, lighting, and per-stage parameters).
  - Fast codecs (`zune-png`, `zune-jpeg`, SIMD `jpeg-encoder`, `fast_image_resize`) and crash-safe atomic `*.tmp` → rename writes.
- **CLI & Runnable Examples:**
  - Multi-threaded `doc-photo-degrade` CLI with deterministic per-file seed striding.
  - Runnable examples (`01_quick_degrade.rs`, `02_strength_sweep.rs`, `03_bench_throughput.rs`, `04_reproject_bboxes.rs`) covering quickstart usage, strength sweeps, throughput benchmarking, and bounding-box reprojection.
- **Dual License:** Released under `MIT OR Apache-2.0`.
