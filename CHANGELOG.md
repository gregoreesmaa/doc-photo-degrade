# Changelog

All notable changes to `doc-photo-degrade` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
