# Changelog

All notable changes to `doc-photo-degrade` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
