# AGENTS.md — `doc-photo-degrade`

Deterministic, margin-safe document-to-phone-photo degradation engine in Rust with optional WGPU compute acceleration. Turns clean document PNG/JPEG renders into photorealistic phone captures paired with exact `DegradeParams` JSON sidecars and ground-truth coordinate reprojection.

---

## Workspace Layout

Two-crate Cargo workspace (`resolver = "2"`, Rust 2021, `MIT OR Apache-2.0`). `doc-photo-degrade` enables `gpu = ["dep:doc-photo-degrade-gpu", "dep:pollster"]` by default; `--no-default-features` strips WGPU completely for pure-CPU builds.

| Path | Responsibility |
|---|---|
| `Cargo.toml` | Workspace manifest and shared dependency versions |
| `crates/doc-photo-degrade/src/lib.rs` | 10-stage pipeline orchestrator, `DegradeConfig`, `DegradeParams`, `Preset`, `StageToggles`, `BackendPreference` |
| `crates/doc-photo-degrade/src/main.rs` | Multi-threaded `doc-photo-degrade` CLI driver, profile/preset flag parsing, shared GPU warmup |
| `crates/doc-photo-degrade/src/degrader.rs` | Ergonomic `Degrader` / `DegraderBuilder`, 7 `DocumentProfile` presets, `DegradeError`, batch helpers |
| `crates/doc-photo-degrade/src/reproject.rs` | Forward-projection of points, bounding boxes, and polygons from `DegradeParams` |
| `crates/doc-photo-degrade/src/geometry.rs` | Margin-safe `content_bbox`, `quiet_margins`, crop/pad framing, homography, tear-free `spine_map`, `ink_grid` |
| `crates/doc-photo-degrade/src/paper.rs` | Paper aging (yellowing, mottling, foxing spots, fibers, cockling, water damage) |
| `crates/doc-photo-degrade/src/ink.rs` | Ink wear (fading, dropout, broken strokes, bleed-through, toner variation) |
| `crates/doc-photo-degrade/src/wear.rs` | Heavy wear (washout, printer banding, streaks, misregistration, eraser smudges, stains) |
| `crates/doc-photo-degrade/src/capture.rs` | Capture/handling defects (exposure, sensor banding, motion blur, tape, punch holes) |
| `crates/doc-photo-degrade/src/banks.rs` | Seeded 256×256 tiled Gaussian banks and precomputed lighting/stamp fields (`--fast-banks`) |
| `crates/doc-photo-degrade/src/codecs.rs` | SIMD `zune-png` / `zune-jpeg` decode, `jpeg-encoder`, `fast_image_resize`, atomic `write_artifact` |
| `crates/doc-photo-degrade/src/gpu_backend.rs` | Feature-gated (`gpu`) wrapper holding `Mutex<GpuContext>` and two-segment resident chain dispatch |
| `crates/doc-photo-degrade-gpu/src/lib.rs` | Public re-exports for the standalone WGPU compute crate (`doc_photo_degrade_gpu`) |
| `crates/doc-photo-degrade-gpu/src/gpu.rs` | Fallible `GpuContext::try_new()`, WGSL compute shaders, single-stage + resident chain runners, `rotscale_frame` |
| `crates/doc-photo-degrade-gpu/src/chain.rs` | `ChainConfig`, tilt-capable Gaussian blur shader, fused rotate+downscale affine stage wiring |
| `crates/doc-photo-degrade/examples/` | Runnable API demos (`01_quick_degrade.rs`, `02_strength_sweep.rs`, `03_bench_throughput.rs`, `04_reproject_bboxes.rs`) |
| `crates/doc-photo-degrade/tests/` | Public-API integration tests (`integration_public_api.rs`) |
| `.github/workflows/ci.yml` | GitHub Actions CPU CI (fmt, clippy, CPU-only tests, example compilation) |

---

## Verification Commands

Run these before declaring any change complete:

```bash
# Formatting
cargo fmt --check

# Lint (all workspace targets, zero warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Full workspace test suite (caps GPU/thread contention)
cargo test --workspace -- --test-threads=4

# Pure-CPU build and tests (verifies `gpu` feature gate stays clean)
cargo test -p doc-photo-degrade --no-default-features

# Doctests across workspace
cargo test --doc --workspace

# Rustdoc with warnings denied (bash/sh)
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --no-default-features
```

On Windows PowerShell, set `RUSTDOCFLAGS` for the doc check with:

```powershell
$env:RUSTDOCFLAGS="-D warnings"; cargo doc --workspace --no-deps --no-default-features; Remove-Item Env:RUSTDOCFLAGS
```

---

## Hard Invariants (Do Not Break)

1. **Per-backend determinism**
   - Every file in a batch derives an independent seed via `base_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))` (sorted input order). Never mix thread scheduling, completion order, or global mutable state into any RNG.
   - **CPU backend:** uses seeded `ChaCha8Rng` (and SplitMix64-avalanched `ChaCha8Rng` streams for `--fast-banks`).
   - **GPU backend:** dense paper-grain and sensor-noise kernels use deterministic WGSL hash streams with distinct stream IDs (`grain_seed32` vs `noise_seed32`).
   - Each backend is bit-identical across repeated runs on the same backend; cross-backend pixel differences are intentional and recorded in `params.backend` (`"cpu"` vs `"gpu"`) and `params.rot_scale_fused`.

2. **Margin-safe & ink-aware geometry**
   - Document content must never be clipped out of frame.
   - **Perspective & barrel distortion:** inward corner motion is kept whole; outward excursion is clamped to a **`0.4%` quiet bleed** (`0.004 * min(w, h)`), with `warp_fit_scale` / `warp_out_px` / `dist_fit_scale` recorded.
   - **Framing:** tight crops are clamped per edge to `quiet_margin - safe_px` (`safe_px = max(round(0.002 * min(w, h)), 3)`); framing runs *after* bend/perspective and *before* rotation so dark desk triangles never corrupt `content_bbox`.
   - **Rotation:** expands the canvas to `ceil(w|cos θ| + h|sin θ|) × ceil(w|sin θ| + h|cos θ|)` with dark desk fill `[25, 25, 30]`.
   - **Sparse destructive artifacts:** tears, coffee rings, pencil marks, and dog-ear folds query `ink_grid` and avoid high-ink cells.

3. **Fallible GPU init & whole-call `Mutex<GpuContext>`**
   - Always initialize WGPU via `GpuContext::try_new()` / `Gpu::new()` returning `Result` — **never panic** on headless or GPU-less machines/CI. Auto mode falls back cleanly to CPU; explicit `--gpu` returns a structured error.
   - Keep `std::sync::Mutex<GpuContext>` serializing each complete WGPU `submit`/`map_async`/`poll` round-trip in `gpu_backend.rs`. Unserialized concurrent submits across worker threads regress wall time **~10×** due to driver contention, while whole-call serialization lets CPU stages run in parallel with zero pixel-output impact.

4. **Atomic file writes & paired JSON sidecars**
   - Always write output images and sidecars via `write_artifact` (`*.tmp` sibling + atomic rename) so interrupted runs never leave half-written files.
   - Every degraded image `<stem><suffix>.png|jpg` must be paired with `<stem><suffix>_params.json` recording all sampled knobs and geometric transforms.

5. **Permissive dependency stack (`MIT OR Apache-2.0`)**
   - All crates and dependencies must remain permissive (MIT/BSD/Apache). Never introduce GPL/AGPL tools, external system binary requirements, or CUDA-only hard dependencies.

---

## Release & Profiling Tips

- **Always use `--release` for benchmarks or real batches:** pixel loops are 10–50× slower in debug builds (~13 min debug sequential vs ~4.3 s release GPU / ~5.4 s release CPU for 10 × 2480 px pages).
- **Per-stage timing breakdown:** set `PHOTO_DEGRADE_TIMING=1` (or `DOC_PHOTO_DEGRADE_TIMING=1`) to print per-stage wall times from the pipeline in `lib.rs`.
- **High-throughput training knobs:** combine `--preset reduced` (skips fine grain, chromatic aberration, sensor noise; coarsens ink & paper aging), `--train-width <N>` (pre-downscales and rescales pixel stats by `k = train_width / src_width`), and `--fast-banks` (L2-resident 256×256 tiled noise banks on CPU).
- **Throughput benchmark example:** `cargo run --release -p doc-photo-degrade --example 03_bench_throughput`.
