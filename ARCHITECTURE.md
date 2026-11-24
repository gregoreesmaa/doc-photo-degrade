# Architecture

This document explains the internal structure of `doc-photo-degrade`: how the workspace crates split responsibilities, how the two-segment resident WGPU compute chains execute, why `Mutex<GpuContext>` serializes whole device round-trips, how margin-safe geometry preserves document content, and how the deterministic PRNG contract works across backends.

---

## 1. Workspace & Crate Split

The workspace (`Cargo.toml`) is organized into two focused crates so downstream users can opt out of GPU dependencies (`default-features = false`) while keeping WGSL shader code and `wgpu` lifetimes completely isolated from document-domain logic:

```text
doc-photo-degrade/
├── crates/
│   ├── doc-photo-degrade/          # Library (`doc_photo_degrade`) + CLI (`doc-photo-degrade`)
│   │   └── src/
│   │       ├── lib.rs              # Pipeline orchestrator, DegradeConfig, DegradeParams, presets
│   │       ├── main.rs             # Multi-threaded CLI driver + GPU warmup
│   │       ├── geometry.rs         # Margin-safe bbox/crop/pad, homography, spine map, ink grid
│   │       ├── paper.rs            # Paper aging (yellowing, mottling, foxing, fibers, cockle, water)
│   │       ├── ink.rs              # Ink wear (fading, dropout, broken strokes, bleed-through, toner)
│   │       ├── wear.rs             # Heavy wear (washout, banding, streaks, misregistration, eraser, stains)
│   │       ├── capture.rs          # Capture/handling (exposure, sensor banding, motion blur, tape, holes)
│   │       ├── banks.rs            # Seeded 256×256 tiled Gaussian banks + precomputed lighting/stamps
│   │       ├── codecs.rs           # zune-png/zune-jpeg decode, SIMD jpeg-encoder, fast_image_resize, atomic IO
│   │       └── gpu_backend.rs      # Feature-gated (`gpu`) bridge wrapping Mutex<GpuContext>
│   └── doc-photo-degrade-gpu/      # Standalone WGPU compute crate (`doc_photo_degrade_gpu`)
│       └── src/
│           ├── lib.rs              # Public re-exports
│           ├── gpu.rs              # GpuContext::try_new(), WGSL compute kernels, single-stage + chain runners
│           └── chain.rs            # ChainConfig, tilt-capable Gaussian blur shader, rotation frame math
```

### Feature Gate

`doc-photo-degrade` enables `gpu = ["dep:doc-photo-degrade-gpu", "dep:pollster"]` by default. Compiling with `--no-default-features` strips `wgpu`, `pollster`, and `doc-photo-degrade-gpu` completely, producing a pure-CPU binary and library.

---

## 2. Two-Segment Resident WGPU Chains

Running each dense per-pixel stage as its own GPU upload → dispatch → readback cycle wastes wall time on PCIe transfers and staging-buffer maps (measured at ~5.9 s for 10 full-res pages). Instead, `GpuContext::chain_full` executes arbitrary subsets of dense stages **resident in device memory** inside a single `wgpu::CommandEncoder` submission, ping-ponging between device storage buffers.

On the GPU path, each page executes **two resident chain submissions** (plus an initial `grain` pass when grain is enabled):

1. **Segment 1 (Pre-Framing):** `run_bend` → `run_persp`
   - Runs page bend (edge curl, book-spine gutter, or hand-held ripple) and inverse-homography perspective warp in one submit.
   - Reads back to host so CPU framing can inspect the warped page.
2. **CPU Mid-Pipeline (Framing + Sparse Physical Effects):**
   - Margin-safe framing (`content_bbox` → `quiet_margins` → desk inset or clamped tight crop).
   - Ink-aware physical artifacts (tears, coffee ring, pencil strokes, folded corner, highlighter, fold creases).
   - Heavy wear (`wear::apply`) and capture/handling defects (`capture::apply`).
3. **Segment 2 (Post-Framing):** `run_blur` (uniform or tilt-shift) → `run_light` → `run_ca` → `run_noise` → `run_affine`
   - All optical, lighting, chromatic aberration, sensor noise, and final geometric stages execute in one resident submit without intermediate host readbacks.

### Why Split at Framing?

The framing stage must compute `content_bbox` (dark document ink vs. clean paper fill `[246, 243, 236]`) **after** page bend and perspective warp have moved ink, but **before** rotation introduces dark desk triangles (`[25, 25, 30]`) at the corners. If dark desk pixels were already present, a naive dark-pixel bounding box would latch onto the desk corners instead of the document content.

### Fused Terminal `run_affine` (Deferred Rotation + Capture Downscale)

On the CPU path, rotation runs eagerly after framing and expands the canvas to `(w2, h2)`, and capture downscale runs later via `fast_image_resize`. On the GPU path, rotation is **deferred** (`rot_stash`) and fused with capture downscale into the terminal `AFFINE_SHADER` dispatch at the end of Segment 2:

- The virtual rotated canvas `(w2, h2)` is **never materialized** in memory.
- Per destination pixel `(x, y)` in `(nw, nh)`, the kernel evaluates the composed inverse map $M^{-1} = R^{-1} \circ S^{-1}$:
  $$x_r = x \cdot s_x^{-1}, \qquad y_r = y \cdot s_y^{-1}$$
  If $(x_r, y_r) \notin [0, w_2) \times [0, h_2)$, it outputs `DESK_FILL`; otherwise it applies the 2×3 inverse rotation rows $R^{-1}$ and performs a manual 4-tap bilinear sample (no hardware texture sampler, keeping filtering bit-specified and deterministic; no perspective divide because row 2 is `[0, 0, 1]`).
- `rot_scale_fused: true` is recorded in `DegradeParams`.

---

## 3. Whole-Call `Mutex<GpuContext>` Rationale

Although `wgpu::Device`, `wgpu::Queue`, and `wgpu::ComputePipeline` are `Send + Sync`, sharing a single device concurrently across $N$ worker threads causes severe driver-level contention during interleaved `submit` / `map_async` / `poll(Wait)` traffic.

- **Measured on 10-thread full-res directory batches:** concurrent unserialized GPU calls inflated wall time **~10×** (30.9 s wall vs. 2.3 s serialized; individual calls stay ~70 ms).
- **Design:** `doc_photo_degrade::Gpu` wraps `GpuContext` in `std::sync::Mutex<GpuContext>`. `Gpu::lock()` holds the mutex for one complete device round-trip (e.g., one `chain_full` call).
- **Result:** All $N$ worker threads continue executing CPU stages (PNG/JPEG decode, paper aging, ink degradation, physical artifacts, heavy wear, capture defects, SIMD JPEG encode, atomic disk writes) in full parallel while GPU submissions queue cleanly with zero contention. Because every file uses an independent seed, lock ordering has **zero effect** on output pixels.

`Gpu::new()` calls `pollster::block_on(GpuContext::try_new())`, returning `Result<Self, String>` with no panics on headless or GPU-less hosts.

---

## 4. Margin-Safe Geometry Math

A core invariant of `doc-photo-degrade` is that **document content never leaves the frame** due to geometric augmentation or cropping:

### 4.1 Tear-Free Book-Spine Gutter (`spine_map`)
In `geometry.rs`, the spine gutter compresses horizontal coordinates smoothly toward `spine_cx` with width `spine_w` and `depth = ((curl_px / w) * 2.0).min(0.15)`:
$$t = \frac{x - c_x}{w_{\text{spine}}}, \qquad g = e^{-t^2}, \qquad s_x = c_x + (x - c_x)(1 - d \cdot g)$$
Because $0 \le d \le 0.15$, $\frac{ds_x}{dx} > 0$ everywhere — the map is continuous and strictly monotonic, so horizontal lines (text baselines, table borders, musical staves) compress naturally into the gutter without tearing or folding over.

### 4.2 Shrink-to-Fit Perspective Warp
Each of the 4 page corners is jittered by up to `perspective_max_frac = strength * 0.07`. Inward motion is preserved completely; outward motion is clamped per axis to a **0.4% quiet-margin bleed sliver**:
$$\text{bleed} = 0.004 \cdot \min(w, h)$$
The resulting scale factor (`warp_fit_scale`) and maximum outward excursion (`warp_out_px`) are recorded in `DegradeParams`. The forward homography is solved via 8×8 Gaussian elimination with partial pivoting (`solve_homography`) and inverted (`invert_3x3`) for inverse-mapped bilinear sampling.

### 4.3 Quiet-Margin Framing (`content_bbox` → `margin_safe_crop`)
1. **Content detection:** `content_bbox` locates the inclusive `[min_x, min_y, max_x, max_y]` box of dark ink pixels. On CPU, `BBoxAcc` folds each written pixel's channel sum directly inside the bend/perspective loop to avoid a standalone full-frame scan. When `train_width` rescaling is active (`rescale_k < 1.0`), the threshold rises from `128.0` to `200.0` (`content_bbox_thresh`) because bilinear pre-downscaling grays thin strokes.
2. **Quiet margins:** `quiet_margins(w, h, bbox)` computes `[left, top, right, bottom]` distances from frame edges to content (`[u32::MAX; 4]` on blank pages).
3. **Framing mode:**
   - **Desk inset (~70%):** Pads each side by `0.8%–4.5%` of `min(w, h)` with dark desk fill `[25, 25, 30]`.
   - **Tight crop (~30%):** Samples desired cuts in `0.4%–2.8%` of `min(w, h)` and clamps each edge via:
     $$\text{cut}_i = \min\!\big(\text{desired}_i,\; \text{quiet}_i \mathbin{\dot{-}} \text{safe}\big), \qquad \text{safe} = \max\!\big(\operatorname{round}(0.002 \cdot \min(w, h)),\; 3\big)$$
     Guaranteeing $\text{cut}_i + \text{safe} \le \text{quiet}_i$ whenever $\text{cut}_i > 0$.

### 4.4 Canvas-Expanding Rotation & Ink-Aware Artifacts
- Rotation ($\theta \in [-10^\circ, +10^\circ] \cdot \text{strength}$) expands the output canvas to:
  $$w_2 = \lceil w |\cos\theta| + h |\sin\theta| \rceil, \qquad h_2 = \lceil w |\sin\theta| + h |\cos\theta| \rceil$$
  so rotated corners are never truncated.
- Sparse physical artifacts (tears, coffee rings, pencil marks, dog-ear folds) query `ink_grid` (cell size `max(round(64 * rescale_k), 8)`) via `pick_quiet_cell` and only place opaque damage in low-ink regions (`< 6%–10%` ink density).

### 4.5 Coordinate Reprojection
Because `DegradeParams` records `src_width`, `src_height`, `rescale_k`, `corner_offsets`, `frame_mode`, `frame_pad_px`, `frame_cut_px`, `rotation_deg`, `scale_x`, `scale_y`, and final `width`/`height`, downstream training loaders can forward-project any source bounding box or polygon through `perspective → framing → rotation → capture downscale` via `transform_point_linear`, `transform_bbox`, and `transform_polygon` (exact when page bend is disabled, e.g. `OfficeScan`/`PhotocopyXerox` or `--skip-bend`; demonstrated in `examples/04_reproject_bboxes.rs`).

---

## 5. Deterministic PRNG Contract

### Batch Seed Stride
For directory inputs, file index `i` (sorted deterministically by `collect_inputs`) receives:
```rust
let file_seed = master_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
```
Per-file streams are strictly independent; thread count and execution order never affect output bytes.

### CPU Streams
- Primary pipeline RNG: `ChaCha8Rng::seed_from_u64(file_seed)`.
- Fast-banks mode (`--fast-banks`): `banks::bank_seed(file_seed, stage_tag)` runs a SplitMix64 avalanche over `TAG_GRAIN = 0x01` and `TAG_NOISE = 0x02` to seed independent `ChaCha8Rng` instances for 256×256 tiled Gaussian/uniform banks (256 KiB, L2-resident). Only the 2D tile offsets `(ox, oy)` are drawn from the primary stream.

### GPU Streams
- Dense WGSL hash kernels (paper grain and sensor noise) take 32-bit stream IDs derived from the 64-bit `file_seed`:
  - `noise_seed32(file_seed) = (file_seed ^ (file_seed >> 32)) as u32`
  - `grain_seed32(file_seed) = noise_seed32(file_seed ^ 0x9E37_79B9_7F4A_7C15)`
- Distinct stage tags guarantee that per-pixel paper grain and sensor noise never correlate.

### Documented Cross-Backend Differences
Semantics (gates, fills, and recorded knob ranges) match across CPU and GPU, with three intentional differences recorded in `DegradeParams`:
1. **Noise/grain streams:** GPU uses per-pixel WGSL hashes instead of the shared CPU `ChaCha8Rng`, shifting downstream draws (`"backend": "gpu"` vs `"cpu"`).
2. **Deferred rotation:** GPU defers rotation into the terminal fused `run_affine` pass (`"rot_scale_fused": true`), so intermediate sparse stages run unrotated before the final affine resample.
3. **Downscale filter:** CPU uses SIMD `fast_image_resize` bilinear resampling; GPU uses the single-sample 4-tap bilinear `AFFINE_SHADER`.

Each backend is **100% bit-exact deterministic** across repeated runs on the same backend.
