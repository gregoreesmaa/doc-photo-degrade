# Contributing to `doc-photo-degrade`

Thank you for your interest in contributing to `doc-photo-degrade`! We welcome bug reports, performance improvements, new document profiles, shader optimizations, and documentation fixes.

---

## Development Setup

### Prerequisites

- **Rust toolchain** (Edition 2021, stable `rustc` & `cargo`).
- **Optional GPU adapter** (Vulkan, DirectX 12, or Metal) to exercise live WGPU compute tests. On headless or GPU-less machines, GPU integration tests skip cleanly without failing.

### Verification Commands

Run the full check suite before opening a pull request:

```bash
# 1. Run all unit & integration tests across both crates (CPU + WGPU if available)
cargo test --workspace

# 2. Verify the pure-CPU build (no wgpu / pollster dependency)
cargo test -p doc-photo-degrade --no-default-features

# 3. Enforce zero Clippy warnings across all workspace targets
cargo clippy --workspace -- -D warnings

# 4. Check Rust formatting
cargo fmt --check
```

For benchmarking or processing real high-resolution pages, always build with `--release`:

```bash
cargo run --release -p doc-photo-degrade -- --input path/to/pages --out-dir out --format jpeg
```

Set `DOC_PHOTO_DEGRADE_TIMING=1` to print per-stage millisecond timings to `stderr`.

---

## Engineering Guardrails

When modifying the pipeline or adding stages, please preserve these core invariants:

1. **Per-Backend Determinism:** Every random draw must derive from the per-file `ChaCha8Rng` (CPU) or a uniquely tagged per-pixel hash stream (WGPU). Never use thread-local RNGs, wall-clock entropy, or `HashMap` iteration order in pixel generation.
2. **Margin-Safe Geometry:** Never allow geometric warps, crops, or opaque physical artifacts to clip or occlude document ink outside the documented 0.4% bleed budget or `margin_safe_crop` guarantee.
3. **Provenance Completeness:** Every new degradation knob must be serialized into `DegradeParams` (with `#[serde(default)]` or aliases when preserving backward compatibility with existing JSON sidecars).
4. **Zero-Panic GPU Fallback:** GPU initialization must remain fallible (`GpuContext::try_new()` / `Gpu::new()`). Never `.unwrap()` adapter or device creation on production paths.
5. **Atomic File Output:** All CLI file writes must go through `write_artifact` (`*.tmp` → `rename`) so interrupted runs never leave truncated images or JSON files.
6. **Permissive Licensing:** Keep all crate dependencies MIT/BSD/Apache-2.0 compatible. No GPL/AGPL tools or external binary dependencies.

---

## Pull Request Checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo test -p doc-photo-degrade --no-default-features` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] `cargo fmt --check` passes
- [ ] New parameters or CLI flags are documented in `README.md` / `ARCHITECTURE.md`
- [ ] Determinism and edge-case unit tests are included for new stages or geometry math

---

## Licensing

By contributing to `doc-photo-degrade`, you agree that your contributions will be dual-licensed under the **MIT License** ([LICENSE-MIT](LICENSE-MIT)) and **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE)).
