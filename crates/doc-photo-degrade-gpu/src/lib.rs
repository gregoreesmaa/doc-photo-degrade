//! WGPU compute backend for `doc-photo-degrade`.
//!
//! Executes the dense per-pixel stages (paper grain, page bend, perspective
//! warp, rotation, defocus/tilt-shift blur, lighting, chromatic aberration,
//! sensor noise, and fused rotate+downscale affine resampling) on the GPU
//! via a single-submit resident ping-pong chain.

pub mod chain;
pub mod gpu;
