//! High-level ergonomic builder API, domain document profiles, and structured
//! error type for `doc-photo-degrade`.
//!
//! # Example
//!
//! ```no_run
//! use doc_photo_degrade::degrader::{Degrader, DocumentProfile};
//! use doc_photo_degrade::BackendPreference;
//!
//! let degrader = Degrader::builder()
//!     .seed(42)
//!     .profile(DocumentProfile::PhonePhoto)
//!     .strength(0.6)
//!     .backend(BackendPreference::Auto)
//!     .build()?;
//!
//! let img = image::RgbImage::from_pixel(128, 128, image::Rgb([255, 255, 255]));
//! let result = degrader.degrade_image(&img, 42)?;
//! # Ok::<(), doc_photo_degrade::degrader::DegradeError>(())
//! ```

use std::fmt;
use std::str::FromStr;

use image::RgbImage;

use crate::{
    BackendPreference, DegradeConfig, DegradeResult, MaskRequest, Preset, RealismConfig,
    StageToggles,
};

#[cfg(feature = "gpu")]
use crate::gpu_backend::Gpu;

/// Golden-ratio 64-bit hash increment used to derive independent per-item seeds
/// in [`Degrader::degrade_batch`].
const GOLDEN_RATIO_U64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Structured error type returned by [`DegraderBuilder::build`] and
/// [`Degrader`] operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum DegradeError {
    /// A configuration field failed validation.
    InvalidConfig { field: &'static str, detail: String },
    /// Input image had zero width or zero height.
    EmptyImage,
    /// WGPU adapter/device initialization failed when GPU execution was required.
    GpuInit(String),
    /// An error occurred during image degradation.
    Processing(String),
    /// An I/O error occurred while reading or writing files.
    Io(std::io::Error),
}

impl fmt::Display for DegradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DegradeError::InvalidConfig { field, detail } => {
                write!(f, "invalid config for `{field}`: {detail}")
            }
            DegradeError::EmptyImage => write!(f, "input image is empty (width or height is 0)"),
            DegradeError::GpuInit(msg) => write!(f, "GPU initialization failed: {msg}"),
            DegradeError::Processing(msg) => write!(f, "degradation failed: {msg}"),
            DegradeError::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for DegradeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DegradeError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DegradeError {
    fn from(err: std::io::Error) -> Self {
        DegradeError::Io(err)
    }
}

/// Domain-specific document degradation profiles.
///
/// Each profile configures [`DegradeConfig`] (`strength`, [`StageToggles`],
/// `scale_min`, and `jpeg_quality`) to match the optical and physical
/// characteristics of a target document domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DocumentProfile {
    /// Handheld smartphone photograph of a paper document (default).
    #[default]
    PhonePhoto,
    /// Aged archival manuscript with heavy paper aging, foxing, and ink wear.
    ArchivalManuscript,
    /// Thermal paper receipt with faded print, low contrast, and handheld capture.
    ThermalReceipt,
    /// Clean flatbed office scanner capture with mild wear and no camera optics.
    OfficeScan,
    /// High-contrast photocopy / Xerox output with toner wear and no chromatic aberration.
    PhotocopyXerox,
    /// Open-book capture emphasizing gutter bend, shadow, and perspective.
    BookGutterBend,
    /// Sheet music capture tuned to preserve fine notation topology.
    SheetMusic,
}

impl DocumentProfile {
    /// All built-in document profiles.
    pub const ALL: &'static [DocumentProfile] = &[
        DocumentProfile::PhonePhoto,
        DocumentProfile::ArchivalManuscript,
        DocumentProfile::ThermalReceipt,
        DocumentProfile::OfficeScan,
        DocumentProfile::PhotocopyXerox,
        DocumentProfile::BookGutterBend,
        DocumentProfile::SheetMusic,
    ];

    /// Parse a profile from its canonical name (or common alias), case-insensitively.
    pub fn from_str_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "phone-photo" | "phone_photo" | "phonephoto" | "phone" => {
                Some(DocumentProfile::PhonePhoto)
            }
            "archival-manuscript"
            | "archival_manuscript"
            | "archivalmanuscript"
            | "archival"
            | "manuscript" => Some(DocumentProfile::ArchivalManuscript),
            "thermal-receipt" | "thermal_receipt" | "thermalreceipt" | "receipt" | "thermal" => {
                Some(DocumentProfile::ThermalReceipt)
            }
            "office-scan" | "office_scan" | "officescan" | "scan" | "office" => {
                Some(DocumentProfile::OfficeScan)
            }
            "photocopy-xerox" | "photocopy_xerox" | "photocopyxerox" | "photocopy" | "xerox" => {
                Some(DocumentProfile::PhotocopyXerox)
            }
            "book-gutter-bend" | "book_gutter_bend" | "bookgutterbend" | "book-gutter"
            | "book_gutter" | "book" => Some(DocumentProfile::BookGutterBend),
            "sheet-music" | "sheet_music" | "sheetmusic" | "music" => {
                Some(DocumentProfile::SheetMusic)
            }
            _ => None,
        }
    }

    /// Canonical kebab-case profile name.
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentProfile::PhonePhoto => "phone-photo",
            DocumentProfile::ArchivalManuscript => "archival-manuscript",
            DocumentProfile::ThermalReceipt => "thermal-receipt",
            DocumentProfile::OfficeScan => "office-scan",
            DocumentProfile::PhotocopyXerox => "photocopy-xerox",
            DocumentProfile::BookGutterBend => "book-gutter-bend",
            DocumentProfile::SheetMusic => "sheet-music",
        }
    }

    /// Apply this profile's defaults onto `cfg`, preserving `seed`, `max_width`,
    /// `train_width`, and `toggles.fast_banks`.
    pub fn apply_to_config(&self, cfg: &mut DegradeConfig) {
        let fast_banks = cfg.toggles.fast_banks;
        match self {
            DocumentProfile::PhonePhoto => {
                cfg.strength = 0.5;
                cfg.scale_min = 0.5;
                cfg.jpeg_quality = None;
                cfg.toggles = StageToggles::full();
                cfg.realism = RealismConfig::phone();
            }
            DocumentProfile::ArchivalManuscript => {
                cfg.strength = 0.75;
                cfg.scale_min = 0.65;
                cfg.jpeg_quality = None;
                cfg.toggles = StageToggles {
                    skip_grain: false,
                    skip_bend: false,
                    skip_chromatic: true,
                    skip_noise: false,
                    coarsen_ink: false,
                    coarsen_aging: false,
                    fast_banks,
                };
                cfg.realism = RealismConfig {
                    enable_awb: true,
                    enable_shot_noise: true,
                    enable_usm: false,
                    enable_desk_surface: true,
                    enable_crumple: true,
                    enable_sheen: false,
                };
            }
            DocumentProfile::ThermalReceipt => {
                cfg.strength = 0.70;
                cfg.scale_min = 0.55;
                cfg.jpeg_quality = None;
                cfg.toggles = StageToggles {
                    skip_grain: false,
                    skip_bend: false,
                    skip_chromatic: false,
                    skip_noise: false,
                    coarsen_ink: false,
                    coarsen_aging: true,
                    fast_banks,
                };
                cfg.realism = RealismConfig {
                    enable_awb: true,
                    enable_shot_noise: true,
                    enable_usm: true,
                    enable_desk_surface: true,
                    enable_crumple: true,
                    enable_sheen: false,
                };
            }
            DocumentProfile::OfficeScan => {
                cfg.strength = 0.20;
                cfg.scale_min = 0.85;
                cfg.jpeg_quality = Some(90);
                cfg.toggles = StageToggles {
                    skip_grain: false,
                    skip_bend: true,
                    skip_chromatic: true,
                    skip_noise: true,
                    coarsen_ink: true,
                    coarsen_aging: true,
                    fast_banks,
                };
                cfg.realism = RealismConfig::flat_scanner();
            }
            DocumentProfile::PhotocopyXerox => {
                cfg.strength = 0.65;
                cfg.scale_min = 0.75;
                cfg.jpeg_quality = None;
                cfg.toggles = StageToggles {
                    skip_grain: false,
                    skip_bend: true,
                    skip_chromatic: true,
                    skip_noise: false,
                    coarsen_ink: false,
                    coarsen_aging: true,
                    fast_banks,
                };
                cfg.realism = RealismConfig::flat_scanner();
            }
            DocumentProfile::BookGutterBend => {
                cfg.strength = 0.60;
                cfg.scale_min = 0.60;
                cfg.jpeg_quality = None;
                cfg.toggles = StageToggles::full();
                cfg.realism = RealismConfig {
                    enable_awb: true,
                    enable_shot_noise: true,
                    enable_usm: true,
                    enable_desk_surface: true,
                    enable_crumple: true,
                    enable_sheen: false,
                };
            }
            DocumentProfile::SheetMusic => {
                cfg.strength = 0.45;
                cfg.scale_min = 0.65;
                cfg.jpeg_quality = None;
                cfg.toggles = StageToggles {
                    skip_grain: false,
                    skip_bend: false,
                    skip_chromatic: false,
                    skip_noise: false,
                    coarsen_ink: true,
                    coarsen_aging: false,
                    fast_banks,
                };
                cfg.realism = RealismConfig::phone();
            }
        }
        cfg.toggles.fast_banks = fast_banks;
    }
}

impl fmt::Display for DocumentProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DocumentProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_name(s).ok_or_else(|| format!("unknown document profile `{s}`"))
    }
}

/// Reusable, thread-safe document degrader owning an optional shared WGPU
/// context and validated [`DegradeConfig`].
pub struct Degrader {
    config: DegradeConfig,
    profile: DocumentProfile,
    #[cfg(feature = "gpu")]
    gpu: Option<Gpu>,
}

impl fmt::Debug for Degrader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Degrader")
            .field("profile", &self.profile)
            .field("active_backend", &self.active_backend())
            .field("config", &self.config)
            .finish()
    }
}

impl Degrader {
    /// Create a new [`DegraderBuilder`] with sensible defaults
    /// (`DocumentProfile::PhonePhoto`, `BackendPreference::Auto`).
    pub fn builder() -> DegraderBuilder {
        DegraderBuilder::default()
    }

    /// Convenience constructor: default [`DocumentProfile::PhonePhoto`]
    /// configuration with `BackendPreference::Auto` (uses GPU when available,
    /// falls back cleanly to CPU otherwise).
    pub fn auto_gpu() -> Result<Self, DegradeError> {
        Self::builder().backend(BackendPreference::Auto).build()
    }

    /// Active compute backend: `"gpu"` or `"cpu"`.
    pub fn active_backend(&self) -> &'static str {
        #[cfg(feature = "gpu")]
        {
            if self.gpu.is_some() {
                "gpu"
            } else {
                "cpu"
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            "cpu"
        }
    }

    /// The [`DocumentProfile`] configured on this degrader.
    pub fn profile(&self) -> DocumentProfile {
        self.profile
    }

    /// Base [`DegradeConfig`] used for degradation calls.
    pub fn config(&self) -> &DegradeConfig {
        &self.config
    }

    /// Degrade a single in-memory RGB image using the given deterministic `seed`.
    pub fn degrade_image(&self, img: &RgbImage, seed: u64) -> Result<DegradeResult, DegradeError> {
        if img.width() == 0 || img.height() == 0 {
            return Err(DegradeError::EmptyImage);
        }
        let mut cfg = self.config.clone();
        cfg.seed = seed;
        #[cfg(feature = "gpu")]
        {
            crate::degrade_image_gpu(img.clone(), &cfg, self.gpu.as_ref())
                .map_err(DegradeError::Processing)
        }
        #[cfg(not(feature = "gpu"))]
        {
            crate::degrade_image(img.clone(), &cfg).map_err(DegradeError::Processing)
        }
    }

    /// Degrade a slice of images in parallel across available CPU cores using
    /// independent golden-ratio derived seeds:
    /// `base_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))`.
    ///
    /// Preserves exact input order and bit-identical per-item determinism
    /// regardless of thread scheduling.
    pub fn degrade_batch(&self, images: &[&RgbImage]) -> Result<Vec<DegradeResult>, DegradeError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let base_seed = self.config.seed;
        if images.len() == 1 {
            return Ok(vec![self.degrade_image(images[0], base_seed)?]);
        }
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1)
            .min(images.len());
        let chunk_size = images.len().div_ceil(n_threads);
        let mut chunk_results: Vec<Result<Vec<DegradeResult>, DegradeError>> =
            Vec::with_capacity(n_threads);

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n_threads);
            for (ci, chunk) in images.chunks(chunk_size).enumerate() {
                let base = ci * chunk_size;
                handles.push(scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for (j, img) in chunk.iter().enumerate() {
                        let i = base + j;
                        let item_seed =
                            base_seed.wrapping_add((i as u64).wrapping_mul(GOLDEN_RATIO_U64));
                        local.push(self.degrade_image(img, item_seed)?);
                    }
                    Ok(local)
                }));
            }
            for h in handles {
                chunk_results.push(h.join().unwrap_or_else(|_| {
                    Err(DegradeError::Processing("worker thread panicked".into()))
                }));
            }
        });

        let mut out = Vec::with_capacity(images.len());
        for res in chunk_results {
            out.extend(res?);
        }
        Ok(out)
    }
}

/// Fluent builder for [`Degrader`].
#[derive(Debug, Clone)]
pub struct DegraderBuilder {
    seed: u64,
    profile: DocumentProfile,
    strength: Option<f32>,
    backend: BackendPreference,
    max_width: u32,
    train_width: Option<u32>,
    scale_min: Option<f32>,
    jpeg_quality: Option<Option<u8>>,
    preset: Option<Preset>,
    toggles: Option<StageToggles>,
    fast_banks: Option<bool>,
    masks: MaskRequest,
}

impl Default for DegraderBuilder {
    fn default() -> Self {
        Self {
            seed: 0,
            profile: DocumentProfile::PhonePhoto,
            strength: None,
            backend: BackendPreference::Auto,
            max_width: 2480,
            train_width: None,
            scale_min: None,
            jpeg_quality: None,
            preset: None,
            toggles: None,
            fast_banks: None,
            masks: MaskRequest::none(),
        }
    }
}

impl DegraderBuilder {
    /// Set the base deterministic seed (used as-is or as the base for
    /// [`Degrader::degrade_batch`]).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Select a domain [`DocumentProfile`].
    pub fn profile(mut self, profile: DocumentProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Override global degradation strength in `[0.0, 1.0]`.
    pub fn strength(mut self, strength: f32) -> Self {
        self.strength = Some(strength);
        self
    }

    /// Select compute backend preference (`Auto`, `Cpu`, or `Gpu`).
    pub fn backend(mut self, backend: BackendPreference) -> Self {
        self.backend = backend;
        self
    }

    /// Cap input width before degradation (`0` = keep native width).
    pub fn max_width(mut self, max_width: u32) -> Self {
        self.max_width = max_width;
        self
    }

    /// Optional training-resolution target width (`None` or `Some(0)` = off).
    pub fn train_width(mut self, train_width: Option<u32>) -> Self {
        self.train_width = train_width;
        self
    }

    /// Enable or disable precomputed fast noise/lighting banks.
    pub fn fast_banks(mut self, fast_banks: bool) -> Self {
        self.fast_banks = Some(fast_banks);
        self
    }

    /// Override minimum random output downscale factor in `(0.0, 1.0]`.
    pub fn scale_min(mut self, scale_min: f32) -> Self {
        self.scale_min = Some(scale_min);
        self
    }

    /// Override JPEG round-trip quality (`Some(1..=100)` or `None` to sample per image).
    pub fn jpeg_quality(mut self, jpeg_quality: Option<u8>) -> Self {
        self.jpeg_quality = Some(jpeg_quality);
        self
    }

    /// Override stage toggles using a [`Preset`].
    pub fn preset(mut self, preset: Preset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Override fine-grained [`StageToggles`] directly.
    pub fn toggles(mut self, toggles: StageToggles) -> Self {
        self.toggles = Some(toggles);
        self
    }

    /// Configure multi-task ground-truth mask export (default: [`MaskRequest::none()`]).
    pub fn export_masks(mut self, masks: MaskRequest) -> Self {
        self.masks = masks;
        self
    }

    /// Validate configuration, initialize the requested backend, and construct a [`Degrader`].
    pub fn build(self) -> Result<Degrader, DegradeError> {
        let mut cfg = DegradeConfig {
            seed: self.seed,
            max_width: self.max_width,
            train_width: self.train_width.unwrap_or(0),
            masks: self.masks,
            ..DegradeConfig::default()
        };

        // Apply profile defaults first, then explicit builder overrides.
        self.profile.apply_to_config(&mut cfg);

        if let Some(s) = self.strength {
            cfg.strength = s;
        }
        if let Some(sm) = self.scale_min {
            cfg.scale_min = sm;
        }
        if let Some(jq) = self.jpeg_quality {
            cfg.jpeg_quality = jq;
        }
        if let Some(preset) = self.preset {
            let fb = cfg.toggles.fast_banks;
            cfg = cfg.with_preset(preset);
            cfg.toggles.fast_banks = fb;
        }
        if let Some(toggles) = self.toggles {
            cfg.toggles = toggles;
        }
        if let Some(fb) = self.fast_banks {
            cfg.toggles.fast_banks = fb;
        }

        // Validate config fields with structured errors.
        if !cfg.strength.is_finite() || !(0.0..=1.0).contains(&cfg.strength) {
            return Err(DegradeError::InvalidConfig {
                field: "strength",
                detail: format!("must be finite and in [0.0, 1.0], got {}", cfg.strength),
            });
        }
        if !cfg.scale_min.is_finite()
            || !(0.0..=1.0).contains(&cfg.scale_min)
            || cfg.scale_min <= 0.0
        {
            return Err(DegradeError::InvalidConfig {
                field: "scale_min",
                detail: format!("must be finite and in (0.0, 1.0], got {}", cfg.scale_min),
            });
        }
        if let Some(q) = cfg.jpeg_quality {
            if !(1..=100).contains(&q) {
                return Err(DegradeError::InvalidConfig {
                    field: "jpeg_quality",
                    detail: format!("must be in 1..=100, got {q}"),
                });
            }
        }

        #[cfg(feature = "gpu")]
        let gpu = match self.backend {
            BackendPreference::Cpu => None,
            BackendPreference::Auto => Gpu::new().ok(),
            BackendPreference::Gpu => Some(Gpu::new().map_err(DegradeError::GpuInit)?),
        };

        #[cfg(not(feature = "gpu"))]
        if matches!(self.backend, BackendPreference::Gpu) {
            return Err(DegradeError::GpuInit(
                "crate was compiled without the `gpu` feature".to_string(),
            ));
        }

        Ok(Degrader {
            config: cfg,
            profile: self.profile,
            #[cfg(feature = "gpu")]
            gpu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn sample_doc(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([250, 248, 242]));
        for y in (h / 4..h * 3 / 4).step_by(8) {
            for x in w / 6..w * 5 / 6 {
                img.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
        img
    }

    #[test]
    fn profile_roundtrip_and_apply() {
        for &profile in DocumentProfile::ALL {
            let name = profile.as_str();
            assert_eq!(DocumentProfile::from_str_name(name), Some(profile));
            assert_eq!(name.parse::<DocumentProfile>().ok(), Some(profile));

            let mut cfg = DegradeConfig::default();
            cfg.toggles.fast_banks = true;
            profile.apply_to_config(&mut cfg);
            assert!(
                cfg.validate().is_ok(),
                "profile {name} produced invalid config"
            );
            assert!(
                cfg.toggles.fast_banks,
                "profile {name} must preserve fast_banks"
            );
        }
        assert_eq!(
            DocumentProfile::from_str_name("OFFICE_SCAN"),
            Some(DocumentProfile::OfficeScan)
        );
        assert_eq!(DocumentProfile::from_str_name("unknown-profile"), None);
    }

    #[test]
    fn builder_validation_errors() {
        let bad_strength = Degrader::builder()
            .backend(BackendPreference::Cpu)
            .strength(1.5)
            .build();
        assert!(matches!(
            bad_strength,
            Err(DegradeError::InvalidConfig {
                field: "strength",
                ..
            })
        ));

        let bad_scale = Degrader::builder()
            .backend(BackendPreference::Cpu)
            .scale_min(0.0)
            .build();
        assert!(matches!(
            bad_scale,
            Err(DegradeError::InvalidConfig {
                field: "scale_min",
                ..
            })
        ));

        let bad_jpeg = Degrader::builder()
            .backend(BackendPreference::Cpu)
            .jpeg_quality(Some(0))
            .build();
        assert!(matches!(
            bad_jpeg,
            Err(DegradeError::InvalidConfig {
                field: "jpeg_quality",
                ..
            })
        ));
    }

    #[test]
    fn degrader_cpu_single_empty_and_batch() {
        let degrader = Degrader::builder()
            .seed(42)
            .profile(DocumentProfile::OfficeScan)
            .strength(0.35)
            .backend(BackendPreference::Cpu)
            .max_width(2480)
            .train_width(None)
            .fast_banks(false)
            .scale_min(1.0)
            .build()
            .expect("CPU builder must succeed");

        assert_eq!(degrader.active_backend(), "cpu");
        assert_eq!(degrader.profile(), DocumentProfile::OfficeScan);
        assert!((degrader.config().strength - 0.35).abs() < 1e-6);

        let empty = RgbImage::new(0, 0);
        assert!(matches!(
            degrader.degrade_image(&empty, 1),
            Err(DegradeError::EmptyImage)
        ));

        let img = sample_doc(64, 64);
        let r1 = degrader.degrade_image(&img, 42).expect("single degrade");
        let r2 = degrader
            .degrade_image(&img, 42)
            .expect("deterministic repeat");
        assert_eq!(r1.image.as_raw(), r2.image.as_raw());
        assert_eq!(r1.params.backend, "cpu");

        let batch = degrader
            .degrade_batch(&[&img, &img, &img, &img, &img, &img])
            .expect("batch degrade");
        assert_eq!(batch.len(), 6);
        for (i, item) in batch.iter().enumerate() {
            let expected_seed = 42u64.wrapping_add((i as u64).wrapping_mul(GOLDEN_RATIO_U64));
            assert_eq!(item.params.file_seed, expected_seed);
            let single = degrader
                .degrade_image(&img, expected_seed)
                .expect("sequential parity");
            assert_eq!(
                item.image.as_raw(),
                single.image.as_raw(),
                "parallel batch item {i} must be bit-identical to sequential degrade_image"
            );
        }
        assert_ne!(batch[0].image.as_raw(), batch[1].image.as_raw());
    }

    #[test]
    fn auto_gpu_constructs_cleanly() {
        let degrader = Degrader::auto_gpu().expect("auto_gpu must fall back to CPU if needed");
        assert!(matches!(degrader.active_backend(), "gpu" | "cpu"));
    }
}
