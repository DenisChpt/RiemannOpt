//! Linear algebra backend abstraction.
//!
//! This module provides a zero-cost abstraction layer over concrete linear
//! algebra libraries. Switch backends via feature flags:
//!
//! - `faer-backend` (default) — pure Rust, near-BLAS performance, no C deps
//! - `nalgebra-backend` — well-known ecosystem, mature
//!
//! # Quick switch
//!
//! ```toml
//! # Use faer (default):
//! riemannopt-core = { version = "0.1" }
//!
//! # Use nalgebra instead:
//! riemannopt-core = { version = "0.1", default-features = false, features = ["nalgebra-backend"] }
//! ```
//!
//! # Type aliases
//!
//! Use [`DefaultBackend`], [`Vec`], and [`Mat`] for backend-agnostic code:
//!
//! ```rust,ignore
//! use riemannopt_core::linalg::{Vec, Mat, VectorOps, MatrixOps};
//!
//! let v = Vec::<f64>::zeros(10);
//! let m = Mat::<f64>::identity(3);
//! ```

pub mod traits;
pub mod types;

// ── Backend modules ──────────────────────────────────────────────────────

pub mod nalgebra_backend;

#[cfg(feature = "faer-backend")]
pub mod parallel_policy;

#[cfg(feature = "faer-backend")]
pub mod faer_backend;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use nalgebra_backend::NalgebraBackend;
pub use traits::{
	DecompositionOps, LinAlgBackend, MatOf, MatrixOps, MatrixView, RealScalar, VecOf, VectorOps,
	VectorView,
};
pub use types::{CholeskyResult, EigenResult, QrResult, SvdResult};

#[cfg(feature = "faer-backend")]
pub use faer_backend::FaerBackend;

// ═══════════════════════════════════════════════════════════════════════════
//  Default backend selection via feature flags
// ═══════════════════════════════════════════════════════════════════════════

/// The active linear algebra backend, selected by feature flags.
///
/// - With `faer-backend` (default): [`FaerBackend`]
/// - With `nalgebra-backend`: [`NalgebraBackend`]
#[cfg(feature = "faer-backend")]
pub type DefaultBackend = FaerBackend;

#[cfg(not(feature = "faer-backend"))]
pub type DefaultBackend = NalgebraBackend;

/// Dense column vector for the active backend.
///
/// `Vec<f64>` resolves to `faer::Col<f64>` or `nalgebra::DVector<f64>`
/// depending on the selected backend.
pub type Vec<T> = <DefaultBackend as LinAlgBackend<T>>::Vector;

/// Dense matrix for the active backend.
///
/// `Mat<T>` resolves to `faer::Mat<T>` or `nalgebra::DMatrix<T>`
/// depending on the selected backend.
pub type Mat<T> = <DefaultBackend as LinAlgBackend<T>>::Matrix;
