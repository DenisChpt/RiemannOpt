//! # RiemannOpt
//!
//! High-performance Riemannian optimization in Rust.
//!
//! This crate is the main facade that re-exports all sub-crates for convenience.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use riemannopt::prelude::*;
//! ```

// Re-export sub-crates
pub use riemannopt_core as core;

#[cfg(feature = "autodiff")]
pub use riemannopt_autodiff as autodiff;

// Re-export nalgebra for user convenience
pub use nalgebra;

/// Prelude module — import everything you need with `use riemannopt::prelude::*`.
pub mod prelude {
	pub use riemannopt_core::prelude::*;

	// Autodiff (optional)
	#[cfg(feature = "autodiff")]
	pub use riemannopt_autodiff::{
		AdSession, AutoDiffMatProblem, AutoDiffProblem, Dual, MVar, SVar, Tape, VVar,
	};
}
