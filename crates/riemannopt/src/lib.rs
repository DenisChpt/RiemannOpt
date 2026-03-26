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
pub use riemannopt_manifolds as manifolds;
pub use riemannopt_optim as optim;

#[cfg(feature = "autodiff")]
pub use riemannopt_autodiff as autodiff;

// Re-export nalgebra for user convenience
pub use nalgebra;

/// Prelude module — import everything you need with `use riemannopt::prelude::*`.
pub mod prelude {
	// Core types and traits
	pub use riemannopt_core::prelude::*;

	// Manifold implementations
	pub use riemannopt_manifolds::{
		Euclidean, FixedRank, Grassmann, Hyperbolic, Oblique, Product, Sphere, Stiefel, SPD,
	};

	// Optimizers
	pub use riemannopt_optim::{
		Adam, AdamConfig, CGConfig, ConjugateGradient, LBFGSConfig, Newton, NewtonConfig,
		SGDConfig, TrustRegion, TrustRegionConfig, LBFGS, SGD,
	};

	// Autodiff (optional)
	#[cfg(feature = "autodiff")]
	pub use riemannopt_autodiff::{backward, check_gradient, Gradients, Tape, TapeGuard, Var};
}
