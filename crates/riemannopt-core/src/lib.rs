//! Core traits and types for Riemannian optimization.
//!
//! This crate provides the foundational traits and types for implementing
//! Riemannian optimization algorithms. It defines the mathematical abstractions
//! needed to work with smooth manifolds, tangent spaces, and Riemannian metrics.
//!
//! # Key Concepts
//!
//! - **Manifolds**: Smooth spaces that locally resemble Euclidean space
//! - **Tangent Spaces**: Linear approximations of manifolds at each point
//! - **Riemannian Metrics**: Inner products on tangent spaces
//! - **Retractions**: Smooth maps from tangent spaces back to the manifold
//!
//! # Modules
//!
//! - [`core`]: Core traits and types (manifold, cost function, error, types)
//! - [`linalg`]: Linear algebra abstractions and backends
//! - [`optimization`]: Optimization algorithms and utilities
//! - [`utils`]: Utility functions and test helpers

#![cfg_attr(not(feature = "std"), no_std)]

// Module declarations
pub mod core;
pub mod linalg;
pub mod optimization;
pub mod utils;

// Re-export key items from core for backward compatibility
pub use core::{
	cost_function, error, manifold, types, ManifoldError, OptimizerError, OptimizerResult, Result,
};

// Re-export optimization module for backward compatibility
pub use optimization::{line_search, optimizer, preconditioner, step_size};
pub use utils::parallel_thresholds;

#[cfg(any(test, feature = "test-utils"))]
pub use utils::{test_manifolds, test_utils};

/// Prelude module for convenient imports.
///
/// # Example
/// ```
/// use riemannopt_core::prelude::*;
/// ```
pub mod prelude {
	// Core types
	pub use crate::core::{
		// From types
		constants,
		// From cost_function
		CostFunction,
		CountingCostFunction,
		DerivativeChecker,
		// From manifold
		Manifold,
		// From error
		ManifoldError,
		OptimizerError,
		OptimizerResult,
		QuadraticCost,
		Result,
		Scalar,
	};

	// Optimization components
	pub use crate::optimization::{
		// From line_search
		BacktrackingLineSearch,
		FixedStepSize,
		LineSearch,
		LineSearchParams,
		LineSearchResult,
		// From optimizer
		OptimizationResult,
		Optimizer,
		StoppingCriterion,
		StrongWolfeLineSearch,
		TerminationReason,
	};

	// Utils
	pub use crate::utils::{
		get_parallel_config,
		set_parallel_config,
		DecompositionKind,
		ParallelDecision,
		ParallelThresholdsBuilder,
		// From parallel_thresholds
		ParallelThresholdsConfig,
		ShouldParallelize,
	};
}
