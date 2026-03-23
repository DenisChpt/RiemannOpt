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
//! - [`memory`]: Memory management utilities
//! - [`compute`]: Computational backends (CPU, specialized)
//! - [`optimization`]: Optimization algorithms and utilities
//! - [`numerical`]: Numerical stability and validation
//! - [`manifold_ops`]: Manifold-specific operations
//! - [`utils`]: Utility functions and test helpers
//! - [`config`]: Configuration utilities
//! - [`profiling`]: Performance profiling
//! - [`gpu`] (feature-gated): GPU acceleration with CUDA support

#![cfg_attr(not(feature = "std"), no_std)]

// Module declarations
pub mod compute;
pub mod config;
pub mod core;
pub mod manifold_ops;
pub mod memory;
pub mod numerical;
pub mod optimization;
pub mod profiling;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export key items from core for backward compatibility
pub use core::{
	cost_function, error, manifold, types, ManifoldError, OptimizerError, OptimizerResult, Result,
};

// Re-export other modules for backward compatibility
pub use compute::cpu::{parallel, simd};
pub use manifold_ops::{fisher, metric, retraction, tangent};
pub use numerical::{stability as numerical_stability, validation as numerical_validation};
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
		DMatrix,
		DSquareMatrix,
		DVector,
		DerivativeChecker,
		Dimension,
		// From manifold
		Manifold,
		// From error
		ManifoldError,
		Matrix,
		OptimizerError,
		OptimizerResult,
		QuadraticCost,
		Result,
		SMatrix,
		SSquareMatrix,
		SVector,
		Scalar,
		Vector,
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

	// Manifold operations
	pub use crate::manifold_ops::{
		// From retraction
		CayleyRetraction,
		DefaultRetraction,
		ExponentialRetraction,
		MatrixMetricTensor,
		MatrixMetricType,
		MatrixMetricWorkspace,
		MatrixRetractionWorkspace,
		MetricOps,
		MetricUtils,
		PolarRetraction,
		ProjectionRetraction,
		QRRetraction,
		Retraction,
		RetractionOps,
		RetractionOrder,
		// From tangent
		TangentSpace,
		// From tangent_workspace
		TangentVectorWorkspace,
		// From metric
		VectorMetricTensor,
		// From metric_workspace
		VectorMetricWorkspace,
		// From retraction_workspace
		VectorRetractionWorkspace,
		VectorTangentSpace,
	};

	// Compute components
	pub use crate::compute::cpu::{
		get_dispatcher,
		ParallelAverage,
		// From parallel
		ParallelBatch,
		ParallelLineSearch,
		ParallelSGD,
		PointBatch,
		// From simd_dispatch
		SimdBackend,
		SimdDispatcher,
		// From simd
		SimdOps,
		SimdParallelOps,
		SimdVector,
		TangentBatch,
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

	#[cfg(feature = "cuda")]
	pub use crate::gpu::{
		DeviceInfo, GpuBackend, GpuBatchOps, GpuError, GpuManifoldOps, GpuMatrix, GpuMatrixOps,
	};
}
