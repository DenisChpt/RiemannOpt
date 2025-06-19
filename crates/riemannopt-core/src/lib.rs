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
pub mod core;
pub mod memory;
pub mod compute;
pub mod optimization;
pub mod numerical;
pub mod manifold_ops;
pub mod utils;
pub mod config;
pub mod profiling;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export key items from core for backward compatibility
pub use core::{
    cost_function, error, manifold, types,
    ManifoldError, OptimizerError, OptimizerResult, Result,
};

// Re-export other modules for backward compatibility
pub use optimization::{
    line_search, optimizer, optimizer_state, preconditioner, step_size,
};
pub use numerical::{
    stability as numerical_stability,
    validation as numerical_validation,
};
pub use manifold_ops::{
    fisher, metric, retraction, tangent,
};
pub use compute::cpu::{
    parallel, simd,
};
pub use utils::{
    parallel_thresholds,
};

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
        // From cost_function
        CostFunction, CountingCostFunction, DerivativeChecker, QuadraticCost,
        // From error
        ManifoldError, OptimizerError, OptimizerResult, Result,
        // From manifold
        Manifold, Point, TangentVector as TangentVectorType,
        // From types
        constants, DMatrix, DSquareMatrix, DVector, Dimension, Matrix, SMatrix, SSquareMatrix,
        SVector, Scalar, Vector,
    };
    
    // Optimization components
    pub use crate::optimization::{
        // From line_search
        BacktrackingLineSearch, FixedStepSize, LineSearch, LineSearchParams, LineSearchResult,
        StrongWolfeLineSearch,
        // From optimizer
        ConvergenceChecker, OptimizationResult, Optimizer, OptimizerState, StoppingCriterion,
        TerminationReason,
        // From optimizer_state
        AdamState, ConjugateGradientMethod, ConjugateGradientState, LBFGSState, MomentumState,
        OptimizerStateData,
    };
    
    // Manifold operations
    pub use crate::manifold_ops::{
        // From metric
        CanonicalMetric, ChristoffelSymbols, MetricTensor, MetricUtils, WeightedMetric,
        // From retraction
        CayleyRetraction, DefaultRetraction, DifferentialRetraction, ExponentialRetraction,
        ParallelTransport, PolarRetraction, ProjectionRetraction, ProjectionTransport,
        QRRetraction, Retraction, RetractionOrder, RetractionVerifier, SchildLadder,
        VectorTransport,
        // From tangent
        RiemannianMetric, TangentBundle, TangentSpace, TangentVector,
    };
    
    // Compute components
    pub use crate::compute::cpu::{
        // From parallel
        ParallelBatch, ParallelLineSearch, ParallelSGD, ParallelAverage,
        PointBatch, TangentBatch, SimdParallelOps,
        // From simd
        SimdOps, SimdVector, SimdVectorOps, SimdMatrixOps,
    };
    
    // Utils
    pub use crate::utils::{
        // From parallel_thresholds
        ParallelThresholdsConfig, ParallelThresholdsBuilder, ParallelDecision,
        get_parallel_config, set_parallel_config, ShouldParallelize, DecompositionKind,
    };
    
    #[cfg(feature = "cuda")]
    pub use crate::gpu::{
        GpuBackend, GpuMatrix, GpuError, DeviceInfo,
        GpuMatrixOps, GpuManifoldOps, GpuBatchOps,
    };
}