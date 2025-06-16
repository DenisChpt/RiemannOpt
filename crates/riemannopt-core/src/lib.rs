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
//! - [`cost_function`]: Cost function interface for optimization
//! - [`error`]: Error types for manifold operations
//! - [`line_search`]: Line search algorithms
//! - [`manifold`]: Core manifold trait and associated types
//! - [`metric`]: Riemannian metric traits
//! - [`optimizer`]: Optimization algorithms framework
//! - [`optimizer_state`]: State management for optimizers
//! - [`retraction`]: Retraction and vector transport methods
//! - [`tangent`]: Tangent space operations
//! - [`types`]: Type aliases and numerical constants

#![cfg_attr(not(feature = "std"), no_std)]

// Re-export key items
pub mod cost_function;
pub mod error;
pub mod line_search;
pub mod manifold;
pub mod metric;
pub mod numerical_validation;
pub mod optimizer;
pub mod optimizer_state;
pub mod retraction;
pub mod tangent;
pub mod types;
pub mod step_size;
pub mod preconditioner;
pub mod fisher;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_manifolds;

// Re-export commonly used items at the crate root
pub use error::{ManifoldError, OptimizerError, OptimizerResult, Result};

/// Prelude module for convenient imports.
///
/// # Example
/// ```
/// use riemannopt_core::prelude::*;
/// ```
pub mod prelude {
    pub use crate::cost_function::{
        CostFunction, CountingCostFunction, DerivativeChecker, QuadraticCost,
    };
    pub use crate::error::{ManifoldError, OptimizerError, OptimizerResult, Result};
    pub use crate::line_search::{
        BacktrackingLineSearch, FixedStepSize, LineSearch, LineSearchParams, LineSearchResult,
        StrongWolfeLineSearch,
    };
    pub use crate::manifold::{Manifold, Point, TangentVector as TangentVectorType};
    pub use crate::metric::{
        CanonicalMetric, ChristoffelSymbols, MetricTensor, MetricUtils, WeightedMetric,
    };
    pub use crate::optimizer::{
        ConvergenceChecker, OptimizationResult, Optimizer, OptimizerState, StoppingCriterion,
        TerminationReason,
    };
    pub use crate::optimizer_state::{
        AdamState, ConjugateGradientMethod, ConjugateGradientState, LBFGSState, MomentumState,
        OptimizerStateData,
    };
    pub use crate::retraction::{
        CayleyRetraction, DefaultRetraction, DifferentialRetraction, ExponentialRetraction,
        ParallelTransport, PolarRetraction, ProjectionRetraction, ProjectionTransport,
        QRRetraction, Retraction, RetractionOrder, RetractionVerifier, SchildLadder,
        VectorTransport,
    };
    pub use crate::tangent::{RiemannianMetric, TangentBundle, TangentSpace, TangentVector};
    pub use crate::types::{
        constants, DMatrix, DSquareMatrix, DVector, Dimension, Matrix, SMatrix, SSquareMatrix,
        SVector, Scalar, Vector,
    };
    // Additional trait re-exports will be added as we implement them
}
