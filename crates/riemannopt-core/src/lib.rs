//! RiemannOpt Core — High-performance Riemannian optimization in Rust.
//!
//! This crate provides the complete mathematical engine for optimization
//! on Riemannian manifolds: traits, manifold implementations, solvers,
//! and linear algebra abstractions.
//!
//! # Modules
//!
//! - [`linalg`]: Backend-agnostic linear algebra (faer / nalgebra)
//! - [`manifold`]: [`Manifold`] trait + implementations (Sphere, Stiefel, Grassmann, …)
//! - [`problem`]: [`Problem`] trait for cost functions on manifolds
//! - [`solver`]: [`Solver`] trait + solvers (SGD, Adam, L-BFGS, CG, Trust-Region, …)
//! - [`error`]: Error types
//! - [`types`]: [`Scalar`] trait and numerical constants

#![cfg_attr(not(feature = "std"), no_std)]

pub mod error;
pub mod linalg;
pub mod manifold;
pub mod problem;
pub mod solver;
pub mod types;

// ────────────────────────────────────────────────────────────────────────────
// Top-level re-exports
// ────────────────────────────────────────────────────────────────────────────

pub use error::{ManifoldError, OptimizerError, Result};
pub use manifold::Manifold;
pub use problem::Problem;
pub use solver::{Solver, SolverResult, StoppingCriterion, TerminationReason};
pub use types::Scalar;

// ────────────────────────────────────────────────────────────────────────────
// Prelude
// ────────────────────────────────────────────────────────────────────────────

/// Convenient wildcard import for common types and traits.
pub mod prelude {
	pub use crate::error::{ManifoldError, OptimizerError, Result};
	pub use crate::manifold::Manifold;
	pub use crate::problem::{CountingProblem, Problem, QuadraticCost};

	// Problem implementations
	pub use crate::problem::euclidean::{
		LogisticRegression, Rastrigin, RidgeRegression, Rosenbrock,
	};
	pub use crate::problem::fixed_rank::{MatrixCompletion, MatrixSensing};
	pub use crate::problem::grassmann::{BrockettCost, RobustPCA};
	pub use crate::problem::hyperbolic::{HyperbolicLogisticRegression, PoincareEmbedding};
	pub use crate::problem::oblique::{DictionaryLearning, ObliqueICA, PhaseRetrieval};
	pub use crate::problem::product::{CoupledFactorization, PoseEstimation};
	pub use crate::problem::psd_cone::{MaxCutSDP, NearestCorrelation};
	pub use crate::problem::spd::{FrechetMean, GaussianMixtureCovariance, MetricLearning};
	pub use crate::problem::sphere::{MaxCutSphere, RayleighQuotient, SphericalKMeans};
	pub use crate::problem::stiefel::{
		ICAContrast, OrderedBrockett, OrthogonalICA, OrthogonalProcrustes,
	};
	pub use crate::solver::{
		Adam, AdamConfig, CGConfig, ConjugateGradient, LBFGSConfig, MomentumMethod, Newton,
		NewtonConfig, SGDConfig, Solver, SolverResult, StoppingCriterion, TerminationReason,
		TrustRegion, TrustRegionConfig, LBFGS, SGD,
	};
	pub use crate::types::Scalar;

	// Manifold implementations
	pub use crate::manifold::{
		Euclidean, FixedRank, Grassmann, Hyperbolic, Oblique, Product, Sphere, Stiefel, SPD,
	};
}
