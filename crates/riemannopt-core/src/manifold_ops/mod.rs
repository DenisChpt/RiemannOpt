//! Manifold-specific geometric operations and computational utilities.
//!
//! This module provides specialized operations that leverage the geometric structure
//! of Riemannian manifolds for efficient and numerically stable computations.
//! These operations form the computational foundation for optimization algorithms
//! and geometric analysis on manifolds.
//!
//! # Geometric Operations
//!
//! ## Retraction Methods (`retraction`)
//! Efficient approximations to the exponential map for moving along tangent directions:
//! - **QR-based retractions**: For matrix manifolds (Stiefel, Grassmann)
//! - **Projection retractions**: For embedded manifolds (spheres, ellipsoids)
//! - **Symmetric retractions**: For SPD matrices and related structures
//! - **Vector transport**: Moving tangent vectors between tangent spaces
//!
//! ## Tangent Space Operations (`tangent`, `tangent_simd`)
//! Efficient projections and manipulations in tangent spaces:
//! - **Orthogonal projections**: Removing normal components
//! - **Tangent vector arithmetic**: Addition, scaling, inner products
//! - **SIMD optimizations**: Vectorized operations for high-dimensional problems
//! - **Workspace management**: Memory-efficient batch operations
//!
//! ## Riemannian Metrics (`metric`, `metric_simd`)
//! Inner product computations respecting manifold geometry:
//! - **Canonical metrics**: Inherited from ambient Euclidean space
//! - **Invariant metrics**: Preserving group symmetries
//! - **Custom metrics**: Problem-specific geometric structures
//! - **Parallel implementations**: High-performance metric evaluations
//!
//! ## Information Geometry (`fisher`)
//! Specialized operations for statistical manifolds:
//! - **Fisher information metrics**: Natural Riemannian structure on probability spaces
//! - **Statistical distances**: Divergences and geometric measures
//! - **Parameter space geometry**: Efficient statistical optimization
//!
//! # Computational Architecture
//!
//! ## Workspace-Based Design
//! All operations support workspace-based memory management for efficiency:
//! ```rust,ignore
//! # use riemannopt_core::prelude::*;
//! let mut workspace = Workspace::new();
//! // Operations reuse pre-allocated memory
//! manifold.retract(&point, &tangent, &mut result, &mut workspace)?;
//! ```
//!
//! ## SIMD Acceleration
//! High-dimensional operations leverage SIMD instructions:
//! - **Vectorized arithmetic**: Parallel floating-point operations
//! - **Cache optimization**: Efficient memory access patterns
//! - **Automatic dispatch**: Runtime selection of optimal implementations
//!
//! ## Generic Programming
//! Operations work with arbitrary manifold and scalar types:
//! ```rust,ignore
//! # use riemannopt_core::prelude::*;
//! fn optimize_on_manifold<M: Manifold<f64>>(
//!     manifold: &M,
//!     initial_point: &M::Point
//! ) -> Result<M::Point> {
//!     // Generic operations work with any manifold
//!     let mut workspace = Workspace::new();
//!     let mut result = initial_point.clone();
//!     // ... optimization logic ...
//!     Ok(result)
//! }
//! ```
//!
//! # Performance Considerations
//!
//! ## Memory Management
//! - **Workspace reuse**: Minimizes allocation overhead
//! - **In-place operations**: Reduces memory traffic
//! - **Buffer pooling**: Efficient memory recycling
//!
//! ## Numerical Stability
//! - **Numerically stable algorithms**: QR, SVD, Cholesky decompositions
//! - **Condition monitoring**: Detection of near-singular cases
//! - **Iterative refinement**: Maintaining manifold constraints
//!
//! ## Scalability
//! - **Parallel algorithms**: Multi-core utilization
//! - **Batch operations**: Amortizing overhead costs
//! - **Cache-aware designs**: Optimizing memory hierarchy usage

pub mod fisher;
pub mod metric;
pub mod metric_simd;
pub mod metric_workspace;
pub mod retraction;
pub mod retraction_workspace;
pub mod tangent;
pub mod tangent_simd;
pub mod tangent_workspace;

// Re-export manifold operations
pub use fisher::*;
pub use metric::*;
pub use metric_simd::*;
pub use metric_workspace::*;
pub use retraction::*;
pub use retraction_workspace::*;
pub use tangent::*;
pub use tangent_simd::*;
pub use tangent_workspace::*;