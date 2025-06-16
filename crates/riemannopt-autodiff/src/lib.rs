//! Automatic differentiation for Riemannian optimization.
//!
//! This crate provides a minimal autodiff engine tailored for Riemannian
//! optimization. It supports forward and reverse mode differentiation with
//! special handling for manifold operations.
//!
//! # Features
//!
//! - **Computation graphs**: Dynamic graph construction for flexible models
//! - **Reverse mode AD**: Efficient gradient computation via backpropagation
//! - **Manifold awareness**: Special operations for Riemannian manifolds
//! - **Memory efficiency**: Gradient checkpointing and graph optimization
//!
//! # Architecture
//!
//! The autodiff engine is built around three core components:
//!
//! 1. **Graph**: Manages the computation graph structure
//! 2. **Operations**: Defines forward and backward operations
//! 3. **Backward**: Implements the backpropagation algorithm

pub mod graph;
pub mod ops;
pub mod backward;
pub mod manifold_ops;
pub mod broadcast;
pub mod integration;

// Re-export key types
pub use graph::{Graph, Node, NodeId, Tensor, Variable};
pub use ops::{Op, OpType};
pub use backward::{backward, GradientMap};
pub use manifold_ops::{
    TangentProjection, StiefelProjection, SphereProjection,
    SphereTangentProjection, ManifoldInnerProduct, ExponentialMap, LogarithmicMap,
};
pub use broadcast::{BroadcastAdd, BroadcastMultiply, broadcast_binary, unbroadcast};
pub use integration::{ManifoldGraph, ManifoldFunction, ManifoldOptimizationProblem};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::graph::{Graph, Node, NodeId, Tensor, Variable};
    pub use crate::ops::{Op, OpType};
    pub use crate::backward::{backward, GradientMap};
    pub use crate::manifold_ops::*;
    pub use crate::broadcast::{BroadcastAdd, BroadcastMultiply};
    pub use crate::integration::{ManifoldGraph, ManifoldFunction, ManifoldOptimizationProblem};
}