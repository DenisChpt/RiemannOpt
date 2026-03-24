//! Core traits and types for Riemannian optimization.

pub mod cached_cost_function;
pub mod cached_cost_function_dyn;
pub mod cost_function;
pub mod error;
pub mod manifold;
pub mod matrix_manifold;
pub mod types;

// Re-export core types
pub use cached_cost_function::*;
pub use cached_cost_function_dyn::*;
pub use cost_function::*;
pub use error::*;
pub use manifold::*;
pub use matrix_manifold::*;
pub use types::*;
