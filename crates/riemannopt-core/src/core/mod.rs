//! Core traits and types for Riemannian optimization.

pub mod cached_cost_function;
pub mod cost_function;
pub mod error;
pub mod manifold;
pub mod types;

// Re-export core types
pub use cached_cost_function::*;
pub use cost_function::*;
pub use error::*;
pub use manifold::*;
pub use types::*;
