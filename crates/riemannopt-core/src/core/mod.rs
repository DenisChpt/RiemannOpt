//! Core traits and types for Riemannian optimization.

pub mod cost_function;
pub mod cost_function_simd;
pub mod cost_function_workspace;
pub mod cached_cost_function_dyn;
pub mod error;
pub mod manifold;
pub mod types;

// Re-export core types
pub use cost_function::*;
pub use cost_function_simd::*;
pub use cost_function_workspace::*;
pub use cached_cost_function_dyn::*;
pub use error::*;
pub use manifold::*;
pub use types::*;