//! Optimization algorithms and utilities.

pub mod callback;
pub mod line_search;
pub mod optimizer;
pub mod preconditioner;
pub mod step_size;
pub mod workspace;

// Re-export optimization components
pub use callback::*;
pub use line_search::*;
pub use optimizer::*;
pub use preconditioner::*;
pub use step_size::*;
pub use workspace::*;
