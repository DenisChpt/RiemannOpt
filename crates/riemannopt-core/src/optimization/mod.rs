//! Optimization algorithms and utilities.

pub mod adaptive;
pub mod line_search;
pub mod optimizer;
pub mod optimizer_state;
pub mod optimizer_workspace;
pub mod preconditioner;
pub mod step_size;

// Re-export optimization components
pub use line_search::*;
pub use optimizer::*;
pub use optimizer_state::*;
pub use optimizer_workspace::*;
pub use preconditioner::*;
pub use step_size::*;