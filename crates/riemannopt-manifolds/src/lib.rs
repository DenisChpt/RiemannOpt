//! RiemannOpt Manifolds - Concrete implementations of Riemannian manifolds.
//!
//! This crate provides implementations of commonly used manifolds in
//! optimization, machine learning, and scientific computing.

pub mod sphere;
pub mod stiefel;
pub mod grassmann;
pub mod spd;
pub mod hyperbolic;
pub mod product;

// Re-export main manifolds for convenience
pub use sphere::Sphere;
pub use stiefel::Stiefel;
pub use grassmann::Grassmann;
pub use spd::SPD;
pub use hyperbolic::Hyperbolic;
pub use product::ProductManifold;
