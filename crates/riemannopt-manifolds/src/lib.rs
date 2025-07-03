//! RiemannOpt Manifolds - Concrete implementations of Riemannian manifolds.
//!
//! This crate provides implementations of commonly used manifolds in
//! optimization, machine learning, and scientific computing.

pub mod sphere;
pub mod stiefel;
pub mod stiefel_small;
pub mod grassmann;
pub mod spd;
pub mod oblique;
pub mod hyperbolic;
pub mod product;
pub mod product_static;
pub mod fixed_rank;
pub mod psd_cone;
pub mod utils;

#[cfg(feature = "parallel")]
pub mod sphere_simd;

// Re-export main manifolds for convenience
pub use sphere::Sphere;
pub use stiefel::Stiefel;
pub use grassmann::Grassmann;
pub use spd::SPD;
pub use oblique::Oblique;
pub use hyperbolic::Hyperbolic;
pub use product::Product;
pub use product_static::{ProductStatic, product_static};
pub use fixed_rank::FixedRank;
pub use psd_cone::PSDCone;
