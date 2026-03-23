//! RiemannOpt Manifolds - Concrete implementations of Riemannian manifolds.
//!
//! This crate provides implementations of commonly used manifolds in
//! optimization, machine learning, and scientific computing.

pub mod euclidean;
pub mod fixed_rank;
pub mod grassmann;
pub mod hyperbolic;
pub mod oblique;
pub mod product;
pub mod product_static;
pub mod psd_cone;
pub mod spd;
pub mod sphere;
pub mod stiefel;
pub mod stiefel_small;
pub mod utils;

#[cfg(feature = "parallel")]
pub mod sphere_simd;

// Re-export main manifolds for convenience
pub use euclidean::Euclidean;
pub use fixed_rank::FixedRank;
pub use grassmann::Grassmann;
pub use hyperbolic::Hyperbolic;
pub use oblique::Oblique;
pub use product::Product;
pub use product_static::{product_static, ProductStatic};
pub use psd_cone::PSDCone;
pub use spd::SPD;
pub use sphere::Sphere;
pub use stiefel::Stiefel;
