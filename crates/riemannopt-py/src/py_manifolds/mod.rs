//! Python wrappers for manifold implementations.
//!
//! This module provides Python-friendly interfaces to the Rust manifold
//! implementations, handling workspace management and type conversions.

use pyo3::prelude::*;

#[macro_use]
pub mod base;
pub mod euclidean;
pub mod fixed_rank;
pub mod grassmann;
pub mod hyperbolic;
pub mod oblique;
pub mod product;
pub mod psd_cone;
pub mod spd;
pub mod sphere;
pub mod stiefel;

pub use euclidean::PyEuclidean;
pub use fixed_rank::{PyFixedRank, PyFixedRankPoint, PyFixedRankTangent};
pub use grassmann::PyGrassmann;
pub use hyperbolic::PyHyperbolic;
pub use oblique::PyOblique;
pub use product::PyProductManifold;
pub use psd_cone::PyPSDCone;
pub use spd::PySPD;
pub use sphere::PySphere;
pub use stiefel::PyStiefel;

/// Register all manifold classes with the Python module.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
	let m = PyModule::new(parent.py(), "manifolds")?;

	// Register manifold classes
	m.add_class::<PySphere>()?;
	m.add_class::<PyStiefel>()?;
	m.add_class::<PyGrassmann>()?;
	m.add_class::<PySPD>()?;
	m.add_class::<PyHyperbolic>()?;
	m.add_class::<PyOblique>()?;
	m.add_class::<PyFixedRank>()?;
	m.add_class::<PyFixedRankPoint>()?;
	m.add_class::<PyFixedRankTangent>()?;
	m.add_class::<PyPSDCone>()?;
	m.add_class::<PyEuclidean>()?;
	m.add_class::<PyProductManifold>()?;

	parent.add_submodule(&m)?;
	Ok(())
}
