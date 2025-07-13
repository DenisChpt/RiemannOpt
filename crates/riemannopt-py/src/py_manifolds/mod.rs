//! Python wrappers for manifold implementations.
//!
//! This module provides Python-friendly interfaces to the Rust manifold
//! implementations, handling workspace management and type conversions.

use pyo3::prelude::*;

#[macro_use]
pub mod base;
pub mod sphere;
pub mod stiefel;
pub mod grassmann;
pub mod spd;
pub mod hyperbolic;
// pub mod euclidean;  // TODO: Implement when Euclidean manifold is available
pub mod product;

pub use sphere::PySphere;
pub use stiefel::PyStiefel;
pub use grassmann::PyGrassmann;
pub use spd::PySPD;
pub use hyperbolic::PyHyperbolic;
// pub use euclidean::PyEuclidean;
pub use product::PyProductManifold;

/// Register all manifold classes with the Python module.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "manifolds")?;
    
    // Register manifold classes
    m.add_class::<PySphere>()?;
    m.add_class::<PyStiefel>()?;
    m.add_class::<PyGrassmann>()?;
    m.add_class::<PySPD>()?;
    m.add_class::<PyHyperbolic>()?;
    // m.add_class::<PyEuclidean>()?;
    m.add_class::<PyProductManifold>()?;
    
    parent.add_submodule(&m)?;
    Ok(())
}