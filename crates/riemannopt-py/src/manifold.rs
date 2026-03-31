use pyo3::prelude::*;

use riemannopt_core::linalg::DefaultBackend;
use riemannopt_core::manifold::{
	Euclidean, Grassmann, Hyperbolic, Manifold, Oblique, Sphere, Stiefel, SPD,
};

use crate::convert::{col_to_numpy_1d, mat_to_numpy_2d};

type T = f64;
type B = DefaultBackend;

// ════════════════════════════════════════════════════════════════════════
//  Dynamic manifold enum
// ════════════════════════════════════════════════════════════════════════

/// Categories of manifolds by their Point type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PointKind {
	Vector,
	Matrix,
}

pub(crate) enum DynManifold {
	Euclidean(Euclidean<T, B>),
	Sphere(Sphere<T, B>),
	Hyperbolic(Hyperbolic<T, B>),
	Stiefel(Stiefel<T, B>),
	Grassmann(Grassmann<T, B>),
	SPD(SPD<T, B>),
	Oblique(Oblique<T, B>),
}

impl DynManifold {
	pub fn point_kind(&self) -> PointKind {
		match self {
			Self::Euclidean(_) | Self::Sphere(_) | Self::Hyperbolic(_) => PointKind::Vector,
			Self::Stiefel(_) | Self::Grassmann(_) | Self::SPD(_) | Self::Oblique(_) => {
				PointKind::Matrix
			}
		}
	}

	pub fn dimension(&self) -> usize {
		match self {
			Self::Euclidean(m) => m.dimension(),
			Self::Sphere(m) => m.dimension(),
			Self::Hyperbolic(m) => m.dimension(),
			Self::Stiefel(m) => m.dimension(),
			Self::Grassmann(m) => m.dimension(),
			Self::SPD(m) => m.dimension(),
			Self::Oblique(m) => m.dimension(),
		}
	}

	pub fn name(&self) -> &str {
		match self {
			Self::Euclidean(m) => m.name(),
			Self::Sphere(m) => m.name(),
			Self::Hyperbolic(m) => m.name(),
			Self::Stiefel(m) => m.name(),
			Self::Grassmann(m) => m.name(),
			Self::SPD(m) => m.name(),
			Self::Oblique(m) => m.name(),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════
//  Macro to reduce boilerplate for delegating to concrete manifolds
// ════════════════════════════════════════════════════════════════════════

/// Dispatch on `DynManifold` for vector-point manifolds.
/// Calls $body with $m bound to the concrete manifold.

// ════════════════════════════════════════════════════════════════════════
//  Python class
// ════════════════════════════════════════════════════════════════════════

#[pyclass(name = "Manifold")]
pub struct PyManifold {
	pub(crate) inner: DynManifold,
}

#[pymethods]
impl PyManifold {
	/// Intrinsic dimension of the manifold.
	fn dimension(&self) -> usize {
		self.inner.dimension()
	}

	/// Human-readable name.
	fn name(&self) -> &str {
		self.inner.name()
	}

	/// Generate a random point on the manifold (returned as `NumPy` array).
	fn random_point(&self, py: Python<'_>) -> Py<PyAny> {
		match &self.inner {
			DynManifold::Euclidean(m) => {
				let mut p = m.allocate_point();
				m.random_point(&mut p);
				col_to_numpy_1d(py, &p).into_any().unbind()
			}
			DynManifold::Sphere(m) => {
				let mut p = m.allocate_point();
				m.random_point(&mut p);
				col_to_numpy_1d(py, &p).into_any().unbind()
			}
			DynManifold::Hyperbolic(m) => {
				let mut p = m.allocate_point();
				m.random_point(&mut p);
				col_to_numpy_1d(py, &p).into_any().unbind()
			}
			DynManifold::Stiefel(m) => {
				let mut p = m.allocate_point();
				m.random_point(&mut p);
				mat_to_numpy_2d(py, &p).into_any().unbind()
			}
			DynManifold::Grassmann(m) => {
				let mut p = m.allocate_point();
				m.random_point(&mut p);
				mat_to_numpy_2d(py, &p).into_any().unbind()
			}
			DynManifold::SPD(m) => {
				let mut p = m.allocate_point();
				m.random_point(&mut p);
				mat_to_numpy_2d(py, &p).into_any().unbind()
			}
			DynManifold::Oblique(m) => {
				let mut p = m.allocate_point();
				m.random_point(&mut p);
				mat_to_numpy_2d(py, &p).into_any().unbind()
			}
		}
	}

	fn __repr__(&self) -> String {
		format!(
			"Manifold('{}', dimension={})",
			self.inner.name(),
			self.inner.dimension()
		)
	}
}

// ════════════════════════════════════════════════════════════════════════
//  Factory functions
// ════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn sphere(n: usize) -> PyManifold {
	PyManifold {
		inner: DynManifold::Sphere(Sphere::new(n)),
	}
}

#[pyfunction]
pub fn euclidean(n: usize) -> PyManifold {
	PyManifold {
		inner: DynManifold::Euclidean(Euclidean::new(n)),
	}
}

#[pyfunction]
pub fn stiefel(n: usize, p: usize) -> PyManifold {
	PyManifold {
		inner: DynManifold::Stiefel(Stiefel::new(n, p)),
	}
}

#[pyfunction]
pub fn grassmann(n: usize, p: usize) -> PyManifold {
	PyManifold {
		inner: DynManifold::Grassmann(Grassmann::new(n, p)),
	}
}

#[pyfunction]
pub fn spd(n: usize) -> PyManifold {
	PyManifold {
		inner: DynManifold::SPD(SPD::new(n)),
	}
}

#[pyfunction]
pub fn hyperbolic(n: usize) -> PyManifold {
	PyManifold {
		inner: DynManifold::Hyperbolic(Hyperbolic::new(n)),
	}
}

#[pyfunction]
pub fn oblique(n: usize, p: usize) -> PyManifold {
	PyManifold {
		inner: DynManifold::Oblique(Oblique::new(n, p)),
	}
}
