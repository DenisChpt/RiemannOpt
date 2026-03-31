use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use riemannopt_autodiff::{AutoDiffMatProblem, AutoDiffProblem};
use riemannopt_core::linalg::DefaultBackend;
use riemannopt_core::problem::QuadraticCost;
use riemannopt_core::problem::euclidean::Rosenbrock;
use riemannopt_core::problem::grassmann::BrockettCost;
use riemannopt_core::problem::sphere::RayleighQuotient;
use riemannopt_core::problem::stiefel::OrthogonalProcrustes;

use crate::autodiff::{replay_mat, replay_vec, RecordedGraph};
use crate::convert::{numpy_1d_to_col, numpy_2d_to_mat, B};

type VecClosure = Box<
	dyn Fn(
			&mut riemannopt_autodiff::AdSession<f64, DefaultBackend>,
			riemannopt_autodiff::VVar,
		) -> riemannopt_autodiff::SVar
		+ Send
		+ Sync,
>;

type MatClosure = Box<
	dyn Fn(
			&mut riemannopt_autodiff::AdSession<f64, DefaultBackend>,
			riemannopt_autodiff::MVar,
		) -> riemannopt_autodiff::SVar
		+ Send
		+ Sync,
>;

// ════════════════════════════════════════════════════════════════════════
//  Dynamic problem enums
// ════════════════════════════════════════════════════════════════════════

pub(crate) enum DynVecProblem {
	AutoDiff(AutoDiffProblem<f64, B, VecClosure>),
	Rayleigh(RayleighQuotient<f64, B>),
	Quadratic(QuadraticCost<f64, B>),
	Rosenbrock(Rosenbrock<f64, B>),
}

impl std::fmt::Debug for DynVecProblem {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::AutoDiff(_) => write!(f, "AutoDiffProblem"),
			Self::Rayleigh(_) => write!(f, "RayleighQuotient"),
			Self::Quadratic(_) => write!(f, "QuadraticCost"),
			Self::Rosenbrock(_) => write!(f, "Rosenbrock"),
		}
	}
}

pub(crate) enum DynMatProblem {
	AutoDiff(AutoDiffMatProblem<f64, B, MatClosure>),
	Brockett(BrockettCost<f64, B>),
	Procrustes(OrthogonalProcrustes<f64, B>),
}

impl std::fmt::Debug for DynMatProblem {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::AutoDiff(_) => write!(f, "AutoDiffMatProblem"),
			Self::Brockett(_) => write!(f, "BrockettCost"),
			Self::Procrustes(_) => write!(f, "Procrustes"),
		}
	}
}

pub(crate) enum DynProblem {
	Vec(DynVecProblem),
	Mat(DynMatProblem),
}

// ════════════════════════════════════════════════════════════════════════
//  Python class
// ════════════════════════════════════════════════════════════════════════

#[pyclass(name = "Problem")]
pub struct PyProblem {
	pub(crate) inner: DynProblem,
}

impl PyProblem {
	pub(crate) fn new_autodiff_vec(graph: RecordedGraph) -> Self {
		let closure: VecClosure = Box::new(move |session, input_vvar| {
			replay_vec(&graph, session, input_vvar)
		});

		Self {
			inner: DynProblem::Vec(DynVecProblem::AutoDiff(AutoDiffProblem::new(closure))),
		}
	}

	pub(crate) fn new_autodiff_mat(graph: RecordedGraph) -> Self {
		let closure: MatClosure = Box::new(move |session, input_mvar| {
			replay_mat(&graph, session, input_mvar)
		});

		Self {
			inner: DynProblem::Mat(DynMatProblem::AutoDiff(AutoDiffMatProblem::new(closure))),
		}
	}
}

#[pymethods]
impl PyProblem {
	fn __repr__(&self) -> String {
		match &self.inner {
			DynProblem::Vec(p) => format!("{p:?}"),
			DynProblem::Mat(p) => format!("{p:?}"),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════
//  Factory functions for built-in problems
// ════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn rayleigh_quotient(a: PyReadonlyArray2<'_, f64>) -> PyProblem {
	let mat = numpy_2d_to_mat(a);
	PyProblem {
		inner: DynProblem::Vec(DynVecProblem::Rayleigh(RayleighQuotient::new(mat))),
	}
}

#[pyfunction]
pub fn quadratic_cost(
	a: PyReadonlyArray2<'_, f64>,
	b: PyReadonlyArray1<'_, f64>,
	c: f64,
) -> PyProblem {
	let a_mat = numpy_2d_to_mat(a);
	let b_vec = numpy_1d_to_col(b);
	PyProblem {
		inner: DynProblem::Vec(DynVecProblem::Quadratic(QuadraticCost::new(
			a_mat, b_vec, c,
		))),
	}
}

#[pyfunction]
pub fn rosenbrock() -> PyProblem {
	PyProblem {
		inner: DynProblem::Vec(DynVecProblem::Rosenbrock(Rosenbrock::new())),
	}
}

#[pyfunction]
pub fn brockett_cost(a: PyReadonlyArray2<'_, f64>) -> PyProblem {
	let mat = numpy_2d_to_mat(a);
	PyProblem {
		inner: DynProblem::Mat(DynMatProblem::Brockett(BrockettCost::new(mat))),
	}
}

#[pyfunction]
pub fn procrustes(a: PyReadonlyArray2<'_, f64>, b: PyReadonlyArray2<'_, f64>) -> PyProblem {
	let a_mat = numpy_2d_to_mat(a);
	let b_mat = numpy_2d_to_mat(b);
	PyProblem {
		inner: DynProblem::Mat(DynMatProblem::Procrustes(OrthogonalProcrustes::new(
			a_mat, b_mat,
		))),
	}
}
