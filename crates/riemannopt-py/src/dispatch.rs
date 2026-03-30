//! Dispatch `solve()` across (solver × manifold × problem) combinations.
//!
//! Strategy: group manifolds by Point type (vector vs matrix), then
//! dispatch solver and problem within each group via macros.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

use riemannopt_core::solver::{Solver, StoppingCriterion};

use crate::convert::{Vec64, Mat64};
use crate::manifold::{DynManifold, PyManifold, PointKind};
use crate::problem::{DynMatProblem, DynProblem, DynVecProblem, PyProblem};
use crate::result::PySolverResult;
use crate::solver::{DynSolver, PySolver, with_solver};
use crate::stopping::PyStoppingCriterion;

// ════════════════════════════════════════════════════════════════════════
//  Vector-manifold dispatch
// ════════════════════════════════════════════════════════════════════════

/// Dispatch for vector-manifold × problem, given a concrete solver `s`.
fn solve_vec<S: Solver<f64>>(
	s: &mut S,
	manifold: &DynManifold,
	problem: &DynVecProblem,
	point: &Vec64,
	stop: &StoppingCriterion<f64>,
) -> PySolverResult {
	macro_rules! on_manifold {
		($m:expr) => {
			match problem {
				DynVecProblem::AutoDiff(p) => PySolverResult::from_vec_result(s.solve(p, $m, point, stop)),
				DynVecProblem::Rayleigh(p) => PySolverResult::from_vec_result(s.solve(p, $m, point, stop)),
				DynVecProblem::Quadratic(p) => PySolverResult::from_vec_result(s.solve(p, $m, point, stop)),
				DynVecProblem::Rosenbrock(p) => PySolverResult::from_vec_result(s.solve(p, $m, point, stop)),
			}
		};
	}

	match manifold {
		DynManifold::Euclidean(m) => on_manifold!(m),
		DynManifold::Sphere(m) => on_manifold!(m),
		DynManifold::Hyperbolic(m) => on_manifold!(m),
		_ => unreachable!("solve_vec called on non-vector manifold"),
	}
}

// ════════════════════════════════════════════════════════════════════════
//  Matrix-manifold dispatch
// ════════════════════════════════════════════════════════════════════════

fn solve_mat<S: Solver<f64>>(
	s: &mut S,
	manifold: &DynManifold,
	problem: &DynMatProblem,
	point: &Mat64,
	stop: &StoppingCriterion<f64>,
) -> PySolverResult {
	macro_rules! on_manifold {
		($m:expr) => {
			match problem {
				DynMatProblem::AutoDiff(p) => PySolverResult::from_mat_result(s.solve(p, $m, point, stop)),
				DynMatProblem::Brockett(p) => PySolverResult::from_mat_result(s.solve(p, $m, point, stop)),
				DynMatProblem::Procrustes(p) => PySolverResult::from_mat_result(s.solve(p, $m, point, stop)),
			}
		};
	}

	match manifold {
		DynManifold::Stiefel(m) => on_manifold!(m),
		DynManifold::Grassmann(m) => on_manifold!(m),
		DynManifold::SPD(m) => on_manifold!(m),
		DynManifold::Oblique(m) => on_manifold!(m),
		_ => unreachable!("solve_mat called on non-matrix manifold"),
	}
}

// ════════════════════════════════════════════════════════════════════════
//  Top-level Python entry point
// ════════════════════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (solver, problem, manifold, initial_point, stopping=None))]
pub fn solve<'py>(
	py: Python<'py>,
	solver: &mut PySolver,
	problem: &PyProblem,
	manifold: &PyManifold,
	initial_point: PyReadonlyArrayDyn<'py, f64>,
	stopping: Option<&PyStoppingCriterion>,
) -> PyResult<PySolverResult> {
	let stop = stopping
		.map(|s| s.inner.clone())
		.unwrap_or_default();

	let kind = manifold.inner.point_kind();

	match kind {
		PointKind::Vector => {
			// Convert numpy -> faer vector
			let ndim = initial_point.shape().len();
			if ndim != 1 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					format!("Expected 1D array for vector manifold, got {ndim}D"),
				));
			}
			let flat = initial_point
				.as_slice()
				.map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
			let point = <Vec64 as riemannopt_core::linalg::VectorOps<f64>>::from_slice(flat);

			// Extract problem
			let vec_problem = match &problem.inner {
				DynProblem::Vec(p) => p,
				DynProblem::Mat(_) => {
					return Err(pyo3::exceptions::PyValueError::new_err(
						"Matrix problem cannot be used with a vector manifold",
					));
				}
			};

			// GIL released — entire solver loop runs in pure Rust
			let result = py.detach(|| {
				with_solver!(&mut solver.inner, |s| {
					solve_vec(s, &manifold.inner, vec_problem, &point, &stop)
				})
			});

			Ok(result)
		}
		PointKind::Matrix => {
			let shape = initial_point.shape();
			if shape.len() != 2 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					format!("Expected 2D array for matrix manifold, got {}D", shape.len()),
				));
			}
			let nrows = shape[0];
			let ncols = shape[1];
			let arr = initial_point.as_array();
			let point = <Mat64 as riemannopt_core::linalg::MatrixOps<f64>>::from_fn(
				nrows, ncols, |i, j| arr[[i, j]],
			);

			let mat_problem = match &problem.inner {
				DynProblem::Mat(p) => p,
				DynProblem::Vec(_) => {
					return Err(pyo3::exceptions::PyValueError::new_err(
						"Vector problem cannot be used with a matrix manifold",
					));
				}
			};

			let result = py.detach(|| {
				with_solver!(&mut solver.inner, |s| {
					solve_mat(s, &manifold.inner, mat_problem, &point, &stop)
				})
			});

			Ok(result)
		}
	}
}
