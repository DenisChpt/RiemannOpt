//! Dispatch `solve()` across (solver × manifold × problem × preconditioner).
//!
//! Strategy: group manifolds by Point type (vector vs matrix), then
//! dispatch solver, problem, and preconditioner within each group via macros.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

use riemannopt_core::solver::{Solver, StoppingCriterion};

use crate::convert::{Mat64, Vec64};
use crate::manifold::{DynManifold, PointKind, PyManifold};
use crate::preconditioner::{with_preconditioner, DynPreconditioner, PyPreconditioner};
use crate::problem::{DynMatProblem, DynProblem, DynVecProblem, PyProblem};
use crate::result::PySolverResult;
use crate::solver::{with_solver, DynSolver, PySolver};
use crate::stopping::PyStoppingCriterion;

// ════════════════════════════════════════════════════════════════════════
//  Vector-manifold dispatch
// ════════════════════════════════════════════════════════════════════════

fn solve_vec<S: Solver<f64>>(
	s: &mut S,
	manifold: &DynManifold,
	problem: &DynVecProblem,
	point: &Vec64,
	pre: &mut DynPreconditioner,
	stop: &StoppingCriterion<f64>,
) -> PySolverResult {
	macro_rules! on_manifold {
		($m:expr) => {
			with_preconditioner!(pre, $m, |p| {
				match problem {
					DynVecProblem::AutoDiff(prob) => {
						PySolverResult::from_vec_result(s.solve(prob, $m, point, p, stop))
					}
					DynVecProblem::Rayleigh(prob) => {
						PySolverResult::from_vec_result(s.solve(prob, $m, point, p, stop))
					}
					DynVecProblem::Quadratic(prob) => {
						PySolverResult::from_vec_result(s.solve(prob, $m, point, p, stop))
					}
					DynVecProblem::Rosenbrock(prob) => {
						PySolverResult::from_vec_result(s.solve(prob, $m, point, p, stop))
					}
				}
			})
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
	pre: &mut DynPreconditioner,
	stop: &StoppingCriterion<f64>,
) -> PySolverResult {
	macro_rules! on_manifold {
		($m:expr) => {
			with_preconditioner!(pre, $m, |p| {
				match problem {
					DynMatProblem::AutoDiff(prob) => {
						PySolverResult::from_mat_result(s.solve(prob, $m, point, p, stop))
					}
					DynMatProblem::Brockett(prob) => {
						PySolverResult::from_mat_result(s.solve(prob, $m, point, p, stop))
					}
					DynMatProblem::Procrustes(prob) => {
						PySolverResult::from_mat_result(s.solve(prob, $m, point, p, stop))
					}
				}
			})
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

/// Solve a Riemannian optimization problem.
///
/// Args:
///     solver: The optimization algorithm to use.
///     problem: The objective function (cost + gradient).
///     manifold: The Riemannian manifold constraint.
///     initial_point: Starting point as a numpy array.
///     stopping: Optional stopping criteria.
///     preconditioner: Optional preconditioner for second-order solvers.
///         When ``None``, no preconditioning is applied.
///
/// Returns:
///     SolverResult with the optimal point and convergence info.
#[pyfunction]
#[pyo3(signature = (solver, problem, manifold, initial_point, stopping=None, preconditioner=None))]
pub fn solve<'py>(
	py: Python<'py>,
	solver: &mut PySolver,
	problem: &PyProblem,
	manifold: &PyManifold,
	initial_point: PyReadonlyArrayDyn<'py, f64>,
	stopping: Option<&PyStoppingCriterion>,
	preconditioner: Option<&mut PyPreconditioner>,
) -> PyResult<PySolverResult> {
	let stop = stopping.map(|s| s.inner.clone()).unwrap_or_default();

	let mut default_pre;
	let pre: &mut DynPreconditioner = match preconditioner {
		Some(py_pre) => &mut py_pre.inner,
		None => {
			default_pre = DynPreconditioner::default();
			&mut default_pre
		}
	};

	let kind = manifold.inner.point_kind();

	match kind {
		PointKind::Vector => {
			let ndim = initial_point.shape().len();
			if ndim != 1 {
				return Err(pyo3::exceptions::PyValueError::new_err(format!(
					"Expected 1D array for vector manifold, got {ndim}D"
				)));
			}
			let flat = initial_point
				.as_slice()
				.map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
			let point = <Vec64 as riemannopt_core::linalg::VectorOps<f64>>::from_slice(flat);

			let vec_problem = match &problem.inner {
				DynProblem::Vec(p) => p,
				DynProblem::Mat(_) => {
					return Err(pyo3::exceptions::PyValueError::new_err(
						"Matrix problem cannot be used with a vector manifold",
					));
				}
			};

			let result = py.detach(|| {
				with_solver!(&mut solver.inner, |s| {
					solve_vec(s, &manifold.inner, vec_problem, &point, pre, &stop)
				})
			});

			Ok(result)
		}
		PointKind::Matrix => {
			let shape = initial_point.shape();
			if shape.len() != 2 {
				return Err(pyo3::exceptions::PyValueError::new_err(format!(
					"Expected 2D array for matrix manifold, got {}D",
					shape.len()
				)));
			}
			let nrows = shape[0];
			let ncols = shape[1];
			let arr = initial_point.as_array();
			let point = <Mat64 as riemannopt_core::linalg::MatrixOps<f64>>::from_fn(
				nrows,
				ncols,
				|i, j| arr[[i, j]],
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
					solve_mat(s, &manifold.inner, mat_problem, &point, pre, &stop)
				})
			});

			Ok(result)
		}
	}
}
