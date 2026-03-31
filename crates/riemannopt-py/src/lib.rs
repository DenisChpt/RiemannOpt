use pyo3::prelude::*;

mod autodiff;
mod convert;
mod dispatch;
mod manifold;
mod problem;
mod result;
mod solver;
mod stopping;

/// Force all internal linear algebra to run on a single thread.
#[pyfunction]
fn disable_parallelism() {
	riemannopt_core::linalg::parallel_policy::Policy::disable_parallelism();
}

/// Re-enable adaptive multi-threading (the default).
#[pyfunction]
fn enable_parallelism() {
	riemannopt_core::linalg::parallel_policy::Policy::enable_parallelism();
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
	riemannopt_core::linalg::parallel_policy::Policy::init_with_calibration();
	m.add_function(wrap_pyfunction!(disable_parallelism, m)?)?;
	m.add_function(wrap_pyfunction!(enable_parallelism, m)?)?;
	// Manifold
	m.add_class::<manifold::PyManifold>()?;
	m.add_function(wrap_pyfunction!(manifold::sphere, m)?)?;
	m.add_function(wrap_pyfunction!(manifold::stiefel, m)?)?;
	m.add_function(wrap_pyfunction!(manifold::grassmann, m)?)?;
	m.add_function(wrap_pyfunction!(manifold::euclidean, m)?)?;
	m.add_function(wrap_pyfunction!(manifold::spd, m)?)?;
	m.add_function(wrap_pyfunction!(manifold::hyperbolic, m)?)?;
	m.add_function(wrap_pyfunction!(manifold::oblique, m)?)?;

	// Solvers
	m.add_class::<solver::PySolver>()?;
	m.add_function(wrap_pyfunction!(solver::sgd, m)?)?;
	m.add_function(wrap_pyfunction!(solver::adam, m)?)?;
	m.add_function(wrap_pyfunction!(solver::lbfgs, m)?)?;
	m.add_function(wrap_pyfunction!(solver::cg, m)?)?;
	m.add_function(wrap_pyfunction!(solver::trust_region, m)?)?;

	// AutoDiff
	m.add_class::<autodiff::PyAdSession>()?;
	m.add_class::<autodiff::ScalarVar>()?;
	m.add_class::<autodiff::VectorVar>()?;
	m.add_class::<autodiff::MatrixVar>()?;

	// Problem
	m.add_class::<problem::PyProblem>()?;
	m.add_function(wrap_pyfunction!(problem::rayleigh_quotient, m)?)?;
	m.add_function(wrap_pyfunction!(problem::quadratic_cost, m)?)?;
	m.add_function(wrap_pyfunction!(problem::rosenbrock, m)?)?;
	m.add_function(wrap_pyfunction!(problem::brockett_cost, m)?)?;
	m.add_function(wrap_pyfunction!(problem::procrustes, m)?)?;

	// Result + Stopping
	m.add_class::<result::PySolverResult>()?;
	m.add_class::<stopping::PyStoppingCriterion>()?;

	// Top-level solve
	m.add_function(wrap_pyfunction!(dispatch::solve, m)?)?;

	Ok(())
}
