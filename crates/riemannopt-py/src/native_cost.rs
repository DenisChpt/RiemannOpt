//! Native Rust cost functions exposed to Python.
//!
//! These cost functions run entirely in Rust with no Python callback overhead.
//! Parameters (matrices A, b, N) are converted from NumPy once at construction
//! time, then all cost/gradient evaluations happen in pure Rust.

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use riemannopt_core::{
	cost_function::CostFunction,
	error::{ManifoldError, Result},
	memory::workspace::Workspace,
};

use crate::array_utils::{dmatrix_to_numpy, dvector_to_numpy, numpy_to_dmatrix, numpy_to_dvector};

// ═══════════════════════════════════════════════════════════════════════════
//  1. Rayleigh Quotient: f(x) = x^T A x,  grad = 2Ax
// ═══════════════════════════════════════════════════════════════════════════

/// Native Rust cost function for the Rayleigh quotient f(x) = x^T A x.
///
/// Use with Sphere manifold. The minimum on S^{n-1} equals the smallest
/// eigenvalue of A and is attained at the corresponding eigenvector.
#[pyclass(name = "RayleighQuotient", module = "riemannopt")]
#[derive(Clone)]
pub struct PyRayleighQuotient {
	a: DMatrix<f64>,
	dim: usize,
}

#[pymethods]
impl PyRayleighQuotient {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
		let mat = numpy_to_dmatrix(a)?;
		let dim = mat.nrows();
		if mat.ncols() != dim {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"Matrix A must be square",
			));
		}
		Ok(Self { a: mat, dim })
	}

	#[pyo3(name = "cost")]
	fn py_cost(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
		let xv = numpy_to_dvector(x)?;
		let ax = &self.a * &xv;
		Ok(xv.dot(&ax))
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray1<'py, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let xv = numpy_to_dvector(x)?;
		let grad = &self.a * &xv * 2.0;
		dvector_to_numpy(py, &grad)
	}

	fn __repr__(&self) -> String {
		format!("RayleighQuotient(dim={})", self.dim)
	}
}

impl std::fmt::Debug for PyRayleighQuotient {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "RayleighQuotient(dim={})", self.dim)
	}
}

impl CostFunction<f64> for PyRayleighQuotient {
	type Point = DVector<f64>;
	type TangentVector = DVector<f64>;

	fn cost(&self, point: &DVector<f64>) -> Result<f64> {
		let ax = &self.a * point;
		Ok(point.dot(&ax))
	}

	fn cost_and_gradient_alloc(&self, point: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
		let ax = &self.a * point;
		let cost = point.dot(&ax);
		let gradient = &ax * 2.0;
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &DVector<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DVector<f64>,
	) -> Result<f64> {
		let ax = &self.a * point;
		let cost = point.dot(&ax);
		*gradient = &ax * 2.0;
		Ok(cost)
	}

	fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
		Ok(&self.a * point * 2.0)
	}

	fn hessian_vector_product(
		&self,
		_point: &DVector<f64>,
		vector: &DVector<f64>,
	) -> Result<DVector<f64>> {
		Ok(&self.a * vector * 2.0)
	}

	fn gradient_fd_alloc(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(
		&self,
		point: &DVector<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DVector<f64>,
	) -> Result<()> {
		*gradient = &self.a * point * 2.0;
		Ok(())
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. Trace Minimization: f(Y) = tr(Y^T A Y),  grad = 2AY
// ═══════════════════════════════════════════════════════════════════════════

/// Native Rust cost function for trace minimization f(Y) = tr(Y^T A Y).
///
/// Use with Grassmann or Stiefel manifold. The minimum equals the sum of
/// the p smallest eigenvalues of A (where Y is n×p).
#[pyclass(name = "TraceMinimization", module = "riemannopt")]
#[derive(Clone)]
pub struct PyTraceMinimization {
	a: DMatrix<f64>,
	rows: usize,
}

#[pymethods]
impl PyTraceMinimization {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
		let mat = numpy_to_dmatrix(a)?;
		let n = mat.nrows();
		if mat.ncols() != n {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"Matrix A must be square",
			));
		}
		Ok(Self { a: mat, rows: n })
	}

	#[pyo3(name = "cost")]
	fn py_cost(&self, y: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
		let ym = numpy_to_dmatrix(y)?;
		let ay = &self.a * &ym;
		Ok((ym.transpose() * ay).trace())
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		y: PyReadonlyArray2<'py, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let ym = numpy_to_dmatrix(y)?;
		let grad = &self.a * &ym * 2.0;
		dmatrix_to_numpy(py, &grad)
	}

	fn __repr__(&self) -> String {
		format!("TraceMinimization(n={})", self.rows)
	}
}

impl std::fmt::Debug for PyTraceMinimization {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "TraceMinimization(n={})", self.rows)
	}
}

impl CostFunction<f64> for PyTraceMinimization {
	type Point = DMatrix<f64>;
	type TangentVector = DMatrix<f64>;

	fn cost(&self, point: &DMatrix<f64>) -> Result<f64> {
		let ay = &self.a * point;
		Ok((point.transpose() * ay).trace())
	}

	fn cost_and_gradient_alloc(&self, point: &DMatrix<f64>) -> Result<(f64, DMatrix<f64>)> {
		let ay = &self.a * point;
		let cost = (point.transpose() * &ay).trace();
		let gradient = &ay * 2.0;
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<f64> {
		let ay = &self.a * point;
		let cost = (point.transpose() * &ay).trace();
		*gradient = &ay * 2.0;
		Ok(cost)
	}

	fn gradient(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		Ok(&self.a * point * 2.0)
	}

	fn hessian_vector_product(
		&self,
		_point: &DMatrix<f64>,
		vector: &DMatrix<f64>,
	) -> Result<DMatrix<f64>> {
		Ok(&self.a * vector * 2.0)
	}

	fn gradient_fd_alloc(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<()> {
		*gradient = &self.a * point * 2.0;
		Ok(())
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. Brockett: f(X) = tr(X^T A X N),  grad = 2AXN  (for symmetric A, N)
// ═══════════════════════════════════════════════════════════════════════════

/// Native Rust cost function for the Brockett problem f(X) = tr(X^T A X N).
///
/// Use with Stiefel manifold.
#[pyclass(name = "Brockett", module = "riemannopt")]
#[derive(Clone)]
pub struct PyBrockett {
	a: DMatrix<f64>,
	n_mat: DMatrix<f64>,
	rows: usize,
	cols: usize,
}

#[pymethods]
impl PyBrockett {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>, n: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
		let a_mat = numpy_to_dmatrix(a)?;
		let n_mat = numpy_to_dmatrix(n)?;
		let rows = a_mat.nrows();
		let cols = n_mat.nrows();
		if a_mat.ncols() != rows {
			return Err(pyo3::exceptions::PyValueError::new_err("A must be square"));
		}
		if n_mat.ncols() != cols {
			return Err(pyo3::exceptions::PyValueError::new_err("N must be square"));
		}
		Ok(Self {
			a: a_mat,
			n_mat,
			rows,
			cols,
		})
	}

	#[pyo3(name = "cost")]
	fn py_cost(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
		let xm = numpy_to_dmatrix(x)?;
		let ax = &self.a * &xm;
		Ok((xm.transpose() * ax * &self.n_mat).trace())
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray2<'py, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let xm = numpy_to_dmatrix(x)?;
		let grad = &self.a * &xm * &self.n_mat * 2.0;
		dmatrix_to_numpy(py, &grad)
	}

	fn __repr__(&self) -> String {
		format!("Brockett(n={}, p={})", self.rows, self.cols)
	}
}

impl std::fmt::Debug for PyBrockett {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Brockett(n={}, p={})", self.rows, self.cols)
	}
}

impl CostFunction<f64> for PyBrockett {
	type Point = DMatrix<f64>;
	type TangentVector = DMatrix<f64>;

	fn cost(&self, point: &DMatrix<f64>) -> Result<f64> {
		let ax = &self.a * point;
		Ok((point.transpose() * ax * &self.n_mat).trace())
	}

	fn cost_and_gradient_alloc(&self, point: &DMatrix<f64>) -> Result<(f64, DMatrix<f64>)> {
		let ax = &self.a * point;
		let cost = (point.transpose() * &ax * &self.n_mat).trace();
		let gradient = &ax * &self.n_mat * 2.0;
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<f64> {
		let ax = &self.a * point;
		let cost = (point.transpose() * &ax * &self.n_mat).trace();
		*gradient = &ax * &self.n_mat * 2.0;
		Ok(cost)
	}

	fn gradient(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		Ok(&self.a * point * &self.n_mat * 2.0)
	}

	fn hessian_vector_product(
		&self,
		_point: &DMatrix<f64>,
		vector: &DMatrix<f64>,
	) -> Result<DMatrix<f64>> {
		Ok(&self.a * vector * &self.n_mat * 2.0)
	}

	fn gradient_fd_alloc(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<()> {
		*gradient = &self.a * point * &self.n_mat * 2.0;
		Ok(())
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. Log-Det Divergence: f(P) = tr(P) - log det(P),  grad = I - P^{-1}
// ═══════════════════════════════════════════════════════════════════════════

/// Native Rust cost function for log-det divergence f(P) = tr(P) - log det(P).
///
/// Use with SPD manifold. The minimum is f* = n at P = I.
#[pyclass(name = "LogDetDivergence", module = "riemannopt")]
#[derive(Clone)]
pub struct PyLogDetDivergence {
	dim: usize,
}

#[pymethods]
impl PyLogDetDivergence {
	#[new]
	fn new(dim: usize) -> Self {
		Self { dim }
	}

	#[pyo3(name = "cost")]
	fn py_cost(&self, p: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
		let pm = numpy_to_dmatrix(p)?;
		let trace = pm.trace();
		let chol = pm.clone().cholesky().ok_or_else(|| {
			pyo3::exceptions::PyValueError::new_err("Matrix is not positive definite")
		})?;
		let log_det: f64 = chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>() * 2.0;
		Ok(trace - log_det)
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		p: PyReadonlyArray2<'py, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let pm = numpy_to_dmatrix(p)?;
		let n = pm.nrows();
		let chol = pm.clone().cholesky().ok_or_else(|| {
			pyo3::exceptions::PyValueError::new_err("Matrix is not positive definite")
		})?;
		let p_inv = chol.inverse();
		let grad = DMatrix::identity(n, n) - p_inv;
		dmatrix_to_numpy(py, &grad)
	}

	fn __repr__(&self) -> String {
		format!("LogDetDivergence(dim={})", self.dim)
	}
}

impl std::fmt::Debug for PyLogDetDivergence {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "LogDetDivergence(dim={})", self.dim)
	}
}

impl CostFunction<f64> for PyLogDetDivergence {
	type Point = DMatrix<f64>;
	type TangentVector = DMatrix<f64>;

	fn cost(&self, point: &DMatrix<f64>) -> Result<f64> {
		let trace = point.trace();
		let chol = point.clone().cholesky().ok_or_else(|| {
			ManifoldError::numerical_error("Matrix is not positive definite in LogDetDivergence")
		})?;
		let log_det: f64 = chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>() * 2.0;
		Ok(trace - log_det)
	}

	fn cost_and_gradient_alloc(&self, point: &DMatrix<f64>) -> Result<(f64, DMatrix<f64>)> {
		let n = point.nrows();
		let trace = point.trace();
		let chol = point.clone().cholesky().ok_or_else(|| {
			ManifoldError::numerical_error("Matrix is not positive definite in LogDetDivergence")
		})?;
		let log_det: f64 = chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>() * 2.0;
		let p_inv = chol.inverse();
		let gradient = DMatrix::identity(n, n) - p_inv;
		Ok((trace - log_det, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<f64> {
		let n = point.nrows();
		let trace = point.trace();
		let chol = point.clone().cholesky().ok_or_else(|| {
			ManifoldError::numerical_error("Matrix is not positive definite in LogDetDivergence")
		})?;
		let log_det: f64 = chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>() * 2.0;
		let p_inv = chol.inverse();
		*gradient = DMatrix::identity(n, n) - p_inv;
		Ok(trace - log_det)
	}

	fn gradient(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		let n = point.nrows();
		let chol = point.clone().cholesky().ok_or_else(|| {
			ManifoldError::numerical_error("Matrix is not positive definite in LogDetDivergence")
		})?;
		let p_inv = chol.inverse();
		Ok(DMatrix::identity(n, n) - p_inv)
	}

	fn hessian_vector_product(
		&self,
		point: &DMatrix<f64>,
		vector: &DMatrix<f64>,
	) -> Result<DMatrix<f64>> {
		let chol = point.clone().cholesky().ok_or_else(|| {
			ManifoldError::numerical_error("Matrix is not positive definite in LogDetDivergence")
		})?;
		let p_inv = chol.inverse();
		Ok(&p_inv * vector * &p_inv)
	}

	fn gradient_fd_alloc(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<()> {
		*gradient = CostFunction::gradient(self, point)?;
		Ok(())
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. Quadratic: f(x) = 0.5 x^T A x + b^T x,  grad = Ax + b
// ═══════════════════════════════════════════════════════════════════════════

/// Native Rust cost function for quadratic f(x) = 0.5 x^T A x + b^T x.
///
/// Use with Euclidean manifold. The minimum is x* = -A^{-1}b with
/// f* = -0.5 b^T A^{-1} b.
#[pyclass(name = "Quadratic", module = "riemannopt")]
#[derive(Clone)]
pub struct PyQuadratic {
	a: DMatrix<f64>,
	b: DVector<f64>,
	dim: usize,
}

#[pymethods]
impl PyQuadratic {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
		let a_mat = numpy_to_dmatrix(a)?;
		let b_vec = numpy_to_dvector(b)?;
		let dim = a_mat.nrows();
		if a_mat.ncols() != dim {
			return Err(pyo3::exceptions::PyValueError::new_err("A must be square"));
		}
		if b_vec.len() != dim {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"b dimension must match A",
			));
		}
		Ok(Self {
			a: a_mat,
			b: b_vec,
			dim,
		})
	}

	#[pyo3(name = "cost")]
	fn py_cost(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
		let xv = numpy_to_dvector(x)?;
		let ax = &self.a * &xv;
		Ok(0.5 * xv.dot(&ax) + self.b.dot(&xv))
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray1<'py, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let xv = numpy_to_dvector(x)?;
		let grad = &self.a * &xv + &self.b;
		dvector_to_numpy(py, &grad)
	}

	fn __repr__(&self) -> String {
		format!("Quadratic(dim={})", self.dim)
	}
}

impl std::fmt::Debug for PyQuadratic {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Quadratic(dim={})", self.dim)
	}
}

impl CostFunction<f64> for PyQuadratic {
	type Point = DVector<f64>;
	type TangentVector = DVector<f64>;

	fn cost(&self, point: &DVector<f64>) -> Result<f64> {
		let ax = &self.a * point;
		Ok(0.5 * point.dot(&ax) + self.b.dot(point))
	}

	fn cost_and_gradient_alloc(&self, point: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
		let ax = &self.a * point;
		let cost = 0.5 * point.dot(&ax) + self.b.dot(point);
		let gradient = ax + &self.b;
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &DVector<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DVector<f64>,
	) -> Result<f64> {
		let ax = &self.a * point;
		let cost = 0.5 * point.dot(&ax) + self.b.dot(point);
		*gradient = ax + &self.b;
		Ok(cost)
	}

	fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
		Ok(&self.a * point + &self.b)
	}

	fn hessian_vector_product(
		&self,
		_point: &DVector<f64>,
		vector: &DVector<f64>,
	) -> Result<DVector<f64>> {
		Ok(&self.a * vector)
	}

	fn gradient_fd_alloc(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(
		&self,
		point: &DVector<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DVector<f64>,
	) -> Result<()> {
		*gradient = &self.a * point + &self.b;
		Ok(())
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Module registration
// ═══════════════════════════════════════════════════════════════════════════

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
	parent.add_class::<PyRayleighQuotient>()?;
	parent.add_class::<PyTraceMinimization>()?;
	parent.add_class::<PyBrockett>()?;
	parent.add_class::<PyLogDetDivergence>()?;
	parent.add_class::<PyQuadratic>()?;
	Ok(())
}
