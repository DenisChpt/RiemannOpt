//! Native Rust cost functions exposed to Python.
//!
//! These cost functions run entirely in Rust with no Python callback overhead.
//! Parameters (matrices A, b, N) are converted from NumPy once at construction
//! time, then all cost/gradient evaluations happen in pure Rust.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use riemannopt_core::{
	cost_function::CostFunction,
	error::{ManifoldError, Result},
	linalg::{DecompositionOps, MatrixOps, MatrixView, VectorOps},
};

use crate::array_utils::{mat_to_numpy, numpy_to_mat, numpy_to_vec, vec_to_numpy, Mat64, Vec64};

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
	a: Mat64,
	dim: usize,
}

#[pymethods]
impl PyRayleighQuotient {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
		let mat = numpy_to_mat(a)?;
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
		let xv = numpy_to_vec(x)?;
		let ax = self.a.mat_vec(&xv);
		Ok(xv.dot(&ax))
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray1<'py, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let xv = numpy_to_vec(x)?;
		let mut grad = self.a.mat_vec(&xv);
		grad.scale_mut(2.0);
		vec_to_numpy(py, &grad)
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
	type Point = Vec64;
	type TangentVector = Vec64;

	fn cost(&self, point: &Vec64) -> Result<f64> {
		let ax = self.a.mat_vec(point);
		Ok(point.dot(&ax))
	}

	fn cost_and_gradient_alloc(&self, point: &Vec64) -> Result<(f64, Vec64)> {
		let ax = self.a.mat_vec(point);
		let cost = point.dot(&ax);
		let mut gradient = ax;
		gradient.scale_mut(2.0);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(&self, point: &Vec64, gradient: &mut Vec64) -> Result<f64> {
		let ax = self.a.mat_vec(point);
		let cost = point.dot(&ax);
		gradient.copy_from(&ax);
		gradient.scale_mut(2.0);
		Ok(cost)
	}

	fn gradient(&self, point: &Vec64) -> Result<Vec64> {
		let mut g = self.a.mat_vec(point);
		g.scale_mut(2.0);
		Ok(g)
	}

	fn hessian_vector_product(&self, _point: &Vec64, vector: &Vec64) -> Result<Vec64> {
		let mut hv = self.a.mat_vec(vector);
		hv.scale_mut(2.0);
		Ok(hv)
	}

	fn gradient_fd_alloc(&self, point: &Vec64) -> Result<Vec64> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(&self, point: &Vec64, gradient: &mut Vec64) -> Result<()> {
		let g = self.a.mat_vec(point);
		gradient.copy_from(&g);
		gradient.scale_mut(2.0);
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
	a: Mat64,
	rows: usize,
}

#[pymethods]
impl PyTraceMinimization {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
		let mat = numpy_to_mat(a)?;
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
		let ym = numpy_to_mat(y)?;
		let ay = self.a.mat_mul(&ym);
		Ok(MatrixOps::transpose_to_owned(&ym).mat_mul(&ay).trace())
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		y: PyReadonlyArray2<'py, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let ym = numpy_to_mat(y)?;
		let mut grad = self.a.mat_mul(&ym);
		grad.scale_mut(2.0);
		mat_to_numpy(py, &grad)
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
	type Point = Mat64;
	type TangentVector = Mat64;

	fn cost(&self, point: &Mat64) -> Result<f64> {
		let ay = self.a.mat_mul(point);
		Ok(MatrixOps::transpose_to_owned(point).mat_mul(&ay).trace())
	}

	fn cost_and_gradient_alloc(&self, point: &Mat64) -> Result<(f64, Mat64)> {
		let ay = self.a.mat_mul(point);
		let cost = MatrixOps::transpose_to_owned(point).mat_mul(&ay).trace();
		let mut gradient = ay;
		gradient.scale_mut(2.0);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(&self, point: &Mat64, gradient: &mut Mat64) -> Result<f64> {
		let ay = self.a.mat_mul(point);
		let cost = MatrixOps::transpose_to_owned(point).mat_mul(&ay).trace();
		gradient.copy_from(&ay);
		gradient.scale_mut(2.0);
		Ok(cost)
	}

	fn gradient(&self, point: &Mat64) -> Result<Mat64> {
		let mut g = self.a.mat_mul(point);
		g.scale_mut(2.0);
		Ok(g)
	}

	fn hessian_vector_product(&self, _point: &Mat64, vector: &Mat64) -> Result<Mat64> {
		let mut hv = self.a.mat_mul(vector);
		hv.scale_mut(2.0);
		Ok(hv)
	}

	fn gradient_fd_alloc(&self, point: &Mat64) -> Result<Mat64> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(&self, point: &Mat64, gradient: &mut Mat64) -> Result<()> {
		*gradient = CostFunction::gradient(self, point)?;
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
	a: Mat64,
	n_mat: Mat64,
	rows: usize,
	cols: usize,
}

#[pymethods]
impl PyBrockett {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>, n: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
		let a_mat = numpy_to_mat(a)?;
		let n_mat = numpy_to_mat(n)?;
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
		let xm = numpy_to_mat(x)?;
		let ax = self.a.mat_mul(&xm);
		Ok(MatrixOps::transpose_to_owned(&xm)
			.mat_mul(&ax)
			.mat_mul(&self.n_mat)
			.trace())
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray2<'py, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let xm = numpy_to_mat(x)?;
		let mut grad = self.a.mat_mul(&xm).mat_mul(&self.n_mat);
		grad.scale_mut(2.0);
		mat_to_numpy(py, &grad)
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
	type Point = Mat64;
	type TangentVector = Mat64;

	fn cost(&self, point: &Mat64) -> Result<f64> {
		let ax = self.a.mat_mul(point);
		Ok(MatrixOps::transpose_to_owned(point)
			.mat_mul(&ax)
			.mat_mul(&self.n_mat)
			.trace())
	}

	fn cost_and_gradient_alloc(&self, point: &Mat64) -> Result<(f64, Mat64)> {
		let ax = self.a.mat_mul(point);
		let cost = MatrixOps::transpose_to_owned(point)
			.mat_mul(&ax)
			.mat_mul(&self.n_mat)
			.trace();
		let mut gradient = ax.mat_mul(&self.n_mat);
		gradient.scale_mut(2.0);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(&self, point: &Mat64, gradient: &mut Mat64) -> Result<f64> {
		let ax = self.a.mat_mul(point);
		let cost = MatrixOps::transpose_to_owned(point)
			.mat_mul(&ax)
			.mat_mul(&self.n_mat)
			.trace();
		let g = ax.mat_mul(&self.n_mat);
		gradient.copy_from(&g);
		gradient.scale_mut(2.0);
		Ok(cost)
	}

	fn gradient(&self, point: &Mat64) -> Result<Mat64> {
		let mut g = self.a.mat_mul(point).mat_mul(&self.n_mat);
		g.scale_mut(2.0);
		Ok(g)
	}

	fn hessian_vector_product(&self, _point: &Mat64, vector: &Mat64) -> Result<Mat64> {
		let mut hv = self.a.mat_mul(vector).mat_mul(&self.n_mat);
		hv.scale_mut(2.0);
		Ok(hv)
	}

	fn gradient_fd_alloc(&self, point: &Mat64) -> Result<Mat64> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(&self, point: &Mat64, gradient: &mut Mat64) -> Result<()> {
		*gradient = CostFunction::gradient(self, point)?;
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
		let pm = numpy_to_mat(p)?;
		let trace_val = pm.trace();
		let chol = pm.cholesky().ok_or_else(|| {
			pyo3::exceptions::PyValueError::new_err("Matrix is not positive definite")
		})?;
		let l = chol.l();
		let mut log_det = 0.0_f64;
		for i in 0..l.nrows() {
			log_det += l.get(i, i).ln();
		}
		log_det *= 2.0;
		Ok(trace_val - log_det)
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		p: PyReadonlyArray2<'py, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let pm = numpy_to_mat(p)?;
		let n = pm.nrows();
		let mut p_inv = <Mat64 as MatrixOps<f64>>::zeros(n, n);
		if !DecompositionOps::inverse(&pm, &mut p_inv) {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"Matrix is not invertible",
			));
		}
		let grad = <Mat64 as MatrixOps<f64>>::identity(n).sub(&p_inv);
		mat_to_numpy(py, &grad)
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
	type Point = Mat64;
	type TangentVector = Mat64;

	fn cost(&self, point: &Mat64) -> Result<f64> {
		let trace_val = point.trace();
		let chol = point.cholesky().ok_or_else(|| {
			ManifoldError::numerical_error("Matrix is not positive definite in LogDetDivergence")
		})?;
		let l = chol.l();
		let mut log_det = 0.0_f64;
		for i in 0..l.nrows() {
			log_det += l.get(i, i).ln();
		}
		log_det *= 2.0;
		Ok(trace_val - log_det)
	}

	fn cost_and_gradient_alloc(&self, point: &Mat64) -> Result<(f64, Mat64)> {
		let n = point.nrows();
		let trace_val = point.trace();
		let chol = point.cholesky().ok_or_else(|| {
			ManifoldError::numerical_error("Matrix is not positive definite in LogDetDivergence")
		})?;
		let l = chol.l();
		let mut log_det = 0.0_f64;
		for i in 0..l.nrows() {
			log_det += l.get(i, i).ln();
		}
		log_det *= 2.0;
		let mut p_inv = <Mat64 as MatrixOps<f64>>::zeros(n, n);
		if !DecompositionOps::inverse(point, &mut p_inv) {
			return Err(ManifoldError::numerical_error(
				"Matrix is not invertible in LogDetDivergence",
			));
		}
		let gradient = <Mat64 as MatrixOps<f64>>::identity(n).sub(&p_inv);
		Ok((trace_val - log_det, gradient))
	}

	fn cost_and_gradient(&self, point: &Mat64, gradient: &mut Mat64) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		*gradient = grad;
		Ok(cost)
	}

	fn gradient(&self, point: &Mat64) -> Result<Mat64> {
		let n = point.nrows();
		let mut p_inv = <Mat64 as MatrixOps<f64>>::zeros(n, n);
		if !DecompositionOps::inverse(point, &mut p_inv) {
			return Err(ManifoldError::numerical_error(
				"Matrix is not invertible in LogDetDivergence",
			));
		}
		Ok(<Mat64 as MatrixOps<f64>>::identity(n).sub(&p_inv))
	}

	fn hessian_vector_product(&self, point: &Mat64, vector: &Mat64) -> Result<Mat64> {
		let n = point.nrows();
		let mut p_inv = <Mat64 as MatrixOps<f64>>::zeros(n, n);
		if !DecompositionOps::inverse(point, &mut p_inv) {
			return Err(ManifoldError::numerical_error(
				"Matrix is not invertible in LogDetDivergence",
			));
		}
		Ok(p_inv.mat_mul(vector).mat_mul(&p_inv))
	}

	fn gradient_fd_alloc(&self, point: &Mat64) -> Result<Mat64> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(&self, point: &Mat64, gradient: &mut Mat64) -> Result<()> {
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
	a: Mat64,
	b: Vec64,
	dim: usize,
}

#[pymethods]
impl PyQuadratic {
	#[new]
	fn new(a: PyReadonlyArray2<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
		let a_mat = numpy_to_mat(a)?;
		let b_vec = numpy_to_vec(b)?;
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
		let xv = numpy_to_vec(x)?;
		let ax = self.a.mat_vec(&xv);
		Ok(0.5 * xv.dot(&ax) + self.b.dot(&xv))
	}

	#[pyo3(name = "gradient")]
	fn py_gradient<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray1<'py, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let xv = numpy_to_vec(x)?;
		let grad = VectorOps::add(&self.a.mat_vec(&xv), &self.b);
		vec_to_numpy(py, &grad)
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
	type Point = Vec64;
	type TangentVector = Vec64;

	fn cost(&self, point: &Vec64) -> Result<f64> {
		let ax = self.a.mat_vec(point);
		Ok(0.5 * point.dot(&ax) + self.b.dot(point))
	}

	fn cost_and_gradient_alloc(&self, point: &Vec64) -> Result<(f64, Vec64)> {
		let ax = self.a.mat_vec(point);
		let cost = 0.5 * point.dot(&ax) + self.b.dot(point);
		let gradient = VectorOps::add(&ax, &self.b);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(&self, point: &Vec64, gradient: &mut Vec64) -> Result<f64> {
		let ax = self.a.mat_vec(point);
		let cost = 0.5 * point.dot(&ax) + self.b.dot(point);
		let g = VectorOps::add(&ax, &self.b);
		gradient.copy_from(&g);
		Ok(cost)
	}

	fn gradient(&self, point: &Vec64) -> Result<Vec64> {
		Ok(VectorOps::add(&self.a.mat_vec(point), &self.b))
	}

	fn hessian_vector_product(&self, _point: &Vec64, vector: &Vec64) -> Result<Vec64> {
		Ok(self.a.mat_vec(vector))
	}

	fn gradient_fd_alloc(&self, point: &Vec64) -> Result<Vec64> {
		CostFunction::gradient(self, point)
	}

	fn gradient_fd(&self, point: &Vec64, gradient: &mut Vec64) -> Result<()> {
		let g = VectorOps::add(&self.a.mat_vec(point), &self.b);
		gradient.copy_from(&g);
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
