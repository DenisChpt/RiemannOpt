//! Python bindings for the Fixed-Rank manifold.
//!
//! The fixed-rank manifold consists of matrices with a fixed rank constraint.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use riemannopt_core::linalg::{MatrixOps, VectorOps};
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::fixed_rank::{FixedRank, FixedRankPoint, FixedRankTangent};

use crate::{
	array_utils::{mat_to_numpy, numpy_to_mat, numpy_to_vec, vec_to_numpy},
	error::to_py_err,
};

use super::base::{PointType, PyManifoldBase};

/// Python wrapper for a point on the Fixed-Rank manifold.
///
/// A point on the fixed-rank manifold is represented using its SVD factorization:
/// X = U * S * V^T
///
/// where:
/// - U: m × k matrix with orthonormal columns (left singular vectors)
/// - S: k-dimensional vector of singular values
/// - V: n × k matrix with orthonormal columns (right singular vectors)
///
/// Attributes
/// ----------
/// u : ndarray
///     Left singular vectors (m × k matrix)
/// s : ndarray
///     Singular values (k-dimensional vector)
/// v : ndarray
///     Right singular vectors (n × k matrix)
#[pyclass(name = "FixedRankPoint", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyFixedRankPoint {
	/// Internal Rust representation
	pub(crate) inner: FixedRankPoint<f64>,
}

#[pymethods]
impl PyFixedRankPoint {
	/// Create a new FixedRankPoint from SVD components.
	///
	/// Args:
	///     u: Left singular vectors (m × k matrix with orthonormal columns)
	///     s: Singular values (k-dimensional positive vector)
	///     v: Right singular vectors (n × k matrix with orthonormal columns)
	///
	/// Returns:
	///     FixedRankPoint: The point on the fixed-rank manifold
	#[new]
	pub fn new(
		u: PyReadonlyArray2<'_, f64>,
		s: PyReadonlyArray1<'_, f64>,
		v: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Self> {
		let u_mat = numpy_to_mat(u)?;
		let s_vec = numpy_to_vec(s)?;
		let v_mat = numpy_to_mat(v)?;

		// Validate dimensions
		let k = s_vec.len();
		if u_mat.ncols() != k {
			return Err(crate::error::dimension_mismatch(
				&[u_mat.nrows(), k],
				&[u_mat.nrows(), u_mat.ncols()],
			));
		}
		if v_mat.ncols() != k {
			return Err(crate::error::dimension_mismatch(
				&[v_mat.nrows(), k],
				&[v_mat.nrows(), v_mat.ncols()],
			));
		}

		Ok(PyFixedRankPoint {
			inner: FixedRankPoint::new(u_mat, s_vec, v_mat),
		})
	}

	/// Get the left singular vectors U.
	#[getter]
	pub fn u<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		mat_to_numpy(py, &self.inner.u)
	}

	/// Get the singular values S.
	#[getter]
	pub fn s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
		vec_to_numpy(py, &self.inner.s)
	}

	/// Get the right singular vectors V.
	#[getter]
	pub fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		mat_to_numpy(py, &self.inner.v)
	}

	/// Convert to full matrix representation X = U * S * V^T.
	///
	/// Returns:
	///     ndarray: The full m × n matrix
	pub fn to_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let mat = self.inner.to_matrix();
		mat_to_numpy(py, &mat)
	}

	/// Create from a full matrix using truncated SVD.
	///
	/// Args:
	///     matrix: The m × n matrix to decompose
	///     k: The target rank
	///
	/// Returns:
	///     FixedRankPoint: The rank-k approximation
	#[staticmethod]
	pub fn from_matrix(matrix: PyReadonlyArray2<'_, f64>, k: usize) -> PyResult<Self> {
		let mat = numpy_to_mat(matrix)?;
		let point = FixedRankPoint::from_matrix(&mat, k).map_err(to_py_err)?;
		Ok(PyFixedRankPoint { inner: point })
	}

	fn __repr__(&self) -> String {
		format!(
			"FixedRankPoint(m={}, n={}, k={})",
			self.inner.u.nrows(),
			self.inner.v.nrows(),
			self.inner.s.len()
		)
	}
}

/// Python wrapper for a tangent vector on the Fixed-Rank manifold.
///
/// A tangent vector at a point X = U*S*V^T is represented as:
/// ξ = U_perp*M*V^T + U*S_dot*V^T + U*N*V_perp^T
///
/// where:
/// - U_perp*M*V^T: Component in the direction of increasing left singular space
/// - U*S_dot*V^T: Component changing the singular values
/// - U*N*V_perp^T: Component in the direction of increasing right singular space
///
/// Attributes
/// ----------
/// u_perp_m : ndarray
///     Left orthogonal component (m-k) × k matrix
/// s_dot : ndarray
///     Singular value derivatives (k-dimensional vector)
/// v_perp_n : ndarray
///     Right orthogonal component (n-k) × k matrix
#[pyclass(name = "FixedRankTangent", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyFixedRankTangent {
	/// Internal Rust representation
	pub(crate) inner: FixedRankTangent<f64>,
}

#[pymethods]
impl PyFixedRankTangent {
	/// Create a new FixedRankTangent from components.
	///
	/// Args:
	///     u_perp_m: Left orthogonal component ((m-k) × k matrix)
	///     s_dot: Singular value derivatives (k-dimensional vector)
	///     v_perp_n: Right orthogonal component ((n-k) × k matrix)
	///
	/// Returns:
	///     FixedRankTangent: The tangent vector
	#[new]
	pub fn new(
		u_perp_m: PyReadonlyArray2<'_, f64>,
		s_dot: PyReadonlyArray1<'_, f64>,
		v_perp_n: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Self> {
		let u_perp_m_mat = numpy_to_mat(u_perp_m)?;
		let s_dot_vec = numpy_to_vec(s_dot)?;
		let v_perp_n_mat = numpy_to_mat(v_perp_n)?;

		Ok(PyFixedRankTangent {
			inner: FixedRankTangent::new(u_perp_m_mat, s_dot_vec, v_perp_n_mat),
		})
	}

	/// Get the left orthogonal component.
	#[getter]
	pub fn u_perp_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		mat_to_numpy(py, &self.inner.u_perp_m)
	}

	/// Get the singular value derivatives.
	#[getter]
	pub fn s_dot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
		vec_to_numpy(py, &self.inner.s_dot)
	}

	/// Get the right orthogonal component.
	#[getter]
	pub fn v_perp_n<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		mat_to_numpy(py, &self.inner.v_perp_n)
	}

	/// Convert to full matrix representation given a base point.
	///
	/// Args:
	///     point: The base point (FixedRankPoint)
	///
	/// Returns:
	///     ndarray: The full m × n tangent matrix
	pub fn to_matrix<'py>(
		&self,
		py: Python<'py>,
		point: &PyFixedRankPoint,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let mat = self.inner.to_matrix(&point.inner);
		mat_to_numpy(py, &mat)
	}

	fn __repr__(&self) -> String {
		let (m_k, k1) = (self.inner.u_perp_m.nrows(), self.inner.u_perp_m.ncols());
		let k2 = self.inner.s_dot.len();
		let (n_k, k3) = (self.inner.v_perp_n.nrows(), self.inner.v_perp_n.ncols());
		format!(
			"FixedRankTangent(u_perp_m={}×{}, s_dot={}, v_perp_n={}×{})",
			m_k, k1, k2, n_k, k3
		)
	}
}

/// Python wrapper for the Fixed-Rank manifold.
///
/// The fixed-rank manifold consists of m×n matrices of rank k:
/// {X ∈ ℝ^{m×n} : rank(X) = k}.
///
/// Parameters
/// ----------
/// m : int
///     Number of rows
/// n : int
///     Number of columns
/// k : int
///     Rank constraint
#[pyclass(name = "FixedRank", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyFixedRank {
	/// The underlying Rust FixedRank manifold
	pub(crate) inner: FixedRank,
	/// Number of rows
	pub(crate) m: usize,
	/// Number of columns
	pub(crate) n: usize,
	/// Rank
	pub(crate) k: usize,
}

impl PyManifoldBase for PyFixedRank {
	fn manifold_name(&self) -> &'static str {
		"FixedRank"
	}

	fn ambient_dim(&self) -> usize {
		self.m * self.n
	}

	fn intrinsic_dim(&self) -> usize {
		self.k * (self.m + self.n - self.k)
	}

	fn point_type(&self) -> PointType {
		PointType::Matrix(self.m, self.n)
	}
}

#[pymethods]
impl PyFixedRank {
	/// Create a new Fixed-Rank manifold.
	///
	/// Args:
	///     m: Number of rows (must be >= 1)
	///     n: Number of columns (must be >= 1)
	///     k: Rank (must be <= min(m, n))
	///
	/// Raises:
	///     ValueError: If m < 1, n < 1, or k > min(m, n)
	#[new]
	pub fn new(m: usize, n: usize, k: usize) -> PyResult<Self> {
		if m < 1 || n < 1 {
			return Err(crate::error::value_error(
				"FixedRank manifold requires m >= 1 and n >= 1",
			));
		}
		if k > m.min(n) {
			return Err(crate::error::value_error(format!(
				"FixedRank manifold requires k <= min(m, n), got k={}, min(m, n)={}",
				k,
				m.min(n)
			)));
		}

		Ok(PyFixedRank {
			inner: FixedRank::new(m, n, k).map_err(crate::error::to_py_err)?,
			m,
			n,
			k,
		})
	}

	/// String representation of the manifold.
	fn __repr__(&self) -> String {
		format!("FixedRank(m={}, n={}, k={})", self.m, self.n, self.k)
	}

	/// Get the number of rows.
	#[getter]
	pub fn m(&self) -> usize {
		self.m
	}

	/// Get the number of columns.
	#[getter]
	pub fn n(&self) -> usize {
		self.n
	}

	/// Get the rank.
	#[getter]
	pub fn k(&self) -> usize {
		self.k
	}

	/// Check if a point is on the manifold.
	///
	/// Args:
	///     point: Matrix to check
	///     atol: Absolute tolerance (default: 1e-10)
	///
	/// Returns:
	///     True if the point has the correct rank
	#[pyo3(signature = (point, atol=1e-10))]
	pub fn contains(&self, point: PyReadonlyArray2<'_, f64>, atol: f64) -> PyResult<bool> {
		let mat = numpy_to_mat(point)?;
		if mat.nrows() != self.m || mat.ncols() != self.n {
			return Ok(false);
		}
		let fp = FixedRankPoint::<f64>::from_matrix(&mat, self.k).map_err(to_py_err)?;
		Ok(self.inner.is_point_on_manifold(&fp, atol))
	}

	pub fn project<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let mat = numpy_to_mat(point)?;
		// Truncated SVD = projection onto rank-k manifold
		let fp = FixedRankPoint::<f64>::from_matrix(&mat, self.k).map_err(to_py_err)?;
		// project_point for FixedRank just re-normalizes the SVD factors
		let mut result = fp.clone();
		Manifold::<f64>::project_point(&self.inner, &fp, &mut result);
		mat_to_numpy(py, &result.to_matrix())
	}

	pub fn project_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;
		let fp = FixedRankPoint::from_matrix(&point_mat, self.k).map_err(to_py_err)?;
		let ft = FixedRankTangent::<f64>::from_ambient(&fp, &tangent_mat);
		let mut result = ft.clone();
		self.inner
			.project_tangent(&fp, &ft, &mut result, &mut ())
			.map_err(to_py_err)?;
		mat_to_numpy(py, &result.to_matrix(&fp))
	}

	pub fn retract<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;
		let fp = FixedRankPoint::from_matrix(&point_mat, self.k).map_err(to_py_err)?;
		let ft = FixedRankTangent::<f64>::from_ambient(&fp, &tangent_mat);
		let mut result = fp.clone();
		self.inner
			.retract(&fp, &ft, &mut result, &mut ())
			.map_err(to_py_err)?;
		mat_to_numpy(py, &result.to_matrix())
	}

	pub fn inverse_retract<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray2<'_, f64>,
		y: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let x_mat = numpy_to_mat(x)?;
		let y_mat = numpy_to_mat(y)?;
		let fp_x = FixedRankPoint::from_matrix(&x_mat, self.k).map_err(to_py_err)?;
		let fp_y = FixedRankPoint::from_matrix(&y_mat, self.k).map_err(to_py_err)?;
		let mut result = FixedRankTangent::<f64>::default();
		self.inner
			.inverse_retract(&fp_x, &fp_y, &mut result, &mut ())
			.map_err(to_py_err)?;
		mat_to_numpy(py, &result.to_matrix(&fp_x))
	}

	pub fn inner(
		&self,
		point: PyReadonlyArray2<'_, f64>,
		u: PyReadonlyArray2<'_, f64>,
		v: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let point_mat = numpy_to_mat(point)?;
		let u_mat = numpy_to_mat(u)?;
		let v_mat = numpy_to_mat(v)?;
		let fp = FixedRankPoint::from_matrix(&point_mat, self.k).map_err(to_py_err)?;
		let fu = FixedRankTangent::<f64>::from_ambient(&fp, &u_mat);
		let fv = FixedRankTangent::<f64>::from_ambient(&fp, &v_mat);
		self.inner
			.inner_product(&fp, &fu, &fv, &mut ())
			.map_err(to_py_err)
	}

	pub fn norm(
		&self,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;
		let fp = FixedRankPoint::from_matrix(&point_mat, self.k).map_err(to_py_err)?;
		let ft = FixedRankTangent::<f64>::from_ambient(&fp, &tangent_mat);
		self.inner.norm(&fp, &ft, &mut ()).map_err(to_py_err)
	}

	pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let mut result = FixedRankPoint::<f64>::default();
		Manifold::<f64>::random_point(&self.inner, &mut result).map_err(to_py_err)?;
		mat_to_numpy(py, &result.to_matrix())
	}

	#[pyo3(signature = (point, scale=1.0))]
	pub fn random_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		scale: f64,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let fp = FixedRankPoint::from_matrix(&point_mat, self.k).map_err(to_py_err)?;
		let mut result = FixedRankTangent::<f64>::default();
		self.inner
			.random_tangent(&fp, &mut result)
			.map_err(to_py_err)?;
		let mut mat = result.to_matrix(&fp);
		if scale != 1.0 {
			mat.scale_mut(scale);
		}
		mat_to_numpy(py, &mat)
	}

	#[getter]
	fn dim(&self) -> usize {
		self.intrinsic_dim()
	}

	#[getter]
	fn ambient_dim(&self) -> usize {
		PyManifoldBase::ambient_dim(self)
	}

	pub fn parallel_transport<'py>(
		&self,
		py: Python<'py>,
		from_point: PyReadonlyArray2<'_, f64>,
		to_point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let from_mat = numpy_to_mat(from_point)?;
		let to_mat = numpy_to_mat(to_point)?;
		let tangent_mat = numpy_to_mat(tangent)?;
		let fp_from = FixedRankPoint::from_matrix(&from_mat, self.k).map_err(to_py_err)?;
		let fp_to = FixedRankPoint::from_matrix(&to_mat, self.k).map_err(to_py_err)?;
		let ft = FixedRankTangent::<f64>::from_ambient(&fp_from, &tangent_mat);
		let mut result = FixedRankTangent::<f64>::default();
		self.inner
			.parallel_transport(&fp_from, &fp_to, &ft, &mut result, &mut ())
			.map_err(to_py_err)?;
		mat_to_numpy(py, &result.to_matrix(&fp_to))
	}
}
