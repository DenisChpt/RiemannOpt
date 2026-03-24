//! # Oblique Manifold OB(n,p)
//!
//! The Oblique manifold OB(n,p) is the product of p unit spheres in ℝⁿ, consisting
//! of all n×p matrices with unit-norm columns. It provides a natural framework for
//! problems with column-wise normalization constraints.
//!
//! ## Mathematical Definition
//!
//! The Oblique manifold is formally defined as:
//! ```text
//! OB(n,p) = {X ∈ ℝⁿˣᵖ : diag(X^T X) = 1_p}
//!         = S^{n-1} × S^{n-1} × ... × S^{n-1}  (p times)
//! ```
//! where S^{n-1} is the unit sphere in ℝⁿ.
//!
//! Equivalently, X ∈ OB(n,p) if and only if ‖x_j‖ = 1 for all columns j = 1,...,p.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space at X ∈ OB(n,p) consists of matrices with columns orthogonal
//! to the corresponding columns of X:
//! ```text
//! T_X OB(n,p) = {V ∈ ℝⁿˣᵖ : diag(X^T V) = 0}
//!             = {V ∈ ℝⁿˣᵖ : x_j^T v_j = 0 for j = 1,...,p}
//! ```
//!
//! ### Riemannian Metric
//! The Oblique manifold inherits the Euclidean metric from ℝⁿˣᵖ:
//! ```text
//! g_X(U, V) = tr(U^T V) = ∑_{j=1}^p u_j^T v_j
//! ```
//!
//! ### Projection Operators
//! - **Manifold projection**: P_{OB}(Y) has columns y_j/‖y_j‖
//! - **Tangent projection**: P_X(V) has columns v_j - (x_j^T v_j)x_j
//!
//! ## Maps and Retractions
//!
//! ### Exponential Map
//! The exponential map acts column-wise using the sphere exponential:
//! ```text
//! exp_X(V) has columns exp_{x_j}(v_j) = cos(‖v_j‖)x_j + sin(‖v_j‖)v_j/‖v_j‖
//! ```
//!
//! ### Logarithmic Map
//! The logarithmic map acts column-wise:
//! ```text
//! log_X(Y) has columns log_{x_j}(y_j) = θ_j/sin(θ_j) · (y_j - cos(θ_j)x_j)
//! ```
//! where θ_j = arccos(x_j^T y_j).
//!
//! ### Retraction
//! The simplest retraction normalizes each column:
//! ```text
//! R_X(V) has columns (x_j + v_j)/‖x_j + v_j‖
//! ```
//!
//! ## Parallel Transport
//!
//! Parallel transport acts independently on each column using sphere transport:
//! ```text
//! Γ_{X→Y}(V) has columns Γ_{x_j→y_j}(v_j)
//! ```
//!
//! ## Geometric Properties
//!
//! - **Dimension**: dim(OB(n,p)) = p(n-1)
//! - **Curvature**: Sectional curvature 0 ≤ K ≤ 1 (from spheres)
//! - **Geodesically complete**: Yes
//! - **Simply connected**: Yes if n ≥ 3; No if n = 2
//! - **Distance**: d²(X,Y) = ∑ᵢ d_S²(xᵢ, yᵢ) where d_S is sphere distance
//!
//! ## Applications
//!
//! 1. **Independent Component Analysis (ICA)**: Unmixing signals
//! 2. **Dictionary Learning**: Sparse coding with normalized atoms
//! 3. **Factor Analysis**: Normalized loading matrices
//! 4. **Direction-of-Arrival**: Array processing in signal processing
//! 5. **Computer Vision**: Normalized feature descriptors
//! 6. **Machine Learning**: Weight normalization in neural networks
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Column-wise operations** for efficiency
//! - **Numerical stability** in normalization (handling zero columns)
//! - **Exact geodesics** through sphere formulas
//! - **Efficient retractions** via simple normalization
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Oblique;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{MatrixOps, VectorOps};
//!
//! // Create OB(3,2) - two unit vectors in ℝ³
//! let oblique = Oblique::new(3, 2)?;
//!
//! // Random point with unit-norm columns
//! let mut x = riemannopt_core::linalg::Mat::<f64>::zeros(3, 2);
//! <Oblique as Manifold<f64>>::random_point(&oblique, &mut x)?;
//! for j in 0..2 {
//!     assert!((x.column(j).norm() - 1.0).abs() < 1e-14);
//! }
//!
//! // Tangent vector
//! let mut v = riemannopt_core::linalg::Mat::<f64>::zeros(3, 2);
//! <Oblique as Manifold<f64>>::random_tangent(&oblique, &x, &mut v)?;
//!
//! // Verify orthogonality: x_j^T v_j = 0
//! for j in 0..2 {
//!     assert!(x.column(j).dot(&v.column(j)).abs() < 1e-14);
//! }
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::Debug;

use riemannopt_core::{
	error::{ManifoldError, Result},
	linalg::{self, LinAlgBackend, MatrixOps, VectorOps},
	manifold::Manifold,
	types::Scalar,
};

/// The Oblique manifold OB(n,p) = S^{n-1} × ... × S^{n-1} (p times).
///
/// This structure represents the manifold of n×p matrices with unit-norm columns,
/// equipped with the product Riemannian metric from the constituent spheres.
///
/// # Type Parameters
///
/// The manifold is generic over the scalar type T through the Manifold trait.
///
/// # Invariants
///
/// - `n ≥ 1`: Dimension of each sphere must be positive
/// - `p ≥ 1`: Number of columns must be positive
/// - All points X satisfy ‖x_j‖ = 1 for j = 1,...,p
/// - All tangent vectors V at X satisfy x_j^T v_j = 0 for j = 1,...,p
#[derive(Debug, Clone)]
pub struct Oblique {
	/// Dimension of the ambient space (rows)
	n: usize,
	/// Number of unit vectors (columns)
	p: usize,
	/// Numerical tolerance for constraint validation
	tolerance: f64,
}

impl Oblique {
	/// Creates a new Oblique manifold OB(n,p).
	///
	/// # Arguments
	///
	/// * `n` - Dimension of each constituent sphere (must be ≥ 1)
	/// * `p` - Number of spheres/columns (must be ≥ 1)
	///
	/// # Returns
	///
	/// An Oblique manifold with dimension p(n-1).
	///
	/// # Errors
	///
	/// Returns `ManifoldError::InvalidParameter` if n = 0 or p = 0.
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::Oblique;
	/// // Create OB(3,2) - two unit vectors in ℝ³
	/// let oblique = Oblique::new(3, 2)?;
	/// assert_eq!(oblique.ambient_dim(), (3, 2));
	/// assert_eq!(oblique.manifold_dim(), 4); // 2*(3-1) = 4
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	pub fn new(n: usize, p: usize) -> Result<Self> {
		if n == 0 || p == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Oblique manifold requires n ≥ 1 and p ≥ 1",
			));
		}
		Ok(Self {
			n,
			p,
			tolerance: 1e-12,
		})
	}

	/// Creates an Oblique manifold with custom numerical tolerance.
	///
	/// # Arguments
	///
	/// * `n` - Dimension of each sphere
	/// * `p` - Number of columns
	/// * `tolerance` - Numerical tolerance for constraint validation
	pub fn with_tolerance(n: usize, p: usize, tolerance: f64) -> Result<Self> {
		if n == 0 || p == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Oblique manifold requires n ≥ 1 and p ≥ 1",
			));
		}
		if tolerance <= 0.0 || tolerance >= 1.0 {
			return Err(ManifoldError::invalid_parameter(
				"Tolerance must be in (0, 1)",
			));
		}
		Ok(Self { n, p, tolerance })
	}

	/// Returns the ambient space dimensions (n, p).
	#[inline]
	pub fn ambient_dim(&self) -> (usize, usize) {
		(self.n, self.p)
	}

	/// Returns the manifold dimension p(n-1).
	#[inline]
	pub fn manifold_dim(&self) -> usize {
		self.p * (self.n - 1)
	}
}

impl Oblique {
	/// Validates that a matrix has unit-norm columns.
	///
	/// # Mathematical Check
	///
	/// Verifies that ‖x_j‖ = 1 for all columns j = 1,...,p.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If matrix dimensions don't match (n,p)
	/// - `NotOnManifold`: If any column doesn't have unit norm
	pub fn check_point<T: Scalar + Float>(&self, x: &linalg::Mat<T>) -> Result<()>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if x.nrows() != self.n || x.ncols() != self.p {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}×{}", self.n, self.p),
				format!("{}×{}", x.nrows(), x.ncols()),
			));
		}

		for j in 0..self.p {
			let col = x.column(j);
			let col_norm = col.norm();
			if <T as Float>::abs(col_norm - T::one()) > <T as Scalar>::from_f64(self.tolerance) {
				return Err(ManifoldError::invalid_point(format!(
					"Column {} has norm {} (expected 1.0, tolerance: {})",
					j, col_norm, self.tolerance
				)));
			}
		}

		Ok(())
	}

	/// Validates that a matrix is in the tangent space at X.
	///
	/// # Mathematical Check
	///
	/// Verifies that x_j^T v_j = 0 for all columns j = 1,...,p.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If dimensions don't match
	/// - `NotOnManifold`: If X is not on Oblique
	/// - `NotInTangentSpace`: If orthogonality constraints violated
	pub fn check_tangent<T: Scalar + Float>(
		&self,
		x: &linalg::Mat<T>,
		v: &linalg::Mat<T>,
	) -> Result<()>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.check_point(x)?;

		if v.nrows() != self.n || v.ncols() != self.p {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}×{}", self.n, self.p),
				format!("{}×{}", v.nrows(), v.ncols()),
			));
		}

		for j in 0..self.p {
			let x_col = x.column(j);
			let v_col = v.column(j);
			let inner = x_col.dot(&v_col);
			if <T as Float>::abs(inner) > <T as Scalar>::from_f64(self.tolerance) {
				return Err(ManifoldError::invalid_tangent(format!(
					"Column {} violates orthogonality: x_j^T v_j = {} (tolerance: {})",
					j, inner, self.tolerance
				)));
			}
		}

		Ok(())
	}

	/// Normalizes columns of a matrix to unit norm.
	///
	/// Zero columns are mapped to the first standard basis vector e_1.
	fn normalize_columns<T: Scalar + Float>(
		&self,
		matrix: &linalg::Mat<T>,
		result: &mut linalg::Mat<T>,
	) where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		result.copy_from(matrix);

		for j in 0..self.p {
			let col_norm = result.column(j).norm();
			if col_norm > <T as Scalar>::from_f64(1e-16) {
				let inv_norm = T::one() / col_norm;
				let col_slice = result.column_as_mut_slice(j);
				for val in col_slice.iter_mut() {
					*val = *val * inv_norm;
				}
			} else {
				// Handle zero columns by setting to e_1
				let col_slice = result.column_as_mut_slice(j);
				for val in col_slice.iter_mut() {
					*val = T::zero();
				}
				if self.n > 0 {
					col_slice[0] = T::one();
				}
			}
		}
	}

	/// Computes the exponential map exp_X(V).
	///
	/// # Mathematical Formula
	///
	/// Acts column-wise using the sphere exponential map:
	/// ```text
	/// (exp_X(V))_j = exp_{x_j}(v_j) = cos(‖v_j‖)x_j + sin(‖v_j‖)v_j/‖v_j‖
	/// ```
	///
	/// # Arguments
	///
	/// * `x` - Point on the Oblique manifold
	/// * `v` - Tangent vector at x
	///
	/// # Returns
	///
	/// The point exp_X(V) on the manifold.
	pub fn exp_map<T: Scalar + Float>(
		&self,
		x: &linalg::Mat<T>,
		v: &linalg::Mat<T>,
	) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.check_tangent(x, v)?;

		let mut result: linalg::Mat<T> = MatrixOps::zeros(self.n, self.p);

		for j in 0..self.p {
			let x_col = x.column(j);
			let v_col = v.column(j);
			let v_norm = v_col.norm();

			let col_slice = result.column_as_mut_slice(j);

			if v_norm < <T as Scalar>::from_f64(1e-16) {
				// Copy x_col
				for i in 0..self.n {
					col_slice[i] = x_col.get(i);
				}
			} else {
				let cos_norm = <T as Float>::cos(v_norm);
				let sin_norm = <T as Float>::sin(v_norm);

				// exp_{x_j}(v_j) = cos(‖v_j‖)x_j + sin(‖v_j‖)v_j/‖v_j‖
				for i in 0..self.n {
					col_slice[i] = cos_norm * x_col.get(i) + (sin_norm / v_norm) * v_col.get(i);
				}
			}
		}

		Ok(result)
	}

	/// Computes the logarithmic map log_X(Y).
	///
	/// # Mathematical Formula
	///
	/// Acts column-wise using the sphere logarithmic map:
	/// ```text
	/// (log_X(Y))_j = log_{x_j}(y_j) = θ_j/sin(θ_j) · (y_j - cos(θ_j)x_j)
	/// ```
	/// where θ_j = arccos(x_j^T y_j).
	///
	/// # Arguments
	///
	/// * `x` - Point on the Oblique manifold
	/// * `y` - Another point on the Oblique manifold
	///
	/// # Returns
	///
	/// The tangent vector log_X(Y) ∈ T_X OB(n,p).
	pub fn log_map<T: Scalar + Float>(
		&self,
		x: &linalg::Mat<T>,
		y: &linalg::Mat<T>,
	) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.check_point(x)?;
		self.check_point(y)?;

		let mut result: linalg::Mat<T> = MatrixOps::zeros(self.n, self.p);

		for j in 0..self.p {
			let x_col = x.column(j);
			let y_col = y.column(j);

			let inner = x_col.dot(&y_col);
			let clamped = <T as Float>::max(<T as Float>::min(inner, T::one()), -T::one());

			// Check if points are the same
			if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-14) {
				continue; // Result column remains zero
			}

			// Check if points are antipodal
			if <T as Float>::abs(clamped + T::one()) < <T as Scalar>::from_f64(1e-14) {
				return Err(ManifoldError::numerical_error(format!(
					"Cannot compute logarithm: columns {} are antipodal",
					j
				)));
			}

			let theta = <T as Float>::acos(clamped);
			let sin_theta = <T as Float>::sin(theta);

			// log_{x_j}(y_j) = θ/sin(θ) · (y_j - cos(θ)x_j)
			let scale = theta / sin_theta;
			let col_slice = result.column_as_mut_slice(j);
			for i in 0..self.n {
				col_slice[i] = scale * (y_col.get(i) - clamped * x_col.get(i));
			}
		}

		Ok(result)
	}

	/// Computes the geodesic distance between two points.
	///
	/// # Mathematical Formula
	///
	/// The distance is the L² norm of column-wise sphere distances:
	/// ```text
	/// d²(X, Y) = ∑_{j=1}^p d_{S^{n-1}}²(x_j, y_j) = ∑_{j=1}^p arccos²(x_j^T y_j)
	/// ```
	///
	/// # Arguments
	///
	/// * `x` - First point on the Oblique manifold
	/// * `y` - Second point on the Oblique manifold
	///
	/// # Returns
	///
	/// The geodesic distance d(X, Y) ≥ 0.
	pub fn geodesic_distance<T: Scalar + Float>(
		&self,
		x: &linalg::Mat<T>,
		y: &linalg::Mat<T>,
	) -> Result<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.check_point(x)?;
		self.check_point(y)?;

		let mut dist_squared = T::zero();

		for j in 0..self.p {
			let x_col = x.column(j);
			let y_col = y.column(j);
			let inner = x_col.dot(&y_col);
			let clamped = <T as Float>::max(<T as Float>::min(inner, T::one()), -T::one());
			let angle = <T as Float>::acos(clamped);
			dist_squared = dist_squared + angle * angle;
		}

		Ok(<T as Float>::sqrt(dist_squared))
	}

	/// Parallel transports a tangent vector along geodesics.
	///
	/// # Mathematical Formula
	///
	/// Transport acts independently on each column using sphere transport.
	///
	/// # Arguments
	///
	/// * `x` - Starting point
	/// * `y` - Ending point
	/// * `v` - Tangent vector at x
	///
	/// # Returns
	///
	/// The parallel transported vector at y.
	pub fn parallel_transport_geodesic<T: Scalar + Float>(
		&self,
		x: &linalg::Mat<T>,
		y: &linalg::Mat<T>,
		v: &linalg::Mat<T>,
	) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.check_tangent(x, v)?;
		self.check_point(y)?;

		let mut result: linalg::Mat<T> = MatrixOps::zeros(self.n, self.p);

		for j in 0..self.p {
			let x_col = x.column(j);
			let y_col = y.column(j);
			let v_col = v.column(j);

			let inner = x_col.dot(&y_col);
			let clamped = <T as Float>::max(<T as Float>::min(inner, T::one()), -T::one());

			let col_slice = result.column_as_mut_slice(j);

			// Same point - no transport needed
			if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-14) {
				for i in 0..self.n {
					col_slice[i] = v_col.get(i);
				}
				continue;
			}

			// Antipodal points - transport is ambiguous but we keep the vector
			if <T as Float>::abs(clamped + T::one()) < <T as Scalar>::from_f64(1e-14) {
				for i in 0..self.n {
					col_slice[i] = v_col.get(i);
				}
				continue;
			}

			// General transport formula on sphere
			let factor = (x_col.dot(&v_col) + y_col.dot(&v_col)) / (T::one() + inner);
			for i in 0..self.n {
				col_slice[i] = v_col.get(i) - factor * (x_col.get(i) + y_col.get(i));
			}
		}

		Ok(result)
	}
}

impl<T> Manifold<T> for Oblique
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Mat<T>;
	type TangentVector = linalg::Mat<T>;

	fn name(&self) -> &str {
		"Oblique"
	}

	fn dimension(&self) -> usize {
		self.p * (self.n - 1)
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.p {
			return false;
		}

		// Check each column has unit norm
		for j in 0..self.p {
			let col = point.column(j);
			let col_norm = col.norm();
			if <T as Float>::abs(col_norm - T::one()) > tol {
				return false;
			}
		}

		true
	}

	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		if !self.is_point_on_manifold(point, tol) {
			return false;
		}

		if vector.nrows() != self.n || vector.ncols() != self.p {
			return false;
		}

		// For each column: x_j^T v_j = 0
		for j in 0..self.p {
			let x_col = point.column(j);
			let v_col = vector.column(j);
			let inner = x_col.dot(&v_col);

			if <T as Float>::abs(inner) > tol {
				return false;
			}
		}

		true
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		self.normalize_columns(point, result);
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		result.copy_from(vector);

		// For each column: v_j - (x_j^T v_j) x_j
		for j in 0..self.p {
			let x_col = point.column(j);
			let v_col = vector.column(j);
			let inner = x_col.dot(&v_col);

			let col_slice = result.column_as_mut_slice(j);
			for i in 0..self.n {
				col_slice[i] = col_slice[i] - inner * x_col.get(i);
			}
		}

		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		// Sum of column-wise inner products
		let mut total = T::zero();
		for j in 0..self.p {
			let u_col = u.column(j);
			let v_col = v.column(j);
			total = total + u_col.dot(&v_col);
		}
		Ok(total)
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		// Normalize each column of (X + V)
		let sum = point.add(tangent);
		result.copy_from(&sum);

		for j in 0..self.p {
			let col_norm = result.column(j).norm();
			if col_norm > T::zero() {
				let inv_norm = T::one() / col_norm;
				let col_slice = result.column_as_mut_slice(j);
				for val in col_slice.iter_mut() {
					*val = *val * inv_norm;
				}
			}
		}

		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		result.fill(T::zero());

		// For each column, compute logarithmic map on sphere
		for j in 0..self.p {
			let x_col = point.column(j);
			let y_col = other.column(j);

			let inner = x_col.dot(&y_col);
			let clamped = <T as Float>::min(<T as Float>::max(inner, -T::one()), T::one());

			if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-10) {
				// Points are identical
				continue;
			}

			let theta = <T as Float>::acos(clamped);
			let sin_theta = <T as Float>::sin(theta);

			if sin_theta > <T as Scalar>::from_f64(1e-10) {
				// v = theta / sin(theta) * (y - cos(theta) * x)
				let scale = theta / sin_theta;
				let col_slice = result.column_as_mut_slice(j);
				for i in 0..self.n {
					col_slice[i] = scale * (y_col.get(i) - clamped * x_col.get(i));
				}
			}
		}

		// Ensure result is in tangent space
		let result_clone = result.clone();
		self.project_tangent(point, &result_clone, result)?;

		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Project to tangent space
		self.project_tangent(point, euclidean_grad, result)
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Ensure result has correct dimensions
		if result.nrows() != self.n || result.ncols() != self.p {
			*result = MatrixOps::zeros(self.n, self.p);
		}

		// Generate random columns and normalize
		for j in 0..self.p {
			let col_slice = result.column_as_mut_slice(j);
			for i in 0..self.n {
				col_slice[i] = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}

			// Compute norm
			let mut norm_sq = T::zero();
			for i in 0..self.n {
				norm_sq = norm_sq + col_slice[i] * col_slice[i];
			}
			let col_norm = <T as Float>::sqrt(norm_sq);

			if col_norm > T::zero() {
				let inv_norm = T::one() / col_norm;
				for val in col_slice.iter_mut() {
					*val = *val * inv_norm;
				}
			}
		}

		Ok(())
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Ensure result has correct dimensions
		if result.nrows() != self.n || result.ncols() != self.p {
			*result = MatrixOps::zeros(self.n, self.p);
		}

		// Generate random matrix
		for j in 0..self.p {
			let col_slice = result.column_as_mut_slice(j);
			for i in 0..self.n {
				col_slice[i] = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// Project to tangent space
		let result_clone = result.clone();
		self.project_tangent(point, &result_clone, result)?;

		Ok(())
	}

	fn parallel_transport(
		&self,
		from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		result.fill(T::zero());

		// Parallel transport each column independently
		for j in 0..self.p {
			let x_col = from.column(j);
			let y_col = to.column(j);
			let v_col = vector.column(j);

			let inner = x_col.dot(&y_col);
			let clamped = <T as Float>::min(<T as Float>::max(inner, -T::one()), T::one());

			let col_slice = result.column_as_mut_slice(j);

			if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-10) {
				// Same point, no transport needed
				for i in 0..self.n {
					col_slice[i] = v_col.get(i);
				}
				continue;
			}

			if <T as Float>::abs(clamped + T::one()) < <T as Scalar>::from_f64(1e-10) {
				// Antipodal points
				for i in 0..self.n {
					col_slice[i] = v_col.get(i);
				}
				continue;
			}

			// Compute transported vector
			let theta = <T as Float>::acos(clamped);
			let sin_theta = <T as Float>::sin(theta);

			// Tangent direction at x towards y
			let mut xi = linalg::Vec::<T>::zeros(self.n);
			for i in 0..self.n {
				*xi.get_mut(i) = y_col.get(i) - clamped * x_col.get(i);
			}
			xi.scale_mut(T::one() / sin_theta);

			// Transport formula: τ(v) = v - sin(θ)⟨v,ξ⟩·x + (cos(θ)-1)⟨v,ξ⟩·ξ
			let v_xi_inner = v_col.dot(&xi);
			let sin_coeff = T::zero() - sin_theta * v_xi_inner;
			let cos_coeff = (<T as Float>::cos(theta) - T::one()) * v_xi_inner;
			for i in 0..self.n {
				col_slice[i] = v_col.get(i) + sin_coeff * x_col.get(i) + cos_coeff * xi.get(i);
			}
		}

		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		let mut dist_squared = T::zero();

		// Sum of squared distances on each sphere
		for j in 0..self.p {
			let x_col = x.column(j);
			let y_col = y.column(j);

			let inner = x_col.dot(&y_col);
			let clamped = <T as Float>::min(<T as Float>::max(inner, -T::one()), T::one());
			let angle = <T as Float>::acos(clamped);

			dist_squared = dist_squared + angle * angle;
		}

		Ok(<T as Float>::sqrt(dist_squared))
	}

	fn has_exact_exp_log(&self) -> bool {
		true // Oblique has exact exp/log on each sphere
	}

	fn is_flat(&self) -> bool {
		false // Product of spheres with positive curvature
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For Oblique manifold, tangent vectors have orthogonal columns to the point columns
		// Scaling preserves this orthogonality
		result.copy_from(tangent);
		result.scale_mut(scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
		// Temporary buffer for projection if needed
		temp: &mut Self::TangentVector,
	) -> Result<()> {
		// Add the tangent vectors
		temp.copy_from(v1);
		temp.add_assign(v2);

		// The sum should already satisfy the tangent space constraint if v1 and v2 do,
		// but we project for numerical stability
		self.project_tangent(point, temp, result)?;

		Ok(())
	}
}

// NOTE: MatrixManifold<T> impl commented out because the trait still requires
// `Point = DMatrix<T>` but we now use `linalg::Mat<T>`. Will be re-enabled
// once riemannopt-core's MatrixManifold trait is migrated to the linalg layer.
//
// impl<T: Scalar> MatrixManifold<T> for Oblique {
// 	fn matrix_dims(&self) -> (usize, usize) {
// 		(self.n, self.p)
// 	}
// }
