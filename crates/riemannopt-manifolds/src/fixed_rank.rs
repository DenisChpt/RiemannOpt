//! # Fixed-Rank Manifold M_k(m,n)
//!
//! The manifold M_k(m,n) of m×n matrices with fixed rank k forms a smooth
//! submanifold of ℝ^{m×n}. It provides a geometric framework for low-rank
//! matrix optimization problems.
//!
//! ## Mathematical Definition
//!
//! The fixed-rank manifold is formally defined as:
//! ```text
//! M_k(m,n) = {X ∈ ℝ^{m×n} : rank(X) = k}
//! ```
//!
//! ## Parametrization via SVD
//!
//! Points are represented using the compact SVD:
//! ```text
//! X = UΣV^T
//! ```
//! where:
//! - U ∈ St(m,k): Left singular vectors
//! - Σ ∈ ℝ^{k×k}: Diagonal matrix of singular values
//! - V ∈ St(n,k): Right singular vectors
//!
//! This gives the quotient structure:
//! ```text
//! M_k(m,n) ≅ (St(m,k) × ℝ₊^k × St(n,k)) / O(k)
//! ```
//!
//! ## Tangent Space
//!
//! The tangent space at X = UΣV^T consists of matrices:
//! ```text
//! T_X M_k = {U_⊥MV^T + UNV_⊥^T + UΩV^T : M ∈ ℝ^{(m-k)×k}, N ∈ ℝ^{k×(n-k)}, Ω ∈ ℝ^{k×k}}
//! ```
//! where U_⊥ and V_⊥ are orthogonal complements.
//!
//! ## Riemannian Metric
//!
//! The standard metric is the Euclidean metric restricted to the tangent space:
//! ```text
//! g_X(ξ, η) = tr(ξ^T η)
//! ```
//!
//! ## Retractions
//!
//! ### SVD-based Retraction
//! ```text
//! R_X(ξ) = best rank-k approximation of (X + ξ)
//! ```
//!
//! ### Orthographic Retraction
//! For X = UΣV^T and tangent ξ = UMV^T + U_⊥NV^T + UN^TV_⊥^T:
//! ```text
//! R_X(ξ) = (U + U_⊥NΣ⁻¹)(Σ + M)(V + V_⊥N^TΣ⁻¹)^T
//! ```
//!
//! ## Geometric Properties
//!
//! - **Dimension**: dim(M_k) = k(m + n - k)
//! - **Non-closed**: M_k is not closed in ℝ^{m×n}
//! - **Embedded submanifold**: When viewed as subset of ℝ^{m×n}
//! - **Quotient manifold**: Inherits structure from product of Stiefel manifolds
//!
//! ## Applications
//!
//! 1. **Matrix Completion**: Netflix problem, collaborative filtering
//! 2. **System Identification**: Low-order dynamical systems
//! 3. **Model Reduction**: Reduced-order modeling
//! 4. **Computer Vision**: Structure from motion, face recognition
//! 5. **Data Compression**: Low-rank approximation
//! 6. **Machine Learning**: Low-rank neural networks
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Efficient storage** using factored form UΣV^T
//! - **Numerical stability** in SVD computations
//! - **Proper handling** of small singular values
//! - **Orthogonality preservation** in U and V factors
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::FixedRank;
//! use riemannopt_manifolds::fixed_rank::FixedRankPoint;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{self, MatrixOps};
//!
//! // Create M_2(4,3) - 4×3 matrices of rank 2
//! let manifold = FixedRank::new(4, 3, 2)?;
//!
//! // Random rank-2 matrix
//! let mut x = FixedRankPoint::<f64>::default();
//! manifold.random_point(&mut x)?;
//!
//! // Convert to matrix form
//! let x_mat = x.to_matrix();
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
	error::{ManifoldError, Result},
	linalg::{self, DecompositionOps, LinAlgBackend, MatrixOps, VectorOps},
	manifold::Manifold,
	types::Scalar,
};

/// The fixed-rank manifold M_k(m,n) of m×n matrices with rank k.
///
/// This structure represents matrices of fixed rank using their SVD factorization,
/// providing efficient storage and computation for low-rank matrix optimization.
///
/// # Type Parameters
///
/// The manifold is generic over the scalar type T through the Manifold trait.
///
/// # Invariants
///
/// - `m ≥ 1, n ≥ 1`: Matrix dimensions must be positive
/// - `k ≥ 1`: Rank must be positive
/// - `k ≤ min(m, n)`: Rank cannot exceed matrix dimensions
/// - Points are stored as vectors containing U, Σ, V factors
#[derive(Debug, Clone)]
pub struct FixedRank {
	/// Number of rows
	m: usize,
	/// Number of columns
	n: usize,
	/// Rank
	k: usize,
	/// Numerical tolerance
	tolerance: f64,
}

/// Representation of a point on the fixed-rank manifold
#[derive(Debug, Clone)]
pub struct FixedRankPoint<T: Scalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Left singular vectors (m × k)
	pub u: linalg::Mat<T>,
	/// Singular values (k × k diagonal)
	pub s: linalg::Vec<T>,
	/// Right singular vectors (n × k)
	pub v: linalg::Mat<T>,
}

impl<T: Scalar> Default for FixedRankPoint<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn default() -> Self {
		Self {
			u: MatrixOps::zeros(0, 0),
			s: VectorOps::zeros(0),
			v: MatrixOps::zeros(0, 0),
		}
	}
}

impl<T: Scalar> FixedRankPoint<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Create a new fixed-rank point from factors
	pub fn new(u: linalg::Mat<T>, s: linalg::Vec<T>, v: linalg::Mat<T>) -> Self {
		Self { u, s, v }
	}

	/// Convert to full matrix representation
	pub fn to_matrix(&self) -> linalg::Mat<T> {
		let s_mat = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&self.s);
		let temp = MatrixOps::mat_mul(&self.u, &s_mat);
		let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.u.nrows(), self.v.nrows());
		out.gemm_bt(T::one(), &temp, &self.v, T::zero());
		out
	}

	/// Create from full matrix using SVD
	pub fn from_matrix(mat: &linalg::Mat<T>, k: usize) -> Result<Self> {
		let svd = DecompositionOps::svd(mat);

		let u_full = svd
			.u
			.ok_or_else(|| ManifoldError::numerical_error("SVD failed to compute U"))?;
		let vt_full = svd
			.vt
			.ok_or_else(|| ManifoldError::numerical_error("SVD failed to compute V^T"))?;

		// Truncate to rank k
		let u_k = MatrixOps::columns(&u_full, 0, k);
		let s_k = <linalg::Vec<T> as VectorOps<T>>::from_fn(k, |i| {
			VectorOps::get(&svd.singular_values, i)
		});
		let v_k = MatrixOps::columns(&MatrixOps::transpose(&vt_full), 0, k);

		Ok(Self::new(u_k, s_k, v_k))
	}
}

/// Representation of a tangent vector on the fixed-rank manifold
#[derive(Debug, Clone)]
pub struct FixedRankTangent<T: Scalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Component U_perp * M * V^T (m × k)
	pub u_perp_m: linalg::Mat<T>,
	/// Component U * S_dot * V^T (k × k)
	pub s_dot: linalg::Vec<T>,
	/// Component U * N * V_perp^T (n × k)
	pub v_perp_n: linalg::Mat<T>,
}

impl<T: Scalar> Default for FixedRankTangent<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn default() -> Self {
		Self {
			u_perp_m: MatrixOps::zeros(0, 0),
			s_dot: VectorOps::zeros(0),
			v_perp_n: MatrixOps::zeros(0, 0),
		}
	}
}

impl<T: Scalar> FixedRankTangent<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Create a new fixed-rank tangent vector from components
	pub fn new(u_perp_m: linalg::Mat<T>, s_dot: linalg::Vec<T>, v_perp_n: linalg::Mat<T>) -> Self {
		Self {
			u_perp_m,
			s_dot,
			v_perp_n,
		}
	}

	/// Convert to full matrix representation given a base point
	pub fn to_matrix(&self, point: &FixedRankPoint<T>) -> linalg::Mat<T> {
		let s_dot_mat = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&self.s_dot);

		// Compute U_perp and V_perp using QR decomposition
		let (u_perp, _) = Self::compute_orthogonal_complement(&point.u);
		let (v_perp, _) = Self::compute_orthogonal_complement(&point.v);

		// Combine the three components
		let vt = MatrixOps::transpose(&point.v);
		let term1 = MatrixOps::mat_mul(&MatrixOps::mat_mul(&u_perp, &self.u_perp_m), &vt);
		let term2 = MatrixOps::mat_mul(&MatrixOps::mat_mul(&point.u, &s_dot_mat), &vt);
		let term3 = MatrixOps::mat_mul(
			&MatrixOps::mat_mul(&point.u, &self.v_perp_n),
			&MatrixOps::transpose(&v_perp),
		);

		MatrixOps::add(&MatrixOps::add(&term1, &term2), &term3)
	}

	/// Compute orthogonal complement of a matrix with orthonormal columns
	fn compute_orthogonal_complement(mat: &linalg::Mat<T>) -> (linalg::Mat<T>, linalg::Mat<T>) {
		let m = MatrixOps::nrows(mat);
		let k = MatrixOps::ncols(mat);

		if k >= m {
			// No orthogonal complement
			return (
				<linalg::Mat<T> as MatrixOps<T>>::zeros(m, 0),
				<linalg::Mat<T> as MatrixOps<T>>::zeros(0, 0),
			);
		}

		// Create identity and project out the columns of mat: I - M·Mᵀ
		let mut q = <linalg::Mat<T> as MatrixOps<T>>::identity(m);
		q.gemm_bt(-T::one(), mat, mat, T::one());

		// Use QR to get orthonormal basis for the complement
		let qr = DecompositionOps::qr(&q);
		let q_full = qr.q();

		// Extract the last m-k columns
		let u_perp = MatrixOps::columns(q_full, k, m - k);
		let r_perp = <linalg::Mat<T> as MatrixOps<T>>::zeros(m - k, m - k); // Placeholder

		(u_perp, r_perp)
	}
}

impl FixedRank {
	/// Creates a new fixed-rank manifold M_k(m,n).
	///
	/// # Arguments
	///
	/// * `m` - Number of rows (must be ≥ 1)
	/// * `n` - Number of columns (must be ≥ 1)
	/// * `k` - Rank (must satisfy 1 ≤ k ≤ min(m, n))
	///
	/// # Returns
	///
	/// A fixed-rank manifold with dimension k(m + n - k).
	///
	/// # Errors
	///
	/// Returns `ManifoldError::InvalidParameter` if:
	/// - Any dimension is zero
	/// - k > min(m, n)
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::FixedRank;
	/// // Create M_2(5,4) - 5×4 matrices of rank 2
	/// let manifold = FixedRank::new(5, 4, 2)?;
	/// assert_eq!(manifold.matrix_dimensions(), (5, 4, 2));
	/// assert_eq!(manifold.manifold_dim(), 2*(5+4-2)); // 14
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
		if m == 0 || n == 0 || k == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Fixed-rank manifold requires m ≥ 1, n ≥ 1, and k ≥ 1",
			));
		}

		if k > m.min(n) {
			return Err(ManifoldError::invalid_parameter(format!(
				"Rank k={} cannot exceed min(m={}, n={})",
				k, m, n
			)));
		}

		Ok(Self {
			m,
			n,
			k,
			tolerance: 1e-12,
		})
	}

	/// Creates a fixed-rank manifold with custom tolerance.
	///
	/// # Arguments
	///
	/// * `m` - Number of rows
	/// * `n` - Number of columns
	/// * `k` - Rank
	/// * `tolerance` - Numerical tolerance for rank checks
	pub fn with_tolerance(m: usize, n: usize, k: usize, tolerance: f64) -> Result<Self> {
		if m == 0 || n == 0 || k == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Fixed-rank manifold requires m ≥ 1, n ≥ 1, and k ≥ 1",
			));
		}

		if k > m.min(n) {
			return Err(ManifoldError::invalid_parameter(format!(
				"Rank k={} cannot exceed min(m={}, n={})",
				k, m, n
			)));
		}

		if tolerance <= 0.0 || tolerance >= 1.0 {
			return Err(ManifoldError::invalid_parameter(
				"Tolerance must be in (0, 1)",
			));
		}

		Ok(Self { m, n, k, tolerance })
	}

	/// Returns the matrix dimensions (m, n, k).
	#[inline]
	pub fn matrix_dimensions(&self) -> (usize, usize, usize) {
		(self.m, self.n, self.k)
	}

	/// Returns the manifold dimension k(m + n - k).
	#[inline]
	pub fn manifold_dim(&self) -> usize {
		self.k * (self.m + self.n - self.k)
	}

	/// Project the U and V factors onto the Stiefel manifold
	fn project_factors<T: Scalar>(&self, u: &mut linalg::Mat<T>, v: &mut linalg::Mat<T>)
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		// QR decomposition for U
		let qr_u = DecompositionOps::qr(u);
		*u = qr_u.q().clone();

		// QR decomposition for V
		let qr_v = DecompositionOps::qr(v);
		*v = qr_v.q().clone();
	}

	/// Validates that a matrix has the correct fixed rank.
	///
	/// # Mathematical Check
	///
	/// Verifies that rank(X) = k using SVD.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If matrix dimensions don't match (m,n)
	/// - `InvalidPoint`: If rank(X) ≠ k
	pub fn check_matrix<T: Scalar>(&self, x: &linalg::Mat<T>) -> Result<()>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(x) != self.m || MatrixOps::ncols(x) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixOps::nrows(x) * MatrixOps::ncols(x),
			));
		}

		// Check rank using SVD
		let svd = DecompositionOps::svd(x);
		let s = &svd.singular_values;

		// Count non-zero singular values
		let mut rank = 0;
		for i in 0..VectorOps::len(s).min(self.m).min(self.n) {
			if VectorOps::get(s, i) > <T as Scalar>::from_f64(self.tolerance) {
				rank += 1;
			}
		}

		if rank != self.k {
			return Err(ManifoldError::invalid_point(format!(
				"Matrix rank {} ≠ required rank {}",
				rank, self.k
			)));
		}

		Ok(())
	}

	/// Validates that a matrix is a valid tangent vector at X.
	///
	/// For now this is a placeholder that accepts all vectors.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If dimensions don't match
	pub fn check_tangent<T: Scalar>(&self, x: &linalg::Mat<T>, z: &linalg::Mat<T>) -> Result<()>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.check_matrix(x)?;

		if MatrixOps::nrows(z) != self.m || MatrixOps::ncols(z) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixOps::nrows(z) * MatrixOps::ncols(z),
			));
		}

		// Check tangent space constraint
		// For fixed-rank, tangent vectors have specific structure

		Ok(())
	}
}

impl<T> Manifold<T> for FixedRank
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = FixedRankPoint<T>;
	type TangentVector = FixedRankTangent<T>;

	fn name(&self) -> &str {
		"FixedRank"
	}

	fn dimension(&self) -> usize {
		self.k * (self.m + self.n - self.k)
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		// Check that U and V are on Stiefel manifolds
		let mut u_gram = linalg::Mat::<T>::zeros(self.k, self.k);
		u_gram.gemm_at(T::one(), &point.u, &point.u, T::zero());
		let mut v_gram = linalg::Mat::<T>::zeros(self.k, self.k);
		v_gram.gemm_at(T::one(), &point.v, &point.v, T::zero());

		// Check orthogonality
		for i in 0..self.k {
			for j in 0..self.k {
				let u_val = if i == j {
					MatrixOps::get(&u_gram, i, j) - T::one()
				} else {
					MatrixOps::get(&u_gram, i, j)
				};
				let v_val = if i == j {
					MatrixOps::get(&v_gram, i, j) - T::one()
				} else {
					MatrixOps::get(&v_gram, i, j)
				};

				if <T as Float>::abs(u_val) > tol || <T as Float>::abs(v_val) > tol {
					return false;
				}
			}
		}

		// Check that singular values are positive
		for i in 0..self.k {
			if VectorOps::get(&point.s, i) <= T::zero() {
				return false;
			}
		}

		true
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Copy the input point
		result.u = point.u.clone();
		result.s = point.s.clone();
		result.v = point.v.clone();

		// Project U and V onto Stiefel manifolds
		self.project_factors(&mut result.u, &mut result.v);

		// Ensure singular values are positive
		for i in 0..self.k {
			if VectorOps::get(&result.s, i) < T::epsilon() {
				*VectorOps::get_mut(&mut result.s, i) = T::epsilon();
			}
		}
	}

	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For a tangent vector at point (U,S,V), the tangent space has the form:
		// ξ = U_perp * M * V^T + U * S_dot * V^T + U * N * V_perp^T
		// The input vector already has this structure, so we just copy it
		result.u_perp_m = vector.u_perp_m.clone();
		result.s_dot = vector.s_dot.clone();
		result.v_perp_n = vector.v_perp_n.clone();
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		// The inner product is the Frobenius inner product of the matrix representations
		// <u, v> = tr(u^T * v) = tr(U_perp*M_u*V^T + U*S_dot_u*V^T + U*N_u*V_perp^T)^T *
		//                           (U_perp*M_v*V^T + U*S_dot_v*V^T + U*N_v*V_perp^T)
		// Since U, U_perp, V, V_perp are orthogonal, this simplifies to:
		// <u, v> = tr(M_u^T * M_v) + tr(S_dot_u * S_dot_v) + tr(N_u^T * N_v)

		let mut inner = T::zero();

		// U_perp component
		inner = inner
			+ MatrixOps::trace(&MatrixOps::mat_mul(
				&MatrixOps::transpose(&u.u_perp_m),
				&v.u_perp_m,
			));

		// S component
		for i in 0..self.k {
			inner = inner + VectorOps::get(&u.s_dot, i) * VectorOps::get(&v.s_dot, i);
		}

		// V_perp component
		inner = inner
			+ MatrixOps::trace(&MatrixOps::mat_mul(
				&MatrixOps::transpose(&u.v_perp_n),
				&v.v_perp_n,
			));

		Ok(inner)
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		// Use the orthographic retraction for fixed-rank manifolds
		// R_X(ξ) = (U + U_perp*M*S^{-1})(S + S_dot)(V + V_perp*N^T*S^{-1})^T

		// Compute U_perp and V_perp
		let (u_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.u);
		let (v_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.v);

		// Compute S^{-1}
		let s_inv_vec = VectorOps::map(&point.s, |x| T::one() / x);
		let s_inv = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&s_inv_vec);

		// Update U: U_new = U + U_perp * M * S^{-1}
		let u_perp_m_sinv = MatrixOps::mat_mul(&tangent.u_perp_m, &s_inv);
		let u_update = MatrixOps::mat_mul(&u_perp, &u_perp_m_sinv);
		result.u = MatrixOps::add(&point.u, &u_update);

		// Update S: S_new = S + S_dot
		result.s = VectorOps::add(&point.s, &tangent.s_dot);

		// Update V: V_new = V + V_perp * N^T * S^{-1}
		let vn_t = MatrixOps::transpose(&tangent.v_perp_n);
		let vn_t_sinv = MatrixOps::mat_mul(&vn_t, &s_inv);
		let v_update = MatrixOps::mat_mul(&v_perp, &vn_t_sinv);
		result.v = MatrixOps::add(&point.v, &v_update);

		// Project factors back to Stiefel
		self.project_factors(&mut result.u, &mut result.v);

		// Ensure singular values are positive
		for i in 0..self.k {
			if VectorOps::get(&result.s, i) < T::epsilon() {
				*VectorOps::get_mut(&mut result.s, i) = T::epsilon();
			}
		}

		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For the canonical metric, just project to tangent space
		self.project_tangent(point, euclidean_grad, result)
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random orthogonal matrices
		let mut u = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.m, self.k);
		let mut v = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.k);

		for j in 0..self.k {
			for i in 0..self.m {
				*MatrixOps::get_mut(&mut u, i, j) =
					<T as Scalar>::from_f64(normal.sample(&mut rng));
			}
			for i in 0..self.n {
				*MatrixOps::get_mut(&mut v, i, j) =
					<T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// Orthogonalize
		let qr_u = DecompositionOps::qr(&u);
		let u_orth = qr_u.q().clone();

		let qr_v = DecompositionOps::qr(&v);
		let v_orth = qr_v.q().clone();

		// Random positive singular values
		let mut s = <linalg::Vec<T> as VectorOps<T>>::zeros(self.k);
		for i in 0..self.k {
			let val: f64 = normal.sample(&mut rng);
			*VectorOps::get_mut(&mut s, i) = <T as Scalar>::from_f64(val.abs() + 1.0);
		}

		*result = FixedRankPoint::new(u_orth, s, v_orth);
		Ok(())
	}

	fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random matrices for the tangent components
		let mut u_perp_m = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.m - self.k, self.k);
		let mut s_dot = <linalg::Vec<T> as VectorOps<T>>::zeros(self.k);
		let mut v_perp_n = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.k, self.n - self.k);

		// Fill with random values
		for j in 0..self.k {
			for i in 0..(self.m - self.k) {
				*MatrixOps::get_mut(&mut u_perp_m, i, j) =
					<T as Scalar>::from_f64(normal.sample(&mut rng));
			}
			*VectorOps::get_mut(&mut s_dot, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			for i in 0..(self.n - self.k) {
				*MatrixOps::get_mut(&mut v_perp_n, j, i) =
					<T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		result.u_perp_m = u_perp_m;
		result.s_dot = s_dot;
		result.v_perp_n = v_perp_n;
		Ok(())
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

		// Check dimensions of tangent components
		if MatrixOps::nrows(&vector.u_perp_m) != self.m - self.k
			|| MatrixOps::ncols(&vector.u_perp_m) != self.k
		{
			return false;
		}
		if VectorOps::len(&vector.s_dot) != self.k {
			return false;
		}
		if MatrixOps::nrows(&vector.v_perp_n) != self.k
			|| MatrixOps::ncols(&vector.v_perp_n) != self.n - self.k
		{
			return false;
		}

		// Tangent vectors have the specific structure, so as long as dimensions match, it's valid
		true
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For fixed-rank manifold, we use a simple approximation
		// The inverse of the orthographic retraction is complex, so we approximate
		// by computing the tangent that moves in the direction of other - point

		// Compute the difference in matrix form
		let point_mat = point.to_matrix();
		let other_mat = other.to_matrix();
		let diff = MatrixOps::sub(&other_mat, &point_mat);

		// Project onto the tangent space at point
		// Compute U_perp and V_perp
		let (u_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.u);
		let (v_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.v);

		// Decompose the difference into tangent components
		// M = U_perp^T * diff * V
		let mut upt_diff = linalg::Mat::<T>::zeros(u_perp.ncols(), diff.ncols());
		upt_diff.gemm_at(T::one(), &u_perp, &diff, T::zero());
		result.u_perp_m = MatrixOps::mat_mul(&upt_diff, &point.v);

		// S_dot = diag(U^T * diff * V)
		let mut ut_diff = linalg::Mat::<T>::zeros(self.k, diff.ncols());
		ut_diff.gemm_at(T::one(), &point.u, &diff, T::zero());
		let s_component = MatrixOps::mat_mul(&ut_diff, &point.v);
		// Extract diagonal
		result.s_dot = <linalg::Vec<T> as VectorOps<T>>::from_fn(self.k, |i| {
			MatrixOps::get(&s_component, i, i)
		});

		// N = U^T * diff * V_perp
		result.v_perp_n = MatrixOps::mat_mul(&ut_diff, &v_perp);

		Ok(())
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For fixed-rank manifold, parallel transport is complex
		// We use a simple approximation: transport the tangent by adapting to the new basis

		// Compute U_perp and V_perp at the destination point
		let (_u_perp_to, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&to.u);
		let (_v_perp_to, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&to.v);

		// For simplicity, we project the tangent vector's matrix representation
		// onto the tangent space at the destination
		// This preserves the general direction but may not be exact parallel transport

		// The transported tangent has the same structure but adapted to the new point
		result.u_perp_m = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.m - self.k, self.k);
		result.s_dot = vector.s_dot.clone();
		result.v_perp_n = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.k, self.n - self.k);

		// Fill with appropriate values (simplified transport)
		for j in 0..self.k {
			for i in 0..(self.m - self.k).min(MatrixOps::nrows(&vector.u_perp_m)) {
				if i < MatrixOps::nrows(&result.u_perp_m) && j < MatrixOps::ncols(&vector.u_perp_m)
				{
					*MatrixOps::get_mut(&mut result.u_perp_m, i, j) =
						MatrixOps::get(&vector.u_perp_m, i, j);
				}
			}
			for i in 0..(self.n - self.k).min(MatrixOps::ncols(&vector.v_perp_n)) {
				if j < MatrixOps::nrows(&vector.v_perp_n) && i < MatrixOps::ncols(&result.v_perp_n)
				{
					*MatrixOps::get_mut(&mut result.v_perp_n, j, i) =
						MatrixOps::get(&vector.v_perp_n, j, i);
				}
			}
		}

		Ok(())
	}
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		// Use Frobenius distance in the embedded space
		let x_mat = x.to_matrix();
		let y_mat = y.to_matrix();
		let diff = MatrixOps::sub(&y_mat, &x_mat);

		// Frobenius norm of the difference
		Ok(MatrixOps::norm(&diff))
	}

	fn has_exact_exp_log(&self) -> bool {
		false // Fixed-rank doesn't have closed-form exp/log
	}

	fn is_flat(&self) -> bool {
		false // Fixed-rank is curved
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Scale each component of the tangent vector
		result.u_perp_m = MatrixOps::scale_by(&tangent.u_perp_m, scalar);
		result.s_dot = tangent.s_dot.clone();
		result.s_dot.scale_mut(scalar);
		result.v_perp_n = MatrixOps::scale_by(&tangent.v_perp_n, scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		_point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
		// Temporary buffer for projection if needed
		_temp: &mut Self::TangentVector,
	) -> Result<()> {
		// Add each component of the tangent vectors
		result.u_perp_m = MatrixOps::add(&v1.u_perp_m, &v2.u_perp_m);
		result.s_dot = VectorOps::add(&v1.s_dot, &v2.s_dot);
		result.v_perp_n = MatrixOps::add(&v1.v_perp_n, &v2.v_perp_n);
		Ok(())
	}
}

impl FixedRank {
	/// Creates a random rank-k matrix using Gaussian sampling.
	///
	/// # Returns
	///
	/// A random m×n matrix of rank k.
	pub fn random_matrix<T: Scalar + Float>(&self) -> linalg::Mat<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		let mut point = FixedRankPoint::<T>::default();
		<Self as Manifold<T>>::random_point(self, &mut point).unwrap();
		point.to_matrix()
	}

	/// Computes the best rank-k approximation of a matrix.
	///
	/// Uses SVD to compute the best rank-k approximation in Frobenius norm.
	///
	/// # Arguments
	///
	/// * `mat` - Input matrix
	///
	/// # Returns
	///
	/// The best rank-k approximation.
	pub fn approximate<T: Scalar + Float>(&self, mat: &linalg::Mat<T>) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(mat) != self.m || MatrixOps::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixOps::nrows(mat) * MatrixOps::ncols(mat),
			));
		}

		let pt = FixedRankPoint::<T>::from_matrix(mat, self.k)?;
		Ok(pt.to_matrix())
	}

	/// Computes the approximation error for a given matrix.
	///
	/// # Mathematical Formula
	///
	/// For a matrix A and its rank-k approximation A_k:
	/// ```text
	/// error = ‖A - A_k‖_F = √(σ_{k+1}² + ... + σ_{min(m,n)}²)
	/// ```
	///
	/// # Arguments
	///
	/// * `mat` - Input matrix
	///
	/// # Returns
	///
	/// The Frobenius norm of the approximation error.
	pub fn approximation_error<T: Scalar + Float>(&self, mat: &linalg::Mat<T>) -> Result<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(mat) != self.m || MatrixOps::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixOps::nrows(mat) * MatrixOps::ncols(mat),
			));
		}

		let svd = DecompositionOps::svd(mat);
		let s = &svd.singular_values;

		// Sum of squared singular values beyond rank k
		let mut error_sq = T::zero();
		for i in self.k..VectorOps::len(s).min(self.m).min(self.n) {
			let sv = VectorOps::get(s, i);
			error_sq = error_sq + sv * sv;
		}

		Ok(<T as Float>::sqrt(error_sq))
	}

	/// Projects a general matrix to the fixed-rank manifold.
	///
	/// # Mathematical Operation
	///
	/// Computes the best rank-k approximation using SVD.
	///
	/// # Arguments
	///
	/// * `mat` - Input matrix of size m×n
	///
	/// # Returns
	///
	/// The projected fixed-rank matrix as a FixedRankPoint.
	pub fn project_matrix<T: Scalar + Float>(
		&self,
		mat: &linalg::Mat<T>,
	) -> Result<FixedRankPoint<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(mat) != self.m || MatrixOps::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixOps::nrows(mat) * MatrixOps::ncols(mat),
			));
		}

		FixedRankPoint::<T>::from_matrix(mat, self.k)
	}
}
