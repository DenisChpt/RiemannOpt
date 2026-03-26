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
	linalg::{self, DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
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
	///
	/// Computes X = U diag(S) V^T using column-scaled GEMM to avoid
	/// allocating a full diagonal matrix.
	pub fn to_matrix(&self) -> linalg::Mat<T> {
		let m = self.u.nrows();
		let k = VectorView::len(&self.s);
		// temp = U * diag(S) via backend-optimized column-scaling
		let mut temp = <linalg::Mat<T> as MatrixOps<T>>::zeros(m, k);
		temp.scale_columns(&self.u, &self.s);
		let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(m, self.v.nrows());
		out.gemm_bt(T::one(), temp.as_view(), self.v.as_view(), T::zero());
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

		// Truncate to rank k — copy element-by-element to avoid allocating via columns()
		let m = MatrixView::nrows(&u_full);
		let mut u_k = <linalg::Mat<T> as MatrixOps<T>>::zeros(m, k);
		for j in 0..k {
			for i in 0..m {
				*MatrixOps::get_mut(&mut u_k, i, j) = MatrixView::get(&u_full, i, j);
			}
		}
		let s_k = <linalg::Vec<T> as VectorOps<T>>::from_fn(k, |i| {
			VectorView::get(&svd.singular_values, i)
		});
		// V = Vt^T[:, 0..k] — transpose element-by-element to avoid full transpose allocation
		let n = MatrixView::ncols(&vt_full);
		let mut v_k = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, k);
		for j in 0..k {
			for i in 0..n {
				*MatrixOps::get_mut(&mut v_k, i, j) = MatrixView::get(&vt_full, j, i);
			}
		}

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
	///
	/// Computes ξ = U_⊥ M V^T + U diag(Ṡ) V^T + U N V_⊥^T using in-place GEMM
	/// to minimize intermediate allocations.
	pub fn to_matrix(&self, point: &FixedRankPoint<T>) -> linalg::Mat<T> {
		let m = point.u.nrows();
		let n = point.v.nrows();
		let k = VectorView::len(&point.s);

		// Compute U_perp and V_perp using QR decomposition
		let (u_perp, _) = Self::compute_orthogonal_complement(&point.u);
		let (v_perp, _) = Self::compute_orthogonal_complement(&point.v);

		// term1: U_perp * M * V^T — use buffer for U_perp * M, then GEMM_BT into result
		let mut buf = <linalg::Mat<T> as MatrixOps<T>>::zeros(m, k);
		buf.gemm(T::one(), u_perp.as_view(), self.u_perp_m.as_view(), T::zero()); // buf = U_perp * M  (m × k)
		let mut result = <linalg::Mat<T> as MatrixOps<T>>::zeros(m, n);
		result.gemm_bt(T::one(), buf.as_view(), point.v.as_view(), T::zero()); // result = buf * V^T

		// term2: U * diag(Ṡ) * V^T — scale columns of U by s_dot, accumulate into result
		// Reuse buf for U * diag(Ṡ)
		buf.scale_columns(&point.u, &self.s_dot);
		result.gemm_bt(T::one(), buf.as_view(), point.v.as_view(), T::one()); // result += buf * V^T

		// term3: U * N * V_perp^T — reuse buf for U * N
		let nk = MatrixView::ncols(&self.v_perp_n); // n-k
		let mut buf2 = <linalg::Mat<T> as MatrixOps<T>>::zeros(m, nk);
		buf2.gemm(T::one(), point.u.as_view(), self.v_perp_n.as_view(), T::zero()); // buf2 = U * N  (m × (n-k))
		result.gemm_bt(T::one(), buf2.as_view(), v_perp.as_view(), T::one()); // result += buf2 * V_perp^T

		result
	}

	/// Compute orthogonal complement of a matrix with orthonormal columns
	fn compute_orthogonal_complement(mat: &linalg::Mat<T>) -> (linalg::Mat<T>, linalg::Mat<T>) {
		let m = MatrixView::nrows(mat);
		let k = MatrixView::ncols(mat);

		if k >= m {
			// No orthogonal complement
			return (
				<linalg::Mat<T> as MatrixOps<T>>::zeros(m, 0),
				<linalg::Mat<T> as MatrixOps<T>>::zeros(0, 0),
			);
		}

		// Create identity and project out the columns of mat: I - M·Mᵀ
		let mut q = <linalg::Mat<T> as MatrixOps<T>>::identity(m);
		q.gemm_bt(-T::one(), mat.as_view(), mat.as_view(), T::one());

		// Use QR to get orthonormal basis for the complement
		let qr = DecompositionOps::qr(&q);
		let q_full = qr.q();

		// Extract the last m-k columns — copy element-by-element to avoid allocating via columns()
		let mut u_perp = <linalg::Mat<T> as MatrixOps<T>>::zeros(m, m - k);
		for j in 0..m - k {
			for i in 0..m {
				*MatrixOps::get_mut(&mut u_perp, i, j) = MatrixView::get(q_full, i, j + k);
			}
		}
		let r_perp = <linalg::Mat<T> as MatrixOps<T>>::zeros(m - k, m - k); // Placeholder

		(u_perp, r_perp)
	}

	/// Project an ambient-space m×n matrix onto the tangent space at a point.
	///
	/// Given point P = U Σ V^T and ambient matrix Z, computes the tangent
	/// components:
	/// - S_dot = U^T Z V  (k × k diagonal change)
	/// - M = U_perp^T Z V  ((m-k) × k, orthogonal complement direction)
	/// - N = U^T Z V_perp  (k × (n-k), orthogonal complement direction)
	pub fn from_ambient(point: &FixedRankPoint<T>, ambient: &linalg::Mat<T>) -> Self {
		let k = VectorView::len(&point.s);
		let n = MatrixView::ncols(ambient);

		// ut_z = U^T * Z  (k × n) — use gemm_at to avoid transposing U
		let mut ut_z = <linalg::Mat<T> as MatrixOps<T>>::zeros(k, n);
		ut_z.gemm_at(T::one(), point.u.as_view(), ambient.as_view(), T::zero());

		// S_dot = diag(U^T Z V) = diag(ut_z * V)
		let mut ut_z_v = <linalg::Mat<T> as MatrixOps<T>>::zeros(k, k);
		ut_z_v.gemm(T::one(), ut_z.as_view(), point.v.as_view(), T::zero());
		let s_dot = VectorOps::from_fn(k, |i| MatrixView::get(&ut_z_v, i, i));

		// M = U_perp^T Z V — use gemm_at to avoid transposing U_perp
		let (u_perp, _) = Self::compute_orthogonal_complement(&point.u);
		let (v_perp, _) = Self::compute_orthogonal_complement(&point.v);

		let mk = MatrixView::ncols(&u_perp); // m-k
		let mut upt_z = <linalg::Mat<T> as MatrixOps<T>>::zeros(mk, n);
		upt_z.gemm_at(T::one(), u_perp.as_view(), ambient.as_view(), T::zero()); // (m-k) × n
		let mut u_perp_m = <linalg::Mat<T> as MatrixOps<T>>::zeros(mk, k);
		u_perp_m.gemm(T::one(), upt_z.as_view(), point.v.as_view(), T::zero()); // (m-k) × k

		// N = U^T Z V_perp — reuse ut_z
		let nk = MatrixView::ncols(&v_perp); // n-k
		let mut v_perp_n = <linalg::Mat<T> as MatrixOps<T>>::zeros(k, nk);
		v_perp_n.gemm(T::one(), ut_z.as_view(), v_perp.as_view(), T::zero()); // k × (n-k)

		Self::new(u_perp_m, s_dot, v_perp_n)
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

	/// Returns the rank k.
	#[inline]
	pub fn rank(&self) -> usize {
		self.k
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
		// QR decomposition with sign correction to ensure continuity.
		// Without sign correction, QR can flip column signs arbitrarily
		// (QR is only unique up to sign of diagonal of R).
		let m = MatrixView::nrows(u);
		let k = MatrixView::ncols(u);
		let n = MatrixView::nrows(v);

		let qr_u = DecompositionOps::qr(u);
		u.copy_from(qr_u.q());
		// Read diagonal of R directly — no clone needed since qr.r() returns a reference
		for j in 0..k.min(MatrixView::ncols(qr_u.r())) {
			if MatrixView::get(qr_u.r(), j, j) < T::zero() {
				for i in 0..m {
					*MatrixOps::get_mut(u, i, j) = T::zero() - MatrixView::get(u, i, j);
				}
			}
		}

		let qr_v = DecompositionOps::qr(v);
		v.copy_from(qr_v.q());
		for j in 0..k.min(MatrixView::ncols(qr_v.r())) {
			if MatrixView::get(qr_v.r(), j, j) < T::zero() {
				for i in 0..n {
					*MatrixOps::get_mut(v, i, j) = T::zero() - MatrixView::get(v, i, j);
				}
			}
		}
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
		if MatrixView::nrows(x) != self.m || MatrixView::ncols(x) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixView::nrows(x) * MatrixView::ncols(x),
			));
		}

		// Check rank using SVD
		let svd = DecompositionOps::svd(x);
		let s = &svd.singular_values;

		// Count non-zero singular values
		let mut rank = 0;
		for i in 0..VectorView::len(s).min(self.m).min(self.n) {
			if VectorView::get(s, i) > <T as Scalar>::from_f64(self.tolerance) {
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

		if MatrixView::nrows(z) != self.m || MatrixView::ncols(z) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixView::nrows(z) * MatrixView::ncols(z),
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
	type Workspace = ();

	fn name(&self) -> &str {
		"FixedRank"
	}

	fn dimension(&self) -> usize {
		self.k * (self.m + self.n - self.k)
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		// Check that U and V are on Stiefel manifolds
		let mut u_gram = linalg::Mat::<T>::zeros(self.k, self.k);
		u_gram.gemm_at(T::one(), point.u.as_view(), point.u.as_view(), T::zero());
		let mut v_gram = linalg::Mat::<T>::zeros(self.k, self.k);
		v_gram.gemm_at(T::one(), point.v.as_view(), point.v.as_view(), T::zero());

		// Check orthogonality
		for i in 0..self.k {
			for j in 0..self.k {
				let u_val = if i == j {
					MatrixView::get(&u_gram, i, j) - T::one()
				} else {
					MatrixView::get(&u_gram, i, j)
				};
				let v_val = if i == j {
					MatrixView::get(&v_gram, i, j) - T::one()
				} else {
					MatrixView::get(&v_gram, i, j)
				};

				if <T as Float>::abs(u_val) > tol || <T as Float>::abs(v_val) > tol {
					return false;
				}
			}
		}

		// Check that singular values are positive
		for i in 0..self.k {
			if VectorView::get(&point.s, i) <= T::zero() {
				return false;
			}
		}

		true
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Copy the input point — use in-place copy to avoid allocation
		// Ensure result has correct dimensions before copying
		if MatrixView::nrows(&result.u) != MatrixView::nrows(&point.u)
			|| MatrixView::ncols(&result.u) != MatrixView::ncols(&point.u)
		{
			result.u = <linalg::Mat<T> as MatrixOps<T>>::zeros(
				MatrixView::nrows(&point.u),
				MatrixView::ncols(&point.u),
			);
		}
		if VectorView::len(&result.s) != VectorView::len(&point.s) {
			result.s = <linalg::Vec<T> as VectorOps<T>>::zeros(VectorView::len(&point.s));
		}
		if MatrixView::nrows(&result.v) != MatrixView::nrows(&point.v)
			|| MatrixView::ncols(&result.v) != MatrixView::ncols(&point.v)
		{
			result.v = <linalg::Mat<T> as MatrixOps<T>>::zeros(
				MatrixView::nrows(&point.v),
				MatrixView::ncols(&point.v),
			);
		}
		result.u.copy_from(&point.u);
		result.s.copy_from(&point.s);
		result.v.copy_from(&point.v);

		// Project U and V onto Stiefel manifolds
		self.project_factors(&mut result.u, &mut result.v);

		// Ensure singular values are positive
		for i in 0..self.k {
			if VectorView::get(&result.s, i) < T::epsilon() {
				*VectorOps::get_mut(&mut result.s, i) = T::epsilon();
			}
		}
	}

	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// Tangent vectors already have the correct structure — copy in-place
		result.u_perp_m.copy_from(&vector.u_perp_m);
		result.s_dot.copy_from(&vector.s_dot);
		result.v_perp_n.copy_from(&vector.v_perp_n);
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> Result<T> {
		// <u, v> = tr(M_u^T M_v) + ⟨s_dot_u, s_dot_v⟩ + tr(N_u^T N_v)
		// Frobenius inner product = Σ a_ij * b_ij (zero alloc)
		let mut inner = T::zero();

		// U_perp component: tr(M_u^T M_v) = Σ_ij M_u[i,j] * M_v[i,j]
		let rows_m = MatrixView::nrows(&u.u_perp_m);
		let cols_m = MatrixView::ncols(&u.u_perp_m);
		for i in 0..rows_m {
			for j in 0..cols_m {
				inner =
					inner + MatrixView::get(&u.u_perp_m, i, j) * MatrixView::get(&v.u_perp_m, i, j);
			}
		}

		// S component: ⟨s_dot_u, s_dot_v⟩
		inner = inner + VectorView::dot(&u.s_dot, &v.s_dot);

		// V_perp component: tr(N_u^T N_v) = Σ_ij N_u[i,j] * N_v[i,j]
		let rows_n = MatrixView::nrows(&u.v_perp_n);
		let cols_n = MatrixView::ncols(&u.v_perp_n);
		for i in 0..rows_n {
			for j in 0..cols_n {
				inner =
					inner + MatrixView::get(&u.v_perp_n, i, j) * MatrixView::get(&v.v_perp_n, i, j);
			}
		}

		Ok(inner)
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) -> Result<()> {
		// Orthographic retraction: R_X(ξ) = (U + U_perp·M·S⁻¹)(S + S_dot)(V + V_perp·Nᵀ·S⁻¹)ᵀ

		// compute_orthogonal_complement allocates (QR inside) — unavoidable
		let (u_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.u);
		let (v_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.v);

		// S⁻¹ diagonal — small k×k alloc
		let s_inv = <linalg::Mat<T> as MatrixOps<T>>::from_fn(self.k, self.k, |i, j| {
			if i == j {
				T::one() / VectorView::get(&point.s, i)
			} else {
				T::zero()
			}
		});

		// U_new = U + U_perp · M · S⁻¹  →  result.u = point.u, then += via GEMM chain
		result.u.copy_from(&point.u);
		// temp = M · S⁻¹  (reuse u_perp_m_sinv would alloc; use GEMM into a small buffer)
		let mut m_sinv =
			<linalg::Mat<T> as MatrixOps<T>>::zeros(MatrixView::nrows(&tangent.u_perp_m), self.k);
		m_sinv.gemm(T::one(), tangent.u_perp_m.as_view(), s_inv.as_view(), T::zero());
		// result.u += U_perp · m_sinv
		result.u.gemm(T::one(), u_perp.as_view(), m_sinv.as_view(), T::one());

		// S_new = S + S_dot (in-place)
		result.s.copy_from(&point.s);
		result.s.add_assign(&tangent.s_dot);

		// V_new = V + V_perp · Nᵀ · S⁻¹
		result.v.copy_from(&point.v);
		// temp = Nᵀ · S⁻¹
		let mut nt_sinv =
			<linalg::Mat<T> as MatrixOps<T>>::zeros(MatrixView::ncols(&tangent.v_perp_n), self.k);
		nt_sinv.gemm_at(T::one(), tangent.v_perp_n.as_view(), s_inv.as_view(), T::zero());
		// result.v += V_perp · nt_sinv
		result.v.gemm(T::one(), v_perp.as_view(), nt_sinv.as_view(), T::one());

		// Project factors back to Stiefel
		self.project_factors(&mut result.u, &mut result.v);

		// Ensure singular values are positive
		for i in 0..self.k {
			if VectorView::get(&result.s, i) < T::epsilon() {
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
		_ws: &mut (),
	) -> Result<()> {
		// For the canonical metric, just project to tangent space
		self.project_tangent(point, euclidean_grad, result, _ws)
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

		// Orthogonalize — copy Q factor into u/v to avoid cloning
		let qr_u = DecompositionOps::qr(&u);
		u.copy_from(qr_u.q());

		let qr_v = DecompositionOps::qr(&v);
		v.copy_from(qr_v.q());

		// Random positive singular values
		let mut s = <linalg::Vec<T> as VectorOps<T>>::zeros(self.k);
		for i in 0..self.k {
			let val: f64 = normal.sample(&mut rng);
			*VectorOps::get_mut(&mut s, i) = <T as Scalar>::from_f64(val.abs() + 1.0);
		}

		*result = FixedRankPoint::new(u, s, v);
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
		if MatrixView::nrows(&vector.u_perp_m) != self.m - self.k
			|| MatrixView::ncols(&vector.u_perp_m) != self.k
		{
			return false;
		}
		if VectorView::len(&vector.s_dot) != self.k {
			return false;
		}
		if MatrixView::nrows(&vector.v_perp_n) != self.k
			|| MatrixView::ncols(&vector.v_perp_n) != self.n - self.k
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
		_ws: &mut (),
	) -> Result<()> {
		// For fixed-rank manifold, we use a simple approximation
		// The inverse of the orthographic retraction is complex, so we approximate
		// by computing the tangent that moves in the direction of other - point

		// Compute the difference in matrix form: diff = other - point
		let point_mat = point.to_matrix();
		let other_mat = other.to_matrix();
		// Compute diff in-place: reuse other_mat by subtracting point_mat element-wise
		let mut diff = other_mat;
		for i in 0..MatrixView::nrows(&diff) {
			for j in 0..MatrixView::ncols(&diff) {
				*MatrixOps::get_mut(&mut diff, i, j) =
					MatrixView::get(&diff, i, j) - MatrixView::get(&point_mat, i, j);
			}
		}

		// Project onto the tangent space at point
		let (u_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.u);
		let (v_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.v);

		// M = U_perp^T * diff * V — use gemm_at, then gemm into result
		let mk = MatrixView::ncols(&u_perp);
		let n = MatrixView::ncols(&diff);
		let mut upt_diff = linalg::Mat::<T>::zeros(mk, n);
		upt_diff.gemm_at(T::one(), u_perp.as_view(), diff.as_view(), T::zero());
		result.u_perp_m = <linalg::Mat<T> as MatrixOps<T>>::zeros(mk, self.k);
		result
			.u_perp_m
			.gemm(T::one(), upt_diff.as_view(), point.v.as_view(), T::zero());

		// S_dot = diag(U^T * diff * V) — use gemm_at, then gemm into buffer
		let mut ut_diff = linalg::Mat::<T>::zeros(self.k, n);
		ut_diff.gemm_at(T::one(), point.u.as_view(), diff.as_view(), T::zero());
		let mut s_component = linalg::Mat::<T>::zeros(self.k, self.k);
		s_component.gemm(T::one(), ut_diff.as_view(), point.v.as_view(), T::zero());
		result.s_dot = <linalg::Vec<T> as VectorOps<T>>::from_fn(self.k, |i| {
			MatrixView::get(&s_component, i, i)
		});

		// N = U^T * diff * V_perp — reuse ut_diff
		let nk = MatrixView::ncols(&v_perp);
		result.v_perp_n = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.k, nk);
		result.v_perp_n.gemm(T::one(), ut_diff.as_view(), v_perp.as_view(), T::zero());

		Ok(())
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		_to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// Simplified transport: copy components in-place (zero alloc).
		// This is an approximation — exact transport on fixed-rank is complex.
		result.u_perp_m.copy_from(&vector.u_perp_m);
		result.s_dot.copy_from(&vector.s_dot);
		result.v_perp_n.copy_from(&vector.v_perp_n);
		Ok(())
	}
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		// Frobenius distance ‖Y - X‖_F without allocating a diff matrix
		let x_mat = x.to_matrix();
		let y_mat = y.to_matrix();
		let mut norm_sq = T::zero();
		for i in 0..MatrixView::nrows(&y_mat) {
			for j in 0..MatrixView::ncols(&y_mat) {
				let d = MatrixView::get(&y_mat, i, j) - MatrixView::get(&x_mat, i, j);
				norm_sq = norm_sq + d * d;
			}
		}
		Ok(Float::sqrt(norm_sq))
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
		// Copy then scale in-place (zero alloc)
		result.u_perp_m.copy_from(&tangent.u_perp_m);
		result.u_perp_m.scale_mut(scalar);
		result.s_dot.copy_from(&tangent.s_dot);
		result.s_dot.scale_mut(scalar);
		result.v_perp_n.copy_from(&tangent.v_perp_n);
		result.v_perp_n.scale_mut(scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		_point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Copy v1 then add v2 in-place (zero alloc)
		result.u_perp_m.copy_from(&v1.u_perp_m);
		result.u_perp_m.add_assign(&v2.u_perp_m);
		result.s_dot.copy_from(&v1.s_dot);
		result.s_dot.add_assign(&v2.s_dot);
		result.v_perp_n.copy_from(&v1.v_perp_n);
		result.v_perp_n.add_assign(&v2.v_perp_n);
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
		if MatrixView::nrows(mat) != self.m || MatrixView::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixView::nrows(mat) * MatrixView::ncols(mat),
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
		if MatrixView::nrows(mat) != self.m || MatrixView::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixView::nrows(mat) * MatrixView::ncols(mat),
			));
		}

		let svd = DecompositionOps::svd(mat);
		let s = &svd.singular_values;

		// Sum of squared singular values beyond rank k
		let mut error_sq = T::zero();
		for i in self.k..VectorView::len(s).min(self.m).min(self.n) {
			let sv = VectorView::get(s, i);
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
		if MatrixView::nrows(mat) != self.m || MatrixView::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.m * self.n,
				MatrixView::nrows(mat) * MatrixView::ncols(mat),
			));
		}

		FixedRankPoint::<T>::from_matrix(mat, self.k)
	}
}
