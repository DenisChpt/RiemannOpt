//! # Positive Semi-Definite Cone S⁺(n)
//!
//! The cone S⁺(n) of n×n symmetric positive semi-definite (PSD) matrices forms
//! a convex cone in the space of symmetric matrices. It provides a fundamental
//! framework for semidefinite programming and covariance estimation.
//!
//! ## Mathematical Definition
//!
//! The PSD cone is formally defined as:
//! ```text
//! S⁺(n) = {X ∈ ℝⁿˣⁿ : X = X^T, X ⪰ 0}
//!        = {X ∈ S(n) : λ_i(X) ≥ 0 for all i}
//! ```
//! where S(n) is the space of symmetric matrices and λ_i(X) are eigenvalues.
//!
//! Equivalently, X ∈ S⁺(n) if and only if v^T X v ≥ 0 for all v ∈ ℝⁿ.
//!
//! ## Geometric Structure
//!
//! ### Boundary and Interior
//! - **Interior**: S⁺⁺(n) = {X ∈ S⁺(n) : X ≻ 0} (positive definite)
//! - **Boundary**: ∂S⁺(n) = {X ∈ S⁺(n) : det(X) = 0} (singular PSD)
//! - **Extreme rays**: Rank-1 matrices X = vv^T
//!
//! ### Tangent Cone
//! At X ∈ S⁺(n), the tangent cone is:
//! ```text
//! T_X S⁺(n) = {V ∈ S(n) : v^T V v ≥ 0 for all v ∈ ker(X)}
//! ```
//! For interior points, T_X S⁺(n) = S(n) (all symmetric matrices).
//!
//! ### Riemannian Metrics
//!
//! #### 1. Euclidean (Frobenius) Metric
//! ```text
//! g_X(U, V) = tr(UV) = ⟨U, V⟩_F
//! ```
//!
//! #### 2. Affine-Invariant Metric (for interior)
//! ```text
//! g_X^{AI}(U, V) = tr(X⁻¹U X⁻¹V)
//! ```
//!
//! #### 3. Log-Euclidean Metric
//! ```text
//! g_X^{LE}(U, V) = g_{log(X)}^E(Dlog_X U, Dlog_X V)
//! ```
//!
//! ### Projection to S⁺(n)
//! For any symmetric matrix A:
//! ```text
//! P_{S⁺}(A) = Q max(0, Λ) Q^T
//! ```
//! where A = QΛQ^T is the eigendecomposition.
//!
//! ## Distance Formulas
//!
//! ### Euclidean Distance
//! ```text
//! d_E(X, Y) = ‖X - Y‖_F
//! ```
//!
//! ### Affine-Invariant Distance (interior)
//! ```text
//! d_{AI}(X, Y) = ‖log(X⁻¹/²YX⁻¹/²)‖_F
//! ```
//!
//! ## Optimization on S⁺(n)
//!
//! ### Riemannian Gradient
//! For f: S⁺(n) → ℝ with Euclidean gradient ∇f(X):
//! ```text
//! grad^E f(X) = P_{T_X}(∇f(X))
//! ```
//!
//! ### Retraction
//! A simple retraction is the metric projection:
//! ```text
//! R_X(V) = P_{S⁺}(X + V)
//! ```
//!
//! ## Applications
//!
//! 1. **Semidefinite Programming (SDP)**: Convex optimization over S⁺(n)
//! 2. **Covariance Estimation**: Regularized sample covariance matrices
//! 3. **Kernel Learning**: Learning positive semi-definite kernel matrices
//! 4. **Quantum State Tomography**: Density matrices in quantum mechanics
//! 5. **Graph Theory**: Laplacian matrices and spectral methods
//! 6. **Control Theory**: Lyapunov functions and stability analysis
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Efficient projection** via eigendecomposition
//! - **Numerical stability** for near-singular matrices
//! - **Proper scaling** for off-diagonal elements in vectorization
//! - **Boundary handling** for rank-deficient matrices
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::PSDCone;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{self, VectorOps, MatrixOps, DecompositionOps};
//!
//! // Create S⁺(3) - 3×3 PSD matrices
//! let psd_cone = PSDCone::new(3)?;
//!
//! // Random PSD matrix (stored as vector)
//! let mut x = linalg::Vec::<f64>::zeros(6);
//! <PSDCone as Manifold<f64>>::random_point(&psd_cone, &mut x)?;
//!
//! // Verify positive semi-definiteness
//! let mut x_mat = linalg::Mat::<f64>::zeros(3, 3);
//! psd_cone.vector_to_matrix::<f64>(&x, &mut x_mat);
//! let eigenvalues: linalg::Vec<f64> = DecompositionOps::symmetric_eigen(&x_mat).eigenvalues;
//! for i in 0..eigenvalues.len() {
//!     assert!(VectorOps::get(&eigenvalues, i) >= -1e-10);
//! }
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

/// Pre-allocated workspace for PSD Cone operations.
///
/// Contains matrix buffers for vec↔mat conversions, avoiding heap
/// allocations in hot-path methods like `retract` and `project_tangent`.
pub struct PSDConeWorkspace<T: Scalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// n×n matrix buffer A (e.g. point as matrix)
	pub mat_a: linalg::Mat<T>,
	/// n×n matrix buffer B (e.g. tangent as matrix, or symmetrized result)
	pub mat_b: linalg::Mat<T>,
}

impl<T: Scalar> Default for PSDConeWorkspace<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn default() -> Self {
		Self {
			mat_a: MatrixOps::zeros(0, 0),
			mat_b: MatrixOps::zeros(0, 0),
		}
	}
}

unsafe impl<T: Scalar> Send for PSDConeWorkspace<T> where linalg::DefaultBackend: LinAlgBackend<T> {}
unsafe impl<T: Scalar> Sync for PSDConeWorkspace<T> where linalg::DefaultBackend: LinAlgBackend<T> {}

impl<T: Scalar> std::fmt::Debug for PSDConeWorkspace<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PSDConeWorkspace").finish()
	}
}

/// The positive semi-definite cone S⁺(n) of n×n symmetric PSD matrices.
///
/// This structure represents the convex cone of symmetric positive semi-definite
/// matrices, equipped with various Riemannian metrics for optimization.
///
/// # Type Parameters
///
/// The manifold is generic over the scalar type T through the Manifold trait.
///
/// # Invariants
///
/// - `n ≥ 1`: Matrix dimension must be positive
/// - All points X satisfy X = X^T and X ⪰ 0
/// - Tangent vectors at interior points are arbitrary symmetric matrices
/// - Tangent vectors at boundary points satisfy additional constraints
#[derive(Debug, Clone)]
pub struct PSDCone {
	/// Dimension of the matrices (n×n)
	n: usize,
	/// Numerical tolerance for constraint validation
	tolerance: f64,
	/// Whether to use strict positive definiteness (interior only)
	strict: bool,
}

impl PSDCone {
	/// Creates a new PSD cone manifold S⁺(n).
	///
	/// # Arguments
	///
	/// * `n` - Dimension of the matrices (must be ≥ 1)
	///
	/// # Returns
	///
	/// A PSD cone manifold with dimension n(n+1)/2.
	///
	/// # Errors
	///
	/// Returns `ManifoldError::InvalidParameter` if n = 0.
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::PSDCone;
	/// // Create S⁺(3) - 3×3 PSD matrices
	/// let psd_cone = PSDCone::new(3)?;
	/// assert_eq!(psd_cone.matrix_size(), 3);
	/// assert_eq!(psd_cone.manifold_dim(), 6); // 3*(3+1)/2 = 6
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	pub fn new(n: usize) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::invalid_parameter("PSD cone requires n ≥ 1"));
		}
		Ok(Self {
			n,
			tolerance: 1e-10,
			strict: false,
		})
	}

	/// Creates a PSD cone with custom parameters.
	///
	/// # Arguments
	///
	/// * `n` - Matrix dimension
	/// * `tolerance` - Numerical tolerance for eigenvalue checks
	/// * `strict` - If true, only accepts positive definite matrices
	pub fn with_parameters(n: usize, tolerance: f64, strict: bool) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::invalid_parameter("PSD cone requires n ≥ 1"));
		}
		if tolerance <= 0.0 || tolerance >= 1.0 {
			return Err(ManifoldError::invalid_parameter(
				"Tolerance must be in (0, 1)",
			));
		}
		Ok(Self {
			n,
			tolerance,
			strict,
		})
	}

	/// Returns the matrix dimension n.
	#[inline]
	pub fn matrix_size(&self) -> usize {
		self.n
	}

	/// Returns the manifold dimension n(n+1)/2.
	#[inline]
	pub fn manifold_dim(&self) -> usize {
		self.n * (self.n + 1) / 2
	}

	/// Validates that a matrix is positive semi-definite.
	///
	/// # Mathematical Check
	///
	/// Verifies that X = X^T and all eigenvalues λ_i(X) ≥ 0.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If matrix is not n×n
	/// - `InvalidPoint`: If matrix is not symmetric or not PSD
	pub fn check_matrix<T: Scalar>(&self, x: &linalg::Mat<T>) -> Result<()>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(x) != self.n || MatrixOps::ncols(x) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}×{}", self.n, self.n),
				format!("{}×{}", MatrixOps::nrows(x), MatrixOps::ncols(x)),
			));
		}

		// Check symmetry
		for i in 0..self.n {
			for j in i + 1..self.n {
				if <T as Float>::abs(MatrixOps::get(x, i, j) - MatrixOps::get(x, j, i))
					> <T as Scalar>::from_f64(self.tolerance)
				{
					return Err(ManifoldError::invalid_point(format!(
						"Matrix not symmetric: |X[{},{}] - X[{},{}]| = {} > {}",
						i,
						j,
						j,
						i,
						<T as Float>::abs(MatrixOps::get(x, i, j) - MatrixOps::get(x, j, i)),
						self.tolerance
					)));
				}
			}
		}

		// Check eigenvalues
		let eigen = DecompositionOps::symmetric_eigen(x);
		let min_eigenvalue =
			VectorOps::iter(&eigen.eigenvalues).fold(<T as Float>::max_value(), |min, val| {
				if val < min {
					val
				} else {
					min
				}
			});

		let threshold = if self.strict {
			<T as Scalar>::from_f64(self.tolerance)
		} else {
			-<T as Scalar>::from_f64(self.tolerance)
		};

		if min_eigenvalue < threshold {
			return Err(ManifoldError::invalid_point(format!(
				"Matrix not {}: minimum eigenvalue {} < {}",
				if self.strict {
					"positive definite"
				} else {
					"positive semi-definite"
				},
				min_eigenvalue,
				threshold
			)));
		}

		Ok(())
	}

	/// Converts a symmetric matrix to its vectorized form.
	///
	/// Uses the standard vectorization with √2 scaling for off-diagonal elements
	/// to preserve the Frobenius inner product.
	pub fn matrix_to_vector<T: Scalar>(&self, mat: &linalg::Mat<T>, vec: &mut linalg::Vec<T>)
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.mat_to_vec(mat, vec);
	}

	/// Converts a vector to its symmetric matrix form.
	pub fn vector_to_matrix<T: Scalar>(&self, vec: &linalg::Vec<T>, mat: &mut linalg::Mat<T>)
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.vec_to_mat(vec, mat);
	}

	/// Convert vector to symmetric matrix, writing into pre-allocated buffer.
	fn vec_to_mat<T: Scalar>(&self, vec: &linalg::Vec<T>, mat: &mut linalg::Mat<T>)
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		let n = self.n;
		let mut idx = 0;
		for i in 0..n {
			for j in i..n {
				if i == j {
					*MatrixOps::get_mut(mat, i, j) = VectorOps::get(vec, idx);
				} else {
					let val = VectorOps::get(vec, idx)
						/ <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
					*MatrixOps::get_mut(mat, i, j) = val;
					*MatrixOps::get_mut(mat, j, i) = val;
				}
				idx += 1;
			}
		}
	}

	/// Convert symmetric matrix to vector form (upper triangular part), writing into `vec`.
	fn mat_to_vec<T: Scalar>(&self, mat: &linalg::Mat<T>, vec: &mut linalg::Vec<T>)
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		let n = self.n;
		let mut idx = 0;
		for i in 0..n {
			for j in i..n {
				if i == j {
					*VectorOps::get_mut(vec, idx) = MatrixOps::get(mat, i, j);
				} else {
					*VectorOps::get_mut(vec, idx) = MatrixOps::get(mat, i, j)
						* <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
				}
				idx += 1;
			}
		}
	}

	/// Project a matrix to the PSD cone
	fn project_to_psd<T: Scalar>(&self, mat: &linalg::Mat<T>) -> linalg::Mat<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		// Symmetrize in-place into a buffer
		let mut sym = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i..self.n {
				let avg = half * (MatrixOps::get(mat, i, j) + MatrixOps::get(mat, j, i));
				*MatrixOps::get_mut(&mut sym, i, j) = avg;
				*MatrixOps::get_mut(&mut sym, j, i) = avg;
			}
		}

		// Eigendecomposition
		let mut eigen = DecompositionOps::symmetric_eigen(&sym);

		// Project eigenvalues to non-negative (in-place, no clone)
		for i in 0..VectorOps::len(&eigen.eigenvalues) {
			if VectorOps::get(&eigen.eigenvalues, i) < T::zero() {
				*VectorOps::get_mut(&mut eigen.eigenvalues, i) = T::zero();
			}
		}

		// Reconstruct: Q * diag(λ) * Q^T
		// temp = Q * diag(λ) via backend-optimized column-scaling
		let q = &eigen.eigenvectors;
		let n = self.n;
		let mut temp = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		temp.scale_columns(q, &eigen.eigenvalues);
		let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		out.gemm_bt(T::one(), &temp, q, T::zero());
		out
	}

	/// Check if a matrix is in the PSD cone
	fn is_psd<T: Scalar>(&self, mat: &linalg::Mat<T>, tol: T) -> bool
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		// Check symmetry
		for i in 0..self.n {
			for j in i + 1..self.n {
				if <T as Float>::abs(MatrixOps::get(mat, i, j) - MatrixOps::get(mat, j, i)) > tol {
					return false;
				}
			}
		}

		// Check positive semi-definiteness
		let eigen = DecompositionOps::symmetric_eigen(mat);
		let threshold = if self.strict { tol } else { -tol };
		let result = VectorOps::iter(&eigen.eigenvalues).all(|lambda| lambda >= threshold);
		result
	}

	/// Projects a matrix onto the PSD cone.
	///
	/// # Mathematical Operation
	///
	/// For a symmetric matrix A with eigendecomposition A = QΛQ^T:
	/// ```text
	/// P_{S⁺}(A) = Q max(0, Λ) Q^T
	/// ```
	///
	/// # Arguments
	///
	/// * `mat` - Input matrix (will be symmetrized first)
	///
	/// # Returns
	///
	/// The projected PSD matrix.
	pub fn project_matrix<T: Scalar>(&self, mat: &linalg::Mat<T>) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(mat) != self.n || MatrixOps::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}×{}", self.n, self.n),
				format!("{}×{}", MatrixOps::nrows(mat), MatrixOps::ncols(mat)),
			));
		}

		Ok(self.project_to_psd(mat))
	}

	/// Computes the distance to the PSD cone.
	///
	/// # Mathematical Formula
	///
	/// For a symmetric matrix A:
	/// ```text
	/// dist(A, S⁺) = ‖A - P_{S⁺}(A)‖_F
	/// ```
	///
	/// # Arguments
	///
	/// * `mat` - Input matrix
	///
	/// # Returns
	///
	/// The Frobenius distance to the nearest PSD matrix.
	pub fn distance_to_cone<T: Scalar>(&self, mat: &linalg::Mat<T>) -> Result<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(mat) != self.n || MatrixOps::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}×{}", self.n, self.n),
				format!("{}×{}", MatrixOps::nrows(mat), MatrixOps::ncols(mat)),
			));
		}

		let projected = self.project_to_psd(mat);
		// Element-wise ‖mat - projected‖_F without allocation
		let mut norm_sq = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let d = MatrixOps::get(mat, i, j) - MatrixOps::get(&projected, i, j);
				norm_sq = norm_sq + d * d;
			}
		}
		Ok(<T as Float>::sqrt(norm_sq))
	}

	/// Computes the minimum eigenvalue of a matrix.
	///
	/// Useful for checking how far a matrix is from being PSD.
	pub fn minimum_eigenvalue<T: Scalar>(&self, mat: &linalg::Mat<T>) -> Result<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if MatrixOps::nrows(mat) != self.n || MatrixOps::ncols(mat) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}×{}", self.n, self.n),
				format!("{}×{}", MatrixOps::nrows(mat), MatrixOps::ncols(mat)),
			));
		}

		// Symmetrize in-place into a buffer
		let mut sym = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i..self.n {
				let avg = half * (MatrixOps::get(mat, i, j) + MatrixOps::get(mat, j, i));
				*MatrixOps::get_mut(&mut sym, i, j) = avg;
				*MatrixOps::get_mut(&mut sym, j, i) = avg;
			}
		}
		let eigen = DecompositionOps::symmetric_eigen(&sym);

		Ok(
			VectorOps::iter(&eigen.eigenvalues).fold(<T as Float>::max_value(), |min, val| {
				if val < min {
					val
				} else {
					min
				}
			}),
		)
	}
}

impl<T> Manifold<T> for PSDCone
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;
	type Workspace = PSDConeWorkspace<T>;

	fn create_workspace(&self, _proto_point: &Self::Point) -> Self::Workspace {
		PSDConeWorkspace {
			mat_a: MatrixOps::zeros(self.n, self.n),
			mat_b: MatrixOps::zeros(self.n, self.n),
		}
	}

	fn name(&self) -> &str {
		"PSDCone"
	}

	fn dimension(&self) -> usize {
		self.n * (self.n + 1) / 2
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if VectorOps::len(point) != self.n * (self.n + 1) / 2 {
			return false;
		}

		let mut mat = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut mat);
		self.is_psd(&mat, tol)
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

		if VectorOps::len(vector) != self.n * (self.n + 1) / 2 {
			return false;
		}

		// For interior points, tangent space is all symmetric matrices
		// For boundary points (with zero eigenvalues), it's more restricted
		let mut mat = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		self.vec_to_mat(vector, &mut mat);

		// Check symmetry
		for i in 0..self.n {
			for j in i + 1..self.n {
				if <T as Float>::abs(MatrixOps::get(&mat, i, j) - MatrixOps::get(&mat, j, i)) > tol
				{
					return false;
				}
			}
		}

		// For boundary points, check additional constraint
		if self.strict {
			return true; // In strict mode, we're always in interior
		}

		// Check if tangent respects the cone constraint at boundary
		let mut x_mat = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut x_mat);
		let eigen = DecompositionOps::symmetric_eigen(&x_mat);

		// Find near-zero eigenvalues (boundary)
		for i in 0..VectorOps::len(&eigen.eigenvalues) {
			let lambda = VectorOps::get(&eigen.eigenvalues, i);
			if <T as Float>::abs(lambda) < <T as Scalar>::from_f64(self.tolerance) {
				// For zero eigenvalue, check v^T * mat * v >= 0
				// Compute directly without extracting column or allocating mat*v
				let n = MatrixOps::nrows(&eigen.eigenvectors);
				let mut vt_mat_v = T::zero();
				for r in 0..n {
					let mut row_sum = T::zero();
					for c in 0..n {
						row_sum = row_sum
							+ MatrixOps::get(&mat, r, c)
								* MatrixOps::get(&eigen.eigenvectors, c, i);
					}
					vt_mat_v =
						vt_mat_v + MatrixOps::get(&eigen.eigenvectors, r, i) * row_sum;
				}
				if vt_mat_v < -tol {
					return false;
				}
			}
		}

		true
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let dim = self.n * (self.n + 1) / 2;
		// Ensure result has correct size
		if VectorOps::len(result) != dim {
			*result = VectorOps::zeros(dim);
		}

		let mut mat = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		if VectorOps::len(point) == self.n * self.n {
			// If given as full matrix, reshape
			mat = <linalg::Mat<T> as MatrixOps<T>>::from_column_slice(
				self.n,
				self.n,
				VectorOps::as_slice(point),
			);
		} else if VectorOps::len(point) == dim {
			self.vec_to_mat(point, &mut mat);
		}
		// else: wrong size, mat stays as zero matrix

		let projected = self.project_to_psd(&mat);
		self.mat_to_vec(&projected, result);
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		let dim = self.n * (self.n + 1) / 2;

		if VectorOps::len(point) != dim || VectorOps::len(vector) != dim {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}", dim),
				format!("{}", VectorOps::len(point).max(VectorOps::len(vector))),
			));
		}

		if self.strict {
			// For interior points, tangent space = all symmetric matrices.
			// The vectorized representation already encodes symmetry,
			// so projection is identity (zero alloc).
			result.copy_from(vector);
		} else {
			// For boundary points, additional projection needed via eigendecomposition.
			// Use workspace buffers for vec_to_mat conversions.
			self.vec_to_mat(vector, &mut ws.mat_b);
			let half = <T as Scalar>::from_f64(0.5);
			// Symmetrize into mat_b
			for i in 0..self.n {
				for j in i..self.n {
					let avg =
						half * (MatrixOps::get(&ws.mat_b, i, j) + MatrixOps::get(&ws.mat_b, j, i));
					*MatrixOps::get_mut(&mut ws.mat_b, i, j) = avg;
					*MatrixOps::get_mut(&mut ws.mat_b, j, i) = avg;
				}
			}

			self.vec_to_mat(point, &mut ws.mat_a);
			let eigen = DecompositionOps::symmetric_eigen(&ws.mat_a);

			for i in 0..VectorOps::len(&eigen.eigenvalues) {
				let lambda = VectorOps::get(&eigen.eigenvalues, i);
				if <T as Float>::abs(lambda) < <T as Scalar>::from_f64(self.tolerance) {
					// Compute v^T mat_b v via element access (avoid column() alloc)
					let mut vt_sym_v = T::zero();
					for r in 0..self.n {
						let mut row_sum = T::zero();
						for c in 0..self.n {
							row_sum = row_sum
								+ MatrixOps::get(&ws.mat_b, r, c)
									* MatrixOps::get(&eigen.eigenvectors, c, i);
						}
						vt_sym_v = vt_sym_v + MatrixOps::get(&eigen.eigenvectors, r, i) * row_sum;
					}
					if vt_sym_v < T::zero() {
						// mat_b -= vt_sym_v * v * v^T
						for r in 0..self.n {
							for c in 0..self.n {
								let correction = vt_sym_v
									* MatrixOps::get(&eigen.eigenvectors, r, i)
									* MatrixOps::get(&eigen.eigenvectors, c, i);
								*MatrixOps::get_mut(&mut ws.mat_b, r, c) =
									MatrixOps::get(&ws.mat_b, r, c) - correction;
							}
						}
					}
				}
			}

			self.mat_to_vec(&ws.mat_b, result);
		}

		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> Result<T> {
		let dim = self.n * (self.n + 1) / 2;
		if VectorOps::len(u) != dim || VectorOps::len(v) != dim {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}", dim),
				format!("{}", VectorOps::len(u).max(VectorOps::len(v))),
			));
		}

		// Standard Frobenius inner product (with √2 scaling already in vectors)
		Ok(VectorOps::dot(u, v))
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		let dim = self.n * (self.n + 1) / 2;
		if VectorOps::len(point) != dim || VectorOps::len(tangent) != dim {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}", dim),
				format!("{}", VectorOps::len(point).max(VectorOps::len(tangent))),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != dim {
			*result = VectorOps::zeros(dim);
		}

		// Use workspace buffers for vec_to_mat conversions
		self.vec_to_mat(point, &mut ws.mat_a);
		self.vec_to_mat(tangent, &mut ws.mat_b);

		// Simple retraction: project(X + V), compute X + V in-place
		ws.mat_a.add_assign(&ws.mat_b);
		let projected = self.project_to_psd(&ws.mat_a);

		self.mat_to_vec(&projected, result);
		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		let dim = self.n * (self.n + 1) / 2;
		if VectorOps::len(point) != dim || VectorOps::len(other) != dim {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}", dim),
				format!("{}", VectorOps::len(point).max(VectorOps::len(other))),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != dim {
			*result = VectorOps::zeros(dim);
		}

		// Simple approximation: project the difference
		// Compute diff in-place in result, then clone for project_tangent which
		// needs separate input/output references.
		result.copy_from(other);
		result.sub_assign(point);
		let diff = result.clone();
		self.project_tangent(point, &diff, result, ws)
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		// For the Euclidean metric, just project to tangent space
		self.project_tangent(point, euclidean_grad, result, ws)
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random symmetric matrix
		let mut mat = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		for i in 0..self.n {
			for j in i..self.n {
				let val: f64 = normal.sample(&mut rng);
				*MatrixOps::get_mut(&mut mat, i, j) = <T as Scalar>::from_f64(val);
				if i != j {
					*MatrixOps::get_mut(&mut mat, j, i) = <T as Scalar>::from_f64(val);
				}
			}
		}

		// Make it PSD by X = A^T A
		let mut psd = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		psd.gemm_at(T::one(), &mat, &mat, T::zero());

		// Scale to reasonable size (in-place)
		psd.scale_mut(T::one() / <T as Scalar>::from_f64(self.n as f64));

		let dim = self.n * (self.n + 1) / 2;
		if VectorOps::len(result) != dim {
			*result = VectorOps::zeros(dim);
		}
		self.mat_to_vec(&psd, result);
		Ok(())
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		let dim = self.n * (self.n + 1) / 2;
		if VectorOps::len(point) != dim {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}", dim),
				format!("{}", VectorOps::len(point)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != dim {
			*result = VectorOps::zeros(dim);
		}

		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random symmetric matrix
		let mut mat = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		for i in 0..self.n {
			for j in i..self.n {
				let val: f64 = normal.sample(&mut rng);
				*MatrixOps::get_mut(&mut mat, i, j) = <T as Scalar>::from_f64(val);
				if i != j {
					*MatrixOps::get_mut(&mut mat, j, i) = <T as Scalar>::from_f64(val);
				}
			}
		}

		let mut tangent = <linalg::Vec<T> as VectorOps<T>>::zeros(dim);
		self.mat_to_vec(&mat, &mut tangent);
		let mut ws = self.create_workspace(point);
		self.project_tangent(point, &tangent, result, &mut ws)
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		// Frobenius distance: ‖y - x‖² = ‖y‖² + ‖x‖² - 2⟨x,y⟩  (zero alloc)
		let xx = VectorOps::dot(x, x);
		let yy = VectorOps::dot(y, y);
		let xy = VectorOps::dot(x, y);
		let dist_sq = xx + yy - (T::one() + T::one()) * xy;
		Ok(<T as Float>::sqrt(<T as Float>::max(dist_sq, T::zero())))
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		// For flat space with Euclidean metric, parallel transport is identity
		// Just project to ensure it's in tangent space at destination
		self.project_tangent(to, vector, result, ws)
	}

	fn has_exact_exp_log(&self) -> bool {
		false // PSD cone doesn't have simple closed-form exp/log
	}

	fn is_flat(&self) -> bool {
		true // With Euclidean metric, PSD cone is flat
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For PSD cone with Euclidean metric, tangent vectors are symmetric matrices
		// Scaling preserves symmetry
		result.copy_from(tangent);
		result.scale_mut(scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		_point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// In the vectorized representation with √2 scaling, tangent vectors are
		// already symmetric. Sum of symmetric = symmetric. No reprojection needed.
		result.copy_from(v1);
		result.add_assign(v2);
		Ok(())
	}
}
