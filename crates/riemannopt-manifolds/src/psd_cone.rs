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
//! let x_mat = psd_cone.vector_to_matrix::<f64>(&x);
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
	pub fn matrix_to_vector<T: Scalar>(&self, mat: &linalg::Mat<T>) -> linalg::Vec<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.mat_to_vec(mat)
	}

	/// Converts a vector to its symmetric matrix form.
	pub fn vector_to_matrix<T: Scalar>(&self, vec: &linalg::Vec<T>) -> linalg::Mat<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.vec_to_mat(vec)
	}

	/// Convert a symmetric matrix to vector form (upper triangular part)
	fn mat_to_vec<T: Scalar>(&self, mat: &linalg::Mat<T>) -> linalg::Vec<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		let n = self.n;
		let dim = n * (n + 1) / 2;
		let mut vec = <linalg::Vec<T> as VectorOps<T>>::zeros(dim);
		let mut idx = 0;

		for i in 0..n {
			for j in i..n {
				if i == j {
					*VectorOps::get_mut(&mut vec, idx) = MatrixOps::get(mat, i, j);
				} else {
					// Store off-diagonal elements with sqrt(2) scaling for proper inner product
					*VectorOps::get_mut(&mut vec, idx) = MatrixOps::get(mat, i, j)
						* <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
				}
				idx += 1;
			}
		}

		vec
	}

	/// Convert vector to symmetric matrix
	fn vec_to_mat<T: Scalar>(&self, vec: &linalg::Vec<T>) -> linalg::Mat<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		let n = self.n;
		let mut mat = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		let mut idx = 0;

		for i in 0..n {
			for j in i..n {
				if i == j {
					*MatrixOps::get_mut(&mut mat, i, j) = VectorOps::get(vec, idx);
				} else {
					// Unscale off-diagonal elements
					let val = VectorOps::get(vec, idx)
						/ <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
					*MatrixOps::get_mut(&mut mat, i, j) = val;
					*MatrixOps::get_mut(&mut mat, j, i) = val;
				}
				idx += 1;
			}
		}

		mat
	}

	/// Project a matrix to the PSD cone
	fn project_to_psd<T: Scalar>(&self, mat: &linalg::Mat<T>) -> linalg::Mat<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		// Symmetrize first
		let sym = MatrixOps::scale_by(
			&MatrixOps::add(mat, &MatrixOps::transpose(mat)),
			<T as Scalar>::from_f64(0.5),
		);

		// Eigendecomposition
		let eigen = DecompositionOps::symmetric_eigen(&sym);
		let mut eigenvalues = eigen.eigenvalues.clone();

		// Project eigenvalues to non-negative
		for i in 0..VectorOps::len(&eigenvalues) {
			if VectorOps::get(&eigenvalues, i) < T::zero() {
				*VectorOps::get_mut(&mut eigenvalues, i) = T::zero();
			}
		}

		// Reconstruct
		let q = &eigen.eigenvectors;
		let d = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&eigenvalues);
		let temp = MatrixOps::mat_mul(q, &d);
		MatrixOps::mat_mul(&temp, &MatrixOps::transpose(q))
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
		let diff = MatrixOps::sub(mat, &projected);
		Ok(MatrixOps::norm(&diff))
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

		// Symmetrize first
		let sym = MatrixOps::scale_by(
			&MatrixOps::add(mat, &MatrixOps::transpose(mat)),
			<T as Scalar>::from_f64(0.5),
		);
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

		let mat = self.vec_to_mat(point);
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
		let mat = self.vec_to_mat(vector);

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
		let x_mat = self.vec_to_mat(point);
		let eigen = DecompositionOps::symmetric_eigen(&x_mat);

		// Find near-zero eigenvalues (boundary)
		for i in 0..VectorOps::len(&eigen.eigenvalues) {
			let lambda = VectorOps::get(&eigen.eigenvalues, i);
			if <T as Float>::abs(lambda) < <T as Scalar>::from_f64(self.tolerance) {
				// For zero eigenvalue, check v^T V v >= 0 where v is eigenvector
				let v = MatrixOps::column(&eigen.eigenvectors, i);
				let mat_v = MatrixOps::mat_vec(&mat, &v);
				let vt_mat_v = VectorOps::dot(&v, &mat_v);
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

		let mat = if VectorOps::len(point) == self.n * self.n {
			// If given as full matrix, reshape
			<linalg::Mat<T> as MatrixOps<T>>::from_column_slice(
				self.n,
				self.n,
				VectorOps::as_slice(point),
			)
		} else if VectorOps::len(point) == dim {
			self.vec_to_mat(point)
		} else {
			// Wrong size, create zero matrix
			<linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n)
		};

		let projected = self.project_to_psd(&mat);
		let projected_vec = self.mat_to_vec(&projected);
		result.copy_from(&projected_vec);
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		let dim = self.n * (self.n + 1) / 2;
		// Ensure result has correct size
		if VectorOps::len(result) != dim {
			*result = VectorOps::zeros(dim);
		}

		// Check dimensions
		if VectorOps::len(point) != dim || VectorOps::len(vector) != dim {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}", dim),
				format!("{}", VectorOps::len(point).max(VectorOps::len(vector))),
			));
		}

		// For PSD cone, tangent projection is symmetrization
		let mat = self.vec_to_mat(vector);
		let sym = MatrixOps::scale_by(
			&MatrixOps::add(&mat, &MatrixOps::transpose(&mat)),
			<T as Scalar>::from_f64(0.5),
		);

		// For boundary points, additional projection may be needed
		if !self.strict {
			let x_mat = self.vec_to_mat(point);
			let eigen = DecompositionOps::symmetric_eigen(&x_mat);

			// Project out components that violate cone constraint
			let mut proj_mat = sym.clone();
			for i in 0..VectorOps::len(&eigen.eigenvalues) {
				let lambda = VectorOps::get(&eigen.eigenvalues, i);
				if <T as Float>::abs(lambda) < <T as Scalar>::from_f64(self.tolerance) {
					// Zero eigenvalue: project to ensure v^T V v >= 0
					let v = MatrixOps::column(&eigen.eigenvectors, i);
					let sym_v = MatrixOps::mat_vec(&sym, &v);
					let vt_mat_v = VectorOps::dot(&v, &sym_v);
					if vt_mat_v < T::zero() {
						// Project out the violating component: vvt * scalar
						let vvt =
							<linalg::Mat<T> as MatrixOps<T>>::from_fn(self.n, self.n, |r, c| {
								VectorOps::get(&v, r) * VectorOps::get(&v, c)
							});
						let correction = MatrixOps::scale_by(&vvt, vt_mat_v);
						proj_mat.sub_assign(&correction);
					}
				}
			}

			let proj_vec = self.mat_to_vec(&proj_mat);
			result.copy_from(&proj_vec);
		} else {
			let sym_vec = self.mat_to_vec(&sym);
			result.copy_from(&sym_vec);
		}

		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
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

		let x_mat = self.vec_to_mat(point);
		let v_mat = self.vec_to_mat(tangent);

		// Simple retraction: project(X + V)
		let new_mat = MatrixOps::add(&x_mat, &v_mat);
		let projected = self.project_to_psd(&new_mat);

		let projected_vec = self.mat_to_vec(&projected);
		result.copy_from(&projected_vec);
		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
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
		let diff = VectorOps::sub(other, point);
		self.project_tangent(point, &diff, result)
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For the Euclidean metric, just project to tangent space
		self.project_tangent(point, euclidean_grad, result)
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
		let psd = MatrixOps::mat_mul(&MatrixOps::transpose(&mat), &mat);

		// Scale to reasonable size
		let psd_scaled =
			MatrixOps::scale_by(&psd, T::one() / <T as Scalar>::from_f64(self.n as f64));

		*result = self.mat_to_vec(&psd_scaled);
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

		let tangent = self.mat_to_vec(&mat);
		self.project_tangent(point, &tangent, result)
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		let dim = self.n * (self.n + 1) / 2;
		if VectorOps::len(x) != dim || VectorOps::len(y) != dim {
			return Err(ManifoldError::dimension_mismatch(
				format!("{}", dim),
				format!("{}", VectorOps::len(x).max(VectorOps::len(y))),
			));
		}

		// Frobenius distance
		let diff = VectorOps::sub(y, x);
		Ok(VectorOps::norm(&diff))
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For flat space with Euclidean metric, parallel transport is identity
		// Just project to ensure it's in tangent space at destination
		self.project_tangent(to, vector, result)
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

		// The sum should already be symmetric if v1 and v2 are,
		// but we project for numerical stability and boundary constraints
		self.project_tangent(point, temp, result)?;

		Ok(())
	}
}
