//! # Symmetric Positive Definite Manifold S⁺⁺(n)
//!
//! The manifold S⁺⁺(n) of n×n symmetric positive definite (SPD) matrices is fundamental
//! in many areas of mathematics and applications. It provides a rich geometric structure
//! for problems involving covariance matrices, metric tensors, and kernel matrices.
//!
//! ## Mathematical Definition
//!
//! The SPD manifold is formally defined as:
//! ```text
//! S⁺⁺(n) = {P ∈ ℝⁿˣⁿ : P = P^T, x^T P x > 0 ∀x ≠ 0}
//! ```
//!
//! Equivalently, P ∈ S⁺⁺(n) if and only if P is symmetric with all positive eigenvalues.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space at P ∈ S⁺⁺(n) consists of all symmetric matrices:
//! ```text
//! T_P S⁺⁺(n) = {V ∈ ℝⁿˣⁿ : V = V^T} ≅ ℝ^{n(n+1)/2}
//! ```
//!
//! ### Riemannian Metrics
//!
//! #### 1. Affine-Invariant Metric (AI)
//! The canonical metric, invariant under congruence transformations:
//! ```text
//! g_P^{AI}(U, V) = tr(P⁻¹ U P⁻¹ V)
//! ```
//!
//! #### 2. Log-Euclidean Metric (LE)
//! Pulls back the Euclidean metric through the matrix logarithm:
//! ```text
//! g_P^{LE}(U, V) = tr((dlog_P U)^T (dlog_P V))
//! ```
//!
//! #### 3. Bures-Wasserstein Metric (BW)
//! Optimal transport metric on covariance matrices:
//! ```text
//! g_P^{BW}(U, V) = ¼ tr(U P⁻¹ V + V P⁻¹ U)
//! ```
//!
//! ### Exponential and Logarithmic Maps
//!
//! For the affine-invariant metric:
//! ```text
//! exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
//! log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
//! ```
//!
//! ### Parallel Transport
//!
//! Along geodesic from P to Q:
//! ```text
//! Γ_{P→Q}(V) = E V E^T
//! ```
//! where E = (PQ)^{1/2} P^{-1/2}.
//!
//! ## Geodesics and Distance
//!
//! ### Affine-Invariant Distance
//! ```text
//! d_{AI}(P, Q) = ‖log(P^{-1/2} Q P^{-1/2})‖_F
//!              = [∑ᵢ log²(λᵢ)]^{1/2}
//! ```
//! where λᵢ are eigenvalues of P⁻¹Q.
//!
//! ### Log-Euclidean Distance
//! ```text
//! d_{LE}(P, Q) = ‖log(P) - log(Q)‖_F
//! ```
//!
//! ### Bures-Wasserstein Distance
//! ```text
//! d_{BW}(P, Q) = [tr(P + Q - 2(P^{1/2} Q P^{1/2})^{1/2})]^{1/2}
//! ```
//!
//! ## Geometric Properties
//!
//! - **Dimension**: dim(S⁺⁺(n)) = n(n+1)/2
//! - **Sectional curvature**: -1/4 ≤ K ≤ 0 (nonpositive)
//! - **Geodesically complete**: Yes
//! - **Simply connected**: Yes
//! - **Symmetric space**: S⁺⁺(n) ≅ GL(n)/O(n)
//!
//! ## Optimization on S⁺⁺(n)
//!
//! ### Riemannian Gradient
//! For f: S⁺⁺(n) → ℝ with Euclidean gradient ∇f(P):
//! ```text
//! grad^{AI} f(P) = P ∇f(P) P
//! grad^{LE} f(P) = P ∇f(P) + ∇f(P) P
//! ```
//!
//! ### Applications
//!
//! 1. **Covariance estimation**: Regularized covariance matrices
//! 2. **Metric learning**: Learning distance metrics
//! 3. **Diffusion tensor imaging**: Processing brain imaging data
//! 4. **Computer vision**: Region covariance descriptors
//! 5. **Machine learning**: Kernel matrix optimization
//! 6. **Signal processing**: Radar/sonar covariance tracking
//!
//! ## Numerical Considerations
//!
//! This implementation provides:
//! - **Numerical stability** through careful eigendecomposition
//! - **Efficient computation** using matrix square roots
//! - **Robust projection** with eigenvalue thresholding
//! - **Multiple metrics** support for different applications
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::SPD;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{self, MatrixOps, DecompositionOps, VectorOps};
//!
//! // Create SPD manifold S⁺⁺(3)
//! let spd = SPD::<f64>::new(3)?;
//!
//! // Random SPD matrix
//! let mut p = linalg::Mat::<f64>::zeros(3, 3);
//! spd.random_point(&mut p)?;
//!
//! // Verify positive definiteness
//! let eigen = DecompositionOps::symmetric_eigen(&p);
//! assert!(VectorOps::iter(&eigen.eigenvalues).all(|l| l > 0.0));
//!
//! // Tangent vector (symmetric)
//! let v = MatrixOps::from_fn(3, 3, |i, j| {
//!     (i + j) as f64
//! });
//! let v_sym = MatrixOps::scale_by(&MatrixOps::add(&v, &MatrixOps::transpose(&v)), 0.5);
//!
//! // Retraction
//! let mut q = linalg::Mat::<f64>::zeros(3, 3);
//! spd.retract(&p, &v_sym, &mut q)?;
//!
//! // Verify result is SPD
//! assert!(spd.is_point_on_manifold(&q, 1e-10));
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
use std::fmt::{self, Debug};

/// The manifold S⁺⁺(n) of symmetric positive definite matrices.
///
/// This structure represents the space of all n×n symmetric positive definite
/// matrices, equipped with the affine-invariant Riemannian metric by default.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `n ≥ 1`: Matrix dimension must be positive
/// - All points P satisfy P = P^T and have positive eigenvalues
/// - All tangent vectors are symmetric matrices
#[derive(Clone)]
pub struct SPD<T = f64> {
	/// Matrix dimension n
	n: usize,
	/// Minimum eigenvalue threshold for numerical stability
	min_eigenvalue: T,
	/// Numerical tolerance for validations
	tolerance: T,
	/// Metric type to use
	metric: SPDMetric,
}

/// Available Riemannian metrics on the SPD manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SPDMetric {
	/// Affine-invariant metric (default)
	AffineInvariant,
	/// Log-Euclidean metric
	LogEuclidean,
	/// Bures-Wasserstein metric
	BuresWasserstein,
}

impl<T: Scalar> Debug for SPD<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "SPD S⁺⁺({}) with {:?} metric", self.n, self.metric)
	}
}

impl<T: Scalar> SPD<T> {
	/// Creates a new SPD manifold S⁺⁺(n) with affine-invariant metric.
	///
	/// # Arguments
	///
	/// * `n` - Matrix dimension (must be ≥ 1)
	///
	/// # Returns
	///
	/// An SPD manifold with dimension n(n+1)/2.
	///
	/// # Errors
	///
	/// Returns `ManifoldError::InvalidParameter` if n = 0.
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::SPD;
	/// // Create S⁺⁺(3) for 3×3 SPD matrices
	/// let spd = SPD::<f64>::new(3)?;
	///
	/// // S⁺⁺(1) is the positive real line
	/// let spd1 = SPD::<f64>::new(1)?;
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	pub fn new(n: usize) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::invalid_parameter(
				"SPD manifold requires n ≥ 1",
			));
		}
		Ok(Self {
			n,
			min_eigenvalue: <T as Scalar>::from_f64(1e-12),
			tolerance: <T as Scalar>::from_f64(1e-12),
			metric: SPDMetric::AffineInvariant,
		})
	}

	/// Creates an SPD manifold with specified metric.
	///
	/// # Arguments
	///
	/// * `n` - Matrix dimension
	/// * `metric` - Choice of Riemannian metric
	pub fn with_metric(n: usize, metric: SPDMetric) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::invalid_parameter(
				"SPD manifold requires n ≥ 1",
			));
		}
		Ok(Self {
			n,
			min_eigenvalue: <T as Scalar>::from_f64(1e-12),
			tolerance: <T as Scalar>::from_f64(1e-12),
			metric,
		})
	}

	/// Creates an SPD manifold with custom parameters.
	///
	/// # Arguments
	///
	/// * `n` - Matrix dimension
	/// * `min_eigenvalue` - Minimum eigenvalue threshold
	/// * `tolerance` - Numerical tolerance
	/// * `metric` - Choice of Riemannian metric
	pub fn with_parameters(
		n: usize,
		min_eigenvalue: T,
		tolerance: T,
		metric: SPDMetric,
	) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::invalid_parameter(
				"SPD manifold requires n ≥ 1",
			));
		}
		if min_eigenvalue <= T::zero() {
			return Err(ManifoldError::invalid_parameter(
				"Minimum eigenvalue must be positive",
			));
		}
		if tolerance <= T::zero() || tolerance >= T::one() {
			return Err(ManifoldError::invalid_parameter(
				"Tolerance must be in (0, 1)",
			));
		}
		Ok(Self {
			n,
			min_eigenvalue,
			tolerance,
			metric,
		})
	}

	/// Returns the matrix dimension n.
	#[inline]
	pub fn matrix_dim(&self) -> usize {
		self.n
	}

	/// Returns the current metric type.
	#[inline]
	pub fn metric_type(&self) -> SPDMetric {
		self.metric
	}
}

impl<T: Scalar> SPD<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Validates that a matrix is symmetric positive definite.
	///
	/// # Mathematical Checks
	///
	/// 1. Symmetry: ‖P - P^T‖ ≤ tolerance
	/// 2. Positive definiteness: λᵢ(P) > min_eigenvalue ∀i
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If matrix is not n×n
	/// - `NotOnManifold`: If matrix is not symmetric or not positive definite
	pub fn check_point(&self, p: &linalg::Mat<T>) -> Result<()> {
		if MatrixOps::nrows(p) != self.n || MatrixOps::ncols(p) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.n,
				MatrixOps::nrows(p) * MatrixOps::ncols(p),
			));
		}

		// Check for finite values
		let has_non_finite =
			(0..p.ncols()).any(|j| (0..p.nrows()).any(|i| !p.get(i, j).is_finite()));
		if has_non_finite {
			return Err(ManifoldError::invalid_point(
				"Matrix contains non-finite values",
			));
		}

		// Check symmetry
		let symmetry_error = MatrixOps::norm(&MatrixOps::sub(p, &MatrixOps::transpose(p)));
		if symmetry_error > self.tolerance {
			return Err(ManifoldError::invalid_point(format!(
				"Matrix not symmetric: ‖P - P^T‖ = {} (tolerance: {})",
				symmetry_error, self.tolerance
			)));
		}

		// Check positive definiteness via eigenvalues
		let eigen = DecompositionOps::symmetric_eigen(p);
		let min_eval = VectorOps::iter(&eigen.eigenvalues)
			.fold(T::infinity(), |min, val| <T as Float>::min(min, val));

		if min_eval <= self.min_eigenvalue {
			return Err(ManifoldError::invalid_point(format!(
				"Matrix not positive definite: min eigenvalue = {} (threshold: {})",
				min_eval, self.min_eigenvalue
			)));
		}

		Ok(())
	}

	/// Validates that a matrix is symmetric (in tangent space).
	///
	/// # Mathematical Check
	///
	/// Verifies that V = V^T within tolerance.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If dimensions don't match
	/// - `NotInTangentSpace`: If matrix is not symmetric
	pub fn check_tangent(&self, p: &linalg::Mat<T>, v: &linalg::Mat<T>) -> Result<()> {
		self.check_point(p)?;

		if MatrixOps::nrows(v) != self.n || MatrixOps::ncols(v) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.n,
				MatrixOps::nrows(v) * MatrixOps::ncols(v),
			));
		}

		// Check symmetry
		let symmetry_error = MatrixOps::norm(&MatrixOps::sub(v, &MatrixOps::transpose(v)));
		if symmetry_error > self.tolerance {
			return Err(ManifoldError::invalid_tangent(format!(
				"Tangent vector not symmetric: ‖V - V^T‖ = {} (tolerance: {})",
				symmetry_error, self.tolerance
			)));
		}

		Ok(())
	}

	/// Computes the matrix square root P^{1/2}.
	///
	/// # Mathematical Formula
	///
	/// For P = V Λ V^T, returns P^{1/2} = V Λ^{1/2} V^T.
	pub fn matrix_sqrt(&self, p: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let eigen = DecompositionOps::symmetric_eigen(p);
		let mut sqrt_vals = <linalg::Vec<T> as VectorOps<T>>::zeros(self.n);
		for i in 0..self.n {
			let x = VectorOps::get(&eigen.eigenvalues, i);
			if x <= T::zero() {
				return Err(ManifoldError::numerical_error(
					"Cannot compute square root of non-positive eigenvalue".to_string(),
				));
			}
			*VectorOps::get_mut(&mut sqrt_vals, i) = <T as Float>::sqrt(x);
		}

		let sqrt_diag = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&sqrt_vals);
		let ev = &eigen.eigenvectors;
		let temp = MatrixOps::mat_mul(ev, &sqrt_diag);
		Ok(MatrixOps::mat_mul(&temp, &MatrixOps::transpose(ev)))
	}

	/// Computes the inverse square root P^{-1/2}.
	pub fn matrix_sqrt_inv(&self, p: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let eigen = DecompositionOps::symmetric_eigen(p);
		let mut sqrt_inv_vals = <linalg::Vec<T> as VectorOps<T>>::zeros(self.n);

		for i in 0..self.n {
			let eval = VectorOps::get(&eigen.eigenvalues, i);
			if eval <= self.min_eigenvalue {
				return Err(ManifoldError::numerical_error(
					"Matrix too close to singular for inverse square root".to_string(),
				));
			}
			*VectorOps::get_mut(&mut sqrt_inv_vals, i) = T::one() / <T as Float>::sqrt(eval);
		}

		let sqrt_inv_diag = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&sqrt_inv_vals);
		let ev = &eigen.eigenvectors;
		let temp = MatrixOps::mat_mul(ev, &sqrt_inv_diag);
		Ok(MatrixOps::mat_mul(&temp, &MatrixOps::transpose(ev)))
	}

	/// Computes the matrix logarithm log(P).
	pub fn matrix_log(&self, p: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let eigen = DecompositionOps::symmetric_eigen(p);
		let mut log_vals = <linalg::Vec<T> as VectorOps<T>>::zeros(self.n);

		for i in 0..self.n {
			let eval = VectorOps::get(&eigen.eigenvalues, i);
			if eval <= T::zero() {
				return Err(ManifoldError::numerical_error(
					"Cannot compute logarithm of non-positive eigenvalue".to_string(),
				));
			}
			*VectorOps::get_mut(&mut log_vals, i) = <T as Float>::ln(eval);
		}

		let log_diag = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&log_vals);
		let ev = &eigen.eigenvectors;
		let temp = MatrixOps::mat_mul(ev, &log_diag);
		Ok(MatrixOps::mat_mul(&temp, &MatrixOps::transpose(ev)))
	}

	/// Computes the matrix exponential exp(X).
	pub fn matrix_exp(&self, x: &linalg::Mat<T>) -> linalg::Mat<T> {
		let eigen = DecompositionOps::symmetric_eigen(x);
		let exp_vals = VectorOps::map(&eigen.eigenvalues, |v| <T as Float>::exp(v));
		let exp_diag = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&exp_vals);
		let ev = &eigen.eigenvectors;
		let temp = MatrixOps::mat_mul(ev, &exp_diag);
		MatrixOps::mat_mul(&temp, &MatrixOps::transpose(ev))
	}

	/// Exponential map at P in direction V.
	///
	/// # Mathematical Formula
	///
	/// exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
	pub fn exp_map(&self, p: &linalg::Mat<T>, v: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let p_sqrt = self.matrix_sqrt(p)?;
		let p_sqrt_inv = self.matrix_sqrt_inv(p)?;

		// Compute P^{-1/2} V P^{-1/2}
		let temp = MatrixOps::mat_mul(&p_sqrt_inv, v);
		let middle = MatrixOps::mat_mul(&temp, &p_sqrt_inv);

		// Compute exp(middle)
		let exp_middle = self.matrix_exp(&middle);

		// Return P^{1/2} exp(middle) P^{1/2}
		let temp2 = MatrixOps::mat_mul(&p_sqrt, &exp_middle);
		Ok(MatrixOps::mat_mul(&temp2, &p_sqrt))
	}

	/// Logarithmic map from P to Q.
	///
	/// # Mathematical Formula
	///
	/// log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
	pub fn log_map(&self, p: &linalg::Mat<T>, q: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let p_sqrt = self.matrix_sqrt(p)?;
		let p_sqrt_inv = self.matrix_sqrt_inv(p)?;

		// Compute P^{-1/2} Q P^{-1/2}
		let temp = MatrixOps::mat_mul(&p_sqrt_inv, q);
		let middle = MatrixOps::mat_mul(&temp, &p_sqrt_inv);

		// Compute log(middle)
		let log_middle = self.matrix_log(&middle)?;

		// Return P^{1/2} log(middle) P^{1/2}
		let temp2 = MatrixOps::mat_mul(&p_sqrt, &log_middle);
		Ok(MatrixOps::mat_mul(&temp2, &p_sqrt))
	}

	/// Computes distance between SPD matrices using selected metric.
	///
	/// # Mathematical Formulas
	///
	/// - Affine-invariant: d(P,Q) = ‖log(P^{-1/2} Q P^{-1/2})‖_F
	/// - Log-Euclidean: d(P,Q) = ‖log(P) - log(Q)‖_F
	/// - Bures-Wasserstein: d(P,Q) = [tr(P + Q - 2(P^{1/2} Q P^{1/2})^{1/2})]^{1/2}
	pub fn distance(&self, p: &linalg::Mat<T>, q: &linalg::Mat<T>) -> Result<T> {
		self.check_point(p)?;
		self.check_point(q)?;

		match self.metric {
			SPDMetric::AffineInvariant => {
				// d(P,Q) = ‖log(P^{-1/2} Q P^{-1/2})‖_F
				let p_sqrt_inv = self.matrix_sqrt_inv(p)?;
				let temp = MatrixOps::mat_mul(&p_sqrt_inv, q);
				let middle = MatrixOps::mat_mul(&temp, &p_sqrt_inv);
				let log_middle = self.matrix_log(&middle)?;
				Ok(MatrixOps::norm(&log_middle))
			}
			SPDMetric::LogEuclidean => {
				// d(P,Q) = ‖log(P) - log(Q)‖_F
				let log_p = self.matrix_log(p)?;
				let log_q = self.matrix_log(q)?;
				Ok(MatrixOps::norm(&MatrixOps::sub(&log_p, &log_q)))
			}
			SPDMetric::BuresWasserstein => {
				// d(P,Q) = [tr(P + Q - 2(P^{1/2} Q P^{1/2})^{1/2})]^{1/2}
				let p_sqrt = self.matrix_sqrt(p)?;
				let temp = MatrixOps::mat_mul(&p_sqrt, q);
				let middle = MatrixOps::mat_mul(&temp, &p_sqrt);
				let middle_sqrt = self.matrix_sqrt(&middle)?;
				let trace_term = MatrixOps::trace(p) + MatrixOps::trace(q)
					- <T as Scalar>::from_f64(2.0) * MatrixOps::trace(&middle_sqrt);
				Ok(<T as Float>::sqrt(trace_term))
			}
		}
	}

	/// Parallel transport along geodesic from P to Q.
	///
	/// # Mathematical Formula
	///
	/// Γ_{P→Q}(V) = E V E^T where E = (PQ)^{1/2} P^{-1/2}
	pub fn parallel_transport(
		&self,
		p: &linalg::Mat<T>,
		q: &linalg::Mat<T>,
		v: &linalg::Mat<T>,
	) -> Result<linalg::Mat<T>> {
		match self.metric {
			SPDMetric::AffineInvariant => {
				// E = (PQ)^{1/2} P^{-1/2}
				let pq = MatrixOps::mat_mul(p, q);
				let pq_sqrt = self.matrix_sqrt(&pq)?;
				let p_sqrt_inv = self.matrix_sqrt_inv(p)?;
				let e = MatrixOps::mat_mul(&pq_sqrt, &p_sqrt_inv);

				// Transport: E V E^T
				let temp = MatrixOps::mat_mul(&e, v);
				Ok(MatrixOps::mat_mul(&temp, &MatrixOps::transpose(&e)))
			}
			_ => {
				// For other metrics, use identity transport (approximation)
				Ok(v.clone())
			}
		}
	}
}

impl<T> Manifold<T> for SPD<T>
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Mat<T>;
	type TangentVector = linalg::Mat<T>;

	fn name(&self) -> &str {
		"SPD"
	}

	fn dimension(&self) -> usize {
		self.n * (self.n + 1) / 2
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if MatrixOps::nrows(point) != self.n || MatrixOps::ncols(point) != self.n {
			return false;
		}

		// Check symmetry
		let symmetry_error = MatrixOps::norm(&MatrixOps::sub(point, &MatrixOps::transpose(point)));
		if symmetry_error > tol {
			return false;
		}

		// Check positive definiteness
		let eigen = DecompositionOps::symmetric_eigen(point);
		let min_eval = VectorOps::iter(&eigen.eigenvalues)
			.fold(T::infinity(), |min, val| <T as Float>::min(min, val));

		min_eval > self.min_eigenvalue && min_eval.is_finite()
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
		if MatrixOps::nrows(vector) != self.n || MatrixOps::ncols(vector) != self.n {
			return false;
		}

		// Tangent space consists of symmetric matrices
		let symmetry_error =
			MatrixOps::norm(&MatrixOps::sub(vector, &MatrixOps::transpose(vector)));
		symmetry_error < tol
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		if MatrixOps::nrows(point) != self.n || MatrixOps::ncols(point) != self.n {
			*result = <linalg::Mat<T> as MatrixOps<T>>::identity(self.n);
			result.scale_mut(self.min_eigenvalue + <T as Scalar>::from_f64(1.0));
			return;
		}

		// Ensure symmetry
		let symmetric = MatrixOps::scale_by(
			&MatrixOps::add(point, &MatrixOps::transpose(point)),
			<T as Scalar>::from_f64(0.5),
		);

		// Eigendecomposition
		let eigen = DecompositionOps::symmetric_eigen(&symmetric);
		let mut eigenvalues = eigen.eigenvalues.clone();

		// Clamp eigenvalues
		for i in 0..VectorOps::len(&eigenvalues) {
			let ev = VectorOps::get(&eigenvalues, i);
			if ev <= self.min_eigenvalue || !ev.is_finite() {
				*VectorOps::get_mut(&mut eigenvalues, i) =
					self.min_eigenvalue + <T as Scalar>::from_f64(1e-8);
			}
		}

		// Reconstruct: P = V Λ V^T
		let lambda_diag = <linalg::Mat<T> as MatrixOps<T>>::from_diagonal(&eigenvalues);
		let ev = &eigen.eigenvectors;
		let temp = MatrixOps::mat_mul(ev, &lambda_diag);
		*result = MatrixOps::mat_mul(&temp, &MatrixOps::transpose(ev));

		// Final symmetry enforcement
		let sym = MatrixOps::scale_by(
			&MatrixOps::add(result, &MatrixOps::transpose(result)),
			<T as Scalar>::from_f64(0.5),
		);
		result.copy_from(&sym);
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		if MatrixOps::nrows(point) != self.n
			|| MatrixOps::ncols(point) != self.n
			|| MatrixOps::nrows(vector) != self.n
			|| MatrixOps::ncols(vector) != self.n
		{
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.n,
				MatrixOps::nrows(point) * MatrixOps::ncols(point),
			));
		}

		// Verify point is on manifold
		self.check_point(point)?;

		// Project to symmetric matrices: (V + V^T)/2
		*result = MatrixOps::scale_by(
			&MatrixOps::add(vector, &MatrixOps::transpose(vector)),
			<T as Scalar>::from_f64(0.5),
		);
		Ok(())
	}

	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		self.check_tangent(point, u)?;
		self.check_tangent(point, v)?;

		match self.metric {
			SPDMetric::AffineInvariant => {
				// <U,V>_P = tr(P^{-1} U P^{-1} V)
				let p_inv = DecompositionOps::try_inverse(point).ok_or_else(|| {
					ManifoldError::numerical_error("Point matrix not invertible".to_string())
				})?;
				let p_inv_u = MatrixOps::mat_mul(&p_inv, u);
				let p_inv_v = MatrixOps::mat_mul(&p_inv, v);
				Ok(MatrixOps::trace(&MatrixOps::mat_mul(&p_inv_u, &p_inv_v)))
			}
			SPDMetric::LogEuclidean => {
				// For Log-Euclidean, use Euclidean metric in log space
				// This is approximated by Frobenius inner product
				Ok(MatrixOps::trace(&MatrixOps::mat_mul(
					&MatrixOps::transpose(u),
					v,
				)))
			}
			SPDMetric::BuresWasserstein => {
				// <U,V>_P = 1/4 tr(U P^{-1} V + V P^{-1} U)
				let p_inv = DecompositionOps::try_inverse(point).ok_or_else(|| {
					ManifoldError::numerical_error("Point matrix not invertible".to_string())
				})?;
				let u_pinv = MatrixOps::mat_mul(u, &p_inv);
				let v_pinv = MatrixOps::mat_mul(v, &p_inv);
				let term1 = MatrixOps::trace(&MatrixOps::mat_mul(&u_pinv, v));
				let term2 = MatrixOps::trace(&MatrixOps::mat_mul(&v_pinv, u));
				Ok(<T as Scalar>::from_f64(0.25) * (term1 + term2))
			}
		}
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		match self.metric {
			SPDMetric::AffineInvariant => {
				// Second-order retraction (manopt/pymanopt default):
				// R_P(V) = sym(P + V + 0.5 * V * P^{-1} * V)
				//
				// This avoids the expensive eigendecomposition of exp_map
				// and uses a single linear solve P\V instead.
				let n = self.n;
				let p_inv_v =
					DecompositionOps::cholesky_solve(point, tangent).unwrap_or_else(|| {
						// Fallback: use pseudoinverse if Cholesky fails
						let fallback_inv = DecompositionOps::try_inverse(point)
							.unwrap_or_else(|| <linalg::Mat<T> as MatrixOps<T>>::identity(n));
						MatrixOps::mat_mul(&fallback_inv, tangent)
					});
				let half = <T as Scalar>::from_f64(0.5);
				let v_pinv_v = MatrixOps::scale_by(&MatrixOps::mat_mul(tangent, &p_inv_v), half);
				let raw = MatrixOps::add(&MatrixOps::add(point, tangent), &v_pinv_v);
				// Symmetrize for numerical stability
				let sym =
					MatrixOps::scale_by(&MatrixOps::add(&raw, &MatrixOps::transpose(&raw)), half);
				result.copy_from(&sym);
				Ok(())
			}
			_ => {
				// For other metrics, use projection-based retraction
				let sum = MatrixOps::add(point, tangent);
				self.project_point(&sum, result);
				Ok(())
			}
		}
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		self.check_point(point)?;
		self.check_point(other)?;

		match self.metric {
			SPDMetric::AffineInvariant => {
				// Use exact logarithmic map
				let log_result = self.log_map(point, other)?;
				result.copy_from(&log_result);
				Ok(())
			}
			_ => {
				// For other metrics, use approximation
				let diff = MatrixOps::sub(other, point);
				self.project_tangent(point, &diff, result)
			}
		}
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		self.check_point(point)?;

		let grad_result = match self.metric {
			SPDMetric::AffineInvariant => {
				// grad^{AI} f(P) = P ∇f(P) P
				let temp = MatrixOps::mat_mul(point, euclidean_grad);
				MatrixOps::mat_mul(&temp, point)
			}
			SPDMetric::LogEuclidean => {
				// grad^{LE} f(P) = P ∇f(P) + ∇f(P) P
				let term1 = MatrixOps::mat_mul(point, euclidean_grad);
				let term2 = MatrixOps::mat_mul(euclidean_grad, point);
				MatrixOps::add(&term1, &term2)
			}
			SPDMetric::BuresWasserstein => {
				// grad^{BW} f(P) = ∇f(P)
				euclidean_grad.clone()
			}
		};

		// Ensure symmetry
		self.project_tangent(point, &grad_result, result)
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Use identity transport (vector transport by projection) like
		// manopt/pymanopt.  The true parallel transport via matrix square
		// roots is expensive and numerically fragile for large n.
		// Re-project to tangent space at the destination to ensure symmetry.
		self.project_tangent(to, vector, result)
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random matrix
		let mut a = <linalg::Mat<T> as MatrixOps<T>>::zeros(self.n, self.n);
		for i in 0..self.n {
			for j in 0..self.n {
				*MatrixOps::get_mut(&mut a, i, j) =
					<T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// Make SPD: P = A^T A + εI
		let ata = MatrixOps::mat_mul(&MatrixOps::transpose(&a), &a);
		let identity = <linalg::Mat<T> as MatrixOps<T>>::identity(self.n);

		*result = MatrixOps::add(&ata, &MatrixOps::scale_by(&identity, self.min_eigenvalue));
		Ok(())
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		self.check_point(point)?;

		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random symmetric matrix
		for i in 0..self.n {
			for j in i..self.n {
				let val = <T as Scalar>::from_f64(normal.sample(&mut rng));
				*MatrixOps::get_mut(result, i, j) = val;
				if i != j {
					*MatrixOps::get_mut(result, j, i) = val;
				}
			}
		}

		// Normalize
		let norm = MatrixOps::norm(result);
		if norm > <T as Scalar>::from_f64(1e-16) {
			result.scale_mut(T::one() / norm);
		}

		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		self.distance(x, y)
	}

	fn has_exact_exp_log(&self) -> bool {
		matches!(self.metric, SPDMetric::AffineInvariant)
	}

	fn is_flat(&self) -> bool {
		false // SPD has negative curvature
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For SPD manifold, tangent vectors are symmetric matrices
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
		// but we project for numerical stability
		self.project_tangent(point, temp, result)?;

		Ok(())
	}
}

// NOTE: MatrixManifold<T> impl is deferred until MatrixManifold
// itself is migrated from DMatrix<T> to linalg::Mat<T>.
// When linalg::Mat<T> == DMatrix<T> (nalgebra backend), the impl
// would be:
//   impl<T: Scalar + Float> MatrixManifold<T> for SPD<T>
//   where linalg::DefaultBackend: LinAlgBackend<T>,
//   { fn matrix_dims(&self) -> (usize, usize) { (self.n, self.n) } }
