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
//! use riemannopt_core::linalg::{self, MatrixOps, DecompositionOps, VectorView};
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
//! assert!(VectorView::iter(&eigen.eigenvalues).all(|l| l > 0.0));
//!
//! // Tangent vector (symmetric)
//! let v = MatrixOps::from_fn(3, 3, |i, j| {
//!     (i + j) as f64
//! });
//! let v_sym = MatrixOps::scale_by(&MatrixOps::add(&v, &MatrixOps::transpose_to_owned(&v)), 0.5);
//!
//! // Retraction
//! let mut ws = spd.create_workspace(&p);
//! let mut q = linalg::Mat::<f64>::zeros(3, 3);
//! spd.retract(&p, &v_sym, &mut q, &mut ws)?;
//!
//! // Verify result is SPD
//! assert!(spd.is_point_on_manifold(&q, 1e-10));
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
use std::fmt::{self, Debug};

/// Pre-allocated workspace for SPD manifold operations.
///
/// Contains matrix buffers for intermediate results in `inner_product`,
/// `retract`, and `euclidean_to_riemannian_gradient`, avoiding heap
/// allocations in hot-path methods.
pub struct SPDWorkspace<T: Scalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// n×n buffer A (e.g. P⁻¹U, temp GEMM result)
	pub buf_a: linalg::Mat<T>,
	/// n×n buffer B (e.g. P⁻¹V, second GEMM buffer)
	pub buf_b: linalg::Mat<T>,
	/// n×n buffer C (e.g. clone for aliasing in project_point)
	pub buf_c: linalg::Mat<T>,
}

impl<T: Scalar> Default for SPDWorkspace<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn default() -> Self {
		Self {
			buf_a: MatrixOps::zeros(0, 0),
			buf_b: MatrixOps::zeros(0, 0),
			buf_c: MatrixOps::zeros(0, 0),
		}
	}
}

unsafe impl<T: Scalar> Send for SPDWorkspace<T> where linalg::DefaultBackend: LinAlgBackend<T> {}
unsafe impl<T: Scalar> Sync for SPDWorkspace<T> where linalg::DefaultBackend: LinAlgBackend<T> {}

impl<T: Scalar> Debug for SPDWorkspace<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("SPDWorkspace").finish()
	}
}

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
		if MatrixView::nrows(p) != self.n || MatrixView::ncols(p) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.n,
				MatrixView::nrows(p) * MatrixView::ncols(p),
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

		// Check symmetry (zero-alloc loop)
		let mut sym_err = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let diff = p.get(i, j) - p.get(j, i);
				sym_err = sym_err + diff * diff;
			}
		}
		let symmetry_error = Float::sqrt(sym_err);
		if symmetry_error > self.tolerance {
			return Err(ManifoldError::invalid_point(format!(
				"Matrix not symmetric: ‖P - P^T‖ = {} (tolerance: {})",
				symmetry_error, self.tolerance
			)));
		}

		// Check positive definiteness via eigenvalues
		let eigen = DecompositionOps::symmetric_eigen(p);
		let min_eval = VectorView::iter(&eigen.eigenvalues)
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

		if MatrixView::nrows(v) != self.n || MatrixView::ncols(v) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.n,
				MatrixView::nrows(v) * MatrixView::ncols(v),
			));
		}

		// Check symmetry (zero-alloc loop)
		let mut sym_err = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let diff = v.get(i, j) - v.get(j, i);
				sym_err = sym_err + diff * diff;
			}
		}
		let symmetry_error = Float::sqrt(sym_err);
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
			let x = VectorView::get(&eigen.eigenvalues, i);
			if x <= T::zero() {
				return Err(ManifoldError::numerical_error(
					"Cannot compute square root of non-positive eigenvalue".to_string(),
				));
			}
			*VectorOps::get_mut(&mut sqrt_vals, i) = <T as Float>::sqrt(x);
		}

		// buf = Q * diag(√λ) via backend-optimized column-scaling
		let ev = &eigen.eigenvectors;
		let n = self.n;
		let mut buf = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		buf.scale_columns(ev, &sqrt_vals);
		let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		out.gemm_bt(T::one(), buf.as_view(), ev.as_view(), T::zero());
		Ok(out)
	}

	/// Computes the inverse square root P^{-1/2}.
	pub fn matrix_sqrt_inv(&self, p: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let eigen = DecompositionOps::symmetric_eigen(p);
		let mut sqrt_inv_vals = <linalg::Vec<T> as VectorOps<T>>::zeros(self.n);

		for i in 0..self.n {
			let eval = VectorView::get(&eigen.eigenvalues, i);
			if eval <= self.min_eigenvalue {
				return Err(ManifoldError::numerical_error(
					"Matrix too close to singular for inverse square root".to_string(),
				));
			}
			*VectorOps::get_mut(&mut sqrt_inv_vals, i) = T::one() / <T as Float>::sqrt(eval);
		}

		let ev = &eigen.eigenvectors;
		let n = self.n;
		let mut buf = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		buf.scale_columns(ev, &sqrt_inv_vals);
		let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		out.gemm_bt(T::one(), buf.as_view(), ev.as_view(), T::zero());
		Ok(out)
	}

	/// Computes the matrix logarithm log(P).
	pub fn matrix_log(&self, p: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let eigen = DecompositionOps::symmetric_eigen(p);
		let mut log_vals = <linalg::Vec<T> as VectorOps<T>>::zeros(self.n);

		for i in 0..self.n {
			let eval = VectorView::get(&eigen.eigenvalues, i);
			if eval <= T::zero() {
				return Err(ManifoldError::numerical_error(
					"Cannot compute logarithm of non-positive eigenvalue".to_string(),
				));
			}
			*VectorOps::get_mut(&mut log_vals, i) = <T as Float>::ln(eval);
		}

		let ev = &eigen.eigenvectors;
		let n = self.n;
		let mut buf = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		buf.scale_columns(ev, &log_vals);
		let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		out.gemm_bt(T::one(), buf.as_view(), ev.as_view(), T::zero());
		Ok(out)
	}

	/// Computes the matrix exponential exp(X).
	pub fn matrix_exp(&self, x: &linalg::Mat<T>) -> linalg::Mat<T> {
		let mut eigen = DecompositionOps::symmetric_eigen(x);
		// Transform eigenvalues in-place: λ_i → exp(λ_i)
		for i in 0..self.n {
			let v = VectorView::get(&eigen.eigenvalues, i);
			*VectorOps::get_mut(&mut eigen.eigenvalues, i) = <T as Float>::exp(v);
		}
		let ev = &eigen.eigenvectors;
		let n = self.n;
		let mut buf = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		buf.scale_columns(ev, &eigen.eigenvalues);
		let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		out.gemm_bt(T::one(), buf.as_view(), ev.as_view(), T::zero());
		out
	}

	/// Exponential map at P in direction V.
	///
	/// # Mathematical Formula
	///
	/// exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
	pub fn exp_map(&self, p: &linalg::Mat<T>, v: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let n = self.n;
		let p_sqrt = self.matrix_sqrt(p)?;
		let p_sqrt_inv = self.matrix_sqrt_inv(p)?;

		let mut buf1 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		let mut buf2 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);

		// buf1 = P^{-1/2} V
		buf1.gemm(T::one(), p_sqrt_inv.as_view(), v.as_view(), T::zero());
		// buf2 = buf1 * P^{-1/2} = P^{-1/2} V P^{-1/2}
		buf2.gemm(T::one(), buf1.as_view(), p_sqrt_inv.as_view(), T::zero());

		// Compute exp(buf2)
		let exp_middle = self.matrix_exp(&buf2);

		// buf1 = P^{1/2} exp(middle)
		buf1.gemm(T::one(), p_sqrt.as_view(), exp_middle.as_view(), T::zero());
		// result = buf1 * P^{1/2}
		let mut result = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		result.gemm(T::one(), buf1.as_view(), p_sqrt.as_view(), T::zero());
		Ok(result)
	}

	/// Logarithmic map from P to Q.
	///
	/// # Mathematical Formula
	///
	/// log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
	pub fn log_map(&self, p: &linalg::Mat<T>, q: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let n = self.n;
		let p_sqrt = self.matrix_sqrt(p)?;
		let p_sqrt_inv = self.matrix_sqrt_inv(p)?;

		let mut buf1 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		let mut buf2 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);

		// buf1 = P^{-1/2} Q
		buf1.gemm(T::one(), p_sqrt_inv.as_view(), q.as_view(), T::zero());
		// buf2 = buf1 * P^{-1/2} = P^{-1/2} Q P^{-1/2}
		buf2.gemm(T::one(), buf1.as_view(), p_sqrt_inv.as_view(), T::zero());

		// Compute log(buf2)
		let log_middle = self.matrix_log(&buf2)?;

		// buf1 = P^{1/2} log(middle)
		buf1.gemm(T::one(), p_sqrt.as_view(), log_middle.as_view(), T::zero());
		// result = buf1 * P^{1/2}
		let mut result = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		result.gemm(T::one(), buf1.as_view(), p_sqrt.as_view(), T::zero());
		Ok(result)
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

		let n = self.n;
		match self.metric {
			SPDMetric::AffineInvariant => {
				// d(P,Q) = ‖log(P^{-1/2} Q P^{-1/2})‖_F
				let p_sqrt_inv = self.matrix_sqrt_inv(p)?;
				let mut buf1 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
				let mut buf2 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
				buf1.gemm(T::one(), p_sqrt_inv.as_view(), q.as_view(), T::zero());
				buf2.gemm(T::one(), buf1.as_view(), p_sqrt_inv.as_view(), T::zero());
				let log_middle = self.matrix_log(&buf2)?;
				Ok(MatrixView::norm(&log_middle))
			}
			SPDMetric::LogEuclidean => {
				// d(P,Q) = ‖log(P) - log(Q)‖_F
				let mut log_p = self.matrix_log(p)?;
				let log_q = self.matrix_log(q)?;
				log_p.sub_assign(&log_q);
				Ok(MatrixView::norm(&log_p))
			}
			SPDMetric::BuresWasserstein => {
				// d(P,Q) = [tr(P + Q - 2(P^{1/2} Q P^{1/2})^{1/2})]^{1/2}
				let p_sqrt = self.matrix_sqrt(p)?;
				let mut buf1 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
				let mut buf2 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
				buf1.gemm(T::one(), p_sqrt.as_view(), q.as_view(), T::zero());
				buf2.gemm(T::one(), buf1.as_view(), p_sqrt.as_view(), T::zero());
				let middle_sqrt = self.matrix_sqrt(&buf2)?;
				let trace_term = MatrixView::trace(p) + MatrixView::trace(q)
					- <T as Scalar>::from_f64(2.0) * MatrixView::trace(&middle_sqrt);
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
		_ws: &mut SPDWorkspace<T>,
	) -> Result<linalg::Mat<T>> {
		match self.metric {
			SPDMetric::AffineInvariant => {
				let n = self.n;
				let mut buf1 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
				let mut buf2 = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);

				// buf1 = P * Q
				buf1.gemm(T::one(), p.as_view(), q.as_view(), T::zero());
				let pq_sqrt = self.matrix_sqrt(&buf1)?;
				let p_sqrt_inv = self.matrix_sqrt_inv(p)?;

				// buf1 = (PQ)^{1/2} P^{-1/2} = E
				buf1.gemm(T::one(), pq_sqrt.as_view(), p_sqrt_inv.as_view(), T::zero());

				// buf2 = E * V
				buf2.gemm(T::one(), buf1.as_view(), v.as_view(), T::zero());
				// out = buf2 * E^T
				let mut out = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
				out.gemm_bt(T::one(), buf2.as_view(), buf1.as_view(), T::zero());
				Ok(out)
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
	type Workspace = SPDWorkspace<T>;

	fn create_workspace(&self, _proto_point: &Self::Point) -> Self::Workspace {
		SPDWorkspace {
			buf_a: MatrixOps::zeros(self.n, self.n),
			buf_b: MatrixOps::zeros(self.n, self.n),
			buf_c: MatrixOps::zeros(self.n, self.n),
		}
	}

	fn name(&self) -> &str {
		"SPD"
	}

	fn dimension(&self) -> usize {
		self.n * (self.n + 1) / 2
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if MatrixView::nrows(point) != self.n || MatrixView::ncols(point) != self.n {
			return false;
		}

		// Check symmetry (zero-alloc loop)
		let mut sym_err = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let diff = point.get(i, j) - point.get(j, i);
				sym_err = sym_err + diff * diff;
			}
		}
		if Float::sqrt(sym_err) > tol {
			return false;
		}

		// Check positive definiteness
		let eigen = DecompositionOps::symmetric_eigen(point);
		let min_eval = VectorView::iter(&eigen.eigenvalues)
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
		if MatrixView::nrows(vector) != self.n || MatrixView::ncols(vector) != self.n {
			return false;
		}

		// Tangent space consists of symmetric matrices (zero-alloc loop)
		let mut sym_err = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let diff = vector.get(i, j) - vector.get(j, i);
				sym_err = sym_err + diff * diff;
			}
		}
		Float::sqrt(sym_err) < tol
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		if MatrixView::nrows(point) != self.n || MatrixView::ncols(point) != self.n {
			*result = <linalg::Mat<T> as MatrixOps<T>>::identity(self.n);
			result.scale_mut(self.min_eigenvalue + <T as Scalar>::from_f64(1.0));
			return;
		}

		// Symmetrize into result, then eigendecompose from result
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i..self.n {
				let avg = half * (point.get(i, j) + point.get(j, i));
				*result.get_mut(i, j) = avg;
				*result.get_mut(j, i) = avg;
			}
		}

		// Eigendecomposition (result already holds the symmetrized matrix)
		let eigen = DecompositionOps::symmetric_eigen(result);
		let mut eigenvalues = eigen.eigenvalues;

		// Clamp eigenvalues
		for i in 0..VectorView::len(&eigenvalues) {
			let ev = VectorView::get(&eigenvalues, i);
			if ev <= self.min_eigenvalue || !ev.is_finite() {
				*VectorOps::get_mut(&mut eigenvalues, i) =
					self.min_eigenvalue + <T as Scalar>::from_f64(1e-8);
			}
		}

		// Reconstruct: P = V diag(λ) V^T via backend-optimized column-scaling
		let ev = &eigen.eigenvectors;
		let n = self.n;
		let mut buf = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		buf.scale_columns(ev, &eigenvalues);
		*result = <linalg::Mat<T> as MatrixOps<T>>::zeros(n, n);
		result.gemm_bt(T::one(), buf.as_view(), ev.as_view(), T::zero());

		// Final symmetry enforcement in-place
		for i in 0..self.n {
			for j in i + 1..self.n {
				let avg = half * (result.get(i, j) + result.get(j, i));
				*result.get_mut(i, j) = avg;
				*result.get_mut(j, i) = avg;
			}
		}
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> Result<()> {
		if MatrixView::nrows(point) != self.n
			|| MatrixView::ncols(point) != self.n
			|| MatrixView::nrows(vector) != self.n
			|| MatrixView::ncols(vector) != self.n
		{
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.n,
				MatrixView::nrows(point) * MatrixView::ncols(point),
			));
		}

		// Verify point is on manifold
		self.check_point(point)?;

		// Project to symmetric matrices: result = (V + V^T)/2
		// In-place: copy V, then symmetrize element-by-element.
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i..self.n {
				let avg = half * (vector.get(i, j) + vector.get(j, i));
				*result.get_mut(i, j) = avg;
				*result.get_mut(j, i) = avg;
			}
		}
		Ok(())
	}

	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> Result<T> {
		self.check_tangent(point, u)?;
		self.check_tangent(point, v)?;

		match self.metric {
			SPDMetric::AffineInvariant => {
				// <U,V>_P = tr(P⁻¹U · P⁻¹V)
				// Solve P X = U → buf_a, and P Y = V → buf_b (zero alloc via workspace)
				if !DecompositionOps::cholesky_solve(point, u, &mut ws.buf_a) {
					if !DecompositionOps::inverse(point, &mut ws.buf_c) {
						return Err(ManifoldError::numerical_error(
							"Point matrix not invertible",
						));
					}
					ws.buf_a.gemm(T::one(), ws.buf_c.as_view(), u.as_view(), T::zero());
				}
				if !DecompositionOps::cholesky_solve(point, v, &mut ws.buf_b) {
					if !DecompositionOps::inverse(point, &mut ws.buf_c) {
						return Err(ManifoldError::numerical_error(
							"Point matrix not invertible",
						));
					}
					ws.buf_b.gemm(T::one(), ws.buf_c.as_view(), v.as_view(), T::zero());
				}
				// tr(buf_a · buf_b) via GEMM into buf_c
				ws.buf_c.gemm(T::one(), ws.buf_a.as_view(), ws.buf_b.as_view(), T::zero());
				Ok(MatrixView::trace(&ws.buf_c))
			}
			SPDMetric::LogEuclidean => {
				let mut inner = T::zero();
				for i in 0..self.n {
					for j in 0..self.n {
						inner = inner + u.get(i, j) * v.get(i, j);
					}
				}
				Ok(inner)
			}
			SPDMetric::BuresWasserstein => {
				// <U,V>_P = 1/4 tr(U P⁻¹ V + V P⁻¹ U)
				// P⁻¹ → buf_c (zero alloc via workspace)
				if !DecompositionOps::inverse(point, &mut ws.buf_c) {
					return Err(ManifoldError::numerical_error(
						"Point matrix not invertible",
					));
				}
				// buf_a = U · P⁻¹
				ws.buf_a.gemm(T::one(), u.as_view(), ws.buf_c.as_view(), T::zero());
				// term1 = tr(buf_a · V)
				ws.buf_b.gemm(T::one(), ws.buf_a.as_view(), v.as_view(), T::zero());
				let term1 = MatrixView::trace(&ws.buf_b);
				// buf_a = V · P⁻¹
				ws.buf_a.gemm(T::one(), v.as_view(), ws.buf_c.as_view(), T::zero());
				// term2 = tr(buf_a · U)
				ws.buf_b.gemm(T::one(), ws.buf_a.as_view(), u.as_view(), T::zero());
				let term2 = MatrixView::trace(&ws.buf_b);
				Ok(<T as Scalar>::from_f64(0.25) * (term1 + term2))
			}
		}
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		match self.metric {
			SPDMetric::AffineInvariant => {
				// R_P(V) = sym(P + V + 0.5 · V · P⁻¹V)
				let n = self.n;
				// Solve P X = V → ws.buf_a (zero alloc)
				if !DecompositionOps::cholesky_solve(point, tangent, &mut ws.buf_a) {
					if !DecompositionOps::inverse(point, &mut ws.buf_a) {
						ws.buf_a = <linalg::Mat<T> as MatrixOps<T>>::identity(n);
					}
					// buf_a = P⁻¹ (or identity fallback), need P⁻¹V
					ws.buf_b.copy_from(&ws.buf_a);
					ws.buf_a.gemm(T::one(), ws.buf_b.as_view(), tangent.as_view(), T::zero());
				}
				let half = <T as Scalar>::from_f64(0.5);
				// result = P + V
				result.copy_from(point);
				result.add_assign(tangent);
				// result += 0.5 · V · P⁻¹V  (in-place GEMM)
				result.gemm(half, tangent.as_view(), ws.buf_a.as_view(), T::one());
				// Symmetrize in-place
				for i in 0..n {
					for j in i + 1..n {
						let avg = half * (result.get(i, j) + result.get(j, i));
						*result.get_mut(i, j) = avg;
						*result.get_mut(j, i) = avg;
					}
				}
				Ok(())
			}
			_ => {
				// For other metrics: result = project(P + V)
				result.copy_from(point);
				result.add_assign(tangent);
				ws.buf_a.copy_from(result);
				self.project_point(&ws.buf_a, result);
				Ok(())
			}
		}
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
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
				// For other metrics: result = project(other - point)
				// Copy diff into ws.buf_a to avoid aliasing in project_tangent
				ws.buf_a.copy_from(other);
				ws.buf_a.sub_assign(point);
				// Inline the tangent projection (symmetrize) to avoid borrow conflict
				// with ws. project_tangent just symmetrizes the matrix.
				let half = <T as Scalar>::from_f64(0.5);
				for i in 0..self.n {
					for j in i..self.n {
						let avg = half * (ws.buf_a.get(i, j) + ws.buf_a.get(j, i));
						*result.get_mut(i, j) = avg;
						*result.get_mut(j, i) = avg;
					}
				}
				Ok(())
			}
		}
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		self.check_point(point)?;

		match self.metric {
			SPDMetric::AffineInvariant => {
				// grad^{AI} f(P) = P ∇f(P) P
				// ws.buf_a = P · ∇f via GEMM, then result = ws.buf_a · P via GEMM
				ws.buf_a.gemm(T::one(), point.as_view(), euclidean_grad.as_view(), T::zero());
				result.gemm(T::one(), ws.buf_a.as_view(), point.as_view(), T::zero());
			}
			SPDMetric::LogEuclidean => {
				// grad^{LE} f(P) = P ∇f(P) + ∇f(P) P
				result.gemm(T::one(), point.as_view(), euclidean_grad.as_view(), T::zero());
				result.gemm(T::one(), euclidean_grad.as_view(), point.as_view(), T::one());
			}
			SPDMetric::BuresWasserstein => {
				// grad^{BW} f(P) = ∇f(P)
				result.copy_from(euclidean_grad);
			}
		}
		// Ensure symmetry in-place
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i + 1..self.n {
				let avg = half * (result.get(i, j) + result.get(j, i));
				*result.get_mut(i, j) = avg;
				*result.get_mut(j, i) = avg;
			}
		}
		Ok(())
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> Result<()> {
		// Use identity transport (vector transport by projection) like
		// manopt/pymanopt.  The true parallel transport via matrix square
		// roots is expensive and numerically fragile for large n.
		// Re-project to tangent space at the destination to ensure symmetry.
		self.project_tangent(to, vector, result, ws)
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
		result.gemm_at(T::one(), a.as_view(), a.as_view(), T::zero());
		// Add εI in-place
		for i in 0..self.n {
			*result.get_mut(i, i) = result.get(i, i) + self.min_eigenvalue;
		}
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
		let norm = MatrixView::norm(result);
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
		_point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		result.copy_from(v1);
		result.add_assign(v2);
		// Symmetrize in-place
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i + 1..self.n {
				let avg = half * (result.get(i, j) + result.get(j, i));
				*result.get_mut(i, j) = avg;
				*result.get_mut(j, i) = avg;
			}
		}
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
