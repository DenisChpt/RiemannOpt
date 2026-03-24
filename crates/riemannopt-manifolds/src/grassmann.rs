//! # Grassmann Manifold Gr(n,p)
//!
//! The Grassmann manifold Gr(n,p) is the space of all p-dimensional linear
//! subspaces of ℝⁿ. It provides a geometric framework for problems involving
//! subspace optimization, dimensionality reduction, and invariant subspace computation.
//!
//! ## Mathematical Definition
//!
//! The Grassmann manifold is formally defined as:
//! ```text
//! Gr(n,p) = {[Y] : Y ∈ ℝⁿˣᵖ, Y^T Y = I_p}
//! ```
//! where [Y] denotes the equivalence class of matrices with the same column space.
//!
//! Two matrices Y₁ and Y₂ represent the same point if Y₁ = Y₂Q for some Q ∈ O(p).
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space at [Y] consists of matrices orthogonal to the subspace:
//! ```text
//! T_{[Y]} Gr(n,p) = {Z ∈ ℝⁿˣᵖ : Y^T Z = 0}
//! ```
//! This is the horizontal space in the principal fiber bundle St(n,p) → Gr(n,p).
//!
//! ### Riemannian Metric
//! The canonical metric is inherited from the embedding in ℝⁿˣᵖ:
//! ```text
//! g_{[Y]}(Z₁, Z₂) = tr(Z₁^T Z₂)
//! ```
//!
//! ### Normal Space
//! The normal space (vertical space) consists of matrices of the form YS:
//! ```text
//! N_{[Y]} Gr(n,p) = {YS : S ∈ ℝᵖˣᵖ}
//! ```
//!
//! ### Projection Operators
//! - **Horizontal projection**: P_h(W) = W - Y(Y^T W) = (I - YY^T)W
//! - **Vertical projection**: P_v(W) = Y(Y^T W)
//!
//! ## Retractions and Exponential Map
//!
//! ### QR-based Retraction
//! The most efficient retraction uses QR decomposition:
//! ```text
//! R_Y(Z) = qf(Y + Z)
//! ```
//! where qf(·) extracts the Q factor from thin QR decomposition.
//!
//! ### SVD-based Retraction
//! A more stable retraction uses SVD:
//! ```text
//! (Y + Z) = UΣV^T, then R_Y(Z) = UV^T
//! ```
//!
//! ### Exponential Map
//! The exponential map involves matrix exponentials:
//! ```text
//! exp_{[Y]}(Z) = [YV cos(Σ) + U sin(Σ)]V^T
//! ```
//! where Z = UΣV^T is the compact SVD.
//!
//! ## Parallel Transport
//!
//! Parallel transport along geodesics can be computed using:
//! ```text
//! Γ_{[Y]→[Ỹ]}(Z) = (I - ỸỸ^T)ZU
//! ```
//! where U comes from the SVD of Ỹ^T Y.
//!
//! ## Geometric Invariants
//!
//! - **Dimension**: dim(Gr(n,p)) = p(n-p)
//! - **Sectional curvature**: 0 ≤ K ≤ 1
//! - **Geodesically complete**: Yes
//! - **Compact**: Yes
//! - **Simply connected**: Yes if p = 1 or p = n-1; No otherwise
//!
//! ## Principal Angles and Distance
//!
//! The distance between subspaces is measured via principal angles θᵢ:
//! ```text
//! d([Y₁], [Y₂]) = ‖θ‖₂
//! ```
//! where θᵢ = arccos(σᵢ) and σᵢ are singular values of Y₁^T Y₂.
//!
//! ## Optimization on Grassmann
//!
//! ### Riemannian Gradient
//! For f: Gr(n,p) → ℝ with Euclidean gradient ∇f(Y):
//! ```text
//! grad f([Y]) = (I - YY^T)∇f(Y)
//! ```
//!
//! ### Applications
//!
//! 1. **Principal Component Analysis**: Finding dominant eigenspaces
//! 2. **Subspace tracking**: Adaptive signal processing
//! 3. **Computer vision**: Multi-view geometry, face recognition
//! 4. **Model reduction**: Finding invariant subspaces
//! 5. **Machine learning**: Metric learning, domain adaptation
//! 6. **Quantum computing**: Optimization over pure state subspaces
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Numerical stability** through careful orthogonalization
//! - **Efficiency** via optimized BLAS operations
//! - **Robustness** to rank-deficient matrices
//! - **Invariance** under orthogonal transformations of representatives
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Grassmann;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{MatrixOps, DecompositionOps};
//!
//! // Create Grassmann manifold Gr(5,2)
//! let grassmann = Grassmann::<f64>::new(5, 2)?;
//!
//! // Random point (2D subspace of ℝ⁵)
//! let mut y = riemannopt_core::linalg::Mat::<f64>::zeros(5, 2);
//! grassmann.random_point(&mut y)?;
//!
//! // Verify orthonormality
//! let yt = MatrixOps::transpose(&y);
//! let yty = yt.mat_mul(&y);
//! let identity = <riemannopt_core::linalg::Mat<f64> as MatrixOps<f64>>::identity(2);
//! assert!(yty.sub(&identity).norm() < 1e-14);
//!
//! // Tangent vector (orthogonal to subspace)
//! let z = riemannopt_core::linalg::Mat::<f64>::from_fn(5, 2, |i, j| 0.1 * (i as f64 - j as f64));
//! let mut z_horizontal = z.clone();
//! grassmann.project_tangent(&y, &z, &mut z_horizontal)?;
//!
//! // Verify horizontality: Y^T Z = 0
//! let ytz = yt.mat_mul(&z_horizontal);
//! assert!(ytz.norm() < 1e-14);
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

/// The Grassmann manifold Gr(n,p) of p-dimensional subspaces in ℝⁿ.
///
/// This structure represents the space of all p-dimensional linear subspaces
/// of n-dimensional Euclidean space, equipped with the canonical Riemannian
/// metric inherited from the Stiefel manifold.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `1 ≤ p ≤ n-1`: Dimension constraints (p=0 or p=n give trivial cases)
/// - Points are represented by n×p matrices with orthonormal columns
/// - The manifold structure is invariant under the O(p) action on the right
#[derive(Clone)]
pub struct Grassmann<T = f64> {
	/// Ambient dimension n
	n: usize,
	/// Subspace dimension p
	p: usize,
	/// Numerical tolerance for validations
	tolerance: T,
}

impl<T: Scalar> Debug for Grassmann<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Grassmann Gr({}, {})", self.n, self.p)
	}
}

impl<T: Scalar> Grassmann<T> {
	/// Creates a new Grassmann manifold Gr(n,p).
	///
	/// # Arguments
	///
	/// * `n` - Ambient dimension (must satisfy p < n)
	/// * `p` - Subspace dimension (must satisfy 0 < p < n)
	///
	/// # Returns
	///
	/// A Grassmann manifold with dimension p(n-p).
	///
	/// # Errors
	///
	/// Returns `ManifoldError::InvalidParameter` if:
	/// - `p = 0` (empty subspace)
	/// - `p ≥ n` (subspace dimension exceeds ambient dimension)
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::Grassmann;
	/// // Create Gr(5,2) - 2D subspaces in ℝ⁵
	/// let gr52 = Grassmann::<f64>::new(5, 2)?;
	///
	/// // Gr(n,1) is the projective space ℝP^{n-1}
	/// let projective = Grassmann::<f64>::new(4, 1)?;
	///
	/// // Gr(n,n-1) is also isomorphic to ℝP^{n-1}
	/// let dual_projective = Grassmann::<f64>::new(4, 3)?;
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	pub fn new(n: usize, p: usize) -> Result<Self> {
		if p == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Grassmann manifold requires p > 0",
			));
		}
		if p >= n {
			return Err(ManifoldError::invalid_parameter(format!(
				"Grassmann manifold Gr(n,p) requires p < n, got n={}, p={}",
				n, p
			)));
		}
		Ok(Self {
			n,
			p,
			tolerance: <T as Scalar>::from_f64(1e-10),
		})
	}

	/// Creates a Grassmann manifold with custom numerical tolerance.
	///
	/// # Arguments
	///
	/// * `n` - Ambient dimension
	/// * `p` - Subspace dimension
	/// * `tolerance` - Numerical tolerance for validations
	pub fn with_tolerance(n: usize, p: usize, tolerance: T) -> Result<Self> {
		if p == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Grassmann manifold requires p > 0",
			));
		}
		if p >= n {
			return Err(ManifoldError::invalid_parameter(format!(
				"Grassmann manifold Gr(n,p) requires p < n, got n={}, p={}",
				n, p
			)));
		}
		if tolerance <= T::zero() || tolerance >= T::one() {
			return Err(ManifoldError::invalid_parameter(
				"Tolerance must be in (0, 1)",
			));
		}
		Ok(Self { n, p, tolerance })
	}

	/// Returns the ambient dimension n.
	#[inline]
	pub fn ambient_dim(&self) -> usize {
		self.n
	}

	/// Returns the subspace dimension p.
	#[inline]
	pub fn subspace_dim(&self) -> usize {
		self.p
	}
}

impl<T> Grassmann<T>
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Validates that a matrix represents a point on Grassmann.
	///
	/// # Mathematical Check
	///
	/// Verifies that Y^T Y = I_p within numerical tolerance.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If matrix dimensions don't match (n,p)
	/// - `NotOnManifold`: If ‖Y^T Y - I_p‖ > tolerance
	pub fn check_point(&self, y: &linalg::Mat<T>) -> Result<()> {
		if y.nrows() != self.n || y.ncols() != self.p {
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.p,
				y.nrows() * y.ncols(),
			));
		}

		// Check orthonormality: Y^T Y = I
		let mut yty = linalg::Mat::<T>::zeros(self.p, self.p);
		yty.gemm_at(T::one(), y, y, T::zero());
		let identity = linalg::Mat::<T>::identity(self.p);
		let constraint_error = yty.sub(&identity).norm();

		if constraint_error > self.tolerance {
			return Err(ManifoldError::invalid_point(format!(
				"Orthonormality violated: ‖Y^T Y - I‖ = {} (tolerance: {})",
				constraint_error, self.tolerance
			)));
		}

		Ok(())
	}

	/// Validates that a matrix lies in the horizontal space at Y.
	///
	/// # Mathematical Check
	///
	/// Verifies that Y^T Z = 0 (horizontality condition).
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If dimensions don't match
	/// - `NotOnManifold`: If Y is not on Grassmann
	/// - `NotInTangentSpace`: If ‖Y^T Z‖ > tolerance
	pub fn check_tangent(&self, y: &linalg::Mat<T>, z: &linalg::Mat<T>) -> Result<()> {
		self.check_point(y)?;

		if z.nrows() != self.n || z.ncols() != self.p {
			return Err(ManifoldError::dimension_mismatch(
				self.n * self.p,
				z.nrows() * z.ncols(),
			));
		}

		// Check horizontality: Y^T Z = 0
		let mut ytz = linalg::Mat::<T>::zeros(self.p, self.p);
		ytz.gemm_at(T::one(), y, z, T::zero());
		let horizontal_error = ytz.norm();

		if horizontal_error > self.tolerance {
			return Err(ManifoldError::invalid_tangent(format!(
				"Horizontality violated: ‖Y^T Z‖ = {} (tolerance: {})",
				horizontal_error, self.tolerance
			)));
		}

		Ok(())
	}

	/// Performs QR-based retraction.
	///
	/// # Mathematical Formula
	///
	/// R_Y(Z) = qf(Y + Z) where qf extracts the Q factor from thin QR.
	///
	/// # Arguments
	///
	/// * `y` - Point on Grassmann manifold
	/// * `z` - Tangent vector at y (horizontal)
	///
	/// # Returns
	///
	/// The retracted point R_Y(Z) on the manifold.
	pub fn qr_retraction(&self, y: &linalg::Mat<T>, z: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		// Compute Y + Z
		let y_plus_z = y.add(z);

		// QR decomposition
		let qr = y_plus_z.qr();
		let mut q = qr.q().clone();

		// Extract first p columns
		if q.ncols() > self.p {
			q = q.columns(0, self.p);
		}

		// Fix signs for continuity
		let r = qr.r();
		for j in 0..self.p.min(r.ncols()) {
			if r.get(j, j) < T::zero() {
				for i in 0..self.n {
					*q.get_mut(i, j) = T::zero() - q.get(i, j);
				}
			}
		}

		Ok(q)
	}

	/// Performs SVD-based retraction (more stable).
	///
	/// # Mathematical Formula
	///
	/// For Y + Z = UΣV^T, R_Y(Z) = UV^T.
	///
	/// # Arguments
	///
	/// * `y` - Point on Grassmann manifold
	/// * `z` - Tangent vector at y
	///
	/// # Returns
	///
	/// The retracted point using SVD.
	pub fn svd_retraction(&self, y: &linalg::Mat<T>, z: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		let y_plus_z = y.add(z);

		// Compute SVD
		let svd = y_plus_z.svd();

		if let (Some(u), Some(vt)) = (svd.u, svd.vt) {
			// Take first p columns of U and rows of V^T
			let u_truncated = if u.ncols() > self.p {
				u.columns(0, self.p)
			} else {
				u
			};

			let vt_truncated = if vt.nrows() > self.p {
				vt.rows(0, self.p)
			} else {
				vt
			};

			Ok(u_truncated.mat_mul(&vt_truncated))
		} else {
			Err(ManifoldError::numerical_error(
				"SVD computation failed in retraction",
			))
		}
	}

	/// Computes geodesic distance between two subspaces.
	///
	/// # Mathematical Formula
	///
	/// d([Y₁], [Y₂]) = ‖θ‖₂ where θᵢ = arccos(σᵢ(Y₁^T Y₂)).
	///
	/// # Arguments
	///
	/// * `y1` - First point on Grassmann
	/// * `y2` - Second point on Grassmann
	///
	/// # Returns
	///
	/// The geodesic distance between the subspaces.
	pub fn geodesic_distance(&self, y1: &linalg::Mat<T>, y2: &linalg::Mat<T>) -> Result<T> {
		self.check_point(y1)?;
		self.check_point(y2)?;

		// Compute Y₁^T Y₂
		let mut y1ty2 = linalg::Mat::<T>::zeros(self.p, self.p);
		y1ty2.gemm_at(T::one(), y1, y2, T::zero());

		// SVD to get principal angles
		let svd = y1ty2.svd();
		let sigma = &svd.singular_values;

		// Principal angles: θᵢ = arccos(σᵢ)
		let mut dist_sq = T::zero();
		for i in 0..self.p {
			// Clamp singular values to [-1, 1]
			let sigma_clamped =
				<T as Float>::max(<T as Float>::min(sigma.get(i), T::one()), -T::one());
			let theta = <T as Float>::acos(sigma_clamped);
			dist_sq = dist_sq + theta * theta;
		}

		Ok(<T as Float>::sqrt(dist_sq))
	}

	/// Parallel transports a tangent vector along a geodesic.
	///
	/// # Mathematical Formula
	///
	/// For geodesic from [Y₁] to [Y₂], transport Z using:
	/// τ(Z) = (I - Y₂Y₂^T)ZU where Y₂^T Y₁ = UΣV^T.
	pub fn parallel_transport_geodesic(
		&self,
		y1: &linalg::Mat<T>,
		y2: &linalg::Mat<T>,
		z: &linalg::Mat<T>,
	) -> Result<linalg::Mat<T>> {
		self.check_tangent(y1, z)?;
		self.check_point(y2)?;

		// Compute Y₂^T Y₁ and its SVD
		let mut y2ty1 = linalg::Mat::<T>::zeros(self.p, self.p);
		y2ty1.gemm_at(T::one(), y2, y1, T::zero());
		let svd = y2ty1.svd();

		if let Some(u) = svd.u {
			// Transport: (I - Y₂Y₂^T)ZU
			let zu = z.mat_mul(&u);
			let mut y2t_zu = linalg::Mat::<T>::zeros(self.p, zu.ncols());
			y2t_zu.gemm_at(T::one(), y2, &zu, T::zero());
			let y2_zu = y2.mat_mul(&y2t_zu);
			Ok(zu.sub(&y2_zu))
		} else {
			// Fallback to simple projection
			let mut result = z.clone();
			self.project_tangent(y2, z, &mut result)?;
			Ok(result)
		}
	}
}

impl<T> Manifold<T> for Grassmann<T>
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Mat<T>;
	type TangentVector = linalg::Mat<T>;

	fn name(&self) -> &str {
		"Grassmann"
	}

	fn dimension(&self) -> usize {
		self.p * (self.n - self.p)
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.p {
			return false;
		}

		// Check Y^T Y = I_p
		let mut yty = linalg::Mat::<T>::zeros(self.p, self.p);
		yty.gemm_at(T::one(), point, point, T::zero());
		let identity = linalg::Mat::<T>::identity(self.p);
		yty.sub(&identity).norm() < tol
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

		// Horizontal space: Y^T Z = 0
		let mut ytz = linalg::Mat::<T>::zeros(self.p, self.p);
		ytz.gemm_at(T::one(), point, vector, T::zero());
		ytz.norm() < tol
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		if point.nrows() != self.n || point.ncols() != self.p {
			*result = MatrixOps::zeros(self.n, self.p);
			return;
		}

		// Use QR decomposition for projection
		let qr = point.qr();
		let mut q = qr.q().clone();

		// Extract first p columns
		if q.ncols() > self.p {
			q = q.columns(0, self.p);
		}

		// Fix signs for continuity
		let r = qr.r();
		for j in 0..self.p.min(r.ncols()) {
			if r.get(j, j) < T::zero() {
				for i in 0..self.n {
					*q.get_mut(i, j) = T::zero() - q.get(i, j);
				}
			}
		}

		*result = q;
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Horizontal projection: result = Z - Y(Y^T Z)
		// Uses in-place GEMM to avoid allocating n×p temporaries.
		// Only allocates one small p×p buffer for Y^T Z.
		let mut ytz = linalg::Mat::<T>::zeros(self.p, self.p);
		ytz.gemm_at(T::one(), point, vector, T::zero()); // ytz = Y^T Z
		result.copy_from(vector);
		result.gemm(-T::one(), point, &ytz, T::one()); // result = Z - Y(Y^T Z)
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		// Canonical metric: tr(U^T V)
		// No tangent validation on hot path.
		let mut inner = T::zero();
		for i in 0..self.n {
			for j in 0..self.p {
				inner = inner + u.get(i, j) * v.get(i, j);
			}
		}
		Ok(inner)
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		// Polar retraction via SVD: Y = X + G;  U Σ V^T = SVD(Y);  return U V^T
		// Compute Y = X + G in-place into result (avoids n×p allocation).
		result.copy_from(point);
		result.add_assign(tangent);

		// SVD (internal faer allocations unavoidable)
		let svd = result.svd();
		match (svd.u, svd.vt) {
			(Some(u), Some(vt)) => {
				// U * V^T via in-place GEMM (avoids n×p allocation)
				result.gemm(T::one(), &u, &vt, T::zero());
				Ok(())
			}
			_ => {
				let retracted = self.qr_retraction(point, tangent)?;
				result.copy_from(&retracted);
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

		// Compute log map approximation using projection
		// For close points: log_Y(Ỹ) ≈ P_h(Ỹ - Y)
		let diff = other.sub(point);
		self.project_tangent(point, &diff, result)
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Riemannian gradient is the horizontal projection of Euclidean gradient
		self.project_tangent(point, euclidean_grad, result)
	}

	fn euclidean_to_riemannian_hessian(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		euclidean_hvp: &Self::TangentVector,
		tangent_vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Grassmann ehess2rhess (cf. pymanopt grassmann.py):
		// rhess = project(point, ehess) - ξ @ (Y^T @ egrad)
		//
		// Step 1: result = project(point, ehess) = ehess - Y(Y^T ehess)
		self.project_tangent(point, euclidean_hvp, result)?;

		// Step 2: curvature correction via in-place GEMM
		// Reuse a small p×p buffer for Y^T egrad
		let mut ytg = linalg::Mat::<T>::zeros(self.p, self.p);
		ytg.gemm_at(T::one(), point, euclidean_grad, T::zero()); // ytg = Y^T egrad

		// result -= ξ (Y^T egrad)  (in-place, no n×p allocation)
		result.gemm(-T::one(), tangent_vector, &ytg, T::one());
		Ok(())
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Projection-based vector transport: τ_Y(Z) = Π_Y(Z) = Z - Y(Y^T Z)
		// This is the standard choice in pymanopt and manopt (grassmannfactory.m line 310).
		// It is compatible with the polar retraction and sufficient for quotient geometry.
		self.project_tangent(to, vector, result)
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random Gaussian matrix
		let mut a: linalg::Mat<T> = MatrixOps::zeros(self.n, self.p);
		for i in 0..self.n {
			for j in 0..self.p {
				*a.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// QR decomposition to get orthonormal basis
		let qr = a.qr();
		let q = qr.q();

		// Extract first p columns
		if q.ncols() > self.p {
			*result = q.columns(0, self.p);
		} else {
			result.copy_from(q);
		}

		Ok(())
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		self.check_point(point)?;

		// Generate random matrix
		let mut rng = rand::rng();
		let normal = StandardNormal;

		let mut z: linalg::Mat<T> = MatrixOps::zeros(self.n, self.p);
		for i in 0..self.n {
			for j in 0..self.p {
				*z.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// Project to horizontal space
		self.project_tangent(point, &z, result)?;

		// Normalize
		let norm = result.norm();
		if norm > <T as Scalar>::from_f64(1e-16) {
			result.scale_mut(T::one() / norm);
		}

		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		self.geodesic_distance(x, y)
	}

	fn has_exact_exp_log(&self) -> bool {
		false // QR retraction is not the exponential map
	}

	fn is_flat(&self) -> bool {
		false
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For Grassmann manifold, tangent vectors are in the horizontal space
		// Scaling preserves the horizontal space property
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

		// The sum should already be in the horizontal space if v1 and v2 are,
		// but we project for numerical stability
		self.project_tangent(point, temp, result)?;

		Ok(())
	}
}

// NOTE: MatrixManifold<T> impl commented out because the trait still requires
// `Point = DMatrix<T>` but we now use `linalg::Mat<T>`. Will be re-enabled
// once riemannopt-core's MatrixManifold trait is migrated to the linalg layer.
//
// impl<T: Scalar> MatrixManifold<T> for Grassmann<T> {
// 	fn matrix_dims(&self) -> (usize, usize) {
// 		(self.n, self.p)
// 	}
// }
