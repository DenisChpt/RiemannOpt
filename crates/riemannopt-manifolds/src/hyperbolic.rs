//! # Hyperbolic Manifold ℍⁿ
//!
//! The hyperbolic manifold ℍⁿ is the n-dimensional hyperbolic space, a complete
//! Riemannian manifold with constant negative sectional curvature. It provides
//! a natural geometry for hierarchical and tree-like data structures.
//!
//! ## Mathematical Definition
//!
//! Hyperbolic space can be represented through several equivalent models:
//!
//! ### Poincaré Ball Model
//! ```text
//! 𝔹ⁿ = {x ∈ ℝⁿ : ‖x‖ < 1}
//! ```
//! with metric tensor:
//! ```text
//! g_x = (2/(1 - ‖x‖²))² · g_E
//! ```
//! where g_E is the Euclidean metric.
//!
//! ### Hyperboloid Model
//! ```text
//! ℍⁿ = {x ∈ ℝⁿ⁺¹ : ⟨x,x⟩_L = -1, x₀ > 0}
//! ```
//! with Lorentzian inner product ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! In the Poincaré ball model:
//! ```text
//! T_x 𝔹ⁿ ≅ ℝⁿ
//! ```
//! The tangent space is isomorphic to ℝⁿ but with a different metric.
//!
//! ### Riemannian Metric
//! The conformal factor λ(x) = 2/(1 - ‖x‖²) gives:
//! ```text
//! ⟨u, v⟩_x = λ(x)² ⟨u, v⟩_E
//! ```
//!
//! ### Geodesics
//! In the Poincaré ball:
//! - Through origin: straight lines
//! - General: circular arcs orthogonal to the boundary
//!
//! ### Distance Formula
//! ```text
//! d(x, y) = arcosh(1 + 2‖x - y‖²/((1 - ‖x‖²)(1 - ‖y‖²)))
//! ```
//!
//! ## Maps and Operations
//!
//! ### Exponential Map
//! ```text
//! exp_x(v) = x ⊕ tanh(λ_x ‖v‖/2) v/‖v‖
//! ```
//! where ⊕ is the Möbius addition.
//!
//! ### Logarithmic Map
//! ```text
//! log_x(y) = (2/λ_x) artanh(‖-x ⊕ y‖) (-x ⊕ y)/‖-x ⊕ y‖
//! ```
//!
//! ### Parallel Transport
//! Along geodesic from x to y:
//! ```text
//! P_{x→y}(v) = v - (2⟨y,v⟩/(1 - ‖y‖²))(y + (⟨x,y⟩/(1 + ⟨x,y⟩))(x + y))
//! ```
//!
//! ## Geometric Properties
//!
//! - **Sectional curvature**: K < 0 (constant negative, default K = -1)
//! - **Scalar curvature**: R = Kn(n-1)
//! - **Ricci curvature**: Ric = K(n-1)g
//! - **Injectivity radius**: ∞ (simply connected)
//! - **Volume growth**: Exponential
//! - **Isometry group**: SO⁺(n,1)
//!
//! ## Applications
//!
//! 1. **Hierarchical embeddings**: Tree-like data structures
//! 2. **Natural language processing**: Word embeddings with hierarchy
//! 3. **Social networks**: Community detection and influence propagation
//! 4. **Bioinformatics**: Phylogenetic trees and protein folding
//! 5. **Computer vision**: Wide-angle and omnidirectional imaging
//! 6. **Recommendation systems**: Capturing latent hierarchies
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Numerical stability** near the boundary using careful projections
//! - **Efficient computation** through optimized Möbius operations
//! - **Boundary handling** with configurable tolerance
//! - **Exact geodesics** through closed-form expressions
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Hyperbolic;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{self, VectorOps};
//!
//! // Create ℍ² (Poincaré disk)
//! let hyperbolic = Hyperbolic::<f64>::new(2)?;
//!
//! // Random point in the ball
//! let mut x = linalg::Vec::<f64>::zeros(2);
//! hyperbolic.random_point(&mut x)?;
//! assert!(x.norm() < 1.0);
//!
//! // Tangent vector
//! let v = VectorOps::from_slice(&[0.1, 0.2]);
//!
//! // Retraction
//! let mut y = linalg::Vec::<f64>::zeros(2);
//! hyperbolic.retract(&x, &v, &mut y, &mut ())?;
//!
//! // Verify y is in the ball
//! assert!(hyperbolic.is_point_on_manifold(&y, 1e-10));
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use riemannopt_core::{
	error::{ManifoldError, Result},
	linalg::{self, LinAlgBackend, VectorOps},
	manifold::Manifold,
	types::Scalar,
};
use std::fmt::{self, Debug};

/// Default boundary tolerance for Poincaré ball boundary stability
const DEFAULT_BOUNDARY_TOLERANCE: f64 = 1e-6;

/// Safety margin for projection to ensure points stay well inside the ball
const PROJECTION_SAFETY_MARGIN: f64 = 0.999;

/// The hyperbolic manifold ℍⁿ using the Poincaré ball model.
///
/// This structure represents n-dimensional hyperbolic space using the Poincaré ball
/// model 𝔹ⁿ_K = {x ∈ ℝⁿ : ‖x‖ < √(-1/K)}, equipped with the hyperbolic metric.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `n ≥ 1`: Dimension must be positive
/// - All points x satisfy ‖x‖ < √(-1/K) where K < 0 is the curvature
/// - Tangent vectors can be any vectors in ℝⁿ
#[derive(Clone)]
pub struct Hyperbolic<T = f64> {
	/// Dimension of the hyperbolic space
	n: usize,
	/// Numerical tolerance for boundary checks
	boundary_tolerance: T,
	/// Curvature parameter (default -1)
	curvature: T,
}

impl<T: Scalar> Debug for Hyperbolic<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Hyperbolic ℍ^{} (Poincaré ball)", self.n)
	}
}

impl<T: Scalar> Hyperbolic<T> {
	/// Creates a new hyperbolic manifold ℍⁿ with standard curvature -1.
	///
	/// # Arguments
	///
	/// * `n` - Dimension of the hyperbolic space (must be ≥ 1)
	///
	/// # Returns
	///
	/// A hyperbolic manifold with dimension n.
	///
	/// # Errors
	///
	/// Returns `ManifoldError::InvalidParameter` if n = 0.
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::Hyperbolic;
	/// // Create ℍ² (Poincaré disk)
	/// let h2 = Hyperbolic::<f64>::new(2)?;
	///
	/// // Create ℍ³ (Poincaré ball in 3D)
	/// let h3 = Hyperbolic::<f64>::new(3)?;
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	pub fn new(n: usize) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Hyperbolic manifold requires dimension n ≥ 1",
			));
		}
		Ok(Self {
			n,
			boundary_tolerance: <T as Scalar>::from_f64(DEFAULT_BOUNDARY_TOLERANCE),
			curvature: -T::one(),
		})
	}

	/// Creates a hyperbolic manifold with custom parameters.
	///
	/// # Arguments
	///
	/// * `n` - Dimension of the space
	/// * `boundary_tolerance` - Distance from boundary to maintain
	/// * `curvature` - Sectional curvature (must be negative)
	pub fn with_parameters(n: usize, boundary_tolerance: T, curvature: T) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Hyperbolic manifold requires dimension n ≥ 1",
			));
		}
		if boundary_tolerance <= T::zero() || boundary_tolerance >= T::one() {
			return Err(ManifoldError::invalid_parameter(
				"Boundary tolerance must be in (0, 1)",
			));
		}
		if curvature >= T::zero() {
			return Err(ManifoldError::invalid_parameter(
				"Curvature must be negative for hyperbolic space",
			));
		}
		Ok(Self {
			n,
			boundary_tolerance,
			curvature,
		})
	}

	/// Returns the dimension of the hyperbolic space.
	#[inline]
	pub fn ambient_dim(&self) -> usize {
		self.n
	}

	/// Returns the boundary tolerance.
	#[inline]
	pub fn boundary_tolerance(&self) -> T {
		self.boundary_tolerance
	}

	/// Returns the curvature.
	#[inline]
	pub fn curvature(&self) -> T {
		self.curvature
	}
}

impl<T> Hyperbolic<T>
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Validates that a point lies in the Poincaré ball.
	///
	/// # Mathematical Check
	///
	/// Verifies that ‖x‖ < 1 within boundary tolerance.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If x.len() ≠ n
	/// - `NotOnManifold`: If ‖x‖ ≥ 1 - boundary_tolerance
	pub fn check_point(&self, x: &linalg::Vec<T>) -> Result<()> {
		if VectorOps::len(x) != self.n {
			return Err(ManifoldError::dimension_mismatch(self.n, VectorOps::len(x)));
		}

		let norm_squared = x.norm_squared();
		// For curvature K < 0, the ball radius is sqrt(-1/K)
		let ball_radius = <T as Float>::sqrt(T::one() / (-self.curvature));
		let boundary = ball_radius - self.boundary_tolerance;

		if norm_squared >= boundary * boundary {
			return Err(ManifoldError::invalid_point(format!(
				"Point not in Poincaré ball: ‖x‖² = {} (boundary: {}²)",
				norm_squared, boundary
			)));
		}

		Ok(())
	}

	/// Validates that a vector is a valid tangent vector.
	///
	/// # Mathematical Check
	///
	/// In the Poincaré ball model, all vectors in ℝⁿ are valid tangent vectors.
	///
	/// # Errors
	///
	/// - `DimensionMismatch`: If dimensions don't match
	/// - `NotOnManifold`: If x is not in the Poincaré ball
	pub fn check_tangent(&self, x: &linalg::Vec<T>, v: &linalg::Vec<T>) -> Result<()> {
		self.check_point(x)?;

		if VectorOps::len(v) != self.n {
			return Err(ManifoldError::dimension_mismatch(self.n, VectorOps::len(v)));
		}

		Ok(())
	}

	/// Computes the exponential map exp_x(v).
	///
	/// # Mathematical Formula
	///
	/// In the Poincaré ball model with curvature K < 0:
	/// exp_x(v) = x ⊕ tanh(√(-K) λ_x ‖v‖/2) v/‖v‖
	/// where ⊕ is the Möbius addition and λ_x is the conformal factor.
	///
	/// # Arguments
	///
	/// * `x` - Point in the Poincaré ball
	/// * `v` - Tangent vector at x
	///
	/// # Returns
	///
	/// The point exp_x(v) in the Poincaré ball.
	pub fn exp_map(&self, x: &linalg::Vec<T>, v: &linalg::Vec<T>) -> Result<linalg::Vec<T>> {
		self.check_tangent(x, v)?;
		Ok(self.exponential_map(x, v))
	}

	/// Computes the logarithmic map log_x(y).
	///
	/// # Mathematical Formula
	///
	/// For x, y in the Poincaré ball:
	/// log_x(y) finds the tangent vector at x pointing towards y
	/// such that exp_x(log_x(y)) = y.
	///
	/// # Arguments
	///
	/// * `x` - Point in the Poincaré ball
	/// * `y` - Another point in the Poincaré ball
	///
	/// # Returns
	///
	/// The tangent vector log_x(y) ∈ T_x ℍⁿ.
	pub fn log_map(&self, x: &linalg::Vec<T>, y: &linalg::Vec<T>) -> Result<linalg::Vec<T>> {
		self.check_point(x)?;
		self.check_point(y)?;
		let mut result: linalg::Vec<T> = VectorOps::zeros(self.n);
		self.logarithmic_map(x, y, &mut result);
		Ok(result)
	}

	/// Computes the hyperbolic distance between two points.
	///
	/// # Mathematical Formula
	///
	/// d(x, y) = arcosh(1 + 2‖x - y‖²/((1 - ‖x‖²)(1 - ‖y‖²)))
	///
	/// # Arguments
	///
	/// * `x` - First point in the Poincaré ball
	/// * `y` - Second point in the Poincaré ball
	///
	/// # Returns
	///
	/// The hyperbolic distance d(x, y) ≥ 0.
	pub fn geodesic_distance(&self, x: &linalg::Vec<T>, y: &linalg::Vec<T>) -> Result<T> {
		self.check_point(x)?;
		self.check_point(y)?;
		Ok(self.hyperbolic_distance(x, y))
	}

	/// Parallel transports a tangent vector along a geodesic.
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
	pub fn parallel_transport(
		&self,
		x: &linalg::Vec<T>,
		y: &linalg::Vec<T>,
		v: &linalg::Vec<T>,
		_ws: &mut (),
	) -> Result<linalg::Vec<T>> {
		self.check_tangent(x, v)?;
		self.check_point(y)?;
		Ok(self.parallel_transport_vector(x, y, v))
	}

	/// Checks if a point is in the Poincare ball (||x|| < √(-1/K)).
	fn is_in_poincare_ball(&self, point: &linalg::Vec<T>, tolerance: T) -> bool {
		if VectorOps::len(point) != self.n {
			return false;
		}

		let norm_squared = point.norm_squared();
		// For curvature K < 0, the ball radius is sqrt(-1/K)
		let ball_radius = <T as Float>::sqrt(T::one() / (-self.curvature));
		let boundary = ball_radius - tolerance;
		let boundary_squared = boundary * boundary;
		norm_squared < boundary_squared
	}

	/// Projects a point to the Poincare ball, writing into `result`.
	///
	/// For general curvature K < 0, the ball has radius √(-1/K).
	/// If the point is outside the ball, project it to a point
	/// slightly inside the boundary for numerical stability.
	fn project_to_poincare_ball(
		&self,
		point: &linalg::Vec<T>,
		result: &mut linalg::Vec<T>,
	) {
		let norm = point.norm();
		let neg_curv = -self.curvature;
		let ball_radius = <T as Float>::sqrt(neg_curv);
		let max_norm = ball_radius - self.boundary_tolerance;

		result.copy_from(point);
		if norm > max_norm {
			// Project to boundary with tolerance (slightly inside)
			let safe_norm = max_norm * <T as Scalar>::from_f64(PROJECTION_SAFETY_MARGIN);
			result.scale_mut(safe_norm / norm);
		}
	}

	/// Computes the conformal factor lambda(x) for general curvature K.
	/// For K = -1: lambda(x) = 2 / (1 - ||x||^2)
	/// For general K < 0: lambda(x) = 2/√(-K) / (1 - ||x||²/(-K))
	fn conformal_factor(&self, point: &linalg::Vec<T>) -> T {
		let norm_squared = point.norm_squared();
		let neg_curv = -self.curvature;
		let sqrt_neg_curv = <T as Float>::sqrt(neg_curv);
		let two = <T as Scalar>::from_f64(2.0);
		(two / sqrt_neg_curv) / (T::one() - norm_squared / neg_curv)
	}

	/// Computes the hyperbolic distance between two points in the Poincare ball.
	///
	/// For curvature K < 0:
	/// d(x,y) = 1/√(-K) * arcosh(1 + 2||x-y||^2 / ((1-||x||^2/(-K))(1-||y||^2/(-K))))
	///
	/// Uses ‖x-y‖² = ‖x‖² + ‖y‖² - 2⟨x,y⟩ to avoid allocating a difference vector.
	fn hyperbolic_distance(&self, x: &linalg::Vec<T>, y: &linalg::Vec<T>) -> T {
		let x_norm_sq = x.norm_squared();
		let y_norm_sq = y.norm_squared();
		let xy = x.dot(y);
		let two = <T as Scalar>::from_f64(2.0);
		let diff_norm_sq = x_norm_sq + y_norm_sq - two * xy;

		let neg_curv = -self.curvature;
		let denominator = (T::one() - x_norm_sq / neg_curv) * (T::one() - y_norm_sq / neg_curv);

		let argument = T::one() + two * diff_norm_sq / denominator;

		// Clamp argument to avoid numerical issues with acosh
		let clamped = <T as Float>::max(argument, T::one());
		let sqrt_neg_curv = <T as Float>::sqrt(neg_curv);
		<T as Float>::acosh(clamped) / sqrt_neg_curv
	}

	/// Computes the exponential map in the Poincare ball model.
	///
	/// The exponential map moves along geodesics from a point in a given direction.
	/// Allocates a single result vector; all intermediate work is done in-place.
	fn exponential_map(&self, point: &linalg::Vec<T>, tangent: &linalg::Vec<T>) -> linalg::Vec<T> {
		let tangent_norm = tangent.norm();

		let mut result: linalg::Vec<T> = VectorOps::zeros(VectorOps::len(point));

		if tangent_norm < <T as Scalar>::from_f64(1e-16) {
			result.copy_from(point);
			return result;
		}

		let lambda = self.conformal_factor(point);
		let sqrt_neg_curv = <T as Float>::sqrt(-self.curvature);
		let scaled_norm = sqrt_neg_curv * tangent_norm * lambda / <T as Scalar>::from_f64(2.0);

		// Exponential map formula in Poincare ball
		let alpha = <T as Float>::tanh(scaled_norm);

		// result = point + (alpha / tangent_norm) * tangent
		result.copy_from(point);
		result.axpy(alpha / tangent_norm, tangent, T::one());

		// denominator = 1 + alpha * dot(point, tangent/norm)
		let denominator = T::one() + alpha * point.dot(tangent) / tangent_norm;

		result.div_scalar_mut(denominator);

		// Project to ball boundary if needed (inlined)
		let norm = result.norm();
		let neg_curv = -self.curvature;
		let ball_radius = <T as Float>::sqrt(neg_curv);
		let max_norm = ball_radius - self.boundary_tolerance;
		if norm > max_norm {
			let safe_norm = max_norm * <T as Scalar>::from_f64(PROJECTION_SAFETY_MARGIN);
			result.scale_mut(safe_norm / norm);
		}

		result
	}

	/// Computes the logarithmic map in the Poincare ball model, writing into `result`.
	///
	/// The logarithmic map finds the tangent vector from point to other.
	/// Zero allocation: all work is done in the caller-provided `result`.
	fn logarithmic_map(
		&self,
		point: &linalg::Vec<T>,
		other: &linalg::Vec<T>,
		result: &mut linalg::Vec<T>,
	) {
		// result = other - point
		result.copy_from(other);
		result.sub_assign(point);

		let diff_norm = result.norm();
		if diff_norm < <T as Scalar>::from_f64(1e-16) {
			result.fill(T::zero());
			return;
		}

		// log_x(y) = d(x,y) * (y-x) / ||y-x||_x
		let dist = self.hyperbolic_distance(point, other);
		let lambda = self.conformal_factor(point);

		// result = (other - point) / diff_norm * dist * 2 / lambda
		result.scale_mut(dist * <T as Scalar>::from_f64(2.0) / (lambda * diff_norm));
	}

	/// Generates a random point in the Poincare ball.
	fn random_poincare_point(&self, result: &mut linalg::Vec<T>) -> Result<()> {
		let mut rng = rand::rng();

		// Ensure result has correct size
		if VectorOps::len(result) != self.n {
			*result = VectorOps::zeros(self.n);
		}

		// Generate random direction
		for i in 0..self.n {
			let val: f64 = StandardNormal.sample(&mut rng);
			*result.get_mut(i) = <T as Scalar>::from_f64(val);
		}

		// Normalize and scale by random radius
		let norm = result.norm();
		if norm > <T as Scalar>::from_f64(1e-16) {
			result.div_scalar_mut(norm);

			// Random radius with appropriate distribution for uniform distribution in ball
			let u: f64 = rand::random();
			let radius = u.powf(1.0 / self.n as f64);
			let neg_curv = -self.curvature;
			let ball_radius = <T as Float>::sqrt(neg_curv).to_f64();
			let max_radius = ball_radius - self.boundary_tolerance.to_f64();
			let scaled_radius = radius * max_radius;

			result.scale_mut(<T as Scalar>::from_f64(scaled_radius));
		} else {
			// Return origin if we got zero vector
			result.fill(T::zero());
		}

		Ok(())
	}

	/// Parallel transport using the Levi-Civita connection.
	///
	/// Transports a tangent vector along the geodesic from one point to another.
	/// Allocates a single result vector; all intermediate work is done in-place via axpy.
	fn parallel_transport_vector(
		&self,
		from: &linalg::Vec<T>,
		to: &linalg::Vec<T>,
		vector: &linalg::Vec<T>,
	) -> linalg::Vec<T> {
		let n = VectorOps::len(vector);
		let mut result: linalg::Vec<T> = VectorOps::zeros(n);

		// Check if from == to without allocating: ‖to-from‖² = ‖to‖²+‖from‖²-2⟨to,from⟩
		let from_norm_sq = from.norm_squared();
		let to_norm_sq = to.norm_squared();
		let from_dot_to = from.dot(to);
		let diff_norm_sq = from_norm_sq + to_norm_sq - <T as Scalar>::from_f64(2.0) * from_dot_to;

		if diff_norm_sq < <T as Scalar>::from_f64(1e-32) {
			result.copy_from(vector);
			return result;
		}

		let from_dot_v = from.dot(vector);
		let to_dot_v = to.dot(vector);

		// P_{x→y}(v) = v + 2/(1 - K||y||²) * (⟨y,v⟩·y - ⟨x,v⟩/(1+⟨x,y⟩) · (y+x))

		let denominator = T::one() + from_dot_to;

		// Avoid division by zero (nearly antipodal)
		if <T as Float>::abs(denominator) < <T as Scalar>::from_f64(1e-16) {
			let lambda_from = self.conformal_factor(from);
			let lambda_to = self.conformal_factor(to);
			result.copy_from(vector);
			result.scale_mut(lambda_from / lambda_to);
			return result;
		}

		let scale_factor = <T as Scalar>::from_f64(2.0) / (T::one() - self.curvature * to_norm_sq);
		let coeff_from_v = from_dot_v / denominator;

		// Build: result = to_dot_v · to − coeff_from_v · (to + from)
		//               = (to_dot_v − coeff_from_v) · to − coeff_from_v · from
		// Then:  result = vector + scale_factor · result

		// Step 1: result = (to_dot_v - coeff_from_v) · to
		result.copy_from(to);
		result.scale_mut(to_dot_v - coeff_from_v);

		// Step 2: result -= coeff_from_v · from
		result.axpy(-coeff_from_v, from, T::one());

		// Step 3: result = vector + scale_factor · result
		result.axpy(T::one(), vector, scale_factor);

		result
	}
}

impl<T> Manifold<T> for Hyperbolic<T>
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;
	type Workspace = ();

	fn name(&self) -> &str {
		"Hyperbolic"
	}

	fn dimension(&self) -> usize {
		self.n
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tolerance: T) -> bool {
		self.is_in_poincare_ball(point, tolerance)
	}

	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_tolerance: T,
	) -> bool {
		// In Poincare ball model, all vectors of correct dimension are tangent vectors
		VectorOps::len(vector) == self.n
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Ensure result has correct size
		if VectorOps::len(result) != self.n {
			*result = VectorOps::zeros(self.n);
		}

		if VectorOps::len(point) != self.n {
			// Handle wrong dimension by padding/truncating
			let mut temp: linalg::Vec<T> = VectorOps::zeros(self.n);
			let copy_len = VectorOps::len(point).min(self.n);
			for i in 0..copy_len {
				*temp.get_mut(i) = point.get(i);
			}
			self.project_to_poincare_ball(&temp, result);
		} else {
			self.project_to_poincare_ball(point, result);
		}
	}

	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// In Poincaré ball model, tangent space at any point is ℝⁿ — identity projection
		result.copy_from(vector);
		Ok(())
	}

	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> Result<T> {
		if VectorOps::len(point) != self.n
			|| VectorOps::len(u) != self.n
			|| VectorOps::len(v) != self.n
		{
			return Err(ManifoldError::dimension_mismatch(
				self.n,
				VectorOps::len(point)
					.max(VectorOps::len(u))
					.max(VectorOps::len(v)),
			));
		}

		// Hyperbolic inner product: <u,v>_x = lambda(x)^2 * <u,v>_euclidean
		let lambda = self.conformal_factor(point);
		let euclidean_inner = u.dot(v);
		Ok(lambda * lambda * euclidean_inner)
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) -> Result<()> {
		// Exponential map in Poincaré ball — in-place into result
		let tangent_norm = tangent.norm();

		if tangent_norm < <T as Scalar>::from_f64(1e-16) {
			result.copy_from(point);
			return Ok(());
		}

		let lambda = self.conformal_factor(point);
		let sqrt_neg_curv = <T as Float>::sqrt(-self.curvature);
		let scaled_norm = sqrt_neg_curv * tangent_norm * lambda / <T as Scalar>::from_f64(2.0);
		let alpha = <T as Float>::tanh(scaled_norm);

		// result = point + (alpha / tangent_norm) * tangent
		result.copy_from(point);
		result.axpy(alpha / tangent_norm, tangent, T::one());

		// result /= (1 + alpha * dot(point, tangent/norm))
		let denominator = T::one() + alpha * point.dot(tangent) / tangent_norm;
		result.div_scalar_mut(denominator);

		// Project to ball boundary if needed
		let norm = result.norm();
		let neg_curv = -self.curvature;
		let ball_radius = <T as Float>::sqrt(neg_curv);
		let max_norm = ball_radius - self.boundary_tolerance;
		if norm > max_norm {
			let safe_norm = max_norm * <T as Scalar>::from_f64(PROJECTION_SAFETY_MARGIN);
			result.scale_mut(safe_norm / norm);
		}

		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// Ensure result has correct size
		if VectorOps::len(result) != self.n {
			*result = VectorOps::zeros(self.n);
		}

		if VectorOps::len(point) != self.n || VectorOps::len(other) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.n,
				VectorOps::len(point).max(VectorOps::len(other)),
			));
		}

		// Use logarithmic map as inverse retraction (zero-alloc: writes directly into result)
		self.logarithmic_map(point, other, result);
		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// Ensure result has correct size
		if VectorOps::len(result) != self.n {
			*result = VectorOps::zeros(self.n);
		}

		if VectorOps::len(point) != self.n || VectorOps::len(euclidean_grad) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.n,
				VectorOps::len(point).max(VectorOps::len(euclidean_grad)),
			));
		}

		// Convert Euclidean gradient to Riemannian gradient
		// For Poincare ball with curvature K: grad_riem = (1 - K||x||^2)^2 / 4 * grad_euclidean
		// Note: Since K < 0, we have 1 - K||x||^2 = 1 + |K|||x||^2
		let norm_squared = point.norm_squared();
		let k_norm_sq = self.curvature * norm_squared; // This is negative
		let factor = (T::one() - k_norm_sq) * (T::one() - k_norm_sq) / <T as Scalar>::from_f64(4.0);

		result.copy_from(euclidean_grad);
		result.scale_mut(factor);
		Ok(())
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		self.random_poincare_point(result)
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		// Ensure result has correct size
		if VectorOps::len(result) != self.n {
			*result = VectorOps::zeros(self.n);
		}

		if VectorOps::len(point) != self.n {
			return Err(ManifoldError::dimension_mismatch(
				self.n,
				VectorOps::len(point),
			));
		}

		let mut rng = rand::rng();
		let mut tangent: linalg::Vec<T> = VectorOps::zeros(self.n);

		for i in 0..self.n {
			let val: f64 = StandardNormal.sample(&mut rng);
			*tangent.get_mut(i) = <T as Scalar>::from_f64(val);
		}

		self.project_tangent(point, &tangent, result, &mut ())?;
		Ok(())
	}

	fn has_exact_exp_log(&self) -> bool {
		true // Poincare ball model has exact exponential and logarithmic maps
	}

	fn parallel_transport(
		&self,
		from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// P_{x→y}(v) = v + scale_factor * (⟨y,v⟩·y - ⟨x,v⟩/(1+⟨x,y⟩) · (y+x))
		// In-place into result.

		let from_dot_to = from.dot(to);
		let denominator = T::one() + from_dot_to;

		if <T as Float>::abs(denominator) < <T as Scalar>::from_f64(1e-16) {
			// Nearly antipodal — simple scaling fallback
			let lambda_from = self.conformal_factor(from);
			let lambda_to = self.conformal_factor(to);
			result.copy_from(vector);
			result.scale_mut(lambda_from / lambda_to);
			return Ok(());
		}

		let to_norm_sq = to.norm_squared();
		let from_dot_v = from.dot(vector);
		let to_dot_v = to.dot(vector);
		let scale_factor = <T as Scalar>::from_f64(2.0) / (T::one() - self.curvature * to_norm_sq);

		// result = vector + scale_factor * (to_dot_v · to - (from_dot_v / denom) · (to + from))
		// Step 1: result = to * to_dot_v
		result.copy_from(to);
		result.scale_mut(to_dot_v);
		// Step 2: result -= (from_dot_v / denom) * to
		result.axpy(-from_dot_v / denominator, to, T::one());
		// Step 3: result -= (from_dot_v / denom) * from
		result.axpy(-from_dot_v / denominator, from, T::one());
		// Step 4: result *= scale_factor, then += vector
		result.scale_mut(scale_factor);
		result.add_assign(vector);

		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		// d(x,y) = 1/√(-K) · acosh(1 + 2‖x-y‖² / ((1-‖x‖²/(-K))(1-‖y‖²/(-K))))
		// ‖x-y‖² = ‖x‖² + ‖y‖² - 2⟨x,y⟩  (zero alloc)
		let x_norm_sq = x.norm_squared();
		let y_norm_sq = y.norm_squared();
		let xy = x.dot(y);
		let diff_norm_sq = x_norm_sq + y_norm_sq - (T::one() + T::one()) * xy;

		let neg_curv = -self.curvature;
		let denominator = (T::one() - x_norm_sq / neg_curv) * (T::one() - y_norm_sq / neg_curv);
		let argument = T::one() + (T::one() + T::one()) * diff_norm_sq / denominator;
		let clamped = <T as Float>::max(argument, T::one());
		Ok(<T as Float>::acosh(clamped) / <T as Float>::sqrt(neg_curv))
	}

	fn is_flat(&self) -> bool {
		false // Hyperbolic space has constant negative curvature
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// In the Poincaré ball model, tangent vectors are just vectors in R^n
		// Scaling preserves the tangent space property
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
		// In the Poincaré ball model, tangent space at a point is just R^n
		// So addition is standard vector addition
		result.copy_from(v1);
		result.add_assign(v2);
		Ok(())
	}
}
