//! # Hyperbolic Manifold ℍⁿ
//!
//! The hyperbolic manifold ℍⁿ is the n-dimensional hyperbolic space, a complete
//! Riemannian manifold with constant negative sectional curvature. It provides
//! a natural geometry for hierarchical and tree-like data structures.
//!
//! ## Mathematical Definition (Poincaré Ball Model)
//!
//! ```text
//! 𝔹ⁿ = {x ∈ ℝⁿ : ‖x‖ < √(-1/K)}
//! ```
//! with metric tensor:
//! ```text
//! g_x = λ(x)² · g_E
//! ```
//! where λ(x) = 2/√(-K) / (1 + K‖x‖²) is the conformal factor, and K < 0.

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, VectorOps, VectorView},
	manifold::Manifold,
	types::Scalar,
};

/// Default boundary tolerance for Poincaré ball boundary stability
const DEFAULT_BOUNDARY_TOLERANCE: f64 = 1e-6;

/// Safety margin for projection to ensure points stay well inside the ball
const PROJECTION_SAFETY_MARGIN: f64 = 0.999;

/// The hyperbolic manifold ℍⁿ using the Poincaré ball model.
///
/// Generic over scalar type `T` and linear algebra backend `B`.
#[derive(Clone)]
pub struct Hyperbolic<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	boundary_tolerance: T,
	curvature: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for Hyperbolic<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			f,
			"Hyperbolic ℍ^{} (Poincaré ball, K={})",
			self.n, self.curvature
		)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Hyperbolic<T, B> {
	/// Creates a new hyperbolic manifold ℍⁿ with standard curvature -1.
	#[inline]
	pub fn new(n: usize) -> Self {
		assert!(n > 0, "Hyperbolic manifold requires dimension n ≥ 1");
		Self {
			n,
			boundary_tolerance: <T as Scalar>::from_f64(DEFAULT_BOUNDARY_TOLERANCE),
			curvature: -T::one(),
			_phantom: PhantomData,
		}
	}

	/// Creates a hyperbolic manifold with custom parameters.
	pub fn with_parameters(n: usize, boundary_tolerance: T, curvature: T) -> Self {
		assert!(n > 0, "Hyperbolic manifold requires dimension n ≥ 1");
		assert!(
			boundary_tolerance > T::zero() && boundary_tolerance < T::one(),
			"Boundary tolerance must be in (0, 1)"
		);
		assert!(
			curvature < T::zero(),
			"Curvature must be strictly negative for hyperbolic space"
		);
		Self {
			n,
			boundary_tolerance,
			curvature,
			_phantom: PhantomData,
		}
	}

	/// Computes the conformal factor λ(x).
	#[inline]
	fn conformal_factor(&self, point: &B::Vector) -> T {
		let norm_squared = point.norm_squared();
		let neg_curv = -self.curvature;
		let sqrt_neg_curv = <T as Float>::sqrt(neg_curv);
		let two = <T as Scalar>::from_f64(2.0);
		(two / sqrt_neg_curv) / (T::one() - norm_squared / neg_curv)
	}

	/// Validates if a point is safely inside the Poincaré ball.
	#[inline]
	pub fn is_in_poincare_ball(&self, point: &B::Vector, tolerance: T) -> bool {
		if VectorView::len(point) != self.n {
			return false;
		}
		let norm_squared = point.norm_squared();
		let ball_radius = <T as Float>::sqrt(T::one() / (-self.curvature));
		let boundary = ball_radius - tolerance;
		norm_squared < boundary * boundary
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold trait implementation — hot path, zero alloc, no Result
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for Hyperbolic<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Vector;
	type TangentVector = B::Vector;
	type Workspace = ();

	#[inline]
	fn name(&self) -> &str {
		"Hyperbolic"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.n
	}

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		self.is_in_poincare_ball(point, tol)
	}

	#[inline]
	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_tol: T,
	) -> bool {
		// In the Poincaré ball, the tangent space is the entire ambient space ℝⁿ
		VectorView::len(vector) == self.n
	}

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let norm = point.norm();
		let neg_curv = -self.curvature;
		let ball_radius = <T as Float>::sqrt(neg_curv);
		let max_norm = ball_radius - self.boundary_tolerance;

		result.copy_from(point);
		if norm > max_norm {
			// Project slightly inside the boundary for numerical stability
			let safe_norm = max_norm * <T as Scalar>::from_f64(PROJECTION_SAFETY_MARGIN);
			result.scale_mut(safe_norm / norm);
		}
	}

	#[inline]
	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		// Tangent space is ℝⁿ — identity projection
		result.copy_from(vector);
	}

	#[inline]
	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> T {
		// g_x(u,v) = λ(x)² ⟨u,v⟩_E
		let lambda = self.conformal_factor(point);
		let euclidean_inner = u.dot(v);
		lambda * lambda * euclidean_inner
	}

	#[inline]
	fn norm(&self, point: &Self::Point, vector: &Self::TangentVector, _ws: &mut ()) -> T {
		// ‖v‖_x = λ(x) ‖v‖_E
		let lambda = self.conformal_factor(point);
		lambda * vector.norm()
	}

	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) {
		// Using the exact exponential map for the retraction
		let tangent_norm = tangent.norm();

		if tangent_norm < <T as Scalar>::from_f64(1e-16) {
			result.copy_from(point);
			return;
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
		result.scale_mut(T::one() / denominator);

		// Enforce boundary constraints (in-place projection)
		let norm = result.norm();
		let max_norm = sqrt_neg_curv - self.boundary_tolerance;
		if norm > max_norm {
			let safe_norm = max_norm * <T as Scalar>::from_f64(PROJECTION_SAFETY_MARGIN);
			result.scale_mut(safe_norm / norm);
		}
	}

	#[inline]
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		// result = other - point
		result.copy_from(other);
		result.axpy(-T::one(), point, T::one());

		let diff_norm = result.norm();
		if diff_norm < <T as Scalar>::from_f64(1e-16) {
			result.fill(T::zero());
			return;
		}

		// log_x(y) = d(x,y) * (y-x) / ‖y-x‖_E * (2 / λ(x))
		let dist = self.distance(point, other);
		let lambda = self.conformal_factor(point);

		let scale = dist * <T as Scalar>::from_f64(2.0) / (lambda * diff_norm);
		result.scale_mut(scale);
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		// grad_R = grad_E / λ(x)²
		let lambda = self.conformal_factor(point);
		let factor = T::one() / (lambda * lambda);

		result.copy_from(euclidean_grad);
		result.scale_mut(factor);
	}

	#[inline]
	fn parallel_transport(
		&self,
		from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		let from_dot_to = from.dot(to);
		let denominator = T::one() + from_dot_to;

		// Fallback for near-antipodal or extremely close points
		if <T as Float>::abs(denominator) < <T as Scalar>::from_f64(1e-16) {
			let lambda_from = self.conformal_factor(from);
			let lambda_to = self.conformal_factor(to);
			result.copy_from(vector);
			result.scale_mut(lambda_from / lambda_to);
			return;
		}

		let to_norm_sq = to.norm_squared();
		let from_dot_v = from.dot(vector);
		let to_dot_v = to.dot(vector);
		let scale_factor = <T as Scalar>::from_f64(2.0) / (T::one() - self.curvature * to_norm_sq);

		// result = vector + scale_factor * (to_dot_v · to - (from_dot_v / denom) · (to + from))

		// 1. result = to_dot_v * to
		result.copy_from(to);
		result.scale_mut(to_dot_v);

		// 2. result -= (from_dot_v / denom) * to
		result.axpy(-from_dot_v / denominator, to, T::one());

		// 3. result -= (from_dot_v / denom) * from
		result.axpy(-from_dot_v / denominator, from, T::one());

		// 4. result = scale_factor * result + vector
		result.scale_mut(scale_factor);
		result.add_assign(vector);
	}

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random direction
		for i in 0..self.n {
			*result.get_mut(i) = <T as Scalar>::from_f64(normal.sample(&mut rng));
		}

		let norm = result.norm();
		if norm > <T as Scalar>::from_f64(1e-16) {
			result.scale_mut(T::one() / norm);

			// Random radius (uniform distribution in ball requires r^(1/n))
			let u: f64 = rand::random();
			let radius = u.powf(1.0 / self.n as f64);
			let neg_curv = -self.curvature;
			let ball_radius = <T as Float>::sqrt(neg_curv).to_f64();
			let max_radius = ball_radius - self.boundary_tolerance.to_f64();
			let scaled_radius = radius * max_radius;

			result.scale_mut(<T as Scalar>::from_f64(scaled_radius));
		} else {
			result.fill(T::zero());
		}
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		// Tangent space is ℝⁿ. Generate random vector and normalize in the Riemannian metric.
		let mut rng = rand::rng();
		let normal = StandardNormal;
		for i in 0..self.n {
			*result.get_mut(i) = <T as Scalar>::from_f64(normal.sample(&mut rng));
		}

		let r_norm = self.norm(point, result, &mut ());
		if r_norm > T::zero() {
			result.scale_mut(T::one() / r_norm);
		}
	}

	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		// Allocate `diff` to safely compute ‖x-y‖² without catastrophic cancellation
		let mut diff = y.clone();
		diff.axpy(-T::one(), x, T::one());
		let diff_norm_sq = diff.norm_squared();

		let x_norm_sq = x.norm_squared();
		let y_norm_sq = y.norm_squared();

		let neg_curv = -self.curvature;
		let denominator = (T::one() - x_norm_sq / neg_curv) * (T::one() - y_norm_sq / neg_curv);

		let argument = T::one() + <T as Scalar>::from_f64(2.0) * diff_norm_sq / denominator;
		let clamped = <T as Float>::max(argument, T::one());

		<T as Float>::acosh(clamped) / <T as Float>::sqrt(neg_curv)
	}

	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		true
	}

	#[inline]
	fn is_flat(&self) -> bool {
		false
	}

	// ════════════════════════════════════════════════════════════════════════
	// Vector ops
	// ════════════════════════════════════════════════════════════════════════

	#[inline]
	fn scale_tangent(&self, scalar: T, v: &mut Self::TangentVector) {
		v.scale_mut(scalar);
	}

	#[inline]
	fn add_tangents(&self, v1: &mut Self::TangentVector, v2: &Self::TangentVector) {
		v1.add_assign(v2);
	}

	#[inline]
	fn axpy_tangent(&self, alpha: T, x: &Self::TangentVector, y: &mut Self::TangentVector) {
		y.axpy(alpha, x, T::one());
	}

	#[inline]
	fn allocate_point(&self) -> Self::Point {
		B::Vector::zeros(self.n)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		B::Vector::zeros(self.n)
	}
}
