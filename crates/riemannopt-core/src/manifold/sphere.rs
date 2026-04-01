//! # Unit Sphere Manifold S^{n-1}
//!
//! The unit sphere S^{n-1} = {x ∈ ℝⁿ : ‖x‖₂ = 1} is the set of all unit vectors
//! in n-dimensional Euclidean space. It serves as a fundamental example in Riemannian
//! geometry and appears ubiquitously in optimization problems involving directional data.
//!
//! ## Mathematical Definition
//!
//! The unit sphere is formally defined as:
//! ```text
//! S^{n-1} = {x ∈ ℝⁿ : x^T x = 1}
//! ```
//!
//! It forms a compact, connected (n-1)-dimensional Riemannian submanifold of ℝⁿ.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! ```text
//! T_x S^{n-1} = {v ∈ ℝⁿ : x^T v = 0}
//! ```
//!
//! ### Riemannian Metric
//! The induced metric from the Euclidean ambient space:
//! ```text
//! g_x(u, v) = u^T v
//! ```
//!
//! ### Retraction (projection)
//! ```text
//! R_x(v) = (x + v) / ‖x + v‖
//! ```
//!
//! ### Exponential Map
//! ```text
//! exp_x(v) = cos(‖v‖)x + sin(‖v‖)(v/‖v‖)
//! ```
//!
//! ### Logarithmic Map
//! ```text
//! log_x(y) = (θ/sin θ)(y − (x^T y)x),  θ = arccos(x^T y)
//! ```
//!
//! ### Parallel Transport
//! ```text
//! Γ_{x→y}(v) = v − ((x+y)^T v / (1 + x^T y))(x + y)
//! ```
//!
//! ## Numerical Considerations
//!
//! - Taylor series for small angles (‖v‖ < √ε) to avoid sin(θ)/θ cancellation
//! - Clamped arccos to handle rounding beyond [-1, 1]
//! - Near-antipodal guard in parallel transport (denominator 1 + x^T y → 0)

use crate::{
	error::{ManifoldError, Result},
	linalg::{self, LinAlgBackend, VectorOps, VectorView},
	manifold::Manifold,
	types::Scalar,
};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

// ════════════════════════════════════════════════════════════════════════════
//  Sphere struct
// ════════════════════════════════════════════════════════════════════════════

/// The unit sphere manifold S^{n-1} = {x ∈ ℝⁿ : ‖x‖₂ = 1}.
///
/// Generic over the scalar type `T` and linear algebra backend `B`.
///
/// # Type Parameters
///
/// * `T` - The scalar type (f32 or f64)
/// * `B` - The linear algebra backend (faer, nalgebra, …)
#[derive(Clone)]
pub struct Sphere<T: Scalar = f64, B: LinAlgBackend<T> = linalg::DefaultBackend> {
	/// Ambient dimension n.
	ambient_dim: usize,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for Sphere<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Sphere(S^{})", self.ambient_dim - 1)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Sphere<T, B> {
	/// Creates a new sphere manifold S^{n-1} in ℝⁿ.
	///
	/// # Panics
	///
	/// Panics if `ambient_dim < 2`.
	#[inline]
	pub fn new(ambient_dim: usize) -> Self {
		assert!(ambient_dim >= 2, "Sphere requires ambient dimension ≥ 2");
		Self {
			ambient_dim,
			_phantom: PhantomData,
		}
	}

	/// Returns the ambient dimension n.
	#[inline]
	pub fn ambient_dimension(&self) -> usize {
		self.ambient_dim
	}

	/// Returns the intrinsic manifold dimension (n-1).
	#[inline]
	pub fn manifold_dimension(&self) -> usize {
		self.ambient_dim - 1
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Validation helpers (for tests / debug builds only)
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Sphere<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	/// Validates that a point lies on the sphere.
	///
	/// Intended for tests and `debug_assert!`. Not called in hot paths.
	pub fn check_point(&self, x: &B::Vector) -> Result<()> {
		if VectorView::len(x) != self.ambient_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.ambient_dim,
				VectorView::len(x),
			));
		}
		let norm = x.norm();
		let deviation = <T as Float>::abs(norm - T::one());
		// Adaptive: c · n · ε
		let tol = <T as Scalar>::from_f64(32.0)
			* T::EPSILON
			* <T as Scalar>::from_f64(self.ambient_dim as f64);
		if deviation > tol {
			return Err(ManifoldError::invalid_point(format!(
				"Point not on sphere: ‖x‖ = {:.6} (deviation: {}, tol: {})",
				norm, deviation, tol,
			)));
		}
		Ok(())
	}

	/// Validates that a vector lies in the tangent space at x.
	pub fn check_tangent(&self, x: &B::Vector, v: &B::Vector) -> Result<()> {
		self.check_point(x)?;
		if VectorView::len(v) != self.ambient_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.ambient_dim,
				VectorView::len(v),
			));
		}
		let inner = <T as Float>::abs(x.dot(v));
		let v_norm = v.norm();
		if v_norm < <T as Scalar>::from_f64(100.0) * <T as Float>::sqrt(T::EPSILON) {
			return Ok(()); // too small to validate
		}
		let tol = <T as Scalar>::from_f64(1e-3) * v_norm;
		if inner > tol {
			return Err(ManifoldError::invalid_tangent(format!(
				"Not in tangent space: |x^T v| = {} (‖v‖ = {}, tol: {})",
				inner, v_norm, tol,
			)));
		}
		Ok(())
	}

	/// Computes the exponential map exp_x(v) → result.
	///
	/// Uses Taylor series for small ‖v‖ to avoid sin(θ)/θ cancellation.
	///
	/// # Formula
	///
	/// - If ‖v‖ < threshold: result ≈ x(1 − ‖v‖²/2) + v(1 − ‖v‖²/6)
	/// - Otherwise: result = cos(‖v‖)x + (sin(‖v‖)/‖v‖)v
	#[inline]
	pub fn exp_map(&self, x: &B::Vector, v: &B::Vector, result: &mut B::Vector) {
		let t = v.norm();
		let threshold = T::SMALL_ANGLE_THRESHOLD;

		if t < threshold {
			let t_sq = t * t;
			let half = <T as Scalar>::from_f64(0.5);
			let sixth = <T as Scalar>::from_f64(1.0 / 6.0);

			// result = (1 − t²/2) · x + (1 − t²/6) · v
			result.copy_from(x);
			result.scale_mut(T::one() - half * t_sq);
			result.axpy(T::one() - sixth * t_sq, v, T::one());
		} else {
			let cos_t = <T as Float>::cos(t);
			let sinc_t = <T as Float>::sin(t) / t;

			// result = cos(t) · x + sinc(t) · v
			result.copy_from(x);
			result.scale_mut(cos_t);
			result.axpy(sinc_t, v, T::one());
		}
	}

	/// Computes the logarithmic map log_x(y) → result.
	///
	/// Uses `atan2` for angle computation (stable over all of [0, π])
	/// and Taylor series for the θ/sin(θ) scaling near θ ≈ 0.
	///
	/// # Formula
	///
	/// δ = y − (x^T y)x  (exactly tangent to x)
	/// θ = 2·atan2(‖x−y‖, ‖x+y‖)
	/// - If θ < threshold: result ≈ (1 + θ²/6) · δ
	/// - Otherwise: result = (θ / sin θ) · δ
	///
	/// Sets result to zero if x ≈ y. Panics (debug) if x ≈ −y.
	#[inline]
	pub fn log_map(&self, x: &B::Vector, y: &B::Vector, result: &mut B::Vector) {
		let xy_inner = x.dot(y);
		let two = <T as Scalar>::from_f64(2.0);
		let threshold = T::SMALL_ANGLE_THRESHOLD;

		// θ via atan2 — stable on the full [0, π] range
		let norm_diff = Float::sqrt(Float::max(T::zero(), (T::one() - xy_inner) * two));
		let norm_sum = Float::sqrt(Float::max(T::zero(), (T::one() + xy_inner) * two));
		let theta = two * Float::atan2(norm_diff, norm_sum);

		// Same point → zero tangent
		if theta < threshold {
			result.fill(T::zero());
			return;
		}

		// Antipodal → undefined
		debug_assert!(
			Float::abs(theta - <T as Scalar>::from_f64(std::f64::consts::PI)) >= threshold,
			"log_map: antipodal points (θ ≈ π)"
		);

		// δ = y − (x^T y)x  (exactly tangent)
		result.copy_from(y);
		result.axpy(-xy_inner, x, T::one());

		// Scale δ by θ/sin(θ), with Taylor fallback for tiny θ
		// (the branch above already handled θ < threshold → zero,
		//  so here threshold ≤ θ < π; Taylor kept for clarity)
		let sixth = <T as Scalar>::from_f64(1.0 / 6.0);
		let scale = if theta < <T as Scalar>::from_f64(1e-4) {
			T::one() + theta * theta * sixth
		} else {
			theta / Float::sin(theta)
		};
		result.scale_mut(scale);
	}

	/// Geodesic distance d(x, y) = 2·atan2(‖x−y‖, ‖x+y‖).
	///
	/// Computed from a single dot product (no allocation).
	/// Stable over the full [0, π] range, unlike arccos.
	#[inline]
	pub fn geodesic_distance(&self, x: &B::Vector, y: &B::Vector) -> T {
		let dot = x.dot(y);
		let two = <T as Scalar>::from_f64(2.0);
		let norm_diff = Float::sqrt(Float::max(T::zero(), (T::one() - dot) * two));
		let norm_sum = Float::sqrt(Float::max(T::zero(), (T::one() + dot) * two));
		two * Float::atan2(norm_diff, norm_sum)
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold trait implementation — hot path, zero alloc, no Result
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for Sphere<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Vector;
	type TangentVector = B::Vector;
	type Workspace = ();

	#[inline]
	fn name(&self) -> &str {
		"Sphere"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.ambient_dim - 1
	}

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		VectorView::len(point) == self.ambient_dim
			&& <T as Float>::abs(point.norm_squared() - T::one()) <= tol
	}

	#[inline]
	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		VectorView::len(vector) == self.ambient_dim && <T as Float>::abs(point.dot(vector)) <= tol
	}

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let norm = point.norm();
		if norm > T::EPSILON {
			result.copy_from(point);
			result.div_scalar_mut(norm);
		} else {
			// Near-zero → canonical e₁
			result.fill(T::zero());
			*result.get_mut(0) = T::one();
		}
	}

	/// P_x(v) = v − ⟨x, v⟩ x
	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		debug_assert!(
			self.is_point_on_manifold(point, T::MANIFOLD_TOLERANCE),
			"project_tangent: point not on sphere"
		);
		let inner = point.dot(vector);
		result.copy_from(vector);
		result.axpy(-inner, point, T::one());
	}

	/// ⟨u, v⟩_x = u^T v  (inherited Euclidean metric)
	#[inline]
	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> T {
		u.dot(v)
	}

	/// ‖v‖_x = ‖v‖₂  (Euclidean norm, avoids sqrt(dot) overhead of default)
	#[inline]
	fn norm(&self, _point: &Self::Point, vector: &Self::TangentVector, _ws: &mut ()) -> T {
		vector.norm()
	}

	/// Projection retraction: R_x(v) = (x + v) / ‖x + v‖
	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) {
		result.copy_from(point);
		result.add_assign(tangent);
		let norm = result.norm();
		debug_assert!(norm > T::EPSILON, "retract: zero norm after step");
		result.div_scalar_mut(norm);
	}

	/// Inverse retraction via logarithmic map.
	#[inline]
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		self.log_map(point, other, result);
	}

	/// grad f = P_x(∇f) = ∇f − ⟨x, ∇f⟩ x
	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		let inner = point.dot(euclidean_grad);
		result.copy_from(euclidean_grad);
		result.axpy(-inner, point, T::one());
	}

	/// Hess f[ξ] = P_x(∇²f[ξ]) − ⟨x, ∇f⟩ ξ   (Weingarten correction)
	#[inline]
	fn euclidean_to_riemannian_hessian(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		euclidean_hvp: &Self::TangentVector,
		tangent_vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		// Step 1: result = P_x(ehvp) = ehvp − ⟨x, ehvp⟩ x
		let inner_hvp = point.dot(euclidean_hvp);
		result.copy_from(euclidean_hvp);
		result.axpy(-inner_hvp, point, T::one());

		// Step 2: result −= ⟨x, egrad⟩ · ξ
		let inner_grad = point.dot(euclidean_grad);
		result.axpy(-inner_grad, tangent_vector, T::one());
	}

	/// Exact parallel transport: Γ_{x→y}(v) = v − ((x+y)^T v / (1 + x^T y))(x + y)
	///
	/// Falls back to copy when x ≈ y. Uses `debug_assert` for near-antipodal.
	#[inline]
	fn parallel_transport(
		&self,
		from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		let from_to_inner = from.dot(to);
		let threshold = T::SMALL_ANGLE_THRESHOLD;

		// Same point → identity transport
		if <T as Float>::abs(from_to_inner - T::one()) < threshold {
			result.copy_from(vector);
			return;
		}

		let denom = T::one() + from_to_inner;
		debug_assert!(
			denom > <T as Scalar>::from_f64(10.0) * <T as Float>::sqrt(T::EPSILON),
			"parallel_transport: near-antipodal points (1 + x^T y = {})",
			denom
		);

		// (from + to)^T v = from^T v + to^T v  (no allocation for the sum)
		let w_dot_v = from.dot(vector) + to.dot(vector);
		let scale = w_dot_v / denom;

		// result = v − scale·from − scale·to
		result.copy_from(vector);
		result.axpy(-scale, from, T::one());
		result.axpy(-scale, to, T::one());
	}

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for i in 0..self.ambient_dim {
			*result.get_mut(i) = <T as Scalar>::from_f64(normal.sample(&mut rng));
		}
		let norm = result.norm();
		if norm > T::zero() {
			result.div_scalar_mut(norm);
		} else {
			// Extremely rare — set to e₁
			result.fill(T::zero());
			*result.get_mut(0) = T::one();
		}
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for i in 0..self.ambient_dim {
			*result.get_mut(i) = <T as Scalar>::from_f64(normal.sample(&mut rng));
		}

		// Project to tangent space: v ← v − ⟨x, v⟩ x
		let inner = point.dot(result);
		result.axpy(-inner, point, T::one());

		// Normalize
		let norm = result.norm();
		if norm > T::zero() {
			result.div_scalar_mut(norm);
		}
	}

	/// d(x, y) = arccos(x^T y)
	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		self.geodesic_distance(x, y)
	}

	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		true
	}

	// ════════════════════════════════════════════════════════════════════════
	// Vector ops — pure algebra, no geometry
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
		VectorOps::zeros(self.ambient_dim)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		VectorOps::zeros(self.ambient_dim)
	}
}
