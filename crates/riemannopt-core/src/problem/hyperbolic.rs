//! Optimization problems on the hyperbolic manifold ℍⁿ (Poincaré ball model).
//!
//! # Problems
//!
//! - [`PoincareEmbedding`] — Graph/tree embedding in hyperbolic space
//! - [`HyperbolicLogisticRegression`] — Logistic regression using geodesic distance

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, RealScalar, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Poincaré Embedding
// ════════════════════════════════════════════════════════════════════════════

/// Poincaré embedding for hierarchical data (graphs/trees) in ℍⁿ.
///
/// ## Mathematical Definition
///
/// Given a set of edges (i,j) with positive/negative labels, learn embeddings
/// u₁, …, u_N ∈ 𝔹ⁿ (Poincaré ball) such that connected nodes are close
/// and disconnected nodes are far apart.
///
/// For a single embedding point u, with positive targets {v⁺} and negative
/// targets {v⁻}:
///
/// ```text
/// f(u) = Σ_{v⁺} d(u, v⁺)² − α Σ_{v⁻} log(exp(−d(u, v⁻)²) + ε)
/// ```
///
/// where d(u,v) is the Poincaré ball distance:
///
/// ```text
/// d(u,v) = arccosh(1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)))
/// ```
///
/// This implementation optimizes a **single point** u given fixed targets.
/// For multi-point optimization, use a Product manifold.
#[derive(Debug, Clone)]
pub struct PoincareEmbedding<T: Scalar, B: LinAlgBackend<T>> {
	/// Positive target points (attract).
	pub positive_targets: Vec<B::Vector>,
	/// Negative target points (repel).
	pub negative_targets: Vec<B::Vector>,
	/// Repulsion strength α.
	pub alpha: T,
	/// Curvature parameter K (negative). Default: -1.
	pub curvature: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> PoincareEmbedding<T, B> {
	/// Creates a Poincaré embedding problem.
	pub fn new(
		positive_targets: Vec<B::Vector>,
		negative_targets: Vec<B::Vector>,
		alpha: T,
	) -> Self {
		Self {
			positive_targets,
			negative_targets,
			alpha,
			curvature: -T::one(),
			_phantom: PhantomData,
		}
	}

	/// Poincaré ball distance: d(u,v) = arccosh(1 + 2‖u−v‖²/((1−‖u‖²)(1−‖v‖²)))
	fn poincare_distance(&self, u: &B::Vector, v: &B::Vector) -> T {
		let diff_sq = {
			let n = VectorView::len(u);
			let mut s = T::zero();
			for i in 0..n {
				let d = u.get(i) - v.get(i);
				s = s + d * d;
			}
			s
		};
		let u_sq = u.norm_squared();
		let v_sq = v.norm_squared();
		let denom = (T::one() - u_sq) * (T::one() - v_sq);
		let two = <T as Scalar>::from_f64(2.0);

		// Clamp for numerical stability
		let arg = T::one() + two * diff_sq / denom.max(T::EPSILON);
		arg.max(T::one() + T::EPSILON).acosh()
	}

	/// Gradient of d(u,v)² w.r.t. u in the Poincaré ball (Euclidean gradient).
	///
	/// ∂d²/∂u = 2d · ∂d/∂u
	/// ∂d/∂u = (1/√(cosh²(d)−1)) · ∂(cosh⁻¹(arg))/∂u
	///
	/// For the Poincaré ball: the Euclidean gradient of the squared distance
	/// includes the conformal factor λ(u)⁻².
	fn distance_sq_egrad(&self, u: &B::Vector, v: &B::Vector, result: &mut B::Vector) {
		let n = VectorView::len(u);
		let u_sq = u.norm_squared();
		let v_sq = v.norm_squared();
		let two = <T as Scalar>::from_f64(2.0);
		let four = <T as Scalar>::from_f64(4.0);

		let alpha_u = T::one() - u_sq;
		let alpha_v = T::one() - v_sq;
		let denom = alpha_u * alpha_v;

		let mut diff_sq = T::zero();
		for i in 0..n {
			let d = u.get(i) - v.get(i);
			diff_sq = diff_sq + d * d;
		}

		let arg = T::one() + two * diff_sq / denom.max(T::EPSILON);
		let dist = arg.max(T::one() + T::EPSILON).acosh();

		if dist < T::EPSILON {
			result.fill(T::zero());
			return;
		}

		// ∂arg/∂uᵢ = (2/(α_u α_v)) · (2(uᵢ−vᵢ) · α_u α_v + 2‖u−v‖² · 2uᵢ α_v) / (α_u α_v)²
		// Simplification: ∂arg/∂uᵢ = (4/(α_u²α_v)) · ((uᵢ−vᵢ)α_u + ‖u−v‖²uᵢ)
		// hmm, let me redo this.
		// arg = 1 + 2‖u−v‖²/(α_u α_v)
		// ∂arg/∂uᵢ = 2·∂(‖u−v‖²)/(α_u α_v)/∂uᵢ
		//           = 2·(2(uᵢ−vᵢ)·α_u·α_v − ‖u−v‖²·(−2uᵢ)·α_v) / (α_u α_v)²
		//           = (4α_v / (α_u α_v)²) · ((uᵢ−vᵢ)·α_u + ‖u−v‖²·uᵢ)
		//           = 4/((α_u)²·α_v) · ((uᵢ−vᵢ)·α_u + diff_sq·uᵢ)
		let inv_sinh_d = T::one() / (arg * arg - T::one()).max(T::EPSILON).sqrt();
		let scale = two * dist * inv_sinh_d * four / (alpha_u * alpha_u * alpha_v).max(T::EPSILON);

		for i in 0..n {
			let ui = u.get(i);
			let vi = v.get(i);
			let darg_dui = (ui - vi) * alpha_u + diff_sq * ui;
			*result.get_mut(i) = *result.get_mut(i) + scale * darg_dui;
		}
	}
}

/// Workspace for [`PoincareEmbedding`].
pub struct PoincareWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	/// Temporary buffer for per-target distance gradient.
	temp_grad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for PoincareWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			temp_grad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for PoincareWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for PoincareWorkspace<T, B> where B::Vector: Sync {}

impl<T, B, M> Problem<T, M> for PoincareEmbedding<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = PoincareWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		PoincareWorkspace {
			egrad: B::Vector::zeros(n),
			temp_grad: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let mut cost = T::zero();

		// Attraction: Σ d(u, v⁺)²
		for v in &self.positive_targets {
			let d = self.poincare_distance(point, v);
			cost = cost + d * d;
		}

		// Repulsion: −α Σ log(exp(−d²) + ε)
		if self.alpha > T::zero() {
			for v in &self.negative_targets {
				let d = self.poincare_distance(point, v);
				let d_sq = d * d;
				cost = cost - self.alpha * ((-d_sq).exp() + T::EPSILON).ln();
			}
		}

		cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.egrad.fill(T::zero());

		// Attraction gradient
		for v in &self.positive_targets {
			self.distance_sq_egrad(point, v, &mut ws.egrad);
		}

		// Repulsion gradient
		if self.alpha > T::zero() {
			for v in &self.negative_targets {
				let d = self.poincare_distance(point, v);
				let d_sq = d * d;
				let weight = -self.alpha * (-d_sq).exp() / ((-d_sq).exp() + T::EPSILON);
				ws.temp_grad.fill(T::zero());
				self.distance_sq_egrad(point, v, &mut ws.temp_grad);
				ws.egrad.axpy(weight, &ws.temp_grad, T::one());
			}
		}

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Hyperbolic Logistic Regression
// ════════════════════════════════════════════════════════════════════════════

/// Logistic regression in hyperbolic space using geodesic distance.
///
/// ## Mathematical Definition
///
/// Given data points x₁, …, xₘ ∈ ℍⁿ with labels y₁, …, yₘ ∈ {−1, +1},
/// and a decision hyperplane defined by a point p ∈ ℍⁿ and a tangent
/// direction a ∈ T_p ℍⁿ, classify using signed geodesic distance:
///
/// ```text
/// f(p) = (1/m) Σᵢ log(1 + exp(−yᵢ · ⟨log_p(xᵢ), a⟩_p)) + (λ/2) ‖p‖²
/// ```
///
/// This formulation learns the decision boundary center p ∈ ℍⁿ with a fixed
/// normal direction a. The signed distance is computed via the logarithmic
/// map and inner product in the tangent space.
///
/// Simpler formulation using Euclidean proxy (Poincaré ball):
///
/// ```text
/// f(p) = (1/m) Σᵢ log(1 + exp(−yᵢ · g(p, xᵢ)))
/// ```
///
/// where g(p, xᵢ) is a score based on the hyperbolic distance.
#[derive(Debug, Clone)]
pub struct HyperbolicLogisticRegression<T: Scalar, B: LinAlgBackend<T>> {
	/// Data points in the Poincaré ball.
	pub data: Vec<B::Vector>,
	/// Labels y ∈ {−1, +1}.
	pub labels: Vec<T>,
	/// Normal direction in the tangent space at origin (fixed).
	pub normal: B::Vector,
	/// Regularization parameter λ ≥ 0.
	pub lambda: T,
	/// 1/m factor.
	inv_m: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> HyperbolicLogisticRegression<T, B> {
	/// Creates a hyperbolic logistic regression problem.
	pub fn new(data: Vec<B::Vector>, labels: Vec<T>, normal: B::Vector, lambda: T) -> Self {
		debug_assert_eq!(data.len(), labels.len());
		let m = data.len();
		Self {
			data,
			labels,
			normal,
			lambda,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`HyperbolicLogisticRegression`].
pub struct HyperbolicLRWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for HyperbolicLRWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for HyperbolicLRWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for HyperbolicLRWorkspace<T, B> where
	B::Vector: Sync
{
}

/// Numerically stable softplus: log(1 + exp(t)).
#[inline]
fn softplus<T: Scalar>(t: T) -> T {
	let abs_t = t.abs();
	abs_t.max(T::zero()) + (T::one() + (-abs_t).exp()).ln()
}

/// Logistic sigmoid σ(t) = 1/(1+exp(−t)).
#[inline]
fn sigmoid<T: Scalar>(t: T) -> T {
	if t >= T::zero() {
		T::one() / (T::one() + (-t).exp())
	} else {
		let e = t.exp();
		e / (T::one() + e)
	}
}

impl<T, B, M> Problem<T, M> for HyperbolicLogisticRegression<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = HyperbolicLRWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		HyperbolicLRWorkspace {
			egrad: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// Score: s_i = ⟨log_p(x_i), a⟩ in the ambient Euclidean space
		// Simplified: use Euclidean inner product of (x_i - p) with normal
		// scaled by conformal factor for a proxy score.
		let n = VectorView::len(point);
		let half = <T as Scalar>::from_f64(0.5);
		let mut loss = T::zero();

		for (xi, &yi) in self.data.iter().zip(&self.labels) {
			// Euclidean proxy score
			let mut score = T::zero();
			for k in 0..n {
				score = score + (xi.get(k) - point.get(k)) * self.normal.get(k);
			}
			loss = loss + softplus(-yi * score);
		}

		self.inv_m * loss + half * self.lambda * point.norm_squared()
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let n = VectorView::len(point);
		ws.egrad.fill(T::zero());

		for (xi, &yi) in self.data.iter().zip(&self.labels) {
			let mut score = T::zero();
			for k in 0..n {
				score = score + (xi.get(k) - point.get(k)) * self.normal.get(k);
			}
			// ∂loss/∂p_k = σ(−y·s) · y · a_k  (the normal direction)
			let weight = -yi * sigmoid(-yi * score);
			ws.egrad.axpy(self.inv_m * weight, &self.normal, T::one());
		}

		// Regularization
		ws.egrad.axpy(self.lambda, point, T::one());

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}
