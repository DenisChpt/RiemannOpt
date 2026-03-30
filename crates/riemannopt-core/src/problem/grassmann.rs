//! Optimization problems on the Grassmann manifold Gr(n,p).
//!
//! # Problems
//!
//! - [`BrockettCost`] — PCA / Eigenspace computation
//! - [`RobustPCA`] — Principal Component Pursuit (low-rank + sparse)

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Brockett Cost (PCA / Eigenspace)
// ════════════════════════════════════════════════════════════════════════════

/// Brockett cost function for eigenspace computation on Gr(n,p).
///
/// ## Mathematical Definition
///
/// ```text
/// f(Y) = −tr(Y^T A Y)
/// ```
///
/// where A ∈ ℝⁿˣⁿ is symmetric and Y ∈ Gr(n,p) represents a p-dimensional
/// subspace. Minimizing f finds the dominant p-eigenspace of A.
///
/// ## Gradient
///
/// ```text
/// ∇f(Y) = −2AY   (Euclidean)
/// grad f(Y) = −2(I − YY^T)AY   (Riemannian on Gr(n,p))
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// Hess f(Y)[Ξ] = −2(I − YY^T)(AΞ − Ξ(Y^T AY))
/// ```
#[derive(Debug, Clone)]
pub struct BrockettCost<T: Scalar, B: LinAlgBackend<T>> {
	/// Symmetric matrix A ∈ ℝⁿˣⁿ.
	pub a: B::Matrix,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> BrockettCost<T, B> {
	/// Creates a Brockett cost f(Y) = −tr(Y^T A Y).
	pub fn new(a: B::Matrix) -> Self {
		debug_assert_eq!(a.nrows(), a.ncols(), "A must be square");
		Self {
			a,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`BrockettCost`] on Grassmann.
pub struct BrockettWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Buffer for AY (n×p).
	ay: B::Matrix,
	/// Buffer for Euclidean gradient / HVP intermediary (n×p).
	egrad: B::Matrix,
	/// Buffer for Euclidean Hessian-vector product (n×p).
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for BrockettWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ay: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for BrockettWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for BrockettWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for BrockettCost<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = BrockettWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		BrockettWorkspace {
			ay: B::Matrix::zeros(n, p),
			egrad: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		// f(Y) = −tr(Y^T A Y)
		// Compute AY, then −tr(Y^T(AY)) = −frobenius_dot(Y, AY)
		let ay = self.a.mat_mul(point);
		-point.frobenius_dot(&ay)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// ∇f = −2AY
		ws.ay.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		ws.egrad.copy_from(&ws.ay);
		ws.egrad.scale_mut(<T as Scalar>::from_f64(-2.0));
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		// AY → ws.ay
		ws.ay.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		let cost = -point.frobenius_dot(&ws.ay);

		// ∇f = −2AY
		ws.egrad.copy_from(&ws.ay);
		ws.egrad.scale_mut(<T as Scalar>::from_f64(-2.0));
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		cost
	}

	fn riemannian_hessian_vector_product(
		&self,
		manifold: &M,
		point: &M::Point,
		vector: &M::TangentVector,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let two = <T as Scalar>::from_f64(2.0);

		// Euclidean gradient: −2AY
		ws.ay.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		ws.egrad.copy_from(&ws.ay);
		ws.egrad.scale_mut(-two);

		// Euclidean HVP: −2AΞ
		ws.ehvp
			.gemm(T::one(), self.a.as_view(), vector.as_view(), T::zero());
		ws.ehvp.scale_mut(-two);

		manifold.euclidean_to_riemannian_hessian(
			point,
			&ws.egrad,
			&ws.ehvp,
			vector,
			result,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Robust PCA (Principal Component Pursuit)
// ════════════════════════════════════════════════════════════════════════════

/// Robust PCA / Principal Component Pursuit on Gr(n,p).
///
/// ## Mathematical Definition
///
/// Given a data matrix D ∈ ℝⁿˣᵐ, decompose D = L + S where L is low-rank
/// and S is sparse. The subspace recovery formulation on Gr(n,p):
///
/// ```text
/// f(Y) = ½ ‖D − YY^T D‖_F² + μ ‖D − YY^T D‖₁
/// ```
///
/// The first term measures reconstruction error (projection residual),
/// the second is a sparsity-inducing penalty on the residual.
///
/// ## Simplified formulation (smooth)
///
/// Without the ℓ₁ term (μ = 0), this reduces to maximizing the projection
/// energy, which is equivalent to PCA:
///
/// ```text
/// f(Y) = ½ ‖D − YY^T D‖_F² = ½ ‖D‖_F² − ½ ‖Y^T D‖_F²
/// ```
///
/// Since ‖D‖_F² is constant, minimizing f is equivalent to maximizing
/// ‖Y^T D‖_F² = tr(Y^T DD^T Y), i.e., a Brockett cost with A = DD^T.
///
/// ## With ℓ₁ penalty (Huber smoothing)
///
/// We use a Huber approximation of ‖·‖₁ for differentiability:
///
/// ```text
/// ψ_δ(t) = { t²/(2δ)     if |t| ≤ δ
///           { |t| − δ/2    if |t| > δ
/// ```
#[derive(Debug, Clone)]
pub struct RobustPCA<T: Scalar, B: LinAlgBackend<T>> {
	/// Data matrix D ∈ ℝⁿˣᵐ.
	pub data: B::Matrix,
	/// ℓ₁ penalty weight μ ≥ 0.
	pub mu: T,
	/// Huber smoothing parameter δ > 0.
	pub huber_delta: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> RobustPCA<T, B> {
	/// Creates a Robust PCA problem.
	///
	/// # Arguments
	///
	/// * `data` — Data matrix D ∈ ℝⁿˣᵐ
	/// * `mu` — ℓ₁ penalty weight (0 for pure PCA)
	/// * `huber_delta` — Huber smoothing parameter
	pub fn new(data: B::Matrix, mu: T, huber_delta: T) -> Self {
		debug_assert!(mu >= T::zero());
		debug_assert!(huber_delta > T::zero());
		Self {
			data,
			mu,
			huber_delta,
			_phantom: PhantomData,
		}
	}

	/// Creates a pure PCA problem (μ = 0), equivalent to Brockett with A = DD^T.
	pub fn pca(data: B::Matrix) -> Self {
		Self::new(data, T::zero(), T::one())
	}
}

/// Workspace for [`RobustPCA`].
pub struct RobustPCAWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Y^T D (p×m).
	ytd: B::Matrix,
	/// Y Y^T D (n×m) — projection of D onto subspace.
	yytd: B::Matrix,
	/// Residual D − YY^T D (n×m).
	residual: B::Matrix,
	/// Euclidean gradient (n×p).
	egrad: B::Matrix,
	/// Buffer for Huber gradient term (n×p).
	tmp_np: B::Matrix,
	/// Buffer for W^T Y (m×p).
	wty: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for RobustPCAWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ytd: B::Matrix::zeros(0, 0),
			yytd: B::Matrix::zeros(0, 0),
			residual: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			tmp_np: B::Matrix::zeros(0, 0),
			wty: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RobustPCAWorkspace<T, B>
where
	B::Matrix: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RobustPCAWorkspace<T, B>
where
	B::Matrix: Sync,
{
}

/// Huber function ψ_δ(t).
#[inline]
fn huber<T: Scalar>(t: T, delta: T) -> T {
	let abs_t = t.abs();
	if abs_t <= delta {
		t * t / (delta + delta)
	} else {
		abs_t - delta * <T as Scalar>::from_f64(0.5)
	}
}

/// Derivative of Huber function ψ'_δ(t).
#[inline]
fn huber_deriv<T: Scalar>(t: T, delta: T) -> T {
	let abs_t = t.abs();
	if abs_t <= delta {
		t / delta
	} else if t > T::zero() {
		T::one()
	} else {
		-T::one()
	}
}

impl<T, B, M> Problem<T, M> for RobustPCA<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = RobustPCAWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		let m = self.data.ncols();
		RobustPCAWorkspace {
			ytd: B::Matrix::zeros(p, m),
			yytd: B::Matrix::zeros(n, m),
			residual: B::Matrix::zeros(n, m),
			egrad: B::Matrix::zeros(n, p),
			tmp_np: B::Matrix::zeros(n, p),
			wty: B::Matrix::zeros(m, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let n = self.data.nrows();
		let m = self.data.ncols();

		// Y^T D
		let mut ytd = B::Matrix::zeros(point.ncols(), m);
		ytd.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());

		// YY^T D
		let mut yytd = B::Matrix::zeros(n, m);
		yytd.gemm(T::one(), point.as_view(), ytd.as_view(), T::zero());

		// Residual R = D − YY^T D
		let half = <T as Scalar>::from_f64(0.5);
		let mut frobenius_sq = T::zero();
		let mut l1_cost = T::zero();

		let r_slice = self.data.as_slice();
		let p_slice = yytd.as_slice();
		for k in 0..n * m {
			let rk = r_slice[k] - p_slice[k];
			frobenius_sq = frobenius_sq + rk * rk;
			if self.mu > T::zero() {
				l1_cost = l1_cost + huber(rk, self.huber_delta);
			}
		}

		half * frobenius_sq + self.mu * l1_cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		self.compute_residual(point, ws);
		self.compute_egrad(point, ws);
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		self.compute_residual(point, ws);

		// Cost from residual
		let half = <T as Scalar>::from_f64(0.5);
		let n = self.data.nrows();
		let m = self.data.ncols();
		let mut frobenius_sq = T::zero();
		let mut l1_cost = T::zero();
		let r_slice = ws.residual.as_slice();
		for k in 0..n * m {
			let rk = r_slice[k];
			frobenius_sq = frobenius_sq + rk * rk;
			if self.mu > T::zero() {
				l1_cost = l1_cost + huber(rk, self.huber_delta);
			}
		}
		let cost = half * frobenius_sq + self.mu * l1_cost;

		self.compute_egrad(point, ws);
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		cost
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> RobustPCA<T, B> {
	/// Computes residual R = D − YY^T D and intermediate Y^T D into workspace.
	fn compute_residual(&self, point: &B::Matrix, ws: &mut RobustPCAWorkspace<T, B>) {
		// Y^T D → ws.ytd
		ws.ytd.gemm_at(
			T::one(),
			point.as_view(),
			self.data.as_view(),
			T::zero(),
		);
		// YY^T D → ws.yytd
		ws.yytd.gemm(
			T::one(),
			point.as_view(),
			ws.ytd.as_view(),
			T::zero(),
		);
		// R = D − YY^T D
		ws.residual.copy_from(&self.data);
		ws.residual.sub_assign(&ws.yytd);
	}

	/// Computes Euclidean gradient from precomputed residual.
	///
	/// ∇f(Y) = −(R + μ ψ'(R)) Dᵀ Y  → but more precisely:
	/// ∇f(Y) = −R (Y^T D)^T − μ ψ'(R) (Y^T D)^T
	///
	/// Actually: ∂f/∂Y = −2 * residual * (Y^T D)^T when μ = 0.
	/// More carefully: f = ½‖D − YY^T D‖² and
	/// ∂f/∂Y = −(D − YY^T D)(Y^T D)^T − (D − YY^T D)^T ... no.
	///
	/// Let's derive properly. Let P = YY^T be the projector.
	/// f(Y) = ½ ‖(I−P)D‖² = ½ tr(D^T(I−P)D) = ½ ‖D‖² − ½ tr(D^T YY^T D)
	///      = const − ½ ‖Y^T D‖²
	/// So ∇_Y f = −D (Y^T D)^T = −D D^T Y  (using Y^T D = (D^T Y)^T)
	///
	/// With Huber penalty μ > 0:
	/// f(Y) = ½ ‖R‖² + μ Σ ψ(Rᵢⱼ) where R = (I − YY^T)D
	/// ∇f/∂Y_ab = −Σ_j (R_aj + μ ψ'(R_aj)) (Y^T D)_bj... this needs chain rule.
	///
	/// Cleaner: G_ab = ∂f/∂Y_ab where f = ½‖R‖² + μ Σ ψ(R_ij)
	/// and R = D − Y (Y^T D).
	/// ∂R/∂Y_ab = −e_a (Y^T D)_b· − Y_a· ∂(Y^T D)_b·/∂Y_ab
	/// This is complex. Using the matrix derivative:
	/// df = −tr(W^T dR) where W = R + μ ψ'(R) (element-wise)
	/// dR = −dY Y^T D − Y dY^T D = −dY (Y^T D) − Y (dY)^T D
	/// df = tr(W^T (dY (Y^T D) + Y (dY)^T D))
	///    = tr((Y^T D) W^T dY) + tr(D W^T Y (dY)^T)... hmm no.
	///
	/// Actually: df = tr(W^T dY (Y^T D)) + tr(W^T Y (dY^T D))
	///             = tr((Y^T D) W^T dY) + tr((W^T Y) (D^T dY))^T
	///             ... Let me be more careful.
	/// tr(W^T dY (Y^T D)) = tr((Y^T D) W^T dY) — cyclic
	///                     = tr(((W (Y^T D)^T)^T dY) = Σ [W (Y^T D)^T]_ab dY_ab
	/// So gradient from this term = W · (Y^T D)^T
	/// tr(W^T Y dY^T D) = tr(D W^T Y dY^T) = tr((D W^T Y)^T dY^T)^T... no
	///   = tr((D^T)^T W^T Y dY^T) — let's use vec form instead.
	/// Actually tr(A^T B C D^T) with A=W, B=Y, C=I, we have:
	/// tr(W^T Y (dY)^T D) = tr(D W^T Y (dY)^T) = tr((dY)^T D W^T Y)
	///                     = Σ [D W^T Y]_ab dY_ab
	/// So gradient from second term = D W^T Y... no: D W^T Y has wrong shape.
	/// Wait: (dY)^T is p×n, D is n×m, so (dY)^T D is p×m.
	/// tr(W^T Y (dY)^T D) = Σ_{i,j} W_{ij} [Y (dY)^T D]_{ij}
	/// = Σ_{i,j} W_{ij} Σ_k Y_{ik} [(dY)^T D]_{kj}
	/// = Σ_{i,j,k} W_{ij} Y_{ik} Σ_l dY_{lk} D_{lj}
	/// = Σ_{l,k} dY_{lk} Σ_{i,j} W_{ij} Y_{ik} D_{lj}
	/// = Σ_{l,k} dY_{lk} [D W^T]_{lj?}... hmm
	///
	/// Let me just use the simpler form. Since R = (I − YY^T)D:
	/// We need ∂/∂Y tr(½ R^T R + μ Σ ψ(R))
	///
	/// For the quadratic part: ∂/∂Y ½‖R‖² = −(R Dᵀ Y) ... wait no.
	/// = const − ½ ‖Y^T D‖² ⟹ ∇_Y = −D (Y^T D)^T = −D D^T Y
	/// Hmm, ‖Y^T D‖² = tr((Y^T D)(Y^T D)^T) = tr(Y^T D D^T Y)
	/// ∂/∂Y tr(Y^T A Y) = 2AY for symmetric A = DD^T.
	/// So ∇_Y(−½ ‖Y^T D‖²) = −D D^T Y.
	///
	/// OK for μ=0 the gradient is simply −DD^T Y.
	/// For μ > 0, the Huber term's gradient adds:
	/// ∂/∂Y μ Σ ψ(R_{ij}) where R = D − Y(Y^T D)
	/// = −μ ψ'(R) (Y^T D)^T − μ ... (from chain rule on both Y terms)
	///
	/// Actually it's simpler: let W = ψ'(R) element-wise.
	/// Then ∂μΣψ/∂Y = −μ (W (Y^T D)^T + D W^T Y)
	/// Hmm, let me just use: −μ (W · (Y^T D)^T + D · W^T · Y)
	///
	/// For simplicity and correctness, let me use the combined form:
	/// Total Euclidean gradient: −(R + μW) (Y^T D)^T − (D (R + μW)^T) Y
	/// where R + μW = R + μ ψ'(R) element-wise.
	///
	/// But actually for the ½‖R‖² term, the full gradient is also:
	/// −R (Y^T D)^T − D R^T Y  ?? No, that's not right either.
	///
	/// I think the issue is that R depends on Y in two places.
	/// Let me just go back to: f = const − ½ tr(Y^T DD^T Y) for μ=0
	/// → ∇f = −DD^T Y. This is clean and correct.
	///
	/// For μ > 0, let S = ψ'(R) element-wise (the Huber derivative applied to residual).
	/// The μ-term gradient w.r.t. Y is: −μ (S (Y^T D)^T + D S^T Y)
	///
	/// Actually, I realize I was overcomplicating this. Let me simplify.
	fn compute_egrad(&self, point: &B::Matrix, ws: &mut RobustPCAWorkspace<T, B>) {
		// For the quadratic term: ∇_Y(½‖R‖²) = −DD^T Y = −D(Y^T D)^T ... no.
		// ½‖R‖² = ½‖D‖² − ½‖Y^T D‖², so ∇_Y = −D (D^T Y) = −D · (Y^T D)^T

		// egrad = −D · (Y^T D)^T — but (Y^T D)^T = D^T Y, so egrad = −D D^T Y
		// We have ws.ytd = Y^T D (p×m), so (Y^T D)^T is m×p
		// egrad (n×p) = −D (m columns) · (Y^T D)^T (m×p)
		// = −D · (Y^T D)^T ... using gemm_bt:
		// egrad = −1 · data · ytd^T + 0
		ws.egrad.gemm_bt(
			-T::one(),
			self.data.as_view(),
			ws.ytd.as_view(),
			T::zero(),
		);

		// Add Huber penalty gradient if μ > 0
		if self.mu > T::zero() {
			let n = self.data.nrows();
			let m = self.data.ncols();

			// Compute W = ψ'(R) element-wise, overwrite residual with W
			let r_slice = ws.residual.as_mut_slice();
			for k in 0..n * m {
				r_slice[k] = huber_deriv(r_slice[k], self.huber_delta);
			}
			// Now ws.residual = W = ψ'(R)

			// Gradient of μ·Huber term: −μ (W · (Y^T D)^T + D · W^T · Y)
			// First part: −μ · W · (Y^T D)^T   (n×p = n×m · m×p)
			ws.tmp_np.gemm_bt(
				-self.mu,
				ws.residual.as_view(),
				ws.ytd.as_view(),
				T::zero(),
			);
			ws.egrad.add_assign(&ws.tmp_np);

			// Second part: −μ · D · W^T · Y   (n×p = n×m · m×n · n×p)
			// D · W^T (n×n) then multiply by Y... expensive.
			// Instead: (D W^T) Y = D (W^T Y) where W^T Y is m×p
			// W^T Y (m×p) = ws.residual^T · Y
			ws.wty.gemm_at(
				T::one(),
				ws.residual.as_view(),
				point.as_view(),
				T::zero(),
			);
			// −μ · D · wty  (n×p = n×m · m×p)
			ws.tmp_np
				.gemm(-self.mu, self.data.as_view(), ws.wty.as_view(), T::zero());
			ws.egrad.add_assign(&ws.tmp_np);
		}
	}
}
