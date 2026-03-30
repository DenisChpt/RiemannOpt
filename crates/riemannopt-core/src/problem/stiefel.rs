//! Optimization problems on the Stiefel manifold St(n,p).
//!
//! # Problems
//!
//! - [`OrthogonalProcrustes`] — Optimal orthogonal alignment
//! - [`OrthogonalICA`] — Independent Component Analysis with orthogonality
//! - [`OrderedBrockett`] — Brockett cost with eigenvalue ordering

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Orthogonal Procrustes
// ════════════════════════════════════════════════════════════════════════════

/// Orthogonal Procrustes problem on St(n,p).
///
/// ## Mathematical Definition
///
/// ```text
/// min_{X ∈ St(n,p)}  ½ ‖AX − B‖_F²
/// ```
///
/// where A ∈ ℝᵐˣⁿ and B ∈ ℝᵐˣᵖ. Finds the best orthogonal transformation
/// aligning the columns of A to B.
///
/// ## Gradient
///
/// ```text
/// ∇f(X) = Aᵀ(AX − B)   (Euclidean)
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// ∇²f · Ξ = AᵀAΞ   (Euclidean, constant)
/// ```
///
/// ## Closed-form solution
///
/// The global optimum is X* = U V^T where A^T B = U Σ V^T (SVD).
/// The Riemannian formulation is useful for constrained variants or
/// when combined with other objectives on a product manifold.
#[derive(Debug, Clone)]
pub struct OrthogonalProcrustes<T: Scalar, B: LinAlgBackend<T>> {
	/// Data matrix A ∈ ℝᵐˣⁿ.
	pub a: B::Matrix,
	/// Target matrix B ∈ ℝᵐˣᵖ.
	pub b: B::Matrix,
	/// Precomputed AᵀA ∈ ℝⁿˣⁿ (constant Hessian).
	ata: B::Matrix,
	/// Precomputed AᵀB ∈ ℝⁿˣᵖ.
	atb: B::Matrix,
}

impl<T: Scalar, B: LinAlgBackend<T>> OrthogonalProcrustes<T, B> {
	/// Creates an orthogonal Procrustes problem.
	pub fn new(a: B::Matrix, b: B::Matrix) -> Self {
		debug_assert_eq!(a.nrows(), b.nrows(), "A and B must have same number of rows");
		let mut ata = B::Matrix::zeros(a.ncols(), a.ncols());
		ata.gemm_at(T::one(), a.as_view(), a.as_view(), T::zero());
		let mut atb = B::Matrix::zeros(a.ncols(), b.ncols());
		atb.gemm_at(T::one(), a.as_view(), b.as_view(), T::zero());
		Self { a, b, ata, atb }
	}
}

/// Workspace for [`OrthogonalProcrustes`].
pub struct ProcrustesWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Euclidean gradient AᵀAX − AᵀB (n×p).
	egrad: B::Matrix,
	/// Buffer for HVP (n×p).
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for ProcrustesWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for ProcrustesWorkspace<T, B> where B::Matrix: Send
{}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for ProcrustesWorkspace<T, B> where B::Matrix: Sync
{}

impl<T, B, M> Problem<T, M> for OrthogonalProcrustes<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = ProcrustesWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		ProcrustesWorkspace {
			egrad: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// f = ½ ‖AX − B‖² = ½ tr(X^T AᵀAX) − tr(X^T AᵀB) + ½ ‖B‖²
		// = ½ tr(X^T (AᵀA) X) − tr(X^T AᵀB) + const
		let half = <T as Scalar>::from_f64(0.5);
		let mut atax = B::Matrix::zeros(self.ata.nrows(), point.ncols());
		atax.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());
		let quad = half * point.frobenius_dot(&atax);
		let lin = point.frobenius_dot(&self.atb);
		let b_norm_sq = half * self.b.frobenius_dot(&self.b);
		quad - lin + b_norm_sq
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// ∇f = AᵀAX − AᵀB
		ws.egrad
			.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());
		ws.egrad.sub_assign(&self.atb);
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
		let half = <T as Scalar>::from_f64(0.5);

		// AᵀAX → egrad
		ws.egrad
			.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());

		// cost = ½ tr(X^T AᵀAX) − tr(X^T AᵀB) + ½‖B‖²
		let quad = half * point.frobenius_dot(&ws.egrad);
		let lin = point.frobenius_dot(&self.atb);
		let b_norm_sq = half * self.b.frobenius_dot(&self.b);
		let cost = quad - lin + b_norm_sq;

		// ∇f = AᵀAX − AᵀB
		ws.egrad.sub_assign(&self.atb);
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
		// Euclidean HVP: AᵀAΞ (constant Hessian)
		ws.ehvp
			.gemm(T::one(), self.ata.as_view(), vector.as_view(), T::zero());

		// Euclidean gradient for curvature correction
		ws.egrad
			.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());
		ws.egrad.sub_assign(&self.atb);

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
//  Orthogonal ICA
// ════════════════════════════════════════════════════════════════════════════

/// Independent Component Analysis with orthogonality constraint on St(n,p).
///
/// ## Mathematical Definition
///
/// ICA seeks to find an unmixing matrix W ∈ St(n,p) that maximizes the
/// non-Gaussianity of the components s = W^T x, measured by a contrast
/// function G. We minimize:
///
/// ```text
/// f(W) = −Σₖ₌₁ᵖ E[G((W^T x)_k)]
/// ```
///
/// Common contrast functions:
/// - **LogCosh**: G(u) = log(cosh(u)) — robust, general-purpose
/// - **Exp**: G(u) = −exp(−u²/2) — good for super-Gaussian sources
/// - **Kurtosis**: G(u) = u⁴/4 — simple but sensitive to outliers
///
/// This implementation uses the sample average over the data matrix
/// X ∈ ℝⁿˣᵐ (n features, m samples).
#[derive(Debug, Clone)]
pub struct OrthogonalICA<T: Scalar, B: LinAlgBackend<T>> {
	/// Data matrix X ∈ ℝⁿˣᵐ (n features, m samples, pre-whitened).
	pub data: B::Matrix,
	/// Contrast function type.
	pub contrast: ICAContrast,
	/// 1/m factor.
	inv_m: T,
	_phantom: PhantomData<B>,
}

/// Contrast function for ICA.
#[derive(Debug, Clone, Copy)]
pub enum ICAContrast {
	/// G(u) = log(cosh(u)), G'(u) = tanh(u), G''(u) = 1 − tanh²(u)
	LogCosh,
	/// G(u) = −exp(−u²/2), G'(u) = u·exp(−u²/2)
	Exp,
	/// G(u) = u⁴/4, G'(u) = u³
	Kurtosis,
}

impl<T: Scalar, B: LinAlgBackend<T>> OrthogonalICA<T, B> {
	/// Creates an ICA problem.
	///
	/// # Arguments
	///
	/// * `data` — Pre-whitened data matrix X ∈ ℝⁿˣᵐ
	/// * `contrast` — Non-Gaussianity measure
	pub fn new(data: B::Matrix, contrast: ICAContrast) -> Self {
		let m = data.ncols();
		Self {
			data,
			contrast,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`OrthogonalICA`].
pub struct ICAWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// W^T X (p×m) — projected data.
	wtx: B::Matrix,
	/// Euclidean gradient (n×p).
	egrad: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for ICAWorkspace<T, B> {
	fn default() -> Self {
		Self {
			wtx: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for ICAWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for ICAWorkspace<T, B> where B::Matrix: Sync {}

impl ICAContrast {
	#[inline]
	pub fn g<T: Scalar>(self, u: T) -> T {
		match self {
			Self::LogCosh => u.cosh().ln(),
			Self::Exp => -(u * u * <T as Scalar>::from_f64(0.5)).exp(),
			Self::Kurtosis => u * u * u * u * <T as Scalar>::from_f64(0.25),
		}
	}

	#[inline]
	pub fn g_prime<T: Scalar>(self, u: T) -> T {
		match self {
			Self::LogCosh => u.tanh(),
			Self::Exp => u * (-(u * u) * <T as Scalar>::from_f64(0.5)).exp(),
			Self::Kurtosis => u * u * u,
		}
	}
}

impl<T, B, M> Problem<T, M> for OrthogonalICA<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = ICAWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let p = proto_point.ncols();
		let m = self.data.ncols();
		let n = proto_point.nrows();
		ICAWorkspace {
			wtx: B::Matrix::zeros(p, m),
			egrad: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let p = point.ncols();
		let m = self.data.ncols();

		// S = W^T X  (p × m)
		let mut wtx = B::Matrix::zeros(p, m);
		wtx.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());

		// f = −(1/m) Σ_k Σ_j G(S_kj)
		let mut total = T::zero();
		for k in 0..p {
			for j in 0..m {
				total = total + self.contrast.g(wtx.get(k, j));
			}
		}
		-self.inv_m * total
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
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
		let p = point.ncols();
		let m = self.data.ncols();

		// S = W^T X
		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());

		// Cost and G'(S) simultaneously
		let mut total = T::zero();
		for k in 0..p {
			for j in 0..m {
				let s = ws.wtx.get(k, j);
				total = total + self.contrast.g(s);
				*ws.wtx.get_mut(k, j) = self.contrast.g_prime(s);
			}
		}
		let cost = -self.inv_m * total;

		// ∇f = −(1/m) X G'(S)^T   (n×p = n×m · m×p, but G'(S) is p×m)
		// so ∇f = −(1/m) X · G'(S)^T where G'(S)^T is m×p
		ws.egrad.gemm_bt(
			-self.inv_m,
			self.data.as_view(),
			ws.wtx.as_view(),
			T::zero(),
		);

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		cost
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> OrthogonalICA<T, B> {
	fn compute_egrad(&self, point: &B::Matrix, ws: &mut ICAWorkspace<T, B>) {
		let p = point.ncols();
		let m = self.data.ncols();

		// S = W^T X
		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());

		// Apply G' element-wise
		for k in 0..p {
			for j in 0..m {
				let s = ws.wtx.get(k, j);
				*ws.wtx.get_mut(k, j) = self.contrast.g_prime(s);
			}
		}

		// ∇f = −(1/m) X G'(S)^T
		ws.egrad.gemm_bt(
			-self.inv_m,
			self.data.as_view(),
			ws.wtx.as_view(),
			T::zero(),
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Ordered Brockett Cost
// ════════════════════════════════════════════════════════════════════════════

/// Brockett cost with eigenvalue ordering on St(n,p).
///
/// ## Mathematical Definition
///
/// ```text
/// f(X) = −tr(X^T A X N)
/// ```
///
/// where A ∈ ℝⁿˣⁿ is symmetric and N = diag(n₁, …, nₚ) with n₁ > n₂ > … > nₚ > 0
/// are distinct positive weights that enforce ordering. At the global minimum,
/// the columns of X are the p dominant eigenvectors of A in order.
///
/// ## Gradient
///
/// ```text
/// ∇f(X) = −2AXN   (Euclidean)
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// ∇²f · Ξ = −2AΞN   (Euclidean, constant w.r.t. X)
/// ```
#[derive(Debug, Clone)]
pub struct OrderedBrockett<T: Scalar, B: LinAlgBackend<T>> {
	/// Symmetric matrix A ∈ ℝⁿˣⁿ.
	pub a: B::Matrix,
	/// Ordering weights n₁ > n₂ > … > nₚ > 0.
	pub weights: B::Vector,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> OrderedBrockett<T, B> {
	/// Creates an ordered Brockett cost.
	///
	/// # Arguments
	///
	/// * `a` — Symmetric matrix A ∈ ℝⁿˣⁿ
	/// * `weights` — Ordering weights (must be strictly decreasing and positive)
	pub fn new(a: B::Matrix, weights: B::Vector) -> Self {
		debug_assert_eq!(a.nrows(), a.ncols(), "A must be square");
		Self {
			a,
			weights,
			_phantom: PhantomData,
		}
	}

	/// Creates with default weights N = diag(p, p-1, …, 1).
	pub fn with_default_weights(a: B::Matrix, p: usize) -> Self {
		let weights = B::Vector::from_fn(p, |i| <T as RealScalar>::from_usize(p - i));
		Self::new(a, weights)
	}
}

/// Workspace for [`OrderedBrockett`].
pub struct OrderedBrockettWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// AX (n×p).
	ax: B::Matrix,
	/// Euclidean gradient −2AXN (n×p).
	egrad: B::Matrix,
	/// AΞ buffer for HVP (n×p).
	axi: B::Matrix,
	/// Euclidean HVP −2AΞN buffer (n×p).
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for OrderedBrockettWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ax: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			axi: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for OrderedBrockettWorkspace<T, B>
where
	B::Matrix: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for OrderedBrockettWorkspace<T, B>
where
	B::Matrix: Sync,
{
}

impl<T, B, M> Problem<T, M> for OrderedBrockett<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = OrderedBrockettWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		OrderedBrockettWorkspace {
			ax: B::Matrix::zeros(n, p),
			egrad: B::Matrix::zeros(n, p),
			axi: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// f(X) = −tr(X^T A X N) = −Σ_k n_k (X_k^T A X_k)
		// where X_k is column k of X.
		let p = point.ncols();
		let mut ax = B::Matrix::zeros(self.a.nrows(), p);
		ax.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		let mut cost = T::zero();
		for k in 0..p {
			let xk_ax_k = point.column_dot(k, &ax, k);
			cost = cost - self.weights.get(k) * xk_ax_k;
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
		// AX → ws.ax
		ws.ax
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());

		// ∇f = −2 AX N: scale each column k by −2 n_k
		ws.egrad.scale_columns(&ws.ax, &self.weights);
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
		let p = point.ncols();

		// AX → ws.ax
		ws.ax
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());

		// Cost: −Σ_k n_k col_k(X)^T col_k(AX)
		let mut cost = T::zero();
		for k in 0..p {
			cost = cost - self.weights.get(k) * point.column_dot(k, &ws.ax, k);
		}

		// ∇f = −2 AX N
		ws.egrad.scale_columns(&ws.ax, &self.weights);
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

		// Euclidean gradient: −2AXN
		ws.ax
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		ws.egrad.scale_columns(&ws.ax, &self.weights);
		ws.egrad.scale_mut(-two);

		// Euclidean HVP: −2AΞN
		ws.axi
			.gemm(T::one(), self.a.as_view(), vector.as_view(), T::zero());
		ws.ehvp.scale_columns(&ws.axi, &self.weights);
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
