//! Optimization problems on the oblique manifold OB(n,p).
//!
//! The oblique manifold is the product of p unit spheres in ℝⁿ:
//! OB(n,p) = {X ∈ ℝⁿˣᵖ : diag(X^T X) = 1} (each column has unit norm).
//!
//! # Problems
//!
//! - [`DictionaryLearning`] — Sparse dictionary learning
//! - [`ObliqueICA`] — Non-orthogonal Independent Component Analysis
//! - [`PhaseRetrieval`] — Signal recovery from magnitude measurements

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Dictionary Learning
// ════════════════════════════════════════════════════════════════════════════

/// Sparse dictionary learning on OB(n,p).
///
/// ## Mathematical Definition
///
/// Given data Y ∈ ℝⁿˣᵐ and sparse codes S ∈ ℝᵖˣᵐ (fixed), learn a dictionary
/// D ∈ OB(n,p) (columns have unit norm) minimizing:
///
/// ```text
/// f(D) = ½ ‖Y − DS‖_F²
/// ```
///
/// ## Gradient
///
/// ```text
/// ∇f(D) = −(Y − DS) Sᵀ = DS Sᵀ − Y Sᵀ
/// ```
///
/// ## Computational Notes
///
/// Precomputes SSᵀ and YSᵀ for O(np²) per gradient instead of O(npm).
#[derive(Debug, Clone)]
pub struct DictionaryLearning<T: Scalar, B: LinAlgBackend<T>> {
	/// Precomputed YSᵀ ∈ ℝⁿˣᵖ.
	yst: B::Matrix,
	/// Precomputed SSᵀ ∈ ℝᵖˣᵖ.
	sst: B::Matrix,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> DictionaryLearning<T, B> {
	/// Creates a dictionary learning problem.
	///
	/// # Arguments
	///
	/// * `y` — Data matrix Y ∈ ℝⁿˣᵐ
	/// * `codes` — Sparse codes S ∈ ℝᵖˣᵐ (typically from LASSO or OMP)
	pub fn new(y: &B::Matrix, codes: &B::Matrix) -> Self {
		let n = y.nrows();
		let p = codes.nrows();

		// YSᵀ (n×p) = Y · Sᵀ
		let mut yst = B::Matrix::zeros(n, p);
		yst.gemm_bt(T::one(), y.as_view(), codes.as_view(), T::zero());

		// SSᵀ (p×p) = S · Sᵀ
		let mut sst = B::Matrix::zeros(p, p);
		sst.gemm_bt(T::one(), codes.as_view(), codes.as_view(), T::zero());

		Self {
			yst,
			sst,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`DictionaryLearning`].
pub struct DictLearnWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Euclidean gradient (n×p).
	egrad: B::Matrix,
	/// Euclidean HVP buffer (n×p).
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for DictLearnWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for DictLearnWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for DictLearnWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for DictionaryLearning<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = DictLearnWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		DictLearnWorkspace {
			egrad: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// f(D) = ½ ‖Y − DS‖² = ½ tr(Y^T Y) − tr(D^T YSᵀ) + ½ tr(D^T D SSᵀ)
		// = const − tr(D^T YSᵀ) + ½ tr((DSSᵀ)^T D)
		let half = <T as Scalar>::from_f64(0.5);
		let lin = point.frobenius_dot(&self.yst);
		// D SSᵀ
		let mut dsst = B::Matrix::zeros(point.nrows(), point.ncols());
		dsst.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());
		let quad = half * point.frobenius_dot(&dsst);
		// const is absorbed (doesn't affect optimization)
		quad - lin
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// ∇f = DSSᵀ − YSᵀ
		ws.egrad
			.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());
		ws.egrad.sub_assign(&self.yst);
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

		// DSSᵀ → egrad
		ws.egrad
			.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());

		let quad = half * point.frobenius_dot(&ws.egrad);
		let lin = point.frobenius_dot(&self.yst);
		let cost = quad - lin;

		// ∇f = DSSᵀ − YSᵀ
		ws.egrad.sub_assign(&self.yst);
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
		// Euclidean HVP: ΞSSᵀ (constant Hessian)
		ws.ehvp
			.gemm(T::one(), vector.as_view(), self.sst.as_view(), T::zero());

		// Euclidean gradient for curvature correction
		ws.egrad
			.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());
		ws.egrad.sub_assign(&self.yst);

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
//  Oblique ICA (Non-orthogonal)
// ════════════════════════════════════════════════════════════════════════════

/// Independent Component Analysis without orthogonality on OB(n,p).
///
/// ## Mathematical Definition
///
/// Same contrast function as orthogonal ICA, but the unmixing matrix W
/// lives on OB(n,p) instead of St(n,p). Each column of W is constrained
/// to have unit norm, but columns need not be orthogonal.
///
/// ```text
/// f(W) = −(1/m) Σₖ₌₁ᵖ Σⱼ₌₁ᵐ G(sₖⱼ),  sₖⱼ = (W^T x_j)_k
/// ```
///
/// This is more general than orthogonal ICA and closer to the true
/// ICA formulation (no whitening assumption needed).
#[derive(Debug, Clone)]
pub struct ObliqueICA<T: Scalar, B: LinAlgBackend<T>> {
	/// Data matrix X ∈ ℝⁿˣᵐ (n features, m samples).
	pub data: B::Matrix,
	/// Contrast function.
	pub contrast: super::stiefel::ICAContrast,
	/// 1/m factor.
	inv_m: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> ObliqueICA<T, B> {
	/// Creates a non-orthogonal ICA problem.
	pub fn new(data: B::Matrix, contrast: super::stiefel::ICAContrast) -> Self {
		let m = data.ncols();
		Self {
			data,
			contrast,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`ObliqueICA`].
pub struct ObliqueICAWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	wtx: B::Matrix,
	egrad: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for ObliqueICAWorkspace<T, B> {
	fn default() -> Self {
		Self {
			wtx: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for ObliqueICAWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for ObliqueICAWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for ObliqueICA<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = ObliqueICAWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let p = proto_point.ncols();
		let m = self.data.ncols();
		let n = proto_point.nrows();
		ObliqueICAWorkspace {
			wtx: B::Matrix::zeros(p, m),
			egrad: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let p = point.ncols();
		let m = self.data.ncols();
		let mut wtx = B::Matrix::zeros(p, m);
		wtx.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());

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
		let p = point.ncols();
		let m = self.data.ncols();

		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());

		for k in 0..p {
			for j in 0..m {
				let s = ws.wtx.get(k, j);
				*ws.wtx.get_mut(k, j) = self.contrast.g_prime(s);
			}
		}

		ws.egrad.gemm_bt(
			-self.inv_m,
			self.data.as_view(),
			ws.wtx.as_view(),
			T::zero(),
		);
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

		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());

		let mut total = T::zero();
		for k in 0..p {
			for j in 0..m {
				let s = ws.wtx.get(k, j);
				total = total + self.contrast.g(s);
				*ws.wtx.get_mut(k, j) = self.contrast.g_prime(s);
			}
		}
		let cost = -self.inv_m * total;

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

// ════════════════════════════════════════════════════════════════════════════
//  Phase Retrieval
// ════════════════════════════════════════════════════════════════════════════

/// Phase retrieval on OB(n,p): recover signals from magnitude measurements.
///
/// ## Mathematical Definition
///
/// Given measurement vectors a₁, …, aₘ ∈ ℝⁿ and magnitude observations
/// bᵢ = |aᵢᵀ x|², recover x (up to global phase) by minimizing:
///
/// ```text
/// f(x) = (1/4m) Σᵢ (|aᵢᵀ x|² − bᵢ)²
/// ```
///
/// On OB(n,1) ≅ S^{n-1}, this removes the global scale ambiguity.
/// For multi-signal recovery, use OB(n,p) with p > 1.
///
/// ## Gradient
///
/// ```text
/// ∇f(x) = (1/m) Σᵢ (|aᵢᵀx|² − bᵢ)(aᵢᵀx) aᵢ
/// ```
///
/// ## Wirtinger Flow
///
/// This is the intensity-based formulation used in Wirtinger Flow
/// and related algorithms.
#[derive(Debug, Clone)]
pub struct PhaseRetrieval<T: Scalar, B: LinAlgBackend<T>> {
	/// Measurement matrix A ∈ ℝᵐˣⁿ (rows are measurement vectors).
	pub measurements: B::Matrix,
	/// Observed intensities bᵢ = |aᵢᵀ x*|².
	pub intensities: B::Vector,
	/// 1/m factor.
	inv_m: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> PhaseRetrieval<T, B> {
	/// Creates a phase retrieval problem.
	///
	/// # Arguments
	///
	/// * `measurements` — Measurement matrix A ∈ ℝᵐˣⁿ
	/// * `intensities` — Observed intensities bᵢ = |aᵢᵀ x*|²
	pub fn new(measurements: B::Matrix, intensities: B::Vector) -> Self {
		let m = measurements.nrows();
		debug_assert_eq!(VectorView::len(&intensities), m);
		Self {
			measurements,
			intensities,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`PhaseRetrieval`].
pub struct PhaseRetrievalWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Ax (m×p) — measurement responses.
	ax: B::Matrix,
	/// Euclidean gradient (n×p).
	egrad: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for PhaseRetrievalWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ax: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for PhaseRetrievalWorkspace<T, B> where
	B::Matrix: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for PhaseRetrievalWorkspace<T, B> where
	B::Matrix: Sync
{
}

impl<T, B, M> Problem<T, M> for PhaseRetrieval<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = PhaseRetrievalWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let m = self.measurements.nrows();
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		PhaseRetrievalWorkspace {
			ax: B::Matrix::zeros(m, p),
			egrad: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let m = self.measurements.nrows();
		let p = point.ncols();

		// AX (m×p)
		let mut ax = B::Matrix::zeros(m, p);
		ax.gemm(
			T::one(),
			self.measurements.as_view(),
			point.as_view(),
			T::zero(),
		);

		let quarter_inv_m = <T as Scalar>::from_f64(0.25) * self.inv_m;
		let mut cost = T::zero();
		for j in 0..p {
			for i in 0..m {
				let aix = ax.get(i, j);
				let residual = aix * aix - self.intensities.get(i);
				cost = cost + residual * residual;
			}
		}
		quarter_inv_m * cost
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
		let m = self.measurements.nrows();
		let p = point.ncols();

		// AX → ws.ax
		ws.ax.gemm(
			T::one(),
			self.measurements.as_view(),
			point.as_view(),
			T::zero(),
		);

		let quarter_inv_m = <T as Scalar>::from_f64(0.25) * self.inv_m;
		let mut cost = T::zero();

		// Compute cost and weighted AX simultaneously
		for j in 0..p {
			for i in 0..m {
				let aix = ws.ax.get(i, j);
				let intensity = aix * aix;
				let residual = intensity - self.intensities.get(i);
				cost = cost + residual * residual;
				// Weight: (|aᵢᵀx|² − bᵢ) · (aᵢᵀx)
				*ws.ax.get_mut(i, j) = residual * aix;
			}
		}

		// egrad = (1/m) Aᵀ (weighted AX)
		ws.egrad.gemm_at(
			self.inv_m,
			self.measurements.as_view(),
			ws.ax.as_view(),
			T::zero(),
		);

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		quarter_inv_m * cost
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> PhaseRetrieval<T, B> {
	fn compute_egrad(&self, point: &B::Matrix, ws: &mut PhaseRetrievalWorkspace<T, B>) {
		let m = self.measurements.nrows();
		let p = point.ncols();

		// AX → ws.ax
		ws.ax.gemm(
			T::one(),
			self.measurements.as_view(),
			point.as_view(),
			T::zero(),
		);

		// Weight each entry
		for j in 0..p {
			for i in 0..m {
				let aix = ws.ax.get(i, j);
				let residual = aix * aix - self.intensities.get(i);
				*ws.ax.get_mut(i, j) = residual * aix;
			}
		}

		// egrad = (1/m) Aᵀ (weighted AX)
		ws.egrad.gemm_at(
			self.inv_m,
			self.measurements.as_view(),
			ws.ax.as_view(),
			T::zero(),
		);
	}
}
