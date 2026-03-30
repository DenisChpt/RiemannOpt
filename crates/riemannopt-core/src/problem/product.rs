//! Optimization problems on product manifolds M₁ × M₂.
//!
//! # Problems
//!
//! - [`PoseEstimation`] — Wahba's problem with translation (rotation + translation)
//! - [`CoupledFactorization`] — Coupled matrix-tensor factorization

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView},
	manifold::{product::Product, Manifold},
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Pose Estimation (Wahba's Problem with Translation)
// ════════════════════════════════════════════════════════════════════════════

/// Pose estimation on St(3,3) × ℝ³ (rotation + translation).
///
/// ## Mathematical Definition
///
/// Given 3D point correspondences {(pᵢ, qᵢ)} with weights wᵢ,
/// find rotation R ∈ SO(3) ≅ St(3,3) and translation t ∈ ℝ³:
///
/// ```text
/// min_{R, t}  ½ Σᵢ wᵢ ‖R pᵢ + t − qᵢ‖²
/// ```
///
/// ## Gradient
///
/// ```text
/// ∂f/∂R = Σᵢ wᵢ (R pᵢ + t − qᵢ) pᵢᵀ
/// ∂f/∂t = Σᵢ wᵢ (R pᵢ + t − qᵢ)
/// ```
///
/// ## Note
///
/// This is implemented for the product manifold
/// `Product<Stiefel<T, B>, Euclidean<T, B>>` where the Stiefel factor
/// represents rotations (3×3 orthogonal matrices) and the Euclidean
/// factor represents translations.
#[derive(Debug, Clone)]
pub struct PoseEstimation<T: Scalar, B: LinAlgBackend<T>> {
	/// Source points P ∈ ℝ³ˣᵐ (3D points as columns).
	pub sources: B::Matrix,
	/// Target points Q ∈ ℝ³ˣᵐ (3D points as columns).
	pub targets: B::Matrix,
	/// Weights w₁, …, wₘ.
	pub weights: B::Vector,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> PoseEstimation<T, B> {
	/// Creates a pose estimation problem.
	///
	/// # Arguments
	///
	/// * `sources` — Source 3D points P ∈ ℝ³ˣᵐ
	/// * `targets` — Target 3D points Q ∈ ℝ³ˣᵐ
	/// * `weights` — Point weights (length m)
	pub fn new(sources: B::Matrix, targets: B::Matrix, weights: B::Vector) -> Self {
		debug_assert_eq!(MatrixView::nrows(&sources), 3);
		debug_assert_eq!(MatrixView::nrows(&targets), 3);
		debug_assert_eq!(MatrixView::ncols(&sources), MatrixView::ncols(&targets));
		debug_assert_eq!(MatrixView::ncols(&sources), VectorView::len(&weights));
		Self {
			sources,
			targets,
			weights,
			_phantom: PhantomData,
		}
	}

	/// Creates with uniform weights.
	pub fn uniform(sources: B::Matrix, targets: B::Matrix) -> Self {
		let m = MatrixView::ncols(&sources);
		let w = T::one() / <T as RealScalar>::from_usize(m);
		let weights = B::Vector::from_fn(m, |_| w);
		Self::new(sources, targets, weights)
	}
}

/// Workspace for [`PoseEstimation`].
pub struct PoseWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// R·P (3×m).
	rp: B::Matrix,
	/// Residuals R·P + t·1ᵀ − Q (3×m).
	residuals: B::Matrix,
	/// Euclidean gradient w.r.t. R (3×3).
	egrad_r: B::Matrix,
	/// Euclidean gradient w.r.t. t (3).
	egrad_t: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for PoseWorkspace<T, B> {
	fn default() -> Self {
		Self {
			rp: B::Matrix::zeros(0, 0),
			residuals: B::Matrix::zeros(0, 0),
			egrad_r: B::Matrix::zeros(0, 0),
			egrad_t: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for PoseWorkspace<T, B>
where
	B::Matrix: Send,
	B::Vector: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for PoseWorkspace<T, B>
where
	B::Matrix: Sync,
	B::Vector: Sync,
{
}

impl<T, B, M1, M2> Problem<T, Product<M1, M2>> for PoseEstimation<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M1: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
	M2: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = PoseWorkspace<T, B>;

	fn create_workspace(
		&self,
		_manifold: &Product<M1, M2>,
		_proto_point: &(B::Matrix, B::Vector),
	) -> Self::Workspace {
		let m = MatrixView::ncols(&self.sources);
		PoseWorkspace {
			rp: B::Matrix::zeros(3, m),
			residuals: B::Matrix::zeros(3, m),
			egrad_r: B::Matrix::zeros(3, 3),
			egrad_t: B::Vector::zeros(3),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &(B::Matrix, B::Vector)) -> T {
		let (r, t) = point;
		let m = MatrixView::ncols(&self.sources);
		let half = <T as Scalar>::from_f64(0.5);

		// RP + t·1ᵀ − Q
		let rp = r.mat_mul(&self.sources);
		let mut cost = T::zero();
		for j in 0..m {
			let w = self.weights.get(j);
			let mut sq = T::zero();
			for i in 0..3 {
				let residual = rp.get(i, j) + t.get(i) - self.targets.get(i, j);
				sq = sq + residual * residual;
			}
			cost = cost + w * sq;
		}
		half * cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &Product<M1, M2>,
		point: &(B::Matrix, B::Vector),
		result: &mut (B::Matrix, B::Vector),
		ws: &mut Self::Workspace,
		manifold_ws: &mut (M1::Workspace, M2::Workspace),
	) {
		let (r, t) = point;
		let m = MatrixView::ncols(&self.sources);

		// RP → ws.rp
		ws.rp
			.gemm(T::one(), r.as_view(), self.sources.as_view(), T::zero());

		// Weighted residuals: wᵢ (R pᵢ + t − qᵢ)
		for j in 0..m {
			let w = self.weights.get(j);
			for i in 0..3 {
				let res = ws.rp.get(i, j) + t.get(i) - self.targets.get(i, j);
				*ws.residuals.get_mut(i, j) = w * res;
			}
		}

		// ∂f/∂R = residuals · Pᵀ  (3×3 = 3×m · m×3)
		ws.egrad_r.gemm_bt(
			T::one(),
			ws.residuals.as_view(),
			self.sources.as_view(),
			T::zero(),
		);

		// ∂f/∂t = Σⱼ residuals_j (sum of columns)
		ws.egrad_t.fill(T::zero());
		for j in 0..m {
			for i in 0..3 {
				*ws.egrad_t.get_mut(i) = ws.egrad_t.get(i) + ws.residuals.get(i, j);
			}
		}

		// Project to tangent spaces
		manifold.manifold1.euclidean_to_riemannian_gradient(
			r,
			&ws.egrad_r,
			&mut result.0,
			&mut manifold_ws.0,
		);
		manifold.manifold2.euclidean_to_riemannian_gradient(
			t,
			&ws.egrad_t,
			&mut result.1,
			&mut manifold_ws.1,
		);
	}

	fn cost_and_gradient(
		&self,
		manifold: &Product<M1, M2>,
		point: &(B::Matrix, B::Vector),
		gradient: &mut (B::Matrix, B::Vector),
		ws: &mut Self::Workspace,
		manifold_ws: &mut (M1::Workspace, M2::Workspace),
	) -> T {
		let (r, t) = point;
		let m = MatrixView::ncols(&self.sources);
		let half = <T as Scalar>::from_f64(0.5);

		ws.rp
			.gemm(T::one(), r.as_view(), self.sources.as_view(), T::zero());

		let mut cost = T::zero();
		for j in 0..m {
			let w = self.weights.get(j);
			let mut sq = T::zero();
			for i in 0..3 {
				let res = ws.rp.get(i, j) + t.get(i) - self.targets.get(i, j);
				*ws.residuals.get_mut(i, j) = w * res;
				sq = sq + res * res;
			}
			cost = cost + w * sq;
		}

		ws.egrad_r.gemm_bt(
			T::one(),
			ws.residuals.as_view(),
			self.sources.as_view(),
			T::zero(),
		);

		ws.egrad_t.fill(T::zero());
		for j in 0..m {
			for i in 0..3 {
				*ws.egrad_t.get_mut(i) = ws.egrad_t.get(i) + ws.residuals.get(i, j);
			}
		}

		manifold.manifold1.euclidean_to_riemannian_gradient(
			r,
			&ws.egrad_r,
			&mut gradient.0,
			&mut manifold_ws.0,
		);
		manifold.manifold2.euclidean_to_riemannian_gradient(
			t,
			&ws.egrad_t,
			&mut gradient.1,
			&mut manifold_ws.1,
		);

		half * cost
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Coupled Matrix-Tensor Factorization
// ════════════════════════════════════════════════════════════════════════════

/// Coupled matrix-tensor factorization on a product of Stiefel manifolds.
///
/// ## Mathematical Definition
///
/// Given a matrix Y ∈ ℝⁿˣᵐ and sharing factor A ∈ St(n,p),
/// with B ∈ St(m,p) and a diagonal core S:
///
/// ```text
/// f(A, B) = ½ ‖Y − A diag(s) Bᵀ‖_F²
/// ```
///
/// This is a low-rank factorization where A and B are constrained
/// to have orthonormal columns. The shared factor A can be coupled
/// across multiple matrices/tensors.
///
/// On the product manifold St(n,p) × St(m,p).
///
/// ## Gradient
///
/// ```text
/// ∂f/∂A = −(Y − A S Bᵀ) B S = −R B S   where R = Y − A S Bᵀ
/// ∂f/∂B = −(Y − A S Bᵀ)ᵀ A S = −Rᵀ A S
/// ```
#[derive(Debug, Clone)]
pub struct CoupledFactorization<T: Scalar, B: LinAlgBackend<T>> {
	/// Data matrix Y ∈ ℝⁿˣᵐ.
	pub data: B::Matrix,
	/// Core diagonal values s ∈ ℝᵖ.
	pub core_diag: B::Vector,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> CoupledFactorization<T, B> {
	/// Creates a coupled factorization problem.
	///
	/// # Arguments
	///
	/// * `data` — Data matrix Y ∈ ℝⁿˣᵐ
	/// * `core_diag` — Diagonal core values s ∈ ℝᵖ
	pub fn new(data: B::Matrix, core_diag: B::Vector) -> Self {
		Self {
			data,
			core_diag,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`CoupledFactorization`].
pub struct CoupledWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// A S (n×p).
	a_scaled: B::Matrix,
	/// A S Bᵀ (n×m).
	approx: B::Matrix,
	/// Residual R = Y − A S Bᵀ (n×m).
	residual: B::Matrix,
	/// Euclidean gradient w.r.t. A (n×p).
	egrad_a: B::Matrix,
	/// Euclidean gradient w.r.t. B (m×p).
	egrad_b: B::Matrix,
	/// Temporary buffer for egrad_a scale_columns (n×p).
	tmp_a: B::Matrix,
	/// Temporary buffer for egrad_b scale_columns (m×p).
	tmp_b: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for CoupledWorkspace<T, B> {
	fn default() -> Self {
		Self {
			a_scaled: B::Matrix::zeros(0, 0),
			approx: B::Matrix::zeros(0, 0),
			residual: B::Matrix::zeros(0, 0),
			egrad_a: B::Matrix::zeros(0, 0),
			egrad_b: B::Matrix::zeros(0, 0),
			tmp_a: B::Matrix::zeros(0, 0),
			tmp_b: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for CoupledWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for CoupledWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M1, M2> Problem<T, Product<M1, M2>> for CoupledFactorization<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M1: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
	M2: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = CoupledWorkspace<T, B>;

	fn create_workspace(
		&self,
		_manifold: &Product<M1, M2>,
		proto_point: &(B::Matrix, B::Matrix),
	) -> Self::Workspace {
		let (a, b) = proto_point;
		let n = MatrixView::nrows(a);
		let m = MatrixView::nrows(b);
		let p = MatrixView::ncols(a);
		CoupledWorkspace {
			a_scaled: B::Matrix::zeros(n, p),
			approx: B::Matrix::zeros(n, m),
			residual: B::Matrix::zeros(n, m),
			egrad_a: B::Matrix::zeros(n, p),
			egrad_b: B::Matrix::zeros(m, p),
			tmp_a: B::Matrix::zeros(n, p),
			tmp_b: B::Matrix::zeros(m, p),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &(B::Matrix, B::Matrix)) -> T {
		let (a, b) = point;
		let half = <T as Scalar>::from_f64(0.5);

		// A S (n×p): scale columns of A by core_diag
		let mut a_scaled = a.clone();
		a_scaled.scale_columns(a, &self.core_diag);

		// A S Bᵀ (n×m)
		let mut approx = B::Matrix::zeros(MatrixView::nrows(a), MatrixView::nrows(b));
		approx.gemm_bt(T::one(), a_scaled.as_view(), b.as_view(), T::zero());

		// R = Y − A S Bᵀ
		approx.sub_assign(&self.data);
		// ‖R‖² — but we subtracted in wrong order, so it's ‖ASBᵀ − Y‖² = same
		half * approx.frobenius_dot(&approx)
	}

	fn riemannian_gradient(
		&self,
		manifold: &Product<M1, M2>,
		point: &(B::Matrix, B::Matrix),
		result: &mut (B::Matrix, B::Matrix),
		ws: &mut Self::Workspace,
		manifold_ws: &mut (M1::Workspace, M2::Workspace),
	) {
		let (a, b) = point;

		self.compute_residual(a, b, ws);

		// ∂f/∂A = −R B S (n×p = n×m · m×p, then scale columns by s)
		ws.egrad_a
			.gemm(-T::one(), ws.residual.as_view(), b.as_view(), T::zero());
		ws.tmp_a.copy_from(&ws.egrad_a);
		ws.egrad_a.scale_columns(&ws.tmp_a, &self.core_diag);

		// ∂f/∂B = −Rᵀ A S (m×p = m×n · n×p, then scale)
		ws.egrad_b
			.gemm_at(-T::one(), ws.residual.as_view(), a.as_view(), T::zero());
		ws.tmp_b.copy_from(&ws.egrad_b);
		ws.egrad_b.scale_columns(&ws.tmp_b, &self.core_diag);

		manifold.manifold1.euclidean_to_riemannian_gradient(
			a,
			&ws.egrad_a,
			&mut result.0,
			&mut manifold_ws.0,
		);
		manifold.manifold2.euclidean_to_riemannian_gradient(
			b,
			&ws.egrad_b,
			&mut result.1,
			&mut manifold_ws.1,
		);
	}

	fn cost_and_gradient(
		&self,
		manifold: &Product<M1, M2>,
		point: &(B::Matrix, B::Matrix),
		gradient: &mut (B::Matrix, B::Matrix),
		ws: &mut Self::Workspace,
		manifold_ws: &mut (M1::Workspace, M2::Workspace),
	) -> T {
		let (a, b) = point;
		let half = <T as Scalar>::from_f64(0.5);

		self.compute_residual(a, b, ws);
		let cost = half * ws.residual.frobenius_dot(&ws.residual);

		ws.egrad_a
			.gemm(-T::one(), ws.residual.as_view(), b.as_view(), T::zero());
		ws.tmp_a.copy_from(&ws.egrad_a);
		ws.egrad_a.scale_columns(&ws.tmp_a, &self.core_diag);

		ws.egrad_b
			.gemm_at(-T::one(), ws.residual.as_view(), a.as_view(), T::zero());
		ws.tmp_b.copy_from(&ws.egrad_b);
		ws.egrad_b.scale_columns(&ws.tmp_b, &self.core_diag);

		manifold.manifold1.euclidean_to_riemannian_gradient(
			a,
			&ws.egrad_a,
			&mut gradient.0,
			&mut manifold_ws.0,
		);
		manifold.manifold2.euclidean_to_riemannian_gradient(
			b,
			&ws.egrad_b,
			&mut gradient.1,
			&mut manifold_ws.1,
		);
		cost
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> CoupledFactorization<T, B> {
	fn compute_residual(&self, a: &B::Matrix, b: &B::Matrix, ws: &mut CoupledWorkspace<T, B>) {
		// A S → a_scaled
		ws.a_scaled.scale_columns(a, &self.core_diag);

		// A S Bᵀ → approx
		ws.approx
			.gemm_bt(T::one(), ws.a_scaled.as_view(), b.as_view(), T::zero());

		// R = Y − approx
		ws.residual.copy_from(&self.data);
		ws.residual.sub_assign(&ws.approx);
	}
}
