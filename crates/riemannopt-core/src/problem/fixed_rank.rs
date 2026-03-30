//! Optimization problems on the fixed-rank manifold M_k(m,n).
//!
//! # Problems
//!
//! - [`MatrixCompletion`] — Low-rank matrix completion from partial observations
//! - [`MatrixSensing`] — Recovery from linear measurements

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::{
		fixed_rank::{FixedRankPoint, FixedRankTangent},
		Manifold,
	},
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Matrix Completion
// ════════════════════════════════════════════════════════════════════════════

/// Low-rank matrix completion from partial observations on M_k(m,n).
///
/// ## Mathematical Definition
///
/// Given observed entries {(i,j, d_{ij})} of an unknown matrix D ∈ ℝᵐˣⁿ,
/// find a rank-k matrix X = UΣVᵀ that best fits the observations:
///
/// ```text
/// f(X) = ½ Σ_{(i,j)∈Ω} (X_{ij} − d_{ij})²
/// ```
///
/// where Ω is the set of observed indices.
///
/// ## Gradient
///
/// The Euclidean gradient is the sparse matrix:
/// ```text
/// [∇f(X)]_{ij} = { X_{ij} − d_{ij}  if (i,j) ∈ Ω
///                 { 0                  otherwise
/// ```
///
/// This is then projected onto the tangent space of M_k(m,n).
#[derive(Debug, Clone)]
pub struct MatrixCompletion<T: Scalar> {
	/// Observed row indices.
	pub rows: Vec<usize>,
	/// Observed column indices.
	pub cols: Vec<usize>,
	/// Observed values d_{ij}.
	pub values: Vec<T>,
	/// Number of rows m.
	pub m: usize,
	/// Number of columns n.
	pub n: usize,
}

impl<T: Scalar> MatrixCompletion<T> {
	/// Creates a matrix completion problem.
	///
	/// # Arguments
	///
	/// * `m`, `n` — Dimensions of the target matrix
	/// * `rows`, `cols` — Observed entry indices
	/// * `values` — Observed entry values
	pub fn new(m: usize, n: usize, rows: Vec<usize>, cols: Vec<usize>, values: Vec<T>) -> Self {
		debug_assert_eq!(rows.len(), cols.len());
		debug_assert_eq!(rows.len(), values.len());
		Self {
			rows,
			cols,
			values,
			m,
			n,
		}
	}

	/// Reconstructs X_{ij} = (UΣVᵀ)_{ij} = Σ_k U_{ik} σ_k V_{jk} for a single entry.
	#[inline]
	fn reconstruct_entry<B: LinAlgBackend<T>>(
		&self,
		point: &FixedRankPoint<T, B>,
		i: usize,
		j: usize,
	) -> T {
		let rank = VectorView::len(&point.s);
		let mut val = T::zero();
		for k in 0..rank {
			val = val + point.u.get(i, k) * point.s.get(k) * point.v.get(j, k);
		}
		val
	}
}

impl<T, B, M> Problem<T, M> for MatrixCompletion<T>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = FixedRankPoint<T, B>, TangentVector = FixedRankTangent<T, B>>,
{
	type Workspace = MatrixCompletionWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let m = self.m;
		let n = self.n;
		let rank = VectorView::len(&proto_point.s);
		MatrixCompletionWorkspace {
			r_omega: B::Matrix::zeros(m, n),
			rv: B::Matrix::zeros(m, rank),
			ut_rv: B::Matrix::zeros(rank, rank),
			u_proj: B::Matrix::zeros(m, rank),
			ut_r: B::Matrix::zeros(rank, n),
			vt_proj: B::Matrix::zeros(rank, n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		let mut cost = T::zero();
		for idx in 0..self.rows.len() {
			let residual = self.reconstruct_entry::<B>(point, self.rows[idx], self.cols[idx])
				- self.values[idx];
			cost = cost + residual * residual;
		}
		half * cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// Build the sparse Euclidean gradient as a FixedRankTangent
		// by projecting element-wise residuals through the tangent space structure.
		//
		// The Euclidean gradient is R_Ω = P_Ω(UΣVᵀ − D), a sparse matrix.
		// Its tangent-space projection involves:
		//   Ṡ = Uᵀ R_Ω V,  M = U_⊥ᵀ R_Ω V,  N = Uᵀ R_Ω V_⊥
		//
		// For efficiency with sparse observations, we compute Uᵀ R_Ω and R_Ω V
		// directly from the sparse entries.

		let rank = VectorView::len(&point.s);

		// s_dot = Uᵀ R_Ω V  (k×k diagonal → k-vector)
		result.s_dot.fill(T::zero());
		for kk in 0..rank {
			let mut val = T::zero();
			for idx in 0..self.rows.len() {
				let i = self.rows[idx];
				let j = self.cols[idx];
				let residual = self.reconstruct_entry::<B>(point, i, j) - self.values[idx];
				val = val + point.u.get(i, kk) * residual * point.v.get(j, kk);
			}
			*result.s_dot.get_mut(kk) = val;
		}

		// Build dense R_Ω (sparse → dense, only observed entries)
		ws.r_omega.fill(T::zero());
		for idx in 0..self.rows.len() {
			let i = self.rows[idx];
			let j = self.cols[idx];
			let residual = self.reconstruct_entry::<B>(point, i, j) - self.values[idx];
			*ws.r_omega.get_mut(i, j) = residual;
		}

		// R_Ω V → m×k
		ws.rv
			.gemm(T::one(), ws.r_omega.as_view(), point.v.as_view(), T::zero());

		// (I − UUᵀ) R_Ω V → u_perp component
		// tmp = U (Uᵀ (R_Ω V))
		ws.ut_rv
			.gemm_at(T::one(), point.u.as_view(), ws.rv.as_view(), T::zero());
		ws.u_proj
			.gemm(T::one(), point.u.as_view(), ws.ut_rv.as_view(), T::zero());
		ws.rv.sub_assign(&ws.u_proj);
		result.u_perp_m.copy_from(&ws.rv);

		// Uᵀ R_Ω (I − VVᵀ) → v_perp component → k×n
		ws.ut_r
			.gemm_at(T::one(), point.u.as_view(), ws.r_omega.as_view(), T::zero());
		// Subtract (Uᵀ R_Ω V) Vᵀ
		ws.vt_proj
			.gemm_bt(T::one(), ws.ut_rv.as_view(), point.v.as_view(), T::zero());
		ws.ut_r.sub_assign(&ws.vt_proj);
		result.v_perp_n.copy_from(&ws.ut_r);

		let _ = manifold;
		let _ = manifold_ws;
	}
}

/// Workspace for [`MatrixCompletion`].
pub struct MatrixCompletionWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Dense sparse gradient R_Ω (m×n).
	r_omega: B::Matrix,
	/// R_Ω V (m×rank).
	rv: B::Matrix,
	/// Uᵀ R_Ω V (rank×rank).
	ut_rv: B::Matrix,
	/// U(Uᵀ R_Ω V) (m×rank).
	u_proj: B::Matrix,
	/// Uᵀ R_Ω (rank×n).
	ut_r: B::Matrix,
	/// (Uᵀ R_Ω V) Vᵀ (rank×n).
	vt_proj: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for MatrixCompletionWorkspace<T, B> {
	fn default() -> Self {
		Self {
			r_omega: B::Matrix::zeros(0, 0),
			rv: B::Matrix::zeros(0, 0),
			ut_rv: B::Matrix::zeros(0, 0),
			u_proj: B::Matrix::zeros(0, 0),
			ut_r: B::Matrix::zeros(0, 0),
			vt_proj: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for MatrixCompletionWorkspace<T, B> where
	B::Matrix: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for MatrixCompletionWorkspace<T, B> where
	B::Matrix: Sync
{
}

// ════════════════════════════════════════════════════════════════════════════
//  Matrix Sensing
// ════════════════════════════════════════════════════════════════════════════

/// Matrix sensing: recover a rank-k matrix from linear measurements.
///
/// ## Mathematical Definition
///
/// Given measurement matrices A₁, …, Aₗ ∈ ℝᵐˣⁿ and observations
/// bᵢ = ⟨Aᵢ, X⟩_F = tr(Aᵢᵀ X), find X ∈ M_k(m,n):
///
/// ```text
/// f(X) = ½ Σᵢ (⟨Aᵢ, X⟩ − bᵢ)²
/// ```
///
/// ## Gradient
///
/// ```text
/// ∇f(X) = Σᵢ (⟨Aᵢ, X⟩ − bᵢ) Aᵢ
/// ```
///
/// The dense gradient is then projected onto the tangent space.
#[derive(Debug, Clone)]
pub struct MatrixSensing<T: Scalar, B: LinAlgBackend<T>> {
	/// Measurement matrices A₁, …, Aₗ.
	pub measurements: Vec<B::Matrix>,
	/// Observed values b₁, …, bₗ.
	pub observations: Vec<T>,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MatrixSensing<T, B> {
	/// Creates a matrix sensing problem.
	pub fn new(measurements: Vec<B::Matrix>, observations: Vec<T>) -> Self {
		debug_assert_eq!(measurements.len(), observations.len());
		Self {
			measurements,
			observations,
			_phantom: PhantomData,
		}
	}

	/// Computes ⟨A, X⟩ = tr(Aᵀ X) = Σ A_ij X_ij for X = UΣVᵀ.
	fn measure(&self, a: &B::Matrix, point: &FixedRankPoint<T, B>) -> T {
		// ⟨A, UΣVᵀ⟩ = tr(Aᵀ U Σ Vᵀ) = tr(Vᵀ Aᵀ U Σ)
		// = Σ_k σ_k (Aᵀ U)_{·k}ᵀ V_{·k} = Σ_k σ_k (Vᵀ Aᵀ U)_{kk}
		let rank = VectorView::len(&point.s);
		let mut result = T::zero();

		// Compute column by column to avoid full matrix multiply
		for k in 0..rank {
			// (Aᵀ U)_{·k} = Aᵀ u_k
			let mut dot = T::zero();
			for i in 0..a.nrows() {
				for j in 0..a.ncols() {
					dot = dot + a.get(i, j) * point.u.get(i, k) * point.v.get(j, k);
				}
			}
			result = result + point.s.get(k) * dot;
		}
		result
	}
}

/// Workspace for [`MatrixSensing`].
pub struct MatrixSensingWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Dense Euclidean gradient (m×n).
	egrad: B::Matrix,
	/// ∇f V (m×rank).
	rv: B::Matrix,
	/// Uᵀ ∇f V (rank×rank).
	ut_rv: B::Matrix,
	/// U(Uᵀ ∇f V) (m×rank).
	u_proj: B::Matrix,
	/// Uᵀ ∇f (rank×n).
	ut_e: B::Matrix,
	/// (Uᵀ ∇f V) Vᵀ (rank×n).
	vt_proj: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for MatrixSensingWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			rv: B::Matrix::zeros(0, 0),
			ut_rv: B::Matrix::zeros(0, 0),
			u_proj: B::Matrix::zeros(0, 0),
			ut_e: B::Matrix::zeros(0, 0),
			vt_proj: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for MatrixSensingWorkspace<T, B> where
	B::Matrix: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for MatrixSensingWorkspace<T, B> where
	B::Matrix: Sync
{
}

impl<T, B, M> Problem<T, M> for MatrixSensing<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = FixedRankPoint<T, B>, TangentVector = FixedRankTangent<T, B>>,
{
	type Workspace = MatrixSensingWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let m = proto_point.u.nrows();
		let n = proto_point.v.nrows();
		let rank = VectorView::len(&proto_point.s);
		MatrixSensingWorkspace {
			egrad: B::Matrix::zeros(m, n),
			rv: B::Matrix::zeros(m, rank),
			ut_rv: B::Matrix::zeros(rank, rank),
			u_proj: B::Matrix::zeros(m, rank),
			ut_e: B::Matrix::zeros(rank, n),
			vt_proj: B::Matrix::zeros(rank, n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		let mut cost = T::zero();
		for (a, &b) in self.measurements.iter().zip(&self.observations) {
			let r = self.measure(a, point) - b;
			cost = cost + r * r;
		}
		half * cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let rank = VectorView::len(&point.s);

		// ∇f = Σᵢ rᵢ Aᵢ  (dense Euclidean gradient)
		ws.egrad.fill(T::zero());
		for (a, &b) in self.measurements.iter().zip(&self.observations) {
			let r = self.measure(a, point) - b;
			ws.egrad.mat_axpy(r, a, T::one());
		}

		// Project to tangent space (same approach as MatrixCompletion)
		ws.rv
			.gemm(T::one(), ws.egrad.as_view(), point.v.as_view(), T::zero());

		// s_dot = Uᵀ ∇f V
		ws.ut_rv
			.gemm_at(T::one(), point.u.as_view(), ws.rv.as_view(), T::zero());
		for k in 0..rank {
			*result.s_dot.get_mut(k) = ws.ut_rv.get(k, k);
		}

		// u_perp_m = (I − UUᵀ) ∇f V
		ws.u_proj
			.gemm(T::one(), point.u.as_view(), ws.ut_rv.as_view(), T::zero());
		ws.rv.sub_assign(&ws.u_proj);
		result.u_perp_m.copy_from(&ws.rv);

		// v_perp_n = Uᵀ ∇f (I − VVᵀ)
		ws.ut_e
			.gemm_at(T::one(), point.u.as_view(), ws.egrad.as_view(), T::zero());
		ws.vt_proj
			.gemm_bt(T::one(), ws.ut_rv.as_view(), point.v.as_view(), T::zero());
		ws.ut_e.sub_assign(&ws.vt_proj);
		result.v_perp_n.copy_from(&ws.ut_e);

		let _ = manifold;
		let _ = manifold_ws;
	}
}
