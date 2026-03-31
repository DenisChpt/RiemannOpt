//! # Fixed-Rank Manifold M_k(m,n)
//!
//! Matrices of size m×n with fixed rank k. Represented via compact SVD: X = UΣVᵀ.

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::marker::PhantomData;

use crate::{
	linalg::{DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Point & Tangent (backend-agnostic)
// ════════════════════════════════════════════════════════════════════════════

/// A point on M_k(m,n), stored as compact SVD: X = U Σ Vᵀ.
#[derive(Clone, Debug)]
pub struct FixedRankPoint<T: Scalar, B: LinAlgBackend<T>> {
	/// U ∈ St(m, k) — left singular vectors
	pub u: B::Matrix,
	/// σ ∈ ℝ₊^k — singular values
	pub s: B::Vector,
	/// V ∈ St(n, k) — right singular vectors
	pub v: B::Matrix,
}

/// A tangent vector ξ = U_⊥ M Vᵀ + U diag(Ṡ) Vᵀ + U N V_⊥ᵀ.
#[derive(Clone, Debug)]
pub struct FixedRankTangent<T: Scalar, B: LinAlgBackend<T>> {
	/// M ∈ ℝ^{(m-k)×k}
	pub u_perp_m: B::Matrix,
	/// Ṡ ∈ ℝ^k
	pub s_dot: B::Vector,
	/// N ∈ ℝ^{k×(n-k)}
	pub v_perp_n: B::Matrix,
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold struct
// ════════════════════════════════════════════════════════════════════════════

/// The fixed-rank manifold M_k(m,n).
#[derive(Clone, Debug)]
pub struct FixedRank<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	m: usize,
	n: usize,
	k: usize,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> FixedRank<T, B> {
	/// Creates M_k(m,n). Panics if k == 0 or k > min(m, n).
	pub fn new(m: usize, n: usize, k: usize) -> Self {
		assert!(k > 0 && k <= m.min(n), "Invalid rank k={k} for ({m}×{n})");
		Self {
			m,
			n,
			k,
			_phantom: PhantomData,
		}
	}

	#[inline]
	pub fn matrix_dimensions(&self) -> (usize, usize, usize) {
		(self.m, self.n, self.k)
	}

	#[inline]
	pub fn rank(&self) -> usize {
		self.k
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Workspace
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated buffers for fixed-rank operations.
pub struct FixedRankWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// m×k buffer (scratch for U QR — destroyed)
	pub tmp_mk: B::Matrix,
	/// n×k buffer (scratch for V QR — destroyed)
	pub tmp_nk: B::Matrix,
	/// k×k buffer (receives R from QR, reused)
	pub tmp_kk: B::Matrix,
	/// k-vector buffer
	pub tmp_vec_k: B::Vector,
	/// Householder factor buffer, max(bs_u, bs_v) × k
	pub qr_h: B::Matrix,
	/// Aligned scratch bytes for QR (shared for U and V)
	pub decomp_scratch: <B::Matrix as DecompositionOps<T>>::ScratchBuffer,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for FixedRankWorkspace<T, B> {
	fn default() -> Self {
		Self {
			tmp_mk: B::Matrix::zeros(0, 0),
			tmp_nk: B::Matrix::zeros(0, 0),
			tmp_kk: B::Matrix::zeros(0, 0),
			tmp_vec_k: B::Vector::zeros(0),
			qr_h: B::Matrix::zeros(0, 0),
			decomp_scratch: B::Matrix::create_qr_scratch(0, 0),
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for FixedRankWorkspace<T, B>
where
	B::Matrix: Send,
	B::Vector: Send,
	<B::Matrix as DecompositionOps<T>>::ScratchBuffer: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for FixedRankWorkspace<T, B>
where
	B::Matrix: Sync,
	B::Vector: Sync,
	<B::Matrix as DecompositionOps<T>>::ScratchBuffer: Sync,
{
}

// ════════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Clamp singular values to ε.
fn clamp_singular_values<T: Scalar + Float, B: LinAlgBackend<T>>(s: &mut B::Vector, k: usize) {
	for i in 0..k {
		if s.get(i) < T::EPSILON {
			*s.get_mut(i) = T::EPSILON;
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold impl
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for FixedRank<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = FixedRankPoint<T, B>;
	type TangentVector = FixedRankTangent<T, B>;
	type Workspace = FixedRankWorkspace<T, B>;

	fn create_workspace(&self, _proto: &Self::Point) -> Self::Workspace {
		let (bs_u, _) = B::Matrix::qr_h_factor_shape(self.m, self.k);
		let (bs_v, _) = B::Matrix::qr_h_factor_shape(self.n, self.k);
		let bs_max = bs_u.max(bs_v);

		// Scratch sized for the larger of the two QRs (m×k vs n×k).
		let dim_max = self.m.max(self.n);
		let decomp_scratch = B::Matrix::create_qr_scratch(dim_max, self.k);

		Self::Workspace {
			tmp_mk: B::Matrix::zeros(self.m, self.k),
			tmp_nk: B::Matrix::zeros(self.n, self.k),
			tmp_kk: B::Matrix::zeros(self.k, self.k),
			tmp_vec_k: B::Vector::zeros(self.k),
			qr_h: B::Matrix::zeros(bs_max, self.k),
			decomp_scratch,
		}
	}

	#[inline]
	fn name(&self) -> &str {
		"FixedRank"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.k * (self.m + self.n - self.k)
	}

	fn is_point_on_manifold(&self, point: &Self::Point, _tol: T) -> bool {
		point.u.nrows() == self.m
			&& point.u.ncols() == self.k
			&& point.v.nrows() == self.n
			&& point.v.ncols() == self.k
			&& VectorView::len(&point.s) == self.k
	}

	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_tol: T,
	) -> bool {
		vector.u_perp_m.nrows() == self.m - self.k
			&& vector.u_perp_m.ncols() == self.k
			&& VectorView::len(&vector.s_dot) == self.k
			&& vector.v_perp_n.nrows() == self.k
			&& vector.v_perp_n.ncols() == self.n - self.k
	}

	/// Re-orthogonalize U and V via QR, clamp singular values.
	///
	/// Cold path — allocates temporary buffers internally (no workspace param).
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let k = self.k;

		// ── QR for U (m×k) ───────────────────────────────────────
		let (bs_u, kk) = B::Matrix::qr_h_factor_shape(self.m, k);
		let mut u_tmp = point.u.clone();
		let mut r_tmp = B::Matrix::zeros(k, k);
		let mut h_tmp = B::Matrix::zeros(bs_u, kk);
		let mut scratch = B::Matrix::create_qr_scratch(self.m, k);
		u_tmp.qr(&mut result.u, &mut r_tmp, &mut h_tmp, &mut scratch);

		// ── QR for V (n×k) ───────────────────────────────────────
		let (bs_v, _) = B::Matrix::qr_h_factor_shape(self.n, k);
		let mut v_tmp = point.v.clone();
		if bs_v > bs_u {
			h_tmp = B::Matrix::zeros(bs_v, kk);
			scratch = B::Matrix::create_qr_scratch(self.n, k);
		}
		v_tmp.qr(&mut result.v, &mut r_tmp, &mut h_tmp, &mut scratch);

		// ── Singular values ──────────────────────────────────────
		result.s.copy_from(&point.s);
		clamp_singular_values::<T, B>(&mut result.s, k);
	}

	#[inline]
	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) {
		result.u_perp_m.copy_from(&vector.u_perp_m);
		result.s_dot.copy_from(&vector.s_dot);
		result.v_perp_n.copy_from(&vector.v_perp_n);
	}

	/// ⟨ξ, η⟩ = tr(M_ξᵀ M_η) + ⟨Ṡ_ξ, Ṡ_η⟩ + tr(N_ξᵀ N_η)
	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> T {
		let ip_m = MatrixView::frobenius_dot(&u.u_perp_m, &v.u_perp_m);
		let ip_s = u.s_dot.dot(&v.s_dot);
		let ip_n = MatrixView::frobenius_dot(&u.v_perp_n, &v.v_perp_n);
		ip_m + ip_s + ip_n
	}

	/// Simplified retraction with QR re-orthogonalization. **Zero allocation.**
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		// ── Σ_new = Σ + Ṡ ────────────────────────────────────────
		result.s.copy_from(&point.s);
		result.s.add_assign(&tangent.s_dot);
		clamp_singular_values::<T, B>(&mut result.s, self.k);

		// ── QR for U: tmp_mk ← point.u, destroyed → result.u = Q
		ws.tmp_mk.copy_from(&point.u);
		ws.tmp_mk.qr(
			&mut result.u,
			&mut ws.tmp_kk,
			&mut ws.qr_h,
			&mut ws.decomp_scratch,
		);

		// ── QR for V: tmp_nk ← point.v, destroyed → result.v = Q
		ws.tmp_nk.copy_from(&point.v);
		ws.tmp_nk.qr(
			&mut result.v,
			&mut ws.tmp_kk, // R reused, we don't need it
			&mut ws.qr_h,   // h_factor reused (sized for max)
			&mut ws.decomp_scratch,
		);
	}

	/// Approximate inverse retraction.
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) {
		result.s_dot.copy_from(&other.s);
		result.s_dot.sub_assign(&point.s);
		result.u_perp_m.fill(T::zero());
		result.v_perp_n.fill(T::zero());
	}

	/// Simplified transport: copy components.
	#[inline]
	fn parallel_transport(
		&self,
		_from: &Self::Point,
		_to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) {
		result.u_perp_m.copy_from(&vector.u_perp_m);
		result.s_dot.copy_from(&vector.s_dot);
		result.v_perp_n.copy_from(&vector.v_perp_n);
	}

	/// Cold path — allocates temporary QR buffers internally.
	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Fill U, V with random entries
		for j in 0..self.k {
			for i in 0..self.m {
				*result.u.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
			for i in 0..self.n {
				*result.v.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// QR-orthogonalize U
		let k = self.k;
		let (bs_u, kk) = B::Matrix::qr_h_factor_shape(self.m, k);
		let mut r_tmp = B::Matrix::zeros(k, k);
		let mut h_tmp = B::Matrix::zeros(bs_u, kk);
		let mut scratch = B::Matrix::create_qr_scratch(self.m, k);
		let mut u_copy = result.u.clone();
		u_copy.qr(&mut result.u, &mut r_tmp, &mut h_tmp, &mut scratch);

		// QR-orthogonalize V
		let (bs_v, _) = B::Matrix::qr_h_factor_shape(self.n, k);
		if bs_v > bs_u {
			h_tmp = B::Matrix::zeros(bs_v, kk);
			scratch = B::Matrix::create_qr_scratch(self.n, k);
		}
		let mut v_copy = result.v.clone();
		v_copy.qr(&mut result.v, &mut r_tmp, &mut h_tmp, &mut scratch);

		// Random positive singular values
		for i in 0..self.k {
			let val: f64 = normal.sample(&mut rng);
			*result.s.get_mut(i) = <T as Scalar>::from_f64(val.abs() + 1.0);
		}
	}

	fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for j in 0..self.k {
			for i in 0..(self.m - self.k) {
				*result.u_perp_m.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
			*result.s_dot.get_mut(j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			for i in 0..(self.n - self.k) {
				*result.v_perp_n.get_mut(j, i) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		let norm_sq = MatrixView::frobenius_dot(&result.u_perp_m, &result.u_perp_m)
			+ result.s_dot.dot(&result.s_dot)
			+ MatrixView::frobenius_dot(&result.v_perp_n, &result.v_perp_n);
		if norm_sq > T::zero() {
			let inv = T::one() / Float::sqrt(norm_sq);
			self.scale_tangent(inv, result);
		}
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let mut dist_sq = T::zero();
		for i in 0..self.k {
			let d = x.s.get(i) - y.s.get(i);
			dist_sq = dist_sq + d * d;
		}
		Float::sqrt(dist_sq)
	}

	#[inline]
	fn scale_tangent(&self, scalar: T, v: &mut Self::TangentVector) {
		v.u_perp_m.scale_mut(scalar);
		v.s_dot.scale_mut(scalar);
		v.v_perp_n.scale_mut(scalar);
	}

	#[inline]
	fn add_tangents(&self, v1: &mut Self::TangentVector, v2: &Self::TangentVector) {
		v1.u_perp_m.add_assign(&v2.u_perp_m);
		v1.s_dot.add_assign(&v2.s_dot);
		v1.v_perp_n.add_assign(&v2.v_perp_n);
	}

	#[inline]
	fn axpy_tangent(&self, alpha: T, x: &Self::TangentVector, y: &mut Self::TangentVector) {
		y.u_perp_m.mat_axpy(alpha, &x.u_perp_m, T::one());
		y.s_dot.axpy(alpha, &x.s_dot, T::one());
		y.v_perp_n.mat_axpy(alpha, &x.v_perp_n, T::one());
	}

	fn allocate_point(&self) -> Self::Point {
		FixedRankPoint {
			u: B::Matrix::zeros(self.m, self.k),
			s: B::Vector::zeros(self.k),
			v: B::Matrix::zeros(self.n, self.k),
		}
	}

	fn allocate_tangent(&self) -> Self::TangentVector {
		FixedRankTangent {
			u_perp_m: B::Matrix::zeros(self.m - self.k, self.k),
			s_dot: B::Vector::zeros(self.k),
			v_perp_n: B::Matrix::zeros(self.k, self.n - self.k),
		}
	}
}
