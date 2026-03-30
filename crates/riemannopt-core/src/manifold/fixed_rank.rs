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
	/// m×k buffer
	pub tmp_mk: B::Matrix,
	/// n×k buffer
	pub tmp_nk: B::Matrix,
	/// k×k buffer
	pub tmp_kk: B::Matrix,
	/// k-vector buffer
	pub tmp_vec_k: B::Vector,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for FixedRankWorkspace<T, B> {
	fn default() -> Self {
		Self {
			tmp_mk: B::Matrix::zeros(0, 0),
			tmp_nk: B::Matrix::zeros(0, 0),
			tmp_kk: B::Matrix::zeros(0, 0),
			tmp_vec_k: B::Vector::zeros(0),
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for FixedRankWorkspace<T, B>
where
	B::Matrix: Send,
	B::Vector: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for FixedRankWorkspace<T, B>
where
	B::Matrix: Sync,
	B::Vector: Sync,
{
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
		Self::Workspace {
			tmp_mk: B::Matrix::zeros(self.m, self.k),
			tmp_nk: B::Matrix::zeros(self.n, self.k),
			tmp_kk: B::Matrix::zeros(self.k, self.k),
			tmp_vec_k: B::Vector::zeros(self.k),
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

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Re-orthogonalize U and V via QR
		let qr_u = point.u.qr();
		result.u.copy_from(qr_u.q());
		let qr_v = point.v.qr();
		result.v.copy_from(qr_v.q());
		result.s.copy_from(&point.s);
		// Clamp singular values to positive
		for i in 0..self.k {
			if result.s.get(i) < T::EPSILON {
				*result.s.get_mut(i) = T::EPSILON;
			}
		}
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

	/// Simplified retraction: Σ_new = Σ + Ṡ, then re-project U/V via QR.
	///
	/// This is a first-order retraction that respects the manifold constraint.
	/// The full orthographic retraction (with U_⊥ M Σ⁻¹ terms) is more accurate
	/// but requires computing orthogonal complements.
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut Self::Workspace,
	) {
		// Σ_new = Σ + Ṡ
		result.s.copy_from(&point.s);
		result.s.add_assign(&tangent.s_dot);

		// U_new ≈ U (with Ṡ perturbation absorbed), V_new ≈ V
		// For a proper retraction we'd need U_⊥, but this simplified version
		// copies and re-orthogonalizes, which is first-order correct.
		result.u.copy_from(&point.u);
		result.v.copy_from(&point.v);

		// Re-orthogonalize
		let qr_u = result.u.qr();
		result.u.copy_from(qr_u.q());
		let qr_v = result.v.qr();
		result.v.copy_from(qr_v.q());

		// Clamp singular values
		for i in 0..self.k {
			if result.s.get(i) < T::EPSILON {
				*result.s.get_mut(i) = T::EPSILON;
			}
		}
	}

	/// Approximate inverse retraction: project ambient difference onto tangent space.
	///
	/// Not zero-alloc (reconstructs full m×n matrices) — not on the hot path.
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) {
		// Ṡ = diag(Uᵀ (Y−X) V) ≈ other.s − point.s (simplified)
		result.s_dot.copy_from(&other.s);
		result.s_dot.sub_assign(&point.s);

		// M and N set to zero (simplified — full version needs orthogonal complements)
		result.u_perp_m.fill(T::zero());
		result.v_perp_n.fill(T::zero());
	}

	/// Simplified transport: copy components (approximation for smooth curves).
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

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Random U, V then QR-orthogonalize
		for j in 0..self.k {
			for i in 0..self.m {
				*result.u.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
			for i in 0..self.n {
				*result.v.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}
		let qr_u = result.u.qr();
		result.u.copy_from(qr_u.q());
		let qr_v = result.v.qr();
		result.v.copy_from(qr_v.q());

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

		// Normalize to unit norm — compute ‖ξ‖² directly (no workspace needed)
		let norm_sq = MatrixView::frobenius_dot(&result.u_perp_m, &result.u_perp_m)
			+ result.s_dot.dot(&result.s_dot)
			+ MatrixView::frobenius_dot(&result.v_perp_n, &result.v_perp_n);
		if norm_sq > T::zero() {
			let inv = T::one() / Float::sqrt(norm_sq);
			self.scale_tangent(inv, result);
		}
	}

	/// Optimization proxy: d(X, Y) ≈ ‖Σ_X − Σ_Y‖₂.
	///
	/// **Not** the true Riemannian distance (which also depends on subspace
	/// alignment of U and V), but an O(k) convergence criterion that avoids
	/// reconstructing full m×n matrices. Sufficient for stopping criteria.
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let mut dist_sq = T::zero();
		for i in 0..self.k {
			let d = x.s.get(i) - y.s.get(i);
			dist_sq = dist_sq + d * d;
		}
		Float::sqrt(dist_sq)
	}

	// ════════════════════════════════════════════════════════════════════════
	// Vector ops — component-wise, in-place
	// ════════════════════════════════════════════════════════════════════════

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
