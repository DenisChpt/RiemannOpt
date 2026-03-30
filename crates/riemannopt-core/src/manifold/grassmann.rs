//! # Grassmann Manifold Gr(n,p)
//!
//! The Grassmann manifold Gr(n,p) is the space of all p-dimensional linear
//! subspaces of ℝⁿ. It provides a geometric framework for problems involving
//! subspace optimization, dimensionality reduction, and invariant subspace computation.
//!
//! ## Mathematical Definition
//!
//! ```text
//! Gr(n,p) = {[Y] : Y ∈ ℝⁿˣᵖ, Y^T Y = I_p}
//! ```

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::{
	linalg::{DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorView},
	manifold::Manifold,
	types::Scalar,
};

/// The Grassmann manifold Gr(n,p) of p-dimensional subspaces in ℝⁿ.
///
/// Generic over scalar type `T` and linear algebra backend `B`.
#[derive(Clone)]
pub struct Grassmann<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	p: usize,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for Grassmann<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Grassmann Gr({}, {})", self.n, self.p)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Grassmann<T, B> {
	/// Creates a new Grassmann manifold Gr(n,p).
	///
	/// # Panics
	///
	/// Panics if `p == 0` or `p >= n`.
	pub fn new(n: usize, p: usize) -> Self {
		assert!(p > 0, "Grassmann manifold requires p > 0");
		assert!(p < n, "Grassmann manifold Gr(n,p) requires p < n");
		Self {
			n,
			p,
			_phantom: PhantomData,
		}
	}

	#[inline]
	pub fn ambient_dim(&self) -> usize {
		self.n
	}

	#[inline]
	pub fn subspace_dim(&self) -> usize {
		self.p
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Workspace Definition (Zero-Allocation Strategy)
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated memory buffers for Grassmann operations.
pub struct GrassmannWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// A small p×p buffer for Y^T Z, Y^T Y, etc.
	pub pp_mat: B::Matrix,
	/// A secondary p×p buffer (needed for SVD/Distance)
	pub pp_mat2: B::Matrix,
	/// A buffer for Y + Z (size n×p)
	pub np_mat: B::Matrix,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for GrassmannWorkspace<T, B> {
	fn default() -> Self {
		Self {
			pp_mat: B::Matrix::zeros(0, 0),
			pp_mat2: B::Matrix::zeros(0, 0),
			np_mat: B::Matrix::zeros(0, 0),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold trait implementation
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for Grassmann<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Matrix;
	type TangentVector = B::Matrix;
	type Workspace = GrassmannWorkspace<T, B>;

	#[inline]
	fn create_workspace(&self, _proto_point: &Self::Point) -> Self::Workspace {
		Self::Workspace {
			pp_mat: B::Matrix::zeros(self.p, self.p),
			pp_mat2: B::Matrix::zeros(self.p, self.p),
			np_mat: B::Matrix::zeros(self.n, self.p),
		}
	}

	#[inline]
	fn name(&self) -> &str {
		"Grassmann"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.p * (self.n - self.p)
	}

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.p {
			return false;
		}

		// Check Y^T Y = I_p
		// Note: For validation functions outside the hot path, we can afford
		// a local allocation, since we don't pass the workspace here.
		let mut yty = B::Matrix::zeros(self.p, self.p);
		yty.gemm_at(T::one(), point.as_view(), point.as_view(), T::zero());

		let mut err_sq = T::zero();
		for i in 0..self.p {
			for j in 0..self.p {
				let diff = yty.get(i, j) - if i == j { T::one() } else { T::zero() };
				err_sq = err_sq + diff * diff;
			}
		}
		<T as Float>::sqrt(err_sq) <= tol
	}

	#[inline]
	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		if vector.nrows() != self.n || vector.ncols() != self.p {
			return false;
		}
		// Horizontal space check: Y^T Z = 0
		let mut ytz = B::Matrix::zeros(self.p, self.p);
		ytz.gemm_at(T::one(), point.as_view(), vector.as_view(), T::zero());
		ytz.norm() <= tol
	}

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Use QR decomposition for projection — write directly into result
		let qr = point.qr();
		let q = qr.q();

		if q.ncols() > self.p {
			let q_trunc = q.columns_to_owned(0, self.p);
			result.copy_from(&q_trunc);
		} else {
			result.copy_from(&q);
		}

		// Fix signs for continuity
		let r = qr.r();
		for j in 0..self.p.min(r.ncols()) {
			if r.get(j, j) < T::zero() {
				for i in 0..self.n {
					*result.get_mut(i, j) = T::zero() - result.get(i, j);
				}
			}
		}
	}

	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Horizontal projection: result = Z - Y(Y^T Z)
		// 1. ws.pp_mat = Y^T Z
		ws.pp_mat
			.gemm_at(T::one(), point.as_view(), vector.as_view(), T::zero());

		// 2. result = Z - Y * ws.pp_mat
		result.copy_from(vector);
		result.gemm(-T::one(), point.as_view(), ws.pp_mat.as_view(), T::one());
	}

	#[inline]
	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> T {
		// Canonical metric: tr(U^T V) = Σ_j u_j · v_j
		let mut inner = T::zero();
		for j in 0..self.p {
			inner = inner + MatrixView::column(u, j).dot(&MatrixView::column(v, j));
		}
		inner
	}

	#[inline]
	fn norm(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T {
		<T as Float>::sqrt(self.inner_product(point, vector, vector, ws))
	}

	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		// Polar retraction via SVD: Y = X + G;  U Σ V^T = SVD(Y);  return U V^T
		// 1. ws.np_mat = X + G
		ws.np_mat.copy_from(point);
		ws.np_mat.add_assign(tangent);

		// 2. Compute SVD
		let svd = ws.np_mat.svd();

		match (svd.u, svd.vt) {
			(Some(u), Some(vt)) => {
				// 3. result = U * V^T
				// Handle thin vs full SVD matrices
				let u_trunc = if u.ncols() > self.p {
					u.columns(0, self.p)
				} else {
					u.as_view()
				};
				let vt_trunc = if vt.nrows() > self.p {
					vt.rows(0, self.p)
				} else {
					vt.as_view()
				};

				result.gemm(T::one(), u_trunc, vt_trunc, T::zero());
			}
			_ => {
				// Fallback to QR retraction if SVD fails
				self.project_point(&ws.np_mat, result);
			}
		}
	}

	#[inline]
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// log_Y(Ỹ) ≈ P_h(Ỹ - Y)
		// Compute diff in result, then project.
		result.copy_from(other);
		result.sub_assign(point);
		
		ws.pp_mat.gemm_at(T::one(), point.as_view(), result.as_view(), T::zero());
		result.gemm(-T::one(), point.as_view(), ws.pp_mat.as_view(), T::one());
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Riemannian gradient is the horizontal projection of Euclidean gradient
		self.project_tangent(point, euclidean_grad, result, ws);
	}

	#[inline]
	fn euclidean_to_riemannian_hessian(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		euclidean_hvp: &Self::TangentVector,
		tangent_vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Grassmann ehess2rhess: rhess = project(point, ehess) - ξ @ (Y^T @ egrad)

		// Step 1: result = project(point, ehess)
		self.project_tangent(point, euclidean_hvp, result, ws);

		// Step 2: ws.pp_mat = Y^T egrad
		ws.pp_mat.gemm_at(
			T::one(),
			point.as_view(),
			euclidean_grad.as_view(),
			T::zero(),
		);

		// Step 3: result -= ξ * ws.pp_mat
		result.gemm(
			-T::one(),
			tangent_vector.as_view(),
			ws.pp_mat.as_view(),
			T::one(),
		);
	}

	#[inline]
	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Projection-based vector transport: τ_Y(Z) = Π_Y(Z) = Z - Y(Y^T Z)
		self.project_tangent(to, vector, result, ws);
	}

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		// Generate random Gaussian matrix
		let mut a: B::Matrix = MatrixOps::zeros(self.n, self.p);
		for i in 0..self.n {
			for j in 0..self.p {
				*a.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// QR decomposition to get orthonormal basis
		let qr = a.qr();
		let q = qr.q();

		if q.ncols() > self.p {
			*result = q.columns_to_owned(0, self.p);
		} else {
			result.copy_from(&q);
		}
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		// Generate random matrix in result
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for i in 0..self.n {
			for j in 0..self.p {
				*result.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		// We can't pass 'result' as both input and output to project_tangent,
		// so we allocate a local temporary (not on hot path).
		let z = result.clone();
		let mut ws = self.create_workspace(point);
		self.project_tangent(point, &z, result, &mut ws);

		let norm = self.norm(point, result, &mut ws);
		if norm > T::EPSILON {
			result.scale_mut(T::one() / norm);
		}
	}

	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		// Compute Y₁^T Y₂
		let mut y1ty2 = B::Matrix::zeros(self.p, self.p);
		y1ty2.gemm_at(T::one(), x.as_view(), y.as_view(), T::zero());

		// SVD to get principal angles
		let svd = y1ty2.svd();
		let sigma = &svd.singular_values;

		// Principal angles: θᵢ = arccos(σᵢ)
		let mut dist_sq = T::zero();
		for i in 0..self.p {
			let s = sigma.get(i);
			let sigma_clamped = <T as Float>::max(<T as Float>::min(s, T::one()), -T::one());
			let theta = <T as Float>::acos(sigma_clamped);
			dist_sq = dist_sq + theta * theta;
		}

		<T as Float>::sqrt(dist_sq)
	}

	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		false
	}

	#[inline]
	fn is_flat(&self) -> bool {
		false
	}

	// ════════════════════════════════════════════════════════════════════════
	// Matrix ops
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
		// MatOps needs an element-wise axpy. We added mat_axpy in our LinAlgBackend.
		y.mat_axpy(alpha, x, T::one());
	}

	#[inline]
	fn allocate_point(&self) -> Self::Point {
		B::Matrix::zeros(self.n, self.p)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		B::Matrix::zeros(self.n, self.p)
	}
}
