//! Optimization problems on the unit sphere S^{n-1}.
//!
//! # Problems
//!
//! - [`RayleighQuotient`] — Dominant eigenvector (min/max eigenvalue)
//! - [`MaxCutSphere`] — Max-Cut SDP relaxation (Burer-Monteiro, rank-1)
//! - [`SphericalKMeans`] — Spherical K-Means clustering

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Rayleigh Quotient
// ════════════════════════════════════════════════════════════════════════════

/// The Rayleigh quotient on the sphere S^{n-1}.
///
/// ## Mathematical Definition
///
/// ```text
/// f(x) = x^T A x,  x ∈ S^{n-1}
/// ```
///
/// Minimizing f finds the eigenvector corresponding to the smallest eigenvalue
/// of A. Maximizing (negating) finds the largest.
///
/// ## Gradient
///
/// ```text
/// ∇f(x) = 2Ax   (Euclidean)
/// grad f(x) = 2(Ax − (x^T Ax)x)   (Riemannian on S^{n-1})
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// Hess f(x)[ξ] = 2(Aξ − (x^T Aξ)x − (x^T Ax)ξ)
/// ```
///
/// The factor of 2 is kept (unlike some references that absorb it into A)
/// so that the Rayleigh quotient equals the eigenvalue at a critical point.
#[derive(Debug, Clone)]
pub struct RayleighQuotient<T: Scalar, B: LinAlgBackend<T>> {
	/// Symmetric matrix A.
	pub a: B::Matrix,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> RayleighQuotient<T, B> {
	/// Creates a Rayleigh quotient problem f(x) = x^T A x.
	pub fn new(a: B::Matrix) -> Self {
		debug_assert_eq!(a.nrows(), a.ncols(), "A must be square");
		Self {
			a,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`RayleighQuotient`].
pub struct RayleighWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Buffer for A·x (length n).
	ax: B::Vector,
	/// Buffer for A·ξ (length n) — used in HVP.
	axi: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for RayleighWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ax: B::Vector::zeros(0),
			axi: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RayleighWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RayleighWorkspace<T, B> where B::Vector: Sync {}

impl<T, B, M> Problem<T, M> for RayleighQuotient<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = RayleighWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		RayleighWorkspace {
			ax: B::Vector::zeros(n),
			axi: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		let ax = self.a.mat_vec(point);
		point.dot(&ax)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// ∇f = 2Ax
		self.a.mat_vec_into(point, &mut ws.ax);
		ws.ax.scale_mut(<T as Scalar>::from_f64(2.0));
		manifold.euclidean_to_riemannian_gradient(point, &ws.ax, result, manifold_ws);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		// Ax → ws.ax (shared)
		self.a.mat_vec_into(point, &mut ws.ax);
		let cost = point.dot(&ws.ax);

		// ∇f = 2Ax
		ws.ax.scale_mut(<T as Scalar>::from_f64(2.0));
		manifold.euclidean_to_riemannian_gradient(point, &ws.ax, gradient, manifold_ws);
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

		// Euclidean gradient: 2Ax
		self.a.mat_vec_into(point, &mut ws.ax);
		ws.ax.scale_mut(two);

		// Euclidean HVP: 2Aξ
		self.a.mat_vec_into(vector, &mut ws.axi);
		ws.axi.scale_mut(two);

		manifold.euclidean_to_riemannian_hessian(
			point,
			&ws.ax,
			&ws.axi,
			vector,
			result,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Max-Cut (Burer-Monteiro, rank-1 relaxation)
// ════════════════════════════════════════════════════════════════════════════

/// Max-Cut SDP relaxation on the sphere (Burer-Monteiro, rank-1).
///
/// ## Mathematical Definition
///
/// Given a graph with weighted adjacency/Laplacian matrix L:
///
/// ```text
/// max  ¼ Σᵢⱼ wᵢⱼ(1 − xᵢxⱼ) = ¼ (W·n − x^T W x)
/// ```
///
/// Equivalently, we minimize:
///
/// ```text
/// f(x) = x^T L x   on S^{n-1}
/// ```
///
/// where L = diag(W·1) − W is the graph Laplacian, or directly
/// f(x) = x^T W x if using the adjacency matrix W.
///
/// This is a Rayleigh quotient, but we provide a separate type for clarity
/// and to include the graph-specific construction helpers.
///
/// ## Gradient
///
/// ```text
/// grad f(x) = 2(Lx − (x^T Lx)x)
/// ```
#[derive(Debug, Clone)]
pub struct MaxCutSphere<T: Scalar, B: LinAlgBackend<T>> {
	/// The Rayleigh quotient subproblem on the Laplacian.
	inner: RayleighQuotient<T, B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MaxCutSphere<T, B> {
	/// Creates a Max-Cut problem from a graph Laplacian L.
	///
	/// The Laplacian L = D − W where D = diag(W·1).
	pub fn from_laplacian(laplacian: B::Matrix) -> Self {
		Self {
			inner: RayleighQuotient::new(laplacian),
		}
	}

	/// Creates a Max-Cut problem from an adjacency/weight matrix W.
	///
	/// Internally computes the Laplacian L = diag(W·1) − W.
	pub fn from_adjacency(w: B::Matrix) -> Self {
		let n = w.nrows();
		debug_assert_eq!(n, w.ncols(), "Adjacency matrix must be square");

		// Compute row sums (= column sums for symmetric W)
		let ones = B::Vector::from_fn(n, |_| T::one());
		let degree = w.mat_vec(&ones);

		// L = diag(degree) − W
		let mut laplacian = B::Matrix::from_diagonal(&degree);
		laplacian.sub_assign(&w);

		Self::from_laplacian(laplacian)
	}
}

impl<T, B, M> Problem<T, M> for MaxCutSphere<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = RayleighWorkspace<T, B>;

	fn create_workspace(&self, manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		<RayleighQuotient<T, B> as Problem<T, M>>::create_workspace(
			&self.inner,
			manifold,
			proto_point,
		)
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		<RayleighQuotient<T, B> as Problem<T, M>>::cost(&self.inner, point)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		<RayleighQuotient<T, B> as Problem<T, M>>::riemannian_gradient(
			&self.inner,
			manifold,
			point,
			result,
			ws,
			manifold_ws,
		);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		<RayleighQuotient<T, B> as Problem<T, M>>::cost_and_gradient(
			&self.inner,
			manifold,
			point,
			gradient,
			ws,
			manifold_ws,
		)
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
		<RayleighQuotient<T, B> as Problem<T, M>>::riemannian_hessian_vector_product(
			&self.inner,
			manifold,
			point,
			vector,
			result,
			ws,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Spherical K-Means
// ════════════════════════════════════════════════════════════════════════════

/// Workspace for [`SphericalKMeans`].
///
/// Pre-allocates the Euclidean gradient buffer so that
/// `euclidean_to_riemannian_gradient` can read from a separate source
/// without a per-iteration allocation.
pub struct SphericalKMeansWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Buffer holding the Euclidean gradient (−Σⱼ dⱼ).
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for SphericalKMeansWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for SphericalKMeansWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for SphericalKMeansWorkspace<T, B> where
	B::Vector: Sync
{
}

/// Spherical K-Means clustering on S^{n-1}.
///
/// ## Mathematical Definition
///
/// Given data points d₁, …, dₘ ∈ ℝⁿ (not necessarily on the sphere),
/// and K cluster centroids c₁, …, c_K ∈ S^{n-1}, minimize:
///
/// ```text
/// f(c₁, …, c_K) = −Σⱼ₌₁ᵐ max_k (dⱼᵀ c_k)
/// ```
///
/// This implementation handles a **single centroid** on S^{n-1}.
/// For K centroids, use a Product manifold of K spheres.
///
/// For a single centroid c ∈ S^{n-1}:
///
/// ```text
/// f(c) = −Σⱼ dⱼᵀc = −(Σⱼ dⱼ)ᵀc
/// ```
///
/// ## Gradient
///
/// ```text
/// ∇f(c) = −Σⱼ dⱼ   (Euclidean, constant)
/// grad f(c) = −Σⱼ dⱼ + (cᵀ Σⱼ dⱼ)c   (Riemannian)
/// ```
#[derive(Debug, Clone)]
pub struct SphericalKMeans<T: Scalar, B: LinAlgBackend<T>> {
	/// Sum of data points Σⱼ dⱼ (precomputed).
	data_sum: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> SphericalKMeans<T, B> {
	/// Creates a single-centroid Spherical K-Means problem from a data matrix.
	///
	/// # Arguments
	///
	/// * `data` — Data matrix D ∈ ℝᵐˣⁿ (m data points of dimension n, row-major)
	pub fn new(data: &B::Matrix) -> Self {
		let n = data.ncols();
		let m = data.nrows();
		// Compute sum of rows: data_sum = Σⱼ dⱼ = Dᵀ 1
		let ones = B::Vector::from_fn(m, |_| T::one());
		let dt = data.transpose_to_owned();
		let data_sum = dt.mat_vec(&ones);
		debug_assert_eq!(VectorView::len(&data_sum), n);
		Self {
			data_sum,
			_phantom: PhantomData,
		}
	}

	/// Creates from a precomputed data sum Σⱼ dⱼ.
	pub fn from_data_sum(data_sum: B::Vector) -> Self {
		Self {
			data_sum,
			_phantom: PhantomData,
		}
	}
}

impl<T, B, M> Problem<T, M> for SphericalKMeans<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = SphericalKMeansWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		SphericalKMeansWorkspace {
			egrad: B::Vector::zeros(VectorView::len(proto_point)),
			_phantom: PhantomData,
		}
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		// f(c) = −(Σⱼ dⱼ)ᵀ c
		-self.data_sum.dot(point)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// Euclidean gradient is constant: −Σⱼ dⱼ
		// Copy into workspace buffer, then project from ws.egrad → result
		ws.egrad.copy_from(&self.data_sum);
		ws.egrad.scale_mut(-T::one());
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
		let cost = -self.data_sum.dot(point);
		// Copy euclidean gradient into workspace buffer, then project into gradient
		ws.egrad.copy_from(&self.data_sum);
		ws.egrad.scale_mut(-T::one());
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		cost
	}
}
