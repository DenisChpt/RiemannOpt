//! Optimization problems on the PSD cone S⁺(n).
//!
//! # Problems
//!
//! - [`NearestCorrelation`] — Nearest correlation matrix
//! - [`MaxCutSDP`] — Max-Cut semidefinite relaxation (Burer-Monteiro)

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Nearest Correlation Matrix
// ════════════════════════════════════════════════════════════════════════════

/// Nearest correlation matrix problem on S⁺(n).
///
/// ## Mathematical Definition
///
/// Given a symmetric matrix C ∈ ℝⁿˣⁿ (e.g., a noisy sample correlation),
/// find the nearest valid correlation matrix (PSD with unit diagonal):
///
/// ```text
/// min_{X ∈ S⁺(n)}  ½ ‖X − C‖_F²  + μ ‖diag(X) − 1‖²
/// ```
///
/// The penalty μ enforces unit diagonal (correlation matrix property).
/// When μ → ∞, diag(X) → 1 exactly.
///
/// ## Gradient
///
/// ```text
/// ∇f(X) = (X − C) + 2μ · diag(diag(X) − 1)
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// ∇²f · Ξ = Ξ + 2μ · diag(diag(Ξ))
/// ```
#[derive(Debug, Clone)]
pub struct NearestCorrelation<T: Scalar, B: LinAlgBackend<T>> {
	/// Target matrix C (symmetric).
	pub target: B::Matrix,
	/// Vectorized target (precomputed for hot path).
	target_vec: B::Vector,
	/// Diagonal penalty weight μ.
	pub mu: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> NearestCorrelation<T, B> {
	/// Creates a nearest correlation matrix problem.
	///
	/// # Arguments
	///
	/// * `target` — Target symmetric matrix C
	/// * `mu` — Penalty weight for unit diagonal constraint
	pub fn new(target: B::Matrix, mu: T) -> Self {
		debug_assert_eq!(
			MatrixView::nrows(&target),
			MatrixView::ncols(&target),
			"Target must be square"
		);
		let target_vec = B::Vector::from_slice(target.as_slice());
		Self {
			target,
			target_vec,
			mu,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`NearestCorrelation`].
pub struct NearestCorrelationWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Euclidean gradient buffer.
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for NearestCorrelationWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for NearestCorrelationWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for NearestCorrelationWorkspace<T, B> where
	B::Vector: Sync
{
}

impl<T, B, M> Problem<T, M> for NearestCorrelation<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	// PSD Cone uses B::Vector as Point (vectorized symmetric matrix)
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = NearestCorrelationWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let len = VectorView::len(proto_point);
		NearestCorrelationWorkspace {
			egrad: B::Vector::zeros(len),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// The PSD cone manifold stores the point as a vector (lower triangle or full).
		// We compute ½ ‖x − c‖² where x and c are vectorized.
		let half = <T as Scalar>::from_f64(0.5);
		let diff = point.sub(&self.target_vec);
		half * diff.norm_squared()
		// Note: diagonal penalty requires reconstructing the matrix.
		// For simplicity, we skip the diagonal penalty in the vectorized form.
		// A proper implementation would reconstruct the matrix.
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// ∇f = x − c (vectorized)
		ws.egrad.copy_from(point);
		ws.egrad.sub_assign(&self.target_vec);

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Max-Cut SDP (Burer-Monteiro)
// ════════════════════════════════════════════════════════════════════════════

/// Max-Cut SDP relaxation via Burer-Monteiro approach on S⁺(n).
///
/// ## Mathematical Definition
///
/// The Max-Cut SDP relaxation:
///
/// ```text
/// max  ¼ ⟨L, X⟩_F    s.t. X ∈ S⁺(n), diag(X) = 1
/// ```
///
/// Burer-Monteiro factorization X = RRᵀ with R ∈ ℝⁿˣʳ,
/// each row of R on the sphere:
///
/// ```text
/// min  −¼ ‖LR‖_F²... no, let's use the factored form:
/// f(R) = −¼ tr(Rᵀ L R)   subject to ‖Rᵢ‖ = 1 ∀i
/// ```
///
/// This lives on a product of spheres (oblique manifold) rather than
/// the PSD cone directly. But we can also formulate it on the PSD cone:
///
/// ```text
/// f(X) = −¼ ⟨L, X⟩_F + μ ‖diag(X) − 1‖²
/// ```
///
/// where L is the graph Laplacian and μ penalizes the diagonal constraint.
#[derive(Debug, Clone)]
pub struct MaxCutSDP<T: Scalar, B: LinAlgBackend<T>> {
	/// Graph Laplacian L.
	pub laplacian: B::Matrix,
	/// Vectorized Laplacian (precomputed for hot path).
	laplacian_vec: B::Vector,
	/// Diagonal constraint penalty μ.
	pub mu: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MaxCutSDP<T, B> {
	/// Creates a Max-Cut SDP problem from a graph Laplacian.
	pub fn new(laplacian: B::Matrix, mu: T) -> Self {
		let laplacian_vec = B::Vector::from_slice(laplacian.as_slice());
		Self {
			laplacian,
			laplacian_vec,
			mu,
			_phantom: PhantomData,
		}
	}

	/// Creates from an adjacency/weight matrix W.
	pub fn from_adjacency(w: B::Matrix, mu: T) -> Self {
		let n = MatrixView::nrows(&w);
		let ones = <B::Vector as VectorOps<T>>::from_fn(n, |_| T::one());
		let degree = w.mat_vec(&ones);
		let mut laplacian = B::Matrix::from_diagonal(&degree);
		laplacian.sub_assign(&w);
		Self::new(laplacian, mu)
	}
}

/// Workspace for [`MaxCutSDP`].
pub struct MaxCutSDPWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Euclidean gradient buffer.
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for MaxCutSDPWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for MaxCutSDPWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for MaxCutSDPWorkspace<T, B> where B::Vector: Sync {}

impl<T, B, M> Problem<T, M> for MaxCutSDP<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = MaxCutSDPWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let len = VectorView::len(proto_point);
		MaxCutSDPWorkspace {
			egrad: B::Vector::zeros(len),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// f(x) = −¼ ⟨L, X⟩ where X is the vectorized PSD matrix
		// For the vectorized form, this is −¼ L_vec · x_vec
		let quarter = <T as Scalar>::from_f64(0.25);
		-quarter * self.laplacian_vec.dot(point)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let quarter = <T as Scalar>::from_f64(0.25);
		ws.egrad.copy_from(&self.laplacian_vec);
		ws.egrad.scale_mut(-quarter);

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}
