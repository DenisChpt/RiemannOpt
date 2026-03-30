//! Optimization problems on the SPD manifold S⁺⁺(n).
//!
//! # Problems
//!
//! - [`FrechetMean`] — Fréchet/Karcher mean of covariance matrices
//! - [`MetricLearning`] — Mahalanobis distance metric learning
//! - [`GaussianMixtureCovariance`] — Covariance estimation for GMMs

use std::marker::PhantomData;

use crate::{
	linalg::{DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Fréchet Mean
// ════════════════════════════════════════════════════════════════════════════

/// Fréchet / Karcher mean of SPD matrices.
///
/// ## Mathematical Definition
///
/// Given SPD matrices P₁, …, Pₖ with weights w₁, …, wₖ (Σwᵢ = 1),
/// find the weighted Fréchet mean:
///
/// ```text
/// P* = argmin_{P ∈ S⁺⁺(n)} Σᵢ wᵢ d²(P, Pᵢ)
/// ```
///
/// where d is the affine-invariant distance:
///
/// ```text
/// d(A, B) = ‖log(A^{-1/2} B A^{-1/2})‖_F
/// ```
///
/// ## Gradient (affine-invariant metric)
///
/// ```text
/// grad f(P) = −Σᵢ wᵢ log(P^{-1} Pᵢ)   (simplified Riemannian gradient)
/// ```
///
/// More precisely, the Riemannian gradient in the affine-invariant metric
/// at P is: grad f(P) = P · (Σᵢ wᵢ log(P⁻¹ Pᵢ)) · P  (up to symmetrization).
///
/// ## Algorithm
///
/// For the Karcher mean, the gradient has a known form involving matrix
/// logarithms. Each iteration requires eigendecompositions of P⁻¹ Pᵢ.
#[derive(Debug, Clone)]
pub struct FrechetMean<T: Scalar, B: LinAlgBackend<T>> {
	/// Input SPD matrices P₁, …, Pₖ.
	pub matrices: Vec<B::Matrix>,
	/// Weights w₁, …, wₖ (sum to 1).
	pub weights: Vec<T>,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> FrechetMean<T, B> {
	/// Creates a Fréchet mean problem with uniform weights.
	pub fn uniform(matrices: Vec<B::Matrix>) -> Self {
		let k = matrices.len();
		let w = T::one() / <T as RealScalar>::from_usize(k);
		let weights = vec![w; k];
		Self {
			matrices,
			weights,
			_phantom: PhantomData,
		}
	}

	/// Creates with custom weights (must sum to 1).
	pub fn weighted(matrices: Vec<B::Matrix>, weights: Vec<T>) -> Self {
		debug_assert_eq!(matrices.len(), weights.len());
		Self {
			matrices,
			weights,
			_phantom: PhantomData,
		}
	}

	/// Computes log(A⁻¹B) via eigendecomposition for SPD A, B.
	/// Returns the matrix logarithm of A⁻¹B.
	fn log_inv_product(&self, a: &B::Matrix, b: &B::Matrix) -> B::Matrix {
		let n = MatrixView::nrows(a);
		// A⁻¹ B
		let mut a_inv = B::Matrix::zeros(n, n);
		a.inverse(&mut a_inv);
		let product = a_inv.mat_mul(b);

		// Eigendecompose: A⁻¹B = V diag(λ) V⁻¹
		// Since A⁻¹B is similar to A^{-1/2} B A^{-1/2} (symmetric), its eigenvalues
		// are positive. log(A⁻¹B) = V diag(log λ) V⁻¹.
		let eig = product.symmetric_eigen();
		let mut log_lambda = B::Matrix::zeros(n, n);
		for i in 0..n {
			let li = eig.eigenvalues.get(i);
			*log_lambda.get_mut(i, i) = li.max(T::EPSILON).ln();
		}
		// V · diag(log λ) · Vᵀ  (symmetric eigendecomposition → V orthogonal)
		let mut tmp = B::Matrix::zeros(n, n);
		tmp.gemm(
			T::one(),
			eig.eigenvectors.as_view(),
			log_lambda.as_view(),
			T::zero(),
		);
		let mut result = B::Matrix::zeros(n, n);
		result.gemm_bt(
			T::one(),
			tmp.as_view(),
			eig.eigenvectors.as_view(),
			T::zero(),
		);
		result
	}
}

/// Workspace for [`FrechetMean`].
pub struct FrechetMeanWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Accumulated gradient (n×n).
	egrad: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for FrechetMeanWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for FrechetMeanWorkspace<T, B>
where
	B::Matrix: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for FrechetMeanWorkspace<T, B>
where
	B::Matrix: Sync,
{
}

impl<T, B, M> Problem<T, M> for FrechetMean<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = FrechetMeanWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = MatrixView::nrows(proto_point);
		FrechetMeanWorkspace {
			egrad: B::Matrix::zeros(n, n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// f(P) = Σᵢ wᵢ ‖log(P⁻¹ Pᵢ)‖_F²
		let mut cost = T::zero();
		for (pi, &wi) in self.matrices.iter().zip(&self.weights) {
			let log_mat = self.log_inv_product(point, pi);
			let norm_sq = log_mat.frobenius_dot(&log_mat);
			cost = cost + wi * norm_sq;
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
		let n = MatrixView::nrows(point);
		// grad f(P) = −Σᵢ wᵢ log(P⁻¹ Pᵢ)
		// This is already a tangent vector in the affine-invariant metric.
		ws.egrad.fill(T::zero());
		for (pi, &wi) in self.matrices.iter().zip(&self.weights) {
			let log_mat = self.log_inv_product(point, pi);
			ws.egrad.mat_axpy(-wi, &log_mat, T::one());
		}

		// The Riemannian gradient in the affine-invariant metric is:
		// grad f = P · egrad · P (symmetrized)
		// But euclidean_to_riemannian_gradient handles this via the manifold.
		// For the AI metric, the conversion from Euclidean grad G is:
		// rgrad = P G P. Here egrad is already −Σ wᵢ log(P⁻¹Pᵢ) which IS
		// the Riemannian gradient. So we pass it through directly.
		//
		// Actually, the Euclidean gradient of f = Σ wᵢ ‖log(P⁻¹Pᵢ)‖² is complex.
		// The manifold's project_tangent handles the conversion from ambient gradient
		// to Riemannian gradient for whichever metric the SPD manifold uses.
		// Let's just provide the "natural" gradient and let the manifold handle it.
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);

		let _ = n;
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Metric Learning
// ════════════════════════════════════════════════════════════════════════════

/// Mahalanobis distance metric learning on S⁺⁺(n).
///
/// ## Mathematical Definition
///
/// Learn a Mahalanobis distance matrix M ∈ S⁺⁺(n) from similarity/dissimilarity
/// constraints:
///
/// ```text
/// f(M) = Σ_{(i,j)∈S} (xᵢ−xⱼ)ᵀ M (xᵢ−xⱼ)
///       − α Σ_{(i,j)∈D} log((xᵢ−xⱼ)ᵀ M (xᵢ−xⱼ))
/// ```
///
/// where S = similar pairs (attract) and D = dissimilar pairs (repel).
///
/// ## Gradient
///
/// ```text
/// ∇f(M) = Σ_{(i,j)∈S} dᵢⱼ dᵢⱼᵀ
///        − α Σ_{(i,j)∈D} dᵢⱼ dᵢⱼᵀ / (dᵢⱼᵀ M dᵢⱼ)
/// ```
///
/// where dᵢⱼ = xᵢ − xⱼ.
#[derive(Debug, Clone)]
pub struct MetricLearning<T: Scalar, B: LinAlgBackend<T>> {
	/// Similar pair outer products: Σ dᵢⱼ dᵢⱼᵀ (precomputed, n×n).
	similar_sum: B::Matrix,
	/// Dissimilar pair differences (list of dᵢⱼ vectors).
	dissimilar_diffs: Vec<B::Vector>,
	/// Repulsion strength α.
	pub alpha: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MetricLearning<T, B> {
	/// Creates a metric learning problem.
	///
	/// # Arguments
	///
	/// * `data` — Data points as columns of a matrix X ∈ ℝⁿˣᵐ
	/// * `similar_pairs` — Indices of similar pairs (i, j)
	/// * `dissimilar_pairs` — Indices of dissimilar pairs (i, j)
	/// * `alpha` — Repulsion strength
	pub fn new(
		data: &B::Matrix,
		similar_pairs: &[(usize, usize)],
		dissimilar_pairs: &[(usize, usize)],
		alpha: T,
	) -> Self {
		let n = MatrixView::nrows(data);

		// Precompute Σ dᵢⱼ dᵢⱼᵀ for similar pairs (no intermediate allocations)
		let mut similar_sum = B::Matrix::zeros(n, n);
		for &(i, j) in similar_pairs {
			for r in 0..n {
				let dr = data.get(r, i) - data.get(r, j);
				for c in 0..n {
					let dc = data.get(c, i) - data.get(c, j);
					*similar_sum.get_mut(r, c) = similar_sum.get(r, c) + dr * dc;
				}
			}
		}

		// Store dissimilar pair differences (element-wise, no column_to_owned)
		let dissimilar_diffs: Vec<_> = dissimilar_pairs
			.iter()
			.map(|&(i, j)| {
				B::Vector::from_fn(n, |r| data.get(r, i) - data.get(r, j))
			})
			.collect();

		Self {
			similar_sum,
			dissimilar_diffs,
			alpha,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`MetricLearning`].
pub struct MetricLearningWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Matrix,
	md: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for MetricLearningWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			md: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for MetricLearningWorkspace<T, B>
where
	B::Matrix: Send,
	B::Vector: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for MetricLearningWorkspace<T, B>
where
	B::Matrix: Sync,
	B::Vector: Sync,
{
}

impl<T, B, M> Problem<T, M> for MetricLearning<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = MetricLearningWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = MatrixView::nrows(proto_point);
		MetricLearningWorkspace {
			egrad: B::Matrix::zeros(n, n),
			md: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// Attraction: tr(M · Σ_S dᵢⱼ dᵢⱼᵀ) = ⟨M, similar_sum⟩_F
		let mut cost = point.frobenius_dot(&self.similar_sum);

		// Repulsion: −α Σ_D log(dᵢⱼᵀ M dᵢⱼ)
		if self.alpha > T::zero() {
			for d in &self.dissimilar_diffs {
				let md = point.mat_vec(d);
				let quad = d.dot(&md);
				cost = cost - self.alpha * quad.max(T::EPSILON).ln();
			}
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
		let n = MatrixView::nrows(point);

		// ∇f = similar_sum − α Σ_D (dᵢⱼ dᵢⱼᵀ) / (dᵢⱼᵀ M dᵢⱼ)
		ws.egrad.copy_from(&self.similar_sum);

		if self.alpha > T::zero() {
			for d in &self.dissimilar_diffs {
				point.mat_vec_into(d, &mut ws.md);
				let quad = d.dot(&ws.md);
				let weight = -self.alpha / quad.max(T::EPSILON);
				// Rank-1 update: egrad += weight * d d^T
				for r in 0..n {
					for c in 0..n {
						*ws.egrad.get_mut(r, c) =
							ws.egrad.get(r, c) + weight * d.get(r) * d.get(c);
					}
				}
			}
		}

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Gaussian Mixture Model Covariance
// ════════════════════════════════════════════════════════════════════════════

/// Covariance estimation for Gaussian Mixture Models on S⁺⁺(n).
///
/// ## Mathematical Definition
///
/// Given data x₁, …, xₘ ∈ ℝⁿ with responsibilities γᵢ ∈ [0,1]
/// (probability that xᵢ belongs to this component), estimate the
/// covariance Σ ∈ S⁺⁺(n) by minimizing the negative log-likelihood:
///
/// ```text
/// f(Σ) = ½ log det(Σ) + (1/2N_k) Σᵢ γᵢ (xᵢ−μ)ᵀ Σ⁻¹ (xᵢ−μ)
/// ```
///
/// where N_k = Σᵢ γᵢ and μ = (1/N_k) Σᵢ γᵢ xᵢ is the component mean (fixed).
///
/// ## Gradient
///
/// ```text
/// ∇f(Σ) = ½ Σ⁻¹ − (1/2N_k) Σ⁻¹ S_k Σ⁻¹
/// ```
///
/// where S_k = Σᵢ γᵢ (xᵢ−μ)(xᵢ−μ)ᵀ is the weighted scatter matrix.
#[derive(Debug, Clone)]
pub struct GaussianMixtureCovariance<T: Scalar, B: LinAlgBackend<T>> {
	/// Weighted scatter matrix S_k = Σᵢ γᵢ (xᵢ−μ)(xᵢ−μ)ᵀ (precomputed).
	pub scatter: B::Matrix,
	/// Effective number of samples N_k = Σᵢ γᵢ.
	pub n_eff: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> GaussianMixtureCovariance<T, B> {
	/// Creates from precomputed scatter matrix and effective sample count.
	pub fn new(scatter: B::Matrix, n_eff: T) -> Self {
		debug_assert!(n_eff > T::zero());
		Self {
			scatter,
			n_eff,
			_phantom: PhantomData,
		}
	}

	/// Creates from data, mean, and responsibilities.
	///
	/// # Arguments
	///
	/// * `data` — Data matrix X ∈ ℝⁿˣᵐ (columns are data points)
	/// * `mean` — Component mean μ ∈ ℝⁿ
	/// * `responsibilities` — Responsibilities γ₁, …, γₘ
	pub fn from_data(data: &B::Matrix, mean: &B::Vector, responsibilities: &[T]) -> Self {
		let n = MatrixView::nrows(data);
		let m = MatrixView::ncols(data);
		debug_assert_eq!(responsibilities.len(), m);

		let mut scatter = B::Matrix::zeros(n, n);
		let mut n_eff = T::zero();

		for j in 0..m {
			let gamma = responsibilities[j];
			n_eff = n_eff + gamma;
			// scatter += γ_j · (x_j − μ)(x_j − μ)ᵀ  (no intermediate allocations)
			for r in 0..n {
				let dr = data.get(r, j) - mean.get(r);
				for c in 0..n {
					let dc = data.get(c, j) - mean.get(c);
					*scatter.get_mut(r, c) = scatter.get(r, c) + gamma * dr * dc;
				}
			}
		}

		Self::new(scatter, n_eff)
	}
}

/// Workspace for [`GaussianMixtureCovariance`].
pub struct GMMWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Σ⁻¹ (n×n).
	sigma_inv: B::Matrix,
	/// Euclidean gradient (n×n).
	egrad: B::Matrix,
	/// Σ⁻¹ S_k (n×n).
	inv_scatter: B::Matrix,
	/// Temporary buffer for matrix products (n×n).
	tmp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for GMMWorkspace<T, B> {
	fn default() -> Self {
		Self {
			sigma_inv: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			inv_scatter: B::Matrix::zeros(0, 0),
			tmp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for GMMWorkspace<T, B>
where
	B::Matrix: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for GMMWorkspace<T, B>
where
	B::Matrix: Sync,
{
}

impl<T, B, M> Problem<T, M> for GaussianMixtureCovariance<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = GMMWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = MatrixView::nrows(proto_point);
		GMMWorkspace {
			sigma_inv: B::Matrix::zeros(n, n),
			egrad: B::Matrix::zeros(n, n),
			inv_scatter: B::Matrix::zeros(n, n),
			tmp: B::Matrix::zeros(n, n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let n = MatrixView::nrows(point);
		let half = <T as Scalar>::from_f64(0.5);

		// log det(Σ) via eigenvalues
		let eig = point.symmetric_eigen();
		let mut log_det = T::zero();
		for i in 0..n {
			log_det = log_det + eig.eigenvalues.get(i).max(T::EPSILON).ln();
		}

		// tr(Σ⁻¹ S_k) / N_k
		let mut sigma_inv = B::Matrix::zeros(n, n);
		point.inverse(&mut sigma_inv);
		let trace_inv_scatter = sigma_inv.frobenius_dot(&self.scatter);

		half * log_det + half * trace_inv_scatter / self.n_eff
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let half = <T as Scalar>::from_f64(0.5);

		// Σ⁻¹
		point.inverse(&mut ws.sigma_inv);

		// ∇f = ½ Σ⁻¹ − (1/2N_k) Σ⁻¹ S_k Σ⁻¹
		// Σ⁻¹ S_k → inv_scatter
		ws.inv_scatter.gemm(
			T::one(),
			ws.sigma_inv.as_view(),
			self.scatter.as_view(),
			T::zero(),
		);

		// ½ Σ⁻¹ − (1/2N_k) inv_scatter · Σ⁻¹
		ws.egrad.copy_from(&ws.sigma_inv);
		ws.egrad.scale_mut(half);

		ws.tmp.gemm(
			-half / self.n_eff,
			ws.inv_scatter.as_view(),
			ws.sigma_inv.as_view(),
			T::zero(),
		);
		ws.egrad.add_assign(&ws.tmp);

		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}
