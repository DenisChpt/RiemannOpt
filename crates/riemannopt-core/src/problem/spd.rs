//! Optimization problems on the SPD manifold S⁺⁺(n).

use std::marker::PhantomData;

use crate::{
	linalg::{
		DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView,
	},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Fréchet Mean
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct FrechetMean<T: Scalar, B: LinAlgBackend<T>> {
	pub matrices: Vec<B::Matrix>,
	pub weights: Vec<T>,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> FrechetMean<T, B> {
	pub fn uniform(matrices: Vec<B::Matrix>) -> Self {
		let k = matrices.len();
		let w = T::one() / <T as RealScalar>::from_usize(k);
		Self {
			matrices,
			weights: vec![w; k],
			_phantom: PhantomData,
		}
	}

	pub fn weighted(matrices: Vec<B::Matrix>, weights: Vec<T>) -> Self {
		debug_assert_eq!(matrices.len(), weights.len());
		Self {
			matrices,
			weights,
			_phantom: PhantomData,
		}
	}

	fn log_inv_product(
		a: &B::Matrix,
		b: &B::Matrix,
		a_inv: &mut B::Matrix,
		product: &mut B::Matrix,
		eigenvalues: &mut B::Vector,
		eigenvectors: &mut B::Matrix,
		tmp: &mut B::Matrix,
		result: &mut B::Matrix,
	) {
		let n = MatrixView::nrows(a);
		a.inverse(a_inv);
		product.gemm(T::one(), a_inv.as_view(), b.as_view(), T::zero());
		DecompositionOps::symmetric_eigen(product, eigenvalues, eigenvectors);
		tmp.fill(T::zero());
		for i in 0..n {
			let log_li = eigenvalues.get(i).max(T::EPSILON).ln();
			for r in 0..n {
				*tmp.get_mut(r, i) = log_li * eigenvectors.get(r, i);
			}
		}
		result.gemm_bt(T::one(), tmp.as_view(), eigenvectors.as_view(), T::zero());
	}
}

pub struct FrechetMeanWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Matrix,
	a_inv: B::Matrix,
	product: B::Matrix,
	eigenvectors: B::Matrix,
	eigenvalues: B::Vector,
	tmp: B::Matrix,
	log_result: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for FrechetMeanWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			a_inv: B::Matrix::zeros(0, 0),
			product: B::Matrix::zeros(0, 0),
			eigenvectors: B::Matrix::zeros(0, 0),
			eigenvalues: B::Vector::zeros(0),
			tmp: B::Matrix::zeros(0, 0),
			log_result: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for FrechetMeanWorkspace<T, B>
where
	B::Matrix: Send,
	B::Vector: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for FrechetMeanWorkspace<T, B>
where
	B::Matrix: Sync,
	B::Vector: Sync,
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
			a_inv: B::Matrix::zeros(n, n),
			product: B::Matrix::zeros(n, n),
			eigenvectors: B::Matrix::zeros(n, n),
			eigenvalues: B::Vector::zeros(n),
			tmp: B::Matrix::zeros(n, n),
			log_result: B::Matrix::zeros(n, n),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — reuses workspace buffers for all k matrices.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let mut cost = T::zero();
		for (pi, &wi) in self.matrices.iter().zip(&self.weights) {
			FrechetMean::<T, B>::log_inv_product(
				point,
				pi,
				&mut ws.a_inv,
				&mut ws.product,
				&mut ws.eigenvalues,
				&mut ws.eigenvectors,
				&mut ws.tmp,
				&mut ws.log_result,
			);
			cost += wi * ws.log_result.frobenius_dot(&ws.log_result);
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
		ws.egrad.fill(T::zero());
		for (pi, &wi) in self.matrices.iter().zip(&self.weights) {
			FrechetMean::<T, B>::log_inv_product(
				point,
				pi,
				&mut ws.a_inv,
				&mut ws.product,
				&mut ws.eigenvalues,
				&mut ws.eigenvectors,
				&mut ws.tmp,
				&mut ws.log_result,
			);
			ws.egrad.mat_axpy(-wi, &ws.log_result, T::one());
		}
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Metric Learning
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MetricLearning<T: Scalar, B: LinAlgBackend<T>> {
	similar_sum: B::Matrix,
	dissimilar_diffs: Vec<B::Vector>,
	pub alpha: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MetricLearning<T, B> {
	pub fn new(
		data: &B::Matrix,
		similar_pairs: &[(usize, usize)],
		dissimilar_pairs: &[(usize, usize)],
		alpha: T,
	) -> Self {
		let n = MatrixView::nrows(data);
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
		let dissimilar_diffs: Vec<_> = dissimilar_pairs
			.iter()
			.map(|&(i, j)| B::Vector::from_fn(n, |r| data.get(r, i) - data.get(r, j)))
			.collect();
		Self {
			similar_sum,
			dissimilar_diffs,
			alpha,
			_phantom: PhantomData,
		}
	}
}

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

	/// **Zero allocation** — uses ws.md for M·d.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let mut cost = point.frobenius_dot(&self.similar_sum);
		if self.alpha > T::zero() {
			for d in &self.dissimilar_diffs {
				point.mat_vec_into(d, &mut ws.md);
				let quad = d.dot(&ws.md);
				cost -= self.alpha * quad.max(T::EPSILON).ln();
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
		ws.egrad.copy_from(&self.similar_sum);
		if self.alpha > T::zero() {
			for d in &self.dissimilar_diffs {
				point.mat_vec_into(d, &mut ws.md);
				let quad = d.dot(&ws.md);
				let weight = -self.alpha / quad.max(T::EPSILON);
				for r in 0..n {
					for c in 0..n {
						*ws.egrad.get_mut(r, c) = ws.egrad.get(r, c) + weight * d.get(r) * d.get(c);
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

#[derive(Debug, Clone)]
pub struct GaussianMixtureCovariance<T: Scalar, B: LinAlgBackend<T>> {
	pub scatter: B::Matrix,
	pub n_eff: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> GaussianMixtureCovariance<T, B> {
	pub fn new(scatter: B::Matrix, n_eff: T) -> Self {
		debug_assert!(n_eff > T::zero());
		Self {
			scatter,
			n_eff,
			_phantom: PhantomData,
		}
	}

	pub fn from_data(data: &B::Matrix, mean: &B::Vector, responsibilities: &[T]) -> Self {
		let n = MatrixView::nrows(data);
		let m = MatrixView::ncols(data);
		debug_assert_eq!(responsibilities.len(), m);
		let mut scatter = B::Matrix::zeros(n, n);
		let mut n_eff = T::zero();
		for (j, &gamma) in responsibilities[..m].iter().enumerate() {
			n_eff += gamma;
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

pub struct GMMWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	sigma_inv: B::Matrix,
	egrad: B::Matrix,
	inv_scatter: B::Matrix,
	tmp: B::Matrix,
	eigenvalues: B::Vector,
	eigenvectors: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for GMMWorkspace<T, B> {
	fn default() -> Self {
		Self {
			sigma_inv: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			inv_scatter: B::Matrix::zeros(0, 0),
			tmp: B::Matrix::zeros(0, 0),
			eigenvalues: B::Vector::zeros(0),
			eigenvectors: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for GMMWorkspace<T, B>
where
	B::Matrix: Send,
	B::Vector: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for GMMWorkspace<T, B>
where
	B::Matrix: Sync,
	B::Vector: Sync,
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
			eigenvalues: B::Vector::zeros(n),
			eigenvectors: B::Matrix::zeros(n, n),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — eigenvalues + inverse via workspace.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let n = MatrixView::nrows(point);
		let half = <T as Scalar>::from_f64(0.5);

		DecompositionOps::symmetric_eigen(point, &mut ws.eigenvalues, &mut ws.eigenvectors);
		let mut log_det = T::zero();
		for i in 0..n {
			log_det += ws.eigenvalues.get(i).max(T::EPSILON).ln();
		}

		point.inverse(&mut ws.sigma_inv);
		let trace_inv_scatter = ws.sigma_inv.frobenius_dot(&self.scatter);

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
		point.inverse(&mut ws.sigma_inv);
		ws.inv_scatter.gemm(
			T::one(),
			ws.sigma_inv.as_view(),
			self.scatter.as_view(),
			T::zero(),
		);
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
