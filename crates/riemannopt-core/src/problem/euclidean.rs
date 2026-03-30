//! Optimization problems on the Euclidean manifold ℝⁿ.
//!
//! # Problems
//!
//! - [`Rosenbrock`] — Non-convex valley test function
//! - [`Rastrigin`] — Highly multimodal test function
//! - [`RidgeRegression`] — Linear regression with ℓ₂ regularization
//! - [`LogisticRegression`] — Binary classification with ℓ₂ regularization

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Rosenbrock
// ════════════════════════════════════════════════════════════════════════════

/// The Rosenbrock function (generalized to n dimensions).
///
/// ## Mathematical Definition
///
/// ```text
/// f(x) = Σᵢ₌₁ⁿ⁻¹ [ a·(xᵢ₊₁ − xᵢ²)² + (b − xᵢ)² ]
/// ```
///
/// Classic parameters: a = 100, b = 1. Global minimum at x* = (b, b², …).
///
/// ## Gradient
///
/// ```text
/// ∂f/∂xᵢ = −4a·xᵢ·(xᵢ₊₁ − xᵢ²) − 2(b − xᵢ)     (first term, i < n−1)
///         +  2a·(xᵢ − xᵢ₋₁²)                        (second term, i > 0)
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// [H·v]ᵢ = (−4a·(xᵢ₊₁ − xᵢ²) + 12a·xᵢ² + 2)·vᵢ
///         − 4a·xᵢ·vᵢ₊₁                               (i < n−1)
///         − 4a·xᵢ₋₁·vᵢ₋₁ + 2a·vᵢ                     (i > 0)
/// ```
#[derive(Debug, Clone)]
pub struct Rosenbrock<T: Scalar, B: LinAlgBackend<T>> {
	/// Coefficient of the quadratic valley term (default: 100).
	pub a: T,
	/// Center of the quadratic penalty term (default: 1).
	pub b: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for Rosenbrock<T, B> {
	fn default() -> Self {
		Self {
			a: <T as Scalar>::from_f64(100.0),
			b: T::one(),
			_phantom: PhantomData,
		}
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Rosenbrock<T, B> {
	/// Creates a Rosenbrock function with classic parameters (a=100, b=1).
	pub fn new() -> Self {
		Self::default()
	}

	/// Creates a Rosenbrock function with custom parameters.
	pub fn with_parameters(a: T, b: T) -> Self {
		Self {
			a,
			b,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`Rosenbrock`]: buffer for Euclidean gradient.
pub struct RosenbrockWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	ehvp: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for RosenbrockWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			ehvp: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RosenbrockWorkspace<T, B> where B::Vector: Send
{}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RosenbrockWorkspace<T, B> where B::Vector: Sync
{}

impl<T, B, M> Problem<T, M> for Rosenbrock<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = RosenbrockWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		RosenbrockWorkspace {
			egrad: B::Vector::zeros(n),
			ehvp: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		let n = VectorView::len(point);
		debug_assert!(n >= 2, "Rosenbrock requires n ≥ 2");
		let mut f = T::zero();
		for i in 0..n - 1 {
			let xi = point.get(i);
			let xi1 = point.get(i + 1);
			let valley = xi1 - xi * xi;
			let ridge = self.b - xi;
			f = f + self.a * valley * valley + ridge * ridge;
		}
		f
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let n = VectorView::len(point);
		let two = <T as Scalar>::from_f64(2.0);
		let four = <T as Scalar>::from_f64(4.0);
		let egrad = &mut ws.egrad;
		egrad.fill(T::zero());

		for i in 0..n - 1 {
			let xi = point.get(i);
			let xi1 = point.get(i + 1);
			let diff = xi1 - xi * xi;

			*egrad.get_mut(i) =
				*egrad.get_mut(i) - four * self.a * xi * diff - two * (self.b - xi);
			*egrad.get_mut(i + 1) = *egrad.get_mut(i + 1) + two * self.a * diff;
		}

		manifold.euclidean_to_riemannian_gradient(point, egrad, result, manifold_ws);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		let n = VectorView::len(point);
		let two = <T as Scalar>::from_f64(2.0);
		let four = <T as Scalar>::from_f64(4.0);
		let egrad = &mut ws.egrad;
		egrad.fill(T::zero());
		let mut f = T::zero();

		for i in 0..n - 1 {
			let xi = point.get(i);
			let xi1 = point.get(i + 1);
			let diff = xi1 - xi * xi;
			let ridge = self.b - xi;

			f = f + self.a * diff * diff + ridge * ridge;

			*egrad.get_mut(i) = *egrad.get_mut(i) - four * self.a * xi * diff - two * ridge;
			*egrad.get_mut(i + 1) = *egrad.get_mut(i + 1) + two * self.a * diff;
		}

		manifold.euclidean_to_riemannian_gradient(point, egrad, gradient, manifold_ws);
		f
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
		let n = VectorView::len(point);
		let two = <T as Scalar>::from_f64(2.0);
		let four = <T as Scalar>::from_f64(4.0);
		let twelve = <T as Scalar>::from_f64(12.0);

		// Euclidean gradient for curvature correction
		let egrad = &mut ws.egrad;
		egrad.fill(T::zero());
		for i in 0..n - 1 {
			let xi = point.get(i);
			let xi1 = point.get(i + 1);
			let diff = xi1 - xi * xi;
			*egrad.get_mut(i) =
				*egrad.get_mut(i) - four * self.a * xi * diff - two * (self.b - xi);
			*egrad.get_mut(i + 1) = *egrad.get_mut(i + 1) + two * self.a * diff;
		}

		// Euclidean HVP directly into result
		result.fill(T::zero());
		for i in 0..n - 1 {
			let xi = point.get(i);
			let xi1 = point.get(i + 1);
			let diff = xi1 - xi * xi;
			let vi = vector.get(i);
			let vi1 = vector.get(i + 1);

			*result.get_mut(i) = *result.get_mut(i)
				+ (-four * self.a * diff + twelve * self.a * xi * xi + two) * vi;
			*result.get_mut(i) = *result.get_mut(i) - four * self.a * xi * vi1;
			*result.get_mut(i + 1) = *result.get_mut(i + 1) - four * self.a * xi * vi;
			*result.get_mut(i + 1) = *result.get_mut(i + 1) + two * self.a * vi1;
		}

		ws.ehvp.copy_from(result);
		manifold.euclidean_to_riemannian_hessian(
			point,
			egrad,
			&ws.ehvp,
			vector,
			result,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Rastrigin
// ════════════════════════════════════════════════════════════════════════════

/// The Rastrigin function (n-dimensional).
///
/// ## Mathematical Definition
///
/// ```text
/// f(x) = An + Σᵢ₌₁ⁿ [xᵢ² − A·cos(2πxᵢ)]
/// ```
///
/// Default A = 10. Global minimum at x* = 0 with f(0) = 0.
/// Highly multimodal: ~10ⁿ local minima.
///
/// ## Gradient
///
/// ```text
/// ∂f/∂xᵢ = 2xᵢ + 2πA·sin(2πxᵢ)
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// [H·v]ᵢ = (2 + 4π²A·cos(2πxᵢ))·vᵢ
/// ```
#[derive(Debug, Clone)]
pub struct Rastrigin<T: Scalar, B: LinAlgBackend<T>> {
	/// Amplitude parameter (default: 10).
	pub amplitude: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for Rastrigin<T, B> {
	fn default() -> Self {
		Self {
			amplitude: <T as Scalar>::from_f64(10.0),
			_phantom: PhantomData,
		}
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Rastrigin<T, B> {
	/// Creates a Rastrigin function with default amplitude A=10.
	pub fn new() -> Self {
		Self::default()
	}

	/// Creates a Rastrigin function with custom amplitude.
	pub fn with_amplitude(amplitude: T) -> Self {
		Self {
			amplitude,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`Rastrigin`].
pub struct RastriginWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	ehvp: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for RastriginWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			ehvp: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RastriginWorkspace<T, B> where B::Vector: Send
{}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RastriginWorkspace<T, B> where B::Vector: Sync
{}

impl<T, B, M> Problem<T, M> for Rastrigin<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = RastriginWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		RastriginWorkspace {
			egrad: B::Vector::zeros(n),
			ehvp: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		let n = VectorView::len(point);
		let two_pi = T::PI + T::PI;
		let mut f = self.amplitude * <T as RealScalar>::from_usize(n);
		for i in 0..n {
			let xi = point.get(i);
			f = f + xi * xi - self.amplitude * (two_pi * xi).cos();
		}
		f
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let n = VectorView::len(point);
		let two = <T as Scalar>::from_f64(2.0);
		let two_pi = T::PI + T::PI;
		let egrad = &mut ws.egrad;
		for i in 0..n {
			let xi = point.get(i);
			*egrad.get_mut(i) = two * xi + two_pi * self.amplitude * (two_pi * xi).sin();
		}
		manifold.euclidean_to_riemannian_gradient(point, egrad, result, manifold_ws);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		let n = VectorView::len(point);
		let two = <T as Scalar>::from_f64(2.0);
		let two_pi = T::PI + T::PI;
		let egrad = &mut ws.egrad;
		let mut f = self.amplitude * <T as RealScalar>::from_usize(n);
		for i in 0..n {
			let xi = point.get(i);
			let angle = two_pi * xi;
			f = f + xi * xi - self.amplitude * angle.cos();
			*egrad.get_mut(i) = two * xi + two_pi * self.amplitude * angle.sin();
		}
		manifold.euclidean_to_riemannian_gradient(point, egrad, gradient, manifold_ws);
		f
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
		let n = VectorView::len(point);
		let two = <T as Scalar>::from_f64(2.0);
		let two_pi = T::PI + T::PI;
		let four_pi_sq = two_pi * two_pi;
		let egrad = &mut ws.egrad;

		for i in 0..n {
			let xi = point.get(i);
			let angle = two_pi * xi;
			*egrad.get_mut(i) = two * xi + two_pi * self.amplitude * angle.sin();
		}

		for i in 0..n {
			let xi = point.get(i);
			let hii = two + four_pi_sq * self.amplitude * (two_pi * xi).cos();
			*result.get_mut(i) = hii * vector.get(i);
		}

		ws.ehvp.copy_from(result);
		manifold.euclidean_to_riemannian_hessian(
			point,
			egrad,
			&ws.ehvp,
			vector,
			result,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Ridge Regression
// ════════════════════════════════════════════════════════════════════════════

/// Linear regression with ℓ₂ regularization (Ridge Regression).
///
/// ## Mathematical Definition
///
/// ```text
/// f(w) = (1/2m) ‖Xw − y‖² + (λ/2) ‖w‖²
/// ```
///
/// where X ∈ ℝᵐˣⁿ (data), y ∈ ℝᵐ (targets), λ ≥ 0 (regularization).
///
/// ## Gradient
///
/// ```text
/// ∇f(w) = (1/m) Xᵀ(Xw − y) + λw
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// ∇²f · v = (1/m) Xᵀ X v + λv
/// ```
#[derive(Debug, Clone)]
pub struct RidgeRegression<T: Scalar, B: LinAlgBackend<T>> {
	/// Data matrix X ∈ ℝᵐˣⁿ.
	pub x: B::Matrix,
	/// Precomputed XᵀX ∈ ℝⁿˣⁿ.
	xtx: B::Matrix,
	/// Precomputed Xᵀy ∈ ℝⁿ.
	xty: B::Vector,
	/// Precomputed ½‖y‖².
	half_y_sq: T,
	/// Target vector y ∈ ℝᵐ.
	pub y: B::Vector,
	/// Regularization parameter λ ≥ 0.
	pub lambda: T,
	/// Precomputed 1/m.
	inv_m: T,
}

impl<T: Scalar, B: LinAlgBackend<T>> RidgeRegression<T, B> {
	/// Creates a ridge regression problem.
	pub fn new(x: B::Matrix, y: B::Vector, lambda: T) -> Self {
		debug_assert!(lambda >= T::zero());
		let m = MatrixView::nrows(&x);
		debug_assert_eq!(VectorView::len(&y), m);
		let inv_m = T::one() / <T as RealScalar>::from_usize(m);
		let half = <T as Scalar>::from_f64(0.5);

		// Precompute XᵀX, Xᵀy
		let n = MatrixView::ncols(&x);
		let mut xtx = B::Matrix::zeros(n, n);
		xtx.gemm_at(T::one(), x.as_view(), x.as_view(), T::zero());
		let xt = x.transpose_to_owned();
		let xty = xt.mat_vec(&y);
		let half_y_sq = half * y.norm_squared();

		Self {
			x,
			xtx,
			xty,
			half_y_sq,
			y,
			lambda,
			inv_m,
		}
	}
}

/// Workspace for [`RidgeRegression`].
pub struct RidgeRegressionWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Buffer for Euclidean gradient (length n).
	egrad: B::Vector,
	/// Buffer for XᵀXw (length n).
	xtxw: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for RidgeRegressionWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			xtxw: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RidgeRegressionWorkspace<T, B>
where
	B::Vector: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RidgeRegressionWorkspace<T, B>
where
	B::Vector: Sync,
{
}

impl<T, B, M> Problem<T, M> for RidgeRegression<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = RidgeRegressionWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		RidgeRegressionWorkspace {
			egrad: B::Vector::zeros(n),
			xtxw: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		// f = (1/2m) (w^T XᵀX w − 2 w^T Xᵀy + ‖y‖²) + (λ/2) ‖w‖²
		let half = <T as Scalar>::from_f64(0.5);
		let xtxw = self.xtx.mat_vec(point);
		let quad = half * self.inv_m * point.dot(&xtxw);
		let lin = self.inv_m * point.dot(&self.xty);
		quad - lin + self.inv_m * self.half_y_sq + half * self.lambda * point.norm_squared()
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// ∇f = (1/m)(XᵀXw − Xᵀy) + λw
		self.xtx.mat_vec_into(point, &mut ws.egrad);
		ws.egrad.sub_assign(&self.xty);
		ws.egrad.scale_mut(self.inv_m);
		ws.egrad.axpy(self.lambda, point, T::one());
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

		// XᵀXw → xtxw
		self.xtx.mat_vec_into(point, &mut ws.xtxw);
		let quad = half * self.inv_m * point.dot(&ws.xtxw);
		let lin = self.inv_m * point.dot(&self.xty);
		let cost =
			quad - lin + self.inv_m * self.half_y_sq + half * self.lambda * point.norm_squared();

		// ∇f = (1/m)(XᵀXw − Xᵀy) + λw
		ws.egrad.copy_from(&ws.xtxw);
		ws.egrad.sub_assign(&self.xty);
		ws.egrad.scale_mut(self.inv_m);
		ws.egrad.axpy(self.lambda, point, T::one());
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
		// ehvp = (1/m) XᵀXv + λv
		self.xtx.mat_vec_into(vector, &mut ws.xtxw);
		ws.xtxw.scale_mut(self.inv_m);
		ws.xtxw.axpy(self.lambda, vector, T::one());

		// egrad for curvature correction
		self.xtx.mat_vec_into(point, &mut ws.egrad);
		ws.egrad.sub_assign(&self.xty);
		ws.egrad.scale_mut(self.inv_m);
		ws.egrad.axpy(self.lambda, point, T::one());

		manifold.euclidean_to_riemannian_hessian(
			point,
			&ws.egrad,
			&ws.xtxw,
			vector,
			result,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Logistic Regression
// ════════════════════════════════════════════════════════════════════════════

/// Binary logistic regression with ℓ₂ regularization.
///
/// ## Mathematical Definition
///
/// ```text
/// f(w) = (1/m) Σᵢ log(1 + exp(−yᵢ · xᵢᵀw)) + (λ/2) ‖w‖²
/// ```
///
/// where X ∈ ℝᵐˣⁿ, y ∈ {−1, +1}ᵐ, λ ≥ 0.
///
/// ## Gradient
///
/// ```text
/// ∇f(w) = −(1/m) Xᵀ (y ⊙ σ(−y ⊙ Xw)) + λw
/// ```
///
/// ## Hessian-vector product
///
/// ```text
/// ∇²f · v = (1/m) Xᵀ diag(σᵢ(1−σᵢ)) X v + λv
/// ```
///
/// ## Numerical Stability
///
/// Uses log-sum-exp trick: log(1 + exp(t)) = max(t,0) + log(1 + exp(−|t|))
#[derive(Debug, Clone)]
pub struct LogisticRegression<T: Scalar, B: LinAlgBackend<T>> {
	/// Data matrix X ∈ ℝᵐˣⁿ.
	pub x: B::Matrix,
	/// Precomputed Xᵀ ∈ ℝⁿˣᵐ.
	xt: B::Matrix,
	/// Label vector y ∈ {−1, +1}ᵐ.
	pub y: B::Vector,
	/// Regularization parameter λ ≥ 0.
	pub lambda: T,
	/// Precomputed 1/m.
	inv_m: T,
}

impl<T: Scalar, B: LinAlgBackend<T>> LogisticRegression<T, B> {
	/// Creates a logistic regression problem.
	pub fn new(x: B::Matrix, y: B::Vector, lambda: T) -> Self {
		debug_assert!(lambda >= T::zero());
		let m = MatrixView::nrows(&x);
		debug_assert_eq!(VectorView::len(&y), m);
		let inv_m = T::one() / <T as RealScalar>::from_usize(m);
		let xt = x.transpose_to_owned();
		Self {
			x,
			xt,
			y,
			lambda,
			inv_m,
		}
	}
}

/// Workspace for [`LogisticRegression`].
pub struct LogisticRegressionWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Buffer for margins z = Xw (length m).
	margins: B::Vector,
	/// Buffer for weights (length m) — reused for σ, weighted residuals.
	weights: B::Vector,
	/// Buffer for Euclidean gradient (length n).
	egrad: B::Vector,
	/// Buffer for Euclidean Hessian-vector product (length n).
	ehvp: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for LogisticRegressionWorkspace<T, B> {
	fn default() -> Self {
		Self {
			margins: B::Vector::zeros(0),
			weights: B::Vector::zeros(0),
			egrad: B::Vector::zeros(0),
			ehvp: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for LogisticRegressionWorkspace<T, B>
where
	B::Vector: Send,
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for LogisticRegressionWorkspace<T, B>
where
	B::Vector: Sync,
{
}

/// Numerically stable log(1 + exp(t)).
#[inline]
fn softplus<T: Scalar>(t: T) -> T {
	let abs_t = t.abs();
	abs_t.max(T::zero()) + (T::one() + (-abs_t).exp()).ln()
}

/// Logistic sigmoid σ(t) = 1/(1 + exp(−t)), numerically stable.
#[inline]
fn sigmoid<T: Scalar>(t: T) -> T {
	if t >= T::zero() {
		T::one() / (T::one() + (-t).exp())
	} else {
		let e = t.exp();
		e / (T::one() + e)
	}
}

impl<T, B, M> Problem<T, M> for LogisticRegression<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = LogisticRegressionWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let m = MatrixView::nrows(&self.x);
		let n = VectorView::len(proto_point);
		LogisticRegressionWorkspace {
			margins: B::Vector::zeros(m),
			weights: B::Vector::zeros(m),
			egrad: B::Vector::zeros(n),
			ehvp: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let m = MatrixView::nrows(&self.x);
		let half = <T as Scalar>::from_f64(0.5);
		// Xw → margins
		let xw = self.x.mat_vec(point);
		let mut loss = T::zero();
		for i in 0..m {
			let zi = self.y.get(i) * xw.get(i);
			loss = loss + softplus(-zi);
		}
		self.inv_m * loss + half * self.lambda * point.norm_squared()
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let m = MatrixView::nrows(&self.x);

		// Xw → margins
		self.x.mat_vec_into(point, &mut ws.margins);

		// weights = −y ⊙ σ(−y ⊙ Xw)
		for i in 0..m {
			let yi = self.y.get(i);
			let zi = yi * ws.margins.get(i);
			*ws.weights.get_mut(i) = -yi * sigmoid(-zi);
		}

		// egrad = (1/m) Xᵀ weights + λw
		self.xt.mat_vec_into(&ws.weights, &mut ws.egrad);
		ws.egrad.scale_mut(self.inv_m);
		ws.egrad.axpy(self.lambda, point, T::one());

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
		let m = MatrixView::nrows(&self.x);
		let half = <T as Scalar>::from_f64(0.5);

		self.x.mat_vec_into(point, &mut ws.margins);

		let mut loss = T::zero();
		for i in 0..m {
			let yi = self.y.get(i);
			let zi = yi * ws.margins.get(i);
			loss = loss + softplus(-zi);
			*ws.weights.get_mut(i) = -yi * sigmoid(-zi);
		}
		let cost = self.inv_m * loss + half * self.lambda * point.norm_squared();

		self.xt.mat_vec_into(&ws.weights, &mut ws.egrad);
		ws.egrad.scale_mut(self.inv_m);
		ws.egrad.axpy(self.lambda, point, T::one());

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
		let m = MatrixView::nrows(&self.x);

		// Compute σ_i = σ(y_i x_i^T w)
		self.x.mat_vec_into(point, &mut ws.margins);
		for i in 0..m {
			let zi = self.y.get(i) * ws.margins.get(i);
			*ws.weights.get_mut(i) = sigmoid(zi);
		}

		// Xv → margins (reuse)
		self.x.mat_vec_into(vector, &mut ws.margins);

		// d_i = σ_i (1 − σ_i) (Xv)_i
		for i in 0..m {
			let si = ws.weights.get(i);
			*ws.margins.get_mut(i) = si * (T::one() - si) * ws.margins.get(i);
		}

		// ehvp = (1/m) Xᵀ d + λv
		self.xt.mat_vec_into(&ws.margins, &mut ws.ehvp);
		ws.ehvp.scale_mut(self.inv_m);
		ws.ehvp.axpy(self.lambda, vector, T::one());

		// egrad for curvature correction
		self.x.mat_vec_into(point, &mut ws.margins);
		for i in 0..m {
			let yi = self.y.get(i);
			let zi = yi * ws.margins.get(i);
			*ws.margins.get_mut(i) = -yi * sigmoid(-zi);
		}
		self.xt.mat_vec_into(&ws.margins, &mut ws.egrad);
		ws.egrad.scale_mut(self.inv_m);
		ws.egrad.axpy(self.lambda, point, T::one());

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
