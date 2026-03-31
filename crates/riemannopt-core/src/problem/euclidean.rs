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
/// f(x) = Σᵢ [ a·(xᵢ₊₁ − xᵢ²)² + (b − xᵢ)² ]
#[derive(Debug, Clone)]
pub struct Rosenbrock<T: Scalar, B: LinAlgBackend<T>> {
	pub a: T,
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
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_parameters(a: T, b: T) -> Self {
		Self {
			a,
			b,
			_phantom: PhantomData,
		}
	}
}

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

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RosenbrockWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RosenbrockWorkspace<T, B> where B::Vector: Sync {}

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
	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
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
			*egrad.get_mut(i) = *egrad.get_mut(i) - four * self.a * xi * diff - two * (self.b - xi);
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

		let egrad = &mut ws.egrad;
		egrad.fill(T::zero());
		for i in 0..n - 1 {
			let xi = point.get(i);
			let xi1 = point.get(i + 1);
			let diff = xi1 - xi * xi;
			*egrad.get_mut(i) = *egrad.get_mut(i) - four * self.a * xi * diff - two * (self.b - xi);
			*egrad.get_mut(i + 1) = *egrad.get_mut(i + 1) + two * self.a * diff;
		}

		result.fill(T::zero());
		for i in 0..n - 1 {
			let xi = point.get(i);
			let xi1 = point.get(i + 1);
			let diff = xi1 - xi * xi;
			let vi = vector.get(i);
			let vi1 = vector.get(i + 1);
			*result.get_mut(i) =
				*result.get_mut(i) + (-four * self.a * diff + twelve * self.a * xi * xi + two) * vi;
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
/// f(x) = An + Σᵢ [xᵢ² − A·cos(2πxᵢ)]
#[derive(Debug, Clone)]
pub struct Rastrigin<T: Scalar, B: LinAlgBackend<T>> {
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
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_amplitude(amplitude: T) -> Self {
		Self {
			amplitude,
			_phantom: PhantomData,
		}
	}
}

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

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RastriginWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RastriginWorkspace<T, B> where B::Vector: Sync {}

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
	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
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

/// Linear regression with ℓ₂ regularization.
///
/// f(w) = (1/2m) ‖Xw − y‖² + (λ/2) ‖w‖²
#[derive(Debug, Clone)]
pub struct RidgeRegression<T: Scalar, B: LinAlgBackend<T>> {
	pub x: B::Matrix,
	xtx: B::Matrix,
	xty: B::Vector,
	half_y_sq: T,
	pub y: B::Vector,
	pub lambda: T,
	inv_m: T,
}

impl<T: Scalar, B: LinAlgBackend<T>> RidgeRegression<T, B> {
	pub fn new(x: B::Matrix, y: B::Vector, lambda: T) -> Self {
		debug_assert!(lambda >= T::zero());
		let m = MatrixView::nrows(&x);
		debug_assert_eq!(VectorView::len(&y), m);
		let inv_m = T::one() / <T as RealScalar>::from_usize(m);
		let half = <T as Scalar>::from_f64(0.5);

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

pub struct RidgeRegressionWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
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

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RidgeRegressionWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RidgeRegressionWorkspace<T, B> where
	B::Vector: Sync
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

	/// **Zero allocation** — uses ws.xtxw buffer for XᵀXw.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		self.xtx.mat_vec_into(point, &mut ws.xtxw);
		let quad = half * self.inv_m * point.dot(&ws.xtxw);
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

		self.xtx.mat_vec_into(point, &mut ws.xtxw);
		let quad = half * self.inv_m * point.dot(&ws.xtxw);
		let lin = self.inv_m * point.dot(&self.xty);
		let cost =
			quad - lin + self.inv_m * self.half_y_sq + half * self.lambda * point.norm_squared();

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
		self.xtx.mat_vec_into(vector, &mut ws.xtxw);
		ws.xtxw.scale_mut(self.inv_m);
		ws.xtxw.axpy(self.lambda, vector, T::one());

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
/// f(w) = (1/m) Σᵢ log(1 + exp(−yᵢ · xᵢᵀw)) + (λ/2) ‖w‖²
#[derive(Debug, Clone)]
pub struct LogisticRegression<T: Scalar, B: LinAlgBackend<T>> {
	pub x: B::Matrix,
	xt: B::Matrix,
	pub y: B::Vector,
	pub lambda: T,
	inv_m: T,
}

impl<T: Scalar, B: LinAlgBackend<T>> LogisticRegression<T, B> {
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

pub struct LogisticRegressionWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	margins: B::Vector,
	weights: B::Vector,
	egrad: B::Vector,
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

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for LogisticRegressionWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for LogisticRegressionWorkspace<T, B> where
	B::Vector: Sync
{
}

#[inline]
fn softplus<T: Scalar>(t: T) -> T {
	let abs_t = t.abs();
	abs_t.max(T::zero()) + (T::one() + (-abs_t).exp()).ln()
}

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

	/// **Zero allocation** — uses ws.margins for Xw.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let m = MatrixView::nrows(&self.x);
		let half = <T as Scalar>::from_f64(0.5);
		self.x.mat_vec_into(point, &mut ws.margins);
		let mut loss = T::zero();
		for i in 0..m {
			let zi = self.y.get(i) * ws.margins.get(i);
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
		self.x.mat_vec_into(point, &mut ws.margins);
		for i in 0..m {
			let yi = self.y.get(i);
			let zi = yi * ws.margins.get(i);
			*ws.weights.get_mut(i) = -yi * sigmoid(-zi);
		}
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

		self.x.mat_vec_into(point, &mut ws.margins);
		for i in 0..m {
			let zi = self.y.get(i) * ws.margins.get(i);
			*ws.weights.get_mut(i) = sigmoid(zi);
		}

		self.x.mat_vec_into(vector, &mut ws.margins);
		for i in 0..m {
			let si = ws.weights.get(i);
			*ws.margins.get_mut(i) = si * (T::one() - si) * ws.margins.get(i);
		}

		self.xt.mat_vec_into(&ws.margins, &mut ws.ehvp);
		ws.ehvp.scale_mut(self.inv_m);
		ws.ehvp.axpy(self.lambda, vector, T::one());

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
