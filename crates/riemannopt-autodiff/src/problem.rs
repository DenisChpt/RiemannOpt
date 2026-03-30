//! `Problem<T, M>` adapters for autodiff-defined cost functions.
//!
//! Two concrete types bridge the gap between a user-provided closure
//! (operating on [`AdSession`] variables) and the solver-facing
//! [`Problem`] trait:
//!
//! * [`AutoDiffProblem`] — for manifolds whose `Point` is `B::Vector`
//!   (Euclidean, Sphere, Hyperbolic, Oblique).
//! * [`AutoDiffMatProblem`] — for manifolds whose `Point` is `B::Matrix`
//!   (Stiefel, Grassmann, SPD, PSD, FixedRank).
//!
//! Both compute Euclidean gradients via reverse-mode AD and then project
//! them to the Riemannian tangent space through the manifold.

use std::fmt::{self, Debug};
use std::marker::PhantomData;

use riemannopt_core::linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView};
use riemannopt_core::manifold::Manifold;
use riemannopt_core::problem::Problem;
use riemannopt_core::types::Scalar;

use crate::session::AdSession;
use crate::var::{MVar, SVar, VVar};

// ═══════════════════════════════════════════════════════════════════════
//  Vector-point variant
// ═══════════════════════════════════════════════════════════════════════

/// Autodiff problem for manifolds where `Point = B::Vector`.
///
/// The closure `F` receives the session and a `VVar` handle representing
/// the current point, and must return an `SVar` representing the scalar
/// cost.
///
/// # Example
///
/// ```ignore
/// let problem = AutoDiffProblem::<f64, FaerBackend, _>::new(|s, x| {
///     let ax = s.mat_vec(a_var, x);
///     s.dot(x, ax)
/// });
/// ```
pub struct AutoDiffProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	F: Fn(&mut AdSession<T, B>, VVar) -> SVar,
{
	f: F,
	_phantom: PhantomData<(T, B)>,
}

impl<T, B, F> Debug for AutoDiffProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	F: Fn(&mut AdSession<T, B>, VVar) -> SVar,
{
	fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt.debug_struct("AutoDiffProblem").finish()
	}
}

impl<T, B, F> AutoDiffProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	F: Fn(&mut AdSession<T, B>, VVar) -> SVar,
{
	/// Creates a new autodiff problem from a closure.
	pub fn new(f: F) -> Self {
		Self {
			f,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`AutoDiffProblem`].
pub struct AutoDiffWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	session: AdSession<T, B>,
	/// Euclidean gradient buffer.
	egrad: B::Vector,
	/// Euclidean Hessian-vector product buffer.
	ehvp: B::Vector,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for AutoDiffWorkspace<T, B> {
	fn default() -> Self {
		Self {
			session: AdSession::new(),
			egrad: B::Vector::zeros(0),
			ehvp: B::Vector::zeros(0),
		}
	}
}

impl<T, B, M, F> Problem<T, M> for AutoDiffProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
	F: Fn(&mut AdSession<T, B>, VVar) -> SVar + Send + Sync + Debug,
{
	type Workspace = AutoDiffWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.len();
		AutoDiffWorkspace {
			session: AdSession::new(),
			egrad: B::Vector::zeros(n),
			ehvp: B::Vector::zeros(n),
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let mut session = AdSession::<T, B>::new();
		let x = session.input_vector(point);
		let loss = (self.f)(&mut session, x);
		session.scalar_value(loss)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.session.reset();
		let x = ws.session.input_vector(point);
		let loss = (self.f)(&mut ws.session, x);
		ws.session.backward(loss);
		ws.egrad.copy_from(ws.session.gradient_vector(x));
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
		ws.session.reset();
		let x = ws.session.input_vector(point);
		let loss = (self.f)(&mut ws.session, x);
		let cost = ws.session.scalar_value(loss);
		ws.session.backward(loss);
		ws.egrad.copy_from(ws.session.gradient_vector(x));
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
		let h = T::FD_CENTRAL_STEP;

		// 1) Euclidean gradient at x (for curvature correction).
		ws.session.reset();
		let x0 = ws.session.input_vector(point);
		let l0 = (self.f)(&mut ws.session, x0);
		ws.session.backward(l0);
		ws.egrad.copy_from(ws.session.gradient_vector(x0));

		// 2) Gradient at x + h·v
		let mut point_plus = point.clone();
		point_plus.axpy(h, vector, T::one());
		ws.session.reset();
		let xp = ws.session.input_vector(&point_plus);
		let lp = (self.f)(&mut ws.session, xp);
		ws.session.backward(lp);
		ws.ehvp.copy_from(ws.session.gradient_vector(xp));

		// 3) Gradient at x - h·v
		let mut point_minus = point.clone();
		point_minus.axpy(T::zero() - h, vector, T::one());
		ws.session.reset();
		let xm = ws.session.input_vector(&point_minus);
		let lm = (self.f)(&mut ws.session, xm);
		ws.session.backward(lm);

		// ehvp = (grad_plus - grad_minus) / (2h)
		ws.ehvp.sub_assign(ws.session.gradient_vector(xm));
		let inv_2h = T::one() / (h + h);
		ws.ehvp.scale_mut(inv_2h);

		// 4) Riemannian correction.
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

// ═══════════════════════════════════════════════════════════════════════
//  Matrix-point variant
// ═══════════════════════════════════════════════════════════════════════

/// Autodiff problem for manifolds where `Point = B::Matrix`.
pub struct AutoDiffMatProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	F: Fn(&mut AdSession<T, B>, MVar) -> SVar,
{
	f: F,
	_phantom: PhantomData<(T, B)>,
}

impl<T, B, F> Debug for AutoDiffMatProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	F: Fn(&mut AdSession<T, B>, MVar) -> SVar,
{
	fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt.debug_struct("AutoDiffMatProblem").finish()
	}
}

impl<T, B, F> AutoDiffMatProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	F: Fn(&mut AdSession<T, B>, MVar) -> SVar,
{
	pub fn new(f: F) -> Self {
		Self {
			f,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`AutoDiffMatProblem`].
pub struct AutoDiffMatWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	session: AdSession<T, B>,
	egrad: B::Matrix,
	ehvp: B::Matrix,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for AutoDiffMatWorkspace<T, B> {
	fn default() -> Self {
		Self {
			session: AdSession::new(),
			egrad: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
		}
	}
}

impl<T, B, M, F> Problem<T, M> for AutoDiffMatProblem<T, B, F>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
	F: Fn(&mut AdSession<T, B>, MVar) -> SVar + Send + Sync + Debug,
{
	type Workspace = AutoDiffMatWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let (r, c) = (proto_point.nrows(), proto_point.ncols());
		AutoDiffMatWorkspace {
			session: AdSession::new(),
			egrad: B::Matrix::zeros(r, c),
			ehvp: B::Matrix::zeros(r, c),
		}
	}

	fn cost(&self, point: &M::Point) -> T {
		let mut session = AdSession::<T, B>::new();
		let x = session.input_matrix(point);
		let loss = (self.f)(&mut session, x);
		session.scalar_value(loss)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.session.reset();
		let x = ws.session.input_matrix(point);
		let loss = (self.f)(&mut ws.session, x);
		ws.session.backward(loss);
		ws.egrad.copy_from(ws.session.gradient_matrix(x));
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
		ws.session.reset();
		let x = ws.session.input_matrix(point);
		let loss = (self.f)(&mut ws.session, x);
		let cost = ws.session.scalar_value(loss);
		ws.session.backward(loss);
		ws.egrad.copy_from(ws.session.gradient_matrix(x));
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
		let h = T::FD_CENTRAL_STEP;

		// Euclidean gradient at x
		ws.session.reset();
		let x0 = ws.session.input_matrix(point);
		let l0 = (self.f)(&mut ws.session, x0);
		ws.session.backward(l0);
		ws.egrad.copy_from(ws.session.gradient_matrix(x0));

		// Gradient at x + h·v
		let mut point_plus = point.clone();
		point_plus.mat_axpy(h, vector, T::one());
		ws.session.reset();
		let xp = ws.session.input_matrix(&point_plus);
		let lp = (self.f)(&mut ws.session, xp);
		ws.session.backward(lp);
		ws.ehvp.copy_from(ws.session.gradient_matrix(xp));

		// Gradient at x - h·v
		let mut point_minus = point.clone();
		point_minus.mat_axpy(T::zero() - h, vector, T::one());
		ws.session.reset();
		let xm = ws.session.input_matrix(&point_minus);
		let lm = (self.f)(&mut ws.session, xm);
		ws.session.backward(lm);

		// ehvp = (grad_plus - grad_minus) / (2h)
		ws.ehvp.sub_assign(ws.session.gradient_matrix(xm));
		let inv_2h = T::one() / (h + h);
		ws.ehvp.scale_mut(inv_2h);

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
