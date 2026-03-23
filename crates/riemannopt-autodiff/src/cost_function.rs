//! Bridges the tape-based AD engine with the [`CostFunction`] trait from
//! `riemannopt-core`, enabling automatic gradient computation for any
//! user-defined function.

use nalgebra::{DMatrix, DVector, Dyn};
use riemannopt_core::{
	cost_function::CostFunction,
	error::{ManifoldError, Result},
	memory::workspace::Workspace,
};

use crate::{backward::backward, Tape, TapeGuard, Var};

/// A [`CostFunction`] whose gradient is computed automatically via reverse-mode
/// AD on each evaluation.
///
/// # Type Parameters
///
/// * `F` — a closure `Fn(Var) -> Var` that builds the computation on the tape.
///
/// # Examples
///
/// ```rust,no_run
/// use riemannopt_autodiff::{AutoDiffCostFunction, Tape, TapeGuard, Var};
/// use riemannopt_core::cost_function::CostFunction;
/// use nalgebra::DVector;
///
/// // f(x) = ||x||^2
/// let cost_fn = AutoDiffCostFunction::new(5, |x| x.dot(x));
///
/// let point = DVector::from_element(5, 1.0);
/// let (c, g) = cost_fn.cost_and_gradient_alloc(&point).unwrap();
/// assert!((c - 5.0).abs() < 1e-12);
/// ```
pub struct AutoDiffCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	f: F,
	dim: usize,
}

impl<F> AutoDiffCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	/// Create a new AD-backed cost function for vectors of dimension `dim`.
	pub fn new(dim: usize, f: F) -> Self {
		Self { f, dim }
	}
}

impl<F> std::fmt::Debug for AutoDiffCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "AutoDiffCostFunction(dim={})", self.dim)
	}
}

impl<F> CostFunction<f64> for AutoDiffCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	type Point = DVector<f64>;
	type TangentVector = DVector<f64>;

	fn cost(&self, point: &DVector<f64>) -> Result<f64> {
		let mut tape = Tape::new();
		let _g = TapeGuard::new(&mut tape);
		let x = tape.var(point.as_slice(), (self.dim, 1));
		let out = (self.f)(x);
		Ok(tape.scalar(out.idx()))
	}

	fn cost_and_gradient_alloc(&self, point: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
		let mut tape = Tape::new();
		let _g = TapeGuard::new(&mut tape);
		let x = tape.var(point.as_slice(), (self.dim, 1));
		let out = (self.f)(x);
		let cost = tape.scalar(out.idx());
		let grads = backward(&tape, out);
		Ok((cost, grads.wrt_vec(x)))
	}

	fn cost_and_gradient(
		&self,
		point: &DVector<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DVector<f64>,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
		self.cost_and_gradient_alloc(point).map(|(_, g)| g)
	}

	fn hessian(&self, _point: &DVector<f64>) -> Result<nalgebra::OMatrix<f64, Dyn, Dyn>> {
		Err(ManifoldError::not_implemented(
			"Hessian not implemented for AutoDiffCostFunction (use hessian_vector_product instead)",
		))
	}

	fn hessian_vector_product(
		&self,
		point: &DVector<f64>,
		vector: &DVector<f64>,
	) -> Result<DVector<f64>> {
		// Forward-difference on the gradient: Hv ≈ (∇f(x+εv) − ∇f(x−εv)) / 2ε
		let eps = 1e-5;
		let gp = self.gradient(&(point + vector * eps))?;
		let gm = self.gradient(&(point - vector * eps))?;
		Ok((gp - gm) / (2.0 * eps))
	}

	fn gradient_fd_alloc(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
		// Use AD — it's exact, no need for finite differences.
		self.gradient(point)
	}

	fn gradient_fd(
		&self,
		point: &DVector<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DVector<f64>,
	) -> Result<()> {
		let g = self.gradient(point)?;
		gradient.copy_from(&g);
		Ok(())
	}
}

/// Same as [`AutoDiffCostFunction`] but for matrix-valued points (e.g. Stiefel, Grassmann).
pub struct AutoDiffMatCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	f: F,
	rows: usize,
	cols: usize,
}

impl<F> AutoDiffMatCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	/// Create a new AD-backed cost function for matrices of shape `(rows, cols)`.
	pub fn new(rows: usize, cols: usize, f: F) -> Self {
		Self { f, rows, cols }
	}
}

impl<F> std::fmt::Debug for AutoDiffMatCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "AutoDiffMatCostFunction({}x{})", self.rows, self.cols)
	}
}

impl<F> CostFunction<f64> for AutoDiffMatCostFunction<F>
where
	F: Fn(Var) -> Var,
{
	type Point = DMatrix<f64>;
	type TangentVector = DMatrix<f64>;

	fn cost(&self, point: &DMatrix<f64>) -> Result<f64> {
		let mut tape = Tape::new();
		let _g = TapeGuard::new(&mut tape);
		let x = tape.var(point.as_slice(), (self.rows, self.cols));
		let out = (self.f)(x);
		Ok(tape.scalar(out.idx()))
	}

	fn cost_and_gradient_alloc(&self, point: &DMatrix<f64>) -> Result<(f64, DMatrix<f64>)> {
		let mut tape = Tape::new();
		let _g = TapeGuard::new(&mut tape);
		let x = tape.var(point.as_slice(), (self.rows, self.cols));
		let out = (self.f)(x);
		let cost = tape.scalar(out.idx());
		let grads = backward(&tape, out);
		Ok((cost, grads.wrt_mat(x, self.rows, self.cols)))
	}

	fn cost_and_gradient(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		self.cost_and_gradient_alloc(point).map(|(_, g)| g)
	}

	fn hessian(&self, _point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		Err(ManifoldError::not_implemented("Hessian not available"))
	}

	fn hessian_vector_product(
		&self,
		point: &DMatrix<f64>,
		vector: &DMatrix<f64>,
	) -> Result<DMatrix<f64>> {
		let eps = 1e-5;
		let gp = self.gradient(&(point + vector * eps))?;
		let gm = self.gradient(&(point - vector * eps))?;
		Ok((gp - gm) / (2.0 * eps))
	}

	fn gradient_fd_alloc(&self, point: &DMatrix<f64>) -> Result<DMatrix<f64>> {
		self.gradient(point)
	}

	fn gradient_fd(
		&self,
		point: &DMatrix<f64>,
		_workspace: &mut Workspace<f64>,
		gradient: &mut DMatrix<f64>,
	) -> Result<()> {
		gradient.copy_from(&self.gradient(point)?);
		Ok(())
	}
}
