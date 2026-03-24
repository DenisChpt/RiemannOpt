//! Bridges the tape-based AD engine with the [`CostFunction`] trait from
//! `riemannopt-core`, enabling automatic gradient computation for any
//! user-defined function.

use riemannopt_core::{
	cost_function::CostFunction,
	error::{ManifoldError, Result},
	linalg::{self, MatrixOps, VectorOps},
};

use crate::{backward::backward, Tape, TapeGuard, Var};

/// Collect matrix elements into a contiguous column-major `Vec<f64>`.
///
/// This is needed because some backends (e.g. faer) may pad columns,
/// making `as_slice()` unavailable.
fn mat_to_col_major(m: &linalg::Mat<f64>) -> Vec<f64> {
	let (r, c) = (MatrixOps::nrows(m), MatrixOps::ncols(m));
	let mut buf = Vec::with_capacity(r * c);
	for j in 0..c {
		for i in 0..r {
			buf.push(MatrixOps::get(m, i, j));
		}
	}
	buf
}

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
/// use riemannopt_core::linalg::VectorOps;
///
/// // f(x) = ||x||^2
/// let cost_fn = AutoDiffCostFunction::new(5, |x| x.dot(x));
///
/// let point = VectorOps::from_fn(5, |_| 1.0);
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
	type Point = linalg::Vec<f64>;
	type TangentVector = linalg::Vec<f64>;

	fn cost(&self, point: &linalg::Vec<f64>) -> Result<f64> {
		let mut tape = Tape::new();
		let _g = TapeGuard::new(&mut tape);
		let x = tape.var(point.as_slice(), (self.dim, 1));
		let out = (self.f)(x);
		Ok(tape.scalar(out.idx()))
	}

	fn cost_and_gradient_alloc(&self, point: &linalg::Vec<f64>) -> Result<(f64, linalg::Vec<f64>)> {
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
		point: &linalg::Vec<f64>,
		gradient: &mut linalg::Vec<f64>,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Vec<f64>) -> Result<linalg::Vec<f64>> {
		self.cost_and_gradient_alloc(point).map(|(_, g)| g)
	}

	fn hessian(&self, _point: &linalg::Vec<f64>) -> Result<linalg::Mat<f64>> {
		Err(ManifoldError::not_implemented(
			"Hessian not implemented for AutoDiffCostFunction (use hessian_vector_product instead)",
		))
	}

	fn hessian_vector_product(
		&self,
		point: &linalg::Vec<f64>,
		vector: &linalg::Vec<f64>,
	) -> Result<linalg::Vec<f64>> {
		// Forward-difference on the gradient: Hv ≈ (∇f(x+εv) − ∇f(x−εv)) / 2ε
		let eps = 1e-5;
		let n = VectorOps::len(point);
		let p_plus = VectorOps::from_fn(n, |i| point.get(i) + vector.get(i) * eps);
		let p_minus = VectorOps::from_fn(n, |i| point.get(i) - vector.get(i) * eps);
		let gp = self.gradient(&p_plus)?;
		let gm = self.gradient(&p_minus)?;
		let inv_2eps = 1.0 / (2.0 * eps);
		Ok(VectorOps::from_fn(n, |i| {
			(gp.get(i) - gm.get(i)) * inv_2eps
		}))
	}

	fn gradient_fd_alloc(&self, point: &linalg::Vec<f64>) -> Result<linalg::Vec<f64>> {
		// Use AD — it's exact, no need for finite differences.
		self.gradient(point)
	}

	fn gradient_fd(
		&self,
		point: &linalg::Vec<f64>,
		gradient: &mut linalg::Vec<f64>,
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
	type Point = linalg::Mat<f64>;
	type TangentVector = linalg::Mat<f64>;

	fn cost(&self, point: &linalg::Mat<f64>) -> Result<f64> {
		let mut tape = Tape::new();
		let _g = TapeGuard::new(&mut tape);
		let data = mat_to_col_major(point);
		let x = tape.var(&data, (self.rows, self.cols));
		let out = (self.f)(x);
		Ok(tape.scalar(out.idx()))
	}

	fn cost_and_gradient_alloc(&self, point: &linalg::Mat<f64>) -> Result<(f64, linalg::Mat<f64>)> {
		let mut tape = Tape::new();
		let _g = TapeGuard::new(&mut tape);
		let data = mat_to_col_major(point);
		let x = tape.var(&data, (self.rows, self.cols));
		let out = (self.f)(x);
		let cost = tape.scalar(out.idx());
		let grads = backward(&tape, out);
		Ok((cost, grads.wrt_mat(x, self.rows, self.cols)))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Mat<f64>,
		gradient: &mut linalg::Mat<f64>,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Mat<f64>) -> Result<linalg::Mat<f64>> {
		self.cost_and_gradient_alloc(point).map(|(_, g)| g)
	}

	fn hessian(&self, _point: &linalg::Mat<f64>) -> Result<linalg::Mat<f64>> {
		Err(ManifoldError::not_implemented("Hessian not available"))
	}

	fn hessian_vector_product(
		&self,
		point: &linalg::Mat<f64>,
		vector: &linalg::Mat<f64>,
	) -> Result<linalg::Mat<f64>> {
		let eps = 1e-5;
		let (r, c) = (MatrixOps::nrows(point), MatrixOps::ncols(point));
		let p_plus = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(r, c, |i, j| {
			point.get(i, j) + vector.get(i, j) * eps
		});
		let p_minus = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(r, c, |i, j| {
			point.get(i, j) - vector.get(i, j) * eps
		});
		let gp = self.gradient(&p_plus)?;
		let gm = self.gradient(&p_minus)?;
		let inv_2eps = 1.0 / (2.0 * eps);
		Ok(<linalg::Mat<f64> as MatrixOps<f64>>::from_fn(
			r,
			c,
			|i, j| (gp.get(i, j) - gm.get(i, j)) * inv_2eps,
		))
	}

	fn gradient_fd_alloc(&self, point: &linalg::Mat<f64>) -> Result<linalg::Mat<f64>> {
		self.gradient(point)
	}

	fn gradient_fd(
		&self,
		point: &linalg::Mat<f64>,
		gradient: &mut linalg::Mat<f64>,
	) -> Result<()> {
		gradient.copy_from(&self.gradient(point)?);
		Ok(())
	}
}
