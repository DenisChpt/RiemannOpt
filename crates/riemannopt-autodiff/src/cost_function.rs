//! Bridges the tape-based AD engine with the [`CostFunction`] trait.
//!
//! Both wrappers embed an [`AdWorkspace`] that is reused across evaluations.
//! In steady state (same graph structure), the buffer pool covers all
//! allocations and no heap allocation occurs.

use std::cell::RefCell;

use riemannopt_core::{
	cost_function::CostFunction,
	error::{ManifoldError, Result},
	linalg::{self, LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView},
	types::Scalar,
};

use crate::backward::{backward_into, Gradients, ScratchWorkspace};
use crate::tape::{Tape, TapeThreadLocal};
use crate::Var;

// ═══════════════════════════════════════════════════════════════════════════
//  AdWorkspace — reused across evaluations
// ═══════════════════════════════════════════════════════════════════════════

struct AdWorkspace<T: RealScalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	tape: Tape<T>,
	grads: Gradients<T>,
	scratch: ScratchWorkspace<T>,
}

impl<T: RealScalar> AdWorkspace<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn new() -> Self {
		Self {
			tape: Tape::new(),
			grads: Gradients::empty(),
			scratch: ScratchWorkspace::new(),
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  AutoDiffCostFunction (vector-valued points)
// ═══════════════════════════════════════════════════════════════════════════

/// A [`CostFunction`] whose gradient is computed via reverse-mode AD.
///
/// Embeds an internal workspace: tape, gradient buffers, and scratch
/// are reused across calls — zero allocation in steady state.
pub struct AutoDiffCostFunction<T: RealScalar, F>
where
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	f: F,
	dim: usize,
	ws: RefCell<AdWorkspace<T>>,
}

impl<T: RealScalar, F> AutoDiffCostFunction<T, F>
where
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub fn new(dim: usize, f: F) -> Self {
		Self {
			f,
			dim,
			ws: RefCell::new(AdWorkspace::new()),
		}
	}
}

impl<T: RealScalar, F> std::fmt::Debug for AutoDiffCostFunction<T, F>
where
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "AutoDiffCostFunction(dim={})", self.dim)
	}
}

impl<T, F> CostFunction<T> for AutoDiffCostFunction<T, F>
where
	T: TapeThreadLocal + Scalar,
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;

	fn cost(&self, point: &linalg::Vec<T>) -> Result<T> {
		let mut ws = self.ws.borrow_mut();
		let AdWorkspace { tape, .. } = &mut *ws;
		tape.clear_for_reuse();
		let _g = crate::TapeGuard::new(tape);
		let x = tape.var(point.as_slice(), (self.dim, 1));
		let out = (self.f)(x);
		Ok(tape.scalar(out.idx()))
	}

	fn cost_and_gradient_alloc(&self, point: &linalg::Vec<T>) -> Result<(T, linalg::Vec<T>)> {
		let mut ws = self.ws.borrow_mut();
		let AdWorkspace {
			tape,
			grads,
			scratch,
		} = &mut *ws;
		tape.clear_for_reuse();
		let _g = crate::TapeGuard::new(tape);
		let x = tape.var(point.as_slice(), (self.dim, 1));
		let out = (self.f)(x);
		let cost = tape.scalar(out.idx());
		tape.compute_graph_metadata();
		backward_into(tape, out, grads, scratch);
		Ok((cost, grads.wrt_vec(x)))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Vec<T>,
		gradient: &mut linalg::Vec<T>,
	) -> Result<T> {
		let mut ws = self.ws.borrow_mut();
		let AdWorkspace {
			tape,
			grads,
			scratch,
		} = &mut *ws;
		tape.clear_for_reuse();
		let _g = crate::TapeGuard::new(tape);
		let x = tape.var(point.as_slice(), (self.dim, 1));
		let out = (self.f)(x);
		let cost = tape.scalar(out.idx());
		tape.compute_graph_metadata();
		backward_into(tape, out, grads, scratch);
		let grad_slice = grads.wrt(x);
		gradient.as_mut_slice().copy_from_slice(grad_slice);
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Vec<T>) -> Result<linalg::Vec<T>> {
		self.cost_and_gradient_alloc(point).map(|(_, g)| g)
	}

	fn hessian(&self, _point: &linalg::Vec<T>) -> Result<linalg::Mat<T>> {
		Err(ManifoldError::not_implemented(
			"Hessian not implemented (use hessian_vector_product instead)",
		))
	}

	fn hessian_vector_product(
		&self,
		point: &linalg::Vec<T>,
		vector: &linalg::Vec<T>,
	) -> Result<linalg::Vec<T>> {
		let eps = T::from_f64_const(1e-5);
		let n = VectorView::len(point);
		let p_plus = VectorOps::from_fn(n, |i| point.get(i) + vector.get(i) * eps);
		let p_minus = VectorOps::from_fn(n, |i| point.get(i) - vector.get(i) * eps);
		let gp = self.gradient(&p_plus)?;
		let gm = self.gradient(&p_minus)?;
		let inv_2eps = T::one() / (T::from_f64_const(2.0) * eps);
		Ok(VectorOps::from_fn(n, |i| {
			(gp.get(i) - gm.get(i)) * inv_2eps
		}))
	}

	fn gradient_fd_alloc(&self, point: &linalg::Vec<T>) -> Result<linalg::Vec<T>> {
		self.gradient(point)
	}

	fn gradient_fd(&self, point: &linalg::Vec<T>, gradient: &mut linalg::Vec<T>) -> Result<()> {
		self.cost_and_gradient(point, gradient)?;
		Ok(())
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  AutoDiffMatCostFunction (matrix-valued points)
// ═══════════════════════════════════════════════════════════════════════════

/// Same as [`AutoDiffCostFunction`] but for matrix-valued points.
pub struct AutoDiffMatCostFunction<T: RealScalar, F>
where
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	f: F,
	rows: usize,
	cols: usize,
	ws: RefCell<AdWorkspace<T>>,
	/// Reusable buffer for column-major extraction.
	col_buf: RefCell<Vec<T>>,
}

impl<T: RealScalar, F> AutoDiffMatCostFunction<T, F>
where
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub fn new(rows: usize, cols: usize, f: F) -> Self {
		Self {
			f,
			rows,
			cols,
			ws: RefCell::new(AdWorkspace::new()),
			col_buf: RefCell::new(Vec::new()),
		}
	}
}

impl<T: RealScalar, F> std::fmt::Debug for AutoDiffMatCostFunction<T, F>
where
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "AutoDiffMatCostFunction({}x{})", self.rows, self.cols)
	}
}

/// Extract column-major data from a matrix into a buffer.
fn mat_to_buf<T: RealScalar>(m: &linalg::Mat<T>, buf: &mut Vec<T>)
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	let (r, c) = (MatrixView::nrows(m), MatrixView::ncols(m));
	let total = r * c;
	buf.resize(total, T::zero());
	for j in 0..c {
		for i in 0..r {
			buf[i + j * r] = MatrixView::get(m, i, j);
		}
	}
}

impl<T, F> CostFunction<T> for AutoDiffMatCostFunction<T, F>
where
	T: TapeThreadLocal + Scalar,
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Mat<T>;
	type TangentVector = linalg::Mat<T>;

	fn cost(&self, point: &linalg::Mat<T>) -> Result<T> {
		let mut ws = self.ws.borrow_mut();
		let mut col_buf = self.col_buf.borrow_mut();
		let AdWorkspace { tape, .. } = &mut *ws;
		tape.clear_for_reuse();
		mat_to_buf(point, &mut col_buf);
		let _g = crate::TapeGuard::new(tape);
		let x = tape.var(&col_buf, (self.rows, self.cols));
		let out = (self.f)(x);
		Ok(tape.scalar(out.idx()))
	}

	fn cost_and_gradient_alloc(&self, point: &linalg::Mat<T>) -> Result<(T, linalg::Mat<T>)> {
		let mut ws = self.ws.borrow_mut();
		let mut col_buf = self.col_buf.borrow_mut();
		let AdWorkspace {
			tape,
			grads,
			scratch,
		} = &mut *ws;
		tape.clear_for_reuse();
		mat_to_buf(point, &mut col_buf);
		let _g = crate::TapeGuard::new(tape);
		let x = tape.var(&col_buf, (self.rows, self.cols));
		let out = (self.f)(x);
		let cost = tape.scalar(out.idx());
		tape.compute_graph_metadata();
		backward_into(tape, out, grads, scratch);
		Ok((cost, grads.wrt_mat(x, self.rows, self.cols)))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Mat<T>,
		gradient: &mut linalg::Mat<T>,
	) -> Result<T> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		self.cost_and_gradient_alloc(point).map(|(_, g)| g)
	}

	fn hessian(&self, _point: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		Err(ManifoldError::not_implemented("Hessian not available"))
	}

	fn hessian_vector_product(
		&self,
		point: &linalg::Mat<T>,
		vector: &linalg::Mat<T>,
	) -> Result<linalg::Mat<T>> {
		let eps = T::from_f64_const(1e-5);
		let (r, c) = (MatrixView::nrows(point), MatrixView::ncols(point));
		let p_plus = <linalg::Mat<T> as MatrixOps<T>>::from_fn(r, c, |i, j| {
			MatrixView::get(point, i, j) + MatrixView::get(vector, i, j) * eps
		});
		let p_minus = <linalg::Mat<T> as MatrixOps<T>>::from_fn(r, c, |i, j| {
			MatrixView::get(point, i, j) - MatrixView::get(vector, i, j) * eps
		});
		let gp = self.gradient(&p_plus)?;
		let gm = self.gradient(&p_minus)?;
		let inv_2eps = T::one() / (T::from_f64_const(2.0) * eps);
		Ok(<linalg::Mat<T> as MatrixOps<T>>::from_fn(r, c, |i, j| {
			(MatrixView::get(&gp, i, j) - MatrixView::get(&gm, i, j)) * inv_2eps
		}))
	}

	fn gradient_fd_alloc(&self, point: &linalg::Mat<T>) -> Result<linalg::Mat<T>> {
		self.gradient(point)
	}

	fn gradient_fd(&self, point: &linalg::Mat<T>, gradient: &mut linalg::Mat<T>) -> Result<()> {
		gradient.copy_from(&self.gradient(point)?);
		Ok(())
	}
}
