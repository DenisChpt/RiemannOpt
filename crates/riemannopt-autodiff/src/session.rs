//! Autodiff session — owns pool, tape, and gradient pool.
//!
//! The session is the user-facing entry point.  It provides methods to
//! create input variables, build a computation graph (eagerly computing
//! forward values), and run the reverse-mode backward pass.
//!
//! # Lifecycle
//!
//! 1. `session.reset()` — resets cursors and clears tape (no dealloc).
//! 2. `session.input_vector(data)` — copies data into pool, returns `VVar`.
//! 3. Graph-building methods (`add_v`, `dot`, `mat_vec`, …) — compute
//!    values eagerly into pool slots and record [`Op`]s on the tape.
//! 4. `session.backward(loss)` — seeds the gradient and traverses the
//!    tape in reverse, accumulating VJPs in the gradient pool.
//! 5. `session.gradient_vector(var)` — reads the accumulated gradient.

use riemannopt_core::linalg::{
	LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView,
};

use crate::backward;
use crate::pool::BufferPool;
use crate::tape::{Op, Tape};
use crate::var::{MVar, SVar, VVar};

/// Autodiff computation session.
///
/// Holds the value pool (forward), the gradient pool (backward), and the
/// flat instruction tape.
pub struct AdSession<T: RealScalar, B: LinAlgBackend<T>> {
	pub(crate) pool: BufferPool<T, B>,
	pub(crate) tape: Tape,
	pub(crate) grad: BufferPool<T, B>,
	input_vector_count: u32,
	input_matrix_count: u32,
}

impl<T: RealScalar, B: LinAlgBackend<T>> AdSession<T, B> {
	/// Creates a new, empty session.
	pub fn new() -> Self {
		Self {
			pool: BufferPool::new(),
			tape: Tape::new(),
			grad: BufferPool::new(),
			input_vector_count: 0,
			input_matrix_count: 0,
		}
	}

	/// Resets the session for a new forward pass.
	///
	/// Cursors are set to zero and the tape is cleared.  No memory is
	/// freed — all buffers are recycled.
	#[inline]
	pub fn reset(&mut self) {
		self.pool.reset();
		self.grad.reset();
		self.tape.clear();
		self.input_vector_count = 0;
		self.input_matrix_count = 0;
	}

	/// Returns the total capacity (in slots) of the pool and gradient arenas.
	///
	/// Useful for verifying that repeated forward/backward passes do not
	/// allocate after the first iteration.  Returns
	/// `(pool_scalars, pool_vectors, pool_matrices,
	///   grad_scalars, grad_vectors, grad_matrices)`.
	pub fn arena_capacities(&self) -> (usize, usize, usize, usize, usize, usize) {
		(
			self.pool.scalars.capacity(),
			self.pool.vectors.capacity(),
			self.pool.matrices.capacity(),
			self.grad.scalars.capacity(),
			self.grad.vectors.capacity(),
			self.grad.matrices.capacity(),
		)
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Input creation
	// ═══════════════════════════════════════════════════════════════════

	/// Registers a vector as an input variable.
	///
	/// The data is copied into the pool.  Inputs are always the first
	/// slots (indices `0..input_vector_count`), which makes gradient
	/// extraction trivial.
	#[inline]
	pub fn input_vector(&mut self, data: &B::Vector) -> VVar {
		let idx = self.pool.alloc_vector(data.len());
		self.pool.vector_mut(VVar(idx)).copy_from(data);
		self.input_vector_count += 1;
		VVar(idx)
	}

	/// Registers a matrix as an input variable.
	#[inline]
	pub fn input_matrix(&mut self, data: &B::Matrix) -> MVar {
		let idx = self.pool.alloc_matrix(data.nrows(), data.ncols());
		self.pool.matrix_mut(MVar(idx)).copy_from(data);
		self.input_matrix_count += 1;
		MVar(idx)
	}

	/// Creates a constant scalar (not differentiated).
	#[inline]
	pub fn constant_scalar(&mut self, val: T) -> SVar {
		SVar(self.pool.alloc_scalar_with(val))
	}

	/// Creates a constant vector (not differentiated).
	#[inline]
	pub fn constant_vector(&mut self, data: &B::Vector) -> VVar {
		let idx = self.pool.alloc_vector(data.len());
		self.pool.vector_mut(VVar(idx)).copy_from(data);
		VVar(idx)
	}

	/// Creates a constant matrix (not differentiated).
	#[inline]
	pub fn constant_matrix(&mut self, data: &B::Matrix) -> MVar {
		let idx = self.pool.alloc_matrix(data.nrows(), data.ncols());
		self.pool.matrix_mut(MVar(idx)).copy_from(data);
		MVar(idx)
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Scalar operations
	// ═══════════════════════════════════════════════════════════════════

	#[inline]
	pub fn add_s(&mut self, a: SVar, b: SVar) -> SVar {
		let val = self.pool.scalar(a) + self.pool.scalar(b);
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::AddS {
			a: a.0,
			b: b.0,
			out,
		});
		SVar(out)
	}

	#[inline]
	pub fn sub_s(&mut self, a: SVar, b: SVar) -> SVar {
		let val = self.pool.scalar(a) - self.pool.scalar(b);
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::SubS {
			a: a.0,
			b: b.0,
			out,
		});
		SVar(out)
	}

	#[inline]
	pub fn mul_s(&mut self, a: SVar, b: SVar) -> SVar {
		let val = self.pool.scalar(a) * self.pool.scalar(b);
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::MulS {
			a: a.0,
			b: b.0,
			out,
		});
		SVar(out)
	}

	#[inline]
	pub fn div_s(&mut self, a: SVar, b: SVar) -> SVar {
		let val = self.pool.scalar(a) / self.pool.scalar(b);
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::DivS {
			a: a.0,
			b: b.0,
			out,
		});
		SVar(out)
	}

	#[inline]
	pub fn neg_s(&mut self, a: SVar) -> SVar {
		let val = T::zero() - self.pool.scalar(a);
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::NegS { a: a.0, out });
		SVar(out)
	}

	#[inline]
	pub fn exp_s(&mut self, a: SVar) -> SVar {
		let val = self.pool.scalar(a).exp();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::ExpS { a: a.0, out });
		SVar(out)
	}

	#[inline]
	pub fn log_s(&mut self, a: SVar) -> SVar {
		let val = self.pool.scalar(a).ln();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::LogS { a: a.0, out });
		SVar(out)
	}

	#[inline]
	pub fn sqrt_s(&mut self, a: SVar) -> SVar {
		let val = self.pool.scalar(a).sqrt();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::SqrtS { a: a.0, out });
		SVar(out)
	}

	#[inline]
	pub fn sin_s(&mut self, a: SVar) -> SVar {
		let val = self.pool.scalar(a).sin();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::SinS { a: a.0, out });
		SVar(out)
	}

	#[inline]
	pub fn cos_s(&mut self, a: SVar) -> SVar {
		let val = self.pool.scalar(a).cos();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::CosS { a: a.0, out });
		SVar(out)
	}

	#[inline]
	pub fn abs_s(&mut self, a: SVar) -> SVar {
		let val = self.pool.scalar(a).abs();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::AbsS { a: a.0, out });
		SVar(out)
	}

	#[inline]
	pub fn pow_s(&mut self, base: SVar, exp: SVar) -> SVar {
		let val = self.pool.scalar(base).powf(self.pool.scalar(exp));
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::PowS {
			base: base.0,
			exp: exp.0,
			out,
		});
		SVar(out)
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Vector operations
	// ═══════════════════════════════════════════════════════════════════

	#[inline]
	pub fn add_v(&mut self, a: VVar, b: VVar) -> VVar {
		let len = self.pool.vector(a).len();
		let out = self.pool.alloc_vector(len);
		{
			let (src, dst) = self.pool.vec_ref_mut(a.0, out);
			dst.copy_from(src);
		}
		{
			let (src, dst) = self.pool.vec_ref_mut(b.0, out);
			dst.add_assign(src);
		}
		self.tape.push(Op::AddV {
			a: a.0,
			b: b.0,
			out,
		});
		VVar(out)
	}

	#[inline]
	pub fn sub_v(&mut self, a: VVar, b: VVar) -> VVar {
		let len = self.pool.vector(a).len();
		let out = self.pool.alloc_vector(len);
		{
			let (src, dst) = self.pool.vec_ref_mut(a.0, out);
			dst.copy_from(src);
		}
		{
			let (src, dst) = self.pool.vec_ref_mut(b.0, out);
			dst.sub_assign(src);
		}
		self.tape.push(Op::SubV {
			a: a.0,
			b: b.0,
			out,
		});
		VVar(out)
	}

	#[inline]
	pub fn neg_v(&mut self, a: VVar) -> VVar {
		let len = self.pool.vector(a).len();
		let out = self.pool.alloc_vector(len);
		{
			let (src, dst) = self.pool.vec_ref_mut(a.0, out);
			dst.copy_from(src);
		}
		self.pool
			.vector_mut(VVar(out))
			.scale_mut(T::zero() - T::one());
		self.tape.push(Op::NegV { a: a.0, out });
		VVar(out)
	}

	#[inline]
	pub fn component_mul_v(&mut self, a: VVar, b: VVar) -> VVar {
		let len = self.pool.vector(a).len();
		let out = self.pool.alloc_vector(len);
		{
			let (src, dst) = self.pool.vec_ref_mut(a.0, out);
			dst.copy_from(src);
		}
		{
			let (src, dst) = self.pool.vec_ref_mut(b.0, out);
			dst.component_mul_assign(src);
		}
		self.tape.push(Op::ComponentMulV {
			a: a.0,
			b: b.0,
			out,
		});
		VVar(out)
	}

	#[inline]
	pub fn scale_v(&mut self, s: SVar, v: VVar) -> VVar {
		let len = self.pool.vector(v).len();
		let alpha = self.pool.scalar(s);
		let out = self.pool.alloc_vector(len);
		{
			let (src, dst) = self.pool.vec_ref_mut(v.0, out);
			dst.copy_from(src);
		}
		self.pool.vector_mut(VVar(out)).scale_mut(alpha);
		self.tape.push(Op::ScaleV {
			s: s.0,
			v: v.0,
			out,
		});
		VVar(out)
	}

	#[inline]
	pub fn dot(&mut self, a: VVar, b: VVar) -> SVar {
		let val = self.pool.vector(a).dot(self.pool.vector(b));
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::DotV {
			a: a.0,
			b: b.0,
			out,
		});
		SVar(out)
	}

	#[inline]
	pub fn norm_v(&mut self, v: VVar) -> SVar {
		let val = self.pool.vector(v).norm();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::NormV { v: v.0, out });
		SVar(out)
	}

	#[inline]
	pub fn norm_sq_v(&mut self, v: VVar) -> SVar {
		let val = self.pool.vector(v).norm_squared();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::NormSqV { v: v.0, out });
		SVar(out)
	}

	#[inline]
	pub fn sum_v(&mut self, v: VVar) -> SVar {
		let val = self.pool.vector(v).iter().fold(T::zero(), |acc, x| acc + x);
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::SumV { v: v.0, out });
		SVar(out)
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Matrix operations
	// ═══════════════════════════════════════════════════════════════════

	#[inline]
	pub fn add_m(&mut self, a: MVar, b: MVar) -> MVar {
		let (r, c) = (self.pool.matrix(a).nrows(), self.pool.matrix(a).ncols());
		let out = self.pool.alloc_matrix(r, c);
		{
			let (src, dst) = self.pool.mat_ref_mut(a.0, out);
			dst.copy_from(src);
		}
		{
			let (src, dst) = self.pool.mat_ref_mut(b.0, out);
			dst.add_assign(src);
		}
		self.tape.push(Op::AddM {
			a: a.0,
			b: b.0,
			out,
		});
		MVar(out)
	}

	#[inline]
	pub fn sub_m(&mut self, a: MVar, b: MVar) -> MVar {
		let (r, c) = (self.pool.matrix(a).nrows(), self.pool.matrix(a).ncols());
		let out = self.pool.alloc_matrix(r, c);
		{
			let (src, dst) = self.pool.mat_ref_mut(a.0, out);
			dst.copy_from(src);
		}
		{
			let (src, dst) = self.pool.mat_ref_mut(b.0, out);
			dst.sub_assign(src);
		}
		self.tape.push(Op::SubM {
			a: a.0,
			b: b.0,
			out,
		});
		MVar(out)
	}

	#[inline]
	pub fn neg_m(&mut self, a: MVar) -> MVar {
		let (r, c) = (self.pool.matrix(a).nrows(), self.pool.matrix(a).ncols());
		let out = self.pool.alloc_matrix(r, c);
		{
			let (src, dst) = self.pool.mat_ref_mut(a.0, out);
			dst.copy_from(src);
		}
		self.pool
			.matrix_mut(MVar(out))
			.scale_mut(T::zero() - T::one());
		self.tape.push(Op::NegM { a: a.0, out });
		MVar(out)
	}

	#[inline]
	pub fn scale_m(&mut self, s: SVar, m: MVar) -> MVar {
		let (r, c) = (self.pool.matrix(m).nrows(), self.pool.matrix(m).ncols());
		let alpha = self.pool.scalar(s);
		let out = self.pool.alloc_matrix(r, c);
		{
			let (src, dst) = self.pool.mat_ref_mut(m.0, out);
			dst.copy_from(src);
		}
		self.pool.matrix_mut(MVar(out)).scale_mut(alpha);
		self.tape.push(Op::ScaleM {
			s: s.0,
			m: m.0,
			out,
		});
		MVar(out)
	}

	#[inline]
	pub fn mat_mul(&mut self, a: MVar, b: MVar) -> MVar {
		let r = self.pool.matrix(a).nrows();
		let c = self.pool.matrix(b).ncols();
		let out = self.pool.alloc_matrix(r, c);
		{
			let (ma, mb, dst) = self.pool.mat_ref2_mut(a.0, b.0, out);
			dst.gemm(T::one(), ma.as_view(), mb.as_view(), T::zero());
		}
		self.tape.push(Op::MatMul {
			a: a.0,
			b: b.0,
			out,
		});
		MVar(out)
	}

	#[inline]
	pub fn mat_vec(&mut self, m: MVar, v: VVar) -> VVar {
		let nrows = self.pool.matrix(m).nrows();
		let out = self.pool.alloc_vector(nrows);
		// Borrow matrices and vectors arenas independently.
		let mat = &self.pool.matrices[m.idx()];
		let (left, right) = self.pool.vectors.split_at_mut(out as usize);
		let src = &left[v.idx()];
		let dst = &mut right[0];
		mat.mat_vec_into(src, dst);
		self.tape.push(Op::MatVec {
			m: m.0,
			v: v.0,
			out,
		});
		VVar(out)
	}

	#[inline]
	pub fn transpose_m(&mut self, a: MVar) -> MVar {
		let (r, c) = (self.pool.matrix(a).nrows(), self.pool.matrix(a).ncols());
		let out = self.pool.alloc_matrix(c, r);
		{
			let (src, dst) = self.pool.mat_ref_mut(a.0, out);
			for j in 0..c {
				for i in 0..r {
					*MatrixOps::get_mut(dst, j, i) = MatrixView::get(src, i, j);
				}
			}
		}
		self.tape.push(Op::TransposeM { a: a.0, out });
		MVar(out)
	}

	#[inline]
	pub fn trace_m(&mut self, m: MVar) -> SVar {
		let val = self.pool.matrix(m).trace();
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::TraceM { m: m.0, out });
		SVar(out)
	}

	#[inline]
	pub fn frob_dot(&mut self, a: MVar, b: MVar) -> SVar {
		let val = self.pool.matrix(a).frobenius_dot(self.pool.matrix(b));
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::FrobDotM {
			a: a.0,
			b: b.0,
			out,
		});
		SVar(out)
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Fused operations
	// ═══════════════════════════════════════════════════════════════════

	/// Computes `A·x + b` in a single fused operation.
	#[inline]
	pub fn linear_layer(&mut self, a: MVar, x: VVar, b: VVar) -> VVar {
		let nrows = self.pool.matrix(a).nrows();
		let out = self.pool.alloc_vector(nrows);
		// out = A·x
		{
			let mat = &self.pool.matrices[a.idx()];
			let (left, right) = self.pool.vectors.split_at_mut(out as usize);
			let xv = &left[x.idx()];
			let ov = &mut right[0];
			mat.mat_vec_into(xv, ov);
		}
		// out += b
		{
			let (src, dst) = self.pool.vec_ref_mut(b.0, out);
			dst.add_assign(src);
		}
		self.tape.push(Op::LinearLayer {
			a: a.0,
			x: x.0,
			b: b.0,
			out,
		});
		VVar(out)
	}

	/// Computes `xᵀ·A·x` (quadratic form) with a scratch buffer for `A·x`.
	#[inline]
	pub fn quad_form(&mut self, x: VVar, a: MVar) -> SVar {
		let n = self.pool.vector(x).len();
		// Scratch vector for A·x (saved for backward)
		let ax = self.pool.alloc_vector(n);
		// Compute A·x into scratch
		{
			let mat = &self.pool.matrices[a.idx()];
			let (left, right) = self.pool.vectors.split_at_mut(ax as usize);
			let xv = &left[x.idx()];
			let axv = &mut right[0];
			mat.mat_vec_into(xv, axv);
		}
		// dot(x, A·x)
		let val = {
			let (xv, axv) = self.pool.vec_ref_mut(x.0, ax);
			xv.dot(axv)
		};
		let out = self.pool.alloc_scalar_with(val);
		self.tape.push(Op::QuadForm {
			x: x.0,
			a: a.0,
			ax,
			out,
		});
		SVar(out)
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Backward pass
	// ═══════════════════════════════════════════════════════════════════

	/// Runs reverse-mode AD, accumulating gradients in the gradient pool.
	///
	/// The output scalar `loss` is seeded with gradient 1.
	pub fn backward(&mut self, loss: SVar) {
		// Initialise the gradient pool — zero everything, seed output = 1.
		self.grad.ensure_scalars_zeroed(self.pool.scalar_cursor());
		self.grad.ensure_vectors_zeroed_like(&self.pool);
		self.grad.ensure_matrices_zeroed_like(&self.pool);
		*self.grad.scalar_mut(loss) = T::one();

		// Traverse tape in reverse
		backward::backward_pass(&self.pool, &mut self.grad, self.tape.ops());
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Value / gradient extraction
	// ═══════════════════════════════════════════════════════════════════

	/// Reads the scalar value stored at `v`.
	#[inline]
	pub fn scalar_value(&self, v: SVar) -> T {
		self.pool.scalar(v)
	}

	/// Returns the gradient w.r.t. a vector input.
	#[inline]
	pub fn gradient_vector(&self, v: VVar) -> &B::Vector {
		self.grad.vector(v)
	}

	/// Returns the gradient w.r.t. a matrix input.
	#[inline]
	pub fn gradient_matrix(&self, v: MVar) -> &B::Matrix {
		self.grad.matrix(v)
	}
}

impl<T: RealScalar, B: LinAlgBackend<T>> Default for AdSession<T, B> {
	fn default() -> Self {
		Self::new()
	}
}
