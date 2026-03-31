//! Reverse-mode backward pass — VJP accumulation.
//!
//! The core routine [`backward_pass`] traverses the tape in reverse,
//! reading forward values from `pool` (immutable) and accumulating
//! gradients into `grad` (mutable).  Because these are **separate**
//! [`BufferPool`]s, the borrow checker is happy with no unsafe code.
//!
//! # Allocation guarantee
//!
//! After the first call (which may grow the gradient pool), every
//! subsequent `backward_pass` with the same graph topology is
//! **zero-allocation**.

use riemannopt_core::linalg::{
	LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView,
};

use crate::pool::BufferPool;
use crate::tape::Op;
use crate::var::{MVar, SVar, VVar};

// ═══════════════════════════════════════════════════════════════════════
//  Main entry point
// ═══════════════════════════════════════════════════════════════════════

/// Traverses the tape in reverse, accumulating VJPs into `grad`.
///
/// `pool` holds the forward values (read-only), `grad` holds the
/// gradient accumulators (read-write).  The caller must have already
/// seeded the output gradient (typically `grad.scalar_mut(loss) = 1`).
pub fn backward_pass<T: RealScalar, B: LinAlgBackend<T>>(
	pool: &BufferPool<T, B>,
	grad: &mut BufferPool<T, B>,
	ops: &[Op],
) {
	for &op in ops.iter().rev() {
		backward_op::<T, B>(pool, grad, op);
	}
}

// ═══════════════════════════════════════════════════════════════════════
//  Per-operation VJP dispatch
// ═══════════════════════════════════════════════════════════════════════

#[inline]
fn backward_op<T: RealScalar, B: LinAlgBackend<T>>(
	pool: &BufferPool<T, B>,
	grad: &mut BufferPool<T, B>,
	op: Op,
) {
	match op {
		// ── Scalar ← Scalar × Scalar ─────────────────────────────────
		Op::AddS { a, b, out } => {
			let go = grad.scalar(SVar(out));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go;
			let gb = grad.scalar_mut(SVar(b));
			*gb += go;
		}
		Op::SubS { a, b, out } => {
			let go = grad.scalar(SVar(out));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go;
			let gb = grad.scalar_mut(SVar(b));
			*gb -= go;
		}
		Op::MulS { a, b, out } => {
			let go = grad.scalar(SVar(out));
			let va = pool.scalar(SVar(a));
			let vb = pool.scalar(SVar(b));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go * vb;
			let gb = grad.scalar_mut(SVar(b));
			*gb += go * va;
		}
		Op::DivS { a, b, out } => {
			let go = grad.scalar(SVar(out));
			let va = pool.scalar(SVar(a));
			let vb = pool.scalar(SVar(b));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go / vb;
			let gb = grad.scalar_mut(SVar(b));
			*gb -= go * va / (vb * vb);
		}
		Op::PowS { base, exp, out } => {
			let go = grad.scalar(SVar(out));
			let vb = pool.scalar(SVar(base));
			let ve = pool.scalar(SVar(exp));
			let out_val = pool.scalar(SVar(out));
			let gb = grad.scalar_mut(SVar(base));
			*gb += go * ve * vb.powf(ve - T::one());
			let ge = grad.scalar_mut(SVar(exp));
			*ge += go * out_val * vb.ln();
		}

		// ── Scalar ← Scalar (unary) ──────────────────────────────────
		Op::NegS { a, out } => {
			let go = grad.scalar(SVar(out));
			let ga = grad.scalar_mut(SVar(a));
			*ga -= go;
		}
		Op::ExpS { a, out } => {
			let go = grad.scalar(SVar(out));
			let out_val = pool.scalar(SVar(out));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go * out_val;
		}
		Op::LogS { a, out } => {
			let go = grad.scalar(SVar(out));
			let va = pool.scalar(SVar(a));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go / va;
		}
		Op::SqrtS { a, out } => {
			let go = grad.scalar(SVar(out));
			let out_val = pool.scalar(SVar(out));
			let two = T::one() + T::one();
			let ga = grad.scalar_mut(SVar(a));
			*ga += go / (two * out_val);
		}
		Op::SinS { a, out } => {
			let go = grad.scalar(SVar(out));
			let va = pool.scalar(SVar(a));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go * va.cos();
		}
		Op::CosS { a, out } => {
			let go = grad.scalar(SVar(out));
			let va = pool.scalar(SVar(a));
			let ga = grad.scalar_mut(SVar(a));
			*ga -= go * va.sin();
		}
		Op::AbsS { a, out } => {
			let go = grad.scalar(SVar(out));
			let va = pool.scalar(SVar(a));
			let ga = grad.scalar_mut(SVar(a));
			*ga += go * va.signum();
		}

		// ── Vector ← Vector × Vector ─────────────────────────────────
		Op::AddV { a, b, out } => {
			// grad_a += grad_out, grad_b += grad_out
			if a == b {
				let two = T::one() + T::one();
				let (ga, go) = grad.vectors_mut_pair(a, out);
				ga.axpy(two, go, T::one());
			} else {
				{
					let (ga, go) = grad.vectors_mut_pair(a, out);
					ga.add_assign(go);
				}
				{
					let (gb, go) = grad.vectors_mut_pair(b, out);
					gb.add_assign(go);
				}
			}
		}
		Op::SubV { a, b, out } => {
			if a == b {
				// grad_a += grad_out - grad_out = 0 → noop
			} else {
				{
					let (ga, go) = grad.vectors_mut_pair(a, out);
					ga.add_assign(go);
				}
				{
					let (gb, go) = grad.vectors_mut_pair(b, out);
					gb.sub_assign(go);
				}
			}
		}
		Op::ComponentMulV { a, b, out } => {
			// grad_a += grad_out ⊙ val(b)
			// grad_b += grad_out ⊙ val(a)
			let vb_slice = pool.vector(VVar(b)).as_slice();
			{
				let (ga, go) = grad.vectors_mut_pair(a, out);
				let ga_s = ga.as_mut_slice();
				let go_s = go.as_slice();
				for i in 0..ga_s.len() {
					ga_s[i] += go_s[i] * vb_slice[i];
				}
			}
			let va_slice = pool.vector(VVar(a)).as_slice();
			if b != a {
				let (gb, go) = grad.vectors_mut_pair(b, out);
				let gb_s = gb.as_mut_slice();
				let go_s = go.as_slice();
				for i in 0..gb_s.len() {
					gb_s[i] += go_s[i] * va_slice[i];
				}
			} else {
				// a == b:  grad_a += grad_out ⊙ val(a) already done above,
				// and grad_b is the same slot, so repeat
				let (ga, go) = grad.vectors_mut_pair(a, out);
				let ga_s = ga.as_mut_slice();
				let go_s = go.as_slice();
				for i in 0..ga_s.len() {
					ga_s[i] += go_s[i] * va_slice[i];
				}
			}
		}
		Op::NegV { a, out } => {
			let (ga, go) = grad.vectors_mut_pair(a, out);
			ga.sub_assign(go);
		}

		// ── Vector ← Scalar × Vector ─────────────────────────────────
		Op::ScaleV { s, v, out } => {
			let go = grad.vector(VVar(out));
			let val_v = pool.vector(VVar(v));
			let dot_val = go.dot(val_v);
			let gs = grad.scalar_mut(SVar(s));
			*gs += dot_val;
			let val_s = pool.scalar(SVar(s));
			let (gv, go) = grad.vectors_mut_pair(v, out);
			gv.axpy(val_s, go, T::one());
		}

		// ── Scalar ← Vector (reductions) ─────────────────────────────
		Op::DotV { a, b, out } => {
			let go = grad.scalar(SVar(out));
			// grad_a += go * val(b)
			let val_b = pool.vector(VVar(b));
			grad.vector_mut(VVar(a)).axpy(go, val_b, T::one());
			// grad_b += go * val(a)
			if b != a {
				let val_a = pool.vector(VVar(a));
				grad.vector_mut(VVar(b)).axpy(go, val_a, T::one());
			} else {
				// a == b → grad_a already got both contributions
				// Actually no: dot(a, a) → d/da = 2*go*a, first axpy gave go*a,
				// need another go*a.
				let val_a = pool.vector(VVar(a));
				grad.vector_mut(VVar(a)).axpy(go, val_a, T::one());
			}
		}
		Op::NormV { v, out } => {
			// d/dv ||v|| = v / ||v||
			let go = grad.scalar(SVar(out));
			let norm_val = pool.scalar(SVar(out));
			if norm_val > T::zero() {
				let scale = go / norm_val;
				let val_v = pool.vector(VVar(v));
				grad.vector_mut(VVar(v)).axpy(scale, val_v, T::one());
			}
		}
		Op::NormSqV { v, out } => {
			// d/dv ||v||² = 2v
			let go = grad.scalar(SVar(out));
			let two = T::one() + T::one();
			let scale = two * go;
			let val_v = pool.vector(VVar(v));
			grad.vector_mut(VVar(v)).axpy(scale, val_v, T::one());
		}
		Op::SumV { v, out } => {
			// d/dv sum(v) = ones
			let go = grad.scalar(SVar(out));
			let gv = grad.vector_mut(VVar(v));
			let s = gv.as_mut_slice();
			for val in s.iter_mut() {
				*val += go;
			}
		}

		// ── Matrix ← Matrix × Matrix ─────────────────────────────────
		Op::AddM { a, b, out } => {
			if a == b {
				let two = T::one() + T::one();
				let (ga, go) = grad.matrices_mut_pair(a, out);
				ga.mat_axpy(two, go, T::one());
			} else {
				{
					let (ga, go) = grad.matrices_mut_pair(a, out);
					ga.add_assign(go);
				}
				{
					let (gb, go) = grad.matrices_mut_pair(b, out);
					gb.add_assign(go);
				}
			}
		}
		Op::SubM { a, b, out } => {
			if a == b {
				// noop
			} else {
				{
					let (ga, go) = grad.matrices_mut_pair(a, out);
					ga.add_assign(go);
				}
				{
					let (gb, go) = grad.matrices_mut_pair(b, out);
					gb.sub_assign(go);
				}
			}
		}
		Op::MatMul { a, b, out } => {
			// grad_a += grad_out · val(b)^T
			// grad_b += val(a)^T · grad_out
			let val_a = pool.matrix(MVar(a));
			let val_b = pool.matrix(MVar(b));
			{
				let (ga, go) = grad.matrices_mut_pair(a, out);
				ga.gemm_bt(T::one(), go.as_view(), val_b.as_view(), T::one());
			}
			{
				let (gb, go) = grad.matrices_mut_pair(b, out);
				gb.gemm_at(T::one(), val_a.as_view(), go.as_view(), T::one());
			}
		}

		// ── Matrix (unary) ───────────────────────────────────────────
		Op::NegM { a, out } => {
			let (ga, go) = grad.matrices_mut_pair(a, out);
			ga.sub_assign(go);
		}
		Op::TransposeM { a, out } => {
			let (ai, oi) = (a as usize, out as usize);
			debug_assert_ne!(ai, oi);
			if ai < oi {
				let (left, right) = grad.matrices.split_at_mut(oi);
				let go = &right[0];
				let ga = &mut left[ai];
				ga.add_transpose_of(T::one(), go);
			} else {
				let (left, right) = grad.matrices.split_at_mut(ai);
				let go = &left[oi];
				let ga = &mut right[0];
				ga.add_transpose_of(T::one(), go);
			}
		}

		// ── Matrix ← Scalar × Matrix ─────────────────────────────────
		Op::ScaleM { s, m, out } => {
			let go = grad.matrix(MVar(out));
			let val_m = pool.matrix(MVar(m));
			let dot_val = go.frobenius_dot(val_m);
			let gs = grad.scalar_mut(SVar(s));
			*gs += dot_val;
			let val_s = pool.scalar(SVar(s));
			let (gm, go) = grad.matrices_mut_pair(m, out);
			gm.mat_axpy(val_s, go, T::one());
		}

		// ── Vector ← Matrix × Vector ─────────────────────────────────
		Op::MatVec { m, v, out } => {
			let val_m = pool.matrix(MVar(m));
			let val_v = pool.vector(VVar(v));

			// grad_v += M^T · grad_out
			// Borrow grad.vectors for both go and gv via split_at_mut.
			{
				debug_assert_ne!(v, out);
				let (vi, oi) = (v as usize, out as usize);
				if vi < oi {
					let (left, right) = grad.vectors.split_at_mut(oi);
					let go = &right[0];
					let gv = &mut left[vi];
					val_m.mat_t_vec_axpy(T::one(), go, T::one(), gv);
				} else {
					let (left, right) = grad.vectors.split_at_mut(vi);
					let go = &left[oi];
					let gv = &mut right[0];
					val_m.mat_t_vec_axpy(T::one(), go, T::one(), gv);
				}
			}
			// grad_m += grad_out ⊗ val(v)^T
			// Borrow grad.vectors (immutable) and grad.matrices (mutable) independently.
			{
				let go = &grad.vectors[out as usize];
				let gm = &mut grad.matrices[m as usize];
				gm.ger(T::one(), go, val_v);
			}
		}

		// ── Scalar ← Matrix (reductions) ─────────────────────────────
		Op::TraceM { m, out } => {
			// d/dm tr(M) = I  →  gm[i,i] += go
			let go = grad.scalar(SVar(out));
			let gm = grad.matrix_mut(MVar(m));
			let n = gm.nrows().min(gm.ncols());
			for i in 0..n {
				*gm.get_mut(i, i) = *gm.get_mut(i, i) + go;
			}
		}
		Op::FrobDotM { a, b, out } => {
			let go = grad.scalar(SVar(out));
			// grad_a += go * val(b)
			let val_b = pool.matrix(MVar(b));
			grad.matrix_mut(MVar(a)).mat_axpy(go, val_b, T::one());
			// grad_b += go * val(a)
			if b != a {
				let val_a = pool.matrix(MVar(a));
				grad.matrix_mut(MVar(b)).mat_axpy(go, val_a, T::one());
			} else {
				let val_a = pool.matrix(MVar(a));
				grad.matrix_mut(MVar(a)).mat_axpy(go, val_a, T::one());
			}
		}

		// ── Fused operators ──────────────────────────────────────────
		Op::LinearLayer { a, x, b, out } => {
			// out = A·x + b
			let val_a = pool.matrix(MVar(a));
			let val_x = pool.vector(VVar(x));

			// grad_b += grad_out
			{
				let (gb, go) = grad.vectors_mut_pair(b, out);
				gb.add_assign(go);
			}
			// grad_x += A^T · grad_out (both in vectors arena)
			{
				debug_assert_ne!(x, out);
				let (xi, oi) = (x as usize, out as usize);
				if xi < oi {
					let (left, right) = grad.vectors.split_at_mut(oi);
					let go = &right[0];
					let gx = &mut left[xi];
					val_a.mat_t_vec_axpy(T::one(), go, T::one(), gx);
				} else {
					let (left, right) = grad.vectors.split_at_mut(xi);
					let go = &left[oi];
					let gx = &mut right[0];
					val_a.mat_t_vec_axpy(T::one(), go, T::one(), gx);
				}
			}
			// grad_A += grad_out ⊗ val(x)^T (cross-arena: vectors read, matrices write)
			{
				let go = &grad.vectors[out as usize];
				let ga = &mut grad.matrices[a as usize];
				ga.ger(T::one(), go, val_x);
			}
		}
		Op::QuadForm { x, a, ax, out } => {
			// out = x^T · A · x, ax = A·x (saved scratch)
			let go = grad.scalar(SVar(out));
			let val_x = pool.vector(VVar(x));
			let val_ax = pool.vector(VVar(ax));
			let two = T::one() + T::one();

			// grad_x += 2·go · A·x
			grad.vector_mut(VVar(x)).axpy(two * go, val_ax, T::one());

			// grad_A += go · (x ⊗ x^T)  (symmetric: x·x^T)
			grad.matrix_mut(MVar(a)).ger(go, val_x, val_x);
		}
	}
}
