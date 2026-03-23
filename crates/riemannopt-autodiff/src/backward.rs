//! Reverse-mode backward pass over the tape.

use crate::tape::{OpCode, Tape};
use crate::Var;

/// Accumulated gradients for every tape entry.
pub struct Gradients {
	pub(crate) grads: Vec<Vec<f64>>,
}

impl Gradients {
	/// Gradient w.r.t. the variable `v` as a flat slice.
	#[inline]
	pub fn wrt(&self, v: Var) -> &[f64] {
		&self.grads[v.idx.0 as usize]
	}

	/// Gradient as a `nalgebra::DVector`.
	pub fn wrt_vec(&self, v: Var) -> nalgebra::DVector<f64> {
		nalgebra::DVector::from_column_slice(self.wrt(v))
	}

	/// Gradient as a `nalgebra::DMatrix` with the given shape.
	pub fn wrt_mat(&self, v: Var, rows: usize, cols: usize) -> nalgebra::DMatrix<f64> {
		nalgebra::DMatrix::from_column_slice(rows, cols, self.wrt(v))
	}
}

/// Run the reverse sweep starting from `output`.
///
/// `output` should be a scalar `(1,1)` node (the loss).
pub fn backward(tape: &Tape, output: Var) -> Gradients {
	let n = tape.entries.len();
	let mut grads: Vec<Vec<f64>> = vec![Vec::new(); n];

	// Seed
	grads[output.idx.0 as usize] = vec![1.0];

	// Reusable scratch buffers to avoid per-node allocations.
	let mut scratch_a: Vec<f64> = Vec::new();
	let mut scratch_b: Vec<f64> = Vec::new();

	for i in (0..n).rev() {
		let entry = &tape.entries[i];
		if grads[i].is_empty() || !entry.requires_grad {
			continue;
		}

		// For Input nodes, the gradient must remain in place so callers can
		// read it via `Gradients::wrt()`.  For all other ops we take ownership
		// without cloning — we never revisit index i in the reverse sweep.
		if matches!(entry.op, OpCode::Input) {
			continue;
		}
		let grad = std::mem::take(&mut grads[i]);

		match entry.op {
			OpCode::Input => unreachable!(),

			// ── binary element-wise ────────────────────────────────
			OpCode::Add(a, b) => {
				accum(&mut grads[a.0 as usize], &grad);
				accum(&mut grads[b.0 as usize], &grad);
			}
			OpCode::Sub(a, b) => {
				accum(&mut grads[a.0 as usize], &grad);
				fill_mapped(&mut scratch_a, &grad, |g| -g);
				accum(&mut grads[b.0 as usize], &scratch_a);
			}
			OpCode::Mul(a, b) => {
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;
				fill_mapped2(&mut scratch_a, &grad, bv, |g, b| g * b);
				fill_mapped2(&mut scratch_b, &grad, av, |g, a| g * a);
				accum(&mut grads[a.0 as usize], &scratch_a);
				accum(&mut grads[b.0 as usize], &scratch_b);
			}
			OpCode::Div(a, b) => {
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;
				fill_mapped2(&mut scratch_a, &grad, bv, |g, b| g / b);
				scratch_b.clear();
				scratch_b.extend(
					grad.iter()
						.zip(av.iter().zip(bv))
						.map(|(g, (a, b))| -g * a / (b * b)),
				);
				accum(&mut grads[a.0 as usize], &scratch_a);
				accum(&mut grads[b.0 as usize], &scratch_b);
			}

			// ── unary element-wise ─────────────────────────────────
			OpCode::Neg(a) => {
				fill_mapped(&mut scratch_a, &grad, |g| -g);
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Exp(a) => {
				let out = &entry.value;
				fill_mapped2(&mut scratch_a, &grad, out, |g, v| g * v);
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Log(a) => {
				let av = &tape.entries[a.0 as usize].value;
				fill_mapped2(&mut scratch_a, &grad, av, |g, a| g / a);
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Sqrt(a) => {
				let out = &entry.value;
				fill_mapped2(&mut scratch_a, &grad, out, |g, v| g * 0.5 / v);
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Sin(a) => {
				let av = &tape.entries[a.0 as usize].value;
				fill_mapped2(&mut scratch_a, &grad, av, |g, a| g * a.cos());
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Cos(a) => {
				let av = &tape.entries[a.0 as usize].value;
				fill_mapped2(&mut scratch_a, &grad, av, |g, a| -g * a.sin());
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Abs(a) => {
				let av = &tape.entries[a.0 as usize].value;
				fill_mapped2(&mut scratch_a, &grad, av, |g, a| g * a.signum());
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Pow(a, n) => {
				let av = &tape.entries[a.0 as usize].value;
				let nf = n as f64;
				scratch_a.clear();
				scratch_a.extend(
					grad.iter()
						.zip(av)
						.map(|(g, a)| g * nf * a.powi(n as i32 - 1)),
				);
				accum(&mut grads[a.0 as usize], &scratch_a);
			}

			// ── reductions ─────────────────────────────────────────
			OpCode::Sum(a) => {
				let len = tape.entries[a.0 as usize].value.len();
				let g0 = grad[0];
				scratch_a.clear();
				scratch_a.resize(len, g0);
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Mean(a) => {
				let len = tape.entries[a.0 as usize].value.len();
				let g0 = grad[0] / len as f64;
				scratch_a.clear();
				scratch_a.resize(len, g0);
				accum(&mut grads[a.0 as usize], &scratch_a);
			}
			OpCode::Dot(a, b) => {
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;
				let g0 = grad[0];
				fill_mapped(&mut scratch_a, bv, |b| g0 * b);
				fill_mapped(&mut scratch_b, av, |a| g0 * a);
				accum(&mut grads[a.0 as usize], &scratch_a);
				accum(&mut grads[b.0 as usize], &scratch_b);
			}
			OpCode::Norm(a) => {
				let av = &tape.entries[a.0 as usize].value;
				let norm_val = entry.value[0];
				let g0 = grad[0];
				if norm_val > 1e-30 {
					fill_mapped(&mut scratch_a, av, |a| g0 * a / norm_val);
					accum(&mut grads[a.0 as usize], &scratch_a);
				}
			}

			// ── linear algebra ─────────────────────────────────────
			OpCode::MatMul(a, b, m, k, n_cols) => {
				let (m, k, n_cols) = (m as usize, k as usize, n_cols as usize);
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;

				// Use nalgebra for MatMul gradients instead of naive triple loops.
				// Column-major layout matches nalgebra's internal representation.
				let grad_out = nalgebra::DMatrixView::from_slice_generic(
					&grad,
					nalgebra::Dyn(m),
					nalgebra::Dyn(n_cols),
				);
				let a_mat = nalgebra::DMatrixView::from_slice_generic(
					av,
					nalgebra::Dyn(m),
					nalgebra::Dyn(k),
				);
				let b_mat = nalgebra::DMatrixView::from_slice_generic(
					bv,
					nalgebra::Dyn(k),
					nalgebra::Dyn(n_cols),
				);

				// grad_A = grad_out * B^T
				let ga_mat = &grad_out * b_mat.transpose();
				// grad_B = A^T * grad_out
				let gb_mat = a_mat.transpose() * &grad_out;

				accum(&mut grads[a.0 as usize], ga_mat.as_slice());
				accum(&mut grads[b.0 as usize], gb_mat.as_slice());
			}
			OpCode::Trace(a) => {
				let (r, _) = tape.entries[a.0 as usize].shape;
				let g0 = grad[0];
				scratch_a.clear();
				scratch_a.resize(r * r, 0.0);
				for ii in 0..r {
					scratch_a[ii + ii * r] = g0;
				}
				accum(&mut grads[a.0 as usize], &scratch_a);
			}

			// ── scalar–tensor broadcast ────────────────────────────
			OpCode::ScalarMul(s, t) => {
				let sv = tape.entries[s.0 as usize].value[0];
				let tv = &tape.entries[t.0 as usize].value;
				let gs_val: f64 = grad.iter().zip(tv).map(|(g, v)| g * v).sum();
				scratch_a.clear();
				scratch_a.push(gs_val);
				fill_mapped(&mut scratch_b, &grad, |g| g * sv);
				accum(&mut grads[s.0 as usize], &scratch_a);
				accum(&mut grads[t.0 as usize], &scratch_b);
			}
			OpCode::ScalarAdd(s, t) => {
				let gs_val: f64 = grad.iter().sum();
				scratch_a.clear();
				scratch_a.push(gs_val);
				accum(&mut grads[s.0 as usize], &scratch_a);
				accum(&mut grads[t.0 as usize], &grad);
			}
		}
	}

	Gradients { grads }
}

/// Accumulate `src` into `dst`, initialising if empty.
#[inline]
fn accum(dst: &mut Vec<f64>, src: &[f64]) {
	if dst.is_empty() {
		*dst = src.to_vec();
	} else {
		debug_assert_eq!(dst.len(), src.len());
		for (d, s) in dst.iter_mut().zip(src) {
			*d += s;
		}
	}
}

/// Fill `buf` by applying `f` element-wise to `src`, reusing the allocation.
#[inline]
fn fill_mapped(buf: &mut Vec<f64>, src: &[f64], f: impl Fn(f64) -> f64) {
	buf.clear();
	buf.extend(src.iter().map(|&x| f(x)));
}

/// Fill `buf` by zipping `a` and `b` through `f`, reusing the allocation.
#[inline]
fn fill_mapped2(buf: &mut Vec<f64>, a: &[f64], b: &[f64], f: impl Fn(f64, f64) -> f64) {
	buf.clear();
	buf.extend(a.iter().zip(b).map(|(&x, &y)| f(x, y)));
}

/// Finite-difference gradient check.  Returns the maximum relative error.
pub fn check_gradient<F>(f: F, x: &[f64], shape: (usize, usize), eps: f64) -> f64
where
	F: Fn(Var) -> Var,
{
	// AD gradient
	let mut tape = Tape::new();
	let _guard = crate::TapeGuard::new(&mut tape);
	let v = tape.var(x, shape);
	let out = f(v);
	let grads = backward(&tape, out);
	let ad_grad = grads.wrt(v).to_vec();
	drop(_guard);

	// Finite-difference gradient
	let n = x.len();
	let mut fd_grad = vec![0.0; n];
	for i in 0..n {
		let mut xp = x.to_vec();
		let mut xm = x.to_vec();
		xp[i] += eps;
		xm[i] -= eps;

		let mut tp = Tape::new();
		let _g = crate::TapeGuard::new(&mut tp);
		let vp = tp.var(&xp, shape);
		let op = f(vp);
		let fp = tp.scalar(op.idx());
		drop(_g);

		let mut tm = Tape::new();
		let _g = crate::TapeGuard::new(&mut tm);
		let vm = tm.var(&xm, shape);
		let om = f(vm);
		let fm = tm.scalar(om.idx());
		drop(_g);

		fd_grad[i] = (fp - fm) / (2.0 * eps);
	}

	let mut max_err = 0.0_f64;
	for (a, f) in ad_grad.iter().zip(&fd_grad) {
		let scale = a.abs().max(f.abs()).max(1e-15);
		let err = (a - f).abs() / scale;
		max_err = max_err.max(err);
	}
	max_err
}
