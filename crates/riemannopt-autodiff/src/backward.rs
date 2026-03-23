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

	for i in (0..n).rev() {
		let entry = &tape.entries[i];
		if grads[i].is_empty() || !entry.requires_grad {
			continue;
		}

		let grad = grads[i].clone(); // need owned copy because we mutate grads below

		match entry.op {
			OpCode::Input => {}

			// ── binary element-wise ────────────────────────────────
			OpCode::Add(a, b) => {
				accum(&mut grads[a.0 as usize], &grad);
				accum(&mut grads[b.0 as usize], &grad);
			}
			OpCode::Sub(a, b) => {
				accum(&mut grads[a.0 as usize], &grad);
				let neg: Vec<f64> = grad.iter().map(|g| -g).collect();
				accum(&mut grads[b.0 as usize], &neg);
			}
			OpCode::Mul(a, b) => {
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;
				let ga: Vec<f64> = grad.iter().zip(bv).map(|(g, b)| g * b).collect();
				let gb: Vec<f64> = grad.iter().zip(av).map(|(g, a)| g * a).collect();
				accum(&mut grads[a.0 as usize], &ga);
				accum(&mut grads[b.0 as usize], &gb);
			}
			OpCode::Div(a, b) => {
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;
				let ga: Vec<f64> = grad.iter().zip(bv).map(|(g, b)| g / b).collect();
				let gb: Vec<f64> = grad
					.iter()
					.zip(av.iter().zip(bv))
					.map(|(g, (a, b))| -g * a / (b * b))
					.collect();
				accum(&mut grads[a.0 as usize], &ga);
				accum(&mut grads[b.0 as usize], &gb);
			}

			// ── unary element-wise ─────────────────────────────────
			OpCode::Neg(a) => {
				let ga: Vec<f64> = grad.iter().map(|g| -g).collect();
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Exp(a) => {
				let out = &entry.value;
				let ga: Vec<f64> = grad.iter().zip(out).map(|(g, v)| g * v).collect();
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Log(a) => {
				let av = &tape.entries[a.0 as usize].value;
				let ga: Vec<f64> = grad.iter().zip(av).map(|(g, a)| g / a).collect();
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Sqrt(a) => {
				let out = &entry.value;
				let ga: Vec<f64> = grad.iter().zip(out).map(|(g, v)| g * 0.5 / v).collect();
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Sin(a) => {
				let av = &tape.entries[a.0 as usize].value;
				let ga: Vec<f64> = grad.iter().zip(av).map(|(g, a)| g * a.cos()).collect();
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Cos(a) => {
				let av = &tape.entries[a.0 as usize].value;
				let ga: Vec<f64> = grad.iter().zip(av).map(|(g, a)| -g * a.sin()).collect();
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Abs(a) => {
				let av = &tape.entries[a.0 as usize].value;
				let ga: Vec<f64> = grad.iter().zip(av).map(|(g, a)| g * a.signum()).collect();
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Pow(a, n) => {
				let av = &tape.entries[a.0 as usize].value;
				let nf = n as f64;
				let ga: Vec<f64> = grad
					.iter()
					.zip(av)
					.map(|(g, a)| g * nf * a.powi(n as i32 - 1))
					.collect();
				accum(&mut grads[a.0 as usize], &ga);
			}

			// ── reductions ─────────────────────────────────────────
			OpCode::Sum(a) => {
				let len = tape.entries[a.0 as usize].value.len();
				let g0 = grad[0];
				let ga = vec![g0; len];
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Mean(a) => {
				let len = tape.entries[a.0 as usize].value.len();
				let g0 = grad[0] / len as f64;
				let ga = vec![g0; len];
				accum(&mut grads[a.0 as usize], &ga);
			}
			OpCode::Dot(a, b) => {
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;
				let g0 = grad[0];
				let ga: Vec<f64> = bv.iter().map(|b| g0 * b).collect();
				let gb: Vec<f64> = av.iter().map(|a| g0 * a).collect();
				accum(&mut grads[a.0 as usize], &ga);
				accum(&mut grads[b.0 as usize], &gb);
			}
			OpCode::Norm(a) => {
				let av = &tape.entries[a.0 as usize].value;
				let norm_val = entry.value[0];
				let g0 = grad[0];
				if norm_val > 1e-30 {
					let ga: Vec<f64> = av.iter().map(|a| g0 * a / norm_val).collect();
					accum(&mut grads[a.0 as usize], &ga);
				}
			}

			// ── linear algebra ─────────────────────────────────────
			OpCode::MatMul(a, b, m, k, n) => {
				let (m, k, n) = (m as usize, k as usize, n as usize);
				let av = &tape.entries[a.0 as usize].value;
				let bv = &tape.entries[b.0 as usize].value;
				// grad_A = grad_out × B^T
				let mut ga = vec![0.0; m * k];
				for j in 0..k {
					for p in 0..n {
						let b_jp = bv[j + p * k];
						for ii in 0..m {
							ga[ii + j * m] += grad[ii + p * m] * b_jp;
						}
					}
				}
				// grad_B = A^T × grad_out
				let mut gb = vec![0.0; k * n];
				for j in 0..n {
					for p in 0..m {
						let g_pj = grad[p + j * m];
						for ii in 0..k {
							gb[ii + j * k] += av[p + ii * m] * g_pj;
						}
					}
				}
				accum(&mut grads[a.0 as usize], &ga);
				accum(&mut grads[b.0 as usize], &gb);
			}
			OpCode::Trace(a) => {
				let (r, _) = tape.entries[a.0 as usize].shape;
				let g0 = grad[0];
				let mut ga = vec![0.0; r * r];
				for ii in 0..r {
					ga[ii + ii * r] = g0;
				}
				accum(&mut grads[a.0 as usize], &ga);
			}

			// ── scalar–tensor broadcast ────────────────────────────
			OpCode::ScalarMul(s, t) => {
				let sv = tape.entries[s.0 as usize].value[0];
				let tv = &tape.entries[t.0 as usize].value;
				let gs = vec![grad.iter().zip(tv).map(|(g, v)| g * v).sum()];
				let gt: Vec<f64> = grad.iter().map(|g| g * sv).collect();
				accum(&mut grads[s.0 as usize], &gs);
				accum(&mut grads[t.0 as usize], &gt);
			}
			OpCode::ScalarAdd(s, t) => {
				let gs = vec![grad.iter().sum()];
				accum(&mut grads[s.0 as usize], &gs);
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
