//! Reverse-mode backward pass over the tape.
//!
//! Gradients are stored in a single contiguous `Vec<f64>` that mirrors the
//! tape's arena layout — no per-node heap allocation.

use crate::tape::{OpCode, Tape};
use crate::Var;

/// Accumulated gradients for every tape entry.
///
/// All gradient data lives in a single contiguous buffer that mirrors the
/// tape's arena layout.  Accessing a node's gradient is a simple slice
/// operation with O(1) offset lookup.
pub struct Gradients {
	/// Contiguous gradient storage (same total length as `Tape::arena`).
	data: Vec<f64>,
	/// Per-node offset into `data`.
	offsets: Vec<u32>,
	/// Per-node element count.
	lengths: Vec<u32>,
	/// Tracks which nodes had gradients accumulated (avoids O(n) zero-checks).
	has_grad: Vec<bool>,
}

impl Gradients {
	/// Gradient w.r.t. the variable `v` as a flat slice.
	#[inline]
	pub fn wrt(&self, v: Var) -> &[f64] {
		let i = v.idx.0 as usize;
		let off = self.offsets[i] as usize;
		let len = self.lengths[i] as usize;
		&self.data[off..off + len]
	}

	/// Gradient as a `nalgebra::DVector`.
	pub fn wrt_vec(&self, v: Var) -> nalgebra::DVector<f64> {
		nalgebra::DVector::from_column_slice(self.wrt(v))
	}

	/// Gradient as a `nalgebra::DMatrix` with the given shape.
	pub fn wrt_mat(&self, v: Var, rows: usize, cols: usize) -> nalgebra::DMatrix<f64> {
		nalgebra::DMatrix::from_column_slice(rows, cols, self.wrt(v))
	}

	/// Mutable gradient slice for node `i`.
	#[inline]
	fn grad_mut(&mut self, i: usize) -> &mut [f64] {
		let off = self.offsets[i] as usize;
		let len = self.lengths[i] as usize;
		&mut self.data[off..off + len]
	}
}

/// Run the reverse sweep starting from `output`.
///
/// `output` should be a scalar `(1,1)` node (the loss).
pub fn backward(tape: &Tape, output: Var) -> Gradients {
	let n = tape.entries.len();
	let arena_len = tape.arena_len();

	// Build offset/length tables from tape entries.
	let mut offsets = Vec::with_capacity(n);
	let mut lengths = Vec::with_capacity(n);
	for e in &tape.entries {
		offsets.push(e.value_offset);
		lengths.push(e.value_len);
	}

	let mut grads = Gradients {
		data: vec![0.0; arena_len],
		offsets,
		lengths,
		has_grad: vec![false; n],
	};

	// Seed the output gradient to 1.0.
	let out_idx = output.idx.0 as usize;
	grads.grad_mut(out_idx)[0] = 1.0;
	grads.has_grad[out_idx] = true;

	// Reusable scratch buffers to avoid per-node allocations.
	let mut scratch_a: Vec<f64> = Vec::new();
	let mut scratch_b: Vec<f64> = Vec::new();
	let mut grad_buf: Vec<f64> = Vec::new();

	for i in (0..n).rev() {
		let entry = &tape.entries[i];
		if !grads.has_grad[i] || !entry.requires_grad {
			continue;
		}

		// Input nodes: keep gradient in place for callers to read via `wrt()`.
		if matches!(entry.op, OpCode::Input) {
			continue;
		}

		// Copy current node's gradient to a temporary buffer so we can
		// mutably accumulate into parent nodes without aliasing.
		let off = entry.value_offset as usize;
		let len = entry.value_len as usize;
		grad_buf.clear();
		grad_buf.extend_from_slice(&grads.data[off..off + len]);

		match entry.op {
			OpCode::Input => unreachable!(),

			// ── binary element-wise ────────────────────────────────
			OpCode::Add(a, b) => {
				accum(&mut grads, a.0 as usize, &grad_buf);
				accum(&mut grads, b.0 as usize, &grad_buf);
			}
			OpCode::Sub(a, b) => {
				accum(&mut grads, a.0 as usize, &grad_buf);
				fill_mapped(&mut scratch_a, &grad_buf, |g| -g);
				accum(&mut grads, b.0 as usize, &scratch_a);
			}
			OpCode::Mul(a, b) => {
				let av = tape.entry_value(a.0 as usize);
				let bv = tape.entry_value(b.0 as usize);
				fill_mapped2(&mut scratch_a, &grad_buf, bv, |g, b| g * b);
				fill_mapped2(&mut scratch_b, &grad_buf, av, |g, a| g * a);
				accum(&mut grads, a.0 as usize, &scratch_a);
				accum(&mut grads, b.0 as usize, &scratch_b);
			}
			OpCode::Div(a, b) => {
				let av = tape.entry_value(a.0 as usize);
				let bv = tape.entry_value(b.0 as usize);
				fill_mapped2(&mut scratch_a, &grad_buf, bv, |g, b| g / b);
				scratch_b.clear();
				scratch_b.extend(
					grad_buf
						.iter()
						.zip(av.iter().zip(bv))
						.map(|(g, (a, b))| -g * a / (b * b)),
				);
				accum(&mut grads, a.0 as usize, &scratch_a);
				accum(&mut grads, b.0 as usize, &scratch_b);
			}

			// ── unary element-wise ─────────────────────────────────
			OpCode::Neg(a) => {
				fill_mapped(&mut scratch_a, &grad_buf, |g| -g);
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Exp(a) => {
				let out = tape.entry_value(i);
				fill_mapped2(&mut scratch_a, &grad_buf, out, |g, v| g * v);
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Log(a) => {
				let av = tape.entry_value(a.0 as usize);
				fill_mapped2(&mut scratch_a, &grad_buf, av, |g, a| g / a);
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Sqrt(a) => {
				let out = tape.entry_value(i);
				fill_mapped2(&mut scratch_a, &grad_buf, out, |g, v| g * 0.5 / v);
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Sin(a) => {
				let av = tape.entry_value(a.0 as usize);
				fill_mapped2(&mut scratch_a, &grad_buf, av, |g, a| g * a.cos());
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Cos(a) => {
				let av = tape.entry_value(a.0 as usize);
				fill_mapped2(&mut scratch_a, &grad_buf, av, |g, a| -g * a.sin());
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Abs(a) => {
				let av = tape.entry_value(a.0 as usize);
				fill_mapped2(&mut scratch_a, &grad_buf, av, |g, a| g * a.signum());
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Pow(a, n) => {
				let av = tape.entry_value(a.0 as usize);
				let nf = n as f64;
				scratch_a.clear();
				scratch_a.extend(
					grad_buf
						.iter()
						.zip(av)
						.map(|(g, a)| g * nf * a.powi(n as i32 - 1)),
				);
				accum(&mut grads, a.0 as usize, &scratch_a);
			}

			// ── reductions ─────────────────────────────────────────
			OpCode::Sum(a) => {
				let parent_len = tape.entries[a.0 as usize].value_len as usize;
				let g0 = grad_buf[0];
				scratch_a.clear();
				scratch_a.resize(parent_len, g0);
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Mean(a) => {
				let parent_len = tape.entries[a.0 as usize].value_len as usize;
				let g0 = grad_buf[0] / parent_len as f64;
				scratch_a.clear();
				scratch_a.resize(parent_len, g0);
				accum(&mut grads, a.0 as usize, &scratch_a);
			}
			OpCode::Dot(a, b) => {
				let av = tape.entry_value(a.0 as usize);
				let bv = tape.entry_value(b.0 as usize);
				let g0 = grad_buf[0];
				fill_mapped(&mut scratch_a, bv, |b| g0 * b);
				fill_mapped(&mut scratch_b, av, |a| g0 * a);
				accum(&mut grads, a.0 as usize, &scratch_a);
				accum(&mut grads, b.0 as usize, &scratch_b);
			}
			OpCode::Norm(a) => {
				let av = tape.entry_value(a.0 as usize);
				let norm_val = tape.entry_value(i)[0];
				let g0 = grad_buf[0];
				if norm_val > 1e-30 {
					fill_mapped(&mut scratch_a, av, |a| g0 * a / norm_val);
					accum(&mut grads, a.0 as usize, &scratch_a);
				}
			}

			// ── linear algebra ─────────────────────────────────────
			OpCode::MatMul(a, b, m, k, n_cols) => {
				let (m, k, n_cols) = (m as usize, k as usize, n_cols as usize);
				let av = tape.entry_value(a.0 as usize);
				let bv = tape.entry_value(b.0 as usize);

				// Use nalgebra for MatMul gradients.
				let grad_out = nalgebra::DMatrixView::from_slice_generic(
					&grad_buf,
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

				accum(&mut grads, a.0 as usize, ga_mat.as_slice());
				accum(&mut grads, b.0 as usize, gb_mat.as_slice());
			}
			OpCode::Trace(a) => {
				let (r, _) = tape.entries[a.0 as usize].shape;
				let g0 = grad_buf[0];
				scratch_a.clear();
				scratch_a.resize(r * r, 0.0);
				for ii in 0..r {
					scratch_a[ii + ii * r] = g0;
				}
				accum(&mut grads, a.0 as usize, &scratch_a);
			}

			// ── scalar–tensor broadcast ────────────────────────────
			OpCode::ScalarMul(s, t) => {
				let sv = tape.entry_value(s.0 as usize)[0];
				let tv = tape.entry_value(t.0 as usize);
				let gs_val: f64 = grad_buf.iter().zip(tv).map(|(g, v)| g * v).sum();
				scratch_a.clear();
				scratch_a.push(gs_val);
				fill_mapped(&mut scratch_b, &grad_buf, |g| g * sv);
				accum(&mut grads, s.0 as usize, &scratch_a);
				accum(&mut grads, t.0 as usize, &scratch_b);
			}
			OpCode::ScalarAdd(s, t) => {
				let gs_val: f64 = grad_buf.iter().sum();
				scratch_a.clear();
				scratch_a.push(gs_val);
				accum(&mut grads, s.0 as usize, &scratch_a);
				accum(&mut grads, t.0 as usize, &grad_buf);
			}
		}
	}

	grads
}

/// Accumulate `src` into the gradient slot for node `node_idx`.
#[inline]
fn accum(grads: &mut Gradients, node_idx: usize, src: &[f64]) {
	let dst = grads.grad_mut(node_idx);
	debug_assert_eq!(dst.len(), src.len());
	for (d, s) in dst.iter_mut().zip(src) {
		*d += s;
	}
	grads.has_grad[node_idx] = true;
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
///
/// Reuses a single [`Tape`] across all perturbations to avoid repeated
/// heap allocations.
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

	// Finite-difference gradient — reuse tape and perturbation buffers
	let n = x.len();
	let mut fd_grad = vec![0.0; n];
	let mut xp = x.to_vec();
	let mut xm = x.to_vec();

	for i in 0..n {
		xp.copy_from_slice(x);
		xm.copy_from_slice(x);
		xp[i] += eps;
		xm[i] -= eps;

		tape.clear();
		let _g = crate::TapeGuard::new(&mut tape);
		let vp = tape.var(&xp, shape);
		let op = f(vp);
		let fp = tape.scalar(op.idx());
		drop(_g);

		tape.clear();
		let _g = crate::TapeGuard::new(&mut tape);
		let vm = tape.var(&xm, shape);
		let om = f(vm);
		let fm = tape.scalar(om.idx());
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
