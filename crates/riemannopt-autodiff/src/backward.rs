//! Reverse-mode backward pass over the tape.
//!
//! Gradients are stored as backend-native [`NodeValue`] entries — one per tape
//! node — enabling SIMD `axpy` for gradient accumulation and BLAS `gemm_at` /
//! `gemm_bt` for `MatMul` backward without intermediate copies.
//!
//! # Zero-Allocation Design
//!
//! The [`backward_into`] entry point accepts pre-allocated [`Gradients`] and
//! [`ScratchWorkspace`] buffers.  In steady state (same graph structure),
//! no heap allocation occurs.
//!
//! # Split-borrow Pattern
//!
//! The inner loop uses `split_at_mut(i)` on `grads.data` so the current
//! node's gradient (`suffix[0]`, shared ref) and its operands' gradients
//! (`prefix[a]`, mutable) can be accessed without aliasing violations.

use num_traits::Float;

use riemannopt_core::linalg::{self, LinAlgBackend, MatrixOps, RealScalar, VectorOps};

use crate::tape::{OpCode, Tape, TapeThreadLocal};
use crate::value::{NodeValue, ValueKind};
use crate::Var;

// ═══════════════════════════════════════════════════════════════════════════
//  Gradients
// ═══════════════════════════════════════════════════════════════════════════

/// Accumulated gradients for every tape node.
///
/// Each entry is a [`NodeValue`] of the same kind as the corresponding
/// tape node, enabling direct SIMD/BLAS operations on gradient data.
pub struct Gradients<T: RealScalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Per-node gradient storage (same length as `Tape::entries`).
	data: Vec<NodeValue<T>>,
	/// Tracks which nodes had gradients accumulated.
	has_grad: Vec<bool>,
}

impl<T: RealScalar> Gradients<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Create empty gradients (used by workspace for lazy init).
	pub fn empty() -> Self {
		Self {
			data: Vec::new(),
			has_grad: Vec::new(),
		}
	}

	/// Reset gradients for reuse with a (possibly different) tape.
	///
	/// Reuses allocated memory — only grows, never shrinks.
	/// Existing `NodeValue` buffers are zeroed in-place when kinds match;
	/// otherwise new zero buffers are allocated.
	pub fn reset(&mut self, tape: &Tape<T>) {
		let n = tape.len();

		// Resize the data vector to match tape length
		self.data.resize_with(n, || NodeValue::Vacant);

		// For each entry, ensure we have a zero-initialized NodeValue of the
		// correct kind. Reuse existing allocation when kinds match.
		for i in 0..n {
			let kind = tape.entries[i].kind;
			let slot = &mut self.data[i];
			if !slot.is_vacant() && slot.kind() == kind {
				slot.fill_zero();
			} else {
				*slot = NodeValue::zeros(kind);
			}
		}

		// Truncate if tape shrunk
		self.data.truncate(n);

		// Reset has_grad flags
		self.has_grad.resize(n, false);
		for v in &mut self.has_grad[..n] {
			*v = false;
		}
		self.has_grad.truncate(n);
	}

	/// Gradient w.r.t. the variable `v` as a flat slice.
	///
	/// For scalar nodes, returns a single-element slice.
	/// For vector nodes, returns the contiguous backing storage.
	/// For matrix nodes, use [`wrt_mat`] instead (faer matrices may not be contiguous).
	#[inline]
	pub fn wrt(&self, v: Var<T>) -> &[T] {
		let i = v.idx.0 as usize;
		match &self.data[i] {
			NodeValue::Scalar(s) => std::slice::from_ref(s),
			NodeValue::Vector(v) => v.as_slice(),
			NodeValue::Matrix(_) => {
				panic!("wrt() not supported for matrix gradients (use wrt_mat instead)")
			}
			NodeValue::Vacant => panic!("wrt() on Vacant gradient"),
		}
	}

	/// Gradient as a backend vector (cloned).
	pub fn wrt_vec(&self, v: Var<T>) -> linalg::Vec<T> {
		VectorOps::from_slice(self.wrt(v))
	}

	/// Gradient as a backend matrix.
	pub fn wrt_mat(&self, v: Var<T>, rows: usize, cols: usize) -> linalg::Mat<T> {
		let i = v.idx.0 as usize;
		match &self.data[i] {
			NodeValue::Matrix(m) => m.clone(),
			NodeValue::Vector(vec) => {
				<linalg::Mat<T> as MatrixOps<T>>::from_column_slice(rows, cols, vec.as_slice())
			}
			_ => panic!("wrt_mat: not a matrix gradient"),
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  ScratchWorkspace
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-allocated scratch buffers for element-wise backward ops.
///
/// Stored in the caller's workspace and reused across evaluations to
/// avoid per-call allocation.  Buffers only grow, never shrink.
pub(crate) struct ScratchWorkspace<T: RealScalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub(crate) vec_a: linalg::Vec<T>,
	pub(crate) vec_b: linalg::Vec<T>,
}

impl<T: RealScalar> ScratchWorkspace<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Create a new workspace with size-1 vectors.
	pub(crate) fn new() -> Self {
		Self {
			vec_a: VectorOps::zeros(1),
			vec_b: VectorOps::zeros(1),
		}
	}

	/// Ensure both scratch vectors have at least `n` elements.
	///
	/// Only grows — never shrinks.  On growth, a new zero vector is
	/// allocated (the old one is dropped, but in steady state this
	/// never triggers).
	pub(crate) fn ensure_vec_size(&mut self, n: usize) {
		if VectorOps::len(&self.vec_a) < n {
			self.vec_a = VectorOps::zeros(n);
		}
		if VectorOps::len(&self.vec_b) < n {
			self.vec_b = VectorOps::zeros(n);
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Backward pass — allocating wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Run the reverse sweep, allocating fresh [`Gradients`].
///
/// For zero-allocation hot paths, use [`backward_into`] with pre-allocated
/// buffers.
/// Ensure a gradient slot is a Matrix of the given shape.
/// If it's a Vector, convert it to an n×1 matrix preserving accumulated values.
fn ensure_mat_grad<T: RealScalar>(slot: &mut NodeValue<T>, rows: usize, cols: usize)
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	match slot {
		NodeValue::Matrix(m) if MatrixOps::nrows(m) == rows && MatrixOps::ncols(m) == cols => {}
		NodeValue::Vector(v) if VectorOps::len(v) == rows && cols == 1 => {
			let data = v.as_slice().to_vec();
			*slot = NodeValue::Matrix(<linalg::Mat<T> as MatrixOps<T>>::from_column_slice(
				rows, cols, &data,
			));
		}
		NodeValue::Vacant => {
			*slot = NodeValue::Matrix(MatrixOps::zeros(rows, cols));
		}
		_ => {
			*slot = NodeValue::Matrix(MatrixOps::zeros(rows, cols));
		}
	}
}

pub fn backward<T: RealScalar>(tape: &Tape<T>, output: Var<T>) -> Gradients<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	let mut grads = Gradients::empty();
	let mut scratch = ScratchWorkspace::new();
	backward_into(tape, output, &mut grads, &mut scratch);
	grads
}

// ═══════════════════════════════════════════════════════════════════════════
//  Backward pass — zero-alloc hot path
// ═══════════════════════════════════════════════════════════════════════════

/// Run the reverse sweep into pre-allocated gradient and scratch buffers.
///
/// This is the zero-allocation entry point: in steady state (same graph
/// structure), no heap allocation occurs.
///
/// # Algorithm
///
/// 1. Reset gradients to zero.
/// 2. Seed the output gradient to `Scalar(1.0)`.
/// 3. Reverse-iterate over tape entries.
/// 4. For each node with accumulated gradient, propagate to operands
///    using `split_at_mut` for zero-copy borrow splitting.
pub(crate) fn backward_into<T: RealScalar>(
	tape: &Tape<T>,
	output: Var<T>,
	grads: &mut Gradients<T>,
	scratch: &mut ScratchWorkspace<T>,
) where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	grads.reset(tape);

	let n = tape.len();

	// Seed the output gradient to 1.
	let out_idx = output.idx.0 as usize;
	grads.data[out_idx] = NodeValue::Scalar(T::one());
	grads.has_grad[out_idx] = true;

	for i in (0..n).rev() {
		let entry = &tape.entries[i];
		if !grads.has_grad[i] || !entry.requires_grad {
			continue;
		}

		if matches!(entry.op, OpCode::Input) {
			continue;
		}

		match entry.op {
			OpCode::Input => unreachable!(),

			// ── Add(a, b): grad_a += gi, grad_b += gi ────────────
			OpCode::Add(a, b) => {
				let (prefix, suffix) = grads.data.split_at_mut(i);
				let gi = &suffix[0];
				prefix[a.0 as usize].axpy_accum(T::one(), gi);
				prefix[b.0 as usize].axpy_accum(T::one(), gi);
				grads.has_grad[a.0 as usize] = true;
				grads.has_grad[b.0 as usize] = true;
			}

			// ── Sub(a, b): grad_a += gi, grad_b -= gi ────────────
			OpCode::Sub(a, b) => {
				let (prefix, suffix) = grads.data.split_at_mut(i);
				let gi = &suffix[0];
				prefix[a.0 as usize].axpy_accum(T::one(), gi);
				prefix[b.0 as usize].axpy_accum(T::zero() - T::one(), gi);
				grads.has_grad[a.0 as usize] = true;
				grads.has_grad[b.0 as usize] = true;
			}

			// ── Neg(a): grad_a -= gi ─────────────────────────────
			OpCode::Neg(a) => {
				let (prefix, suffix) = grads.data.split_at_mut(i);
				let gi = &suffix[0];
				prefix[a.0 as usize].axpy_accum(T::zero() - T::one(), gi);
				grads.has_grad[a.0 as usize] = true;
			}

			// ── Mul(a, b): element-wise, scratch-based ───────────
			// grad_a += gi * b_val, grad_b += gi * a_val
			OpCode::Mul(a, b) => {
				let a_idx = a.0 as usize;
				let b_idx = b.0 as usize;
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let a_val = tape.values[a_idx].as_scalar();
						let b_val = tape.values[b_idx].as_scalar();
						*grads.data[a_idx].as_scalar_mut() += gi_s * b_val;
						*grads.data[b_idx].as_scalar_mut() += gi_s * a_val;
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let b_val = tape.values[b_idx].as_vec();
						scratch.vec_a.copy_from(gi_vec);
						scratch.vec_a.component_mul_assign(b_val);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);

						let a_val = tape.values[a_idx].as_vec();
						scratch.vec_b.copy_from(gi_vec);
						scratch.vec_b.component_mul_assign(a_val);
						prefix[b_idx].as_vec_mut().add_assign(&scratch.vec_b);
					}
					ValueKind::Matrix(r, c) => {
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_mat = suffix[0].as_mat();
						let b_val = tape.values[b_idx].as_mat();
						let a_val = tape.values[a_idx].as_mat();

						let mut tmp = <linalg::Mat<T> as MatrixOps<T>>::zeros(r, c);
						tmp.copy_from(gi_mat);
						tmp.mat_component_mul_assign(b_val);
						prefix[a_idx].as_mat_mut().add_assign(&tmp);

						tmp.copy_from(gi_mat);
						tmp.mat_component_mul_assign(a_val);
						prefix[b_idx].as_mat_mut().add_assign(&tmp);
					}
				}
				grads.has_grad[a_idx] = true;
				grads.has_grad[b_idx] = true;
			}

			// ── Div(a, b): grad_a += gi/b, grad_b += -gi*a/b^2 ──
			OpCode::Div(a, b) => {
				let a_idx = a.0 as usize;
				let b_idx = b.0 as usize;
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let a_val = tape.values[a_idx].as_scalar();
						let b_val = tape.values[b_idx].as_scalar();
						*grads.data[a_idx].as_scalar_mut() += gi_s / b_val;
						*grads.data[b_idx].as_scalar_mut() +=
							(T::zero() - gi_s) * a_val / (b_val * b_val);
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let a_val = tape.values[a_idx].as_vec();
						let b_val = tape.values[b_idx].as_vec();

						// grad_a = gi / b
						scratch.vec_a.copy_from(gi_vec);
						scratch.vec_a.component_div_assign(b_val);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);

						// grad_b = -gi * a / b^2
						let gi_sl = gi_vec.as_slice();
						let a_sl = a_val.as_slice();
						let b_sl = b_val.as_slice();
						let dst = scratch.vec_b.as_mut_slice();
						for j in 0..n {
							dst[j] = (T::zero() - gi_sl[j]) * a_sl[j] / (b_sl[j] * b_sl[j]);
						}
						prefix[b_idx].as_vec_mut().add_assign(&scratch.vec_b);
					}
					_ => panic!("Div backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
				grads.has_grad[b_idx] = true;
			}

			// ── Exp(a): grad_a += gi * out_val ───────────────────
			OpCode::Exp(a) => {
				let a_idx = a.0 as usize;
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let out_val = tape.values[i].as_scalar();
						*grads.data[a_idx].as_scalar_mut() += gi_s * out_val;
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let out_val = tape.values[i].as_vec();
						scratch.vec_a.copy_from(gi_vec);
						scratch.vec_a.component_mul_assign(out_val);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Exp backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Log(a): grad_a += gi / a_val ─────────────────────
			OpCode::Log(a) => {
				let a_idx = a.0 as usize;
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let a_val = tape.values[a_idx].as_scalar();
						*grads.data[a_idx].as_scalar_mut() += gi_s / a_val;
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let a_val = tape.values[a_idx].as_vec();
						scratch.vec_a.copy_from(gi_vec);
						scratch.vec_a.component_div_assign(a_val);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Log backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Sqrt(a): grad_a += gi * 0.5 / out_val ───────────
			OpCode::Sqrt(a) => {
				let a_idx = a.0 as usize;
				let half = T::from_f64_const(0.5);
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let out_val = tape.values[i].as_scalar();
						*grads.data[a_idx].as_scalar_mut() += gi_s * half / out_val;
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let out_val = tape.values[i].as_vec();
						let gi_sl = gi_vec.as_slice();
						let out_sl = out_val.as_slice();
						let dst = scratch.vec_a.as_mut_slice();
						for j in 0..n {
							dst[j] = gi_sl[j] * half / out_sl[j];
						}
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Sqrt backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Sin(a): grad_a += gi * cos(a_val) ────────────────
			OpCode::Sin(a) => {
				let a_idx = a.0 as usize;
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let a_val = tape.values[a_idx].as_scalar();
						*grads.data[a_idx].as_scalar_mut() += gi_s * Float::cos(a_val);
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let a_val = tape.values[a_idx].as_vec();
						scratch.vec_a.copy_from(a_val);
						scratch.vec_a.map_mut(Float::cos);
						scratch.vec_a.component_mul_assign(gi_vec);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Sin backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Cos(a): grad_a += gi * (-sin(a_val)) ────────────
			OpCode::Cos(a) => {
				let a_idx = a.0 as usize;
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let a_val = tape.values[a_idx].as_scalar();
						*grads.data[a_idx].as_scalar_mut() +=
							(T::zero() - gi_s) * Float::sin(a_val);
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let a_val = tape.values[a_idx].as_vec();
						scratch.vec_a.copy_from(a_val);
						scratch.vec_a.map_mut(|x| T::zero() - Float::sin(x));
						scratch.vec_a.component_mul_assign(gi_vec);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Cos backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Abs(a): grad_a += gi * signum(a_val) ─────────────
			OpCode::Abs(a) => {
				let a_idx = a.0 as usize;
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let a_val = tape.values[a_idx].as_scalar();
						*grads.data[a_idx].as_scalar_mut() += gi_s * Float::signum(a_val);
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let a_val = tape.values[a_idx].as_vec();
						scratch.vec_a.copy_from(a_val);
						scratch.vec_a.map_mut(Float::signum);
						scratch.vec_a.component_mul_assign(gi_vec);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Abs backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Pow(a, n): grad_a += gi * n * a_val^(n-1) ───────
			OpCode::Pow(a, pow) => {
				let a_idx = a.0 as usize;
				let nf = T::from_f64_const(pow as f64);
				let kind = tape.entries[i].kind;
				match kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let a_val = tape.values[a_idx].as_scalar();
						*grads.data[a_idx].as_scalar_mut() +=
							gi_s * nf * Float::powi(a_val, pow as i32 - 1);
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let a_val = tape.values[a_idx].as_vec();
						scratch.vec_a.copy_from(a_val);
						scratch
							.vec_a
							.map_mut(|x| nf * Float::powi(x, pow as i32 - 1));
						scratch.vec_a.component_mul_assign(gi_vec);
						prefix[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Pow backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Sum(a): grad_a += gi_scalar * ones ───────────────
			OpCode::Sum(a) => {
				let a_idx = a.0 as usize;
				let g0 = grads.data[i].as_scalar();
				let a_kind = tape.entries[a_idx].kind;
				match a_kind {
					ValueKind::Scalar => {
						*grads.data[a_idx].as_scalar_mut() += g0;
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						scratch.vec_a.fill(g0);
						grads.data[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					ValueKind::Matrix(r, c) => {
						let mut tmp = <linalg::Mat<T> as MatrixOps<T>>::zeros(r, c);
						tmp.fill(g0);
						grads.data[a_idx].as_mat_mut().add_assign(&tmp);
					}
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Mean(a): grad_a += gi_scalar / n * ones ──────────
			OpCode::Mean(a) => {
				let a_idx = a.0 as usize;
				let a_kind = tape.entries[a_idx].kind;
				let count = a_kind.len();
				let g0 = grads.data[i].as_scalar() / <T as RealScalar>::from_usize(count);
				match a_kind {
					ValueKind::Scalar => {
						*grads.data[a_idx].as_scalar_mut() += g0;
					}
					ValueKind::Vector(n) => {
						scratch.ensure_vec_size(n);
						scratch.vec_a.fill(g0);
						grads.data[a_idx].as_vec_mut().add_assign(&scratch.vec_a);
					}
					_ => panic!("Mean backward: unsupported for matrices"),
				}
				grads.has_grad[a_idx] = true;
			}

			// ── Dot(a, b): g0 = gi; grad_a += g0*b_val, grad_b += g0*a_val
			// Read the scalar g0 by value first, then mutate operands.
			OpCode::Dot(a, b) => {
				let a_idx = a.0 as usize;
				let b_idx = b.0 as usize;
				let g0 = grads.data[i].as_scalar();
				let a_val = tape.values[a_idx].as_vec();
				let b_val = tape.values[b_idx].as_vec();
				// grad_a += g0 * b_val
				grads.data[a_idx].as_vec_mut().axpy(g0, b_val, T::one());
				// grad_b += g0 * a_val
				grads.data[b_idx].as_vec_mut().axpy(g0, a_val, T::one());
				grads.has_grad[a_idx] = true;
				grads.has_grad[b_idx] = true;
			}

			// ── Norm(a): grad_a += (gi / ‖a‖) * a_val ───────────
			OpCode::Norm(a) => {
				let a_idx = a.0 as usize;
				let g0 = grads.data[i].as_scalar();
				let norm_val = tape.values[i].as_scalar();
				let threshold = T::from_f64_const(1e-30);
				if norm_val > threshold {
					let scale = g0 / norm_val;
					let a_val = tape.values[a_idx].as_vec();
					grads.data[a_idx].as_vec_mut().axpy(scale, a_val, T::one());
				}
				grads.has_grad[a_idx] = true;
			}

			// ── MatMul(a, b, m, k, n) ───────────────────────────
			// grad_A = grad_out * B^T   (via gemm_bt)
			// grad_B = A^T * grad_out   (via gemm_at)
			OpCode::MatMul(a, b, _m, _k, _n) => {
				let a_idx = a.0 as usize;
				let b_idx = b.0 as usize;

				// Use split_at_mut so we can read suffix[0] (grad_out)
				// while mutating prefix[a] and prefix[b].
				// Convert to matrices (vectors become n×1)
				let grad_out_m = grads.data[i].to_mat();
				let a_mat = tape.values[a_idx].to_mat();
				let b_mat = tape.values[b_idx].to_mat();

				// grad_A += grad_out * B^T — ensure grad slot is a matrix
				ensure_mat_grad(
					&mut grads.data[a_idx],
					MatrixOps::nrows(&a_mat),
					MatrixOps::ncols(&a_mat),
				);
				grads.data[a_idx]
					.as_mat_mut()
					.gemm_bt(T::one(), &grad_out_m, &b_mat, T::one());

				// grad_B += A^T * grad_out — ensure grad slot is a matrix
				ensure_mat_grad(
					&mut grads.data[b_idx],
					MatrixOps::nrows(&b_mat),
					MatrixOps::ncols(&b_mat),
				);
				grads.data[b_idx]
					.as_mat_mut()
					.gemm_at(T::one(), &a_mat, &grad_out_m, T::one());

				grads.has_grad[a_idx] = true;
				grads.has_grad[b_idx] = true;
			}

			// ── Trace(a): grad_a += g0 * I ───────────────────────
			OpCode::Trace(a) => {
				let a_idx = a.0 as usize;
				let g0 = grads.data[i].as_scalar();
				let (r, _c) = match tape.entries[a_idx].kind {
					ValueKind::Matrix(r, c) => (r, c),
					_ => panic!("Trace backward: operand is not a matrix"),
				};
				let grad_a = grads.data[a_idx].as_mat_mut();
				for ii in 0..r {
					*grad_a.get_mut(ii, ii) += g0;
				}
				grads.has_grad[a_idx] = true;
			}

			// ── ScalarMul(s, t): s is scalar, t is tensor ────────
			// grad_s += dot(gi, t_val)
			// grad_t += s_val * gi
			OpCode::ScalarMul(s, t) => {
				let s_idx = s.0 as usize;
				let t_idx = t.0 as usize;
				let s_val = tape.values[s_idx].as_scalar();
				let t_kind = tape.entries[t_idx].kind;

				match t_kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						let t_val = tape.values[t_idx].as_scalar();
						*grads.data[s_idx].as_scalar_mut() += gi_s * t_val;
						*grads.data[t_idx].as_scalar_mut() += gi_s * s_val;
					}
					ValueKind::Vector(_) => {
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_vec = suffix[0].as_vec();
						let t_val = tape.values[t_idx].as_vec();
						let gs_val = VectorOps::dot(gi_vec, t_val);
						// grad_s += dot(gi, t_val)
						*prefix[s_idx].as_scalar_mut() += gs_val;
						// grad_t += s_val * gi
						prefix[t_idx].as_vec_mut().axpy(s_val, gi_vec, T::one());
					}
					ValueKind::Matrix(r, c) => {
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi_mat = suffix[0].as_mat();
						let t_mat = tape.values[t_idx].as_mat();
						// dot(gi_mat, t_mat) via element-wise access
						let mut gs_val = T::zero();
						for jj in 0..c {
							for ii in 0..r {
								gs_val = gs_val
									+ MatrixOps::get(gi_mat, ii, jj)
										* MatrixOps::get(t_mat, ii, jj);
							}
						}
						*prefix[s_idx].as_scalar_mut() += gs_val;
						// grad_t += s_val * gi_mat
						prefix[t_idx].as_mat_mut().mat_axpy(s_val, gi_mat, T::one());
					}
				}
				grads.has_grad[s_idx] = true;
				grads.has_grad[t_idx] = true;
			}

			// ── ScalarAdd(s, t): s is scalar, t is tensor ────────
			// grad_s += sum(gi)
			// grad_t += gi
			OpCode::ScalarAdd(s, t) => {
				let s_idx = s.0 as usize;
				let t_idx = t.0 as usize;
				let t_kind = tape.entries[t_idx].kind;

				match t_kind {
					ValueKind::Scalar => {
						let gi_s = grads.data[i].as_scalar();
						*grads.data[s_idx].as_scalar_mut() += gi_s;
						*grads.data[t_idx].as_scalar_mut() += gi_s;
					}
					ValueKind::Vector(_) => {
						// grad_s += sum(gi)
						let gi_vec = grads.data[i].as_vec();
						let gs_val: T = gi_vec.iter().fold(T::zero(), |acc, x| acc + x);
						*grads.data[s_idx].as_scalar_mut() += gs_val;
						// grad_t += gi
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi = &suffix[0];
						prefix[t_idx].axpy_accum(T::one(), gi);
					}
					ValueKind::Matrix(r, c) => {
						// grad_s += sum(gi)
						let gi_mat = grads.data[i].as_mat();
						let mut gs_val = T::zero();
						for jj in 0..c {
							for ii in 0..r {
								gs_val = gs_val + MatrixOps::get(gi_mat, ii, jj);
							}
						}
						*grads.data[s_idx].as_scalar_mut() += gs_val;
						// grad_t += gi
						let (prefix, suffix) = grads.data.split_at_mut(i);
						let gi = &suffix[0];
						prefix[t_idx].axpy_accum(T::one(), gi);
					}
				}
				grads.has_grad[s_idx] = true;
				grads.has_grad[t_idx] = true;
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Gradient checking
// ═══════════════════════════════════════════════════════════════════════════

/// Finite-difference gradient check.  Returns the maximum relative error
/// between the AD gradient and a centered finite-difference approximation.
///
/// Reuses a single [`Tape`] across all perturbations.
///
/// # Arguments
///
/// * `f` — the function to differentiate (maps `Var<T>` → `Var<T>`)
/// * `x` — the evaluation point as a flat slice
/// * `shape` — `(rows, cols)` of the input variable
/// * `eps` — perturbation step for finite differences
///
/// # Returns
///
/// The maximum component-wise relative error:
/// `max_i |ad_i - fd_i| / max(|ad_i|, |fd_i|, tiny)`.
pub fn check_gradient<T, F>(f: F, x: &[T], shape: (usize, usize), eps: T) -> T
where
	T: TapeThreadLocal,
	F: Fn(Var<T>) -> Var<T>,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	// AD gradient
	let mut tape = Tape::new();
	let _guard = crate::TapeGuard::new(&mut tape);
	let v = tape.var(x, shape);
	let out = f(v);
	let grads = backward(&tape, out);
	// Extract AD gradient as flat slice (handle both vector and matrix nodes)
	let ad_grad: Vec<T> = match &grads.data[v.idx.0 as usize] {
		NodeValue::Scalar(s) => vec![*s],
		NodeValue::Vector(vec) => vec.as_slice().to_vec(),
		NodeValue::Matrix(m) => {
			let (r, c) = (MatrixOps::nrows(m), MatrixOps::ncols(m));
			let mut buf = Vec::with_capacity(r * c);
			for j in 0..c {
				for i in 0..r {
					buf.push(MatrixOps::get(m, i, j));
				}
			}
			buf
		}
		NodeValue::Vacant => panic!("check_gradient: gradient is Vacant"),
	};
	drop(_guard);

	// Finite-difference gradient — reuse tape and perturbation buffers
	let n = x.len();
	let mut fd_grad = vec![T::zero(); n];
	let mut xp = x.to_vec();
	let mut xm = x.to_vec();
	let two = T::from_f64_const(2.0);

	for i in 0..n {
		xp.copy_from_slice(x);
		xm.copy_from_slice(x);
		xp[i] = xp[i] + eps;
		xm[i] = xm[i] - eps;

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

		fd_grad[i] = (fp - fm) / (two * eps);
	}

	let mut max_err = T::zero();
	let tiny = T::from_f64_const(1e-15);
	for (&a, &fd) in ad_grad.iter().zip(&fd_grad) {
		let scale = Float::max(Float::max(Float::abs(a), Float::abs(fd)), tiny);
		let err = Float::abs(a - fd) / scale;
		max_err = Float::max(max_err, err);
	}
	max_err
}
