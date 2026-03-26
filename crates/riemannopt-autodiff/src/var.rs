//! [`Var`] — a lightweight handle to a tape entry with operator overloading.
//!
//! All forward-pass operations use backend-native types (SIMD).

use num_traits::Float;

use riemannopt_core::linalg::{
	self, LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView,
};

use crate::tape::{with_tape, NodeIdx, OpCode, Tape, TapeThreadLocal};
use crate::value::{NodeValue, ValueKind};

/// A differentiable variable — just an index into the active [`Tape`].
#[derive(Debug, Clone, Copy)]
pub struct Var<T: RealScalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub(crate) idx: NodeIdx,
	pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T: RealScalar> Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	#[inline]
	pub fn idx(self) -> NodeIdx {
		self.idx
	}

	#[inline]
	pub(crate) fn new(idx: NodeIdx) -> Self {
		Self {
			idx,
			_marker: std::marker::PhantomData,
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Construction on Tape
// ═══════════════════════════════════════════════════════════════════════════

impl<T: RealScalar> Tape<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Record a differentiable input variable from a slice.
	pub fn var(&mut self, data: &[T], shape: (usize, usize)) -> Var<T> {
		debug_assert_eq!(data.len(), shape.0 * shape.1);
		let (kind, value) = slice_to_node_value(data, shape);
		let idx = self.push(OpCode::Input, value, kind, true);
		Var::new(idx)
	}

	/// Record a non-differentiable constant from a slice.
	pub fn constant(&mut self, data: &[T], shape: (usize, usize)) -> Var<T> {
		debug_assert_eq!(data.len(), shape.0 * shape.1);
		let (kind, value) = slice_to_node_value(data, shape);
		let idx = self.push(OpCode::Input, value, kind, false);
		Var::new(idx)
	}

	/// Scalar variable (1x1).
	pub fn scalar_var(&mut self, v: T) -> Var<T> {
		let idx = self.push(OpCode::Input, NodeValue::Scalar(v), ValueKind::Scalar, true);
		Var::new(idx)
	}

	/// Scalar constant (1x1, no grad).
	pub fn scalar_const(&mut self, v: T) -> Var<T> {
		let idx = self.push(
			OpCode::Input,
			NodeValue::Scalar(v),
			ValueKind::Scalar,
			false,
		);
		Var::new(idx)
	}
}

fn slice_to_node_value<T: RealScalar>(
	data: &[T],
	shape: (usize, usize),
) -> (ValueKind, NodeValue<T>)
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	if shape == (1, 1) {
		(ValueKind::Scalar, NodeValue::Scalar(data[0]))
	} else if shape.1 == 1 {
		(
			ValueKind::Vector(shape.0),
			NodeValue::Vector(VectorOps::from_slice(data)),
		)
	} else {
		(
			ValueKind::Matrix(shape.0, shape.1),
			NodeValue::Matrix(<linalg::Mat<T> as MatrixOps<T>>::from_column_slice(
				shape.0, shape.1, data,
			)),
		)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a NodeValue to a matrix view (vectors become n×1 matrices).
fn vec_to_mat<T: RealScalar>(nv: &NodeValue<T>, rows: usize, cols: usize) -> linalg::Mat<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	match nv {
		NodeValue::Matrix(m) => m.clone(),
		NodeValue::Vector(v) => {
			<linalg::Mat<T> as MatrixOps<T>>::from_column_slice(rows, cols, v.as_slice())
		}
		_ => panic!("vec_to_mat: unsupported variant"),
	}
}

/// Get a vector buffer for a unary op from the pool, copying the operand's data.
///
/// Donation (zero-copy reuse of the operand's buffer) is not safe during
/// forward tracing because the graph may diverge after the donation point.
/// A future replay mode (tape-driven, not closure-driven) will enable donation.
#[inline]
fn get_vec_unary<T: RealScalar>(tape: &mut Tape<T>, operand: NodeIdx, n: usize) -> linalg::Vec<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	let src = tape.values[operand.0 as usize].as_vec();
	let mut buf = tape.pool.get_vec(n);
	buf.copy_from(src);
	buf
}

/// Get a vector buffer for a binary op from the pool, copying `lhs` data.
#[inline]
fn get_vec_binary<T: RealScalar>(
	tape: &mut Tape<T>,
	lhs: NodeIdx,
	_rhs: NodeIdx,
	n: usize,
) -> linalg::Vec<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	let src = tape.values[lhs.0 as usize].as_vec();
	let mut buf = tape.pool.get_vec(n);
	buf.copy_from(src);
	buf
}

#[inline]
fn entry_rg(entries: &[crate::tape::TapeEntry], idx: NodeIdx) -> bool {
	entries[idx.0 as usize].requires_grad
}

#[inline]
fn entry_kind(entries: &[crate::tape::TapeEntry], idx: NodeIdx) -> ValueKind {
	entries[idx.0 as usize].kind
}

// ═══════════════════════════════════════════════════════════════════════════
//  Binary ops — uses split borrows on Tape fields
// ═══════════════════════════════════════════════════════════════════════════

impl<T: TapeThreadLocal> std::ops::Add for Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Output = Var<T>;
	fn add(self, rhs: Var<T>) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let a_kind = entry_kind(&tape.entries, self.idx);
			let b_kind = entry_kind(&tape.entries, rhs.idx);
			let req_grad = entry_rg(&tape.entries, self.idx) || entry_rg(&tape.entries, rhs.idx);

			// Scalar + tensor broadcast
			if a_kind == ValueKind::Scalar && b_kind != ValueKind::Scalar {
				let s = tape.values[self.idx.0 as usize].as_scalar();
				let bv = tape.values[rhs.idx.0 as usize].as_vec();
				let n = VectorView::len(bv);
				let mut r = tape.pool.get_vec(n);
				r.copy_from(bv);
				let sl = r.as_mut_slice();
				for v in sl.iter_mut() {
					*v = *v + s;
				}
				return Var::new(tape.push(
					OpCode::ScalarAdd(self.idx, rhs.idx),
					NodeValue::Vector(r),
					b_kind,
					req_grad,
				));
			}
			if b_kind == ValueKind::Scalar && a_kind != ValueKind::Scalar {
				let s = tape.values[rhs.idx.0 as usize].as_scalar();
				let av = tape.values[self.idx.0 as usize].as_vec();
				let n = VectorView::len(av);
				let mut r = tape.pool.get_vec(n);
				r.copy_from(av);
				let sl = r.as_mut_slice();
				for v in sl.iter_mut() {
					*v = *v + s;
				}
				return Var::new(tape.push(
					OpCode::ScalarAdd(rhs.idx, self.idx),
					NodeValue::Vector(r),
					a_kind,
					req_grad,
				));
			}

			assert_eq!(a_kind, b_kind, "Add: shape mismatch");
			match a_kind {
				ValueKind::Scalar => {
					let a = tape.values[self.idx.0 as usize].as_scalar();
					let b = tape.values[rhs.idx.0 as usize].as_scalar();
					Var::new(tape.push(
						OpCode::Add(self.idx, rhs.idx),
						NodeValue::Scalar(a + b),
						ValueKind::Scalar,
						req_grad,
					))
				}
				ValueKind::Vector(n) => {
					let mut r = get_vec_binary(tape, self.idx, rhs.idx, n);
					let bv = tape.values[rhs.idx.0 as usize].as_vec();
					r.add_assign(bv); // SIMD
					Var::new(tape.push(
						OpCode::Add(self.idx, rhs.idx),
						NodeValue::Vector(r),
						ValueKind::Vector(n),
						req_grad,
					))
				}
				ValueKind::Matrix(rows, cols) => {
					let am = tape.values[self.idx.0 as usize].as_mat();
					let bm = tape.values[rhs.idx.0 as usize].as_mat();
					let r = MatrixOps::add(am, bm);
					Var::new(tape.push(
						OpCode::Add(self.idx, rhs.idx),
						NodeValue::Matrix(r),
						ValueKind::Matrix(rows, cols),
						req_grad,
					))
				}
			}
		})
	}
}

impl<T: TapeThreadLocal> std::ops::Sub for Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Output = Var<T>;
	fn sub(self, rhs: Var<T>) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let a_kind = entry_kind(&tape.entries, self.idx);
			let req_grad = entry_rg(&tape.entries, self.idx) || entry_rg(&tape.entries, rhs.idx);
			match a_kind {
				ValueKind::Scalar => {
					let a = tape.values[self.idx.0 as usize].as_scalar();
					let b = tape.values[rhs.idx.0 as usize].as_scalar();
					Var::new(tape.push(
						OpCode::Sub(self.idx, rhs.idx),
						NodeValue::Scalar(a - b),
						ValueKind::Scalar,
						req_grad,
					))
				}
				ValueKind::Vector(n) => {
					let mut r = get_vec_binary(tape, self.idx, rhs.idx, n);
					let bv = tape.values[rhs.idx.0 as usize].as_vec();
					r.sub_assign(bv); // SIMD
					Var::new(tape.push(
						OpCode::Sub(self.idx, rhs.idx),
						NodeValue::Vector(r),
						ValueKind::Vector(n),
						req_grad,
					))
				}
				ValueKind::Matrix(rows, cols) => {
					let am = tape.values[self.idx.0 as usize].as_mat();
					let bm = tape.values[rhs.idx.0 as usize].as_mat();
					let r = MatrixOps::sub(am, bm);
					Var::new(tape.push(
						OpCode::Sub(self.idx, rhs.idx),
						NodeValue::Matrix(r),
						ValueKind::Matrix(rows, cols),
						req_grad,
					))
				}
			}
		})
	}
}

impl<T: TapeThreadLocal> std::ops::Mul for Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Output = Var<T>;
	fn mul(self, rhs: Var<T>) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let a_kind = entry_kind(&tape.entries, self.idx);
			let b_kind = entry_kind(&tape.entries, rhs.idx);
			let req_grad = entry_rg(&tape.entries, self.idx) || entry_rg(&tape.entries, rhs.idx);

			// Scalar * tensor broadcast
			if a_kind == ValueKind::Scalar && b_kind != ValueKind::Scalar {
				let s = tape.values[self.idx.0 as usize].as_scalar();
				let bv = tape.values[rhs.idx.0 as usize].as_vec();
				let n = VectorView::len(bv);
				let mut r = tape.pool.get_vec(n);
				r.copy_from(bv);
				r.scale_mut(s);
				return Var::new(tape.push(
					OpCode::ScalarMul(self.idx, rhs.idx),
					NodeValue::Vector(r),
					b_kind,
					req_grad,
				));
			}
			if b_kind == ValueKind::Scalar && a_kind != ValueKind::Scalar {
				let s = tape.values[rhs.idx.0 as usize].as_scalar();
				let av = tape.values[self.idx.0 as usize].as_vec();
				let n = VectorView::len(av);
				let mut r = tape.pool.get_vec(n);
				r.copy_from(av);
				r.scale_mut(s);
				return Var::new(tape.push(
					OpCode::ScalarMul(rhs.idx, self.idx),
					NodeValue::Vector(r),
					a_kind,
					req_grad,
				));
			}

			assert_eq!(a_kind, b_kind, "Mul: shape mismatch");
			match a_kind {
				ValueKind::Scalar => {
					let a = tape.values[self.idx.0 as usize].as_scalar();
					let b = tape.values[rhs.idx.0 as usize].as_scalar();
					Var::new(tape.push(
						OpCode::Mul(self.idx, rhs.idx),
						NodeValue::Scalar(a * b),
						ValueKind::Scalar,
						req_grad,
					))
				}
				ValueKind::Vector(n) => {
					let mut r = get_vec_binary(tape, self.idx, rhs.idx, n);
					let bv = tape.values[rhs.idx.0 as usize].as_vec();
					r.component_mul_assign(bv); // SIMD via faer::zip!
					Var::new(tape.push(
						OpCode::Mul(self.idx, rhs.idx),
						NodeValue::Vector(r),
						ValueKind::Vector(n),
						req_grad,
					))
				}
				ValueKind::Matrix(rows, cols) => {
					let am = tape.values[self.idx.0 as usize].as_mat();
					let bm = tape.values[rhs.idx.0 as usize].as_mat();
					let mut r = tape.pool.get_mat(rows, cols);
					r.copy_from(am);
					r.mat_component_mul_assign(bm); // SIMD via faer::zip!
					Var::new(tape.push(
						OpCode::Mul(self.idx, rhs.idx),
						NodeValue::Matrix(r),
						ValueKind::Matrix(rows, cols),
						req_grad,
					))
				}
			}
		})
	}
}

impl<T: TapeThreadLocal> std::ops::Div for Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Output = Var<T>;
	fn div(self, rhs: Var<T>) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let a_kind = entry_kind(&tape.entries, self.idx);
			let req_grad = entry_rg(&tape.entries, self.idx) || entry_rg(&tape.entries, rhs.idx);
			match a_kind {
				ValueKind::Scalar => {
					let a = tape.values[self.idx.0 as usize].as_scalar();
					let b = tape.values[rhs.idx.0 as usize].as_scalar();
					Var::new(tape.push(
						OpCode::Div(self.idx, rhs.idx),
						NodeValue::Scalar(a / b),
						ValueKind::Scalar,
						req_grad,
					))
				}
				ValueKind::Vector(n) => {
					let mut r = get_vec_binary(tape, self.idx, rhs.idx, n);
					let bv = tape.values[rhs.idx.0 as usize].as_vec();
					r.component_div_assign(bv); // SIMD via faer::zip!
					Var::new(tape.push(
						OpCode::Div(self.idx, rhs.idx),
						NodeValue::Vector(r),
						ValueKind::Vector(n),
						req_grad,
					))
				}
				_ => panic!("Div: unsupported for matrices"),
			}
		})
	}
}

impl<T: TapeThreadLocal> std::ops::Neg for Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Output = Var<T>;
	fn neg(self) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let kind = entry_kind(&tape.entries, self.idx);
			let req_grad = entry_rg(&tape.entries, self.idx);
			match kind {
				ValueKind::Scalar => {
					let v = tape.values[self.idx.0 as usize].as_scalar();
					Var::new(tape.push(
						OpCode::Neg(self.idx),
						NodeValue::Scalar(T::zero() - v),
						ValueKind::Scalar,
						req_grad,
					))
				}
				ValueKind::Vector(n) => {
					let v = tape.values[self.idx.0 as usize].as_vec();
					let r = VectorOps::neg(v);
					Var::new(tape.push(
						OpCode::Neg(self.idx),
						NodeValue::Vector(r),
						ValueKind::Vector(n),
						req_grad,
					))
				}
				_ => panic!("Neg: unsupported for matrices"),
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Var op T  /  T op Var
// ═══════════════════════════════════════════════════════════════════════════

impl<T: TapeThreadLocal> std::ops::Mul<T> for Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Output = Var<T>;
	fn mul(self, rhs: T) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let kind = entry_kind(&tape.entries, self.idx);
			let req_grad = entry_rg(&tape.entries, self.idx);
			let s_idx = tape.push(
				OpCode::Input,
				NodeValue::Scalar(rhs),
				ValueKind::Scalar,
				false,
			);
			match kind {
				ValueKind::Scalar => {
					let v = tape.values[self.idx.0 as usize].as_scalar();
					Var::new(tape.push(
						OpCode::ScalarMul(s_idx, self.idx),
						NodeValue::Scalar(v * rhs),
						ValueKind::Scalar,
						req_grad,
					))
				}
				ValueKind::Vector(n) => {
					let v = tape.values[self.idx.0 as usize].as_vec();
					let mut r = tape.pool.get_vec(n);
					r.copy_from(v);
					r.scale_mut(rhs);
					Var::new(tape.push(
						OpCode::ScalarMul(s_idx, self.idx),
						NodeValue::Vector(r),
						kind,
						req_grad,
					))
				}
				_ => panic!("Mul<T>: unsupported for matrices"),
			}
		})
	}
}

impl std::ops::Mul<Var<f64>> for f64 {
	type Output = Var<f64>;
	#[inline]
	fn mul(self, rhs: Var<f64>) -> Var<f64> {
		rhs * self
	}
}

impl std::ops::Mul<Var<f32>> for f32 {
	type Output = Var<f32>;
	#[inline]
	fn mul(self, rhs: Var<f32>) -> Var<f32> {
		rhs * self
	}
}

impl<T: TapeThreadLocal> std::ops::Add<T> for Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Output = Var<T>;
	fn add(self, rhs: T) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let kind = entry_kind(&tape.entries, self.idx);
			let req_grad = entry_rg(&tape.entries, self.idx);
			let s_idx = tape.push(
				OpCode::Input,
				NodeValue::Scalar(rhs),
				ValueKind::Scalar,
				false,
			);
			match kind {
				ValueKind::Scalar => {
					let v = tape.values[self.idx.0 as usize].as_scalar();
					Var::new(tape.push(
						OpCode::ScalarAdd(s_idx, self.idx),
						NodeValue::Scalar(v + rhs),
						ValueKind::Scalar,
						req_grad,
					))
				}
				ValueKind::Vector(n) => {
					let v = tape.values[self.idx.0 as usize].as_vec();
					let mut r = tape.pool.get_vec(n);
					r.copy_from(v);
					let sl = r.as_mut_slice();
					for val in sl.iter_mut() {
						*val = *val + rhs;
					}
					Var::new(tape.push(
						OpCode::ScalarAdd(s_idx, self.idx),
						NodeValue::Vector(r),
						kind,
						req_grad,
					))
				}
				_ => panic!("Add<T>: unsupported for matrices"),
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Unary math — map_mut on backend vectors
// ═══════════════════════════════════════════════════════════════════════════

fn unary_op<T: TapeThreadLocal>(
	tape: &mut Tape<T>,
	self_idx: NodeIdx,
	op: OpCode,
	f: impl FnMut(T) -> T,
) -> Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	let kind = entry_kind(&tape.entries, self_idx);
	let req_grad = entry_rg(&tape.entries, self_idx);
	match kind {
		ValueKind::Scalar => {
			let v = tape.values[self_idx.0 as usize].as_scalar();
			let mut f = f;
			Var::new(tape.push(op, NodeValue::Scalar(f(v)), ValueKind::Scalar, req_grad))
		}
		ValueKind::Vector(n) => {
			let mut r = get_vec_unary(tape, self_idx, n);
			r.map_mut(f);
			Var::new(tape.push(op, NodeValue::Vector(r), kind, req_grad))
		}
		_ => panic!("unary op: unsupported for matrices"),
	}
}

impl<T: TapeThreadLocal> Var<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub fn exp(self) -> Var<T> {
		with_tape(|t| unary_op(t, self.idx, OpCode::Exp(self.idx), Float::exp))
	}

	pub fn log(self) -> Var<T> {
		with_tape(|t| unary_op(t, self.idx, OpCode::Log(self.idx), Float::ln))
	}

	pub fn sqrt(self) -> Var<T> {
		with_tape(|t| unary_op(t, self.idx, OpCode::Sqrt(self.idx), Float::sqrt))
	}

	pub fn sin(self) -> Var<T> {
		with_tape(|t| unary_op(t, self.idx, OpCode::Sin(self.idx), Float::sin))
	}

	pub fn cos(self) -> Var<T> {
		with_tape(|t| unary_op(t, self.idx, OpCode::Cos(self.idx), Float::cos))
	}

	pub fn abs(self) -> Var<T> {
		with_tape(|t| unary_op(t, self.idx, OpCode::Abs(self.idx), Float::abs))
	}

	pub fn powi(self, n: u64) -> Var<T> {
		with_tape(|t| {
			unary_op(t, self.idx, OpCode::Pow(self.idx, n), |x| {
				Float::powi(x, n as i32)
			})
		})
	}

	// ── Reductions ────────────────────────────────────────────────────

	pub fn sum(self) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let kind = entry_kind(&tape.entries, self.idx);
			let req_grad = entry_rg(&tape.entries, self.idx);
			let s = match kind {
				ValueKind::Scalar => tape.values[self.idx.0 as usize].as_scalar(),
				ValueKind::Vector(_) => tape.values[self.idx.0 as usize]
					.as_vec()
					.iter()
					.fold(T::zero(), |a, x| a + x),
				ValueKind::Matrix(r, c) => {
					let m = tape.values[self.idx.0 as usize].as_mat();
					let mut s = T::zero();
					for j in 0..c {
						for i in 0..r {
							s = s + MatrixView::get(m, i, j);
						}
					}
					s
				}
			};
			Var::new(tape.push(
				OpCode::Sum(self.idx),
				NodeValue::Scalar(s),
				ValueKind::Scalar,
				req_grad,
			))
		})
	}

	pub fn mean(self) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let kind = entry_kind(&tape.entries, self.idx);
			let req_grad = entry_rg(&tape.entries, self.idx);
			let n = kind.len();
			let s = match kind {
				ValueKind::Scalar => tape.values[self.idx.0 as usize].as_scalar(),
				ValueKind::Vector(_) => {
					let v = tape.values[self.idx.0 as usize].as_vec();
					v.iter().fold(T::zero(), |a, x| a + x) / <T as RealScalar>::from_usize(n)
				}
				_ => panic!("mean: unsupported for matrices"),
			};
			Var::new(tape.push(
				OpCode::Mean(self.idx),
				NodeValue::Scalar(s),
				ValueKind::Scalar,
				req_grad,
			))
		})
	}

	/// Dot product — SIMD via `VectorOps::dot`.
	pub fn dot(self, other: Var<T>) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let av = tape.values[self.idx.0 as usize].as_vec();
			let bv = tape.values[other.idx.0 as usize].as_vec();
			let d = VectorView::dot(av, bv); // SIMD
			let req_grad = entry_rg(&tape.entries, self.idx) || entry_rg(&tape.entries, other.idx);
			Var::new(tape.push(
				OpCode::Dot(self.idx, other.idx),
				NodeValue::Scalar(d),
				ValueKind::Scalar,
				req_grad,
			))
		})
	}

	/// L2 norm — SIMD via `VectorOps::norm`.
	pub fn norm(self) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let v = tape.values[self.idx.0 as usize].as_vec();
			let n = VectorView::norm(v); // SIMD
			let req_grad = entry_rg(&tape.entries, self.idx);
			Var::new(tape.push(
				OpCode::Norm(self.idx),
				NodeValue::Scalar(n),
				ValueKind::Scalar,
				req_grad,
			))
		})
	}

	/// Trace of a square matrix.
	pub fn trace(self) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let kind = entry_kind(&tape.entries, self.idx);
			let (r, c) = match kind {
				ValueKind::Matrix(r, c) => (r, c),
				_ => panic!("trace: not a matrix"),
			};
			assert_eq!(r, c, "trace: not square");
			let m = tape.values[self.idx.0 as usize].as_mat();
			let t = MatrixView::trace(m);
			let req_grad = entry_rg(&tape.entries, self.idx);
			Var::new(tape.push(
				OpCode::Trace(self.idx),
				NodeValue::Scalar(t),
				ValueKind::Scalar,
				req_grad,
			))
		})
	}

	/// Matrix multiply — via `MatrixOps::gemm`.
	pub fn matmul(self, other: Var<T>) -> Var<T> {
		with_tape(|tape: &mut Tape<T>| {
			let a_kind = entry_kind(&tape.entries, self.idx);
			let b_kind = entry_kind(&tape.entries, other.idx);
			let (m, k1) = match a_kind {
				ValueKind::Matrix(r, c) => (r, c),
				ValueKind::Vector(len) => (len, 1),
				_ => panic!("matmul: left must be matrix or vector"),
			};
			let (k2, n) = match b_kind {
				ValueKind::Matrix(r, c) => (r, c),
				ValueKind::Vector(len) => (len, 1),
				_ => panic!("matmul: right must be matrix or vector"),
			};
			assert_eq!(k1, k2, "matmul: inner dim mismatch {k1} vs {k2}");

			// Convert vectors to 1-column matrices for gemm
			let am = vec_to_mat(&tape.values[self.idx.0 as usize], m, k1);
			let bm = vec_to_mat(&tape.values[other.idx.0 as usize], k2, n);
			let mut r = tape.pool.get_mat(m, n);
			r.gemm(T::one(), am.as_view(), bm.as_view(), T::zero());
			let req_grad = entry_rg(&tape.entries, self.idx) || entry_rg(&tape.entries, other.idx);
			Var::new(tape.push(
				OpCode::MatMul(self.idx, other.idx, m as u32, k1 as u32, n as u32),
				NodeValue::Matrix(r),
				ValueKind::Matrix(m, n),
				req_grad,
			))
		})
	}
}
