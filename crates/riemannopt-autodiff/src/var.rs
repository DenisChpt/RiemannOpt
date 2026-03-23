//! [`Var`] — a lightweight handle to a tape entry with operator overloading.

use crate::tape::{with_tape, NodeIdx, OpCode, Tape};

/// A differentiable variable.
///
/// `Var` is `Copy` — it is just an index into the active [`Tape`](crate::Tape).
/// Arithmetic operators (`+`, `-`, `*`, `/`, negation) are overloaded to
/// record operations on the tape automatically.
#[derive(Debug, Clone, Copy)]
pub struct Var {
	pub(crate) idx: NodeIdx,
}

// ── construction helpers (called via Tape, but need public Var) ────────────

impl Var {
	/// The underlying index.
	#[inline]
	pub fn idx(self) -> NodeIdx {
		self.idx
	}
}

/// Extension methods on [`Tape`](crate::Tape) to create variables.
impl crate::Tape {
	/// Record a differentiable input variable.
	pub fn var(&mut self, data: &[f64], shape: (usize, usize)) -> Var {
		debug_assert_eq!(data.len(), shape.0 * shape.1);
		let idx = self.push(OpCode::Input, data, shape, true);
		Var { idx }
	}

	/// Record a non-differentiable constant.
	pub fn constant(&mut self, data: &[f64], shape: (usize, usize)) -> Var {
		debug_assert_eq!(data.len(), shape.0 * shape.1);
		let idx = self.push(OpCode::Input, data, shape, false);
		Var { idx }
	}

	/// Scalar variable (1×1).
	pub fn scalar_var(&mut self, v: f64) -> Var {
		self.var(&[v], (1, 1))
	}

	/// Scalar constant (1×1, no grad).
	pub fn scalar_const(&mut self, v: f64) -> Var {
		self.constant(&[v], (1, 1))
	}
}

// ── helpers: push unary / binary nodes on the tape ────────────────────────

fn push_unary(
	tape: &mut Tape,
	op: OpCode,
	parent: NodeIdx,
	value: &[f64],
	shape: (usize, usize),
) -> Var {
	let rg = tape.entries[parent.0 as usize].requires_grad;
	let idx = tape.push(op, value, shape, rg);
	Var { idx }
}

fn push_binary(
	tape: &mut Tape,
	op: OpCode,
	a: NodeIdx,
	b: NodeIdx,
	value: &[f64],
	shape: (usize, usize),
) -> Var {
	let rg = tape.entries[a.0 as usize].requires_grad || tape.entries[b.0 as usize].requires_grad;
	let idx = tape.push(op, value, shape, rg);
	Var { idx }
}

// ── Var + Var (element-wise) ──────────────────────────────────────────────

impl std::ops::Add for Var {
	type Output = Var;
	fn add(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a_e = &tape.entries[self.idx.0 as usize];
			let b_e = &tape.entries[rhs.idx.0 as usize];
			let a_off = a_e.value_offset as usize;
			let a_len = a_e.value_len as usize;
			let b_off = b_e.value_offset as usize;
			let b_len = b_e.value_len as usize;
			let a_shape = a_e.shape;
			let b_shape = b_e.shape;
			let a_rg = a_e.requires_grad;
			let b_rg = b_e.requires_grad;

			// scalar + tensor broadcast
			if a_shape == (1, 1) && b_shape != (1, 1) {
				let s = tape.arena[a_off];
				let val: Vec<f64> = tape.arena[b_off..b_off + b_len]
					.iter()
					.map(|x| x + s)
					.collect();
				let idx = tape.push(
					OpCode::ScalarAdd(self.idx, rhs.idx),
					&val,
					b_shape,
					a_rg || b_rg,
				);
				return Var { idx };
			}
			if b_shape == (1, 1) && a_shape != (1, 1) {
				let s = tape.arena[b_off];
				let val: Vec<f64> = tape.arena[a_off..a_off + a_len]
					.iter()
					.map(|x| x + s)
					.collect();
				let idx = tape.push(
					OpCode::ScalarAdd(rhs.idx, self.idx),
					&val,
					a_shape,
					a_rg || b_rg,
				);
				return Var { idx };
			}
			assert_eq!(
				a_shape, b_shape,
				"Add: shape mismatch {:?} vs {:?}",
				a_shape, b_shape
			);
			let val: Vec<f64> = tape.arena[a_off..a_off + a_len]
				.iter()
				.zip(&tape.arena[b_off..b_off + b_len])
				.map(|(x, y)| x + y)
				.collect();
			let idx = tape.push(OpCode::Add(self.idx, rhs.idx), &val, a_shape, a_rg || b_rg);
			Var { idx }
		})
	}
}

impl std::ops::Sub for Var {
	type Output = Var;
	fn sub(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a_e = &tape.entries[self.idx.0 as usize];
			let b_e = &tape.entries[rhs.idx.0 as usize];
			let a_off = a_e.value_offset as usize;
			let a_len = a_e.value_len as usize;
			let b_off = b_e.value_offset as usize;
			let a_shape = a_e.shape;
			let b_shape = b_e.shape;
			let a_rg = a_e.requires_grad;
			let b_rg = b_e.requires_grad;

			assert_eq!(a_shape, b_shape, "Sub: shape mismatch");
			let val: Vec<f64> = tape.arena[a_off..a_off + a_len]
				.iter()
				.zip(&tape.arena[b_off..b_off + a_len])
				.map(|(x, y)| x - y)
				.collect();
			let idx = tape.push(OpCode::Sub(self.idx, rhs.idx), &val, a_shape, a_rg || b_rg);
			Var { idx }
		})
	}
}

impl std::ops::Mul for Var {
	type Output = Var;
	fn mul(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a_e = &tape.entries[self.idx.0 as usize];
			let b_e = &tape.entries[rhs.idx.0 as usize];
			let a_off = a_e.value_offset as usize;
			let a_len = a_e.value_len as usize;
			let b_off = b_e.value_offset as usize;
			let b_len = b_e.value_len as usize;
			let a_shape = a_e.shape;
			let b_shape = b_e.shape;
			let a_rg = a_e.requires_grad;
			let b_rg = b_e.requires_grad;

			// scalar * tensor broadcast
			if a_shape == (1, 1) && b_shape != (1, 1) {
				let s = tape.arena[a_off];
				let val: Vec<f64> = tape.arena[b_off..b_off + b_len]
					.iter()
					.map(|x| x * s)
					.collect();
				let idx = tape.push(
					OpCode::ScalarMul(self.idx, rhs.idx),
					&val,
					b_shape,
					a_rg || b_rg,
				);
				return Var { idx };
			}
			if b_shape == (1, 1) && a_shape != (1, 1) {
				let s = tape.arena[b_off];
				let val: Vec<f64> = tape.arena[a_off..a_off + a_len]
					.iter()
					.map(|x| x * s)
					.collect();
				let idx = tape.push(
					OpCode::ScalarMul(rhs.idx, self.idx),
					&val,
					a_shape,
					a_rg || b_rg,
				);
				return Var { idx };
			}
			assert_eq!(a_shape, b_shape, "Mul: shape mismatch");
			let val: Vec<f64> = tape.arena[a_off..a_off + a_len]
				.iter()
				.zip(&tape.arena[b_off..b_off + b_len])
				.map(|(x, y)| x * y)
				.collect();
			let idx = tape.push(OpCode::Mul(self.idx, rhs.idx), &val, a_shape, a_rg || b_rg);
			Var { idx }
		})
	}
}

impl std::ops::Div for Var {
	type Output = Var;
	fn div(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a_e = &tape.entries[self.idx.0 as usize];
			let b_e = &tape.entries[rhs.idx.0 as usize];
			let a_off = a_e.value_offset as usize;
			let a_len = a_e.value_len as usize;
			let b_off = b_e.value_offset as usize;
			let a_shape = a_e.shape;
			let b_shape = b_e.shape;
			let a_rg = a_e.requires_grad;
			let b_rg = b_e.requires_grad;

			assert_eq!(a_shape, b_shape, "Div: shape mismatch");
			let val: Vec<f64> = tape.arena[a_off..a_off + a_len]
				.iter()
				.zip(&tape.arena[b_off..b_off + a_len])
				.map(|(x, y)| x / y)
				.collect();
			let idx = tape.push(OpCode::Div(self.idx, rhs.idx), &val, a_shape, a_rg || b_rg);
			Var { idx }
		})
	}
}

impl std::ops::Neg for Var {
	type Output = Var;
	fn neg(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;

			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| -x).collect();
			push_unary(tape, OpCode::Neg(self.idx), self.idx, &val, shape)
		})
	}
}

// ── Var op f64  /  f64 op Var ─────────────────────────────────────────────

impl std::ops::Mul<f64> for Var {
	type Output = Var;
	fn mul(self, rhs: f64) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let rg = e.requires_grad;

			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| x * rhs).collect();
			// Create constant scalar node
			let s_idx = tape.push(OpCode::Input, &[rhs], (1, 1), false);
			let idx = tape.push(OpCode::ScalarMul(s_idx, self.idx), &val, shape, rg);
			Var { idx }
		})
	}
}

impl std::ops::Mul<Var> for f64 {
	type Output = Var;
	fn mul(self, rhs: Var) -> Var {
		rhs * self
	}
}

impl std::ops::Add<f64> for Var {
	type Output = Var;
	fn add(self, rhs: f64) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let rg = e.requires_grad;

			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| x + rhs).collect();
			let s_idx = tape.push(OpCode::Input, &[rhs], (1, 1), false);
			let idx = tape.push(OpCode::ScalarAdd(s_idx, self.idx), &val, shape, rg);
			Var { idx }
		})
	}
}

// ── math / linalg methods on Var ──────────────────────────────────────────

impl Var {
	/// Element-wise exponential.
	pub fn exp(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| x.exp()).collect();
			push_unary(tape, OpCode::Exp(self.idx), self.idx, &val, shape)
		})
	}

	/// Element-wise natural logarithm.
	pub fn log(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| x.ln()).collect();
			push_unary(tape, OpCode::Log(self.idx), self.idx, &val, shape)
		})
	}

	/// Element-wise square root.
	pub fn sqrt(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let val: Vec<f64> = tape.arena[off..off + len]
				.iter()
				.map(|x| x.sqrt())
				.collect();
			push_unary(tape, OpCode::Sqrt(self.idx), self.idx, &val, shape)
		})
	}

	/// Element-wise sine.
	pub fn sin(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| x.sin()).collect();
			push_unary(tape, OpCode::Sin(self.idx), self.idx, &val, shape)
		})
	}

	/// Element-wise cosine.
	pub fn cos(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| x.cos()).collect();
			push_unary(tape, OpCode::Cos(self.idx), self.idx, &val, shape)
		})
	}

	/// Element-wise absolute value.
	pub fn abs(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let val: Vec<f64> = tape.arena[off..off + len].iter().map(|x| x.abs()).collect();
			push_unary(tape, OpCode::Abs(self.idx), self.idx, &val, shape)
		})
	}

	/// Element-wise integer power.
	pub fn powi(self, n: u64) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let shape = e.shape;
			let val: Vec<f64> = tape.arena[off..off + len]
				.iter()
				.map(|x| x.powi(n as i32))
				.collect();
			push_unary(tape, OpCode::Pow(self.idx, n), self.idx, &val, shape)
		})
	}

	/// Sum all elements → scalar.
	pub fn sum(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let s: f64 = tape.arena[off..off + len].iter().sum();
			push_unary(tape, OpCode::Sum(self.idx), self.idx, &[s], (1, 1))
		})
	}

	/// Mean of all elements → scalar.
	pub fn mean(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let n = len as f64;
			let s: f64 = tape.arena[off..off + len].iter().sum::<f64>() / n;
			push_unary(tape, OpCode::Mean(self.idx), self.idx, &[s], (1, 1))
		})
	}

	/// Dot product `a · b` → scalar.
	pub fn dot(self, other: Var) -> Var {
		with_tape(|tape| {
			let a_e = &tape.entries[self.idx.0 as usize];
			let b_e = &tape.entries[other.idx.0 as usize];
			let a_off = a_e.value_offset as usize;
			let a_len = a_e.value_len as usize;
			let b_off = b_e.value_offset as usize;
			let b_len = b_e.value_len as usize;
			assert_eq!(a_len, b_len, "dot: length mismatch");

			let d: f64 = tape.arena[a_off..a_off + a_len]
				.iter()
				.zip(&tape.arena[b_off..b_off + b_len])
				.map(|(x, y)| x * y)
				.sum();
			push_binary(
				tape,
				OpCode::Dot(self.idx, other.idx),
				self.idx,
				other.idx,
				&[d],
				(1, 1),
			)
		})
	}

	/// L2 norm `‖a‖₂` → scalar.
	pub fn norm(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let len = e.value_len as usize;
			let n: f64 = tape.arena[off..off + len]
				.iter()
				.map(|x| x * x)
				.sum::<f64>()
				.sqrt();
			push_unary(tape, OpCode::Norm(self.idx), self.idx, &[n], (1, 1))
		})
	}

	/// Trace of a square matrix → scalar.
	pub fn trace(self) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let off = e.value_offset as usize;
			let (r, c) = e.shape;
			assert_eq!(r, c, "trace: not square");
			let mut t = 0.0;
			for i in 0..r {
				t += tape.arena[off + i + i * r]; // column-major
			}
			push_unary(tape, OpCode::Trace(self.idx), self.idx, &[t], (1, 1))
		})
	}

	/// Matrix multiply `A(m,k) × B(k,n)`.
	pub fn matmul(self, other: Var) -> Var {
		with_tape(|tape| {
			let a_e = &tape.entries[self.idx.0 as usize];
			let b_e = &tape.entries[other.idx.0 as usize];
			let a_off = a_e.value_offset as usize;
			let a_len = a_e.value_len as usize;
			let b_off = b_e.value_offset as usize;
			let b_len = b_e.value_len as usize;
			let (m, k1) = a_e.shape;
			let (k2, n) = b_e.shape;
			assert_eq!(k1, k2, "matmul: inner dimension mismatch {k1} vs {k2}");
			let k = k1;

			// Copy operand data to avoid aliasing with arena during push
			let a_data: Vec<f64> = tape.arena[a_off..a_off + a_len].to_vec();
			let b_data: Vec<f64> = tape.arena[b_off..b_off + b_len].to_vec();

			// column-major matmul
			let mut val = vec![0.0; m * n];
			for j in 0..n {
				for p in 0..k {
					let b_pk = b_data[p + j * k];
					for i in 0..m {
						val[i + j * m] += a_data[i + p * m] * b_pk;
					}
				}
			}
			push_binary(
				tape,
				OpCode::MatMul(self.idx, other.idx, m as u32, k as u32, n as u32),
				self.idx,
				other.idx,
				&val,
				(m, n),
			)
		})
	}
}
