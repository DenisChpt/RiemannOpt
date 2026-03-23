//! [`Var`] — a lightweight handle to a tape entry with operator overloading.

use crate::tape::{with_tape, NodeIdx, OpCode, TapeEntry};

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
		let idx = self.push(TapeEntry {
			op: OpCode::Input,
			value: data.to_vec(),
			shape,
			requires_grad: true,
		});
		Var { idx }
	}

	/// Record a non-differentiable constant.
	pub fn constant(&mut self, data: &[f64], shape: (usize, usize)) -> Var {
		debug_assert_eq!(data.len(), shape.0 * shape.1);
		let idx = self.push(TapeEntry {
			op: OpCode::Input,
			value: data.to_vec(),
			shape,
			requires_grad: false,
		});
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

// ── helper: create a binary node ──────────────────────────────────────────

fn binary(op: OpCode, a: Var, b: Var, value: Vec<f64>, shape: (usize, usize)) -> Var {
	with_tape(|tape| {
		let rg = tape.entries[a.idx.0 as usize].requires_grad
			|| tape.entries[b.idx.0 as usize].requires_grad;
		let idx = tape.push(TapeEntry {
			op,
			value,
			shape,
			requires_grad: rg,
		});
		Var { idx }
	})
}

fn unary(op: OpCode, a: Var, value: Vec<f64>, shape: (usize, usize)) -> Var {
	with_tape(|tape| {
		let rg = tape.entries[a.idx.0 as usize].requires_grad;
		let idx = tape.push(TapeEntry {
			op,
			value,
			shape,
			requires_grad: rg,
		});
		Var { idx }
	})
}

// ── Var + Var (element-wise) ──────────────────────────────────────────────

impl std::ops::Add for Var {
	type Output = Var;
	fn add(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let b = &tape.entries[rhs.idx.0 as usize];
			// scalar + tensor broadcast
			if a.shape == (1, 1) && b.shape != (1, 1) {
				let s = a.value[0];
				let val: Vec<f64> = b.value.iter().map(|x| x + s).collect();
				let shape = b.shape;
				let rg = a.requires_grad || b.requires_grad;
				let idx = tape.push(TapeEntry {
					op: OpCode::ScalarAdd(self.idx, rhs.idx),
					value: val,
					shape,
					requires_grad: rg,
				});
				return Var { idx };
			}
			if b.shape == (1, 1) && a.shape != (1, 1) {
				let s = b.value[0];
				let val: Vec<f64> = a.value.iter().map(|x| x + s).collect();
				let shape = a.shape;
				let rg = a.requires_grad || b.requires_grad;
				let idx = tape.push(TapeEntry {
					op: OpCode::ScalarAdd(rhs.idx, self.idx),
					value: val,
					shape,
					requires_grad: rg,
				});
				return Var { idx };
			}
			assert_eq!(
				a.shape, b.shape,
				"Add: shape mismatch {:?} vs {:?}",
				a.shape, b.shape
			);
			let val: Vec<f64> = a.value.iter().zip(&b.value).map(|(x, y)| x + y).collect();
			let shape = a.shape;
			let rg = a.requires_grad || b.requires_grad;
			let idx = tape.push(TapeEntry {
				op: OpCode::Add(self.idx, rhs.idx),
				value: val,
				shape,
				requires_grad: rg,
			});
			Var { idx }
		})
	}
}

impl std::ops::Sub for Var {
	type Output = Var;
	fn sub(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let b = &tape.entries[rhs.idx.0 as usize];
			assert_eq!(a.shape, b.shape, "Sub: shape mismatch");
			let val: Vec<f64> = a.value.iter().zip(&b.value).map(|(x, y)| x - y).collect();
			let shape = a.shape;
			let rg = a.requires_grad || b.requires_grad;
			let idx = tape.push(TapeEntry {
				op: OpCode::Sub(self.idx, rhs.idx),
				value: val,
				shape,
				requires_grad: rg,
			});
			Var { idx }
		})
	}
}

impl std::ops::Mul for Var {
	type Output = Var;
	fn mul(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let b = &tape.entries[rhs.idx.0 as usize];
			// scalar * tensor broadcast
			if a.shape == (1, 1) && b.shape != (1, 1) {
				let s = a.value[0];
				let val: Vec<f64> = b.value.iter().map(|x| x * s).collect();
				let shape = b.shape;
				let rg = a.requires_grad || b.requires_grad;
				let idx = tape.push(TapeEntry {
					op: OpCode::ScalarMul(self.idx, rhs.idx),
					value: val,
					shape,
					requires_grad: rg,
				});
				return Var { idx };
			}
			if b.shape == (1, 1) && a.shape != (1, 1) {
				let s = b.value[0];
				let val: Vec<f64> = a.value.iter().map(|x| x * s).collect();
				let shape = a.shape;
				let rg = a.requires_grad || b.requires_grad;
				let idx = tape.push(TapeEntry {
					op: OpCode::ScalarMul(rhs.idx, self.idx),
					value: val,
					shape,
					requires_grad: rg,
				});
				return Var { idx };
			}
			assert_eq!(a.shape, b.shape, "Mul: shape mismatch");
			let val: Vec<f64> = a.value.iter().zip(&b.value).map(|(x, y)| x * y).collect();
			let shape = a.shape;
			let rg = a.requires_grad || b.requires_grad;
			let idx = tape.push(TapeEntry {
				op: OpCode::Mul(self.idx, rhs.idx),
				value: val,
				shape,
				requires_grad: rg,
			});
			Var { idx }
		})
	}
}

impl std::ops::Div for Var {
	type Output = Var;
	fn div(self, rhs: Var) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let b = &tape.entries[rhs.idx.0 as usize];
			assert_eq!(a.shape, b.shape, "Div: shape mismatch");
			let val: Vec<f64> = a.value.iter().zip(&b.value).map(|(x, y)| x / y).collect();
			let shape = a.shape;
			let rg = a.requires_grad || b.requires_grad;
			let idx = tape.push(TapeEntry {
				op: OpCode::Div(self.idx, rhs.idx),
				value: val,
				shape,
				requires_grad: rg,
			});
			Var { idx }
		})
	}
}

impl std::ops::Neg for Var {
	type Output = Var;
	fn neg(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| -x).collect();
			unary(OpCode::Neg(self.idx), self, val, a.shape)
		})
	}
}

// ── Var op f64  /  f64 op Var ─────────────────────────────────────────────

impl std::ops::Mul<f64> for Var {
	type Output = Var;
	fn mul(self, rhs: f64) -> Var {
		with_tape(|tape| {
			let e = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = e.value.iter().map(|x| x * rhs).collect();
			let shape = e.shape;
			let rg = e.requires_grad;
			// Create constant scalar node
			let s_idx = tape.push(TapeEntry {
				op: OpCode::Input,
				value: vec![rhs],
				shape: (1, 1),
				requires_grad: false,
			});
			let idx = tape.push(TapeEntry {
				op: OpCode::ScalarMul(s_idx, self.idx),
				value: val,
				shape,
				requires_grad: rg,
			});
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
			let val: Vec<f64> = e.value.iter().map(|x| x + rhs).collect();
			let shape = e.shape;
			let rg = e.requires_grad;
			let s_idx = tape.push(TapeEntry {
				op: OpCode::Input,
				value: vec![rhs],
				shape: (1, 1),
				requires_grad: false,
			});
			let idx = tape.push(TapeEntry {
				op: OpCode::ScalarAdd(s_idx, self.idx),
				value: val,
				shape,
				requires_grad: rg,
			});
			Var { idx }
		})
	}
}

// ── math / linalg methods on Var ──────────────────────────────────────────

impl Var {
	/// Element-wise exponential.
	pub fn exp(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| x.exp()).collect();
			unary(OpCode::Exp(self.idx), self, val, a.shape)
		})
	}

	/// Element-wise natural logarithm.
	pub fn log(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| x.ln()).collect();
			unary(OpCode::Log(self.idx), self, val, a.shape)
		})
	}

	/// Element-wise square root.
	pub fn sqrt(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| x.sqrt()).collect();
			unary(OpCode::Sqrt(self.idx), self, val, a.shape)
		})
	}

	/// Element-wise sine.
	pub fn sin(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| x.sin()).collect();
			unary(OpCode::Sin(self.idx), self, val, a.shape)
		})
	}

	/// Element-wise cosine.
	pub fn cos(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| x.cos()).collect();
			unary(OpCode::Cos(self.idx), self, val, a.shape)
		})
	}

	/// Element-wise absolute value.
	pub fn abs(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| x.abs()).collect();
			unary(OpCode::Abs(self.idx), self, val, a.shape)
		})
	}

	/// Element-wise integer power.
	pub fn powi(self, n: u64) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let val: Vec<f64> = a.value.iter().map(|x| x.powi(n as i32)).collect();
			unary(OpCode::Pow(self.idx, n), self, val, a.shape)
		})
	}

	/// Sum all elements → scalar.
	pub fn sum(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let s: f64 = a.value.iter().sum();
			unary(OpCode::Sum(self.idx), self, vec![s], (1, 1))
		})
	}

	/// Mean of all elements → scalar.
	pub fn mean(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let n = a.value.len() as f64;
			let s: f64 = a.value.iter().sum::<f64>() / n;
			unary(OpCode::Mean(self.idx), self, vec![s], (1, 1))
		})
	}

	/// Dot product `a · b` → scalar.
	pub fn dot(self, other: Var) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let b = &tape.entries[other.idx.0 as usize];
			assert_eq!(a.value.len(), b.value.len(), "dot: length mismatch");
			let d: f64 = a.value.iter().zip(&b.value).map(|(x, y)| x * y).sum();
			binary(
				OpCode::Dot(self.idx, other.idx),
				self,
				other,
				vec![d],
				(1, 1),
			)
		})
	}

	/// L2 norm `‖a‖₂` → scalar.
	pub fn norm(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let n: f64 = a.value.iter().map(|x| x * x).sum::<f64>().sqrt();
			unary(OpCode::Norm(self.idx), self, vec![n], (1, 1))
		})
	}

	/// Trace of a square matrix → scalar.
	pub fn trace(self) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let (r, c) = a.shape;
			assert_eq!(r, c, "trace: not square");
			let mut t = 0.0;
			for i in 0..r {
				t += a.value[i + i * r]; // column-major
			}
			unary(OpCode::Trace(self.idx), self, vec![t], (1, 1))
		})
	}

	/// Matrix multiply `A(m,k) × B(k,n)`.
	pub fn matmul(self, other: Var) -> Var {
		with_tape(|tape| {
			let a = &tape.entries[self.idx.0 as usize];
			let b = &tape.entries[other.idx.0 as usize];
			let (m, k1) = a.shape;
			let (k2, n) = b.shape;
			assert_eq!(k1, k2, "matmul: inner dimension mismatch {k1} vs {k2}");
			let k = k1;
			// column-major matmul
			let mut val = vec![0.0; m * n];
			for j in 0..n {
				for p in 0..k {
					let b_pk = b.value[p + j * k];
					for i in 0..m {
						val[i + j * m] += a.value[i + p * m] * b_pk;
					}
				}
			}
			binary(
				OpCode::MatMul(self.idx, other.idx, m as u32, k as u32, n as u32),
				self,
				other,
				val,
				(m, n),
			)
		})
	}
}
