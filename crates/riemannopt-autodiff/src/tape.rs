//! Flat tape (Wengert list) of operations.
//!
//! The tape stores **no mathematical data** — only a dense array of [`Op`]
//! enums whose fields are `u32` indices into the [`BufferPool`](crate::pool::BufferPool).
//! This keeps the tape cache-friendly and allows the CPU's branch predictor
//! to prefetch the next instruction efficiently.

/// A single recorded operation.
///
/// Every variant stores only `u32` pool indices.  The `out` field always
/// refers to a *freshly allocated* slot (never aliased with an input),
/// which is enforced by `debug_assert_ne!` in the backward pass.
#[derive(Copy, Clone, Debug)]
pub enum Op {
	// ── Scalar ← Scalar × Scalar ─────────────────────────────────────
	AddS {
		a: u32,
		b: u32,
		out: u32,
	},
	SubS {
		a: u32,
		b: u32,
		out: u32,
	},
	MulS {
		a: u32,
		b: u32,
		out: u32,
	},
	DivS {
		a: u32,
		b: u32,
		out: u32,
	},
	PowS {
		base: u32,
		exp: u32,
		out: u32,
	},

	// ── Scalar ← Scalar (unary) ──────────────────────────────────────
	NegS {
		a: u32,
		out: u32,
	},
	ExpS {
		a: u32,
		out: u32,
	},
	LogS {
		a: u32,
		out: u32,
	},
	SqrtS {
		a: u32,
		out: u32,
	},
	SinS {
		a: u32,
		out: u32,
	},
	CosS {
		a: u32,
		out: u32,
	},
	AbsS {
		a: u32,
		out: u32,
	},

	// ── Vector ← Vector × Vector ─────────────────────────────────────
	AddV {
		a: u32,
		b: u32,
		out: u32,
	},
	SubV {
		a: u32,
		b: u32,
		out: u32,
	},
	ComponentMulV {
		a: u32,
		b: u32,
		out: u32,
	},

	// ── Vector ← Vector (unary) ──────────────────────────────────────
	NegV {
		a: u32,
		out: u32,
	},

	// ── Vector ← Scalar × Vector ─────────────────────────────────────
	ScaleV {
		s: u32,
		v: u32,
		out: u32,
	},

	// ── Scalar ← Vector (reductions) ─────────────────────────────────
	DotV {
		a: u32,
		b: u32,
		out: u32,
	},
	NormV {
		v: u32,
		out: u32,
	},
	NormSqV {
		v: u32,
		out: u32,
	},
	SumV {
		v: u32,
		out: u32,
	},

	// ── Matrix ← Matrix × Matrix ─────────────────────────────────────
	AddM {
		a: u32,
		b: u32,
		out: u32,
	},
	SubM {
		a: u32,
		b: u32,
		out: u32,
	},
	MatMul {
		a: u32,
		b: u32,
		out: u32,
	},

	// ── Matrix ← Matrix (unary) ──────────────────────────────────────
	NegM {
		a: u32,
		out: u32,
	},
	TransposeM {
		a: u32,
		out: u32,
	},

	// ── Matrix ← Scalar × Matrix ─────────────────────────────────────
	ScaleM {
		s: u32,
		m: u32,
		out: u32,
	},

	// ── Vector ← Matrix × Vector ─────────────────────────────────────
	MatVec {
		m: u32,
		v: u32,
		out: u32,
	},

	// ── Scalar ← Matrix (reductions) ─────────────────────────────────
	TraceM {
		m: u32,
		out: u32,
	},
	FrobDotM {
		a: u32,
		b: u32,
		out: u32,
	},

	// ── Fused operators ──────────────────────────────────────────────
	/// out = A·x + b  (linear layer / affine transform)
	LinearLayer {
		a: u32,
		x: u32,
		b: u32,
		out: u32,
	},
	/// out = xᵀ·A·x  (quadratic form); `ax` holds A·x for the backward pass
	QuadForm {
		x: u32,
		a: u32,
		ax: u32,
		out: u32,
	},
}

// Compile-time size guard: keep Op ≤ 24 bytes for cache friendliness.
const _: () = assert!(
	std::mem::size_of::<Op>() <= 24,
	"Op enum exceeds 24 bytes — consider shrinking variants"
);

/// A flat list of [`Op`]s recorded during the forward pass.
///
/// After the first iteration the internal `Vec` has enough capacity
/// and `clear()` reuses it — zero allocation in steady state.
pub struct Tape {
	ops: Vec<Op>,
}

impl Tape {
	/// Creates an empty tape.
	#[inline]
	pub fn new() -> Self {
		Self { ops: Vec::new() }
	}

	/// Creates a tape with pre-allocated capacity.
	#[inline]
	pub fn with_capacity(cap: usize) -> Self {
		Self {
			ops: Vec::with_capacity(cap),
		}
	}

	/// Appends an operation.
	#[inline]
	pub fn push(&mut self, op: Op) {
		self.ops.push(op);
	}

	/// Returns the recorded operations as a slice.
	#[inline]
	pub fn ops(&self) -> &[Op] {
		&self.ops
	}

	/// Number of recorded operations.
	#[inline]
	pub fn len(&self) -> usize {
		self.ops.len()
	}

	/// Whether the tape is empty.
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.ops.is_empty()
	}

	/// Clears all operations, retaining allocated capacity.
	#[inline]
	pub fn clear(&mut self) {
		self.ops.clear();
	}
}

impl Default for Tape {
	fn default() -> Self {
		Self::new()
	}
}
