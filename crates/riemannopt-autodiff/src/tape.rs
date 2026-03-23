//! Tape-based arena for reverse-mode automatic differentiation.
//!
//! The [`Tape`] stores computation entries in a flat `Vec` (Wengert list)
//! with all forward-pass values kept in a single contiguous arena.  This
//! avoids per-node heap allocations and improves cache locality.
//!
//! Entries are appended during the forward pass in topological order, so the
//! backward pass is a simple reverse iteration — no graph traversal required.

use std::cell::RefCell;

// ---------------------------------------------------------------------------
// NodeIdx
// ---------------------------------------------------------------------------

/// Lightweight, `Copy` handle into the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeIdx(pub(crate) u32);

// ---------------------------------------------------------------------------
// OpCode — every operation the tape can record
// ---------------------------------------------------------------------------

/// Describes how a tape entry was produced.
///
/// Leaf nodes use [`OpCode::Input`].  All other variants store the indices of
/// their operands so the backward pass can look them up in O(1).
#[derive(Debug, Clone, Copy)]
pub enum OpCode {
	/// Leaf: user-provided input or constant.
	Input,

	// -- binary element-wise --
	Add(NodeIdx, NodeIdx),
	Sub(NodeIdx, NodeIdx),
	Mul(NodeIdx, NodeIdx),
	Div(NodeIdx, NodeIdx),

	// -- unary element-wise --
	Neg(NodeIdx),
	Exp(NodeIdx),
	Log(NodeIdx),
	Sqrt(NodeIdx),
	Sin(NodeIdx),
	Cos(NodeIdx),
	Abs(NodeIdx),
	/// `x.pow(n)` where `n` is a compile-time constant.
	Pow(NodeIdx, u64),

	// -- reductions (always produce a scalar (1,1)) --
	/// Sum of all elements.
	Sum(NodeIdx),
	/// Mean of all elements.
	Mean(NodeIdx),
	/// Euclidean dot product  `a · b`.
	Dot(NodeIdx, NodeIdx),
	/// L2 norm  `‖a‖₂`.
	Norm(NodeIdx),

	// -- linear algebra --
	/// `A(m,k) × B(k,n)` — shapes stored to avoid parent lookup in backward.
	MatMul(NodeIdx, NodeIdx, u32, u32, u32),
	/// Trace of a square matrix → scalar.
	Trace(NodeIdx),

	// -- scalar–tensor broadcast --
	/// `scalar * tensor`  (first operand is `(1,1)`).
	ScalarMul(NodeIdx, NodeIdx),
	/// `scalar + tensor`  (first operand is `(1,1)`).
	ScalarAdd(NodeIdx, NodeIdx),
}

// ---------------------------------------------------------------------------
// TapeEntry
// ---------------------------------------------------------------------------

/// A single recorded computation.
///
/// Forward-pass values are stored in the [`Tape`]'s contiguous arena;
/// this entry only stores an offset and length into that arena.
#[derive(Debug)]
pub struct TapeEntry {
	pub op: OpCode,
	/// Start offset into [`Tape::arena`].
	pub value_offset: u32,
	/// Number of `f64` elements for this node's value.
	pub value_len: u32,
	/// `(rows, cols)`.  Scalars are `(1, 1)`, column vectors `(n, 1)`.
	pub shape: (usize, usize),
	/// Propagate gradients through this node?
	pub requires_grad: bool,
}

// ---------------------------------------------------------------------------
// Tape
// ---------------------------------------------------------------------------

/// The Wengert list (tape) with arena-based value storage.
///
/// All entries are in topological order by construction: each entry only
/// references earlier entries.  All forward-pass values live in a single
/// contiguous `Vec<f64>` arena — no per-node heap allocation.
pub struct Tape {
	pub(crate) entries: Vec<TapeEntry>,
	/// Contiguous storage for all forward-pass values.
	pub(crate) arena: Vec<f64>,
}

impl Tape {
	/// Create a new empty tape.
	pub fn new() -> Self {
		Self {
			entries: Vec::with_capacity(256),
			arena: Vec::with_capacity(4096),
		}
	}

	/// Reset the tape for reuse, keeping allocated memory.
	pub fn clear(&mut self) {
		self.entries.clear();
		self.arena.clear();
	}

	/// Number of entries.
	#[inline]
	pub fn len(&self) -> usize {
		self.entries.len()
	}

	/// Is the tape empty?
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	/// Total number of `f64` values stored in the arena.
	#[inline]
	pub fn arena_len(&self) -> usize {
		self.arena.len()
	}

	/// Append an entry whose value is stored in the arena.
	#[inline]
	pub(crate) fn push(
		&mut self,
		op: OpCode,
		value: &[f64],
		shape: (usize, usize),
		requires_grad: bool,
	) -> NodeIdx {
		let offset = self.arena.len();
		self.arena.extend_from_slice(value);
		let idx = NodeIdx(self.entries.len() as u32);
		self.entries.push(TapeEntry {
			op,
			value_offset: offset as u32,
			value_len: value.len() as u32,
			shape,
			requires_grad,
		});
		idx
	}

	/// Read the value stored at `idx`.
	#[inline]
	pub fn value(&self, idx: NodeIdx) -> &[f64] {
		let e = &self.entries[idx.0 as usize];
		&self.arena[e.value_offset as usize..(e.value_offset + e.value_len) as usize]
	}

	/// Read the forward-pass value for a given entry index (not `NodeIdx`).
	#[inline]
	pub(crate) fn entry_value(&self, entry_idx: usize) -> &[f64] {
		let e = &self.entries[entry_idx];
		&self.arena[e.value_offset as usize..(e.value_offset + e.value_len) as usize]
	}

	/// Read the scalar value stored at `idx` (panics if not `(1,1)`).
	#[inline]
	pub fn scalar(&self, idx: NodeIdx) -> f64 {
		let e = &self.entries[idx.0 as usize];
		debug_assert_eq!(e.shape, (1, 1));
		self.arena[e.value_offset as usize]
	}

	/// Shape of the value at `idx`.
	#[inline]
	pub fn shape(&self, idx: NodeIdx) -> (usize, usize) {
		self.entries[idx.0 as usize].shape
	}
}

impl Default for Tape {
	fn default() -> Self {
		Self::new()
	}
}

// ---------------------------------------------------------------------------
// Thread-local active tape (for operator overloading)
// ---------------------------------------------------------------------------

thread_local! {
	static ACTIVE_TAPE: RefCell<Option<*mut Tape>> = const { RefCell::new(None) };
}

/// RAII guard that sets the thread-local active tape for the duration of its
/// lifetime.  All [`Var`](crate::Var) operations will record on this tape.
///
/// # Safety
///
/// The guard stores a raw pointer to the tape.  Soundness relies on:
/// - The `Tape` outlives the guard (guaranteed by borrow rules).
/// - Only one guard is active per thread at a time (enforced by `set`).
pub struct TapeGuard {
	prev: Option<*mut Tape>,
}

impl TapeGuard {
	/// Activate `tape` as the current recording target.
	///
	/// # Panics
	///
	/// Panics if another tape is already active on this thread.
	pub fn new(tape: &mut Tape) -> Self {
		let prev = ACTIVE_TAPE.with(|cell| {
			let prev = *cell.borrow();
			*cell.borrow_mut() = Some(tape as *mut Tape);
			prev
		});
		Self { prev }
	}
}

impl Drop for TapeGuard {
	fn drop(&mut self) {
		ACTIVE_TAPE.with(|cell| *cell.borrow_mut() = self.prev);
	}
}

/// Execute `f` with mutable access to the active tape.
///
/// # Panics
///
/// Panics if no tape is active.
#[inline]
pub fn with_tape<F, R>(f: F) -> R
where
	F: FnOnce(&mut Tape) -> R,
{
	ACTIVE_TAPE.with(|cell| {
		let ptr = cell
			.borrow()
			.expect("No active tape — wrap operations in a TapeGuard scope");
		// SAFETY: The TapeGuard guarantees the pointer is valid for the
		// duration of any Var operation.
		let tape = unsafe { &mut *ptr };
		f(tape)
	})
}
