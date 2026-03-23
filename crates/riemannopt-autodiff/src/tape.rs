//! Tape-based arena for reverse-mode automatic differentiation.
//!
//! The [`Tape`] stores computation entries in a flat `Vec` (Wengert list).
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
#[derive(Debug)]
pub struct TapeEntry {
    pub op: OpCode,
    /// Forward-pass value stored as a flat column-major buffer.
    pub value: Vec<f64>,
    /// `(rows, cols)`.  Scalars are `(1, 1)`, column vectors `(n, 1)`.
    pub shape: (usize, usize),
    /// Propagate gradients through this node?
    pub requires_grad: bool,
}

// ---------------------------------------------------------------------------
// Tape
// ---------------------------------------------------------------------------

/// The Wengert list (tape).
///
/// All entries are in topological order by construction: each entry only
/// references earlier entries.
pub struct Tape {
    pub(crate) entries: Vec<TapeEntry>,
}

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(256),
        }
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

    /// Append an entry, returning its index.
    #[inline]
    pub(crate) fn push(&mut self, entry: TapeEntry) -> NodeIdx {
        let idx = NodeIdx(self.entries.len() as u32);
        self.entries.push(entry);
        idx
    }

    /// Read the value stored at `idx`.
    #[inline]
    pub fn value(&self, idx: NodeIdx) -> &[f64] {
        &self.entries[idx.0 as usize].value
    }

    /// Read the scalar value stored at `idx` (panics if not `(1,1)`).
    #[inline]
    pub fn scalar(&self, idx: NodeIdx) -> f64 {
        debug_assert_eq!(self.entries[idx.0 as usize].shape, (1, 1));
        self.entries[idx.0 as usize].value[0]
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
            let prev = cell.borrow().clone();
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
