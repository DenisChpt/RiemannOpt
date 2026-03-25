//! Tape-based computation graph for reverse-mode automatic differentiation.
//!
//! The [`Tape`] records operations in topological order (Wengert list).
//! Values are stored as backend-native types ([`NodeValue`]) — not in a flat
//! scalar arena — enabling SIMD/BLAS operations without intermediate copies.
//!
//! # Buffer lifecycle
//!
//! On first evaluation the tape allocates values from a [`BufferPool`].
//! On subsequent evaluations (`clear_for_reuse`), non-saved buffers are
//! recycled back to the pool.  In steady state, zero heap allocation occurs.

use std::cell::RefCell;

use riemannopt_core::linalg::{self, LinAlgBackend, RealScalar};

use crate::buffer_pool::BufferPool;
use crate::value::{NodeValue, SavePolicy, ValueKind};

// ═══════════════════════════════════════════════════════════════════════════
//  GraphPlan — cached graph structure that survives clear_for_reuse
// ═══════════════════════════════════════════════════════════════════════════

/// Cached graph structure from a previous forward pass.
///
/// Survives `clear_for_reuse()`. On the next forward, each `push()` validates
/// that the current op+kind match the plan. If so, `last_forward_use` and
/// `must_survive` are available for buffer donation during tracing.
/// If the graph diverges, the plan is invalidated.
#[derive(Default)]
pub(crate) struct GraphPlan {
	/// Full OpCode for each node (includes operand indices for exact match).
	ops: Vec<OpCode>,
	/// ValueKind for each node (for validation).
	kinds: Vec<ValueKind>,
	/// Liveness: last node that reads this node during forward.
	last_forward_use: Vec<u32>,
	/// Whether this node's value must survive for backward.
	must_survive: Vec<bool>,
}

impl GraphPlan {
	/// Number of nodes in the cached plan.
	fn len(&self) -> usize {
		self.ops.len()
	}

	/// Whether the plan has been populated.
	fn is_populated(&self) -> bool {
		!self.ops.is_empty()
	}

	/// Check if node at `cursor` matches the expected op and kind.
	fn matches(&self, cursor: usize, op: &OpCode, kind: ValueKind) -> bool {
		cursor < self.ops.len() && self.ops[cursor] == *op && self.kinds[cursor] == kind
	}

	/// Snapshot the current tape state into the plan.
	fn capture(&mut self, entries: &[TapeEntry], must_survive: &[bool]) {
		let n = entries.len();
		self.ops.clear();
		self.ops.reserve(n);
		self.kinds.clear();
		self.kinds.reserve(n);
		self.last_forward_use.clear();
		self.last_forward_use.reserve(n);
		self.must_survive.clear();
		self.must_survive.reserve(n);
		for (i, entry) in entries.iter().enumerate() {
			self.ops.push(entry.op);
			self.kinds.push(entry.kind);
			self.last_forward_use.push(entry.last_forward_use);
			self.must_survive
				.push(must_survive.get(i).copied().unwrap_or(false));
		}
	}

	/// Invalidate the plan (graph structure changed).
	fn invalidate(&mut self) {
		self.ops.clear();
		self.kinds.clear();
		self.last_forward_use.clear();
		self.must_survive.clear();
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  NodeIdx
// ═══════════════════════════════════════════════════════════════════════════

/// Lightweight, `Copy` handle into the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeIdx(pub(crate) u32);

// ═══════════════════════════════════════════════════════════════════════════
//  OpCode
// ═══════════════════════════════════════════════════════════════════════════

/// Describes how a tape entry was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
	Pow(NodeIdx, u64),

	// -- reductions (produce a scalar) --
	Sum(NodeIdx),
	Mean(NodeIdx),
	Dot(NodeIdx, NodeIdx),
	Norm(NodeIdx),

	// -- linear algebra --
	MatMul(NodeIdx, NodeIdx, u32, u32, u32),
	Trace(NodeIdx),

	// -- scalar-tensor broadcast --
	ScalarMul(NodeIdx, NodeIdx),
	ScalarAdd(NodeIdx, NodeIdx),
}

impl OpCode {
	/// Which forward-pass values this operation needs saved for backward.
	pub fn save_policy(&self) -> SavePolicy {
		match self {
			// Nothing needed
			Self::Input
			| Self::Add(..)
			| Self::Sub(..)
			| Self::Neg(_)
			| Self::Sum(_)
			| Self::Mean(_)
			| Self::ScalarAdd(..)
			| Self::Trace(_) => SavePolicy::NotNeeded,

			// Save output
			Self::Exp(_) | Self::Sqrt(_) => SavePolicy::SaveOutput,

			// Save first input
			Self::Log(_) | Self::Sin(_) | Self::Cos(_) | Self::Abs(_) | Self::Pow(..) => {
				SavePolicy::SaveInput(0)
			}

			// Save both inputs
			Self::Mul(..)
			| Self::Div(..)
			| Self::Dot(..)
			| Self::MatMul(..)
			| Self::ScalarMul(..) => SavePolicy::SaveBothInputs,

			// Save output + first input
			Self::Norm(_) => SavePolicy::SaveOutputAndInput(0),
		}
	}

	/// Iterator over operand indices referenced by this op.
	pub fn operands(&self) -> impl Iterator<Item = NodeIdx> {
		let mut ops = [None, None];
		match *self {
			Self::Input => {}
			Self::Neg(a)
			| Self::Exp(a)
			| Self::Log(a)
			| Self::Sqrt(a)
			| Self::Sin(a)
			| Self::Cos(a)
			| Self::Abs(a)
			| Self::Pow(a, _)
			| Self::Sum(a)
			| Self::Mean(a)
			| Self::Norm(a)
			| Self::Trace(a) => {
				ops[0] = Some(a);
			}
			Self::Add(a, b)
			| Self::Sub(a, b)
			| Self::Mul(a, b)
			| Self::Div(a, b)
			| Self::Dot(a, b)
			| Self::MatMul(a, b, _, _, _)
			| Self::ScalarMul(a, b)
			| Self::ScalarAdd(a, b) => {
				ops[0] = Some(a);
				ops[1] = Some(b);
			}
		}
		ops.into_iter().flatten()
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  TapeEntry
// ═══════════════════════════════════════════════════════════════════════════

/// Metadata for a single recorded computation.
#[derive(Debug)]
pub struct TapeEntry {
	pub op: OpCode,
	pub kind: ValueKind,
	pub requires_grad: bool,
	/// Last node index that reads this node's value during forward pass.
	/// Computed post-trace. Enables buffer donation when `last_forward_use`
	/// equals the current node and save policy is `NotNeeded`.
	pub last_forward_use: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tape<T>
// ═══════════════════════════════════════════════════════════════════════════

/// The computation graph with backend-native value storage.
pub struct Tape<T: RealScalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub(crate) entries: Vec<TapeEntry>,
	pub(crate) values: Vec<NodeValue<T>>,
	pub(crate) pool: BufferPool<T>,
	/// Per-op save policy — structural property of each operation, never overwritten.
	pub(crate) op_policies: Vec<SavePolicy>,
	/// Per-node survival flag — derived from op_policies.
	pub(crate) must_survive: Vec<bool>,
	/// Cached plan from the previous trace — survives `clear_for_reuse()`.
	plan: GraphPlan,
	/// Cursor: how many nodes of the current trace match the cached plan.
	plan_cursor: usize,
	/// Whether the current trace still matches the cached plan so far.
	/// Set at the start of each run based on whether a plan exists.
	/// Invalidated at the first mismatch during push().
	plan_valid: bool,
	/// Whether the plan has been confirmed by a full matching run.
	/// Donation is only allowed when `plan_confirmed == true`.
	///
	/// Lifecycle:
	/// - Run 0: no plan → trace + capture plan. `plan_confirmed = false`.
	/// - Run 1: plan exists → trace + validate against plan.
	///   If full match at end → `plan_confirmed = true`.
	///   If mismatch → plan invalidated, recapture.
	/// - Run 2+: `plan_confirmed = true` → donation active during forward.
	plan_confirmed: bool,
	/// Whether graph metadata (must_survive, last_forward_use) is computed
	/// for the *current* trace.
	metadata_valid: bool,
}

impl<T: RealScalar> Tape<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub fn new() -> Self {
		Self {
			entries: Vec::with_capacity(64),
			values: Vec::with_capacity(64),
			pool: BufferPool::new(),
			op_policies: Vec::with_capacity(64),
			must_survive: Vec::with_capacity(64),
			plan: GraphPlan::default(),
			plan_cursor: 0,
			plan_valid: false,
			plan_confirmed: false,
			metadata_valid: false,
		}
	}

	/// Reset the tape for a new evaluation, recycling non-saved buffers.
	///
	/// Buffers whose save policy is `NotNeeded` are returned to the pool.
	/// Buffers that were saved for backward are also recycled now (backward
	/// is complete by this point).
	pub fn clear_for_reuse(&mut self) {
		// Recycle all non-vacant values back to the pool
		for value in &mut self.values {
			match value.take_buffer() {
				NodeValue::Vector(v) => self.pool.return_vec(v),
				NodeValue::Matrix(m) => self.pool.return_mat(m),
				_ => {} // Scalar/Vacant — nothing to recycle
			}
		}
		self.entries.clear();
		self.values.clear();
		self.op_policies.clear();
		self.must_survive.clear();
		self.plan_cursor = 0;
		self.plan_valid = self.plan.is_populated();
		self.metadata_valid = false;
	}

	/// Full clear, also dropping pooled buffers and plan.
	pub fn clear(&mut self) {
		self.entries.clear();
		self.values.clear();
		self.op_policies.clear();
		self.must_survive.clear();
		self.pool.clear();
		self.plan.invalidate();
		self.plan_cursor = 0;
		self.plan_valid = false;
		self.plan_confirmed = false;
		self.metadata_valid = false;
	}

	/// Number of nodes.
	#[inline]
	pub fn len(&self) -> usize {
		self.entries.len()
	}

	/// Whether the tape is empty.
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	// ── Node construction ─────────────────────────────────────────────

	/// Push a new node onto the tape.
	///
	/// Validates against the cached [`GraphPlan`] if one exists.
	/// If the op+kind match the plan, donation remains possible for
	/// subsequent nodes. If they diverge, the plan is invalidated.
	pub(crate) fn push(
		&mut self,
		op: OpCode,
		value: NodeValue<T>,
		kind: ValueKind,
		requires_grad: bool,
	) -> NodeIdx {
		let cursor = self.entries.len();
		let idx = NodeIdx(cursor as u32);

		// Validate against cached plan
		if self.plan_valid {
			if self.plan.matches(cursor, &op, kind) {
				self.plan_cursor = cursor + 1;
			} else {
				// Graph structure diverged — invalidate plan
				self.plan_valid = false;
				self.plan.invalidate();
			}
		}

		self.entries.push(TapeEntry {
			op,
			kind,
			requires_grad,
			last_forward_use: idx.0,
		});
		self.values.push(value);
		self.metadata_valid = false;
		idx
	}

	// ── Value access ──────────────────────────────────────────────────

	/// Read the value at `idx`.
	#[inline]
	pub fn value(&self, idx: NodeIdx) -> &NodeValue<T> {
		&self.values[idx.0 as usize]
	}

	/// Read the scalar value at `idx`.
	#[inline]
	pub fn scalar(&self, idx: NodeIdx) -> T {
		self.values[idx.0 as usize].as_scalar()
	}

	/// Shape of the value at `idx`.
	#[inline]
	pub fn kind(&self, idx: NodeIdx) -> ValueKind {
		self.entries[idx.0 as usize].kind
	}

	// ── Post-trace graph analysis ─────────────────────────────────────

	/// Compute graph metadata after the forward pass completes.
	///
	/// Fills:
	/// - `op_policies[i]` — structural save policy of op `i` (never overwritten)
	/// - `must_survive[i]` — whether node `i`'s value is needed by a downstream backward
	/// - `entries[i].last_forward_use` — latest node that reads node `i` as operand
	pub(crate) fn compute_graph_metadata(&mut self) {
		let n = self.entries.len();

		// 1. op_policies — one per node, derived from OpCode.
		self.op_policies.clear();
		self.op_policies.reserve(n);
		for entry in &self.entries {
			self.op_policies.push(entry.op.save_policy());
		}

		// 2. must_survive — derived from op_policies of all nodes.
		self.must_survive.clear();
		self.must_survive.resize(n, false);
		for i in 0..n {
			let op = self.entries[i].op;
			match self.op_policies[i] {
				SavePolicy::SaveOutput => {
					self.must_survive[i] = true;
				}
				SavePolicy::SaveInput(idx) => {
					if let Some(operand) = op.operands().nth(idx as usize) {
						self.must_survive[operand.0 as usize] = true;
					}
				}
				SavePolicy::SaveBothInputs => {
					for operand in op.operands() {
						self.must_survive[operand.0 as usize] = true;
					}
				}
				SavePolicy::SaveOutputAndInput(idx) => {
					self.must_survive[i] = true;
					if let Some(operand) = op.operands().nth(idx as usize) {
						self.must_survive[operand.0 as usize] = true;
					}
				}
				SavePolicy::NotNeeded => {}
			}
		}

		// 3. last_forward_use
		for entry in &mut self.entries {
			entry.last_forward_use = 0;
		}
		for i in 0..n {
			for operand in self.entries[i].op.operands() {
				let a = operand.0 as usize;
				if (i as u32) > self.entries[a].last_forward_use {
					self.entries[a].last_forward_use = i as u32;
				}
			}
		}

		// Plan lifecycle:
		// - If plan_valid && plan matches full trace length → confirm the plan.
		//   Next run with this plan will allow donation.
		// - Otherwise → capture new plan, reset confirmation.
		if self.plan_valid && self.plan_cursor == n && self.plan.len() == n {
			// Full match — confirm the plan for donation on subsequent runs.
			self.plan_confirmed = true;
		} else {
			// Graph changed or first trace — capture new plan, await confirmation.
			self.plan.capture(&self.entries, &self.must_survive);
			self.plan_confirmed = false;
		}
		self.metadata_valid = true;
	}
}

impl<T: RealScalar> Default for Tape<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn default() -> Self {
		Self::new()
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Thread-local active tape (per scalar type)
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for per-type thread-local tape dispatch.
pub trait TapeThreadLocal: RealScalar
where
	linalg::DefaultBackend: LinAlgBackend<Self>,
{
	fn with_active<F, R>(f: F) -> R
	where
		F: FnOnce(&mut Tape<Self>) -> R;

	fn set_active(tape: *mut Tape<Self>) -> Option<*mut Tape<Self>>;
	fn restore_active(prev: Option<*mut Tape<Self>>);
}

macro_rules! impl_tape_thread_local {
	($scalar:ty, $tls:ident) => {
		thread_local! {
			static $tls: RefCell<Option<*mut Tape<$scalar>>> = const { RefCell::new(None) };
		}

		impl TapeThreadLocal for $scalar {
			#[inline]
			fn with_active<F, R>(f: F) -> R
			where
				F: FnOnce(&mut Tape<Self>) -> R,
			{
				$tls.with(|cell| {
					let ptr = cell
						.borrow()
						.expect("No active tape — wrap operations in a TapeGuard scope");
					let tape = unsafe { &mut *ptr };
					f(tape)
				})
			}

			#[inline]
			fn set_active(tape: *mut Tape<Self>) -> Option<*mut Tape<Self>> {
				$tls.with(|cell| {
					let prev = *cell.borrow();
					*cell.borrow_mut() = Some(tape);
					prev
				})
			}

			#[inline]
			fn restore_active(prev: Option<*mut Tape<Self>>) {
				$tls.with(|cell| *cell.borrow_mut() = prev);
			}
		}
	};
}

impl_tape_thread_local!(f32, ACTIVE_TAPE_F32);
impl_tape_thread_local!(f64, ACTIVE_TAPE_F64);

// ═══════════════════════════════════════════════════════════════════════════
//  TapeGuard
// ═══════════════════════════════════════════════════════════════════════════

/// RAII guard that sets the thread-local active tape.
pub struct TapeGuard<T: TapeThreadLocal>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	prev: Option<*mut Tape<T>>,
	_marker: std::marker::PhantomData<T>,
}

impl<T: TapeThreadLocal> TapeGuard<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub fn new(tape: &mut Tape<T>) -> Self {
		let prev = T::set_active(tape as *mut Tape<T>);
		Self {
			prev,
			_marker: std::marker::PhantomData,
		}
	}
}

impl<T: TapeThreadLocal> Drop for TapeGuard<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn drop(&mut self) {
		T::restore_active(self.prev);
	}
}

/// Execute `f` with mutable access to the active tape.
#[inline]
pub fn with_tape<T: TapeThreadLocal, F, R>(f: F) -> R
where
	F: FnOnce(&mut Tape<T>) -> R,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	T::with_active(f)
}
