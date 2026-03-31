//! Typed buffer pool — the zero-allocation arena for autodiff.
//!
//! Three separate arenas hold scalars (`T`), vectors (`B::Vector`), and
//! matrices (`B::Matrix`).  Allocation is O(1) cursor-advance; reset is
//! three pointer stores.  After the first forward pass the pool has enough
//! capacity and no heap allocation occurs in steady state.
//!
//! # Borrow strategy
//!
//! The three arenas are stored as separate `pub(crate)` fields so that
//! callers (notably [`AdSession`](crate::session::AdSession)) can borrow
//! them independently — e.g. read a matrix while writing a vector.

use std::marker::PhantomData;

use riemannopt_core::linalg::{
	LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView,
};

use crate::var::{MVar, SVar, VVar};

/// Backend-agnostic memory arena with typed slots for scalars, vectors,
/// and matrices.
pub struct BufferPool<T: RealScalar, B: LinAlgBackend<T>> {
	pub(crate) scalars: Vec<T>,
	pub(crate) scalar_cursor: usize,

	pub(crate) vectors: Vec<B::Vector>,
	pub(crate) vector_cursor: usize,

	pub(crate) matrices: Vec<B::Matrix>,
	pub(crate) matrix_cursor: usize,

	_phantom: PhantomData<B>,
}

impl<T: RealScalar, B: LinAlgBackend<T>> BufferPool<T, B> {
	/// Creates an empty pool.
	pub fn new() -> Self {
		Self {
			scalars: Vec::new(),
			scalar_cursor: 0,
			vectors: Vec::new(),
			vector_cursor: 0,
			matrices: Vec::new(),
			matrix_cursor: 0,
			_phantom: PhantomData,
		}
	}

	// ── Reset ────────────────────────────────────────────────────────

	/// Resets all cursors to zero.  No memory is freed — buffers are
	/// recycled on the next forward pass.
	#[inline]
	pub fn reset(&mut self) {
		self.scalar_cursor = 0;
		self.vector_cursor = 0;
		self.matrix_cursor = 0;
	}

	// ── Scalar arena ─────────────────────────────────────────────────

	/// Allocates a scalar slot and returns its index.
	#[inline]
	pub fn alloc_scalar(&mut self) -> u32 {
		let idx = self.scalar_cursor;
		if idx >= self.scalars.len() {
			self.scalars.push(T::zero());
		}
		self.scalar_cursor += 1;
		idx as u32
	}

	/// Allocates a scalar slot pre-filled with `val`.
	#[inline]
	pub fn alloc_scalar_with(&mut self, val: T) -> u32 {
		let idx = self.alloc_scalar();
		self.scalars[idx as usize] = val;
		idx
	}

	/// Reads a scalar value.
	#[inline(always)]
	pub fn scalar(&self, v: SVar) -> T {
		self.scalars[v.idx()]
	}

	/// Mutable reference to a scalar slot.
	#[inline(always)]
	pub fn scalar_mut(&mut self, v: SVar) -> &mut T {
		&mut self.scalars[v.idx()]
	}

	/// Number of active scalar slots.
	#[inline]
	pub fn scalar_cursor(&self) -> usize {
		self.scalar_cursor
	}

	// ── Vector arena ─────────────────────────────────────────────────

	/// Allocates a vector slot of length `len`.
	///
	/// On the first pass, pushes `B::Vector::zeros(len)`.  On subsequent
	/// passes the existing buffer is reused (dimensions are checked in
	/// debug mode only).
	#[inline]
	pub fn alloc_vector(&mut self, len: usize) -> u32 {
		let idx = self.vector_cursor;
		if idx >= self.vectors.len() {
			self.vectors.push(B::Vector::zeros(len));
		} else {
			debug_assert_eq!(
				self.vectors[idx].len(),
				len,
				"vector pool: dimension mismatch at slot {idx} (expected {len}, got {})",
				self.vectors[idx].len()
			);
		}
		self.vector_cursor += 1;
		idx as u32
	}

	/// Reads a vector.
	#[inline(always)]
	pub fn vector(&self, v: VVar) -> &B::Vector {
		&self.vectors[v.idx()]
	}

	/// Mutable reference to a vector slot.
	#[inline(always)]
	pub fn vector_mut(&mut self, v: VVar) -> &mut B::Vector {
		&mut self.vectors[v.idx()]
	}

	/// Number of active vector slots.
	#[inline]
	pub fn vector_cursor(&self) -> usize {
		self.vector_cursor
	}

	/// Two disjoint mutable references to vector slots.
	///
	/// # Panics
	/// Panics if `a == b`.
	#[inline]
	pub fn vectors_mut_pair(&mut self, a: u32, b: u32) -> (&mut B::Vector, &mut B::Vector) {
		assert_ne!(a, b, "vectors_mut_pair: cannot borrow the same slot twice");
		let (ai, bi) = (a as usize, b as usize);
		if ai < bi {
			let (left, right) = self.vectors.split_at_mut(bi);
			(&mut left[ai], &mut right[0])
		} else {
			let (left, right) = self.vectors.split_at_mut(ai);
			(&mut right[0], &mut left[bi])
		}
	}

	// ── Matrix arena ─────────────────────────────────────────────────

	/// Allocates a matrix slot of shape `(nrows, ncols)`.
	#[inline]
	pub fn alloc_matrix(&mut self, nrows: usize, ncols: usize) -> u32 {
		let idx = self.matrix_cursor;
		if idx >= self.matrices.len() {
			self.matrices.push(B::Matrix::zeros(nrows, ncols));
		} else {
			debug_assert_eq!(
				(self.matrices[idx].nrows(), self.matrices[idx].ncols()),
				(nrows, ncols),
				"matrix pool: shape mismatch at slot {idx}"
			);
		}
		self.matrix_cursor += 1;
		idx as u32
	}

	/// Reads a matrix.
	#[inline(always)]
	pub fn matrix(&self, v: MVar) -> &B::Matrix {
		&self.matrices[v.idx()]
	}

	/// Mutable reference to a matrix slot.
	#[inline(always)]
	pub fn matrix_mut(&mut self, v: MVar) -> &mut B::Matrix {
		&mut self.matrices[v.idx()]
	}

	/// Number of active matrix slots.
	#[inline]
	pub fn matrix_cursor(&self) -> usize {
		self.matrix_cursor
	}

	/// Two disjoint mutable references to matrix slots.
	///
	/// # Panics
	/// Panics if `a == b`.
	#[inline]
	pub fn matrices_mut_pair(&mut self, a: u32, b: u32) -> (&mut B::Matrix, &mut B::Matrix) {
		assert_ne!(a, b, "matrices_mut_pair: cannot borrow the same slot twice");
		let (ai, bi) = (a as usize, b as usize);
		if ai < bi {
			let (left, right) = self.matrices.split_at_mut(bi);
			(&mut left[ai], &mut right[0])
		} else {
			let (left, right) = self.matrices.split_at_mut(ai);
			(&mut right[0], &mut left[bi])
		}
	}

	// ── Forward-pass helpers ────────────────────────────────────────
	// These methods leverage the invariant that `out_idx` was just
	// allocated and is therefore strictly greater than any input index.

	/// Returns `(&input, &mut output)` for two vector slots where
	/// `out > inp` (guaranteed by allocation order).
	#[inline]
	pub fn vec_ref_mut(&mut self, inp: u32, out: u32) -> (&B::Vector, &mut B::Vector) {
		debug_assert!(
			(out as usize) > (inp as usize),
			"vec_ref_mut: out ({out}) must be > inp ({inp})"
		);
		let (left, right) = self.vectors.split_at_mut(out as usize);
		(&left[inp as usize], &mut right[0])
	}

	/// Returns `(&in_a, &in_b, &mut output)` for three vector slots
	/// where `out > max(a, b)`.
	#[inline]
	pub fn vec_ref2_mut(
		&mut self,
		a: u32,
		b: u32,
		out: u32,
	) -> (&B::Vector, &B::Vector, &mut B::Vector) {
		debug_assert!((out as usize) > (a as usize));
		debug_assert!((out as usize) > (b as usize));
		let (left, right) = self.vectors.split_at_mut(out as usize);
		(&left[a as usize], &left[b as usize], &mut right[0])
	}

	/// Returns `(&in_a, &in_b, &mut output)` for three matrix slots
	/// where `out > max(a, b)`.
	#[inline]
	pub fn mat_ref2_mut(
		&mut self,
		a: u32,
		b: u32,
		out: u32,
	) -> (&B::Matrix, &B::Matrix, &mut B::Matrix) {
		debug_assert!((out as usize) > (a as usize));
		debug_assert!((out as usize) > (b as usize));
		let (left, right) = self.matrices.split_at_mut(out as usize);
		(&left[a as usize], &left[b as usize], &mut right[0])
	}

	/// Returns `(&input_mat, &mut output_mat)` where `out > inp`.
	#[inline]
	pub fn mat_ref_mut(&mut self, inp: u32, out: u32) -> (&B::Matrix, &mut B::Matrix) {
		debug_assert!((out as usize) > (inp as usize));
		let (left, right) = self.matrices.split_at_mut(out as usize);
		(&left[inp as usize], &mut right[0])
	}

	// ── Bulk initialisation (for gradient pool) ─────────────────────

	/// Ensures the pool has at least `n` scalar slots, all zeroed.
	pub fn ensure_scalars_zeroed(&mut self, n: usize) {
		self.scalars.resize(n, T::zero());
		for s in &mut self.scalars[..n] {
			*s = T::zero();
		}
		self.scalar_cursor = n;
	}

	/// Ensures the pool has vector slots matching the shapes of `source`.
	/// All vectors are filled with zero.
	pub fn ensure_vectors_zeroed_like(&mut self, source: &Self) {
		let n = source.vector_cursor;
		while self.vectors.len() < n {
			let src_len = source.vectors[self.vectors.len()].len();
			self.vectors.push(B::Vector::zeros(src_len));
		}
		for i in 0..n {
			debug_assert_eq!(self.vectors[i].len(), source.vectors[i].len());
			self.vectors[i].fill(T::zero());
		}
		self.vector_cursor = n;
	}

	/// Ensures the pool has matrix slots matching the shapes of `source`.
	/// All matrices are filled with zero.
	pub fn ensure_matrices_zeroed_like(&mut self, source: &Self) {
		let n = source.matrix_cursor;
		while self.matrices.len() < n {
			let src = &source.matrices[self.matrices.len()];
			self.matrices
				.push(B::Matrix::zeros(src.nrows(), src.ncols()));
		}
		for i in 0..n {
			debug_assert_eq!(
				(self.matrices[i].nrows(), self.matrices[i].ncols()),
				(source.matrices[i].nrows(), source.matrices[i].ncols())
			);
			self.matrices[i].fill(T::zero());
		}
		self.matrix_cursor = n;
	}
}

impl<T: RealScalar, B: LinAlgBackend<T>> Default for BufferPool<T, B> {
	fn default() -> Self {
		Self::new()
	}
}
