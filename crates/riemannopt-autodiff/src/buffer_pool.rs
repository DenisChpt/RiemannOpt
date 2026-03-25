//! Buffer pool for recycling backend-allocated vectors and matrices.
//!
//! Organizes buffers into size-class buckets to minimize allocation overhead.
//! In steady state (same computation graph), the pool covers all allocations
//! and no heap allocation occurs.

use riemannopt_core::linalg::{self, LinAlgBackend, MatrixOps, RealScalar, VectorOps};

// ═══════════════════════════════════════════════════════════════════════════
//  Size class helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Map a vector length to a bucket index (0..NUM_VEC_BUCKETS).
#[inline]
fn vec_bucket(len: usize) -> usize {
	match len {
		0..=16 => 0,
		17..=64 => 1,
		65..=256 => 2,
		257..=1024 => 3,
		_ => 4,
	}
}

const NUM_VEC_BUCKETS: usize = 5;

// ═══════════════════════════════════════════════════════════════════════════
//  BufferPool
// ═══════════════════════════════════════════════════════════════════════════

/// Recycles `linalg::Vec<T>` and `linalg::Mat<T>` buffers by size class.
///
/// Vectors are bucketed by capacity class (0-16, 17-64, 65-256, 257-1024, 1025+).
/// Matrices are stored with their shape metadata for best-fit matching.
pub struct BufferPool<T: RealScalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	vec_buckets: [Vec<linalg::Vec<T>>; NUM_VEC_BUCKETS],
	/// Matrices stored as (rows, cols, buffer).
	mat_pool: Vec<(usize, usize, linalg::Mat<T>)>,
}

impl<T: RealScalar> BufferPool<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	pub fn new() -> Self {
		Self {
			vec_buckets: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
			mat_pool: Vec::new(),
		}
	}

	/// Get a zero-initialized vector of length `n`.
	///
	/// Tries to recycle a buffer from the matching bucket. If none available,
	/// allocates a new one.
	pub fn get_vec(&mut self, n: usize) -> linalg::Vec<T> {
		// faer::Col<T> has fixed size — only exact matches are reusable.
		// Search the matching bucket first, then all buckets.
		let bucket = vec_bucket(n);
		for b in bucket..NUM_VEC_BUCKETS {
			if let Some(pos) = self.vec_buckets[b]
				.iter()
				.position(|v| VectorOps::len(v) == n)
			{
				let mut v = self.vec_buckets[b].swap_remove(pos);
				v.fill(T::zero());
				return v;
			}
		}
		// Also check smaller buckets (a buffer might have been misclassified
		// or the bucket boundary doesn't match exactly)
		for b in 0..bucket {
			if let Some(pos) = self.vec_buckets[b]
				.iter()
				.position(|v| VectorOps::len(v) == n)
			{
				let mut v = self.vec_buckets[b].swap_remove(pos);
				v.fill(T::zero());
				return v;
			}
		}
		VectorOps::zeros(n)
	}

	/// Return a vector buffer to the pool for future reuse.
	pub fn return_vec(&mut self, v: linalg::Vec<T>) {
		let n = VectorOps::len(&v);
		if n == 0 {
			return;
		}
		let bucket = vec_bucket(n);
		self.vec_buckets[bucket].push(v);
	}

	/// Get a zero-initialized matrix of shape `(rows, cols)`.
	pub fn get_mat(&mut self, rows: usize, cols: usize) -> linalg::Mat<T> {
		// Find exact shape match first
		if let Some(pos) = self
			.mat_pool
			.iter()
			.position(|(r, c, _)| *r == rows && *c == cols)
		{
			let (_, _, mut m) = self.mat_pool.swap_remove(pos);
			m.fill(T::zero());
			return m;
		}
		// No match — allocate fresh
		MatrixOps::zeros(rows, cols)
	}

	/// Return a matrix buffer to the pool.
	pub fn return_mat(&mut self, m: linalg::Mat<T>) {
		let rows = MatrixOps::nrows(&m);
		let cols = MatrixOps::ncols(&m);
		if rows == 0 || cols == 0 {
			return;
		}
		self.mat_pool.push((rows, cols, m));
	}

	/// Clear the pool, dropping all buffers.
	pub fn clear(&mut self) {
		for bucket in &mut self.vec_buckets {
			bucket.clear();
		}
		self.mat_pool.clear();
	}
}

impl<T: RealScalar> Default for BufferPool<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn default() -> Self {
		Self::new()
	}
}
