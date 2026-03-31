//! Typed variable handles for the autodiff tape.
//!
//! Each handle is a lightweight index (4 bytes, `Copy`) into the
//! corresponding arena of the [`BufferPool`](crate::pool::BufferPool).
//! The type-level distinction (`SVar` / `VVar` / `MVar`) catches
//! pool-type mismatches at compile time.

/// Scalar variable — index into `BufferPool::scalars`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SVar(pub(crate) u32);

/// Vector variable — index into `BufferPool::vectors`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VVar(pub(crate) u32);

/// Matrix variable — index into `BufferPool::matrices`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MVar(pub(crate) u32);

impl SVar {
	/// Returns the pool index as `usize`.
	#[inline(always)]
	pub fn idx(self) -> usize {
		self.0 as usize
	}
}

impl VVar {
	/// Returns the pool index as `usize`.
	#[inline(always)]
	pub fn idx(self) -> usize {
		self.0 as usize
	}
}

impl MVar {
	/// Returns the pool index as `usize`.
	#[inline(always)]
	pub fn idx(self) -> usize {
		self.0 as usize
	}
}
