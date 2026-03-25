//! Core value types for the autodiff engine.
//!
//! [`NodeValue`] stores forward-pass values and gradients as backend-native
//! types (`linalg::Vec<T>`, `linalg::Mat<T>`), enabling SIMD operations
//! without intermediate copies.

use riemannopt_core::linalg::{self, LinAlgBackend, MatrixOps, RealScalar, VectorOps};

// ═══════════════════════════════════════════════════════════════════════════
//  ValueKind — semantic typing of tape nodes
// ═══════════════════════════════════════════════════════════════════════════

/// The semantic type of a tape node's value.
///
/// This is the source of truth for runtime type dispatch — not a derived
/// metadata. Every [`NodeValue`] has a corresponding `ValueKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
	Scalar,
	Vector(usize),
	Matrix(usize, usize),
}

impl ValueKind {
	/// Total number of scalar elements.
	#[inline]
	pub fn len(self) -> usize {
		match self {
			Self::Scalar => 1,
			Self::Vector(n) => n,
			Self::Matrix(r, c) => r * c,
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  SavePolicy — backward strategy per operation
// ═══════════════════════════════════════════════════════════════════════════

/// Declares which values an operation needs saved for its backward pass.
///
/// This is a property of the operation (OpCode), not of the node.
/// It determines which forward-pass values must survive until backward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavePolicy {
	/// No values needed (Add, Sub, Neg, Sum, Mean, ScalarAdd, Trace).
	NotNeeded,
	/// Save the output value of this node (Exp, Sqrt).
	SaveOutput,
	/// Save input at the given operand index (Log→0, Sin→0, Cos→0, Abs→0, Pow→0).
	SaveInput(u8),
	/// Save both inputs (Mul, Div, Dot, MatMul, ScalarMul).
	SaveBothInputs,
	/// Save both the output and an input (Norm: needs a_val + ‖a‖).
	SaveOutputAndInput(u8),
}

// ═══════════════════════════════════════════════════════════════════════════
//  NodeValue — typed backend storage
// ═══════════════════════════════════════════════════════════════════════════

/// A value stored on the tape or in the gradient buffer.
///
/// Each variant wraps a backend-native type, enabling SIMD/BLAS operations
/// without copies to/from a flat arena.
///
/// # Invariant
///
/// `Vacant` is a logically empty slot. Any read access (`as_vec`, `as_mat`,
/// `as_scalar`) on a `Vacant` value will panic in debug mode. Only the
/// buffer pool and lifecycle manager should create or transition to `Vacant`.
pub enum NodeValue<T: RealScalar>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Logically empty slot — buffer has been donated or not yet assigned.
	Vacant,
	/// A single scalar value.
	Scalar(T),
	/// A column vector (faer::Col<T>).
	Vector(linalg::Vec<T>),
	/// A dense matrix (faer::Mat<T>).
	Matrix(linalg::Mat<T>),
}

impl<T: RealScalar> NodeValue<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	// ── Construction ──────────────────────────────────────────────────

	/// Create a zero-initialized value of the given kind.
	pub fn zeros(kind: ValueKind) -> Self {
		match kind {
			ValueKind::Scalar => Self::Scalar(T::zero()),
			ValueKind::Vector(n) => Self::Vector(VectorOps::zeros(n)),
			ValueKind::Matrix(r, c) => Self::Matrix(MatrixOps::zeros(r, c)),
		}
	}

	/// Create a zero-initialized value with the same kind as `self`.
	pub fn zeros_like(&self) -> Self {
		Self::zeros(self.kind())
	}

	// ── Queries ───────────────────────────────────────────────────────

	/// The semantic kind of this value.
	pub fn kind(&self) -> ValueKind {
		match self {
			Self::Vacant => panic!("NodeValue::kind() called on Vacant"),
			Self::Scalar(_) => ValueKind::Scalar,
			Self::Vector(v) => ValueKind::Vector(VectorOps::len(v)),
			Self::Matrix(m) => ValueKind::Matrix(MatrixOps::nrows(m), MatrixOps::ncols(m)),
		}
	}

	/// Whether this slot is vacant (no logical value).
	#[inline]
	pub fn is_vacant(&self) -> bool {
		matches!(self, Self::Vacant)
	}

	// ── Typed accessors (panic on wrong variant) ──────────────────────

	#[inline]
	pub fn as_scalar(&self) -> T {
		match self {
			Self::Scalar(v) => *v,
			Self::Vacant => panic!("as_scalar() on Vacant"),
			_ => panic!("as_scalar() on non-scalar NodeValue"),
		}
	}

	#[inline]
	pub fn as_scalar_mut(&mut self) -> &mut T {
		match self {
			Self::Scalar(v) => v,
			_ => panic!("as_scalar_mut() on non-scalar NodeValue"),
		}
	}

	#[inline]
	pub fn as_vec(&self) -> &linalg::Vec<T> {
		match self {
			Self::Vector(v) => v,
			Self::Vacant => panic!("as_vec() on Vacant"),
			_ => panic!("as_vec() on non-vector NodeValue"),
		}
	}

	#[inline]
	pub fn as_vec_mut(&mut self) -> &mut linalg::Vec<T> {
		match self {
			Self::Vector(v) => v,
			_ => panic!("as_vec_mut() on non-vector NodeValue"),
		}
	}

	#[inline]
	pub fn as_mat(&self) -> &linalg::Mat<T> {
		match self {
			Self::Matrix(m) => m,
			Self::Vacant => panic!("as_mat() on Vacant"),
			_ => panic!("as_mat() on non-matrix NodeValue"),
		}
	}

	#[inline]
	pub fn as_mat_mut(&mut self) -> &mut linalg::Mat<T> {
		match self {
			Self::Matrix(m) => m,
			_ => panic!("as_mat_mut() on non-matrix NodeValue"),
		}
	}

	/// Get or convert to a matrix. Vectors become n×1 matrices (allocates a copy).
	/// Matrices are returned by reference wrapper.
	pub fn to_mat(&self) -> linalg::Mat<T> {
		match self {
			Self::Matrix(m) => m.clone(),
			Self::Vector(v) => {
				let n = VectorOps::len(v);
				<linalg::Mat<T> as MatrixOps<T>>::from_column_slice(n, 1, v.as_slice())
			}
			_ => panic!("to_mat: unsupported variant"),
		}
	}

	// ── In-place operations ──────────────────────────────────────────

	/// `self += alpha * other` — the critical gradient accumulation path.
	///
	/// Dispatches to `VectorOps::axpy` (SIMD via faer::zip!) for vectors,
	/// `MatrixOps::mat_axpy` for matrices, and scalar arithmetic for scalars.
	pub fn axpy_accum(&mut self, alpha: T, other: &Self) {
		match (self, other) {
			(Self::Scalar(s), Self::Scalar(o)) => {
				*s = *s + alpha * *o;
			}
			(Self::Vector(v), Self::Vector(o)) => {
				v.axpy(alpha, o, T::one());
			}
			(Self::Matrix(m), Self::Matrix(o)) => {
				m.mat_axpy(alpha, o, T::one());
			}
			(Self::Vacant, _) | (_, Self::Vacant) => {
				panic!("axpy_accum on Vacant");
			}
			_ => panic!("axpy_accum: shape mismatch"),
		}
	}

	/// Zero all elements without changing the allocation.
	pub fn fill_zero(&mut self) {
		match self {
			Self::Vacant => {}
			Self::Scalar(v) => *v = T::zero(),
			Self::Vector(v) => v.fill(T::zero()),
			Self::Matrix(m) => m.fill(T::zero()),
		}
	}

	/// Extract the buffer, leaving `Vacant` behind.
	///
	/// Used for buffer donation: a dead node gives its buffer to a new node.
	pub fn take_buffer(&mut self) -> Self {
		std::mem::replace(self, Self::Vacant)
	}

	/// Overwrite this value with another, reusing the allocation if shapes match.
	///
	/// If the existing buffer has the right shape, `copy_from` is used (no alloc).
	/// Otherwise, the new value replaces the old one entirely.
	pub fn overwrite(&mut self, new_val: Self) {
		match (self, &new_val) {
			(Self::Scalar(s), Self::Scalar(o)) => *s = *o,
			(Self::Vector(v), Self::Vector(o)) if VectorOps::len(v) == VectorOps::len(o) => {
				v.copy_from(o);
			}
			(Self::Matrix(m), Self::Matrix(o))
				if MatrixOps::nrows(m) == MatrixOps::nrows(o)
					&& MatrixOps::ncols(m) == MatrixOps::ncols(o) =>
			{
				m.copy_from(o);
			}
			(slot, _) => {
				*slot = new_val;
			}
		}
	}
}

// Manual Debug impl to avoid requiring Debug on linalg types
impl<T: RealScalar> std::fmt::Debug for NodeValue<T>
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Vacant => write!(f, "Vacant"),
			Self::Scalar(v) => write!(f, "Scalar({v})"),
			Self::Vector(v) => write!(f, "Vector(len={})", VectorOps::len(v)),
			Self::Matrix(m) => {
				write!(f, "Matrix({}x{})", MatrixOps::nrows(m), MatrixOps::ncols(m))
			}
		}
	}
}
