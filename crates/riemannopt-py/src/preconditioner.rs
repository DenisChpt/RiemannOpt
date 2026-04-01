//! Python-facing preconditioner types.
//!
//! Follows the same `Dyn*` / `Py*` / `with_*!` pattern as
//! `solver.rs`, `manifold.rs`, and `problem.rs`.
//!
//! # Jacobi design note
//!
//! `JacobiPreconditioner<T, M>` is generic over the manifold because it
//! stores `M::TangentVector`. We don't know `M` at Python construction
//! time — only at dispatch time inside the `on_manifold!` macro.
//!
//! Solution: `DynPreconditioner::Jacobi` stores the raw diagonal as a
//! `Vec<f64>`. The `with_preconditioner!` macro accepts the concrete
//! manifold, constructs the typed `JacobiPreconditioner` on the fly, and
//! passes it to `solve`. This is essentially free (one memcpy of the
//! diagonal per solve call) and Jacobi has no accumulating state.
//!
//! # Python API
//!
//! ```python
//! from riemannopt import Preconditioner
//!
//! # No preconditioning (default when omitted):
//! result = solve(solver, problem, manifold, x0)
//!
//! # L-BFGS (accumulates curvature, state preserved across calls):
//! pre = Preconditioner.lbfgs(memory=10)
//! result = solve(solver, problem, manifold, x0, preconditioner=pre)
//!
//! # Jacobi (fixed diagonal scaling):
//! diag = np.abs(np.diag(A))           # e.g. diag(Hessian)
//! pre = Preconditioner.jacobi(diag)
//! result = solve(solver, problem, manifold, x0, preconditioner=pre)
//! ```

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

use riemannopt_core::manifold::Manifold;
use riemannopt_core::preconditioner::{
	AsElementWise, IdentityPreconditioner, JacobiPreconditioner, LbfgsPreconditioner,
};
use riemannopt_core::types::Scalar;

// ════════════════════════════════════════════════════════════════════════
//  Dynamic dispatch enum
// ════════════════════════════════════════════════════════════════════════

/// Untyped preconditioner enum for Python dispatch.
#[derive(Debug)]
pub enum DynPreconditioner {
	Identity(IdentityPreconditioner),
	Lbfgs(LbfgsPreconditioner),
	/// Jacobi stores the raw diagonal; the typed preconditioner is
	/// constructed at dispatch time when the manifold type is known.
	Jacobi {
		diagonal: Vec<f64>,
		floor: f64,
	},
}

impl Default for DynPreconditioner {
	fn default() -> Self {
		Self::Identity(IdentityPreconditioner)
	}
}

/// Constructs a typed `JacobiPreconditioner` from raw diagonal data and
/// a concrete manifold. Called inside `with_preconditioner!`.
///
/// Copies the diagonal into a manifold-allocated tangent vector via
/// contiguous slice access (`AsElementWise`). Panics if the diagonal
/// length does not match the tangent vector dimension.
pub fn build_jacobi<M: Manifold<f64>>(
	manifold: &M,
	raw_diagonal: &[f64],
	floor: f64,
) -> JacobiPreconditioner<f64, M>
where
	M::TangentVector: AsElementWise<f64>,
{
	let mut diag = manifold.allocate_tangent();
	{
		let slice = diag.ew_as_mut_slice();
		assert_eq!(
			slice.len(),
			raw_diagonal.len(),
			"Jacobi diagonal length ({}) does not match tangent dimension ({})",
			raw_diagonal.len(),
			slice.len(),
		);
		slice.copy_from_slice(raw_diagonal);
	}
	JacobiPreconditioner::fixed(diag).with_floor(<f64 as Scalar>::from_f64(floor))
}

/// Dispatches on `DynPreconditioner`, binding the inner concrete type.
///
/// For Identity and L-BFGS, the stored preconditioner is used directly.
/// For Jacobi, a typed `JacobiPreconditioner` is constructed on the fly
/// from the raw diagonal and the concrete manifold `$m`.
///
/// Usage:
/// ```ignore
/// with_preconditioner!(pre, manifold_ref, |p| {
///     s.solve(problem, manifold_ref, point, p, stop)
/// })
/// ```
macro_rules! with_preconditioner {
	($pre:expr, $m:expr, |$p:ident| $body:expr) => {
		match $pre {
			DynPreconditioner::Identity(ref mut $p) => $body,
			DynPreconditioner::Lbfgs(ref mut $p) => $body,
			DynPreconditioner::Jacobi {
				ref diagonal,
				floor,
			} => {
				let mut __jacobi = $crate::preconditioner::build_jacobi($m, diagonal, *floor);
				let $p = &mut __jacobi;
				$body
			}
		}
	};
}

pub(crate) use with_preconditioner;

// ════════════════════════════════════════════════════════════════════════
//  Python class
// ════════════════════════════════════════════════════════════════════════

/// A preconditioner for Riemannian solvers.
///
/// Preconditioners improve convergence of second-order solvers (Trust
/// Region, Newton, Conjugate Gradient) by approximating the inverse
/// Hessian. First-order solvers (SGD, Adam) ignore them.
///
/// Create instances via the static methods:
///
/// - ``Preconditioner.identity()`` — no preconditioning (default)
/// - ``Preconditioner.lbfgs(memory)`` — limited-memory BFGS
/// - ``Preconditioner.jacobi(diagonal)`` — diagonal scaling
#[pyclass(name = "Preconditioner")]
#[derive(Debug)]
pub struct PyPreconditioner {
	pub inner: DynPreconditioner,
}

#[pymethods]
impl PyPreconditioner {
	/// Creates an identity preconditioner (P = I, no effect).
	///
	/// This is what the solver uses when no preconditioner is passed.
	#[staticmethod]
	fn identity() -> Self {
		Self {
			inner: DynPreconditioner::Identity(IdentityPreconditioner),
		}
	}

	/// Creates an L-BFGS preconditioner.
	///
	/// Maintains a circular buffer of curvature pairs and applies the
	/// two-loop recursion to approximate H⁻¹. Recommended default when
	/// preconditioning is desired and no problem-specific diagonal is
	/// available.
	///
	/// The preconditioner accumulates state across solver iterations.
	/// Reuse the same object across multiple ``solve()`` calls if you
	/// want to warm-start from the previous curvature information.
	///
	/// Args:
	///     memory: Number of curvature pairs to store (5–20 typical).
	///     initial_scale: Optional fixed γ for H₀ = γI. When ``None``
	///         (default), uses automatic Shanno–Phua scaling.
	#[staticmethod]
	#[pyo3(signature = (memory=10, initial_scale=None))]
	fn lbfgs(memory: usize, initial_scale: Option<f64>) -> PyResult<Self> {
		if memory == 0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"L-BFGS memory must be at least 1",
			));
		}

		let mut pre = LbfgsPreconditioner::new(memory);
		if let Some(gamma) = initial_scale {
			pre = pre.with_initial_scale(gamma);
		}

		Ok(Self {
			inner: DynPreconditioner::Lbfgs(pre),
		})
	}

	/// Creates a Jacobi (diagonal) preconditioner.
	///
	/// Applies P⁻¹ v = v / diagonal element-wise. Use when a good
	/// approximation of diag(Hessian) is available, e.g.
	/// ``np.abs(np.diag(A))`` for a quadratic cost f(x) = ½xᵀAx.
	///
	/// The diagonal is copied once. The preconditioner is stateless —
	/// it is reconstructed internally on each ``solve()`` call (the
	/// cost is negligible: one memcpy of the diagonal).
	///
	/// Args:
	///     diagonal: 1-D numpy array of strictly positive scaling factors.
	///         Length must match the manifold's tangent vector dimension.
	///     floor: Minimum value for diagonal entries (clamped to avoid
	///         division by near-zero). Default: 1e-12.
	#[staticmethod]
	#[pyo3(signature = (diagonal, floor=1e-12))]
	fn jacobi<'py>(diagonal: PyReadonlyArrayDyn<'py, f64>, floor: f64) -> PyResult<Self> {
		let shape = diagonal.shape();
		if shape.len() != 1 {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"Expected 1-D array for Jacobi diagonal, got {}D",
				shape.len(),
			)));
		}

		let slice = diagonal
			.as_slice()
			.map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;

		if slice.iter().any(|&v| v <= 0.0) {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"Jacobi diagonal entries must be strictly positive",
			));
		}

		if floor <= 0.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"Floor must be strictly positive",
			));
		}

		Ok(Self {
			inner: DynPreconditioner::Jacobi {
				diagonal: slice.to_vec(),
				floor,
			},
		})
	}

	fn __repr__(&self) -> String {
		match &self.inner {
			DynPreconditioner::Identity(_) => "Preconditioner.identity()".to_string(),
			DynPreconditioner::Lbfgs(_) => "Preconditioner.lbfgs(...)".to_string(),
			DynPreconditioner::Jacobi { diagonal, .. } => {
				format!("Preconditioner.jacobi(dim={})", diagonal.len())
			}
		}
	}
}
