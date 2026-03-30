//! Zero-allocation reverse-mode automatic differentiation for
//! Riemannian optimization.
//!
//! This crate provides a typed buffer pool, flat instruction tape, and
//! in-place VJP backward pass that integrates with the [`Problem`] trait
//! from `riemannopt-core`.  After the first forward pass the entire
//! forward–backward cycle runs without heap allocation.
//!
//! # Architecture
//!
//! * **[`BufferPool`]** — three typed arenas (scalars, vectors, matrices)
//!   with O(1) cursor-based allocation and zero-cost reset.
//! * **[`Tape`]** — a flat `Vec<Op>` of enum instructions.  No data,
//!   no `dyn Trait`, no `Box`.
//! * **[`AdSession`]** — user-facing API that owns pool + tape + grad pool.
//!   Eagerly computes forward values and records ops for the backward pass.
//! * **[`AutoDiffProblem`] / [`AutoDiffMatProblem`]** — implement
//!   `Problem<T, M>` so any solver can optimise an AD-defined cost.
//! * **[`Dual`]** — dual numbers for forward-mode AD (Hessian-vector
//!   products via forward-over-reverse).
//!
//! [`Problem`]: riemannopt_core::problem::Problem

pub mod backward;
pub mod dual;
pub mod pool;
pub mod problem;
pub mod session;
pub mod tape;
pub mod var;

// ── Public re-exports ────────────────────────────────────────────────

pub use dual::Dual;
pub use pool::BufferPool;
pub use problem::{AutoDiffMatProblem, AutoDiffProblem};
pub use session::AdSession;
pub use tape::{Op, Tape};
pub use var::{MVar, SVar, VVar};
