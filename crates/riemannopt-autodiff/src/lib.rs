//! Tape-based reverse-mode automatic differentiation for Riemannian
//! optimization.
//!
//! # Architecture
//!
//! The engine uses a **Wengert list** (tape) stored as a flat `Vec<TapeEntry>`.
//! During the forward pass entries are appended in topological order.
//! The backward pass simply iterates in reverse — no graph traversal, no
//! `HashMap`, no `RefCell` on the hot path.
//!
//! # Quick Start
//!
//! ```
//! use riemannopt_autodiff::{Tape, TapeGuard, Var, backward};
//!
//! let mut tape = Tape::new();
//! let _g = TapeGuard::new(&mut tape);
//!
//! // f(x) = ‖x‖² = x · x
//! let x = tape.var(&[1.0, 2.0, 3.0], (3, 1));
//! let f = x.dot(x);                         // records on the tape
//!
//! let grads = backward(&tape, f);
//! let df_dx = grads.wrt(x);                 // [2, 4, 6]
//! assert!((df_dx[0] - 2.0).abs() < 1e-12);
//! assert!((df_dx[1] - 4.0).abs() < 1e-12);
//! assert!((df_dx[2] - 6.0).abs() < 1e-12);
//! ```
//!
//! # Integration with Optimizers
//!
//! Use [`AutoDiffCostFunction`] to plug an AD-defined objective into any
//! RiemannOpt optimizer:
//!
//! ```rust,no_run
//! use riemannopt_autodiff::AutoDiffCostFunction;
//!
//! let cost_fn = AutoDiffCostFunction::new(10, |x| {
//!     // x^T A x  written with AD primitives
//!     x.dot(x)
//! });
//! // `cost_fn` implements `CostFunction<f64>` — pass it to any optimizer.
//! ```

pub mod backward;
pub mod cost_function;
pub mod tape;
pub mod var;

pub use backward::{backward, check_gradient, Gradients};
pub use cost_function::{AutoDiffCostFunction, AutoDiffMatCostFunction};
pub use tape::{NodeIdx, OpCode, Tape, TapeEntry, TapeGuard};
pub use var::Var;
