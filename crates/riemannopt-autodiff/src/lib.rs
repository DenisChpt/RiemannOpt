//! Tape-based reverse-mode automatic differentiation for Riemannian
//! optimization.
//!
//! # Architecture
//!
//! Values are stored as backend-native types (`faer::Col<T>`, `faer::Mat<T>`)
//! enabling SIMD/BLAS operations without intermediate copies. A buffer pool
//! recycles allocations across evaluations — zero allocation in steady state.
//!
//! # Quick Start
//!
//! ```
//! use riemannopt_autodiff::{Tape, TapeGuard, Var, backward};
//!
//! let mut tape = Tape::<f64>::new();
//! let _g = TapeGuard::new(&mut tape);
//!
//! let x = tape.var(&[1.0_f64, 2.0, 3.0], (3, 1));
//! let f = x.dot(x);
//!
//! let grads = backward(&tape, f);
//! let df_dx = grads.wrt(x);
//! assert!((df_dx[0] - 2.0_f64).abs() < 1e-12);
//! assert!((df_dx[1] - 4.0_f64).abs() < 1e-12);
//! assert!((df_dx[2] - 6.0_f64).abs() < 1e-12);
//! ```
//!
//! # Integration with Optimizers
//!
//! ```rust,no_run
//! use riemannopt_autodiff::{AutoDiffCostFunction, Var};
//!
//! let cost_fn = AutoDiffCostFunction::new(10, |x: Var<f64>| {
//!     x.dot(x)
//! });
//! // `cost_fn` implements `CostFunction<f64>` — pass it to any optimizer.
//! ```

pub mod backward;
pub(crate) mod buffer_pool;
pub mod cost_function;
pub mod tape;
pub(crate) mod value;
pub mod var;

pub use backward::{backward, check_gradient, Gradients};
pub use cost_function::{AutoDiffCostFunction, AutoDiffMatCostFunction};
pub use tape::{NodeIdx, OpCode, Tape, TapeEntry, TapeGuard, TapeThreadLocal};
pub use var::Var;
