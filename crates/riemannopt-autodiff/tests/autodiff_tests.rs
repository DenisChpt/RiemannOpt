//! Comprehensive tests for the tape-based autodiff engine.
//!
//! Every operation is verified by comparing the AD gradient against a
//! central-difference finite-difference approximation.

use riemannopt_autodiff::{backward, check_gradient, Tape, TapeGuard, Var};

const EPS: f64 = 1e-6;
const GRAD_TOL: f64 = 1e-4;

// ── helper ────────────────────────────────────────────────────────────────

fn assert_grad_ok<F: Fn(Var) -> Var>(f: F, x: &[f64], shape: (usize, usize)) {
	let err = check_gradient(f, x, shape, EPS);
	assert!(
		err < GRAD_TOL,
		"gradient check failed: max relative error = {err:.2e}"
	);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Basic arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_add() {
	assert_grad_ok(|x| (x + x).sum(), &[2.0, 3.0], (2, 1));
}

#[test]
fn test_sub() {
	let mut tape = Tape::new();
	let _g = TapeGuard::new(&mut tape);
	let a = tape.var(&[5.0], (1, 1));
	let b = tape.var(&[3.0], (1, 1));
	let c = a - b;
	assert!((tape.scalar(c.idx()) - 2.0).abs() < 1e-14);
	let grads = backward(&tape, c);
	assert!((grads.wrt(a)[0] - 1.0).abs() < 1e-14); // dc/da = 1
	assert!((grads.wrt(b)[0] - (-1.0)).abs() < 1e-14); // dc/db = -1
}

#[test]
fn test_mul_elementwise() {
	assert_grad_ok(|x| (x * x).sum(), &[2.0, 3.0, 4.0], (3, 1));
}

#[test]
fn test_div() {
	// Manual div test
	let mut tape = Tape::new();
	let _g = TapeGuard::new(&mut tape);
	let a = tape.var(&[6.0], (1, 1));
	let b = tape.var(&[3.0], (1, 1));
	let c = a / b;
	assert!((tape.scalar(c.idx()) - 2.0).abs() < 1e-14);
	let grads = backward(&tape, c);
	assert!((grads.wrt(a)[0] - 1.0 / 3.0).abs() < 1e-12);
	assert!((grads.wrt(b)[0] - (-6.0 / 9.0)).abs() < 1e-12);
}

#[test]
fn test_neg() {
	assert_grad_ok(|x| (-x).sum(), &[1.0, 2.0, 3.0], (3, 1));
}

#[test]
fn test_scalar_mul() {
	assert_grad_ok(|x| (x * 3.0).sum(), &[1.0, 2.0], (2, 1));
}

#[test]
fn test_scalar_add() {
	assert_grad_ok(|x| (x + 5.0).sum(), &[1.0, 2.0], (2, 1));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Math functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_exp() {
	assert_grad_ok(|x| x.exp().sum(), &[0.5, 1.0, -0.5], (3, 1));
}

#[test]
fn test_log() {
	assert_grad_ok(|x| x.log().sum(), &[1.0, 2.0, 3.0], (3, 1));
}

#[test]
fn test_sqrt() {
	assert_grad_ok(|x| x.sqrt().sum(), &[1.0, 4.0, 9.0], (3, 1));
}

#[test]
fn test_sin() {
	assert_grad_ok(|x| x.sin().sum(), &[0.5, 1.0, 2.0], (3, 1));
}

#[test]
fn test_cos() {
	assert_grad_ok(|x| x.cos().sum(), &[0.5, 1.0, 2.0], (3, 1));
}

#[test]
fn test_abs() {
	assert_grad_ok(|x| x.abs().sum(), &[1.0, -2.0, 3.0], (3, 1));
}

#[test]
fn test_pow() {
	assert_grad_ok(|x| x.powi(3).sum(), &[1.0, 2.0, 0.5], (3, 1));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Reductions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sum() {
	let mut tape = Tape::new();
	let _g = TapeGuard::new(&mut tape);
	let x = tape.var(&[1.0, 2.0, 3.0], (3, 1));
	let s = x.sum();
	assert!((tape.scalar(s.idx()) - 6.0).abs() < 1e-14);
	let grads = backward(&tape, s);
	for &g in grads.wrt(x) {
		assert!((g - 1.0).abs() < 1e-14);
	}
}

#[test]
fn test_mean() {
	assert_grad_ok(|x| x.mean(), &[2.0, 4.0, 6.0], (3, 1));
}

#[test]
fn test_dot() {
	assert_grad_ok(|x| x.dot(x), &[1.0, 2.0, 3.0], (3, 1));
	// Manual check: d/dx (x·x) = 2x
	let mut tape = Tape::new();
	let _g = TapeGuard::new(&mut tape);
	let x = tape.var(&[1.0, 2.0, 3.0], (3, 1));
	let d = x.dot(x);
	assert!((tape.scalar(d.idx()) - 14.0).abs() < 1e-14);
	let grads = backward(&tape, d);
	let g = grads.wrt(x);
	assert!((g[0] - 2.0).abs() < 1e-14);
	assert!((g[1] - 4.0).abs() < 1e-14);
	assert!((g[2] - 6.0).abs() < 1e-14);
}

#[test]
fn test_norm() {
	assert_grad_ok(|x| x.norm(), &[3.0, 4.0], (2, 1));
	// ||[3,4]|| = 5, grad = [3/5, 4/5]
	let mut tape = Tape::new();
	let _g = TapeGuard::new(&mut tape);
	let x = tape.var(&[3.0, 4.0], (2, 1));
	let n = x.norm();
	assert!((tape.scalar(n.idx()) - 5.0).abs() < 1e-14);
	let grads = backward(&tape, n);
	assert!((grads.wrt(x)[0] - 0.6).abs() < 1e-12);
	assert!((grads.wrt(x)[1] - 0.8).abs() < 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Linear algebra
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_trace() {
	// tr([[1,2],[3,4]]) = 5, grad = I
	assert_grad_ok(|x| x.trace(), &[1.0, 3.0, 2.0, 4.0], (2, 2)); // col-major
}

#[test]
fn test_matmul_gradient() {
	// f(X) = sum(X @ Y) where Y is constant
	// Use a 2×3 × 3×1 matmul
	let y_data = [1.0, 2.0, 3.0]; // 3×1
	assert_grad_ok(
		move |x| {
			// Create a constant Y on the active tape
			crate::with_tape_for_test(|tape| {
				let y = tape.constant(&y_data, (3, 1));
				x.matmul(y).sum()
			})
		},
		&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], // 2×3 col-major
		(2, 3),
	);
}

// helper for creating constants in tests
mod crate_helper {
	use super::*;
	pub fn with_tape_for_test<F, R>(f: F) -> R
	where
		F: FnOnce(&mut Tape) -> R,
	{
		riemannopt_autodiff::tape::with_tape(f)
	}
}
use crate_helper::with_tape_for_test;

// ═══════════════════════════════════════════════════════════════════════════
//  Composite expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_rayleigh_quotient() {
	// f(x) = x^T A x  where A = diag(1,2,3)
	// grad = 2*A*x = [2, 8, 18]
	let a = [1.0, 2.0, 3.0];
	assert_grad_ok(
		move |x| {
			// x * a * x summed (element-wise multiply then dot)
			with_tape_for_test(|tape| {
				let diag = tape.constant(&a, (3, 1));
				(x * diag * x).sum() // sum of x_i * a_i * x_i
			})
		},
		&[1.0, 2.0, 3.0],
		(3, 1),
	);
}

#[test]
fn test_chain_rule() {
	// f(x) = exp(sum(x^2))
	assert_grad_ok(|x| (x * x).sum().exp(), &[1.0, 2.0], (2, 1));
}

#[test]
fn test_complex_expression() {
	// f(x) = log(1 + exp(x·x))  (softplus of squared norm)
	assert_grad_ok(|x| (x.dot(x).exp() + 1.0).log(), &[0.5, 0.3, -0.2], (3, 1));
}

// ═══════════════════════════════════════════════════════════════════════════
//  CostFunction integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_autodiff_cost_function() {
	use riemannopt_autodiff::AutoDiffCostFunction;
	use riemannopt_core::cost_function::CostFunction;
	use riemannopt_core::linalg::VectorOps;

	// f(x) = ||x||^2
	let cf = AutoDiffCostFunction::new(3, |x| x.dot(x));
	let point = VectorOps::from_slice(&[1.0, 2.0, 3.0]);

	let cost = cf.cost(&point).unwrap();
	assert!((cost - 14.0).abs() < 1e-12);

	let (c, g) = cf.cost_and_gradient_alloc(&point).unwrap();
	assert!((c - 14.0).abs() < 1e-12);
	assert!((VectorOps::get(&g, 0) - 2.0).abs() < 1e-12);
	assert!((VectorOps::get(&g, 1) - 4.0).abs() < 1e-12);
	assert!((VectorOps::get(&g, 2) - 6.0).abs() < 1e-12);
}

#[test]
fn test_autodiff_with_optimizer() {
	use riemannopt_autodiff::AutoDiffCostFunction;
	use riemannopt_core::{
		cost_function::CostFunction,
		linalg::VectorOps,
		optimization::optimizer::{Optimizer, StoppingCriterion},
	};
	use riemannopt_manifolds::Euclidean;
	use riemannopt_optim::ConjugateGradient;

	// min ||x||^2 on R^5 — solution is x=0
	let cf = AutoDiffCostFunction::new(5, |x| x.dot(x));
	let eucl = Euclidean::<f64>::new(5).unwrap();
	let x0 = VectorOps::from_fn(5, |_| 1.0);

	let mut opt = ConjugateGradient::with_default_config();
	let crit = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-10);

	let result = opt.optimize(&cf, &eucl, &x0, &crit).unwrap();
	assert!(
		cf.cost(&result.point).unwrap() < 1e-6,
		"AutoDiff CG: cost={:.2e}, iterations={}",
		cf.cost(&result.point).unwrap(),
		result.iterations
	);
}

#[test]
fn test_autodiff_mat_cost_function() {
	use riemannopt_autodiff::AutoDiffMatCostFunction;
	use riemannopt_core::cost_function::CostFunction;
	use riemannopt_core::linalg::{self, MatrixOps};

	// f(X) = tr(X^T X) = ||X||_F^2
	let _cf = AutoDiffMatCostFunction::new(3, 2, |x| x.matmul(x).trace());

	// Actually need X^T X but matmul(x,x) requires compatible dims.
	// Use sum of squares instead: f(X) = sum(X*X)
	let cf2 = AutoDiffMatCostFunction::new(3, 2, |x| (x * x).sum());

	let point = <linalg::Mat<f64> as MatrixOps<f64>>::from_column_slice(
		3,
		2,
		&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
	);
	let cost = cf2.cost(&point).unwrap();
	assert!((cost - 91.0).abs() < 1e-12); // 1+4+9+16+25+36

	let (c, g) = cf2.cost_and_gradient_alloc(&point).unwrap();
	assert!((c - 91.0).abs() < 1e-12);
	// grad of sum(X*X) = 2X
	assert!((MatrixOps::get(&g, 0, 0) - 2.0).abs() < 1e-12);
	assert!((MatrixOps::get(&g, 2, 1) - 12.0).abs() < 1e-12);
}
