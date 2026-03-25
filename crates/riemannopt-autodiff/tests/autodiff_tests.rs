//! Comprehensive tests for the tape-based autodiff engine.
//!
//! Every operation is verified by comparing the AD gradient against a
//! central-difference finite-difference approximation.

use riemannopt_autodiff::{backward, check_gradient, Tape, TapeGuard, Var};

const EPS: f64 = 1e-6;
const GRAD_TOL: f64 = 1e-4;

fn assert_grad_ok<F: Fn(Var<f64>) -> Var<f64>>(f: F, x: &[f64], shape: (usize, usize)) {
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
	let mut tape = Tape::<f64>::new();
	let _g = TapeGuard::new(&mut tape);
	let a = tape.var(&[5.0], (1, 1));
	let b = tape.var(&[3.0], (1, 1));
	let c = a - b;
	assert!((tape.scalar(c.idx()) - 2.0).abs() < 1e-14);
	let grads = backward(&tape, c);
	assert!((grads.wrt(a)[0] - 1.0).abs() < 1e-14);
	assert!((grads.wrt(b)[0] - (-1.0)).abs() < 1e-14);
}

#[test]
fn test_mul_elementwise() {
	assert_grad_ok(|x| (x * x).sum(), &[2.0, 3.0, 4.0], (3, 1));
}

#[test]
fn test_div() {
	let mut tape = Tape::<f64>::new();
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
	let mut tape = Tape::<f64>::new();
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
	let mut tape = Tape::<f64>::new();
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
	let mut tape = Tape::<f64>::new();
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
	assert_grad_ok(|x| x.trace(), &[1.0, 3.0, 2.0, 4.0], (2, 2));
}

#[test]
fn test_matmul_gradient() {
	let y_data = [1.0, 2.0, 3.0];
	assert_grad_ok(
		move |x| {
			riemannopt_autodiff::tape::with_tape::<f64, _, _>(|tape| {
				let y = tape.constant(&y_data, (3, 1));
				x.matmul(y).sum()
			})
		},
		&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
		(2, 3),
	);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Composite expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_rayleigh_quotient() {
	let a = [1.0, 2.0, 3.0];
	assert_grad_ok(
		move |x| {
			riemannopt_autodiff::tape::with_tape::<f64, _, _>(|tape| {
				let diag = tape.constant(&a, (3, 1));
				(x * diag * x).sum()
			})
		},
		&[1.0, 2.0, 3.0],
		(3, 1),
	);
}

#[test]
fn test_chain_rule() {
	assert_grad_ok(|x| (x * x).sum().exp(), &[1.0, 2.0], (2, 1));
}

#[test]
fn test_complex_expression() {
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

	let cf = AutoDiffCostFunction::new(3, |x: Var<f64>| x.dot(x));
	let point = VectorOps::from_slice(&[1.0_f64, 2.0, 3.0]);

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

	let cf = AutoDiffCostFunction::new(5, |x: Var<f64>| x.dot(x));
	let eucl = Euclidean::<f64>::new(5).unwrap();
	let x0 = VectorOps::from_fn(5, |_| 1.0_f64);

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

	let cf = AutoDiffMatCostFunction::new(3, 2, |x: Var<f64>| (x * x).sum());

	let point = <linalg::Mat<f64> as MatrixOps<f64>>::from_column_slice(
		3,
		2,
		&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
	);
	let cost = cf.cost(&point).unwrap();
	assert!((cost - 91.0).abs() < 1e-12);

	let (c, g) = cf.cost_and_gradient_alloc(&point).unwrap();
	assert!((c - 91.0).abs() < 1e-12);
	assert!((MatrixOps::get(&g, 0, 0) - 2.0).abs() < 1e-12);
	assert!((MatrixOps::get(&g, 2, 1) - 12.0).abs() < 1e-12);
}
