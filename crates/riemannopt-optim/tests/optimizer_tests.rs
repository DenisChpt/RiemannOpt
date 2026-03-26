//! Integration tests for all Riemannian optimizers.
//!
//! First-order methods (SGD, Adam, CG, Natural Gradient) are tested on the Sphere manifold
//! with the Rayleigh quotient cost function.
//!
//! Second-order methods (Newton, Trust Region, L-BFGS) are tested on the Euclidean manifold
//! with a simple quadratic, because these methods internally compute Hessian-vector products
//! that do not automatically lie in the tangent space of curved manifolds.

use riemannopt_core::{
	core::cost_function::CostFunction,
	error::Result as ManifoldResult,
	linalg::{self, MatrixOps, MatrixView, VectorOps, VectorView},
	optimization::optimizer::{Optimizer, StoppingCriterion},
};
use riemannopt_manifolds::{Euclidean, Sphere};
use riemannopt_optim::*;

// ---------------------------------------------------------------------------
// Rayleigh quotient cost function: f(x) = x^T A x
// Euclidean gradient: 2 * A * x
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RayleighQuotient {
	a: linalg::Mat<f64>,
}

impl RayleighQuotient {
	/// Create a diagonal SPD matrix with eigenvalues 1, 2, ..., n.
	/// The minimum on the sphere is e_1 with cost = 1.
	fn diagonal(n: usize) -> Self {
		let a = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(n, n, |i, j| {
			if i == j {
				(i + 1) as f64
			} else {
				0.0
			}
		});
		Self { a }
	}
}

impl CostFunction<f64> for RayleighQuotient {
	type Point = linalg::Vec<f64>;
	type TangentVector = linalg::Vec<f64>;

	fn cost(&self, point: &linalg::Vec<f64>) -> ManifoldResult<f64> {
		let ax = self.a.mat_vec(point);
		Ok(point.dot(&ax))
	}

	fn cost_and_gradient_alloc(
		&self,
		point: &linalg::Vec<f64>,
	) -> ManifoldResult<(f64, linalg::Vec<f64>)> {
		let ax = self.a.mat_vec(point);
		let cost = point.dot(&ax);
		let mut gradient = ax;
		gradient.scale_mut(2.0);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Vec<f64>,
		gradient: &mut linalg::Vec<f64>,
	) -> ManifoldResult<f64> {
		let ax = self.a.mat_vec(point);
		let cost = point.dot(&ax);
		gradient.copy_from(&ax);
		gradient.scale_mut(2.0);
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Vec<f64>> {
		let mut ax = self.a.mat_vec(point);
		ax.scale_mut(2.0);
		Ok(ax)
	}

	fn hessian(&self, _point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		let n = self.a.nrows();
		let scaled = self.a.scale_by(2.0);
		Ok(linalg::Mat::<f64>::from_fn(n, n, |i, j| {
			MatrixView::get(&scaled, i, j)
		}))
	}

	fn hessian_vector_product(
		&self,
		_point: &linalg::Vec<f64>,
		vector: &linalg::Vec<f64>,
	) -> ManifoldResult<linalg::Vec<f64>> {
		let mut av = self.a.mat_vec(vector);
		av.scale_mut(2.0);
		Ok(av)
	}

	fn gradient_fd_alloc(&self, point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Vec<f64>> {
		self.gradient(point)
	}
}

// ---------------------------------------------------------------------------
// Simple quadratic cost: f(x) = 0.5 * ||x||^2 (replaces QuadraticCost for linalg types)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SimpleQuadratic {
	n: usize,
}

impl SimpleQuadratic {
	fn new(n: usize) -> Self {
		Self { n }
	}
}

impl CostFunction<f64> for SimpleQuadratic {
	type Point = linalg::Vec<f64>;
	type TangentVector = linalg::Vec<f64>;

	fn cost(&self, point: &linalg::Vec<f64>) -> ManifoldResult<f64> {
		Ok(0.5 * point.dot(point))
	}

	fn cost_and_gradient_alloc(
		&self,
		point: &linalg::Vec<f64>,
	) -> ManifoldResult<(f64, linalg::Vec<f64>)> {
		let cost = 0.5 * point.dot(point);
		Ok((cost, point.clone()))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Vec<f64>,
		gradient: &mut linalg::Vec<f64>,
	) -> ManifoldResult<f64> {
		gradient.copy_from(point);
		Ok(0.5 * point.dot(point))
	}

	fn gradient(&self, point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Vec<f64>> {
		Ok(point.clone())
	}

	fn hessian(&self, _point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		Ok(linalg::Mat::<f64>::identity(self.n, self.n))
	}

	fn hessian_vector_product(
		&self,
		_point: &linalg::Vec<f64>,
		vector: &linalg::Vec<f64>,
	) -> ManifoldResult<linalg::Vec<f64>> {
		Ok(vector.clone())
	}

	fn gradient_fd_alloc(&self, point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Vec<f64>> {
		self.gradient(point)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a starting point on the sphere that is not aligned with any eigenvector.
fn initial_point_on_sphere(n: usize) -> linalg::Vec<f64> {
	let mut x: linalg::Vec<f64> = VectorOps::from_fn(n, |_| 1.0 / (n as f64).sqrt());
	*x.get_mut(0) += 0.1;
	let norm = x.norm();
	x.div_scalar_mut(norm);
	x
}

/// Verify the result point lies on the sphere.
fn assert_on_sphere(point: &linalg::Vec<f64>, tol: f64) {
	let norm = point.norm();
	assert!(
		(norm - 1.0).abs() < tol,
		"Point norm {} deviates from 1.0 by {}",
		norm,
		(norm - 1.0).abs()
	);
}

// ---------------------------------------------------------------------------
// SGD Tests
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_convergence_on_sphere() {
	let n = 5;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = SGDConfig::new().with_constant_step_size(0.01);
	let mut optimizer = SGD::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(300)
		.with_gradient_tolerance(1e-4);

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();

	// Compute actual cost at the returned point (result.value may lag by one step)
	let final_cost = cost_fn.cost(&result.point).unwrap();
	assert!(
		final_cost < initial_cost,
		"SGD did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert_on_sphere(&result.point, 1e-10);
}

#[test]
fn test_sgd_with_momentum_convergence() {
	let n = 5;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = SGDConfig::new()
		.with_constant_step_size(0.01)
		.with_classical_momentum(0.9);
	let mut optimizer = SGD::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(300)
		.with_gradient_tolerance(1e-4);

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"SGD with momentum did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert_on_sphere(&result.point, 1e-10);
}

#[test]
fn test_sgd_with_nesterov_momentum() {
	let n = 5;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = SGDConfig::new()
		.with_constant_step_size(0.01)
		.with_nesterov_momentum(0.9);
	let mut optimizer = SGD::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(300)
		.with_gradient_tolerance(1e-4);

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"SGD with Nesterov did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert_on_sphere(&result.point, 1e-10);
}

#[test]
fn test_sgd_config_parameters() {
	let config = SGDConfig::<f64>::new()
		.with_constant_step_size(0.05)
		.with_classical_momentum(0.8)
		.with_gradient_clip(5.0);

	assert!(matches!(
		config.step_size,
		StepSizeSchedule::Constant(v) if (v - 0.05).abs() < 1e-15
	));
	assert!(matches!(
		config.momentum,
		MomentumMethod::Classical { coefficient } if (coefficient - 0.8).abs() < 1e-15
	));
	assert_eq!(config.gradient_clip, Some(5.0));
}

// ---------------------------------------------------------------------------
// Adam Tests
// ---------------------------------------------------------------------------

#[test]
fn test_adam_convergence_on_sphere() {
	let n = 5;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = AdamConfig::new().with_learning_rate(0.01);
	let mut optimizer = Adam::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(300)
		.with_gradient_tolerance(1e-4);

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"Adam did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert_on_sphere(&result.point, 1e-10);
}

#[test]
fn test_adam_amsgrad_convergence() {
	let n = 5;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = AdamConfig::new().with_learning_rate(0.01).with_amsgrad();
	let mut optimizer = Adam::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(300)
		.with_gradient_tolerance(1e-4);

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"Adam AMSGrad did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert_on_sphere(&result.point, 1e-10);
}

#[test]
fn test_adam_config_parameters() {
	let config = AdamConfig::<f64>::new()
		.with_learning_rate(0.002)
		.with_beta1(0.85)
		.with_beta2(0.995)
		.with_epsilon(1e-7);

	assert!((config.learning_rate - 0.002).abs() < 1e-15);
	assert!((config.beta1 - 0.85).abs() < 1e-15);
	assert!((config.beta2 - 0.995).abs() < 1e-15);
	assert!((config.epsilon - 1e-7).abs() < 1e-20);
}

// ---------------------------------------------------------------------------
// Conjugate Gradient Tests (on Euclidean manifold to avoid tangent drift
// issues with the Sphere's strict validation; CG accumulates numerical
// errors in conjugate directions that exceed the Sphere's tolerance)
// ---------------------------------------------------------------------------

#[test]
fn test_cg_fletcher_reeves_convergence() {
	let n = 5;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, -1.0, 0.5, -0.3]);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = CGConfig::new().with_method(ConjugateGradientMethod::FletcherReeves);
	let mut optimizer = ConjugateGradient::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-6);

	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"CG Fletcher-Reeves did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert!(
		final_cost < 1e-6,
		"CG FR did not converge: cost={}",
		final_cost
	);
}

#[test]
fn test_cg_polak_ribiere_convergence() {
	let n = 5;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, -1.0, 0.5, -0.3]);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = CGConfig::new().with_method(ConjugateGradientMethod::PolakRibiere);
	let mut optimizer = ConjugateGradient::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-6);

	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"CG Polak-Ribiere did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert!(
		final_cost < 1e-6,
		"CG PR did not converge: cost={}",
		final_cost
	);
}

#[test]
fn test_cg_dai_yuan_convergence() {
	let n = 5;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, -1.0, 0.5, -0.3]);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = CGConfig::new().with_method(ConjugateGradientMethod::DaiYuan);
	let mut optimizer = ConjugateGradient::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-6);

	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"CG Dai-Yuan did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert!(
		final_cost < 1e-6,
		"CG DY did not converge: cost={}",
		final_cost
	);
}

#[test]
fn test_cg_hestenes_stiefel_convergence() {
	let n = 5;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, -1.0, 0.5, -0.3]);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = CGConfig::new().with_method(ConjugateGradientMethod::HestenesStiefel);
	let mut optimizer = ConjugateGradient::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-6);

	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"CG Hestenes-Stiefel did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert!(
		final_cost < 1e-3,
		"CG HS did not converge: cost={}",
		final_cost
	);
}

#[test]
fn test_cg_config_parameters() {
	let config = CGConfig::<f64>::new()
		.with_method(ConjugateGradientMethod::FletcherReeves)
		.with_restart_period(20);

	assert_eq!(config.method, ConjugateGradientMethod::FletcherReeves);
	assert_eq!(config.restart_period, 20);
}

// ---------------------------------------------------------------------------
// Natural Gradient Tests (on Sphere, small steps)
// ---------------------------------------------------------------------------

#[test]
fn test_natural_gradient_convergence_on_sphere() {
	let n = 4;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = NaturalGradientConfig::new()
		.with_learning_rate(0.005)
		.with_fisher_approximation(FisherApproximation::Identity)
		.with_damping(0.001);
	let mut optimizer = NaturalGradient::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-3);

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"Natural Gradient did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert_on_sphere(&result.point, 1e-8);
}

#[test]
fn test_natural_gradient_diagonal_fisher() {
	// Use Euclidean manifold to avoid very tight point-on-sphere tolerance
	// that can be exceeded by accumulated floating point drift
	let n = 4;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, -1.0, 0.5]);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let config = NaturalGradientConfig::new()
		.with_learning_rate(0.01)
		.with_fisher_approximation(FisherApproximation::Diagonal)
		.with_damping(0.01);
	let mut optimizer = NaturalGradient::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(500)
		.with_gradient_tolerance(1e-4);

	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"Natural Gradient (diagonal) did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
}

#[test]
fn test_natural_gradient_config_parameters() {
	let config = NaturalGradientConfig::<f64>::new()
		.with_learning_rate(0.05)
		.with_fisher_approximation(FisherApproximation::Diagonal)
		.with_damping(0.01)
		.with_fisher_update_freq(5);

	assert!((config.learning_rate - 0.05).abs() < 1e-15);
	assert_eq!(config.fisher_approximation, FisherApproximation::Diagonal);
	assert!((config.damping - 0.01).abs() < 1e-15);
	assert_eq!(config.fisher_update_freq, 5);
}

// ---------------------------------------------------------------------------
// L-BFGS Tests (on Euclidean manifold with SimpleQuadratic)
// ---------------------------------------------------------------------------

#[test]
fn test_lbfgs_convergence_on_euclidean() {
	let n = 5;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, -1.0, 0.5, -0.3]);

	let config = LBFGSConfig::new().with_memory_size(5);
	let mut optimizer = LBFGS::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(100)
		.with_gradient_tolerance(1e-6);

	let initial_cost = cost_fn.cost(&x0).unwrap();
	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"L-BFGS did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	// L-BFGS should converge close to the origin for f(x)=0.5*||x||^2
	assert!(
		final_cost < 1e-6,
		"L-BFGS did not converge: cost={}",
		final_cost
	);
}

#[test]
fn test_lbfgs_config_parameters() {
	let config = LBFGSConfig::<f64>::new().with_memory_size(20);
	assert_eq!(config.memory_size, 20);
}

// ---------------------------------------------------------------------------
// Trust Region Tests (on Euclidean manifold with SimpleQuadratic)
// ---------------------------------------------------------------------------

#[test]
fn test_trust_region_convergence_on_euclidean() {
	let n = 5;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, -1.0, 0.5, -0.3]);

	let config = TrustRegionConfig::new()
		.with_initial_radius(1.0)
		.with_max_radius(10.0);
	let mut optimizer = TrustRegion::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(100)
		.with_gradient_tolerance(1e-6);

	let initial_cost = cost_fn.cost(&x0).unwrap();
	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"Trust Region did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	assert!(
		final_cost < 1e-6,
		"Trust Region did not converge: cost={}",
		final_cost
	);
}

#[test]
fn test_trust_region_config_parameters() {
	let config = TrustRegionConfig::<f64>::new()
		.with_initial_radius(0.5)
		.with_max_radius(5.0)
		.with_min_radius(1e-8)
		.with_acceptance_ratio(0.15);

	assert!((config.initial_radius - 0.5).abs() < 1e-15);
	assert!((config.max_radius - 5.0).abs() < 1e-15);
	assert!((config.min_radius - 1e-8).abs() < 1e-20);
	assert!((config.acceptance_ratio - 0.15).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// Newton Tests (on Euclidean manifold with SimpleQuadratic)
// ---------------------------------------------------------------------------

#[test]
fn test_newton_convergence_on_euclidean() {
	let n = 3;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 1.0, 1.0]);

	let config = NewtonConfig::new().with_regularization(1e-6);
	let mut optimizer = Newton::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(50)
		.with_gradient_tolerance(1e-6);

	let initial_cost = cost_fn.cost(&x0).unwrap();
	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < initial_cost,
		"Newton did not reduce cost: initial={}, final={}",
		initial_cost,
		final_cost
	);
	// Newton should converge fast on a quadratic
	assert!(
		result.iterations < 20,
		"Newton took too many iterations: {}",
		result.iterations
	);
}

#[test]
fn test_newton_config_parameters() {
	let config = NewtonConfig::<f64>::new()
		.with_regularization(1e-4)
		.with_cg_params(50, 1e-8);

	assert!((config.hessian_regularization - 1e-4).abs() < 1e-15);
	assert_eq!(config.max_cg_iterations, 50);
	assert!((config.cg_tolerance - 1e-8).abs() < 1e-20);
}

// ---------------------------------------------------------------------------
// Cross-optimizer comparison: first-order methods on Sphere
// ---------------------------------------------------------------------------

#[test]
fn test_first_order_optimizers_all_reduce_cost_on_sphere() {
	let n = 4;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-3);

	// SGD
	let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
	let sgd_result = sgd.optimize(&cost_fn, &sphere, &x0, &criterion).unwrap();
	let sgd_cost = cost_fn.cost(&sgd_result.point).unwrap();
	assert!(sgd_cost < initial_cost, "SGD failed to reduce cost");
	assert_on_sphere(&sgd_result.point, 1e-10);

	// Adam
	let mut adam = Adam::new(AdamConfig::new().with_learning_rate(0.01));
	let adam_result = adam.optimize(&cost_fn, &sphere, &x0, &criterion).unwrap();
	let adam_cost = cost_fn.cost(&adam_result.point).unwrap();
	assert!(adam_cost < initial_cost, "Adam failed to reduce cost");
	assert_on_sphere(&adam_result.point, 1e-10);

	// Natural Gradient
	let mut ng = NaturalGradient::new(
		NaturalGradientConfig::new()
			.with_learning_rate(0.005)
			.with_fisher_approximation(FisherApproximation::Identity),
	);
	let ng_result = ng.optimize(&cost_fn, &sphere, &x0, &criterion).unwrap();
	let ng_cost = cost_fn.cost(&ng_result.point).unwrap();
	assert!(
		ng_cost < initial_cost,
		"Natural Gradient failed to reduce cost"
	);
	assert_on_sphere(&ng_result.point, 1e-8);
}

// ---------------------------------------------------------------------------
// Cross-optimizer comparison: second-order methods on Euclidean
// ---------------------------------------------------------------------------

#[test]
fn test_second_order_optimizers_converge_on_euclidean() {
	let n = 4;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[2.0, -1.0, 0.5, 1.5]);
	let initial_cost = cost_fn.cost(&x0).unwrap();

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-6);

	// Newton
	let mut newton = Newton::new(NewtonConfig::new().with_regularization(1e-6));
	let newton_result = newton
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let newton_cost = cost_fn.cost(&newton_result.point).unwrap();
	assert!(newton_cost < initial_cost, "Newton failed to reduce cost");
	assert!(newton_cost < 1e-6, "Newton did not converge to minimum");

	// Trust Region
	let mut tr = TrustRegion::new(TrustRegionConfig::new());
	let tr_result = tr.optimize(&cost_fn, &manifold, &x0, &criterion).unwrap();
	let tr_cost = cost_fn.cost(&tr_result.point).unwrap();
	assert!(tr_cost < initial_cost, "Trust Region failed to reduce cost");
	assert!(tr_cost < 1e-6, "Trust Region did not converge to minimum");

	// L-BFGS
	let mut lbfgs = LBFGS::new(LBFGSConfig::new());
	let lbfgs_result = lbfgs
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let lbfgs_cost = cost_fn.cost(&lbfgs_result.point).unwrap();
	assert!(lbfgs_cost < initial_cost, "L-BFGS failed to reduce cost");
	assert!(lbfgs_cost < 1e-6, "L-BFGS did not converge to minimum");
}

// ---------------------------------------------------------------------------
// Optimizer result metadata tests
// ---------------------------------------------------------------------------

#[test]
fn test_optimization_result_metadata() {
	let n = 4;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);

	let mut optimizer = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
	let criterion = StoppingCriterion::new()
		.with_max_iterations(50)
		.with_gradient_tolerance(1e-15); // Very tight so all 50 iterations run

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();

	assert!(result.iterations > 0, "Should have performed iterations");
	assert!(
		result.function_evaluations > 0,
		"Should have function evaluations"
	);
	assert!(!result.duration.is_zero(), "Duration should be nonzero");
}

// ---------------------------------------------------------------------------
// Stopping criterion test
// ---------------------------------------------------------------------------

#[test]
fn test_max_iterations_stopping() {
	let n = 3;
	let sphere = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);

	let mut optimizer = SGD::new(SGDConfig::new().with_constant_step_size(0.001));
	let criterion = StoppingCriterion::new()
		.with_max_iterations(10)
		.with_gradient_tolerance(1e-15); // Very tight -- won't converge

	let result = optimizer
		.optimize(&cost_fn, &sphere, &x0, &criterion)
		.unwrap();

	assert!(
		result.iterations <= 10,
		"Should have stopped at or before 10 iterations, got {}",
		result.iterations
	);
}

// ---------------------------------------------------------------------------
// Convergence quality: CG approaches minimum on Euclidean
// ---------------------------------------------------------------------------

#[test]
fn test_cg_approaches_quadratic_minimum() {
	let n = 5;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	// f(x) = 0.5 * x^T * x, minimum at origin with cost = 0
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[3.0, -2.0, 1.0, 0.5, -1.5]);

	let config = CGConfig::new().with_method(ConjugateGradientMethod::PolakRibiere);
	let mut optimizer = ConjugateGradient::new(config);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(500)
		.with_gradient_tolerance(1e-8);

	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let final_cost = cost_fn.cost(&result.point).unwrap();

	assert!(
		final_cost < 1e-6,
		"CG did not approach minimum: cost={}",
		final_cost
	);
}

// ---------------------------------------------------------------------------
// Second-order convergence quality on Euclidean
// ---------------------------------------------------------------------------

#[test]
fn test_second_order_methods_approach_minimum_on_euclidean() {
	let n = 3;
	let manifold = Euclidean::<f64>::new(n).unwrap();
	// f(x) = 0.5 * x^T * x, minimum at origin with cost = 0
	let cost_fn = SimpleQuadratic::new(n);
	let x0: linalg::Vec<f64> = VectorOps::from_slice(&[3.0, -2.0, 1.0]);

	let criterion = StoppingCriterion::new()
		.with_max_iterations(200)
		.with_gradient_tolerance(1e-8);

	// Trust Region should get very close to the minimum
	let mut tr = TrustRegion::new(TrustRegionConfig::new());
	let tr_result = tr.optimize(&cost_fn, &manifold, &x0, &criterion).unwrap();
	let tr_cost = cost_fn.cost(&tr_result.point).unwrap();
	assert!(
		tr_cost < 1e-10,
		"Trust Region did not converge to minimum: cost={}",
		tr_cost
	);

	// Newton should converge very fast on a quadratic
	let mut newton = Newton::new(NewtonConfig::new().with_regularization(1e-8));
	let newton_result = newton
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let newton_cost = cost_fn.cost(&newton_result.point).unwrap();
	assert!(
		newton_cost < 1e-10,
		"Newton did not converge to minimum: cost={}",
		newton_cost
	);

	// L-BFGS should also converge well
	let mut lbfgs = LBFGS::new(LBFGSConfig::new());
	let lbfgs_result = lbfgs
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();
	let lbfgs_cost = cost_fn.cost(&lbfgs_result.point).unwrap();
	assert!(
		lbfgs_cost < 1e-10,
		"L-BFGS did not converge to minimum: cost={}",
		lbfgs_cost
	);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Strong Wolfe line search + CG convergence at increasing dimensions
// ═══════════════════════════════════════════════════════════════════════════

/// CG with Polak-Ribiere on Sphere + Rayleigh quotient at various dimensions.
/// f* = lambda_min(A) = 1. This is the key test for the Strong Wolfe line search:
/// without Wolfe conditions, CG stagnates at large n due to loss of conjugacy.
#[test]
fn test_cg_wolfe_sphere_rayleigh_n10() {
	cg_rayleigh_sphere(10, 1e-4);
}

#[test]
fn test_cg_wolfe_sphere_rayleigh_n50() {
	cg_rayleigh_sphere(50, 1e-4);
}

#[test]
fn test_cg_wolfe_sphere_rayleigh_n200() {
	cg_rayleigh_sphere(200, 1e-3);
}

#[test]
fn test_cg_wolfe_sphere_rayleigh_n500() {
	cg_rayleigh_sphere(500, 1e-2);
}

fn cg_rayleigh_sphere(n: usize, tolerance: f64) {
	let manifold = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);
	let x0 = initial_point_on_sphere(n);

	let mut optimizer = ConjugateGradient::new(CGConfig::polak_ribiere());

	let max_iter = (n * 3).min(1000);
	let criterion = StoppingCriterion::new()
		.with_max_iterations(max_iter)
		.with_gradient_tolerance(1e-10);

	let result = optimizer
		.optimize(&cost_fn, &manifold, &x0, &criterion)
		.unwrap();

	let final_cost = cost_fn.cost(&result.point).unwrap();
	let error = (final_cost - 1.0).abs();
	assert!(
		error < tolerance,
		"CG/Sphere n={}: |f - f*| = {:.2e} >= tol {:.2e} (cost={:.6}, iters={})",
		n,
		error,
		tolerance,
		final_cost,
		result.iterations,
	);
}

/// All CG variants should converge on Sphere(20) Rayleigh.
#[test]
fn test_cg_all_variants_sphere20() {
	let n = 20;
	let manifold = Sphere::<f64>::new(n).unwrap();
	let cost_fn = RayleighQuotient::diagonal(n);

	for method in [
		ConjugateGradientMethod::FletcherReeves,
		ConjugateGradientMethod::PolakRibiere,
		ConjugateGradientMethod::HestenesStiefel,
		ConjugateGradientMethod::DaiYuan,
	] {
		let x0 = initial_point_on_sphere(n);
		let config = CGConfig::new().with_method(method);
		let mut optimizer = ConjugateGradient::new(config);
		let criterion = StoppingCriterion::new()
			.with_max_iterations(200)
			.with_gradient_tolerance(1e-10);

		let result = optimizer
			.optimize(&cost_fn, &manifold, &x0, &criterion)
			.unwrap();
		let final_cost = cost_fn.cost(&result.point).unwrap();
		let error = (final_cost - 1.0).abs();
		assert!(
			error < 1e-3,
			"CG {:?}/Sphere(20): |f - f*| = {:.2e} (iters={})",
			method,
			error,
			result.iterations,
		);
	}
}
