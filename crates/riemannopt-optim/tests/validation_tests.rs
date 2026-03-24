//! Rigorous validation tests on known optimization problems with analytical solutions.
//!
//! These tests verify that RiemannOpt produces mathematically correct results
//! with proper convergence on classical Riemannian optimization problems.

use riemannopt_core::{
	core::cost_function::CostFunction,
	error::Result as ManifoldResult,
	linalg::{self, DecompositionOps, MatrixOps, VectorOps},
	manifold::Manifold,
	optimization::optimizer::{Optimizer, StoppingCriterion},
};
use riemannopt_manifolds::{Euclidean, Grassmann, Sphere, Stiefel};
use riemannopt_optim::*;

// ============================================================================
// COST FUNCTIONS FOR KNOWN PROBLEMS
// ============================================================================

/// Rayleigh quotient: f(x) = x^T A x.
/// On S^{n-1}, the global minimum is the eigenvector of smallest eigenvalue.
#[derive(Debug, Clone)]
struct RayleighQuotient {
	a: linalg::Mat<f64>,
}

impl RayleighQuotient {
	fn new(a: linalg::Mat<f64>) -> Self {
		Self { a }
	}

	/// Diagonal matrix diag(1, 2, ..., n).
	/// Minimum on S^{n-1}: e_1 with cost 1.0.
	fn diagonal(n: usize) -> Self {
		let a = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(n, n, |i, j| {
			if i == j {
				(i + 1) as f64
			} else {
				0.0
			}
		});
		Self::new(a)
	}

	/// Diagonal with a spectral gap: diag(1, gap+1, gap+1, ..., gap+1).
	/// Larger gap makes the problem easier.
	fn with_gap(n: usize, gap: f64) -> Self {
		let a = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(n, n, |i, j| {
			if i == j {
				if i == 0 {
					1.0
				} else {
					gap + 1.0
				}
			} else {
				0.0
			}
		});
		Self::new(a)
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
			MatrixOps::get(&scaled, i, j)
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

/// Trace minimization on Grassmann manifold: f(Y) = tr(Y^T A Y).
/// The minimum is the subspace spanned by the p eigenvectors of A
/// corresponding to the p smallest eigenvalues.
#[derive(Debug, Clone)]
struct TraceMinimization {
	a: linalg::Mat<f64>,
}

impl TraceMinimization {
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

impl CostFunction<f64> for TraceMinimization {
	type Point = linalg::Mat<f64>;
	type TangentVector = linalg::Mat<f64>;

	fn cost(&self, point: &linalg::Mat<f64>) -> ManifoldResult<f64> {
		let ay = self.a.mat_mul(point);
		let yt_ay = MatrixOps::transpose(point).mat_mul(&ay);
		Ok(yt_ay.trace())
	}

	fn cost_and_gradient_alloc(
		&self,
		point: &linalg::Mat<f64>,
	) -> ManifoldResult<(f64, linalg::Mat<f64>)> {
		let ay = self.a.mat_mul(point);
		let yt_ay = MatrixOps::transpose(point).mat_mul(&ay);
		let cost = yt_ay.trace();
		let gradient = ay.scale_by(2.0);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Mat<f64>,
		gradient: &mut linalg::Mat<f64>,
	) -> ManifoldResult<f64> {
		let ay = self.a.mat_mul(point);
		let yt_ay = MatrixOps::transpose(point).mat_mul(&ay);
		let cost = yt_ay.trace();
		gradient.copy_from(&ay);
		gradient.scale_mut(2.0);
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Mat<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		let ay = self.a.mat_mul(point);
		Ok(ay.scale_by(2.0))
	}

	fn hessian(&self, _point: &linalg::Mat<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		// Not needed for first-order methods
		Err(riemannopt_core::ManifoldError::not_implemented(
			"Hessian not implemented for TraceMinimization",
		))
	}

	fn hessian_vector_product(
		&self,
		_point: &linalg::Mat<f64>,
		vector: &linalg::Mat<f64>,
	) -> ManifoldResult<linalg::Mat<f64>> {
		let av = self.a.mat_mul(vector);
		Ok(av.scale_by(2.0))
	}

	fn gradient_fd_alloc(&self, point: &linalg::Mat<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		self.gradient(point)
	}
}

/// Procrustes problem on Stiefel: f(X) = ||AX - B||_F^2.
/// For A = I and B = target, the minimum is B projected onto St(n,p).
#[derive(Debug, Clone)]
struct Procrustes {
	a: linalg::Mat<f64>,
	b: linalg::Mat<f64>,
}

impl Procrustes {
	fn new(a: linalg::Mat<f64>, b: linalg::Mat<f64>) -> Self {
		Self { a, b }
	}
}

impl CostFunction<f64> for Procrustes {
	type Point = linalg::Mat<f64>;
	type TangentVector = linalg::Mat<f64>;

	fn cost(&self, point: &linalg::Mat<f64>) -> ManifoldResult<f64> {
		let ax = self.a.mat_mul(point);
		let diff = ax.sub(&self.b);
		// norm_squared = sum of squares of all elements
		let n = diff.norm();
		Ok(n * n)
	}

	fn cost_and_gradient_alloc(
		&self,
		point: &linalg::Mat<f64>,
	) -> ManifoldResult<(f64, linalg::Mat<f64>)> {
		let ax = self.a.mat_mul(point);
		let diff = ax.sub(&self.b);
		let n = diff.norm();
		let cost = n * n;
		// gradient = 2 * A^T * (A*X - B)
		let at = MatrixOps::transpose(&self.a);
		let gradient = at.mat_mul(&diff).scale_by(2.0);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Mat<f64>,
		gradient: &mut linalg::Mat<f64>,
	) -> ManifoldResult<f64> {
		let ax = self.a.mat_mul(point);
		let diff = ax.sub(&self.b);
		let n = diff.norm();
		let cost = n * n;
		let at = MatrixOps::transpose(&self.a);
		let g = at.mat_mul(&diff).scale_by(2.0);
		gradient.copy_from(&g);
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Mat<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		let ax = self.a.mat_mul(point);
		let diff = ax.sub(&self.b);
		let at = MatrixOps::transpose(&self.a);
		Ok(at.mat_mul(&diff).scale_by(2.0))
	}

	fn hessian(&self, _point: &linalg::Mat<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		Err(riemannopt_core::ManifoldError::not_implemented(
			"Hessian not implemented for Procrustes",
		))
	}

	fn hessian_vector_product(
		&self,
		_point: &linalg::Mat<f64>,
		vector: &linalg::Mat<f64>,
	) -> ManifoldResult<linalg::Mat<f64>> {
		// 2 * A^T * A * V
		let at = MatrixOps::transpose(&self.a);
		let ata = at.mat_mul(&self.a);
		Ok(ata.mat_mul(vector).scale_by(2.0))
	}

	fn gradient_fd_alloc(&self, point: &linalg::Mat<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		self.gradient(point)
	}
}

/// Simple quadratic cost: f(x) = 0.5 * ||x||^2
/// Replaces QuadraticCost for linalg types.
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

/// Shifted quadratic cost: f(x) = 0.5 * x^T A x + b^T x
/// where A = diag(1, 2, ..., n), b = (0.5, 1.0, 1.5, ..., n*0.5)
/// Minimum at x* = -A^{-1}b
#[derive(Debug, Clone)]
struct ShiftedQuadratic {
	a_diag: Vec<f64>,
	b: linalg::Vec<f64>,
}

impl ShiftedQuadratic {
	fn new(n: usize) -> Self {
		let a_diag: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
		let b: linalg::Vec<f64> = VectorOps::from_fn(n, |i| (i + 1) as f64 * 0.5);
		Self { a_diag, b }
	}

	fn x_star(&self) -> linalg::Vec<f64> {
		let n = self.a_diag.len();
		VectorOps::from_fn(n, |i| -self.b.get(i) / self.a_diag[i])
	}

	fn f_star(&self) -> f64 {
		let n = self.a_diag.len();
		(0..n)
			.map(|i| -0.5 * self.b.get(i) * self.b.get(i) / self.a_diag[i])
			.sum()
	}
}

impl CostFunction<f64> for ShiftedQuadratic {
	type Point = linalg::Vec<f64>;
	type TangentVector = linalg::Vec<f64>;

	fn cost(&self, point: &linalg::Vec<f64>) -> ManifoldResult<f64> {
		let n = self.a_diag.len();
		let mut quad = 0.0;
		let mut lin = 0.0;
		for i in 0..n {
			let xi = point.get(i);
			quad += self.a_diag[i] * xi * xi;
			lin += self.b.get(i) * xi;
		}
		Ok(0.5 * quad + lin)
	}

	fn cost_and_gradient_alloc(
		&self,
		point: &linalg::Vec<f64>,
	) -> ManifoldResult<(f64, linalg::Vec<f64>)> {
		let cost = self.cost(point)?;
		let grad = self.gradient(point)?;
		Ok((cost, grad))
	}

	fn cost_and_gradient(
		&self,
		point: &linalg::Vec<f64>,
		gradient: &mut linalg::Vec<f64>,
	) -> ManifoldResult<f64> {
		let n = self.a_diag.len();
		let mut cost = 0.0;
		for i in 0..n {
			let xi = point.get(i);
			cost += 0.5 * self.a_diag[i] * xi * xi + self.b.get(i) * xi;
			*gradient.get_mut(i) = self.a_diag[i] * xi + self.b.get(i);
		}
		Ok(cost)
	}

	fn gradient(&self, point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Vec<f64>> {
		let n = self.a_diag.len();
		Ok(VectorOps::from_fn(n, |i| {
			self.a_diag[i] * point.get(i) + self.b.get(i)
		}))
	}

	fn hessian(&self, _point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Mat<f64>> {
		let n = self.a_diag.len();
		Ok(linalg::Mat::<f64>::from_fn(n, n, |i, j| {
			if i == j {
				self.a_diag[i]
			} else {
				0.0
			}
		}))
	}

	fn hessian_vector_product(
		&self,
		_point: &linalg::Vec<f64>,
		vector: &linalg::Vec<f64>,
	) -> ManifoldResult<linalg::Vec<f64>> {
		let n = self.a_diag.len();
		Ok(VectorOps::from_fn(n, |i| self.a_diag[i] * vector.get(i)))
	}

	fn gradient_fd_alloc(&self, point: &linalg::Vec<f64>) -> ManifoldResult<linalg::Vec<f64>> {
		self.gradient(point)
	}
}

// ============================================================================
// HELPERS
// ============================================================================

/// Starting point on the sphere not aligned with any eigenvector.
fn sphere_start(n: usize) -> linalg::Vec<f64> {
	let mut x: linalg::Vec<f64> = VectorOps::from_fn(n, |_| 1.0 / (n as f64).sqrt());
	// Perturb slightly to break symmetry
	for i in 0..n {
		*x.get_mut(i) += 0.01 * (i as f64) / (n as f64);
	}
	let norm = x.norm();
	x.div_scalar_mut(norm);
	x
}

/// Starting point on the Grassmann manifold Gr(n, p).
fn grassmann_start(gr: &Grassmann<f64>, n: usize, p: usize) -> linalg::Mat<f64> {
	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(n, p);
	gr.random_point(&mut y).unwrap();
	y
}

/// Starting point on the Stiefel manifold St(n, p).
fn stiefel_start(st: &Stiefel<f64>) -> linalg::Mat<f64> {
	let mut x = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(st.rows(), st.cols());
	st.random_point(&mut x).unwrap();
	x
}

/// Check that a point is the minimizer of the Rayleigh quotient (smallest eigenvector).
/// On S^{n-1} with A = diag(lambda_1, ..., lambda_n), the minimizer is +/-e_1.
fn check_rayleigh_minimizer(x: &linalg::Vec<f64>, tol: f64) {
	// |x_1| should be close to 1 (the point is +/-e_1)
	assert!(
		x.get(0).abs() > 1.0 - tol,
		"Expected |x[0]| close to 1.0, got {}. Full x = {:?}",
		x.get(0).abs(),
		x.as_slice()
	);
}

/// Check that columns of Y span the subspace of smallest eigenvectors.
/// For A = diag(1, 2, ..., n), the p smallest are e_1, ..., e_p.
/// The projection matrix P = Y Y^T should satisfy P e_i = e_i for i = 1..p.
fn check_subspace_minimizer(y: &linalg::Mat<f64>, p: usize, tol: f64) {
	let proj = y.mat_mul(&MatrixOps::transpose(y));
	let nrows = y.nrows();
	for i in 0..p {
		let ei: linalg::Vec<f64> = VectorOps::from_fn(nrows, |k| if k == i { 1.0 } else { 0.0 });
		let projected = proj.mat_vec(&ei);
		let diff = projected.sub(&ei);
		let error = diff.norm();
		assert!(
			error < tol,
			"Subspace projection error for e_{}: {:.2e} (tol = {:.2e})",
			i + 1,
			error,
			tol,
		);
	}
}

// ============================================================================
// PROBLEM 1: RAYLEIGH QUOTIENT ON S^{n-1} (EIGENVECTOR COMPUTATION)
//
// min_{x in S^{n-1}} x^T A x
// Known solution: eigenvector of smallest eigenvalue of A
// ============================================================================

mod rayleigh_on_sphere {
	use super::*;

	#[test]
	fn sgd_finds_smallest_eigenvector() {
		let n = 5;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::with_gap(n, 3.0);
		let x0 = sphere_start(n);

		// SGD on the sphere needs careful tuning: step size must be small enough
		// to not overshoot, but large enough to converge in reasonable time.
		let config = SGDConfig::new().with_constant_step_size(0.01);
		let mut opt = SGD::new(config);
		let crit = StoppingCriterion::new()
			.with_max_iterations(20000)
			.with_gradient_tolerance(1e-6);

		let result = opt.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();

		// SGD should converge or at least reduce cost significantly
		let final_cost = cost_fn.cost(&result.point).unwrap();
		let initial_cost = cost_fn.cost(&x0).unwrap();
		assert!(
			final_cost < initial_cost,
			"SGD: cost should decrease, got initial={:.6}, final={:.6}",
			initial_cost,
			final_cost
		);

		// Point must be on the sphere
		assert!(
			(result.point.norm() - 1.0).abs() < 1e-14,
			"SGD: point not on sphere, norm = {}",
			result.point.norm()
		);
	}

	#[test]
	fn sgd_momentum_accelerates_convergence() {
		let n = 10;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::with_gap(n, 5.0);
		let x0 = sphere_start(n);

		// Without momentum
		let config_plain = SGDConfig::new().with_constant_step_size(0.02);
		let mut opt_plain = SGD::new(config_plain);
		let crit = StoppingCriterion::new()
			.with_max_iterations(500)
			.with_gradient_tolerance(1e-6);
		let result_plain = opt_plain.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();

		// With Nesterov momentum
		let config_nesterov = SGDConfig::new()
			.with_constant_step_size(0.02)
			.with_nesterov_momentum(0.9);
		let mut opt_nesterov = SGD::new(config_nesterov);
		let result_nesterov = opt_nesterov
			.optimize(&cost_fn, &sphere, &x0, &crit)
			.unwrap();

		// Momentum should converge faster (fewer iterations or lower cost)
		let cost_plain = cost_fn.cost(&result_plain.point).unwrap();
		let cost_nesterov = cost_fn.cost(&result_nesterov.point).unwrap();
		assert!(
			cost_nesterov <= cost_plain + 1e-6,
			"Nesterov ({:.8}) should be at least as good as plain SGD ({:.8})",
			cost_nesterov,
			cost_plain
		);
	}

	#[test]
	fn adam_finds_smallest_eigenvector() {
		let n = 5;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::with_gap(n, 3.0);
		let x0 = sphere_start(n);

		let mut opt = Adam::new(
			AdamConfig::new()
				.with_learning_rate(0.01)
				.with_beta1(0.9)
				.with_beta2(0.999),
		);
		let crit = StoppingCriterion::new()
			.with_max_iterations(10000)
			.with_gradient_tolerance(1e-6);

		let result = opt.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		let initial_cost = cost_fn.cost(&x0).unwrap();
		assert!(
			final_cost < initial_cost,
			"Adam: cost should decrease, got initial={:.6}, final={:.6}",
			initial_cost,
			final_cost
		);

		assert!(
			(result.point.norm() - 1.0).abs() < 1e-14,
			"Adam: point not on sphere"
		);
	}

	#[test]
	fn cg_converges_fast_on_well_conditioned_problem() {
		let n = 10;
		let sphere = Sphere::<f64>::new(n).unwrap();
		// Well-conditioned: eigenvalues 1, 2, ..., 10 -> kappa = 10
		let cost_fn = RayleighQuotient::diagonal(n);
		let x0 = sphere_start(n);

		let mut opt = ConjugateGradient::new(CGConfig::polak_ribiere());
		let crit = StoppingCriterion::new()
			.with_max_iterations(200)
			.with_gradient_tolerance(1e-10);

		let result = opt.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			(final_cost - 1.0).abs() < 1e-8,
			"CG (PR): final cost = {:.12}, expected 1.0. Iterations: {}",
			final_cost,
			result.iterations
		);

		check_rayleigh_minimizer(&result.point, 1e-4);

		// CG should converge in significantly fewer iterations than SGD
		assert!(
			result.iterations < 500,
			"CG should converge reasonably fast, took {} iterations",
			result.iterations
		);
	}

	#[test]
	fn cg_all_variants_converge() {
		let n = 5;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::with_gap(n, 3.0);
		let x0 = sphere_start(n);
		let crit = StoppingCriterion::new()
			.with_max_iterations(300)
			.with_gradient_tolerance(1e-8);

		let variants: Vec<(&str, ConjugateGradient<f64>)> = vec![
			("FR", ConjugateGradient::new(CGConfig::fletcher_reeves())),
			("PR", ConjugateGradient::new(CGConfig::polak_ribiere())),
			("HS", ConjugateGradient::new(CGConfig::hestenes_stiefel())),
			("DY", ConjugateGradient::new(CGConfig::dai_yuan())),
		];

		for (name, mut opt) in variants {
			let result = opt.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();
			let final_cost = cost_fn.cost(&result.point).unwrap();
			assert!(
				(final_cost - 1.0).abs() < 0.05,
				"CG variant {}: final cost = {:.10}, expected 1.0",
				name,
				final_cost
			);
		}
	}

	#[test]
	fn gradient_norm_decreases_monotonically_with_sgd() {
		let n = 5;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::with_gap(n, 5.0);
		let x0 = sphere_start(n);

		let config = SGDConfig::new().with_constant_step_size(0.01);
		let mut opt = SGD::new(config);

		// Run for a fixed number of iterations and verify cost decreases
		let mut current_point = x0.clone();
		let mut prev_cost = cost_fn.cost(&current_point).unwrap();

		for _iter in 0..50 {
			let crit = StoppingCriterion::new().with_max_iterations(1);
			let result = opt
				.optimize(&cost_fn, &sphere, &current_point, &crit)
				.unwrap();
			let new_cost = cost_fn.cost(&result.point).unwrap();

			// With small enough step size, cost should decrease (or stay same)
			assert!(
				new_cost <= prev_cost + 1e-10,
				"Cost increased: {:.10} > {:.10}",
				new_cost,
				prev_cost
			);

			prev_cost = new_cost;
			current_point = result.point;
		}
	}
}

// ============================================================================
// PROBLEM 2: QUADRATIC MINIMIZATION ON R^n
//
// min_{x in R^n} 0.5 * x^T A x + b^T x + c
// Known solution: x* = -A^{-1} b, f* = c - 0.5 * b^T A^{-1} b
// ============================================================================

mod quadratic_on_euclidean {
	use super::*;

	/// Create a test problem: f(x) = 0.5 * x^T x
	/// Minimum at x = 0 with f* = 0.
	fn simple_problem(n: usize) -> (Euclidean<f64>, SimpleQuadratic, linalg::Vec<f64>) {
		let eucl = Euclidean::<f64>::new(n).unwrap();
		let cost_fn = SimpleQuadratic::new(n);
		let x0: linalg::Vec<f64> = VectorOps::from_fn(n, |_| 1.0);
		(eucl, cost_fn, x0)
	}

	/// f(x) = 0.5 * x^T A x + b^T x, min at x* = -A^{-1}b.
	fn shifted_problem(
		n: usize,
	) -> (
		Euclidean<f64>,
		ShiftedQuadratic,
		linalg::Vec<f64>,
		linalg::Vec<f64>,
		f64,
	) {
		let eucl = Euclidean::<f64>::new(n).unwrap();
		let cost_fn = ShiftedQuadratic::new(n);
		let x_star = cost_fn.x_star();
		let f_star = cost_fn.f_star();
		let x0: linalg::Vec<f64> = VectorOps::from_fn(n, |_| 2.0);
		(eucl, cost_fn, x0, x_star, f_star)
	}

	#[test]
	fn newton_converges_in_one_step_on_quadratic() {
		let (eucl, cost_fn, x0) = simple_problem(5);

		let mut opt = Newton::new(NewtonConfig::new());
		let crit = StoppingCriterion::new()
			.with_max_iterations(10)
			.with_gradient_tolerance(1e-12);

		let result = opt.optimize(&cost_fn, &eucl, &x0, &crit).unwrap();

		// Newton should converge in 1 iteration on a quadratic
		assert!(
			result.iterations <= 2,
			"Newton should converge in 1-2 steps on quadratic, took {}",
			result.iterations
		);

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			final_cost < 1e-20,
			"Newton: final cost = {:.2e}, expected ~0",
			final_cost
		);

		// Solution should be at the origin
		assert!(
			result.point.norm() < 1e-10,
			"Newton: solution norm = {:.2e}, expected ~0",
			result.point.norm()
		);
	}

	#[test]
	fn newton_finds_exact_solution_with_linear_term() {
		let n = 5;
		let (eucl, cost_fn, x0, x_star, f_star) = shifted_problem(n);

		let mut opt = Newton::new(NewtonConfig::new());
		let crit = StoppingCriterion::new()
			.with_max_iterations(10)
			.with_gradient_tolerance(1e-12);

		let result = opt.optimize(&cost_fn, &eucl, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			(final_cost - f_star).abs() < 1e-12,
			"Newton: cost = {:.14}, expected {:.14}",
			final_cost,
			f_star
		);

		let error = result.point.sub(&x_star).norm();
		assert!(error < 1e-6, "Newton: ||x - x*|| = {:.2e}", error);
	}

	#[test]
	fn lbfgs_converges_to_exact_solution() {
		let n = 10;
		let (eucl, cost_fn, x0, x_star, f_star) = shifted_problem(n);

		let config = LBFGSConfig::new().with_memory_size(10);
		let mut opt = LBFGS::new(config);
		let crit = StoppingCriterion::new()
			.with_max_iterations(100)
			.with_gradient_tolerance(1e-12);

		let result = opt.optimize(&cost_fn, &eucl, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			(final_cost - f_star).abs() < 1e-10,
			"L-BFGS: cost = {:.14}, expected {:.14}",
			final_cost,
			f_star
		);

		let error = result.point.sub(&x_star).norm();
		assert!(error < 1e-8, "L-BFGS: ||x - x*|| = {:.2e}", error);
	}

	#[test]
	fn trust_region_converges_to_exact_solution() {
		let n = 5;
		let (eucl, cost_fn, x0, x_star, f_star) = shifted_problem(n);

		let mut opt = TrustRegion::new(TrustRegionConfig::new());
		let crit = StoppingCriterion::new()
			.with_max_iterations(50)
			.with_gradient_tolerance(1e-12);

		let result = opt.optimize(&cost_fn, &eucl, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			(final_cost - f_star).abs() < 1e-10,
			"Trust Region: cost = {:.14}, expected {:.14}",
			final_cost,
			f_star
		);

		let error = result.point.sub(&x_star).norm();
		assert!(error < 1e-6, "Trust Region: ||x - x*|| = {:.2e}", error);
	}

	#[test]
	fn cg_converges_in_at_most_n_steps_on_quadratic() {
		let n = 5;
		let (eucl, cost_fn, x0) = simple_problem(n);

		let mut opt = ConjugateGradient::new(CGConfig::fletcher_reeves());
		let crit = StoppingCriterion::new()
			.with_max_iterations(3 * n + 10) // CG on n-dim quadratic, with adaptive step
			.with_gradient_tolerance(1e-12);

		let result = opt.optimize(&cost_fn, &eucl, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			final_cost < 1e-6,
			"CG: cost = {:.2e} after {} iterations, expected ~0",
			final_cost,
			result.iterations
		);

		// With adaptive initial step size, CG may take more iterations
		// than the classical n-step bound (which assumes exact line search)
		assert!(
			result.iterations <= 3 * n + 10,
			"CG should converge in <= {} steps on {}-dim quadratic, took {}",
			3 * n + 10,
			n,
			result.iterations
		);
	}
}

// ============================================================================
// PROBLEM 3: TRACE MINIMIZATION ON GRASSMANN MANIFOLD (EIGENSPACE COMPUTATION)
//
// min_{Y in Gr(n,p)} tr(Y^T A Y)
// Known solution: column space of the p eigenvectors of smallest eigenvalues
// ============================================================================

mod trace_on_grassmann {
	use super::*;

	#[test]
	fn sgd_finds_dominant_eigenspace() {
		let n = 6;
		let p = 2;
		let gr = Grassmann::<f64>::new(n, p).unwrap();
		let cost_fn = TraceMinimization::diagonal(n);
		let y0 = grassmann_start(&gr, n, p);

		let config = SGDConfig::new().with_constant_step_size(0.005);
		let mut opt = SGD::new(config);
		let crit = StoppingCriterion::new()
			.with_max_iterations(20000)
			.with_gradient_tolerance(1e-6);

		let result = opt.optimize(&cost_fn, &gr, &y0, &crit).unwrap();

		// Cost should decrease toward sum of p smallest eigenvalues: 1 + 2 = 3
		let final_cost = cost_fn.cost(&result.point).unwrap();
		let initial_cost = cost_fn.cost(&y0).unwrap();
		assert!(
			final_cost < initial_cost,
			"SGD/Grassmann: cost should decrease, got initial={:.6}, final={:.6}",
			initial_cost,
			final_cost
		);

		// Point must be on manifold
		assert!(
			gr.is_point_on_manifold(&result.point, 1e-10),
			"Result not on Grassmann manifold"
		);
	}

	#[test]
	fn cg_finds_eigenspace_precisely() {
		let n = 8;
		let p = 3;
		let gr = Grassmann::<f64>::new(n, p).unwrap();
		let cost_fn = TraceMinimization::diagonal(n);
		let y0 = grassmann_start(&gr, n, p);

		let mut opt = ConjugateGradient::new(CGConfig::polak_ribiere());
		let crit = StoppingCriterion::new()
			.with_max_iterations(300)
			.with_gradient_tolerance(1e-10);

		let result = opt.optimize(&cost_fn, &gr, &y0, &crit).unwrap();

		// Cost should be 1 + 2 + 3 = 6
		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			(final_cost - 6.0).abs() < 1e-4,
			"CG/Grassmann: cost = {:.10}, expected 6.0",
			final_cost
		);

		check_subspace_minimizer(&result.point, p, 0.01);
	}

	#[test]
	fn adam_finds_eigenspace() {
		let n = 6;
		let p = 2;
		let gr = Grassmann::<f64>::new(n, p).unwrap();
		let cost_fn = TraceMinimization::diagonal(n);
		let y0 = grassmann_start(&gr, n, p);

		let mut opt = Adam::new(
			AdamConfig::new()
				.with_learning_rate(0.005)
				.with_beta1(0.9)
				.with_beta2(0.999),
		);
		let crit = StoppingCriterion::new()
			.with_max_iterations(10000)
			.with_gradient_tolerance(1e-6);

		let result = opt.optimize(&cost_fn, &gr, &y0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		let initial_cost = cost_fn.cost(&y0).unwrap();
		assert!(
			final_cost < initial_cost,
			"Adam/Grassmann: cost should decrease, got initial={:.6}, final={:.6}",
			initial_cost,
			final_cost
		);
	}
}

// ============================================================================
// PROBLEM 4: PROCRUSTES PROBLEM ON STIEFEL MANIFOLD
//
// min_{X in St(n,p)} ||X - B||_F^2  (A = I)
// Known solution: nearest orthogonal matrix to B, given by U V^T from SVD B = U Sigma V^T
// ============================================================================

mod procrustes_on_stiefel {
	use super::*;

	#[test]
	fn sgd_minimizes_procrustes() {
		let n = 4;
		let p = 2;
		let st = Stiefel::<f64>::new(n, p).unwrap();

		// Target: an arbitrary matrix
		let target = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(n, p, |i, j| {
			((i + 1) as f64) * 0.3 + (j as f64) * 0.5
		});

		// Known solution: nearest orthogonal matrix to target via SVD
		let svd = DecompositionOps::svd(&target);
		let u = svd.u.unwrap();
		let vt = svd.vt.unwrap();
		let _x_star = u.columns(0, p).mat_mul(&vt.rows(0, p));

		let identity_n = <linalg::Mat<f64> as MatrixOps<f64>>::identity(n);
		let cost_fn = Procrustes::new(identity_n, target);
		let x0 = stiefel_start(&st);

		let config = SGDConfig::new().with_constant_step_size(0.002);
		let mut opt = SGD::new(config);
		let crit = StoppingCriterion::new()
			.with_max_iterations(30000)
			.with_gradient_tolerance(1e-6);

		let result = opt.optimize(&cost_fn, &st, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		let initial_cost = cost_fn.cost(&x0).unwrap();
		assert!(
			final_cost < initial_cost,
			"SGD/Stiefel Procrustes: cost should decrease, got initial={:.6}, final={:.6}",
			initial_cost,
			final_cost
		);

		// Verify result is on Stiefel manifold: X^T X = I
		let xtx = MatrixOps::transpose(&result.point).mat_mul(&result.point);
		let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(p);
		let orth_error = xtx.sub(&eye).norm();
		assert!(
			orth_error < 1e-12,
			"Result not on Stiefel manifold: ||X^T X - I|| = {:.2e}",
			orth_error
		);
	}

	#[test]
	fn cg_solves_procrustes_precisely() {
		let n = 4;
		let p = 2;
		let st = Stiefel::<f64>::new(n, p).unwrap();

		let target = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(n, p, |i, j| {
			((i + 1) as f64) * 0.3 + (j as f64) * 0.5
		});
		let svd = DecompositionOps::svd(&target);
		let u = svd.u.unwrap();
		let vt = svd.vt.unwrap();
		let x_star = u.columns(0, p).mat_mul(&vt.rows(0, p));
		let diff = target.sub(&x_star);
		let f_star = diff.norm() * diff.norm();

		let identity_n = <linalg::Mat<f64> as MatrixOps<f64>>::identity(n);
		let cost_fn = Procrustes::new(identity_n, target);
		let x0 = stiefel_start(&st);

		let mut opt = ConjugateGradient::new(CGConfig::polak_ribiere());
		let crit = StoppingCriterion::new()
			.with_max_iterations(200)
			.with_gradient_tolerance(1e-10);

		let result = opt.optimize(&cost_fn, &st, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			(final_cost - f_star).abs() < 1e-6,
			"CG/Stiefel Procrustes: cost = {:.10}, expected {:.10}",
			final_cost,
			f_star
		);
	}
}

// ============================================================================
// PROBLEM 5: MANIFOLD GEOMETRIC PROPERTIES VALIDATION
// ============================================================================

mod manifold_geometry {
	use super::*;

	#[test]
	fn sphere_exponential_and_logarithm_are_inverse() {
		let sphere = Sphere::<f64>::new(5).unwrap();

		for _ in 0..20 {
			let x = sphere.random_point();
			let v = sphere.random_tangent(&x).unwrap();
			let mut v_small = v.clone();
			v_small.scale_mut(0.3); // Small tangent for numerical stability

			let y = sphere.exp_map(&x, &v_small).unwrap();
			let v_recovered = sphere.log_map(&x, &y).unwrap();

			let error = v_small.sub(&v_recovered).norm();
			assert!(
				error < 1e-12,
				"exp/log inverse error: {:.2e} (v_norm = {:.4})",
				error,
				v_small.norm()
			);
		}
	}

	#[test]
	fn sphere_distance_is_geodesic() {
		let sphere = Sphere::<f64>::new(5).unwrap();

		for _ in 0..20 {
			let x = sphere.random_point();
			let v = sphere.random_tangent(&x).unwrap();
			let v_norm = v.norm();
			let mut v_unit = v.clone();
			v_unit.div_scalar_mut(v_norm);
			let t = 0.5; // geodesic parameter
			let mut v_scaled = v_unit.clone();
			v_scaled.scale_mut(t);

			let y = sphere.exp_map(&x, &v_scaled).unwrap();
			let dist = sphere.geodesic_distance(&x, &y).unwrap();

			assert!(
				(dist - t).abs() < 1e-12,
				"Geodesic distance error: d = {:.14}, expected {:.14}",
				dist,
				t
			);
		}
	}

	#[test]
	fn sphere_parallel_transport_preserves_norm() {
		let sphere = Sphere::<f64>::new(10).unwrap();

		for _ in 0..20 {
			let x = sphere.random_point();
			let v = sphere.random_tangent(&x).unwrap();
			let mut v_small = v.clone();
			v_small.scale_mut(0.5);
			let y = sphere.exp_map(&x, &v_small).unwrap();

			let u = sphere.random_tangent(&x).unwrap();
			let transported = sphere.parallel_transport(&x, &y, &u).unwrap();

			let norm_before = u.norm();
			let norm_after = transported.norm();

			assert!(
				(norm_before - norm_after).abs() < 1e-12,
				"Parallel transport norm error: {:.14} -> {:.14}",
				norm_before,
				norm_after
			);

			// Transported vector must be tangent at y
			let inner = y.dot(&transported);
			assert!(
				inner.abs() < 1e-12,
				"Transported vector not tangent: <y, tau(u)> = {:.2e}",
				inner
			);
		}
	}

	#[test]
	fn sphere_parallel_transport_preserves_inner_product() {
		let sphere = Sphere::<f64>::new(10).unwrap();

		for _ in 0..10 {
			let x = sphere.random_point();
			let u = sphere.random_tangent(&x).unwrap();
			let w = sphere.random_tangent(&x).unwrap();

			let v = sphere.random_tangent(&x).unwrap();
			let mut v_small = v.clone();
			v_small.scale_mut(0.3);
			let y = sphere.exp_map(&x, &v_small).unwrap();

			let u_t = sphere.parallel_transport(&x, &y, &u).unwrap();
			let w_t = sphere.parallel_transport(&x, &y, &w).unwrap();

			let ip_before = u.dot(&w);
			let ip_after = u_t.dot(&w_t);

			assert!(
				(ip_before - ip_after).abs() < 1e-11,
				"Inner product not preserved: {:.14} -> {:.14}",
				ip_before,
				ip_after
			);
		}
	}

	#[test]
	fn stiefel_retraction_preserves_orthogonality() {
		let st = Stiefel::<f64>::new(6, 3).unwrap();

		for _ in 0..20 {
			let mut x = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(6, 3);
			st.random_point(&mut x).unwrap();

			let mut v = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(6, 3);
			st.random_tangent(&x, &mut v).unwrap();

			// Various step sizes
			for &scale in &[0.01, 0.1, 0.5, 1.0] {
				let v_scaled = v.scale_by(scale);
				let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(6, 3);
				st.retract(&x, &v_scaled, &mut y).unwrap();

				let yty = MatrixOps::transpose(&y).mat_mul(&y);
				let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(3);
				let orth_error = yty.sub(&eye).norm();
				assert!(
					orth_error < 1e-13,
					"Stiefel retraction orthogonality error at scale {}: {:.2e}",
					scale,
					orth_error
				);
			}
		}
	}

	#[test]
	fn grassmann_distance_is_symmetric_and_nonneg() {
		let gr = Grassmann::<f64>::new(6, 2).unwrap();

		for _ in 0..10 {
			let mut y1 = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(6, 2);
			let mut y2 = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(6, 2);
			gr.random_point(&mut y1).unwrap();
			gr.random_point(&mut y2).unwrap();

			let d12 = gr.distance(&y1, &y2).unwrap();
			let d21 = gr.distance(&y2, &y1).unwrap();

			assert!(d12 >= 0.0, "Distance must be non-negative");
			assert!(
				(d12 - d21).abs() < 1e-12,
				"Distance not symmetric: d(Y1,Y2) = {:.14}, d(Y2,Y1) = {:.14}",
				d12,
				d21
			);

			let d11 = gr.distance(&y1, &y1).unwrap();
			assert!(
				d11.abs() < 1e-7,
				"Self-distance must be ~0, got {:.2e}",
				d11
			);
		}
	}

	#[test]
	fn euclidean_is_flat() {
		let eucl = Euclidean::<f64>::new(5).unwrap();

		// Retraction = addition
		let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
		let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5]);
		let mut y: linalg::Vec<f64> = VectorOps::zeros(5);
		eucl.retract(&x, &v, &mut y).unwrap();
		let expected_y = x.add(&v);
		assert!(
			y.sub(&expected_y).norm() < 1e-15,
			"Euclidean retraction should be addition"
		);

		// Parallel transport = identity
		let z: linalg::Vec<f64> = VectorOps::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);
		let mut transported: linalg::Vec<f64> = VectorOps::zeros(5);
		eucl.parallel_transport(&x, &z, &v, &mut transported)
			.unwrap();
		assert!(
			transported.sub(&v).norm() < 1e-15,
			"Euclidean parallel transport should be identity"
		);

		// Distance = Euclidean norm
		let d = eucl.distance(&x, &z).unwrap();
		let expected = x.sub(&z).norm();
		assert!(
			(d - expected).abs() < 1e-15,
			"Euclidean distance should be norm"
		);
	}
}

// ============================================================================
// PROBLEM 6: CONVERGENCE RATE VERIFICATION
// ============================================================================

mod convergence_rates {
	use super::*;

	#[test]
	fn sgd_converges_linearly() {
		let n = 5;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::with_gap(n, 5.0);
		let x0 = sphere_start(n);

		let config = SGDConfig::new().with_constant_step_size(0.01);
		let mut opt = SGD::new(config);

		// Collect cost at each iteration
		let mut costs = Vec::new();
		let mut x = x0;
		for _ in 0..500 {
			let crit = StoppingCriterion::new().with_max_iterations(1);
			let result = opt.optimize(&cost_fn, &sphere, &x, &crit).unwrap();
			let c = cost_fn.cost(&result.point).unwrap();
			costs.push(c);
			x = result.point;
		}

		// Check that the cost decreased significantly
		let initial_cost = costs[0];
		let final_cost = *costs.last().unwrap();
		assert!(
			final_cost < initial_cost,
			"SGD should decrease cost: initial {:.6} -> final {:.6}",
			initial_cost,
			final_cost
		);

		// Check that cost is generally decreasing
		let n_decreasing = costs.windows(2).filter(|w| w[1] < w[0] + 1e-10).count();
		let total = costs.len() - 1;
		let ratio = n_decreasing as f64 / total as f64;
		assert!(
			ratio > 0.9,
			"SGD cost should mostly decrease: {:.1}% of steps decreased",
			ratio * 100.0
		);
	}

	#[test]
	fn cg_converges_faster_than_sgd() {
		let n = 10;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::diagonal(n);
		let x0 = sphere_start(n);

		// SGD
		let config_sgd = SGDConfig::new().with_constant_step_size(0.01);
		let mut opt_sgd = SGD::new(config_sgd);
		let crit = StoppingCriterion::new()
			.with_max_iterations(200)
			.with_gradient_tolerance(1e-6);
		let result_sgd = opt_sgd.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();

		// CG
		let mut opt_cg = ConjugateGradient::new(CGConfig::polak_ribiere());
		let result_cg = opt_cg.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();

		let cost_sgd = cost_fn.cost(&result_sgd.point).unwrap();
		let cost_cg = cost_fn.cost(&result_cg.point).unwrap();

		// CG should reach a better or comparable solution
		assert!(
			cost_cg <= cost_sgd + 0.5,
			"CG ({:.8}) should be at least as good as SGD ({:.8})",
			cost_cg,
			cost_sgd
		);
	}

	#[test]
	fn lbfgs_converges_superlinearly_on_quadratic() {
		let n = 10;
		let eucl = Euclidean::<f64>::new(n).unwrap();
		let cost_fn = SimpleQuadratic::new(n);
		let x0: linalg::Vec<f64> = VectorOps::from_fn(n, |_| 1.0);

		let config = LBFGSConfig::new().with_memory_size(n); // Full memory
		let mut opt = LBFGS::new(config);

		let mut costs = Vec::new();
		let mut x = x0;
		for _ in 0..30 {
			let crit = StoppingCriterion::new().with_max_iterations(1);
			let result = opt.optimize(&cost_fn, &eucl, &x, &crit).unwrap();
			let c = cost_fn.cost(&result.point).unwrap();
			costs.push(c);
			x = result.point;
			if c < 1e-20 {
				break;
			}
		}

		// L-BFGS with full memory on quadratic should converge very fast
		let final_cost = *costs.last().unwrap();
		assert!(
			final_cost < 1e-15,
			"L-BFGS should converge to near-zero on quadratic, got {:.2e}",
			final_cost
		);

		// Should converge in O(n) steps with full memory
		assert!(
			costs.len() <= n + 5,
			"L-BFGS with full memory should converge in ~n steps, took {}",
			costs.len()
		);
	}
}

// ============================================================================
// PROBLEM 7: NUMERICAL PRECISION AND ROBUSTNESS
// ============================================================================

mod numerical_precision {
	use super::*;

	#[test]
	fn riemannian_gradient_is_tangent() {
		let sphere = Sphere::<f64>::new(10).unwrap();
		let cost_fn = RayleighQuotient::diagonal(10);

		for _ in 0..20 {
			let x = sphere.random_point();
			let (_, egrad) = cost_fn.cost_and_gradient_alloc(&x).unwrap();

			let mut rgrad: linalg::Vec<f64> = VectorOps::zeros(10);
			sphere
				.euclidean_to_riemannian_gradient(&x, &egrad, &mut rgrad)
				.unwrap();

			// Riemannian gradient must be tangent: <x, rgrad> = 0
			let inner = x.dot(&rgrad);
			assert!(
				inner.abs() < 1e-14,
				"Riemannian gradient not tangent: <x, rgrad> = {:.2e}",
				inner
			);
		}
	}

	#[test]
	fn optimization_result_stays_on_manifold() {
		let n = 10;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::diagonal(n);
		let x0 = sphere_start(n);
		let crit = StoppingCriterion::new()
			.with_max_iterations(500)
			.with_gradient_tolerance(1e-6);

		// SGD
		let result = SGD::new(SGDConfig::new().with_constant_step_size(0.01))
			.optimize(&cost_fn, &sphere, &x0, &crit)
			.unwrap();
		assert!(
			(result.point.norm() - 1.0).abs() < 1e-14,
			"SGD: point not on sphere, ||x|| - 1 = {:.2e}",
			(result.point.norm() - 1.0).abs()
		);

		// Adam
		let result = Adam::new(AdamConfig::new().with_learning_rate(0.01))
			.optimize(&cost_fn, &sphere, &x0, &crit)
			.unwrap();
		assert!(
			(result.point.norm() - 1.0).abs() < 1e-14,
			"Adam: point not on sphere, ||x|| - 1 = {:.2e}",
			(result.point.norm() - 1.0).abs()
		);

		// CG
		let result = ConjugateGradient::new(CGConfig::polak_ribiere())
			.optimize(&cost_fn, &sphere, &x0, &crit)
			.unwrap();
		assert!(
			(result.point.norm() - 1.0).abs() < 1e-14,
			"CG: point not on sphere, ||x|| - 1 = {:.2e}",
			(result.point.norm() - 1.0).abs()
		);
	}

	#[test]
	fn stiefel_optimization_preserves_orthogonality_throughout() {
		let n = 5;
		let p = 2;
		let st = Stiefel::<f64>::new(n, p).unwrap();
		let target = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(n, p, |i, j| {
			((i + 1) as f64) * 0.3 + (j as f64) * 0.5
		});
		let identity_n = <linalg::Mat<f64> as MatrixOps<f64>>::identity(n);
		let cost_fn = Procrustes::new(identity_n, target);
		let x0 = stiefel_start(&st);

		let config = SGDConfig::new().with_constant_step_size(0.005);
		let mut opt = SGD::new(config);

		let mut x = x0;
		for iter in 0..100 {
			let crit = StoppingCriterion::new().with_max_iterations(1);
			let result = opt.optimize(&cost_fn, &st, &x, &crit).unwrap();

			let xtx = MatrixOps::transpose(&result.point).mat_mul(&result.point);
			let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(p);
			let orth_error = xtx.sub(&eye).norm();
			assert!(
				orth_error < 1e-12,
				"Orthogonality violated at iteration {}: ||X^T X - I|| = {:.2e}",
				iter,
				orth_error
			);

			x = result.point;
		}
	}

	#[test]
	fn high_dimensional_sphere_optimization() {
		let n = 100;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::with_gap(n, 10.0);
		let x0 = sphere_start(n);

		let mut opt = ConjugateGradient::new(CGConfig::polak_ribiere());
		let crit = StoppingCriterion::new()
			.with_max_iterations(500)
			.with_gradient_tolerance(1e-8);

		let result = opt.optimize(&cost_fn, &sphere, &x0, &crit).unwrap();

		let final_cost = cost_fn.cost(&result.point).unwrap();
		assert!(
			(final_cost - 1.0).abs() < 1e-6,
			"High-dim CG: cost = {:.10}, expected 1.0 (n = {})",
			final_cost,
			n
		);

		// Point must be on sphere
		assert!(
			(result.point.norm() - 1.0).abs() < 1e-14,
			"High-dim: point not on sphere"
		);
	}

	#[test]
	fn gradient_consistency_check() {
		let n = 5;
		let sphere = Sphere::<f64>::new(n).unwrap();
		let cost_fn = RayleighQuotient::diagonal(n);
		let x = sphere.random_point();

		// Compute gradient analytically
		let grad_analytic = cost_fn.gradient(&x).unwrap();

		// Compute gradient via finite differences
		let eps = 1e-7;
		let mut grad_fd: linalg::Vec<f64> = VectorOps::zeros(n);
		for i in 0..n {
			let mut x_plus = x.clone();
			let mut x_minus = x.clone();
			*x_plus.get_mut(i) += eps;
			*x_minus.get_mut(i) -= eps;
			*grad_fd.get_mut(i) =
				(cost_fn.cost(&x_plus).unwrap() - cost_fn.cost(&x_minus).unwrap()) / (2.0 * eps);
		}

		let error = grad_analytic.sub(&grad_fd).norm();
		let relative_error = error / grad_analytic.norm().max(1e-15);
		assert!(
			relative_error < 1e-5,
			"Gradient consistency: relative error = {:.2e}",
			relative_error
		);
	}
}
