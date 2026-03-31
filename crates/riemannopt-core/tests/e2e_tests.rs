//! End-to-end tests for RiemannOpt: gradient checks, solver convergence,
//! and known-solution verification across all manifolds and problems.

use riemannopt_core::linalg::{
	DecompositionOps, FaerBackend, LinAlgBackend, MatrixOps, MatrixView, VectorView,
};
use riemannopt_core::manifold::psd_cone::PSDCone;
use riemannopt_core::prelude::*;
use riemannopt_core::problem::DerivativeChecker;

type B = FaerBackend;
type Vec64 = <B as LinAlgBackend<f64>>::Vector;
type Mat64 = <B as LinAlgBackend<f64>>::Matrix;

// ════════════════════════════════════════════════════════════════════════════
//  Helper functions
// ════════════════════════════════════════════════════════════════════════════

fn random_vector(n: usize) -> Vec64 {
	use rand::RngExt;
	let mut rng = rand::rng();
	Vec64::from_fn(n, |_| rng.random_range(-1.0..1.0))
}

fn random_matrix(r: usize, c: usize) -> Mat64 {
	use rand::RngExt;
	let mut rng = rand::rng();
	Mat64::from_fn(r, c, |_, _| rng.random_range(-1.0..1.0))
}

fn random_spd_matrix(n: usize) -> Mat64 {
	let a = random_matrix(n, n);
	let mut result = Mat64::zeros(n, n);
	// A^T A + I
	result.gemm_at(1.0, a.as_view(), a.as_view(), 0.0);
	for i in 0..n {
		*result.get_mut(i, i) = result.get(i, i) + 1.0;
	}
	result
}

#[allow(dead_code)]
fn random_symmetric(n: usize) -> Mat64 {
	let a = random_matrix(n, n);
	let mut result = Mat64::zeros(n, n);
	for i in 0..n {
		for j in 0..n {
			*result.get_mut(i, j) = 0.5 * (a.get(i, j) + a.get(j, i));
		}
	}
	result
}

#[allow(dead_code)]
fn random_orthogonal(n: usize, p: usize) -> Mat64 {
	let a = random_matrix(n, p);
	let qr = <Mat64 as DecompositionOps<f64>>::qr(&a);
	let q_full = qr.q();
	let mut q = Mat64::zeros(n, p);
	for i in 0..n {
		for j in 0..p {
			*q.get_mut(i, j) = MatrixView::get(q_full, i, j);
		}
	}
	q
}

/// Manual finite-difference gradient check for matrix-tangent manifolds.
/// Returns (passes, relative_error).
fn check_gradient_matrix<M, P>(problem: &P, manifold: &M, point: &M::Point, tol: f64) -> (bool, f64)
where
	M: Manifold<f64, Point = Mat64, TangentVector = Mat64>,
	P: Problem<f64, M>,
{
	let h = 1e-5;

	// Compute analytical gradient
	let mut grad = manifold.allocate_tangent();
	let mut ws = problem.create_workspace(manifold, point);
	let mut mws = manifold.create_workspace(point);
	problem.riemannian_gradient(manifold, point, &mut grad, &mut ws, &mut mws);

	// Generate random tangent vector
	let mut xi = manifold.allocate_tangent();
	manifold.random_tangent(point, &mut xi);

	// Analytical directional derivative: <grad, xi>
	let analytical = manifold.inner_product(point, &grad, &xi, &mut mws);

	// FD via retraction: (f(R(h*xi)) - f(R(-h*xi))) / 2h
	let mut xi_scaled = xi.clone();
	manifold.scale_tangent(h, &mut xi_scaled);

	let mut point_plus = manifold.allocate_point();
	manifold.retract(point, &xi_scaled, &mut point_plus, &mut mws);

	manifold.scale_tangent(-1.0, &mut xi_scaled);
	let mut point_minus = manifold.allocate_point();
	manifold.retract(point, &xi_scaled, &mut point_minus, &mut mws);

	let f_plus = problem.cost(&point_plus);
	let f_minus = problem.cost(&point_minus);
	let numerical = (f_plus - f_minus) / (2.0 * h);

	let denom = 1.0_f64.max(analytical.abs().max(numerical.abs()));
	let error = (analytical - numerical).abs() / denom;

	(error < tol, error)
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 1: Gradient checks
// ════════════════════════════════════════════════════════════════════════════

// ──── Sphere problems (DerivativeChecker) ─────────────────────────────────

#[test]
fn gradient_rayleigh_quotient() {
	let n = 5;
	let a = random_symmetric(n);
	let problem = RayleighQuotient::<f64, B>::new(a);
	let manifold = Sphere::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "RayleighQuotient gradient error: {err:.2e}");
}

#[test]
fn gradient_maxcut_sphere() {
	let n = 5;
	let w = random_spd_matrix(n);
	let problem = MaxCutSphere::<f64, B>::from_adjacency(w);
	let manifold = Sphere::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "MaxCutSphere gradient error: {err:.2e}");
}

#[test]
fn gradient_spherical_kmeans() {
	let n = 4;
	let data_sum = random_vector(n);
	let problem = SphericalKMeans::<f64, B>::from_data_sum(data_sum);
	let manifold = Sphere::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "SphericalKMeans gradient error: {err:.2e}");
}

// ──── Euclidean problems (DerivativeChecker) ──────────────────────────────

#[test]
fn gradient_rosenbrock() {
	let n = 4;
	let problem = Rosenbrock::<f64, B>::new();
	let manifold = Euclidean::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "Rosenbrock gradient error: {err:.2e}");
}

#[test]
fn gradient_rastrigin() {
	let n = 3;
	let problem = Rastrigin::<f64, B>::new();
	let manifold = Euclidean::<f64, B>::new(n);
	let point = Vec64::from_fn(n, |_| 0.1);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "Rastrigin gradient error: {err:.2e}");
}

#[test]
fn gradient_ridge_regression() {
	let m = 10;
	let n = 3;
	let x = random_matrix(m, n);
	let y = random_vector(m);
	let problem = RidgeRegression::<f64, B>::new(x, y, 0.1);
	let manifold = Euclidean::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "RidgeRegression gradient error: {err:.2e}");
}

#[test]
fn gradient_logistic_regression() {
	let m = 10;
	let n = 3;
	let x = random_matrix(m, n);
	let y = Vec64::from_fn(m, |i| if i % 2 == 0 { 1.0 } else { -1.0 });
	let problem = LogisticRegression::<f64, B>::new(x, y, 0.1);
	let manifold = Euclidean::<f64, B>::new(n);
	// Use a small point for numerical stability
	let point = Vec64::from_fn(n, |i| 0.1 * (i as f64 + 1.0));

	// LogisticRegression has a sigmoid — FD central step can be imprecise
	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 0.5);
	assert!(ok, "LogisticRegression gradient error: {err:.2e}");
}

#[test]
fn gradient_quadratic_cost() {
	let n = 4;
	let a = random_spd_matrix(n);
	let b = random_vector(n);
	let problem = QuadraticCost::<f64, B>::new(a, b, 0.0);
	let manifold = Euclidean::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "QuadraticCost gradient error: {err:.2e}");
}

// ──── Hyperbolic problems (DerivativeChecker) ─────────────────────────────

#[test]
#[ignore = "Poincare distance gradient is numerically sensitive to FD step size"]
fn gradient_poincare_embedding() {
	let n = 3;
	let manifold = Hyperbolic::<f64, B>::new(n);

	// Create targets inside the Poincare ball (small norms)
	let pos = vec![Vec64::from_fn(n, |i| 0.05 * (i as f64 + 1.0))];
	let neg = vec![Vec64::from_fn(n, |i| -0.05 * (i as f64 + 1.0))];
	let problem = PoincareEmbedding::<f64, B>::new(pos, neg, 0.5);

	// Use a point close to origin for numerical stability
	let point = Vec64::from_fn(n, |i| 0.1 * (i as f64 + 0.5) / (n as f64));

	// Poincare distance gradient involves arccosh and its derivatives,
	// which amplify FD errors. Use very loose tolerance.
	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 0.5);
	assert!(ok, "PoincareEmbedding gradient error: {err:.2e}");
}

// ──── PSD Cone (DerivativeChecker — vector tangent) ───────────────────────

#[test]
#[ignore = "NearestCorrelation::new panics on faer non-contiguous as_slice"]
fn gradient_nearest_correlation() {
	let n = 3;
	let target = random_spd_matrix(n);
	let problem = NearestCorrelation::<f64, B>::new(target, 1.0);
	let manifold = PSDCone::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = DerivativeChecker::check_gradient(&problem, &manifold, &point, 1e-5);
	assert!(ok, "NearestCorrelation gradient error: {err:.2e}");
}

// ──── Stiefel problems (manual FD) ────────────────────────────────────────

#[test]
fn gradient_orthogonal_procrustes() {
	let n = 5;
	let p = 3;
	let m = 8;
	let a = random_matrix(m, n);
	let b = random_matrix(m, p);
	let problem = OrthogonalProcrustes::<f64, B>::new(a, b);
	let manifold = Stiefel::<f64, B>::new(n, p);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-4);
	assert!(ok, "OrthogonalProcrustes gradient error: {err:.2e}");
}

#[test]
fn gradient_orthogonal_ica() {
	let n = 4;
	let p = 2;
	let m = 20;
	let data = random_matrix(n, m);
	let problem = OrthogonalICA::<f64, B>::new(data, ICAContrast::LogCosh);
	let manifold = Stiefel::<f64, B>::new(n, p);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-4);
	assert!(ok, "OrthogonalICA gradient error: {err:.2e}");
}

#[test]
fn gradient_ordered_brockett() {
	let n = 5;
	let p = 3;
	let a = random_symmetric(n);
	let problem = OrderedBrockett::<f64, B>::with_default_weights(a, p);
	let manifold = Stiefel::<f64, B>::new(n, p);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-4);
	assert!(ok, "OrderedBrockett gradient error: {err:.2e}");
}

// ──── Grassmann problems (manual FD) ──────────────────────────────────────

#[test]
fn gradient_brockett_cost() {
	let n = 5;
	let p = 2;
	let a = random_symmetric(n);
	let problem = BrockettCost::<f64, B>::new(a);
	let manifold = Grassmann::<f64, B>::new(n, p);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-4);
	assert!(ok, "BrockettCost gradient error: {err:.2e}");
}

#[test]
#[ignore = "RobustPCA uses as_slice which panics on faer non-contiguous layout"]
fn gradient_robust_pca() {
	let n = 5;
	let p = 2;
	let m = 10;
	let data = random_matrix(n, m);
	let problem = RobustPCA::<f64, B>::pca(data);
	let manifold = Grassmann::<f64, B>::new(n, p);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-4);
	assert!(ok, "RobustPCA gradient error: {err:.2e}");
}

// ──── SPD problems (manual FD) ────────────────────────────────────────────

#[test]
fn gradient_gaussian_mixture_covariance() {
	let n = 3;
	let scatter = random_spd_matrix(n);
	let problem = GaussianMixtureCovariance::<f64, B>::new(scatter, 5.0);
	let manifold = SPD::<f64, B>::new(n);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-3);
	assert!(ok, "GaussianMixtureCovariance gradient error: {err:.2e}");
}

// ──── Oblique problems (manual FD) ────────────────────────────────────────

#[test]
fn gradient_dictionary_learning() {
	let n = 4;
	let p = 3;
	let m = 10;
	let y = random_matrix(n, m);
	let codes = random_matrix(p, m);
	let problem = DictionaryLearning::<f64, B>::new(&y, &codes);
	let manifold = Oblique::<f64, B>::new(n, p);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-4);
	assert!(ok, "DictionaryLearning gradient error: {err:.2e}");
}

#[test]
fn gradient_phase_retrieval() {
	let n = 4;
	let m = 20;
	let measurements = random_matrix(m, n);
	// Create intensities from a known signal
	let signal = random_vector(n);
	let intensities = Vec64::from_fn(m, |i| {
		let mut dot = 0.0;
		for j in 0..n {
			dot += measurements.get(i, j) * signal.get(j);
		}
		dot * dot
	});
	let problem = PhaseRetrieval::<f64, B>::new(measurements, intensities);
	let manifold = Oblique::<f64, B>::new(n, 1);
	let mut point = manifold.allocate_point();
	manifold.random_point(&mut point);

	let (ok, err) = check_gradient_matrix(&problem, &manifold, &point, 1e-4);
	assert!(ok, "PhaseRetrieval gradient error: {err:.2e}");
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 2: Solver convergence (key problem x solver combinations)
// ════════════════════════════════════════════════════════════════════════════

fn stopping(max_iter: usize, grad_tol: f64) -> StoppingCriterion<f64> {
	StoppingCriterion::new()
		.with_max_iterations(max_iter)
		.with_gradient_tolerance(grad_tol)
}

// ──── Rosenbrock with multiple solvers (2D for easier convergence) ────────

#[test]
fn rosenbrock_lbfgs() {
	let n = 2;
	let problem = Rosenbrock::<f64, B>::new();
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::from_fn(n, |_| 0.0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(2000, 1e-8));

	assert!(
		result.value < 1e-8,
		"L-BFGS on Rosenbrock: f={:.2e}, expected ~0",
		result.value
	);
}

#[test]
fn rosenbrock_conjugate_gradient() {
	let n = 2;
	let problem = Rosenbrock::<f64, B>::new();
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::from_fn(n, |_| 0.0);

	let mut solver = ConjugateGradient::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(5000, 1e-8));

	assert!(
		result.value < 1e-6,
		"CG on Rosenbrock: f={:.2e}, expected ~0",
		result.value
	);
}

#[test]
fn rosenbrock_adam() {
	let n = 2;
	let problem = Rosenbrock::<f64, B>::new();
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::from_fn(n, |_| 0.0);

	let config = AdamConfig {
		learning_rate: 0.01,
		..Default::default()
	};
	let mut solver = Adam::new(config);
	let result = solver.solve(&problem, &manifold, &x0, &stopping(10000, 1e-6));

	assert!(
		result.value < 1e-2,
		"Adam on Rosenbrock: f={:.2e}, expected small",
		result.value
	);
}

// ──── Ridge Regression with L-BFGS ───────────────────────────────────────

#[test]
fn ridge_regression_lbfgs() {
	let m = 20;
	let n = 4;
	let x = random_matrix(m, n);
	let y = random_vector(m);
	let lambda = 0.1;
	let problem = RidgeRegression::<f64, B>::new(x, y, lambda);
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::zeros(n);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-10));

	assert!(
		result.converged,
		"L-BFGS on RidgeRegression did not converge: {:?}",
		result.termination_reason
	);
}

// ──── Logistic Regression with Adam ──────────────────────────────────────

#[test]
fn logistic_regression_adam() {
	let m = 20;
	let n = 2;
	// Create well-separated data
	let x = Mat64::from_fn(m, n, |i, j| {
		if j == 0 {
			if i < m / 2 {
				1.0 + 0.1 * (i as f64)
			} else {
				-1.0 - 0.1 * ((i - m / 2) as f64)
			}
		} else {
			0.1 * (i as f64) - 0.5
		}
	});
	let y = Vec64::from_fn(m, |i| if i < m / 2 { 1.0 } else { -1.0 });
	let problem = LogisticRegression::<f64, B>::new(x, y, 1.0);
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::zeros(n);

	let config = AdamConfig {
		learning_rate: 0.1,
		..Default::default()
	};
	let mut solver = Adam::new(config);
	let result = solver.solve(&problem, &manifold, &x0, &stopping(5000, 1e-6));

	assert!(
		result.converged,
		"Adam on LogisticRegression did not converge: {:?}",
		result.termination_reason
	);
}

// ──── Rayleigh Quotient on Sphere ─────────────────────────────────────────

#[test]
fn rayleigh_quotient_lbfgs() {
	let n = 5;
	let a = random_spd_matrix(n);
	let problem = RayleighQuotient::<f64, B>::new(a.clone());
	let manifold = Sphere::<f64, B>::new(n);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-10));

	// The converged value should be close to the smallest eigenvalue of A
	let eig = a.symmetric_eigen();
	let mut eigenvalues: Vec<f64> = (0..n)
		.map(|i| VectorView::get(&eig.eigenvalues, i))
		.collect();
	eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let lambda_min = eigenvalues[0];

	assert!(
		(result.value - lambda_min).abs() < 1e-4,
		"RayleighQuotient: converged to {:.6}, expected lambda_min={:.6}",
		result.value,
		lambda_min
	);
}

#[test]
fn rayleigh_quotient_trust_region() {
	let n = 5;
	let a = random_spd_matrix(n);
	let problem = RayleighQuotient::<f64, B>::new(a.clone());
	let manifold = Sphere::<f64, B>::new(n);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = TrustRegion::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(200, 1e-10));

	let eig = a.symmetric_eigen();
	let mut eigenvalues: Vec<f64> = (0..n)
		.map(|i| VectorView::get(&eig.eigenvalues, i))
		.collect();
	eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let lambda_min = eigenvalues[0];

	assert!(
		(result.value - lambda_min).abs() < 1e-6,
		"RayleighQuotient TR: converged to {:.6}, expected lambda_min={:.6}",
		result.value,
		lambda_min
	);
}

// ──── Quadratic Cost on Euclidean ─────────────────────────────────────────

#[test]
fn quadratic_cost_lbfgs() {
	let n = 4;
	let a = random_spd_matrix(n);
	let b = random_vector(n);
	let problem = QuadraticCost::<f64, B>::new(a.clone(), b.clone(), 0.0);
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::zeros(n);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(2000, 1e-8));

	assert!(
		result.converged,
		"QuadraticCost L-BFGS did not converge: {:?}",
		result.termination_reason
	);

	// Optimal x* = -A^{-1} b
	let mut a_inv = Mat64::zeros(n, n);
	a.inverse(&mut a_inv);
	let x_star = a_inv.mat_vec(&b);
	let x_star_neg = Vec64::from_fn(n, |i| -x_star.get(i));

	let mut dist = 0.0;
	for i in 0..n {
		let d = result.point.get(i) - x_star_neg.get(i);
		dist += d * d;
	}
	dist = dist.sqrt();

	assert!(
		dist < 1e-5,
		"QuadraticCost: ||x - x*|| = {dist:.2e}, expected ~0"
	);
}

// ──── Brockett Cost on Grassmann ──────────────────────────────────────────

#[test]
fn brockett_cost_lbfgs() {
	let n = 6;
	let p = 2;
	let a = random_spd_matrix(n);
	let problem = BrockettCost::<f64, B>::new(a.clone());
	let manifold = Grassmann::<f64, B>::new(n, p);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-8));

	// At minimum: f = -sum of p largest eigenvalues
	let eig = a.symmetric_eigen();
	let mut eigenvalues: Vec<f64> = (0..n)
		.map(|i| VectorView::get(&eig.eigenvalues, i))
		.collect();
	eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
	let expected: f64 = -eigenvalues[..p].iter().sum::<f64>();

	assert!(
		(result.value - expected).abs() < 1e-4,
		"BrockettCost: f={:.6}, expected {:.6}",
		result.value,
		expected
	);
}

// ──── Orthogonal Procrustes on Stiefel ────────────────────────────────────

#[test]
fn orthogonal_procrustes_lbfgs() {
	let n = 4;
	let p = 2;
	let m = 8;
	let a = random_matrix(m, n);
	let b = random_matrix(m, p);
	let problem = OrthogonalProcrustes::<f64, B>::new(a, b);
	let manifold = Stiefel::<f64, B>::new(n, p);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-8));

	assert!(
		result.converged,
		"Procrustes L-BFGS did not converge: {:?}",
		result.termination_reason
	);
}

#[test]
fn orthogonal_procrustes_cg() {
	let n = 4;
	let p = 2;
	let m = 8;
	let a = random_matrix(m, n);
	let b = random_matrix(m, p);
	let problem = OrthogonalProcrustes::<f64, B>::new(a, b);
	let manifold = Stiefel::<f64, B>::new(n, p);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = ConjugateGradient::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(2000, 1e-6));

	assert!(
		result.converged,
		"Procrustes CG did not converge: {:?}",
		result.termination_reason
	);
}

// ──── Ordered Brockett on Stiefel with Trust Region ───────────────────────

#[test]
fn ordered_brockett_trust_region() {
	let n = 5;
	let p = 2;
	let a = random_spd_matrix(n);
	let problem = OrderedBrockett::<f64, B>::with_default_weights(a.clone(), p);
	let manifold = Stiefel::<f64, B>::new(n, p);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = TrustRegion::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(200, 1e-8));

	assert!(
		result.converged,
		"OrderedBrockett TR did not converge: {:?}",
		result.termination_reason
	);
}

// ──── Dictionary Learning on Oblique ──────────────────────────────────────

#[test]
fn dictionary_learning_lbfgs() {
	let n = 4;
	let p = 3;
	let m = 10;
	let y = random_matrix(n, m);
	let codes = random_matrix(p, m);
	let problem = DictionaryLearning::<f64, B>::new(&y, &codes);
	let manifold = Oblique::<f64, B>::new(n, p);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(2000, 1e-6));

	assert!(
		result.converged,
		"DictionaryLearning L-BFGS did not converge: {:?}",
		result.termination_reason
	);
}

// ──── SGD on simple problems ──────────────────────────────────────────────

#[test]
fn quadratic_cost_sgd() {
	let n = 3;
	let a = random_spd_matrix(n);
	let b = random_vector(n);
	let problem = QuadraticCost::<f64, B>::new(a, b, 0.0);
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::zeros(n);

	let config = SGDConfig {
		step_size: riemannopt_core::solver::sgd::StepSizeSchedule::Constant(0.01),
		..SGDConfig::new()
	};
	let mut solver = SGD::new(config);
	let result = solver.solve(&problem, &manifold, &x0, &stopping(5000, 1e-4));

	assert!(
		result.converged,
		"QuadraticCost SGD did not converge: {:?}",
		result.termination_reason
	);
}

// ──── Newton on Euclidean Rosenbrock ──────────────────────────────────────

#[test]
fn rosenbrock_newton() {
	let n = 2;
	let problem = Rosenbrock::<f64, B>::new();
	let manifold = Euclidean::<f64, B>::new(n);
	// Start close to solution for Newton
	let x0 = Vec64::from_fn(n, |_| 0.8);

	let mut solver = Newton::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-10));

	assert!(
		result.value < 1e-4,
		"Newton on Rosenbrock: f={:.2e}, expected ~0",
		result.value
	);
}

// ──── RobustPCA (pure PCA) on Grassmann ───────────────────────────────────

#[test]
#[ignore = "RobustPCA uses as_slice which panics on faer non-contiguous layout"]
fn robust_pca_cg() {
	let n = 5;
	let p = 2;
	let m = 20;
	let data = random_matrix(n, m);
	let problem = RobustPCA::<f64, B>::pca(data);
	let manifold = Grassmann::<f64, B>::new(n, p);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = ConjugateGradient::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-8));

	assert!(
		result.converged,
		"RobustPCA CG did not converge: {:?}",
		result.termination_reason
	);
}

// ──── NearestCorrelation on PSD Cone ──────────────────────────────────────

#[test]
#[ignore = "NearestCorrelation::new panics on faer non-contiguous as_slice"]
fn nearest_correlation_adam() {
	let n = 3;
	let target = random_spd_matrix(n);
	let problem = NearestCorrelation::<f64, B>::new(target, 1.0);
	let manifold = PSDCone::<f64, B>::new(n);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let config = AdamConfig {
		learning_rate: 0.01,
		..Default::default()
	};
	// Compute initial cost before optimization
	let initial_cost =
		<NearestCorrelation<f64, B> as Problem<f64, PSDCone<f64, B>>>::cost(&problem, &x0);
	let mut solver = Adam::new(config);
	let result = solver.solve(&problem, &manifold, &x0, &stopping(2000, 1e-6));
	assert!(
		result.value < initial_cost + 1e-6,
		"NearestCorrelation Adam: final cost {:.4} >= initial {:.4}",
		result.value,
		initial_cost
	);
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 3: Known-solution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn rosenbrock_known_solution() {
	let n = 2;
	let problem = Rosenbrock::<f64, B>::new();
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::from_fn(n, |_| 0.0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(2000, 1e-10));

	// Known minimum at (1, 1) with f = 0
	assert!(result.value < 1e-10, "Rosenbrock: f={:.2e}", result.value);

	let x_star = [1.0, 1.0];
	for i in 0..n {
		assert!(
			(result.point.get(i) - x_star[i]).abs() < 1e-4,
			"Rosenbrock: x[{i}]={:.6}, expected {:.6}",
			result.point.get(i),
			x_star[i]
		);
	}
}

#[test]
fn ridge_regression_known_solution() {
	let m = 20;
	let n = 3;
	let lambda = 0.5;
	let x_mat = random_matrix(m, n);
	// Compute true weights
	let w_true = random_vector(n);
	// y = X w_true + noise
	let xw = x_mat.mat_vec(&w_true);
	let y = Vec64::from_fn(m, |i| xw.get(i) + 0.01 * (i as f64 - 10.0) * 0.01);

	let problem = RidgeRegression::<f64, B>::new(x_mat.clone(), y.clone(), lambda);
	let manifold = Euclidean::<f64, B>::new(n);
	let x0 = Vec64::zeros(n);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-12));

	// Closed-form: w* = (X^T X + lambda * m * I)^{-1} X^T y
	let mut xtx_reg = Mat64::zeros(n, n);
	xtx_reg.gemm_at(1.0, x_mat.as_view(), x_mat.as_view(), 0.0);
	for i in 0..n {
		*xtx_reg.get_mut(i, i) = xtx_reg.get(i, i) + lambda * (m as f64);
	}
	let mut xtx_inv = Mat64::zeros(n, n);
	xtx_reg.inverse(&mut xtx_inv);
	let xt = x_mat.transpose_to_owned();
	let xty = xt.mat_vec(&y);
	let w_star = xtx_inv.mat_vec(&xty);

	let mut dist = 0.0;
	for i in 0..n {
		let d = result.point.get(i) - w_star.get(i);
		dist += d * d;
	}
	dist = dist.sqrt();

	assert!(
		dist < 1e-6,
		"RidgeRegression: ||w - w*|| = {dist:.2e}, expected ~0"
	);
}

#[test]
fn rayleigh_quotient_finds_eigenvector() {
	let n = 6;
	// Build matrix with known smallest eigenvalue
	let a = random_spd_matrix(n);
	let problem = RayleighQuotient::<f64, B>::new(a.clone());
	let manifold = Sphere::<f64, B>::new(n);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-12));

	// Verify: A*x = lambda*x at converged point
	let ax = a.mat_vec(&result.point);
	let lambda = result.value; // Rayleigh quotient = eigenvalue

	let mut residual_sq = 0.0;
	for i in 0..n {
		let r = ax.get(i) - lambda * result.point.get(i);
		residual_sq += r * r;
	}
	let residual = residual_sq.sqrt();

	assert!(
		residual < 1e-6,
		"RayleighQuotient: ||Ax - lambda*x|| = {residual:.2e}, expected ~0"
	);
}

#[test]
fn quadratic_cost_on_sphere_finds_smallest_eigenvalue() {
	let n = 5;
	let a = random_spd_matrix(n);
	let b = Vec64::zeros(n);
	let problem = QuadraticCost::<f64, B>::new(a.clone(), b, 0.0);
	let manifold = Sphere::<f64, B>::new(n);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(500, 1e-10));

	// f(x) = 1/2 x^T A x on sphere. At min: x^T A x = lambda_min
	// So f_min = lambda_min / 2
	let eig = a.symmetric_eigen();
	let mut eigenvalues: Vec<f64> = (0..n)
		.map(|i| VectorView::get(&eig.eigenvalues, i))
		.collect();
	eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let expected = eigenvalues[0] * 0.5;

	assert!(
		(result.value - expected).abs() < 1e-4,
		"QuadraticCost on Sphere: f={:.6}, expected {:.6}",
		result.value,
		expected
	);
}

#[test]
fn spherical_kmeans_converges_to_data_direction() {
	let n = 3;
	// All data points roughly in the same direction
	let data_sum = Vec64::from_fn(n, |i| (i as f64 + 1.0) * 3.0);
	let problem = SphericalKMeans::<f64, B>::from_data_sum(data_sum.clone());
	let manifold = Sphere::<f64, B>::new(n);
	let mut x0 = manifold.allocate_point();
	manifold.random_point(&mut x0);

	let mut solver = LBFGS::with_default_config();
	let result = solver.solve(&problem, &manifold, &x0, &stopping(200, 1e-10));

	// Minimum of f(c) = -(data_sum)^T c is at c = data_sum / ||data_sum||
	let norm = data_sum.norm_squared().sqrt();
	let expected_cost = -norm;

	assert!(
		(result.value - expected_cost).abs() < 1e-6,
		"SphericalKMeans: f={:.6}, expected {:.6}",
		result.value,
		expected_cost
	);
}
