//! Performance benchmarks for solvers × problems × manifolds.
//!
//! Run with: `cargo bench -p riemannopt-core`
//!
//! These benchmarks serve as a regression baseline.  Any significant
//! slowdown after a refactor should be investigated.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use riemannopt_core::linalg::{FaerBackend, LinAlgBackend, MatrixOps};
use riemannopt_core::manifold::{Euclidean, Grassmann, Manifold, Sphere, Stiefel};
use riemannopt_core::prelude::*;

type B = FaerBackend;
type Vec64 = <B as LinAlgBackend<f64>>::Vector;
type Mat64 = <B as LinAlgBackend<f64>>::Matrix;

fn random_spd(n: usize) -> Mat64 {
	use rand::RngExt;
	let mut rng = rand::rng();
	let a = Mat64::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0));
	let mut r = Mat64::zeros(n, n);
	r.gemm_at(1.0, a.as_view(), a.as_view(), 0.0);
	for i in 0..n {
		*r.get_mut(i, i) = r.get(i, i) + 1.0;
	}
	r
}

fn random_data(m: usize, n: usize) -> (Mat64, Vec64) {
	use rand::RngExt;
	let mut rng = rand::rng();
	let x = Mat64::from_fn(m, n, |_, _| rng.random_range(-1.0..1.0));
	let y = Vec64::from_fn(m, |_| rng.random_range(-1.0..1.0));
	(x, y)
}

// ════════════════════════════════════════════════════════════════════════
//  Solver benchmarks — measures full solve() time
// ════════════════════════════════════════════════════════════════════════

fn bench_rosenbrock_lbfgs(c: &mut Criterion) {
	let mut group = c.benchmark_group("rosenbrock_lbfgs");
	for &n in &[2, 4, 10, 20] {
		group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, &n| {
			let euclidean = Euclidean::<f64, B>::new(n);
			let problem = Rosenbrock::<f64, B>::new();
			let mut solver = LBFGS::new(LBFGSConfig::new());
			let point = Vec64::from_fn(n, |_| 0.0);
			let stop = StoppingCriterion::new()
				.with_max_iterations(200)
				.with_gradient_tolerance(1e-8);
			b.iter(|| solver.solve(&problem, &euclidean, &point, &stop));
		});
	}
	group.finish();
}

fn bench_rayleigh_lbfgs(c: &mut Criterion) {
	let mut group = c.benchmark_group("rayleigh_lbfgs");
	for &n in &[10, 50, 100, 200] {
		group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, &n| {
			let sphere = Sphere::<f64, B>::new(n);
			let a = random_spd(n);
			let problem = RayleighQuotient::<f64, B>::new(a);
			let mut solver = LBFGS::new(LBFGSConfig::new());
			let mut point = sphere.allocate_point();
			sphere.random_point(&mut point);
			let stop = StoppingCriterion::new()
				.with_max_iterations(200)
				.with_gradient_tolerance(1e-6);
			b.iter(|| solver.solve(&problem, &sphere, &point, &stop));
		});
	}
	group.finish();
}

fn bench_rayleigh_trust_region(c: &mut Criterion) {
	let mut group = c.benchmark_group("rayleigh_trust_region");
	for &n in &[10, 50, 100] {
		group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, &n| {
			let sphere = Sphere::<f64, B>::new(n);
			let a = random_spd(n);
			let problem = RayleighQuotient::<f64, B>::new(a);
			let mut solver = TrustRegion::with_default_config();
			let mut point = sphere.allocate_point();
			sphere.random_point(&mut point);
			let stop = StoppingCriterion::new()
				.with_max_iterations(100)
				.with_gradient_tolerance(1e-6);
			b.iter(|| solver.solve(&problem, &sphere, &point, &stop));
		});
	}
	group.finish();
}

fn bench_ridge_regression_lbfgs(c: &mut Criterion) {
	let mut group = c.benchmark_group("ridge_lbfgs");
	for &(m, n) in &[(50, 5), (200, 20), (500, 50)] {
		let label = format!("{m}x{n}");
		group.bench_with_input(BenchmarkId::new("size", &label), &(m, n), |b, &(m, n)| {
			let (x, y) = random_data(m, n);
			let euclidean = Euclidean::<f64, B>::new(n);
			let problem = RidgeRegression::<f64, B>::new(x, y, 0.01);
			let mut solver = LBFGS::new(LBFGSConfig::new());
			let point = Vec64::zeros(n);
			let stop = StoppingCriterion::new()
				.with_max_iterations(100)
				.with_gradient_tolerance(1e-8);
			b.iter(|| solver.solve(&problem, &euclidean, &point, &stop));
		});
	}
	group.finish();
}

fn bench_procrustes_lbfgs(c: &mut Criterion) {
	let mut group = c.benchmark_group("procrustes_lbfgs");
	for &(n, p) in &[(10, 3), (30, 5), (50, 10)] {
		let label = format!("{n}x{p}");
		group.bench_with_input(
			BenchmarkId::new("size", &label),
			&(n, p),
			|b, &(n, p)| {
				use rand::RngExt;
				let mut rng = rand::rng();
				let a = Mat64::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0));
				let b_mat = Mat64::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
				let stiefel = Stiefel::<f64, B>::new(n, p);
				let problem = OrthogonalProcrustes::<f64, B>::new(a, b_mat);
				let mut solver = LBFGS::new(LBFGSConfig::new());
				let mut point = stiefel.allocate_point();
				stiefel.random_point(&mut point);
				let stop = StoppingCriterion::new()
					.with_max_iterations(200)
					.with_gradient_tolerance(1e-6);
				b.iter(|| solver.solve(&problem, &stiefel, &point, &stop));
			},
		);
	}
	group.finish();
}

fn bench_brockett_grassmann_lbfgs(c: &mut Criterion) {
	let mut group = c.benchmark_group("brockett_grassmann_lbfgs");
	for &(n, p) in &[(10, 2), (30, 5), (50, 10)] {
		let label = format!("{n}x{p}");
		group.bench_with_input(
			BenchmarkId::new("size", &label),
			&(n, p),
			|b, &(n, p)| {
				let a = random_spd(n);
				let grassmann = Grassmann::<f64, B>::new(n, p);
				let problem = BrockettCost::<f64, B>::new(a);
				let mut solver = LBFGS::new(LBFGSConfig::new());
				let mut point = grassmann.allocate_point();
				grassmann.random_point(&mut point);
				let stop = StoppingCriterion::new()
					.with_max_iterations(200)
					.with_gradient_tolerance(1e-6);
				b.iter(|| solver.solve(&problem, &grassmann, &point, &stop));
			},
		);
	}
	group.finish();
}

// ════════════════════════════════════════════════════════════════════════
//  cost_and_gradient micro-benchmarks (single evaluation)
// ════════════════════════════════════════════════════════════════════════

fn bench_cost_gradient_rayleigh(c: &mut Criterion) {
	let mut group = c.benchmark_group("cost_grad_rayleigh");
	for &n in &[10, 50, 100, 500] {
		group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, &n| {
			let sphere = Sphere::<f64, B>::new(n);
			let a = random_spd(n);
			let problem = RayleighQuotient::<f64, B>::new(a);
			let mut point = sphere.allocate_point();
			sphere.random_point(&mut point);
			let mut grad = sphere.allocate_tangent();
			let mut prob_ws = problem.create_workspace(&sphere, &point);
			let mut man_ws = sphere.create_workspace(&point);
			b.iter(|| {
				problem.cost_and_gradient(
					&sphere, &point, &mut grad, &mut prob_ws, &mut man_ws,
				)
			});
		});
	}
	group.finish();
}

fn bench_cost_gradient_ridge(c: &mut Criterion) {
	let mut group = c.benchmark_group("cost_grad_ridge");
	for &(m, n) in &[(100, 10), (500, 50), (1000, 100)] {
		let label = format!("{m}x{n}");
		group.bench_with_input(BenchmarkId::new("size", &label), &(m, n), |b, &(m, n)| {
			let (x, y) = random_data(m, n);
			let euclidean = Euclidean::<f64, B>::new(n);
			let problem = RidgeRegression::<f64, B>::new(x, y, 0.01);
			let point = Vec64::zeros(n);
			let mut grad = euclidean.allocate_tangent();
			let mut prob_ws = problem.create_workspace(&euclidean, &point);
			let mut man_ws = euclidean.create_workspace(&point);
			b.iter(|| {
				problem.cost_and_gradient(
					&euclidean, &point, &mut grad, &mut prob_ws, &mut man_ws,
				)
			});
		});
	}
	group.finish();
}

fn bench_cost_gradient_procrustes(c: &mut Criterion) {
	let mut group = c.benchmark_group("cost_grad_procrustes");
	for &(n, p) in &[(20, 5), (50, 10), (100, 20)] {
		let label = format!("{n}x{p}");
		group.bench_with_input(
			BenchmarkId::new("size", &label),
			&(n, p),
			|b, &(n, p)| {
				use rand::RngExt;
				let mut rng = rand::rng();
				let a = Mat64::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0));
				let b_mat = Mat64::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
				let stiefel = Stiefel::<f64, B>::new(n, p);
				let problem = OrthogonalProcrustes::<f64, B>::new(a, b_mat);
				let mut point = stiefel.allocate_point();
				stiefel.random_point(&mut point);
				let mut grad = stiefel.allocate_tangent();
				let mut prob_ws = problem.create_workspace(&stiefel, &point);
				let mut man_ws = stiefel.create_workspace(&point);
				b.iter(|| {
					problem.cost_and_gradient(
						&stiefel, &point, &mut grad, &mut prob_ws, &mut man_ws,
					)
				});
			},
		);
	}
	group.finish();
}

criterion_group!(
	solver_benches,
	bench_rosenbrock_lbfgs,
	bench_rayleigh_lbfgs,
	bench_rayleigh_trust_region,
	bench_ridge_regression_lbfgs,
	bench_procrustes_lbfgs,
	bench_brockett_grassmann_lbfgs,
);

criterion_group!(
	cost_grad_benches,
	bench_cost_gradient_rayleigh,
	bench_cost_gradient_ridge,
	bench_cost_gradient_procrustes,
);

criterion_main!(solver_benches, cost_grad_benches);
