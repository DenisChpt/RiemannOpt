//! Performance benchmarks for autodiff forward+backward cycle.
//!
//! Run with: `cargo bench -p riemannopt-autodiff`
//!
//! Compares:
//! - AutoDiff cost_and_gradient vs hand-coded analytical
//! - Session overhead across problem sizes
//! - Zero-alloc verification (2nd iteration should be as fast as 1st)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use riemannopt_core::linalg::{FaerBackend, LinAlgBackend, MatrixOps};
use riemannopt_core::manifold::{Euclidean, Manifold, Sphere};
use riemannopt_core::problem::Problem;

use riemannopt_autodiff::{AdSession, AutoDiffProblem};

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

// ════════════════════════════════════════════════════════════════════════
//  AutoDiff session: forward + backward micro-benchmarks
// ════════════════════════════════════════════════════════════════════════

fn bench_session_quadform(c: &mut Criterion) {
	let mut group = c.benchmark_group("ad_session_quadform");
	for &n in &[10, 50, 100, 500] {
		group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, &n| {
			let a = random_spd(n);
			let x = Vec64::from_fn(n, |i| (i as f64 + 1.0) * 0.01);
			let mut session = AdSession::<f64, B>::new();
			// Warmup to allocate buffers
			{
				let xv = session.input_vector(&x);
				let am = session.constant_matrix(&a);
				let loss = session.quad_form(xv, am);
				session.backward(loss);
			}
			b.iter(|| {
				session.reset();
				let xv = session.input_vector(&x);
				let am = session.constant_matrix(&a);
				let loss = session.quad_form(xv, am);
				session.backward(loss);
			});
		});
	}
	group.finish();
}

fn bench_session_matvec_dot(c: &mut Criterion) {
	let mut group = c.benchmark_group("ad_session_matvec_dot");
	for &n in &[10, 50, 100, 500] {
		group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, &n| {
			let a = random_spd(n);
			let x = Vec64::from_fn(n, |i| (i as f64 + 1.0) * 0.01);
			let mut session = AdSession::<f64, B>::new();
			// Warmup
			{
				let xv = session.input_vector(&x);
				let am = session.constant_matrix(&a);
				let ax = session.mat_vec(am, xv);
				let loss = session.dot(ax, ax);
				session.backward(loss);
			}
			b.iter(|| {
				session.reset();
				let xv = session.input_vector(&x);
				let am = session.constant_matrix(&a);
				let ax = session.mat_vec(am, xv);
				let loss = session.dot(ax, ax);
				session.backward(loss);
			});
		});
	}
	group.finish();
}

// ════════════════════════════════════════════════════════════════════════
//  AutoDiff Problem: cost_and_gradient vs hand-coded
// ════════════════════════════════════════════════════════════════════════

fn bench_rayleigh_autodiff_vs_analytical(c: &mut Criterion) {
	let mut group = c.benchmark_group("rayleigh_ad_vs_analytical");
	for &n in &[10, 50, 100] {
		let a = random_spd(n);
		let sphere = Sphere::<f64, B>::new(n);
		let mut point = sphere.allocate_point();
		sphere.random_point(&mut point);
		let mut grad = sphere.allocate_tangent();

		// Analytical
		{
			let a_clone = a.clone();
			let problem = riemannopt_core::problem::sphere::RayleighQuotient::<f64, B>::new(a_clone);
			let mut ws = problem.create_workspace(&sphere, &point);
			let mut man_ws = sphere.create_workspace(&point);
			group.bench_with_input(
				BenchmarkId::new("analytical", n),
				&n,
				|b, _| {
					b.iter(|| {
						problem.cost_and_gradient(
							&sphere, &point, &mut grad, &mut ws, &mut man_ws,
						)
					});
				},
			);
		}

		// AutoDiff
		{
			let a_clone = a.clone();
			let ad_problem = AutoDiffProblem::<f64, B, _>::new(move |s, xv| {
				let am = s.constant_matrix(&a_clone);
				s.quad_form(xv, am)
			});
			let mut ws = ad_problem.create_workspace(&sphere, &point);
			let mut man_ws = sphere.create_workspace(&point);
			// Warmup
			ad_problem.cost_and_gradient(
				&sphere, &point, &mut grad, &mut ws, &mut man_ws,
			);
			group.bench_with_input(
				BenchmarkId::new("autodiff", n),
				&n,
				|b, _| {
					b.iter(|| {
						ad_problem.cost_and_gradient(
							&sphere, &point, &mut grad, &mut ws, &mut man_ws,
						)
					});
				},
			);
		}
	}
	group.finish();
}

fn bench_ridge_autodiff_vs_analytical(c: &mut Criterion) {
	let mut group = c.benchmark_group("ridge_ad_vs_analytical");
	for &(m, n) in &[(100, 10), (500, 50)] {
		let label = format!("{m}x{n}");
		use rand::RngExt;
		let mut rng = rand::rng();
		let x_data = Mat64::from_fn(m, n, |_, _| rng.random_range(-1.0..1.0));
		let y = Vec64::from_fn(m, |_| rng.random_range(-1.0..1.0));
		let euclidean = Euclidean::<f64, B>::new(n);
		let point = Vec64::zeros(n);
		let mut grad = euclidean.allocate_tangent();

		// Analytical
		{
			let problem =
				riemannopt_core::problem::euclidean::RidgeRegression::<f64, B>::new(
					x_data.clone(), y.clone(), 0.01,
				);
			let mut ws = problem.create_workspace(&euclidean, &point);
			let mut man_ws = euclidean.create_workspace(&point);
			group.bench_with_input(
				BenchmarkId::new("analytical", &label),
				&(m, n),
				|b, _| {
					b.iter(|| {
						problem.cost_and_gradient(
							&euclidean, &point, &mut grad, &mut ws, &mut man_ws,
						)
					});
				},
			);
		}

		// AutoDiff
		{
			let x_cl = x_data.clone();
			let y_cl = y.clone();
			let m_f = m;
			let ad_problem = AutoDiffProblem::<f64, B, _>::new(move |s, wv| {
				let xm = s.constant_matrix(&x_cl);
				let yv = s.constant_vector(&y_cl);
				let pred = s.mat_vec(xm, wv);
				let residual = s.sub_v(pred, yv);
				let mse = s.dot(residual, residual);
				let inv_2m = s.constant_scalar(1.0 / (2.0 * m_f as f64));
				let mse_scaled = s.mul_s(inv_2m, mse);
				let reg_coeff = s.constant_scalar(0.005); // lambda/2
				let w_sq = s.dot(wv, wv);
				let reg = s.mul_s(reg_coeff, w_sq);
				s.add_s(mse_scaled, reg)
			});
			let mut ws = ad_problem.create_workspace(&euclidean, &point);
			let mut man_ws = euclidean.create_workspace(&point);
			// Warmup
			ad_problem.cost_and_gradient(
				&euclidean, &point, &mut grad, &mut ws, &mut man_ws,
			);
			group.bench_with_input(
				BenchmarkId::new("autodiff", &label),
				&(m, n),
				|b, _| {
					b.iter(|| {
						ad_problem.cost_and_gradient(
							&euclidean, &point, &mut grad, &mut ws, &mut man_ws,
						)
					});
				},
			);
		}
	}
	group.finish();
}

criterion_group!(
	ad_benches,
	bench_session_quadform,
	bench_session_matvec_dot,
	bench_rayleigh_autodiff_vs_analytical,
	bench_ridge_autodiff_vs_analytical,
);

criterion_main!(ad_benches);
