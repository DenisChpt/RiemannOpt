//! End-to-end optimization benchmarks.
//!
//! This benchmark suite tests complete optimization algorithms on realistic
//! problems to measure overall performance and convergence characteristics.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nalgebra::Dyn;
use rand::prelude::*;
use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    manifold::Manifold,
    retraction::{DefaultRetraction, Retraction},
    types::{DMatrix, DVector},
};
use std::time::Duration;

/// Test sphere manifold for optimization problems
#[derive(Debug, Clone)]
struct TestSphere {
    dim: usize,
}

impl TestSphere {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for TestSphere {
    fn name(&self) -> &str {
        "Test Sphere"
    }

    fn dimension(&self) -> usize {
        self.dim - 1
    }

    fn is_point_on_manifold(&self, point: &DVector<f64>, tol: f64) -> bool {
        (point.norm() - 1.0).abs() < tol
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<f64>,
        vector: &DVector<f64>,
        tol: f64,
    ) -> bool {
        point.dot(vector).abs() < tol
    }

    fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>) {
        let norm = point.norm();
        if norm > f64::EPSILON {
            *result = point / norm;
        } else {
            result.fill(0.0);
            result[0] = 1.0;
        }
    }

    fn project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let inner = point.dot(vector);
        *result = vector - point * inner;
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        u: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<f64> {
        Ok(u.dot(v))
    }

    fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        // Exponential map on sphere
        let norm_v = tangent.norm();
        if norm_v < f64::EPSILON {
            *result = point.clone();
        } else {
            let cos_norm = norm_v.cos();
            let sin_norm = norm_v.sin();
            *result = point * cos_norm + tangent * (sin_norm / norm_v);
        }
        Ok(())
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let inner = point.dot(other).min(1.0).max(-1.0);
        let theta = inner.acos();

        if theta.abs() < f64::EPSILON {
            result.fill(0.0);
        } else {
            let v = other - point * inner;
            let v_norm = v.norm();
            if v_norm > f64::EPSILON {
                *result = v * (theta / v_norm);
            } else {
                result.fill(0.0);
            }
        }
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
        result: &mut DVector<f64>,
    ) -> Result<()> {
        self.project_tangent(point, euclidean_grad, result)
    }

    fn random_point(&self) -> DVector<f64> {
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        let mut result = DVector::zeros(self.dim);
        self.project_point(&v, &mut result);
        result
    }

    fn random_tangent(&self, point: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        self.project_tangent(point, &v, result)
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
    ) -> Result<()> {
        self.project_tangent(to, vector, result)
    }
}

/// Rayleigh quotient minimization problem
#[derive(Debug)]
struct RayleighQuotient {
    matrix: DMatrix<f64>,
}

impl RayleighQuotient {
    fn new(dim: usize) -> Self {
        let mut rng = thread_rng();
        let mut matrix = DMatrix::from_fn(dim, dim, |_, _| rng.gen::<f64>());
        // Make symmetric
        matrix = &matrix + &matrix.transpose();
        Self { matrix }
    }
}

impl CostFunction<f64, Dyn> for RayleighQuotient {
    fn cost(&self, x: &DVector<f64>) -> Result<f64> {
        let ax = &self.matrix * x;
        Ok(x.dot(&ax))
    }

    fn cost_and_gradient(&self, x: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        let ax = &self.matrix * x;
        let cost = x.dot(&ax);
        let gradient = 2.0 * ax;
        Ok((cost, gradient))
    }
}

/// Rosenbrock function on sphere
#[derive(Debug)]
struct SphericalRosenbrock {
    dim: usize,
}

impl SphericalRosenbrock {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl CostFunction<f64, Dyn> for SphericalRosenbrock {
    fn cost(&self, x: &DVector<f64>) -> Result<f64> {
        let mut cost = 0.0;
        for i in 0..self.dim - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            cost += a * a + 100.0 * b * b;
        }
        Ok(cost)
    }

    fn cost_and_gradient(&self, x: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        let mut cost = 0.0;
        let mut gradient = DVector::zeros(self.dim);

        for i in 0..self.dim - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            cost += a * a + 100.0 * b * b;

            gradient[i] += -2.0 * a - 400.0 * x[i] * b;
            if i > 0 {
                gradient[i] += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
            }
        }
        if self.dim > 1 {
            gradient[self.dim - 1] += 200.0 * (x[self.dim - 1] - x[self.dim - 2] * x[self.dim - 2]);
        }

        Ok((cost, gradient))
    }
}

/// Simple gradient descent step for benchmarking
fn gradient_descent_step<M, F>(
    manifold: &M,
    cost_fn: &F,
    retraction: &impl Retraction<f64, Dyn>,
    point: &DVector<f64>,
    step_size: f64,
) -> Result<(DVector<f64>, f64, f64)>
where
    M: Manifold<f64, Dyn>,
    F: CostFunction<f64, Dyn>,
{
    // Compute cost and gradient
    let (cost, euclidean_grad) = cost_fn.cost_and_gradient(point)?;

    // Convert to Riemannian gradient
    let mut riemannian_grad = DVector::zeros(point.len());
    manifold.euclidean_to_riemannian_gradient(point, &euclidean_grad, &mut riemannian_grad)?;
    let gradient_norm = manifold
        .inner_product(point, &riemannian_grad, &riemannian_grad)?
        .sqrt();

    // Compute search direction (negative gradient)
    let search_dir = -&riemannian_grad * step_size;

    // Take step using retraction
    let new_point = retraction.retract(manifold, point, &search_dir)?;

    Ok((new_point, cost, gradient_norm))
}

/// Run simple gradient descent
fn simple_gradient_descent<M, F>(
    manifold: &M,
    cost_fn: &F,
    retraction: &impl Retraction<f64, Dyn>,
    initial_point: &DVector<f64>,
    step_size: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(DVector<f64>, f64, usize)>
where
    M: Manifold<f64, Dyn>,
    F: CostFunction<f64, Dyn>,
{
    let mut point = initial_point.clone();
    let mut iterations = 0;
    let mut final_cost = 0.0;

    for i in 0..max_iterations {
        let (new_point, cost, grad_norm) =
            gradient_descent_step(manifold, cost_fn, retraction, &point, step_size)?;

        point = new_point;
        final_cost = cost;
        iterations = i + 1;

        if grad_norm < tolerance {
            break;
        }
    }

    Ok((point, final_cost, iterations))
}

fn bench_rayleigh_quotient(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayleigh_quotient");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for dim in [10, 50, 100] {
        let manifold = TestSphere::new(dim);
        let cost_fn = RayleighQuotient::new(dim);
        let retraction = DefaultRetraction;

        // Benchmark with fixed step size
        group.bench_with_input(BenchmarkId::new("fixed_step", dim), &dim, |b, _| {
            b.iter_batched(
                || manifold.random_point(),
                |initial_point| {
                    let result = simple_gradient_descent(
                        &manifold,
                        &cost_fn,
                        &retraction,
                        &initial_point,
                        0.01,
                        100,
                        1e-6,
                    )
                    .unwrap();
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });

        // Benchmark with different step sizes
        for step_size in [0.001, 0.01, 0.1] {
            group.bench_with_input(
                BenchmarkId::new(format!("step_{}", step_size), dim),
                &dim,
                |b, _| {
                    b.iter_batched(
                        || manifold.random_point(),
                        |initial_point| {
                            let result = simple_gradient_descent(
                                &manifold,
                                &cost_fn,
                                &retraction,
                                &initial_point,
                                step_size,
                                100,
                                1e-6,
                            )
                            .unwrap();
                            black_box(result)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_spherical_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("spherical_rosenbrock");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for dim in [10, 20, 50] {
        let manifold = TestSphere::new(dim);
        let cost_fn = SphericalRosenbrock::new(dim);
        let retraction = DefaultRetraction;

        group.bench_with_input(BenchmarkId::new("gradient_descent", dim), &dim, |b, _| {
            b.iter_batched(
                || manifold.random_point(),
                |initial_point| {
                    let result = simple_gradient_descent(
                        &manifold,
                        &cost_fn,
                        &retraction,
                        &initial_point,
                        0.01,
                        200,
                        1e-6,
                    )
                    .unwrap();
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_convergence_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence_analysis");
    group.sample_size(10);

    let dim = 50;
    let manifold = TestSphere::new(dim);
    let retraction = DefaultRetraction;

    // Test different tolerance levels
    for tolerance in [1e-3, 1e-6, 1e-9] {
        let cost_fn = RayleighQuotient::new(dim);

        group.bench_with_input(
            BenchmarkId::new("tolerance", format!("{:e}", tolerance)),
            &tolerance,
            |b, &tol| {
                b.iter_batched(
                    || manifold.random_point(),
                    |initial_point| {
                        let result = simple_gradient_descent(
                            &manifold,
                            &cost_fn,
                            &retraction,
                            &initial_point,
                            0.01,
                            1000,
                            tol,
                        )
                        .unwrap();
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_step_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_size_impact");

    let dim = 50;
    let manifold = TestSphere::new(dim);
    let cost_fn = RayleighQuotient::new(dim);
    let retraction = DefaultRetraction;

    // Different step sizes
    for step_size in [0.0001, 0.001, 0.01, 0.1, 0.5] {
        group.bench_with_input(
            BenchmarkId::new("step", step_size),
            &step_size,
            |b, &step| {
                b.iter_batched(
                    || manifold.random_point(),
                    |initial_point| {
                        let result = simple_gradient_descent(
                            &manifold,
                            &cost_fn,
                            &retraction,
                            &initial_point,
                            step,
                            100,
                            1e-6,
                        )
                        .unwrap();
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_large_scale_problems(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_problems");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let retraction = DefaultRetraction;

    for dim in [100, 500, 1000] {
        let manifold = TestSphere::new(dim);
        let cost_fn = RayleighQuotient::new(dim);

        group.bench_with_input(BenchmarkId::new("rayleigh_quotient", dim), &dim, |b, _| {
            b.iter_batched(
                || manifold.random_point(),
                |initial_point| {
                    let result = simple_gradient_descent(
                        &manifold,
                        &cost_fn,
                        &retraction,
                        &initial_point,
                        1.0 / (dim as f64).sqrt(),
                        50,
                        1e-4,
                    )
                    .unwrap();
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark to profile hot paths in optimization
fn bench_hot_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_paths");

    let dim = 100;
    let manifold = TestSphere::new(dim);
    let cost_fn = RayleighQuotient::new(dim);
    let retraction = DefaultRetraction;
    let point = manifold.random_point();

    // Benchmark individual operations that are called frequently
    group.bench_function("gradient_computation", |b| {
        b.iter(|| {
            let (_, grad) = cost_fn.cost_and_gradient(&point).unwrap();
            black_box(grad)
        });
    });

    group.bench_function("gradient_conversion", |b| {
        let (_, euclidean_grad) = cost_fn.cost_and_gradient(&point).unwrap();
        b.iter(|| {
            let mut riemannian_grad = DVector::zeros(point.len());
            manifold.euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad).unwrap();
            black_box(riemannian_grad)
        });
    });

    group.bench_function("retraction", |b| {
        let mut tangent = DVector::zeros(point.len());
        manifold.random_tangent(&point, &mut tangent).unwrap();
        tangent *= 0.01;
        b.iter(|| {
            let new_point = retraction.retract(&manifold, &point, &tangent).unwrap();
            black_box(new_point)
        });
    });

    group.bench_function("inner_product", |b| {
        let mut u = DVector::zeros(point.len());
        let mut v = DVector::zeros(point.len());
        manifold.random_tangent(&point, &mut u).unwrap();
        manifold.random_tangent(&point, &mut v).unwrap();
        b.iter(|| {
            let result = manifold.inner_product(&point, &u, &v).unwrap();
            black_box(result)
        });
    });

    // Benchmark a full optimization step
    group.bench_function("full_step", |b| {
        b.iter(|| {
            let (new_point, _, _) =
                gradient_descent_step(&manifold, &cost_fn, &retraction, &point, 0.01).unwrap();
            black_box(new_point)
        });
    });

    group.finish();
}

/// Compare different optimization problems
fn bench_problem_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("problem_comparison");

    let dim = 50;
    let manifold = TestSphere::new(dim);
    let retraction = DefaultRetraction;

    // Rayleigh quotient
    group.bench_function("rayleigh_quotient", |b| {
        let cost_fn = RayleighQuotient::new(dim);
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = simple_gradient_descent(
                    &manifold,
                    &cost_fn,
                    &retraction,
                    &initial_point,
                    0.01,
                    100,
                    1e-6,
                )
                .unwrap();
                black_box(result)
            },
            BatchSize::SmallInput,
        );
    });

    // Spherical Rosenbrock
    group.bench_function("spherical_rosenbrock", |b| {
        let cost_fn = SphericalRosenbrock::new(dim);
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = simple_gradient_descent(
                    &manifold,
                    &cost_fn,
                    &retraction,
                    &initial_point,
                    0.01,
                    100,
                    1e-6,
                )
                .unwrap();
                black_box(result)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rayleigh_quotient,
    bench_spherical_rosenbrock,
    bench_convergence_analysis,
    bench_step_size_impact,
    bench_large_scale_problems,
    bench_hot_paths,
    bench_problem_comparison
);
criterion_main!(benches);
