//! Algorithm comparison benchmarks.
//!
//! This benchmark suite compares different optimization algorithms
//! on the same problems to evaluate their relative performance.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nalgebra::Dyn;
use rand::prelude::*;
use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    manifold::Manifold,
    retraction::{DefaultRetraction, ExponentialRetraction, ProjectionRetraction, Retraction},
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

    fn project_point(&self, point: &DVector<f64>) -> DVector<f64> {
        let norm = point.norm();
        if norm > f64::EPSILON {
            point / norm
        } else {
            let mut p = DVector::zeros(self.dim);
            p[0] = 1.0;
            p
        }
    }

    fn project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>) -> Result<DVector<f64>> {
        let inner = point.dot(vector);
        Ok(vector - point * inner)
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        u: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<f64> {
        Ok(u.dot(v))
    }

    fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>) -> Result<DVector<f64>> {
        // Exponential map on sphere
        let norm_v = tangent.norm();
        if norm_v < f64::EPSILON {
            Ok(point.clone())
        } else {
            let cos_norm = norm_v.cos();
            let sin_norm = norm_v.sin();
            Ok(point * cos_norm + tangent * (sin_norm / norm_v))
        }
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>) -> Result<DVector<f64>> {
        let inner = point.dot(other).min(1.0).max(-1.0);
        let theta = inner.acos();

        if theta.abs() < f64::EPSILON {
            Ok(DVector::zeros(self.dim))
        } else {
            let v = other - point * inner;
            let v_norm = v.norm();
            if v_norm > f64::EPSILON {
                Ok(v * (theta / v_norm))
            } else {
                Ok(DVector::zeros(self.dim))
            }
        }
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        self.project_tangent(point, euclidean_grad)
    }

    fn random_point(&self) -> DVector<f64> {
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        self.project_point(&v)
    }

    fn random_tangent(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        self.project_tangent(point, &v)
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        self.project_tangent(to, vector)
    }

    fn has_exact_exp_log(&self) -> bool {
        true
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

/// Simple gradient descent
fn gradient_descent<M, F>(
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
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&point)?;
        let riemannian_grad = manifold.euclidean_to_riemannian_gradient(&point, &euclidean_grad)?;
        let gradient_norm = manifold
            .inner_product(&point, &riemannian_grad, &riemannian_grad)?
            .sqrt();

        final_cost = cost;
        iterations = i + 1;

        if gradient_norm < tolerance {
            break;
        }

        let search_dir = -&riemannian_grad * step_size;
        point = retraction.retract(manifold, &point, &search_dir)?;
    }

    Ok((point, final_cost, iterations))
}

/// Gradient descent with momentum
fn momentum_gradient_descent<M, F>(
    manifold: &M,
    cost_fn: &F,
    retraction: &impl Retraction<f64, Dyn>,
    initial_point: &DVector<f64>,
    step_size: f64,
    momentum: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(DVector<f64>, f64, usize)>
where
    M: Manifold<f64, Dyn>,
    F: CostFunction<f64, Dyn>,
{
    let mut point = initial_point.clone();
    let mut velocity = DVector::zeros(initial_point.len());
    let mut iterations = 0;
    let mut final_cost = 0.0;

    for i in 0..max_iterations {
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&point)?;
        let riemannian_grad = manifold.euclidean_to_riemannian_gradient(&point, &euclidean_grad)?;
        let gradient_norm = manifold
            .inner_product(&point, &riemannian_grad, &riemannian_grad)?
            .sqrt();

        final_cost = cost;
        iterations = i + 1;

        if gradient_norm < tolerance {
            break;
        }

        // Update velocity with momentum
        velocity = velocity * momentum - &riemannian_grad * step_size;

        // Ensure velocity is in tangent space
        velocity = manifold.project_tangent(&point, &velocity)?;

        // Take step
        let new_point = retraction.retract(manifold, &point, &velocity)?;

        // Transport velocity to new point
        velocity = manifold.parallel_transport(&point, &new_point, &velocity)?;

        point = new_point;
    }

    Ok((point, final_cost, iterations))
}

/// Nesterov accelerated gradient
fn nesterov_gradient_descent<M, F>(
    manifold: &M,
    cost_fn: &F,
    retraction: &impl Retraction<f64, Dyn>,
    initial_point: &DVector<f64>,
    step_size: f64,
    momentum: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(DVector<f64>, f64, usize)>
where
    M: Manifold<f64, Dyn>,
    F: CostFunction<f64, Dyn>,
{
    let mut point = initial_point.clone();
    let mut velocity = DVector::zeros(initial_point.len());
    let mut iterations = 0;
    let mut final_cost = 0.0;

    for i in 0..max_iterations {
        // Look ahead
        let look_ahead_velocity = manifold.project_tangent(&point, &velocity)?;
        let look_ahead_point =
            retraction.retract(manifold, &point, &(&look_ahead_velocity * momentum))?;

        // Compute gradient at look-ahead point
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&look_ahead_point)?;
        let riemannian_grad =
            manifold.euclidean_to_riemannian_gradient(&look_ahead_point, &euclidean_grad)?;

        // Transport gradient back to current point
        let transported_grad =
            manifold.parallel_transport(&look_ahead_point, &point, &riemannian_grad)?;
        let gradient_norm = manifold
            .inner_product(&point, &transported_grad, &transported_grad)?
            .sqrt();

        final_cost = cost;
        iterations = i + 1;

        if gradient_norm < tolerance {
            break;
        }

        // Update velocity
        velocity = velocity * momentum - &transported_grad * step_size;
        velocity = manifold.project_tangent(&point, &velocity)?;

        // Take step
        let new_point = retraction.retract(manifold, &point, &velocity)?;

        // Transport velocity to new point
        velocity = manifold.parallel_transport(&point, &new_point, &velocity)?;

        point = new_point;
    }

    Ok((point, final_cost, iterations))
}

fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    let dim = 50;
    let manifold = TestSphere::new(dim);
    let cost_fn = RayleighQuotient::new(dim);
    let retraction = DefaultRetraction;

    // Standard gradient descent
    group.bench_function("gradient_descent", |b| {
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = gradient_descent(
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

    // Gradient descent with momentum
    group.bench_function("momentum_gd", |b| {
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = momentum_gradient_descent(
                    &manifold,
                    &cost_fn,
                    &retraction,
                    &initial_point,
                    0.01,
                    0.9,
                    200,
                    1e-6,
                )
                .unwrap();
                black_box(result)
            },
            BatchSize::SmallInput,
        );
    });

    // Nesterov accelerated gradient
    group.bench_function("nesterov_gd", |b| {
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = nesterov_gradient_descent(
                    &manifold,
                    &cost_fn,
                    &retraction,
                    &initial_point,
                    0.01,
                    0.9,
                    200,
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

fn bench_retraction_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("retraction_comparison");
    group.sample_size(20);

    let dim = 50;
    let manifold = TestSphere::new(dim);
    let cost_fn = RayleighQuotient::new(dim);

    // Default retraction
    group.bench_function("default_retraction", |b| {
        let retraction = DefaultRetraction;
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = gradient_descent(
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

    // Projection retraction
    group.bench_function("projection_retraction", |b| {
        let retraction = ProjectionRetraction;
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = gradient_descent(
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

    // Exponential retraction
    group.bench_function("exponential_retraction", |b| {
        let retraction = ExponentialRetraction::new();
        b.iter_batched(
            || manifold.random_point(),
            |initial_point| {
                let result = gradient_descent(
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

fn bench_momentum_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("momentum_values");

    let dim = 50;
    let manifold = TestSphere::new(dim);
    let cost_fn = RayleighQuotient::new(dim);
    let retraction = DefaultRetraction;

    for momentum in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99] {
        group.bench_with_input(
            BenchmarkId::new("momentum", momentum),
            &momentum,
            |b, &mom| {
                b.iter_batched(
                    || manifold.random_point(),
                    |initial_point| {
                        let result = momentum_gradient_descent(
                            &manifold,
                            &cost_fn,
                            &retraction,
                            &initial_point,
                            0.01,
                            mom,
                            150,
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

fn bench_algorithm_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_scaling");
    group.sample_size(10);

    let retraction = DefaultRetraction;

    for dim in [10, 50, 100, 200] {
        let manifold = TestSphere::new(dim);
        let cost_fn = RayleighQuotient::new(dim);

        // Gradient descent
        group.bench_with_input(BenchmarkId::new("gradient_descent", dim), &dim, |b, _| {
            b.iter_batched(
                || manifold.random_point(),
                |initial_point| {
                    let result = gradient_descent(
                        &manifold,
                        &cost_fn,
                        &retraction,
                        &initial_point,
                        0.01,
                        100,
                        1e-4,
                    )
                    .unwrap();
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });

        // Momentum
        group.bench_with_input(BenchmarkId::new("momentum", dim), &dim, |b, _| {
            b.iter_batched(
                || manifold.random_point(),
                |initial_point| {
                    let result = momentum_gradient_descent(
                        &manifold,
                        &cost_fn,
                        &retraction,
                        &initial_point,
                        0.01,
                        0.9,
                        100,
                        1e-4,
                    )
                    .unwrap();
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });

        // Nesterov
        group.bench_with_input(BenchmarkId::new("nesterov", dim), &dim, |b, _| {
            b.iter_batched(
                || manifold.random_point(),
                |initial_point| {
                    let result = nesterov_gradient_descent(
                        &manifold,
                        &cost_fn,
                        &retraction,
                        &initial_point,
                        0.01,
                        0.9,
                        100,
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

criterion_group!(
    benches,
    bench_algorithm_comparison,
    bench_retraction_comparison,
    bench_momentum_values,
    bench_algorithm_scaling
);
criterion_main!(benches);
