//! End-to-end optimization benchmarks.
//!
//! This benchmark suite tests complete optimization algorithms on realistic
//! problems to measure overall performance and convergence characteristics.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    manifold::Manifold,
    memory::Workspace,
    retraction::{DefaultRetraction, Retraction},
    types::DVector,
    utils::test_helpers::{TestSphere, RayleighQuotient, SphericalRosenbrock},
};
use std::time::Duration;

/// Simple gradient descent step for benchmarking
fn gradient_descent_step<M, F>(
    manifold: &M,
    cost_fn: &F,
    retraction: &impl Retraction<f64>,
    point: &DVector<f64>,
    step_size: f64,
    workspace: &mut riemannopt_core::memory::workspace::Workspace<f64>,
    euclidean_grad: &mut DVector<f64>,
    riemannian_grad: &mut DVector<f64>,
    search_dir: &mut DVector<f64>,
    new_point: &mut DVector<f64>,
) -> Result<(f64, f64)>
where
    M: Manifold<f64, Point = DVector<f64>, TangentVector = DVector<f64>>,
    F: CostFunction<f64, Point = DVector<f64>, TangentVector = DVector<f64>>,
{
    // Compute cost and gradient using workspace
    let cost = cost_fn.cost_and_gradient(point, workspace, euclidean_grad)?;

    // Convert to Riemannian gradient
    manifold.euclidean_to_riemannian_gradient(point, euclidean_grad, riemannian_grad, workspace)?;
    let gradient_norm = manifold
        .inner_product(point, riemannian_grad, riemannian_grad)?
        .sqrt();

    // Compute search direction (negative gradient)
    search_dir.copy_from(riemannian_grad);
    search_dir.scale_mut(-step_size);

    // Take step using retraction
    retraction.retract(manifold, point, search_dir, new_point)?;

    Ok((cost, gradient_norm))
}

/// Run simple gradient descent
fn simple_gradient_descent<M, F>(
    manifold: &M,
    cost_fn: &F,
    retraction: &impl Retraction<f64>,
    initial_point: &DVector<f64>,
    step_size: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(DVector<f64>, f64, usize)>
where
    M: Manifold<f64, Point = DVector<f64>, TangentVector = DVector<f64>>,
    F: CostFunction<f64, Point = DVector<f64>, TangentVector = DVector<f64>>,
{
    let n = initial_point.len();
    let mut point = initial_point.clone();
    let mut iterations = 0;
    let mut final_cost = 0.0;
    
    // Pre-allocate workspace and buffers
    let mut workspace = riemannopt_core::memory::Workspace::with_size(n);
    let mut euclidean_grad = DVector::zeros(n);
    let mut riemannian_grad = DVector::zeros(n);
    let mut search_dir = DVector::zeros(n);
    let mut new_point = DVector::zeros(n);

    for i in 0..max_iterations {
        let (cost, grad_norm) = gradient_descent_step(
            manifold,
            cost_fn,
            retraction,
            &point,
            step_size,
            &mut workspace,
            &mut euclidean_grad,
            &mut riemannian_grad,
            &mut search_dir,
            &mut new_point,
        )?;

        // Swap buffers to avoid copying
        std::mem::swap(&mut point, &mut new_point);
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
            let (_, grad) = cost_fn.cost_and_gradient_alloc(&point).unwrap();
            black_box(grad)
        });
    });

    group.bench_function("gradient_conversion", |b| {
        let (_, euclidean_grad) = cost_fn.cost_and_gradient_alloc(&point).unwrap();
        b.iter(|| {
            let mut riemannian_grad = DVector::zeros(point.len());
            let mut workspace = Workspace::<f64>::new();
            manifold.euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut workspace).unwrap();
            black_box(riemannian_grad)
        });
    });

    group.bench_function("retraction", |b| {
        let mut tangent = DVector::zeros(point.len());
        let mut workspace = Workspace::new();
        manifold.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        tangent *= 0.01;
        b.iter(|| {
            let mut new_point = DVector::zeros(point.len());
            retraction.retract(&manifold, &point, &tangent, &mut new_point).unwrap();
            black_box(new_point)
        });
    });

    group.bench_function("inner_product", |b| {
        let mut u = DVector::zeros(point.len());
        let mut v = DVector::zeros(point.len());
        let mut workspace = Workspace::new();
        manifold.random_tangent(&point, &mut u, &mut workspace).unwrap();
        manifold.random_tangent(&point, &mut v, &mut workspace).unwrap();
        b.iter(|| {
            let result = manifold.inner_product(&point, &u, &v).unwrap();
            black_box(result)
        });
    });

    // Benchmark a full optimization step
    group.bench_function("full_step", |b| {
        let n = point.len();
        let mut workspace = riemannopt_core::memory::Workspace::with_size(n);
        let mut euclidean_grad = DVector::zeros(n);
        let mut riemannian_grad = DVector::zeros(n);
        let mut search_dir = DVector::zeros(n);
        let mut new_point = DVector::zeros(n);
        
        b.iter(|| {
            let (cost, grad_norm) = gradient_descent_step(
                &manifold,
                &cost_fn,
                &retraction,
                &point,
                0.01,
                &mut workspace,
                &mut euclidean_grad,
                &mut riemannian_grad,
                &mut search_dir,
                &mut new_point,
            ).unwrap();
            black_box((cost, grad_norm))
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
