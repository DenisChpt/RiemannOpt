//! Memory profiling benchmarks.
//!
//! This benchmark suite focuses on memory allocation patterns and
//! efficiency of various manifold operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Dyn;
use riemannopt_core::{
    manifold::Manifold,
    optimizer_state::{AdamState, LBFGSState, MomentumState},
    types::DVector,
};

/// Memory-efficient sphere implementation
#[derive(Debug)]
struct MemoryTestSphere {
    dim: usize,
}

impl MemoryTestSphere {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for MemoryTestSphere {
    fn name(&self) -> &str {
        "Memory Test Sphere"
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

    fn project_tangent(
        &self,
        point: &DVector<f64>,
        vector: &DVector<f64>,
    ) -> riemannopt_core::error::Result<DVector<f64>> {
        let inner = point.dot(vector);
        Ok(vector - point * inner)
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        u: &DVector<f64>,
        v: &DVector<f64>,
    ) -> riemannopt_core::error::Result<f64> {
        Ok(u.dot(v))
    }

    fn retract(
        &self,
        point: &DVector<f64>,
        tangent: &DVector<f64>,
    ) -> riemannopt_core::error::Result<DVector<f64>> {
        let new_point = point + tangent;
        Ok(self.project_point(&new_point))
    }

    fn inverse_retract(
        &self,
        point: &DVector<f64>,
        other: &DVector<f64>,
    ) -> riemannopt_core::error::Result<DVector<f64>> {
        let diff = other - point;
        self.project_tangent(point, &diff)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
    ) -> riemannopt_core::error::Result<DVector<f64>> {
        self.project_tangent(point, euclidean_grad)
    }

    fn random_point(&self) -> DVector<f64> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        self.project_point(&v)
    }

    fn random_tangent(&self, point: &DVector<f64>) -> riemannopt_core::error::Result<DVector<f64>> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        self.project_tangent(point, &v)
    }
}

fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    for dim in [100, 500, 1000, 5000] {
        let sphere = MemoryTestSphere::new(dim);

        // Benchmark repeated allocations in projection
        group.bench_with_input(
            BenchmarkId::new("repeated_projections", dim),
            &dim,
            |b, _| {
                let points: Vec<DVector<f64>> = (0..10)
                    .map(|_| DVector::from_fn(dim, |_, _| rand::random::<f64>()))
                    .collect();

                let mut idx = 0;
                b.iter(|| {
                    // This allocates a new vector each time
                    let result = sphere.project_point(black_box(&points[idx % points.len()]));
                    idx += 1;
                    result
                });
            },
        );

        // Benchmark in-place operations (simulated)
        group.bench_with_input(
            BenchmarkId::new("in_place_operations", dim),
            &dim,
            |b, _| {
                let mut point = sphere.random_point();
                let tangents: Vec<DVector<f64>> = (0..10)
                    .map(|_| sphere.random_tangent(&point).unwrap() * 0.01)
                    .collect();

                let mut idx = 0;
                b.iter(|| {
                    // Simulate in-place retraction
                    let tangent = &tangents[idx % tangents.len()];
                    point += tangent;
                    let norm = point.norm();
                    point /= norm;
                    idx += 1;
                    black_box(point.norm()) // Return a value instead of reference
                });
            },
        );
    }

    group.finish();
}

fn bench_optimizer_state_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_state_memory");

    for dim in [100, 1000, 10000] {
        // Benchmark momentum state
        group.bench_with_input(
            BenchmarkId::new("momentum_state_init", dim),
            &dim,
            |b, &dim| {
                b.iter(|| {
                    let state = MomentumState::<f64, Dyn>::new(dim as f64, false);
                    black_box(state)
                });
            },
        );

        // Benchmark Adam state (requires more memory)
        group.bench_with_input(BenchmarkId::new("adam_state_init", dim), &dim, |b, &dim| {
            b.iter(|| {
                let state = AdamState::<f64, Dyn>::new(dim as f64, 0.9, 0.999, false);
                black_box(state)
            });
        });

        // Benchmark L-BFGS state with different memory sizes
        for memory_size in [5, 10, 20] {
            group.bench_with_input(
                BenchmarkId::new(format!("lbfgs_state_init_m{}", memory_size), dim),
                &dim,
                |b, &dim| {
                    b.iter(|| {
                        let state = LBFGSState::<f64, Dyn>::new(dim);
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_vector_cloning(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_cloning");

    for dim in [100, 1000, 10000] {
        let vector = DVector::<f64>::from_fn(dim, |_, _| rand::random());

        group.bench_with_input(BenchmarkId::new("clone", dim), &dim, |b, _| {
            b.iter(|| {
                let cloned = black_box(&vector).clone();
                black_box(cloned)
            });
        });

        // Benchmark view creation (no allocation)
        group.bench_with_input(BenchmarkId::new("view", dim), &dim, |b, _| {
            b.iter(|| {
                let view = black_box(&vector).as_slice();
                black_box(view)
            });
        });
    }

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let dim = 1000;
    let sphere = MemoryTestSphere::new(dim);

    for batch_size in [1, 10, 100, 1000] {
        // Generate batch data
        let points: Vec<DVector<f64>> = (0..batch_size).map(|_| sphere.random_point()).collect();

        let tangents: Vec<DVector<f64>> = points
            .iter()
            .map(|p| sphere.random_tangent(p).unwrap() * 0.1)
            .collect();

        // Benchmark sequential processing
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<DVector<f64>> = points
                        .iter()
                        .zip(tangents.iter())
                        .map(|(p, t)| sphere.retract(p, t).unwrap())
                        .collect();
                    black_box(results)
                });
            },
        );

        // Benchmark pre-allocated processing
        group.bench_with_input(
            BenchmarkId::new("preallocated", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(batch_size);
                    for (p, t) in points.iter().zip(tangents.iter()) {
                        results.push(sphere.retract(p, t).unwrap());
                    }
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

fn bench_temporary_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporary_allocations");

    let dim = 1000;
    let sphere = MemoryTestSphere::new(dim);
    let point = sphere.random_point();

    // Benchmark operations that create temporaries
    group.bench_function("gradient_conversion_temps", |b| {
        let euclidean_grad = DVector::from_fn(dim, |_, _| rand::random::<f64>());

        b.iter(|| {
            // This creates temporaries: point * inner
            let result = sphere
                .euclidean_to_riemannian_gradient(&point, &euclidean_grad)
                .unwrap();
            black_box(result)
        });
    });

    // Benchmark chained operations
    group.bench_function("chained_operations", |b| {
        let v1 = sphere.random_tangent(&point).unwrap();
        let v2 = sphere.random_tangent(&point).unwrap();

        b.iter(|| {
            // Multiple temporaries created
            let result = &(&v1 * 2.0) + &(&v2 * 3.0);
            let normalized = &result / result.norm();
            black_box(normalized)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_allocation_patterns,
    bench_optimizer_state_memory,
    bench_vector_cloning,
    bench_batch_operations,
    bench_temporary_allocations
);
criterion_main!(benches);
