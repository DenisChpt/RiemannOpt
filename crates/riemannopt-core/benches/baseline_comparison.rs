//! Baseline comparison benchmarks.
//!
//! This benchmark suite compares our implementations against naive
//! baseline implementations to measure performance improvements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Dyn;
use rand::prelude::*;
use riemannopt_core::{error::Result, manifold::Manifold, types::DVector};

/// Naive sphere implementation for baseline comparison
#[derive(Debug)]
struct NaiveSphere {
    _dim: usize,
}

impl NaiveSphere {
    fn new(dim: usize) -> Self {
        Self { _dim: dim }
    }

    // Naive projection that always normalizes
    fn naive_project_point(&self, point: &DVector<f64>) -> DVector<f64> {
        point / point.norm()
    }

    // Naive tangent projection without optimization
    fn naive_project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>) -> DVector<f64> {
        let inner = point.dot(vector);
        vector - point * inner
    }

    // Naive retraction without special cases
    fn naive_retract(&self, point: &DVector<f64>, tangent: &DVector<f64>) -> DVector<f64> {
        let new_point = point + tangent;
        self.naive_project_point(&new_point)
    }
}

/// Optimized sphere implementation
#[derive(Debug)]
struct OptimizedSphere {
    dim: usize,
}

impl OptimizedSphere {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for OptimizedSphere {
    fn name(&self) -> &str {
        "Optimized Sphere"
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
        // Optimized exponential map
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
}

fn bench_projection_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("projection_comparison");

    for dim in [50, 100, 500] {
        let naive = NaiveSphere::new(dim);
        let optimized = OptimizedSphere::new(dim);

        // Generate test data
        let points: Vec<DVector<f64>> = (0..100)
            .map(|_| DVector::from_fn(dim, |_, _| thread_rng().gen::<f64>() * 2.0 - 1.0))
            .collect();

        group.bench_with_input(BenchmarkId::new("naive", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let result = naive.naive_project_point(black_box(&points[idx % points.len()]));
                idx += 1;
                result
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let result = optimized.project_point(black_box(&points[idx % points.len()]));
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_tangent_projection_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("tangent_projection_comparison");

    for dim in [50, 100, 500] {
        let naive = NaiveSphere::new(dim);
        let optimized = OptimizedSphere::new(dim);

        // Generate test data
        let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = optimized.random_point();
                let vector = DVector::from_fn(dim, |_, _| thread_rng().gen::<f64>() * 2.0 - 1.0);
                (point, vector)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("naive", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, vector) = &test_data[idx % test_data.len()];
                let result = naive.naive_project_tangent(black_box(point), black_box(vector));
                idx += 1;
                result
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, vector) = &test_data[idx % test_data.len()];
                let result = optimized
                    .project_tangent(black_box(point), black_box(vector))
                    .unwrap();
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_retraction_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("retraction_comparison");

    for dim in [50, 100, 500] {
        let naive = NaiveSphere::new(dim);
        let optimized = OptimizedSphere::new(dim);

        // Generate test data
        let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = optimized.random_point();
                let tangent = optimized.random_tangent(&point).unwrap() * 0.1;
                (point, tangent)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("naive", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, tangent) = &test_data[idx % test_data.len()];
                let result = naive.naive_retract(black_box(point), black_box(tangent));
                idx += 1;
                result
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, tangent) = &test_data[idx % test_data.len()];
                let result = optimized
                    .retract(black_box(point), black_box(tangent))
                    .unwrap();
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_numerical_stability_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_comparison");

    let dim = 100;
    let naive = NaiveSphere::new(dim);
    let optimized = OptimizedSphere::new(dim);

    // Test with very small vectors
    let small_vectors: Vec<DVector<f64>> = (0..1000)
        .map(|_| DVector::from_fn(dim, |_, _| thread_rng().gen::<f64>() * 1e-10))
        .collect();

    group.bench_function("naive_small_vectors", |b| {
        let mut idx = 0;
        b.iter(|| {
            let vector = &small_vectors[idx % small_vectors.len()];
            let result = naive.naive_project_point(black_box(vector));
            idx += 1;
            result
        });
    });

    group.bench_function("optimized_small_vectors", |b| {
        let mut idx = 0;
        b.iter(|| {
            let vector = &small_vectors[idx % small_vectors.len()];
            let result = optimized.project_point(black_box(vector));
            idx += 1;
            result
        });
    });

    // Test with very large vectors
    let large_vectors: Vec<DVector<f64>> = (0..1000)
        .map(|_| DVector::from_fn(dim, |_, _| thread_rng().gen::<f64>() * 1e10))
        .collect();

    group.bench_function("naive_large_vectors", |b| {
        let mut idx = 0;
        b.iter(|| {
            let vector = &large_vectors[idx % large_vectors.len()];
            let result = naive.naive_project_point(black_box(vector));
            idx += 1;
            result
        });
    });

    group.bench_function("optimized_large_vectors", |b| {
        let mut idx = 0;
        b.iter(|| {
            let vector = &large_vectors[idx % large_vectors.len()];
            let result = optimized.project_point(black_box(vector));
            idx += 1;
            result
        });
    });

    group.finish();
}

fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");

    let dim = 100;
    let optimized = OptimizedSphere::new(dim);

    // Test gradient descent simulation
    let num_iterations = 100;
    let step_size = 0.01;

    group.bench_function("gradient_descent_simulation", |b| {
        b.iter(|| {
            let mut x = optimized.random_point();
            let target = optimized.random_point();

            for _ in 0..num_iterations {
                // Gradient: x - target (projected to tangent space)
                let euclidean_grad = &x - &target;
                let riem_grad = optimized
                    .euclidean_to_riemannian_gradient(&x, &euclidean_grad)
                    .unwrap();

                // Take step
                let step = riem_grad * (-step_size);
                x = optimized.retract(&x, &step).unwrap();
            }

            black_box(x)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_projection_comparison,
    bench_tangent_projection_comparison,
    bench_retraction_comparison,
    bench_numerical_stability_comparison,
    bench_algorithm_comparison
);
criterion_main!(benches);
