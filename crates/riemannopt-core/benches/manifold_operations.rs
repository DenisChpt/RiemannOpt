//! Benchmarks for core manifold operations.
//!
//! This benchmark suite tests the performance of fundamental manifold
//! operations like projection, inner product, and gradient conversion.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Dyn;
use rand::prelude::*;
use riemannopt_core::{error::Result, manifold::Manifold, types::DVector};

/// Simple sphere manifold for benchmarking
#[derive(Debug)]
struct BenchSphere {
    dim: usize,
}

impl BenchSphere {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for BenchSphere {
    fn name(&self) -> &str {
        "Benchmark Sphere"
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
}

fn bench_point_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_projection");

    for dim in [10, 50, 100, 500, 1000] {
        let sphere = BenchSphere::new(dim);
        let mut rng = thread_rng();

        // Generate random points
        let points: Vec<DVector<f64>> = (0..100)
            .map(|_| DVector::from_fn(dim, |_, _| rng.gen::<f64>() * 2.0 - 1.0))
            .collect();

        group.bench_with_input(BenchmarkId::new("sphere", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let result = sphere.project_point(black_box(&points[idx % points.len()]));
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_tangent_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("tangent_projection");

    for dim in [10, 50, 100, 500, 1000] {
        let sphere = BenchSphere::new(dim);

        // Generate points on manifold and random vectors
        let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = sphere.random_point();
                let vector = DVector::from_fn(dim, |_, _| thread_rng().gen::<f64>() * 2.0 - 1.0);
                (point, vector)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("sphere", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, vector) = &test_data[idx % test_data.len()];
                let result = sphere
                    .project_tangent(black_box(point), black_box(vector))
                    .unwrap();
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");

    for dim in [10, 50, 100, 500, 1000] {
        let sphere = BenchSphere::new(dim);

        // Generate test data
        let test_data: Vec<(DVector<f64>, DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = sphere.random_point();
                let u = sphere.random_tangent(&point).unwrap();
                let v = sphere.random_tangent(&point).unwrap();
                (point, u, v)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("sphere", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, u, v) = &test_data[idx % test_data.len()];
                let result = sphere
                    .inner_product(black_box(point), black_box(u), black_box(v))
                    .unwrap();
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_gradient_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_conversion");

    for dim in [10, 50, 100, 500, 1000] {
        let sphere = BenchSphere::new(dim);

        // Generate test data
        let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = sphere.random_point();
                let grad = DVector::from_fn(dim, |_, _| thread_rng().gen::<f64>() * 2.0 - 1.0);
                (point, grad)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("sphere", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, grad) = &test_data[idx % test_data.len()];
                let result = sphere
                    .euclidean_to_riemannian_gradient(black_box(point), black_box(grad))
                    .unwrap();
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_manifold_checks(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_checks");

    for dim in [10, 50, 100, 500, 1000] {
        let sphere = BenchSphere::new(dim);
        let tolerance = 1e-10;

        // Generate test data
        let points: Vec<DVector<f64>> = (0..100).map(|_| sphere.random_point()).collect();

        let vectors: Vec<(DVector<f64>, DVector<f64>)> = points
            .iter()
            .map(|p| (p.clone(), sphere.random_tangent(p).unwrap()))
            .collect();

        group.bench_with_input(BenchmarkId::new("is_on_manifold", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let result = sphere.is_point_on_manifold(
                    black_box(&points[idx % points.len()]),
                    black_box(tolerance),
                );
                idx += 1;
                result
            });
        });

        group.bench_with_input(
            BenchmarkId::new("is_in_tangent_space", dim),
            &dim,
            |b, _| {
                let mut idx = 0;
                b.iter(|| {
                    let (point, vector) = &vectors[idx % vectors.len()];
                    let result = sphere.is_vector_in_tangent_space(
                        black_box(point),
                        black_box(vector),
                        black_box(tolerance),
                    );
                    idx += 1;
                    result
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_transport(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_transport");

    for dim in [10, 50, 100, 500] {
        let sphere = BenchSphere::new(dim);

        // Generate test data
        let test_data: Vec<(DVector<f64>, DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let from = sphere.random_point();
                let direction = sphere.random_tangent(&from).unwrap() * 0.1;
                let to = sphere.retract(&from, &direction).unwrap();
                let vector = sphere.random_tangent(&from).unwrap();
                (from, to, vector)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("sphere", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (from, to, vector) = &test_data[idx % test_data.len()];
                let result = sphere
                    .parallel_transport(black_box(from), black_box(to), black_box(vector))
                    .unwrap();
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_point_projection,
    bench_tangent_projection,
    bench_inner_product,
    bench_gradient_conversion,
    bench_manifold_checks,
    bench_parallel_transport
);
criterion_main!(benches);
