//! Benchmarks for retraction operations.
//!
//! This benchmark suite compares different retraction methods and their
//! performance characteristics across various manifold dimensions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Dyn;
use rand::prelude::*;
use riemannopt_core::{
    error::Result,
    manifold::Manifold,
    retraction::{DefaultRetraction, ExponentialRetraction, ProjectionRetraction, Retraction},
    types::DVector,
};

/// Test manifold with multiple retraction implementations
#[derive(Debug)]
struct RetractionTestManifold {
    dim: usize,
}

impl RetractionTestManifold {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for RetractionTestManifold {
    fn name(&self) -> &str {
        "Retraction Test Manifold"
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
            result.copy_from(&(point / norm));
        } else {
            result.fill(0.0);
            result[0] = 1.0;
        }
    }

    fn project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let inner = point.dot(vector);
        result.copy_from(&(vector - point * inner));
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
        // Default to exponential map
        let norm_v = tangent.norm();
        if norm_v < f64::EPSILON {
            result.copy_from(point);
        } else {
            let cos_norm = norm_v.cos();
            let sin_norm = norm_v.sin();
            result.copy_from(&(point * cos_norm + tangent * (sin_norm / norm_v)));
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
                result.copy_from(&(v * (theta / v_norm)));
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

    fn has_exact_exp_log(&self) -> bool {
        true
    }
}

fn bench_retraction_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("retraction_methods");

    for dim in [10, 50, 100, 500] {
        let manifold = RetractionTestManifold::new(dim);

        // Generate test data
        let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = manifold.random_point();
                let mut tangent = DVector::zeros(dim);
                manifold.random_tangent(&point, &mut tangent).unwrap();
                tangent *= 0.1;
                (point, tangent)
            })
            .collect();

        // Benchmark default retraction
        group.bench_with_input(BenchmarkId::new("default", dim), &dim, |b, _| {
            let retraction = DefaultRetraction;
            let mut idx = 0;
            b.iter(|| {
                let (point, tangent) = &test_data[idx % test_data.len()];
                let mut result = DVector::zeros(manifold.dim);
                retraction
                    .retract(&manifold, black_box(point), black_box(tangent), &mut result)
                    .unwrap();
                idx += 1;
                black_box(result)
            });
        });

        // Benchmark projection retraction
        group.bench_with_input(BenchmarkId::new("projection", dim), &dim, |b, _| {
            let retraction = ProjectionRetraction;
            let mut idx = 0;
            b.iter(|| {
                let (point, tangent) = &test_data[idx % test_data.len()];
                let mut result = DVector::zeros(manifold.dim);
                retraction
                    .retract(&manifold, black_box(point), black_box(tangent), &mut result)
                    .unwrap();
                idx += 1;
                black_box(result)
            });
        });

        // Benchmark exponential retraction
        group.bench_with_input(BenchmarkId::new("exponential", dim), &dim, |b, _| {
            let retraction = ExponentialRetraction::new();
            let mut idx = 0;
            b.iter(|| {
                let (point, tangent) = &test_data[idx % test_data.len()];
                let mut result = DVector::zeros(manifold.dim);
                retraction
                    .retract(&manifold, black_box(point), black_box(tangent), &mut result)
                    .unwrap();
                idx += 1;
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_inverse_retraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("inverse_retraction");

    for dim in [10, 50, 100, 500] {
        let manifold = RetractionTestManifold::new(dim);

        // Generate test data
        let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = manifold.random_point();
                let mut tangent = DVector::zeros(dim);
                manifold.random_tangent(&point, &mut tangent).unwrap();
                tangent *= 0.1;
                let mut other = DVector::zeros(dim);
                manifold.retract(&point, &tangent, &mut other).unwrap();
                (point, other)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("logarithm", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let (point, other) = &test_data[idx % test_data.len()];
                let mut result = DVector::zeros(dim);
                manifold
                    .inverse_retract(black_box(point), black_box(other), &mut result)
                    .unwrap();
                idx += 1;
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_retraction_small_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("retraction_small_steps");

    let dim = 100;
    let manifold = RetractionTestManifold::new(dim);

    for scale in [1e-6, 1e-4, 1e-2, 0.1, 0.5] {
        // Generate test data with specific step sizes
        let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
            .map(|_| {
                let point = manifold.random_point();
                let mut tangent = DVector::zeros(dim);
                manifold.random_tangent(&point, &mut tangent).unwrap();
                let tangent_norm = tangent.norm();
                tangent *= scale / tangent_norm;
                (point, tangent)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("projection", scale), &scale, |b, _| {
            let retraction = ProjectionRetraction;
            let mut idx = 0;
            b.iter(|| {
                let (point, tangent) = &test_data[idx % test_data.len()];
                let mut result = DVector::zeros(manifold.dim);
                retraction
                    .retract(&manifold, black_box(point), black_box(tangent), &mut result)
                    .unwrap();
                idx += 1;
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("exponential", scale), &scale, |b, _| {
            let retraction = ExponentialRetraction::new();
            let mut idx = 0;
            b.iter(|| {
                let (point, tangent) = &test_data[idx % test_data.len()];
                let mut result = DVector::zeros(manifold.dim);
                retraction
                    .retract(&manifold, black_box(point), black_box(tangent), &mut result)
                    .unwrap();
                idx += 1;
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_retraction_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("retraction_accuracy");

    // Test accuracy vs performance trade-off
    let dim = 50;
    let manifold = RetractionTestManifold::new(dim);

    // Generate test data
    let num_samples = 1000;
    let test_data: Vec<(DVector<f64>, DVector<f64>)> = (0..num_samples)
        .map(|_| {
            let point = manifold.random_point();
            let mut tangent = DVector::zeros(dim);
            manifold.random_tangent(&point, &mut tangent).unwrap();
            tangent *= 0.01;
            (point, tangent)
        })
        .collect();

    // Benchmark accuracy check for projection retraction
    group.bench_function("projection_accuracy_check", |b| {
        let retraction = ProjectionRetraction;
        b.iter(|| {
            let mut max_error: f64 = 0.0;
            let mut exact = DVector::zeros(dim);
            for (point, tangent) in &test_data {
                let mut result = DVector::zeros(dim);
                retraction.retract(&manifold, point, tangent, &mut result).unwrap();
                manifold.retract(point, tangent, &mut exact).unwrap();
                let error = (&result - &exact).norm();
                max_error = max_error.max(error);
            }
            black_box(max_error)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_retraction_methods,
    bench_inverse_retraction,
    bench_retraction_small_steps,
    bench_retraction_accuracy
);
criterion_main!(benches);
