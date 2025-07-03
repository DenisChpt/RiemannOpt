//! Benchmarks for tangent space operations.
//!
//! This benchmark suite tests the performance of tangent space operations
//! including normalization, orthogonalization, and basis construction.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Dyn;
use rand::prelude::*;
use riemannopt_core::{
    manifold::Manifold,
    memory::Workspace,
    tangent::{gram_schmidt, normalize},
    types::DVector,
};

/// Simple Euclidean space for tangent benchmarks
#[derive(Debug)]
struct EuclideanSpace {
    dim: usize,
}

impl EuclideanSpace {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for EuclideanSpace {
    fn name(&self) -> &str {
        "Euclidean Space"
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn is_point_on_manifold(&self, _point: &DVector<f64>, _tol: f64) -> bool {
        true
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &DVector<f64>,
        _vector: &DVector<f64>,
        _tol: f64,
    ) -> bool {
        true
    }

    fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) {
        result.copy_from(point);
    }

    fn project_tangent(
        &self,
        _point: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> riemannopt_core::error::Result<()> {
        result.copy_from(vector);
        Ok(())
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
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> riemannopt_core::error::Result<()> {
        result.copy_from(&(point + tangent));
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &DVector<f64>,
        other: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> riemannopt_core::error::Result<()> {
        result.copy_from(&(other - point));
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> riemannopt_core::error::Result<()> {
        result.copy_from(euclidean_grad);
        Ok(())
    }

    fn random_point(&self) -> DVector<f64> {
        let mut rng = thread_rng();
        DVector::from_fn(self.dim, |_, _| rng.gen::<f64>() * 2.0 - 1.0)
    }

    fn random_tangent(
        &self,
        _point: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> riemannopt_core::error::Result<()> {
        let mut rng = thread_rng();
        for i in 0..self.dim {
            result[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        Ok(())
    }
}

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    for dim in [10, 50, 100, 500, 1000] {
        let manifold = EuclideanSpace::new(dim);

        // Generate random vectors
        let vectors: Vec<DVector<f64>> = (0..100)
            .map(|_| {
                let mut rng = thread_rng();
                DVector::from_fn(dim, |_, _| rng.gen::<f64>() * 2.0 - 1.0)
            })
            .collect();

        let point = manifold.random_point();

        group.bench_with_input(BenchmarkId::new("euclidean", dim), &dim, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let result = normalize(
                    &manifold,
                    black_box(&point),
                    black_box(&vectors[idx % vectors.len()]),
                )
                .unwrap();
                idx += 1;
                result
            });
        });
    }

    group.finish();
}

fn bench_gram_schmidt(c: &mut Criterion) {
    let mut group = c.benchmark_group("gram_schmidt");

    for (dim, num_vectors) in [(10, 5), (50, 10), (100, 20), (500, 50)] {
        let manifold = EuclideanSpace::new(dim);
        let point = manifold.random_point();

        // Generate sets of random vectors
        let vector_sets: Vec<Vec<DVector<f64>>> = (0..10)
            .map(|_| {
                (0..num_vectors)
                    .map(|_| {
                        let mut rng = thread_rng();
                        DVector::from_fn(dim, |_, _| rng.gen::<f64>() * 2.0 - 1.0)
                    })
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new(format!("{}x{}", dim, num_vectors), dim),
            &dim,
            |b, _| {
                let mut idx = 0;
                b.iter(|| {
                    let result = gram_schmidt(
                        &manifold,
                        black_box(&point),
                        black_box(&vector_sets[idx % vector_sets.len()]),
                    )
                    .unwrap();
                    idx += 1;
                    result
                });
            },
        );
    }

    group.finish();
}

// Removed tangent space construction benchmark as TangentSpace is not public

// Removed tangent basis benchmark as TangentSpace is not public

fn bench_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    let dim = 100;
    let manifold = EuclideanSpace::new(dim);
    let point = manifold.random_point();

    // Generate test vectors
    let vectors: Vec<(DVector<f64>, DVector<f64>)> = (0..100)
        .map(|_| {
            let mut u = DVector::zeros(dim);
            let mut v = DVector::zeros(dim);
            let mut workspace = Workspace::new();
            manifold.random_tangent(&point, &mut u, &mut workspace).unwrap();
            manifold.random_tangent(&point, &mut v, &mut workspace).unwrap();
            (u, v)
        })
        .collect();

    group.bench_function("add", |b| {
        let mut idx = 0;
        b.iter(|| {
            let (u, v) = &vectors[idx % vectors.len()];
            let result = u + v;
            idx += 1;
            result
        });
    });

    group.bench_function("scale", |b| {
        let mut idx = 0;
        b.iter(|| {
            let (u, _) = &vectors[idx % vectors.len()];
            let result = u * 2.5;
            idx += 1;
            result
        });
    });

    group.bench_function("inner_product", |b| {
        let mut idx = 0;
        b.iter(|| {
            let (u, v) = &vectors[idx % vectors.len()];
            let result = manifold.inner_product(&point, u, v).unwrap();
            idx += 1;
            result
        });
    });

    group.bench_function("norm", |b| {
        let mut idx = 0;
        b.iter(|| {
            let (u, _) = &vectors[idx % vectors.len()];
            let result = manifold.norm(&point, u).unwrap();
            idx += 1;
            result
        });
    });

    group.finish();
}

fn bench_orthogonalization_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonalization");

    let dim = 100;
    let num_vectors = 20;
    let manifold = EuclideanSpace::new(dim);
    let point = manifold.random_point();

    // Generate test data
    let vector_sets: Vec<Vec<DVector<f64>>> = (0..10)
        .map(|_| {
            (0..num_vectors)
                .map(|_| {
                    let mut tangent = DVector::zeros(dim);
                    let mut workspace = Workspace::new();
                    manifold.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
                    tangent
                })
                .collect()
        })
        .collect();

    // Classical Gram-Schmidt
    group.bench_function("classical_gram_schmidt", |b| {
        let mut idx = 0;
        b.iter(|| {
            let vectors = &vector_sets[idx % vector_sets.len()];
            let result = gram_schmidt(&manifold, &point, vectors).unwrap();
            idx += 1;
            black_box(result)
        });
    });

    // Modified Gram-Schmidt (simulated - would be a separate implementation)
    group.bench_function("modified_gram_schmidt", |b| {
        let mut idx = 0;
        b.iter(|| {
            let vectors = vector_sets[idx % vector_sets.len()].clone();
            let mut result = vectors;

            // Modified Gram-Schmidt process
            for i in 0..result.len() {
                // Normalize
                let norm = result[i].norm();
                if norm > 1e-10 {
                    result[i] /= norm;
                }

                // Store the vector to avoid borrow checker issues
                let vec_i = result[i].clone();

                // Orthogonalize remaining vectors
                for j in (i + 1)..result.len() {
                    let proj = vec_i.dot(&result[j]);
                    result[j] -= &vec_i * proj;
                }
            }

            idx += 1;
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_normalize,
    bench_gram_schmidt,
    bench_vector_operations,
    bench_orthogonalization_methods
);
criterion_main!(benches);
