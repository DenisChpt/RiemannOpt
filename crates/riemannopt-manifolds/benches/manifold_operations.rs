//! Benchmarks for manifold operations
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use riemannopt_manifolds::{Sphere, Stiefel, Grassmann};
use riemannopt_core::manifold::Manifold;
use riemannopt_core::memory::workspace::Workspace;
use nalgebra::{DVector, DMatrix};

fn benchmark_sphere_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere");
    
    for &n in &[10, 100, 1000] {
        let sphere = Sphere::<f64>::new(n).unwrap();
        let point = sphere.random_point();
        let mut vector = DVector::zeros(n);
        let mut workspace = Workspace::<f64>::new();
        sphere.random_tangent(&point, &mut vector, &mut workspace).unwrap();
        
        group.bench_with_input(BenchmarkId::new("projection", n), &n, |b, _| {
            let mut result = DVector::zeros(n);
            let mut workspace = Workspace::<f64>::new();
            b.iter(|| {
                sphere.project_point(black_box(&point), &mut result, &mut workspace)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("retraction", n), &n, |b, _| {
            let mut result = DVector::zeros(n);
            let mut workspace = Workspace::<f64>::new();
            b.iter(|| {
                sphere.retract(black_box(&point), black_box(&vector), &mut result, &mut workspace).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("tangent_projection", n), &n, |b, _| {
            let mut result = DVector::zeros(n);
            let mut workspace = Workspace::<f64>::new();
            b.iter(|| {
                sphere.project_tangent(black_box(&point), black_box(&vector), &mut result, &mut workspace).unwrap()
            });
        });
    }
    
    group.finish();
}

fn benchmark_stiefel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("stiefel");
    
    let configs = [(10, 3), (50, 10), (100, 20)];
    
    for &(n, p) in &configs {
        let stiefel = Stiefel::<f64>::new(n, p).unwrap();
        let point = stiefel.random_point();
        let mut vector = DMatrix::zeros(n, p);
        let mut workspace = Workspace::<f64>::new();
        stiefel.random_tangent(&point, &mut vector, &mut workspace).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("qr_retraction", format!("{}x{}", n, p)), 
            &(n, p), 
            |b, _| {
                let mut result = DMatrix::zeros(n, p);
                let mut workspace = Workspace::<f64>::new();
                b.iter(|| {
                    stiefel.retract(black_box(&point), black_box(&vector), &mut result, &mut workspace).unwrap()
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("projection", format!("{}x{}", n, p)), 
            &(n, p), 
            |b, _| {
                let mut result = DMatrix::zeros(n, p);
                let mut workspace = Workspace::<f64>::new();
                b.iter(|| {
                    stiefel.project_point(black_box(&point), &mut result, &mut workspace)
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_grassmann_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("grassmann");
    
    let configs = [(10, 3), (20, 5), (50, 10)];
    
    for &(n, p) in &configs {
        let grassmann = Grassmann::<f64>::new(n, p).unwrap();
        let x = grassmann.random_point();
        let y = grassmann.random_point();
        
        group.bench_with_input(
            BenchmarkId::new("distance", format!("{}x{}", n, p)), 
            &(n, p), 
            |b, _| {
                let mut workspace = Workspace::<f64>::new();
                b.iter(|| {
                    <Grassmann<f64> as Manifold<f64>>::distance(&grassmann, black_box(&x), black_box(&y), &mut workspace).unwrap()
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches, 
    benchmark_sphere_operations,
    benchmark_stiefel_operations,
    benchmark_grassmann_operations
);
criterion_main!(benches);