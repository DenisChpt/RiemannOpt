//! Benchmarks for SIMD operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nalgebra::{DVector, DMatrix};
use riemannopt_core::compute::cpu::{SimdBackend, get_dispatcher};

fn benchmark_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    
    for size in [100, 1000, 10000].iter() {
        let a = DVector::<f64>::from_element(*size, 1.0);
        let b = DVector::<f64>::from_element(*size, 2.0);
        
        // Benchmark nalgebra dot product
        group.bench_with_input(BenchmarkId::new("nalgebra", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.dot(&b))
            });
        });
        
        // Benchmark SIMD dot product
        let dispatcher = get_dispatcher::<f64>();
        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(dispatcher.dot_product(&a, &b))
            });
        });
    }
    
    group.finish();
}

fn benchmark_vector_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_norm");
    
    for size in [100, 1000, 10000].iter() {
        let v = DVector::<f64>::from_element(*size, 3.0);
        
        // Benchmark nalgebra norm
        group.bench_with_input(BenchmarkId::new("nalgebra", size), size, |bench, _| {
            bench.iter(|| {
                black_box(v.norm())
            });
        });
        
        // Benchmark SIMD norm
        let dispatcher = get_dispatcher::<f64>();
        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(dispatcher.norm(&v))
            });
        });
    }
    
    group.finish();
}

fn benchmark_matrix_vector_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_vector_multiply");
    
    for size in [50, 100, 200].iter() {
        let a = DMatrix::<f64>::identity(*size, *size);
        let x = DVector::<f64>::from_element(*size, 1.0);
        let mut y = DVector::<f64>::zeros(*size);
        
        // Benchmark nalgebra gemv
        group.bench_with_input(BenchmarkId::new("nalgebra", size), size, |bench, _| {
            bench.iter(|| {
                y = &a * &x;
                black_box(&y);
            });
        });
        
        // Benchmark SIMD gemv
        let dispatcher = get_dispatcher::<f64>();
        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                dispatcher.gemv(&a, &x, &mut y, 1.0, 0.0);
                black_box(&y);
            });
        });
    }
    
    group.finish();
}

fn benchmark_vector_operations(c: &mut Criterion) {
    use riemannopt_core::compute::simd_dispatch::ScalarBackend;
    
    let mut group = c.benchmark_group("vector_ops_comparison");
    
    for size in [100, 1000, 10000].iter() {
        let a = DVector::<f64>::from_element(*size, 1.0);
        let b = DVector::<f64>::from_element(*size, 2.0);
        let mut result = DVector::<f64>::zeros(*size);
        
        // Benchmark scalar backend
        let scalar_backend = ScalarBackend::<f64>::new();
        group.bench_with_input(BenchmarkId::new("scalar_add", size), size, |bench, _| {
            bench.iter(|| {
                scalar_backend.add(&a, &b, &mut result);
                black_box(&result);
            });
        });
        
        // Benchmark SIMD backend
        let simd_dispatcher = get_dispatcher::<f64>();
        group.bench_with_input(BenchmarkId::new("simd_add", size), size, |bench, _| {
            bench.iter(|| {
                simd_dispatcher.add(&a, &b, &mut result);
                black_box(&result);
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_dot_product,
    benchmark_vector_norm,
    benchmark_matrix_vector_multiply,
    benchmark_vector_operations
);
criterion_main!(benches);