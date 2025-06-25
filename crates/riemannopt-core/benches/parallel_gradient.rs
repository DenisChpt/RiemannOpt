//! Benchmarks for parallel gradient computation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nalgebra::DVector;
use riemannopt_core::{
    cost_function::{CostFunction, QuadraticCost},
    compute::cpu::parallel::ParallelConfig,
};

fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");
    
    // Test different dimensions
    for &dim in &[50, 100, 200, 500, 1000, 2000] {
        // Create a simple quadratic cost function
        let a = nalgebra::DMatrix::<f64>::identity(dim, dim) * 2.0;
        let b = nalgebra::DVector::zeros(dim);
        let cost = QuadraticCost::new(a, b, 0.0);
        
        let x = DVector::from_element(dim, 1.0);
        
        // Benchmark sequential gradient
        group.bench_with_input(
            BenchmarkId::new("sequential", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    let grad = cost.gradient_fd_alloc(&x).unwrap();
                    black_box(grad)
                });
            },
        );
        
        // Benchmark parallel gradient with default config
        let config = ParallelConfig::default();
        group.bench_with_input(
            BenchmarkId::new("parallel_default", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    let grad = cost.gradient_fd_parallel(&x, &config).unwrap();
                    black_box(grad)
                });
            },
        );
        
        // Benchmark parallel gradient with custom config
        let config_custom = ParallelConfig::new()
            .with_min_dimension(50)
            .with_chunk_size(dim / 20);
        group.bench_with_input(
            BenchmarkId::new("parallel_custom", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    let grad = cost.gradient_fd_parallel(&x, &config_custom).unwrap();
                    black_box(grad)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_simd_parallel_gradient(c: &mut Criterion) {
    use riemannopt_core::core::cost_function_simd::gradient_fd_simd_parallel;
    
    let mut group = c.benchmark_group("simd_parallel_gradient");
    
    for &dim in &[100, 500, 1000, 5000] {
        let a = nalgebra::DMatrix::<f64>::identity(dim, dim) * 3.0;
        let b = nalgebra::DVector::zeros(dim);
        
        let x = DVector::from_element(dim, 1.0);
        
        let cost_fn = |p: &DVector<f64>| -> riemannopt_core::error::Result<f64> {
            Ok(0.5 * p.dot(&(&a * p)) - b.dot(p))
        };
        
        let config = ParallelConfig::default();
        
        group.bench_with_input(
            BenchmarkId::new("simd_parallel", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    let grad = gradient_fd_simd_parallel(&cost_fn, &x, &config).unwrap();
                    black_box(grad)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_gradient_computation, bench_simd_parallel_gradient);
criterion_main!(benches);