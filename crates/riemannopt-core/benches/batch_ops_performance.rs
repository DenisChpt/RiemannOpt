//! Benchmarks comparing cache-friendly batch operations vs original implementation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use nalgebra::{DMatrix, DVector};
use riemannopt_core::compute::cpu::parallel::ParallelBatch;
use riemannopt_core::compute::cpu::batch_ops::CacheFriendlyBatch;
use rand::prelude::*;

/// Original implementation with allocations for comparison
mod original_impl {
    use nalgebra::{DMatrix, DVector};
    use rayon::prelude::*;
    use riemannopt_core::types::Scalar;
    
    pub fn gradient_with_allocations<T, F>(
        points: &DMatrix<T>,
        grad_func: F,
    ) -> DMatrix<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> DVector<T> + Sync,
    {
        let n_points = points.ncols();
        let dim = points.nrows();
        let mut result = DMatrix::<T>::zeros(dim, n_points);
        
        let chunk_size = ((n_points + rayon::current_num_threads() - 1) / rayon::current_num_threads()).max(1);
        
        result.column_iter_mut()
            .enumerate()
            .collect::<Vec<_>>()
            .par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                for (i, col) in chunk {
                    let point = points.column(*i).clone_owned(); // Allocation here!
                    let grad = grad_func(&point);
                    col.copy_from(&grad);
                }
            });
        
        result
    }
    
    pub fn map_with_allocations<T, F>(
        points: &DMatrix<T>,
        op: F,
    ) -> DMatrix<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> DVector<T> + Sync,
    {
        let n_points = points.ncols();
        let dim = points.nrows();
        let mut result = DMatrix::<T>::zeros(dim, n_points);
        
        let chunk_size = ((n_points + rayon::current_num_threads() - 1) / rayon::current_num_threads()).max(1);
        
        result.column_iter_mut()
            .enumerate()
            .collect::<Vec<_>>()
            .par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                for (i, col) in chunk {
                    let point = points.column(*i).clone_owned(); // Allocation here!
                    let res = op(&point);
                    col.copy_from(&res);
                }
            });
        
        result
    }
    
    pub fn map_pairs_with_allocations<T, F>(
        points: &DMatrix<T>,
        tangents: &DMatrix<T>,
        op: F,
    ) -> DMatrix<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>, &DVector<T>) -> DVector<T> + Sync,
    {
        let n_points = points.ncols();
        let dim = points.nrows();
        let mut result = DMatrix::<T>::zeros(dim, n_points);
        
        let chunk_size = ((n_points + rayon::current_num_threads() - 1) / rayon::current_num_threads()).max(1);
        
        result.column_iter_mut()
            .enumerate()
            .collect::<Vec<_>>()
            .par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                for (i, col) in chunk {
                    let point = points.column(*i).clone_owned();   // Allocation 1!
                    let tangent = tangents.column(*i).clone_owned(); // Allocation 2!
                    let res = op(&point, &tangent);
                    col.copy_from(&res);
                }
            });
        
        result
    }
}

fn bench_gradient_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");
    
    // Test different batch sizes and dimensions
    let configs = vec![
        (100, 10),    // 100 points, 10 dimensions
        (100, 100),   // 100 points, 100 dimensions
        (1000, 10),   // 1000 points, 10 dimensions
        (1000, 100),  // 1000 points, 100 dimensions
        (10000, 50),  // 10000 points, 50 dimensions
    ];
    
    for (n_points, dim) in configs {
        // Create random batch of points
        let mut rng = thread_rng();
        let points = DMatrix::<f64>::from_fn(dim, n_points, |_, _| rng.gen());
        
        // Simple gradient function: gradient of f(x) = ||x||^2 is 2x
        let grad_func = |x: &DVector<f64>| x * 2.0;
        
        group.throughput(Throughput::Elements((n_points * dim) as u64));
        
        // Benchmark original implementation with allocations
        group.bench_with_input(
            BenchmarkId::new("original_with_allocations", format!("{}x{}", n_points, dim)),
            &(n_points, dim),
            |b, _| {
                b.iter(|| {
                    let result = original_impl::gradient_with_allocations(&points, &grad_func);
                    black_box(result)
                });
            },
        );
        
        // Benchmark new cache-friendly implementation
        group.bench_with_input(
            BenchmarkId::new("cache_friendly_zero_alloc", format!("{}x{}", n_points, dim)),
            &(n_points, dim),
            |b, _| {
                b.iter(|| {
                    let mut output = DMatrix::<f64>::zeros(dim, n_points);
                    let grad_func_inplace = |x: nalgebra::DVectorView<f64>, mut out: nalgebra::DVectorViewMut<f64>| {
                        let result = x * 2.0;
                        out.copy_from(&result);
                    };
                    CacheFriendlyBatch::gradient(&points, &mut output, grad_func_inplace).unwrap();
                    black_box(output)
                });
            },
        );
        
        // Benchmark current ParallelBatch (which now uses cache-friendly)
        group.bench_with_input(
            BenchmarkId::new("parallel_batch_current", format!("{}x{}", n_points, dim)),
            &(n_points, dim),
            |b, _| {
                b.iter(|| {
                    let mut output = DMatrix::<f64>::zeros(dim, n_points);
                    let grad_func_inplace = |x: nalgebra::DVectorView<f64>, mut out: nalgebra::DVectorViewMut<f64>| {
                        let result = x * 2.0;
                        out.copy_from(&result);
                    };
                    ParallelBatch::gradient(&points, &mut output, grad_func_inplace).unwrap();
                    black_box(output)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_map_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_operations");
    
    let configs = vec![
        (100, 50),
        (1000, 50),
        (1000, 100),
        (5000, 100),
    ];
    
    for (n_points, dim) in configs {
        let mut rng = thread_rng();
        let points = DMatrix::<f64>::from_fn(dim, n_points, |_, _| rng.gen());
        
        // Simple operation: normalize each point
        let op = |x: &DVector<f64>| {
            let norm = x.norm();
            if norm > 0.0 {
                x / norm
            } else {
                x.clone()
            }
        };
        
        group.throughput(Throughput::Elements((n_points * dim) as u64));
        
        // Original with allocations
        group.bench_with_input(
            BenchmarkId::new("original_with_allocations", format!("{}x{}", n_points, dim)),
            &(n_points, dim),
            |b, _| {
                b.iter(|| {
                    let result = original_impl::map_with_allocations(&points, &op);
                    black_box(result)
                });
            },
        );
        
        // Cache-friendly implementation
        group.bench_with_input(
            BenchmarkId::new("cache_friendly_zero_alloc", format!("{}x{}", n_points, dim)),
            &(n_points, dim),
            |b, _| {
                b.iter(|| {
                    let mut output = DMatrix::<f64>::zeros(dim, n_points);
                    let op_inplace = |x: nalgebra::DVectorView<f64>, mut out: nalgebra::DVectorViewMut<f64>| {
                        let norm = x.norm();
                        if norm > 0.0 {
                            let result = x / norm;
                            out.copy_from(&result);
                        } else {
                            out.copy_from(&x);
                        }
                    };
                    CacheFriendlyBatch::map(&points, &mut output, op_inplace).unwrap();
                    black_box(output)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_map_pairs_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_pairs_operations");
    
    let configs = vec![
        (100, 50),
        (500, 100),
        (1000, 100),
        (2000, 50),
    ];
    
    for (n_points, dim) in configs {
        let mut rng = thread_rng();
        let points = DMatrix::<f64>::from_fn(dim, n_points, |_, _| rng.gen());
        let tangents = DMatrix::<f64>::from_fn(dim, n_points, |_, _| rng.gen());
        
        // Operation: retraction x + t
        let op = |x: &DVector<f64>, t: &DVector<f64>| x + t;
        
        group.throughput(Throughput::Elements((n_points * dim * 2) as u64));
        
        // Original with double allocations
        group.bench_with_input(
            BenchmarkId::new("original_double_allocations", format!("{}x{}", n_points, dim)),
            &(n_points, dim),
            |b, _| {
                b.iter(|| {
                    let result = original_impl::map_pairs_with_allocations(&points, &tangents, &op);
                    black_box(result)
                });
            },
        );
        
        // Cache-friendly implementation
        group.bench_with_input(
            BenchmarkId::new("cache_friendly_zero_alloc", format!("{}x{}", n_points, dim)),
            &(n_points, dim),
            |b, _| {
                b.iter(|| {
                    let mut output = DMatrix::<f64>::zeros(dim, n_points);
                    let op_inplace = |p: nalgebra::DVectorView<f64>, t: nalgebra::DVectorView<f64>, mut out: nalgebra::DVectorViewMut<f64>| {
                        let result = p + t;
                        out.copy_from(&result);
                    };
                    CacheFriendlyBatch::map_pairs(&points, &tangents, &mut output, op_inplace).unwrap();
                    black_box(output)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access_patterns");
    
    // Large dimension to stress memory bandwidth
    let n_points = 1000;
    let dim = 1000;
    
    let mut rng = thread_rng();
    let points = DMatrix::<f64>::from_fn(dim, n_points, |_, _| rng.gen());
    
    // Memory-intensive operation
    let heavy_op = |x: &DVector<f64>| {
        let mut result = x.clone();
        for _ in 0..5 {
            result = &result + x;
            result = result.component_mul(&result);
            result /= result.norm() + 1.0;
        }
        result
    };
    
    group.throughput(Throughput::Bytes((n_points * dim * 8) as u64));
    
    group.bench_function("original_column_major_access", |b| {
        b.iter(|| {
            let result = original_impl::map_with_allocations(&points, &heavy_op);
            black_box(result)
        });
    });
    
    group.bench_function("cache_friendly_aos_access", |b| {
        b.iter(|| {
            let mut output = DMatrix::<f64>::zeros(dim, n_points);
            let heavy_op_inplace = |x: nalgebra::DVectorView<f64>, mut out: nalgebra::DVectorViewMut<f64>| {
                let mut result = x.clone_owned();
                for _ in 0..5 {
                    result = &result + &x;
                    result = result.component_mul(&result);
                    result /= result.norm() + 1.0;
                }
                out.copy_from(&result);
            };
            CacheFriendlyBatch::map(&points, &mut output, heavy_op_inplace).unwrap();
            black_box(output)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_gradient_operations,
    bench_map_operations,
    bench_map_pairs_operations,
    bench_memory_access_patterns
);
criterion_main!(benches);