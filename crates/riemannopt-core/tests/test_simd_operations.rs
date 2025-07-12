//! Comprehensive tests for SIMD operations in RiemannOpt.
//!
//! These tests verify the correctness and performance characteristics
//! of SIMD-accelerated operations across different backends and data types.

use approx::{assert_relative_eq, assert_ulps_eq};
use nalgebra::{Const, DMatrix, DVector, Dyn, VecStorage};
use riemannopt_core::compute::cpu::{get_dispatcher, SimdBackend};
use riemannopt_core::compute::simd_dispatch::ScalarBackend;
use std::time::Instant;

#[cfg(test)]
mod correctness_tests {
    use approx::assert_ulps_eq;

    use super::*;

    /// Test that SIMD and scalar backends produce identical results for f32
    fn test_f32_backend_equivalence_impl(test_sizes: &[usize]) {
        let scalar_backend = ScalarBackend::<f32>::new();
        let simd_dispatcher = get_dispatcher::<f32>();
        
        // Use appropriate tolerance for f32 - relative epsilon of 1e-5 
        // f32 has limited precision, so we need more tolerance
        let epsilon = 1e-5_f32;
        let ulps = 32u32;

        for &size in test_sizes {
            // Generate test vectors
            let a = DVector::from_fn(size, |i, _| (i as f32).sin());
            let b = DVector::from_fn(size, |i, _| (i as f32).cos());

            // Test dot product
            let scalar_dot = scalar_backend.dot_product(&a, &b);
            let simd_dot = simd_dispatcher.dot_product(&a, &b);
            // For dot products with accumulation, use relative tolerance
            assert_relative_eq!(scalar_dot, simd_dot, epsilon = epsilon);

            // Test norm
            let scalar_norm = scalar_backend.norm(&a);
            let simd_norm = simd_dispatcher.norm(&a);
            assert_relative_eq!(scalar_norm, simd_norm, epsilon = epsilon);
            
            // Test norm squared
            let scalar_norm_sq = scalar_backend.norm_squared(&a);
            let simd_norm_sq = simd_dispatcher.norm_squared(&a);
            assert_ulps_eq!(scalar_norm_sq, simd_norm_sq, max_ulps = ulps);

            // Test add
            let mut scalar_result = DVector::zeros(size);
            let mut simd_result = DVector::zeros(size);
            scalar_backend.add(&a, &b, &mut scalar_result);
            simd_dispatcher.add(&a, &b, &mut simd_result);
            for i in 0..size {
                assert_ulps_eq!(scalar_result[i], simd_result[i], max_ulps = ulps);
            }

            // Test axpy
            let alpha = 2.5_f32;
            let mut scalar_y = b.clone();
            let mut simd_y = b.clone();
            scalar_backend.axpy(alpha, &a, &mut scalar_y);
            simd_dispatcher.axpy(alpha, &a, &mut simd_y);
            for i in 0..size {
                assert_ulps_eq!(scalar_y[i], simd_y[i], max_ulps = ulps);
            }

            // Test scale
            let mut scalar_v = a.clone();
            let mut simd_v = a.clone();
            scalar_backend.scale(&mut scalar_v, alpha);
            simd_dispatcher.scale(&mut simd_v, alpha);
            for i in 0..size {
                assert_ulps_eq!(scalar_v[i], simd_v[i], max_ulps = ulps);
            }

            // Test normalize
            let mut scalar_v = a.clone();
            let mut simd_v = a.clone();
            let scalar_old_norm = scalar_backend.normalize(&mut scalar_v);
            let simd_old_norm = simd_dispatcher.normalize(&mut simd_v);
            assert_relative_eq!(scalar_old_norm, simd_old_norm, epsilon = epsilon);
            for i in 0..size {
                assert_relative_eq!(scalar_v[i], simd_v[i], epsilon = epsilon);
            }

            // Test hadamard product
            let mut scalar_result = DVector::zeros(size);
            let mut simd_result = DVector::zeros(size);
            scalar_backend.hadamard_product(&a, &b, &mut scalar_result);
            simd_dispatcher.hadamard_product(&a, &b, &mut simd_result);
            for i in 0..size {
                assert_ulps_eq!(scalar_result[i], simd_result[i], max_ulps = ulps);
            }

            // Test max_abs_diff
            let scalar_max_diff = scalar_backend.max_abs_diff(&a, &b);
            let simd_max_diff = simd_dispatcher.max_abs_diff(&a, &b);
            assert_relative_eq!(scalar_max_diff, simd_max_diff, epsilon = epsilon);
        }
    }

    /// Test that SIMD and scalar backends produce identical results for f64
    fn test_f64_backend_equivalence_impl(test_sizes: &[usize]) {
        let scalar_backend = ScalarBackend::<f64>::new();
        let simd_dispatcher = get_dispatcher::<f64>();
        
        // Use appropriate tolerance for f64 - much higher precision
        let epsilon = 1e-10_f64;

        for &size in test_sizes {
            // Generate test vectors
            let a = DVector::from_fn(size, |i, _| (i as f64).sin());
            let b = DVector::from_fn(size, |i, _| (i as f64).cos());

            // Test dot product
            let scalar_dot = scalar_backend.dot_product(&a, &b);
            let simd_dot = simd_dispatcher.dot_product(&a, &b);
            assert_relative_eq!(scalar_dot, simd_dot, epsilon = epsilon);

            // Test norm
            let scalar_norm = scalar_backend.norm(&a);
            let simd_norm = simd_dispatcher.norm(&a);
            assert_relative_eq!(scalar_norm, simd_norm, epsilon = epsilon);
            
            // Test norm squared
            let scalar_norm_sq = scalar_backend.norm_squared(&a);
            let simd_norm_sq = simd_dispatcher.norm_squared(&a);
            assert_relative_eq!(scalar_norm_sq, simd_norm_sq, epsilon = epsilon);

            // Test add
            let mut scalar_result = DVector::zeros(size);
            let mut simd_result = DVector::zeros(size);
            scalar_backend.add(&a, &b, &mut scalar_result);
            simd_dispatcher.add(&a, &b, &mut simd_result);
            for i in 0..size {
                assert_relative_eq!(scalar_result[i], simd_result[i], epsilon = epsilon);
            }

            // Test axpy
            let alpha = 2.5_f64;
            let mut scalar_y = b.clone();
            let mut simd_y = b.clone();
            scalar_backend.axpy(alpha, &a, &mut scalar_y);
            simd_dispatcher.axpy(alpha, &a, &mut simd_y);
            for i in 0..size {
                assert_relative_eq!(scalar_y[i], simd_y[i], epsilon = epsilon);
            }

            // Test scale
            let mut scalar_v = a.clone();
            let mut simd_v = a.clone();
            scalar_backend.scale(&mut scalar_v, alpha);
            simd_dispatcher.scale(&mut simd_v, alpha);
            for i in 0..size {
                assert_relative_eq!(scalar_v[i], simd_v[i], epsilon = epsilon);
            }

            // Test normalize
            let mut scalar_v = a.clone();
            let mut simd_v = a.clone();
            let scalar_old_norm = scalar_backend.normalize(&mut scalar_v);
            let simd_old_norm = simd_dispatcher.normalize(&mut simd_v);
            assert_relative_eq!(scalar_old_norm, simd_old_norm, epsilon = epsilon);
            for i in 0..size {
                assert_relative_eq!(scalar_v[i], simd_v[i], epsilon = epsilon);
            }

            // Test hadamard product
            let mut scalar_result = DVector::zeros(size);
            let mut simd_result = DVector::zeros(size);
            scalar_backend.hadamard_product(&a, &b, &mut scalar_result);
            simd_dispatcher.hadamard_product(&a, &b, &mut simd_result);
            for i in 0..size {
                assert_relative_eq!(scalar_result[i], simd_result[i], epsilon = epsilon);
            }

            // Test max_abs_diff
            let scalar_max_diff = scalar_backend.max_abs_diff(&a, &b);
            let simd_max_diff = simd_dispatcher.max_abs_diff(&a, &b);
            assert_relative_eq!(scalar_max_diff, simd_max_diff, epsilon = epsilon);
        }
    }

    #[test]
    fn test_f32_backend_equivalence() {
        // Test various sizes including:
        // - Small sizes that might not benefit from SIMD
        // - Sizes that are multiples of SIMD width (8 for f32x8)
        // - Sizes that are not multiples of SIMD width
        let test_sizes = vec![
            1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000,
        ];
        test_f32_backend_equivalence_impl(&test_sizes);
    }

    #[test]
    fn test_f64_backend_equivalence() {
        // Test various sizes including:
        // - Small sizes that might not benefit from SIMD
        // - Sizes that are multiples of SIMD width (4 for f64x4)
        // - Sizes that are not multiples of SIMD width
        let test_sizes = vec![
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
            257, 1000,
        ];
        test_f64_backend_equivalence_impl(&test_sizes);
    }

    #[test]
    fn test_matrix_operations() {
        let scalar_backend = ScalarBackend::<f64>::new();
        let simd_dispatcher = get_dispatcher::<f64>();

        // Test matrix sizes
        let test_cases = vec![(10, 10), (16, 16), (17, 17), (32, 64), (100, 50)];

        for (rows, cols) in test_cases {
            let a = DMatrix::from_fn(rows, cols, |i, j| ((i * cols + j) as f64).sin());
            let x = DVector::from_fn(cols, |i, _| (i as f64).cos());
            let mut scalar_y = DVector::zeros(rows);
            let mut simd_y = DVector::zeros(rows);

            let alpha = 2.0;
            let beta = 0.5;

            // Initialize y vectors
            for i in 0..rows {
                scalar_y[i] = (i as f64) * 0.1;
                simd_y[i] = (i as f64) * 0.1;
            }

            // Test gemv
            scalar_backend.gemv(&a, &x, &mut scalar_y, alpha, beta);
            simd_dispatcher.gemv(&a, &x, &mut simd_y, alpha, beta);

            for i in 0..rows {
                assert_relative_eq!(scalar_y[i], simd_y[i], epsilon = 1e-10);
            }

            // Test Frobenius norm
            let scalar_norm = scalar_backend.frobenius_norm(&a);
            let simd_norm = simd_dispatcher.frobenius_norm(&a);
            assert_relative_eq!(scalar_norm, simd_norm, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_edge_cases() {
        let dispatcher = get_dispatcher::<f64>();

        // Test empty vectors
        let empty = DVector::<f64>::zeros(0);
        assert_eq!(dispatcher.dot_product(&empty, &empty), 0.0);
        assert_eq!(dispatcher.norm(&empty), 0.0);

        // Test zero vector
        let zeros = DVector::<f64>::zeros(100);
        assert_eq!(dispatcher.norm(&zeros), 0.0);
        let mut normalized_zeros = zeros.clone();
        let old_norm = dispatcher.normalize(&mut normalized_zeros);
        assert_eq!(old_norm, 0.0);
        // Normalized zero vector should remain zero
        for i in 0..100 {
            assert_eq!(normalized_zeros[i], 0.0);
        }

        // Test single element
        let single = DVector::from_vec(vec![3.0]);
        assert_eq!(dispatcher.norm(&single), 3.0);

        // Test very small values (denormalized numbers)
        let small = DVector::from_vec(vec![1e-308, 1e-308, 1e-308]);
        let norm = dispatcher.norm(&small);
        assert!(norm > 0.0);
        assert!(norm < 1e-307);

        // Test very large values
        let large = DVector::from_vec(vec![1e100, 1e100, 1e100]);
        let norm = dispatcher.norm(&large);
        assert!(norm > 1e100);
        assert!(norm < 1e101);
    }

    #[test]
    fn test_alignment_sensitivity() {
        // Test that SIMD operations work correctly with different memory alignments
        let dispatcher = get_dispatcher::<f32>();

        // Create misaligned vectors by using subslices
        let buffer = vec![0.0_f32; 1024];

        for offset in 0..8 {
            let size = 100;
            let a_data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
            let b_data: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();

            // Create vectors with different alignments
            let mut a_buffer = buffer.clone();
            let mut b_buffer = buffer.clone();

            a_buffer[offset..offset + size].copy_from_slice(&a_data);
            b_buffer[offset..offset + size].copy_from_slice(&b_data);

            let a = DVector::from_data(VecStorage::new(
                Dyn(size),
                Const::<1>,
                a_buffer[offset..offset + size].to_vec(),
            ));
            let b = DVector::from_data(VecStorage::new(
                Dyn(size),
                Const::<1>,
                b_buffer[offset..offset + size].to_vec(),
            ));

            // All operations should work correctly regardless of alignment
            let dot = dispatcher.dot_product(&a, &b);
            let norm = dispatcher.norm(&a);

            // Compare with reference implementation
            let ref_dot: f32 = a_data.iter().zip(&b_data).map(|(x, y)| x * y).sum();
            let ref_norm: f32 = a_data.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Use appropriate f32 tolerance and ULP comparison for better numerical stability
            assert_ulps_eq!(dot, ref_dot, max_ulps = 10);
            assert_ulps_eq!(norm, ref_norm, max_ulps = 10);
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    /// Measure speedup of SIMD operations compared to scalar
    #[test]
    #[ignore] // Run with --ignored to include performance tests
    fn benchmark_simd_speedup() {
        let scalar_backend = ScalarBackend::<f32>::new();
        let simd_dispatcher = get_dispatcher::<f32>();

        let sizes = vec![100, 1000, 10000, 100000];

        println!("\nSIMD Performance Comparison (f32):");
        println!(
            "{:<10} {:<15} {:<15} {:<10}",
            "Size", "Scalar (μs)", "SIMD (μs)", "Speedup"
        );
        println!("{:-<50}", "");

        for size in sizes {
            let a = DVector::from_fn(size, |i, _| (i as f32).sin());
            let b = DVector::from_fn(size, |i, _| (i as f32).cos());

            // Warmup
            for _ in 0..10 {
                let _ = scalar_backend.dot_product(&a, &b);
                let _ = simd_dispatcher.dot_product(&a, &b);
            }

            // Benchmark scalar
            let start = Instant::now();
            let iterations = 1000;
            for _ in 0..iterations {
                let _ = scalar_backend.dot_product(&a, &b);
            }
            let scalar_time = start.elapsed().as_micros() as f64 / iterations as f64;

            // Benchmark SIMD
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = simd_dispatcher.dot_product(&a, &b);
            }
            let simd_time = start.elapsed().as_micros() as f64 / iterations as f64;

            let speedup = scalar_time / simd_time;

            println!(
                "{:<10} {:<15.2} {:<15.2} {:<10.2}x",
                size, scalar_time, simd_time, speedup
            );
        }
    }

    #[test]
    #[ignore] // Run with --ignored to include performance tests
    fn benchmark_operations() {
        let dispatcher = get_dispatcher::<f64>();
        let size = 10000;
        let iterations = 1000;

        let a = DVector::from_fn(size, |i, _| (i as f64).sin());
        let b = DVector::from_fn(size, |i, _| (i as f64).cos());
        let mut result = DVector::zeros(size);

        println!("\nOperation Benchmarks (f64, size={})", size);
        println!("{:<20} {:<15}", "Operation", "Time (μs)");
        println!("{:-<35}", "");

        // Dot product
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = dispatcher.dot_product(&a, &b);
        }
        let time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("{:<20} {:<15.2}", "dot_product", time);

        // Norm
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = dispatcher.norm(&a);
        }
        let time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("{:<20} {:<15.2}", "norm", time);

        // Add
        let start = Instant::now();
        for _ in 0..iterations {
            dispatcher.add(&a, &b, &mut result);
        }
        let time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("{:<20} {:<15.2}", "add", time);

        // Scale
        let mut v = a.clone();
        let start = Instant::now();
        for _ in 0..iterations {
            dispatcher.scale(&mut v, 2.0);
        }
        let time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("{:<20} {:<15.2}", "scale", time);

        // Hadamard product
        let start = Instant::now();
        for _ in 0..iterations {
            dispatcher.hadamard_product(&a, &b, &mut result);
        }
        let time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("{:<20} {:<15.2}", "hadamard_product", time);
    }
}

// Note: SIMD manifold operations are in internal module, so we test through dispatcher
#[cfg(test)]
mod simd_manifold_tests {
    use super::*;

    #[test]
    fn test_sphere_projection_via_dispatcher() {
        // Test various sizes
        let sizes = vec![3, 8, 16, 17, 32, 100];
        let dispatcher = get_dispatcher::<f64>();

        for size in sizes {
            let mut point = DVector::from_fn(size, |i, _| ((i + 1) as f64) * 0.5);

            // Store original for comparison
            let original = point.clone();

            // Normalize using SIMD dispatcher (equivalent to sphere projection)
            let old_norm = dispatcher.normalize(&mut point);

            // Verify it's on the unit sphere
            let norm = dispatcher.norm(&point);
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);

            // Verify old norm is correct
            assert_relative_eq!(old_norm, original.norm(), epsilon = 1e-10);

            // Verify direction is preserved
            let original_normalized = &original / original.norm();
            for i in 0..size {
                assert_relative_eq!(point[i], original_normalized[i], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_simd_operations_composition() {
        // Test that composing multiple SIMD operations works correctly
        // This is more relevant for SIMD testing than numerical algorithms
        let dispatcher = get_dispatcher::<f64>();
        let test_sizes = vec![10, 50, 100, 256];

        for size in test_sizes {
            let a = DVector::from_fn(size, |i, _| ((i as f64) * 0.1).sin());
            let b = DVector::from_fn(size, |i, _| ((i as f64) * 0.2).cos());
            let c = DVector::from_fn(size, |i, _| ((i as f64) * 0.3).tan());
            
            // Test composition: result = 2*a + 3*b - c
            let mut result = a.clone();
            dispatcher.scale(&mut result, 2.0);  // result = 2*a
            dispatcher.axpy(3.0, &b, &mut result);  // result = 2*a + 3*b
            dispatcher.axpy(-1.0, &c, &mut result);  // result = 2*a + 3*b - c
            
            // Verify against direct computation
            for i in 0..size {
                let expected = 2.0 * a[i] + 3.0 * b[i] - c[i];
                assert_relative_eq!(result[i], expected, epsilon = 1e-10);
            }
            
            // Test more complex operations
            let mut v1 = a.clone();
            let mut v2 = b.clone();
            
            // Normalize both vectors
            let _norm1 = dispatcher.normalize(&mut v1);
            let _norm2 = dispatcher.normalize(&mut v2);
            
            // Verify they are unit vectors
            assert_relative_eq!(dispatcher.norm(&v1), 1.0, epsilon = 1e-10);
            assert_relative_eq!(dispatcher.norm(&v2), 1.0, epsilon = 1e-10);
            
            // Test hadamard product of normalized vectors
            let mut hadamard_result = DVector::zeros(size);
            dispatcher.hadamard_product(&v1, &v2, &mut hadamard_result);
            
            // Verify hadamard product
            for i in 0..size {
                assert_relative_eq!(hadamard_result[i], v1[i] * v2[i], epsilon = 1e-10);
            }
            
            // Test max_abs_diff
            let max_diff = dispatcher.max_abs_diff(&v1, &v2);
            let mut expected_max_diff = 0.0_f64;
            for i in 0..size {
                expected_max_diff = expected_max_diff.max((v1[i] - v2[i]).abs());
            }
            assert_relative_eq!(max_diff, expected_max_diff, epsilon = 1e-10);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_simd_in_optimization_context() {
        // Simulate a simple gradient descent step using SIMD operations
        let dispatcher = get_dispatcher::<f64>();

        let dim = 100;
        let mut x = DVector::from_fn(dim, |i, _| (i as f64) * 0.01);
        let target = DVector::from_fn(dim, |i, _| ((i as f64) * 0.02).sin());

        // Simple quadratic cost: f(x) = 0.5 * ||x - target||^2
        // Gradient: grad f(x) = x - target

        let step_size = 0.01; // Smaller step size for better convergence
        let iterations = 1000; // More iterations

        for _ in 0..iterations {
            // Compute gradient = x - target
            let mut gradient = x.clone();
            dispatcher.axpy(-1.0, &target, &mut gradient);

            // Update x: x = x - step_size * gradient
            dispatcher.axpy(-step_size, &gradient, &mut x);
        }

        // Check convergence by computing ||x - target||
        let mut diff = x.clone();
        dispatcher.axpy(-1.0, &target, &mut diff);

        let final_error = dispatcher.norm(&diff);
        // Use a reasonable tolerance for gradient descent convergence
        let tolerance = 1e-3;
        assert!(
            final_error < tolerance,
            "Failed to converge: error = {} (expected < {})",
            final_error,
            tolerance
        );
    }
}

#[test]
fn test_simd_feature_detection() {
    use riemannopt_core::config::features::simd_config;

    let config = simd_config();
    println!("\nSIMD Configuration:");
    println!("  Enabled: {}", config.enabled);
    println!("  Min vector length: {}", config.min_vector_length);
    println!("  Use AVX-512: {}", config.use_avx512);
    println!("  Use FMA: {}", config.use_fma);

    // Verify dispatcher selection based on config
    let f32_dispatcher = get_dispatcher::<f32>();
    let f64_dispatcher = get_dispatcher::<f64>();

    // Test that dispatchers work correctly
    let v = DVector::from_vec(vec![1.0_f32, 2.0, 3.0]);
    let norm = f32_dispatcher.norm(&v);
    assert_ulps_eq!(norm, (14.0_f32).sqrt(), max_ulps = 5);

    let v = DVector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let norm = f64_dispatcher.norm(&v);
    assert_relative_eq!(norm, (14.0_f64).sqrt(), epsilon = 1e-10);
}
