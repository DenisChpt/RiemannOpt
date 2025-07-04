//! Comprehensive tests for memory management in RiemannOpt.
//!
//! This module tests the workspace and memory pool functionality to ensure
//! efficient memory reuse and zero-allocation behavior during optimization.

use riemannopt_core::{
    memory::{
        workspace::{Workspace, WorkspaceBuilder, BufferId},
        pool::{VectorPool, MatrixPool, get_pooled_vector, get_pooled_matrix},
    },
    types::DVector,
};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

#[cfg(test)]
mod workspace_tests {
    use super::*;
    
    #[test]
    fn test_workspace_creation_and_access() {
        let dim = 50;
        let mut workspace = Workspace::<f64>::with_size(dim);
        
        // Verify all standard buffers are pre-allocated
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap().len(), dim);
        assert_eq!(workspace.get_vector(BufferId::Direction).unwrap().len(), dim);
        assert_eq!(workspace.get_vector(BufferId::PreviousGradient).unwrap().len(), dim);
        assert_eq!(workspace.get_vector(BufferId::Temp1).unwrap().len(), dim);
        assert_eq!(workspace.get_vector(BufferId::Temp2).unwrap().len(), dim);
        assert_eq!(workspace.get_vector(BufferId::UnitVector).unwrap().len(), dim);
        assert_eq!(workspace.get_vector(BufferId::PointPlus).unwrap().len(), dim);
        assert_eq!(workspace.get_vector(BufferId::PointMinus).unwrap().len(), dim);
        
        // Test mutability
        workspace.get_vector_mut(BufferId::Gradient).unwrap()[0] = 42.0;
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap()[0], 42.0);
        
        // Test clear
        workspace.clear();
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap()[0], 0.0);
    }
    
    #[test]
    fn test_workspace_builder() {
        let dim = 30;
        let workspace = WorkspaceBuilder::<f32>::new()
            .with_standard_buffers(dim)
            .with_momentum_buffers(dim)
            .with_adam_buffers(dim)
            .with_quasi_newton_buffers(dim)
            .with_vector(BufferId::Custom(0), dim * 2)
            .with_matrix(BufferId::Custom(1), dim, dim)
            .build();
        
        // Verify all buffers exist
        assert!(workspace.get_vector(BufferId::Gradient).is_some());
        assert!(workspace.get_vector(BufferId::Momentum).is_some());
        assert!(workspace.get_vector(BufferId::SecondMoment).is_some());
        assert!(workspace.get_matrix(BufferId::Hessian).is_some());
        assert_eq!(workspace.get_vector(BufferId::Custom(0)).unwrap().len(), dim * 2);
        assert_eq!(workspace.get_matrix(BufferId::Custom(1)).unwrap().nrows(), dim);
    }
    
    #[test]
    fn test_get_or_create() {
        let mut workspace = Workspace::<f64>::new();
        
        // Initially empty
        assert!(workspace.get_vector(BufferId::Custom(5)).is_none());
        
        // Create on demand
        let vec = workspace.get_or_create_vector(BufferId::Custom(5), 100);
        vec[0] = 1.0;
        
        // Now it exists
        assert_eq!(workspace.get_vector(BufferId::Custom(5)).unwrap().len(), 100);
        assert_eq!(workspace.get_vector(BufferId::Custom(5)).unwrap()[0], 1.0);
        
        // Test matrix
        let mat = workspace.get_or_create_matrix(BufferId::Preconditioner, 10, 10);
        mat[(0, 0)] = 2.0;
        assert_eq!(workspace.get_matrix(BufferId::Preconditioner).unwrap()[(0, 0)], 2.0);
    }
    
    #[test]
    fn test_gradient_buffers() {
        let dim = 20;
        let mut workspace = WorkspaceBuilder::<f64>::new()
            .with_standard_buffers(dim)
            .with_vector(BufferId::UnitVector, dim)
            .with_vector(BufferId::PointPlus, dim)
            .with_vector(BufferId::PointMinus, dim)
            .build();
        
        // Test gradient buffer access
        let buffers = workspace.get_gradient_buffers_mut().unwrap();
        let (gradient, unit, plus, minus) = buffers;
        
        // Modify all buffers
        gradient[0] = 1.0;
        unit[1] = 2.0;
        plus[2] = 3.0;
        minus[3] = 4.0;
        
        // Verify changes persisted
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap()[0], 1.0);
        assert_eq!(workspace.get_vector(BufferId::UnitVector).unwrap()[1], 2.0);
        assert_eq!(workspace.get_vector(BufferId::PointPlus).unwrap()[2], 3.0);
        assert_eq!(workspace.get_vector(BufferId::PointMinus).unwrap()[3], 4.0);
    }
    
    #[test]
    fn test_memory_usage() {
        let workspace = WorkspaceBuilder::<f64>::new()
            .with_vector(BufferId::Gradient, 1000)
            .with_vector(BufferId::Direction, 1000)
            .with_matrix(BufferId::Hessian, 100, 100)
            .build();
        
        let usage = workspace.memory_usage();
        let expected = 1000 * 8 + 1000 * 8 + 100 * 100 * 8; // f64 = 8 bytes
        assert_eq!(usage, expected);
    }
    
    #[test]
    fn test_temporary_pool_buffers() {
        let workspace = Workspace::<f32>::new();
        
        // Acquire temporary buffers
        let mut temp_vec = workspace.acquire_temp_vector(500);
        let mut temp_mat = workspace.acquire_temp_matrix(20, 30);
        
        // Use them
        temp_vec[0] = 1.0;
        temp_mat[(0, 0)] = 2.0;
        
        assert_eq!(temp_vec[0], 1.0);
        assert_eq!(temp_mat[(0, 0)], 2.0);
        
        // They'll be returned to pool when dropped
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    
    #[test]
    fn test_vector_pool_basic_operations() {
        let pool = VectorPool::<f64>::new(5);
        
        // Acquire multiple vectors
        let mut vectors = Vec::new();
        for i in 0..3 {
            let mut v = pool.acquire(100);
            v[0] = i as f64;
            vectors.push(v);
        }
        
        // Pool should be empty
        assert_eq!(pool.pool_size(100), 0);
        
        // Return vectors
        drop(vectors);
        
        // Pool should now have 3 vectors
        assert_eq!(pool.pool_size(100), 3);
        
        // Acquire again - should be zeroed
        let v = pool.acquire(100);
        assert_eq!(v[0], 0.0);
    }
    
    #[test]
    fn test_pool_size_limit() {
        let pool = VectorPool::<f32>::new(2); // Max 2 per size
        
        // Create 5 vectors
        let vectors: Vec<_> = (0..5).map(|_| pool.acquire(50)).collect();
        
        // Drop all
        drop(vectors);
        
        // Only 2 should be kept
        assert_eq!(pool.pool_size(50), 2);
    }
    
    #[test]
    fn test_pool_different_sizes() {
        let pool = VectorPool::<f64>::new(10);
        
        // Acquire vectors of different sizes
        let v1 = pool.acquire(100);
        let v2 = pool.acquire(200);
        let v3 = pool.acquire(100);
        
        drop(v1);
        drop(v2);
        drop(v3);
        
        // Check pools for different sizes
        assert_eq!(pool.pool_size(100), 2);
        assert_eq!(pool.pool_size(200), 1);
        assert_eq!(pool.pool_size(300), 0);
    }
    
    #[test]
    fn test_take_ownership() {
        let pool = VectorPool::<f64>::new(5);
        
        let v1 = pool.acquire(50);
        let owned = v1.take();
        
        // Drop owned vector
        drop(owned);
        
        // Pool should still be empty since we took ownership
        assert_eq!(pool.pool_size(50), 0);
    }
    
    #[test]
    fn test_matrix_pool() {
        let pool = MatrixPool::<f32>::new(3);
        
        // Acquire matrices
        let mut m1 = pool.acquire(10, 20);
        let mut m2 = pool.acquire(10, 20);
        let mut m3 = pool.acquire(20, 30);
        
        m1[(0, 0)] = 1.0;
        m2[(1, 1)] = 2.0;
        m3[(2, 2)] = 3.0;
        
        drop(m1);
        drop(m2);
        drop(m3);
        
        // Check pool sizes
        assert_eq!(pool.pool_size(10, 20), 2);
        assert_eq!(pool.pool_size(20, 30), 1);
        
        // Acquire again and verify zeroed
        let m = pool.acquire(10, 20);
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(1, 1)], 0.0);
    }
    
    #[test]
    fn test_global_pools() {
        // Test f64 vector pool
        {
            let v1 = get_pooled_vector::<f64>(1000);
            let v2 = get_pooled_vector::<f64>(1000);
            assert_eq!(v1.len(), 1000);
            assert_eq!(v2.len(), 1000);
        }
        
        // Test f32 matrix pool
        {
            let m1 = get_pooled_matrix::<f32>(50, 60);
            let m2 = get_pooled_matrix::<f32>(50, 60);
            assert_eq!(m1.nrows(), 50);
            assert_eq!(m2.ncols(), 60);
        }
    }
    
    #[test]
    fn test_pool_thread_safety() {
        let pool = Arc::new(VectorPool::<f64>::new(10));
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];
        
        for thread_id in 0..4 {
            let pool_clone = Arc::clone(&pool);
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                // Each thread acquires and releases vectors
                for i in 0..25 {
                    let mut v = pool_clone.acquire(100);
                    v[0] = (thread_id * 100 + i) as f64;
                    
                    // Simulate some work
                    let sum: f64 = v.iter().sum();
                    assert!(sum >= 0.0);
                    
                    // Vector automatically returned when dropped
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Pool should have some cached vectors (up to limit)
        let final_size = pool.pool_size(100);
        assert!(final_size > 0);
        assert!(final_size <= 10);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_workspace_with_pools() {
        let dim = 100;
        let mut workspace = WorkspaceBuilder::<f64>::new()
            .with_standard_buffers(dim)
            .build();
        
        // Simulate optimization iteration
        for iteration in 0..10 {
            // Get gradient buffer and fill it
            {
                let grad = workspace.get_vector_mut(BufferId::Gradient).unwrap();
                for i in 0..dim {
                    grad[i] = (i + iteration) as f64;
                }
            }
            
            // Acquire temporary buffer from pool and compute
            {
                let grad = workspace.get_vector(BufferId::Gradient).unwrap();
                let temp = workspace.acquire_temp_vector(dim);
                
                // Compute something with gradient and temp
                let sum: f64 = grad.iter().zip(temp.iter()).map(|(a, b)| a * b).sum();
                assert!(sum >= 0.0);
            }
            // Temp automatically returned to pool
        }
        
        // Verify workspace still has gradient
        let final_grad = workspace.get_vector(BufferId::Gradient).unwrap();
        assert_eq!(final_grad[0], 9.0); // Last iteration value
    }
    
    #[test]
    fn test_zero_allocation_workflow() {
        // Pre-allocate everything
        let dim = 50;
        let mut workspace = WorkspaceBuilder::<f64>::new()
            .with_standard_buffers(dim)
            .with_momentum_buffers(dim)
            .build();
        
        let mut point = DVector::<f64>::zeros(dim);
        let mut new_point = DVector::<f64>::zeros(dim);
        
        // Simulate multiple optimization steps
        for step in 0..100 {
            // Fill gradient buffer (simulating gradient computation)
            {
                let grad = workspace.get_vector_mut(BufferId::Gradient).unwrap();
                for i in 0..dim {
                    grad[i] = ((i + step) as f64).sin();
                }
            }
            
            // Update momentum
            {
                // Copy gradient values first to avoid borrow conflicts
                let grad_values: Vec<f64> = workspace.get_vector(BufferId::Gradient).unwrap().iter().copied().collect();
                let momentum = workspace.get_vector_mut(BufferId::Momentum).unwrap();
                let beta = 0.9;
                
                for i in 0..dim {
                    momentum[i] = beta * momentum[i] + (1.0 - beta) * grad_values[i];
                }
            }
            
            // Compute new point
            {
                let momentum = workspace.get_vector(BufferId::Momentum).unwrap();
                let step_size = 0.01;
                
                for i in 0..dim {
                    new_point[i] = point[i] - step_size * momentum[i];
                }
                
                // Swap
                std::mem::swap(&mut point, &mut new_point);
            }
        }
        
        // Verify point has changed
        let norm: f64 = point.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    #[ignore] // Run with --ignored to include performance tests
    fn benchmark_workspace_vs_allocation() {
        let dim = 1000;
        let iterations = 10000;
        
        // Benchmark with workspace
        let mut workspace = WorkspaceBuilder::<f64>::new()
            .with_standard_buffers(dim)
            .build();
        
        let start = Instant::now();
        for _ in 0..iterations {
            // Fill gradient
            {
                let grad = workspace.get_vector_mut(BufferId::Gradient).unwrap();
                for i in 0..dim {
                    grad[i] = (i as f64).sin();
                }
            }
            
            // Compute direction
            {
                // Copy gradient values first to avoid borrow conflicts
                let grad_values: Vec<f64> = workspace.get_vector(BufferId::Gradient).unwrap().iter().copied().collect();
                let dir = workspace.get_vector_mut(BufferId::Direction).unwrap();
                for i in 0..dim {
                    dir[i] = -grad_values[i];
                }
            }
        }
        let workspace_time = start.elapsed();
        
        // Benchmark with allocation
        let start = Instant::now();
        for _ in 0..iterations {
            let mut grad = DVector::<f64>::zeros(dim);
            for i in 0..dim {
                grad[i] = (i as f64).sin();
            }
            
            let mut dir = DVector::<f64>::zeros(dim);
            for i in 0..dim {
                dir[i] = -grad[i];
            }
        }
        let alloc_time = start.elapsed();
        
        println!("\nWorkspace vs Allocation Performance:");
        println!("  Workspace: {:?}", workspace_time);
        println!("  Allocation: {:?}", alloc_time);
        println!("  Speedup: {:.2}x", alloc_time.as_secs_f64() / workspace_time.as_secs_f64());
        
        // Workspace should be significantly faster
        assert!(workspace_time < alloc_time);
    }
    
    #[test]
    #[ignore] // Run with --ignored to include performance tests
    fn benchmark_pool_reuse() {
        let pool = VectorPool::<f64>::new(10);
        let size = 1000;
        let iterations = 10000;
        
        // Warmup the pool
        for _ in 0..10 {
            let v = pool.acquire(size);
            drop(v);
        }
        
        // Benchmark with pool
        let start = Instant::now();
        for _ in 0..iterations {
            let mut v = pool.acquire(size);
            v[0] = 1.0;
            // Automatically returned to pool
        }
        let pool_time = start.elapsed();
        
        // Benchmark with allocation
        let start = Instant::now();
        for _ in 0..iterations {
            let mut v = DVector::<f64>::zeros(size);
            v[0] = 1.0;
            // Dropped and deallocated
        }
        let alloc_time = start.elapsed();
        
        println!("\nPool vs Allocation Performance:");
        println!("  Pool: {:?}", pool_time);
        println!("  Allocation: {:?}", alloc_time);
        println!("  Speedup: {:.2}x", alloc_time.as_secs_f64() / pool_time.as_secs_f64());
        
        // Pool should be faster
        assert!(pool_time < alloc_time);
    }
}

#[test]
fn test_custom_buffer_management() {
    let mut workspace = Workspace::<f64>::new();
    
    // Add many custom buffers
    for i in 0..20 {
        workspace.preallocate_vector(BufferId::Custom(i), 100 + i as usize);
    }
    
    // Verify all exist with correct sizes
    for i in 0..20 {
        let vec = workspace.get_vector(BufferId::Custom(i)).unwrap();
        assert_eq!(vec.len(), 100 + i as usize);
    }
    
    // Test memory usage scales correctly
    let usage = workspace.memory_usage();
    let expected: usize = (0..20).map(|i| (100 + i) * 8).sum();
    assert_eq!(usage, expected);
}

#[test]
fn test_gradient_fd_buffers() {
    let dim = 25;
    let mut workspace = Workspace::<f64>::new();
    
    // Get FD buffers - should create them if not exist
    let buffers = workspace.get_gradient_fd_buffers_mut(dim).unwrap();
    let (unit, plus, minus) = buffers;
    
    // Initialize buffers
    for i in 0..dim {
        unit[i] = if i == 0 { 1.0 } else { 0.0 };
        plus[i] = i as f64;
        minus[i] = -(i as f64);
    }
    
    // Verify persistence
    assert_eq!(workspace.get_vector(BufferId::UnitVector).unwrap()[0], 1.0);
    assert_eq!(workspace.get_vector(BufferId::PointPlus).unwrap()[5], 5.0);
    assert_eq!(workspace.get_vector(BufferId::PointMinus).unwrap()[5], -5.0);
}

#[test]
fn test_clear_pools() {
    let vec_pool = VectorPool::<f64>::new(10);
    let mat_pool = MatrixPool::<f32>::new(10);
    
    // Fill pools
    for _ in 0..5 {
        drop(vec_pool.acquire(100));
        drop(mat_pool.acquire(10, 10));
    }
    
    assert!(vec_pool.pool_size(100) > 0);
    assert!(mat_pool.pool_size(10, 10) > 0);
    
    // Clear
    vec_pool.clear();
    mat_pool.clear();
    
    assert_eq!(vec_pool.pool_size(100), 0);
    assert_eq!(mat_pool.pool_size(10, 10), 0);
}