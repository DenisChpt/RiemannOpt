//! Example demonstrating parallel gradient computation with adaptive strategy.

use nalgebra::DVector;
use riemannopt_core::{
    core::cost_function::{CostFunction, CostFunctionParallel, QuadraticCost},
    compute::cpu::parallel::ParallelConfig,
};

fn main() {
    println!("Parallel Gradient Computation Example");
    println!("=====================================\n");

    // Test different problem sizes
    let dimensions = vec![50, 100, 500, 1000, 5000];

    for dim in dimensions {
        println!("Testing dimension: {}", dim);
        
        // Create a simple quadratic cost function
        let a = nalgebra::DMatrix::<f64>::identity(dim, dim) * 2.0;
        let b = nalgebra::DVector::zeros(dim);
        let cost = QuadraticCost::new(a, b, 0.0);
        
        let x = DVector::from_element(dim, 1.0);
        
        // Time sequential gradient
        let start = std::time::Instant::now();
        let _grad_seq = cost.gradient(&x).unwrap();
        let seq_time = start.elapsed();
        
        // Time parallel gradient with default config (adaptive)
        let config = ParallelConfig::default();
        let start = std::time::Instant::now();
        let _grad_par = cost.gradient_fd_parallel_dvec(&x, &config).unwrap();
        let par_time = start.elapsed();
        
        // Time parallel gradient with custom config
        let config_custom = ParallelConfig::new()
            .with_min_dimension(50)
            .with_chunk_size(dim / 20);
        let start = std::time::Instant::now();
        let _grad_custom = cost.gradient_fd_parallel_dvec(&x, &config_custom).unwrap();
        let custom_time = start.elapsed();
        
        println!("  Sequential:      {:?}", seq_time);
        println!("  Parallel (auto): {:?}", par_time);
        println!("  Parallel (custom): {:?}", custom_time);
        println!("  Speedup (auto):  {:.2}x", seq_time.as_secs_f64() / par_time.as_secs_f64());
        println!("  Speedup (custom): {:.2}x\n", seq_time.as_secs_f64() / custom_time.as_secs_f64());
    }
    
    // Demonstrate adaptive strategy behavior
    println!("Adaptive Strategy Behavior");
    println!("--------------------------");
    
    let strategy = riemannopt_core::compute::cpu::parallel_strategy::get_adaptive_strategy();
    
    for &dim in &[10, 50, 100, 500, 1000, 5000] {
        let should_parallel = strategy.should_parallelize(dim);
        let chunk_size = strategy.optimal_chunk_size(dim);
        
        println!("Dimension {}: parallel={}, chunk_size={}", 
                 dim, should_parallel, chunk_size);
    }
}