//! Example demonstrating the adaptive backend system.

use nalgebra::DVector;
use riemannopt_core::{
    compute::backend::{BackendSelector, BackendSelection},
};

fn main() {
    println!("Adaptive Backend Selection Example");
    println!("=================================\n");
    
    // Create different backend selectors
    let mut auto_selector = BackendSelector::<f64>::new(BackendSelection::Auto);
    let mut fixed_selector = BackendSelector::<f64>::new(
        BackendSelection::Fixed("cpu".to_string())
    );
    let mut adaptive_selector = BackendSelector::<f64>::new(
        BackendSelection::Adaptive { cpu_threshold: 1000 }
    );
    
    // Test with different problem sizes
    let sizes = vec![10, 100, 1000, 10000];
    
    for size in sizes {
        println!("Problem size: {}", size);
        println!("-------------");
        
        // Select backend with auto strategy
        let backend = auto_selector.select_backend(size);
        println!("  Auto selection: {}", backend.name());
        
        // Select backend with fixed strategy
        let backend = fixed_selector.select_backend(size);
        println!("  Fixed selection: {}", backend.name());
        
        // Select backend with adaptive strategy
        let backend = adaptive_selector.select_backend(size);
        println!("  Adaptive selection: {}", backend.name());
        
        // Perform a simple operation to demonstrate backend usage
        let a = DVector::from_element(size, 1.0);
        let b = DVector::from_element(size, 2.0);
        
        let start = std::time::Instant::now();
        let dot_product = backend.dot(&a, &b).unwrap();
        let elapsed = start.elapsed();
        
        println!("  Dot product result: {}", dot_product);
        println!("  Computation time: {:?}\n", elapsed);
    }
    
    // Show available backends
    println!("Available backends:");
    for backend_name in auto_selector.available_backends() {
        println!("  - {}", backend_name);
    }
    
    // Demonstrate performance hints
    println!("\nBackend Performance Hints:");
    let backend = auto_selector.current_backend();
    println!("  Prefers batched operations: {}", backend.prefers_batched_operations());
    println!("  Optimal batch size: {}", backend.optimal_batch_size());
    println!("  Can transfer to device: {}", backend.can_transfer());
    
    // Demonstrate batch operations
    println!("\nBatch Operations Example:");
    let pairs = vec![
        (DVector::from_vec(vec![1.0, 2.0, 3.0]), DVector::from_vec(vec![4.0, 5.0, 6.0])),
        (DVector::from_vec(vec![7.0, 8.0, 9.0]), DVector::from_vec(vec![10.0, 11.0, 12.0])),
        (DVector::from_vec(vec![13.0, 14.0, 15.0]), DVector::from_vec(vec![16.0, 17.0, 18.0])),
    ];
    
    let start = std::time::Instant::now();
    let results = backend.batch_dot(&pairs).unwrap();
    let batch_elapsed = start.elapsed();
    
    println!("  Batch dot products: {:?}", results);
    println!("  Batch computation time: {:?}", batch_elapsed);
    
    // Compare with sequential computation
    let start = std::time::Instant::now();
    let _sequential_results: Vec<f64> = pairs.iter()
        .map(|(a, b)| backend.dot(a, b).unwrap())
        .collect();
    let seq_elapsed = start.elapsed();
    
    println!("  Sequential computation time: {:?}", seq_elapsed);
    println!("  Speedup from batching: {:.2}x", 
             seq_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64());
}