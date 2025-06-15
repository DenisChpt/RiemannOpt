//! Example: Riemannian Adam optimization on the sphere.
//!
//! This example demonstrates using the Adam optimizer to find the minimum
//! of a function defined on the unit sphere S^{n-1}.

use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    manifold::Manifold,
    optimizer::StoppingCriterion,
    types::DVector,
};
use riemannopt_manifolds::Sphere;
use riemannopt_optim::{Adam, AdamConfig};
use nalgebra::Dyn;

/// Simple quadratic function on the sphere
/// f(x) = ||x - target||^2
#[derive(Debug)]
struct SphericalQuadratic {
    target: DVector<f64>,
}

impl SphericalQuadratic {
    fn new(target: DVector<f64>) -> Self {
        Self { target }
    }
}

impl CostFunction<f64, Dyn> for SphericalQuadratic {
    fn cost(&self, x: &DVector<f64>) -> Result<f64> {
        let diff = x - &self.target;
        Ok(0.5 * diff.norm_squared())
    }
    
    fn cost_and_gradient(&self, x: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        let cost = self.cost(x)?;
        let grad = x - &self.target;
        Ok((cost, grad))
    }
}

fn main() -> Result<()> {
    println!("=== Riemannian Adam on Sphere S^2 ===\n");
    
    // Create a 3D sphere (S^2)
    let sphere = Sphere::new(3)?;
    
    // Target point on the sphere
    let target = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let cost_fn = SphericalQuadratic::new(target.clone());
    
    // Random initial point
    let initial_point = sphere.random_point();
    let initial_cost = cost_fn.cost(&initial_point)?;
    
    println!("Initial point: [{:.3}, {:.3}, {:.3}]", 
             initial_point[0], initial_point[1], initial_point[2]);
    println!("Initial cost: {:.6}", initial_cost);
    println!("Target point: [{:.3}, {:.3}, {:.3}]", 
             target[0], target[1], target[2]);
    
    // Configure Adam optimizer with more conservative parameters
    let config = AdamConfig::new()
        .with_learning_rate(0.001)  // Smaller learning rate
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_epsilon(1e-4)  // Larger epsilon for stability
        .with_gradient_clip(1.0);
    
    let mut optimizer = Adam::new(config);
    
    // Set stopping criteria
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(500)
        .with_gradient_tolerance(1e-5)
        .with_function_tolerance(1e-6);
    
    println!("\nOptimizing with Riemannian Adam...");
    
    // Run optimization
    let result = optimizer.optimize(
        &cost_fn,
        &sphere,
        &initial_point,
        &stopping_criterion,
    )?;
    
    println!("\n=== Optimization Results ===");
    println!("Final point: [{:.6}, {:.6}, {:.6}]", 
             result.point[0], result.point[1], result.point[2]);
    println!("Final cost: {:.6}", result.value);
    println!("Iterations: {}", result.iterations);
    println!("Converged: {}", result.converged);
    println!("Gradient norm: {:.6}", result.gradient_norm.unwrap_or(0.0));
    println!("Time: {:.3}s", result.duration.as_secs_f64());
    
    // Check that we're still on the sphere
    let final_norm = result.point.norm();
    println!("\nFinal point norm: {:.12} (should be 1.0)", final_norm);
    
    // Distance to target
    let distance = sphere.distance(&result.point, &target)?;
    println!("Distance to target: {:.6}", distance);
    
    // Compare with SGD
    println!("\n=== Comparison with SGD ===");
    
    use riemannopt_optim::{SGD, SGDConfig, StepSizeSchedule};
    
    let sgd_config = SGDConfig::new()
        .with_step_size(StepSizeSchedule::Constant(0.01))
        .with_gradient_clip(1.0);
    
    let mut sgd = SGD::new(sgd_config);
    
    let sgd_result = sgd.optimize(
        &cost_fn,
        &sphere,
        &initial_point,
        &stopping_criterion,
    )?;
    
    println!("SGD final cost: {:.6}", sgd_result.value);
    println!("SGD iterations: {}", sgd_result.iterations);
    println!("SGD time: {:.3}s", sgd_result.duration.as_secs_f64());
    
    // Compare convergence rates
    let adam_improvement = initial_cost - result.value;
    let sgd_improvement = initial_cost - sgd_result.value;
    
    println!("\nAdam improvement: {:.6}", adam_improvement);
    println!("SGD improvement: {:.6}", sgd_improvement);
    println!("Adam vs SGD: {:.1}x better", 
             adam_improvement / sgd_improvement.max(1e-10));
    
    Ok(())
}