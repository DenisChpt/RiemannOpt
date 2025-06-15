//! Example: SGD optimization on the sphere manifold
//!
//! This example demonstrates how to use the Riemannian SGD optimizer
//! to solve a simple optimization problem on the unit sphere.

use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    manifold::Manifold,
    optimizer::StoppingCriterion,
    types::DVector,
};
use riemannopt_manifolds::Sphere;
use riemannopt_optim::{SGD, SGDConfig};
use nalgebra::Dyn;

/// A simple quadratic cost function on the sphere.
/// 
/// This function tries to minimize the distance from a target point on the sphere.
#[derive(Debug)]
struct SphereCost {
    target: DVector<f64>,
}

impl SphereCost {
    fn new(target: DVector<f64>) -> Self {
        Self { target }
    }
}

impl CostFunction<f64, Dyn> for SphereCost {
    fn cost(&self, point: &DVector<f64>) -> Result<f64> {
        // Cost is the squared distance to target
        let diff = point - &self.target;
        Ok(0.5 * diff.norm_squared())
    }
    
    fn cost_and_gradient(&self, point: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        let cost = self.cost(point)?;
        let gradient = point - &self.target;
        Ok((cost, gradient))
    }
}

fn main() -> Result<()> {
    println!("🌐 Riemannian SGD Optimization on the Unit Sphere");
    println!("================================================\n");

    // Create the sphere manifold S^2 (2-sphere in 3D)
    let sphere = Sphere::new(3)?;
    println!("📍 Manifold: {} (dimension: {})", <Sphere as Manifold<f64, Dyn>>::name(&sphere), <Sphere as Manifold<f64, Dyn>>::dimension(&sphere));

    // Define target point on the sphere
    let target = DVector::from_vec(vec![1.0, 0.0, 0.0]); // North pole
    let target_normalized = <Sphere as Manifold<f64, Dyn>>::project_point(&sphere, &target);
    println!("🎯 Target point: [{:.3}, {:.3}, {:.3}]", 
             target_normalized[0], target_normalized[1], target_normalized[2]);

    // Create cost function
    let cost_fn = SphereCost::new(target_normalized.clone());

    // Create initial point (start from a different location)
    let initial_point = <Sphere as Manifold<f64, Dyn>>::project_point(&sphere, &DVector::from_vec(vec![0.0, 1.0, 1.0]));
    println!("🚀 Initial point: [{:.3}, {:.3}, {:.3}]", 
             initial_point[0], initial_point[1], initial_point[2]);
    
    let initial_cost = cost_fn.cost(&initial_point)?;
    println!("💰 Initial cost: {:.6}\n", initial_cost);

    // Configure SGD optimizer with momentum
    let config = SGDConfig::<f64>::new()
        .with_exponential_decay(0.1, 0.95)
        .with_classical_momentum(0.9)
        .with_gradient_clip(1.0);

    let mut sgd = SGD::new(config);
    println!("⚙️  Optimizer: {}", sgd.name());
    println!("📊 Configuration:");
    println!("   • Step size: Exponential decay (initial=0.1, rate=0.95)");
    println!("   • Momentum: Classical (coefficient=0.9)");
    println!("   • Gradient clipping: threshold=1.0\n");

    // Set up stopping criteria
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(1000)
        .with_gradient_tolerance(1e-8)
        .with_function_tolerance(1e-12);

    // Run optimization
    println!("🔄 Starting optimization...\n");
    let result = sgd.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;

    // Display results
    println!("✅ Optimization completed!");
    println!("🎉 Status: {}", if result.converged { "Converged" } else { "Did not converge" });
    println!("🔢 Iterations: {}", result.iterations);
    println!("⏱️  Duration: {:.3}s", result.duration.as_secs_f64());
    println!("🔍 Function evaluations: {}", result.function_evaluations);
    println!("∇ Gradient evaluations: {}", result.gradient_evaluations);
    
    println!("\n📈 Final results:");
    println!("🎯 Final point: [{:.6}, {:.6}, {:.6}]", 
             result.point[0], result.point[1], result.point[2]);
    println!("💰 Final cost: {:.10}", result.value);
    
    if let Some(grad_norm) = result.gradient_norm {
        println!("📉 Gradient norm: {:.2e}", grad_norm);
    }

    // Verify the point is on the sphere
    let point_norm = result.point.norm();
    println!("🧮 Point norm: {:.10} (should be ≈ 1.0)", point_norm);
    
    // Compute distance to target
    let distance_to_target = <Sphere as Manifold<f64, Dyn>>::distance(&sphere, &result.point, &target_normalized)?;
    println!("📏 Distance to target: {:.6}", distance_to_target);

    println!("\n✨ Optimization successfully completed!");

    Ok(())
}