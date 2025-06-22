//! Simple optimization example demonstrating basic usage of riemannopt-core.
//!
//! This example shows how to:
//! - Define a simple manifold (Euclidean space)
//! - Create a cost function
//! - Use line search
//! - Track optimization progress

use nalgebra::{DVector, Dyn};
use riemannopt_core::{
    prelude::*,
    manifold::{Manifold, Point, TangentVector},
    cost_function::{CostFunction, CountingCostFunction},
    optimization::line_search::{BacktrackingLineSearch, LineSearchParams},
    error::{ManifoldError, Result},
};
use std::time::Instant;

/// A simple Euclidean manifold for demonstration.
#[derive(Debug)]
struct EuclideanManifold {
    dim: usize,
}

impl EuclideanManifold {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for EuclideanManifold {
    fn name(&self) -> &str {
        "Euclidean"
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn is_point_on_manifold(&self, _point: &Point<f64, Dyn>, _tol: f64) -> bool {
        true
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &Point<f64, Dyn>,
        _vector: &TangentVector<f64, Dyn>,
        _tol: f64,
    ) -> bool {
        true
    }

    fn project_point(&self, point: &Point<f64, Dyn>, result: &mut Point<f64, Dyn>) {
        result.copy_from(point);
    }

    fn project_tangent(
        &self,
        _point: &Point<f64, Dyn>,
        vector: &TangentVector<f64, Dyn>,
        result: &mut TangentVector<f64, Dyn>,
    ) -> Result<()> {
        result.copy_from(vector);
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Point<f64, Dyn>,
        v1: &TangentVector<f64, Dyn>,
        v2: &TangentVector<f64, Dyn>,
    ) -> Result<f64> {
        Ok(v1.dot(v2))
    }

    fn retract(
        &self,
        point: &Point<f64, Dyn>,
        tangent: &TangentVector<f64, Dyn>,
        result: &mut Point<f64, Dyn>,
    ) -> Result<()> {
        *result = point + tangent;
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Point<f64, Dyn>,
        other: &Point<f64, Dyn>,
        result: &mut TangentVector<f64, Dyn>,
    ) -> Result<()> {
        *result = other - point;
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &Point<f64, Dyn>,
        euclidean_grad: &TangentVector<f64, Dyn>,
        result: &mut TangentVector<f64, Dyn>,
    ) -> Result<()> {
        result.copy_from(euclidean_grad);
        Ok(())
    }

    fn random_point(&self) -> Point<f64, Dyn> {
        DVector::from_fn(self.dim, |_, _| rand::random::<f64>() * 2.0 - 1.0)
    }

    fn random_tangent(&self, _point: &Point<f64, Dyn>, result: &mut TangentVector<f64, Dyn>) -> Result<()> {
        *result = DVector::from_fn(self.dim, |_, _| {
            rand::random::<f64>() * 2.0 - 1.0
        });
        Ok(())
    }
}

/// Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
#[derive(Debug)]
struct RosenbrockFunction;

impl CostFunction<f64, Dyn> for RosenbrockFunction {
    fn cost(&self, point: &Point<f64, Dyn>) -> Result<f64> {
        if point.len() < 2 {
            return Err(ManifoldError::dimension_mismatch(2, point.len()));
        }
        let x = point[0];
        let y = point[1];
        Ok((1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2))
    }

    fn gradient(&self, point: &Point<f64, Dyn>) -> Result<TangentVector<f64, Dyn>> {
        if point.len() < 2 {
            return Err(ManifoldError::dimension_mismatch(2, point.len()));
        }
        let x = point[0];
        let y = point[1];
        
        let mut grad = DVector::zeros(point.len());
        grad[0] = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        grad[1] = 200.0 * (y - x * x);
        Ok(grad)
    }
}

/// Simple gradient descent optimizer
fn gradient_descent(
    manifold: &impl Manifold<f64, Dyn>,
    cost_fn: &impl CostFunction<f64, Dyn>,
    initial_point: Point<f64, Dyn>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(Point<f64, Dyn>, f64, usize)> {
    let mut point = initial_point;
    let mut line_search = BacktrackingLineSearch::new();
    let ls_params = LineSearchParams::backtracking();
    
    println!("Starting optimization from point: [{:.3}, {:.3}]", point[0], point[1]);
    println!("Initial cost: {:.6}", cost_fn.cost(&point)?);
    println!();
    
    for iter in 0..max_iterations {
        let value = cost_fn.cost(&point)?;
        let euclidean_grad = cost_fn.gradient(&point)?;
        let mut gradient = DVector::zeros(point.len());
        manifold.euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut gradient)?;
        let grad_norm = manifold.norm(&point, &gradient)?;
        
        if iter % 10 == 0 {
            println!(
                "Iteration {:3}: cost = {:.6}, |grad| = {:.6}, point = [{:.3}, {:.3}]",
                iter, value, grad_norm, point[0], point[1]
            );
        }
        
        if grad_norm < tolerance {
            println!("\nConverged! Gradient norm below tolerance.");
            return Ok((point, value, iter));
        }
        
        // Compute descent direction (negative gradient)
        let direction = -&gradient;
        
        // Perform line search
        let ls_result = line_search.search(
            cost_fn,
            manifold,
            &point,
            value,
            &gradient,
            &direction,
            &ls_params,
        )?;
        
        if !ls_result.success {
            println!("\nLine search failed at iteration {}", iter);
            return Ok((point, value, iter));
        }
        
        point = ls_result.new_point;
    }
    
    println!("\nMaximum iterations reached.");
    let final_value = cost_fn.cost(&point)?;
    Ok((point, final_value, max_iterations))
}

fn main() -> Result<()> {
    println!("=== Riemannian Optimization Example: Rosenbrock Function ===\n");
    
    // Create a 2D Euclidean manifold
    let manifold = EuclideanManifold::new(2);
    println!("Manifold: {} (dimension: {})", manifold.name(), manifold.dimension());
    
    // Create the Rosenbrock cost function
    let cost_fn = RosenbrockFunction;
    
    // Count function evaluations
    let cost_fn = CountingCostFunction::new(cost_fn);
    
    // Starting point (far from optimum)
    let initial_point = DVector::from_vec(vec![-1.2, 1.0]);
    
    // Run optimization
    let start_time = Instant::now();
    let (optimal_point, optimal_value, iterations) = gradient_descent(
        &manifold,
        &cost_fn,
        initial_point,
        1000,
        1e-6,
    )?;
    let elapsed = start_time.elapsed();
    
    // Print results
    println!("\n=== Optimization Results ===");
    println!("Optimal point: [{:.6}, {:.6}]", optimal_point[0], optimal_point[1]);
    println!("Optimal value: {:.6}", optimal_value);
    println!("Iterations: {}", iterations);
    let (cost_evals, grad_evals, _) = cost_fn.counts();
    println!("Function evaluations: {}", cost_evals);
    println!("Gradient evaluations: {}", grad_evals);
    println!("Time elapsed: {:.3} seconds", elapsed.as_secs_f64());
    
    // The Rosenbrock function has a global minimum at (1, 1) with value 0
    println!("\nTrue optimum: [1.0, 1.0] with value 0.0");
    let distance_to_optimum = ((optimal_point[0] - 1.0).powi(2) + (optimal_point[1] - 1.0).powi(2)).sqrt();
    println!("Distance to true optimum: {:.6}", distance_to_optimum);
    
    Ok(())
}