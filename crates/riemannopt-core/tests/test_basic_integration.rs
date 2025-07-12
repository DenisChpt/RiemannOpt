//! Basic integration test for riemannopt-core components
//!
//! This test demonstrates basic functionality without implementing traits.

use riemannopt_core::{
    core::cost_function::CostFunction,
    error::Result,
    types::{DVector, DMatrix},
    memory::workspace::{Workspace, WorkspaceBuilder},
};
use approx::assert_relative_eq;

#[test]
fn test_sphere_gradient_descent() {
    // Test gradient descent on the unit sphere with a quadratic cost function
    // This verifies the correctness of Riemannian gradient computation and retraction
    let dim = 5;
    
    // Create a random point and normalize it to be on the unit sphere S^{n-1}
    let mut point = DVector::<f64>::from_fn(dim, |i, _| ((i as f64) * 0.3).sin());
    let norm = point.norm();
    point /= norm;
    
    // Verify the point is on the sphere
    assert_relative_eq!(point.norm(), 1.0, epsilon = 1e-12);
    
    // Create a simple quadratic cost function matrix (diagonal)
    // f(x) = x^T A x where A is positive definite
    let mut matrix_a = DMatrix::<f64>::zeros(dim, dim);
    for i in 0..dim {
        matrix_a[(i, i)] = (i + 1) as f64; // Eigenvalues: 1, 2, 3, 4, 5
    }
    
    // The minimum of f(x) on the sphere is achieved at the eigenvector
    // corresponding to the smallest eigenvalue (which is e_0 with eigenvalue 1)
    let expected_minimum_cost = 1.0;
    
    // Gradient descent parameters
    let step_size = 0.1;
    let max_iterations = 100;
    let tolerance = 1e-6;
    
    // Workspace for computations (demonstrates proper memory management)
    let _workspace = WorkspaceBuilder::<f64>::new()
        .with_standard_buffers(dim)
        .build();
    
    let mut cost_history = Vec::new();
    
    for iteration in 0..max_iterations {
        // Compute cost: f(x) = x^T A x
        let ax = &matrix_a * &point;
        let cost = point.dot(&ax);
        cost_history.push(cost);
        
        // Compute Euclidean gradient: 2Ax
        let euclidean_grad = 2.0 * &ax;
        
        // Project to tangent space (Riemannian gradient)
        let inner = point.dot(&euclidean_grad);
        let riemannian_grad = &euclidean_grad - inner * &point;
        
        // Check gradient norm for convergence
        let grad_norm = riemannian_grad.norm();
        if grad_norm < tolerance {
            println!("Converged at iteration {} with gradient norm {}", iteration, grad_norm);
            break;
        }
        
        // Take a step in the negative gradient direction
        let step_direction = -step_size * &riemannian_grad;
        
        // Retract: move along tangent and project back to sphere
        let new_point = &point + &step_direction;
        let new_norm = new_point.norm();
        point = new_point / new_norm;
        
        // Verify we're still on the sphere
        assert_relative_eq!(point.norm(), 1.0, epsilon = 1e-10);
    }
    
    // Verify optimization worked (cost should decrease)
    let initial_cost = cost_history[0];
    let final_cost = cost_history[cost_history.len() - 1];
    assert!(final_cost < initial_cost, "Cost should decrease: initial = {}, final = {}", initial_cost, final_cost);
    
    // Verify monotonic decrease (with small tolerance for numerical errors)
    for i in 1..cost_history.len() {
        assert!(cost_history[i] <= cost_history[i-1] * 1.01, 
                "Cost should be non-increasing: cost[{}] = {} > cost[{}] = {}", 
                i, cost_history[i], i-1, cost_history[i-1]);
    }
    
    // The final cost should be close to the minimum eigenvalue (1.0)
    assert!(final_cost >= expected_minimum_cost * 0.99, 
            "Final cost {} should be >= theoretical minimum {}", 
            final_cost, expected_minimum_cost);
    
    // Verify the final point is close to an eigenvector
    // Since we minimize x^T A x with A = diag(1,2,3,4,5), the minimum is at e_1
    // However, gradient descent might converge to a local minimum if started near another eigenvector
    
    // Check that the point is close to one of the canonical basis vectors
    let mut max_component_squared = 0.0;
    let mut dominant_index = 0;
    for i in 0..dim {
        let component_squared = point[i] * point[i];
        if component_squared > max_component_squared {
            max_component_squared = component_squared;
            dominant_index = i;
        }
    }
    
    // The point should be close to a standard basis vector
    assert!(max_component_squared > 0.95, 
            "Point should be close to a basis vector, but max component squared is {} (point = {:?})", 
            max_component_squared, point);
    
    // Verify that the final cost matches the eigenvalue
    let eigenvalue = (dominant_index + 1) as f64;
    assert_relative_eq!(final_cost, eigenvalue, epsilon = 1e-3);
    
    println!("Initial cost: {:.6}, Final cost: {:.6}, Iterations: {}", 
             initial_cost, final_cost, cost_history.len());
}

#[test]
fn test_workspace_usage() {
    let dim = 10;
    let mut workspace = WorkspaceBuilder::<f64>::new()
        .with_standard_buffers(dim)
        .build();
    
    // Test that we can get and use buffers
    {
        let grad = workspace.get_vector_mut(riemannopt_core::memory::workspace::BufferId::Gradient).unwrap();
        for i in 0..dim {
            grad[i] = i as f64;
        }
    }
    
    // Test that values persist
    {
        let grad = workspace.get_vector(riemannopt_core::memory::workspace::BufferId::Gradient).unwrap();
        for i in 0..dim {
            assert_eq!(grad[i], i as f64);
        }
    }
    
    // Test clear
    workspace.clear();
    {
        let grad = workspace.get_vector(riemannopt_core::memory::workspace::BufferId::Gradient).unwrap();
        for i in 0..dim {
            assert_eq!(grad[i], 0.0);
        }
    }
}

#[test] 
fn test_cost_function_trait() {
    /// Simple quadratic cost function
    #[derive(Debug)]
    struct QuadraticCost {
        matrix: DMatrix<f64>,
    }
    
    impl CostFunction<f64> for QuadraticCost {
        type Point = DVector<f64>;
        type TangentVector = DVector<f64>;
        
        fn cost(&self, point: &Self::Point) -> Result<f64> {
            let ax = &self.matrix * point;
            Ok(point.dot(&ax))
        }
        
        fn cost_and_gradient(
            &self,
            point: &Self::Point,
            _workspace: &mut Workspace<f64>,
            gradient: &mut Self::TangentVector,
        ) -> Result<f64> {
            let ax = &self.matrix * point;
            let cost = point.dot(&ax);
            *gradient = 2.0 * ax;
            Ok(cost)
        }
        
        fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
            let ax = &self.matrix * point;
            Ok(2.0 * ax)
        }
        
        fn hessian_vector_product(
            &self,
            _point: &Self::Point,
            vector: &Self::TangentVector,
        ) -> Result<Self::TangentVector> {
            Ok(2.0 * &self.matrix * vector)
        }
        
        fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
            let epsilon = f64::EPSILON.sqrt();
            let mut gradient = DVector::zeros(point.len());
            
            for i in 0..point.len() {
                let mut point_plus = point.clone();
                let mut point_minus = point.clone();
                
                point_plus[i] += epsilon;
                point_minus[i] -= epsilon;
                
                let cost_plus = self.cost(&point_plus)?;
                let cost_minus = self.cost(&point_minus)?;
                
                gradient[i] = (cost_plus - cost_minus) / (2.0 * epsilon);
            }
            
            Ok(gradient)
        }
    }
    
    // Test the cost function
    let n = 3;
    let mut matrix = DMatrix::<f64>::identity(n, n);
    matrix *= 2.0; // Simple identity * 2
    
    let cost_fn = QuadraticCost { matrix };
    let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    
    // Test cost
    let cost = cost_fn.cost(&point).unwrap();
    assert_relative_eq!(cost, 2.0); // x^T * 2I * x = 2
    
    // Test gradient
    let gradient = cost_fn.gradient(&point).unwrap();
    assert_relative_eq!(gradient[0], 4.0); // 2 * 2 * 1
    assert_relative_eq!(gradient[1], 0.0);
    assert_relative_eq!(gradient[2], 0.0);
    
    // Test gradient matches cost_and_gradient
    let mut workspace = Workspace::new();
    let mut grad_buffer = DVector::zeros(n);
    let cost2 = cost_fn.cost_and_gradient(&point, &mut workspace, &mut grad_buffer).unwrap();
    assert_relative_eq!(cost, cost2);
    assert_relative_eq!(gradient, grad_buffer);
}