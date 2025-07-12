//! Integration tests for automatic differentiation with manifolds.

use nalgebra::{DMatrix, DVector};
use riemannopt_autodiff::prelude::*;
use riemannopt_core::prelude::Manifold;
use riemannopt_manifolds::Sphere;
use std::collections::HashMap;

/// Tests gradient computation on the sphere manifold.
#[test]
fn test_sphere_optimization() {
    // Create autodiff graph
    let graph = Graph::new();
    let mut func = ManifoldFunction::new(graph);
    
    // Add point on sphere as input
    let x = func.add_input(
        DMatrix::from_row_slice(3, 1, &[1.0/3.0_f64.sqrt(); 3]),
        "sphere",
    );
    
    // Objective: maximize first component (same as minimize negative)
    let neg = func.graph.apply_op(
        Box::new(riemannopt_autodiff::ops::Negate),
        &[x],
    );
    let first_component = func.graph.apply_op(
        Box::new(FirstComponent),
        &[neg],
    );
    func.set_output(first_component);
    
    // Evaluate at a point
    let mut inputs = HashMap::new();
    let x_val = DMatrix::from_row_slice(3, 1, &[0.6, 0.8, 0.0]);
    inputs.insert(x, x_val.clone());
    
    let (value, grads) = func.value_and_grad(&inputs);
    
    // Check value
    assert_eq!(value, -0.6);
    
    // Check gradient is in tangent space
    let grad = grads.get(&x).unwrap();
    let inner_product = x_val.transpose() * grad;
    assert!(inner_product[(0, 0)].abs() < 1e-10);
}

/// Tests gradient computation on the Stiefel manifold.
#[test]
fn test_stiefel_optimization() {
    // Create autodiff graph
    let graph = Graph::new();
    let mut func = ManifoldFunction::new(graph);
    
    // Add orthogonal matrix as input
    let x = func.add_input(
        DMatrix::identity(3, 2),
        "stiefel",
    );
    
    // Objective: Frobenius norm squared
    let squared = func.graph.apply_op(
        Box::new(riemannopt_autodiff::ops::Multiply),
        &[x, x],
    );
    let output = func.graph.apply_op(
        Box::new(riemannopt_autodiff::ops::Sum::all()),
        &[squared],
    );
    func.set_output(output);
    
    // Evaluate
    let mut inputs = HashMap::new();
    inputs.insert(x, DMatrix::identity(3, 2));
    
    let (value, grads) = func.value_and_grad(&inputs);
    
    // For identity matrix, Frobenius norm squared is 2
    assert_eq!(value, 2.0);
    
    // Gradient should exist
    assert!(grads.contains_key(&x));
}

/// Tests that autodiff respects manifold constraints.
#[test]
fn test_manifold_constraint_preservation() {
    let graph = Graph::new();
    
    // Create point on sphere
    let x = graph.variable(DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]));
    
    // Apply operation that might violate constraint
    let scaled = graph.apply_op(
        Box::new(ScalarMultiply { scalar: 2.0 }),
        &[x.id],
    );
    
    // Project back to sphere
    let projected = graph.project_onto_manifold(scaled, "sphere");
    
    // Forward pass
    let result = graph.forward(projected).unwrap();
    
    // Check that result is on sphere
    let norm = result.norm();
    assert!((norm - 1.0).abs() < 1e-10);
}

/// Custom operation for testing: extracts first component.
#[derive(Debug, Clone)]
struct FirstComponent;

impl riemannopt_autodiff::ops::Op for FirstComponent {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1);
        Tensor::from_element(1, 1, inputs[0][(0, 0)])
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        let mut grad = Tensor::zeros(inputs[0].nrows(), inputs[0].ncols());
        grad[(0, 0)] = grad_output[(0, 0)];
        vec![grad]
    }
    
    fn name(&self) -> &str {
        "FirstComponent"
    }
}

/// Custom operation: scalar multiplication.
#[derive(Debug, Clone)]
struct ScalarMultiply {
    scalar: f64,
}

impl riemannopt_autodiff::ops::Op for ScalarMultiply {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1);
        &inputs[0] * self.scalar
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        vec![grad_output * self.scalar]
    }
    
    fn name(&self) -> &str {
        "ScalarMultiply"
    }
}

/// Tests integration with actual manifold types.
#[test]
fn test_real_manifold_integration() {
    // Create sphere manifold
    let sphere = Sphere::new(3).unwrap();
    
    // Create autodiff graph
    let graph = Graph::new();
    
    // Variable on sphere
    let x_val = nalgebra::DVector::from_column_slice(&[0.6, 0.8, 0.0]);
    let x_mat = DMatrix::from_row_slice(3, 1, &[0.6, 0.8, 0.0]);
    let x = graph.variable(x_mat.clone());
    
    // Gradient
    let grad_val = nalgebra::DVector::from_column_slice(&[1.0, 0.0, 0.0]);
    let grad_mat = DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]);
    let grad = graph.variable(grad_mat.clone());
    
    // Project gradient to tangent space using sphere manifold
    let mut tangent_grad = DVector::zeros(3);
    let mut workspace = riemannopt_core::memory::workspace::Workspace::new();
    sphere.project_tangent(&x_val, &grad_val, &mut tangent_grad, &mut workspace).unwrap();
    
    // Do the same with autodiff
    let auto_tangent = graph.riemannian_gradient(x.id, grad.id, "sphere");
    let auto_result = graph.forward(auto_tangent).unwrap();
    
    // Results should match
    for i in 0..3 {
        assert!((tangent_grad[i] - auto_result[(i, 0)]).abs() < 1e-10);
    }
}