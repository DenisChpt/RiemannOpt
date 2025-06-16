//! Integration of automatic differentiation with manifold operations.
//!
//! This module provides the bridge between the autodiff engine and the
//! manifold operations, enabling gradient computation through manifold constraints.

use crate::graph::{Graph, NodeId, Tensor};
use crate::manifold_ops::{
    TangentProjection, StiefelProjection, SphereProjection,
    SphereTangentProjection, ExponentialMap, LogarithmicMap, ManifoldInnerProduct,
};
use std::collections::HashMap;

/// Extension trait for Graph to add manifold-aware operations.
pub trait ManifoldGraph {
    /// Projects a point onto a manifold.
    fn project_onto_manifold(&self, x: NodeId, manifold_type: &str) -> NodeId;
    
    /// Projects a vector onto the tangent space at a point.
    fn project_tangent(&self, point: NodeId, vector: NodeId, dim: usize) -> NodeId;
    
    /// Computes the Riemannian inner product.
    fn manifold_inner_product(&self, point: NodeId, u: NodeId, v: NodeId, manifold_type: &str) -> NodeId;
    
    /// Applies the exponential map.
    fn exponential_map(&self, point: NodeId, tangent: NodeId, manifold_type: &str) -> NodeId;
    
    /// Applies the logarithmic map.
    fn logarithmic_map(&self, x: NodeId, y: NodeId, manifold_type: &str) -> NodeId;
    
    /// Computes the Riemannian gradient from Euclidean gradient.
    fn riemannian_gradient(&self, point: NodeId, euclidean_grad: NodeId, manifold_type: &str) -> NodeId;
}

impl ManifoldGraph for Graph {
    fn project_onto_manifold(&self, x: NodeId, manifold_type: &str) -> NodeId {
        match manifold_type {
            "sphere" => self.apply_op(Box::new(SphereProjection), &[x]),
            "stiefel" => self.apply_op(Box::new(StiefelProjection), &[x]),
            _ => panic!("Unsupported manifold type: {}", manifold_type),
        }
    }
    
    fn project_tangent(&self, point: NodeId, vector: NodeId, dim: usize) -> NodeId {
        self.apply_op(Box::new(TangentProjection::new(dim)), &[point, vector])
    }
    
    fn manifold_inner_product(&self, point: NodeId, u: NodeId, v: NodeId, manifold_type: &str) -> NodeId {
        self.apply_op(
            Box::new(ManifoldInnerProduct::new(manifold_type)),
            &[point, u, v],
        )
    }
    
    fn exponential_map(&self, point: NodeId, tangent: NodeId, manifold_type: &str) -> NodeId {
        self.apply_op(
            Box::new(ExponentialMap::new(manifold_type)),
            &[point, tangent],
        )
    }
    
    fn logarithmic_map(&self, x: NodeId, y: NodeId, manifold_type: &str) -> NodeId {
        self.apply_op(
            Box::new(LogarithmicMap::new(manifold_type)),
            &[x, y],
        )
    }
    
    fn riemannian_gradient(&self, point: NodeId, euclidean_grad: NodeId, manifold_type: &str) -> NodeId {
        // For most manifolds, the Riemannian gradient is the projection of
        // the Euclidean gradient onto the tangent space
        match manifold_type {
            "sphere" => {
                // Use specific sphere tangent projection
                self.apply_op(Box::new(SphereTangentProjection), &[point, euclidean_grad])
            }
            "stiefel" | "grassmann" => {
                // Project gradient onto tangent space
                let dim = self.get_value(point).map(|v| v.nrows()).unwrap_or(0);
                self.project_tangent(point, euclidean_grad, dim)
            }
            "spd" => {
                // For SPD manifolds with affine-invariant metric,
                // the gradient transformation is more complex
                // For now, use tangent projection as placeholder
                let dim = self.get_value(point).map(|v| v.nrows()).unwrap_or(0);
                self.project_tangent(point, euclidean_grad, dim)
            }
            _ => euclidean_grad, // Default to Euclidean gradient
        }
    }
}

/// Represents a differentiable function on a manifold.
pub struct ManifoldFunction {
    /// The computation graph
    pub graph: Graph,
    /// Input variables
    pub inputs: Vec<NodeId>,
    /// Output node
    pub output: NodeId,
    /// Manifold type for each input
    pub manifold_types: HashMap<NodeId, String>,
}

impl ManifoldFunction {
    /// Creates a new manifold function.
    pub fn new(graph: Graph) -> Self {
        let dummy_output = graph.variable(Tensor::zeros(1, 1)).id;
        Self {
            graph,
            inputs: Vec::new(),
            output: dummy_output,
            manifold_types: HashMap::new(),
        }
    }
    
    /// Adds an input variable on a specific manifold.
    pub fn add_input(&mut self, value: Tensor, manifold_type: &str) -> NodeId {
        let var = self.graph.variable(value);
        self.inputs.push(var.id);
        self.manifold_types.insert(var.id, manifold_type.to_string());
        var.id
    }
    
    /// Sets the output node.
    pub fn set_output(&mut self, output: NodeId) {
        self.output = output;
    }
    
    /// Computes the value and Riemannian gradients.
    pub fn value_and_grad(&self, inputs: &HashMap<NodeId, Tensor>) -> (f64, HashMap<NodeId, Tensor>) {
        // Set input values
        for (&node_id, value) in inputs {
            self.graph.set_value(node_id, value.clone());
        }
        
        // Forward pass
        let output_value = self.graph.forward(self.output)
            .expect("Forward pass failed");
        
        // Assume scalar output
        let value = output_value[(0, 0)];
        
        // Backward pass
        let euclidean_grads = crate::backward::backward(&self.graph, self.output, None);
        
        // Convert to Riemannian gradients
        let mut riemannian_grads = HashMap::new();
        for &input_id in &self.inputs {
            if let Some(euclidean_grad) = euclidean_grads.get(&input_id) {
                if let Some(manifold_type) = self.manifold_types.get(&input_id) {
                    // Create a new node for the Euclidean gradient
                    let grad_node = self.graph.constant(euclidean_grad.clone());
                    
                    // Convert to Riemannian gradient
                    let riem_grad_node = self.graph.riemannian_gradient(
                        input_id,
                        grad_node,
                        manifold_type,
                    );
                    
                    // Compute the Riemannian gradient value
                    if let Some(riem_grad) = self.graph.forward(riem_grad_node) {
                        riemannian_grads.insert(input_id, riem_grad);
                    }
                }
            }
        }
        
        (value, riemannian_grads)
    }
}

/// Helper for creating optimization problems on manifolds.
pub struct ManifoldOptimizationProblem {
    /// The objective function
    pub objective: ManifoldFunction,
    /// Constraints (if any)
    pub constraints: Vec<ManifoldFunction>,
}

impl ManifoldOptimizationProblem {
    /// Creates a new optimization problem.
    pub fn new(objective: ManifoldFunction) -> Self {
        Self {
            objective,
            constraints: Vec::new(),
        }
    }
    
    /// Adds a constraint function.
    pub fn add_constraint(&mut self, constraint: ManifoldFunction) {
        self.constraints.push(constraint);
    }
    
    /// Evaluates the objective and constraints.
    pub fn evaluate(&self, inputs: &HashMap<NodeId, Tensor>) -> (f64, Vec<f64>) {
        let (obj_value, _) = self.objective.value_and_grad(inputs);
        
        let constraint_values: Vec<f64> = self.constraints
            .iter()
            .map(|c| c.value_and_grad(inputs).0)
            .collect();
        
        (obj_value, constraint_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    
    #[test]
    fn test_manifold_graph_sphere() {
        let graph = Graph::new();
        
        // Create a point not on the sphere
        let x = graph.variable(DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 2.0]));
        
        // Project onto sphere
        let x_proj = graph.project_onto_manifold(x.id, "sphere");
        
        // Forward pass
        let result = graph.forward(x_proj).unwrap();
        
        // Check that result has unit norm
        let norm = result.norm();
        assert!((norm - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_riemannian_gradient() {
        let graph = Graph::new();
        
        // Point on sphere
        let x = graph.variable(DMatrix::from_row_slice(3, 1, &[0.6, 0.8, 0.0]));
        
        // Euclidean gradient
        let grad = graph.variable(DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]));
        
        // Convert to Riemannian gradient
        let riem_grad = graph.riemannian_gradient(x.id, grad.id, "sphere");
        
        // Forward pass
        let result = graph.forward(riem_grad).unwrap();
        
        // For sphere, Riemannian gradient should be orthogonal to x
        let x_val = graph.get_value(x.id).unwrap();
        let inner_product = x_val.transpose() * &result;
        assert!(inner_product[(0, 0)].abs() < 1e-10);
    }
    
    #[test]
    fn test_manifold_function() {
        let graph = Graph::new();
        let mut func = ManifoldFunction::new(graph);
        
        // Add input on sphere
        let x = func.add_input(
            DMatrix::from_row_slice(3, 1, &[0.6, 0.8, 0.0]),
            "sphere",
        );
        
        // Simple objective: sum of squares
        let squared = func.graph.apply_op(
            Box::new(crate::ops::Multiply),
            &[x, x],
        );
        let output = func.graph.apply_op(
            Box::new(crate::ops::Sum::all()),
            &[squared],
        );
        func.set_output(output);
        
        // Evaluate
        let mut inputs = HashMap::new();
        inputs.insert(x, DMatrix::from_row_slice(3, 1, &[0.6, 0.8, 0.0]));
        
        let (value, grads) = func.value_and_grad(&inputs);
        
        // Value should be 0.36 + 0.64 + 0 = 1.0
        assert!((value - 1.0).abs() < 1e-10);
        
        // Gradient should exist
        assert!(grads.contains_key(&x));
    }
}