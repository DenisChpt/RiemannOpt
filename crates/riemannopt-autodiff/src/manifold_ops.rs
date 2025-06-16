//! Manifold-specific operations for automatic differentiation.
//!
//! This module provides operations that are specific to Riemannian manifolds,
//! such as projections, retractions, and metric operations.

use crate::graph::Tensor;
use crate::ops::Op;
use nalgebra::DMatrix;
use std::fmt::Debug;

/// Projection onto the tangent space of a manifold.
#[derive(Debug, Clone)]
pub struct TangentProjection {
    /// The dimension of the manifold
    pub dim: usize,
}

impl TangentProjection {
    /// Creates a new tangent projection operation.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Op for TangentProjection {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "TangentProjection requires exactly 2 inputs: point and vector");
        let _point = &inputs[0];
        let vector = &inputs[1];
        
        // For Euclidean space, projection is identity
        // This is a placeholder - specific manifolds would override this
        vector.clone()
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        let point = &inputs[0];
        let _vector = &inputs[1];
        
        // Gradient w.r.t. point is zero for simple projections
        // Gradient w.r.t. vector is the projected gradient
        vec![
            Tensor::zeros(point.nrows(), point.ncols()),
            grad_output.clone(),
        ]
    }
    
    fn name(&self) -> &str {
        "TangentProjection"
    }
}

/// Projection onto a Stiefel manifold (orthonormal matrices).
#[derive(Debug, Clone)]
pub struct StiefelProjection;

impl Op for StiefelProjection {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "StiefelProjection requires exactly 1 input");
        let matrix = &inputs[0];
        
        // Project onto Stiefel manifold using QR decomposition
        let qr = matrix.clone().qr();
        qr.q() * DMatrix::identity(matrix.nrows(), matrix.ncols())
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // Gradient of QR decomposition projection
        // This is complex - for now, return a simple approximation
        vec![grad_output.clone()]
    }
    
    fn name(&self) -> &str {
        "StiefelProjection"
    }
}

/// Projection onto the unit sphere.
#[derive(Debug, Clone)]
pub struct SphereProjection;

/// Projection onto the tangent space of the sphere.
#[derive(Debug, Clone)]
pub struct SphereTangentProjection;

impl Op for SphereProjection {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "SphereProjection requires exactly 1 input");
        let vector = &inputs[0];
        
        // Normalize to unit norm
        let norm = vector.norm();
        if norm > 1e-10 {
            vector / norm
        } else {
            vector.clone()
        }
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        output: &Tensor,
    ) -> Vec<Tensor> {
        let vector = &inputs[0];
        let norm = vector.norm();
        
        if norm > 1e-10 {
            // Gradient of normalization
            // d/dx (x/||x||) = (I - xx^T/||x||^2) / ||x||
            let outer = output * output.transpose();
            let proj = DMatrix::identity(vector.nrows(), vector.nrows()) - outer;
            vec![(proj * grad_output) / norm]
        } else {
            vec![grad_output.clone()]
        }
    }
    
    fn name(&self) -> &str {
        "SphereProjection"
    }
}

impl Op for SphereTangentProjection {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "SphereTangentProjection requires 2 inputs: point and vector");
        let point = &inputs[0];
        let vector = &inputs[1];
        
        // Project vector onto tangent space of sphere at point
        // v_tan = v - <v, x>x (assuming x is normalized)
        let inner = (point.transpose() * vector)[(0, 0)];
        vector - point * inner
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        let point = &inputs[0];
        let vector = &inputs[1];
        
        // Gradient w.r.t. point: -<v, grad>x - <v, x>grad
        let v_dot_grad = (vector.transpose() * grad_output)[(0, 0)];
        let v_dot_x = (vector.transpose() * point)[(0, 0)];
        let grad_point = -(point * v_dot_grad + grad_output * v_dot_x);
        
        // Gradient w.r.t. vector: grad - <grad, x>x
        let grad_dot_x = (grad_output.transpose() * point)[(0, 0)];
        let grad_vector = grad_output - point * grad_dot_x;
        
        vec![grad_point, grad_vector]
    }
    
    fn name(&self) -> &str {
        "SphereTangentProjection"
    }
}

/// Inner product on a manifold.
#[derive(Debug, Clone)]
pub struct ManifoldInnerProduct {
    /// Type of manifold
    pub manifold_type: String,
}

impl ManifoldInnerProduct {
    /// Creates a new inner product operation for the given manifold type.
    pub fn new(manifold_type: impl Into<String>) -> Self {
        Self {
            manifold_type: manifold_type.into(),
        }
    }
}

impl Op for ManifoldInnerProduct {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 3, "ManifoldInnerProduct requires 3 inputs: point, u, v");
        let _point = &inputs[0];
        let u = &inputs[1];
        let v = &inputs[2];
        
        // For now, use Euclidean inner product
        // Specific manifolds would implement their own metrics
        let inner = u.component_mul(v).sum();
        Tensor::from_element(1, 1, inner)
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        let point = &inputs[0];
        let u = &inputs[1];
        let v = &inputs[2];
        
        let scalar = grad_output[(0, 0)];
        
        // Gradients for Euclidean inner product
        vec![
            Tensor::zeros(point.nrows(), point.ncols()), // No gradient w.r.t. point for Euclidean
            v * scalar,  // Gradient w.r.t. u
            u * scalar,  // Gradient w.r.t. v
        ]
    }
    
    fn name(&self) -> &str {
        "ManifoldInnerProduct"
    }
}

/// Exponential map on a manifold.
#[derive(Debug, Clone)]
pub struct ExponentialMap {
    /// Type of manifold
    pub manifold_type: String,
}

impl ExponentialMap {
    /// Creates a new exponential map for the given manifold type.
    pub fn new(manifold_type: impl Into<String>) -> Self {
        Self {
            manifold_type: manifold_type.into(),
        }
    }
}

impl Op for ExponentialMap {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "ExponentialMap requires 2 inputs: point and tangent");
        let point = &inputs[0];
        let tangent = &inputs[1];
        
        // For Euclidean space, exp map is just addition
        // Specific manifolds would implement geodesic exponential
        point + tangent
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // For Euclidean space, gradients pass through unchanged
        vec![grad_output.clone(), grad_output.clone()]
    }
    
    fn name(&self) -> &str {
        "ExponentialMap"
    }
}

/// Logarithmic map on a manifold.
#[derive(Debug, Clone)]
pub struct LogarithmicMap {
    /// Type of manifold
    pub manifold_type: String,
}

impl LogarithmicMap {
    /// Creates a new logarithmic map for the given manifold type.
    pub fn new(manifold_type: impl Into<String>) -> Self {
        Self {
            manifold_type: manifold_type.into(),
        }
    }
}

impl Op for LogarithmicMap {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "LogarithmicMap requires 2 inputs: x and y");
        let x = &inputs[0];
        let y = &inputs[1];
        
        // For Euclidean space, log map is just subtraction
        // Specific manifolds would implement geodesic logarithm
        y - x
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // For Euclidean space
        vec![-grad_output, grad_output.clone()]
    }
    
    fn name(&self) -> &str {
        "LogarithmicMap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_sphere_projection() {
        let op = SphereProjection;
        let input = DMatrix::from_row_slice(3, 1, &[3.0, 4.0, 0.0]);
        
        let result = op.forward(&[input.clone()]);
        
        // Should be normalized to unit length
        assert_relative_eq!(result.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 0)], 0.6, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 0.8, epsilon = 1e-10);
    }
    
    #[test]
    fn test_manifold_inner_product() {
        let op = ManifoldInnerProduct::new("euclidean");
        let point = DMatrix::zeros(2, 2);
        let u = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let v = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        
        let result = op.forward(&[point, u, v]);
        
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_eq!(result[(0, 0)], 70.0);
    }
    
    #[test]
    fn test_exponential_map() {
        let op = ExponentialMap::new("euclidean");
        let point = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let tangent = DMatrix::from_row_slice(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        
        let result = op.forward(&[point, tangent]);
        
        assert_relative_eq!(result[(0, 0)], 1.1, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 1)], 2.2, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 3.3, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 1)], 4.4, epsilon = 1e-10);
    }
}