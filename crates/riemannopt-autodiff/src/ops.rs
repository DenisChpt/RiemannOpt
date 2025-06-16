//! Operations for the computation graph.
//!
//! This module defines the operations that can be performed in the
//! computation graph, including their forward and backward computations.

use crate::graph::Tensor;
use nalgebra::DMatrix;
use std::fmt::Debug;

/// Trait for operations in the computation graph.
pub trait Op: Debug {
    /// Performs the forward computation.
    fn forward(&self, inputs: &[Tensor]) -> Tensor;
    
    /// Computes the gradient with respect to each input.
    /// 
    /// # Arguments
    /// * `grad_output` - The gradient flowing from the output
    /// * `inputs` - The input values used in the forward pass
    /// * `output` - The output value from the forward pass
    /// 
    /// # Returns
    /// A vector of gradients, one for each input
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        output: &Tensor,
    ) -> Vec<Tensor>;
    
    /// Returns the name of this operation.
    fn name(&self) -> &str;
}

/// Enumeration of operation types for easier matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Add,
    Multiply,
    MatMul,
    Transpose,
    Sum,
    Mean,
    Pow,
    Exp,
    Log,
    Tanh,
    Sigmoid,
    ReLU,
    Negate,
    Reshape,
}

/// Element-wise addition operation.
#[derive(Debug, Clone)]
pub struct Add;

impl Op for Add {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "Add requires exactly 2 inputs");
        &inputs[0] + &inputs[1]
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // Gradient of addition is just the output gradient for both inputs
        vec![grad_output.clone(), grad_output.clone()]
    }
    
    fn name(&self) -> &str {
        "Add"
    }
}

/// Element-wise multiplication operation.
#[derive(Debug, Clone)]
pub struct Multiply;

impl Op for Multiply {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "Multiply requires exactly 2 inputs");
        inputs[0].component_mul(&inputs[1])
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // d/dx (x * y) = y
        // d/dy (x * y) = x
        vec![
            grad_output.component_mul(&inputs[1]),
            grad_output.component_mul(&inputs[0]),
        ]
    }
    
    fn name(&self) -> &str {
        "Multiply"
    }
}

/// Matrix multiplication operation.
#[derive(Debug, Clone)]
pub struct MatMul;

impl Op for MatMul {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "MatMul requires exactly 2 inputs");
        &inputs[0] * &inputs[1]
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // d/dA (A * B) = grad_output * B^T
        // d/dB (A * B) = A^T * grad_output
        vec![
            grad_output * inputs[1].transpose(),
            inputs[0].transpose() * grad_output,
        ]
    }
    
    fn name(&self) -> &str {
        "MatMul"
    }
}

/// Matrix transpose operation.
#[derive(Debug, Clone)]
pub struct Transpose;

impl Op for Transpose {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Transpose requires exactly 1 input");
        inputs[0].transpose()
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // Gradient of transpose is transpose of gradient
        vec![grad_output.transpose()]
    }
    
    fn name(&self) -> &str {
        "Transpose"
    }
}

/// Sum operation (reduces to scalar).
#[derive(Debug, Clone)]
pub struct Sum {
    /// Optional axis to sum along (None means sum all elements)
    pub axis: Option<usize>,
    /// Whether to keep dimensions
    pub keepdim: bool,
}

impl Sum {
    /// Creates a new Sum operation that sums all elements.
    pub fn all() -> Self {
        Self {
            axis: None,
            keepdim: false,
        }
    }
    
    /// Creates a new Sum operation along a specific axis.
    pub fn along_axis(axis: usize, keepdim: bool) -> Self {
        Self {
            axis: Some(axis),
            keepdim,
        }
    }
}

impl Op for Sum {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Sum requires exactly 1 input");
        let input = &inputs[0];
        
        match self.axis {
            None => {
                // Sum all elements to a 1x1 matrix
                Tensor::from_element(1, 1, input.sum())
            }
            Some(0) => {
                // Sum along rows (result is 1 x ncols)
                let mut result = DMatrix::zeros(1, input.ncols());
                for j in 0..input.ncols() {
                    let sum: f64 = (0..input.nrows()).map(|i| input[(i, j)]).sum();
                    result[(0, j)] = sum;
                }
                result
            }
            Some(1) => {
                // Sum along columns (result is nrows x 1)
                let mut result = DMatrix::zeros(input.nrows(), 1);
                for i in 0..input.nrows() {
                    let sum: f64 = (0..input.ncols()).map(|j| input[(i, j)]).sum();
                    result[(i, 0)] = sum;
                }
                result
            }
            _ => panic!("Invalid axis for Sum operation"),
        }
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        let input_shape = (inputs[0].nrows(), inputs[0].ncols());
        
        match self.axis {
            None => {
                // Broadcast gradient to input shape
                vec![Tensor::from_element(
                    input_shape.0,
                    input_shape.1,
                    grad_output[(0, 0)],
                )]
            }
            Some(0) => {
                // Broadcast along rows
                let grad = DMatrix::from_fn(input_shape.0, input_shape.1, |_, j| {
                    grad_output[(0, j)]
                });
                vec![grad]
            }
            Some(1) => {
                // Broadcast along columns
                let grad = DMatrix::from_fn(input_shape.0, input_shape.1, |i, _| {
                    grad_output[(i, 0)]
                });
                vec![grad]
            }
            _ => panic!("Invalid axis for Sum backward"),
        }
    }
    
    fn name(&self) -> &str {
        "Sum"
    }
}

/// Negation operation.
#[derive(Debug, Clone)]
pub struct Negate;

impl Op for Negate {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Negate requires exactly 1 input");
        -&inputs[0]
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // Gradient of negation is negative gradient
        vec![-grad_output]
    }
    
    fn name(&self) -> &str {
        "Negate"
    }
}

/// ReLU (Rectified Linear Unit) activation.
#[derive(Debug, Clone)]
pub struct ReLU;

impl Op for ReLU {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "ReLU requires exactly 1 input");
        inputs[0].map(|x| x.max(0.0))
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // Gradient is 1 where input > 0, else 0
        let mask = inputs[0].map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        vec![grad_output.component_mul(&mask)]
    }
    
    fn name(&self) -> &str {
        "ReLU"
    }
}

/// Exponential operation.
#[derive(Debug, Clone)]
pub struct Exp;

impl Op for Exp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Exp requires exactly 1 input");
        inputs[0].map(|x| x.exp())
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        output: &Tensor,
    ) -> Vec<Tensor> {
        // d/dx exp(x) = exp(x)
        vec![grad_output.component_mul(output)]
    }
    
    fn name(&self) -> &str {
        "Exp"
    }
}

/// Natural logarithm operation.
#[derive(Debug, Clone)]
pub struct Log;

impl Op for Log {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Log requires exactly 1 input");
        inputs[0].map(|x| x.ln())
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // d/dx log(x) = 1/x
        let grad = grad_output.component_div(&inputs[0]);
        vec![grad]
    }
    
    fn name(&self) -> &str {
        "Log"
    }
}

/// Power operation (x^n for constant n).
#[derive(Debug, Clone)]
pub struct Pow {
    pub exponent: f64,
}

impl Op for Pow {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Pow requires exactly 1 input");
        inputs[0].map(|x| x.powf(self.exponent))
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // d/dx x^n = n * x^(n-1)
        let grad = inputs[0].map(|x| self.exponent * x.powf(self.exponent - 1.0));
        vec![grad_output.component_mul(&grad)]
    }
    
    fn name(&self) -> &str {
        "Pow"
    }
}

/// Mean operation.
#[derive(Debug, Clone)]
pub struct Mean {
    /// Optional axis to compute mean along (None means mean of all elements)
    pub axis: Option<usize>,
    /// Whether to keep dimensions
    pub keepdim: bool,
}

impl Mean {
    /// Creates a new Mean operation that computes mean of all elements.
    pub fn all() -> Self {
        Self {
            axis: None,
            keepdim: false,
        }
    }
}

impl Op for Mean {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Mean requires exactly 1 input");
        let input = &inputs[0];
        
        match self.axis {
            None => {
                let n = (input.nrows() * input.ncols()) as f64;
                Tensor::from_element(1, 1, input.sum() / n)
            }
            _ => {
                // For now, only support mean of all elements
                panic!("Mean along specific axis not yet implemented");
            }
        }
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        let input_shape = (inputs[0].nrows(), inputs[0].ncols());
        let n = (input_shape.0 * input_shape.1) as f64;
        
        match self.axis {
            None => {
                // Gradient is uniformly distributed
                vec![Tensor::from_element(
                    input_shape.0,
                    input_shape.1,
                    grad_output[(0, 0)] / n,
                )]
            }
            _ => {
                panic!("Mean backward along specific axis not yet implemented");
            }
        }
    }
    
    fn name(&self) -> &str {
        "Mean"
    }
}

/// Sigmoid activation function.
#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Op for Sigmoid {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Sigmoid requires exactly 1 input");
        inputs[0].map(|x| 1.0 / (1.0 + (-x).exp()))
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        output: &Tensor,
    ) -> Vec<Tensor> {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let grad = output.component_mul(&output.map(|s| 1.0 - s));
        vec![grad_output.component_mul(&grad)]
    }
    
    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Hyperbolic tangent activation function.
#[derive(Debug, Clone)]
pub struct Tanh;

impl Op for Tanh {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 1, "Tanh requires exactly 1 input");
        inputs[0].map(|x| x.tanh())
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        _inputs: &[Tensor],
        output: &Tensor,
    ) -> Vec<Tensor> {
        // d/dx tanh(x) = 1 - tanh(x)^2
        let grad = output.map(|t| 1.0 - t * t);
        vec![grad_output.component_mul(&grad)]
    }
    
    fn name(&self) -> &str {
        "Tanh"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_forward() {
        let op = Add;
        let a = Tensor::from_element(2, 2, 1.0);
        let b = Tensor::from_element(2, 2, 2.0);
        
        let result = op.forward(&[a, b]);
        assert_eq!(result[(0, 0)], 3.0);
    }

    #[test]
    fn test_add_backward() {
        let op = Add;
        let a = Tensor::from_element(2, 2, 1.0);
        let b = Tensor::from_element(2, 2, 2.0);
        let grad_output = Tensor::from_element(2, 2, 1.0);
        let output = Tensor::from_element(2, 2, 3.0);
        
        let grads = op.backward(&grad_output, &[a, b], &output);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0], grad_output);
        assert_eq!(grads[1], grad_output);
    }

    #[test]
    fn test_multiply_forward() {
        let op = Multiply;
        let a = Tensor::from_element(2, 2, 2.0);
        let b = Tensor::from_element(2, 2, 3.0);
        
        let result = op.forward(&[a, b]);
        assert_eq!(result[(0, 0)], 6.0);
    }

    #[test]
    fn test_matmul_forward() {
        let op = MatMul;
        let a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DMatrix::from_row_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        
        let result = op.forward(&[a, b]);
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        // First row: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert_eq!(result[(0, 0)], 58.0);
    }

    #[test]
    fn test_transpose_forward() {
        let op = Transpose;
        let a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let result = op.forward(&[a]);
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result[(0, 0)], 1.0);
        assert_eq!(result[(1, 0)], 2.0);
        assert_eq!(result[(0, 1)], 4.0);
    }

    #[test]
    fn test_sum_all() {
        let op = Sum::all();
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        
        let result = op.forward(&[a]);
        assert_eq!(result.nrows(), 1);
        assert_eq!(result.ncols(), 1);
        assert_eq!(result[(0, 0)], 10.0);
    }

    #[test]
    fn test_relu_forward() {
        let op = ReLU;
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, -3.0, 4.0]);
        
        let result = op.forward(&[a]);
        assert_eq!(result[(0, 0)], 0.0);
        assert_eq!(result[(0, 1)], 2.0);
        assert_eq!(result[(1, 0)], 0.0);
        assert_eq!(result[(1, 1)], 4.0);
    }

    #[test]
    fn test_exp_forward() {
        let op = Exp;
        let a = DMatrix::from_row_slice(1, 1, &[2.0]);
        
        let result = op.forward(&[a]);
        assert_relative_eq!(result[(0, 0)], 2.0_f64.exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_mean_forward() {
        let op = Mean::all();
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        
        let result = op.forward(&[a]);
        assert_eq!(result[(0, 0)], 2.5);
    }
    
    #[test]
    fn test_sigmoid_forward() {
        let op = Sigmoid;
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 2.0]);
        
        let result = op.forward(&[a]);
        assert_relative_eq!(result[(0, 0)], 0.5, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 1)], 1.0 / (1.0 + (-1.0_f64).exp()), epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 1.0 / (1.0 + 1.0_f64.exp()), epsilon = 1e-10);
    }
    
    #[test]
    fn test_tanh_forward() {
        let op = Tanh;
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.5]);
        
        let result = op.forward(&[a]);
        assert_relative_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 1)], 1.0_f64.tanh(), epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], (-1.0_f64).tanh(), epsilon = 1e-10);
        assert_relative_eq!(result[(1, 1)], 0.5_f64.tanh(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_sigmoid_backward() {
        let op = Sigmoid;
        let input = DMatrix::from_element(1, 1, 0.0);
        let output = op.forward(&[input.clone()]);
        let grad_output = DMatrix::from_element(1, 1, 1.0);
        
        let grads = op.backward(&grad_output, &[input], &output);
        
        // At x=0, sigmoid(0) = 0.5, so gradient = 0.5 * (1 - 0.5) = 0.25
        assert_relative_eq!(grads[0][(0, 0)], 0.25, epsilon = 1e-10);
    }
    
    #[test]
    fn test_tanh_backward() {
        let op = Tanh;
        let input = DMatrix::from_element(1, 1, 0.0);
        let output = op.forward(&[input.clone()]);
        let grad_output = DMatrix::from_element(1, 1, 1.0);
        
        let grads = op.backward(&grad_output, &[input], &output);
        
        // At x=0, tanh(0) = 0, so gradient = 1 - 0^2 = 1
        assert_relative_eq!(grads[0][(0, 0)], 1.0, epsilon = 1e-10);
    }
}