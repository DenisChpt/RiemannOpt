//! Broadcasting support for tensor operations.
//!
//! This module provides utilities for broadcasting tensors to compatible shapes,
//! following NumPy-style broadcasting rules.

use crate::graph::Tensor;
use nalgebra::DMatrix;

/// Result of a broadcast operation.
#[derive(Debug, Clone)]
pub struct BroadcastResult {
    /// The broadcasted tensors
    pub tensors: Vec<Tensor>,
    /// The output shape after broadcasting
    pub output_shape: (usize, usize),
    /// Information needed to reverse the broadcast in backward pass
    pub broadcast_info: Vec<BroadcastInfo>,
}

/// Information about how a tensor was broadcasted.
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    /// Original shape of the tensor
    pub original_shape: (usize, usize),
    /// Axes that were added (None for scalar expansion)
    pub added_axes: Vec<usize>,
    /// Axes that were repeated
    pub repeated_axes: Vec<(usize, usize)>, // (axis, repeat_count)
}

/// Broadcasts two tensors to a common shape.
///
/// # Broadcasting Rules:
/// 1. If shapes differ in rank, prepend 1s to the smaller shape
/// 2. Two dimensions are compatible if they are equal or one is 1
/// 3. The output shape is the max of each dimension pair
pub fn broadcast_binary(a: &Tensor, b: &Tensor) -> Result<BroadcastResult, String> {
    let shape_a = (a.nrows(), a.ncols());
    let shape_b = (b.nrows(), b.ncols());
    
    // Check compatibility and compute output shape
    let output_shape = broadcast_shape(shape_a, shape_b)?;
    
    // Broadcast each tensor
    let (broadcast_a, info_a) = broadcast_to(a, output_shape)?;
    let (broadcast_b, info_b) = broadcast_to(b, output_shape)?;
    
    Ok(BroadcastResult {
        tensors: vec![broadcast_a, broadcast_b],
        output_shape,
        broadcast_info: vec![info_a, info_b],
    })
}

/// Computes the output shape for broadcasting two shapes.
fn broadcast_shape(shape_a: (usize, usize), shape_b: (usize, usize)) -> Result<(usize, usize), String> {
    // For 2D matrices, broadcasting rules are simpler
    let (rows_a, cols_a) = shape_a;
    let (rows_b, cols_b) = shape_b;
    
    // Check row compatibility
    let out_rows = if rows_a == rows_b {
        rows_a
    } else if rows_a == 1 {
        rows_b
    } else if rows_b == 1 {
        rows_a
    } else {
        return Err(format!(
            "Incompatible shapes for broadcasting: ({}, {}) and ({}, {})",
            rows_a, cols_a, rows_b, cols_b
        ));
    };
    
    // Check column compatibility
    let out_cols = if cols_a == cols_b {
        cols_a
    } else if cols_a == 1 {
        cols_b
    } else if cols_b == 1 {
        cols_a
    } else {
        return Err(format!(
            "Incompatible shapes for broadcasting: ({}, {}) and ({}, {})",
            rows_a, cols_a, rows_b, cols_b
        ));
    };
    
    Ok((out_rows, out_cols))
}

/// Broadcasts a tensor to a target shape.
fn broadcast_to(tensor: &Tensor, target_shape: (usize, usize)) -> Result<(Tensor, BroadcastInfo), String> {
    let original_shape = (tensor.nrows(), tensor.ncols());
    let (orig_rows, orig_cols) = original_shape;
    let (target_rows, target_cols) = target_shape;
    
    // Check if already correct shape
    if original_shape == target_shape {
        return Ok((
            tensor.clone(),
            BroadcastInfo {
                original_shape,
                added_axes: vec![],
                repeated_axes: vec![],
            },
        ));
    }
    
    let mut result = tensor.clone();
    let mut repeated_axes = Vec::new();
    
    // Broadcast rows
    if orig_rows == 1 && target_rows > 1 {
        // Repeat rows
        let mut expanded = DMatrix::zeros(target_rows, orig_cols);
        for i in 0..target_rows {
            for j in 0..orig_cols {
                expanded[(i, j)] = result[(0, j)];
            }
        }
        result = expanded;
        repeated_axes.push((0, target_rows));
    } else if orig_rows != target_rows {
        return Err(format!(
            "Cannot broadcast shape ({}, {}) to ({}, {})",
            orig_rows, orig_cols, target_rows, target_cols
        ));
    }
    
    // Broadcast columns
    if orig_cols == 1 && target_cols > 1 {
        // Repeat columns
        let current_rows = result.nrows();
        let mut expanded = DMatrix::zeros(current_rows, target_cols);
        for i in 0..current_rows {
            for j in 0..target_cols {
                expanded[(i, j)] = result[(i, 0)];
            }
        }
        result = expanded;
        repeated_axes.push((1, target_cols));
    } else if orig_cols != target_cols {
        return Err(format!(
            "Cannot broadcast shape ({}, {}) to ({}, {})",
            orig_rows, orig_cols, target_rows, target_cols
        ));
    }
    
    Ok((
        result,
        BroadcastInfo {
            original_shape,
            added_axes: vec![],
            repeated_axes,
        },
    ))
}

/// Reverses a broadcast operation for the backward pass.
///
/// This sums over the axes that were broadcasted to get gradients
/// in the original shape.
pub fn unbroadcast(grad: &Tensor, info: &BroadcastInfo) -> Tensor {
    let mut result = grad.clone();
    
    // Reverse broadcasts by summing over repeated axes
    for &(axis, _repeat_count) in info.repeated_axes.iter().rev() {
        result = match axis {
            0 => {
                // Sum over rows to get back to 1 row
                let mut summed = DMatrix::zeros(1, result.ncols());
                for j in 0..result.ncols() {
                    let sum: f64 = (0..result.nrows()).map(|i| result[(i, j)]).sum();
                    summed[(0, j)] = sum;
                }
                summed
            }
            1 => {
                // Sum over columns to get back to 1 column
                let mut summed = DMatrix::zeros(result.nrows(), 1);
                for i in 0..result.nrows() {
                    let sum: f64 = (0..result.ncols()).map(|j| result[(i, j)]).sum();
                    summed[(i, 0)] = sum;
                }
                summed
            }
            _ => unreachable!("Invalid axis for 2D broadcast"),
        };
    }
    
    result
}

/// Broadcast-aware element-wise addition.
#[derive(Debug, Clone)]
pub struct BroadcastAdd;

impl crate::ops::Op for BroadcastAdd {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "BroadcastAdd requires exactly 2 inputs");
        
        match broadcast_binary(&inputs[0], &inputs[1]) {
            Ok(broadcast_result) => {
                &broadcast_result.tensors[0] + &broadcast_result.tensors[1]
            }
            Err(_) => {
                // Fallback to regular addition if shapes are already compatible
                &inputs[0] + &inputs[1]
            }
        }
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        // For addition, gradient is 1 for both inputs
        // But we need to unbroadcast to original shapes
        match broadcast_binary(&inputs[0], &inputs[1]) {
            Ok(broadcast_result) => {
                vec![
                    unbroadcast(grad_output, &broadcast_result.broadcast_info[0]),
                    unbroadcast(grad_output, &broadcast_result.broadcast_info[1]),
                ]
            }
            Err(_) => {
                // No broadcasting needed
                vec![grad_output.clone(), grad_output.clone()]
            }
        }
    }
    
    fn name(&self) -> &str {
        "BroadcastAdd"
    }
}

/// Broadcast-aware element-wise multiplication.
#[derive(Debug, Clone)]
pub struct BroadcastMultiply;

impl crate::ops::Op for BroadcastMultiply {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "BroadcastMultiply requires exactly 2 inputs");
        
        match broadcast_binary(&inputs[0], &inputs[1]) {
            Ok(broadcast_result) => {
                broadcast_result.tensors[0].component_mul(&broadcast_result.tensors[1])
            }
            Err(_) => {
                // Fallback to regular multiplication if shapes are already compatible
                inputs[0].component_mul(&inputs[1])
            }
        }
    }
    
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[Tensor],
        _output: &Tensor,
    ) -> Vec<Tensor> {
        match broadcast_binary(&inputs[0], &inputs[1]) {
            Ok(broadcast_result) => {
                let grad_a = grad_output.component_mul(&broadcast_result.tensors[1]);
                let grad_b = grad_output.component_mul(&broadcast_result.tensors[0]);
                
                vec![
                    unbroadcast(&grad_a, &broadcast_result.broadcast_info[0]),
                    unbroadcast(&grad_b, &broadcast_result.broadcast_info[1]),
                ]
            }
            Err(_) => {
                // No broadcasting needed
                vec![
                    grad_output.component_mul(&inputs[1]),
                    grad_output.component_mul(&inputs[0]),
                ]
            }
        }
    }
    
    fn name(&self) -> &str {
        "BroadcastMultiply"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;
    
    #[test]
    fn test_broadcast_shape() {
        // Compatible shapes
        assert_eq!(broadcast_shape((2, 3), (2, 3)).unwrap(), (2, 3));
        assert_eq!(broadcast_shape((1, 3), (2, 3)).unwrap(), (2, 3));
        assert_eq!(broadcast_shape((2, 1), (2, 3)).unwrap(), (2, 3));
        assert_eq!(broadcast_shape((1, 1), (2, 3)).unwrap(), (2, 3));
        
        // Incompatible shapes
        assert!(broadcast_shape((2, 3), (3, 3)).is_err());
        assert!(broadcast_shape((2, 3), (2, 4)).is_err());
    }
    
    #[test]
    fn test_broadcast_to() {
        // Broadcast scalar
        let scalar = DMatrix::from_element(1, 1, 5.0);
        let (broadcasted, info) = broadcast_to(&scalar, (3, 4)).unwrap();
        assert_eq!(broadcasted.nrows(), 3);
        assert_eq!(broadcasted.ncols(), 4);
        assert_eq!(broadcasted[(0, 0)], 5.0);
        assert_eq!(broadcasted[(2, 3)], 5.0);
        assert_eq!(info.repeated_axes.len(), 2);
        
        // Broadcast row vector
        let row = DMatrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        let (broadcasted, info) = broadcast_to(&row, (4, 3)).unwrap();
        assert_eq!(broadcasted.nrows(), 4);
        assert_eq!(broadcasted.ncols(), 3);
        assert_eq!(broadcasted[(0, 0)], 1.0);
        assert_eq!(broadcasted[(3, 2)], 3.0);
        assert_eq!(info.repeated_axes.len(), 1);
    }
    
    #[test]
    fn test_unbroadcast() {
        // Test unbroadcasting a scalar broadcast
        let grad = DMatrix::from_element(3, 4, 2.0);
        let info = BroadcastInfo {
            original_shape: (1, 1),
            added_axes: vec![],
            repeated_axes: vec![(0, 3), (1, 4)],
        };
        
        let unbroadcasted = unbroadcast(&grad, &info);
        assert_eq!(unbroadcasted.nrows(), 1);
        assert_eq!(unbroadcasted.ncols(), 1);
        assert_eq!(unbroadcasted[(0, 0)], 24.0); // 2.0 * 3 * 4
    }
    
    #[test]
    fn test_broadcast_add() {
        let op = BroadcastAdd;
        
        // Test scalar + matrix
        let scalar = DMatrix::from_element(1, 1, 5.0);
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        
        let result = op.forward(&[scalar, matrix]);
        assert_eq!(result[(0, 0)], 6.0);
        assert_eq!(result[(0, 1)], 7.0);
        assert_eq!(result[(1, 0)], 8.0);
        assert_eq!(result[(1, 1)], 9.0);
    }
}