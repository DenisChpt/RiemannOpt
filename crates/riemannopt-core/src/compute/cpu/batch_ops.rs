//! High-performance batch operations with cache-friendly memory layout.
//!
//! This module provides optimized batch processing for manifold operations
//! using Array of Structures (AoS) layout to maximize cache locality and
//! achieve zero-allocation in hot paths through in-place operations.

use crate::types::Scalar;
use nalgebra::{DMatrix, DVector, DVectorView, DVectorViewMut};
use rayon::prelude::*;
use std::fmt;

/// Cache line size in bytes (typical for modern CPUs)
const CACHE_LINE_SIZE: usize = 64;

/// Batch data structure with cache-friendly Array of Structures (AoS) layout.
///
/// Instead of storing data as a matrix where columns are far apart in memory,
/// this structure stores each point contiguously: [x1, y1, z1, x2, y2, z2, ...]
#[derive(Debug, Clone)]
pub struct BatchData<T: Scalar> {
    /// Contiguous storage for all points
    data: Vec<T>,
    /// Dimension of each point
    dim: usize,
    /// Number of points in the batch
    n_points: usize,
}

impl<T: Scalar> BatchData<T> {
    /// Creates a new BatchData with specified capacity.
    pub fn new(dim: usize, n_points: usize) -> Self {
        let capacity = dim * n_points;
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, T::zero());
        
        Self {
            data,
            dim,
            n_points,
        }
    }
    
    /// Creates BatchData from a DMatrix (column-major to AoS conversion).
    pub fn from_matrix(matrix: &DMatrix<T>) -> Self {
        let dim = matrix.nrows();
        let n_points = matrix.ncols();
        let mut batch = Self::new(dim, n_points);
        
        // Convert from column-major to AoS layout
        for (i, col) in matrix.column_iter().enumerate() {
            let start = i * dim;
            for (j, &val) in col.iter().enumerate() {
                batch.data[start + j] = val;
            }
        }
        
        batch
    }
    
    /// Creates a mutable BatchData view from a mutable DMatrix reference.
    pub fn from_matrix_mut(matrix: &mut DMatrix<T>) -> BatchDataMut<T> {
        let dim = matrix.nrows();
        let n_points = matrix.ncols();
        
        BatchDataMut {
            matrix,
            dim,
            n_points,
        }
    }
    
    /// Converts BatchData back to DMatrix (AoS to column-major conversion).
    pub fn to_matrix(&self) -> DMatrix<T> {
        let mut matrix = DMatrix::zeros(self.dim, self.n_points);
        
        for i in 0..self.n_points {
            let start = i * self.dim;
            let point_slice = &self.data[start..start + self.dim];
            
            for (j, &val) in point_slice.iter().enumerate() {
                matrix[(j, i)] = val;
            }
        }
        
        matrix
    }
    
    /// Gets an immutable view of a point.
    #[inline]
    pub fn point_view(&self, index: usize) -> DVectorView<T> {
        debug_assert!(index < self.n_points);
        let start = index * self.dim;
        DVectorView::from_slice(&self.data[start..start + self.dim], self.dim)
    }
    
    /// Gets a mutable view of a point.
    #[inline]
    pub fn point_view_mut(&mut self, index: usize) -> DVectorViewMut<T> {
        debug_assert!(index < self.n_points);
        let start = index * self.dim;
        DVectorViewMut::from_slice(&mut self.data[start..start + self.dim], self.dim)
    }
    
    /// Gets an immutable slice for a point (zero-copy access).
    #[inline]
    pub fn point_slice(&self, index: usize) -> &[T] {
        debug_assert!(index < self.n_points);
        let start = index * self.dim;
        &self.data[start..start + self.dim]
    }
    
    /// Gets a mutable slice for a point (zero-copy access).
    #[inline]
    pub fn point_slice_mut(&mut self, index: usize) -> &mut [T] {
        debug_assert!(index < self.n_points);
        let start = index * self.dim;
        &mut self.data[start..start + self.dim]
    }
    
    /// Returns the dimension of points.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// Returns the number of points.
    #[inline]
    pub fn n_points(&self) -> usize {
        self.n_points
    }
    
    /// Calculates optimal chunk size for parallel processing.
    #[allow(dead_code)]
    fn optimal_chunk_size(&self) -> usize {
        let num_threads = rayon::current_num_threads();
        let points_per_thread = (self.n_points + num_threads - 1) / num_threads;
        
        // Ensure chunk size is cache-aligned
        let bytes_per_point = self.dim * std::mem::size_of::<T>();
        let cache_lines_per_point = (bytes_per_point + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
        let points_per_cache_line = if cache_lines_per_point > 0 {
            1
        } else {
            CACHE_LINE_SIZE / bytes_per_point
        };
        
        points_per_thread.max(points_per_cache_line).max(1)
    }
}

/// Mutable view wrapper for DMatrix to enable in-place operations.
pub struct BatchDataMut<'a, T: Scalar> {
    matrix: &'a mut DMatrix<T>,
    #[allow(dead_code)]
    dim: usize,
    n_points: usize,
}

impl<'a, T: Scalar> BatchDataMut<'a, T> {
    /// Gets a mutable view of a column in the matrix.
    #[inline]
    pub fn column_mut(&mut self, index: usize) -> DVectorViewMut<T> {
        debug_assert!(index < self.n_points);
        self.matrix.column_mut(index)
    }
}

/// Error type for batch operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchError {
    /// Input and output dimensions don't match
    DimensionMismatch { 
        input_rows: usize, 
        input_cols: usize,
        output_rows: usize, 
        output_cols: usize 
    },
    /// Points and tangents have different dimensions
    PointTangentMismatch {
        points_cols: usize,
        tangents_cols: usize,
    },
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchError::DimensionMismatch { input_rows, input_cols, output_rows, output_cols } => {
                write!(f, "Dimension mismatch: input is {}x{}, output is {}x{}", 
                       input_rows, input_cols, output_rows, output_cols)
            }
            BatchError::PointTangentMismatch { points_cols, tangents_cols } => {
                write!(f, "Points and tangents must have same number of columns: {} vs {}", 
                       points_cols, tangents_cols)
            }
        }
    }
}

impl std::error::Error for BatchError {}

/// High-performance parallel batch operations with zero allocations.
pub struct CacheFriendlyBatch;

impl CacheFriendlyBatch {
    /// Validates that input and output matrices have compatible dimensions.
    #[inline]
    fn validate_dimensions<T: Scalar>(
        input: &DMatrix<T>, 
        output: &DMatrix<T>
    ) -> Result<(), BatchError> {
        if input.nrows() != output.nrows() || input.ncols() != output.ncols() {
            return Err(BatchError::DimensionMismatch {
                input_rows: input.nrows(),
                input_cols: input.ncols(),
                output_rows: output.nrows(),
                output_cols: output.ncols(),
            });
        }
        Ok(())
    }
    
    /// Computes gradients with zero allocations using in-place operations.
    ///
    /// # Arguments
    /// * `points` - Input matrix where each column is a point
    /// * `output` - Pre-allocated output matrix to store gradients
    /// * `grad_func` - Function that computes gradient in-place
    ///
    /// # Returns
    /// Result indicating success or dimension mismatch error
    pub fn gradient<T, F>(
        points: &DMatrix<T>,
        output: &mut DMatrix<T>,
        grad_func: F,
    ) -> Result<(), BatchError>
    where
        T: Scalar,
        F: Fn(DVectorView<T>, DVectorViewMut<T>) + Sync + Send,
    {
        // Validate dimensions
        Self::validate_dimensions(points, output)?;
        
        // For optimal performance with zero allocations
        let n_points = points.ncols();
        let dim = points.nrows();
        
        // Use rayon's parallel iterator with column indices
        let indices: Vec<_> = (0..n_points).collect();
        indices.par_iter().for_each(|&i| {
            // Get input view
            let point_view = points.column(i);
            
            // Get mutable output column
            // SAFETY: We access different columns in parallel, which is safe
            // because columns are non-overlapping memory regions
            unsafe {
                let output_ptr = output.as_ptr() as *mut T;
                let col_start = i * dim;
                let col_ptr = output_ptr.add(col_start);
                let col_slice = std::slice::from_raw_parts_mut(col_ptr, dim);
                let col_view = DVectorViewMut::from_slice(col_slice, dim);
                
                // Apply the gradient function
                grad_func(point_view, col_view);
            }
        });
        
        Ok(())
    }
    
    /// Maps an operation over points with zero allocations using in-place operations.
    ///
    /// # Arguments
    /// * `points` - Input matrix where each column is a point
    /// * `output` - Pre-allocated output matrix to store results
    /// * `op` - Operation to apply in-place
    ///
    /// # Returns
    /// Result indicating success or dimension mismatch error
    pub fn map<T, F>(
        points: &DMatrix<T>,
        output: &mut DMatrix<T>,
        op: F,
    ) -> Result<(), BatchError>
    where
        T: Scalar,
        F: Fn(DVectorView<T>, DVectorViewMut<T>) + Sync + Send,
    {
        // Validate dimensions
        Self::validate_dimensions(points, output)?;
        
        // Zero-allocation parallel processing
        let n_points = points.ncols();
        let dim = points.nrows();
        
        // Use rayon's parallel iterator with column indices
        let indices: Vec<_> = (0..n_points).collect();
        indices.par_iter().for_each(|&i| {
            let point_view = points.column(i);
            
            unsafe {
                let output_ptr = output.as_ptr() as *mut T;
                let col_start = i * dim;
                let col_ptr = output_ptr.add(col_start);
                let col_slice = std::slice::from_raw_parts_mut(col_ptr, dim);
                let col_view = DVectorViewMut::from_slice(col_slice, dim);
                
                op(point_view, col_view);
            }
        });
        
        Ok(())
    }
    
    /// Maps an operation over pairs of points and tangent vectors with zero allocations.
    ///
    /// # Arguments
    /// * `points` - Input matrix where each column is a point
    /// * `tangents` - Input matrix where each column is a tangent vector
    /// * `output` - Pre-allocated output matrix to store results
    /// * `op` - Operation to apply in-place
    ///
    /// # Returns
    /// Result indicating success or dimension mismatch error
    pub fn map_pairs<T, F>(
        points: &DMatrix<T>,
        tangents: &DMatrix<T>,
        output: &mut DMatrix<T>,
        op: F,
    ) -> Result<(), BatchError>
    where
        T: Scalar,
        F: Fn(DVectorView<T>, DVectorView<T>, DVectorViewMut<T>) + Sync + Send,
    {
        // Validate dimensions
        if points.ncols() != tangents.ncols() {
            return Err(BatchError::PointTangentMismatch {
                points_cols: points.ncols(),
                tangents_cols: tangents.ncols(),
            });
        }
        Self::validate_dimensions(points, output)?;
        
        // Zero-allocation parallel processing
        let n_points = points.ncols();
        let dim = points.nrows();
        
        // Use rayon's parallel iterator with column indices
        let indices: Vec<_> = (0..n_points).collect();
        indices.par_iter().for_each(|&i| {
            let point_view = points.column(i);
            let tangent_view = tangents.column(i);
            
            unsafe {
                let output_ptr = output.as_ptr() as *mut T;
                let col_start = i * dim;
                let col_ptr = output_ptr.add(col_start);
                let col_slice = std::slice::from_raw_parts_mut(col_ptr, dim);
                let col_view = DVectorViewMut::from_slice(col_slice, dim);
                
                op(point_view, tangent_view, col_view);
            }
        });
        
        Ok(())
    }
    
    /// Legacy API: Evaluates a function on multiple points (allocates for compatibility).
    pub fn evaluate<T, F>(
        points: &DMatrix<T>,
        func: F,
    ) -> Vec<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> T + Sync,
    {
        (0..points.ncols())
            .into_par_iter()
            .map(|i| {
                let point_view = points.column(i);
                func(&point_view.into_owned())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_batch_data_conversion() {
        let matrix = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        
        let batch = BatchData::from_matrix(&matrix);
        assert_eq!(batch.dim(), 3);
        assert_eq!(batch.n_points(), 2);
        
        // Check AoS layout
        assert_eq!(batch.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        
        // Convert back
        let matrix2 = batch.to_matrix();
        assert_eq!(matrix, matrix2);
    }
    
    #[test]
    fn test_point_views() {
        let mut batch = BatchData::new(3, 2);
        batch.data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        // Test immutable view
        let view0 = batch.point_view(0);
        assert_eq!(view0[0], 1.0);
        assert_eq!(view0[1], 2.0);
        assert_eq!(view0[2], 3.0);
        
        // Test mutable view
        {
            let mut view1 = batch.point_view_mut(1);
            view1[0] = 7.0;
            view1[1] = 8.0;
            view1[2] = 9.0;
        }
        
        assert_eq!(batch.data[3], 7.0);
        assert_eq!(batch.data[4], 8.0);
        assert_eq!(batch.data[5], 9.0);
    }
    
    #[test]
    fn test_cache_friendly_gradient() {
        let points = DMatrix::from_columns(&[
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ]);
        
        let mut output = DMatrix::zeros(2, 2);
        
        // Gradient of f(x) = x^T x is 2x
        let grad_func = |x: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
            let result = x * 2.0;
            out.copy_from(&result);
        };
        
        CacheFriendlyBatch::gradient(&points, &mut output, grad_func).unwrap();
        
        assert_eq!(output.ncols(), 2);
        assert_relative_eq!(output[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(output[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(output[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(output[(1, 1)], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_cache_friendly_map() {
        let points = DMatrix::from_columns(&[
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        ]);
        
        let mut output = DMatrix::zeros(2, 2);
        
        // Simple scaling operation
        let op = |x: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
            let result = x * 2.0;
            out.copy_from(&result);
        };
        
        CacheFriendlyBatch::map(&points, &mut output, op).unwrap();
        
        assert_eq!(output.ncols(), 2);
        assert_relative_eq!(output[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(output[(1, 0)], 4.0, epsilon = 1e-10);
        assert_relative_eq!(output[(0, 1)], 6.0, epsilon = 1e-10);
        assert_relative_eq!(output[(1, 1)], 8.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_cache_friendly_map_pairs() {
        let points = DMatrix::from_columns(&[
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        ]);
        let tangents = DMatrix::from_columns(&[
            DVector::from_vec(vec![0.1, 0.2]),
            DVector::from_vec(vec![0.3, 0.4]),
        ]);
        
        let mut output = DMatrix::zeros(2, 2);
        
        // Add point and tangent
        let op = |p: DVectorView<f64>, t: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
            let result = p + t;
            out.copy_from(&result);
        };
        
        CacheFriendlyBatch::map_pairs(&points, &tangents, &mut output, op).unwrap();
        
        assert_eq!(output.ncols(), 2);
        assert_relative_eq!(output[(0, 0)], 1.1, epsilon = 1e-10);
        assert_relative_eq!(output[(1, 0)], 2.2, epsilon = 1e-10);
        assert_relative_eq!(output[(0, 1)], 3.3, epsilon = 1e-10);
        assert_relative_eq!(output[(1, 1)], 4.4, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dimension_validation() {
        let points = DMatrix::from_element(3, 2, 1.0);
        let mut output_wrong = DMatrix::zeros(2, 2); // Wrong dimensions
        let mut output_correct = DMatrix::zeros(3, 2);
        
        let op = |_: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
            out.fill(0.0);
        };
        
        // Should fail with wrong dimensions
        let result = CacheFriendlyBatch::map(&points, &mut output_wrong, op);
        assert!(result.is_err());
        
        // Should succeed with correct dimensions
        let result = CacheFriendlyBatch::map(&points, &mut output_correct, op);
        assert!(result.is_ok());
    }
}