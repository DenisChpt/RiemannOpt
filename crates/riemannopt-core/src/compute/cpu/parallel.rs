//! Parallel computing support for Riemannian optimization.
//!
//! This module provides parallel implementations of manifold operations
//! and batch processing capabilities using Rayon.

use crate::types::Scalar;
use crate::simd::{SimdOps, SimdVectorOps, SimdMatrixOps};
use crate::compute::cpu::batch_ops::{BatchError, CacheFriendlyBatch};
use nalgebra::{DMatrix, DVector, DVectorView, DVectorViewMut};
use rayon::prelude::*;

/// Type alias for batch of points (each column is a point).
pub type PointBatch<T> = DMatrix<T>;

/// Type alias for batch of tangent vectors.
pub type TangentBatch<T> = DMatrix<T>;

/// Configuration for parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum dimension to trigger parallel execution
    pub min_dimension_for_parallel: usize,
    /// Number of threads to use (None = use rayon default)
    pub num_threads: Option<usize>,
    /// Chunk size for parallel iterations
    pub chunk_size: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_dimension_for_parallel: 100,
            num_threads: None,
            chunk_size: None,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the minimum dimension for parallel execution.
    pub fn with_min_dimension(mut self, min_dim: usize) -> Self {
        self.min_dimension_for_parallel = min_dim;
        self
    }
    
    /// Set the number of threads.
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }
    
    /// Set the chunk size for parallel iterations.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }
    
    /// Check if parallel execution should be used for given dimension.
    pub fn should_parallelize(&self, dimension: usize) -> bool {
        dimension >= self.min_dimension_for_parallel
    }
}

/// Parallel batch operations for optimization.
pub struct ParallelBatch;

impl ParallelBatch {
    /// Evaluates a function on multiple points in parallel.
    ///
    /// # Arguments
    /// * `points` - Matrix where each column is a point
    /// * `func` - Function to evaluate at each point
    ///
    /// # Returns
    /// Vector of function values
    pub fn evaluate<T, F>(
        points: &PointBatch<T>,
        func: F,
    ) -> Vec<T>
    where
        T: Scalar,
        F: Fn(DVectorView<T>) -> T + Sync,
    {
        (0..points.ncols())
            .into_par_iter()
            .map(|i| {
                let point = points.column(i);
                func(point)
            })
            .collect()
    }
    
    /// Computes gradients at multiple points in parallel with zero allocations.
    ///
    /// # Arguments
    /// * `points` - Matrix where each column is a point
    /// * `output` - Pre-allocated output matrix to store gradients
    /// * `grad_func` - Function that computes gradient in-place
    ///
    /// # Returns
    /// Result indicating success or dimension mismatch error
    pub fn gradient<T, F>(
        points: &PointBatch<T>,
        output: &mut TangentBatch<T>,
        grad_func: F,
    ) -> Result<(), BatchError>
    where
        T: Scalar,
        F: Fn(DVectorView<T>, DVectorViewMut<T>) + Sync + Send,
    {
        CacheFriendlyBatch::gradient(points, output, grad_func)
    }
    
    /// Applies an operation to each point in parallel with zero allocations.
    ///
    /// # Arguments
    /// * `points` - Matrix where each column is a point
    /// * `output` - Pre-allocated output matrix to store results
    /// * `op` - Operation to apply in-place
    ///
    /// # Returns
    /// Result indicating success or dimension mismatch error
    pub fn map<T, F>(
        points: &PointBatch<T>,
        output: &mut PointBatch<T>,
        op: F,
    ) -> Result<(), BatchError>
    where
        T: Scalar,
        F: Fn(DVectorView<T>, DVectorViewMut<T>) + Sync + Send,
    {
        CacheFriendlyBatch::map(points, output, op)
    }
    
    /// Applies an operation to pairs of points and tangent vectors with zero allocations.
    ///
    /// # Arguments
    /// * `points` - Matrix where each column is a point
    /// * `tangents` - Matrix where each column is a tangent vector
    /// * `output` - Pre-allocated output matrix to store results
    /// * `op` - Operation to apply in-place
    ///
    /// # Returns
    /// Result indicating success or dimension mismatch error
    pub fn map_pairs<T, F>(
        points: &PointBatch<T>,
        tangents: &TangentBatch<T>,
        output: &mut PointBatch<T>,
        op: F,
    ) -> Result<(), BatchError>
    where
        T: Scalar,
        F: Fn(DVectorView<T>, DVectorView<T>, DVectorViewMut<T>) + Sync + Send,
    {
        CacheFriendlyBatch::map_pairs(points, tangents, output, op)
    }
}

/// Parallel line search for batch optimization.
pub struct ParallelLineSearch<T: Scalar> {
    /// Maximum number of iterations
    pub max_iters: usize,
    /// Armijo constant
    pub c1: T,
    /// Initial step size
    pub alpha0: T,
    /// Backtracking factor
    pub rho: T,
}

impl<T: Scalar> Default for ParallelLineSearch<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> ParallelLineSearch<T> {
    /// Creates a new parallel line search with default parameters.
    pub fn new() -> Self {
        Self {
            max_iters: 20,
            c1: T::from_f32(1e-4).unwrap(),
            alpha0: T::one(),
            rho: T::from_f32(0.5).unwrap(),
        }
    }
    
    /// Performs parallel line search for a batch of points.
    ///
    /// # Arguments
    /// * `points` - Current points
    /// * `directions` - Search directions
    /// * `values` - Function values at current points
    /// * `dir_derivatives` - Directional derivatives
    /// * `retract_fn` - Function to retract along direction
    /// * `eval_fn` - Function to evaluate objective
    ///
    /// # Returns
    /// Step sizes for each point
    pub fn search_batch<R, F>(
        &self,
        points: &PointBatch<T>,
        directions: &TangentBatch<T>,
        values: &[T],
        dir_derivatives: &[T],
        retract_fn: R,
        eval_fn: F,
    ) -> Vec<T>
    where
        R: Fn(&DVector<T>, &DVector<T>, T) -> DVector<T> + Sync,
        F: Fn(&DVector<T>) -> T + Sync,
    {
        let n_points = points.ncols();
        
        // Perform line search for each point in parallel
        (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = points.column(i).clone_owned();
                let direction = directions.column(i).clone_owned();
                let value = values[i];
                let dir_deriv = dir_derivatives[i];
                
                // Armijo line search
                let mut alpha = self.alpha0;
                for _ in 0..self.max_iters {
                    let new_point = retract_fn(&point, &direction, alpha);
                    let new_value = eval_fn(&new_point);
                    
                    if new_value <= value + self.c1 * alpha * dir_deriv {
                        return alpha;
                    }
                    
                    alpha *= self.rho;
                }
                
                alpha
            })
            .collect()
    }
}

/// Parallel stochastic gradient descent utilities.
pub struct ParallelSGD;

impl ParallelSGD {
    /// Processes mini-batches in parallel.
    ///
    /// # Arguments
    /// * `data` - Full dataset
    /// * `batch_size` - Size of each mini-batch
    /// * `process_batch` - Function to process a batch
    pub fn process_batches<T, D, F, R>(
        data: &[D],
        batch_size: usize,
        process_batch: F,
    ) -> Vec<R>
    where
        T: Scalar,
        D: Send + Sync,
        F: Fn(&[D]) -> R + Send + Sync,
        R: Send,
    {
        data.par_chunks(batch_size)
            .map(process_batch)
            .collect()
    }
    
    /// Computes average gradient over mini-batches in parallel.
    pub fn average_gradients<T>(
        gradients: &[DVector<T>],
    ) -> DVector<T>
    where
        T: Scalar,
    {
        let n = gradients.len();
        if n == 0 {
            return DVector::zeros(0);
        }
        
        let dim = gradients[0].len();
        let sum: DVector<T> = gradients
            .par_iter()
            .map(|g| g.clone())
            .reduce(
                || DVector::zeros(dim),
                |a, b| a + b,
            );
        
        sum / <T as crate::types::Scalar>::from_usize(n)
    }
}

/// Parallel model averaging for distributed optimization.
pub struct ParallelAverage;

impl ParallelAverage {
    /// Averages parameters from multiple models.
    pub fn average_parameters<T>(
        parameters: &[DVector<T>],
    ) -> DVector<T>
    where
        T: Scalar,
    {
        ParallelSGD::average_gradients(parameters)
    }
    
    /// Weighted average of parameters.
    pub fn weighted_average<T>(
        parameters: &[DVector<T>],
        weights: &[T],
    ) -> DVector<T>
    where
        T: Scalar,
    {
        assert_eq!(parameters.len(), weights.len());
        
        let dim = parameters[0].len();
        let weighted_sum: DVector<T> = parameters
            .par_iter()
            .zip(weights.par_iter())
            .map(|(param, &weight)| param * weight)
            .reduce(
                || DVector::zeros(dim),
                |a, b| a + b,
            );
        
        let total_weight: T = weights.par_iter().copied().reduce(|| T::zero(), |a, b| a + b);
        weighted_sum / total_weight
    }
}

/// SIMD-enhanced parallel operations
pub struct SimdParallelOps;

impl SimdParallelOps {
    /// Parallel dot product with SIMD for large batches
    pub fn batch_dot_product<T>(
        a_batch: &PointBatch<T>,
        b_batch: &PointBatch<T>,
    ) -> Vec<T>
    where
        T: Scalar + SimdOps,
    {
        assert_eq!(a_batch.ncols(), b_batch.ncols());
        
        // Use parallel index iteration to avoid allocations
        (0..a_batch.ncols())
            .into_par_iter()
            .map(|i| {
                let a_col = a_batch.column(i);
                let b_col = b_batch.column(i);
                SimdVectorOps::dot_product(a_col, b_col)
            })
            .collect()
    }
    
    /// Parallel normalization with SIMD
    pub fn batch_normalize<T>(
        points: &mut PointBatch<T>,
    ) -> Vec<T>
    where
        T: Scalar + SimdOps,
    {
        let n_points = points.ncols();
        
        // Pass 1: Compute all norms in parallel (read-only)
        let norms: Vec<T> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let col = points.column(i);
                SimdVectorOps::norm(col)
            })
            .collect();
        
        // Pass 2: Normalize in parallel (write-only)
        // We need to collect indices to iterate in parallel
        let indices: Vec<_> = (0..n_points).collect();
        indices.par_iter().zip(norms.par_iter()).for_each(|(&i, &norm)| {
            if norm > T::zero() {
                // SAFETY: We're accessing different columns in parallel, which is safe
                // because columns are non-overlapping memory regions
                unsafe {
                    let points_ptr = points.as_ptr() as *mut T;
                    let col_start = i * points.nrows();
                    let col_ptr = points_ptr.add(col_start);
                    let col_slice = std::slice::from_raw_parts_mut(col_ptr, points.nrows());
                    
                    let inv_norm = T::one() / norm;
                    for elem in col_slice.iter_mut() {
                        *elem *= inv_norm;
                    }
                }
            }
        });
        
        norms
    }
    
    /// Parallel matrix-vector multiplication with SIMD
    pub fn batch_gemv<T>(
        matrices: &[DMatrix<T>],
        vectors: &[DVector<T>],
        alpha: T,
        beta: T,
    ) -> Vec<DVector<T>>
    where
        T: Scalar + SimdOps,
    {
        assert_eq!(matrices.len(), vectors.len());
        
        matrices
            .par_iter()
            .zip(vectors.par_iter())
            .map(|(matrix, vector)| {
                let mut result = DVector::zeros(matrix.nrows());
                SimdMatrixOps::gemv(matrix, vector, &mut result, alpha, beta);
                result
            })
            .collect()
    }
    
    /// Parallel Frobenius norm computation with SIMD
    pub fn batch_frobenius_norm<T>(
        matrices: &[DMatrix<T>],
    ) -> Vec<T>
    where
        T: Scalar + SimdOps,
    {
        matrices
            .par_iter()
            .map(|matrix| SimdMatrixOps::frobenius_norm(matrix))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_parallel_evaluate() {
        let points = PointBatch::from_columns(&[
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ]);
        
        // Simple quadratic function
        let func = |x: DVectorView<f64>| x.dot(&x);
        
        let values = ParallelBatch::evaluate(&points, func);
        
        assert_eq!(values.len(), 3);
        assert_relative_eq!(values[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(values[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(values[2], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_parallel_gradient() {
        let points = PointBatch::from_columns(&[
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ]);
        
        let mut gradients = TangentBatch::<f64>::zeros(2, 2);
        
        // Gradient of f(x) = x^T x is 2x
        let grad_func = |x: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
            let result = x * 2.0;
            out.copy_from(&result);
        };
        
        ParallelBatch::gradient(&points, &mut gradients, grad_func).unwrap();
        
        assert_eq!(gradients.ncols(), 2);
        assert_relative_eq!(gradients[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(gradients[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(gradients[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(gradients[(1, 1)], 2.0, epsilon = 1e-10);
    }
    
    
    #[test]
    fn test_parallel_map() {
        let points = PointBatch::from_columns(&[
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        ]);
        
        let mut results = PointBatch::<f64>::zeros(2, 2);
        
        // Simple scaling operation
        let op = |x: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
            let result = x * 2.0;
            out.copy_from(&result);
        };
        
        ParallelBatch::map(&points, &mut results, op).unwrap();
        
        assert_eq!(results.ncols(), 2);
        assert_relative_eq!(results[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(results[(1, 0)], 4.0, epsilon = 1e-10);
        assert_relative_eq!(results[(0, 1)], 6.0, epsilon = 1e-10);
        assert_relative_eq!(results[(1, 1)], 8.0, epsilon = 1e-10);
    }
    
    
    #[test]
    fn test_average_gradients() {
        let gradients = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
            DVector::from_vec(vec![5.0, 6.0]),
        ];
        
        let average = ParallelSGD::average_gradients(&gradients);
        
        assert_relative_eq!(average[0], 3.0, epsilon = 1e-10); // (1+3+5)/3
        assert_relative_eq!(average[1], 4.0, epsilon = 1e-10); // (2+4+6)/3
    }
    
    #[test]
    fn test_weighted_average() {
        let parameters = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        ];
        let weights = vec![0.3, 0.7];
        
        let average = ParallelAverage::weighted_average(&parameters, &weights);
        
        assert_relative_eq!(average[0], 2.4, epsilon = 1e-10); // 0.3*1 + 0.7*3
        assert_relative_eq!(average[1], 3.4, epsilon = 1e-10); // 0.3*2 + 0.7*4
    }
    
    #[test]
    fn test_simd_batch_dot_product() {
        let a = PointBatch::from_columns(&[
            DVector::from_vec(vec![1.0_f32, 2.0, 3.0]),
            DVector::from_vec(vec![4.0, 5.0, 6.0]),
        ]);
        let b = PointBatch::from_columns(&[
            DVector::from_vec(vec![2.0_f32, 3.0, 4.0]),
            DVector::from_vec(vec![1.0, 2.0, 3.0]),
        ]);
        
        let dots = SimdParallelOps::batch_dot_product(&a, &b);
        
        assert_eq!(dots.len(), 2);
        assert_relative_eq!(dots[0], 20.0, epsilon = 1e-6); // 1*2 + 2*3 + 3*4
        assert_relative_eq!(dots[1], 32.0, epsilon = 1e-6); // 4*1 + 5*2 + 6*3
    }
    
    #[test]
    fn test_simd_batch_normalize() {
        let mut points = PointBatch::from_columns(&[
            DVector::from_vec(vec![3.0_f64, 4.0]),
            DVector::from_vec(vec![5.0, 12.0]),
        ]);
        
        let norms = SimdParallelOps::batch_normalize(&mut points);
        
        assert_eq!(norms.len(), 2);
        assert_relative_eq!(norms[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(norms[1], 13.0, epsilon = 1e-10);
        
        // Check normalization
        assert_relative_eq!(points[(0, 0)], 0.6, epsilon = 1e-10);
        assert_relative_eq!(points[(1, 0)], 0.8, epsilon = 1e-10);
        assert_relative_eq!(points[(0, 1)], 5.0/13.0, epsilon = 1e-10);
        assert_relative_eq!(points[(1, 1)], 12.0/13.0, epsilon = 1e-10);
    }
}