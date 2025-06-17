//! Parallel computing support for Riemannian optimization.
//!
//! This module provides parallel implementations of manifold operations
//! and batch processing capabilities using Rayon.

use crate::types::Scalar;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Type alias for batch of points (each column is a point).
pub type PointBatch<T> = DMatrix<T>;

/// Type alias for batch of tangent vectors.
pub type TangentBatch<T> = DMatrix<T>;

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
        F: Fn(&DVector<T>) -> T + Sync,
    {
        (0..points.ncols())
            .into_par_iter()
            .map(|i| {
                let point = points.column(i);
                func(&point.clone_owned())
            })
            .collect()
    }
    
    /// Computes gradients at multiple points in parallel.
    ///
    /// # Arguments
    /// * `points` - Matrix where each column is a point
    /// * `grad_func` - Function that computes gradient at a point
    ///
    /// # Returns
    /// Matrix where each column is a gradient
    pub fn gradient<T, F>(
        points: &PointBatch<T>,
        grad_func: F,
    ) -> TangentBatch<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> DVector<T> + Sync,
    {
        let n_points = points.ncols();
        let dim = points.nrows();
        
        let gradients: Vec<_> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = points.column(i);
                grad_func(&point.clone_owned())
            })
            .collect();
        
        // Combine into batch
        let mut result = TangentBatch::<T>::zeros(dim, n_points);
        for (i, grad) in gradients.into_iter().enumerate() {
            result.set_column(i, &grad);
        }
        
        result
    }
    
    /// Applies an operation to each point in parallel.
    ///
    /// # Arguments
    /// * `points` - Matrix where each column is a point
    /// * `op` - Operation to apply to each point
    ///
    /// # Returns
    /// Matrix where each column is the result
    pub fn map<T, F>(
        points: &PointBatch<T>,
        op: F,
    ) -> PointBatch<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> DVector<T> + Sync,
    {
        let n_points = points.ncols();
        let dim = points.nrows();
        
        let results: Vec<_> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = points.column(i);
                op(&point.clone_owned())
            })
            .collect();
        
        // Combine results
        let mut result = PointBatch::<T>::zeros(dim, n_points);
        for (i, vec) in results.into_iter().enumerate() {
            result.set_column(i, &vec);
        }
        
        result
    }
    
    /// Applies an operation to pairs of points and tangent vectors.
    ///
    /// # Arguments
    /// * `points` - Matrix where each column is a point
    /// * `tangents` - Matrix where each column is a tangent vector
    /// * `op` - Operation to apply to each pair
    ///
    /// # Returns
    /// Matrix where each column is the result
    pub fn map_pairs<T, F>(
        points: &PointBatch<T>,
        tangents: &TangentBatch<T>,
        op: F,
    ) -> PointBatch<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>, &DVector<T>) -> DVector<T> + Sync,
    {
        assert_eq!(
            points.ncols(),
            tangents.ncols(),
            "Points and tangents must have same number of columns"
        );
        
        let n_points = points.ncols();
        let dim = points.nrows();
        
        let results: Vec<_> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = points.column(i);
                let tangent = tangents.column(i);
                op(&point.clone_owned(), &tangent.clone_owned())
            })
            .collect();
        
        // Combine results
        let mut result = PointBatch::<T>::zeros(dim, n_points);
        for (i, vec) in results.into_iter().enumerate() {
            result.set_column(i, &vec);
        }
        
        result
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
                    
                    alpha = alpha * self.rho;
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
        let func = |x: &DVector<f64>| x.dot(x);
        
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
        
        // Gradient of f(x) = x^T x is 2x
        let grad_func = |x: &DVector<f64>| x * 2.0;
        
        let gradients = ParallelBatch::gradient(&points, grad_func);
        
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
        
        // Simple scaling operation
        let op = |x: &DVector<f64>| x * 2.0;
        
        let results = ParallelBatch::map(&points, op);
        
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
}