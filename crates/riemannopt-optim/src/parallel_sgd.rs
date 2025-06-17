//! Parallel Stochastic Gradient Descent utilities.
//!
//! This module provides utilities for parallel batch processing in SGD,
//! designed to work with the existing SGD optimizer.

use riemannopt_core::{
    types::Scalar,
    parallel::{ParallelBatch, PointBatch},
};
use nalgebra::DVector;

/// Utilities for parallel SGD operations.
pub struct ParallelSGDUtils;

impl ParallelSGDUtils {
    /// Computes the average gradient from a batch of points.
    ///
    /// This is useful for mini-batch SGD where we want to compute
    /// gradients at multiple points and average them.
    pub fn batch_gradient_average<T, F>(
        points: &PointBatch<T>,
        grad_fn: F,
    ) -> DVector<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> DVector<T> + Sync,
    {
        let gradients = ParallelBatch::gradient(points, grad_fn);
        
        // Average the gradients
        let n_points = gradients.ncols();
        let dim = gradients.nrows();
        
        if n_points == 0 {
            return DVector::zeros(dim);
        }
        
        let mut avg_grad = DVector::zeros(dim);
        for i in 0..n_points {
            avg_grad += gradients.column(i);
        }
        
        avg_grad / <T as Scalar>::from_usize(n_points)
    }
    
    /// Computes a weighted average of gradients.
    pub fn weighted_gradient_average<T, F>(
        points: &PointBatch<T>,
        weights: &[T],
        grad_fn: F,
    ) -> DVector<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> DVector<T> + Sync,
    {
        assert_eq!(points.ncols(), weights.len(), "Number of points must match number of weights");
        
        let gradients = ParallelBatch::gradient(points, grad_fn);
        
        let dim = gradients.nrows();
        let mut weighted_grad = DVector::zeros(dim);
        
        for (i, weight) in weights.iter().enumerate().take(gradients.ncols()) {
            weighted_grad += gradients.column(i) * *weight;
        }
        
        let total_weight: T = weights.iter().copied().fold(T::zero(), |a, b| a + b);
        weighted_grad / total_weight
    }
    
    /// Evaluates function values at multiple points in parallel.
    pub fn batch_evaluate<T, F>(
        points: &PointBatch<T>,
        func: F,
    ) -> Vec<T>
    where
        T: Scalar,
        F: Fn(&DVector<T>) -> T + Sync,
    {
        ParallelBatch::evaluate(points, func)
    }
    
    /// Performs stochastic sampling from a dataset.
    pub fn sample_batch<T>(
        data: &[DVector<T>],
        batch_size: usize,
    ) -> PointBatch<T>
    where
        T: Scalar,
    {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let mut rng = thread_rng();
        let sample: Vec<_> = data.choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();
        
        if sample.is_empty() {
            return PointBatch::zeros(0, 0);
        }
        
        let dim = sample[0].len();
        let mut batch = PointBatch::zeros(dim, sample.len());
        
        for (i, point) in sample.into_iter().enumerate() {
            batch.set_column(i, &point);
        }
        
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_batch_gradient_average() {
        // Create batch of points
        let points = PointBatch::from_columns(&[
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ]);
        
        // Gradient of f(x) = ||x||^2 is 2x
        let grad_fn = |x: &DVector<f64>| x * 2.0;
        
        let avg_grad = ParallelSGDUtils::batch_gradient_average(&points, grad_fn);
        
        // Average gradient should be 2 * average of points
        assert_relative_eq!(avg_grad[0], 4.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(avg_grad[1], 4.0 / 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_weighted_gradient_average() {
        let points = PointBatch::from_columns(&[
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ]);
        
        let weights = vec![0.7, 0.3];
        let grad_fn = |x: &DVector<f64>| x * 2.0;
        
        let weighted_grad = ParallelSGDUtils::weighted_gradient_average(&points, &weights, grad_fn);
        
        // Weighted average: 0.7 * [2, 0] + 0.3 * [0, 2] = [1.4, 0.6]
        assert_relative_eq!(weighted_grad[0], 1.4, epsilon = 1e-10);
        assert_relative_eq!(weighted_grad[1], 0.6, epsilon = 1e-10);
    }
    
    #[test]
    fn test_sample_batch() {
        let data = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 1.0]),
            DVector::from_vec(vec![0.0, 0.0]),
        ];
        
        let batch = ParallelSGDUtils::sample_batch(&data, 2);
        
        assert_eq!(batch.ncols(), 2);
        assert_eq!(batch.nrows(), 2);
    }
}