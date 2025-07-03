//! Riemannian metric structures and implementations.
//!
//! This module provides various Riemannian metric implementations that can be
//! used to define the geometry of manifolds. A Riemannian metric assigns an
//! inner product to each tangent space, allowing measurement of lengths and angles.
//!
//! # Mathematical Background
//!
//! A Riemannian metric g on a manifold M assigns to each point p ∈ M an inner
//! product g_p on the tangent space T_p M. This allows us to:
//! - Measure lengths of curves
//! - Define angles between tangent vectors
//! - Compute volumes
//! - Define geodesics (shortest paths)
//!
//! The metric tensor g_{ij} in local coordinates provides the components of the metric,
//! and the Christoffel symbols Γ^k_{ij} encode how the metric changes across the manifold.

use crate::{
    error::{ManifoldError, Result},
    types::{DVector, DMatrix, Scalar},
};
use num_traits::Float;
use std::fmt::Debug;

/// Represents a metric tensor for vector manifolds.
///
/// The metric tensor is a symmetric positive-definite matrix that
/// encodes the inner product structure at a point.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorMetricTensor<T: Scalar> {
    /// The metric tensor matrix (symmetric positive-definite)
    pub matrix: DMatrix<T>,
}

impl<T: Scalar> VectorMetricTensor<T> {
    /// Creates a new metric tensor from a symmetric positive-definite matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A symmetric positive-definite matrix
    ///
    /// # Returns
    ///
    /// A new metric tensor or an error if the matrix is invalid
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not symmetric or not positive-definite.
    pub fn new(matrix: DMatrix<T>) -> Result<Self> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(ManifoldError::dimension_mismatch(
                "Metric tensor must be square",
                format!("Got {}x{} matrix", matrix.nrows(), matrix.ncols()),
            ));
        }

        // Check symmetry
        for i in 0..n {
            for j in i + 1..n {
                if <T as Float>::abs(matrix[(i, j)] - matrix[(j, i)]) > T::epsilon() {
                    return Err(ManifoldError::numerical_error(
                        "Metric tensor must be symmetric",
                    ));
                }
            }
        }

        // Check positive-definiteness using eigenvalue decomposition
        let eigen = matrix.clone().symmetric_eigen();
        let min_eigenvalue = eigen.eigenvalues.iter()
            .fold(T::infinity(), |min, &val| <T as Float>::min(min, val));
        
        if min_eigenvalue <= T::epsilon() {
            return Err(ManifoldError::numerical_error(
                "Metric tensor must be positive definite",
            ));
        }

        Ok(Self { matrix })
    }

    /// Creates an identity metric tensor of the given dimension.
    ///
    /// This represents the standard Euclidean metric.
    pub fn identity(dim: usize) -> Self {
        Self {
            matrix: DMatrix::identity(dim, dim),
        }
    }

    /// Computes the inner product between two vectors using this metric.
    ///
    /// # Arguments
    ///
    /// * `u` - First vector
    /// * `v` - Second vector
    ///
    /// # Returns
    ///
    /// The inner product g(u, v) = u^T G v
    pub fn inner_product(&self, u: &DVector<T>, v: &DVector<T>) -> Result<T> {
        if u.len() != self.matrix.nrows() || v.len() != self.matrix.nrows() {
            return Err(ManifoldError::dimension_mismatch(
                "Vectors must have same dimension as metric tensor",
                format!("Metric: {}, vectors: {}, {}", self.matrix.nrows(), u.len(), v.len()),
            ));
        }

        // Compute u^T * G * v
        let gv = &self.matrix * v;
        Ok(u.dot(&gv))
    }

    /// Computes the norm of a vector using this metric.
    pub fn norm(&self, v: &DVector<T>) -> Result<T> {
        let norm_sq = self.inner_product(v, v)?;
        Ok(<T as Float>::sqrt(norm_sq))
    }

    /// Returns the inverse of the metric tensor.
    ///
    /// This is useful for converting between covariant and contravariant representations.
    pub fn inverse(&self) -> Result<DMatrix<T>> {
        self.matrix.clone().try_inverse()
            .ok_or_else(|| ManifoldError::numerical_error("Metric tensor is not invertible"))
    }

    /// Returns the dimension of the space this metric is defined on.
    pub fn dimension(&self) -> usize {
        self.matrix.nrows()
    }
}

/// Represents a metric tensor for matrix manifolds.
///
/// For matrix manifolds, the metric often has a special structure that
/// can be exploited for efficiency.
#[derive(Debug, Clone)]
pub struct MatrixMetricTensor<T: Scalar> {
    /// The metric type
    metric_type: MatrixMetricType,
    /// Parameters for the metric (interpretation depends on metric_type)
    params: Vec<T>,
    /// Dimensions of the matrix manifold
    nrows: usize,
    ncols: usize,
}

/// Types of metrics commonly used for matrix manifolds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMetricType {
    /// Standard Frobenius inner product: <U, V> = tr(U^T V)
    Frobenius,
    /// Weighted Frobenius: <U, V> = tr(U^T W V) for some weight matrix W
    WeightedFrobenius,
    /// Canonical metric for SPD manifolds: <U, V>_P = tr(P^{-1} U P^{-1} V)
    SPDCanonical,
    /// Canonical metric for Stiefel manifolds
    StiefelCanonical,
}

impl<T: Scalar> MatrixMetricTensor<T> {
    /// Creates a new Frobenius metric for matrices of the given size.
    pub fn frobenius(nrows: usize, ncols: usize) -> Self {
        Self {
            metric_type: MatrixMetricType::Frobenius,
            params: vec![],
            nrows,
            ncols,
        }
    }

    /// Creates a weighted Frobenius metric.
    ///
    /// The weight matrix should be provided as a flattened vector.
    pub fn weighted_frobenius(nrows: usize, ncols: usize, weights: Vec<T>) -> Result<Self> {
        if weights.len() != nrows * ncols {
            return Err(ManifoldError::dimension_mismatch(
                "Weight vector has incorrect size",
                format!("Expected {}, got {}", nrows * ncols, weights.len()),
            ));
        }

        Ok(Self {
            metric_type: MatrixMetricType::WeightedFrobenius,
            params: weights,
            nrows,
            ncols,
        })
    }

    /// Computes the inner product between two matrices using this metric.
    pub fn inner_product(&self, u: &DMatrix<T>, v: &DMatrix<T>) -> Result<T> {
        if u.nrows() != self.nrows || u.ncols() != self.ncols ||
           v.nrows() != self.nrows || v.ncols() != self.ncols {
            return Err(ManifoldError::dimension_mismatch(
                "Matrices must have correct dimensions for metric",
                format!("Expected {}x{}, got {}x{} and {}x{}", 
                    self.nrows, self.ncols, u.nrows(), u.ncols(), v.nrows(), v.ncols()),
            ));
        }

        match self.metric_type {
            MatrixMetricType::Frobenius => {
                // Standard Frobenius: tr(U^T V)
                Ok((u.transpose() * v).trace())
            }
            MatrixMetricType::WeightedFrobenius => {
                // Weighted: sum_ij w_ij * u_ij * v_ij
                let mut result = T::zero();
                for i in 0..self.nrows {
                    for j in 0..self.ncols {
                        let idx = i * self.ncols + j;
                        result += self.params[idx] * u[(i, j)] * v[(i, j)];
                    }
                }
                Ok(result)
            }
            _ => Err(ManifoldError::not_implemented(
                "This metric type requires additional context"
            )),
        }
    }

    /// Returns the type of this metric.
    pub fn metric_type(&self) -> MatrixMetricType {
        self.metric_type
    }

    /// Returns the dimensions of matrices this metric operates on.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

/// Utilities for working with metric tensors.
pub struct MetricUtils;

impl MetricUtils {
    /// Converts a covariant vector to a contravariant vector.
    ///
    /// Given a covector (gradient) g and metric tensor G, returns G^{-1} g.
    pub fn raise_index<T: Scalar>(
        metric: &VectorMetricTensor<T>,
        covector: &DVector<T>,
    ) -> Result<DVector<T>> {
        let g_inv = metric.inverse()?;
        Ok(&g_inv * covector)
    }

    /// Converts a contravariant vector to a covariant vector.
    ///
    /// Given a vector v and metric tensor G, returns G v.
    pub fn lower_index<T: Scalar>(
        metric: &VectorMetricTensor<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        Ok(&metric.matrix * vector)
    }

    /// Computes the angle between two vectors using a metric.
    pub fn angle<T: Scalar>(
        metric: &VectorMetricTensor<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        let norm_u = metric.norm(u)?;
        let norm_v = metric.norm(v)?;
        
        if norm_u < T::epsilon() || norm_v < T::epsilon() {
            return Err(ManifoldError::numerical_error(
                "Cannot compute angle with zero vector"
            ));
        }
        
        let cos_angle = metric.inner_product(u, v)? / (norm_u * norm_v);
        // Clamp to avoid numerical issues
        let cos_angle = <T as Float>::max(cos_angle, -T::one());
        let cos_angle = <T as Float>::min(cos_angle, T::one());
        
        Ok(<T as Float>::acos(cos_angle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vector_metric_tensor() {
        // Test identity metric
        let metric = VectorMetricTensor::<f64>::identity(3);
        let u = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0, 0.0]);

        // Orthogonal vectors should have zero inner product
        assert_eq!(metric.inner_product(&u, &v).unwrap(), 0.0);

        // Self inner product should equal squared norm
        assert_eq!(metric.inner_product(&u, &u).unwrap(), 1.0);
        assert_eq!(metric.norm(&u).unwrap(), 1.0);
    }

    #[test]
    fn test_weighted_metric() {
        // Create a diagonal metric with weights [2, 3, 4]
        let weights = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0, 4.0]));
        let metric = VectorMetricTensor::new(weights).unwrap();

        let v = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        
        // Inner product should be 2*1^2 + 3*1^2 + 4*1^2 = 9
        assert_eq!(metric.inner_product(&v, &v).unwrap(), 9.0);
        assert_eq!(metric.norm(&v).unwrap(), 3.0);
    }

    #[test]
    fn test_matrix_frobenius_metric() {
        let metric = MatrixMetricTensor::<f64>::frobenius(2, 2);
        
        let u = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]); // Identity
        let v = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]); // Anti-diagonal
        
        // tr(I * Anti-diag) = 0
        assert_eq!(metric.inner_product(&u, &v).unwrap(), 0.0);
        
        // tr(I * I) = 2
        assert_eq!(metric.inner_product(&u, &u).unwrap(), 2.0);
    }

    #[test]
    fn test_metric_angle() {
        let metric = VectorMetricTensor::<f64>::identity(2);
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        
        let angle = MetricUtils::angle(&metric, &u, &v).unwrap();
        assert_relative_eq!(angle, std::f64::consts::PI / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_raise_lower_index() {
        // Use a non-trivial metric
        let g = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let metric = VectorMetricTensor::new(g).unwrap();
        
        let v = DVector::from_vec(vec![1.0, 2.0]);
        
        // Lower then raise should give back the original
        let v_lower = MetricUtils::lower_index(&metric, &v).unwrap();
        let v_raised = MetricUtils::raise_index(&metric, &v_lower).unwrap();
        
        assert_relative_eq!(v[0], v_raised[0], epsilon = 1e-10);
        assert_relative_eq!(v[1], v_raised[1], epsilon = 1e-10);
    }
}