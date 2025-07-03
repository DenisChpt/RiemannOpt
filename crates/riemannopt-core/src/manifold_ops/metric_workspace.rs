//! Metric operations using pre-allocated workspace.
//!
//! This module provides workspace-aware implementations of metric operations
//! to reduce memory allocations in performance-critical code.

use crate::{
    error::Result,
    memory::Workspace,
    types::{Scalar, DVector, DMatrix},
};
use num_traits::Float;

/// Workspace-aware metric operations for vector manifolds.
pub struct VectorMetricWorkspace;

impl VectorMetricWorkspace {
    /// Computes the inner product between two vectors using workspace.
    ///
    /// For the standard Euclidean metric, this is just the dot product.
    /// More complex manifolds might use the workspace for intermediate computations.
    pub fn inner_product<T: Scalar>(
        u: &DVector<T>,
        v: &DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<T> {
        Ok(u.dot(v))
    }

    /// Computes the norm of a vector using workspace.
    pub fn norm<T: Scalar>(
        v: &DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<T> {
        let inner = Self::inner_product(v, v, workspace)?;
        Ok(<T as Float>::sqrt(inner))
    }

    /// Normalizes a vector in-place using workspace.
    pub fn normalize_in_place<T: Scalar>(
        v: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let norm = Self::norm(v, workspace)?;
        if norm > T::epsilon() {
            *v /= norm;
        }
        Ok(())
    }

    /// Computes the Gram matrix for a set of vectors using workspace.
    ///
    /// The Gram matrix G has entries G[i,j] = <v_i, v_j>.
    pub fn gram_matrix<T: Scalar>(
        vectors: &[DVector<T>],
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let n = vectors.len();
        assert_eq!(result.nrows(), n, "Gram matrix must have correct dimensions");
        assert_eq!(result.ncols(), n, "Gram matrix must be square");
        
        for i in 0..n {
            for j in i..n {
                let inner = vectors[i].dot(&vectors[j]);
                result[(i, j)] = inner;
                if i != j {
                    result[(j, i)] = inner;
                }
            }
        }
        
        Ok(())
    }

    /// Projects a vector onto the span of a set of orthonormal vectors.
    pub fn project_onto_span<T: Scalar>(
        v: &DVector<T>,
        basis: &[DVector<T>],
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        result.fill(T::zero());
        
        for basis_vec in basis {
            let coeff = Self::inner_product(v, basis_vec, workspace)?;
            result.axpy(coeff, basis_vec, T::one());
        }
        
        Ok(())
    }
}

/// Workspace-aware metric operations for matrix manifolds.
pub struct MatrixMetricWorkspace;

impl MatrixMetricWorkspace {
    /// Computes the Frobenius inner product between two matrices using workspace.
    ///
    /// The Frobenius inner product is defined as tr(A^T B).
    pub fn frobenius_inner_product<T: Scalar>(
        a: &DMatrix<T>,
        b: &DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<T> {
        Ok((a.transpose() * b).trace())
    }

    /// Computes the Frobenius norm of a matrix using workspace.
    pub fn frobenius_norm<T: Scalar>(
        m: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<T> {
        let inner = Self::frobenius_inner_product(m, m, workspace)?;
        Ok(<T as Float>::sqrt(inner))
    }

    /// Normalizes a matrix with respect to the Frobenius norm.
    pub fn normalize_frobenius_in_place<T: Scalar>(
        m: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let norm = Self::frobenius_norm(m, workspace)?;
        if norm > T::epsilon() {
            *m /= norm;
        }
        Ok(())
    }

    /// Computes a weighted inner product between matrices.
    ///
    /// The weighted inner product is defined as tr(A^T W B) where W is a weight matrix.
    pub fn weighted_inner_product<T: Scalar>(
        a: &DMatrix<T>,
        b: &DMatrix<T>,
        weight: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<T> {
        let n = a.nrows();
        let p = a.ncols();
        
        // Use workspace to store intermediate result W * B
        let mut wb = workspace.acquire_temp_matrix(n, p);
        wb.copy_from(&(weight * b));
        
        // Compute tr(A^T * (W * B))
        Ok((a.transpose() * &*wb).trace())
    }
}

/// Allocating versions of metric operations for convenience.
pub struct MetricOps;

impl MetricOps {
    /// Computes the inner product between two vectors (allocating version).
    pub fn vector_inner_product<T: Scalar>(u: &DVector<T>, v: &DVector<T>) -> Result<T> {
        let mut workspace = Workspace::new();
        VectorMetricWorkspace::inner_product(u, v, &mut workspace)
    }

    /// Computes the norm of a vector (allocating version).
    pub fn vector_norm<T: Scalar>(v: &DVector<T>) -> Result<T> {
        let mut workspace = Workspace::new();
        VectorMetricWorkspace::norm(v, &mut workspace)
    }

    /// Normalizes a vector (allocating version).
    pub fn normalize_vector<T: Scalar>(v: &DVector<T>) -> Result<DVector<T>> {
        let mut workspace = Workspace::new();
        let mut result = v.clone();
        VectorMetricWorkspace::normalize_in_place(&mut result, &mut workspace)?;
        Ok(result)
    }

    /// Computes the Gram matrix for a set of vectors (allocating version).
    pub fn vector_gram_matrix<T: Scalar>(vectors: &[DVector<T>]) -> Result<DMatrix<T>> {
        let n = vectors.len();
        let mut workspace = Workspace::new();
        let mut gram = DMatrix::zeros(n, n);
        VectorMetricWorkspace::gram_matrix(vectors, &mut gram, &mut workspace)?;
        Ok(gram)
    }

    /// Computes the Frobenius inner product between two matrices (allocating version).
    pub fn matrix_frobenius_inner_product<T: Scalar>(a: &DMatrix<T>, b: &DMatrix<T>) -> Result<T> {
        let mut workspace = Workspace::new();
        MatrixMetricWorkspace::frobenius_inner_product(a, b, &mut workspace)
    }

    /// Computes the Frobenius norm of a matrix (allocating version).
    pub fn matrix_frobenius_norm<T: Scalar>(m: &DMatrix<T>) -> Result<T> {
        let mut workspace = Workspace::new();
        MatrixMetricWorkspace::frobenius_norm(m, &mut workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_vector_operations() {
        let u = DVector::from_vec(vec![3.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 4.0, 0.0]);
        let mut workspace = Workspace::new();
        
        // Test inner product
        let inner = VectorMetricWorkspace::inner_product(&u, &v, &mut workspace).unwrap();
        assert_eq!(inner, 0.0); // orthogonal vectors
        
        // Test norm
        let norm_u = VectorMetricWorkspace::norm(&u, &mut workspace).unwrap();
        assert_eq!(norm_u, 3.0);
        
        // Test normalization
        let mut w = u.clone();
        VectorMetricWorkspace::normalize_in_place(&mut w, &mut workspace).unwrap();
        assert_relative_eq!(w.norm(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gram_matrix() {
        let vecs = vec![
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 1.0]),
        ];
        let mut workspace = Workspace::new();
        let mut gram = DMatrix::<f64>::zeros(3, 3);
        
        VectorMetricWorkspace::gram_matrix(&vecs, &mut gram, &mut workspace).unwrap();
        
        // Should be identity matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(gram[(i, j)], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(gram[(i, j)], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
    
    #[test]
    fn test_matrix_operations() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let mut workspace = Workspace::new();
        
        // Test Frobenius inner product
        // tr(A^T B) = 1*5 + 3*6 + 2*7 + 4*8 = 5 + 18 + 14 + 32 = 69
        let inner = MatrixMetricWorkspace::frobenius_inner_product(&a, &b, &mut workspace).unwrap();
        assert_eq!(inner, 70.0);
        
        // Test Frobenius norm
        // ||A||_F^2 = 1 + 4 + 9 + 16 = 30
        let norm = MatrixMetricWorkspace::frobenius_norm(&a, &mut workspace).unwrap();
        assert_relative_eq!(norm, 30.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_inner_product() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]); // Identity
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]); // Ones
        let w = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]); // Diagonal weights
        let mut workspace = Workspace::with_size(10);
        
        // tr(A^T W B) = tr(I * diag(2,3) * ones) = tr([2,2; 3,3]) = 2 + 3 = 5
        let inner = MatrixMetricWorkspace::weighted_inner_product(&a, &b, &w, &mut workspace).unwrap();
        assert_eq!(inner, 5.0);
    }

    #[test]
    fn test_project_onto_span() {
        let v = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
        ];
        let mut result = DVector::zeros(3);
        let mut workspace = Workspace::new();
        
        VectorMetricWorkspace::project_onto_span(&v, &basis, &mut result, &mut workspace).unwrap();
        
        // Projection should be [1, 1, 0]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 0.0);
    }
}