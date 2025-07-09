//! Tangent space operations using pre-allocated workspace.
//!
//! This module provides workspace-aware implementations of tangent space operations
//! to reduce memory allocations in performance-critical code.

use crate::{
    error::Result,
    memory::{Workspace, BufferId},
    types::{Scalar, DVector, DMatrix},
};

/// Workspace-aware tangent vector operations for general vectors.
pub struct TangentVectorWorkspace;

impl TangentVectorWorkspace {
    /// Projects a vector onto the tangent space of a sphere.
    ///
    /// For a sphere, the tangent space at point p consists of all vectors v
    /// such that <p, v> = 0.
    pub fn project_sphere<T: Scalar>(
        point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let n = point.len();
        assert_eq!(vector.len(), n, "Vector dimensions must match");
        assert_eq!(result.len(), n, "Result dimensions must match");
        
        // Compute projection: v - <v, p> * p
        let inner = point.dot(vector);
        
        // Use workspace for temporary storage
        let temp = workspace.get_or_create_vector(BufferId::Temp1, n);
        temp.copy_from(point);
        temp.scale_mut(inner);
        
        result.copy_from(vector);
        result.axpy(-T::one(), &*temp, T::one());
        
        Ok(())
    }
    
    /// Normalizes a tangent vector in-place.
    pub fn normalize_in_place<T: Scalar>(
        vector: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let norm = vector.norm();
        if norm > T::epsilon() {
            *vector /= norm;
        }
        Ok(())
    }
    
    /// Computes the parallel transport of a vector along a geodesic on the sphere.
    ///
    /// This is a simplified version that assumes we're on a sphere manifold.
    pub fn parallel_transport_sphere<T: Scalar>(
        from_point: &DVector<T>,
        to_point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let n = from_point.len();
        assert_eq!(to_point.len(), n, "Point dimensions must match");
        assert_eq!(vector.len(), n, "Vector dimensions must match");
        assert_eq!(result.len(), n, "Result dimensions must match");
        
        // For the sphere, parallel transport along a geodesic from p to q
        // of a tangent vector v at p is given by:
        // PT(v) = v - <v, q> / (1 + <p, q>) * (p + q)
        
        let p_dot_q = from_point.dot(to_point);
        let v_dot_q = vector.dot(to_point);
        
        if num_traits::Float::abs(T::one() + p_dot_q) < T::epsilon() {
            // Points are antipodal, parallel transport is undefined
            result.fill(T::zero());
            return Ok(());
        }
        
        let scale = v_dot_q / (T::one() + p_dot_q);
        
        // Use workspace for p + q
        let temp = workspace.get_or_create_vector(BufferId::Temp1, n);
        temp.copy_from(from_point);
        temp.axpy(T::one(), to_point, T::one());
        
        result.copy_from(vector);
        result.axpy(-scale, &*temp, T::one());
        
        Ok(())
    }
    
    /// Computes a random tangent vector at a point on the sphere.
    pub fn random_tangent_sphere<T: Scalar>(
        point: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let n = point.len();
        assert_eq!(result.len(), n, "Result dimensions must match");
        
        // Generate a random vector
        // Note: This is a placeholder - proper random generation would be needed
        for i in 0..n {
            result[i] = <T as Scalar>::from_f64(((i + 1) as f64).sin());
        }
        
        // Project onto tangent space
        let temp_vector = {
            let temp_result = workspace.get_or_create_vector(BufferId::Temp2, result.len());
            temp_result.copy_from(result);
            temp_result.clone_owned()
        };
        Self::project_sphere(point, &temp_vector, result, workspace)?;
        
        // Normalize
        Self::normalize_in_place(result, workspace)?;
        
        Ok(())
    }
}

/// Workspace-aware tangent operations for matrix manifolds.
pub struct MatrixTangentWorkspace;

impl MatrixTangentWorkspace {
    /// Projects a matrix onto the tangent space of the Stiefel manifold.
    ///
    /// For the Stiefel manifold, the tangent space at X consists of matrices Z
    /// such that X^T Z + Z^T X = 0 (skew-symmetric).
    pub fn project_stiefel<T: Scalar>(
        point: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let n = point.nrows();
        let p = point.ncols();
        assert_eq!(vector.nrows(), n, "Vector row dimensions must match");
        assert_eq!(vector.ncols(), p, "Vector column dimensions must match");
        
        // Compute projection: Z - X * (X^T * Z + Z^T * X) / 2
        let mut temp1 = workspace.acquire_temp_matrix(p, p);
        let mut temp2 = workspace.acquire_temp_matrix(p, p);
        
        // temp1 = X^T * Z
        temp1.gemm(T::one(), &point.transpose(), vector, T::zero());
        
        // temp2 = Z^T * X
        temp2.gemm(T::one(), &vector.transpose(), point, T::zero());
        
        // temp1 = (X^T * Z + Z^T * X) / 2
        *temp1 += &*temp2;
        *temp1 *= <T as Scalar>::from_f64(0.5);
        
        // result = Z - X * temp1
        result.copy_from(vector);
        result.gemm(-T::one(), point, &temp1, T::one());
        
        Ok(())
    }
    
    /// Normalizes a matrix tangent vector with respect to the Frobenius norm.
    pub fn normalize_frobenius_in_place<T: Scalar>(
        matrix: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let norm = matrix.norm();
        if norm > T::epsilon() {
            *matrix /= norm;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_sphere_projection() {
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let vector = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        let mut result = DVector::zeros(3);
        let mut workspace = Workspace::new();
        
        TangentVectorWorkspace::project_sphere(&point, &vector, &mut result, &mut workspace).unwrap();
        
        // Result should be orthogonal to point
        let inner = point.dot(&result);
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
        
        // Should be [0, 1, 0]
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_parallel_transport_sphere() {
        let sqrt2 = 2.0_f64.sqrt() / 2.0;
        let from_point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let to_point = DVector::from_vec(vec![sqrt2, sqrt2, 0.0]);
        let vector = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let mut result = DVector::zeros(3);
        let mut workspace = Workspace::new();
        
        TangentVectorWorkspace::parallel_transport_sphere(
            &from_point,
            &to_point,
            &vector,
            &mut result,
            &mut workspace,
        ).unwrap();
        
        // Result should be in tangent space at to_point
        let inner = to_point.dot(&result);
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_stiefel_projection() {
        let point = DMatrix::from_row_slice(3, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        let vector = DMatrix::from_row_slice(3, 2, &[
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
        ]);
        let mut result = DMatrix::zeros(3, 2);
        let mut workspace = Workspace::with_size(20);
        
        MatrixTangentWorkspace::project_stiefel(&point, &vector, &mut result, &mut workspace).unwrap();
        
        // Check that X^T * Z + Z^T * X ≈ 0
        let check = point.transpose() * &result + result.transpose() * &point;
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(check[(i, j)], 0.0, epsilon = 1e-10);
            }
        }
    }
}