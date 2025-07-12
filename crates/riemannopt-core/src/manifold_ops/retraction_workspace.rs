//! Retraction operations using pre-allocated workspace.
//!
//! This module provides workspace-aware implementations of retraction operations
//! to reduce memory allocations in performance-critical code.

use crate::{
    error::Result,
    memory::{Workspace, BufferId},
    types::{Scalar, DVector, DMatrix},
};

/// Workspace-aware retraction operations for vector manifolds.
pub struct VectorRetractionWorkspace;

impl VectorRetractionWorkspace {
    /// Performs retraction along a direction for Euclidean space.
    ///
    /// For Euclidean manifolds, this is simply point + t * direction.
    /// More complex manifolds would use the workspace for intermediate computations.
    pub fn euclidean_retract<T: Scalar>(
        point: &DVector<T>,
        direction: &DVector<T>,
        t: T,
        _workspace: &mut Workspace<T>,
    ) -> Result<DVector<T>> {
        Ok(point + direction * t)
    }

    /// Computes the inverse retraction (logarithm) for Euclidean space.
    ///
    /// For Euclidean manifolds, this is simply target - base.
    pub fn euclidean_inverse_retract<T: Scalar>(
        base: &DVector<T>,
        target: &DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<DVector<T>> {
        Ok(target - base)
    }

    /// Computes geodesic distance for Euclidean space.
    pub fn euclidean_distance<T: Scalar>(
        point1: &DVector<T>,
        point2: &DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<T> {
        let diff = Self::euclidean_inverse_retract(point1, point2, workspace)?;
        Ok(diff.norm())
    }

    /// Performs batch retraction for multiple directions efficiently.
    ///
    /// This is useful for algorithms that need to evaluate multiple search directions.
    pub fn batch_retract<T: Scalar>(
        point: &DVector<T>,
        directions: &[DVector<T>],
        t: T,
        results: &mut [DVector<T>],
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        assert_eq!(
            directions.len(),
            results.len(),
            "Results vector must have same length as directions"
        );

        let n = point.len();
        let temp = workspace.get_or_create_buffer(BufferId::Temp1, || DVector::<T>::zeros(n));

        for (direction, result) in directions.iter().zip(results.iter_mut()) {
            // temp = point + t * direction
            temp.copy_from(point);
            temp.axpy(t, direction, T::one());
            result.copy_from(&*temp);
        }

        Ok(())
    }

    /// Computes a geodesic curve between two points.
    ///
    /// Returns points along the geodesic at parameters t_i in [0, 1].
    pub fn geodesic_curve<T: Scalar>(
        start: &DVector<T>,
        end: &DVector<T>,
        t_values: &[T],
        results: &mut [DVector<T>],
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        assert_eq!(
            t_values.len(),
            results.len(),
            "Results must have same length as t_values"
        );

        let direction = Self::euclidean_inverse_retract(start, end, workspace)?;

        for (t, result) in t_values.iter().zip(results.iter_mut()) {
            let point = Self::euclidean_retract(start, &direction, *t, workspace)?;
            result.copy_from(&point);
        }

        Ok(())
    }
}

/// Workspace-aware retraction operations for matrix manifolds.
pub struct MatrixRetractionWorkspace;

impl MatrixRetractionWorkspace {
    /// Performs retraction along a direction for matrix Euclidean space.
    pub fn euclidean_retract<T: Scalar>(
        point: &DMatrix<T>,
        direction: &DMatrix<T>,
        t: T,
        _workspace: &mut Workspace<T>,
    ) -> Result<DMatrix<T>> {
        Ok(point + direction * t)
    }

    /// Computes the inverse retraction for matrix Euclidean space.
    pub fn euclidean_inverse_retract<T: Scalar>(
        base: &DMatrix<T>,
        target: &DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<DMatrix<T>> {
        Ok(target - base)
    }

    /// Computes Frobenius distance between matrices.
    pub fn frobenius_distance<T: Scalar>(
        mat1: &DMatrix<T>,
        mat2: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<T> {
        let diff = Self::euclidean_inverse_retract(mat1, mat2, workspace)?;
        Ok(diff.norm())
    }

    /// Performs QR-based retraction for matrix manifolds with orthogonality constraints.
    ///
    /// This is particularly useful for Stiefel and Grassmann manifolds.
    pub fn qr_retract<T: Scalar>(
        point: &DMatrix<T>,
        direction: &DMatrix<T>,
        t: T,
        workspace: &mut Workspace<T>,
    ) -> Result<DMatrix<T>> {
        let n = point.nrows();
        let p = point.ncols();
        
        // Compute point + t * direction
        let mut temp = workspace.acquire_temp_matrix(n, p);
        temp.copy_from(point);
        // temp = point + t * direction
        *temp += direction * t;
        
        // Perform QR decomposition
        let qr = temp.clone_owned().qr();
        let q = qr.q();
        
        // Return the Q factor (orthonormal columns)
        if q.ncols() > p {
            Ok(q.columns(0, p).into_owned())
        } else {
            Ok(q)
        }
    }
}

/// Allocating versions of retraction operations for convenience.
pub struct RetractionOps;

impl RetractionOps {
    /// Performs Euclidean retraction for vectors (allocating version).
    pub fn vector_retract<T: Scalar>(
        point: &DVector<T>,
        direction: &DVector<T>,
        t: T,
    ) -> Result<DVector<T>> {
        let mut workspace = Workspace::new();
        VectorRetractionWorkspace::euclidean_retract(point, direction, t, &mut workspace)
    }

    /// Computes inverse retraction for vectors (allocating version).
    pub fn vector_inverse_retract<T: Scalar>(
        base: &DVector<T>,
        target: &DVector<T>,
    ) -> Result<DVector<T>> {
        let mut workspace = Workspace::new();
        VectorRetractionWorkspace::euclidean_inverse_retract(base, target, &mut workspace)
    }

    /// Computes geodesic distance between vectors (allocating version).
    pub fn vector_distance<T: Scalar>(
        point1: &DVector<T>,
        point2: &DVector<T>,
    ) -> Result<T> {
        let mut workspace = Workspace::new();
        VectorRetractionWorkspace::euclidean_distance(point1, point2, &mut workspace)
    }

    /// Performs Euclidean retraction for matrices (allocating version).
    pub fn matrix_retract<T: Scalar>(
        point: &DMatrix<T>,
        direction: &DMatrix<T>,
        t: T,
    ) -> Result<DMatrix<T>> {
        let mut workspace = Workspace::new();
        MatrixRetractionWorkspace::euclidean_retract(point, direction, t, &mut workspace)
    }

    /// Computes inverse retraction for matrices (allocating version).
    pub fn matrix_inverse_retract<T: Scalar>(
        base: &DMatrix<T>,
        target: &DMatrix<T>,
    ) -> Result<DMatrix<T>> {
        let mut workspace = Workspace::new();
        MatrixRetractionWorkspace::euclidean_inverse_retract(base, target, &mut workspace)
    }

    /// Computes Frobenius distance between matrices (allocating version).
    pub fn matrix_distance<T: Scalar>(
        mat1: &DMatrix<T>,
        mat2: &DMatrix<T>,
    ) -> Result<T> {
        let mut workspace = Workspace::new();
        MatrixRetractionWorkspace::frobenius_distance(mat1, mat2, &mut workspace)
    }

    /// Performs QR retraction for matrices (allocating version).
    pub fn matrix_qr_retract<T: Scalar>(
        point: &DMatrix<T>,
        direction: &DMatrix<T>,
        t: T,
    ) -> Result<DMatrix<T>> {
        let mut workspace = Workspace::new();
        MatrixRetractionWorkspace::qr_retract(point, direction, t, &mut workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vector_retraction() {
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let direction = DVector::from_vec(vec![1.0, 0.0, -1.0]);
        let t = 0.5;
        let mut workspace = Workspace::new();

        let result = VectorRetractionWorkspace::euclidean_retract(
            &point,
            &direction,
            t,
            &mut workspace,
        )
        .unwrap();

        assert_eq!(result[0], 1.5); // 1 + 0.5 * 1
        assert_eq!(result[1], 2.0); // 2 + 0.5 * 0
        assert_eq!(result[2], 2.5); // 3 + 0.5 * (-1)
    }

    #[test]
    fn test_vector_distance() {
        let point1 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let point2 = DVector::from_vec(vec![3.0, 4.0, 0.0]);
        let mut workspace = Workspace::new();

        let dist = VectorRetractionWorkspace::euclidean_distance(
            &point1,
            &point2,
            &mut workspace,
        )
        .unwrap();

        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_retract() {
        let point = DVector::from_vec(vec![1.0, 1.0]);
        let directions = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![-1.0, -1.0]),
        ];
        let t = 0.1;
        let mut results = vec![DVector::zeros(2); 3];
        let mut workspace = Workspace::new();

        VectorRetractionWorkspace::batch_retract(
            &point,
            &directions,
            t,
            &mut results,
            &mut workspace,
        )
        .unwrap();

        assert_eq!(results[0][0], 1.1);
        assert_eq!(results[0][1], 1.0);
        assert_eq!(results[1][0], 1.0);
        assert_eq!(results[1][1], 1.1);
        assert_eq!(results[2][0], 0.9);
        assert_eq!(results[2][1], 0.9);
    }

    #[test]
    fn test_geodesic_curve() {
        let start = DVector::from_vec(vec![0.0, 0.0]);
        let end = DVector::from_vec(vec![2.0, 2.0]);
        let t_values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut results = vec![DVector::zeros(2); 5];
        let mut workspace = Workspace::new();

        VectorRetractionWorkspace::geodesic_curve(
            &start,
            &end,
            &t_values,
            &mut results,
            &mut workspace,
        )
        .unwrap();

        for (_i, (t, result)) in t_values.iter().zip(results.iter()).enumerate() {
            assert_relative_eq!(result[0], 2.0 * t, epsilon = 1e-10);
            assert_relative_eq!(result[1], 2.0 * t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_matrix_operations() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DMatrix::from_row_slice(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let t = 2.0;
        let mut workspace = Workspace::new();

        let result = MatrixRetractionWorkspace::euclidean_retract(&a, &b, t, &mut workspace).unwrap();

        assert_eq!(result[(0, 0)], 1.2); // 1 + 2 * 0.1
        assert_eq!(result[(0, 1)], 2.4); // 2 + 2 * 0.2
        assert_eq!(result[(1, 0)], 3.6); // 3 + 2 * 0.3
        assert_eq!(result[(1, 1)], 4.8); // 4 + 2 * 0.4
    }

    #[test]
    fn test_qr_retraction() {
        let point = DMatrix::from_row_slice(3, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        let direction = DMatrix::from_row_slice(3, 2, &[
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
        ]);
        let t = 0.1;
        let mut workspace = Workspace::with_size(20);

        let result = MatrixRetractionWorkspace::qr_retract(
            &point,
            &direction,
            t,
            &mut workspace,
        )
        .unwrap();

        // Check that result has orthonormal columns
        let gram = result.transpose() * &result;
        assert_relative_eq!(gram[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(gram[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(gram[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(gram[(1, 0)], 0.0, epsilon = 1e-10);
    }
}