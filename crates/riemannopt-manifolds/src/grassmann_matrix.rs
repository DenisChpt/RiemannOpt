//! Matrix-based implementation of the Grassmann manifold.
//!
//! This module provides a `MatrixManifold` implementation for the Grassmann manifold,
//! operating directly on matrix representations of subspaces.

use nalgebra::DMatrix;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
    error::{ManifoldError, Result},
    memory::Workspace,
    types::Scalar,
};

use crate::matrix_manifold::{MatrixManifold, MatrixManifoldExt};
use crate::impl_manifold_for_matrix_manifold;

/// The Grassmann manifold Gr(n,p) using matrix operations.
///
/// The Grassmann manifold is the set of p-dimensional linear subspaces in R^n.
/// Elements are represented as n×p matrices with orthonormal columns, where
/// two matrices represent the same point if their column spaces are identical.
#[derive(Debug, Clone)]
pub struct GrassmannMatrix {
    n: usize,
    p: usize,
}

impl GrassmannMatrix {
    /// Creates a new Grassmann manifold Gr(n,p).
    ///
    /// # Arguments
    ///
    /// * `n` - Ambient dimension (must be >= p)
    /// * `p` - Subspace dimension (must be > 0)
    ///
    /// # Returns
    ///
    /// A new GrassmannMatrix instance.
    ///
    /// # Errors
    ///
    /// Returns an error if n < p or if p = 0.
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if p == 0 {
            return Err(ManifoldError::invalid_point(
                "Grassmann manifold requires p > 0"
            ));
        }
        if n < p {
            return Err(ManifoldError::invalid_point(
                format!("Grassmann manifold Gr(n,p) requires n >= p, got n={}, p={}", n, p)
            ));
        }
        Ok(Self { n, p })
    }

    /// QR-based projection onto the Grassmann manifold.
    fn qr_projection<T: Scalar>(&self, matrix: &DMatrix<T>) -> DMatrix<T> {
        let qr = matrix.clone().qr();
        let mut q = qr.q();
        
        // Ensure we only take the first p columns
        if q.ncols() > self.p {
            q = q.columns(0, self.p).into_owned();
        }
        
        q
    }
}

impl<T: Scalar> MatrixManifold<T> for GrassmannMatrix {
    fn name(&self) -> &str {
        "GrassmannMatrix"
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.p
    }

    fn dimension(&self) -> usize {
        self.p * (self.n - self.p)
    }

    fn is_point_on_manifold(&self, point: &DMatrix<T>, tolerance: T) -> bool {
        if point.nrows() != self.n || point.ncols() != self.p {
            return false;
        }
        
        // Check Y^T Y = I_p
        let y_t_y = point.transpose() * point;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        let diff = &y_t_y - &identity;
        
        diff.norm() < tolerance
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DMatrix<T>,
        vector: &DMatrix<T>,
        tolerance: T,
    ) -> bool {
        if !self.is_point_on_manifold(point, tolerance) {
            return false;
        }
        
        if vector.nrows() != self.n || vector.ncols() != self.p {
            return false;
        }
        
        // Horizontal space: Y^T V = 0
        let y_t_v = point.transpose() * vector;
        
        y_t_v.norm() < tolerance
    }

    fn project_point(
        &self,
        matrix: &DMatrix<T>,
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) {
        let projected = self.qr_projection(matrix);
        result.copy_from(&projected);
    }

    fn project_tangent(
        &self,
        point: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Projection to horizontal space: V - Y(Y^T V)
        let mut y_t_v = workspace.acquire_temp_matrix(self.p, self.p);
        y_t_v.copy_from(&(point.transpose() * vector));
        
        result.copy_from(vector);
        result.gemm(-T::one(), point, &*y_t_v, T::one());
        
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &DMatrix<T>,
        u: &DMatrix<T>,
        v: &DMatrix<T>,
    ) -> Result<T> {
        // Canonical metric: tr(U^T V)
        Ok((u.transpose() * v).trace())
    }

    fn retract(
        &self,
        point: &DMatrix<T>,
        tangent: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // QR retraction for Grassmann
        let mut y_plus_v = workspace.acquire_temp_matrix(self.n, self.p);
        y_plus_v.copy_from(&(point + tangent));
        
        let qr = y_plus_v.clone().qr();
        let q = qr.q();
        
        if q.ncols() > self.p {
            result.copy_from(&q.columns(0, self.p));
        } else {
            result.copy_from(&q);
        }
        
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &DMatrix<T>,
        other: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Approximate inverse retraction
        let mut diff = workspace.acquire_temp_matrix(self.n, self.p);
        diff.copy_from(&(other - point));
        
        self.project_tangent(point, &*diff, result, workspace)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DMatrix<T>,
        euclidean_grad: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For canonical metric on Grassmann, project to horizontal space
        self.project_tangent(point, euclidean_grad, result, workspace)
    }

    fn random_point(&self) -> DMatrix<T> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random n×p matrix
        let mut matrix = DMatrix::<T>::zeros(self.n, self.p);
        for i in 0..self.n {
            for j in 0..self.p {
                matrix[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // Project to Grassmann via QR
        self.qr_projection(&matrix)
    }

    fn random_tangent(
        &self,
        point: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random matrix
        let mut tangent = DMatrix::<T>::zeros(self.n, self.p);
        for i in 0..self.n {
            for j in 0..self.p {
                tangent[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // Project to horizontal space
        self.project_tangent(point, &tangent, result, workspace)
    }

    fn parallel_transport(
        &self,
        _from: &DMatrix<T>,
        to: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Simple parallel transport: project to tangent space at destination
        self.project_tangent(to, vector, result, workspace)
    }

    fn distance(
        &self,
        x: &DMatrix<T>,
        y: &DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<T> {
        // Principal angle distance
        let x_t_y = x.transpose() * y;
        let svd = x_t_y.svd(true, true);
        
        let mut dist_squared = T::zero();
        for &sigma in svd.singular_values.iter() {
            let clamped = Float::min(Float::max(sigma, -T::one()), T::one());
            let angle = Float::acos(clamped);
            dist_squared += angle * angle;
        }
        
        Ok(Float::sqrt(dist_squared))
    }

    fn has_exact_exp_log(&self) -> bool {
        false // QR retraction is not exact
    }
}

impl<T: Scalar> MatrixManifoldExt<T> for GrassmannMatrix {
    fn vector_to_matrix(&self, vector: &[T]) -> DMatrix<T> {
        DMatrix::from_column_slice(self.n, self.p, vector)
    }

    fn matrix_to_vector(&self, matrix: &DMatrix<T>) -> Vec<T> {
        matrix.as_slice().to_vec()
    }

    fn vector_length(&self) -> usize {
        self.n * self.p
    }
}

// Generate the vector-based Manifold implementation
impl_manifold_for_matrix_manifold!(GrassmannMatrix);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grassmann_matrix_creation() {
        let grassmann = GrassmannMatrix::new(5, 2).unwrap();
        assert_eq!(<GrassmannMatrix as MatrixManifold<f64>>::nrows(&grassmann), 5);
        assert_eq!(<GrassmannMatrix as MatrixManifold<f64>>::ncols(&grassmann), 2);
        assert_eq!(<GrassmannMatrix as MatrixManifold<f64>>::dimension(&grassmann), 6); // 2*(5-2) = 6
        
        // Error cases
        assert!(GrassmannMatrix::new(2, 5).is_err()); // n < p
        assert!(GrassmannMatrix::new(5, 0).is_err()); // p = 0
    }

    #[test]
    fn test_point_on_manifold() {
        let grassmann = GrassmannMatrix::new(4, 2).unwrap();
        
        // Create orthonormal matrix
        let point = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]);
        
        assert!(grassmann.is_point_on_manifold(&point, 1e-10));
    }

    #[test]
    fn test_tangent_space() {
        let grassmann = GrassmannMatrix::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let point = grassmann.random_point();
        
        // Create vector in horizontal space
        let mut tangent = DMatrix::zeros(3, 2);
        grassmann.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        
        assert!(grassmann.is_vector_in_tangent_space(&point, &tangent, 1e-10));
        
        // Check Y^T V = 0
        let y_t_v = point.transpose() * &tangent;
        assert!(y_t_v.norm() < 1e-10);
    }

    #[test]
    fn test_retraction() {
        let grassmann = GrassmannMatrix::new(4, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let point = grassmann.random_point();
        let mut tangent = DMatrix::zeros(4, 2);
        grassmann.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        
        // Scale for small step
        tangent *= 0.1;
        
        let mut new_point = DMatrix::zeros(4, 2);
        grassmann.retract(&point, &tangent, &mut new_point, &mut workspace).unwrap();
        
        assert!(grassmann.is_point_on_manifold(&new_point, 1e-10));
    }
}