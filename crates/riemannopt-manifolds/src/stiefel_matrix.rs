//! Matrix-based implementation of the Stiefel manifold.
//!
//! This module provides a `MatrixManifold` implementation for the Stiefel manifold,
//! allowing it to operate directly on matrix representations without vectorization overhead.

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

/// The Stiefel manifold St(n,p) using matrix operations.
///
/// This implementation operates directly on n×p matrices, avoiding the
/// overhead of vector-matrix conversions in the base Stiefel implementation.
#[derive(Debug, Clone)]
pub struct StiefelMatrix {
    n: usize,
    p: usize,
}

impl StiefelMatrix {
    /// Creates a new Stiefel manifold St(n,p).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows (must be >= p)
    /// * `p` - Number of columns (must be > 0)
    ///
    /// # Returns
    ///
    /// A new StiefelMatrix instance.
    ///
    /// # Errors
    ///
    /// Returns an error if n < p or if p = 0.
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if p == 0 {
            return Err(ManifoldError::invalid_point(
                "Stiefel manifold requires p > 0"
            ));
        }
        if n < p {
            return Err(ManifoldError::invalid_point(
                format!("Stiefel manifold St(n,p) requires n >= p, got n={}, p={}", n, p)
            ));
        }
        Ok(Self { n, p })
    }

    /// QR-based projection onto the Stiefel manifold.
    fn qr_projection<T: Scalar>(&self, matrix: &DMatrix<T>) -> DMatrix<T> {
        // Use QR decomposition to get orthonormal columns
        let qr = matrix.clone().qr();
        let mut q = qr.q();
        
        // Ensure we only take the first p columns
        if q.ncols() > self.p {
            q = q.columns(0, self.p).into_owned();
        }
        
        q
    }

    /// Cayley transform for retraction.
    #[allow(dead_code)]
    fn cayley_transform<T: Scalar>(
        &self,
        x: &DMatrix<T>,
        v: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<DMatrix<T>> {
        // Cayley retraction: R_X(V) = X + V(I + 0.5 * X^T V)^{-1}
        let mut x_t_v = workspace.acquire_temp_matrix(self.p, self.p);
        x_t_v.copy_from(&(x.transpose() * v));
        
        // I + 0.5 * X^T V
        let mut eye_plus = workspace.acquire_temp_matrix(self.p, self.p);
        eye_plus.copy_from(&DMatrix::<T>::identity(self.p, self.p));
        // eye_plus += 0.5 * x_t_v
        let scaled_x_t_v = &*x_t_v * <T as Scalar>::from_f64(0.5);
        let identity = DMatrix::<T>::identity(self.p, self.p);
        let temp = &identity + &scaled_x_t_v;
        eye_plus.copy_from(&temp);
        
        // Solve (I + 0.5 * X^T V) * Y = I to get Y = (I + 0.5 * X^T V)^{-1}
        let inv = eye_plus.clone().try_inverse()
            .ok_or_else(|| ManifoldError::numerical_error("Cayley transform singular"))?;
        
        // Result = X + V * (I + 0.5 * X^T V)^{-1}
        let mut result = x.clone();
        result.gemm(T::one(), v, &inv, T::one());
        
        Ok(result)
    }
}

impl<T: Scalar> MatrixManifold<T> for StiefelMatrix {
    fn name(&self) -> &str {
        "StiefelMatrix"
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.p
    }

    fn dimension(&self) -> usize {
        self.n * self.p - self.p * (self.p + 1) / 2
    }

    fn is_point_on_manifold(&self, point: &DMatrix<T>, tolerance: T) -> bool {
        if point.nrows() != self.n || point.ncols() != self.p {
            return false;
        }
        
        // Check X^T X = I_p
        let x_t_x = point.transpose() * point;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        let diff = &x_t_x - &identity;
        
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
        
        // Check X^T V + V^T X = 0 (skew-symmetric)
        let x_t_v = point.transpose() * vector;
        let v_t_x = vector.transpose() * point;
        let sum = &x_t_v + &v_t_x;
        
        sum.norm() < tolerance
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
        // Tangent space projection: V - X(X^T V + V^T X)/2
        let mut x_t_v = workspace.acquire_temp_matrix(self.p, self.p);
        x_t_v.copy_from(&(point.transpose() * vector));
        
        let mut v_t_x = workspace.acquire_temp_matrix(self.p, self.p);
        v_t_x.copy_from(&(vector.transpose() * point));
        
        let mut sym_part = workspace.acquire_temp_matrix(self.p, self.p);
        sym_part.copy_from(&(&*x_t_v + &*v_t_x));
        let temp = &*sym_part * <T as Scalar>::from_f64(0.5);
        sym_part.copy_from(&temp);
        
        result.copy_from(vector);
        result.gemm(-T::one(), point, &*sym_part, T::one());
        
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
        // QR retraction: R_X(V) = qf(X + V)
        let mut x_plus_v = workspace.acquire_temp_matrix(self.n, self.p);
        x_plus_v.copy_from(&(point + tangent));
        
        let qr = x_plus_v.clone().qr();
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
        // Approximate inverse retraction: project Y - X to tangent space
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
        // For canonical metric, Riemannian gradient = tangent space projection
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
        
        // Project to Stiefel via QR
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
        
        // Project to tangent space
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
        // Simple parallel transport: project vector to tangent space at destination
        // This is not the exact parallel transport but a reasonable approximation
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
        false // QR retraction is not exact exponential map
    }
}

impl<T: Scalar> MatrixManifoldExt<T> for StiefelMatrix {
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
impl_manifold_for_matrix_manifold!(StiefelMatrix);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stiefel_matrix_creation() {
        let stiefel = StiefelMatrix::new(5, 3).unwrap();
        assert_eq!(<StiefelMatrix as MatrixManifold<f64>>::nrows(&stiefel), 5);
        assert_eq!(<StiefelMatrix as MatrixManifold<f64>>::ncols(&stiefel), 3);
        assert_eq!(<StiefelMatrix as MatrixManifold<f64>>::dimension(&stiefel), 9); // 5*3 - 3*4/2 = 15 - 6 = 9
        
        // Error cases
        assert!(StiefelMatrix::new(3, 5).is_err()); // n < p
        assert!(StiefelMatrix::new(5, 0).is_err()); // p = 0
    }

    #[test]
    fn test_point_on_manifold() {
        let stiefel = StiefelMatrix::new(4, 2).unwrap();
        
        // Create orthonormal matrix
        let point = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]);
        
        assert!(stiefel.is_point_on_manifold(&point, 1e-10));
        
        // Non-orthonormal matrix
        let bad_point = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
        ]);
        
        assert!(!stiefel.is_point_on_manifold(&bad_point, 1e-10));
    }

    #[test]
    fn test_projection() {
        let stiefel = StiefelMatrix::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        // Random matrix
        let matrix = DMatrix::from_column_slice(3, 2, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        
        let mut projected = DMatrix::zeros(3, 2);
        stiefel.project_point(&matrix, &mut projected, &mut workspace);
        
        // Check that result is on manifold
        assert!(stiefel.is_point_on_manifold(&projected, 1e-10));
    }

    #[test]
    fn test_tangent_projection() {
        let stiefel = StiefelMatrix::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let point = stiefel.random_point();
        let vector = DMatrix::from_column_slice(3, 2, &[
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
        ]);
        
        let mut tangent = DMatrix::zeros(3, 2);
        stiefel.project_tangent(&point, &vector, &mut tangent, &mut workspace).unwrap();
        
        // Check tangent space condition
        assert!(stiefel.is_vector_in_tangent_space(&point, &tangent, 1e-10));
    }

    #[test]
    fn test_retraction() {
        let stiefel = StiefelMatrix::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let point = stiefel.random_point();
        let mut tangent = DMatrix::zeros(3, 2);
        stiefel.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        
        // Scale tangent for small step
        tangent *= 0.1;
        
        let mut new_point = DMatrix::zeros(3, 2);
        stiefel.retract(&point, &tangent, &mut new_point, &mut workspace).unwrap();
        
        // Check result is on manifold
        assert!(stiefel.is_point_on_manifold(&new_point, 1e-10));
    }

    #[test]
    fn test_vector_conversion() {
        let stiefel = StiefelMatrix::new(3, 2).unwrap();
        
        let matrix = DMatrix::from_column_slice(3, 2, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        
        let vec = stiefel.matrix_to_vector(&matrix);
        let matrix_back = stiefel.vector_to_matrix(&vec);
        
        assert_eq!(matrix, matrix_back);
    }
}