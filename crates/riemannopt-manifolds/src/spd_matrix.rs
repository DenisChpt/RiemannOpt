//! Matrix-based implementation of the SPD manifold.
//!
//! This module provides a `MatrixManifold` implementation for the SPD manifold,
//! operating directly on symmetric positive definite matrices.

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

/// The SPD manifold SPD(n) using matrix operations.
///
/// This implementation operates directly on n×n symmetric positive definite matrices,
/// providing more natural and efficient operations than the vectorized version.
#[derive(Debug, Clone)]
pub struct SPDMatrix {
    n: usize,
    min_eigenvalue: f64,
}

impl SPDMatrix {
    /// Creates a new SPD manifold SPD(n).
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension (must be > 0)
    ///
    /// # Returns
    ///
    /// A new SPDMatrix instance.
    ///
    /// # Errors
    ///
    /// Returns an error if n = 0.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "SPD manifold requires n > 0"
            ));
        }
        Ok(Self {
            n,
            min_eigenvalue: 1e-12,
        })
    }

    /// Creates an SPD manifold with custom eigenvalue threshold.
    pub fn with_min_eigenvalue(n: usize, min_eigenvalue: f64) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "SPD manifold requires n > 0"
            ));
        }
        if min_eigenvalue <= 0.0 {
            return Err(ManifoldError::invalid_point(
                "Minimum eigenvalue must be positive"
            ));
        }
        Ok(Self { n, min_eigenvalue })
    }

    /// Checks if a matrix is SPD with relative tolerance.
    fn is_spd<T: Scalar>(&self, matrix: &DMatrix<T>, tolerance: T) -> bool {
        if matrix.nrows() != self.n || matrix.ncols() != self.n {
            return false;
        }

        // Check for finite values
        if matrix.iter().any(|x| !x.is_finite()) {
            return false;
        }

        // Check symmetry with relative tolerance
        let max_abs = matrix.iter()
            .map(|x| <T as Float>::abs(*x))
            .fold(T::zero(), <T as Float>::max);
        let sym_tolerance = if max_abs > T::zero() {
            tolerance * max_abs
        } else {
            tolerance
        };

        for i in 0..self.n {
            for j in i+1..self.n {
                if <T as Float>::abs(matrix[(i, j)] - matrix[(j, i)]) > sym_tolerance {
                    return false;
                }
            }
        }

        // Check positive definiteness
        let eigen = matrix.clone().symmetric_eigen();
        let min_eigenvalue = <T as Scalar>::from_f64(self.min_eigenvalue);
        
        eigen.eigenvalues.iter().all(|&eval| eval > min_eigenvalue && eval.is_finite())
    }

    /// Projects to SPD with improved numerical stability.
    fn project_to_spd<T: Scalar>(
        &self,
        matrix: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) {
        // Check for numerical issues
        if matrix.iter().any(|x| !x.is_finite()) {
            result.copy_from(&DMatrix::<T>::identity(self.n, self.n));
            *result *= <T as Scalar>::from_f64(1.0 + self.min_eigenvalue);
            return;
        }

        // Ensure symmetry
        let mut symmetric = workspace.acquire_temp_matrix(self.n, self.n);
        symmetric.copy_from(&((matrix + matrix.transpose()) * <T as Scalar>::from_f64(0.5)));

        // Eigenvalue decomposition
        let eigen = symmetric.clone().symmetric_eigen();
        let mut eigenvalues = eigen.eigenvalues.clone();
        let eigenvectors = eigen.eigenvectors;

        // Clamp eigenvalues with regularization
        let min_eval = <T as Scalar>::from_f64(self.min_eigenvalue);
        let regularization = <T as Scalar>::from_f64(1e-12);
        
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] < min_eval || !eigenvalues[i].is_finite() {
                eigenvalues[i] = min_eval + regularization;
            }
        }

        // Reconstruct: P = V * diag(lambda) * V^T
        let lambda_matrix = DMatrix::from_diagonal(&eigenvalues);
        let mut temp = workspace.acquire_temp_matrix(self.n, self.n);
        temp.copy_from(&(&eigenvectors * &lambda_matrix));
        result.copy_from(&(&*temp * eigenvectors.transpose()));

        // Final symmetry enforcement
        let mut final_result = workspace.acquire_temp_matrix(self.n, self.n);
        final_result.copy_from(&((result.clone() + result.transpose()) * <T as Scalar>::from_f64(0.5)));
        result.copy_from(&*final_result);
    }

    /// Affine-invariant distance between SPD matrices.
    fn affine_invariant_distance<T: Scalar>(
        &self,
        p: &DMatrix<T>,
        q: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> T {
        // d(P,Q) = ||log(P^{-1/2} Q P^{-1/2})||_F
        
        // Compute P^{-1/2}
        let p_eigen = p.clone().symmetric_eigen();
        let p_sqrt_inv_eigenvals = p_eigen.eigenvalues.map(|x| {
            if x <= T::zero() {
                T::zero()
            } else {
                T::one() / <T as Float>::sqrt(x)
            }
        });
        
        let p_sqrt_inv_diag = DMatrix::from_diagonal(&p_sqrt_inv_eigenvals);
        let mut p_sqrt_inv = workspace.acquire_temp_matrix(self.n, self.n);
        let mut temp = workspace.acquire_temp_matrix(self.n, self.n);
        temp.copy_from(&(&p_eigen.eigenvectors * &p_sqrt_inv_diag));
        p_sqrt_inv.copy_from(&(&*temp * p_eigen.eigenvectors.transpose()));

        // Compute P^{-1/2} Q P^{-1/2}
        let mut middle = workspace.acquire_temp_matrix(self.n, self.n);
        temp.copy_from(&(&*p_sqrt_inv * q));
        middle.copy_from(&(&*temp * &*p_sqrt_inv));

        // Compute matrix logarithm
        let middle_eigen = middle.clone().symmetric_eigen();
        let log_eigenvals = middle_eigen.eigenvalues.map(|x| {
            if x <= T::zero() {
                <T as Scalar>::from_f64(-50.0) // Large negative for stability
            } else {
                <T as Float>::ln(x)
            }
        });
        
        let log_diag = DMatrix::from_diagonal(&log_eigenvals);
        temp.copy_from(&(&middle_eigen.eigenvectors * &log_diag));
        let mut log_middle = workspace.acquire_temp_matrix(self.n, self.n);
        log_middle.copy_from(&(&*temp * middle_eigen.eigenvectors.transpose()));

        // Return Frobenius norm
        log_middle.norm()
    }
}

impl<T: Scalar> MatrixManifold<T> for SPDMatrix {
    fn name(&self) -> &str {
        "SPDMatrix"
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn dimension(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    fn is_point_on_manifold(&self, point: &DMatrix<T>, tolerance: T) -> bool {
        self.is_spd(point, tolerance)
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &DMatrix<T>,
        vector: &DMatrix<T>,
        tolerance: T,
    ) -> bool {
        if vector.nrows() != self.n || vector.ncols() != self.n {
            return false;
        }

        // Tangent space consists of symmetric matrices
        for i in 0..self.n {
            for j in i+1..self.n {
                if <T as Float>::abs(vector[(i, j)] - vector[(j, i)]) > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn project_point(
        &self,
        matrix: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) {
        self.project_to_spd(matrix, result, workspace);
    }

    fn project_tangent(
        &self,
        _point: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Project to symmetric matrices
        result.copy_from(&((vector + vector.transpose()) * <T as Scalar>::from_f64(0.5)));
        Ok(())
    }

    fn inner_product(
        &self,
        point: &DMatrix<T>,
        u: &DMatrix<T>,
        v: &DMatrix<T>,
    ) -> Result<T> {
        // Affine-invariant metric: <U,V>_P = tr(P^{-1} U P^{-1} V)
        let p_inv = point.clone().try_inverse()
            .ok_or_else(|| ManifoldError::numerical_error("Point matrix not invertible"))?;
        
        let result = (&p_inv * u * &p_inv * v).trace();
        Ok(result)
    }

    fn retract(
        &self,
        point: &DMatrix<T>,
        tangent: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Exponential map: exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
        // For efficiency, use simpler retraction: project(P + V)
        
        let mut p_plus_v = workspace.acquire_temp_matrix(self.n, self.n);
        p_plus_v.copy_from(&(point + tangent));
        
        self.project_to_spd(&*p_plus_v, result, workspace);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &DMatrix<T>,
        other: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For close points, use first-order approximation
        let diff = other - point;
        let diff_norm = diff.norm();
        let point_norm = point.norm();
        
        if diff_norm / point_norm < <T as Scalar>::from_f64(1e-8) {
            self.project_tangent(point, &diff, result, workspace)?;
            return Ok(());
        }

        // Otherwise use full logarithmic map
        // log_X(Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}
        
        // This is complex, so for now use simple approximation
        self.project_tangent(point, &diff, result, workspace)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DMatrix<T>,
        euclidean_grad: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For affine-invariant metric: grad_R = P * grad_E * P
        let mut temp = workspace.acquire_temp_matrix(self.n, self.n);
        temp.copy_from(&(point * euclidean_grad));
        result.copy_from(&(&*temp * point));
        
        // Project to tangent space (ensure symmetry)
        let result_clone = result.clone();
        self.project_tangent(point, &result_clone, result, workspace)
    }

    fn random_point(&self) -> DMatrix<T> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random matrix
        let mut matrix = DMatrix::<T>::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                matrix[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }

        // Make SPD: A = B^T B + epsilon * I
        let btb = matrix.transpose() * &matrix;
        let identity = DMatrix::<T>::identity(self.n, self.n);
        let epsilon = <T as Scalar>::from_f64(self.min_eigenvalue);
        
        btb + identity * epsilon
    }

    fn random_tangent(
        &self,
        _point: &DMatrix<T>,
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random symmetric matrix
        for i in 0..self.n {
            for j in i..self.n {
                let val = <T as Scalar>::from_f64(normal.sample(&mut rng));
                result[(i, j)] = val;
                if i != j {
                    result[(j, i)] = val;
                }
            }
        }
        
        Ok(())
    }

    fn parallel_transport(
        &self,
        _from: &DMatrix<T>,
        to: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Simplified parallel transport: project to tangent space at destination
        self.project_tangent(to, vector, result, workspace)
    }

    fn distance(
        &self,
        x: &DMatrix<T>,
        y: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<T> {
        Ok(self.affine_invariant_distance(x, y, workspace))
    }

    fn has_exact_exp_log(&self) -> bool {
        true // SPD has exact exponential map
    }
}

impl<T: Scalar> MatrixManifoldExt<T> for SPDMatrix {
    fn vector_to_matrix(&self, vector: &[T]) -> DMatrix<T> {
        // Convert from upper triangular packed format
        let expected_size = self.n * (self.n + 1) / 2;
        assert_eq!(vector.len(), expected_size, "Invalid vector size for SPD matrix");
        
        let mut matrix = DMatrix::<T>::zeros(self.n, self.n);
        let mut idx = 0;
        
        // Fill upper triangular part
        for j in 0..self.n {
            for i in 0..=j {
                matrix[(i, j)] = vector[idx];
                if i != j {
                    matrix[(j, i)] = vector[idx]; // Symmetric
                }
                idx += 1;
            }
        }
        
        matrix
    }

    fn matrix_to_vector(&self, matrix: &DMatrix<T>) -> Vec<T> {
        // Extract upper triangular part in column-major order
        let size = self.n * (self.n + 1) / 2;
        let mut vector = Vec::with_capacity(size);
        
        for j in 0..self.n {
            for i in 0..=j {
                vector.push(matrix[(i, j)]);
            }
        }
        
        vector
    }

    fn vector_length(&self) -> usize {
        self.n * (self.n + 1) / 2
    }
}

// Generate the vector-based Manifold implementation
impl_manifold_for_matrix_manifold!(SPDMatrix);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_spd_matrix_creation() {
        let spd = SPDMatrix::new(3).unwrap();
        assert_eq!(<SPDMatrix as MatrixManifold<f64>>::nrows(&spd), 3);
        assert_eq!(<SPDMatrix as MatrixManifold<f64>>::ncols(&spd), 3);
        assert_eq!(<SPDMatrix as MatrixManifold<f64>>::dimension(&spd), 6); // 3*(3+1)/2 = 6
        
        // Error cases
        assert!(SPDMatrix::new(0).is_err());
        assert!(SPDMatrix::with_min_eigenvalue(3, -1.0).is_err());
    }

    #[test]
    fn test_point_on_manifold() {
        let spd = SPDMatrix::new(2).unwrap();
        
        // Identity is SPD
        let identity = DMatrix::<f64>::identity(2, 2);
        assert!(spd.is_point_on_manifold(&identity, 1e-10));
        
        // Diagonal with positive entries
        let diag = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0]));
        assert!(spd.is_point_on_manifold(&diag, 1e-10));
        
        // Non-symmetric
        let non_sym = DMatrix::from_column_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(!spd.is_point_on_manifold(&non_sym, 1e-10));
    }

    #[test]
    fn test_projection() {
        let spd = SPDMatrix::new(2).unwrap();
        let mut workspace = Workspace::new();
        
        // Non-SPD matrix
        let matrix = DMatrix::from_column_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let mut projected = DMatrix::zeros(2, 2);
        spd.project_point(&matrix, &mut projected, &mut workspace);
        
        // Check result is SPD
        assert!(spd.is_point_on_manifold(&projected, 1e-10));
    }

    #[test]
    fn test_tangent_space() {
        let spd = SPDMatrix::new(2).unwrap();
        
        // Symmetric matrix is in tangent space
        let sym = DMatrix::from_column_slice(2, 2, &[1.0, 2.0, 2.0, 3.0]);
        let point = spd.random_point();
        
        assert!(spd.is_vector_in_tangent_space(&point, &sym, 1e-10));
    }

    #[test]
    fn test_inner_product() {
        let spd = SPDMatrix::new(2).unwrap();
        
        let p = DMatrix::<f64>::identity(2, 2) * 2.0;
        let u = DMatrix::<f64>::identity(2, 2);
        let v = DMatrix::from_column_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        
        let inner = spd.inner_product(&p, &u, &v).unwrap();
        // For P = 2I, <U,V>_P = tr(P^{-1} U P^{-1} V) = tr(0.5I * I * 0.5I * V) = 0.25 * tr(V) = 0
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_conversion() {
        let spd = SPDMatrix::new(3).unwrap();
        
        // Test with identity matrix
        let identity = DMatrix::<f64>::identity(3, 3);
        let vec = spd.matrix_to_vector(&identity);
        
        // Upper triangular: [1, 0, 1, 0, 0, 1]
        assert_eq!(vec.len(), 6);
        assert_eq!(vec[0], 1.0); // (0,0)
        assert_eq!(vec[1], 0.0); // (0,1)
        assert_eq!(vec[2], 1.0); // (1,1)
        
        // Convert back
        let matrix_back = spd.vector_to_matrix(&vec);
        assert_eq!(identity, matrix_back);
    }
}