//! Positive Semi-Definite cone manifold
//!
//! The manifold of symmetric positive semi-definite matrices.

use nalgebra::{DMatrix, DVector, Dyn};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    types::Scalar,
};

/// The positive semi-definite cone S^n_+ of n×n symmetric PSD matrices.
///
/// # Mathematical Definition
///
/// The PSD cone is defined as:
/// ```text
/// S^n_+ = {X ∈ ℝ^{n×n} : X = X^T, X ⪰ 0}
/// ```
///
/// where X ⪰ 0 means all eigenvalues of X are non-negative.
///
/// # Properties
///
/// - **Dimension**: n(n+1)/2
/// - **Tangent space**: Space of symmetric matrices at interior points
/// - **Metric**: Various metrics available (Euclidean, Log-Euclidean, Bures-Wasserstein)
///
/// # Applications
///
/// - Semidefinite programming
/// - Covariance matrix estimation
/// - Kernel learning
/// - Quantum state tomography
#[derive(Debug, Clone)]
pub struct PSDCone {
    n: usize,
}

impl PSDCone {
    /// Create a new PSD cone manifold.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the matrices (n×n)
    ///
    /// # Errors
    ///
    /// Returns an error if n = 0.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "PSD cone requires n > 0"
            ));
        }
        Ok(Self { n })
    }

    /// Get the size n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Convert a symmetric matrix to vector form (upper triangular part)
    fn mat_to_vec<T: Scalar>(&self, mat: &DMatrix<T>) -> DVector<T> {
        let n = self.n;
        let dim = n * (n + 1) / 2;
        let mut vec = DVector::zeros(dim);
        let mut idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    vec[idx] = mat[(i, j)];
                } else {
                    // Store off-diagonal elements with sqrt(2) scaling for proper inner product
                    vec[idx] = mat[(i, j)] * <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
                }
                idx += 1;
            }
        }
        
        vec
    }

    /// Convert vector to symmetric matrix
    fn vec_to_mat<T: Scalar>(&self, vec: &DVector<T>) -> DMatrix<T> {
        let n = self.n;
        let mut mat = DMatrix::zeros(n, n);
        let mut idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    mat[(i, j)] = vec[idx];
                } else {
                    // Unscale off-diagonal elements
                    let val = vec[idx] / <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
                    mat[(i, j)] = val;
                    mat[(j, i)] = val;
                }
                idx += 1;
            }
        }
        
        mat
    }

    /// Project a matrix to the PSD cone
    fn project_to_psd<T: Scalar>(&self, mat: &DMatrix<T>) -> DMatrix<T> {
        // Symmetrize first
        let sym = (mat.clone() + mat.transpose()) / <T as Scalar>::from_f64(2.0);
        
        // Eigendecomposition
        let eigen = sym.clone().symmetric_eigen();
        let mut eigenvalues = eigen.eigenvalues.clone();
        
        // Project eigenvalues to non-negative
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] < T::zero() {
                eigenvalues[i] = T::zero();
            }
        }
        
        // Reconstruct
        let q = eigen.eigenvectors;
        let d = DMatrix::from_diagonal(&eigenvalues);
        &q * &d * q.transpose()
    }

    /// Check if a matrix is in the PSD cone
    fn is_psd<T: Scalar>(&self, mat: &DMatrix<T>, tol: T) -> bool {
        // Check symmetry
        for i in 0..self.n {
            for j in i+1..self.n {
                if Float::abs(mat[(i, j)] - mat[(j, i)]) > tol {
                    return false;
                }
            }
        }
        
        // Check positive semi-definiteness
        let eigen = mat.clone().symmetric_eigen();
        eigen.eigenvalues.iter().all(|&lambda| lambda >= -tol)
    }
}

impl<T: Scalar> Manifold<T, Dyn> for PSDCone {
    fn name(&self) -> &str {
        "PSDCone"
    }

    fn dimension(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    fn ambient_dimension(&self) -> usize {
        <Self as Manifold<T, Dyn>>::dimension(self)
    }

    fn is_point_on_manifold(&self, point: &Point<T, Dyn>, tol: T) -> bool {
        if point.len() != <Self as Manifold<T, Dyn>>::dimension(self) {
            return false;
        }
        
        let mat = self.vec_to_mat(point);
        self.is_psd(&mat, tol)
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        tol: T,
    ) -> bool {
        if !self.is_point_on_manifold(point, tol) {
            return false;
        }
        
        if vector.len() != <Self as Manifold<T, Dyn>>::dimension(self) {
            return false;
        }
        
        // For interior points, tangent space is all symmetric matrices
        // For boundary points (with zero eigenvalues), it's more restricted
        let mat = self.vec_to_mat(vector);
        
        // Check symmetry
        for i in 0..self.n {
            for j in i+1..self.n {
                if Float::abs(mat[(i, j)] - mat[(j, i)]) > tol {
                    return false;
                }
            }
        }
        
        true
    }

    fn project_point(&self, point: &Point<T, Dyn>, result: &mut Point<T, Dyn>) {
        // Ensure result has correct size
        let expected_size = <Self as Manifold<T, Dyn>>::ambient_dimension(self);
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        let mat = if point.len() == self.n * self.n {
            // If given as full matrix, reshape
            DMatrix::from_vec(self.n, self.n, point.as_slice().to_vec())
        } else {
            self.vec_to_mat(point)
        };
        
        let projected = self.project_to_psd(&mat);
        let projected_vec = self.mat_to_vec(&projected);
        result.copy_from(&projected_vec);
    }

    fn project_tangent(
        &self,
        _point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = <Self as Manifold<T, Dyn>>::ambient_dimension(self);
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        // For PSD cone, tangent projection is just symmetrization
        let mat = if vector.len() == self.n * self.n {
            DMatrix::from_vec(self.n, self.n, vector.as_slice().to_vec())
        } else {
            self.vec_to_mat(vector)
        };
        
        let sym = (mat.clone() + mat.transpose()) / <T as Scalar>::from_f64(2.0);
        let sym_vec = self.mat_to_vec(&sym);
        result.copy_from(&sym_vec);
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Point<T, Dyn>,
        u: &TangentVector<T, Dyn>,
        v: &TangentVector<T, Dyn>,
    ) -> Result<T> {
        // Standard Frobenius inner product
        Ok(u.dot(v))
    }

    fn retract(
        &self,
        point: &Point<T, Dyn>,
        tangent: &TangentVector<T, Dyn>,
        result: &mut Point<T, Dyn>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = <Self as Manifold<T, Dyn>>::ambient_dimension(self);
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        let x_mat = self.vec_to_mat(point);
        let v_mat = self.vec_to_mat(tangent);
        
        // Simple retraction: project(X + V)
        let new_mat = &x_mat + &v_mat;
        let projected = self.project_to_psd(&new_mat);
        
        let projected_vec = self.mat_to_vec(&projected);
        result.copy_from(&projected_vec);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Point<T, Dyn>,
        other: &Point<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = <Self as Manifold<T, Dyn>>::ambient_dimension(self);
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        // Simple approximation: project the difference
        let diff = other - point;
        self.project_tangent(point, &diff, result)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Point<T, Dyn>,
        euclidean_grad: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = <Self as Manifold<T, Dyn>>::ambient_dimension(self);
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        // For the standard metric, just project to tangent space
        self.project_tangent(point, euclidean_grad, result)
    }

    fn random_point(&self) -> Point<T, Dyn> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random symmetric matrix
        let mut mat = DMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in i..self.n {
                let val: f64 = normal.sample(&mut rng);
                mat[(i, j)] = <T as Scalar>::from_f64(val);
                if i != j {
                    mat[(j, i)] = <T as Scalar>::from_f64(val);
                }
            }
        }
        
        // Make it PSD by X = A^T A
        let psd = mat.transpose() * &mat;
        
        // Scale to reasonable size
        let psd_scaled = psd / <T as Scalar>::from_f64(self.n as f64);
        
        self.mat_to_vec(&psd_scaled)
    }

    fn random_tangent(&self, point: &Point<T, Dyn>, result: &mut TangentVector<T, Dyn>) -> Result<()> {
        // Ensure result has correct size
        let expected_size = <Self as Manifold<T, Dyn>>::ambient_dimension(self);
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random symmetric matrix
        let mut mat = DMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in i..self.n {
                let val: f64 = normal.sample(&mut rng);
                mat[(i, j)] = <T as Scalar>::from_f64(val);
                if i != j {
                    mat[(j, i)] = <T as Scalar>::from_f64(val);
                }
            }
        }
        
        let tangent = self.mat_to_vec(&mat);
        self.project_tangent(point, &tangent, result)
    }

    fn distance(&self, x: &Point<T, Dyn>, y: &Point<T, Dyn>) -> Result<T> {
        // Frobenius distance
        let diff = y - x;
        Ok(Float::sqrt(diff.dot(&diff)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_manifold() -> PSDCone {
        PSDCone::new(3).unwrap()
    }

    #[test]
    fn test_psd_cone_creation() {
        let manifold = create_test_manifold();
        assert_eq!(manifold.n(), 3);
        assert_eq!(<PSDCone as Manifold<f64, Dyn>>::dimension(&manifold), 6); // 3*(3+1)/2
    }

    #[test]
    fn test_psd_cone_projection() {
        let manifold = create_test_manifold();
        
        // Create a non-PSD matrix (negative eigenvalue)
        let mat = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.5, 0.3,
            0.5, -1.0, 0.2,
            0.3, 0.2, 0.5
        ]);
        let vec = manifold.mat_to_vec(&mat);
        
        let mut projected = DVector::zeros(<PSDCone as Manifold<f64, Dyn>>::ambient_dimension(&manifold));
        <PSDCone as Manifold<f64, Dyn>>::project_point(&manifold, &vec, &mut projected);
        let proj_mat = manifold.vec_to_mat(&projected);
        
        // Check that projection is PSD
        assert!(manifold.is_psd(&proj_mat, 1e-10));
    }

    #[test]
    fn test_psd_cone_tangent_projection() {
        let manifold = create_test_manifold();
        
        let point = <PSDCone as Manifold<f64, Dyn>>::random_point(&manifold);
        
        // Create a non-symmetric matrix
        let mat = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            0.0, 1.0, 2.0,
            0.0, 0.0, 1.0
        ]);
        let vec = manifold.mat_to_vec(&mat);
        
        let mut tangent = DVector::zeros(<PSDCone as Manifold<f64, Dyn>>::ambient_dimension(&manifold));
        <PSDCone as Manifold<f64, Dyn>>::project_tangent(&manifold, &point, &vec, &mut tangent).unwrap();
        let tan_mat = manifold.vec_to_mat(&tangent);
        
        // Check symmetry
        for i in 0..3 {
            for j in i+1..3 {
                assert_relative_eq!(tan_mat[(i, j)], tan_mat[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_psd_cone_retraction() {
        let manifold = create_test_manifold();
        
        let point = <PSDCone as Manifold<f64, Dyn>>::random_point(&manifold);
        let mut tangent = DVector::zeros(<PSDCone as Manifold<f64, Dyn>>::ambient_dimension(&manifold));
        <PSDCone as Manifold<f64, Dyn>>::random_tangent(&manifold, &point, &mut tangent).unwrap();
        let scaled_tangent = 0.1 * &tangent;
        let mut retracted = DVector::zeros(<PSDCone as Manifold<f64, Dyn>>::ambient_dimension(&manifold));
        <PSDCone as Manifold<f64, Dyn>>::retract(&manifold, &point, &scaled_tangent, &mut retracted).unwrap();
        
        // Check that result is on manifold
        assert!(<PSDCone as Manifold<f64, Dyn>>::is_point_on_manifold(&manifold, &retracted, 1e-6));
    }

    #[test]
    fn test_psd_cone_inner_product() {
        let manifold = create_test_manifold();
        
        let point = <PSDCone as Manifold<f64, Dyn>>::random_point(&manifold);
        let mut u = DVector::zeros(<PSDCone as Manifold<f64, Dyn>>::ambient_dimension(&manifold));
        let mut v = DVector::zeros(<PSDCone as Manifold<f64, Dyn>>::ambient_dimension(&manifold));
        <PSDCone as Manifold<f64, Dyn>>::random_tangent(&manifold, &point, &mut u).unwrap();
        <PSDCone as Manifold<f64, Dyn>>::random_tangent(&manifold, &point, &mut v).unwrap();
        
        let ip_uv = <PSDCone as Manifold<f64, Dyn>>::inner_product(&manifold, &point, &u, &v).unwrap();
        let ip_vu = <PSDCone as Manifold<f64, Dyn>>::inner_product(&manifold, &point, &v, &u).unwrap();
        
        // Check symmetry
        assert_relative_eq!(ip_uv, ip_vu, epsilon = 1e-10);
    }
}