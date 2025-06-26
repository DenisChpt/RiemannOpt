//! Matrix-based implementation of the Oblique manifold.
//!
//! This module provides a `MatrixManifold` implementation for the Oblique manifold,
//! operating directly on matrices whose columns have unit norm.

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

/// The Oblique manifold OB(n,p) using matrix operations.
///
/// The Oblique manifold is the product of p unit spheres in R^n, represented
/// as n×p matrices where each column has unit norm.
#[derive(Debug, Clone)]
pub struct ObliqueMatrix {
    n: usize,
    p: usize,
}

impl ObliqueMatrix {
    /// Creates a new Oblique manifold OB(n,p).
    ///
    /// # Arguments
    ///
    /// * `n` - Dimension of each sphere (must be > 0)
    /// * `p` - Number of spheres/columns (must be > 0)
    ///
    /// # Returns
    ///
    /// A new ObliqueMatrix instance.
    ///
    /// # Errors
    ///
    /// Returns an error if n = 0 or p = 0.
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if n == 0 || p == 0 {
            return Err(ManifoldError::invalid_point(
                "Oblique manifold requires n > 0 and p > 0"
            ));
        }
        Ok(Self { n, p })
    }

    /// Normalizes columns of a matrix.
    fn normalize_columns<T: Scalar>(
        &self,
        matrix: &DMatrix<T>,
        result: &mut DMatrix<T>,
    ) {
        result.copy_from(matrix);
        
        for j in 0..self.p {
            let col_norm = result.column(j).norm();
            if col_norm > T::zero() {
                let mut col_mut = result.column_mut(j);
                col_mut /= col_norm;
            } else {
                // Handle zero columns by setting to e_1
                let mut col_mut = result.column_mut(j);
                col_mut.fill(T::zero());
                if self.n > 0 {
                    col_mut[0] = T::one();
                }
            }
        }
    }
}

impl<T: Scalar> MatrixManifold<T> for ObliqueMatrix {
    fn name(&self) -> &str {
        "ObliqueMatrix"
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.p
    }

    fn dimension(&self) -> usize {
        self.p * (self.n - 1)
    }

    fn is_point_on_manifold(&self, point: &DMatrix<T>, tolerance: T) -> bool {
        if point.nrows() != self.n || point.ncols() != self.p {
            return false;
        }
        
        // Check each column has unit norm
        for j in 0..self.p {
            let col_norm = point.column(j).norm();
            if <T as Float>::abs(col_norm - T::one()) > tolerance {
                return false;
            }
        }
        
        true
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
        
        // For each column: x_j^T v_j = 0
        for j in 0..self.p {
            let x_col = point.column(j);
            let v_col = vector.column(j);
            let inner = x_col.dot(&v_col);
            
            if <T as Float>::abs(inner) > tolerance {
                return false;
            }
        }
        
        true
    }

    fn project_point(
        &self,
        matrix: &DMatrix<T>,
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) {
        self.normalize_columns(matrix, result);
    }

    fn project_tangent(
        &self,
        point: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        result.copy_from(vector);
        
        // For each column: v_j - (x_j^T v_j) x_j
        for j in 0..self.p {
            let x_col = point.column(j);
            let v_col = vector.column(j);
            let inner = x_col.dot(&v_col);
            
            let mut result_col = result.column_mut(j);
            result_col.axpy(-inner, &x_col, T::one());
        }
        
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &DMatrix<T>,
        u: &DMatrix<T>,
        v: &DMatrix<T>,
    ) -> Result<T> {
        // Sum of column-wise inner products
        let mut total = T::zero();
        for j in 0..self.p {
            total += u.column(j).dot(&v.column(j));
        }
        Ok(total)
    }

    fn retract(
        &self,
        point: &DMatrix<T>,
        tangent: &DMatrix<T>,
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Normalize each column of (X + V)
        result.copy_from(&(point + tangent));
        
        for j in 0..self.p {
            let col_norm = result.column(j).norm();
            if col_norm > T::zero() {
                let mut col_mut = result.column_mut(j);
                col_mut /= col_norm;
            }
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
        result.fill(T::zero());
        
        // For each column, compute logarithmic map on sphere
        for j in 0..self.p {
            let x_col = point.column(j);
            let y_col = other.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = Float::min(Float::max(inner, -T::one()), T::one());
            
            if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-10) {
                // Points are identical
                continue;
            }
            
            let theta = Float::acos(clamped);
            let sin_theta = Float::sin(theta);
            
            if sin_theta > <T as Scalar>::from_f64(1e-10) {
                // v = theta / sin(theta) * (y - cos(theta) * x)
                let scale = theta / sin_theta;
                let mut result_col = result.column_mut(j);
                result_col.copy_from(&y_col);
                result_col.axpy(-clamped, &x_col, T::one());
                result_col *= scale;
            }
        }
        
        // Ensure result is in tangent space
        let result_clone = result.clone();
        self.project_tangent(point, &result_clone, result, workspace)?;
        
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DMatrix<T>,
        euclidean_grad: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Project to tangent space
        self.project_tangent(point, euclidean_grad, result, workspace)
    }

    fn random_point(&self) -> DMatrix<T> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        let mut matrix = DMatrix::<T>::zeros(self.n, self.p);
        
        // Generate random columns and normalize
        for j in 0..self.p {
            for i in 0..self.n {
                matrix[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            
            let col_norm = matrix.column(j).norm();
            if col_norm > T::zero() {
                let mut col_mut = matrix.column_mut(j);
                col_mut /= col_norm;
            }
        }
        
        matrix
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
        for i in 0..self.n {
            for j in 0..self.p {
                result[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // Project to tangent space
        let result_clone = result.clone();
        self.project_tangent(point, &result_clone, result, workspace)?;
        
        Ok(())
    }

    fn parallel_transport(
        &self,
        from: &DMatrix<T>,
        to: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        result.fill(T::zero());
        
        // Parallel transport each column independently
        for j in 0..self.p {
            let x_col = from.column(j);
            let y_col = to.column(j);
            let v_col = vector.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = Float::min(Float::max(inner, -T::one()), T::one());
            
            if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-10) {
                // Same point, no transport needed
                result.column_mut(j).copy_from(&v_col);
                continue;
            }
            
            if <T as Float>::abs(clamped + T::one()) < <T as Scalar>::from_f64(1e-10) {
                // Antipodal points
                result.column_mut(j).copy_from(&v_col);
                continue;
            }
            
            // Compute transported vector
            let theta = Float::acos(clamped);
            let sin_theta = Float::sin(theta);
            
            // Tangent direction at x towards y
            let mut xi = y_col.clone_owned();
            xi.axpy(-clamped, &x_col, T::one());
            xi /= sin_theta;
            
            // Transport formula
            let v_xi_inner = v_col.dot(&xi);
            let mut result_col = result.column_mut(j);
            result_col.copy_from(&v_col);
            result_col.axpy(
                -sin_theta * v_xi_inner,
                &x_col,
                T::one()
            );
            result_col.axpy(
                (T::one() - Float::cos(theta)) * v_xi_inner,
                &xi,
                T::one()
            );
        }
        
        Ok(())
    }

    fn distance(
        &self,
        x: &DMatrix<T>,
        y: &DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<T> {
        let mut dist_squared = T::zero();
        
        // Sum of squared distances on each sphere
        for j in 0..self.p {
            let x_col = x.column(j);
            let y_col = y.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = Float::min(Float::max(inner, -T::one()), T::one());
            let angle = Float::acos(clamped);
            
            dist_squared += angle * angle;
        }
        
        Ok(Float::sqrt(dist_squared))
    }

    fn has_exact_exp_log(&self) -> bool {
        true // Oblique has exact exp/log on each sphere
    }
}

impl<T: Scalar> MatrixManifoldExt<T> for ObliqueMatrix {
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
impl_manifold_for_matrix_manifold!(ObliqueMatrix);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_oblique_matrix_creation() {
        let oblique = ObliqueMatrix::new(3, 4).unwrap();
        assert_eq!(<ObliqueMatrix as MatrixManifold<f64>>::nrows(&oblique), 3);
        assert_eq!(<ObliqueMatrix as MatrixManifold<f64>>::ncols(&oblique), 4);
        assert_eq!(<ObliqueMatrix as MatrixManifold<f64>>::dimension(&oblique), 8); // 4*(3-1) = 8
        
        // Error cases
        assert!(ObliqueMatrix::new(0, 4).is_err());
        assert!(ObliqueMatrix::new(3, 0).is_err());
    }

    #[test]
    fn test_point_on_manifold() {
        let oblique = ObliqueMatrix::new(3, 2).unwrap();
        
        // Create matrix with unit norm columns
        let point = DMatrix::from_column_slice(3, 2, &[
            1.0, 0.0, 0.0,  // First column: [1, 0, 0]
            0.0, 1.0, 0.0,  // Second column: [0, 1, 0]
        ]);
        
        assert!(oblique.is_point_on_manifold(&point, 1e-10));
        
        // Non-unit column
        let bad_point = DMatrix::from_column_slice(3, 2, &[
            2.0, 0.0, 0.0,  // Norm = 2
            0.0, 1.0, 0.0,
        ]);
        
        assert!(!oblique.is_point_on_manifold(&bad_point, 1e-10));
    }

    #[test]
    fn test_projection() {
        let oblique = ObliqueMatrix::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let matrix = DMatrix::from_column_slice(3, 2, &[
            3.0, 0.0, 0.0,
            0.0, 4.0, 0.0,
        ]);
        
        let mut projected = DMatrix::zeros(3, 2);
        oblique.project_point(&matrix, &mut projected, &mut workspace);
        
        assert!(oblique.is_point_on_manifold(&projected, 1e-10));
        
        // Check columns are normalized versions of input
        assert_relative_eq!(projected.column(0).norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(projected.column(1).norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_space() {
        let oblique = ObliqueMatrix::new(3, 2).unwrap();
        
        // Create tangent vector
        let tangent = DMatrix::from_column_slice(3, 2, &[
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
        ]);
        
        // For first column of point = [1,0,0], tangent [0,1,0] is valid
        // For second column of point = [0,1,0], tangent [-1,0,0] is valid
        let test_point = DMatrix::from_column_slice(3, 2, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);
        
        assert!(oblique.is_vector_in_tangent_space(&test_point, &tangent, 1e-10));
    }

    #[test]
    fn test_retraction() {
        let oblique = ObliqueMatrix::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let point = <ObliqueMatrix as MatrixManifold<f64>>::random_point(&oblique);
        let mut tangent = DMatrix::zeros(3, 2);
        oblique.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        
        // Scale for small step
        tangent *= 0.1;
        
        let mut new_point = DMatrix::zeros(3, 2);
        oblique.retract(&point, &tangent, &mut new_point, &mut workspace).unwrap();
        
        assert!(oblique.is_point_on_manifold(&new_point, 1e-10));
    }

    #[test]
    fn test_distance() {
        let oblique = ObliqueMatrix::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let x = DMatrix::from_column_slice(3, 2, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);
        
        let y = DMatrix::from_column_slice(3, 2, &[
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ]);
        
        let dist = oblique.distance(&x, &y, &mut workspace).unwrap();
        
        // Distance should be sqrt(2) * pi/2 (90 degrees on each sphere)
        let expected = (2.0_f64).sqrt() * std::f64::consts::PI / 2.0;
        assert_relative_eq!(dist, expected, epsilon = 1e-10);
    }
}