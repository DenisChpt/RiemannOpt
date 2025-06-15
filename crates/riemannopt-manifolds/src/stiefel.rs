//! Stiefel manifold St(n,p) = {X in R^{n x p} : X^T X = I_p}
//!
//! The Stiefel manifold represents the space of n x p orthonormal matrices.
//! It naturally appears in:
//! - Principal Component Analysis (PCA)
//! - Independent Component Analysis (ICA)
//! - Neural network weight constraints
//! - Dimensionality reduction
//! - Orthogonal dictionary learning

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::Scalar,
};
use nalgebra::{DMatrix, DVector, Dyn};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

/// The Stiefel manifold St(n,p) of n x p orthonormal matrices.
///
/// This manifold represents all n x p matrices X such that X^T X = I_p,
/// where I_p is the p x p identity matrix. The tangent space at X consists
/// of all n x p matrices V such that X^T V + V^T X = 0.
///
/// # Mathematical Properties
///
/// - **Dimension**: np - p(p+1)/2
/// - **Tangent space**: T_X St(n,p) = {V in R^{n x p} : X^T V + V^T X = 0}
/// - **Riemannian metric**: Inherited from Euclidean space (canonical metric)
/// - **Retractions**: QR decomposition, polar decomposition, Cayley transform
///
/// # Applications
///
/// - **PCA**: Finding orthonormal principal components
/// - **Neural networks**: Orthogonal weight constraints
/// - **Computer vision**: Orthonormal frame estimation
/// - **Signal processing**: Orthogonal basis learning
#[derive(Debug, Clone)]
pub struct Stiefel {
    /// Number of rows (n)
    n: usize,
    /// Number of columns (p)
    p: usize,
}

impl Stiefel {
    /// Creates a new Stiefel manifold St(n,p).
    ///
    /// # Arguments
    /// * `n` - Number of rows (ambient dimension)
    /// * `p` - Number of columns (must be d n)
    ///
    /// # Returns
    /// A Stiefel manifold with intrinsic dimension np - p(p+1)/2
    ///
    /// # Errors
    /// Returns an error if p > n or either dimension is 0
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if n == 0 || p == 0 {
            return Err(ManifoldError::invalid_point(
                "Stiefel manifold requires n > 0 and p > 0",
            ));
        }
        if p > n {
            return Err(ManifoldError::invalid_point(
                "Stiefel manifold requires p <= n",
            ));
        }
        Ok(Self { n, p })
    }

    /// Returns the number of rows (n)
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of columns (p)
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the ambient dimensions (n, p)
    pub fn ambient_dimensions(&self) -> (usize, usize) {
        (self.n, self.p)
    }

    /// Checks if a matrix satisfies the orthonormality constraint X^T X = I
    fn is_orthonormal<T>(&self, matrix: &DMatrix<T>, tolerance: T) -> bool
    where
        T: Scalar,
    {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            return false;
        }

        let gram = matrix.transpose() * matrix;
        // Check gram matrix against identity
        
        for i in 0..self.p {
            for j in 0..self.p {
                let expected = if i == j { T::one() } else { T::zero() };
                if <T as Float>::abs(gram[(i, j)] - expected) > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Projects a matrix to the Stiefel manifold using QR decomposition
    fn qr_projection<T>(&self, matrix: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            // Handle wrong dimensions by padding or truncating
            let mut result = DMatrix::<T>::zeros(self.n, self.p);
            let copy_rows = matrix.nrows().min(self.n);
            let copy_cols = matrix.ncols().min(self.p);
            
            for i in 0..copy_rows {
                for j in 0..copy_cols {
                    result[(i, j)] = matrix[(i, j)];
                }
            }
            
            // If we have zero columns, fill with random data
            if result.column(0).norm() < T::epsilon() {
                result[(0, 0)] = T::one();
            }
            
            let qr = result.qr();
            // Take only the first p columns of Q
            qr.q().columns(0, self.p).into_owned()
        } else {
            let qr = matrix.clone().qr();
            // Take only the first p columns of Q
            qr.q().columns(0, self.p).into_owned()
        }
    }

    /// Generates a random tangent vector at the given point
    fn random_tangent_matrix<T>(&self, point: &DMatrix<T>) -> Result<DMatrix<T>>
    where
        T: Scalar,
    {
        let mut rng = rand::thread_rng();
        
        // Generate random matrix
        let mut random_matrix = DMatrix::<T>::zeros(self.n, self.p);
        for i in 0..self.n {
            for j in 0..self.p {
                let val: f64 = StandardNormal.sample(&mut rng);
                random_matrix[(i, j)] = <T as Scalar>::from_f64(val);
            }
        }
        
        // Project to tangent space: V - X(X^T V + V^T X)/2
        let xtv = point.transpose() * &random_matrix;
        let vtx = random_matrix.transpose() * point;
        let symmetric = (&xtv + &vtx) * <T as Scalar>::from_f64(0.5);
        
        Ok(random_matrix - point * symmetric)
    }
}

impl<T> Manifold<T, Dyn> for Stiefel
where
    T: Scalar,
{
    fn name(&self) -> &str {
        "Stiefel"
    }

    fn dimension(&self) -> usize {
        self.n * self.p - self.p * (self.p + 1) / 2
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tolerance: T) -> bool {
        // Reshape vector to matrix
        if point.len() != self.n * self.p {
            return false;
        }
        
        let matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        self.is_orthonormal(&matrix, tolerance)
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        tolerance: T,
    ) -> bool {
        if point.len() != self.n * self.p || vector.len() != self.n * self.p {
            return false;
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        let v_matrix = DMatrix::from_vec(self.n, self.p, vector.data.as_vec().clone());
        
        // Check if X^T V + V^T X = 0 (skew-symmetric constraint)
        let xtv = x_matrix.transpose() * &v_matrix;
        let vtx = v_matrix.transpose() * &x_matrix;
        let sum = xtv + vtx;
        
        // Check if sum is approximately zero
        for i in 0..self.p {
            for j in 0..self.p {
                if <T as Float>::abs(sum[(i, j)]) > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn project_point(&self, point: &DVector<T>) -> DVector<T> {
        let matrix = if point.len() == self.n * self.p {
            DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone())
        } else {
            // Handle wrong size by creating a random matrix
            let mut matrix = DMatrix::<T>::zeros(self.n, self.p);
            let copy_len = point.len().min(self.n * self.p);
            for i in 0..copy_len {
                let row = i / self.p;
                let col = i % self.p;
                matrix[(row, col)] = point[i];
            }
            matrix
        };
        
        let projected = self.qr_projection(&matrix);
        DVector::from_vec(projected.data.as_vec().clone())
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        if point.len() != self.n * self.p || vector.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Point and vector must have correct dimensions for Stiefel manifold",
                format!("point: {}, vector: {}", point.len(), vector.len()),
            ));
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        let v_matrix = DMatrix::from_vec(self.n, self.p, vector.data.as_vec().clone());
        
        // Project to tangent space: V - X(X^T V + V^T X)/2
        let xtv = x_matrix.transpose() * &v_matrix;
        let vtx = v_matrix.transpose() * &x_matrix;
        let symmetric = (&xtv + &vtx) * <T as Scalar>::from_f64(0.5);
        
        let projected = v_matrix - &x_matrix * symmetric;
        Ok(DVector::from_vec(projected.data.as_vec().clone()))
    }

    fn inner_product(
        &self,
        _point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        // Use Euclidean inner product (canonical metric)
        Ok(u.dot(v))
    }

    fn retract(&self, point: &DVector<T>, tangent: &DVector<T>) -> Result<DVector<T>> {
        if point.len() != self.n * self.p || tangent.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Point and tangent must have correct dimensions",
                format!("point: {}, tangent: {}", point.len(), tangent.len()),
            ));
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        let v_matrix = DMatrix::from_vec(self.n, self.p, tangent.data.as_vec().clone());
        
        // Use QR retraction: R(X, V) = qr(X + V).Q
        let candidate = x_matrix + v_matrix;
        let retracted = self.qr_projection(&candidate);
        Ok(DVector::from_vec(retracted.data.as_vec().clone()))
    }

    fn inverse_retract(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
    ) -> Result<DVector<T>> {
        if point.len() != self.n * self.p || other.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("point: {}, other: {}", point.len(), other.len()),
            ));
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        let y_matrix = DMatrix::from_vec(self.n, self.p, other.data.as_vec().clone());
        
        // Approximate inverse retraction: V ≈ Y - X
        let v_matrix = y_matrix - &x_matrix;
        
        // Project to tangent space
        let xtv = x_matrix.transpose() * &v_matrix;
        let vtx = v_matrix.transpose() * &x_matrix;
        let symmetric = (&xtv + &vtx) * <T as Scalar>::from_f64(0.5);
        
        let projected = v_matrix - &x_matrix * symmetric;
        Ok(DVector::from_vec(projected.data.as_vec().clone()))
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        grad: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Project Euclidean gradient to tangent space
        self.project_tangent(point, grad)
    }

    fn random_point(&self) -> DVector<T> {
        let mut rng = rand::thread_rng();
        let mut matrix = DMatrix::<T>::zeros(self.n, self.p);
        
        // Generate random matrix
        for i in 0..self.n {
            for j in 0..self.p {
                let val: f64 = StandardNormal.sample(&mut rng);
                matrix[(i, j)] = <T as Scalar>::from_f64(val);
            }
        }
        
        let projected = self.qr_projection(&matrix);
        DVector::from_vec(projected.data.as_vec().clone())
    }

    fn random_tangent(&self, point: &DVector<T>) -> Result<DVector<T>> {
        if point.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Point must have correct dimensions",
                format!("expected: {}, actual: {}", self.n * self.p, point.len()),
            ));
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        let tangent = self.random_tangent_matrix(&x_matrix)?;
        Ok(DVector::from_vec(tangent.data.as_vec().clone()))
    }

    fn has_exact_exp_log(&self) -> bool {
        false // Stiefel manifold doesn't have closed-form exp/log maps
    }

    fn parallel_transport(
        &self,
        _from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Use projection-based parallel transport
        // P_{x->y}(v) = proj_tangent_y(v)
        self.project_tangent(to, vector)
    }

    fn distance(&self, point1: &DVector<T>, point2: &DVector<T>) -> Result<T> {
        if point1.len() != self.n * self.p || point2.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("point1: {}, point2: {}", point1.len(), point2.len()),
            ));
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, point1.data.as_vec().clone());
        let y_matrix = DMatrix::from_vec(self.n, self.p, point2.data.as_vec().clone());
        
        // Geodesic distance using principal angles
        let m = x_matrix.transpose() * &y_matrix;
        let svd = m.svd(true, true);
        
        let mut distance_squared = T::zero();
        let singular_values = svd.singular_values;
        
        for i in 0..singular_values.len() {
            let sigma = singular_values[i];
            // Clamp to avoid numerical issues
            let clamped = <T as Float>::max(
                <T as Float>::min(sigma, T::one()),
                -T::one(),
            );
            let angle = <T as Float>::acos(clamped);
            distance_squared = distance_squared + angle * angle;
        }
        
        Ok(<T as Float>::sqrt(distance_squared))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_stiefel_creation() {
        let stiefel = Stiefel::new(5, 3).unwrap();
        assert_eq!(<Stiefel as Manifold<f64, Dyn>>::dimension(&stiefel), 15 - 6); // 5*3 - 3*4/2 = 9
        assert_eq!(stiefel.n(), 5);
        assert_eq!(stiefel.p(), 3);
        
        // Test invalid dimensions
        assert!(Stiefel::new(3, 5).is_err()); // p > n
        assert!(Stiefel::new(0, 3).is_err()); // n = 0
        assert!(Stiefel::new(3, 0).is_err()); // p = 0
    }

    #[test]
    fn test_orthonormality_check() {
        let stiefel = Stiefel::new(4, 2).unwrap();
        
        // Create orthonormal matrix
        let mut matrix = DMatrix::zeros(4, 2);
        matrix[(0, 0)] = 1.0;
        matrix[(1, 1)] = 1.0;
        
        assert!(stiefel.is_orthonormal(&matrix, 1e-10));
        
        // Create non-orthonormal matrix
        let mut matrix = DMatrix::zeros(4, 2);
        matrix[(0, 0)] = 1.0;
        matrix[(0, 1)] = 1.0; // This makes it non-orthonormal
        
        assert!(!stiefel.is_orthonormal(&matrix, 1e-10));
    }

    #[test]
    fn test_point_on_manifold() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        
        // Create orthonormal matrix as vector (column-major order)
        let matrix = DMatrix::from_vec(3, 2, vec![
            1.0, 0.0, 0.0,  // First column: [1, 0, 0]
            0.0, 1.0, 0.0   // Second column: [0, 1, 0]
        ]);
        let point = DVector::from_vec(matrix.data.as_vec().clone());
        
        assert!(stiefel.is_point_on_manifold(&point, 1e-10));
    }

    #[test]
    fn test_tangent_space() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        
        // Create identity-like matrix (column-major order)
        let x_matrix = DMatrix::from_vec(3, 2, vec![
            1.0, 0.0, 0.0,  // First column
            0.0, 1.0, 0.0   // Second column
        ]);
        let point = DVector::from_vec(x_matrix.data.as_vec().clone());
        
        // Create tangent vector: X^T V + V^T X = 0 (column-major order)
        let v_matrix = DMatrix::from_vec(3, 2, vec![
            0.0, 0.0, 1.0,  // First column
            0.0, 0.0, 1.0   // Second column
        ]);
        let tangent = DVector::from_vec(v_matrix.data.as_vec().clone());
        
        assert!(stiefel.is_vector_in_tangent_space(&point, &tangent, 1e-10));
    }

    #[test]
    fn test_projection() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        
        // Create non-orthonormal matrix
        let point = DVector::from_vec(vec![2.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let projected = stiefel.project_point(&point);
        
        assert!(stiefel.is_point_on_manifold(&projected, 1e-10));
    }

    #[test]
    fn test_tangent_projection() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        
        let point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let projected = stiefel.project_tangent(&point, &vector).unwrap();
        assert!(stiefel.is_vector_in_tangent_space(&point, &projected, 1e-10));
    }

    #[test]
    fn test_retraction_properties() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        let point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let zero_tangent = DVector::zeros(6);
        
        // Test centering: R(x, 0) = x
        let retracted = stiefel.retract(&point, &zero_tangent).unwrap();
        assert_relative_eq!(
            (retracted - &point).norm(), 
            0.0, 
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_random_generation() {
        let stiefel = Stiefel::new(4, 2).unwrap();
        
        // Test random point
        let random_point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        assert!(stiefel.is_point_on_manifold(&random_point, 1e-10));
        
        // Test random tangent
        let tangent = stiefel.random_tangent(&random_point).unwrap();
        assert!(stiefel.is_vector_in_tangent_space(&random_point, &tangent, 1e-10));
    }

    #[test]
    fn test_gradient_conversion() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        let point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let euclidean_grad = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let riemannian_grad = stiefel
            .euclidean_to_riemannian_gradient(&point, &euclidean_grad)
            .unwrap();
        
        assert!(stiefel.is_vector_in_tangent_space(&point, &riemannian_grad, 1e-10));
    }

    #[test]
    fn test_distance() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        let point1 = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let point2 = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        
        // Distance should be non-negative
        let dist = stiefel.distance(&point1, &point2).unwrap();
        assert!(dist >= 0.0);
        
        // Distance to self should be zero
        let self_dist = stiefel.distance(&point1, &point1).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-10);
    }
}