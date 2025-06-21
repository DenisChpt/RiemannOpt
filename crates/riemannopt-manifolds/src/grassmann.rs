//! Grassmann manifold Gr(n,p) - the space of p-dimensional subspaces in R^n
//!
//! The Grassmann manifold represents the quotient space St(n,p)/O(p), where
//! St(n,p) is the Stiefel manifold and O(p) is the orthogonal group.
//! Points represent p-dimensional linear subspaces rather than specific
//! orthonormal bases, making it fundamental in:
//! - Subspace tracking and identification
//! - Principal component analysis (PCA)
//! - Computer vision (motion estimation)
//! - Signal processing (subspace methods)
//! - Machine learning (dimensionality reduction)

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::Scalar,
    core::MatrixManifold,
};
use nalgebra::{DMatrix, DVector, Dyn};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

/// The Grassmann manifold Gr(n,p) of p-dimensional subspaces in R^n.
///
/// This manifold represents equivalence classes of orthonormal matrices
/// under right multiplication by orthogonal matrices. Each point corresponds
/// to a unique p-dimensional linear subspace of R^n.
///
/// # Mathematical Properties
///
/// - **Dimension**: p(n-p) (quotient dimension)
/// - **Tangent space**: T_X Gr(n,p) = {V in R^{n x p} : X^T V = 0} (horizontal space)
/// - **Riemannian metric**: Inherited from ambient Euclidean space
/// - **Geodesic distance**: Sum of principal angles between subspaces
///
/// # Representation
///
/// Points are represented by orthonormal matrices X in R^{n x p}, where
/// X and XR represent the same point for any R in O(p). We maintain
/// a canonical representation to ensure uniqueness.
///
/// # Applications
///
/// - **Computer vision**: Motion subspaces, face recognition
/// - **Signal processing**: Subspace-based parameter estimation
/// - **Machine learning**: Principal subspace analysis, dimensionality reduction
/// - **Statistics**: Canonical correlation analysis, factor analysis
#[derive(Debug, Clone)]
pub struct Grassmann {
    /// Ambient dimension (n)
    n: usize,
    /// Subspace dimension (p)
    p: usize,
}

impl Grassmann {
    /// Creates a new Grassmann manifold Gr(n,p).
    ///
    /// # Arguments
    /// * `n` - Ambient dimension (must be > 0)
    /// * `p` - Subspace dimension (must satisfy 0 < p < n)
    ///
    /// # Returns
    /// A Grassmann manifold with intrinsic dimension p(n-p)
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid
    ///
    /// # Examples
    /// ```
    /// use riemannopt_manifolds::Grassmann;
    /// 
    /// // Create Gr(5,2) - 2D subspaces in R^5
    /// let grassmann = Grassmann::new(5, 2).unwrap();
    /// assert_eq!(grassmann.subspace_dimension(), 2);
    /// assert_eq!(grassmann.ambient_dimension(), 5);
    /// ```
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "Grassmann manifold requires ambient dimension n > 0",
            ));
        }
        if p == 0 || p >= n {
            return Err(ManifoldError::invalid_point(
                "Grassmann manifold requires 0 < p < n",
            ));
        }
        Ok(Self { n, p })
    }

    /// Returns the ambient dimension (n)
    pub fn ambient_dimension(&self) -> usize {
        self.n
    }

    /// Returns the subspace dimension (p)
    pub fn subspace_dimension(&self) -> usize {
        self.p
    }

    /// Returns the dimensions (n, p)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.n, self.p)
    }

    /// Computes the canonical representation of a subspace.
    ///
    /// Given an orthonormal matrix X, returns a canonical representative
    /// of the same subspace. This ensures uniqueness of representation.
    fn canonical_representation<T>(&self, matrix: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        // Use QR decomposition to get canonical form
        let qr = matrix.clone().qr();
        let mut q = qr.q().columns(0, self.p).into_owned();
        
        // Ensure positive diagonal elements in R for canonical form
        let r = qr.r();
        for i in 0..self.p.min(r.nrows()) {
            if r[(i, i)] < T::zero() {
                // Flip sign of column if diagonal element is negative
                for j in 0..self.n {
                    q[(j, i)] = -q[(j, i)];
                }
            }
        }
        
        q
    }

    /// Projects a matrix to the Grassmann manifold.
    ///
    /// This computes the canonical orthonormal basis for the subspace
    /// spanned by the columns of the input matrix.
    fn project_to_manifold<T>(&self, matrix: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            // Handle dimension mismatch by padding/truncating
            let mut padded = DMatrix::<T>::zeros(self.n, self.p);
            let copy_rows = matrix.nrows().min(self.n);
            let copy_cols = matrix.ncols().min(self.p);
            
            for i in 0..copy_rows {
                for j in 0..copy_cols {
                    padded[(i, j)] = matrix[(i, j)];
                }
            }
            
            // Ensure we have non-zero columns
            for j in 0..self.p {
                if padded.column(j).norm() < T::epsilon() {
                    if j < self.n {
                        padded[(j, j)] = T::one();
                    }
                }
            }
            
            self.canonical_representation(&padded)
        } else {
            self.canonical_representation(matrix)
        }
    }

    /// Projects a vector to the horizontal tangent space at a point.
    ///
    /// The horizontal tangent space consists of matrices V such that X^T V = 0,
    /// representing variations in the subspace that don't correspond to
    /// rotations within the subspace.
    fn project_to_horizontal_tangent<T>(
        &self,
        point: &DMatrix<T>,
        vector: &DMatrix<T>,
    ) -> DMatrix<T>
    where
        T: Scalar,
    {
        // Project to horizontal space: V - X(X^T V)
        let xtv = point.transpose() * vector;
        vector - point * xtv
    }

    /// Computes principal angles between two subspaces.
    ///
    /// Returns the cosines of principal angles, sorted in descending order.
    fn principal_angles_cosines<T>(
        &self,
        x1: &DMatrix<T>,
        x2: &DMatrix<T>,
    ) -> DVector<T>
    where
        T: Scalar,
    {
        // Compute SVD of X1^T X2 to get principal angles
        let inner = x1.transpose() * x2;
        let svd = inner.svd(true, true);
        
        // Singular values are cosines of principal angles
        // Clamp to [0,1] to avoid numerical issues
        let mut cosines = svd.singular_values.clone();
        for i in 0..cosines.len() {
            cosines[i] = <T as Float>::max(
                <T as Float>::min(cosines[i], T::one()),
                T::zero(),
            );
        }
        
        cosines
    }

    /// Generates a random tangent vector in the horizontal space.
    fn random_horizontal_tangent<T>(&self, point: &DMatrix<T>) -> Result<DMatrix<T>>
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
        
        // Project to horizontal tangent space
        Ok(self.project_to_horizontal_tangent(point, &random_matrix))
    }

    /// Checks if a matrix represents a valid point on the Grassmann manifold.
    fn is_valid_subspace_representative<T>(&self, matrix: &DMatrix<T>, tolerance: T) -> bool
    where
        T: Scalar,
    {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            return false;
        }
        
        // Check orthonormality: X^T X = I
        let gram = matrix.transpose() * matrix;
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
}

impl<T> Manifold<T, Dyn> for Grassmann
where
    T: Scalar,
{
    fn name(&self) -> &str {
        "Grassmann"
    }

    fn dimension(&self) -> usize {
        self.p * (self.n - self.p)
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tolerance: T) -> bool {
        if point.len() != self.n * self.p {
            return false;
        }
        
        let matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        self.is_valid_subspace_representative(&matrix, tolerance)
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
        
        // Check if V is in horizontal tangent space: X^T V = 0
        let xtv = x_matrix.transpose() * &v_matrix;
        
        for i in 0..self.p {
            for j in 0..self.p {
                if <T as Float>::abs(xtv[(i, j)]) > tolerance {
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
            // Handle wrong dimensions
            let mut matrix = DMatrix::<T>::zeros(self.n, self.p);
            let copy_len = point.len().min(self.n * self.p);
            for i in 0..copy_len {
                let row = i / self.p;
                let col = i % self.p;
                matrix[(row, col)] = point[i];
            }
            matrix
        };
        
        let projected = self.project_to_manifold(&matrix);
        DVector::from_vec(projected.data.as_vec().clone())
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        if point.len() != self.n * self.p || vector.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Point and vector must have correct dimensions for Grassmann manifold",
                format!("point: {}, vector: {}", point.len(), vector.len()),
            ));
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, point.data.as_vec().clone());
        let v_matrix = DMatrix::from_vec(self.n, self.p, vector.data.as_vec().clone());
        
        // Project to horizontal tangent space
        let projected = self.project_to_horizontal_tangent(&x_matrix, &v_matrix);
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
        
        // QR retraction: R(X, V) = qr(X + V).Q
        let candidate = x_matrix + v_matrix;
        let retracted = self.project_to_manifold(&candidate);
        
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
        
        // Approximate inverse retraction for Grassmann manifold
        // This is more complex than for Stiefel due to the quotient structure
        let v_matrix = y_matrix - &x_matrix;
        let projected = self.project_to_horizontal_tangent(&x_matrix, &v_matrix);
        
        Ok(DVector::from_vec(projected.data.as_vec().clone()))
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        grad: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Project Euclidean gradient to horizontal tangent space
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
        
        // Project to manifold to get canonical representation
        let projected = self.project_to_manifold(&matrix);
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
        let tangent = self.random_horizontal_tangent(&x_matrix)?;
        Ok(DVector::from_vec(tangent.data.as_vec().clone()))
    }

    fn has_exact_exp_log(&self) -> bool {
        false // Grassmann manifold doesn't have simple closed-form exp/log maps
    }

    fn parallel_transport(
        &self,
        _from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Use projection-based parallel transport for simplicity
        // More sophisticated parallel transport could be implemented using
        // the geodesic connection, but projection works well in practice
        self.project_tangent(to, vector)
    }

    fn distance(&self, point1: &DVector<T>, point2: &DVector<T>) -> Result<T> {
        if point1.len() != self.n * self.p || point2.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("point1: {}, point2: {}", point1.len(), point2.len()),
            ));
        }
        
        let x1_matrix = DMatrix::from_vec(self.n, self.p, point1.data.as_vec().clone());
        let x2_matrix = DMatrix::from_vec(self.n, self.p, point2.data.as_vec().clone());
        
        // Compute geodesic distance using principal angles
        let cosines = self.principal_angles_cosines(&x1_matrix, &x2_matrix);
        
        let mut distance_squared = T::zero();
        for i in 0..cosines.len() {
            let cos_theta = cosines[i];
            // Compute arccos, handling numerical issues
            let cos_clamped = <T as Float>::max(
                <T as Float>::min(cos_theta, T::one()),
                T::zero(),
            );
            let angle = <T as Float>::acos(cos_clamped);
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
    fn test_grassmann_creation() {
        let grassmann = Grassmann::new(5, 2).unwrap();
        assert_eq!(<Grassmann as Manifold<f64, Dyn>>::dimension(&grassmann), 6); // 2*(5-2) = 6
        assert_eq!(grassmann.ambient_dimension(), 5);
        assert_eq!(grassmann.subspace_dimension(), 2);
        
        // Test invalid dimensions
        assert!(Grassmann::new(3, 3).is_err()); // p >= n
        assert!(Grassmann::new(3, 0).is_err()); // p = 0
        assert!(Grassmann::new(0, 2).is_err()); // n = 0
    }

    #[test]
    fn test_canonical_representation() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        
        // Create a matrix and its rotated version
        let matrix = DMatrix::from_vec(4, 2, vec![
            1.0, 0.0, 0.0, 0.0,  // First column
            0.0, 1.0, 0.0, 0.0   // Second column
        ]);
        
        // Apply orthogonal transformation (rotation by pi/4)
        let cos45 = std::f64::consts::FRAC_1_SQRT_2;
        let rotation = DMatrix::from_vec(2, 2, vec![
            cos45, -cos45,
            cos45, cos45
        ]);
        let rotated = &matrix * rotation;
        
        // Both should give the same canonical representation
        let canon1 = grassmann.canonical_representation(&matrix);
        let canon2 = grassmann.canonical_representation(&rotated);
        
        // The subspaces spanned should be the same (up to canonical form)
        // Check that they span the same subspace by verifying projection matrices
        let proj1 = &canon1 * canon1.transpose();
        let proj2 = &canon2 * canon2.transpose();
        
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(proj1[(i, j)], proj2[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_point_on_manifold() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        
        // Create valid orthonormal matrix (column-major)
        let matrix = DMatrix::from_vec(4, 2, vec![
            1.0, 0.0, 0.0, 0.0,  // First column
            0.0, 1.0, 0.0, 0.0   // Second column
        ]);
        let point = DVector::from_vec(matrix.data.as_vec().clone());
        
        assert!(grassmann.is_point_on_manifold(&point, 1e-10));
        
        // Test non-orthonormal matrix
        let bad_matrix = DMatrix::from_vec(4, 2, vec![
            2.0, 0.0, 0.0, 0.0,  // First column (not unit)
            0.0, 1.0, 0.0, 0.0   // Second column
        ]);
        let bad_point = DVector::from_vec(bad_matrix.data.as_vec().clone());
        
        assert!(!grassmann.is_point_on_manifold(&bad_point, 1e-10));
    }

    #[test]
    fn test_horizontal_tangent_space() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        
        // Create point (column-major)
        let x_matrix = DMatrix::from_vec(4, 2, vec![
            1.0, 0.0, 0.0, 0.0,  // First column
            0.0, 1.0, 0.0, 0.0   // Second column
        ]);
        let point = DVector::from_vec(x_matrix.data.as_vec().clone());
        
        // Create vector in horizontal tangent space (X^T V = 0)
        let v_matrix = DMatrix::from_vec(4, 2, vec![
            0.0, 0.0, 1.0, 0.0,  // First column
            0.0, 0.0, 0.0, 1.0   // Second column
        ]);
        let tangent = DVector::from_vec(v_matrix.data.as_vec().clone());
        
        assert!(grassmann.is_vector_in_tangent_space(&point, &tangent, 1e-10));
        
        // Test vector not in horizontal space
        let bad_v_matrix = DMatrix::from_vec(4, 2, vec![
            1.0, 0.0, 0.0, 0.0,  // This violates X^T V = 0
            0.0, 0.0, 0.0, 1.0
        ]);
        let bad_tangent = DVector::from_vec(bad_v_matrix.data.as_vec().clone());
        
        assert!(!grassmann.is_vector_in_tangent_space(&point, &bad_tangent, 1e-10));
    }

    #[test]
    fn test_projection_operations() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        
        // Test point projection
        let bad_point = DVector::from_vec(vec![2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let projected_point = grassmann.project_point(&bad_point);
        assert!(grassmann.is_point_on_manifold(&projected_point, 1e-10));
        
        // Test tangent projection
        let point = grassmann.random_point();
        let bad_vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let projected_tangent = grassmann.project_tangent(&point, &bad_vector).unwrap();
        
        assert!(grassmann.is_vector_in_tangent_space(&point, &projected_tangent, 1e-10));
    }

    #[test]
    fn test_retraction_properties() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        let point = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        let zero_tangent = DVector::zeros(8);
        
        // Test centering property: R(x, 0) = x
        let retracted = grassmann.retract(&point, &zero_tangent).unwrap();
        
        // For Grassmann manifold, the retraction should preserve the subspace
        // Check that both points represent the same subspace
        let x_matrix = DMatrix::from_vec(4, 2, point.data.as_vec().clone());
        let r_matrix = DMatrix::from_vec(4, 2, retracted.data.as_vec().clone());
        
        let proj_x = &x_matrix * x_matrix.transpose();
        let proj_r = &r_matrix * r_matrix.transpose();
        
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(proj_x[(i, j)], proj_r[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_distance_properties() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        
        let point1 = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        let point2 = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        
        // Distance should be non-negative
        let dist = grassmann.distance(&point1, &point2).unwrap();
        assert!(dist >= 0.0);
        
        // Distance to self should be zero (within numerical tolerance)
        let self_dist = grassmann.distance(&point1, &point1).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-7);
        
        // Distance should be symmetric
        let dist_rev = grassmann.distance(&point2, &point1).unwrap();
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-10);
    }

    #[test]
    fn test_random_generation() {
        let grassmann = Grassmann::new(5, 3).unwrap();
        
        // Test random point generation
        let random_point = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        assert!(grassmann.is_point_on_manifold(&random_point, 1e-10));
        
        // Test random tangent generation
        let tangent = grassmann.random_tangent(&random_point).unwrap();
        assert!(grassmann.is_vector_in_tangent_space(&random_point, &tangent, 1e-10));
    }

    #[test]
    fn test_gradient_conversion() {
        let grassmann = Grassmann::new(3, 2).unwrap();
        let point = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        let euclidean_grad = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let riemannian_grad = grassmann
            .euclidean_to_riemannian_gradient(&point, &euclidean_grad)
            .unwrap();
        
        assert!(grassmann.is_vector_in_tangent_space(&point, &riemannian_grad, 1e-10));
    }

    #[test]
    fn test_subspace_invariance() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        
        // Create two representations of the same subspace
        let matrix1 = DMatrix::from_vec(4, 2, vec![
            1.0, 0.0, 0.0, 0.0,  // Standard basis
            0.0, 1.0, 0.0, 0.0
        ]);
        
        // Rotate within the subspace (should represent same point)
        let cos45 = std::f64::consts::FRAC_1_SQRT_2;
        let rotation = DMatrix::from_vec(2, 2, vec![
            cos45, -cos45,
            cos45, cos45
        ]);
        let matrix2 = &matrix1 * rotation;
        
        let point1 = grassmann.project_point(&DVector::from_vec(matrix1.data.as_vec().clone()));
        let point2 = grassmann.project_point(&DVector::from_vec(matrix2.data.as_vec().clone()));
        
        // Distance between equivalent representations should be small
        let dist = grassmann.distance(&point1, &point2).unwrap();
        assert!(dist < 1e-10, "Distance between equivalent subspaces: {}", dist);
    }

    #[test]
    fn test_principal_angles() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        
        // Create two orthogonal subspaces
        let matrix1 = DMatrix::from_vec(4, 2, vec![
            1.0, 0.0, 0.0, 0.0,  // span{e1, e2}
            0.0, 1.0, 0.0, 0.0
        ]);
        
        let matrix2 = DMatrix::from_vec(4, 2, vec![
            0.0, 0.0, 1.0, 0.0,  // span{e3, e4}
            0.0, 0.0, 0.0, 1.0
        ]);
        
        let cosines = grassmann.principal_angles_cosines(&matrix1, &matrix2);
        
        // For orthogonal subspaces, all principal angles should be pi/2
        for i in 0..cosines.len() {
            assert_relative_eq!(cosines[i], 0.0, epsilon = 1e-10);
        }
        
        // Distance should be sqrt(pi^2/2 + pi^2/2) = pi/sqrt(2)
        let point1 = DVector::from_vec(matrix1.data.as_vec().clone());
        let point2 = DVector::from_vec(matrix2.data.as_vec().clone());
        let dist = grassmann.distance(&point1, &point2).unwrap();
        let expected_dist = std::f64::consts::PI * std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(dist, expected_dist, epsilon = 1e-10);
    }
}

// MatrixManifold implementation for efficient matrix operations
impl<T: Scalar + Float> MatrixManifold<T> for Grassmann {
    fn matrix_dims(&self) -> (usize, usize) {
        (self.n, self.p)
    }
    
    fn project_matrix(&self, matrix: &DMatrix<T>) -> Result<DMatrix<T>> {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", matrix.nrows(), matrix.ncols()),
            ));
        }
        
        // Use QR decomposition to get orthonormal basis
        Ok(self.canonical_representation(matrix))
    }
    
    fn project_tangent_matrix(&self, point: &DMatrix<T>, matrix: &DMatrix<T>) -> Result<DMatrix<T>> {
        if point.nrows() != self.n || point.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", point.nrows(), point.ncols()),
            ));
        }
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", matrix.nrows(), matrix.ncols()),
            ));
        }
        
        // Project to horizontal space: V - XX^T V
        let xv = point.transpose() * matrix;
        Ok(matrix - point * xv)
    }
    
    fn inner_product_matrix(&self, _point: &DMatrix<T>, u: &DMatrix<T>, v: &DMatrix<T>) -> Result<T> {
        if u.nrows() != self.n || u.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", u.nrows(), u.ncols()),
            ));
        }
        if v.nrows() != self.n || v.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", v.nrows(), v.ncols()),
            ));
        }
        
        // Frobenius inner product: trace(U^T V)
        Ok((u.transpose() * v).trace())
    }
    
    fn retract_matrix(&self, point: &DMatrix<T>, tangent: &DMatrix<T>) -> Result<DMatrix<T>> {
        if point.nrows() != self.n || point.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", point.nrows(), point.ncols()),
            ));
        }
        if tangent.nrows() != self.n || tangent.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", tangent.nrows(), tangent.ncols()),
            ));
        }
        
        // QR retraction on Grassmann: R_X(V) = qf([X V])
        let combined = DMatrix::from_columns(&[
            point.column_iter().collect::<Vec<_>>().as_slice(),
            tangent.column_iter().collect::<Vec<_>>().as_slice(),
        ].concat());
        
        let qr = combined.qr();
        let q = qr.q();
        
        // Extract first p columns and canonicalize
        let result = q.columns(0, self.p).into_owned();
        Ok(self.canonical_representation(&result))
    }
    
    fn inverse_retract_matrix(&self, point: &DMatrix<T>, other: &DMatrix<T>) -> Result<DMatrix<T>> {
        if point.nrows() != self.n || point.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", point.nrows(), point.ncols()),
            ));
        }
        if other.nrows() != self.n || other.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", other.nrows(), other.ncols()),
            ));
        }
        
        // For Grassmann, inverse retraction via projection
        // This is an approximation
        let diff = other - point;
        self.project_tangent_matrix(point, &diff)
    }
    
    fn euclidean_to_riemannian_gradient_matrix(
        &self,
        point: &DMatrix<T>,
        euclidean_grad: &DMatrix<T>,
    ) -> Result<DMatrix<T>> {
        // Riemannian gradient is the projection to horizontal space
        self.project_tangent_matrix(point, euclidean_grad)
    }
    
    fn parallel_transport_matrix(
        &self,
        from: &DMatrix<T>,
        to: &DMatrix<T>,
        tangent: &DMatrix<T>,
    ) -> Result<DMatrix<T>> {
        if from.nrows() != self.n || from.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", from.nrows(), from.ncols()),
            ));
        }
        if to.nrows() != self.n || to.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", to.nrows(), to.ncols()),
            ));
        }
        if tangent.nrows() != self.n || tangent.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", tangent.nrows(), tangent.ncols()),
            ));
        }
        
        // Simple parallel transport via projection
        self.project_tangent_matrix(to, tangent)
    }
    
    fn random_point_matrix(&self) -> DMatrix<T> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random Gaussian matrix
        let mut a = DMatrix::<T>::zeros(self.n, self.p);
        for i in 0..self.n {
            for j in 0..self.p {
                let sample: f64 = normal.sample(&mut rng);
                a[(i, j)] = T::from(sample).unwrap();
            }
        }
        
        // Get canonical representation
        self.canonical_representation(&a)
    }
    
    fn random_tangent_matrix(&self, point: &DMatrix<T>) -> Result<DMatrix<T>> {
        if point.nrows() != self.n || point.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", point.nrows(), point.ncols()),
            ));
        }
        
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random matrix
        let mut v = DMatrix::<T>::zeros(self.n, self.p);
        for i in 0..self.n {
            for j in 0..self.p {
                let sample: f64 = normal.sample(&mut rng);
                v[(i, j)] = T::from(sample).unwrap();
            }
        }
        
        // Project to horizontal space
        self.project_tangent_matrix(point, &v)
    }
    
    fn is_point_on_manifold_matrix(&self, matrix: &DMatrix<T>, tolerance: T) -> bool {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            return false;
        }
        
        // Check X^T X = I (orthonormality)
        let gram = matrix.transpose() * matrix;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        let diff = &gram - &identity;
        
        diff.norm() <= tolerance
    }
    
    fn is_vector_in_tangent_space_matrix(
        &self,
        point: &DMatrix<T>,
        tangent: &DMatrix<T>,
        tolerance: T,
    ) -> bool {
        if point.nrows() != self.n || point.ncols() != self.p {
            return false;
        }
        if tangent.nrows() != self.n || tangent.ncols() != self.p {
            return false;
        }
        
        // Check X^T V = 0 (horizontal condition)
        let xtv = point.transpose() * tangent;
        
        // Check if xtv is approximately zero
        xtv.norm() <= tolerance
    }
}