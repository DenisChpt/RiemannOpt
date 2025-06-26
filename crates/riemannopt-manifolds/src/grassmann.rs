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
    manifold::{Manifold, Point, TangentVector},
    types::{Scalar, DVector},
    compute::{get_dispatcher, SimdBackend},
    memory::Workspace,
};
use crate::utils::{vector_to_matrix_view, vector_to_matrix_view_mut};
use nalgebra::{DMatrix, Dyn};
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
                format!("Grassmann manifold Gr(n,p) requires ambient dimension n > 0, got n={}", n),
            ));
        }
        if p == 0 || p >= n {
            return Err(ManifoldError::invalid_point(
                format!("Grassmann manifold Gr(n,p) requires 0 < p < n, got n={}, p={}", n, p),
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
    fn canonical_representation<T>(&self, matrix: &DMatrix<T>, _workspace: &mut Workspace<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        // Clone is necessary here as QR decomposition consumes the matrix
        let qr = matrix.clone().qr();
        let mut q = qr.q().columns(0, self.p).into_owned();
        
        // Ensure positive diagonal elements in R for canonical form
        let r = qr.r();
        for i in 0..std::cmp::min(self.p, r.nrows()) {
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
    fn project_to_manifold<T>(&self, matrix: &DMatrix<T>, workspace: &mut Workspace<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            // Use workspace buffer for dimension mismatch
            let mut padded = workspace.acquire_temp_matrix(self.n, self.p);
            padded.fill(T::zero());
            
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
            
            self.canonical_representation(&padded.clone_owned(), workspace)
        } else {
            self.canonical_representation(matrix, workspace)
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
	#[allow(dead_code)]
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

    /// Projects a vector to the horizontal tangent space at a point using a workspace.
    ///
    /// This variant avoids allocations by using pre-allocated buffers from the workspace.
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as vector)
    /// * `vector` - Vector to project
    /// * `result` - Output buffer for the projected vector
    /// * `workspace` - Pre-allocated workspace for temporary buffers
    fn project_tangent_with_workspace<T>(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        T: Scalar,
    {
        if point.len() != self.n * self.p || vector.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for Gr({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements, vector has {} elements", point.len(), vector.len()),
            ));
        }
        
        // Use workspace buffer for X^T V
        let mut xtv = workspace.acquire_temp_matrix(self.p, self.p);
        
        // Use matrix views instead of cloning
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let v_matrix = vector_to_matrix_view(vector, self.n, self.p);
        
        // Compute X^T V
        xtv.copy_from(&(x_matrix.transpose() * &v_matrix));
        
        // Project to horizontal space: V - X(X^T V) directly into result
        result.copy_from(vector);
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &x_matrix * &*xtv;
        
        Ok(())
    }

    /// Retracts a tangent vector at a point using a workspace.
    ///
    /// This variant avoids allocations by using pre-allocated buffers from the workspace.
    /// Uses QR decomposition for the retraction.
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as vector)
    /// * `tangent` - Tangent vector
    /// * `result` - Output buffer for the retracted point
    /// * `workspace` - Pre-allocated workspace for temporary buffers
    fn retract_with_workspace<T>(
        &self,
        point: &DVector<T>,
        tangent: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        T: Scalar,
    {
        if point.len() != self.n * self.p || tangent.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for Gr({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements, tangent has {} elements", point.len(), tangent.len()),
            ));
        }
        
        // Use workspace buffer for the candidate matrix
        let mut candidate = workspace.acquire_temp_matrix(self.n, self.p);
        
        // Use matrix views
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let v_matrix = vector_to_matrix_view(tangent, self.n, self.p);
        
        // QR retraction: R(X, V) = qr(X + V).Q
        candidate.copy_from(&x_matrix);
        *candidate += &v_matrix;
        
        // Compute canonical representation
        // Clone is necessary here as QR decomposition consumes the matrix
        let qr = candidate.clone_owned().qr();
        let q_full = qr.q();
        let mut q = q_full.columns(0, self.p).into_owned();
        
        // Ensure positive diagonal elements in R for canonical form
        let r = qr.r();
        for i in 0..std::cmp::min(self.p, r.nrows()) {
            if r[(i, i)] < T::zero() {
                for j in 0..self.n {
                    q[(j, i)] = -q[(j, i)];
                }
            }
        }
        
        result.copy_from(&DVector::from_vec(q.as_slice().to_vec()));
        Ok(())
    }

    /// Computes principal angles between two subspaces using a workspace.
    ///
    /// This variant avoids allocations by using pre-allocated buffers from the workspace.
    fn principal_angles_cosines_with_workspace<T>(
        &self,
        x1: &DMatrix<T>,
        x2: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> DVector<T>
    where
        T: Scalar,
    {
        // Use workspace buffer for X1^T X2
        let mut inner = workspace.acquire_temp_matrix(self.p, self.p);
        inner.copy_from(&(x1.transpose() * x2));
        
        // SVD requires owned matrix, so we must clone here
        let svd = inner.clone_owned().svd(true, true);
        
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

    /// Computes distance between two points using a workspace.
    ///
    /// This variant avoids allocations by using pre-allocated buffers from the workspace.
    fn distance_with_workspace<T>(
        &self,
        x: &Point<T, Dyn>,
        y: &Point<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<T>
    where
        T: Scalar,
    {
        if x.len() != self.n * self.p || y.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for Gr({},{})", self.n * self.p, self.n, self.p),
                format!("x has {} elements, y has {} elements", x.len(), y.len()),
            ));
        }
        
        // Use matrix views to avoid cloning
        let x1_matrix = vector_to_matrix_view(x, self.n, self.p);
        let x2_matrix = vector_to_matrix_view(y, self.n, self.p);
        
        // Compute geodesic distance using principal angles
        let cosines = self.principal_angles_cosines_with_workspace(&x1_matrix.clone_owned(), &x2_matrix.clone_owned(), workspace);
        
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
    
    fn ambient_dimension(&self) -> usize {
        self.n * self.p
    }

    fn is_point_on_manifold(&self, point: &Point<T, Dyn>, tolerance: T) -> bool {
        if point.len() != self.n * self.p {
            return false;
        }
        
        // Use matrix view to avoid cloning
        let matrix = vector_to_matrix_view(point, self.n, self.p);
        self.is_valid_subspace_representative(&matrix.clone_owned(), tolerance)
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        tolerance: T,
    ) -> bool {
        if point.len() != self.n * self.p || vector.len() != self.n * self.p {
            return false;
        }
        
        // Use matrix views to avoid cloning
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let v_matrix = vector_to_matrix_view(vector, self.n, self.p);
        
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

    fn project_point(&self, point: &Point<T, Dyn>, result: &mut Point<T, Dyn>, workspace: &mut Workspace<T>) {
        // Ensure result has correct size
        let expected_size = self.n * self.p;
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }

        let matrix = if point.len() == self.n * self.p {
            // Use matrix view to avoid cloning
            let view = vector_to_matrix_view(point, self.n, self.p);
            view.clone_owned()
        } else {
            // Handle wrong dimensions using workspace
            let mut matrix = workspace.acquire_temp_matrix(self.n, self.p);
            matrix.fill(T::zero());
            let copy_len = point.len().min(self.n * self.p);
            for i in 0..copy_len {
                let row = i / self.p;
                let col = i % self.p;
                matrix[(row, col)] = point[i];
            }
            matrix.clone_owned()
        };
        
        let projected = self.project_to_manifold(&matrix, workspace);
        result.copy_from(&DVector::from_vec(projected.as_slice().to_vec()))
    }

    fn project_tangent(
        &self,
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.n * self.p;
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        self.project_tangent_with_workspace(point, vector, result, workspace)
    }

    fn inner_product(
        &self,
        _point: &Point<T, Dyn>,
        u: &TangentVector<T, Dyn>,
        v: &TangentVector<T, Dyn>,
    ) -> Result<T> {
        // Use Euclidean inner product (canonical metric)
        let dispatcher = get_dispatcher::<T>();
        Ok(dispatcher.dot_product(u, v))
    }

    fn retract(&self, point: &Point<T, Dyn>, tangent: &TangentVector<T, Dyn>, result: &mut Point<T, Dyn>, workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.n * self.p;
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        self.retract_with_workspace(point, tangent, result, workspace)
    }

    fn inverse_retract(
        &self,
        point: &Point<T, Dyn>,
        other: &Point<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.n * self.p;
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        if point.len() != self.n * self.p || other.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for Gr({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements, other has {} elements", point.len(), other.len()),
            ));
        }
        
        // Use matrix views to avoid cloning
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let _y_matrix = vector_to_matrix_view(other, self.n, self.p);
        
        // Approximate inverse retraction: V ≈ Y - X, then project to horizontal space
        result.copy_from(other);
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &x_matrix;
        
        // Now project to horizontal tangent space in-place
        let v_matrix = vector_to_matrix_view(result, self.n, self.p);
        let mut xtv = workspace.acquire_temp_matrix(self.p, self.p);
        xtv.copy_from(&(x_matrix.transpose() * &v_matrix));
        
        // Final projection: result = result - X * (X^T * result)
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &x_matrix * &*xtv;
        
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Point<T, Dyn>,
        euclidean_grad: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.n * self.p;
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        // Project Euclidean gradient to horizontal tangent space
        self.project_tangent(point, euclidean_grad, result, workspace)
    }

    fn random_point(&self) -> Point<T, Dyn> {
        let mut rng = rand::thread_rng();
        let mut matrix = DMatrix::<T>::zeros(self.n, self.p);
        
        // Generate random matrix
        for i in 0..self.n {
            for j in 0..self.p {
                let val: f64 = StandardNormal.sample(&mut rng);
                matrix[(i, j)] = <T as Scalar>::from_f64(val);
            }
        }
        
        // Create temporary workspace for projection
        let mut workspace = Workspace::new();
        let projected = self.project_to_manifold(&matrix, &mut workspace);
        DVector::from_vec(projected.as_slice().to_vec())
    }

    fn random_tangent(&self, point: &Point<T, Dyn>, result: &mut TangentVector<T, Dyn>, _workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.n * self.p;
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        if point.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for Gr({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements", point.len()),
            ));
        }
        
        // Use matrix view to avoid cloning
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let tangent = self.random_horizontal_tangent(&x_matrix.clone_owned())?;
        result.copy_from(&DVector::from_vec(tangent.as_slice().to_vec()));
        Ok(())
    }

    fn has_exact_exp_log(&self) -> bool {
        false // Grassmann manifold doesn't have simple closed-form exp/log maps
    }

    fn parallel_transport(
        &self,
        _from: &Point<T, Dyn>,
        to: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.n * self.p;
        if result.len() != expected_size {
            *result = DVector::zeros(expected_size);
        }
        
        // Use projection-based parallel transport for simplicity
        // More sophisticated parallel transport could be implemented using
        // the geodesic connection, but projection works well in practice
        self.project_tangent(to, vector, result, workspace)
    }

    fn distance(&self, x: &Point<T, Dyn>, y: &Point<T, Dyn>, workspace: &mut Workspace<T>) -> Result<T> {
        self.distance_with_workspace(x, y, workspace)
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
        let mut workspace = Workspace::new();
        let canon1 = grassmann.canonical_representation(&matrix, &mut workspace);
        let canon2 = grassmann.canonical_representation(&rotated, &mut workspace);
        
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
        let point = DVector::from_vec(matrix.as_slice().to_vec());
        
        assert!(grassmann.is_point_on_manifold(&point, 1e-10));
        
        // Test non-orthonormal matrix
        let bad_matrix = DMatrix::from_vec(4, 2, vec![
            2.0, 0.0, 0.0, 0.0,  // First column (not unit)
            0.0, 1.0, 0.0, 0.0   // Second column
        ]);
        let bad_point = DVector::from_vec(bad_matrix.as_slice().to_vec());
        
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
        let point = DVector::from_vec(x_matrix.as_slice().to_vec());
        
        // Create vector in horizontal tangent space (X^T V = 0)
        let v_matrix = DMatrix::from_vec(4, 2, vec![
            0.0, 0.0, 1.0, 0.0,  // First column
            0.0, 0.0, 0.0, 1.0   // Second column
        ]);
        let tangent = DVector::from_vec(v_matrix.as_slice().to_vec());
        
        assert!(grassmann.is_vector_in_tangent_space(&point, &tangent, 1e-10));
        
        // Test vector not in horizontal space
        let bad_v_matrix = DMatrix::from_vec(4, 2, vec![
            1.0, 0.0, 0.0, 0.0,  // This violates X^T V = 0
            0.0, 0.0, 0.0, 1.0
        ]);
        let bad_tangent = DVector::from_vec(bad_v_matrix.as_slice().to_vec());
        
        assert!(!grassmann.is_vector_in_tangent_space(&point, &bad_tangent, 1e-10));
    }

    #[test]
    fn test_projection_operations() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        let mut workspace = Workspace::new();
        
        // Test point projection
        let bad_point = DVector::from_vec(vec![2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut projected_point = DVector::zeros(8); // n * p = 4 * 2 = 8
        grassmann.project_point(&bad_point, &mut projected_point, &mut workspace);
        assert!(grassmann.is_point_on_manifold(&projected_point, 1e-10));
        
        // Test tangent projection
        let point = grassmann.random_point();
        let bad_vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut projected_tangent = DVector::zeros(8); // n * p = 4 * 2 = 8
        grassmann.project_tangent(&point, &bad_vector, &mut projected_tangent, &mut workspace).unwrap();
        
        assert!(grassmann.is_vector_in_tangent_space(&point, &projected_tangent, 1e-10));
    }

    #[test]
    fn test_retraction_properties() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        let mut workspace = Workspace::new();
        let point = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        let zero_tangent = DVector::zeros(8);
        
        // Test centering property: R(x, 0) = x
        let mut retracted = DVector::zeros(8); // n * p = 4 * 2 = 8
        grassmann.retract(&point, &zero_tangent, &mut retracted, &mut workspace).unwrap();
        
        // For Grassmann manifold, the retraction should preserve the subspace
        // Check that both points represent the same subspace
        let x_matrix = DMatrix::from_vec(4, 2, point.as_slice().to_vec());
        let r_matrix = DMatrix::from_vec(4, 2, retracted.as_slice().to_vec());
        
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
        let mut workspace = Workspace::new();
        
        let point1 = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        let point2 = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        
        // Distance should be non-negative
        let dist = grassmann.distance(&point1, &point2, &mut workspace).unwrap();
        assert!(dist >= 0.0);
        
        // Distance to self should be zero (within numerical tolerance)
        let self_dist = grassmann.distance(&point1, &point1, &mut workspace).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-7);
        
        // Distance should be symmetric
        let dist_rev = grassmann.distance(&point2, &point1, &mut workspace).unwrap();
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-10);
    }

    #[test]
    fn test_random_generation() {
        let grassmann = Grassmann::new(5, 3).unwrap();
        let mut workspace = Workspace::new();
        
        // Test random point generation
        let random_point = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        assert!(grassmann.is_point_on_manifold(&random_point, 1e-10));
        
        // Test random tangent generation
        let mut tangent = DVector::zeros(15);
        grassmann.random_tangent(&random_point, &mut tangent, &mut workspace).unwrap();
        assert!(grassmann.is_vector_in_tangent_space(&random_point, &tangent, 1e-10));
    }

    #[test]
    fn test_gradient_conversion() {
        let grassmann = Grassmann::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        let point = <Grassmann as Manifold<f64, Dyn>>::random_point(&grassmann);
        let euclidean_grad = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let mut riemannian_grad = DVector::zeros(6);
        grassmann
            .euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut workspace)
            .unwrap();
        
        assert!(grassmann.is_vector_in_tangent_space(&point, &riemannian_grad, 1e-10));
    }

    #[test]
    fn test_subspace_invariance() {
        let grassmann = Grassmann::new(4, 2).unwrap();
        let mut workspace = Workspace::new();
        
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
        
        let mut point1 = DVector::zeros(8);
        grassmann.project_point(&DVector::from_vec(matrix1.as_slice().to_vec()), &mut point1, &mut workspace);
        let mut point2 = DVector::zeros(8);
        grassmann.project_point(&DVector::from_vec(matrix2.as_slice().to_vec()), &mut point2, &mut workspace);
        
        // Distance between equivalent representations should be small
        let dist = grassmann.distance(&point1, &point2, &mut workspace).unwrap();
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
        
        let mut workspace = Workspace::new();
        let cosines = grassmann.principal_angles_cosines_with_workspace(&matrix1, &matrix2, &mut workspace);
        
        // For orthogonal subspaces, all principal angles should be pi/2
        for i in 0..cosines.len() {
            assert_relative_eq!(cosines[i], 0.0, epsilon = 1e-10);
        }
        
        // Distance should be sqrt(pi^2/2 + pi^2/2) = pi/sqrt(2)
        let point1 = DVector::from_vec(matrix1.as_slice().to_vec());
        let point2 = DVector::from_vec(matrix2.as_slice().to_vec());
        let dist = grassmann.distance(&point1, &point2, &mut workspace).unwrap();
        let expected_dist = std::f64::consts::PI * std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(dist, expected_dist, epsilon = 1e-10);
    }
}

