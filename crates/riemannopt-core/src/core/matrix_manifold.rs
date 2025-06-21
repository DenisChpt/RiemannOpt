//! Matrix manifold trait for manifolds that naturally work with matrix representations.
//!
//! This module provides a trait for manifolds whose elements are naturally represented
//! as matrices (e.g., Stiefel, Grassmann, SPD manifolds). This allows for more efficient
//! operations by avoiding conversions between matrix and vector representations.

use nalgebra::{DMatrix, DVector};
use crate::{
    error::Result,
    types::Scalar,
};

/// Trait for manifolds whose elements are naturally represented as matrices.
///
/// This trait extends the base `Manifold` trait with matrix-specific operations,
/// allowing implementations to work directly with `DMatrix` representations
/// instead of converting to/from `DVector`.
///
/// # Benefits
///
/// - **Efficiency**: Avoids unnecessary matrix-vector conversions
/// - **Natural API**: Methods accept and return matrices matching the manifold structure
/// - **Memory optimization**: Can operate on matrix data in-place
/// - **Clarity**: Matrix operations are more intuitive for matrix manifolds
///
/// # Examples
///
/// ```rust,ignore
/// // For a Stiefel manifold St(n, p)
/// let point = DMatrix::identity(n, p);  // n×p matrix
/// let tangent = DMatrix::zeros(n, p);   // n×p matrix in tangent space
/// 
/// // Direct matrix operations
/// let projected = manifold.project_matrix(&point)?;
/// let new_point = manifold.retract_matrix(&point, &tangent)?;
/// ```
pub trait MatrixManifold<T: Scalar> {
    /// Get the dimensions of the matrix representation (rows, columns).
    fn matrix_dims(&self) -> (usize, usize);
    
    /// Project a matrix onto the manifold.
    ///
    /// # Arguments
    /// * `matrix` - Matrix to project
    ///
    /// # Returns
    /// Projected matrix that satisfies the manifold constraints
    fn project_matrix(&self, matrix: &DMatrix<T>) -> Result<DMatrix<T>>;
    
    /// Project a matrix to the tangent space at a point.
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as matrix)
    /// * `matrix` - Matrix to project
    ///
    /// # Returns
    /// Projected matrix in the tangent space at `point`
    fn project_tangent_matrix(&self, point: &DMatrix<T>, matrix: &DMatrix<T>) -> Result<DMatrix<T>>;
    
    /// Compute the Riemannian metric between two tangent vectors.
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as matrix)
    /// * `u` - First tangent vector (as matrix)
    /// * `v` - Second tangent vector (as matrix)
    ///
    /// # Returns
    /// Inner product ⟨u, v⟩ in the tangent space at `point`
    fn inner_product_matrix(&self, point: &DMatrix<T>, u: &DMatrix<T>, v: &DMatrix<T>) -> Result<T>;
    
    /// Retract a tangent vector to get a new point on the manifold.
    ///
    /// # Arguments
    /// * `point` - Current point on the manifold (as matrix)
    /// * `tangent` - Tangent vector (as matrix)
    ///
    /// # Returns
    /// New point on the manifold
    fn retract_matrix(&self, point: &DMatrix<T>, tangent: &DMatrix<T>) -> Result<DMatrix<T>>;
    
    /// Compute the inverse retraction (logarithmic map).
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as matrix)
    /// * `other` - Another point on the manifold (as matrix)
    ///
    /// # Returns
    /// Tangent vector at `point` that retracts to `other`
    fn inverse_retract_matrix(&self, point: &DMatrix<T>, other: &DMatrix<T>) -> Result<DMatrix<T>>;
    
    /// Convert Euclidean gradient to Riemannian gradient.
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as matrix)
    /// * `euclidean_grad` - Euclidean gradient (as matrix)
    ///
    /// # Returns
    /// Riemannian gradient (as matrix)
    fn euclidean_to_riemannian_gradient_matrix(
        &self,
        point: &DMatrix<T>,
        euclidean_grad: &DMatrix<T>,
    ) -> Result<DMatrix<T>>;
    
    /// Parallel transport a tangent vector along a retraction.
    ///
    /// # Arguments
    /// * `from` - Starting point (as matrix)
    /// * `to` - Ending point (as matrix)
    /// * `tangent` - Tangent vector at `from` (as matrix)
    ///
    /// # Returns
    /// Transported tangent vector at `to`
    fn parallel_transport_matrix(
        &self,
        from: &DMatrix<T>,
        to: &DMatrix<T>,
        tangent: &DMatrix<T>,
    ) -> Result<DMatrix<T>>;
    
    /// Generate a random point on the manifold.
    ///
    /// # Returns
    /// Random point as a matrix
    fn random_point_matrix(&self) -> DMatrix<T>;
    
    /// Generate a random tangent vector at a point.
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as matrix)
    ///
    /// # Returns
    /// Random tangent vector as a matrix
    fn random_tangent_matrix(&self, point: &DMatrix<T>) -> Result<DMatrix<T>>;
    
    /// Check if a matrix is on the manifold.
    ///
    /// # Arguments
    /// * `matrix` - Matrix to check
    /// * `tolerance` - Numerical tolerance
    ///
    /// # Returns
    /// `true` if the matrix satisfies manifold constraints within tolerance
    fn is_point_on_manifold_matrix(&self, matrix: &DMatrix<T>, tolerance: T) -> bool;
    
    /// Check if a matrix is in the tangent space at a point.
    ///
    /// # Arguments
    /// * `point` - Point on the manifold (as matrix)
    /// * `tangent` - Potential tangent vector (as matrix)
    /// * `tolerance` - Numerical tolerance
    ///
    /// # Returns
    /// `true` if the matrix is in the tangent space within tolerance
    fn is_vector_in_tangent_space_matrix(
        &self,
        point: &DMatrix<T>,
        tangent: &DMatrix<T>,
        tolerance: T,
    ) -> bool;
    
    // Conversion methods to/from vector representation
    
    /// Convert a matrix to vector representation.
    ///
    /// This is used for compatibility with the base `Manifold` trait.
    fn matrix_to_vec(&self, matrix: &DMatrix<T>) -> DVector<T> {
        DVector::from_vec(matrix.as_slice().to_vec())
    }
    
    /// Convert a vector to matrix representation.
    ///
    /// This is used for compatibility with the base `Manifold` trait.
    fn vec_to_matrix(&self, vec: &DVector<T>) -> Result<DMatrix<T>> {
        let (n, p) = self.matrix_dims();
        if vec.len() != n * p {
            return Err(crate::error::ManifoldError::dimension_mismatch(
                format!("{}×{}", n, p),
                format!("{}", vec.len()),
            ));
        }
        Ok(DMatrix::from_vec(n, p, vec.as_slice().to_vec()))
    }
}

/// Extension trait that provides default implementations for the base `Manifold` trait
/// in terms of `MatrixManifold` operations.
///
/// This allows matrix manifolds to implement only the matrix-based methods
/// while automatically getting vector-based methods.
pub trait MatrixManifoldExt<T: Scalar>: MatrixManifold<T> {
    /// Project a vector by converting to matrix, projecting, and converting back.
    fn project_vec(&self, vec: &DVector<T>) -> Result<DVector<T>> {
        let matrix = self.vec_to_matrix(vec)?;
        let projected = self.project_matrix(&matrix)?;
        Ok(self.matrix_to_vec(&projected))
    }
    
    /// Retract a tangent vector by converting to matrices, retracting, and converting back.
    fn retract_vec(&self, point: &DVector<T>, tangent: &DVector<T>) -> Result<DVector<T>> {
        let point_mat = self.vec_to_matrix(point)?;
        let tangent_mat = self.vec_to_matrix(tangent)?;
        let result = self.retract_matrix(&point_mat, &tangent_mat)?;
        Ok(self.matrix_to_vec(&result))
    }
    
    // ... similar implementations for other methods
}

// Implement the extension trait for all types that implement MatrixManifold
impl<T: Scalar, M: MatrixManifold<T>> MatrixManifoldExt<T> for M {}