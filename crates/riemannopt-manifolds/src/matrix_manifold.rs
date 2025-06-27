//! Matrix Manifold trait for manifolds that naturally work with matrix representations.
//!
//! This module provides a trait `MatrixManifold` that allows manifolds to operate
//! directly on matrix representations, avoiding the costly conversions between
//! vectors and matrices that plague the base `Manifold` trait.
//!
//! # Motivation
//!
//! Many manifolds in optimization are naturally matrix manifolds:
//! - Stiefel manifold St(n,p): n×p matrices with orthonormal columns
//! - Grassmann manifold Gr(n,p): equivalence classes of n×p orthonormal matrices
//! - SPD manifold SPD(n): n×n symmetric positive definite matrices
//! - Fixed-rank manifold: n×m matrices of fixed rank r
//!
//! The base `Manifold` trait operates on flat vectors, requiring constant conversions
//! that allocate memory and copy data. This trait eliminates these inefficiencies.

use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};
use num_traits::Float;

use riemannopt_core::{
    error::Result,
    memory::Workspace,
    types::Scalar,
};

/// A manifold that naturally operates on matrix representations.
///
/// This trait is designed for manifolds whose elements are naturally represented
/// as matrices. It provides all the same operations as the base `Manifold` trait
/// but operates directly on matrix types, avoiding vectorization overhead.
///
/// # Type Parameters
///
/// * `T` - The scalar type (f32 or f64)
/// * `R` - Number of rows in the matrix representation
/// * `C` - Number of columns in the matrix representation
pub trait MatrixManifold<T: Scalar> {
    /// Returns the name of this manifold.
    fn name(&self) -> &str;

    /// Returns the number of rows in the matrix representation.
    fn nrows(&self) -> usize;

    /// Returns the number of columns in the matrix representation.
    fn ncols(&self) -> usize;

    /// Returns the intrinsic dimension of the manifold.
    ///
    /// This is the dimension of the tangent space, which may be less than
    /// the ambient dimension (nrows * ncols).
    fn dimension(&self) -> usize;

    /// Checks if a matrix represents a valid point on the manifold.
    ///
    /// # Arguments
    ///
    /// * `point` - A matrix to check
    /// * `tolerance` - Numerical tolerance for the check
    ///
    /// # Returns
    ///
    /// `true` if the matrix is on the manifold within the given tolerance.
    fn is_point_on_manifold(&self, point: &DMatrix<T>, tolerance: T) -> bool;

    /// Checks if a matrix represents a valid tangent vector.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `vector` - A matrix to check
    /// * `tolerance` - Numerical tolerance for the check
    ///
    /// # Returns
    ///
    /// `true` if the matrix is in the tangent space at `point`.
    fn is_vector_in_tangent_space(
        &self,
        point: &DMatrix<T>,
        vector: &DMatrix<T>,
        tolerance: T,
    ) -> bool;

    /// Projects a matrix onto the manifold.
    ///
    /// # Arguments
    ///
    /// * `matrix` - An arbitrary matrix
    /// * `result` - Output matrix that will contain the projection
    /// * `workspace` - Workspace for temporary allocations
    fn project_point(
        &self,
        matrix: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    );

    /// Projects a matrix onto the tangent space at a point.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `vector` - An arbitrary matrix
    /// * `result` - Output matrix that will contain the projection
    /// * `workspace` - Workspace for temporary allocations
    fn project_tangent(
        &self,
        point: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Computes the Riemannian inner product between two tangent vectors.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `u` - First tangent vector
    /// * `v` - Second tangent vector
    ///
    /// # Returns
    ///
    /// The inner product ⟨u,v⟩_point.
    fn inner_product(
        &self,
        point: &DMatrix<T>,
        u: &DMatrix<T>,
        v: &DMatrix<T>,
    ) -> Result<T>;

    /// Computes the Riemannian norm of a tangent vector.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `vector` - A tangent vector
    ///
    /// # Returns
    ///
    /// The norm ||vector||_point.
    fn norm(&self, point: &DMatrix<T>, vector: &DMatrix<T>) -> Result<T> {
        self.inner_product(point, vector, vector)
            .map(|ip| <T as Float>::sqrt(ip))
    }

    /// Performs a retraction from the tangent space to the manifold.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `tangent` - A tangent vector at `point`
    /// * `result` - Output matrix for the new point
    /// * `workspace` - Workspace for temporary allocations
    fn retract(
        &self,
        point: &DMatrix<T>,
        tangent: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Computes the inverse retraction (logarithmic map).
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `other` - Another point on the manifold
    /// * `result` - Output matrix for the tangent vector
    /// * `workspace` - Workspace for temporary allocations
    fn inverse_retract(
        &self,
        point: &DMatrix<T>,
        other: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Converts a Euclidean gradient to a Riemannian gradient.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `euclidean_grad` - The Euclidean gradient
    /// * `result` - Output matrix for the Riemannian gradient
    /// * `workspace` - Workspace for temporary allocations
    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DMatrix<T>,
        euclidean_grad: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Generates a random point on the manifold.
    fn random_point(&self) -> DMatrix<T>;

    /// Generates a random tangent vector at a point.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `result` - Output matrix for the random tangent vector
    /// * `workspace` - Workspace for temporary allocations
    fn random_tangent(
        &self,
        point: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Performs parallel transport of a vector along a geodesic.
    ///
    /// # Arguments
    ///
    /// * `from` - Starting point
    /// * `to` - Ending point
    /// * `vector` - Tangent vector to transport
    /// * `result` - Output matrix for the transported vector
    /// * `workspace` - Workspace for temporary allocations
    fn parallel_transport(
        &self,
        from: &DMatrix<T>,
        to: &DMatrix<T>,
        vector: &DMatrix<T>,
        result: &mut DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Computes the Riemannian distance between two points.
    ///
    /// # Arguments
    ///
    /// * `x` - First point
    /// * `y` - Second point
    /// * `workspace` - Workspace for temporary allocations
    ///
    /// # Returns
    ///
    /// The geodesic distance d(x,y).
    fn distance(
        &self,
        x: &DMatrix<T>,
        y: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<T>;

    /// Returns whether this manifold has an exact exponential map and logarithm.
    fn has_exact_exp_log(&self) -> bool {
        false
    }

    /// Projects a matrix onto the manifold using views (zero-copy when possible).
    ///
    /// # Arguments
    ///
    /// * `matrix` - View of an arbitrary matrix
    /// * `result` - Mutable view of output matrix
    /// * `workspace` - Workspace for temporary allocations
    fn project_point_view(
        &self,
        matrix: DMatrixView<T>,
        mut result: DMatrixViewMut<T>,
        workspace: &mut Workspace<T>,
    ) {
        // Default implementation: copy to owned matrices
        let matrix_owned = matrix.clone_owned();
        let mut result_owned = DMatrix::zeros(self.nrows(), self.ncols());
        self.project_point(&matrix_owned, &mut result_owned, workspace);
        result.copy_from(&result_owned);
    }

    /// Projects a matrix onto the tangent space using views (zero-copy when possible).
    ///
    /// # Arguments
    ///
    /// * `point` - View of a point on the manifold
    /// * `vector` - View of an arbitrary matrix
    /// * `result` - Mutable view of output matrix
    /// * `workspace` - Workspace for temporary allocations
    fn project_tangent_view(
        &self,
        point: DMatrixView<T>,
        vector: DMatrixView<T>,
        mut result: DMatrixViewMut<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Default implementation: copy to owned matrices
        let point_owned = point.clone_owned();
        let vector_owned = vector.clone_owned();
        let mut result_owned = DMatrix::zeros(self.nrows(), self.ncols());
        self.project_tangent(&point_owned, &vector_owned, &mut result_owned, workspace)?;
        result.copy_from(&result_owned);
        Ok(())
    }
}

/// Extension trait to convert between vector and matrix representations.
///
/// This trait provides convenience methods for working with both the vector-based
/// `Manifold` trait and the matrix-based `MatrixManifold` trait.
///
/// # Performance Note
///
/// These conversion methods allocate memory. For performance-critical paths,
/// prefer using the `MatrixManifold` trait directly to avoid conversions.
pub trait MatrixManifoldExt<T: Scalar> {
    /// Converts a flat vector to a matrix representation.
    fn vector_to_matrix(&self, vector: &[T]) -> DMatrix<T>;

    /// Converts a matrix to a flat vector representation.
    fn matrix_to_vector(&self, matrix: &DMatrix<T>) -> Vec<T>;

    /// Gets the expected vector length for this manifold.
    fn vector_length(&self) -> usize;
}

/// Macro to implement the vector-based Manifold trait for types that implement MatrixManifold.
///
/// This macro generates a blanket implementation that handles the vector-matrix conversions
/// automatically, allowing matrix manifolds to be used seamlessly with the existing
/// optimization infrastructure.
///
/// # Performance Note
///
/// The generated implementation involves vector-matrix conversions that allocate memory.
/// For performance-critical applications, prefer using the `MatrixManifold` trait directly.
#[macro_export]
macro_rules! impl_manifold_for_matrix_manifold {
    ($type:ty) => {
        impl<T: Scalar> riemannopt_core::manifold::Manifold<T, nalgebra::Dyn> for $type
        where
            Self: MatrixManifold<T> + MatrixManifoldExt<T>,
        {
            fn name(&self) -> &str {
                MatrixManifold::name(self)
            }

            fn dimension(&self) -> usize {
                MatrixManifold::dimension(self)
            }

            fn ambient_dimension(&self) -> usize {
                self.vector_length()
            }

            fn is_point_on_manifold(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                tolerance: T,
            ) -> bool {
                if point.len() != self.vector_length() {
                    return false;
                }
                let matrix = self.vector_to_matrix(point.as_slice());
                MatrixManifold::is_point_on_manifold(self, &matrix, tolerance)
            }

            fn is_vector_in_tangent_space(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                vector: &riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                tolerance: T,
            ) -> bool {
                if point.len() != self.vector_length() || vector.len() != self.vector_length() {
                    return false;
                }
                let point_matrix = self.vector_to_matrix(point.as_slice());
                let vector_matrix = self.vector_to_matrix(vector.as_slice());
                MatrixManifold::is_vector_in_tangent_space(
                    self,
                    &point_matrix,
                    &vector_matrix,
                    tolerance,
                )
            }

            fn project_point(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                result: &mut riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) {
                let matrix = self.vector_to_matrix(point.as_slice());
                let mut result_matrix = workspace.acquire_temp_matrix(self.nrows(), self.ncols());
                MatrixManifold::project_point(self, &matrix, &mut result_matrix, workspace);
                let result_vec = self.matrix_to_vector(&result_matrix);
                result.copy_from_slice(&result_vec);
            }

            fn project_tangent(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                vector: &riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                result: &mut riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) -> riemannopt_core::error::Result<()> {
                let point_matrix = self.vector_to_matrix(point.as_slice());
                let vector_matrix = self.vector_to_matrix(vector.as_slice());
                let mut result_matrix = workspace.acquire_temp_matrix(self.nrows(), self.ncols());
                MatrixManifold::project_tangent(
                    self,
                    &point_matrix,
                    &vector_matrix,
                    &mut result_matrix,
                    workspace,
                )?;
                let result_vec = self.matrix_to_vector(&result_matrix);
                result.copy_from_slice(&result_vec);
                Ok(())
            }

            fn inner_product(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                u: &riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                v: &riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
            ) -> riemannopt_core::error::Result<T> {
                let point_matrix = self.vector_to_matrix(point.as_slice());
                let u_matrix = self.vector_to_matrix(u.as_slice());
                let v_matrix = self.vector_to_matrix(v.as_slice());
                MatrixManifold::inner_product(self, &point_matrix, &u_matrix, &v_matrix)
            }

            fn retract(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                tangent: &riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                result: &mut riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) -> riemannopt_core::error::Result<()> {
                let point_matrix = self.vector_to_matrix(point.as_slice());
                let tangent_matrix = self.vector_to_matrix(tangent.as_slice());
                let mut result_matrix = workspace.acquire_temp_matrix(self.nrows(), self.ncols());
                MatrixManifold::retract(
                    self,
                    &point_matrix,
                    &tangent_matrix,
                    &mut result_matrix,
                    workspace,
                )?;
                let result_vec = self.matrix_to_vector(&result_matrix);
                result.copy_from_slice(&result_vec);
                Ok(())
            }

            fn inverse_retract(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                other: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                result: &mut riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) -> riemannopt_core::error::Result<()> {
                let point_matrix = self.vector_to_matrix(point.as_slice());
                let other_matrix = self.vector_to_matrix(other.as_slice());
                let mut result_matrix = workspace.acquire_temp_matrix(self.nrows(), self.ncols());
                MatrixManifold::inverse_retract(
                    self,
                    &point_matrix,
                    &other_matrix,
                    &mut result_matrix,
                    workspace,
                )?;
                let result_vec = self.matrix_to_vector(&result_matrix);
                result.copy_from_slice(&result_vec);
                Ok(())
            }

            fn euclidean_to_riemannian_gradient(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                euclidean_grad: &riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                result: &mut riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) -> riemannopt_core::error::Result<()> {
                let point_matrix = self.vector_to_matrix(point.as_slice());
                let grad_matrix = self.vector_to_matrix(euclidean_grad.as_slice());
                let mut result_matrix = workspace.acquire_temp_matrix(self.nrows(), self.ncols());
                MatrixManifold::euclidean_to_riemannian_gradient(
                    self,
                    &point_matrix,
                    &grad_matrix,
                    &mut result_matrix,
                    workspace,
                )?;
                let result_vec = self.matrix_to_vector(&result_matrix);
                result.copy_from_slice(&result_vec);
                Ok(())
            }

            fn random_point(&self) -> riemannopt_core::manifold::Point<T, nalgebra::Dyn> {
                let matrix = MatrixManifold::random_point(self);
                let vec = self.matrix_to_vector(&matrix);
                nalgebra::DVector::from_vec(vec)
            }

            fn random_tangent(
                &self,
                point: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                result: &mut riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) -> riemannopt_core::error::Result<()> {
                let point_matrix = self.vector_to_matrix(point.as_slice());
                let mut result_matrix = workspace.acquire_temp_matrix(self.nrows(), self.ncols());
                MatrixManifold::random_tangent(
                    self,
                    &point_matrix,
                    &mut result_matrix,
                    workspace,
                )?;
                let result_vec = self.matrix_to_vector(&result_matrix);
                result.copy_from_slice(&result_vec);
                Ok(())
            }

            fn parallel_transport(
                &self,
                from: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                to: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                vector: &riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                result: &mut riemannopt_core::manifold::TangentVector<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) -> riemannopt_core::error::Result<()> {
                let from_matrix = self.vector_to_matrix(from.as_slice());
                let to_matrix = self.vector_to_matrix(to.as_slice());
                let vector_matrix = self.vector_to_matrix(vector.as_slice());
                let mut result_matrix = workspace.acquire_temp_matrix(self.nrows(), self.ncols());
                MatrixManifold::parallel_transport(
                    self,
                    &from_matrix,
                    &to_matrix,
                    &vector_matrix,
                    &mut result_matrix,
                    workspace,
                )?;
                let result_vec = self.matrix_to_vector(&result_matrix);
                result.copy_from_slice(&result_vec);
                Ok(())
            }

            fn distance(
                &self,
                x: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                y: &riemannopt_core::manifold::Point<T, nalgebra::Dyn>,
                workspace: &mut riemannopt_core::memory::Workspace<T>,
            ) -> riemannopt_core::error::Result<T> {
                let x_matrix = self.vector_to_matrix(x.as_slice());
                let y_matrix = self.vector_to_matrix(y.as_slice());
                MatrixManifold::distance(self, &x_matrix, &y_matrix, workspace)
            }

            fn has_exact_exp_log(&self) -> bool {
                MatrixManifold::has_exact_exp_log(self)
            }
        }
    };
}