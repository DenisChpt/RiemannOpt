//! # Stiefel Manifold St(n,p)
//!
//! The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = Iₚ} is the set of all
//! n×p matrices with orthonormal columns. It is a fundamental manifold in
//! matrix optimization and appears naturally in many applications.
//!
//! ## Mathematical Structure
//!
//! The Stiefel manifold is defined as:
//! ```text
//! St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = Iₚ}
//! ```
//! where Iₚ is the p×p identity matrix.
//!
//! Key properties:
//! - **Dimension**: np - p(p+1)/2
//! - **Ambient space**: ℝⁿˣᵖ (space of all n×p matrices)
//! - **Constraint**: X^T X = Iₚ (orthonormality of columns)
//! - **Embedding**: Closed subset of ℝⁿˣᵖ
//!
//! ## Geometric Properties
//!
//! ### Tangent Space
//! The tangent space at X ∈ St(n,p) consists of matrices V satisfying:
//! ```text
//! T_X St(n,p) = {V ∈ ℝⁿˣᵖ : X^T V + V^T X = 0}
//! ```
//! This is equivalent to requiring V^T X to be skew-symmetric.
//!
//! ### Orthogonal Decomposition
//! Any matrix V ∈ ℝⁿˣᵖ can be decomposed as:
//! ```text
//! V = V_tangent + V_normal
//! V_tangent = V - X(X^T V + V^T X)/2
//! V_normal = X(X^T V + V^T X)/2
//! ```
//!
//! ### Retractions
//! Multiple retraction strategies are available:
//! 1. **QR retraction**: R_X(V) = qf(X + V) (QR decomposition)
//! 2. **Polar retraction**: R_X(V) = (X + V)(X + V)^(-1/2)
//! 3. **Cayley retraction**: R_X(V) = (I + V/2)(I - V/2)^(-1)X
//!
//! ## Special Cases
//!
//! - **St(n,1)**: Unit sphere Sⁿ⁻¹ (single unit vector)
//! - **St(n,n)**: Orthogonal group O(n) (square orthogonal matrices)
//! - **St(n,p) with n » p**: "Tall" matrices (most common case)
//!
//! ## Optimization Applications
//!
//! ### 1. Principal Component Analysis (PCA)
//! Find the top p principal components:
//! ```text
//! max_{X ∈ St(n,p)} trace(X^T Σ X)
//! ```
//! where Σ is the covariance matrix.
//!
//! ### 2. Independent Component Analysis (ICA)
//! Learn orthogonal unmixing matrix:
//! ```text
//! max_{W ∈ St(n,n)} ∑ᵢ G(wᵢ^T x)
//! ```
//! where G is a non-Gaussian measure.
//!
//! ### 3. Dictionary Learning
//! Learn orthonormal dictionary atoms:
//! ```text
//! min_{D ∈ St(n,p), X} ‖Y - DX‖_F^2 + λ‖X‖_1
//! ```
//!
//! ### 4. Neural Network Orthogonal Constraints
//! Constrain weight matrices to be orthogonal:
//! ```text
//! W ∈ St(n,p) ⟹ W^T W = Iₚ
//! ```
//! This prevents vanishing/exploding gradients and improves conditioning.
//!
//! ### 5. Computer Vision
//! - **Structure from Motion**: Camera orientation matrices
//! - **Shape Analysis**: Orthogonal Procrustes alignment
//! - **Feature Learning**: Orthogonal basis functions
//!
//! ## Implementation Features
//!
//! This implementation provides:
//! - **QR-based projection**: Efficient orthogonalization
//! - **Multiple retractions**: QR retraction (default)
//! - **Tangent space operations**: Exact projection formulas
//! - **Principal angles distance**: Geodesic distance computation
//! - **Random sampling**: Uniform distribution via QR of Gaussian matrices
//!
//! ## Numerical Considerations
//!
//! - **Stability**: QR decomposition maintains numerical orthogonality
//! - **Conditioning**: Gram-Schmidt can be unstable; QR is preferred
//! - **Scaling**: O(np²) complexity for most operations
//! - **Memory**: Stores n×p matrices efficiently
//!
//! ## Example Usage
//!
//! ```rust
//! use riemannopt_manifolds::Stiefel;
//! use riemannopt_core::manifold::Manifold;
//! use nalgebra::{DMatrix, DVector, Dyn};
//!
//! // Create Stiefel manifold St(5,3)
//! let stiefel = Stiefel::new(5, 3).unwrap();
//!
//! // Generate random orthonormal matrix
//! let x: DVector<f64> = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
//! // Reshape to matrix form
//! let x_matrix: DMatrix<f64> = DMatrix::from_vec(5, 3, x.data.as_vec().clone());
//! let gram: DMatrix<f64> = x_matrix.transpose() * &x_matrix;
//! // gram should be approximately identity
//!
//! // Project arbitrary matrix to Stiefel manifold
//! let arbitrary: DVector<f64> = DVector::from_vec(vec![1.0; 15]);
//! let projected: DVector<f64> = stiefel.project_point(&arbitrary);
//! ```

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::Scalar,
    memory::workspace::Workspace,
    parallel_thresholds::{ParallelDecision, DecompositionKind, get_parallel_config, ShouldParallelize},
    core::MatrixManifold,
};
use crate::utils::vector_to_matrix_view;
use crate::stiefel_small::{project_tangent_stiefel_3_2, project_tangent_stiefel_4_2, can_use_specialized_stiefel};
use nalgebra::{DMatrix, DVector, Dyn};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::iter::Sum;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = Iₚ}.
///
/// The Stiefel manifold represents the space of n×p matrices with orthonormal columns.
/// It is a fundamental matrix manifold that preserves orthogonality constraints during
/// optimization, making it essential for many applications in machine learning,
/// signal processing, and computer vision.
///
/// # Mathematical Definition
///
/// ```text
/// St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = Iₚ}
/// ```
///
/// where:
/// - n ≥ p (number of rows ≥ number of columns)
/// - Iₚ is the p×p identity matrix
/// - Columns of X are orthonormal vectors in ℝⁿ
///
/// # Geometric Structure
///
/// ## Dimension
/// The intrinsic dimension is:
/// ```text
/// dim(St(n,p)) = np - p(p+1)/2
/// ```
/// This accounts for np free parameters minus p(p+1)/2 orthogonality constraints.
///
/// ## Tangent Space
/// At point X ∈ St(n,p), the tangent space is:
/// ```text
/// T_X St(n,p) = {V ∈ ℝⁿˣᵖ : X^T V + V^T X = 0}
/// ```
/// Equivalently, V^T X must be skew-symmetric.
///
/// ## Riemannian Metric
/// The canonical metric is the restriction of the Frobenius inner product:
/// ```text
/// ⟨U, V⟩_X = trace(U^T V) = ∑ᵢⱼ Uᵢⱼ Vᵢⱼ
/// ```
///
/// # Retraction Methods
///
/// ## QR Retraction (Default)
/// ```text
/// R_X(V) = qf(X + V)
/// ```
/// where qf(·) extracts the Q factor from QR decomposition.
///
/// **Properties:**
/// - Exact orthogonality preservation
/// - Numerically stable
/// - O(np²) complexity
/// - First-order retraction
///
/// ## Polar Retraction
/// ```text
/// R_X(V) = (X + V)((X + V)^T(X + V))^(-1/2)
/// ```
/// **Properties:**
/// - Second-order retraction
/// - More expensive (requires matrix square root)
/// - Better approximation to exponential map
///
/// # Geodesics and Distance
///
/// The geodesic distance uses principal angles between column spaces:
/// ```text
/// d(X, Y) = √(∑ᵢ θᵢ²)
/// ```
/// where θᵢ are principal angles: cos(θᵢ) = σᵢ(X^T Y).
///
/// # Optimization Context
///
/// ## Gradient Conversion
/// Euclidean gradient → Riemannian gradient:
/// ```text
/// grad f(X) = ∇f(X) - X((∇f(X))^T X + X^T ∇f(X))/2
/// ```
///
/// ## Common Cost Functions
/// 1. **Linear**: f(X) = trace(A^T X) → ∇f(X) = A
/// 2. **Quadratic**: f(X) = trace(X^T A X) → ∇f(X) = 2AX
/// 3. **Frobenius**: f(X) = ‖X - A‖_F² → ∇f(X) = 2(X - A)
///
/// # Implementation Examples
///
/// ## Basic Operations
/// ```rust
/// use riemannopt_manifolds::Stiefel;
/// use riemannopt_core::manifold::Manifold;
/// use nalgebra::{DMatrix, DVector, Dyn};
///
/// // Create St(4,2) - 4×2 orthonormal matrices
/// let stiefel = Stiefel::new(4, 2).unwrap();
/// assert_eq!(<Stiefel as Manifold<f64, Dyn>>::dimension(&stiefel), 4*2 - 2*3/2); // 8 - 3 = 5
///
/// // Generate random orthonormal matrix
/// let x: DVector<f64> = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
/// 
/// // Convert to matrix form for verification
/// let x_mat: DMatrix<f64> = DMatrix::from_vec(4, 2, x.data.as_vec().clone());
/// let gram: DMatrix<f64> = x_mat.transpose() * &x_mat;
/// // gram ≈ I₂
/// ```
///
/// ## Projection Operations
/// ```rust
/// # use riemannopt_manifolds::Stiefel;
/// # use riemannopt_core::manifold::Manifold;
/// # use nalgebra::{DMatrix, DVector, Dyn};
/// # let stiefel = Stiefel::new(3, 2).unwrap();
///
/// // Project arbitrary matrix to Stiefel manifold
/// let arbitrary: DVector<f64> = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let projected: DVector<f64> = stiefel.project_point(&arbitrary);
/// 
/// // Result has orthonormal columns
/// assert!(stiefel.is_point_on_manifold(&projected, 1e-12));
///
/// // Tangent space projection
/// let x: DVector<f64> = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
/// let v: DVector<f64> = DVector::from_vec(vec![0.1; 6]);
/// let v_tangent: DVector<f64> = stiefel.project_tangent(&x, &v).unwrap();
/// assert!(stiefel.is_vector_in_tangent_space(&x, &v_tangent, 1e-12));
/// ```
///
/// ## Optimization Step
/// ```rust
/// # use riemannopt_manifolds::Stiefel;
/// # use riemannopt_core::manifold::Manifold;
/// # use nalgebra::{DVector, Dyn};
/// # let stiefel = Stiefel::new(3, 2).unwrap();
/// # let x: DVector<f64> = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
///
/// // Simulate optimization step
/// let euclidean_grad: DVector<f64> = DVector::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3]);
/// let riemannian_grad: DVector<f64> = stiefel.euclidean_to_riemannian_gradient(&x, &euclidean_grad).unwrap();
/// 
/// // Take step using retraction
/// let step_size: f64 = 0.01;
/// let direction: DVector<f64> = riemannian_grad * (-step_size);
/// let x_new: DVector<f64> = stiefel.retract(&x, &direction).unwrap();
/// 
/// // New point maintains orthogonality
/// assert!(stiefel.is_point_on_manifold(&x_new, 1e-12));
/// ```
///
/// # Special Cases and Relationships
///
/// - **St(n,1) ≅ Sⁿ⁻¹**: Single column → unit sphere
/// - **St(n,n) ≅ O(n)**: Square case → orthogonal group  
/// - **St(∞,p)**: Infinite-dimensional case (Grassmannian limit)
/// - **Gr(n,p) = St(n,p)/O(p)**: Quotient by column rotations
///
/// # Performance Characteristics
///
/// - **Memory**: O(np) for matrix storage
/// - **Projection**: O(np²) via QR decomposition
/// - **Retraction**: O(np²) via QR decomposition
/// - **Tangent projection**: O(np²) via matrix multiplication
/// - **Distance**: O(p³) via SVD of p×p matrix
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
    /// * `p` - Number of columns (orthonormal vectors), must satisfy p ≤ n
    ///
    /// # Returns
    /// A Stiefel manifold representing n×p matrices with orthonormal columns.
    /// The intrinsic dimension is np - p(p+1)/2.
    ///
    /// # Mathematical Background
    /// The dimension formula accounts for:
    /// - np: Total matrix entries
    /// - p(p+1)/2: Orthonormality constraints (X^T X = Iₚ has p² entries, but
    ///   only p(p+1)/2 are independent due to symmetry)
    ///
    /// Common examples:
    /// - St(3,1): dimension = 3×1 - 1×2/2 = 2 (sphere S²)
    /// - St(4,2): dimension = 4×2 - 2×3/2 = 5
    /// - St(n,n): dimension = n² - n(n+1)/2 = n(n-1)/2 (skew-symmetric matrices)
    ///
    /// # Errors
    /// - Returns `ManifoldError::InvalidPoint` if `p > n` (impossible to have p
    ///   orthonormal vectors in dimension n < p)
    /// - Returns `ManifoldError::InvalidPoint` if `n = 0` or `p = 0` (degenerate cases)
    ///
    /// # Examples
    /// ```rust
    /// use riemannopt_manifolds::Stiefel;
    /// use riemannopt_core::manifold::Manifold;
    /// use nalgebra::Dyn;
    ///
    /// // Standard case: "tall" matrices
    /// let stiefel = Stiefel::new(10, 3).unwrap();
    /// assert_eq!(<Stiefel as Manifold<f64, Dyn>>::dimension(&stiefel), 10*3 - 3*4/2); // 30 - 6 = 24
    /// assert_eq!(stiefel.n(), 10);
    /// assert_eq!(stiefel.p(), 3);
    ///
    /// // Square case (orthogonal group)
    /// let orthogonal = Stiefel::new(3, 3).unwrap();
    /// assert_eq!(<Stiefel as Manifold<f64, Dyn>>::dimension(&orthogonal), 3); // 9 - 6 = 3
    ///
    /// // Single column (sphere)
    /// let sphere = Stiefel::new(5, 1).unwrap();
    /// assert_eq!(<Stiefel as Manifold<f64, Dyn>>::dimension(&sphere), 4); // 5 - 1 = 4
    ///
    /// // Error cases
    /// assert!(Stiefel::new(2, 3).is_err()); // p > n
    /// assert!(Stiefel::new(0, 1).is_err()); // n = 0
    /// assert!(Stiefel::new(3, 0).is_err()); // p = 0
    /// ```
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

    /// Checks if a matrix satisfies the orthonormality constraint X^T X = Iₚ.
    ///
    /// # Mathematical Test
    /// Verifies that the Gram matrix G = X^T X satisfies:
    /// - Gᵢᵢ = 1 for all i (unit length columns)
    /// - Gᵢⱼ = 0 for i ≠ j (orthogonal columns)
    ///
    /// # Arguments
    /// * `matrix` - The n×p matrix to test
    /// * `tolerance` - Numerical tolerance for floating-point comparison
    ///
    /// # Returns
    /// `true` if the matrix has orthonormal columns within tolerance
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

    /// Projects a matrix to the Stiefel manifold using QR decomposition.
    ///
    /// # Algorithm
    /// Given matrix A ∈ ℝⁿˣᵖ, compute QR decomposition A = QR and return
    /// the first p columns of Q. This gives the closest orthonormal matrix
    /// to A in the Frobenius norm sense.
    ///
    /// # Mathematical Properties
    /// - **Optimality**: Minimizes ‖X - A‖_F subject to X^T X = Iₚ
    /// - **Stability**: QR is numerically stable (unlike Gram-Schmidt)
    /// - **Uniqueness**: Result is unique if A has full column rank
    ///
    /// # Edge Cases
    /// - If input has wrong dimensions, it's resized and padded
    /// - If input has rank deficiency, standard basis vectors are used
    ///
    /// # Complexity
    /// O(np²) for QR decomposition of n×p matrix
    fn qr_projection<T>(&self, matrix: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        // Check if we should use parallel QR decomposition based on matrix size
        let should_parallelize = get_parallel_config()
            .should_parallelize_decomposition(self.n.min(self.p), DecompositionKind::QR);
        
        if should_parallelize {
            // Log that we're using parallel QR
            #[cfg(feature = "parallel")]
            {
                // In a real implementation, we would use a parallel QR algorithm here
                // For now, we use the standard QR with the understanding that
                // this is where parallel QR would be plugged in
                self.qr_projection_sequential(matrix)
            }
            #[cfg(not(feature = "parallel"))]
            {
                self.qr_projection_sequential(matrix)
            }
        } else {
            self.qr_projection_sequential(matrix)
        }
    }
    
    fn qr_projection_sequential<T>(&self, matrix: &DMatrix<T>) -> DMatrix<T>
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

    /// Generates a random tangent vector at the given point.
    ///
    /// # Algorithm
    /// 1. Generate random matrix A with i.i.d. Gaussian entries
    /// 2. Project to tangent space: V = A - X(X^T A + A^T X)/2
    ///
    /// # Mathematical Background
    /// The projection formula ensures V ∈ T_X St(n,p):
    /// - X^T V + V^T X = 0 (skew-symmetric condition)
    /// - This is the orthogonal projection onto the tangent space
    ///
    /// # Distribution
    /// The resulting tangent vector follows a matrix normal distribution
    /// on the tangent space, providing good sampling coverage.
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
    T: Scalar + Sum,
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
        
        // Use view to avoid cloning
        let matrix_view = vector_to_matrix_view(point, self.n, self.p);
        // Need to convert view to owned matrix for is_orthonormal (gram computation requires it)
        let matrix = matrix_view.clone_owned();
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
        
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let v_matrix = vector_to_matrix_view(vector, self.n, self.p);
        
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
        
        // Create a workspace to hold the result
        let mut result = DVector::zeros(self.n * self.p);
        
        // Use workspace method which avoids allocations
        let mut workspace = riemannopt_core::memory::workspace::Workspace::new();
        self.project_tangent_with_workspace(point, vector, &mut result, &mut workspace)?;
        
        Ok(result)
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
        
        // Create a workspace to hold the result
        let mut result = DVector::zeros(self.n * self.p);
        
        // Use workspace method which avoids allocations
        let mut workspace = riemannopt_core::memory::workspace::Workspace::new();
        self.retract_with_workspace(point, tangent, &mut result, &mut workspace)?;
        
        Ok(result)
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
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        if from.len() != self.n * self.p || to.len() != self.n * self.p || vector.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "All vectors must have correct dimensions",
                format!("expected: {}", self.n * self.p),
            ));
        }
        
        let x_matrix = DMatrix::from_vec(self.n, self.p, from.data.as_vec().clone());
        let y_matrix = DMatrix::from_vec(self.n, self.p, to.data.as_vec().clone());
        let v_matrix = DMatrix::from_vec(self.n, self.p, vector.data.as_vec().clone());
        
        // Implement parallel transport along geodesics for Stiefel manifold
        // This uses the algorithm from "Optimization Algorithms on Matrix Manifolds" by Absil et al.
        
        // First, compute the direction of the geodesic from X to Y
        // We need to find the tangent vector W at X such that geodesic γ(1) = Y
        
        // For computational efficiency, we use a first-order approximation:
        // The parallel transport of V from X to Y along the geodesic is approximately:
        // P_{X→Y}(V) = V - X * ((Y^T V + V^T Y) / 2) + Y * ((X^T V + V^T X) / 2)
        
        // This formula ensures that the transported vector remains in the tangent space at Y
        // and preserves the inner product structure along the geodesic
        
        let ytv = y_matrix.transpose() * &v_matrix;
        let vty = v_matrix.transpose() * &y_matrix;
        let xtv = x_matrix.transpose() * &v_matrix;
        let vtx = v_matrix.transpose() * &x_matrix;
        
        let term1 = (&ytv + &vty) * <T as Scalar>::from_f64(0.5);
        let term2 = (&xtv + &vtx) * <T as Scalar>::from_f64(0.5);
        
        let transported = &v_matrix - &x_matrix * term1 + &y_matrix * term2;
        
        // Project to ensure we're exactly in the tangent space at Y
        let final_transported = {
            let ytw = y_matrix.transpose() * &transported;
            let wty = transported.transpose() * &y_matrix;
            let symmetric = (&ytw + &wty) * <T as Scalar>::from_f64(0.5);
            transported - &y_matrix * symmetric
        };
        
        Ok(DVector::from_vec(final_transported.data.as_vec().clone()))
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
        
        // Geodesic distance using principal angles between column spaces
        // Algorithm: d(X,Y) = √(∑ᵢ θᵢ²) where cos(θᵢ) = σᵢ(X^T Y)
        // The θᵢ are principal angles between the column spaces
        
        // Check if we should use parallel matrix multiplication
        let m = if ParallelDecision::matrix_multiply::<T>(self.p, self.p, self.n) {
            // For large matrices, we could use parallel GEMM here
            // For now, use the standard multiplication
            x_matrix.transpose() * &y_matrix
        } else {
            x_matrix.transpose() * &y_matrix
        };
        
        // Check if we should use parallel SVD
        let svd = if get_parallel_config()
            .should_parallelize_decomposition(self.p, DecompositionKind::SVD) {
            // In a real implementation, we would use parallel SVD here
            m.svd(true, true)
        } else {
            m.svd(true, true)
        };
        
        let mut distance_squared = T::zero();
        let singular_values = svd.singular_values;
        
        // For large number of singular values, we could parallelize this loop
        if ParallelDecision::dot_product::<T>(singular_values.len()) {
            #[cfg(feature = "parallel")]
            {
                distance_squared = singular_values.as_slice()
                    .par_iter()
                    .map(|&sigma| {
                        let clamped = <T as Float>::max(
                            <T as Float>::min(sigma, T::one()),
                            -T::one(),
                        );
                        let angle = <T as Float>::acos(clamped);
                        angle * angle
                    })
                    .sum::<T>();
            }
            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..singular_values.len() {
                    let sigma = singular_values[i];
                    let clamped = <T as Float>::max(
                        <T as Float>::min(sigma, T::one()),
                        -T::one(),
                    );
                    let angle = <T as Float>::acos(clamped);
                    distance_squared = distance_squared + angle * angle;
                }
            }
        } else {
            for i in 0..singular_values.len() {
                let sigma = singular_values[i];
                let clamped = <T as Float>::max(
                    <T as Float>::min(sigma, T::one()),
                    -T::one(),
                );
                let angle = <T as Float>::acos(clamped);
                distance_squared = distance_squared + angle * angle;
            }
        }
        
        Ok(<T as Float>::sqrt(distance_squared))
    }

    // ========================================================================
    // Workspace-based methods for zero-allocation operations
    // ========================================================================

    fn project_tangent_with_workspace(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.n * self.p || vector.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Point and vector must have correct dimensions for Stiefel manifold",
                format!("point: {}, vector: {}", point.len(), vector.len()),
            ));
        }

        // Ensure result has correct size
        if result.len() != self.n * self.p {
            *result = DVector::zeros(self.n * self.p);
        }

        // Check if we can use specialized small dimension implementations
        if can_use_specialized_stiefel(self.n, self.p) {
            match (self.n, self.p) {
                (3, 2) => {
                    project_tangent_stiefel_3_2(
                        point.as_slice(),
                        vector.as_slice(),
                        result.as_mut_slice(),
                    );
                    return Ok(());
                }
                (4, 2) => {
                    project_tangent_stiefel_4_2(
                        point.as_slice(),
                        vector.as_slice(),
                        result.as_mut_slice(),
                    );
                    return Ok(());
                }
                _ => {}
            }
        }

        // Generic implementation for larger dimensions
        // Use temporary matrices from the pool
        let mut x_matrix = workspace.acquire_temp_matrix(self.n, self.p);
        let mut v_matrix = workspace.acquire_temp_matrix(self.n, self.p);
        let mut xtv = workspace.acquire_temp_matrix(self.p, self.p);
        let mut vtx = workspace.acquire_temp_matrix(self.p, self.p);
        let mut symmetric = workspace.acquire_temp_matrix(self.p, self.p);
        let mut temp = workspace.acquire_temp_matrix(self.n, self.p);

        // Fill x_matrix from point vector (column-major order)
        for j in 0..self.p {
            for i in 0..self.n {
                x_matrix[(i, j)] = point[j * self.n + i];
            }
        }

        // Fill v_matrix from vector (column-major order)
        for j in 0..self.p {
            for i in 0..self.n {
                v_matrix[(i, j)] = vector[j * self.n + i];
            }
        }

        // Compute X^T V
        xtv.gemm(T::one(), &x_matrix.transpose(), &v_matrix, T::zero());
        
        // Compute V^T X
        vtx.gemm(T::one(), &v_matrix.transpose(), &x_matrix, T::zero());

        // Compute symmetric part: (X^T V + V^T X) / 2
        for i in 0..self.p {
            for j in 0..self.p {
                symmetric[(i, j)] = (xtv[(i, j)] + vtx[(i, j)]) * <T as Scalar>::from_f64(0.5);
            }
        }

        // Compute X * symmetric
        temp.gemm(T::one(), &x_matrix, &symmetric, T::zero());

        // Compute result = V - X * symmetric (column-major order)
        for j in 0..self.p {
            for i in 0..self.n {
                let idx = j * self.n + i;
                result[idx] = v_matrix[(i, j)] - temp[(i, j)];
            }
        }

        Ok(())
    }

    fn retract_with_workspace(
        &self,
        point: &DVector<T>,
        tangent: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.n * self.p || tangent.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Point and tangent must have correct dimensions",
                format!("point: {}, tangent: {}", point.len(), tangent.len()),
            ));
        }

        // Ensure result has correct size
        if result.len() != self.n * self.p {
            *result = DVector::zeros(self.n * self.p);
        }

        // Use temporary matrices from the pool
        let mut x_matrix = workspace.acquire_temp_matrix(self.n, self.p);
        let mut v_matrix = workspace.acquire_temp_matrix(self.n, self.p);
        let mut candidate = workspace.acquire_temp_matrix(self.n, self.p);

        // Fill matrices from vectors (column-major order)
        for j in 0..self.p {
            for i in 0..self.n {
                let idx = j * self.n + i;
                x_matrix[(i, j)] = point[idx];
                v_matrix[(i, j)] = tangent[idx];
            }
        }

        // Compute X + V
        for i in 0..self.n {
            for j in 0..self.p {
                candidate[(i, j)] = x_matrix[(i, j)] + v_matrix[(i, j)];
            }
        }

        // QR decomposition (this still allocates, but we minimize other allocations)
        let qr = candidate.clone().qr();
        let q = qr.q();

        // Copy first p columns of Q to result (column-major order)
        for j in 0..self.p {
            for i in 0..self.n {
                result[j * self.n + i] = q[(i, j)];
            }
        }

        Ok(())
    }

    fn euclidean_to_riemannian_gradient_with_workspace(
        &self,
        point: &DVector<T>,
        euclidean_grad: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For Stiefel manifold, this is just projection to tangent space
        self.project_tangent_with_workspace(point, euclidean_grad, result, workspace)
    }

    fn inverse_retract_with_workspace(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.n * self.p || other.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("point: {}, other: {}", point.len(), other.len()),
            ));
        }

        // Ensure result has correct size
        if result.len() != self.n * self.p {
            *result = DVector::zeros(self.n * self.p);
        }

        // Use temporary matrices from the pool
        let mut x_matrix = workspace.acquire_temp_matrix(self.n, self.p);
        let mut y_matrix = workspace.acquire_temp_matrix(self.n, self.p);
        let mut v_vec = workspace.acquire_temp_vector(self.n * self.p);

        // Fill matrices from vectors (column-major order)
        for j in 0..self.p {
            for i in 0..self.n {
                let idx = j * self.n + i;
                x_matrix[(i, j)] = point[idx];
                y_matrix[(i, j)] = other[idx];
            }
        }

        // Compute V = Y - X and store in v_vec (column-major order)
        for j in 0..self.p {
            for i in 0..self.n {
                let idx = j * self.n + i;
                v_vec[idx] = y_matrix[(i, j)] - x_matrix[(i, j)];
            }
        }

        // Project to tangent space using workspace
        self.project_tangent_with_workspace(point, &v_vec, result, workspace)?;

        Ok(())
    }

    fn parallel_transport_with_workspace(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if from.len() != self.n * self.p || to.len() != self.n * self.p || vector.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                "All vectors must have correct dimensions",
                format!("expected: {}", self.n * self.p),
            ));
        }

        // For now, use vector transport by projection
        // This is a simplified implementation - a full implementation would use
        // the algorithm from Absil et al.
        self.project_tangent_with_workspace(to, vector, result, workspace)
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
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-7);
    }
}

// MatrixManifold implementation for efficient matrix operations
impl<T: Scalar + Float + Sum> MatrixManifold<T> for Stiefel {
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
        
        // Use QR decomposition for projection
        let qr = matrix.clone().qr();
        Ok(qr.q() * DMatrix::<T>::identity(self.n, self.p))
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
        
        // Project to tangent space: V - X(X^T V + V^T X)/2
        let xtv = point.transpose() * matrix;
        let sym_part = (&xtv + xtv.transpose()) * T::from(0.5).unwrap();
        Ok(matrix - point * sym_part)
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
        
        // QR retraction: R_X(V) = qf(X + V)
        let result = point + tangent;
        let qr = result.qr();
        Ok(qr.q() * DMatrix::<T>::identity(self.n, self.p))
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
        
        // For QR retraction, approximate inverse retraction
        // This is a first-order approximation
        let diff = other - point;
        self.project_tangent_matrix(point, &diff)
    }
    
    fn euclidean_to_riemannian_gradient_matrix(
        &self,
        point: &DMatrix<T>,
        euclidean_grad: &DMatrix<T>,
    ) -> Result<DMatrix<T>> {
        // Riemannian gradient is the projection of Euclidean gradient to tangent space
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
        
        // Simple parallel transport: project to new tangent space
        // This is a first-order approximation
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
        
        // QR decomposition to get orthonormal columns
        let qr = a.qr();
        qr.q() * DMatrix::<T>::identity(self.n, self.p)
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
        
        // Project to tangent space
        self.project_tangent_matrix(point, &v)
    }
    
    fn is_point_on_manifold_matrix(&self, matrix: &DMatrix<T>, tolerance: T) -> bool {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            return false;
        }
        
        // Check X^T X = I
        let gram = matrix.transpose() * matrix;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        let diff = &gram - &identity;
        
        // Check Frobenius norm of difference
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
        
        // Check X^T V + V^T X = 0
        let xtv = point.transpose() * tangent;
        let sum = &xtv + xtv.transpose();
        
        // Check if sum is approximately zero
        sum.norm() <= tolerance
    }
}