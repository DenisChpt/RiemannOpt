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
//! use riemannopt_core::memory::workspace::Workspace;
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
//! let mut projected = DVector::<f64>::zeros(15);
//! let mut workspace = Workspace::new();
//! stiefel.project_point(&arbitrary, &mut projected, &mut workspace);
//! ```

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    types::{Scalar, DVector},
    compute::{get_dispatcher, SimdBackend},
    memory::Workspace,
};
use crate::utils::{vector_to_matrix_view, vector_to_matrix_view_mut};
// use crate::stiefel_small::{project_tangent_stiefel_3_2, project_tangent_stiefel_4_2, can_use_specialized_stiefel};
use nalgebra::{DMatrix, Dyn};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::iter::Sum;

// #[cfg(feature = "parallel")]
// use rayon::prelude::*;

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
/// # use riemannopt_core::memory::workspace::Workspace;
/// # use nalgebra::{DMatrix, DVector, Dyn};
/// # let stiefel = Stiefel::new(3, 2).unwrap();
///
/// // Project arbitrary matrix to Stiefel manifold
/// let arbitrary: DVector<f64> = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let mut projected = DVector::<f64>::zeros(6);
/// let mut workspace = Workspace::new();
/// stiefel.project_point(&arbitrary, &mut projected, &mut workspace);
/// 
/// // Result has orthonormal columns
/// assert!(stiefel.is_point_on_manifold(&projected, 1e-12));
///
/// // Tangent space projection
/// let x: DVector<f64> = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
/// let v: DVector<f64> = DVector::from_vec(vec![0.1; 6]);
/// let mut v_tangent = DVector::<f64>::zeros(6);
/// stiefel.project_tangent(&x, &v, &mut v_tangent, &mut workspace).unwrap();
/// assert!(stiefel.is_vector_in_tangent_space(&x, &v_tangent, 1e-12));
/// ```
///
/// ## Optimization Step
/// ```rust
/// # use riemannopt_manifolds::Stiefel;
/// # use riemannopt_core::manifold::Manifold;
/// # use riemannopt_core::memory::workspace::Workspace;
/// # use nalgebra::{DVector, Dyn};
/// # let stiefel = Stiefel::new(3, 2).unwrap();
/// # let x: DVector<f64> = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
///
/// // Simulate optimization step
/// let euclidean_grad: DVector<f64> = DVector::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3]);
/// let mut riemannian_grad = DVector::<f64>::zeros(6);
/// let mut workspace = Workspace::new();
/// stiefel.euclidean_to_riemannian_gradient(&x, &euclidean_grad, &mut riemannian_grad, &mut workspace).unwrap();
/// 
/// // Take step using retraction
/// let step_size: f64 = 0.01;
/// let direction: DVector<f64> = &riemannian_grad * (-step_size);
/// let mut x_new = DVector::<f64>::zeros(6);
/// stiefel.retract(&x, &direction, &mut x_new, &mut workspace).unwrap();
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
                format!("Stiefel manifold St(n,p) requires n > 0 and p > 0, got n={}, p={}", n, p),
            ));
        }
        if p > n {
            return Err(ManifoldError::invalid_point(
                format!("Stiefel manifold St(n,p) requires p <= n, got n={}, p={} (p > n)", n, p),
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
    fn qr_projection<T>(&self, matrix: &DMatrix<T>, workspace: &mut Workspace<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        self.qr_projection_with_workspace(matrix, workspace)
    }
    
    fn qr_projection_with_workspace<T>(&self, matrix: &DMatrix<T>, workspace: &mut Workspace<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        if matrix.nrows() != self.n || matrix.ncols() != self.p {
            // Use workspace buffer for wrong dimensions case
            let mut result = workspace.acquire_temp_matrix(self.n, self.p);
            result.fill(T::zero());
            
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
            
            // Must clone here for QR decomposition which consumes the matrix
            let qr = result.clone_owned().qr();
            // Take only the first p columns of Q
            qr.q().columns(0, self.p).into_owned()
        } else {
            // Must clone here for QR decomposition which consumes the matrix
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

    /// Projects a vector to the tangent space at a point using a workspace.
    ///
    /// This variant avoids allocations by using pre-allocated buffers from the workspace.
    /// 
    /// # Arguments
    /// * `point` - Point on the manifold (as vector)
    /// * `vector` - Vector to project
    /// * `result` - Output buffer for the projected vector
    /// * `workspace` - Pre-allocated workspace for temporary buffers
    pub fn project_tangent_with_workspace<T>(
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
                format!("n*p={} elements for St({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements, vector has {} elements", point.len(), vector.len()),
            ));
        }
        
        // Use workspace buffers for temporary matrices
        let mut xtv = workspace.acquire_temp_matrix(self.p, self.p);
        let mut vtx = workspace.acquire_temp_matrix(self.p, self.p);
        
        // Use matrix views instead of cloning
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let v_matrix = vector_to_matrix_view(vector, self.n, self.p);
        
        // Compute X^T V and V^T X
        xtv.copy_from(&(x_matrix.transpose() * &v_matrix));
        vtx.copy_from(&(v_matrix.transpose() * &x_matrix));
        
        // Compute symmetric part: (X^T V + V^T X) / 2
        let mut symmetric = workspace.acquire_temp_matrix(self.p, self.p);
        symmetric.copy_from(&(&*xtv + &*vtx));
        *symmetric *= <T as Scalar>::from_f64(0.5);
        
        // Compute projection: V - X * symmetric directly into result
        result.copy_from(vector);
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &x_matrix * &*symmetric;
        
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
    pub fn retract_with_workspace<T>(
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
                format!("n*p={} elements for St({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements, tangent has {} elements", point.len(), tangent.len()),
            ));
        }
        
        // Use workspace buffer for the candidate matrix
        let mut candidate = workspace.acquire_temp_matrix(self.n, self.p);
        
        // Use matrix views and compute X + V
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let v_matrix = vector_to_matrix_view(tangent, self.n, self.p);
        candidate.copy_from(&x_matrix);
        *candidate += &v_matrix;
        
        // QR retraction: extract Q factor
        // Clone is necessary here as QR decomposition consumes the matrix
        let qr = candidate.clone_owned().qr();
        let q_full = qr.q();
        let q = q_full.columns(0, self.p);
        
        // Copy directly into result
        let q_owned = q.into_owned();
        result.copy_from(&DVector::from_vec(q_owned.data.as_vec().clone()));
        
        Ok(())
    }

    /// Computes the inverse retraction (logarithmic map) using a workspace.
    ///
    /// This variant avoids allocations by using pre-allocated buffers from the workspace.
    ///
    /// # Arguments
    /// * `point` - Base point on the manifold
    /// * `other` - Target point on the manifold
    /// * `result` - Output buffer for the tangent vector
    /// * `workspace` - Pre-allocated workspace for temporary buffers
    pub fn inverse_retract_with_workspace<T>(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        T: Scalar,
    {
        if point.len() != self.n * self.p || other.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for St({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements, other has {} elements", point.len(), other.len()),
            ));
        }
        
        // Use workspace buffers
        let mut xtv = workspace.acquire_temp_matrix(self.p, self.p);
        let mut vtx = workspace.acquire_temp_matrix(self.p, self.p);
        
        // Use matrix views
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let _y_matrix = vector_to_matrix_view(other, self.n, self.p);
        
        // Approximate inverse retraction: V ≈ Y - X
        // First copy Y to result, then subtract X
        result.copy_from(other);
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &x_matrix;
        
        // Now project to tangent space in-place
        let v_matrix = vector_to_matrix_view(result, self.n, self.p);
        xtv.copy_from(&(x_matrix.transpose() * &v_matrix));
        vtx.copy_from(&(v_matrix.transpose() * &x_matrix));
        
        let mut symmetric = workspace.acquire_temp_matrix(self.p, self.p);
        symmetric.copy_from(&(&*xtv + &*vtx));
        *symmetric *= <T as Scalar>::from_f64(0.5);
        
        // Final projection: result = result - X * symmetric
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &x_matrix * &*symmetric;
        
        Ok(())
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

    fn is_point_on_manifold(&self, point: &Point<T, Dyn>, tolerance: T) -> bool {
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
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
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

    fn project_point(&self, point: &Point<T, Dyn>, result: &mut Point<T, Dyn>, workspace: &mut Workspace<T>) {
        let matrix = if point.len() == self.n * self.p {
            // Use matrix view to avoid cloning
            let view = vector_to_matrix_view(point, self.n, self.p);
            view.clone_owned()
        } else {
            // Handle wrong size by creating a matrix in workspace
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
        
        let projected = self.qr_projection(&matrix, workspace);
        result.copy_from(&DVector::from_vec(projected.data.as_vec().clone()))
    }

    fn project_tangent(
        &self,
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        self.project_tangent_with_workspace(point, vector, result, workspace)
    }

    fn inner_product(
        &self,
        _point: &Point<T, Dyn>,
        u: &TangentVector<T, Dyn>,
        v: &TangentVector<T, Dyn>,
    ) -> Result<T> {
        // Use Euclidean inner product (canonical metric) with SIMD dispatcher
        let dispatcher = get_dispatcher::<T>();
        Ok(dispatcher.dot_product(u, v))
    }

    fn retract(&self, point: &Point<T, Dyn>, tangent: &TangentVector<T, Dyn>, result: &mut Point<T, Dyn>, workspace: &mut Workspace<T>) -> Result<()> {
        self.retract_with_workspace(point, tangent, result, workspace)
    }

    fn inverse_retract(
        &self,
        point: &Point<T, Dyn>,
        other: &Point<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        self.inverse_retract_with_workspace(point, other, result, workspace)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Point<T, Dyn>,
        euclidean_grad: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Project Euclidean gradient to tangent space
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
        
        // Create temporary workspace for QR projection
        let mut workspace = Workspace::new();
        let projected = self.qr_projection(&matrix, &mut workspace);
        DVector::from_vec(projected.data.as_vec().clone())
    }

    fn random_tangent(&self, point: &Point<T, Dyn>, result: &mut TangentVector<T, Dyn>, _workspace: &mut Workspace<T>) -> Result<()> {
        if point.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for St({},{})", self.n * self.p, self.n, self.p),
                format!("point has {} elements", point.len()),
            ));
        }
        
        // Use matrix view to avoid cloning
        let x_matrix = vector_to_matrix_view(point, self.n, self.p);
        let tangent = self.random_tangent_matrix(&x_matrix.clone_owned())?;
        result.copy_from(&DVector::from_vec(tangent.data.as_vec().clone()));
        Ok(())
    }

    fn has_exact_exp_log(&self) -> bool {
        false // Stiefel manifold doesn't have closed-form exp/log maps
    }

    fn parallel_transport(
        &self,
        from: &Point<T, Dyn>,
        to: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if from.len() != self.n * self.p || to.len() != self.n * self.p || vector.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for St({},{})", self.n * self.p, self.n, self.p),
                format!("from has {} elements, to has {} elements, vector has {} elements", from.len(), to.len(), vector.len()),
            ));
        }
        
        // Use matrix views to avoid cloning
        let x_matrix = vector_to_matrix_view(from, self.n, self.p);
        let y_matrix = vector_to_matrix_view(to, self.n, self.p);
        let v_matrix = vector_to_matrix_view(vector, self.n, self.p);
        
        // Use workspace buffers for temporary matrices
        let mut ytv = workspace.acquire_temp_matrix(self.p, self.p);
        let mut vty = workspace.acquire_temp_matrix(self.p, self.p);
        let mut xtv = workspace.acquire_temp_matrix(self.p, self.p);
        let mut vtx = workspace.acquire_temp_matrix(self.p, self.p);
        
        // Compute all the necessary products
        ytv.copy_from(&(y_matrix.transpose() * &v_matrix));
        vty.copy_from(&(v_matrix.transpose() * &y_matrix));
        xtv.copy_from(&(x_matrix.transpose() * &v_matrix));
        vtx.copy_from(&(v_matrix.transpose() * &x_matrix));
        
        let mut term1 = workspace.acquire_temp_matrix(self.p, self.p);
        let mut term2 = workspace.acquire_temp_matrix(self.p, self.p);
        
        term1.copy_from(&(&*ytv + &*vty));
        *term1 *= <T as Scalar>::from_f64(0.5);
        
        term2.copy_from(&(&*xtv + &*vtx));
        *term2 *= <T as Scalar>::from_f64(0.5);
        
        // Compute transported = V - X * term1 + Y * term2
        result.copy_from(vector);
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &x_matrix * &*term1;
        result_matrix += &y_matrix * &*term2;
        
        // Project to ensure we're exactly in the tangent space at Y
        let transported_view = vector_to_matrix_view(result, self.n, self.p);
        let mut ytw = workspace.acquire_temp_matrix(self.p, self.p);
        let mut wty = workspace.acquire_temp_matrix(self.p, self.p);
        
        ytw.copy_from(&(y_matrix.transpose() * &transported_view));
        wty.copy_from(&(transported_view.transpose() * &y_matrix));
        
        let mut symmetric = workspace.acquire_temp_matrix(self.p, self.p);
        symmetric.copy_from(&(&*ytw + &*wty));
        *symmetric *= <T as Scalar>::from_f64(0.5);
        
        // Final projection in-place
        let mut result_matrix = vector_to_matrix_view_mut(result, self.n, self.p);
        result_matrix -= &y_matrix * &*symmetric;
        
        Ok(())
    }

    fn distance(&self, x: &Point<T, Dyn>, y: &Point<T, Dyn>, workspace: &mut Workspace<T>) -> Result<T> {
        if x.len() != self.n * self.p || y.len() != self.n * self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("n*p={} elements for St({},{})", self.n * self.p, self.n, self.p),
                format!("x has {} elements, y has {} elements", x.len(), y.len()),
            ));
        }
        
        // Use matrix views to avoid cloning
        let x_matrix = vector_to_matrix_view(x, self.n, self.p);
        let y_matrix = vector_to_matrix_view(y, self.n, self.p);
        
        // Use workspace buffer for the product
        let mut m = workspace.acquire_temp_matrix(self.p, self.p);
        m.copy_from(&(x_matrix.transpose() * &y_matrix));
        
        // SVD requires owned matrix, so we must clone here
        let svd = m.clone_owned().svd(true, true);
        
        let mut distance_squared = T::zero();
        let singular_values = svd.singular_values;
        
        // Process singular values sequentially for now
        for i in 0..singular_values.len() {
            let sigma = singular_values[i];
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
        let mut workspace = Workspace::new();
        
        // Create non-orthonormal matrix
        let point = DVector::from_vec(vec![2.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let mut projected = DVector::zeros(6);
        stiefel.project_point(&point, &mut projected, &mut workspace);
        
        assert!(stiefel.is_point_on_manifold(&projected, 1e-10));
    }

    #[test]
    fn test_tangent_projection() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        
        let point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let mut projected = DVector::zeros(6);
        stiefel.project_tangent(&point, &vector, &mut projected, &mut workspace).unwrap();
        assert!(stiefel.is_vector_in_tangent_space(&point, &projected, 1e-10));
    }

    #[test]
    fn test_retraction_properties() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        let point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let zero_tangent = DVector::zeros(6);
        
        // Test centering: R(x, 0) = x
        let mut retracted = DVector::zeros(6);
        stiefel.retract(&point, &zero_tangent, &mut retracted, &mut workspace).unwrap();
        assert_relative_eq!(
            (&retracted - &point).norm(), 
            0.0, 
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_random_generation() {
        let stiefel = Stiefel::new(4, 2).unwrap();
        let mut workspace = Workspace::new();
        
        // Test random point
        let random_point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        assert!(stiefel.is_point_on_manifold(&random_point, 1e-10));
        
        // Test random tangent
        let mut tangent = DVector::zeros(8);
        stiefel.random_tangent(&random_point, &mut tangent, &mut workspace).unwrap();
        assert!(stiefel.is_vector_in_tangent_space(&random_point, &tangent, 1e-10));
    }

    #[test]
    fn test_gradient_conversion() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        let point = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let euclidean_grad = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let mut riemannian_grad = DVector::zeros(6);
        stiefel
            .euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut workspace)
            .unwrap();
        
        assert!(stiefel.is_vector_in_tangent_space(&point, &riemannian_grad, 1e-10));
    }

    #[test]
    fn test_distance() {
        let stiefel = Stiefel::new(3, 2).unwrap();
        let mut workspace = Workspace::new();
        let point1 = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        let point2 = <Stiefel as Manifold<f64, Dyn>>::random_point(&stiefel);
        
        // Distance should be non-negative
        let dist = stiefel.distance(&point1, &point2, &mut workspace).unwrap();
        assert!(dist >= 0.0);
        
        // Distance to self should be zero
        let self_dist = stiefel.distance(&point1, &point1, &mut workspace).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-7);
    }
}