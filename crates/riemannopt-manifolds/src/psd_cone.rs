//! # Positive Semi-Definite Cone S⁺(n)
//!
//! The cone S⁺(n) of n×n symmetric positive semi-definite (PSD) matrices forms
//! a convex cone in the space of symmetric matrices. It provides a fundamental
//! framework for semidefinite programming and covariance estimation.
//!
//! ## Mathematical Definition
//!
//! The PSD cone is formally defined as:
//! ```text
//! S⁺(n) = {X ∈ ℝⁿˣⁿ : X = X^T, X ⪰ 0}
//!        = {X ∈ S(n) : λ_i(X) ≥ 0 for all i}
//! ```
//! where S(n) is the space of symmetric matrices and λ_i(X) are eigenvalues.
//!
//! Equivalently, X ∈ S⁺(n) if and only if v^T X v ≥ 0 for all v ∈ ℝⁿ.
//!
//! ## Geometric Structure
//!
//! ### Boundary and Interior
//! - **Interior**: S⁺⁺(n) = {X ∈ S⁺(n) : X ≻ 0} (positive definite)
//! - **Boundary**: ∂S⁺(n) = {X ∈ S⁺(n) : det(X) = 0} (singular PSD)
//! - **Extreme rays**: Rank-1 matrices X = vv^T
//!
//! ### Tangent Cone
//! At X ∈ S⁺(n), the tangent cone is:
//! ```text
//! T_X S⁺(n) = {V ∈ S(n) : v^T V v ≥ 0 for all v ∈ ker(X)}
//! ```
//! For interior points, T_X S⁺(n) = S(n) (all symmetric matrices).
//!
//! ### Riemannian Metrics
//!
//! #### 1. Euclidean (Frobenius) Metric
//! ```text
//! g_X(U, V) = tr(UV) = ⟨U, V⟩_F
//! ```
//!
//! #### 2. Affine-Invariant Metric (for interior)
//! ```text
//! g_X^{AI}(U, V) = tr(X⁻¹U X⁻¹V)
//! ```
//!
//! #### 3. Log-Euclidean Metric
//! ```text
//! g_X^{LE}(U, V) = g_{log(X)}^E(Dlog_X U, Dlog_X V)
//! ```
//!
//! ### Projection to S⁺(n)
//! For any symmetric matrix A:
//! ```text
//! P_{S⁺}(A) = Q max(0, Λ) Q^T
//! ```
//! where A = QΛQ^T is the eigendecomposition.
//!
//! ## Distance Formulas
//!
//! ### Euclidean Distance
//! ```text
//! d_E(X, Y) = ‖X - Y‖_F
//! ```
//!
//! ### Affine-Invariant Distance (interior)
//! ```text
//! d_{AI}(X, Y) = ‖log(X⁻¹/²YX⁻¹/²)‖_F
//! ```
//!
//! ## Optimization on S⁺(n)
//!
//! ### Riemannian Gradient
//! For f: S⁺(n) → ℝ with Euclidean gradient ∇f(X):
//! ```text
//! grad^E f(X) = P_{T_X}(∇f(X))
//! ```
//!
//! ### Retraction
//! A simple retraction is the metric projection:
//! ```text
//! R_X(V) = P_{S⁺}(X + V)
//! ```
//!
//! ## Applications
//!
//! 1. **Semidefinite Programming (SDP)**: Convex optimization over S⁺(n)
//! 2. **Covariance Estimation**: Regularized sample covariance matrices
//! 3. **Kernel Learning**: Learning positive semi-definite kernel matrices
//! 4. **Quantum State Tomography**: Density matrices in quantum mechanics
//! 5. **Graph Theory**: Laplacian matrices and spectral methods
//! 6. **Control Theory**: Lyapunov functions and stability analysis
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Efficient projection** via eigendecomposition
//! - **Numerical stability** for near-singular matrices
//! - **Proper scaling** for off-diagonal elements in vectorization
//! - **Boundary handling** for rank-deficient matrices
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::PSDCone;
//! use riemannopt_core::manifold::Manifold;
//! use nalgebra::DMatrix;
//!
//! // Create S⁺(3) - 3×3 PSD matrices
//! let psd_cone = PSDCone::new(3)?;
//!
//! // Random PSD matrix
//! let x = psd_cone.random_point();
//!
//! // Verify positive semi-definiteness
//! let x_mat = psd_cone.vector_to_matrix(&x);
//! let eigenvalues = x_mat.symmetric_eigen().eigenvalues;
//! assert!(eigenvalues.iter().all(|&λ| λ >= -1e-10));
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::Scalar,
};

/// The positive semi-definite cone S⁺(n) of n×n symmetric PSD matrices.
///
/// This structure represents the convex cone of symmetric positive semi-definite
/// matrices, equipped with various Riemannian metrics for optimization.
///
/// # Type Parameters
///
/// The manifold is generic over the scalar type T through the Manifold trait.
///
/// # Invariants
///
/// - `n ≥ 1`: Matrix dimension must be positive
/// - All points X satisfy X = X^T and X ⪰ 0
/// - Tangent vectors at interior points are arbitrary symmetric matrices
/// - Tangent vectors at boundary points satisfy additional constraints
#[derive(Debug, Clone)]
pub struct PSDCone {
    /// Dimension of the matrices (n×n)
    n: usize,
    /// Numerical tolerance for constraint validation
    tolerance: f64,
    /// Whether to use strict positive definiteness (interior only)
    strict: bool,
}

impl PSDCone {
    /// Creates a new PSD cone manifold S⁺(n).
    ///
    /// # Arguments
    ///
    /// * `n` - Dimension of the matrices (must be ≥ 1)
    ///
    /// # Returns
    ///
    /// A PSD cone manifold with dimension n(n+1)/2.
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if n = 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::PSDCone;
    /// // Create S⁺(3) - 3×3 PSD matrices
    /// let psd_cone = PSDCone::new(3)?;
    /// assert_eq!(psd_cone.matrix_size(), 3);
    /// assert_eq!(psd_cone.manifold_dim(), 6); // 3*(3+1)/2 = 6
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_parameter(
                "PSD cone requires n ≥ 1"
            ));
        }
        Ok(Self { n, tolerance: 1e-10, strict: false })
    }

    /// Creates a PSD cone with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension
    /// * `tolerance` - Numerical tolerance for eigenvalue checks
    /// * `strict` - If true, only accepts positive definite matrices
    pub fn with_parameters(n: usize, tolerance: f64, strict: bool) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_parameter(
                "PSD cone requires n ≥ 1"
            ));
        }
        if tolerance <= 0.0 || tolerance >= 1.0 {
            return Err(ManifoldError::invalid_parameter(
                "Tolerance must be in (0, 1)"
            ));
        }
        Ok(Self { n, tolerance, strict })
    }

    /// Returns the matrix dimension n.
    #[inline]
    pub fn matrix_size(&self) -> usize {
        self.n
    }

    /// Returns the manifold dimension n(n+1)/2.
    #[inline]
    pub fn manifold_dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    /// Validates that a matrix is positive semi-definite.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that X = X^T and all eigenvalues λ_i(X) ≥ 0.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix is not n×n
    /// - `InvalidPoint`: If matrix is not symmetric or not PSD
    pub fn check_matrix<T: Scalar>(&self, x: &DMatrix<T>) -> Result<()> {
        if x.nrows() != self.n || x.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.n),
                format!("{}×{}", x.nrows(), x.ncols())
            ));
        }

        // Check symmetry
        for i in 0..self.n {
            for j in i+1..self.n {
                if <T as Float>::abs(x[(i, j)] - x[(j, i)]) > <T as Scalar>::from_f64(self.tolerance) {
                    return Err(ManifoldError::invalid_point(format!(
                        "Matrix not symmetric: |X[{},{}] - X[{},{}]| = {} > {}",
                        i, j, j, i,
                        <T as Float>::abs(x[(i, j)] - x[(j, i)]),
                        self.tolerance
                    )));
                }
            }
        }

        // Check eigenvalues
        let eigen = x.clone().symmetric_eigen();
        let min_eigenvalue = eigen.eigenvalues.iter()
            .fold(<T as Float>::max_value(), |min, &val| if val < min { val } else { min });
        
        let threshold = if self.strict {
            <T as Scalar>::from_f64(self.tolerance)
        } else {
            -<T as Scalar>::from_f64(self.tolerance)
        };

        if min_eigenvalue < threshold {
            return Err(ManifoldError::invalid_point(format!(
                "Matrix not {}: minimum eigenvalue {} < {}",
                if self.strict { "positive definite" } else { "positive semi-definite" },
                min_eigenvalue,
                threshold
            )));
        }

        Ok(())
    }

    /// Converts a symmetric matrix to its vectorized form.
    ///
    /// Uses the standard vectorization with √2 scaling for off-diagonal elements
    /// to preserve the Frobenius inner product.
    pub fn matrix_to_vector<T: Scalar>(&self, mat: &DMatrix<T>) -> DVector<T> {
        self.mat_to_vec(mat)
    }

    /// Converts a vector to its symmetric matrix form.
    pub fn vector_to_matrix<T: Scalar>(&self, vec: &DVector<T>) -> DMatrix<T> {
        self.vec_to_mat(vec)
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
                if <T as Float>::abs(mat[(i, j)] - mat[(j, i)]) > tol {
                    return false;
                }
            }
        }
        
        // Check positive semi-definiteness
        let eigen = mat.clone().symmetric_eigen();
        let threshold = if self.strict { tol } else { -tol };
        eigen.eigenvalues.iter().all(|&lambda| lambda >= threshold)
    }

    /// Projects a matrix onto the PSD cone.
    ///
    /// # Mathematical Operation
    ///
    /// For a symmetric matrix A with eigendecomposition A = QΛQ^T:
    /// ```text
    /// P_{S⁺}(A) = Q max(0, Λ) Q^T
    /// ```
    ///
    /// # Arguments
    ///
    /// * `mat` - Input matrix (will be symmetrized first)
    ///
    /// # Returns
    ///
    /// The projected PSD matrix.
    pub fn project_matrix<T: Scalar>(&self, mat: &DMatrix<T>) -> Result<DMatrix<T>> {
        if mat.nrows() != self.n || mat.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.n),
                format!("{}×{}", mat.nrows(), mat.ncols())
            ));
        }
        
        Ok(self.project_to_psd(mat))
    }

    /// Computes the distance to the PSD cone.
    ///
    /// # Mathematical Formula
    ///
    /// For a symmetric matrix A:
    /// ```text
    /// dist(A, S⁺) = ‖A - P_{S⁺}(A)‖_F
    /// ```
    ///
    /// # Arguments
    ///
    /// * `mat` - Input matrix
    ///
    /// # Returns
    ///
    /// The Frobenius distance to the nearest PSD matrix.
    pub fn distance_to_cone<T: Scalar>(&self, mat: &DMatrix<T>) -> Result<T> {
        if mat.nrows() != self.n || mat.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.n),
                format!("{}×{}", mat.nrows(), mat.ncols())
            ));
        }
        
        let projected = self.project_to_psd(mat);
        let diff = mat - &projected;
        Ok(<T as Float>::sqrt(diff.dot(&diff)))
    }

    /// Computes the minimum eigenvalue of a matrix.
    ///
    /// Useful for checking how far a matrix is from being PSD.
    pub fn minimum_eigenvalue<T: Scalar>(&self, mat: &DMatrix<T>) -> Result<T> {
        if mat.nrows() != self.n || mat.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.n),
                format!("{}×{}", mat.nrows(), mat.ncols())
            ));
        }
        
        // Symmetrize first
        let sym = (mat + &mat.transpose()) / <T as Scalar>::from_f64(2.0);
        let eigen = sym.symmetric_eigen();
        
        Ok(eigen.eigenvalues.iter()
            .fold(<T as Float>::max_value(), |min, &val| if val < min { val } else { min }))
    }
}

impl<T: Scalar> Manifold<T> for PSDCone {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;

    fn name(&self) -> &str {
        "PSDCone"
    }

    fn dimension(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.len() != self.n * (self.n + 1) / 2 {
            return false;
        }
        
        let mat = self.vec_to_mat(point);
        self.is_psd(&mat, tol)
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        tol: T,
    ) -> bool {
        if !self.is_point_on_manifold(point, tol) {
            return false;
        }
        
        if vector.len() != self.n * (self.n + 1) / 2 {
            return false;
        }
        
        // For interior points, tangent space is all symmetric matrices
        // For boundary points (with zero eigenvalues), it's more restricted
        let mat = self.vec_to_mat(vector);
        
        // Check symmetry
        for i in 0..self.n {
            for j in i+1..self.n {
                if <T as Float>::abs(mat[(i, j)] - mat[(j, i)]) > tol {
                    return false;
                }
            }
        }
        
        // For boundary points, check additional constraint
        if self.strict {
            return true; // In strict mode, we're always in interior
        }
        
        // Check if tangent respects the cone constraint at boundary
        let x_mat = self.vec_to_mat(point);
        let eigen = x_mat.clone().symmetric_eigen();
        
        // Find near-zero eigenvalues (boundary)
        for (i, &lambda) in eigen.eigenvalues.iter().enumerate() {
            if <T as Float>::abs(lambda) < <T as Scalar>::from_f64(self.tolerance) {
                // For zero eigenvalue, check v^T V v >= 0 where v is eigenvector
                let v = eigen.eigenvectors.column(i);
                let vt_mat_v = v.dot(&(&mat * v));
                if vt_mat_v < -tol {
                    return false;
                }
            }
        }
        
        true
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
        // Ensure result has correct size
        if result.len() != self.n * (self.n + 1) / 2 {
            *result = DVector::zeros(self.n * (self.n + 1) / 2);
        }
        
        let mat = if point.len() == self.n * self.n {
            // If given as full matrix, reshape
            DMatrix::from_vec(self.n, self.n, point.as_slice().to_vec())
        } else if point.len() == self.n * (self.n + 1) / 2 {
            self.vec_to_mat(point)
        } else {
            // Wrong size, create zero matrix
            DMatrix::zeros(self.n, self.n)
        };
        
        let projected = self.project_to_psd(&mat);
        let projected_vec = self.mat_to_vec(&projected);
        result.copy_from(&projected_vec);
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n * (self.n + 1) / 2 {
            *result = DVector::zeros(self.n * (self.n + 1) / 2);
        }
        
        // Check dimensions
        if point.len() != self.n * (self.n + 1) / 2 || vector.len() != self.n * (self.n + 1) / 2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}", self.n * (self.n + 1) / 2),
                format!("{}", point.len().max(vector.len()))
            ));
        }
        
        // For PSD cone, tangent projection is symmetrization
        let mat = self.vec_to_mat(vector);
        let sym = (mat.clone() + mat.transpose()) / <T as Scalar>::from_f64(2.0);
        
        // For boundary points, additional projection may be needed
        if !self.strict {
            let x_mat = self.vec_to_mat(point);
            let eigen = x_mat.clone().symmetric_eigen();
            
            // Project out components that violate cone constraint
            let mut proj_mat = sym.clone();
            for (i, &lambda) in eigen.eigenvalues.iter().enumerate() {
                if <T as Float>::abs(lambda) < <T as Scalar>::from_f64(self.tolerance) {
                    // Zero eigenvalue: project to ensure v^T V v >= 0
                    let v = eigen.eigenvectors.column(i);
                    let vt_mat_v = v.dot(&(&sym * v));
                    if vt_mat_v < T::zero() {
                        // Project out the violating component
                        let vvt = &v * v.transpose();
                        proj_mat = proj_mat - vvt * vt_mat_v;
                    }
                }
            }
            
            let proj_vec = self.mat_to_vec(&proj_mat);
            result.copy_from(&proj_vec);
        } else {
            let sym_vec = self.mat_to_vec(&sym);
            result.copy_from(&sym_vec);
        }
        
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        if u.len() != self.n * (self.n + 1) / 2 || v.len() != self.n * (self.n + 1) / 2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}", self.n * (self.n + 1) / 2),
                format!("{}", u.len().max(v.len()))
            ));
        }
        
        // Standard Frobenius inner product (with √2 scaling already in vectors)
        Ok(u.dot(v))
    }

    fn retract(
        &self,
        point: &Self::Point,
        tangent: &Self::TangentVector,
        result: &mut Self::Point,
    ) -> Result<()> {
        if point.len() != self.n * (self.n + 1) / 2 || tangent.len() != self.n * (self.n + 1) / 2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}", self.n * (self.n + 1) / 2),
                format!("{}", point.len().max(tangent.len()))
            ));
        }
        
        // Ensure result has correct size
        if result.len() != self.n * (self.n + 1) / 2 {
            *result = DVector::zeros(self.n * (self.n + 1) / 2);
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
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        if point.len() != self.n * (self.n + 1) / 2 || other.len() != self.n * (self.n + 1) / 2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}", self.n * (self.n + 1) / 2),
                format!("{}", point.len().max(other.len()))
            ));
        }
        
        // Ensure result has correct size
        if result.len() != self.n * (self.n + 1) / 2 {
            *result = DVector::zeros(self.n * (self.n + 1) / 2);
        }
        
        // Simple approximation: project the difference
        let diff = other - point;
        self.project_tangent(point, &diff, result)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For the Euclidean metric, just project to tangent space
        self.project_tangent(point, euclidean_grad, result)
    }

    fn random_point(&self, result: &mut Self::Point) -> Result<()> {
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
        
        *result = self.mat_to_vec(&psd_scaled);
        Ok(())
    }

    fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
        if point.len() != self.n * (self.n + 1) / 2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}", self.n * (self.n + 1) / 2),
                format!("{}", point.len())
            ));
        }
        
        // Ensure result has correct size
        if result.len() != self.n * (self.n + 1) / 2 {
            *result = DVector::zeros(self.n * (self.n + 1) / 2);
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

    fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
        if x.len() != self.n * (self.n + 1) / 2 || y.len() != self.n * (self.n + 1) / 2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}", self.n * (self.n + 1) / 2),
                format!("{}", x.len().max(y.len()))
            ));
        }
        
        // Frobenius distance
        let diff = y - x;
        Ok(<T as Float>::sqrt(diff.dot(&diff)))
    }

    fn parallel_transport(
        &self,
        _from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For flat space with Euclidean metric, parallel transport is identity
        // Just project to ensure it's in tangent space at destination
        self.project_tangent(to, vector, result)
    }

    fn has_exact_exp_log(&self) -> bool {
        false // PSD cone doesn't have simple closed-form exp/log
    }

    fn is_flat(&self) -> bool {
        true // With Euclidean metric, PSD cone is flat
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For PSD cone with Euclidean metric, tangent vectors are symmetric matrices
        // Scaling preserves symmetry
        result.copy_from(tangent);
        *result *= scalar;
        Ok(())
    }

    fn add_tangents(
        &self,
        point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        // Temporary buffer for projection if needed
        temp: &mut Self::TangentVector,
    ) -> Result<()> {
        // Add the tangent vectors
        temp.copy_from(v1);
        *temp += v2;
        
        // The sum should already be symmetric if v1 and v2 are,
        // but we project for numerical stability and boundary constraints
        self.project_tangent(point, temp, result)?;
        
        Ok(())
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
        assert_eq!(manifold.matrix_size(), 3);
        assert_eq!(<PSDCone as Manifold<f64>>::dimension(&manifold), 6); // 3*(3+1)/2
        assert_eq!(manifold.manifold_dim(), 6);
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
        let vec = manifold.matrix_to_vector(&mat);
        
        let mut projected = DVector::zeros(<PSDCone as Manifold<f64>>::dimension(&manifold));
        manifold.project_point(&vec, &mut projected);
        let proj_mat = manifold.vector_to_matrix(&projected);
        
        // Check that projection is PSD
        assert!(manifold.is_psd(&proj_mat, 1e-10));
        
        // Test public projection method
        let proj_mat2 = manifold.project_matrix(&mat).unwrap();
        assert!(manifold.is_psd(&proj_mat2, 1e-10));
    }

    #[test]
    fn test_psd_cone_tangent_projection() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        
        // Create a non-symmetric matrix
        let mat = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            0.0, 1.0, 2.0,
            0.0, 0.0, 1.0
        ]);
        let vec = manifold.matrix_to_vector(&mat);
        
        let mut tangent = DVector::zeros(<PSDCone as Manifold<f64>>::dimension(&manifold));
        manifold.project_tangent(&point, &vec, &mut tangent).unwrap();
        let tan_mat = manifold.vector_to_matrix(&tangent);
        
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
        
        let point = manifold.random_point();
        let mut tangent = DVector::zeros(<PSDCone as Manifold<f64>>::dimension(&manifold));
        manifold.random_tangent(&point, &mut tangent).unwrap();
        let scaled_tangent = &tangent * 0.1;
        let mut retracted = DVector::zeros(<PSDCone as Manifold<f64>>::dimension(&manifold));
        manifold.retract(&point, &scaled_tangent, &mut retracted).unwrap();
        
        // Check that result is on manifold
        assert!(manifold.is_point_on_manifold(&retracted, 1e-6));
    }

    #[test]
    fn test_psd_cone_inner_product() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let mut u = DVector::zeros(<PSDCone as Manifold<f64>>::dimension(&manifold));
        let mut v = DVector::zeros(<PSDCone as Manifold<f64>>::dimension(&manifold));
        manifold.random_tangent(&point, &mut u).unwrap();
        manifold.random_tangent(&point, &mut v).unwrap();
        
        let ip_uv = manifold.inner_product(&point, &u, &v).unwrap();
        let ip_vu = manifold.inner_product(&point, &v, &u).unwrap();
        
        // Check symmetry
        assert_relative_eq!(ip_uv, ip_vu, epsilon = 1e-10);
    }

    #[test]
    fn test_psd_cone_public_methods() {
        let manifold = create_test_manifold();
        
        // Test check_matrix
        let psd_mat = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.5, 0.3,
            0.5, 1.0, 0.2,
            0.3, 0.2, 1.0
        ]);
        assert!(manifold.check_matrix(&psd_mat).is_ok());
        
        // Test minimum_eigenvalue
        let min_eig = manifold.minimum_eigenvalue(&psd_mat).unwrap();
        assert!(min_eig > 0.0);
        
        // Test distance_to_cone
        let non_psd_mat = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 1.0
        ]);
        let dist = manifold.distance_to_cone(&non_psd_mat).unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn test_psd_cone_properties() {
        let manifold = PSDCone::new(4).unwrap();
        
        assert_eq!(<PSDCone as Manifold<f64>>::name(&manifold), "PSDCone");
        assert_eq!(<PSDCone as Manifold<f64>>::dimension(&manifold), 10); // 4*(4+1)/2 = 10
        assert!(!<PSDCone as Manifold<f64>>::has_exact_exp_log(&manifold));
        assert!(<PSDCone as Manifold<f64>>::is_flat(&manifold));
    }

    #[test]
    fn test_psd_cone_with_parameters() {
        let manifold = PSDCone::with_parameters(3, 1e-8, true).unwrap();
        assert_eq!(manifold.tolerance, 1e-8);
        assert!(manifold.strict);
        
        // Test invalid parameters
        assert!(PSDCone::with_parameters(0, 1e-8, false).is_err());
        assert!(PSDCone::with_parameters(3, 0.0, false).is_err());
        assert!(PSDCone::with_parameters(3, 1.0, false).is_err());
    }
}