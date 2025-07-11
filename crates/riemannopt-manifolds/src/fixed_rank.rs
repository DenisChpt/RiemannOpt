//! # Fixed-Rank Manifold M_k(m,n)
//!
//! The manifold M_k(m,n) of m×n matrices with fixed rank k forms a smooth
//! submanifold of ℝ^{m×n}. It provides a geometric framework for low-rank
//! matrix optimization problems.
//!
//! ## Mathematical Definition
//!
//! The fixed-rank manifold is formally defined as:
//! ```text
//! M_k(m,n) = {X ∈ ℝ^{m×n} : rank(X) = k}
//! ```
//!
//! ## Parametrization via SVD
//!
//! Points are represented using the compact SVD:
//! ```text
//! X = UΣV^T
//! ```
//! where:
//! - U ∈ St(m,k): Left singular vectors
//! - Σ ∈ ℝ^{k×k}: Diagonal matrix of singular values
//! - V ∈ St(n,k): Right singular vectors
//!
//! This gives the quotient structure:
//! ```text
//! M_k(m,n) ≅ (St(m,k) × ℝ₊^k × St(n,k)) / O(k)
//! ```
//!
//! ## Tangent Space
//!
//! The tangent space at X = UΣV^T consists of matrices:
//! ```text
//! T_X M_k = {U_⊥MV^T + UNV_⊥^T + UΩV^T : M ∈ ℝ^{(m-k)×k}, N ∈ ℝ^{k×(n-k)}, Ω ∈ ℝ^{k×k}}
//! ```
//! where U_⊥ and V_⊥ are orthogonal complements.
//!
//! ## Riemannian Metric
//!
//! The standard metric is the Euclidean metric restricted to the tangent space:
//! ```text
//! g_X(ξ, η) = tr(ξ^T η)
//! ```
//!
//! ## Retractions
//!
//! ### SVD-based Retraction
//! ```text
//! R_X(ξ) = best rank-k approximation of (X + ξ)
//! ```
//!
//! ### Orthographic Retraction
//! For X = UΣV^T and tangent ξ = UMV^T + U_⊥NV^T + UN^TV_⊥^T:
//! ```text
//! R_X(ξ) = (U + U_⊥NΣ⁻¹)(Σ + M)(V + V_⊥N^TΣ⁻¹)^T
//! ```
//!
//! ## Geometric Properties
//!
//! - **Dimension**: dim(M_k) = k(m + n - k)
//! - **Non-closed**: M_k is not closed in ℝ^{m×n}
//! - **Embedded submanifold**: When viewed as subset of ℝ^{m×n}
//! - **Quotient manifold**: Inherits structure from product of Stiefel manifolds
//!
//! ## Applications
//!
//! 1. **Matrix Completion**: Netflix problem, collaborative filtering
//! 2. **System Identification**: Low-order dynamical systems
//! 3. **Model Reduction**: Reduced-order modeling
//! 4. **Computer Vision**: Structure from motion, face recognition
//! 5. **Data Compression**: Low-rank approximation
//! 6. **Machine Learning**: Low-rank neural networks
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Efficient storage** using factored form UΣV^T
//! - **Numerical stability** in SVD computations
//! - **Proper handling** of small singular values
//! - **Orthogonality preservation** in U and V factors
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::FixedRank;
//! use riemannopt_core::manifold::Manifold;
//! use nalgebra::DMatrix;
//!
//! // Create M_2(4,3) - 4×3 matrices of rank 2
//! let manifold = FixedRank::new(4, 3, 2)?;
//!
//! // Random rank-2 matrix
//! let x = manifold.random_point();
//!
//! // Convert to matrix form
//! let x_mat = x.to_matrix();
//! assert_eq!(x_mat.rank(1e-10), 2);
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::{DMatrix, DVector, SVD};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::Scalar,
};

/// The fixed-rank manifold M_k(m,n) of m×n matrices with rank k.
///
/// This structure represents matrices of fixed rank using their SVD factorization,
/// providing efficient storage and computation for low-rank matrix optimization.
///
/// # Type Parameters
///
/// The manifold is generic over the scalar type T through the Manifold trait.
///
/// # Invariants
///
/// - `m ≥ 1, n ≥ 1`: Matrix dimensions must be positive
/// - `k ≥ 1`: Rank must be positive
/// - `k ≤ min(m, n)`: Rank cannot exceed matrix dimensions
/// - Points are stored as vectors containing U, Σ, V factors
#[derive(Debug, Clone)]
pub struct FixedRank {
    /// Number of rows
    m: usize,
    /// Number of columns
    n: usize,
    /// Rank
    k: usize,
    /// Numerical tolerance
    tolerance: f64,
}

/// Representation of a point on the fixed-rank manifold
#[derive(Debug, Clone)]
pub struct FixedRankPoint<T: Scalar> {
    /// Left singular vectors (m × k)
    pub u: DMatrix<T>,
    /// Singular values (k × k diagonal)
    pub s: DVector<T>,
    /// Right singular vectors (n × k)
    pub v: DMatrix<T>,
}

impl<T: Scalar> FixedRankPoint<T> {
    /// Create a new fixed-rank point from factors
    pub fn new(u: DMatrix<T>, s: DVector<T>, v: DMatrix<T>) -> Self {
        Self { u, s, v }
    }

    /// Convert to full matrix representation
    pub fn to_matrix(&self) -> DMatrix<T> {
        let s_mat = DMatrix::from_diagonal(&self.s);
        &self.u * s_mat * self.v.transpose()
    }

    /// Create from full matrix using SVD
    pub fn from_matrix(mat: &DMatrix<T>, k: usize) -> Result<Self> {
        let svd = SVD::new(mat.clone(), true, true);
        
        let u = svd.u.ok_or_else(|| ManifoldError::numerical_error("SVD failed to compute U"))?;
        let v_t = svd.v_t.ok_or_else(|| ManifoldError::numerical_error("SVD failed to compute V^T"))?;
        let s = &svd.singular_values;
        
        // Truncate to rank k
        let u_k = u.columns(0, k).into();
        let s_k = s.rows(0, k).into();
        let v_k = v_t.transpose().columns(0, k).into();
        
        Ok(Self::new(u_k, s_k, v_k))
    }
}

/// Representation of a tangent vector on the fixed-rank manifold
#[derive(Debug, Clone)]
pub struct FixedRankTangent<T: Scalar> {
    /// Component U_perp * M * V^T (m × k)
    pub u_perp_m: DMatrix<T>,
    /// Component U * S_dot * V^T (k × k)
    pub s_dot: DVector<T>,
    /// Component U * N * V_perp^T (n × k)  
    pub v_perp_n: DMatrix<T>,
}

impl<T: Scalar> FixedRankTangent<T> {
    /// Create a new fixed-rank tangent vector from components
    pub fn new(u_perp_m: DMatrix<T>, s_dot: DVector<T>, v_perp_n: DMatrix<T>) -> Self {
        Self { u_perp_m, s_dot, v_perp_n }
    }
    
    /// Convert to full matrix representation given a base point
    pub fn to_matrix(&self, point: &FixedRankPoint<T>) -> DMatrix<T> {
        let s_dot_mat = DMatrix::from_diagonal(&self.s_dot);
        
        // Compute U_perp and V_perp using QR decomposition
        let (u_perp, _) = Self::compute_orthogonal_complement(&point.u);
        let (v_perp, _) = Self::compute_orthogonal_complement(&point.v);
        
        // Combine the three components
        &u_perp * &self.u_perp_m * point.v.transpose() +
        &point.u * s_dot_mat * point.v.transpose() +
        &point.u * &self.v_perp_n * v_perp.transpose()
    }
    
    /// Compute orthogonal complement of a matrix with orthonormal columns
    fn compute_orthogonal_complement(mat: &DMatrix<T>) -> (DMatrix<T>, DMatrix<T>) {
        let m = mat.nrows();
        let k = mat.ncols();
        
        if k >= m {
            // No orthogonal complement
            return (DMatrix::zeros(m, 0), DMatrix::zeros(0, 0));
        }
        
        // Create identity and project out the columns of mat
        let mut q = DMatrix::identity(m, m);
        q -= mat * mat.transpose();
        
        // Use QR to get orthonormal basis for the complement
        let qr = q.qr();
        let q_full = qr.q();
        
        // Extract the last m-k columns
        let u_perp = q_full.columns(k, m - k).into();
        let r_perp = DMatrix::zeros(m - k, m - k); // Placeholder
        
        (u_perp, r_perp)
    }
}

impl FixedRank {
    /// Creates a new fixed-rank manifold M_k(m,n).
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows (must be ≥ 1)
    /// * `n` - Number of columns (must be ≥ 1)
    /// * `k` - Rank (must satisfy 1 ≤ k ≤ min(m, n))
    ///
    /// # Returns
    ///
    /// A fixed-rank manifold with dimension k(m + n - k).
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if:
    /// - Any dimension is zero
    /// - k > min(m, n)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::FixedRank;
    /// // Create M_2(5,4) - 5×4 matrices of rank 2
    /// let manifold = FixedRank::new(5, 4, 2)?;
    /// assert_eq!(manifold.matrix_dimensions(), (5, 4, 2));
    /// assert_eq!(manifold.manifold_dim(), 2*(5+4-2)); // 14
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Fixed-rank manifold requires m ≥ 1, n ≥ 1, and k ≥ 1"
            ));
        }
        
        if k > m.min(n) {
            return Err(ManifoldError::invalid_parameter(
                format!("Rank k={} cannot exceed min(m={}, n={})", k, m, n)
            ));
        }
        
        Ok(Self { m, n, k, tolerance: 1e-12 })
    }

    /// Creates a fixed-rank manifold with custom tolerance.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows
    /// * `n` - Number of columns
    /// * `k` - Rank
    /// * `tolerance` - Numerical tolerance for rank checks
    pub fn with_tolerance(m: usize, n: usize, k: usize, tolerance: f64) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Fixed-rank manifold requires m ≥ 1, n ≥ 1, and k ≥ 1"
            ));
        }
        
        if k > m.min(n) {
            return Err(ManifoldError::invalid_parameter(
                format!("Rank k={} cannot exceed min(m={}, n={})", k, m, n)
            ));
        }
        
        if tolerance <= 0.0 || tolerance >= 1.0 {
            return Err(ManifoldError::invalid_parameter(
                "Tolerance must be in (0, 1)"
            ));
        }
        
        Ok(Self { m, n, k, tolerance })
    }

    /// Returns the matrix dimensions (m, n, k).
    #[inline]
    pub fn matrix_dimensions(&self) -> (usize, usize, usize) {
        (self.m, self.n, self.k)
    }

    /// Returns the manifold dimension k(m + n - k).
    #[inline]
    pub fn manifold_dim(&self) -> usize {
        self.k * (self.m + self.n - self.k)
    }

    /// Project the U and V factors onto the Stiefel manifold
    fn project_factors<T: Scalar>(&self, u: &mut DMatrix<T>, v: &mut DMatrix<T>) {
        // QR decomposition for U
        let qr_u = u.clone().qr();
        *u = qr_u.q();
        
        // QR decomposition for V
        let qr_v = v.clone().qr();
        *v = qr_v.q();
    }

    /// Validates that a matrix has the correct fixed rank.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that rank(X) = k using SVD.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix dimensions don't match (m,n)
    /// - `InvalidPoint`: If rank(X) ≠ k
    pub fn check_matrix<T: Scalar>(&self, x: &DMatrix<T>) -> Result<()> {
        if x.nrows() != self.m || x.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.m * self.n,
                x.nrows() * x.ncols()
            ));
        }

        // Check rank using SVD
        let svd = x.clone().svd(true, true);
        let s = &svd.singular_values;
        
        // Count non-zero singular values
        let mut rank = 0;
        for i in 0..s.len().min(self.m).min(self.n) {
            if s[i] > <T as Scalar>::from_f64(self.tolerance) {
                rank += 1;
            }
        }
        
        if rank != self.k {
            return Err(ManifoldError::invalid_point(format!(
                "Matrix rank {} ≠ required rank {}",
                rank, self.k
            )));
        }

        Ok(())
    }

    /// Validates that a matrix is a valid tangent vector at X.
    ///
    /// For now this is a placeholder that accepts all vectors.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions don't match
    pub fn check_tangent<T: Scalar>(&self, x: &DMatrix<T>, z: &DMatrix<T>) -> Result<()> {
        self.check_matrix(x)?;

        if z.nrows() != self.m || z.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.m * self.n,
                z.nrows() * z.ncols()
            ));
        }

        // Check tangent space constraint
        // For fixed-rank, tangent vectors have specific structure

        Ok(())
    }
}

impl<T: Scalar> Manifold<T> for FixedRank {
    type Point = FixedRankPoint<T>;
    type TangentVector = FixedRankTangent<T>;

    fn name(&self) -> &str {
        "FixedRank"
    }

    fn dimension(&self) -> usize {
        self.k * (self.m + self.n - self.k)
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        // Check that U and V are on Stiefel manifolds
        let u_gram = point.u.transpose() * &point.u;
        let v_gram = point.v.transpose() * &point.v;
        
        // Check orthogonality
        for i in 0..self.k {
            for j in 0..self.k {
                let u_val = if i == j { u_gram[(i, j)] - T::one() } else { u_gram[(i, j)] };
                let v_val = if i == j { v_gram[(i, j)] - T::one() } else { v_gram[(i, j)] };
                
                if <T as Float>::abs(u_val) > tol || <T as Float>::abs(v_val) > tol {
                    return false;
                }
            }
        }
        
        // Check that singular values are positive
        for i in 0..self.k {
            if point.s[i] <= T::zero() {
                return false;
            }
        }
        
        true
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
        // Copy the input point
        result.u = point.u.clone();
        result.s = point.s.clone();
        result.v = point.v.clone();
        
        // Project U and V onto Stiefel manifolds
        self.project_factors(&mut result.u, &mut result.v);
        
        // Ensure singular values are positive
        for i in 0..self.k {
            if result.s[i] < T::epsilon() {
                result.s[i] = T::epsilon();
            }
        }
    }

    fn project_tangent(
        &self,
        _point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For a tangent vector at point (U,S,V), the tangent space has the form:
        // ξ = U_perp * M * V^T + U * S_dot * V^T + U * N * V_perp^T
        // The input vector already has this structure, so we just copy it
        result.u_perp_m = vector.u_perp_m.clone();
        result.s_dot = vector.s_dot.clone();
        result.v_perp_n = vector.v_perp_n.clone();
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        // The inner product is the Frobenius inner product of the matrix representations
        // <u, v> = tr(u^T * v) = tr(U_perp*M_u*V^T + U*S_dot_u*V^T + U*N_u*V_perp^T)^T * 
        //                           (U_perp*M_v*V^T + U*S_dot_v*V^T + U*N_v*V_perp^T)
        // Since U, U_perp, V, V_perp are orthogonal, this simplifies to:
        // <u, v> = tr(M_u^T * M_v) + tr(S_dot_u * S_dot_v) + tr(N_u^T * N_v)
        
        let mut inner = T::zero();
        
        // U_perp component
        inner += (u.u_perp_m.transpose() * &v.u_perp_m).trace();
        
        // S component
        for i in 0..self.k {
            inner += u.s_dot[i] * v.s_dot[i];
        }
        
        // V_perp component
        inner += (u.v_perp_n.transpose() * &v.v_perp_n).trace();
        
        Ok(inner)
    }

    fn retract(
        &self,
        point: &Self::Point,
        tangent: &Self::TangentVector,
        result: &mut Self::Point,
    ) -> Result<()> {
        // Use the orthographic retraction for fixed-rank manifolds
        // R_X(ξ) = (U + U_perp*M*S^{-1})(S + S_dot)(V + V_perp*N^T*S^{-1})^T
        
        // Compute U_perp and V_perp
        let (u_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.u);
        let (v_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.v);
        
        // Compute S^{-1}
        let s_inv = DMatrix::from_diagonal(&point.s.map(|x| T::one() / x));
        
        // Update U: U_new = U + U_perp * M * S^{-1}
        result.u = &point.u + &u_perp * &tangent.u_perp_m * &s_inv;
        
        // Update S: S_new = S + S_dot
        result.s = &point.s + &tangent.s_dot;
        
        // Update V: V_new = V + V_perp * N^T * S^{-1}
        result.v = &point.v + &v_perp * tangent.v_perp_n.transpose() * &s_inv;
        
        // Project factors back to Stiefel
        self.project_factors(&mut result.u, &mut result.v);
        
        // Ensure singular values are positive
        for i in 0..self.k {
            if result.s[i] < T::epsilon() {
                result.s[i] = T::epsilon();
            }
        }
        
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For the canonical metric, just project to tangent space
        self.project_tangent(point, euclidean_grad, result)
    }

    fn random_point(&self) -> Self::Point {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random orthogonal matrices
        let mut u = DMatrix::zeros(self.m, self.k);
        let mut v = DMatrix::zeros(self.n, self.k);
        
        for j in 0..self.k {
            for i in 0..self.m {
                u[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            for i in 0..self.n {
                v[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // Orthogonalize
        let qr_u = u.qr();
        let u_orth = qr_u.q();
        
        let qr_v = v.qr();
        let v_orth = qr_v.q();
        
        // Random positive singular values
        let mut s = DVector::zeros(self.k);
        for i in 0..self.k {
            let val: f64 = normal.sample(&mut rng);
            s[i] = <T as Scalar>::from_f64(val.abs() + 1.0);
        }
        
        FixedRankPoint::new(u_orth, s, v_orth)
    }

    fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random matrices for the tangent components
        let mut u_perp_m = DMatrix::zeros(self.m - self.k, self.k);
        let mut s_dot = DVector::zeros(self.k);
        let mut v_perp_n = DMatrix::zeros(self.k, self.n - self.k);
        
        // Fill with random values
        for j in 0..self.k {
            for i in 0..(self.m - self.k) {
                u_perp_m[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            s_dot[j] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            for i in 0..(self.n - self.k) {
                v_perp_n[(j, i)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        result.u_perp_m = u_perp_m;
        result.s_dot = s_dot;
        result.v_perp_n = v_perp_n;
        Ok(())
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
        
        // Check dimensions of tangent components
        if vector.u_perp_m.nrows() != self.m - self.k || vector.u_perp_m.ncols() != self.k {
            return false;
        }
        if vector.s_dot.len() != self.k {
            return false;
        }
        if vector.v_perp_n.nrows() != self.k || vector.v_perp_n.ncols() != self.n - self.k {
            return false;
        }
        
        // Tangent vectors have the specific structure, so as long as dimensions match, it's valid
        true
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For fixed-rank manifold, we use a simple approximation
        // The inverse of the orthographic retraction is complex, so we approximate
        // by computing the tangent that moves in the direction of other - point
        
        // Compute the difference in matrix form
        let point_mat = point.to_matrix();
        let other_mat = other.to_matrix();
        let diff = &other_mat - &point_mat;
        
        // Project onto the tangent space at point
        // Compute U_perp and V_perp
        let (u_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.u);
        let (v_perp, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&point.v);
        
        // Decompose the difference into tangent components
        // M = U_perp^T * diff * V
        result.u_perp_m = u_perp.transpose() * &diff * &point.v;
        
        // S_dot = U^T * diff * V (diagonal part)
        let s_component = point.u.transpose() * &diff * &point.v;
        result.s_dot = s_component.diagonal();
        
        // N = U^T * diff * V_perp
        result.v_perp_n = point.u.transpose() * &diff * &v_perp;
        
        Ok(())
    }

    fn parallel_transport(
        &self,
        _from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For fixed-rank manifold, parallel transport is complex
        // We use a simple approximation: transport the tangent by adapting to the new basis
        
        // Compute U_perp and V_perp at the destination point
        let (_u_perp_to, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&to.u);
        let (_v_perp_to, _) = FixedRankTangent::<T>::compute_orthogonal_complement(&to.v);
        
        // For simplicity, we project the tangent vector's matrix representation
        // onto the tangent space at the destination
        // This preserves the general direction but may not be exact parallel transport
        
        // The transported tangent has the same structure but adapted to the new point
        result.u_perp_m = DMatrix::zeros(self.m - self.k, self.k);
        result.s_dot = vector.s_dot.clone();
        result.v_perp_n = DMatrix::zeros(self.k, self.n - self.k);
        
        // Fill with appropriate values (simplified transport)
        for j in 0..self.k {
            for i in 0..(self.m - self.k).min(vector.u_perp_m.nrows()) {
                if i < result.u_perp_m.nrows() && j < vector.u_perp_m.ncols() {
                    result.u_perp_m[(i, j)] = vector.u_perp_m[(i, j)];
                }
            }
            for i in 0..(self.n - self.k).min(vector.v_perp_n.ncols()) {
                if j < vector.v_perp_n.nrows() && i < result.v_perp_n.ncols() {
                    result.v_perp_n[(j, i)] = vector.v_perp_n[(j, i)];
                }
            }
        }
        
        Ok(())
    }
    fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
        // Use Frobenius distance in the embedded space
        let x_mat = x.to_matrix();
        let y_mat = y.to_matrix();
        let diff = &y_mat - &x_mat;
        
        // Frobenius norm of the difference
        let mut sum = T::zero();
        for i in 0..diff.nrows() {
            for j in 0..diff.ncols() {
                sum = sum + diff[(i, j)] * diff[(i, j)];
            }
        }
        
        Ok(<T as Float>::sqrt(sum))
    }

    fn has_exact_exp_log(&self) -> bool {
        false // Fixed-rank doesn't have closed-form exp/log
    }

    fn is_flat(&self) -> bool {
        false // Fixed-rank is curved
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // Scale each component of the tangent vector
        result.u_perp_m = &tangent.u_perp_m * scalar;
        result.s_dot = &tangent.s_dot * scalar;
        result.v_perp_n = &tangent.v_perp_n * scalar;
        Ok(())
    }

    fn add_tangents(
        &self,
        _point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        // Temporary buffer for projection if needed
        _temp: &mut Self::TangentVector,
    ) -> Result<()> {
        // Add each component of the tangent vectors
        result.u_perp_m = &v1.u_perp_m + &v2.u_perp_m;
        result.s_dot = &v1.s_dot + &v2.s_dot;
        result.v_perp_n = &v1.v_perp_n + &v2.v_perp_n;
        Ok(())
    }
}

impl FixedRank {
    /// Creates a random rank-k matrix using Gaussian sampling.
    ///
    /// # Returns
    /// 
    /// A random m×n matrix of rank k.
    pub fn random_matrix<T: Scalar>(&self) -> DMatrix<T> {
        let point = <Self as Manifold<T>>::random_point(self);
        point.to_matrix()
    }

    /// Computes the best rank-k approximation of a matrix.
    ///
    /// Uses SVD to compute the best rank-k approximation in Frobenius norm.
    ///
    /// # Arguments
    ///
    /// * `mat` - Input matrix
    ///
    /// # Returns
    ///
    /// The best rank-k approximation.
    pub fn approximate<T: Scalar>(&self, mat: &DMatrix<T>) -> Result<DMatrix<T>> {
        if mat.nrows() != self.m || mat.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.m * self.n,
                mat.nrows() * mat.ncols()
            ));
        }
        
        let pt = FixedRankPoint::<T>::from_matrix(mat, self.k)?;
        Ok(pt.to_matrix())
    }

    /// Computes the approximation error for a given matrix.
    ///
    /// # Mathematical Formula
    ///
    /// For a matrix A and its rank-k approximation A_k:
    /// ```text
    /// error = ‖A - A_k‖_F = √(σ_{k+1}² + ... + σ_{min(m,n)}²)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `mat` - Input matrix
    ///
    /// # Returns
    ///
    /// The Frobenius norm of the approximation error.
    pub fn approximation_error<T: Scalar>(&self, mat: &DMatrix<T>) -> Result<T> {
        if mat.nrows() != self.m || mat.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.m * self.n,
                mat.nrows() * mat.ncols()
            ));
        }
        
        let svd = mat.clone().svd(true, true);
        let s = &svd.singular_values;
        
        // Sum of squared singular values beyond rank k
        let mut error_sq = T::zero();
        for i in self.k..s.len().min(self.m).min(self.n) {
            error_sq = error_sq + s[i] * s[i];
        }
        
        Ok(<T as Float>::sqrt(error_sq))
    }

    /// Projects a general matrix to the fixed-rank manifold.
    ///
    /// # Mathematical Operation
    ///
    /// Computes the best rank-k approximation using SVD.
    ///
    /// # Arguments
    ///
    /// * `mat` - Input matrix of size m×n
    ///
    /// # Returns
    ///
    /// The projected fixed-rank matrix as a FixedRankPoint.
    pub fn project_matrix<T: Scalar>(&self, mat: &DMatrix<T>) -> Result<FixedRankPoint<T>> {
        if mat.nrows() != self.m || mat.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.m * self.n,
                mat.nrows() * mat.ncols()
            ));
        }
        
        FixedRankPoint::<T>::from_matrix(mat, self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_manifold() -> FixedRank {
        FixedRank::new(6, 4, 2).unwrap()
    }

    #[test]
    fn test_fixed_rank_creation() {
        let manifold = create_test_manifold();
        let (m, n, k) = manifold.matrix_dimensions();
        assert_eq!(m, 6);
        assert_eq!(n, 4);
        assert_eq!(k, 2);
        assert_eq!(<FixedRank as Manifold<f64>>::dimension(&manifold), 16); // 2*(6+4-2)
        
        // Test invalid creation
        assert!(FixedRank::new(0, 4, 2).is_err());
        assert!(FixedRank::new(4, 0, 2).is_err());
        assert!(FixedRank::new(4, 4, 0).is_err());
        assert!(FixedRank::new(4, 4, 5).is_err()); // k > min(m,n)
    }

    #[test]
    fn test_fixed_rank_point_conversion() {
        let _manifold = create_test_manifold();
        
        let u = DMatrix::from_fn(6, 2, |i, j| ((i + j) as f64).sin());
        let s = DVector::from_fn(2, |i, _| (i + 1) as f64);
        let v = DMatrix::from_fn(4, 2, |i, j| ((i * j) as f64).cos());
        
        let point = FixedRankPoint::new(u.clone(), s.clone(), v.clone());
        let mat = point.to_matrix();
        let point2 = FixedRankPoint::<f64>::from_matrix(&mat, 2).unwrap();
        
        // Check that conversion preserves the matrix
        let mat2 = point2.to_matrix();
        assert_relative_eq!(mat, mat2, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_rank_projection() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let mut projected = point.clone();
        manifold.project_point(&point, &mut projected);
        
        assert!(manifold.is_point_on_manifold(&projected, 1e-6));
    }

    #[test]
    fn test_fixed_rank_tangent_projection() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let tangent = FixedRankTangent::new(
            DMatrix::from_element(manifold.m - manifold.k, manifold.k, 0.1),
            DVector::from_element(manifold.k, 0.1),
            DMatrix::from_element(manifold.k, manifold.n - manifold.k, 0.1)
        );
        let mut tangent2 = tangent.clone();
        manifold.project_tangent(&point, &tangent, &mut tangent2).unwrap();
        
        // Check that projection is idempotent
        let mut tangent3 = tangent2.clone();
        manifold.project_tangent(&point, &tangent2, &mut tangent3).unwrap();
        assert_relative_eq!(tangent2.u_perp_m, tangent3.u_perp_m, epsilon = 1e-10);
        assert_relative_eq!(tangent2.s_dot, tangent3.s_dot, epsilon = 1e-10);
        assert_relative_eq!(tangent2.v_perp_n, tangent3.v_perp_n, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_rank_retraction() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let mut tangent = FixedRankTangent::new(
            DMatrix::zeros(manifold.m - manifold.k, manifold.k),
            DVector::zeros(manifold.k),
            DMatrix::zeros(manifold.k, manifold.n - manifold.k)
        );
        manifold.random_tangent(&point, &mut tangent).unwrap();
        
        // Scale the tangent
        let mut scaled_tangent = tangent.clone();
        manifold.scale_tangent(&point, 0.1, &tangent, &mut scaled_tangent).unwrap();
        
        let mut retracted = point.clone();
        manifold.retract(&point, &scaled_tangent, &mut retracted).unwrap();
        
        assert!(manifold.is_point_on_manifold(&retracted, 1e-6));
    }

    #[test]
    fn test_fixed_rank_properties() {
        let manifold = create_test_manifold();
        
        assert_eq!(<FixedRank as Manifold<f64>>::name(&manifold), "FixedRank");
        assert!(!<FixedRank as Manifold<f64>>::has_exact_exp_log(&manifold));
        assert!(!<FixedRank as Manifold<f64>>::is_flat(&manifold));
    }

    #[test]
    fn test_fixed_rank_inner_product() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let mut u = FixedRankTangent::new(
            DMatrix::zeros(manifold.m - manifold.k, manifold.k),
            DVector::zeros(manifold.k),
            DMatrix::zeros(manifold.k, manifold.n - manifold.k)
        );
        let mut v = u.clone();
        manifold.random_tangent(&point, &mut u).unwrap();
        manifold.random_tangent(&point, &mut v).unwrap();
        
        let ip_uv = manifold.inner_product(&point, &u, &v).unwrap();
        let ip_vu = manifold.inner_product(&point, &v, &u).unwrap();
        
        // Check symmetry
        assert_relative_eq!(ip_uv, ip_vu, epsilon = 1e-10);
        
        // Check positive definiteness
        let ip_uu = manifold.inner_product(&point, &u, &u).unwrap();
        assert!(ip_uu >= 0.0);
    }

    #[test]
    fn test_fixed_rank_matrix_operations() {
        let manifold = FixedRank::new(5, 4, 2).unwrap();
        
        // Test random matrix generation
        let mat = manifold.random_matrix::<f64>();
        assert_eq!(mat.nrows(), 5);
        assert_eq!(mat.ncols(), 4);
        assert!(mat.rank(1e-10) <= 2);
        
        // Test approximation
        let full_rank_mat = DMatrix::from_fn(5, 4, |i, j| ((i + j) as f64).sin());
        let approx = manifold.approximate(&full_rank_mat).unwrap();
        assert_eq!(approx.nrows(), 5);
        assert_eq!(approx.ncols(), 4);
        assert!(approx.rank(1e-10) <= 2);
        
        // Test approximation error
        let error = manifold.approximation_error(&full_rank_mat).unwrap();
        assert!(error >= 0.0);
        
        // Test projection
        let proj_point = manifold.project_matrix(&full_rank_mat).unwrap();
        assert_eq!(proj_point.u.nrows(), 5);
        assert_eq!(proj_point.u.ncols(), 2);
        assert_eq!(proj_point.s.len(), 2);
        assert_eq!(proj_point.v.nrows(), 4);
        assert_eq!(proj_point.v.ncols(), 2);
    }

    #[test]
    fn test_fixed_rank_with_tolerance() {
        let manifold = FixedRank::with_tolerance(4, 3, 2, 1e-8).unwrap();
        assert_eq!(manifold.tolerance, 1e-8);
        
        // Test invalid tolerance
        assert!(FixedRank::with_tolerance(4, 3, 2, 0.0).is_err());
        assert!(FixedRank::with_tolerance(4, 3, 2, 1.0).is_err());
    }

    #[test]
    fn test_fixed_rank_distance() {
        let manifold = create_test_manifold();
        let x = manifold.random_point();
        let y = manifold.random_point();
        
        // Distance to self should be zero
        let d_xx = manifold.distance(&x, &x).unwrap();
        assert_relative_eq!(d_xx, 0.0, epsilon = 1e-10);
        
        // Distance should be symmetric
        let d_xy = manifold.distance(&x, &y).unwrap();
        let d_yx = manifold.distance(&y, &x).unwrap();
        assert_relative_eq!(d_xy, d_yx, epsilon = 1e-10);
        
        // Distance should be non-negative
        assert!(d_xy >= 0.0);
    }
}