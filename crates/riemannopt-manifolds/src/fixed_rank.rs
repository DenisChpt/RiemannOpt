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
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DMatrix;
//!
//! // Create M_2(4,3) - 4×3 matrices of rank 2
//! let manifold = FixedRank::new(4, 3, 2)?;
//!
//! // Random rank-2 matrix
//! let x = manifold.random_point();
//!
//! // Convert to matrix form
//! let x_mat = manifold.vector_to_matrix(&x);
//! assert_eq!(x_mat.rank(1e-10), 2);
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::{DMatrix, DVector, SVD};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    memory::workspace::Workspace,
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

    /// Convert to vector representation for manifold operations
    pub fn to_vector(&self) -> DVector<T> {
        let m = self.u.nrows();
        let n = self.v.nrows();
        let k = self.s.len();
        
        let mut vec = DVector::zeros(m * k + k + n * k);
        let mut idx = 0;
        
        // Pack U
        for j in 0..k {
            for i in 0..m {
                vec[idx] = self.u[(i, j)];
                idx += 1;
            }
        }
        
        // Pack S
        for i in 0..k {
            vec[idx] = self.s[i];
            idx += 1;
        }
        
        // Pack V
        for j in 0..k {
            for i in 0..n {
                vec[idx] = self.v[(i, j)];
                idx += 1;
            }
        }
        
        vec
    }

    /// Create from vector representation
    pub fn from_vector(vec: &DVector<T>, m: usize, n: usize, k: usize) -> Self {
        let mut idx = 0;
        
        // Unpack U
        let mut u = DMatrix::zeros(m, k);
        for j in 0..k {
            for i in 0..m {
                u[(i, j)] = vec[idx];
                idx += 1;
            }
        }
        
        // Unpack S
        let mut s = DVector::zeros(k);
        for i in 0..k {
            s[i] = vec[idx];
            idx += 1;
        }
        
        // Unpack V
        let mut v = DMatrix::zeros(n, k);
        for j in 0..k {
            for i in 0..n {
                v[(i, j)] = vec[idx];
                idx += 1;
            }
        }
        
        Self::new(u, s, v)
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

    /// Returns the size of the vectorized representation.
    #[inline]
    fn vector_size(&self) -> usize {
        self.m * self.k + self.k + self.n * self.k
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
    type Point = DVector<T>;
    type TangentVector = DVector<T>;

    fn name(&self) -> &str {
        "FixedRank"
    }

    fn dimension(&self) -> usize {
        self.k * (self.m + self.n - self.k)
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.len() != self.vector_size() {
            return false;
        }
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        
        // Check that U and V are on Stiefel manifolds
        let u_gram = pt.u.transpose() * &pt.u;
        let v_gram = pt.v.transpose() * &pt.v;
        
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
            if pt.s[i] <= T::zero() {
                return false;
            }
        }
        
        true
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, _workspace: &mut Workspace<T>) {
        let vec_size = self.vector_size();
        
        // Ensure result has correct size
        if result.len() != vec_size {
            *result = DVector::zeros(vec_size);
        }
        
        let mut pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        
        // Project U and V onto Stiefel manifolds
        self.project_factors(&mut pt.u, &mut pt.v);
        
        // Ensure singular values are positive
        for i in 0..self.k {
            if pt.s[i] < T::epsilon() {
                pt.s[i] = T::epsilon();
            }
        }
        
        let projected = pt.to_vector();
        result.copy_from(&projected);
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let vec_size = self.vector_size();
        if point.len() != vec_size || vector.len() != vec_size {
            return Err(ManifoldError::dimension_mismatch(
                vec_size,
                point.len().max(vector.len())
            ));
        }
        
        // Ensure result has correct size
        if result.len() != vec_size {
            *result = DVector::zeros(vec_size);
        }
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let tangent = FixedRankPoint::from_vector(vector, self.m, self.n, self.k);
        
        // Project U and V components to tangent spaces of Stiefel manifolds
        let u_proj = &tangent.u - &pt.u * (pt.u.transpose() * &tangent.u);
        let v_proj = &tangent.v - &pt.v * (pt.v.transpose() * &tangent.v);
        
        let proj_tangent = FixedRankPoint::new(u_proj, tangent.s.clone(), v_proj);
        
        let projected = proj_tangent.to_vector();
        result.copy_from(&projected);
        Ok(())
    }

    fn inner_product(
        &self,
        point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let u_tan = FixedRankPoint::from_vector(u, self.m, self.n, self.k);
        let v_tan = FixedRankPoint::from_vector(v, self.m, self.n, self.k);
        
        // Compute the inner product with scaling by singular values
        let mut inner = T::zero();
        
        // U component
        for i in 0..self.m {
            for j in 0..self.k {
                inner += u_tan.u[(i, j)] * v_tan.u[(i, j)];
            }
        }
        
        // S component (scaled)
        for i in 0..self.k {
            inner += u_tan.s[i] * v_tan.s[i] / pt.s[i];
        }
        
        // V component
        for i in 0..self.n {
            for j in 0..self.k {
                inner += u_tan.v[(i, j)] * v_tan.v[(i, j)];
            }
        }
        
        Ok(inner)
    }

    fn retract(
        &self,
        point: &Self::Point,
        tangent: &Self::TangentVector,
        result: &mut Self::Point,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let vec_size = self.vector_size();
        if point.len() != vec_size || tangent.len() != vec_size {
            return Err(ManifoldError::dimension_mismatch(
                vec_size,
                point.len().max(tangent.len())
            ));
        }
        
        // Ensure result has correct size
        if result.len() != vec_size {
            *result = DVector::zeros(vec_size);
        }
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let tan = FixedRankPoint::from_vector(tangent, self.m, self.n, self.k);
        
        // Retract using projection
        let new_u = &pt.u + &tan.u;
        let new_s = &pt.s + &tan.s;
        let new_v = &pt.v + &tan.v;
        
        let mut new_pt = FixedRankPoint::new(new_u, new_s, new_v);
        
        // Project factors back to Stiefel
        self.project_factors(&mut new_pt.u, &mut new_pt.v);
        
        // Ensure singular values are positive
        for i in 0..self.k {
            if new_pt.s[i] < T::epsilon() {
                new_pt.s[i] = T::epsilon();
            }
        }
        
        let retracted = new_pt.to_vector();
        result.copy_from(&retracted);
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For the canonical metric, just project to tangent space
        self.project_tangent(point, euclidean_grad, result, workspace)
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
        
        let pt = FixedRankPoint::new(u_orth, s, v_orth);
        pt.to_vector()
    }

    fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector, _workspace: &mut Workspace<T>) -> Result<()> {
        let vec_size = self.vector_size();
        if point.len() != vec_size {
            return Err(ManifoldError::dimension_mismatch(
                vec_size,
                point.len()
            ));
        }
        
        // Ensure result has correct size
        if result.len() != vec_size {
            *result = DVector::zeros(vec_size);
        }
        
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        
        // Generate random matrices
        let mut u_tan = DMatrix::zeros(self.m, self.k);
        let mut v_tan = DMatrix::zeros(self.n, self.k);
        let mut s_tan = DVector::zeros(self.k);
        
        for j in 0..self.k {
            for i in 0..self.m {
                u_tan[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            for i in 0..self.n {
                v_tan[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            s_tan[j] = <T as Scalar>::from_f64(normal.sample(&mut rng));
        }
        
        // Project to tangent spaces of Stiefel manifolds
        u_tan = &u_tan - &pt.u * (pt.u.transpose() * &u_tan);
        v_tan = &v_tan - &pt.v * (pt.v.transpose() * &v_tan);
        
        let tangent = FixedRankPoint::new(u_tan, s_tan, v_tan);
        let tangent_vec = tangent.to_vector();
        result.copy_from(&tangent_vec);
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
        
        if vector.len() != self.vector_size() {
            return false;
        }
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let vec = FixedRankPoint::from_vector(vector, self.m, self.n, self.k);
        
        // Check that U and V components are in tangent spaces of Stiefel manifolds
        let u_proj = pt.u.transpose() * &vec.u;
        let v_proj = pt.v.transpose() * &vec.v;
        
        // Check that projections are skew-symmetric
        for i in 0..self.k {
            for j in 0..self.k {
                if <T as Float>::abs(u_proj[(i, j)] + u_proj[(j, i)]) > tol {
                    return false;
                }
                if <T as Float>::abs(v_proj[(i, j)] + v_proj[(j, i)]) > tol {
                    return false;
                }
            }
        }
        
        true
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let vec_size = self.vector_size();
        if point.len() != vec_size || other.len() != vec_size {
            return Err(ManifoldError::dimension_mismatch(
                vec_size,
                point.len().max(other.len())
            ));
        }
        
        // Ensure result has correct size
        if result.len() != vec_size {
            *result = DVector::zeros(vec_size);
        }
        
        // Simple approximation: project the difference
        let diff = other - point;
        self.project_tangent(point, &diff, result, workspace)
    }

    fn parallel_transport(
        &self,
        from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let vec_size = self.vector_size();
        if from.len() != vec_size || to.len() != vec_size || vector.len() != vec_size {
            return Err(ManifoldError::dimension_mismatch(
                vec_size,
                from.len().max(to.len()).max(vector.len())
            ));
        }
        
        // Ensure result has correct size
        if result.len() != vec_size {
            *result = DVector::zeros(vec_size);
        }
        
        // For fixed-rank manifold, we use projection to tangent space at destination
        // This is an approximation; exact parallel transport requires more computation
        self.project_tangent(to, vector, result, workspace)
    }
    fn distance(&self, x: &Self::Point, y: &Self::Point, _workspace: &mut Workspace<T>) -> Result<T> {
        if x.len() != self.vector_size() || y.len() != self.vector_size() {
            return Err(ManifoldError::dimension_mismatch(
                self.vector_size(),
                x.len().max(y.len())
            ));
        }
        
        // Use Frobenius distance in the embedded space
        let diff = y - x;
        Ok(<T as Float>::sqrt(diff.dot(&diff)))
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
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For fixed-rank manifold represented as vectorized (U, S, V),
        // scaling preserves the tangent space constraints
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
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Add the tangent vectors
        result.copy_from(v1);
        *result += v2;
        
        // The sum should already satisfy the tangent space constraints if v1 and v2 do,
        // but we project for numerical stability
        // Create a temporary clone to avoid borrowing issues
        let temp = result.clone();
        self.project_tangent(point, &temp, result, workspace)?;
        
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
        let pt = FixedRankPoint::<T>::from_vector(&point, self.m, self.n, self.k);
        pt.to_matrix()
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
    /// The projected fixed-rank matrix as a vector.
    pub fn project_matrix<T: Scalar>(&self, mat: &DMatrix<T>) -> Result<DVector<T>> {
        if mat.nrows() != self.m || mat.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.m * self.n,
                mat.nrows() * mat.ncols()
            ));
        }
        
        let pt = FixedRankPoint::<T>::from_matrix(mat, self.k)?;
        Ok(pt.to_vector())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use riemannopt_core::memory::workspace::Workspace;

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
        let vec = point.to_vector();
        let point2 = FixedRankPoint::<f64>::from_vector(&vec, 6, 4, 2);
        
        // Check reconstruction
        assert_relative_eq!(point2.u, u, epsilon = 1e-10);
        assert_relative_eq!(point2.s, s, epsilon = 1e-10);
        assert_relative_eq!(point2.v, v, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_rank_projection() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let mut projected = DVector::zeros(manifold.vector_size());
        let mut workspace = Workspace::<f64>::new();
        manifold.project_point(&point, &mut projected, &mut workspace);
        
        assert!(manifold.is_point_on_manifold(&projected, 1e-6));
    }

    #[test]
    fn test_fixed_rank_tangent_projection() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let vector = DVector::<f64>::from_vec(vec![0.1; manifold.vector_size()]);
        let mut tangent = DVector::zeros(manifold.vector_size());
        let mut workspace = Workspace::<f64>::new();
        manifold.project_tangent(&point, &vector, &mut tangent, &mut workspace).unwrap();
        
        // Check that projection is idempotent
        let mut tangent2 = DVector::zeros(manifold.vector_size());
        let mut workspace = Workspace::<f64>::new();
        manifold.project_tangent(&point, &tangent, &mut tangent2, &mut workspace).unwrap();
        assert_relative_eq!(&tangent, &tangent2, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_rank_retraction() {
        let manifold = create_test_manifold();
        
        let point = manifold.random_point();
        let mut tangent = DVector::zeros(manifold.vector_size());
        let mut workspace = Workspace::<f64>::new();
        manifold.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        let scaled_tangent = 0.1 * &tangent;
        let mut retracted = DVector::zeros(manifold.vector_size());
        let mut workspace = Workspace::<f64>::new();
        manifold.retract(&point, &scaled_tangent, &mut retracted, &mut workspace).unwrap();
        
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
        let mut u = DVector::zeros(manifold.vector_size());
        let mut v = DVector::zeros(manifold.vector_size());
        let mut workspace = Workspace::<f64>::new();
        manifold.random_tangent(&point, &mut u, &mut workspace).unwrap();
        manifold.random_tangent(&point, &mut v, &mut workspace).unwrap();
        
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
        let proj_vec = manifold.project_matrix(&full_rank_mat).unwrap();
        assert_eq!(proj_vec.len(), manifold.vector_size());
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
        let mut workspace = Workspace::<f64>::new();
        
        let x = manifold.random_point();
        let y = manifold.random_point();
        
        // Distance to self should be zero
        let d_xx = manifold.distance(&x, &x, &mut workspace).unwrap();
        assert_relative_eq!(d_xx, 0.0, epsilon = 1e-10);
        
        // Distance should be symmetric
        let d_xy = manifold.distance(&x, &y, &mut workspace).unwrap();
        let d_yx = manifold.distance(&y, &x, &mut workspace).unwrap();
        assert_relative_eq!(d_xy, d_yx, epsilon = 1e-10);
        
        // Distance should be non-negative
        assert!(d_xy >= 0.0);
    }
}