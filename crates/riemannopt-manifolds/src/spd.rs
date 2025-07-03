//! # Symmetric Positive Definite Manifold S⁺⁺(n)
//!
//! The manifold S⁺⁺(n) of n×n symmetric positive definite (SPD) matrices is fundamental
//! in many areas of mathematics and applications. It provides a rich geometric structure
//! for problems involving covariance matrices, metric tensors, and kernel matrices.
//!
//! ## Mathematical Definition
//!
//! The SPD manifold is formally defined as:
//! ```text
//! S⁺⁺(n) = {P ∈ ℝⁿˣⁿ : P = P^T, x^T P x > 0 ∀x ≠ 0}
//! ```
//!
//! Equivalently, P ∈ S⁺⁺(n) if and only if P is symmetric with all positive eigenvalues.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space at P ∈ S⁺⁺(n) consists of all symmetric matrices:
//! ```text
//! T_P S⁺⁺(n) = {V ∈ ℝⁿˣⁿ : V = V^T} ≅ ℝ^{n(n+1)/2}
//! ```
//!
//! ### Riemannian Metrics
//!
//! #### 1. Affine-Invariant Metric (AI)
//! The canonical metric, invariant under congruence transformations:
//! ```text
//! g_P^{AI}(U, V) = tr(P⁻¹ U P⁻¹ V)
//! ```
//!
//! #### 2. Log-Euclidean Metric (LE)
//! Pulls back the Euclidean metric through the matrix logarithm:
//! ```text
//! g_P^{LE}(U, V) = tr((dlog_P U)^T (dlog_P V))
//! ```
//!
//! #### 3. Bures-Wasserstein Metric (BW)
//! Optimal transport metric on covariance matrices:
//! ```text
//! g_P^{BW}(U, V) = ¼ tr(U P⁻¹ V + V P⁻¹ U)
//! ```
//!
//! ### Exponential and Logarithmic Maps
//!
//! For the affine-invariant metric:
//! ```text
//! exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
//! log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
//! ```
//!
//! ### Parallel Transport
//!
//! Along geodesic from P to Q:
//! ```text
//! Γ_{P→Q}(V) = E V E^T
//! ```
//! where E = (PQ)^{1/2} P^{-1/2}.
//!
//! ## Geodesics and Distance
//!
//! ### Affine-Invariant Distance
//! ```text
//! d_{AI}(P, Q) = ‖log(P^{-1/2} Q P^{-1/2})‖_F
//!              = [∑ᵢ log²(λᵢ)]^{1/2}
//! ```
//! where λᵢ are eigenvalues of P⁻¹Q.
//!
//! ### Log-Euclidean Distance
//! ```text
//! d_{LE}(P, Q) = ‖log(P) - log(Q)‖_F
//! ```
//!
//! ### Bures-Wasserstein Distance
//! ```text
//! d_{BW}(P, Q) = [tr(P + Q - 2(P^{1/2} Q P^{1/2})^{1/2})]^{1/2}
//! ```
//!
//! ## Geometric Properties
//!
//! - **Dimension**: dim(S⁺⁺(n)) = n(n+1)/2
//! - **Sectional curvature**: -1/4 ≤ K ≤ 0 (nonpositive)
//! - **Geodesically complete**: Yes
//! - **Simply connected**: Yes
//! - **Symmetric space**: S⁺⁺(n) ≅ GL(n)/O(n)
//!
//! ## Optimization on S⁺⁺(n)
//!
//! ### Riemannian Gradient
//! For f: S⁺⁺(n) → ℝ with Euclidean gradient ∇f(P):
//! ```text
//! grad^{AI} f(P) = P ∇f(P) P
//! grad^{LE} f(P) = P ∇f(P) + ∇f(P) P
//! ```
//!
//! ### Applications
//!
//! 1. **Covariance estimation**: Regularized covariance matrices
//! 2. **Metric learning**: Learning distance metrics
//! 3. **Diffusion tensor imaging**: Processing brain imaging data
//! 4. **Computer vision**: Region covariance descriptors
//! 5. **Machine learning**: Kernel matrix optimization
//! 6. **Signal processing**: Radar/sonar covariance tracking
//!
//! ## Numerical Considerations
//!
//! This implementation provides:
//! - **Numerical stability** through careful eigendecomposition
//! - **Efficient computation** using matrix square roots
//! - **Robust projection** with eigenvalue thresholding
//! - **Multiple metrics** support for different applications
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::SPD;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DMatrix;
//!
//! // Create SPD manifold S⁺⁺(3)
//! let spd = SPD::<f64>::new(3)?;
//!
//! // Random SPD matrix
//! let p = spd.random_point();
//! 
//! // Verify positive definiteness
//! let eigen = p.clone().symmetric_eigen();
//! assert!(eigen.eigenvalues.iter().all(|&λ| λ > 0.0));
//!
//! // Tangent vector (symmetric)
//! let v = DMatrix::from_fn(3, 3, |i, j| {
//!     let val = (i + j) as f64;
//!     if i <= j { val } else { v[(j, i)] } // Ensure symmetry
//! });
//!
//! // Exponential map
//! let mut workspace = Workspace::<f64>::new();
//! let mut q = DMatrix::zeros(3, 3);
//! spd.retract(&p, &v, &mut q, &mut workspace)?;
//! 
//! // Verify result is SPD
//! assert!(spd.is_point_on_manifold(&q, 1e-10));
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    memory::workspace::Workspace,
    types::Scalar,
};
use std::fmt::{self, Debug};

/// The manifold S⁺⁺(n) of symmetric positive definite matrices.
///
/// This structure represents the space of all n×n symmetric positive definite
/// matrices, equipped with the affine-invariant Riemannian metric by default.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `n ≥ 1`: Matrix dimension must be positive
/// - All points P satisfy P = P^T and have positive eigenvalues
/// - All tangent vectors are symmetric matrices
#[derive(Clone)]
pub struct SPD<T = f64> {
    /// Matrix dimension n
    n: usize,
    /// Minimum eigenvalue threshold for numerical stability
    min_eigenvalue: T,
    /// Numerical tolerance for validations
    tolerance: T,
    /// Metric type to use
    metric: SPDMetric,
}

/// Available Riemannian metrics on the SPD manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SPDMetric {
    /// Affine-invariant metric (default)
    AffineInvariant,
    /// Log-Euclidean metric
    LogEuclidean,
    /// Bures-Wasserstein metric
    BuresWasserstein,
}

impl<T: Scalar> Debug for SPD<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SPD S⁺⁺({}) with {:?} metric", self.n, self.metric)
    }
}

impl<T: Scalar> SPD<T> {
    /// Creates a new SPD manifold S⁺⁺(n) with affine-invariant metric.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension (must be ≥ 1)
    ///
    /// # Returns
    ///
    /// An SPD manifold with dimension n(n+1)/2.
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if n = 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::SPD;
    /// // Create S⁺⁺(3) for 3×3 SPD matrices
    /// let spd = SPD::<f64>::new(3)?;
    /// 
    /// // S⁺⁺(1) is the positive real line
    /// let spd1 = SPD::<f64>::new(1)?;
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_parameter(
                "SPD manifold requires n ≥ 1",
            ));
        }
        Ok(Self {
            n,
            min_eigenvalue: <T as Scalar>::from_f64(1e-12),
            tolerance: <T as Scalar>::from_f64(1e-12),
            metric: SPDMetric::AffineInvariant,
        })
    }

    /// Creates an SPD manifold with specified metric.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension
    /// * `metric` - Choice of Riemannian metric
    pub fn with_metric(n: usize, metric: SPDMetric) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_parameter(
                "SPD manifold requires n ≥ 1",
            ));
        }
        Ok(Self {
            n,
            min_eigenvalue: <T as Scalar>::from_f64(1e-12),
            tolerance: <T as Scalar>::from_f64(1e-12),
            metric,
        })
    }

    /// Creates an SPD manifold with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension
    /// * `min_eigenvalue` - Minimum eigenvalue threshold
    /// * `tolerance` - Numerical tolerance
    /// * `metric` - Choice of Riemannian metric
    pub fn with_parameters(
        n: usize,
        min_eigenvalue: T,
        tolerance: T,
        metric: SPDMetric,
    ) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_parameter(
                "SPD manifold requires n ≥ 1",
            ));
        }
        if min_eigenvalue <= T::zero() {
            return Err(ManifoldError::invalid_parameter(
                "Minimum eigenvalue must be positive",
            ));
        }
        if tolerance <= T::zero() || tolerance >= T::one() {
            return Err(ManifoldError::invalid_parameter(
                "Tolerance must be in (0, 1)",
            ));
        }
        Ok(Self {
            n,
            min_eigenvalue,
            tolerance,
            metric,
        })
    }

    /// Returns the matrix dimension n.
    #[inline]
    pub fn matrix_dim(&self) -> usize {
        self.n
    }

    /// Returns the current metric type.
    #[inline]
    pub fn metric_type(&self) -> SPDMetric {
        self.metric
    }

    /// Validates that a matrix is symmetric positive definite.
    ///
    /// # Mathematical Checks
    ///\n    /// 1. Symmetry: ‖P - P^T‖ ≤ tolerance\n    /// 2. Positive definiteness: λᵢ(P) > min_eigenvalue ∀i
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix is not n×n
    /// - `NotOnManifold`: If matrix is not symmetric or not positive definite
    pub fn check_point(&self, p: &DMatrix<T>) -> Result<()> {
        if p.nrows() != self.n || p.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.n,
                p.nrows() * p.ncols()
            ));
        }

        // Check for finite values
        if p.iter().any(|x| !x.is_finite()) {
            return Err(ManifoldError::invalid_point(
                "Matrix contains non-finite values",
            ));
        }

        // Check symmetry
        let symmetry_error = (p - &p.transpose()).norm();
        if symmetry_error > self.tolerance {
            return Err(ManifoldError::invalid_point(format!(
                "Matrix not symmetric: ‖P - P^T‖ = {} (tolerance: {})",
                symmetry_error, self.tolerance
            )));
        }

        // Check positive definiteness via eigenvalues
        let eigen = p.clone().symmetric_eigen();
        let min_eval = eigen.eigenvalues.iter()
            .fold(T::infinity(), |min, &val| <T as Float>::min(min, val));
        
        if min_eval <= self.min_eigenvalue {
            return Err(ManifoldError::invalid_point(format!(
                "Matrix not positive definite: min eigenvalue = {} (threshold: {})",
                min_eval, self.min_eigenvalue
            )));
        }

        Ok(())
    }

    /// Validates that a matrix is symmetric (in tangent space).
    ///
    /// # Mathematical Check
    ///
    /// Verifies that V = V^T within tolerance.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions don't match
    /// - `NotInTangentSpace`: If matrix is not symmetric
    pub fn check_tangent(&self, p: &DMatrix<T>, v: &DMatrix<T>) -> Result<()> {
        self.check_point(p)?;

        if v.nrows() != self.n || v.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.n,
                v.nrows() * v.ncols()
            ));
        }

        // Check symmetry
        let symmetry_error = (v - &v.transpose()).norm();
        if symmetry_error > self.tolerance {
            return Err(ManifoldError::invalid_tangent(format!(
                "Tangent vector not symmetric: ‖V - V^T‖ = {} (tolerance: {})",
                symmetry_error, self.tolerance
            )));
        }

        Ok(())
    }

    /// Computes the matrix square root P^{1/2}.
    ///
    /// # Mathematical Formula
    ///
    /// For P = V Λ V^T, returns P^{1/2} = V Λ^{1/2} V^T.
    pub fn matrix_sqrt(&self, p: &DMatrix<T>) -> Result<DMatrix<T>> {
        let eigen = p.clone().symmetric_eigen();
        let mut sqrt_vals = DVector::zeros(self.n);
        for (i, &x) in eigen.eigenvalues.iter().enumerate() {
            if x <= T::zero() {
                return Err(ManifoldError::numerical_error(
                    "Cannot compute square root of non-positive eigenvalue".to_string(),
                ));
            }
            sqrt_vals[i] = <T as Float>::sqrt(x);
        }
        
        let sqrt_diag = DMatrix::from_diagonal(&sqrt_vals);
        Ok(&eigen.eigenvectors * sqrt_diag * eigen.eigenvectors.transpose())
    }

    /// Computes the inverse square root P^{-1/2}.
    pub fn matrix_sqrt_inv(&self, p: &DMatrix<T>) -> Result<DMatrix<T>> {
        let eigen = p.clone().symmetric_eigen();
        let mut sqrt_inv_vals = DVector::zeros(self.n);
        
        for (i, &eval) in eigen.eigenvalues.iter().enumerate() {
            if eval <= self.min_eigenvalue {
                return Err(ManifoldError::numerical_error(
                    "Matrix too close to singular for inverse square root".to_string(),
                ));
            }
            sqrt_inv_vals[i] = T::one() / <T as Float>::sqrt(eval);
        }
        
        let sqrt_inv_diag = DMatrix::from_diagonal(&sqrt_inv_vals);
        Ok(&eigen.eigenvectors * sqrt_inv_diag * eigen.eigenvectors.transpose())
    }

    /// Computes the matrix logarithm log(P).
    pub fn matrix_log(&self, p: &DMatrix<T>) -> Result<DMatrix<T>> {
        let eigen = p.clone().symmetric_eigen();
        let mut log_vals = DVector::zeros(self.n);
        
        for (i, &eval) in eigen.eigenvalues.iter().enumerate() {
            if eval <= T::zero() {
                return Err(ManifoldError::numerical_error(
                    "Cannot compute logarithm of non-positive eigenvalue".to_string(),
                ));
            }
            log_vals[i] = <T as Float>::ln(eval);
        }
        
        let log_diag = DMatrix::from_diagonal(&log_vals);
        Ok(&eigen.eigenvectors * log_diag * eigen.eigenvectors.transpose())
    }

    /// Computes the matrix exponential exp(X).
    pub fn matrix_exp(&self, x: &DMatrix<T>) -> DMatrix<T> {
        let eigen = x.clone().symmetric_eigen();
        let exp_vals = eigen.eigenvalues.map(|x| <T as Float>::exp(x));
        let exp_diag = DMatrix::from_diagonal(&exp_vals);
        &eigen.eigenvectors * exp_diag * eigen.eigenvectors.transpose()
    }

    /// Exponential map at P in direction V.
    ///
    /// # Mathematical Formula
    ///
    /// exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
    pub fn exp_map(&self, p: &DMatrix<T>, v: &DMatrix<T>) -> Result<DMatrix<T>> {
        let p_sqrt = self.matrix_sqrt(p)?;
        let p_sqrt_inv = self.matrix_sqrt_inv(p)?;
        
        // Compute P^{-1/2} V P^{-1/2}
        let middle = &p_sqrt_inv * v * &p_sqrt_inv;
        
        // Compute exp(middle)
        let exp_middle = self.matrix_exp(&middle);
        
        // Return P^{1/2} exp(middle) P^{1/2}
        Ok(&p_sqrt * exp_middle * &p_sqrt)
    }

    /// Logarithmic map from P to Q.
    ///
    /// # Mathematical Formula
    ///
    /// log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
    pub fn log_map(&self, p: &DMatrix<T>, q: &DMatrix<T>) -> Result<DMatrix<T>> {
        let p_sqrt = self.matrix_sqrt(p)?;
        let p_sqrt_inv = self.matrix_sqrt_inv(p)?;
        
        // Compute P^{-1/2} Q P^{-1/2}
        let middle = &p_sqrt_inv * q * &p_sqrt_inv;
        
        // Compute log(middle)
        let log_middle = self.matrix_log(&middle)?;
        
        // Return P^{1/2} log(middle) P^{1/2}
        Ok(&p_sqrt * log_middle * &p_sqrt)
    }

    /// Computes distance between SPD matrices using selected metric.
    ///
    /// # Mathematical Formulas
    ///
    /// - Affine-invariant: d(P,Q) = ‖log(P^{-1/2} Q P^{-1/2})‖_F
    /// - Log-Euclidean: d(P,Q) = ‖log(P) - log(Q)‖_F
    /// - Bures-Wasserstein: d(P,Q) = [tr(P + Q - 2(P^{1/2} Q P^{1/2})^{1/2})]^{1/2}
    pub fn distance(&self, p: &DMatrix<T>, q: &DMatrix<T>) -> Result<T> {
        self.check_point(p)?;
        self.check_point(q)?;

        match self.metric {
            SPDMetric::AffineInvariant => {
                // d(P,Q) = ‖log(P^{-1/2} Q P^{-1/2})‖_F
                let p_sqrt_inv = self.matrix_sqrt_inv(p)?;
                let middle = &p_sqrt_inv * q * &p_sqrt_inv;
                let log_middle = self.matrix_log(&middle)?;
                Ok(log_middle.norm())
            }
            SPDMetric::LogEuclidean => {
                // d(P,Q) = ‖log(P) - log(Q)‖_F
                let log_p = self.matrix_log(p)?;
                let log_q = self.matrix_log(q)?;
                Ok((&log_p - &log_q).norm())
            }
            SPDMetric::BuresWasserstein => {
                // d(P,Q) = [tr(P + Q - 2(P^{1/2} Q P^{1/2})^{1/2})]^{1/2}
                let p_sqrt = self.matrix_sqrt(p)?;
                let middle = &p_sqrt * q * &p_sqrt;
                let middle_sqrt = self.matrix_sqrt(&middle)?;
                let trace_term = p.trace() + q.trace() - <T as Scalar>::from_f64(2.0) * middle_sqrt.trace();
                Ok(<T as Float>::sqrt(trace_term))
            }
        }
    }

    /// Parallel transport along geodesic from P to Q.
    ///
    /// # Mathematical Formula
    ///
    /// Γ_{P→Q}(V) = E V E^T where E = (PQ)^{1/2} P^{-1/2}
    pub fn parallel_transport(
        &self,
        p: &DMatrix<T>,
        q: &DMatrix<T>,
        v: &DMatrix<T>,
    ) -> Result<DMatrix<T>> {
        match self.metric {
            SPDMetric::AffineInvariant => {
                // E = (PQ)^{1/2} P^{-1/2}
                let pq = p * q;
                let pq_sqrt = self.matrix_sqrt(&pq)?;
                let p_sqrt_inv = self.matrix_sqrt_inv(p)?;
                let e = &pq_sqrt * &p_sqrt_inv;
                
                // Transport: E V E^T
                Ok(&e * v * &e.transpose())
            }
            _ => {
                // For other metrics, use identity transport (approximation)
                Ok(v.clone())
            }
        }
    }
}

impl<T: Scalar> Manifold<T> for SPD<T> {
    type Point = DMatrix<T>;
    type TangentVector = DMatrix<T>;

    fn name(&self) -> &str {
        "SPD"
    }

    fn dimension(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.nrows() != self.n || point.ncols() != self.n {
            return false;
        }

        // Check symmetry
        let symmetry_error = (point - &point.transpose()).norm();
        if symmetry_error > tol {
            return false;
        }

        // Check positive definiteness
        let eigen = point.clone().symmetric_eigen();
        let min_eval = eigen.eigenvalues.iter()
            .fold(T::infinity(), |min, &val| <T as Float>::min(min, val));
        
        min_eval > self.min_eigenvalue && min_eval.is_finite()
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
        if vector.nrows() != self.n || vector.ncols() != self.n {
            return false;
        }

        // Tangent space consists of symmetric matrices
        let symmetry_error = (vector - &vector.transpose()).norm();
        symmetry_error < tol
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, _workspace: &mut Workspace<T>) {
        if point.nrows() != self.n || point.ncols() != self.n {
            *result = DMatrix::identity(self.n, self.n);
            *result *= self.min_eigenvalue + <T as Scalar>::from_f64(1.0);
            return;
        }

        // Ensure symmetry
        let symmetric = (point + &point.transpose()) * <T as Scalar>::from_f64(0.5);

        // Eigendecomposition
        let eigen = symmetric.clone().symmetric_eigen();
        let mut eigenvalues = eigen.eigenvalues.clone();

        // Clamp eigenvalues
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] <= self.min_eigenvalue || !eigenvalues[i].is_finite() {
                eigenvalues[i] = self.min_eigenvalue + <T as Scalar>::from_f64(1e-8);
            }
        }

        // Reconstruct: P = V Λ V^T
        let lambda_diag = DMatrix::from_diagonal(&eigenvalues);
        *result = &eigen.eigenvectors * lambda_diag * eigen.eigenvectors.transpose();
        
        // Final symmetry enforcement
        *result = (result.clone() + result.transpose()) * <T as Scalar>::from_f64(0.5);
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.nrows() != self.n || point.ncols() != self.n ||
           vector.nrows() != self.n || vector.ncols() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.n,
                point.nrows() * point.ncols()
            ));
        }

        // Verify point is on manifold
        self.check_point(point)?;

        // Project to symmetric matrices: (V + V^T)/2
        *result = (vector + &vector.transpose()) * <T as Scalar>::from_f64(0.5);
        Ok(())
    }

    fn inner_product(
        &self,
        point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        self.check_tangent(point, u)?;
        self.check_tangent(point, v)?;

        match self.metric {
            SPDMetric::AffineInvariant => {
                // <U,V>_P = tr(P^{-1} U P^{-1} V)
                let p_inv = point.clone().try_inverse()
                    .ok_or_else(|| ManifoldError::numerical_error("Point matrix not invertible".to_string()))?;
                Ok((&p_inv * u * &p_inv * v).trace())
            }
            SPDMetric::LogEuclidean => {
                // For Log-Euclidean, use Euclidean metric in log space
                // This is approximated by Frobenius inner product
                Ok((u.transpose() * v).trace())
            }
            SPDMetric::BuresWasserstein => {
                // <U,V>_P = 1/4 tr(U P^{-1} V + V P^{-1} U)
                let p_inv = point.clone().try_inverse()
                    .ok_or_else(|| ManifoldError::numerical_error("Point matrix not invertible".to_string()))?;
                let term1 = (u * &p_inv * v).trace();
                let term2 = (v * &p_inv * u).trace();
                Ok(<T as Scalar>::from_f64(0.25) * (term1 + term2))
            }
        }
    }

    fn retract(
        &self,
        point: &Self::Point,
        tangent: &Self::TangentVector,
        result: &mut Self::Point,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        match self.metric {
            SPDMetric::AffineInvariant => {
                // Use exact exponential map
                let exp_result = self.exp_map(point, tangent)?;
                result.copy_from(&exp_result);
                Ok(())
            }
            _ => {
                // For other metrics, use projection-based retraction
                let sum = point + tangent;
                self.project_point(&sum, result, workspace);
                Ok(())
            }
        }
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        self.check_point(point)?;
        self.check_point(other)?;

        match self.metric {
            SPDMetric::AffineInvariant => {
                // Use exact logarithmic map
                let log_result = self.log_map(point, other)?;
                result.copy_from(&log_result);
                Ok(())
            }
            _ => {
                // For other metrics, use approximation
                let diff = other - point;
                self.project_tangent(point, &diff, result, workspace)
            }
        }
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        self.check_point(point)?;

        let grad_result = match self.metric {
            SPDMetric::AffineInvariant => {
                // grad^{AI} f(P) = P ∇f(P) P
                point * euclidean_grad * point
            }
            SPDMetric::LogEuclidean => {
                // grad^{LE} f(P) = P ∇f(P) + ∇f(P) P
                point * euclidean_grad + euclidean_grad * point
            }
            SPDMetric::BuresWasserstein => {
                // grad^{BW} f(P) = ∇f(P)
                euclidean_grad.clone()
            }
        };

        // Ensure symmetry
        self.project_tangent(point, &grad_result, result, workspace)
    }

    fn parallel_transport(
        &self,
        from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let transported = self.parallel_transport(from, to, vector)?;
        result.copy_from(&transported);
        Ok(())
    }

    fn random_point(&self) -> Self::Point {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random matrix
        let mut a = DMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                a[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }

        // Make SPD: P = A^T A + εI
        let ata = a.transpose() * &a;
        let identity = DMatrix::<T>::identity(self.n, self.n);
        
        ata + identity * self.min_eigenvalue
    }

    fn random_tangent(
        &self,
        point: &Self::Point,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        self.check_point(point)?;
        
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random symmetric matrix
        for i in 0..self.n {
            for j in i..self.n {
                let val = <T as Scalar>::from_f64(normal.sample(&mut rng));
                result[(i, j)] = val;
                if i != j {
                    result[(j, i)] = val;
                }
            }
        }
        
        // Normalize
        let norm = result.norm();
        if norm > <T as Scalar>::from_f64(1e-16) {
            *result /= norm;
        }
        
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, _workspace: &mut Workspace<T>) -> Result<T> {
        self.distance(x, y)
    }

    fn has_exact_exp_log(&self) -> bool {
        matches!(self.metric, SPDMetric::AffineInvariant)
    }

    fn is_flat(&self) -> bool {
        false // SPD has negative curvature
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};
    use riemannopt_core::memory::workspace::Workspace;

    #[test]
    fn test_spd_creation() {
        // Valid SPD manifolds
        let spd22 = SPD::<f64>::new(2).unwrap();
        assert_eq!(spd22.matrix_dim(), 2);
        assert_eq!(spd22.dimension(), 3); // 2*(2+1)/2 = 3
        assert_eq!(spd22.metric_type(), SPDMetric::AffineInvariant);
        
        let spd33 = SPD::<f64>::new(3).unwrap();
        assert_eq!(spd33.dimension(), 6); // 3*(3+1)/2 = 6
        
        // With custom metric
        let spd_le = SPD::<f64>::with_metric(2, SPDMetric::LogEuclidean).unwrap();
        assert_eq!(spd_le.metric_type(), SPDMetric::LogEuclidean);
        
        // Invalid cases
        assert!(SPD::<f64>::new(0).is_err());
        assert!(SPD::<f64>::with_parameters(2, -1.0, 1e-10, SPDMetric::AffineInvariant).is_err());
    }

    #[test]
    fn test_point_validation() {
        let spd = SPD::<f64>::new(2).unwrap();
        
        // Identity is SPD
        let identity = DMatrix::<f64>::identity(2, 2);
        assert!(spd.check_point(&identity).is_ok());
        
        // Diagonal with positive entries
        let diag = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0]));
        assert!(spd.check_point(&diag).is_ok());
        
        // Non-symmetric
        let non_sym = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(spd.check_point(&non_sym).is_err());
        
        // Symmetric but not positive definite
        let not_pd = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(spd.check_point(&not_pd).is_err()); // Eigenvalues are -1 and 3
    }

    #[test]
    fn test_tangent_projection() {
        let spd = SPD::<f64>::new(3).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let p = spd.random_point();
        
        // Arbitrary matrix
        let v = DMatrix::from_fn(3, 3, |i, j| (i + j) as f64);
        
        let mut v_symmetric = DMatrix::zeros(3, 3);
        spd.project_tangent(&p, &v, &mut v_symmetric, &mut workspace).unwrap();
        
        // Check symmetry
        let symmetry_error = (&v_symmetric - &v_symmetric.transpose()).norm();
        assert!(symmetry_error < 1e-14);
        
        // Verify projection formula: (V + V^T)/2
        let expected = (&v + &v.transpose()) * 0.5;
        assert_relative_eq!(v_symmetric, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_matrix_operations() {
        let spd = SPD::<f64>::new(2).unwrap();
        
        // Test matrix square root
        let p = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 9.0]);
        let p_sqrt = spd.matrix_sqrt(&p).unwrap();
        
        // P^{1/2} should have eigenvalues 2 and 3
        assert_relative_eq!(p_sqrt[(0, 0)], 2.0, epsilon = 1e-14);
        assert_relative_eq!(p_sqrt[(1, 1)], 3.0, epsilon = 1e-14);
        
        // Verify P^{1/2} * P^{1/2} = P
        let p_reconstructed = &p_sqrt * &p_sqrt;
        assert_relative_eq!(p_reconstructed, p, epsilon = 1e-14);
        
        // Test matrix logarithm
        let log_p = spd.matrix_log(&p).unwrap();
        assert_relative_eq!(log_p[(0, 0)], (4.0_f64).ln(), epsilon = 1e-14);
        assert_relative_eq!(log_p[(1, 1)], (9.0_f64).ln(), epsilon = 1e-14);
        
        // Test exp(log(P)) = P
        let p_from_log = spd.matrix_exp(&log_p);
        assert_relative_eq!(p_from_log, p, epsilon = 1e-14);
    }

    #[test]
    fn test_exponential_logarithm() {
        let spd = SPD::<f64>::new(2).unwrap();
        
        let p = DMatrix::<f64>::identity(2, 2) * 2.0;
        let v = DMatrix::from_row_slice(2, 2, &[0.1, 0.2, 0.2, 0.3]);
        
        // Test exp followed by log
        let q = spd.exp_map(&p, &v).unwrap();
        assert!(spd.check_point(&q).is_ok());
        
        let v_recovered = spd.log_map(&p, &q).unwrap();
        assert_relative_eq!(v, v_recovered, epsilon = 1e-10);
        
        // Test identity: exp_P(0) = P
        let zero = DMatrix::zeros(2, 2);
        let p_recovered = spd.exp_map(&p, &zero).unwrap();
        assert_relative_eq!(p, p_recovered, epsilon = 1e-14);
    }

    #[test]
    fn test_inner_product_metrics() {
        let spd_ai = SPD::with_metric(2, SPDMetric::AffineInvariant).unwrap();
        let spd_le = SPD::<f64>::with_metric(2, SPDMetric::LogEuclidean).unwrap();
        let spd_bw = SPD::with_metric(2, SPDMetric::BuresWasserstein).unwrap();
        
        let p = DMatrix::<f64>::identity(2, 2) * 2.0;
        let u = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let v = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        
        // Different metrics give different inner products
        let inner_ai = spd_ai.inner_product(&p, &u, &v).unwrap();
        let inner_le = spd_le.inner_product(&p, &u, &v).unwrap();
        let inner_bw = spd_bw.inner_product(&p, &u, &v).unwrap();
        
        // For AI metric: <U,V>_P = tr(P^{-1} U P^{-1} V) = tr(0.5I * U * 0.5I * V) = 0.25 * tr(UV) = 0
        assert_relative_eq!(inner_ai, 0.0, epsilon = 1e-10);
        
        // For LE metric: <U,V> = tr(U^T V) = 0
        assert_relative_eq!(inner_le, 0.0, epsilon = 1e-10);
        
        // For BW metric: <U,V>_P = 0.25 * tr(U P^{-1} V + V P^{-1} U)
        // = 0.25 * tr(U * 0.5I * V + V * 0.5I * U) = 0.25 * 0.5 * 2 * tr(UV) = 0
        assert_relative_eq!(inner_bw, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_metrics() {
        let spd = SPD::<f64>::new(2).unwrap();
        
        let p = DMatrix::<f64>::identity(2, 2);
        let q = DMatrix::<f64>::identity(2, 2) * 4.0;
        
        // Affine-invariant distance
        let dist_ai = spd.distance(&p, &q).unwrap();
        // d(I, 4I) = ‖log(I^{-1/2} * 4I * I^{-1/2})‖_F = ‖log(4I)‖_F = √2 * ln(4)
        let expected_ai = (2.0_f64).sqrt() * (4.0_f64).ln();
        assert_relative_eq!(dist_ai, expected_ai, epsilon = 1e-10);
        
        // Distance properties
        let dist_pp = spd.distance(&p, &p).unwrap();
        assert_relative_eq!(dist_pp, 0.0, epsilon = 1e-14);
        
        // Symmetry
        let dist_pq = spd.distance(&p, &q).unwrap();
        let dist_qp = spd.distance(&q, &p).unwrap();
        assert_relative_eq!(dist_pq, dist_qp, epsilon = 1e-14);
    }

    #[test]
    fn test_random_point() {
        let spd = SPD::<f64>::new(3).unwrap();
        
        for _ in 0..10 {
            let p = spd.random_point();
            assert!(spd.check_point(&p).is_ok());
            
            // Check eigenvalues are positive
            let eigen = p.clone().symmetric_eigen();
            assert!(eigen.eigenvalues.iter().all(|&λ| λ > 0.0));
        }
    }

    #[test]
    fn test_euclidean_to_riemannian_gradient() {
        let spd = SPD::<f64>::new(2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let p = DMatrix::<f64>::identity(2, 2) * 2.0;
        let grad = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 2.0]);
        
        let mut rgrad = grad.clone();
        spd.euclidean_to_riemannian_gradient(&p, &grad, &mut rgrad, &mut workspace).unwrap();
        
        // Check it's symmetric
        assert!(spd.check_tangent(&p, &rgrad).is_ok());
        
        // For AI metric: grad^{AI} = P * grad * P = 2I * grad * 2I = 4 * grad
        let expected = &p * &grad * &p;
        let expected_symmetric = (&expected + &expected.transpose()) * 0.5;
        assert_relative_eq!(rgrad, expected_symmetric, epsilon = 1e-14);
    }

    #[test]
    fn test_parallel_transport() {
        let spd = SPD::<f64>::new(2).unwrap();
        
        let p = spd.random_point();
        let q = spd.random_point();
        let v = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 2.0]);
        
        let v_transported = spd.parallel_transport(&p, &q, &v).unwrap();
        
        // Check it's symmetric (in tangent space at q)
        let symmetry_error = (&v_transported - &v_transported.transpose()).norm();
        assert!(symmetry_error < 1e-14);
        
        // For AI metric, parallel transport preserves inner product magnitude
        // (approximately for our implementation)
        let norm_p = spd.inner_product(&p, &v, &v).unwrap().sqrt();
        let norm_q = spd.inner_product(&q, &v_transported, &v_transported).unwrap().sqrt();
        assert_relative_eq!(norm_p, norm_q, epsilon = 1e-8);
    }

    #[test]
    fn test_special_cases() {
        // S⁺⁺(1) is the positive real line
        let spd1 = SPD::<f64>::new(1).unwrap();
        assert_eq!(spd1.dimension(), 1); // 1*(1+1)/2 = 1
        
        let p = DMatrix::from_vec(1, 1, vec![2.0]);
        let v = DMatrix::from_vec(1, 1, vec![0.5]);
        
        // Exponential map: exp_p(v) = p * exp(v/p) = 2 * exp(0.5/2) = 2 * exp(0.25)
        let q = spd1.exp_map(&p, &v).unwrap();
        let expected = 2.0 * (0.25_f64).exp();
        assert_relative_eq!(q[(0, 0)], expected, epsilon = 1e-10);
    }
}