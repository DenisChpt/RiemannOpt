//! # Stiefel Manifold St(n,p)
//!
//! The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p} is the set of all
//! n×p matrices with orthonormal columns. It generalizes both the sphere (p=1)
//! and the orthogonal group (n=p), making it fundamental for problems involving
//! orthogonality constraints.
//!
//! ## Mathematical Definition
//!
//! The Stiefel manifold is formally defined as:
//! ```text
//! St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p}
//! ```
//! where I_p is the p×p identity matrix.
//!
//! It forms a compact embedded submanifold of ℝⁿˣᵖ with dimension np - p(p+1)/2.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space at X ∈ St(n,p) consists of matrices satisfying a symmetry constraint:
//! ```text
//! T_X St(n,p) = {Z ∈ ℝⁿˣᵖ : X^T Z + Z^T X = 0}
//!             = {Z ∈ ℝⁿˣᵖ : X^T Z is skew-symmetric}
//! ```
//!
//! ### Riemannian Metric
//! The canonical metric is inherited from the Euclidean space:
//! ```text
//! g_X(Z₁, Z₂) = tr(Z₁^T Z₂) = ⟨Z₁, Z₂⟩_F
//! ```
//! where ⟨·,·⟩_F denotes the Frobenius inner product.
//!
//! ### Normal Space
//! The normal space at X consists of matrices of the form XS where S is symmetric:
//! ```text
//! N_X St(n,p) = {XS : S ∈ ℝᵖˣᵖ, S^T = S}
//! ```
//!
//! ### Projection Operators
//! - **Orthogonal projection**: P_St(A) = UV^T from SVD A = UΣV^T
//! - **Tangent projection**: P_X(Z) = Z - X sym(X^T Z)
//!   where sym(A) = (A + A^T)/2
//!
//! ## Retractions and Vector Transport
//!
//! ### QR-based Retraction
//! The QR decomposition provides an efficient retraction:
//! ```text
//! R_X(Z) = qf(X + Z)
//! ```
//! where qf(·) extracts the Q factor from QR decomposition.
//!
//! ### Polar Retraction
//! Using the polar decomposition (X + Z) = UP:
//! ```text
//! R_X(Z) = U
//! ```
//!
//! ### Cayley Retraction
//! For Z ∈ T_X St(n,p):
//! ```text
//! R_X(Z) = (X + Z)(I + ½Z^T Z)^{-1/2}
//! ```
//!
//! ### Vector Transport
//! Differentiated retraction provides vector transport:
//! ```text
//! τ_Z(W) = P_{R_X(Z)}(W)
//! ```
//!
//! ## Geometric Invariants
//!
//! - **Dimension**: dim(St(n,p)) = np - p(p+1)/2
//! - **Sectional curvature**: 0 ≤ K ≤ 1
//! - **Geodesically complete**: Yes
//! - **Simply connected**: Yes if p < n, No if p = n > 1
//!
//! ## Optimization on Stiefel
//!
//! ### Riemannian Gradient
//! For f: St(n,p) → ℝ with Euclidean gradient ∇f(X):
//! ```text
//! grad f(X) = ∇f(X) - X sym(X^T ∇f(X))
//! ```
//!
//! ### Applications
//!
//! 1. **Eigenvalue problems**: Finding p dominant eigenvectors
//! 2. **Procrustes problems**: Optimal alignment of point sets
//! 3. **Dimensionality reduction**: PCA, CCA, LDA with orthogonality
//! 4. **Computer vision**: Essential matrix estimation
//! 5. **Signal processing**: Subspace tracking, blind source separation
//! 6. **Machine learning**: Orthogonal neural network layers
//!
//! ## Numerical Considerations
//!
//! This implementation provides:
//! - **Numerically stable** QR and polar decompositions
//! - **Efficient operations** using BLAS when available
//! - **Orthogonality preservation** up to machine precision
//! - **Robust retractions** handling edge cases
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Stiefel;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DMatrix;
//!
//! // Create Stiefel manifold St(5,2)
//! let stiefel = Stiefel::<f64>::new(5, 2)?;
//!
//! // Random point on manifold
//! let x = stiefel.random_point();
//! 
//! // Verify orthonormality: X^T X = I
//! let xtx = x.transpose() * &x;
//! assert!((xtx - DMatrix::<f64>::identity(2, 2)).norm() < 1e-14);
//!
//! // Project gradient to tangent space
//! let grad = DMatrix::from_fn(5, 2, |i, j| (i + j) as f64);
//! let mut rgrad = grad.clone();
//! let mut workspace = Workspace::<f64>::new();
//! stiefel.euclidean_to_riemannian_gradient(&x, &grad, &mut rgrad, &mut workspace)?;
//!
//! // Verify tangent space constraint
//! let constraint = x.transpose() * &rgrad;
//! assert!((constraint + constraint.transpose()).norm() < 1e-14);
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

/// The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p}.
///
/// This structure represents the manifold of n×p matrices with orthonormal columns,
/// equipped with the canonical Riemannian metric inherited from ℝⁿˣᵖ.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `n ≥ p ≥ 1`: Dimensions must satisfy this constraint
/// - All points X satisfy X^T X = I_p up to numerical tolerance
/// - All tangent vectors Z at X satisfy X^T Z + Z^T X = 0
#[derive(Clone)]
pub struct Stiefel<T = f64> {
    /// Number of rows (n)
    n: usize,
    /// Number of columns (p)
    p: usize,
    /// Numerical tolerance for constraint validation
    tolerance: T,
}

impl<T: Scalar> Debug for Stiefel<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stiefel St({}, {})", self.n, self.p)
    }
}

impl<T: Scalar> Stiefel<T> {
    /// Creates a new Stiefel manifold St(n,p).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows (must be ≥ p)
    /// * `p` - Number of columns (must be ≥ 1)
    ///
    /// # Returns
    ///
    /// A Stiefel manifold with dimension np - p(p+1)/2.
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if:
    /// - `p = 0`
    /// - `n < p`
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::Stiefel;
    /// // Create St(5,2) - 5×2 matrices with orthonormal columns
    /// let stiefel = Stiefel::<f64>::new(5, 2)?;
    /// 
    /// // Special case: St(n,1) is the sphere S^{n-1}
    /// let sphere = Stiefel::<f64>::new(3, 1)?;
    /// 
    /// // Special case: St(n,n) is the orthogonal group O(n)
    /// let orthogonal = Stiefel::<f64>::new(3, 3)?;
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if p == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Stiefel manifold requires p ≥ 1",
            ));
        }
        if n < p {
            return Err(ManifoldError::invalid_parameter(format!(
                "Stiefel manifold St(n,p) requires n ≥ p, got n={}, p={}",
                n, p
            )));
        }
        Ok(Self {
            n,
            p,
            tolerance: <T as Scalar>::from_f64(1e-12),
        })
    }

    /// Creates a Stiefel manifold with custom numerical tolerance.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows
    /// * `p` - Number of columns
    /// * `tolerance` - Numerical tolerance for constraint validation
    pub fn with_tolerance(n: usize, p: usize, tolerance: T) -> Result<Self> {
        if p == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Stiefel manifold requires p ≥ 1",
            ));
        }
        if n < p {
            return Err(ManifoldError::invalid_parameter(format!(
                "Stiefel manifold St(n,p) requires n ≥ p, got n={}, p={}",
                n, p
            )));
        }
        if tolerance <= T::zero() || tolerance >= T::one() {
            return Err(ManifoldError::invalid_parameter(
                "Tolerance must be in (0, 1)",
            ));
        }
        Ok(Self { n, p, tolerance })
    }

    /// Returns the number of rows n.
    #[inline]
    pub fn rows(&self) -> usize {
        self.n
    }

    /// Returns the number of columns p.
    #[inline]
    pub fn cols(&self) -> usize {
        self.p
    }

    /// Validates that a matrix lies on the Stiefel manifold.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that X^T X = I_p within numerical tolerance.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix dimensions don't match (n,p)
    /// - `NotOnManifold`: If ‖X^T X - I_p‖ > tolerance
    pub fn check_point(&self, x: &DMatrix<T>) -> Result<()> {
        if x.nrows() != self.n || x.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.p,
                x.nrows() * x.ncols()
            ));
        }

        // Check orthonormality: X^T X = I
        let xtx = x.transpose() * x;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        let constraint_error = (&xtx - &identity).norm();
        
        if constraint_error > self.tolerance {
            return Err(ManifoldError::invalid_point(format!(
                "Orthonormality violated: ‖X^T X - I‖ = {} (tolerance: {})",
                constraint_error, self.tolerance
            )));
        }

        Ok(())
    }

    /// Validates that a matrix lies in the tangent space at X.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that X^T Z + Z^T X = 0 (skew-symmetry constraint).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions don't match
    /// - `NotOnManifold`: If X is not on Stiefel
    /// - `NotInTangentSpace`: If skew-symmetry constraint violated
    pub fn check_tangent(&self, x: &DMatrix<T>, z: &DMatrix<T>) -> Result<()> {
        self.check_point(x)?;

        if z.nrows() != self.n || z.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.p,
                z.nrows() * z.ncols()
            ));
        }

        // Check skew-symmetry: X^T Z + Z^T X = 0
        let xtz = x.transpose() * z;
        let skew_error = (&xtz + &xtz.transpose()).norm();
        
        if skew_error > self.tolerance {
            return Err(ManifoldError::invalid_tangent(format!(
                "Skew-symmetry violated: ‖X^T Z + Z^T X‖ = {} (tolerance: {})",
                skew_error, self.tolerance
            )));
        }

        Ok(())
    }

    /// Computes the symmetric part of a matrix: sym(A) = (A + A^T)/2.
    #[inline]
    fn symmetrize(a: &DMatrix<T>) -> DMatrix<T> {
        (a + &a.transpose()) * <T as Scalar>::from_f64(0.5)
    }

    /// Performs QR-based retraction.
    ///
    /// # Mathematical Formula
    ///
    /// R_X(Z) = qf(X + Z) where qf extracts the Q factor from QR decomposition.
    ///
    /// # Arguments
    ///
    /// * `x` - Point on Stiefel manifold
    /// * `z` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The retracted point R_X(Z) on the manifold.
    pub fn qr_retraction(&self, x: &DMatrix<T>, z: &DMatrix<T>) -> Result<DMatrix<T>> {
        // Compute X + Z
        let x_plus_z = x + z;
        
        // QR decomposition
        let qr = x_plus_z.qr();
        let mut q = qr.q();
        
        // Extract first p columns if needed
        if q.ncols() > self.p {
            q = q.columns(0, self.p).clone_owned();
        }
        
        // Fix signs to ensure continuity (R has positive diagonal)
        let r = qr.r();
        for j in 0..self.p.min(r.ncols()) {
            if r[(j, j)] < T::zero() {
                for i in 0..self.n {
                    q[(i, j)] = -q[(i, j)];
                }
            }
        }
        
        Ok(q)
    }

    /// Performs polar retraction.
    ///
    /// # Mathematical Formula
    ///
    /// For X + Z = UP (polar decomposition), R_X(Z) = U.
    ///
    /// # Arguments
    ///
    /// * `x` - Point on Stiefel manifold
    /// * `z` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The retracted point using polar decomposition.
    pub fn polar_retraction(&self, x: &DMatrix<T>, z: &DMatrix<T>) -> Result<DMatrix<T>> {
        let x_plus_z = x + z;
        
        // Compute (X+Z)^T(X+Z)
        let gram = x_plus_z.transpose() * &x_plus_z;
        
        // Eigendecomposition for matrix square root
        let eigen = gram.symmetric_eigen();
        let d = &eigen.eigenvalues;
        let v = &eigen.eigenvectors;
        
        // Compute (gram)^{-1/2}
        let mut d_sqrt_inv = DVector::zeros(self.p);
        for i in 0..self.p {
            if d[i] > self.tolerance {
                d_sqrt_inv[i] = T::one() / <T as Float>::sqrt(d[i]);
            } else {
                return Err(ManifoldError::numerical_error(
                    "Polar retraction failed: singular Gram matrix",
                ));
            }
        }
        
        let gram_sqrt_inv = v * DMatrix::from_diagonal(&d_sqrt_inv) * v.transpose();
        
        Ok(&x_plus_z * gram_sqrt_inv)
    }

    /// Computes geodesic distance between two points (Frobenius-based).
    ///
    /// # Mathematical Formula
    ///
    /// For the canonical metric, the geodesic distance involves principal angles:
    /// d(X, Y) = ‖θ‖₂ where θ contains principal angles between span(X) and span(Y).
    ///
    /// # Note
    ///
    /// This implementation uses a Frobenius-based approximation for efficiency.
    /// For exact geodesic distance, principal angle computation is required.
    pub fn geodesic_distance(&self, x: &DMatrix<T>, y: &DMatrix<T>) -> Result<T> {
        self.check_point(x)?;
        self.check_point(y)?;

        // Compute X^T Y
        let xty = x.transpose() * y;
        
        // SVD to get principal angles
        let svd = xty.svd(true, true);
        let sigma = &svd.singular_values;
        
        // Principal angles: θᵢ = arccos(σᵢ)
        let mut dist_sq = T::zero();
        for i in 0..self.p {
            // Clamp singular values to [-1, 1]
            let sigma_clamped = <T as Float>::max(
                <T as Float>::min(sigma[i], T::one()),
                -T::one()
            );
            let theta = <T as Float>::acos(sigma_clamped);
            dist_sq = dist_sq + theta * theta;
        }
        
        Ok(<T as Float>::sqrt(dist_sq))
    }

    /// Parallel transports a tangent vector along a geodesic.
    ///
    /// This uses the differentiated retraction for vector transport.
    pub fn parallel_transport(
        &self,
        x: &DMatrix<T>,
        y: &DMatrix<T>,
        v: &DMatrix<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<DMatrix<T>> {
        self.check_tangent(x, v)?;
        self.check_point(y)?;

        // For Stiefel, we use projection-based transport
        // τ_{X→Y}(V) = P_Y(V) where P_Y is tangent projection at Y
        
        // Compute Y^T V
        let ytv = y.transpose() * v;
        
        // Project: V - Y * sym(Y^T V)
        let sym_ytv = Self::symmetrize(&ytv);
        Ok(v - y * sym_ytv)
    }
}

impl<T: Scalar> Manifold<T> for Stiefel<T> {
    type Point = DMatrix<T>;
    type TangentVector = DMatrix<T>;

    fn name(&self) -> &str {
        "Stiefel"
    }

    fn dimension(&self) -> usize {
        self.n * self.p - self.p * (self.p + 1) / 2
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.nrows() != self.n || point.ncols() != self.p {
            return false;
        }
        
        // Check X^T X = I
        let xtx = point.transpose() * point;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        (&xtx - &identity).norm() < tol
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
        if vector.nrows() != self.n || vector.ncols() != self.p {
            return false;
        }
        
        // Check X^T Z + Z^T X = 0
        let xtz = point.transpose() * vector;
        (&xtz + &xtz.transpose()).norm() < tol
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, _workspace: &mut Workspace<T>) {
        if point.nrows() != self.n || point.ncols() != self.p {
            *result = DMatrix::zeros(self.n, self.p);
            return;
        }
        
        // Use SVD for projection: A = UΣV^T → UV^T
        let svd = point.clone().svd(true, true);
        if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
            // Take first p columns of U if needed
            let u_truncated = if u.ncols() > self.p {
                u.columns(0, self.p).clone_owned()
            } else {
                u
            };
            
            *result = u_truncated * vt;
        } else {
            // Fallback to QR
            let qr = point.clone().qr();
            let mut q = qr.q();
            if q.ncols() > self.p {
                q = q.columns(0, self.p).clone_owned();
            }
            *result = q;
        }
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.nrows() != self.n || point.ncols() != self.p ||
           vector.nrows() != self.n || vector.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.p,
                point.nrows() * point.ncols()
            ));
        }

        // Check that point is on manifold
        let xtx = point.transpose() * point;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        if (&xtx - &identity).norm() > self.tolerance {
            return Err(ManifoldError::invalid_point(
                "Point must be on Stiefel for tangent projection",
            ));
        }

        // Project: Z - X * sym(X^T Z)
        let xtz = point.transpose() * vector;
        let sym_xtz = Self::symmetrize(&xtz);
        *result = vector - point * sym_xtz;
        
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
        
        // Canonical metric: tr(U^T V)
        let mut inner = T::zero();
        for i in 0..self.n {
            for j in 0..self.p {
                inner = inner + u[(i, j)] * v[(i, j)];
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
        // Use QR retraction by default
        let retracted = self.qr_retraction(point, tangent)?;
        result.copy_from(&retracted);
        Ok(())
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

        // Compute log map approximation
        // For close points: log_X(Y) ≈ P_X(Y - X)
        let diff = other - point;
        self.project_tangent(point, &diff, result, workspace)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Riemannian gradient is the tangent projection of Euclidean gradient
        self.project_tangent(point, euclidean_grad, result, workspace)
    }

    fn parallel_transport(
        &self,
        from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let transported = self.parallel_transport(from, to, vector, workspace)?;
        result.copy_from(&transported);
        Ok(())
    }

    fn random_point(&self) -> Self::Point {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random Gaussian matrix
        let mut a = DMatrix::zeros(self.n, self.p);
        for i in 0..self.n {
            for j in 0..self.p {
                a[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // QR decomposition to get orthonormal columns
        let qr = a.qr();
        let mut q = qr.q();
        
        // Extract first p columns if needed
        if q.ncols() > self.p {
            q = q.columns(0, self.p).clone_owned();
        }
        
        q
    }

    fn random_tangent(
        &self,
        point: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        self.check_point(point)?;
        
        // Generate random matrix
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        let mut z = DMatrix::zeros(self.n, self.p);
        for i in 0..self.n {
            for j in 0..self.p {
                z[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // Project to tangent space
        self.project_tangent(point, &z, result, workspace)?;
        
        // Normalize
        let norm = result.norm();
        if norm > <T as Scalar>::from_f64(1e-16) {
            *result /= norm;
        }
        
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, _workspace: &mut Workspace<T>) -> Result<T> {
        self.geodesic_distance(x, y)
    }

    fn has_exact_exp_log(&self) -> bool {
        false // Stiefel doesn't have closed-form exp/log in general
    }

    fn is_flat(&self) -> bool {
        false
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For Stiefel manifold, tangent vectors satisfy X^T Z + Z^T X = 0
        // Scaling preserves this skew-symmetry constraint
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
        
        // The sum should already satisfy the tangent space constraint if v1 and v2 do,
        // but we project for numerical stability
        // Create a temporary clone to avoid borrowing issues
        let temp = result.clone();
        self.project_tangent(point, &temp, result, workspace)?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use riemannopt_core::memory::workspace::Workspace;

    #[test]
    fn test_stiefel_creation() {
        // Valid Stiefel manifolds
        let st32 = Stiefel::<f64>::new(3, 2).unwrap();
        assert_eq!(st32.rows(), 3);
        assert_eq!(st32.cols(), 2);
        assert_eq!(st32.dimension(), 3 * 2 - 2 * 3 / 2); // 6 - 3 = 3
        
        let st55 = Stiefel::<f64>::new(5, 5).unwrap();
        assert_eq!(st55.dimension(), 5 * 5 - 5 * 6 / 2); // 25 - 15 = 10
        
        // Invalid cases
        assert!(Stiefel::<f64>::new(2, 3).is_err()); // n < p
        assert!(Stiefel::<f64>::new(5, 0).is_err()); // p = 0
    }

    #[test]
    fn test_point_validation() {
        let stiefel = Stiefel::<f64>::new(4, 2).unwrap();
        
        // Create orthonormal matrix
        let a = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ]);
        
        assert!(stiefel.check_point(&a).is_ok());
        
        // Non-orthonormal matrix
        let b = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.5,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ]);
        
        assert!(stiefel.check_point(&b).is_err());
    }

    #[test]
    fn test_tangent_projection() {
        let stiefel = Stiefel::<f64>::new(3, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        // Point on manifold
        let x = DMatrix::from_row_slice(3, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        
        // Arbitrary matrix
        let z = DMatrix::from_row_slice(3, 2, &[
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
        ]);
        
        let mut z_tangent = DMatrix::zeros(3, 2);
        stiefel.project_tangent(&x, &z, &mut z_tangent, &mut workspace).unwrap();
        
        // Check tangent space constraint
        let xtz = x.transpose() * &z_tangent;
        let skew_error = (&xtz + &xtz.transpose()).norm();
        assert!(skew_error < 1e-14);
    }

    #[test]
    fn test_qr_retraction() {
        let stiefel = Stiefel::<f64>::new(4, 2).unwrap();
        
        let x = stiefel.random_point();
        assert!(stiefel.check_point(&x).is_ok());
        
        // Small tangent vector
        let z = DMatrix::from_fn(4, 2, |i, j| 0.1 * ((i + j) as f64));
        let mut z_tangent = z.clone();
        let mut workspace = Workspace::<f64>::new();
        stiefel.project_tangent(&x, &z, &mut z_tangent, &mut workspace).unwrap();
        
        // Retract
        let y = stiefel.qr_retraction(&x, &z_tangent).unwrap();
        
        // Check result is on manifold
        assert!(stiefel.check_point(&y).is_ok());
        
        // Check first-order approximation: R_X(0) = X
        let zero = DMatrix::zeros(4, 2);
        let x_recovered = stiefel.qr_retraction(&x, &zero).unwrap();
        assert_relative_eq!(x, x_recovered, epsilon = 1e-14);
    }

    #[test]
    fn test_inner_product() {
        let stiefel = Stiefel::<f64>::new(3, 2).unwrap();
        
        let x = stiefel.random_point();
        let mut workspace = Workspace::<f64>::new();
        
        // Generate two tangent vectors
        let u = DMatrix::from_fn(3, 2, |i, j| (i as f64) * 0.1 + (j as f64) * 0.2);
        let v = DMatrix::from_fn(3, 2, |i, j| (i as f64) * 0.3 - (j as f64) * 0.1);
        
        let mut u_tangent = u.clone();
        let mut v_tangent = v.clone();
        stiefel.project_tangent(&x, &u, &mut u_tangent, &mut workspace).unwrap();
        stiefel.project_tangent(&x, &v, &mut v_tangent, &mut workspace).unwrap();
        
        // Compute inner product
        let inner = stiefel.inner_product(&x, &u_tangent, &v_tangent).unwrap();
        
        // Should equal trace(U^T V)
        let expected = (u_tangent.transpose() * &v_tangent).trace();
        assert_relative_eq!(inner, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_random_point() {
        let stiefel = Stiefel::<f64>::new(10, 3).unwrap();
        
        for _ in 0..10 {
            let x = stiefel.random_point();
            assert!(stiefel.check_point(&x).is_ok());
            
            // Check orthonormality precisely
            let xtx = x.transpose() * &x;
            let identity = DMatrix::<f64>::identity(3, 3);
            assert_relative_eq!(xtx, identity, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_euclidean_to_riemannian_gradient() {
        let stiefel = Stiefel::<f64>::new(4, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let x = stiefel.random_point();
        
        // Euclidean gradient
        let grad = DMatrix::from_fn(4, 2, |i, j| (i + j) as f64);
        
        let mut rgrad = grad.clone();
        stiefel.euclidean_to_riemannian_gradient(&x, &grad, &mut rgrad, &mut workspace).unwrap();
        
        // Check it's in tangent space
        assert!(stiefel.check_tangent(&x, &rgrad).is_ok());
        
        // Check projection formula
        let xtg = x.transpose() * &grad;
        let sym_xtg = (&xtg + &xtg.transpose()) * 0.5;
        let expected = &grad - &x * sym_xtg;
        assert_relative_eq!(rgrad, expected, epsilon = 1e-14);
    }

    #[test] 
    fn test_special_cases() {
        // St(n,1) should behave like sphere
        let st31 = Stiefel::<f64>::new(3, 1).unwrap();
        assert_eq!(st31.dimension(), 2); // Same as S^2
        
        // St(n,n) is orthogonal group
        let st33 = Stiefel::<f64>::new(3, 3).unwrap();
        assert_eq!(st33.dimension(), 3); // Dimension of SO(3)
    }
}