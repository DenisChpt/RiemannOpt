//! # Grassmann Manifold Gr(n,p)
//!
//! The Grassmann manifold Gr(n,p) is the space of all p-dimensional linear
//! subspaces of ℝⁿ. It provides a geometric framework for problems involving
//! subspace optimization, dimensionality reduction, and invariant subspace computation.
//!
//! ## Mathematical Definition
//!
//! The Grassmann manifold is formally defined as:
//! ```text
//! Gr(n,p) = {[Y] : Y ∈ ℝⁿˣᵖ, Y^T Y = I_p}
//! ```
//! where [Y] denotes the equivalence class of matrices with the same column space.
//!
//! Two matrices Y₁ and Y₂ represent the same point if Y₁ = Y₂Q for some Q ∈ O(p).
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space at [Y] consists of matrices orthogonal to the subspace:
//! ```text
//! T_{[Y]} Gr(n,p) = {Z ∈ ℝⁿˣᵖ : Y^T Z = 0}
//! ```
//! This is the horizontal space in the principal fiber bundle St(n,p) → Gr(n,p).
//!
//! ### Riemannian Metric
//! The canonical metric is inherited from the embedding in ℝⁿˣᵖ:
//! ```text
//! g_{[Y]}(Z₁, Z₂) = tr(Z₁^T Z₂)
//! ```
//!
//! ### Normal Space
//! The normal space (vertical space) consists of matrices of the form YS:
//! ```text
//! N_{[Y]} Gr(n,p) = {YS : S ∈ ℝᵖˣᵖ}
//! ```
//!
//! ### Projection Operators
//! - **Horizontal projection**: P_h(W) = W - Y(Y^T W) = (I - YY^T)W
//! - **Vertical projection**: P_v(W) = Y(Y^T W)
//!
//! ## Retractions and Exponential Map
//!
//! ### QR-based Retraction
//! The most efficient retraction uses QR decomposition:
//! ```text
//! R_Y(Z) = qf(Y + Z)
//! ```
//! where qf(·) extracts the Q factor from thin QR decomposition.
//!
//! ### SVD-based Retraction
//! A more stable retraction uses SVD:
//! ```text
//! (Y + Z) = UΣV^T, then R_Y(Z) = UV^T
//! ```
//!
//! ### Exponential Map
//! The exponential map involves matrix exponentials:
//! ```text
//! exp_{[Y]}(Z) = [YV cos(Σ) + U sin(Σ)]V^T
//! ```
//! where Z = UΣV^T is the compact SVD.
//!
//! ## Parallel Transport
//!
//! Parallel transport along geodesics can be computed using:
//! ```text
//! Γ_{[Y]→[Ỹ]}(Z) = (I - ỸỸ^T)ZU
//! ```
//! where U comes from the SVD of Ỹ^T Y.
//!
//! ## Geometric Invariants
//!
//! - **Dimension**: dim(Gr(n,p)) = p(n-p)
//! - **Sectional curvature**: 0 ≤ K ≤ 1
//! - **Geodesically complete**: Yes
//! - **Compact**: Yes
//! - **Simply connected**: Yes if p = 1 or p = n-1; No otherwise
//!
//! ## Principal Angles and Distance
//!
//! The distance between subspaces is measured via principal angles θᵢ:
//! ```text
//! d([Y₁], [Y₂]) = ‖θ‖₂
//! ```
//! where θᵢ = arccos(σᵢ) and σᵢ are singular values of Y₁^T Y₂.
//!
//! ## Optimization on Grassmann
//!
//! ### Riemannian Gradient
//! For f: Gr(n,p) → ℝ with Euclidean gradient ∇f(Y):
//! ```text
//! grad f([Y]) = (I - YY^T)∇f(Y)
//! ```
//!
//! ### Applications
//!
//! 1. **Principal Component Analysis**: Finding dominant eigenspaces
//! 2. **Subspace tracking**: Adaptive signal processing
//! 3. **Computer vision**: Multi-view geometry, face recognition
//! 4. **Model reduction**: Finding invariant subspaces
//! 5. **Machine learning**: Metric learning, domain adaptation
//! 6. **Quantum computing**: Optimization over pure state subspaces
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Numerical stability** through careful orthogonalization
//! - **Efficiency** via optimized BLAS operations
//! - **Robustness** to rank-deficient matrices
//! - **Invariance** under orthogonal transformations of representatives
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Grassmann;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DMatrix;
//!
//! // Create Grassmann manifold Gr(5,2)
//! let grassmann = Grassmann::<f64>::new(5, 2)?;
//!
//! // Random point (2D subspace of ℝ⁵)
//! let y = grassmann.random_point();
//! 
//! // Verify orthonormality
//! let yty = y.transpose() * &y;
//! assert!((yty - DMatrix::<f64>::identity(2, 2)).norm() < 1e-14);
//!
//! // Tangent vector (orthogonal to subspace)
//! let z = DMatrix::from_fn(5, 2, |i, j| 0.1 * (i as f64 - j as f64));
//! let mut z_horizontal = z.clone();
//! let mut workspace = Workspace::<f64>::new();
//! grassmann.project_tangent(&y, &z, &mut z_horizontal, &mut workspace)?;
//!
//! // Verify horizontality: Y^T Z = 0
//! let ytz = y.transpose() * &z_horizontal;
//! assert!(ytz.norm() < 1e-14);
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::DMatrix;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    core::matrix_manifold::MatrixManifold,
    memory::workspace::Workspace,
    types::Scalar,
};
use std::fmt::{self, Debug};

/// The Grassmann manifold Gr(n,p) of p-dimensional subspaces in ℝⁿ.
///
/// This structure represents the space of all p-dimensional linear subspaces
/// of n-dimensional Euclidean space, equipped with the canonical Riemannian
/// metric inherited from the Stiefel manifold.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `1 ≤ p ≤ n-1`: Dimension constraints (p=0 or p=n give trivial cases)
/// - Points are represented by n×p matrices with orthonormal columns
/// - The manifold structure is invariant under the O(p) action on the right
#[derive(Clone)]
pub struct Grassmann<T = f64> {
    /// Ambient dimension n
    n: usize,
    /// Subspace dimension p
    p: usize,
    /// Numerical tolerance for validations
    tolerance: T,
}

impl<T: Scalar> Debug for Grassmann<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Grassmann Gr({}, {})", self.n, self.p)
    }
}

impl<T: Scalar> Grassmann<T> {
    /// Creates a new Grassmann manifold Gr(n,p).
    ///
    /// # Arguments
    ///
    /// * `n` - Ambient dimension (must satisfy p < n)
    /// * `p` - Subspace dimension (must satisfy 0 < p < n)
    ///
    /// # Returns
    ///
    /// A Grassmann manifold with dimension p(n-p).
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if:
    /// - `p = 0` (empty subspace)
    /// - `p ≥ n` (subspace dimension exceeds ambient dimension)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::Grassmann;
    /// // Create Gr(5,2) - 2D subspaces in ℝ⁵
    /// let gr52 = Grassmann::<f64>::new(5, 2)?;
    /// 
    /// // Gr(n,1) is the projective space ℝP^{n-1}
    /// let projective = Grassmann::<f64>::new(4, 1)?;
    /// 
    /// // Gr(n,n-1) is also isomorphic to ℝP^{n-1}
    /// let dual_projective = Grassmann::<f64>::new(4, 3)?;
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if p == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Grassmann manifold requires p > 0",
            ));
        }
        if p >= n {
            return Err(ManifoldError::invalid_parameter(format!(
                "Grassmann manifold Gr(n,p) requires p < n, got n={}, p={}",
                n, p
            )));
        }
        Ok(Self {
            n,
            p,
            tolerance: <T as Scalar>::from_f64(1e-12),
        })
    }

    /// Creates a Grassmann manifold with custom numerical tolerance.
    ///
    /// # Arguments
    ///
    /// * `n` - Ambient dimension
    /// * `p` - Subspace dimension
    /// * `tolerance` - Numerical tolerance for validations
    pub fn with_tolerance(n: usize, p: usize, tolerance: T) -> Result<Self> {
        if p == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Grassmann manifold requires p > 0",
            ));
        }
        if p >= n {
            return Err(ManifoldError::invalid_parameter(format!(
                "Grassmann manifold Gr(n,p) requires p < n, got n={}, p={}",
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

    /// Returns the ambient dimension n.
    #[inline]
    pub fn ambient_dim(&self) -> usize {
        self.n
    }

    /// Returns the subspace dimension p.
    #[inline]
    pub fn subspace_dim(&self) -> usize {
        self.p
    }

    /// Validates that a matrix represents a point on Grassmann.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that Y^T Y = I_p within numerical tolerance.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix dimensions don't match (n,p)
    /// - `NotOnManifold`: If ‖Y^T Y - I_p‖ > tolerance
    pub fn check_point(&self, y: &DMatrix<T>) -> Result<()> {
        if y.nrows() != self.n || y.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.p,
                y.nrows() * y.ncols()
            ));
        }

        // Check orthonormality: Y^T Y = I
        let yty = y.transpose() * y;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        let constraint_error = (&yty - &identity).norm();
        
        if constraint_error > self.tolerance {
            return Err(ManifoldError::invalid_point(format!(
                "Orthonormality violated: ‖Y^T Y - I‖ = {} (tolerance: {})",
                constraint_error, self.tolerance
            )));
        }

        Ok(())
    }

    /// Validates that a matrix lies in the horizontal space at Y.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that Y^T Z = 0 (horizontality condition).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions don't match
    /// - `NotOnManifold`: If Y is not on Grassmann
    /// - `NotInTangentSpace`: If ‖Y^T Z‖ > tolerance
    pub fn check_tangent(&self, y: &DMatrix<T>, z: &DMatrix<T>) -> Result<()> {
        self.check_point(y)?;

        if z.nrows() != self.n || z.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                self.n * self.p,
                z.nrows() * z.ncols()
            ));
        }

        // Check horizontality: Y^T Z = 0
        let ytz = y.transpose() * z;
        let horizontal_error = ytz.norm();
        
        if horizontal_error > self.tolerance {
            return Err(ManifoldError::invalid_tangent(format!(
                "Horizontality violated: ‖Y^T Z‖ = {} (tolerance: {})",
                horizontal_error, self.tolerance
            )));
        }

        Ok(())
    }

    /// Performs QR-based retraction.
    ///
    /// # Mathematical Formula
    ///
    /// R_Y(Z) = qf(Y + Z) where qf extracts the Q factor from thin QR.
    ///
    /// # Arguments
    ///
    /// * `y` - Point on Grassmann manifold
    /// * `z` - Tangent vector at y (horizontal)
    ///
    /// # Returns
    ///
    /// The retracted point R_Y(Z) on the manifold.
    pub fn qr_retraction(&self, y: &DMatrix<T>, z: &DMatrix<T>) -> Result<DMatrix<T>> {
        // Compute Y + Z
        let y_plus_z = y + z;
        
        // QR decomposition
        let qr = y_plus_z.qr();
        let mut q = qr.q();
        
        // Extract first p columns
        if q.ncols() > self.p {
            q = q.columns(0, self.p).clone_owned();
        }
        
        // Fix signs for continuity
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

    /// Performs SVD-based retraction (more stable).
    ///
    /// # Mathematical Formula
    ///
    /// For Y + Z = UΣV^T, R_Y(Z) = UV^T.
    ///
    /// # Arguments
    ///
    /// * `y` - Point on Grassmann manifold
    /// * `z` - Tangent vector at y
    ///
    /// # Returns
    ///
    /// The retracted point using SVD.
    pub fn svd_retraction(&self, y: &DMatrix<T>, z: &DMatrix<T>) -> Result<DMatrix<T>> {
        let y_plus_z = y + z;
        
        // Compute SVD
        let svd = y_plus_z.svd(true, true);
        
        if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
            // Take first p columns of U and rows of V^T
            let u_truncated = if u.ncols() > self.p {
                u.columns(0, self.p).clone_owned()
            } else {
                u
            };
            
            let vt_truncated = if vt.nrows() > self.p {
                vt.rows(0, self.p).clone_owned()
            } else {
                vt
            };
            
            Ok(u_truncated * vt_truncated)
        } else {
            Err(ManifoldError::numerical_error(
                "SVD computation failed in retraction",
            ))
        }
    }

    /// Computes geodesic distance between two subspaces.
    ///
    /// # Mathematical Formula
    ///
    /// d([Y₁], [Y₂]) = ‖θ‖₂ where θᵢ = arccos(σᵢ(Y₁^T Y₂)).
    ///
    /// # Arguments
    ///
    /// * `y1` - First point on Grassmann
    /// * `y2` - Second point on Grassmann
    ///
    /// # Returns
    ///
    /// The geodesic distance between the subspaces.
    pub fn geodesic_distance(&self, y1: &DMatrix<T>, y2: &DMatrix<T>) -> Result<T> {
        self.check_point(y1)?;
        self.check_point(y2)?;

        // Compute Y₁^T Y₂
        let y1ty2 = y1.transpose() * y2;
        
        // SVD to get principal angles
        let svd = y1ty2.clone().svd(true, true);
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
    /// # Mathematical Formula
    ///
    /// For geodesic from [Y₁] to [Y₂], transport Z using:
    /// τ(Z) = (I - Y₂Y₂^T)ZU where Y₂^T Y₁ = UΣV^T.
    pub fn parallel_transport(
        &self,
        y1: &DMatrix<T>,
        y2: &DMatrix<T>,
        z: &DMatrix<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<DMatrix<T>> {
        self.check_tangent(y1, z)?;
        self.check_point(y2)?;

        // Compute Y₂^T Y₁ and its SVD
        let y2ty1 = y2.transpose() * y1;
        let svd = y2ty1.clone().svd(true, true);
        
        if let Some(u) = svd.u {
            // Transport: (I - Y₂Y₂^T)ZU
            let zu = z * &u;
            let y2_zu = y2 * &(y2.transpose() * &zu);
            Ok(&zu - &y2_zu)
        } else {
            // Fallback to simple projection
            let mut result = z.clone();
            self.project_tangent(y2, z, &mut result, workspace)?;
            Ok(result)
        }
    }
}

impl<T: Scalar> Manifold<T> for Grassmann<T> {
    type Point = DMatrix<T>;
    type TangentVector = DMatrix<T>;

    fn name(&self) -> &str {
        "Grassmann"
    }

    fn dimension(&self) -> usize {
        self.p * (self.n - self.p)
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.nrows() != self.n || point.ncols() != self.p {
            return false;
        }
        
        // Check Y^T Y = I_p
        let yty = point.transpose() * point;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        (&yty - &identity).norm() < tol
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
        
        // Horizontal space: Y^T Z = 0
        let ytz = point.transpose() * vector;
        ytz.norm() < tol
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, _workspace: &mut Workspace<T>) {
        if point.nrows() != self.n || point.ncols() != self.p {
            *result = DMatrix::zeros(self.n, self.p);
            return;
        }
        
        // Use QR decomposition for projection
        let qr = point.clone().qr();
        let mut q = qr.q();
        
        // Extract first p columns
        if q.ncols() > self.p {
            q = q.columns(0, self.p).clone_owned();
        }
        
        // Fix signs for continuity
        let r = qr.r();
        for j in 0..self.p.min(r.ncols()) {
            if r[(j, j)] < T::zero() {
                for i in 0..self.n {
                    q[(i, j)] = -q[(i, j)];
                }
            }
        }
        
        *result = q;
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
        let yty = point.transpose() * point;
        let identity = DMatrix::<T>::identity(self.p, self.p);
        if (&yty - &identity).norm() > self.tolerance {
            return Err(ManifoldError::invalid_point(
                "Point must be on Grassmann for tangent projection",
            ));
        }

        // Horizontal projection: Z - Y(Y^T Z)
        let ytz = point.transpose() * vector;
        *result = vector - point * ytz;
        
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

        // Compute log map approximation using projection
        // For close points: log_Y(Ỹ) ≈ P_h(Ỹ - Y)
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
        // Riemannian gradient is the horizontal projection of Euclidean gradient
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
        
        // QR decomposition to get orthonormal basis
        let qr = a.qr();
        let mut q = qr.q();
        
        // Extract first p columns
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
        
        // Project to horizontal space
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
        false // QR retraction is not the exponential map
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
        // For Grassmann manifold, tangent vectors are in the horizontal space
        // Scaling preserves the horizontal space property
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
        
        // The sum should already be in the horizontal space if v1 and v2 are,
        // but we project for numerical stability
        // Create a temporary clone to avoid borrowing issues
        let temp = result.clone();
        self.project_tangent(point, &temp, result, workspace)?;
        
        Ok(())
    }
}

impl<T: Scalar> MatrixManifold<T> for Grassmann<T> {
    fn matrix_dims(&self) -> (usize, usize) {
        (self.n, self.p)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use riemannopt_core::memory::workspace::Workspace;

    #[test]
    fn test_grassmann_creation() {
        // Valid Grassmann manifolds
        let gr32 = Grassmann::<f64>::new(3, 2).unwrap();
        assert_eq!(gr32.ambient_dim(), 3);
        assert_eq!(gr32.subspace_dim(), 2);
        assert_eq!(gr32.dimension(), 2); // 2*(3-2) = 2
        
        let gr52 = Grassmann::<f64>::new(5, 2).unwrap();
        assert_eq!(gr52.dimension(), 6); // 2*(5-2) = 6
        
        let gr53 = Grassmann::<f64>::new(5, 3).unwrap();
        assert_eq!(gr53.dimension(), 6); // 3*(5-3) = 6
        
        // Invalid cases
        assert!(Grassmann::<f64>::new(2, 3).is_err()); // p > n
        assert!(Grassmann::<f64>::new(3, 0).is_err()); // p = 0
        assert!(Grassmann::<f64>::new(3, 3).is_err()); // p = n
    }

    #[test]
    fn test_point_validation() {
        let grassmann = Grassmann::<f64>::new(4, 2).unwrap();
        
        // Create orthonormal matrix
        let y = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ]);
        
        assert!(grassmann.check_point(&y).is_ok());
        
        // Non-orthonormal matrix
        let y_bad = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.5,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ]);
        
        assert!(grassmann.check_point(&y_bad).is_err());
    }

    #[test]
    fn test_tangent_projection() {
        let grassmann = Grassmann::<f64>::new(4, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        // Canonical basis vectors
        let y = DMatrix::from_column_slice(4, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ]);
        
        // Arbitrary matrix
        let z = DMatrix::from_column_slice(4, 2, &[
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
            0.7, 0.8,
        ]);
        
        let mut z_horizontal = DMatrix::zeros(4, 2);
        grassmann.project_tangent(&y, &z, &mut z_horizontal, &mut workspace).unwrap();
        
        // Check horizontality: Y^T Z_h = 0
        let ytz = y.transpose() * &z_horizontal;
        assert!(ytz.norm() < 1e-14);
        
        // Verify projection formula: Z_h = Z - Y(Y^T Z)
        let ytz_orig = y.transpose() * &z;
        let expected = &z - &y * ytz_orig;
        assert_relative_eq!(z_horizontal, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_qr_retraction() {
        let grassmann = Grassmann::<f64>::new(5, 2).unwrap();
        
        let y = grassmann.random_point();
        assert!(grassmann.check_point(&y).is_ok());
        
        // Small tangent vector
        let z = DMatrix::from_fn(5, 2, |i, j| 0.01 * ((i + j) as f64));
        let mut z_horizontal = z.clone();
        let mut workspace = Workspace::<f64>::new();
        grassmann.project_tangent(&y, &z, &mut z_horizontal, &mut workspace).unwrap();
        
        // Retract
        let y_new = grassmann.qr_retraction(&y, &z_horizontal).unwrap();
        
        // Check result is on manifold
        assert!(grassmann.check_point(&y_new).is_ok());
        
        // Check first-order approximation: R_Y(0) = Y
        let zero = DMatrix::zeros(5, 2);
        let y_recovered = grassmann.qr_retraction(&y, &zero).unwrap();
        assert_relative_eq!(y, y_recovered, epsilon = 1e-14);
    }

    #[test]
    fn test_inner_product() {
        let grassmann = Grassmann::<f64>::new(4, 3).unwrap();
        
        let y = grassmann.random_point();
        let mut workspace = Workspace::<f64>::new();
        
        // Generate two tangent vectors
        let u = DMatrix::from_fn(4, 3, |i, j| (i as f64) * 0.1 + (j as f64) * 0.2);
        let v = DMatrix::from_fn(4, 3, |i, j| (i as f64) * 0.3 - (j as f64) * 0.1);
        
        let mut u_horizontal = u.clone();
        let mut v_horizontal = v.clone();
        grassmann.project_tangent(&y, &u, &mut u_horizontal, &mut workspace).unwrap();
        grassmann.project_tangent(&y, &v, &mut v_horizontal, &mut workspace).unwrap();
        
        // Compute inner product
        let inner = grassmann.inner_product(&y, &u_horizontal, &v_horizontal).unwrap();
        
        // Should equal trace(U^T V)
        let expected = (u_horizontal.transpose() * &v_horizontal).trace();
        assert_relative_eq!(inner, expected, epsilon = 1e-14);
        
        // Check symmetry
        let inner_rev = grassmann.inner_product(&y, &v_horizontal, &u_horizontal).unwrap();
        assert_relative_eq!(inner, inner_rev, epsilon = 1e-14);
    }

    #[test]
    fn test_geodesic_distance() {
        let grassmann = Grassmann::<f64>::new(4, 2).unwrap();
        
        let y1 = grassmann.random_point();
        let y2 = grassmann.random_point();
        
        // Distance properties
        let d12 = grassmann.geodesic_distance(&y1, &y2).unwrap();
        let d21 = grassmann.geodesic_distance(&y2, &y1).unwrap();
        
        // Symmetry
        assert_relative_eq!(d12, d21, epsilon = 1e-14);
        
        // Non-negativity
        assert!(d12 >= 0.0);
        
        // Identity of indiscernibles
        let d11 = grassmann.geodesic_distance(&y1, &y1).unwrap();
        assert_relative_eq!(d11, 0.0, epsilon = 1e-14);
        
        // Maximum distance on Grassmann is π/2 * sqrt(p)
        let max_dist = std::f64::consts::FRAC_PI_2 * (2.0_f64).sqrt();
        assert!(d12 <= max_dist * 1.01); // Small tolerance for numerical errors
    }

    #[test]
    fn test_random_point() {
        let grassmann = Grassmann::<f64>::new(6, 3).unwrap();
        
        for _ in 0..10 {
            let y = grassmann.random_point();
            assert!(grassmann.check_point(&y).is_ok());
            
            // Check orthonormality precisely
            let yty = y.transpose() * &y;
            let identity = DMatrix::<f64>::identity(3, 3);
            assert_relative_eq!(yty, identity, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_euclidean_to_riemannian_gradient() {
        let grassmann = Grassmann::<f64>::new(5, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let y = grassmann.random_point();
        
        // Euclidean gradient
        let grad = DMatrix::from_fn(5, 2, |i, j| (i + j) as f64);
        
        let mut rgrad = grad.clone();
        grassmann.euclidean_to_riemannian_gradient(&y, &grad, &mut rgrad, &mut workspace).unwrap();
        
        // Check it's in horizontal space
        assert!(grassmann.check_tangent(&y, &rgrad).is_ok());
        
        // Check projection formula
        let yty = y.transpose() * &grad;
        let expected = &grad - &y * yty;
        assert_relative_eq!(rgrad, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_parallel_transport() {
        let grassmann = Grassmann::<f64>::new(4, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let y1 = grassmann.random_point();
        let y2 = grassmann.random_point();
        
        // Create tangent vector at y1
        let mut z = DMatrix::zeros(4, 2);
        grassmann.random_tangent(&y1, &mut z, &mut workspace).unwrap();
        
        // Transport to y2
        let z_transported = grassmann.parallel_transport(&y1, &y2, &z, &mut workspace).unwrap();
        
        // Check it's in tangent space at y2
        assert!(grassmann.check_tangent(&y2, &z_transported).is_ok());
        
        // For Grassmann, parallel transport should preserve norm (approximately)
        let norm1 = z.norm();
        let norm2 = z_transported.norm();
        assert_relative_eq!(norm1, norm2, epsilon = 1e-8);
    }

    #[test]
    fn test_special_cases() {
        // Gr(n,1) is projective space ℝP^{n-1}
        let gr31 = Grassmann::<f64>::new(3, 1).unwrap();
        assert_eq!(gr31.dimension(), 2); // Same as ℝP²
        
        // Gr(n,n-1) is also projective space (via duality)
        let gr43 = Grassmann::<f64>::new(4, 3).unwrap();
        assert_eq!(gr43.dimension(), 3); // Same as ℝP³
        
        // Gr(4,2) has maximal dimension for fixed n+p
        let gr42 = Grassmann::<f64>::new(4, 2).unwrap();
        assert_eq!(gr42.dimension(), 4); // 2*(4-2) = 4
    }

    #[test]
    fn test_inverse_retraction() {
        let grassmann = Grassmann::<f64>::new(5, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let y = grassmann.random_point();
        
        // Small tangent vector
        let mut z = DMatrix::zeros(5, 2);
        grassmann.random_tangent(&y, &mut z, &mut workspace).unwrap();
        z *= 0.01; // Scale down
        
        // Retract then inverse retract
        let mut y_new = DMatrix::zeros(5, 2);
        grassmann.retract(&y, &z, &mut y_new, &mut workspace).unwrap();
        
        let mut z_recovered = DMatrix::zeros(5, 2);
        grassmann.inverse_retract(&y, &y_new, &mut z_recovered, &mut workspace).unwrap();
        
        // Should approximately recover the tangent vector
        assert_relative_eq!(z, z_recovered, epsilon = 1e-6);
    }
}