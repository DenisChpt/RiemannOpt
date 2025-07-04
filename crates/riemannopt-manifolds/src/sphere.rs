//! # Unit Sphere Manifold S^{n-1}
//!
//! The unit sphere S^{n-1} = {x ∈ ℝⁿ : ‖x‖₂ = 1} is the set of all unit vectors
//! in n-dimensional Euclidean space. It serves as a fundamental example in Riemannian
//! geometry and appears ubiquitously in optimization problems involving directional data.
//!
//! ## Mathematical Definition
//!
//! The unit sphere is formally defined as:
//! ```text
//! S^{n-1} = {x ∈ ℝⁿ : x^T x = 1}
//! ```
//!
//! It forms a compact, connected (n-1)-dimensional Riemannian submanifold of ℝⁿ.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! At any point x ∈ S^{n-1}, the tangent space is the orthogonal complement of x:
//! ```text
//! T_x S^{n-1} = {v ∈ ℝⁿ : x^T v = 0} = x^⊥
//! ```
//! This is an (n-1)-dimensional linear subspace of ℝⁿ.
//!
//! ### Riemannian Metric
//! The sphere inherits the Euclidean metric from the ambient space:
//! ```text
//! g_x(u, v) = u^T v = ⟨u, v⟩_ℝⁿ
//! ```
//! for all u, v ∈ T_x S^{n-1}.
//!
//! ### Projection Operators
//! - **Projection onto manifold**: P_S(y) = y/‖y‖₂ for y ≠ 0
//! - **Projection onto tangent space**: P_x(v) = (I - xx^T)v = v - ⟨v,x⟩x
//!
//! ### Geodesics and Exponential Map
//! Geodesics on the sphere are great circles. The exponential map has the closed form:
//! ```text
//! exp_x(v) = cos(‖v‖₂)x + sin(‖v‖₂)(v/‖v‖₂)
//! ```
//! for v ∈ T_x S^{n-1}, v ≠ 0. For v = 0, exp_x(0) = x.
//!
//! ### Logarithmic Map
//! For x, y ∈ S^{n-1} with x ≠ -y:
//! ```text
//! log_x(y) = θ/sin(θ) · (y - cos(θ)x)
//! ```
//! where θ = arccos(x^T y) ∈ [0, π] is the geodesic distance.
//!
//! ### Parallel Transport
//! Parallel transport from x to y along the geodesic is given by:
//! ```text
//! Γ_{x→y}(v) = v - (x^T v + y^T v)/(1 + x^T y) · (x + y)
//! ```
//! for x ≠ -y and v ∈ T_x S^{n-1}.
//!
//! ## Geometric Invariants
//!
//! - **Sectional curvature**: K ≡ 1 (constant positive curvature)
//! - **Scalar curvature**: R = (n-1)(n-2) 
//! - **Ricci curvature**: Ric = (n-2)g
//! - **Diameter**: diam(S^{n-1}) = π
//! - **Volume**: vol(S^{n-1}) = 2π^{n/2}/Γ(n/2)
//! - **Injectivity radius**: inj(S^{n-1}) = π
//!
//! ## Optimization on the Sphere
//!
//! ### Riemannian Gradient
//! For a smooth function f: S^{n-1} → ℝ with Euclidean gradient ∇f(x):
//! ```text
//! grad f(x) = P_x(∇f(x)) = ∇f(x) - (x^T ∇f(x))x
//! ```
//!
//! ### Riemannian Hessian
//! The Riemannian Hessian involves both the Euclidean Hessian H_f(x) and curvature:
//! ```text
//! Hess f(x)[v] = P_x(H_f(x)v) - (x^T ∇f(x))v
//! ```
//!
//! ## Applications
//!
//! 1. **Principal Component Analysis**: max_{x ∈ S^{n-1}} x^T Σ x
//! 2. **Sparse PCA**: max_{x ∈ S^{n-1}} x^T Σ x - λ‖x‖₁
//! 3. **Spherical k-means**: Clustering directional data
//! 4. **Independent Component Analysis**: Finding orthogonal projections
//! 5. **Quantum state optimization**: Pure states as complex unit vectors
//! 6. **Computer vision**: 3D rotations via quaternions (S³)
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Numerical stability** near antipodal points (x ≈ -y)
//! - **Machine precision** constraint satisfaction ‖x‖₂ = 1
//! - **Efficient operations** leveraging closed-form expressions
//! - **Robust edge cases** handling for zero vectors and singularities
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Sphere;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DVector;
//!
//! // Create unit sphere in ℝ³
//! let sphere = Sphere::<f64>::new(3)?;
//! 
//! // Random point on S²
//! let x = sphere.random_point();
//! assert!((x.norm() - 1.0).abs() < 1e-14);
//! 
//! // Tangent vector
//! let v = DVector::from_vec(vec![0.0, 1.0, 0.0]);
//! let mut v_tangent = v.clone();
//! let mut workspace = Workspace::<f64>::new();
//! sphere.project_tangent(&x, &v, &mut v_tangent, &mut workspace)?;
//! assert!(x.dot(&v_tangent).abs() < 1e-14);
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    memory::workspace::Workspace,
    types::{DVector, Scalar},
};
use std::fmt::{self, Debug};

/// The unit sphere manifold S^{n-1} = {x ∈ ℝⁿ : ‖x‖₂ = 1}.
///
/// This structure represents the (n-1)-dimensional unit sphere embedded in ℝⁿ,
/// equipped with the induced Riemannian metric from the ambient Euclidean space.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `ambient_dim ≥ 2`: The sphere must be at least S¹ (circle)
/// - All points x satisfy ‖x‖₂ = 1 up to numerical tolerance
/// - All tangent vectors v at x satisfy x^T v = 0 up to numerical tolerance
#[derive(Clone)]
pub struct Sphere<T = f64> {
    /// Ambient dimension n for the sphere S^{n-1}
    ambient_dim: usize,
    /// Numerical tolerance for constraint validation
    tolerance: T,
}

impl<T: Scalar> Debug for Sphere<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sphere S^{} in R^{}", self.ambient_dim - 1, self.ambient_dim)
    }
}

impl<T: Scalar> Sphere<T> {
    /// Creates a new sphere manifold S^{n-1} embedded in ℝⁿ.
    ///
    /// # Arguments
    ///
    /// * `ambient_dim` - Dimension n of the ambient space ℝⁿ (must be ≥ 2)
    ///
    /// # Returns
    ///
    /// A sphere manifold with dimension (n-1).
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if `ambient_dim < 2`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::Sphere;
    /// // Create the unit circle S¹ in ℝ²
    /// let circle = Sphere::<f64>::new(2)?;
    /// 
    /// // Create the unit sphere S² in ℝ³
    /// let sphere = Sphere::<f64>::new(3)?;
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(ambient_dim: usize) -> Result<Self> {
        if ambient_dim < 2 {
            return Err(ManifoldError::invalid_parameter(format!(
                "Sphere requires ambient dimension ≥ 2 (got {})",
                ambient_dim
            )));
        }
        Ok(Self {
            ambient_dim,
            tolerance: <T as Scalar>::from_f64(1e-12),
        })
    }

    /// Creates a sphere with custom numerical tolerance.
    ///
    /// # Arguments
    ///
    /// * `ambient_dim` - Dimension of the ambient space
    /// * `tolerance` - Numerical tolerance for constraint validation
    ///
    /// # Returns
    ///
    /// A sphere manifold with specified tolerance.
    pub fn with_tolerance(ambient_dim: usize, tolerance: T) -> Result<Self> {
        if ambient_dim < 2 {
            return Err(ManifoldError::invalid_parameter(
                "Sphere requires ambient dimension ≥ 2",
            ));
        }
        if tolerance <= T::zero() || tolerance >= T::one() {
            return Err(ManifoldError::invalid_parameter(
                "Tolerance must be in (0, 1)",
            ));
        }
        Ok(Self {
            ambient_dim,
            tolerance,
        })
    }

    /// Returns the ambient dimension n.
    #[inline]
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dim
    }

    /// Returns the intrinsic manifold dimension (n-1).
    #[inline]
    pub fn manifold_dimension(&self) -> usize {
        self.ambient_dim - 1
    }

    /// Validates that a point lies on the sphere.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that ‖x‖₂ = 1 within numerical tolerance.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If x.len() ≠ ambient_dim
    /// - `NotOnManifold`: If |‖x‖₂ - 1| > tolerance
    pub fn check_point(&self, x: &DVector<T>) -> Result<()> {
        if x.len() != self.ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.ambient_dim,
                x.len()
            ));
        }

        let norm_squared = x.norm_squared();
        if <T as Float>::abs(norm_squared - T::one()) > self.tolerance {
            return Err(ManifoldError::invalid_point(format!(
                "Point not on sphere: ‖x‖² = {} (tolerance: {})",
                norm_squared, self.tolerance
            )));
        }

        Ok(())
    }

    /// Validates that a vector lies in the tangent space at x.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that x^T v = 0 within numerical tolerance.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions don't match
    /// - `NotOnManifold`: If x is not on the sphere
    /// - `NotInTangentSpace`: If |x^T v| > tolerance
    pub fn check_tangent(&self, x: &DVector<T>, v: &DVector<T>) -> Result<()> {
        self.check_point(x)?;

        if v.len() != self.ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.ambient_dim,
                v.len()
            ));
        }

        let inner_product = x.dot(v);
        if <T as Float>::abs(inner_product) > self.tolerance {
            return Err(ManifoldError::invalid_tangent(format!(
                "Vector not in tangent space: x^T v = {} (tolerance: {})",
                inner_product, self.tolerance
            )));
        }

        Ok(())
    }

    /// Computes the exponential map exp_x(v).
    ///
    /// # Mathematical Formula
    ///
    /// For v ∈ T_x S^{n-1}:
    /// - If ‖v‖ = 0: exp_x(v) = x
    /// - Otherwise: exp_x(v) = cos(‖v‖)x + sin(‖v‖)(v/‖v‖)
    ///
    /// # Arguments
    ///
    /// * `x` - Point on the sphere
    /// * `v` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The point exp_x(v) on the sphere.
    pub fn exp_map(&self, x: &DVector<T>, v: &DVector<T>) -> Result<DVector<T>> {
        self.check_tangent(x, v)?;

        let v_norm = v.norm();
        if v_norm < self.tolerance {
            return Ok(x.clone());
        }

        let cos_norm = <T as Float>::cos(v_norm);
        let sin_norm = <T as Float>::sin(v_norm);
        
        Ok(x * cos_norm + v * (sin_norm / v_norm))
    }

    /// Computes the logarithmic map log_x(y).
    ///
    /// # Mathematical Formula
    ///
    /// For x, y ∈ S^{n-1} with x ≠ -y:
    /// - If x = y: log_x(y) = 0
    /// - Otherwise: log_x(y) = θ/sin(θ) · (y - cos(θ)x)
    ///   where θ = arccos(x^T y)
    ///
    /// # Arguments
    ///
    /// * `x` - Point on the sphere
    /// * `y` - Another point on the sphere
    ///
    /// # Returns
    ///
    /// The tangent vector log_x(y) ∈ T_x S^{n-1}.
    ///
    /// # Errors
    ///
    /// Returns error if x and y are antipodal (x ≈ -y).
    pub fn log_map(&self, x: &DVector<T>, y: &DVector<T>) -> Result<DVector<T>> {
        self.check_point(x)?;
        self.check_point(y)?;

        let inner = x.dot(y);
        
        // Check if points are the same
        if <T as Float>::abs(inner - T::one()) < self.tolerance {
            return Ok(DVector::zeros(self.ambient_dim));
        }
        
        // Check if points are antipodal
        if <T as Float>::abs(inner + T::one()) < self.tolerance {
            return Err(ManifoldError::numerical_error(
                "Cannot compute logarithm between antipodal points",
            ));
        }

        // Ensure inner product is in valid range for arccos
        let inner_clamped = <T as Float>::max(
            <T as Float>::min(inner, T::one()),
            -T::one()
        );
        
        let theta = <T as Float>::acos(inner_clamped);
        let sin_theta = <T as Float>::sin(theta);
        
        if sin_theta < self.tolerance {
            // Points are very close, use first-order approximation
            Ok(y - x)
        } else {
            let scale = theta / sin_theta;
            Ok((y - x * inner_clamped) * scale)
        }
    }

    /// Computes the geodesic distance between two points.
    ///
    /// # Mathematical Formula
    ///
    /// d(x, y) = arccos(x^T y)
    ///
    /// # Arguments
    ///
    /// * `x` - First point on the sphere
    /// * `y` - Second point on the sphere
    ///
    /// # Returns
    ///
    /// The geodesic distance d(x, y) ∈ [0, π].
    pub fn geodesic_distance(&self, x: &DVector<T>, y: &DVector<T>) -> Result<T> {
        self.check_point(x)?;
        self.check_point(y)?;

        let inner = x.dot(y);
        let inner_clamped = <T as Float>::max(
            <T as Float>::min(inner, T::one()),
            -T::one()
        );
        
        Ok(<T as Float>::acos(inner_clamped))
    }

    /// Parallel transports a tangent vector along a geodesic.
    ///
    /// # Mathematical Formula
    ///
    /// For x ≠ -y and v ∈ T_x S^{n-1}:
    /// Γ_{x→y}(v) = v - [(x^T v + y^T v)/(1 + x^T y)] · (x + y)
    ///
    /// # Arguments
    ///
    /// * `x` - Starting point
    /// * `y` - Ending point
    /// * `v` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The parallel transported vector at y.
    pub fn parallel_transport(
        &self,
        x: &DVector<T>,
        y: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<DVector<T>> {
        self.check_tangent(x, v)?;
        self.check_point(y)?;

        let inner_xy = x.dot(y);
        
        // Check if points are the same
        if <T as Float>::abs(inner_xy - T::one()) < self.tolerance {
            return Ok(v.clone());
        }
        
        // Check if points are antipodal
        if <T as Float>::abs(inner_xy + T::one()) < self.tolerance {
            return Err(ManifoldError::numerical_error(
                "Cannot parallel transport between antipodal points",
            ));
        }

        let factor = (x.dot(v) + y.dot(v)) / (T::one() + inner_xy);
        Ok(v - &(x + y) * factor)
    }
}

impl<T: Scalar> Manifold<T> for Sphere<T> {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;

    fn name(&self) -> &str {
        "Sphere"
    }

    fn dimension(&self) -> usize {
        self.ambient_dim - 1
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        point.len() == self.ambient_dim && {
            let norm_sq = point.norm_squared();
            <T as Float>::abs(norm_sq - T::one()) < tol
        }
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
        vector.len() == self.ambient_dim && <T as Float>::abs(point.dot(vector)) < tol
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, _workspace: &mut Workspace<T>) {
        if point.len() != self.ambient_dim {
            result.resize_vertically_mut(self.ambient_dim, T::zero());
        }
        
        let norm = point.norm();
        if norm < <T as Scalar>::from_f64(1e-16) {
            // Handle near-zero vector by projecting to first coordinate
            result.fill(T::zero());
            result[0] = T::one();
        } else {
            // Project by normalizing
            result.copy_from(point);
            *result /= norm;
        }
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.ambient_dim || vector.len() != self.ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.ambient_dim,
                point.len().max(vector.len())
            ));
        }

        // Check that point is on manifold
        let norm_sq = point.norm_squared();
        if <T as Float>::abs(norm_sq - T::one()) > self.tolerance {
            return Err(ManifoldError::invalid_point(
                "Point must be on sphere for tangent projection",
            ));
        }

        // Project: v - <v,x>x
        let inner = point.dot(vector);
        result.copy_from(vector);
        result.axpy(-inner, point, T::one());
        
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
        
        // The sphere inherits the Euclidean inner product
        Ok(u.dot(v))
    }

    fn retract(
        &self,
        point: &Self::Point,
        tangent: &Self::TangentVector,
        result: &mut Self::Point,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Use exponential map as the retraction
        let exp_result = self.exp_map(point, tangent)?;
        result.copy_from(&exp_result);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Use logarithmic map as the inverse retraction
        let log_result = self.log_map(point, other)?;
        result.copy_from(&log_result);
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Riemannian gradient is the projection of Euclidean gradient
        self.project_tangent(point, euclidean_grad, result, workspace)
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
        
        // Generate random Gaussian vector
        let mut x = DVector::zeros(self.ambient_dim);
        for i in 0..self.ambient_dim {
            x[i] = <T as Scalar>::from_f64(normal.sample(&mut rng));
        }
        
        // Normalize to get uniform distribution on sphere
        let norm = x.norm();
        if norm < <T as Scalar>::from_f64(1e-16) {
            // Extremely rare case: regenerate
            x[0] = T::one();
        } else {
            x /= norm;
        }
        
        x
    }

    fn random_tangent(
        &self,
        point: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        self.check_point(point)?;
        
        // Generate random vector in ambient space
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        let mut v = DVector::zeros(self.ambient_dim);
        for i in 0..self.ambient_dim {
            v[i] = <T as Scalar>::from_f64(normal.sample(&mut rng));
        }
        
        // Project to tangent space
        self.project_tangent(point, &v, result, workspace)?;
        
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
        true
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
        // For the sphere, tangent vectors are just vectors in the ambient space
        // orthogonal to the point. Scaling preserves this orthogonality.
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
        // Add the vectors
        result.copy_from(v1);
        *result += v2;
        
        // The sum should already be in the tangent space if v1 and v2 are,
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
    use nalgebra::DVector;
    use riemannopt_core::memory::workspace::Workspace;

    #[test]
    fn test_sphere_creation() {
        // Valid spheres
        let sphere2 = Sphere::<f64>::new(2).unwrap();
        assert_eq!(sphere2.ambient_dimension(), 2);
        assert_eq!(sphere2.dimension(), 1);
        
        let sphere3 = Sphere::<f64>::new(3).unwrap();
        assert_eq!(sphere3.ambient_dimension(), 3);
        assert_eq!(sphere3.dimension(), 2);
        
        // Invalid sphere
        assert!(Sphere::<f64>::new(1).is_err());
    }

    #[test]
    fn test_point_projection() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        // Project non-zero vector
        let point = DVector::from_vec(vec![3.0, 4.0, 0.0]);
        let mut projected = DVector::zeros(3);
        sphere.project_point(&point, &mut projected, &mut workspace);
        
        assert_relative_eq!(projected.norm(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(projected[0], 0.6, epsilon = 1e-14);
        assert_relative_eq!(projected[1], 0.8, epsilon = 1e-14);
        assert_relative_eq!(projected[2], 0.0, epsilon = 1e-14);
        
        // Project already normalized vector
        let unit_point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        sphere.project_point(&unit_point, &mut projected, &mut workspace);
        assert_relative_eq!(projected, unit_point, epsilon = 1e-14);
    }

    #[test]
    fn test_tangent_projection() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.5, 1.0, 2.0]);
        let mut v_tangent = DVector::zeros(3);
        
        sphere.project_tangent(&x, &v, &mut v_tangent, &mut workspace).unwrap();
        
        // Check orthogonality
        assert_relative_eq!(x.dot(&v_tangent), 0.0, epsilon = 1e-14);
        
        // Check projection formula
        let expected = &v - &x * x.dot(&v);
        assert_relative_eq!(v_tangent, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_exponential_logarithm() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        
        let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 0.5, 0.0]);
        
        // Test exp followed by log
        let y = sphere.exp_map(&x, &v).unwrap();
        assert_relative_eq!(y.norm(), 1.0, epsilon = 1e-14);
        
        let v_recovered = sphere.log_map(&x, &y).unwrap();
        assert_relative_eq!(v, v_recovered, epsilon = 1e-14);
        
        // Test zero tangent vector
        let x_recovered = sphere.exp_map(&x, &DVector::zeros(3)).unwrap();
        assert_relative_eq!(x, x_recovered, epsilon = 1e-14);
    }

    #[test]
    fn test_geodesic_distance() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        
        let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let y = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        
        let dist = sphere.geodesic_distance(&x, &y).unwrap();
        assert_relative_eq!(dist, std::f64::consts::FRAC_PI_2, epsilon = 1e-14);
        
        // Distance to same point
        let dist_same = sphere.geodesic_distance(&x, &x).unwrap();
        assert_relative_eq!(dist_same, 0.0, epsilon = 1e-14);
        
        // Distance to antipodal point
        let z = DVector::from_vec(vec![-1.0, 0.0, 0.0]);
        let dist_antipodal = sphere.geodesic_distance(&x, &z).unwrap();
        assert_relative_eq!(dist_antipodal, std::f64::consts::PI, epsilon = 1e-14);
    }

    #[test]
    fn test_parallel_transport() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        
        let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let y = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        
        let v_transported = sphere.parallel_transport(&x, &y, &v).unwrap();
        
        // Check it's in tangent space at y
        assert_relative_eq!(y.dot(&v_transported), 0.0, epsilon = 1e-14);
        
        // Check norm preservation (for orthogonal transport)
        assert_relative_eq!(v_transported.norm(), v.norm(), epsilon = 1e-14);
    }

    #[test]
    fn test_random_point() {
        let sphere = Sphere::<f64>::new(10).unwrap();
        
        for _ in 0..100 {
            let x = sphere.random_point();
            assert_relative_eq!(x.norm(), 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_manifold_properties() {
        let sphere = Sphere::<f64>::new(4).unwrap();
        
        assert_eq!(sphere.name(), "Sphere");
        assert_eq!(sphere.dimension(), 3);
        assert!(sphere.has_exact_exp_log());
        assert!(!sphere.is_flat());
    }
}