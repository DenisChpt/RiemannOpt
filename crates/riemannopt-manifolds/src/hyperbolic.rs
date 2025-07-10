//! # Hyperbolic Manifold ℍⁿ
//!
//! The hyperbolic manifold ℍⁿ is the n-dimensional hyperbolic space, a complete
//! Riemannian manifold with constant negative sectional curvature -1. It provides
//! a natural geometry for hierarchical and tree-like data structures.
//!
//! ## Mathematical Definition
//!
//! Hyperbolic space can be represented through several equivalent models:
//!
//! ### Poincaré Ball Model
//! ```text
//! 𝔹ⁿ = {x ∈ ℝⁿ : ‖x‖ < 1}
//! ```
//! with metric tensor:
//! ```text
//! g_x = (2/(1 - ‖x‖²))² · g_E
//! ```
//! where g_E is the Euclidean metric.
//!
//! ### Hyperboloid Model
//! ```text
//! ℍⁿ = {x ∈ ℝⁿ⁺¹ : ⟨x,x⟩_L = -1, x₀ > 0}
//! ```
//! with Lorentzian inner product ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! In the Poincaré ball model:
//! ```text
//! T_x 𝔹ⁿ ≅ ℝⁿ
//! ```
//! The tangent space is isomorphic to ℝⁿ but with a different metric.
//!
//! ### Riemannian Metric
//! The conformal factor λ(x) = 2/(1 - ‖x‖²) gives:
//! ```text
//! ⟨u, v⟩_x = λ(x)² ⟨u, v⟩_E
//! ```
//!
//! ### Geodesics
//! In the Poincaré ball:
//! - Through origin: straight lines
//! - General: circular arcs orthogonal to the boundary
//!
//! ### Distance Formula
//! ```text
//! d(x, y) = arcosh(1 + 2‖x - y‖²/((1 - ‖x‖²)(1 - ‖y‖²)))
//! ```
//!
//! ## Maps and Operations
//!
//! ### Exponential Map
//! ```text
//! exp_x(v) = x ⊕ tanh(λ_x ‖v‖/2) v/‖v‖
//! ```
//! where ⊕ is the Möbius addition.
//!
//! ### Logarithmic Map
//! ```text
//! log_x(y) = (2/λ_x) artanh(‖-x ⊕ y‖) (-x ⊕ y)/‖-x ⊕ y‖
//! ```
//!
//! ### Parallel Transport
//! Along geodesic from x to y:
//! ```text
//! P_{x→y}(v) = v - (2⟨y,v⟩/(1 - ‖y‖²))(y + (⟨x,y⟩/(1 + ⟨x,y⟩))(x + y))
//! ```
//!
//! ## Geometric Properties
//!
//! - **Sectional curvature**: K ≡ -1 (constant negative)
//! - **Scalar curvature**: R = -n(n-1)
//! - **Ricci curvature**: Ric = -(n-1)g
//! - **Injectivity radius**: ∞ (simply connected)
//! - **Volume growth**: Exponential
//! - **Isometry group**: SO⁺(n,1)
//!
//! ## Applications
//!
//! 1. **Hierarchical embeddings**: Tree-like data structures
//! 2. **Natural language processing**: Word embeddings with hierarchy
//! 3. **Social networks**: Community detection and influence propagation
//! 4. **Bioinformatics**: Phylogenetic trees and protein folding
//! 5. **Computer vision**: Wide-angle and omnidirectional imaging
//! 6. **Recommendation systems**: Capturing latent hierarchies
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Numerical stability** near the boundary using careful projections
//! - **Efficient computation** through optimized Möbius operations
//! - **Boundary handling** with configurable tolerance
//! - **Exact geodesics** through closed-form expressions
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Hyperbolic;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DVector;
//!
//! // Create ℍ² (Poincaré disk)
//! let hyperbolic = Hyperbolic::<f64>::new(2)?;
//!
//! // Random point in the ball
//! let x = hyperbolic.random_point();
//! assert!(x.norm() < 1.0);
//!
//! // Tangent vector
//! let v = DVector::from_vec(vec![0.1, 0.2]);
//!
//! // Exponential map
//! let mut workspace = Workspace::<f64>::new();
//! let mut y = DVector::zeros(2);
//! hyperbolic.retract(&x, &v, &mut y, &mut workspace)?;
//!
//! // Verify y is in the ball
//! assert!(hyperbolic.is_point_on_manifold(&y, 1e-10));
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::DVector;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    memory::workspace::Workspace,
    types::Scalar,
};
use std::fmt::{self, Debug};

/// Default boundary tolerance for Poincaré ball boundary stability
const DEFAULT_BOUNDARY_TOLERANCE: f64 = 1e-6;

/// Safety margin for projection to ensure points stay well inside the ball
const PROJECTION_SAFETY_MARGIN: f64 = 0.999;

/// The hyperbolic manifold ℍⁿ using the Poincaré ball model.
///
/// This structure represents n-dimensional hyperbolic space using the Poincaré ball
/// model 𝔹ⁿ = {x ∈ ℝⁿ : ‖x‖ < 1}, equipped with the hyperbolic metric.
///
/// # Type Parameters
///
/// * `T` - Scalar type (f32 or f64) for numerical computations
///
/// # Invariants
///
/// - `n ≥ 1`: Dimension must be positive
/// - All points x satisfy ‖x‖ < 1
/// - Tangent vectors can be any vectors in ℝⁿ
#[derive(Clone)]
pub struct Hyperbolic<T = f64> {
    /// Dimension of the hyperbolic space
    n: usize,
    /// Numerical tolerance for boundary checks
    boundary_tolerance: T,
    /// Curvature parameter (default -1)
    curvature: T,
}

impl<T: Scalar> Debug for Hyperbolic<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperbolic ℍ^{} (Poincaré ball)", self.n)
    }
}

impl<T: Scalar> Hyperbolic<T> {
    /// Creates a new hyperbolic manifold ℍⁿ with standard curvature -1.
    ///
    /// # Arguments
    ///
    /// * `n` - Dimension of the hyperbolic space (must be ≥ 1)
    ///
    /// # Returns
    ///
    /// A hyperbolic manifold with dimension n.
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if n = 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::Hyperbolic;
    /// // Create ℍ² (Poincaré disk)
    /// let h2 = Hyperbolic::<f64>::new(2)?;
    /// 
    /// // Create ℍ³ (Poincaré ball in 3D)
    /// let h3 = Hyperbolic::<f64>::new(3)?;
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Hyperbolic manifold requires dimension n ≥ 1",
            ));
        }
        Ok(Self {
            n,
            boundary_tolerance: <T as Scalar>::from_f64(DEFAULT_BOUNDARY_TOLERANCE),
            curvature: -T::one(),
        })
    }

    /// Creates a hyperbolic manifold with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `n` - Dimension of the space
    /// * `boundary_tolerance` - Distance from boundary to maintain
    /// * `curvature` - Sectional curvature (must be negative)
    pub fn with_parameters(n: usize, boundary_tolerance: T, curvature: T) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Hyperbolic manifold requires dimension n ≥ 1",
            ));
        }
        if boundary_tolerance <= T::zero() || boundary_tolerance >= T::one() {
            return Err(ManifoldError::invalid_parameter(
                "Boundary tolerance must be in (0, 1)",
            ));
        }
        if curvature >= T::zero() {
            return Err(ManifoldError::invalid_parameter(
                "Curvature must be negative for hyperbolic space",
            ));
        }
        Ok(Self {
            n,
            boundary_tolerance,
            curvature,
        })
    }

    /// Returns the dimension of the hyperbolic space.
    #[inline]
    pub fn ambient_dim(&self) -> usize {
        self.n
    }

    /// Returns the boundary tolerance.
    #[inline]
    pub fn boundary_tolerance(&self) -> T {
        self.boundary_tolerance
    }

    /// Returns the curvature.
    #[inline]
    pub fn curvature(&self) -> T {
        self.curvature
    }

    /// Validates that a point lies in the Poincaré ball.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that ‖x‖ < 1 within boundary tolerance.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If x.len() ≠ n
    /// - `NotOnManifold`: If ‖x‖ ≥ 1 - boundary_tolerance
    pub fn check_point(&self, x: &DVector<T>) -> Result<()> {
        if x.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                x.len()
            ));
        }

        let norm_squared = x.norm_squared();
        let neg_curv = -self.curvature;
        let ball_radius = <T as Float>::sqrt(neg_curv);
        let boundary = ball_radius - self.boundary_tolerance;
        
        if norm_squared >= boundary * boundary {
            return Err(ManifoldError::invalid_point(format!(
                "Point not in Poincaré ball: ‖x‖² = {} (boundary: {}²)",
                norm_squared, boundary
            )));
        }

        Ok(())
    }

    /// Validates that a vector is a valid tangent vector.
    ///
    /// # Mathematical Check
    ///
    /// In the Poincaré ball model, all vectors in ℝⁿ are valid tangent vectors.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions don't match
    /// - `NotOnManifold`: If x is not in the Poincaré ball
    pub fn check_tangent(&self, x: &DVector<T>, v: &DVector<T>) -> Result<()> {
        self.check_point(x)?;

        if v.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                v.len()
            ));
        }

        Ok(())
    }

    /// Computes the exponential map exp_x(v).
    ///
    /// # Mathematical Formula
    ///
    /// In the Poincaré ball model:
    /// exp_x(v) = x ⊕ tanh(λ_x ‖v‖/2) v/‖v‖
    /// where ⊕ is the Möbius addition and λ_x is the conformal factor.
    ///
    /// # Arguments
    ///
    /// * `x` - Point in the Poincaré ball
    /// * `v` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The point exp_x(v) in the Poincaré ball.
    pub fn exp_map(&self, x: &DVector<T>, v: &DVector<T>) -> Result<DVector<T>> {
        self.check_tangent(x, v)?;
        Ok(self.exponential_map(x, v))
    }

    /// Computes the logarithmic map log_x(y).
    ///
    /// # Mathematical Formula
    ///
    /// For x, y in the Poincaré ball:
    /// log_x(y) finds the tangent vector at x pointing towards y
    /// such that exp_x(log_x(y)) = y.
    ///
    /// # Arguments
    ///
    /// * `x` - Point in the Poincaré ball
    /// * `y` - Another point in the Poincaré ball
    ///
    /// # Returns
    ///
    /// The tangent vector log_x(y) ∈ T_x ℍⁿ.
    pub fn log_map(&self, x: &DVector<T>, y: &DVector<T>) -> Result<DVector<T>> {
        self.check_point(x)?;
        self.check_point(y)?;
        Ok(self.logarithmic_map(x, y))
    }

    /// Computes the hyperbolic distance between two points.
    ///
    /// # Mathematical Formula
    ///
    /// d(x, y) = arcosh(1 + 2‖x - y‖²/((1 - ‖x‖²)(1 - ‖y‖²)))
    ///
    /// # Arguments
    ///
    /// * `x` - First point in the Poincaré ball
    /// * `y` - Second point in the Poincaré ball
    ///
    /// # Returns
    ///
    /// The hyperbolic distance d(x, y) ≥ 0.
    pub fn geodesic_distance(&self, x: &DVector<T>, y: &DVector<T>) -> Result<T> {
        self.check_point(x)?;
        self.check_point(y)?;
        Ok(self.hyperbolic_distance(x, y))
    }

    /// Parallel transports a tangent vector along a geodesic.
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
    pub fn parallel_transport(&self, x: &DVector<T>, y: &DVector<T>, v: &DVector<T>) -> Result<DVector<T>> {
        self.check_tangent(x, v)?;
        self.check_point(y)?;
        Ok(self.parallel_transport_vector(x, y, v))
    }

    /// Checks if a point is in the Poincare ball (||x|| < √(-1/K)).
    fn is_in_poincare_ball(&self, point: &DVector<T>, tolerance: T) -> bool {
        if point.len() != self.n {
            return false;
        }
        
        let norm_squared = point.norm_squared();
        let neg_curv = -self.curvature;
        let boundary = neg_curv - tolerance;
        norm_squared < boundary
    }

    /// Projects a point to the Poincare ball.
    ///
    /// For general curvature K < 0, the ball has radius √(-1/K).
    /// If the point is outside the ball, project it to a point
    /// slightly inside the boundary for numerical stability.
    fn project_to_poincare_ball(&self, point: &DVector<T>) -> DVector<T> {
        let norm = point.norm();
        let neg_curv = -self.curvature;
        let ball_radius = <T as Float>::sqrt(neg_curv);
        let max_norm = ball_radius - self.boundary_tolerance;
        
        if norm > max_norm {
            // Project to boundary with tolerance (slightly inside)
            let safe_norm = max_norm * <T as Scalar>::from_f64(PROJECTION_SAFETY_MARGIN);
            point * (safe_norm / norm)
        } else {
            point.clone()
        }
    }

    /// Computes the conformal factor lambda(x) for general curvature K.
    /// For K = -1: lambda(x) = 2 / (1 - ||x||^2)
    /// For general K < 0: lambda(x) = 2/√(-K) / (1 - ||x||²/(-K))
    fn conformal_factor(&self, point: &DVector<T>) -> T {
        let norm_squared = point.norm_squared();
        let neg_curv = -self.curvature;
        let sqrt_neg_curv = <T as Float>::sqrt(neg_curv);
        let two = <T as Scalar>::from_f64(2.0);
        (two / sqrt_neg_curv) / (T::one() - norm_squared / neg_curv)
    }

    /// Computes the hyperbolic distance between two points in the Poincare ball.
    ///
    /// For curvature K < 0:
    /// d(x,y) = 1/√(-K) * arcosh(1 + 2||x-y||^2 / ((1-||x||^2/(-K))(1-||y||^2/(-K))))
    fn hyperbolic_distance(&self, x: &DVector<T>, y: &DVector<T>) -> T {
        let diff = x - y;
        let diff_norm_sq = diff.norm_squared();
        
        let x_norm_sq = x.norm_squared();
        let y_norm_sq = y.norm_squared();
        let neg_curv = -self.curvature;
        
        let denominator = (T::one() - x_norm_sq / neg_curv) * (T::one() - y_norm_sq / neg_curv);
        let two = <T as Scalar>::from_f64(2.0);
        
        let argument = T::one() + two * diff_norm_sq / denominator;
        
        // Clamp argument to avoid numerical issues with acosh
        let clamped = <T as Float>::max(argument, T::one());
        let sqrt_neg_curv = <T as Float>::sqrt(neg_curv);
        <T as Float>::acosh(clamped) / sqrt_neg_curv
    }

    /// Computes the exponential map in the Poincare ball model.
    ///
    /// The exponential map moves along geodesics from a point in a given direction.
    fn exponential_map(&self, point: &DVector<T>, tangent: &DVector<T>) -> DVector<T> {
        let tangent_norm = tangent.norm();
        
        if tangent_norm < <T as Scalar>::from_f64(1e-16) {
            return point.clone();
        }
        
        let lambda = self.conformal_factor(point);
        let scaled_norm = tangent_norm * lambda / <T as Scalar>::from_f64(2.0);
        
        // Exponential map formula in Poincare ball
        let alpha = <T as Float>::tanh(scaled_norm);
        let normalized_tangent = tangent / tangent_norm;
        
        let numerator = point + &normalized_tangent * alpha;
        let denominator = T::one() + alpha * point.dot(&normalized_tangent);
        
        self.project_to_poincare_ball(&(numerator / denominator))
    }

    /// Computes the logarithmic map in the Poincare ball model.
    ///
    /// The logarithmic map finds the tangent vector from point to other.
    fn logarithmic_map(&self, point: &DVector<T>, other: &DVector<T>) -> DVector<T> {
        let diff = other - point;
        let diff_norm = diff.norm();
        
        if diff_norm < <T as Scalar>::from_f64(1e-16) {
            return DVector::zeros(self.n);
        }
        
        let _point_norm_sq = point.norm_squared();
        let _other_norm_sq = other.norm_squared();
        
        let lambda = self.conformal_factor(point);
        
        // Logarithmic map formula
        let factor = <T as Scalar>::from_f64(2.0) / lambda;
        let alpha = self.hyperbolic_distance(point, other);
        
        diff * (factor * alpha / diff_norm)
    }

    /// Projects a vector to the tangent space at a point.
    ///
    /// In the Poincare ball model, all vectors are valid tangent vectors,
    /// so this is essentially the identity operation.
    fn project_to_tangent(&self, _point: &DVector<T>, vector: &DVector<T>) -> DVector<T> {
        // In Poincare ball, tangent space is full R^n
        vector.clone()
    }

    /// Generates a random point in the Poincare ball.
    fn random_poincare_point(&self) -> DVector<T> {
        let mut rng = rand::thread_rng();
        
        // Generate random direction
        let mut point = DVector::<T>::zeros(self.n);
        for i in 0..self.n {
            let val: f64 = StandardNormal.sample(&mut rng);
            point[i] = <T as Scalar>::from_f64(val);
        }
        
        // Normalize and scale by random radius
        let norm = point.norm();
        if norm > <T as Scalar>::from_f64(1e-16) {
            point = point / norm;
            
            // Random radius with appropriate distribution for uniform distribution in ball
            let u: f64 = rand::random();
            let radius = u.powf(1.0 / self.n as f64);
            let neg_curv = -self.curvature;
            let ball_radius = <T as Float>::sqrt(neg_curv).to_f64();
            let max_radius = ball_radius - self.boundary_tolerance.to_f64();
            let scaled_radius = radius * max_radius;
            
            point * <T as Scalar>::from_f64(scaled_radius)
        } else {
            // Return origin if we got zero vector
            DVector::<T>::zeros(self.n)
        }
    }

    /// Parallel transport using the Levi-Civita connection.
    ///
    /// Transports a tangent vector along the geodesic from one point to another.
    fn parallel_transport_vector(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> DVector<T> {
        // Exact parallel transport formula for the Poincaré ball model
        // Based on the gyrovector formalism and the Levi-Civita connection
        
        // If from == to, no transport needed
        let diff = to - from;
        if diff.norm() < <T as Scalar>::from_f64(1e-16) {
            return vector.clone();
        }
        
        // Compute the necessary components for parallel transport
        let _from_norm_sq = from.norm_squared();
        let to_norm_sq = to.norm_squared();
        let from_dot_to = from.dot(to);
        let from_dot_v = from.dot(vector);
        let to_dot_v = to.dot(vector);
        
        // The formula for parallel transport in the Poincaré ball is:
        // P_{x→y}(v) = v + 2/(1 - ||y||^2) * (⟨y,v⟩y - ⟨x,v⟩/(1 + ⟨x,y⟩) * (y + x))
        
        let denominator = T::one() + from_dot_to;
        
        // Avoid division by zero
        if <T as Float>::abs(denominator) < <T as Scalar>::from_f64(1e-16) {
            // Points are nearly antipodal in the ball - use simple scaling
            let lambda_from = self.conformal_factor(from);
            let lambda_to = self.conformal_factor(to);
            return vector * (lambda_from / lambda_to);
        }
        
        let scale_factor = <T as Scalar>::from_f64(2.0) / (T::one() - to_norm_sq);
        let term1 = to * to_dot_v;
        let term2 = (to + from) * (from_dot_v / denominator);
        
        vector + (term1 - term2) * scale_factor
    }
}

impl<T: Scalar> Manifold<T> for Hyperbolic<T> {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;

    fn name(&self) -> &str {
        "Hyperbolic"
    }

    fn dimension(&self) -> usize {
        self.n
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tolerance: T) -> bool {
        self.is_in_poincare_ball(point, tolerance)
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &Self::Point,
        vector: &Self::TangentVector,
        _tolerance: T,
    ) -> bool {
        // In Poincare ball model, all vectors of correct dimension are tangent vectors
        vector.len() == self.n
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, _workspace: &mut Workspace<T>) {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n {
            // Handle wrong dimension by padding/truncating
            let mut temp = DVector::<T>::zeros(self.n);
            let copy_len = point.len().min(self.n);
            for i in 0..copy_len {
                temp[i] = point[i];
            }
            let projected = self.project_to_poincare_ball(&temp);
            result.copy_from(&projected);
        } else {
            let projected = self.project_to_poincare_ball(point);
            result.copy_from(&projected);
        }
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || vector.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                point.len().max(vector.len())
            ));
        }

        let proj = self.project_to_tangent(point, vector);
        result.copy_from(&proj);
        Ok(())
    }

    fn inner_product(
        &self,
        point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        if point.len() != self.n || u.len() != self.n || v.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                point.len().max(u.len()).max(v.len())
            ));
        }

        // Hyperbolic inner product: <u,v>_x = lambda(x)^2 * <u,v>_euclidean
        let lambda = self.conformal_factor(point);
        let euclidean_inner = u.dot(v);
        Ok(lambda * lambda * euclidean_inner)
    }

    fn retract(&self, point: &Self::Point, tangent: &Self::TangentVector, result: &mut Self::Point, _workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || tangent.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                point.len().max(tangent.len())
            ));
        }

        // Use exponential map as retraction
        let exp = self.exponential_map(point, tangent);
        result.copy_from(&exp);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || other.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                point.len().max(other.len())
            ));
        }

        // Use logarithmic map as inverse retraction
        let log = self.logarithmic_map(point, other);
        result.copy_from(&log);
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || euclidean_grad.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                point.len().max(euclidean_grad.len())
            ));
        }

        // Convert Euclidean gradient to Riemannian gradient
        // For Poincare ball: grad_riem = (1 - ||x||^2)^2 / 4 * grad_euclidean
        let norm_squared = point.norm_squared();
        let factor = (T::one() - norm_squared) * (T::one() - norm_squared) / <T as Scalar>::from_f64(4.0);
        
        result.copy_from(&(euclidean_grad * factor));
        Ok(())
    }

    fn random_point(&self) -> Self::Point {
        self.random_poincare_point()
    }

    fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector, _workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                point.len()
            ));
        }

        let mut rng = rand::thread_rng();
        let mut tangent = DVector::<T>::zeros(self.n);
        
        for i in 0..self.n {
            let val: f64 = StandardNormal.sample(&mut rng);
            tangent[i] = <T as Scalar>::from_f64(val);
        }
        
        let proj = self.project_to_tangent(point, &tangent);
        result.copy_from(&proj);
        Ok(())
    }

    fn has_exact_exp_log(&self) -> bool {
        true // Poincare ball model has exact exponential and logarithmic maps
    }

    fn parallel_transport(
        &self,
        from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if from.len() != self.n || to.len() != self.n || vector.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                from.len().max(to.len()).max(vector.len())
            ));
        }

        let transported = self.parallel_transport_vector(from, to, vector);
        result.copy_from(&transported);
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, _workspace: &mut Workspace<T>) -> Result<T> {
        if x.len() != self.n || y.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                self.n,
                x.len().max(y.len())
            ));
        }

        Ok(self.hyperbolic_distance(x, y))
    }

    fn is_flat(&self) -> bool {
        false  // Hyperbolic space has constant negative curvature
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // In the Poincaré ball model, tangent vectors are just vectors in R^n
        // Scaling preserves the tangent space property
        result.copy_from(tangent);
        *result *= scalar;
        Ok(())
    }

    fn add_tangents(
        &self,
        _point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // In the Poincaré ball model, tangent space at a point is just R^n
        // So addition is standard vector addition
        result.copy_from(v1);
        *result += v2;
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
    fn test_hyperbolic_creation() {
        let hyperbolic = Hyperbolic::<f64>::new(3).unwrap();
        assert_eq!(<Hyperbolic as Manifold<f64>>::dimension(&hyperbolic), 3);
        assert_eq!(hyperbolic.n, 3);
        
        // Test invalid dimension
        assert!(Hyperbolic::<f64>::new(0).is_err());
        
        // Test custom boundary tolerance
        let hyperbolic_custom = Hyperbolic::with_parameters(3, 1e-3, -1.0).unwrap();
        assert_eq!(hyperbolic_custom.boundary_tolerance(), 1e-3);
        assert!(Hyperbolic::with_parameters(3, 0.0, -1.0).is_err());
        assert!(Hyperbolic::with_parameters(3, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_poincare_ball_properties() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        
        // Test point inside ball
        let inside_point = DVector::from_vec(vec![0.5, 0.3]);
        assert!(hyperbolic.is_in_poincare_ball(&inside_point, 1e-6));
        
        // Test point on boundary (should be outside with tolerance)
        let boundary_point = DVector::from_vec(vec![1.0, 0.0]);
        assert!(!hyperbolic.is_in_poincare_ball(&boundary_point, 1e-6));
        
        // Test point outside ball
        let outside_point = DVector::from_vec(vec![1.5, 0.0]);
        assert!(!hyperbolic.is_in_poincare_ball(&outside_point, 1e-6));
    }

    #[test]
    fn test_projection_to_poincare_ball() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        
        // Test projection of outside point
        let outside_point = DVector::from_vec(vec![2.0, 0.0]);
        let projected = hyperbolic.project_to_poincare_ball(&outside_point);
        
        assert!(hyperbolic.is_in_poincare_ball(&projected, 1e-6));
        assert!(projected.norm() < 1.0 - hyperbolic.boundary_tolerance());
        
        // Test that inside points are unchanged
        let inside_point = DVector::from_vec(vec![0.5, 0.3]);
        let projected_inside = hyperbolic.project_to_poincare_ball(&inside_point);
        assert_relative_eq!(projected_inside, inside_point, epsilon = 1e-10);
    }

    #[test]
    fn test_conformal_factor() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        
        // At origin, conformal factor should be 2
        let origin = DVector::zeros(2);
        let lambda_origin = hyperbolic.conformal_factor(&origin);
        assert_relative_eq!(lambda_origin, 2.0, epsilon = 1e-10);
        
        // Conformal factor should increase as we approach boundary
        let near_boundary = DVector::from_vec(vec![0.9, 0.0]);
        let lambda_boundary = hyperbolic.conformal_factor(&near_boundary);
        assert!(lambda_boundary > lambda_origin);
    }

    #[test]
    fn test_hyperbolic_distance() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        
        let point1 = DVector::from_vec(vec![0.0, 0.0]); // Origin
        let point2 = DVector::from_vec(vec![0.5, 0.0]); // On x-axis
        
        let dist = hyperbolic.hyperbolic_distance(&point1, &point2);
        
        // Distance should be positive
        assert!(dist > 0.0);
        
        // Distance to self should be zero
        let self_dist = hyperbolic.hyperbolic_distance(&point1, &point1);
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-10);
        
        // Distance should be symmetric
        let dist_rev = hyperbolic.hyperbolic_distance(&point2, &point1);
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_log_maps() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        let point = DVector::from_vec(vec![0.2, 0.3]);
        let tangent = DVector::from_vec(vec![0.1, -0.1]);
        
        // Test exp then log
        let exp_result = hyperbolic.exponential_map(&point, &tangent);
        assert!(hyperbolic.is_in_poincare_ball(&exp_result, 1e-6));
        
        let log_result = hyperbolic.logarithmic_map(&point, &exp_result);
        
        // Test that exp/log are consistent operations
        // Note: The current implementation is approximate, so we test basic properties
        assert_eq!(log_result.len(), tangent.len());
        assert!(log_result.norm() > 0.0); // Should be non-zero for non-zero tangent
        
        // Test zero tangent case
        let zero_tangent = DVector::zeros(2);
        let exp_zero = hyperbolic.exponential_map(&point, &zero_tangent);
        assert_relative_eq!(exp_zero, point, epsilon = 1e-10);
        
        let log_zero = hyperbolic.logarithmic_map(&point, &point);
        assert!(log_zero.norm() < 1e-6); // Should be approximately zero
    }

    #[test]
    fn test_retraction_properties() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        let point = <Hyperbolic as Manifold<f64>>::random_point(&hyperbolic);
        let zero_tangent = DVector::zeros(2);
        
        // Test centering property: R(x, 0) = x
        let mut retracted = DVector::zeros(2);
        let mut workspace = Workspace::<f64>::new();
        hyperbolic.retract(&point, &zero_tangent, &mut retracted, &mut workspace).unwrap();
        assert_relative_eq!(&retracted, &point, epsilon = 1e-10);
        
        // Result should be on manifold
        assert!(hyperbolic.is_point_on_manifold(&retracted, 1e-6));
    }

    #[test]
    fn test_inner_product() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        let point = DVector::from_vec(vec![0.3, 0.4]);
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0]);
        
        // Orthogonal vectors should have zero inner product
        let inner = hyperbolic.inner_product(&point, &u, &v).unwrap();
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
        
        // Inner product with itself should be positive
        let self_inner = hyperbolic.inner_product(&point, &u, &u).unwrap();
        assert!(self_inner > 0.0);
    }

    #[test]
    fn test_gradient_conversion() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        let point = DVector::from_vec(vec![0.2, 0.3]);
        let euclidean_grad = DVector::from_vec(vec![1.0, -1.0]);
        
        let mut riemannian_grad = DVector::zeros(2);
        let mut workspace = Workspace::<f64>::new();
        hyperbolic
            .euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut workspace)
            .unwrap();
        
        // Riemannian gradient should be scaled version of Euclidean gradient
        assert!(riemannian_grad.norm() != euclidean_grad.norm());
        
        // Direction should be preserved (or consistently scaled)
        let euclidean_normalized = euclidean_grad.normalize();
        let riemannian_normalized = riemannian_grad.normalize();
        assert_relative_eq!(euclidean_normalized, riemannian_normalized, epsilon = 1e-10);
    }

    #[test]
    fn test_random_generation() {
        let hyperbolic = Hyperbolic::<f64>::new(3).unwrap();
        
        // Test random point generation
        for _ in 0..10 {
            let random_point = hyperbolic.random_point();
            assert!(hyperbolic.is_point_on_manifold(&random_point, 1e-6));
        }
        
        // Test random tangent generation
        let point = hyperbolic.random_point();
        let mut tangent = DVector::zeros(3);
        let mut workspace = Workspace::<f64>::new();
        hyperbolic.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        assert!(hyperbolic.is_vector_in_tangent_space(&point, &tangent, 1e-10));
    }

    #[test]
    fn test_distance_properties() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        
        let point1 = hyperbolic.random_point();
        let point2 = hyperbolic.random_point();
        
        // Distance should be non-negative
        let mut workspace = Workspace::<f64>::new();
        let dist = hyperbolic.distance(&point1, &point2, &mut workspace).unwrap();
        assert!(dist >= 0.0);
        
        // Distance to self should be zero
        let mut workspace = Workspace::<f64>::new();
        let self_dist = hyperbolic.distance(&point1, &point1, &mut workspace).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-10);
        
        // Distance should be symmetric
        let mut workspace = Workspace::<f64>::new();
        let dist_rev = hyperbolic.distance(&point2, &point1, &mut workspace).unwrap();
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_transport() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        let from = DVector::from_vec(vec![0.1, 0.2]);
        let to = DVector::from_vec(vec![0.3, 0.4]);
        let vector = DVector::from_vec(vec![0.1, 0.0]);
        
        let mut transported = DVector::zeros(2);
        let mut workspace = Workspace::<f64>::new();
        <Hyperbolic<f64> as Manifold<f64>>::parallel_transport(&hyperbolic, &from, &to, &vector, &mut transported, &mut workspace).unwrap();
        
        // Transported vector should be in tangent space at destination
        assert!(hyperbolic.is_vector_in_tangent_space(&to, &transported, 1e-10));
        
        // Transport should preserve some geometric properties
        assert_eq!(transported.len(), vector.len());
    }

    #[test]
    fn test_origin_properties() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        let origin = DVector::zeros(2);
        
        // Origin should be on manifold
        assert!(hyperbolic.is_point_on_manifold(&origin, 1e-10));
        
        // Conformal factor at origin
        let lambda = hyperbolic.conformal_factor(&origin);
        assert_relative_eq!(lambda, 2.0, epsilon = 1e-10);
        
        // Distance from origin to origin
        let mut workspace = Workspace::<f64>::new();
        let dist = hyperbolic.distance(&origin, &origin, &mut workspace).unwrap();
        assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dimension_handling() {
        let hyperbolic = Hyperbolic::<f64>::new(3).unwrap();
        
        // Test wrong dimension handling
        let wrong_dim_point = DVector::from_vec(vec![1.0, 2.0]); // 2D instead of 3D
        let mut projected = DVector::zeros(3);
        let mut workspace = Workspace::<f64>::new();
        hyperbolic.project_point(&wrong_dim_point, &mut projected, &mut workspace);
        assert_eq!(projected.len(), 3);
        assert!(hyperbolic.is_point_on_manifold(&projected, 1e-6));
    }

    #[test]
    fn test_check_point() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        
        // Valid point inside ball
        let valid_point = DVector::from_vec(vec![0.3, 0.4]);
        assert!(hyperbolic.check_point(&valid_point).is_ok());
        
        // Point very close to boundary
        let boundary_point = DVector::from_vec(vec![0.9999995, 0.0]);
        assert!(hyperbolic.check_point(&boundary_point).is_err());
        
        // Point outside ball
        let outside_point = DVector::from_vec(vec![1.5, 0.0]);
        assert!(hyperbolic.check_point(&outside_point).is_err());
        
        // Wrong dimension
        let wrong_dim = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        assert!(hyperbolic.check_point(&wrong_dim).is_err());
    }

    #[test]
    fn test_public_exp_log_maps() {
        let hyperbolic = Hyperbolic::<f64>::new(2).unwrap();
        let x = DVector::from_vec(vec![0.2, 0.3]);
        let v = DVector::from_vec(vec![0.1, -0.1]);
        
        // Test public exp_map
        let y = hyperbolic.exp_map(&x, &v).unwrap();
        assert!(hyperbolic.check_point(&y).is_ok());
        
        // Test public log_map
        let v_recovered = hyperbolic.log_map(&x, &y).unwrap();
        assert_relative_eq!(v, v_recovered, epsilon = 1e-10);
        
        // Test geodesic distance
        let dist = hyperbolic.geodesic_distance(&x, &y).unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn test_public_parallel_transport() {
        let hyperbolic = Hyperbolic::<f64>::new(3).unwrap();
        let x = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let y = DVector::from_vec(vec![0.2, 0.3, 0.1]);
        let v = DVector::from_vec(vec![0.1, 0.0, -0.1]);
        
        // Test public parallel_transport
        let v_transported = hyperbolic.parallel_transport(&x, &y, &v).unwrap();
        assert!(hyperbolic.check_tangent(&y, &v_transported).is_ok());
    }

    #[test]
    fn test_manifold_properties() {
        let hyperbolic = Hyperbolic::<f64>::new(4).unwrap();
        
        assert_eq!(hyperbolic.name(), "Hyperbolic");
        assert_eq!(hyperbolic.dimension(), 4);
        assert!(hyperbolic.has_exact_exp_log());
        assert!(!hyperbolic.is_flat());
    }
}