//! Core manifold trait and associated types.
//!
//! This module defines the fundamental `Manifold` trait that all Riemannian
//! manifolds must implement. A manifold is a topological space that locally
//! resembles Euclidean space and can be equipped with a Riemannian metric.
//!
//! # Mathematical Background
//!
//! A Riemannian manifold (ℳ, g) consists of:
//! - A smooth manifold ℳ
//! - A Riemannian metric g that assigns an inner product to each tangent space T_p ℳ
//!
//! ## Key Concepts
//!
//! - **Tangent space**: T_p ℳ is the linear approximation of ℳ at point p
//! - **Retraction**: A smooth map R_p: T_p ℳ → ℳ that approximates the exponential map
//! - **Riemannian gradient**: The unique vector in T_p ℳ representing the derivative
//! - **Parallel transport**: Moving vectors along curves while preserving angles
//! - **Geodesics**: Curves that locally minimize distance on the manifold
//!
//! # Examples
//!
//! ## Basic Manifold Usage
//!
//! ```rust,ignore
//! # use riemannopt_core::manifold::Manifold;
//! # use riemannopt_core::memory::Workspace;
//! # use nalgebra::DVector;
//! 
//! #[derive(Debug)]
//! struct MySphere { radius: f64 }
//! 
//! impl Manifold<f64> for MySphere {
//!     type Point = DVector<f64>;
//!     type TangentVector = DVector<f64>;
//!     // ... implement required methods ...
//! }
//! 
//! let sphere = MySphere { radius: 1.0 };
//! let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
//! let tangent = DVector::from_vec(vec![0.0, 0.1, 0.0]);
//!
//! // Check if point is on manifold
//! assert!(sphere.is_point_on_manifold(&point, 1e-10));
//!
//! // Perform retraction
//! let mut new_point = point.clone();
//! let mut workspace = Workspace::new();
//! sphere.retract(&point, &tangent, &mut new_point, &mut workspace)?;
//! ```
//!
//! ## Common Manifolds
//!
//! This library provides implementations for several important manifolds:
//!
//! - **Unit Sphere S^{n-1}**: Points with unit norm in ℝⁿ
//! - **Stiefel St(n,p)**: n×p matrices with orthonormal columns  
//! - **Grassmann Gr(n,p)**: p-dimensional subspaces of ℝⁿ
//! - **SPD(n)**: Symmetric positive definite matrices
//! - **Euclidean ℝⁿ**: Flat space with identity metric

use crate::{error::Result, types::Scalar, memory::workspace::Workspace};
use num_traits::Float;
use std::fmt::Debug;

/// Trait for Riemannian manifolds.
///
/// A Riemannian manifold (ℳ, g) is a smooth manifold ℳ equipped with a Riemannian 
/// metric g that provides an inner product structure on each tangent space T_p ℳ.
///
/// This trait defines the interface that all manifolds must implement to be used
/// with Riemannian optimization algorithms. It provides methods for:
///
/// ## Core Operations
/// - **Manifold membership**: Check if a point p ∈ ℳ
/// - **Projection**: Project ambient points onto the manifold: Π: ℝⁿ → ℳ
/// - **Tangent projection**: Project vectors to tangent space: P_p: ℝⁿ → T_p ℳ
/// - **Retraction**: Approximate exponential map: R_p: T_p ℳ → ℳ
/// - **Riemannian metric**: Inner product ⟨·,·⟩_p: T_p ℳ × T_p ℳ → ℝ
///
/// ## Mathematical Properties
/// 
/// The trait ensures the following mathematical properties:
/// 
/// 1. **Projection idempotency**: Π(Π(x)) = Π(x) for all x ∈ ℝⁿ
/// 2. **Tangent orthogonality**: ⟨P_p(v), p⟩ = 0 for embedded manifolds
/// 3. **Retraction constraints**: R_p(0) = p and dR_p(0) = id_{T_p ℳ}
/// 4. **Metric properties**: ⟨·,·⟩_p is symmetric, bilinear, and positive definite
///
/// # Type Parameters
///
/// - `T`: The scalar type (f32 or f64)
/// - `D`: The dimension type of the manifold's representation
///
/// # Implementation Notes
///
/// When implementing this trait:
/// - Ensure numerical stability for all operations
/// - Maintain manifold constraints (e.g., orthogonality, normalization)
/// - Use efficient algorithms (e.g., QR decomposition for Stiefel)
/// - Consider caching expensive computations
///
/// # Examples
///
/// ## Unit Sphere Implementation
///
/// ```rust,ignore
/// use riemannopt_core::prelude::*;
/// use nalgebra::{Const, DVector};
///
/// /// Unit sphere S^{n-1} = {x ∈ ℝⁿ : ||x|| = 1}
/// struct Sphere<T: Scalar> {
///     ambient_dim: usize,
/// }
///
/// impl<T: Scalar> Manifold<T, nalgebra::Dyn> for Sphere<T> {
///     fn name(&self) -> &str { "Sphere" }
///     
///     fn dimension(&self) -> usize {
///         self.ambient_dim - 1  // S^{n-1} has dimension n-1
///     }
///     
///     fn is_point_on_manifold(&self, point: &DVector<T>, tol: T) -> bool {
///         // Check ||x|| = 1
///         (point.norm_squared() - T::one()).abs() < tol
///     }
///     
///     fn project(&self, point: &DVector<T>) -> Result<DVector<T>> {
///         // Π(x) = x / ||x||
///         let norm = point.norm();
///         if norm < T::epsilon() {
///             return Err(ManifoldError::NumericalError);
///         }
///         Ok(point / norm)
///     }
///     
///     fn tangent_projection(&self, point: &DVector<T>, vector: &DVector<T>) 
///         -> Result<DVector<T>> {
///         // P_p(v) = v - ⟨v, p⟩ p  (orthogonal projection)
///         Ok(vector - point * vector.dot(point))
///     }
/// }
/// ```
///
/// ## Stiefel Manifold Implementation
///
/// ```rust,ignore
/// /// Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p}
/// struct Stiefel<T: Scalar> {
///     n: usize,  // Number of rows
///     p: usize,  // Number of columns
/// }
///
/// impl<T: Scalar> Manifold<T, nalgebra::Dyn> for Stiefel<T> {
///     fn name(&self) -> &str { "Stiefel" }
///     
///     fn dimension(&self) -> usize {
///         self.n * self.p - self.p * (self.p + 1) / 2
///     }
///     
///     fn project(&self, matrix: &DMatrix<T>) -> Result<DMatrix<T>> {
///         // QR decomposition: X = QR, return Q
///         let qr = matrix.qr();
///         Ok(qr.q())
///     }
///     
///     fn retract(&self, point: &DMatrix<T>, tangent: &DMatrix<T>) 
///         -> Result<DMatrix<T>> {
///         // QR-based retraction: R_X(V) = qf(X + V)
///         let sum = point + tangent;
///         self.project(&sum)
///     }
/// }
/// ```
pub trait Manifold<T: Scalar>: Debug + Send + Sync {
    /// The type of data for a point (e.g., DVector<T> or DMatrix<T>).
    type Point: Clone + Debug + Send + Sync;
    /// The type of data for a tangent vector.
    type TangentVector: Clone + Debug + Send + Sync;
    /// Returns a human-readable name for the manifold.
    fn name(&self) -> &str;

    /// Returns the intrinsic dimension of the manifold.
    ///
    /// For example, the sphere S^{n-1} embedded in R^n has dimension n-1.
    fn dimension(&self) -> usize;


    /// Checks if a point lies on the manifold within a given tolerance.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    /// * `tol` - Tolerance for the membership test
    ///
    /// # Returns
    ///
    /// `true` if the point is on the manifold within tolerance, `false` otherwise.
    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool;

    /// Checks if a vector is in the tangent space at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `vector` - The vector to check
    /// * `tol` - Tolerance for the membership test
    ///
    /// # Returns
    ///
    /// `true` if the vector is in T_point M within tolerance, `false` otherwise.
    fn is_vector_in_tangent_space(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        tol: T,
    ) -> bool;

    /// Projects a point onto the manifold.
    ///
    /// This method takes a point in the ambient space and writes the closest
    /// point on the manifold (in the Euclidean sense) to the output buffer.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to project
    /// * `result` - Pre-allocated output buffer for the projected point
    /// * `workspace` - Pre-allocated workspace for temporary computations
    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, workspace: &mut Workspace<T>);

    /// Projects a vector onto the tangent space at a given point.
    ///
    /// The tangent space T_p ℳ at point p is the linear space of all possible
    /// directions of motion on the manifold at p. This method computes the
    /// orthogonal projection onto this space:
    ///
    /// P_p: ℝⁿ → T_p ℳ
    ///
    /// For embedded manifolds, this typically involves:
    /// - **Sphere**: P_p(v) = v - ⟨v,p⟩p (remove normal component)
    /// - **Stiefel**: P_p(V) = V - X(X^T V + V^T X)/2 (skew-symmetric projection)
    /// - **Grassmann**: P_p(V) = (I - XX^T)V (orthogonal complement projection)
    ///
    /// # Mathematical Properties
    ///
    /// The projection satisfies:
    /// 1. **Idempotency**: P_p(P_p(v)) = P_p(v)
    /// 2. **Linearity**: P_p(αu + βv) = αP_p(u) + βP_p(v)
    /// 3. **Orthogonality**: For embedded manifolds, ⟨P_p(v), n⟩ = 0 where n is normal to ℳ
    ///
    /// # Arguments
    ///
    /// * `point` - A point p ∈ ℳ on the manifold
    /// * `vector` - The ambient vector v ∈ ℝⁿ to project
    /// * `result` - Pre-allocated output buffer for the projected tangent vector
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Errors
    ///
    /// Returns an error if `point` is not on the manifold within numerical tolerance.
    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Computes the Riemannian inner product between two tangent vectors.
    ///
    /// The Riemannian metric g provides an inner product on each tangent space:
    /// g_p: T_p ℳ × T_p ℳ → ℝ
    ///
    /// Common metrics include:
    /// - **Canonical metric**: For embedded manifolds, inherit the Euclidean metric
    ///   - Sphere: ⟨u,v⟩_p = u^T v (standard Euclidean inner product)
    ///   - Stiefel: ⟨U,V⟩_X = trace(U^T V) (Frobenius inner product)
    /// - **Invariant metrics**: Metrics that preserve group symmetries
    /// - **Pullback metrics**: Induced from the ambient space
    ///
    /// # Mathematical Properties
    ///
    /// The Riemannian metric satisfies:
    /// 1. **Symmetry**: ⟨u,v⟩_p = ⟨v,u⟩_p
    /// 2. **Bilinearity**: ⟨αu₁ + βu₂,v⟩_p = α⟨u₁,v⟩_p + β⟨u₂,v⟩_p
    /// 3. **Positive definiteness**: ⟨u,u⟩_p > 0 for all u ≠ 0
    /// 4. **Smoothness**: The metric varies smoothly with the point p
    ///
    /// # Arguments
    ///
    /// * `point` - A point p ∈ ℳ on the manifold
    /// * `u` - First tangent vector u ∈ T_p ℳ
    /// * `v` - Second tangent vector v ∈ T_p ℳ
    ///
    /// # Returns
    ///
    /// The Riemannian inner product ⟨u,v⟩_p ∈ ℝ.
    ///
    /// # Errors
    ///
    /// Returns an error if `point` is not on the manifold or if `u` or `v`
    /// are not in the tangent space at `point`.
    fn inner_product(
        &self,
        point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T>;

    /// Computes the norm of a tangent vector.
    ///
    /// This is equivalent to sqrt(inner_product(point, v, v)).
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `vector` - A tangent vector
    ///
    /// # Returns
    ///
    /// The norm ||v||_g.
    fn norm(&self, point: &Self::Point, vector: &Self::TangentVector) -> Result<T> {
        self.inner_product(point, vector, vector)
            .map(|ip| <T as Float>::sqrt(ip))
    }

    /// Performs a retraction from the tangent space to the manifold.
    ///
    /// A retraction R_p: T_p ℳ → ℳ is a smooth mapping that provides a way to 
    /// "move" from a point p on the manifold in the direction of a tangent vector v.
    /// It serves as an efficient approximation to the exponential map.
    ///
    /// # Mathematical Definition
    ///
    /// A retraction R_p must satisfy:
    /// 1. **Identity property**: R_p(0) = p (staying at p with zero step)
    /// 2. **Tangent condition**: dR_p(0) = id_{T_p ℳ} (first-order agreement with exp)
    /// 3. **Manifold constraint**: R_p(v) ∈ ℳ for all v ∈ T_p ℳ
    ///
    /// # Common Retractions
    ///
    /// Different manifolds use different retraction strategies:
    /// - **Sphere**: R_p(v) = (p + v)/||p + v|| (projection retraction)
    /// - **Stiefel**: R_X(V) = qf(X + V) where qf is QR decomposition Q-factor
    /// - **SPD**: R_X(V) = X + V + ½VX⁻¹V (symmetric retraction)
    /// - **Grassmann**: Via lifting to Stiefel and projection
    ///
    /// # Computational Considerations
    ///
    /// - **Efficiency**: Retractions are typically much faster than exponential maps
    /// - **Stability**: Good retractions maintain numerical manifold constraints
    /// - **Order**: Higher-order retractions provide better approximation to exp_p
    ///
    /// # Arguments
    ///
    /// * `point` - A point p ∈ ℳ on the manifold
    /// * `tangent` - A tangent vector v ∈ T_p ℳ (direction and magnitude of step)
    /// * `result` - Pre-allocated output buffer for the retracted point
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `point` is not on the manifold
    /// - `tangent` is not in the tangent space at `point`
    /// - Numerical issues prevent computation (e.g., singularities)
    fn retract(&self, point: &Self::Point, tangent: &Self::TangentVector, result: &mut Self::Point, workspace: &mut Workspace<T>) -> Result<()>;

    /// Computes the inverse retraction (logarithmic map).
    ///
    /// This is the inverse of the retraction, mapping from the manifold
    /// back to the tangent space.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `other` - Another point on the manifold
    /// * `result` - Pre-allocated output buffer for the tangent vector
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Errors
    ///
    /// Returns an error if the logarithm is not well-defined (e.g., points
    /// are too far apart or at cut locus).
    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Converts the Euclidean gradient to the Riemannian gradient.
    ///
    /// Given the Euclidean gradient ∇f of a function f: M → R, this computes
    /// the Riemannian gradient grad f, which is the unique tangent vector
    /// satisfying g(grad f, v) = df(v) for all v in T_p M.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `euclidean_grad` - The Euclidean gradient at `point`
    /// * `result` - Pre-allocated output buffer for the Riemannian gradient
    /// * `workspace` - Pre-allocated workspace for temporary computations
    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Performs parallel transport of a vector along a retraction.
    ///
    /// Parallel transport moves a tangent vector from one point to another
    /// while preserving its "direction" in a manifold-specific sense.
    ///
    /// # Arguments
    ///
    /// * `from` - Starting point on the manifold
    /// * `to` - Ending point on the manifold
    /// * `vector` - Tangent vector at `from` to transport
    /// * `result` - Pre-allocated output buffer for the transported vector
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Default Implementation
    ///
    /// The default implementation uses vector transport by projection,
    /// which may not be true parallel transport but is often sufficient.
    fn parallel_transport(
        &self,
        _from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Default: vector transport by projection
        self.project_tangent(to, vector, result, workspace)
    }

    /// Generates a random point on the manifold.
    ///
    /// This is useful for testing and initialization.
    ///
    /// # Returns
    ///
    /// A random point uniformly distributed on the manifold (if possible).
    fn random_point(&self) -> Self::Point;

    /// Generates a random tangent vector at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `result` - Pre-allocated output buffer for the random tangent vector
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Returns
    ///
    /// A random tangent vector at `point` with unit norm.
    fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector, workspace: &mut Workspace<T>) -> Result<()>;

    /// Computes the geodesic distance between two points.
    ///
    /// # Arguments
    ///
    /// * `x` - First point on the manifold
    /// * `y` - Second point on the manifold
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Returns
    ///
    /// The geodesic distance d(x, y).
    ///
    /// # Default Implementation
    ///
    /// Uses the norm of the logarithmic map.
    fn distance(&self, x: &Self::Point, y: &Self::Point, workspace: &mut Workspace<T>) -> Result<T>;

    /// Checks if the manifold has a closed-form exponential map.
    ///
    /// Some manifolds (like spheres) have efficient closed-form exponential
    /// maps, while others require numerical approximation.
    fn has_exact_exp_log(&self) -> bool {
        false
    }

    /// Checks if the manifold has curvature.
    ///
    /// Flat manifolds (like Euclidean space) have zero curvature.
    fn is_flat(&self) -> bool {
        false
    }

    // ============================================================================
    // Vector Operations for Optimization
    // ============================================================================
    
    /// Scales a tangent vector by a scalar.
    ///
    /// Computes: result = scalar * tangent
    ///
    /// This operation is fundamental for optimization algorithms that need to
    /// scale gradients or search directions. For most manifolds, this is simply
    /// scalar multiplication, but some manifolds may require special handling.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold (for manifolds where metric depends on position)
    /// * `scalar` - The scalar factor
    /// * `tangent` - The tangent vector to scale
    /// * `result` - Pre-allocated output buffer for the scaled vector
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Errors
    ///
    /// Returns an error if the tangent vector is not in the tangent space.
    fn scale_tangent(
        &self,
        point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Adds two tangent vectors.
    ///
    /// Computes: result = v1 + v2
    ///
    /// For most manifolds embedded in Euclidean space, this is standard vector
    /// addition. However, the result must be in the tangent space, so projection
    /// may be necessary for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `v1` - First tangent vector
    /// * `v2` - Second tangent vector  
    /// * `result` - Pre-allocated output buffer for the sum
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Errors
    ///
    /// Returns an error if either vector is not in the tangent space.
    fn add_tangents(
        &self,
        point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>;

    /// Computes a linear combination of tangent vectors (axpy operation).
    ///
    /// Computes: result = y + alpha * x
    ///
    /// This is a fundamental operation for many optimization algorithms,
    /// combining scaling and addition in a single step for efficiency.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `alpha` - Scalar coefficient
    /// * `x` - Tangent vector to scale
    /// * `y` - Tangent vector to add
    /// * `result` - Pre-allocated output buffer for the result
    /// * `workspace` - Pre-allocated workspace for temporary computations
    ///
    /// # Default Implementation
    ///
    /// The default implementation uses scale_tangent and add_tangents, but
    /// specific manifolds may provide more efficient implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if either vector is not in the tangent space.
    fn axpy_tangent(
        &self,
        point: &Self::Point,
        alpha: T,
        x: &Self::TangentVector,
        y: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Default implementation: compute alpha * x, then add y
        let mut scaled_x = x.clone(); // Temporary, will use workspace buffer in optimized version
        self.scale_tangent(point, alpha, x, &mut scaled_x, workspace)?;
        self.add_tangents(point, &scaled_x, y, result, workspace)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_manifolds::TestEuclideanManifold;
    use crate::types::DVector;

    #[test]
    fn test_manifold_basic_properties() {
        let manifold = TestEuclideanManifold::new(10);
        assert_eq!(<TestEuclideanManifold as Manifold<f64>>::name(&manifold), "TestEuclidean");
        assert_eq!(<TestEuclideanManifold as Manifold<f64>>::dimension(&manifold), 10);
        assert!(!<TestEuclideanManifold as Manifold<f64>>::has_exact_exp_log(&manifold));
        assert!(!<TestEuclideanManifold as Manifold<f64>>::is_flat(&manifold));
    }

    #[test]
    fn test_default_implementations() {
        let manifold = TestEuclideanManifold::new(3);
        let point = DVector::zeros(3);
        let vector = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let mut workspace = Workspace::new();

        // Test norm (uses inner_product)
        let norm = manifold.norm(&point, &vector).unwrap();
        assert_eq!(norm, 1.0);

        // Test parallel transport (uses project_tangent by default)
        let mut transported = DVector::zeros(3);
        manifold
            .parallel_transport(&point, &point, &vector, &mut transported, &mut workspace)
            .unwrap();
        assert_eq!(transported, vector);

        // Test distance (uses inverse_retract and norm)
        let other_point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let dist = manifold.distance(&point, &other_point, &mut workspace).unwrap();
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_manifold_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TestEuclideanManifold>();
    }
}
