//! Core manifold trait and associated types.
//!
//! This module defines the fundamental `Manifold` trait that all Riemannian
//! manifolds must implement. A manifold is a topological space that locally
//! resembles Euclidean space and can be equipped with a Riemannian metric.
//!
//! # Mathematical Background
//!
//! A Riemannian manifold (M, g) consists of:
//! - A smooth manifold M
//! - A Riemannian metric g that assigns an inner product to each tangent space
//!
//! Key concepts:
//! - **Tangent space**: T_p M is the linear approximation of M at point p
//! - **Retraction**: A smooth map R_p: T_p M → M that approximates the exponential map
//! - **Riemannian gradient**: The unique vector in T_p M representing the derivative
//! - **Parallel transport**: Moving vectors along curves while preserving angles

use crate::{error::Result, types::Scalar, memory::workspace::Workspace};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use num_traits::Float;
use std::fmt::Debug;

/// Type alias for manifold points and tangent vectors.
pub type Point<T, D> = OVector<T, D>;
pub type TangentVector<T, D> = OVector<T, D>;

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
pub trait Manifold<T, D>: Debug + Send + Sync
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Returns a human-readable name for the manifold.
    fn name(&self) -> &str;

    /// Returns the intrinsic dimension of the manifold.
    ///
    /// For example, the sphere S^{n-1} embedded in R^n has dimension n-1.
    fn dimension(&self) -> usize;

    /// Returns the dimension of the ambient space.
    ///
    /// For embedded manifolds, this is the dimension of the space in which
    /// the manifold is embedded.
    /// 
    /// Returns 0 for dynamic dimensions where the size cannot be determined at compile time.
    fn ambient_dimension(&self) -> usize {
        D::try_to_usize().unwrap_or(0)
    }

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
    fn is_point_on_manifold(&self, point: &Point<T, D>, tol: T) -> bool;

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
        point: &Point<T, D>,
        vector: &TangentVector<T, D>,
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
    fn project_point(&self, point: &Point<T, D>, result: &mut Point<T, D>, workspace: &mut Workspace<T>);

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
        point: &Point<T, D>,
        vector: &TangentVector<T, D>,
        result: &mut TangentVector<T, D>,
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
        point: &Point<T, D>,
        u: &TangentVector<T, D>,
        v: &TangentVector<T, D>,
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
    fn norm(&self, point: &Point<T, D>, vector: &TangentVector<T, D>) -> Result<T> {
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
    fn retract(&self, point: &Point<T, D>, tangent: &TangentVector<T, D>, result: &mut Point<T, D>, workspace: &mut Workspace<T>) -> Result<()>;

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
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVector<T, D>,
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
        point: &Point<T, D>,
        euclidean_grad: &TangentVector<T, D>,
        result: &mut TangentVector<T, D>,
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
        _from: &Point<T, D>,
        to: &Point<T, D>,
        vector: &TangentVector<T, D>,
        result: &mut TangentVector<T, D>,
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
    fn random_point(&self) -> Point<T, D>;

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
    fn random_tangent(&self, point: &Point<T, D>, result: &mut TangentVector<T, D>, workspace: &mut Workspace<T>) -> Result<()>;

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
    fn distance(&self, x: &Point<T, D>, y: &Point<T, D>, workspace: &mut Workspace<T>) -> Result<T> {
        let mut log = TangentVector::<T, D>::zeros_generic(x.shape_generic().0, nalgebra::Const::<1>);
        self.inverse_retract(x, y, &mut log, workspace)?;
        self.norm(x, &log)
    }

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

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_manifolds::TestEuclideanManifold;
    use crate::types::DVector;

    #[test]
    fn test_manifold_basic_properties() {
        let manifold = TestEuclideanManifold::new(10);
        assert_eq!(<TestEuclideanManifold as Manifold<f64, _>>::name(&manifold), "TestEuclidean");
        assert_eq!(<TestEuclideanManifold as Manifold<f64, _>>::dimension(&manifold), 10);
        assert!(!<TestEuclideanManifold as Manifold<f64, _>>::has_exact_exp_log(&manifold));
        assert!(!<TestEuclideanManifold as Manifold<f64, _>>::is_flat(&manifold));
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
