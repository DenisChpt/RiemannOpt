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

use crate::{error::Result, types::Scalar};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use num_traits::Float;
use std::fmt::Debug;

/// Type alias for manifold points and tangent vectors.
pub type Point<T, D> = OVector<T, D>;
pub type TangentVector<T, D> = OVector<T, D>;

/// Trait for Riemannian manifolds.
///
/// This trait defines the interface that all manifolds must implement to be used
/// with Riemannian optimization algorithms. It provides methods for:
/// - Checking manifold membership
/// - Computing tangent space projections
/// - Performing retractions and their inverses
/// - Computing Riemannian metrics and gradients
/// - Parallel transport of vectors
///
/// # Type Parameters
///
/// - `T`: The scalar type (f32 or f64)
/// - `D`: The dimension of the manifold's representation
///
/// # Example
///
/// ```rust,ignore
/// use riemannopt_core::prelude::*;
/// use nalgebra::Const;
///
/// // Example implementation for the unit sphere
/// struct Sphere<T: Scalar, const N: usize>;
///
/// impl<T: Scalar, const N: usize> Manifold<T, Const<N>> for Sphere<T, N> {
///     fn name(&self) -> &str {
///         "Sphere"
///     }
///     
///     fn dimension(&self) -> usize {
///         N - 1  // Sphere S^{n-1} embedded in R^n
///     }
///     
///     fn is_point_on_manifold(&self, point: &Point<T, Const<N>>, tol: T) -> bool {
///         (point.norm_squared() - T::one()).abs() < tol
///     }
///     
///     // ... other required methods
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
    /// This method takes a point in the ambient space and returns the closest
    /// point on the manifold (in the Euclidean sense).
    ///
    /// # Arguments
    ///
    /// * `point` - The point to project
    ///
    /// # Returns
    ///
    /// The projected point on the manifold.
    fn project_point(&self, point: &Point<T, D>) -> Point<T, D>;

    /// Projects a vector onto the tangent space at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `vector` - The vector to project
    ///
    /// # Returns
    ///
    /// The projection of `vector` onto T_point M.
    ///
    /// # Errors
    ///
    /// Returns an error if `point` is not on the manifold.
    fn project_tangent(
        &self,
        point: &Point<T, D>,
        vector: &TangentVector<T, D>,
    ) -> Result<TangentVector<T, D>>;

    /// Computes the Riemannian inner product between two tangent vectors.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `u` - First tangent vector
    /// * `v` - Second tangent vector
    ///
    /// # Returns
    ///
    /// The inner product g_point(u, v).
    ///
    /// # Errors
    ///
    /// Returns an error if `point` is not on the manifold or if `u` or `v`
    /// are not in the tangent space.
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
    /// A retraction at point p is a smooth mapping R_p: T_p M → M such that:
    /// - R_p(0) = p
    /// - dR_p(0) = identity on T_p M
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `tangent` - A tangent vector at `point`
    ///
    /// # Returns
    ///
    /// A new point on the manifold.
    ///
    /// # Errors
    ///
    /// Returns an error if the inputs are invalid.
    fn retract(&self, point: &Point<T, D>, tangent: &TangentVector<T, D>) -> Result<Point<T, D>>;

    /// Computes the inverse retraction (logarithmic map).
    ///
    /// This is the inverse of the retraction, mapping from the manifold
    /// back to the tangent space.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `other` - Another point on the manifold
    ///
    /// # Returns
    ///
    /// A tangent vector v such that retract(point, v) ≈ other.
    ///
    /// # Errors
    ///
    /// Returns an error if the logarithm is not well-defined (e.g., points
    /// are too far apart or at cut locus).
    fn inverse_retract(
        &self,
        point: &Point<T, D>,
        other: &Point<T, D>,
    ) -> Result<TangentVector<T, D>>;

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
    ///
    /// # Returns
    ///
    /// The Riemannian gradient.
    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Point<T, D>,
        euclidean_grad: &TangentVector<T, D>,
    ) -> Result<TangentVector<T, D>>;

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
    ///
    /// # Returns
    ///
    /// The transported vector in T_to M.
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
    ) -> Result<TangentVector<T, D>> {
        // Default: vector transport by projection
        self.project_tangent(to, vector)
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
    ///
    /// # Returns
    ///
    /// A random tangent vector at `point` with unit norm.
    fn random_tangent(&self, point: &Point<T, D>) -> Result<TangentVector<T, D>>;

    /// Computes the geodesic distance between two points.
    ///
    /// # Arguments
    ///
    /// * `x` - First point on the manifold
    /// * `y` - Second point on the manifold
    ///
    /// # Returns
    ///
    /// The geodesic distance d(x, y).
    ///
    /// # Default Implementation
    ///
    /// Uses the norm of the logarithmic map.
    fn distance(&self, x: &Point<T, D>, y: &Point<T, D>) -> Result<T> {
        let log = self.inverse_retract(x, y)?;
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

        // Test norm (uses inner_product)
        let norm = manifold.norm(&point, &vector).unwrap();
        assert_eq!(norm, 1.0);

        // Test parallel transport (uses project_tangent by default)
        let transported = manifold
            .parallel_transport(&point, &point, &vector)
            .unwrap();
        assert_eq!(transported, vector);

        // Test distance (uses inverse_retract and norm)
        let other_point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let dist = manifold.distance(&point, &other_point).unwrap();
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_manifold_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TestEuclideanManifold>();
    }
}
