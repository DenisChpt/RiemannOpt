//! Core manifold trait and concrete implementations.
//!
//! This module defines the fundamental [`Manifold`] trait and provides
//! implementations for common Riemannian manifolds.

pub mod euclidean;
pub mod fixed_rank;
pub mod grassmann;
pub mod hyperbolic;
pub mod oblique;
pub mod product;
pub mod psd_cone;
pub mod sphere;
pub mod spd;
pub mod stiefel;
pub mod utils;

// Re-export manifold types at module level
pub use euclidean::Euclidean;
pub use fixed_rank::FixedRank;
pub use grassmann::Grassmann;
pub use hyperbolic::Hyperbolic;
pub use oblique::Oblique;
pub use product::Product;
pub use psd_cone::PSDCone;
pub use sphere::Sphere;
pub use spd::SPD;
pub use stiefel::Stiefel;

use crate::types::Scalar;
use num_traits::Float;
use std::fmt::Debug;

/// Trait for Riemannian manifolds.
///
/// # Type Parameters
///
/// - `T`: The scalar type (f32 or f64)
/// - `D`: The dimension type of the manifold's representation
///
pub trait Manifold<T: Scalar>: Debug + Send + Sync {
	/// The type of data for a point (e.g., linalg::Vec<T> or linalg::Mat<T>).
	type Point: Clone + Debug + Send + Sync;
	/// The type of data for a tangent vector.
	type TangentVector: Clone + Debug + Send + Sync;
	/// Pre-allocated workspace for hot-path operations.
	///
	/// Manifolds that need temporary buffers define a concrete
	/// workspace type here. Manifolds with no extra buffer needs use `()`,
	/// which is a zero-sized type — the compiler eliminates it entirely.
	type Workspace: Default + Send + Sync;

	/// Creates a workspace sized for the given prototype point.
	///
	/// The default implementation returns `Default::default()`, which is
	/// a no-op for `()`.
	#[inline]
	fn create_workspace(&self, _proto_point: &Self::Point) -> Self::Workspace {
		Self::Workspace::default()
	}
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
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point);

	/// Projects a vector onto the tangent space at a given point.
	///
	/// The tangent space T_p ℳ at point p is the linear space of all possible
	/// directions of motion on the manifold at p. This method computes the
	/// orthogonal projection onto this space:
	///
	/// P_p: ℝⁿ → T_p ℳ
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
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	);

	/// Computes the Riemannian inner product between two tangent vectors.
	///
	/// The Riemannian metric g provides an inner product on each tangent space:
	/// g_p: T_p ℳ × T_p ℳ → ℝ
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
	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T;

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
	#[inline]
	fn norm(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T {
		<T as Float>::sqrt(self.inner_product(point, vector, vector, ws))
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
	/// # Arguments
	///
	/// * `point` - A point p ∈ ℳ on the manifold
	/// * `tangent` - A tangent vector v ∈ T_p ℳ (direction and magnitude of step)
	/// * `result` - Pre-allocated output buffer for the retracted point
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	);

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
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	);

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
	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.project_tangent(point, euclidean_grad, result, ws);
	}

	/// Converts an Euclidean Hessian-vector product to the Riemannian Hessian-vector product.
	///
	/// Given the Euclidean gradient ∇f, the Euclidean Hessian-vector product ∇²f[ξ],
	/// and a tangent vector ξ ∈ T_p ℳ, computes the Riemannian Hessian-vector product
	/// Hess f[ξ] which accounts for the manifold curvature.
	///
	/// # Default Implementation
	///
	/// Projects the Euclidean Hessian-vector product onto the tangent space.
	/// This is exact for Euclidean space and a reasonable approximation for
	/// manifolds where the curvature correction is small. Manifolds with
	/// significant curvature (Grassmann, Stiefel, Sphere) should override this.
	#[inline]
	fn euclidean_to_riemannian_hessian(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		euclidean_hvp: &Self::TangentVector,
		tangent_vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Default: just project the Euclidean HVP onto the tangent space
		let _ = euclidean_grad;
		let _ = tangent_vector;
		self.project_tangent(point, euclidean_hvp, result, ws);
	}

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
	///
	/// # Default Implementation
	///
	/// The default implementation uses vector transport by projection,
	/// which may not be true parallel transport but is often sufficient.
	#[inline]
	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Default: vector transport by projection
		self.project_tangent(to, vector, result, ws);
	}

	/// Generates a random point on the manifold.
	///
	/// This is useful for testing and initialization.
	///
	/// # Arguments
	///
	/// * `result` - Pre-allocated output buffer for the random point
	fn random_point(&self, result: &mut Self::Point);

	/// Generates a random tangent vector at a given point.
	///
	/// # Arguments
	///
	/// * `point` - A point on the manifold
	/// * `result` - Pre-allocated output buffer for the random tangent vector
	///
	/// # Returns
	///
	/// A random tangent vector at `point` with unit norm.
	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector);

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
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T;

	/// Checks if the manifold has a closed-form exponential map.
	///
	/// Some manifolds (like spheres) have efficient closed-form exponential
	/// maps, while others require numerical approximation.
	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		false
	}

	/// Checks if the manifold has curvature.
	///
	/// Flat manifolds (like Euclidean space) have zero curvature.
	#[inline]
	fn is_flat(&self) -> bool {
		false
	}

	// ============================================================================
	// Vector Operations for Optimization
	// ============================================================================

	/// Scales a tangent vector in-place: v ← scalar · v
	///
	/// Pure linear algebra in the ambient representation — the base point
	/// is irrelevant because T_p ℳ is a vector subspace of ℝⁿ.
	///
	/// # Arguments
	///
	/// * `scalar` - The scalar factor
	/// * `v` - The tangent vector to scale in-place
	fn scale_tangent(&self, scalar: T, v: &mut Self::TangentVector);

	/// Adds a tangent vector in-place: v1 ← v1 + v2.
	///
	/// Pure linear algebra in the ambient representation — no re-projection
	/// needed because the sum of two tangent vectors in the same T_p ℳ
	/// remains in T_p ℳ (it is a vector space).
	///
	/// # Arguments
	///
	/// * `v1` - Tangent vector, accumulated in-place
	/// * `v2` - Tangent vector to add
	fn add_tangents(&self, v1: &mut Self::TangentVector, v2: &Self::TangentVector);

	/// Computes in-place: y ← y + alpha · x
	///
	/// No temporary buffer needed — this is a fused scale-and-accumulate
	/// that any backend can execute in a single pass.
	fn axpy_tangent(&self, alpha: T, x: &Self::TangentVector, y: &mut Self::TangentVector);

	/// Allocates a new uninitialized point sized for this manifold.
	fn allocate_point(&self) -> Self::Point;

	/// Allocates a new uninitialized tangent vector sized for this manifold.
	fn allocate_tangent(&self) -> Self::TangentVector;

	// ════════════════════════════════════════════════════════════════════════
	// Copy operations (for solvers that swap buffers)
	// ════════════════════════════════════════════════════════════════════════

	/// Copies `src` into `dst` in-place (overwrites dst contents).
	///
	/// Default uses `Clone::clone_from`. Override for zero-alloc backends
	/// where the underlying storage supports in-place memcpy.
	#[inline]
	fn copy_point(&self, dst: &mut Self::Point, src: &Self::Point) {
		dst.clone_from(src);
	}

	/// Copies `src` into `dst` in-place (overwrites dst contents).
	#[inline]
	fn copy_tangent(&self, dst: &mut Self::TangentVector, src: &Self::TangentVector) {
		dst.clone_from(src);
	}
}
