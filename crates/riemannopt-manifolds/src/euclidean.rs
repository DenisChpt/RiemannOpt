//! # Euclidean Manifold ℝ^n
//!
//! The Euclidean manifold is the standard n-dimensional vector space equipped
//! with the usual Euclidean metric. While trivial as a manifold, it provides
//! a baseline for optimization algorithms and enables unconstrained optimization
//! within the Riemannian framework.
//!
//! ## Mathematical Definition
//!
//! The Euclidean manifold is:
//! ```text
//! ℝ^n = {x = (x₁, ..., xₙ) : xᵢ ∈ ℝ}
//! ```
//!
//! ## Riemannian Structure
//!
//! All Riemannian operations are trivial:
//! - **Tangent space**: TₓM = ℝ^n for all x
//! - **Metric**: ⟨u, v⟩ₓ = u^T v (standard inner product)
//! - **Projection**: Pₓ(v) = v (identity)
//! - **Retraction**: Rₓ(v) = x + v (addition)
//! - **Exponential map**: expₓ(v) = x + v
//! - **Logarithmic map**: logₓ(y) = y - x
//! - **Parallel transport**: Γₓ→y(v) = v (identity)
//!
//! ## Properties
//!
//! - **Dimension**: dim(ℝ^n) = n
//! - **Curvature**: Flat (zero curvature everywhere)
//! - **Completeness**: Complete metric space
//! - **Simply connected**: Yes
//! - **Geodesics**: Straight lines
//!
//! ## Applications
//!
//! 1. **Baseline comparison**: Testing optimization algorithms
//! 2. **Unconstrained optimization**: Standard optimization problems
//! 3. **Component of product manifolds**: Combined with constrained manifolds
//! 4. **Regularization**: Euclidean penalty terms
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Euclidean;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::VectorOps;
//!
//! // Create 10-dimensional Euclidean space
//! let manifold = Euclidean::<f64>::new(10)?;
//!
//! // Random point
//! let mut x = riemannopt_core::linalg::Vec::<f64>::zeros(10);
//! manifold.random_point(&mut x)?;
//!
//! // Random tangent vector
//! let mut v = riemannopt_core::linalg::Vec::<f64>::zeros(10);
//! manifold.random_tangent(&x, &mut v)?;
//!
//! // Retraction is just addition
//! let mut y = riemannopt_core::linalg::Vec::<f64>::zeros(10);
//! manifold.retract(&x, &v, &mut y, &mut ())?;
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
	error::{ManifoldError, Result},
	linalg::{self, LinAlgBackend, VectorOps},
	manifold::Manifold,
	types::Scalar,
};

/// The Euclidean manifold ℝ^n.
///
/// This structure represents the standard n-dimensional Euclidean space
/// with trivial Riemannian operations.
#[derive(Debug, Clone)]
pub struct Euclidean<T: Scalar> {
	/// Dimension of the space
	n: usize,
	_phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Euclidean<T> {
	/// Create a new Euclidean manifold of dimension n.
	///
	/// # Arguments
	///
	/// * `n` - Dimension of the space (must be positive)
	///
	/// # Errors
	///
	/// Returns an error if n is 0.
	pub fn new(n: usize) -> Result<Self> {
		if n == 0 {
			return Err(ManifoldError::dimension_mismatch("positive dimension", n));
		}
		Ok(Self {
			n,
			_phantom: std::marker::PhantomData,
		})
	}

	/// Get the dimension of the space.
	pub fn dim(&self) -> usize {
		self.n
	}
}

impl<T> Manifold<T> for Euclidean<T>
where
	T: Scalar + Float,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;
	type Workspace = ();

	fn name(&self) -> &str {
		"Euclidean"
	}

	fn dimension(&self) -> usize {
		self.n
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Projection is identity in Euclidean space
		result.copy_from(point);
	}

	fn project_tangent(
		&self,
		_point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// Projection is identity in Euclidean space
		result.copy_from(tangent);
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> Result<T> {
		// Standard inner product
		Ok(VectorOps::dot(u, v))
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) -> Result<()> {
		// Retraction is addition: result = point + tangent
		result.copy_from(point);
		result.add_assign(tangent);
		Ok(())
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		_to: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// Parallel transport is identity in flat space
		result.copy_from(tangent);
		Ok(())
	}

	fn inverse_retract(
		&self,
		x: &Self::Point,
		y: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// Inverse retraction is subtraction: result = y - x
		result.copy_from(y);
		result.sub_assign(x);
		Ok(())
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut rng = rand::rng();

		// Ensure result has correct size
		if VectorOps::len(result) != self.n {
			*result = VectorOps::zeros(self.n);
		}

		// Generate random point from standard normal distribution
		for i in 0..self.n {
			let sample: f64 = StandardNormal.sample(&mut rng);
			*result.get_mut(i) = T::from(sample).unwrap_or_else(T::zero);
		}

		Ok(())
	}

	fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		// Random tangent is just a random vector
		self.random_point(result)
	}

	fn is_point_on_manifold(&self, point: &Self::Point, _tol: T) -> bool {
		// Check dimension
		VectorOps::len(point) == self.n
	}

	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_tol: T,
	) -> bool {
		// Check dimension
		VectorOps::len(vector) == self.n
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		// ‖y - x‖² = ‖y‖² + ‖x‖² - 2⟨x,y⟩  (zero alloc)
		let xx = VectorOps::dot(x, x);
		let yy = VectorOps::dot(y, y);
		let xy = VectorOps::dot(x, y);
		let dist_sq = xx + yy - (T::one() + T::one()) * xy;
		// Guard against negative values from floating-point errors
		Ok(<T as Float>::sqrt(<T as Float>::max(dist_sq, T::zero())))
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Standard scalar multiplication
		result.copy_from(tangent);
		result.scale_mut(scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		_point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Standard vector addition
		result.copy_from(v1);
		result.add_assign(v2);
		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		_point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		// In Euclidean space, Euclidean and Riemannian gradients are the same
		result.copy_from(euclidean_grad);
		Ok(())
	}
}
