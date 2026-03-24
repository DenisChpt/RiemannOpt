//! # Static Product Manifold M₁ × M₂
//!
//! This module provides a generic product manifold implementation with static
//! dispatch for optimal performance when combining two known manifold types.
//!
//! ## Mathematical Definition
//!
//! For manifolds M₁ and M₂, the product manifold is:
//! ```text
//! M = M₁ × M₂ = {(x₁, x₂) : x₁ ∈ M₁, x₂ ∈ M₂}
//! ```
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space decomposes as:
//! ```text
//! T_{(x₁,x₂)} M = T_{x₁} M₁ × T_{x₂} M₂
//! ```
//!
//! ### Riemannian Metric
//! The product metric is:
//! ```text
//! g_{(x₁,x₂)}((u₁,u₂), (v₁,v₂)) = g₁(u₁, v₁) + g₂(u₂, v₂)
//! ```
//!
//! ### Geodesics
//! Geodesics are component-wise:
//! ```text
//! γ(t) = (γ₁(t), γ₂(t))
//! ```
//!
//! ## Distance Formula
//!
//! Using the product metric:
//! ```text
//! d²((x₁,x₂), (y₁,y₂)) = d₁²(x₁, y₁) + d₂²(x₂, y₂)
//! ```
//!
//! ## Performance Benefits
//!
//! Static dispatch provides:
//! - **Zero-cost abstractions**: No virtual function calls
//! - **Inline optimization**: Compiler can inline all operations
//! - **Type safety**: Compile-time verification of manifold compatibility
//! - **Better performance**: Typically 2-5x faster than dynamic dispatch
//!
//! ## Applications
//!
//! 1. **Optimization**: Block coordinate methods
//! 2. **Robotics**: Position × Orientation spaces (ℝ³ × SO(3))
//! 3. **Machine Learning**: Multi-modal representations
//! 4. **Computer Vision**: Shape × Appearance models
//! 5. **Signal Processing**: Amplitude × Phase manifolds
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::{ProductStatic, Sphere};
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{self, VectorOps};
//!
//! // Create S² × S³ statically
//! let sphere1 = Sphere::<f64>::new(3)?;
//! let sphere2 = Sphere::<f64>::new(4)?;
//! let product = ProductStatic::new(sphere1, sphere2);
//!
//! // Operations are fully type-safe and optimized
//! let mut x = linalg::Vec::<f64>::zeros(7);
//! product.random_point(&mut x)?;
//!
//! // Access components directly
//! let (x1, x2) = product.split_point(&x)?;
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use num_traits::Float;
use std::marker::PhantomData;

use riemannopt_core::{
	error::{ManifoldError, Result},
	linalg::{self, LinAlgBackend, VectorOps},
	manifold::Manifold,
	types::Scalar,
};

/// A static product manifold M₁ × M₂ with compile-time dispatch.
///
/// This structure provides optimal performance for combining two known
/// manifold types through static dispatch and inline optimization.
///
/// # Type Parameters
///
/// - `T`: Scalar type (f32 or f64)
/// - `M1`: First manifold type
/// - `M2`: Second manifold type
///
/// # Invariants
///
/// - Both manifolds must use the same scalar type
/// - Points and tangent vectors are concatenated: [x₁; x₂]
/// - Operations are performed component-wise
#[derive(Debug, Clone)]
pub struct ProductStatic<T, M1, M2>
where
	T: Scalar,
	M1: Manifold<T>,
	M2: Manifold<T>,
{
	/// First component manifold
	pub manifold1: M1,
	/// Second component manifold
	pub manifold2: M2,
	/// Dimension of first manifold's representation
	dim1: usize,
	/// Dimension of second manifold's representation
	dim2: usize,
	/// Total dimension
	total_dim: usize,
	_phantom: PhantomData<T>,
}

impl<T, M1, M2> ProductStatic<T, M1, M2>
where
	T: Scalar,
	linalg::DefaultBackend: LinAlgBackend<T>,
	M1: Manifold<T, Point = linalg::Vec<T>, TangentVector = linalg::Vec<T>>,
	M2: Manifold<T, Point = linalg::Vec<T>, TangentVector = linalg::Vec<T>>,
{
	/// Creates a new static product manifold M₁ × M₂.
	///
	/// # Arguments
	///
	/// * `manifold1` - First component manifold
	/// * `manifold2` - Second component manifold
	///
	/// # Returns
	///
	/// A product manifold with combined operations.
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::{ProductStatic, Sphere, SPD};
	/// let sphere = Sphere::<f64>::new(3).unwrap();
	/// let sphere2 = Sphere::<f64>::new(4).unwrap();
	/// let product = ProductStatic::new(sphere, sphere2);
	/// ```
	pub fn new(manifold1: M1, manifold2: M2) -> Self {
		// Get dimensions by creating test points
		let mut test1 = VectorOps::zeros(0);
		let mut test2 = VectorOps::zeros(0);
		manifold1.random_point(&mut test1).unwrap();
		manifold2.random_point(&mut test2).unwrap();
		let dim1 = VectorOps::len(&test1);
		let dim2 = VectorOps::len(&test2);
		let total_dim = dim1 + dim2;

		Self {
			manifold1,
			manifold2,
			dim1,
			dim2,
			total_dim,
			_phantom: PhantomData,
		}
	}

	/// Returns the dimensions of both component manifolds.
	#[inline]
	pub fn component_dimensions(&self) -> (usize, usize) {
		(self.dim1, self.dim2)
	}

	/// Returns references to the component manifolds.
	#[inline]
	pub fn components(&self) -> (&M1, &M2) {
		(&self.manifold1, &self.manifold2)
	}

	/// Returns mutable references to the component manifolds.
	#[inline]
	pub fn components_mut(&mut self) -> (&mut M1, &mut M2) {
		(&mut self.manifold1, &mut self.manifold2)
	}

	/// Splits a product space vector into component vectors.
	///
	/// # Arguments
	///
	/// * `vector` - Combined vector from product space
	///
	/// # Returns
	///
	/// Tuple of (component1, component2) vectors.
	///
	/// # Errors
	///
	/// Returns error if vector dimension doesn't match.
	pub fn split_vector(
		&self,
		vector: &linalg::Vec<T>,
	) -> Result<(linalg::Vec<T>, linalg::Vec<T>)> {
		if VectorOps::len(vector) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(vector),
			));
		}

		let comp1 =
			<linalg::Vec<T> as VectorOps<T>>::from_fn(self.dim1, |i| VectorOps::get(vector, i));
		let comp2 = <linalg::Vec<T> as VectorOps<T>>::from_fn(self.dim2, |i| {
			VectorOps::get(vector, self.dim1 + i)
		});

		Ok((comp1, comp2))
	}

	/// Combines component vectors into a product space vector.
	///
	/// # Arguments
	///
	/// * `comp1` - First component vector
	/// * `comp2` - Second component vector
	///
	/// # Returns
	///
	/// Combined vector in product space.
	///
	/// # Errors
	///
	/// Returns error if component dimensions don't match.
	pub fn combine_vectors(
		&self,
		comp1: &linalg::Vec<T>,
		comp2: &linalg::Vec<T>,
	) -> Result<linalg::Vec<T>> {
		if VectorOps::len(comp1) != self.dim1 {
			return Err(ManifoldError::dimension_mismatch(
				self.dim1,
				VectorOps::len(comp1),
			));
		}
		if VectorOps::len(comp2) != self.dim2 {
			return Err(ManifoldError::dimension_mismatch(
				self.dim2,
				VectorOps::len(comp2),
			));
		}

		let mut combined = <linalg::Vec<T> as VectorOps<T>>::zeros(self.total_dim);
		for i in 0..self.dim1 {
			*VectorOps::get_mut(&mut combined, i) = VectorOps::get(comp1, i);
		}
		for i in 0..self.dim2 {
			*VectorOps::get_mut(&mut combined, self.dim1 + i) = VectorOps::get(comp2, i);
		}

		Ok(combined)
	}

	/// Splits a product space vector into components using workspace.
	///
	/// This version avoids allocations by using pre-allocated buffers.
	pub fn split_vector_mut(
		&self,
		vector: &linalg::Vec<T>,
		comp1: &mut linalg::Vec<T>,
		comp2: &mut linalg::Vec<T>,
	) -> Result<()> {
		if VectorOps::len(vector) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(vector),
			));
		}

		// Ensure output vectors have correct size
		if VectorOps::len(comp1) != self.dim1 {
			*comp1 = VectorOps::zeros(self.dim1);
		}
		if VectorOps::len(comp2) != self.dim2 {
			*comp2 = VectorOps::zeros(self.dim2);
		}

		for i in 0..self.dim1 {
			*VectorOps::get_mut(comp1, i) = VectorOps::get(vector, i);
		}
		for i in 0..self.dim2 {
			*VectorOps::get_mut(comp2, i) = VectorOps::get(vector, self.dim1 + i);
		}

		Ok(())
	}

	/// Combines component vectors into result using workspace.
	///
	/// This version avoids allocations by using a pre-allocated buffer.
	pub fn combine_vectors_mut(
		&self,
		comp1: &linalg::Vec<T>,
		comp2: &linalg::Vec<T>,
		result: &mut linalg::Vec<T>,
	) -> Result<()> {
		if VectorOps::len(comp1) != self.dim1 {
			return Err(ManifoldError::dimension_mismatch(
				self.dim1,
				VectorOps::len(comp1),
			));
		}
		if VectorOps::len(comp2) != self.dim2 {
			return Err(ManifoldError::dimension_mismatch(
				self.dim2,
				VectorOps::len(comp2),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		for i in 0..self.dim1 {
			*VectorOps::get_mut(result, i) = VectorOps::get(comp1, i);
		}
		for i in 0..self.dim2 {
			*VectorOps::get_mut(result, self.dim1 + i) = VectorOps::get(comp2, i);
		}

		Ok(())
	}

	/// Splits a point and returns components.
	///
	/// Convenience method for splitting points.
	#[inline]
	pub fn split_point(&self, point: &linalg::Vec<T>) -> Result<(linalg::Vec<T>, linalg::Vec<T>)> {
		self.split_vector(point)
	}

	/// Combines point components.
	///
	/// Convenience method for combining points.
	#[inline]
	pub fn combine_points(
		&self,
		p1: &linalg::Vec<T>,
		p2: &linalg::Vec<T>,
	) -> Result<linalg::Vec<T>> {
		self.combine_vectors(p1, p2)
	}
}

impl<T, M1, M2> Manifold<T> for ProductStatic<T, M1, M2>
where
	T: Scalar,
	linalg::DefaultBackend: LinAlgBackend<T>,
	M1: Manifold<T, Point = linalg::Vec<T>, TangentVector = linalg::Vec<T>>,
	M2: Manifold<T, Point = linalg::Vec<T>, TangentVector = linalg::Vec<T>>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;

	fn name(&self) -> &str {
		"ProductStatic"
	}

	fn dimension(&self) -> usize {
		self.manifold1.dimension() + self.manifold2.dimension()
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if VectorOps::len(point) != self.total_dim {
			return false;
		}

		match self.split_vector(point) {
			Ok((p1, p2)) => {
				self.manifold1.is_point_on_manifold(&p1, tol)
					&& self.manifold2.is_point_on_manifold(&p2, tol)
			}
			Err(_) => false,
		}
	}

	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		if VectorOps::len(point) != self.total_dim || VectorOps::len(vector) != self.total_dim {
			return false;
		}

		match (self.split_vector(point), self.split_vector(vector)) {
			(Ok((p1, p2)), Ok((v1, v2))) => {
				self.manifold1.is_vector_in_tangent_space(&p1, &v1, tol)
					&& self.manifold2.is_vector_in_tangent_space(&p2, &v2, tol)
			}
			_ => false,
		}
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		// Handle dimension mismatch by padding or truncating
		let padded_point = if VectorOps::len(point) != self.total_dim {
			let mut p = <linalg::Vec<T> as VectorOps<T>>::zeros(self.total_dim);
			let copy_len = VectorOps::len(point).min(self.total_dim);
			for i in 0..copy_len {
				*VectorOps::get_mut(&mut p, i) = VectorOps::get(point, i);
			}
			p
		} else {
			point.clone()
		};

		// Split and project
		if let Ok((p1, p2)) = self.split_vector(&padded_point) {
			let mut proj1 = p1.clone();
			let mut proj2 = p2.clone();

			self.manifold1.project_point(&p1, &mut proj1);
			self.manifold2.project_point(&p2, &mut proj2);

			// Combine results
			let _ = self.combine_vectors_mut(&proj1, &proj2, result);
		}
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		if VectorOps::len(point) != self.total_dim || VectorOps::len(vector) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point).max(VectorOps::len(vector)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (p1, p2) = self.split_vector(point)?;
		let (v1, v2) = self.split_vector(vector)?;

		let mut proj1 = v1.clone();
		let mut proj2 = v2.clone();

		self.manifold1.project_tangent(&p1, &v1, &mut proj1)?;
		self.manifold2.project_tangent(&p2, &v2, &mut proj2)?;

		self.combine_vectors_mut(&proj1, &proj2, result)?;
		Ok(())
	}

	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		if VectorOps::len(point) != self.total_dim
			|| VectorOps::len(u) != self.total_dim
			|| VectorOps::len(v) != self.total_dim
		{
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point)
					.max(VectorOps::len(u))
					.max(VectorOps::len(v)),
			));
		}

		let (p1, p2) = self.split_vector(point)?;
		let (u1, u2) = self.split_vector(u)?;
		let (v1, v2) = self.split_vector(v)?;

		let ip1 = self.manifold1.inner_product(&p1, &u1, &v1)?;
		let ip2 = self.manifold2.inner_product(&p2, &u2, &v2)?;

		Ok(ip1 + ip2)
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		if VectorOps::len(point) != self.total_dim || VectorOps::len(tangent) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point).max(VectorOps::len(tangent)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (p1, p2) = self.split_vector(point)?;
		let (t1, t2) = self.split_vector(tangent)?;

		let mut ret1 = p1.clone();
		let mut ret2 = p2.clone();

		self.manifold1.retract(&p1, &t1, &mut ret1)?;
		self.manifold2.retract(&p2, &t2, &mut ret2)?;

		self.combine_vectors_mut(&ret1, &ret2, result)?;
		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		if VectorOps::len(point) != self.total_dim || VectorOps::len(other) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point).max(VectorOps::len(other)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (p1, p2) = self.split_vector(point)?;
		let (o1, o2) = self.split_vector(other)?;

		let mut tan1 = <linalg::Vec<T> as VectorOps<T>>::zeros(self.dim1);
		let mut tan2 = <linalg::Vec<T> as VectorOps<T>>::zeros(self.dim2);

		self.manifold1.inverse_retract(&p1, &o1, &mut tan1)?;
		self.manifold2.inverse_retract(&p2, &o2, &mut tan2)?;

		self.combine_vectors_mut(&tan1, &tan2, result)?;
		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		if VectorOps::len(point) != self.total_dim
			|| VectorOps::len(euclidean_grad) != self.total_dim
		{
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point).max(VectorOps::len(euclidean_grad)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (p1, p2) = self.split_vector(point)?;
		let (g1, g2) = self.split_vector(euclidean_grad)?;

		let mut rgrad1 = g1.clone();
		let mut rgrad2 = g2.clone();

		self.manifold1
			.euclidean_to_riemannian_gradient(&p1, &g1, &mut rgrad1)?;
		self.manifold2
			.euclidean_to_riemannian_gradient(&p2, &g2, &mut rgrad2)?;

		self.combine_vectors_mut(&rgrad1, &rgrad2, result)?;
		Ok(())
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut p1 = VectorOps::zeros(0);
		let mut p2 = VectorOps::zeros(0);

		self.manifold1.random_point(&mut p1)?;
		self.manifold2.random_point(&mut p2)?;

		self.combine_vectors_mut(&p1, &p2, result)?;
		Ok(())
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		if VectorOps::len(point) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (p1, p2) = self.split_vector(point)?;

		let mut tan1 = <linalg::Vec<T> as VectorOps<T>>::zeros(self.dim1);
		let mut tan2 = <linalg::Vec<T> as VectorOps<T>>::zeros(self.dim2);

		self.manifold1.random_tangent(&p1, &mut tan1)?;
		self.manifold2.random_tangent(&p2, &mut tan2)?;

		self.combine_vectors_mut(&tan1, &tan2, result)?;
		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		if VectorOps::len(x) != self.total_dim || VectorOps::len(y) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(x).max(VectorOps::len(y)),
			));
		}

		let (x1, x2) = self.split_vector(x)?;
		let (y1, y2) = self.split_vector(y)?;

		let d1 = self.manifold1.distance(&x1, &y1)?;
		let d2 = self.manifold2.distance(&x2, &y2)?;

		Ok(<T as Float>::sqrt(d1 * d1 + d2 * d2))
	}

	fn parallel_transport(
		&self,
		from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		if VectorOps::len(from) != self.total_dim
			|| VectorOps::len(to) != self.total_dim
			|| VectorOps::len(vector) != self.total_dim
		{
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(from)
					.max(VectorOps::len(to))
					.max(VectorOps::len(vector)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (f1, f2) = self.split_vector(from)?;
		let (t1, t2) = self.split_vector(to)?;
		let (v1, v2) = self.split_vector(vector)?;

		let mut trans1 = v1.clone();
		let mut trans2 = v2.clone();

		self.manifold1
			.parallel_transport(&f1, &t1, &v1, &mut trans1)?;
		self.manifold2
			.parallel_transport(&f2, &t2, &v2, &mut trans2)?;

		self.combine_vectors_mut(&trans1, &trans2, result)?;
		Ok(())
	}

	fn has_exact_exp_log(&self) -> bool {
		self.manifold1.has_exact_exp_log() && self.manifold2.has_exact_exp_log()
	}

	fn is_flat(&self) -> bool {
		self.manifold1.is_flat() && self.manifold2.is_flat()
	}

	fn scale_tangent(
		&self,
		point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		if VectorOps::len(point) != self.total_dim || VectorOps::len(tangent) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point).max(VectorOps::len(tangent)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (p1, p2) = self.split_vector(point)?;
		let (t1, t2) = self.split_vector(tangent)?;

		let mut scaled1 = t1.clone();
		let mut scaled2 = t2.clone();

		self.manifold1
			.scale_tangent(&p1, scalar, &t1, &mut scaled1)?;
		self.manifold2
			.scale_tangent(&p2, scalar, &t2, &mut scaled2)?;

		self.combine_vectors_mut(&scaled1, &scaled2, result)?;
		Ok(())
	}

	fn add_tangents(
		&self,
		point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
		// Temporary buffer for projection if needed
		_temp: &mut Self::TangentVector,
	) -> Result<()> {
		if VectorOps::len(point) != self.total_dim
			|| VectorOps::len(v1) != self.total_dim
			|| VectorOps::len(v2) != self.total_dim
		{
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(point)
					.max(VectorOps::len(v1))
					.max(VectorOps::len(v2)),
			));
		}

		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let (p1, p2) = self.split_vector(point)?;
		let (v1_1, v1_2) = self.split_vector(v1)?;
		let (v2_1, v2_2) = self.split_vector(v2)?;

		let mut sum1 = v1_1.clone();
		let mut sum2 = v1_2.clone();

		// Create temp buffers for components
		let mut temp1 = <linalg::Vec<T> as VectorOps<T>>::zeros(self.dim1);
		let mut temp2 = <linalg::Vec<T> as VectorOps<T>>::zeros(self.dim2);

		self.manifold1
			.add_tangents(&p1, &v1_1, &v2_1, &mut sum1, &mut temp1)?;
		self.manifold2
			.add_tangents(&p2, &v1_2, &v2_2, &mut sum2, &mut temp2)?;

		self.combine_vectors_mut(&sum1, &sum2, result)?;
		Ok(())
	}
}

/// Creates a static product manifold with type inference.
///
/// This is a convenience function that infers types automatically.
///
/// # Example
///
/// ```rust
/// use riemannopt_manifolds::{product_static, Sphere};
///
/// let sphere1 = Sphere::<f64>::new(3).unwrap();
/// let sphere2 = Sphere::<f64>::new(4).unwrap();
/// let product = product_static(sphere1, sphere2);
/// ```
pub fn product_static<T, M1, M2>(manifold1: M1, manifold2: M2) -> ProductStatic<T, M1, M2>
where
	T: Scalar,
	linalg::DefaultBackend: LinAlgBackend<T>,
	M1: Manifold<T, Point = linalg::Vec<T>, TangentVector = linalg::Vec<T>>,
	M2: Manifold<T, Point = linalg::Vec<T>, TangentVector = linalg::Vec<T>>,
{
	ProductStatic::new(manifold1, manifold2)
}

// Alias for backward compatibility
pub type ProductManifoldStatic<T, M1, M2> = ProductStatic<T, M1, M2>;

// Convenience function for backward compatibility
pub use product_static as product;
