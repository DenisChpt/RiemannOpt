//! # Product Manifold M₁ × M₂ × ... × Mₙ
//!
//! The product manifold enables combining multiple manifolds into a single
//! manifold structure. This is fundamental for optimization problems involving
//! variables on different geometric spaces simultaneously.
//!
//! ## Mathematical Definition
//!
//! For manifolds M₁, M₂, ..., Mₙ, the product manifold is:
//! ```text
//! M = M₁ × M₂ × ... × Mₙ = {(x₁, x₂, ..., xₙ) : xᵢ ∈ Mᵢ}
//! ```
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space decomposes naturally:
//! ```text
//! T_{(x₁,...,xₙ)} M = T_{x₁} M₁ × T_{x₂} M₂ × ... × T_{xₙ} Mₙ
//! ```
//!
//! ### Riemannian Metric
//! The product metric is the sum of component metrics:
//! ```text
//! g_x((u₁,...,uₙ), (v₁,...,vₙ)) = Σᵢ gᵢ(uᵢ, vᵢ)
//! ```
//!
//! ### Geodesics
//! Geodesics in product manifolds are component-wise geodesics:
//! ```text
//! γ(t) = (γ₁(t), γ₂(t), ..., γₙ(t))
//! ```
//!
//! ## Distance Formula
//!
//! Using the product metric:
//! ```text
//! d²(x, y) = Σᵢ d²ᵢ(xᵢ, yᵢ)
//! ```
//!
//! ## Retractions
//!
//! Component-wise retractions:
//! ```text
//! R_x(v) = (R₁(v₁), R₂(v₂), ..., Rₙ(vₙ))
//! ```
//!
//! ## Applications
//!
//! 1. **Multi-task Learning**: Different constraints for different tasks
//! 2. **Neural Networks**: Mixing Euclidean and non-Euclidean layers
//! 3. **Robotics**: Position (ℝ³) and orientation (SO(3)) constraints
//! 4. **Computer Vision**: Multiple geometric transformations
//! 5. **Signal Processing**: Frequency and phase constraints
//! 6. **Optimization**: Block coordinate descent methods
//!
//! ## Implementation Notes
//!
//! This implementation provides a dynamic product manifold that can combine
//! any number of manifolds at runtime. For better performance with a fixed
//! number of manifolds, consider using `ProductStatic`.
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::{Product, Sphere};
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::linalg::{self, VectorOps};
//!
//! // Create S² × S³ - two sphere manifolds
//! let sphere1 = Sphere::<f64>::new(3)?;
//! let sphere2 = Sphere::<f64>::new(4)?;
//! let product = Product::new(vec![Box::new(sphere1), Box::new(sphere2)]);
//!
//! // Points are concatenated vectors
//! let mut x: linalg::Vec<f64> = VectorOps::zeros(product.component_dimensions().iter().sum());
//! <Product as Manifold<f64>>::random_point(&product, &mut x)?;
//!
//! // Access components
//! let x_sphere1 = product.split_point::<f64>(&x, 0)?;
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use riemannopt_core::{
	error::{ManifoldError, Result},
	linalg::{self, LinAlgBackend, VectorOps},
	manifold::Manifold,
	types::Scalar,
};

// Type aliases for the concrete f64 backend types used internally
type Vec64 = linalg::Vec<f64>;

/// A dynamic product manifold combining multiple manifolds.
///
/// This structure represents the Cartesian product of multiple Riemannian
/// manifolds, where points are concatenated vectors from all manifolds.
///
/// # Type Parameters
///
/// The manifold is generic over the scalar type T through the Manifold trait.
///
/// # Invariants
///
/// - All component manifolds must be valid
/// - Points and tangent vectors are concatenated in order
/// - Operations are performed component-wise
#[derive(Debug)]
pub struct Product {
	/// Component manifolds stored as trait objects
	manifolds: Vec<Box<dyn Manifold<f64, Point = Vec64, TangentVector = Vec64>>>,
	/// Dimensions of each component manifold
	dimensions: Vec<usize>,
	/// Cumulative dimensions for efficient indexing
	cumulative_dims: Vec<usize>,
	/// Total dimension
	total_dim: usize,
}

impl Product {
	/// Creates a new product manifold from component manifolds.
	///
	/// # Arguments
	///
	/// * `manifolds` - Vector of boxed manifold trait objects
	///
	/// # Returns
	///
	/// A product manifold combining all components.
	///
	/// # Panics
	///
	/// Panics if the manifolds vector is empty.
	///
	/// # Example
	///
	/// ```rust
	/// # use riemannopt_manifolds::{Product, Sphere};
	/// let sphere1 = Sphere::<f64>::new(3).unwrap();
	/// let sphere2 = Sphere::<f64>::new(4).unwrap();
	/// let product = Product::new(vec![Box::new(sphere1), Box::new(sphere2)]);
	/// ```
	pub fn new(
		manifolds: Vec<Box<dyn Manifold<f64, Point = Vec64, TangentVector = Vec64>>>,
	) -> Self {
		assert!(
			!manifolds.is_empty(),
			"Product manifold requires at least one component"
		);

		let dimensions: Vec<usize> = manifolds
			.iter()
			.map(|m| {
				// Get dimension by creating a default point and getting a random point
				let mut p = <Vec64 as VectorOps<f64>>::zeros(0);
				m.random_point(&mut p).unwrap();
				VectorOps::len(&p)
			})
			.collect();

		let mut cumulative_dims = vec![0];
		let mut cum_sum = 0;
		for &dim in &dimensions {
			cum_sum += dim;
			cumulative_dims.push(cum_sum);
		}

		let total_dim = cum_sum;

		Self {
			manifolds,
			dimensions,
			cumulative_dims,
			total_dim,
		}
	}

	/// Returns the number of component manifolds.
	#[inline]
	pub fn num_components(&self) -> usize {
		self.manifolds.len()
	}

	/// Returns the dimensions of all component manifolds.
	#[inline]
	pub fn component_dimensions(&self) -> &[usize] {
		&self.dimensions
	}

	/// Internal: split an f64 vector into components
	fn split_f64(&self, vector: &Vec64, component_idx: Option<usize>) -> Result<Vec<Vec64>> {
		if VectorOps::len(vector) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(vector),
			));
		}

		if let Some(idx) = component_idx {
			if idx >= self.manifolds.len() {
				return Err(ManifoldError::invalid_parameter(format!(
					"Component index {} out of bounds",
					idx
				)));
			}

			let start = self.cumulative_dims[idx];
			let dim = self.dimensions[idx];
			let component =
				<Vec64 as VectorOps<f64>>::from_fn(dim, |i| VectorOps::get(vector, start + i));
			Ok(vec![component])
		} else {
			let mut components = Vec::with_capacity(self.manifolds.len());
			for i in 0..self.manifolds.len() {
				let start = self.cumulative_dims[i];
				let dim = self.dimensions[i];
				let component =
					<Vec64 as VectorOps<f64>>::from_fn(dim, |j| VectorOps::get(vector, start + j));
				components.push(component);
			}
			Ok(components)
		}
	}

	/// Internal: combine f64 component vectors into one
	fn combine_f64(&self, components: &[Vec64]) -> Result<Vec64> {
		if components.len() != self.manifolds.len() {
			return Err(ManifoldError::invalid_parameter(format!(
				"Expected {} components, got {}",
				self.manifolds.len(),
				components.len()
			)));
		}

		for (i, comp) in components.iter().enumerate() {
			if VectorOps::len(comp) != self.dimensions[i] {
				return Err(ManifoldError::dimension_mismatch(
					self.dimensions[i],
					VectorOps::len(comp),
				));
			}
		}

		let mut combined = <Vec64 as VectorOps<f64>>::zeros(self.total_dim);
		for (i, comp) in components.iter().enumerate() {
			let start = self.cumulative_dims[i];
			let dim = self.dimensions[i];
			for j in 0..dim {
				*VectorOps::get_mut(&mut combined, start + j) = VectorOps::get(comp, j);
			}
		}

		Ok(combined)
	}

	/// Splits a product space vector into component vectors.
	///
	/// # Arguments
	///
	/// * `vector` - Combined vector from product space
	/// * `component_idx` - Index of component to extract (None for all)
	///
	/// # Returns
	///
	/// Vector of component vectors, or single component if index specified.
	///
	/// # Errors
	///
	/// Returns error if:
	/// - Vector dimension doesn't match total dimension
	/// - Component index is out of bounds
	pub fn split_vector<T: Scalar>(
		&self,
		vector: &linalg::Vec<T>,
		component_idx: Option<usize>,
	) -> Result<Vec<linalg::Vec<T>>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if VectorOps::len(vector) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(vector),
			));
		}

		if let Some(idx) = component_idx {
			if idx >= self.manifolds.len() {
				return Err(ManifoldError::invalid_parameter(format!(
					"Component index {} out of bounds",
					idx
				)));
			}

			let start = self.cumulative_dims[idx];
			let dim = self.dimensions[idx];
			let component = <linalg::Vec<T> as VectorOps<T>>::from_fn(dim, |i| {
				VectorOps::get(vector, start + i)
			});
			Ok(vec![component])
		} else {
			let mut components = Vec::with_capacity(self.manifolds.len());
			for i in 0..self.manifolds.len() {
				let start = self.cumulative_dims[i];
				let dim = self.dimensions[i];
				let component = <linalg::Vec<T> as VectorOps<T>>::from_fn(dim, |j| {
					VectorOps::get(vector, start + j)
				});
				components.push(component);
			}
			Ok(components)
		}
	}

	/// Combines component vectors into a product space vector.
	///
	/// # Arguments
	///
	/// * `components` - Vector of component vectors
	///
	/// # Returns
	///
	/// Combined vector in product space.
	///
	/// # Errors
	///
	/// Returns error if:
	/// - Number of components doesn't match manifold count
	/// - Component dimensions don't match expected dimensions
	pub fn combine_vectors<T: Scalar>(
		&self,
		components: &[linalg::Vec<T>],
	) -> Result<linalg::Vec<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		if components.len() != self.manifolds.len() {
			return Err(ManifoldError::invalid_parameter(format!(
				"Expected {} components, got {}",
				self.manifolds.len(),
				components.len()
			)));
		}

		for (i, comp) in components.iter().enumerate() {
			if VectorOps::len(comp) != self.dimensions[i] {
				return Err(ManifoldError::dimension_mismatch(
					self.dimensions[i],
					VectorOps::len(comp),
				));
			}
		}

		let mut combined = <linalg::Vec<T> as VectorOps<T>>::zeros(self.total_dim);
		for (i, comp) in components.iter().enumerate() {
			let start = self.cumulative_dims[i];
			let dim = self.dimensions[i];
			for j in 0..dim {
				*VectorOps::get_mut(&mut combined, start + j) = VectorOps::get(comp, j);
			}
		}

		Ok(combined)
	}

	/// Splits a point and returns a specific component.
	///
	/// # Arguments
	///
	/// * `point` - Point in product manifold
	/// * `idx` - Component index
	///
	/// # Returns
	///
	/// The component at the specified index.
	pub fn split_point<T: Scalar>(
		&self,
		point: &linalg::Vec<T>,
		idx: usize,
	) -> Result<linalg::Vec<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		let components = self.split_vector(point, Some(idx))?;
		Ok(components.into_iter().next().unwrap())
	}

	/// Helper to convert a generic T vector to f64
	fn to_f64<T: Scalar>(vec: &linalg::Vec<T>) -> Vec64
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		<Vec64 as VectorOps<f64>>::from_fn(VectorOps::len(vec), |i| {
			<T as Scalar>::to_f64(VectorOps::get(vec, i))
		})
	}

	/// Helper to convert an f64 vector to generic T
	fn from_f64<T: Scalar>(vec: &Vec64) -> linalg::Vec<T>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		<linalg::Vec<T> as VectorOps<T>>::from_fn(VectorOps::len(vec), |i| {
			<T as Scalar>::from_f64(VectorOps::get(vec, i))
		})
	}
}

impl<T> Manifold<T> for Product
where
	T: Scalar,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;

	fn name(&self) -> &str {
		"Product"
	}

	fn dimension(&self) -> usize {
		self.manifolds.iter().map(|m| m.dimension()).sum()
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if VectorOps::len(point) != self.total_dim {
			return false;
		}

		let point_f64 = Self::to_f64(point);
		let tol_f64 = <T as Scalar>::to_f64(tol);

		if let Ok(components) = self.split_f64(&point_f64, None) {
			components
				.iter()
				.zip(self.manifolds.iter())
				.all(|(comp, manifold)| manifold.is_point_on_manifold(comp, tol_f64))
		} else {
			false
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

		let point_f64 = Self::to_f64(point);
		let vector_f64 = Self::to_f64(vector);
		let tol_f64 = <T as Scalar>::to_f64(tol);

		match (
			self.split_f64(&point_f64, None),
			self.split_f64(&vector_f64, None),
		) {
			(Ok(point_comps), Ok(vector_comps)) => point_comps
				.iter()
				.zip(vector_comps.iter())
				.zip(self.manifolds.iter())
				.all(|((p, v), m)| m.is_vector_in_tangent_space(p, v, tol_f64)),
			_ => false,
		}
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Ensure result has correct size
		if VectorOps::len(result) != self.total_dim {
			*result = VectorOps::zeros(self.total_dim);
		}

		let point_f64 = Self::to_f64(point);

		// Handle dimension mismatch by padding or truncating
		let padded_point = if VectorOps::len(&point_f64) != self.total_dim {
			let mut p = <Vec64 as VectorOps<f64>>::zeros(self.total_dim);
			let copy_len = VectorOps::len(&point_f64).min(self.total_dim);
			for i in 0..copy_len {
				*VectorOps::get_mut(&mut p, i) = VectorOps::get(&point_f64, i);
			}
			p
		} else {
			point_f64
		};

		if let Ok(components) = self.split_f64(&padded_point, None) {
			let mut projected_components = Vec::with_capacity(self.manifolds.len());
			for (comp, manifold) in components.iter().zip(self.manifolds.iter()) {
				let mut proj_comp = comp.clone();
				manifold.project_point(comp, &mut proj_comp);
				projected_components.push(proj_comp);
			}

			if let Ok(combined) = self.combine_f64(&projected_components) {
				*result = Self::from_f64(&combined);
			}
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

		let point_f64 = Self::to_f64(point);
		let vector_f64 = Self::to_f64(vector);

		let point_comps = self.split_f64(&point_f64, None)?;
		let vector_comps = self.split_f64(&vector_f64, None)?;

		let mut projected_components = Vec::with_capacity(self.manifolds.len());

		for ((p, v), manifold) in point_comps
			.iter()
			.zip(vector_comps.iter())
			.zip(self.manifolds.iter())
		{
			let mut proj_v = v.clone();
			manifold.project_tangent(p, v, &mut proj_v)?;
			projected_components.push(proj_v);
		}

		let combined = self.combine_f64(&projected_components)?;
		*result = Self::from_f64(&combined);
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

		let point_f64 = Self::to_f64(point);
		let u_f64 = Self::to_f64(u);
		let v_f64 = Self::to_f64(v);

		let point_comps = self.split_f64(&point_f64, None)?;
		let u_comps = self.split_f64(&u_f64, None)?;
		let v_comps = self.split_f64(&v_f64, None)?;

		let mut total_inner = 0.0;

		for ((p, (u_i, v_i)), manifold) in point_comps
			.iter()
			.zip(u_comps.iter().zip(v_comps.iter()))
			.zip(self.manifolds.iter())
		{
			total_inner += manifold.inner_product(p, u_i, v_i)?;
		}

		Ok(<T as Scalar>::from_f64(total_inner))
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

		let point_f64 = Self::to_f64(point);
		let tangent_f64 = Self::to_f64(tangent);

		let point_comps = self.split_f64(&point_f64, None)?;
		let tangent_comps = self.split_f64(&tangent_f64, None)?;

		let mut retracted_components = Vec::with_capacity(self.manifolds.len());

		for ((p, t), manifold) in point_comps
			.iter()
			.zip(tangent_comps.iter())
			.zip(self.manifolds.iter())
		{
			let mut ret = p.clone();
			manifold.retract(p, t, &mut ret)?;
			retracted_components.push(ret);
		}

		let combined = self.combine_f64(&retracted_components)?;
		*result = Self::from_f64(&combined);
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

		let point_f64 = Self::to_f64(point);
		let other_f64 = Self::to_f64(other);

		let point_comps = self.split_f64(&point_f64, None)?;
		let other_comps = self.split_f64(&other_f64, None)?;

		let mut tangent_components = Vec::with_capacity(self.manifolds.len());

		for ((p, o), manifold) in point_comps
			.iter()
			.zip(other_comps.iter())
			.zip(self.manifolds.iter())
		{
			let mut tan = <Vec64 as VectorOps<f64>>::zeros(VectorOps::len(p));
			manifold.inverse_retract(p, o, &mut tan)?;
			tangent_components.push(tan);
		}

		let combined = self.combine_f64(&tangent_components)?;
		*result = Self::from_f64(&combined);
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

		let point_f64 = Self::to_f64(point);
		let grad_f64 = Self::to_f64(euclidean_grad);

		let point_comps = self.split_f64(&point_f64, None)?;
		let grad_comps = self.split_f64(&grad_f64, None)?;

		let mut rgrad_components = Vec::with_capacity(self.manifolds.len());

		for ((p, g), manifold) in point_comps
			.iter()
			.zip(grad_comps.iter())
			.zip(self.manifolds.iter())
		{
			let mut rgrad = g.clone();
			manifold.euclidean_to_riemannian_gradient(p, g, &mut rgrad)?;
			rgrad_components.push(rgrad);
		}

		let combined = self.combine_f64(&rgrad_components)?;
		*result = Self::from_f64(&combined);
		Ok(())
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut components = Vec::with_capacity(self.manifolds.len());

		for manifold in &self.manifolds {
			let mut component = <Vec64 as VectorOps<f64>>::zeros(manifold.dimension());
			manifold.random_point(&mut component)?;
			components.push(component);
		}

		let combined = self.combine_f64(&components)?;
		*result = Self::from_f64(&combined);
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

		let point_f64 = Self::to_f64(point);
		let point_comps = self.split_f64(&point_f64, None)?;

		let mut tangent_components = Vec::with_capacity(self.manifolds.len());

		for (p, manifold) in point_comps.iter().zip(self.manifolds.iter()) {
			let mut tan = <Vec64 as VectorOps<f64>>::zeros(VectorOps::len(p));
			manifold.random_tangent(p, &mut tan)?;
			tangent_components.push(tan);
		}

		let combined = self.combine_f64(&tangent_components)?;
		*result = Self::from_f64(&combined);
		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		if VectorOps::len(x) != self.total_dim || VectorOps::len(y) != self.total_dim {
			return Err(ManifoldError::dimension_mismatch(
				self.total_dim,
				VectorOps::len(x).max(VectorOps::len(y)),
			));
		}

		let x_f64 = Self::to_f64(x);
		let y_f64 = Self::to_f64(y);

		let x_comps = self.split_f64(&x_f64, None)?;
		let y_comps = self.split_f64(&y_f64, None)?;

		let mut dist_sq = 0.0;

		for ((x_i, y_i), manifold) in x_comps
			.iter()
			.zip(y_comps.iter())
			.zip(self.manifolds.iter())
		{
			let d = manifold.distance(x_i, y_i)?;
			dist_sq += d * d;
		}

		Ok(<T as Scalar>::from_f64(dist_sq.sqrt()))
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

		let from_f64 = Self::to_f64(from);
		let to_f64 = Self::to_f64(to);
		let vector_f64 = Self::to_f64(vector);

		let from_comps = self.split_f64(&from_f64, None)?;
		let to_comps = self.split_f64(&to_f64, None)?;
		let vector_comps = self.split_f64(&vector_f64, None)?;

		let mut transported_components = Vec::with_capacity(self.manifolds.len());

		for (((f, t), v), manifold) in from_comps
			.iter()
			.zip(to_comps.iter())
			.zip(vector_comps.iter())
			.zip(self.manifolds.iter())
		{
			let mut trans = v.clone();
			manifold.parallel_transport(f, t, v, &mut trans)?;
			transported_components.push(trans);
		}

		let combined = self.combine_f64(&transported_components)?;
		*result = Self::from_f64(&combined);
		Ok(())
	}

	fn has_exact_exp_log(&self) -> bool {
		self.manifolds.iter().all(|m| m.has_exact_exp_log())
	}

	fn is_flat(&self) -> bool {
		self.manifolds.iter().all(|m| m.is_flat())
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

		let point_f64 = Self::to_f64(point);
		let tangent_f64 = Self::to_f64(tangent);
		let scalar_f64 = scalar.to_f64();

		let point_comps = self.split_f64(&point_f64, None)?;
		let tangent_comps = self.split_f64(&tangent_f64, None)?;

		let mut scaled_components = Vec::with_capacity(self.manifolds.len());

		for ((p, t), manifold) in point_comps
			.iter()
			.zip(tangent_comps.iter())
			.zip(self.manifolds.iter())
		{
			let mut scaled = t.clone();
			manifold.scale_tangent(p, scalar_f64, t, &mut scaled)?;
			scaled_components.push(scaled);
		}

		let combined = self.combine_f64(&scaled_components)?;
		*result = Self::from_f64(&combined);
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

		let point_f64 = Self::to_f64(point);
		let v1_f64 = Self::to_f64(v1);
		let v2_f64 = Self::to_f64(v2);

		let point_comps = self.split_f64(&point_f64, None)?;
		let v1_comps = self.split_f64(&v1_f64, None)?;
		let v2_comps = self.split_f64(&v2_f64, None)?;

		let mut sum_components = Vec::with_capacity(self.manifolds.len());

		for (((p, v1_i), v2_i), manifold) in point_comps
			.iter()
			.zip(v1_comps.iter())
			.zip(v2_comps.iter())
			.zip(self.manifolds.iter())
		{
			let mut sum = v1_i.clone();
			let mut temp_f64 = <Vec64 as VectorOps<f64>>::zeros(VectorOps::len(v1_i));
			manifold.add_tangents(p, v1_i, v2_i, &mut sum, &mut temp_f64)?;
			sum_components.push(sum);
		}

		let combined = self.combine_f64(&sum_components)?;
		*result = Self::from_f64(&combined);
		Ok(())
	}
}

// Alias for backward compatibility
pub type ProductManifold = Product;
