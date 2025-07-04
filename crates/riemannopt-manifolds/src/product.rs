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
//! use riemannopt_manifolds::{Product, Sphere, Stiefel};
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DVector;
//!
//! // Create S² × St(5,2) - sphere and Stiefel manifold
//! let sphere = Sphere::<f64>::new(3)?;
//! let stiefel = Stiefel::<f64>::new(5, 2)?;
//! let product = Product::new(vec![Box::new(sphere), Box::new(stiefel)]);
//!
//! // Points are concatenated vectors
//! let mut x = product.random_point();
//! 
//! // Access components
//! let (x_sphere, x_stiefel) = product.split_point(&x, 0)?;
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::DVector;

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    memory::workspace::Workspace,
    types::Scalar,
};

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
    manifolds: Vec<Box<dyn Manifold<f64, Point = DVector<f64>, TangentVector = DVector<f64>>>>,
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
    /// # use riemannopt_manifolds::{Product, Sphere, SPD};
    /// let sphere = Sphere::<f64>::new(3).unwrap();
    /// let spd = SPD::<f64>::new(2).unwrap();
    /// let product = Product::new(vec![Box::new(sphere), Box::new(spd)]);
    /// ```
    pub fn new(manifolds: Vec<Box<dyn Manifold<f64, Point = DVector<f64>, TangentVector = DVector<f64>>>>) -> Self {
        assert!(!manifolds.is_empty(), "Product manifold requires at least one component");
        
        let dimensions: Vec<usize> = manifolds.iter().map(|m| {
            // Get dimension by creating a random point
            let p = m.random_point();
            p.len()
        }).collect();
        
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
        vector: &DVector<T>,
        component_idx: Option<usize>,
    ) -> Result<Vec<DVector<T>>> {
        if vector.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                vector.len()
            ));
        }

        if let Some(idx) = component_idx {
            if idx >= self.manifolds.len() {
                return Err(ManifoldError::invalid_parameter(
                    format!("Component index {} out of bounds", idx)
                ));
            }
            
            let start = self.cumulative_dims[idx];
            let dim = self.dimensions[idx];
            let component = vector.rows(start, dim).clone_owned();
            Ok(vec![component])
        } else {
            let mut components = Vec::with_capacity(self.manifolds.len());
            for i in 0..self.manifolds.len() {
                let start = self.cumulative_dims[i];
                let dim = self.dimensions[i];
                components.push(vector.rows(start, dim).clone_owned());
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
        components: &[DVector<T>],
    ) -> Result<DVector<T>> {
        if components.len() != self.manifolds.len() {
            return Err(ManifoldError::invalid_parameter(
                format!(
                    "Expected {} components, got {}",
                    self.manifolds.len(),
                    components.len()
                )
            ));
        }

        for (i, comp) in components.iter().enumerate() {
            if comp.len() != self.dimensions[i] {
                return Err(ManifoldError::dimension_mismatch(
                    self.dimensions[i],
                    comp.len()
                ));
            }
        }

        let mut combined = DVector::zeros(self.total_dim);
        for (i, comp) in components.iter().enumerate() {
            let start = self.cumulative_dims[i];
            let dim = self.dimensions[i];
            combined.rows_mut(start, dim).copy_from(comp);
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
        point: &DVector<T>,
        idx: usize,
    ) -> Result<DVector<T>> {
        let components = self.split_vector(point, Some(idx))?;
        Ok(components.into_iter().next().unwrap())
    }

    /// Helper to convert between scalar types
    fn convert_scalar<T1: Scalar, T2: Scalar>(vec: &DVector<T1>) -> DVector<T2> {
        DVector::from_iterator(
            vec.len(),
            vec.iter().map(|&x| <T2 as Scalar>::from_f64(<T1 as Scalar>::to_f64(x)))
        )
    }
}

impl<T: Scalar> Manifold<T> for Product {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;

    fn name(&self) -> &str {
        "Product"
    }

    fn dimension(&self) -> usize {
        self.manifolds.iter().map(|m| m.dimension()).sum()
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.len() != self.total_dim {
            return false;
        }

        // Convert to f64 for trait objects
        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let tol_f64 = <T as Scalar>::to_f64(tol);

        if let Ok(components) = self.split_vector(&point_f64, None) {
            components.iter()
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
        if point.len() != self.total_dim || vector.len() != self.total_dim {
            return false;
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let vector_f64 = Self::convert_scalar::<T, f64>(vector);
        let tol_f64 = <T as Scalar>::to_f64(tol);

        match (self.split_vector(&point_f64, None), self.split_vector(&vector_f64, None)) {
            (Ok(point_comps), Ok(vector_comps)) => {
                point_comps.iter()
                    .zip(vector_comps.iter())
                    .zip(self.manifolds.iter())
                    .all(|((p, v), m)| m.is_vector_in_tangent_space(p, v, tol_f64))
            }
            _ => false,
        }
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, _workspace: &mut Workspace<T>) {
        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        
        // Handle dimension mismatch by padding or truncating
        let padded_point = if point_f64.len() != self.total_dim {
            let mut p = DVector::zeros(self.total_dim);
            let copy_len = point_f64.len().min(self.total_dim);
            p.rows_mut(0, copy_len).copy_from(&point_f64.rows(0, copy_len));
            p
        } else {
            point_f64.clone()
        };

        if let Ok(components) = self.split_vector(&padded_point, None) {
            let mut projected_components = Vec::with_capacity(self.manifolds.len());
            let mut workspace_f64 = Workspace::<f64>::new();
            
            for (comp, manifold) in components.iter().zip(self.manifolds.iter()) {
                let mut proj_comp = comp.clone();
                manifold.project_point(comp, &mut proj_comp, &mut workspace_f64);
                projected_components.push(proj_comp);
            }

            if let Ok(combined) = self.combine_vectors(&projected_components) {
                *result = Self::convert_scalar::<f64, T>(&combined);
            }
        }
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.total_dim || vector.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len().max(vector.len())
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let vector_f64 = Self::convert_scalar::<T, f64>(vector);

        let point_comps = self.split_vector(&point_f64, None)?;
        let vector_comps = self.split_vector(&vector_f64, None)?;

        let mut projected_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for ((p, v), manifold) in point_comps.iter()
            .zip(vector_comps.iter())
            .zip(self.manifolds.iter())
        {
            let mut proj_v = v.clone();
            manifold.project_tangent(p, v, &mut proj_v, &mut workspace_f64)?;
            projected_components.push(proj_v);
        }

        let combined = self.combine_vectors(&projected_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
        Ok(())
    }

    fn inner_product(
        &self,
        point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        if point.len() != self.total_dim || u.len() != self.total_dim || v.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len().max(u.len()).max(v.len())
            ));
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let u_f64 = Self::convert_scalar::<T, f64>(u);
        let v_f64 = Self::convert_scalar::<T, f64>(v);

        let point_comps = self.split_vector(&point_f64, None)?;
        let u_comps = self.split_vector(&u_f64, None)?;
        let v_comps = self.split_vector(&v_f64, None)?;

        let mut total_inner = 0.0;

        for ((p, (u_i, v_i)), manifold) in point_comps.iter()
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
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.total_dim || tangent.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len().max(tangent.len())
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let tangent_f64 = Self::convert_scalar::<T, f64>(tangent);

        let point_comps = self.split_vector(&point_f64, None)?;
        let tangent_comps = self.split_vector(&tangent_f64, None)?;

        let mut retracted_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for ((p, t), manifold) in point_comps.iter()
            .zip(tangent_comps.iter())
            .zip(self.manifolds.iter())
        {
            let mut ret = p.clone();
            manifold.retract(p, t, &mut ret, &mut workspace_f64)?;
            retracted_components.push(ret);
        }

        let combined = self.combine_vectors(&retracted_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.total_dim || other.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len().max(other.len())
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let other_f64 = Self::convert_scalar::<T, f64>(other);

        let point_comps = self.split_vector(&point_f64, None)?;
        let other_comps = self.split_vector(&other_f64, None)?;

        let mut tangent_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for ((p, o), manifold) in point_comps.iter()
            .zip(other_comps.iter())
            .zip(self.manifolds.iter())
        {
            let mut tan = DVector::zeros(p.len());
            manifold.inverse_retract(p, o, &mut tan, &mut workspace_f64)?;
            tangent_components.push(tan);
        }

        let combined = self.combine_vectors(&tangent_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.total_dim || euclidean_grad.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len().max(euclidean_grad.len())
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let grad_f64 = Self::convert_scalar::<T, f64>(euclidean_grad);

        let point_comps = self.split_vector(&point_f64, None)?;
        let grad_comps = self.split_vector(&grad_f64, None)?;

        let mut rgrad_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for ((p, g), manifold) in point_comps.iter()
            .zip(grad_comps.iter())
            .zip(self.manifolds.iter())
        {
            let mut rgrad = g.clone();
            manifold.euclidean_to_riemannian_gradient(p, g, &mut rgrad, &mut workspace_f64)?;
            rgrad_components.push(rgrad);
        }

        let combined = self.combine_vectors(&rgrad_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
        Ok(())
    }

    fn random_point(&self) -> Self::Point {
        let mut components = Vec::with_capacity(self.manifolds.len());
        
        for manifold in &self.manifolds {
            components.push(manifold.random_point());
        }

        let combined = self.combine_vectors(&components).unwrap();
        Self::convert_scalar::<f64, T>(&combined)
    }

    fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector, _workspace: &mut Workspace<T>) -> Result<()> {
        if point.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len()
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let point_comps = self.split_vector(&point_f64, None)?;

        let mut tangent_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for (p, manifold) in point_comps.iter().zip(self.manifolds.iter()) {
            let mut tan = DVector::zeros(p.len());
            manifold.random_tangent(p, &mut tan, &mut workspace_f64)?;
            tangent_components.push(tan);
        }

        let combined = self.combine_vectors(&tangent_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, _workspace: &mut Workspace<T>) -> Result<T> {
        if x.len() != self.total_dim || y.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                x.len().max(y.len())
            ));
        }

        let x_f64 = Self::convert_scalar::<T, f64>(x);
        let y_f64 = Self::convert_scalar::<T, f64>(y);

        let x_comps = self.split_vector(&x_f64, None)?;
        let y_comps = self.split_vector(&y_f64, None)?;

        let mut dist_sq = 0.0;
        let mut workspace_f64 = Workspace::<f64>::new();

        for ((x_i, y_i), manifold) in x_comps.iter()
            .zip(y_comps.iter())
            .zip(self.manifolds.iter())
        {
            let d = manifold.distance(x_i, y_i, &mut workspace_f64)?;
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
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if from.len() != self.total_dim || to.len() != self.total_dim || vector.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                from.len().max(to.len()).max(vector.len())
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let from_f64 = Self::convert_scalar::<T, f64>(from);
        let to_f64 = Self::convert_scalar::<T, f64>(to);
        let vector_f64 = Self::convert_scalar::<T, f64>(vector);

        let from_comps = self.split_vector(&from_f64, None)?;
        let to_comps = self.split_vector(&to_f64, None)?;
        let vector_comps = self.split_vector(&vector_f64, None)?;

        let mut transported_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for (((f, t), v), manifold) in from_comps.iter()
            .zip(to_comps.iter())
            .zip(vector_comps.iter())
            .zip(self.manifolds.iter())
        {
            let mut trans = v.clone();
            manifold.parallel_transport(f, t, v, &mut trans, &mut workspace_f64)?;
            transported_components.push(trans);
        }

        let combined = self.combine_vectors(&transported_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
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
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.total_dim || tangent.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len().max(tangent.len())
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let tangent_f64 = Self::convert_scalar::<T, f64>(tangent);
        let scalar_f64 = scalar.to_f64();

        let point_comps = self.split_vector(&point_f64, None)?;
        let tangent_comps = self.split_vector(&tangent_f64, None)?;

        let mut scaled_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for ((p, t), manifold) in point_comps.iter()
            .zip(tangent_comps.iter())
            .zip(self.manifolds.iter())
        {
            let mut scaled = t.clone();
            manifold.scale_tangent(p, scalar_f64, t, &mut scaled, &mut workspace_f64)?;
            scaled_components.push(scaled);
        }

        let combined = self.combine_vectors(&scaled_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
        Ok(())
    }

    fn add_tangents(
        &self,
        point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if point.len() != self.total_dim || v1.len() != self.total_dim || v2.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                point.len().max(v1.len()).max(v2.len())
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        let point_f64 = Self::convert_scalar::<T, f64>(point);
        let v1_f64 = Self::convert_scalar::<T, f64>(v1);
        let v2_f64 = Self::convert_scalar::<T, f64>(v2);

        let point_comps = self.split_vector(&point_f64, None)?;
        let v1_comps = self.split_vector(&v1_f64, None)?;
        let v2_comps = self.split_vector(&v2_f64, None)?;

        let mut sum_components = Vec::with_capacity(self.manifolds.len());
        let mut workspace_f64 = Workspace::<f64>::new();

        for (((p, v1_i), v2_i), manifold) in point_comps.iter()
            .zip(v1_comps.iter())
            .zip(v2_comps.iter())
            .zip(self.manifolds.iter())
        {
            let mut sum = v1_i.clone();
            manifold.add_tangents(p, v1_i, v2_i, &mut sum, &mut workspace_f64)?;
            sum_components.push(sum);
        }

        let combined = self.combine_vectors(&sum_components)?;
        *result = Self::convert_scalar::<f64, T>(&combined);
        Ok(())
    }
}

// Alias for backward compatibility
pub type ProductManifold = Product;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sphere;
    use approx::assert_relative_eq;
    use riemannopt_core::memory::workspace::Workspace;

    fn create_test_product() -> Product {
        let sphere1 = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        Product::new(vec![Box::new(sphere1), Box::new(sphere2)])
    }

    #[test]
    fn test_product_creation() {
        let product = create_test_product();
        assert_eq!(product.num_components(), 2);
        assert_eq!(<Product as Manifold<f64>>::dimension(&product), 2 + 3); // S^2 dim=2, S^3 dim=3
    }

    #[test]
    fn test_split_combine() {
        let product = create_test_product();
        let point = product.random_point();
        
        // Split and recombine
        let components = product.split_vector::<f64>(&point, None).unwrap();
        assert_eq!(components.len(), 2);
        
        let recombined = product.combine_vectors(&components).unwrap();
        assert_relative_eq!(point, recombined, epsilon = 1e-14);
    }

    #[test]
    fn test_product_operations() {
        let product = create_test_product();
        let mut workspace = Workspace::<f64>::new();
        
        // Test point validation
        let point = product.random_point();
        assert!(product.is_point_on_manifold(&point, 1e-6));
        
        // Test tangent projection
        let mut tangent = DVector::zeros(<Product as Manifold<f64>>::dimension(&product));
        product.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        assert!(product.is_vector_in_tangent_space(&point, &tangent, 1e-6));
        
        // Test retraction
        let scaled_tangent = tangent * 0.1;
        let mut retracted = DVector::zeros(<Product as Manifold<f64>>::dimension(&product));
        product.retract(&point, &scaled_tangent, &mut retracted, &mut workspace).unwrap();
        assert!(product.is_point_on_manifold(&retracted, 1e-6));
    }

    #[test]
    fn test_product_inner_product() {
        let product = create_test_product();
        let mut workspace = Workspace::<f64>::new();
        
        let point = product.random_point();
        let mut u = DVector::zeros(<Product as Manifold<f64>>::dimension(&product));
        let mut v = DVector::zeros(<Product as Manifold<f64>>::dimension(&product));
        product.random_tangent(&point, &mut u, &mut workspace).unwrap();
        product.random_tangent(&point, &mut v, &mut workspace).unwrap();
        
        let ip_uv = product.inner_product(&point, &u, &v).unwrap();
        let ip_vu = product.inner_product(&point, &v, &u).unwrap();
        
        // Check symmetry
        assert_relative_eq!(ip_uv, ip_vu, epsilon = 1e-10);
    }

    #[test]
    fn test_product_distance() {
        let product = create_test_product();
        let mut workspace = Workspace::<f64>::new();
        
        let x = product.random_point();
        let y = product.random_point();
        
        // Distance to self should be zero
        let d_xx = product.distance(&x, &x, &mut workspace).unwrap();
        assert_relative_eq!(d_xx, 0.0, epsilon = 1e-10);
        
        // Distance should be symmetric
        let d_xy = product.distance(&x, &y, &mut workspace).unwrap();
        let d_yx = product.distance(&y, &x, &mut workspace).unwrap();
        assert_relative_eq!(d_xy, d_yx, epsilon = 1e-10);
        
        // Distance should be non-negative
        assert!(d_xy >= 0.0);
    }

    #[test]
    fn test_product_properties() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        
        // Flat components
        let flat_prod = Product::new(vec![Box::new(sphere)]);
        assert!(!<Product as Manifold<f64>>::is_flat(&flat_prod)); // Sphere is not flat
        
        // Has exact exp/log
        assert!(!<Product as Manifold<f64>>::has_exact_exp_log(&flat_prod)); // Sphere doesn't have exact exp/log
    }
}