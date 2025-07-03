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
//! use riemannopt_manifolds::{ProductStatic, Sphere, Stiefel};
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//!
//! // Create S² × St(5,2) statically
//! let sphere = Sphere::<f64>::new(3)?;
//! let stiefel = Stiefel::<f64>::new(5, 2)?;
//! let product = ProductStatic::new(sphere, stiefel);
//!
//! // Operations are fully type-safe and optimized
//! let x = product.random_point();
//! let mut workspace = Workspace::<f64>::new();
//!
//! // Access components directly
//! let (x1, x2) = product.split_point(&x)?;
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::DVector;
use num_traits::Float;
use std::marker::PhantomData;

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    memory::workspace::Workspace,
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
    M1: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
    M2: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
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
        let test1 = manifold1.random_point();
        let test2 = manifold2.random_point();
        let dim1 = test1.len();
        let dim2 = test2.len();
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
    pub fn split_vector(&self, vector: &DVector<T>) -> Result<(DVector<T>, DVector<T>)> {
        if vector.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                vector.len()
            ));
        }

        let comp1 = vector.rows(0, self.dim1).clone_owned();
        let comp2 = vector.rows(self.dim1, self.dim2).clone_owned();
        
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
    pub fn combine_vectors(&self, comp1: &DVector<T>, comp2: &DVector<T>) -> Result<DVector<T>> {
        if comp1.len() != self.dim1 {
            return Err(ManifoldError::dimension_mismatch(
                self.dim1,
                comp1.len()
            ));
        }
        if comp2.len() != self.dim2 {
            return Err(ManifoldError::dimension_mismatch(
                self.dim2,
                comp2.len()
            ));
        }

        let mut combined = DVector::zeros(self.total_dim);
        combined.rows_mut(0, self.dim1).copy_from(comp1);
        combined.rows_mut(self.dim1, self.dim2).copy_from(comp2);
        
        Ok(combined)
    }

    /// Splits a product space vector into components using workspace.
    ///
    /// This version avoids allocations by using pre-allocated buffers.
    pub fn split_vector_mut(
        &self,
        vector: &DVector<T>,
        comp1: &mut DVector<T>,
        comp2: &mut DVector<T>,
    ) -> Result<()> {
        if vector.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                vector.len()
            ));
        }

        // Ensure output vectors have correct size
        if comp1.len() != self.dim1 {
            *comp1 = DVector::zeros(self.dim1);
        }
        if comp2.len() != self.dim2 {
            *comp2 = DVector::zeros(self.dim2);
        }

        comp1.copy_from(&vector.rows(0, self.dim1));
        comp2.copy_from(&vector.rows(self.dim1, self.dim2));
        
        Ok(())
    }

    /// Combines component vectors into result using workspace.
    ///
    /// This version avoids allocations by using a pre-allocated buffer.
    pub fn combine_vectors_mut(
        &self,
        comp1: &DVector<T>,
        comp2: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        if comp1.len() != self.dim1 {
            return Err(ManifoldError::dimension_mismatch(
                self.dim1,
                comp1.len()
            ));
        }
        if comp2.len() != self.dim2 {
            return Err(ManifoldError::dimension_mismatch(
                self.dim2,
                comp2.len()
            ));
        }

        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        result.rows_mut(0, self.dim1).copy_from(comp1);
        result.rows_mut(self.dim1, self.dim2).copy_from(comp2);
        
        Ok(())
    }

    /// Splits a point and returns components.
    ///
    /// Convenience method for splitting points.
    #[inline]
    pub fn split_point(&self, point: &DVector<T>) -> Result<(DVector<T>, DVector<T>)> {
        self.split_vector(point)
    }

    /// Combines point components.
    ///
    /// Convenience method for combining points.
    #[inline]
    pub fn combine_points(&self, p1: &DVector<T>, p2: &DVector<T>) -> Result<DVector<T>> {
        self.combine_vectors(p1, p2)
    }
}

impl<T, M1, M2> Manifold<T> for ProductStatic<T, M1, M2>
where
    T: Scalar,
    M1: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
    M2: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
{
    type Point = DVector<T>;
    type TangentVector = DVector<T>;

    fn name(&self) -> &str {
        "ProductStatic"
    }

    fn dimension(&self) -> usize {
        self.manifold1.dimension() + self.manifold2.dimension()
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.len() != self.total_dim {
            return false;
        }

        match self.split_vector(point) {
            Ok((p1, p2)) => {
                self.manifold1.is_point_on_manifold(&p1, tol) &&
                self.manifold2.is_point_on_manifold(&p2, tol)
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
        if point.len() != self.total_dim || vector.len() != self.total_dim {
            return false;
        }

        match (self.split_vector(point), self.split_vector(vector)) {
            (Ok((p1, p2)), Ok((v1, v2))) => {
                self.manifold1.is_vector_in_tangent_space(&p1, &v1, tol) &&
                self.manifold2.is_vector_in_tangent_space(&p2, &v2, tol)
            }
            _ => false,
        }
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, workspace: &mut Workspace<T>) {
        // Ensure result has correct size
        if result.len() != self.total_dim {
            *result = DVector::zeros(self.total_dim);
        }

        // Handle dimension mismatch by padding or truncating
        let padded_point = if point.len() != self.total_dim {
            let mut p = DVector::zeros(self.total_dim);
            let copy_len = point.len().min(self.total_dim);
            p.rows_mut(0, copy_len).copy_from(&point.rows(0, copy_len));
            p
        } else {
            point.clone()
        };

        // Split and project
        if let Ok((p1, p2)) = self.split_vector(&padded_point) {
            let mut proj1 = p1.clone();
            let mut proj2 = p2.clone();
            
            self.manifold1.project_point(&p1, &mut proj1, workspace);
            self.manifold2.project_point(&p2, &mut proj2, workspace);
            
            // Combine results
            result.rows_mut(0, self.dim1).copy_from(&proj1);
            result.rows_mut(self.dim1, self.dim2).copy_from(&proj2);
        }
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
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

        let (p1, p2) = self.split_vector(point)?;
        let (v1, v2) = self.split_vector(vector)?;

        let mut proj1 = v1.clone();
        let mut proj2 = v2.clone();
        
        self.manifold1.project_tangent(&p1, &v1, &mut proj1, workspace)?;
        self.manifold2.project_tangent(&p2, &v2, &mut proj2, workspace)?;
        
        self.combine_vectors_mut(&proj1, &proj2, result)?;
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
        workspace: &mut Workspace<T>,
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

        let (p1, p2) = self.split_vector(point)?;
        let (t1, t2) = self.split_vector(tangent)?;

        let mut ret1 = p1.clone();
        let mut ret2 = p2.clone();
        
        self.manifold1.retract(&p1, &t1, &mut ret1, workspace)?;
        self.manifold2.retract(&p2, &t2, &mut ret2, workspace)?;
        
        self.combine_vectors_mut(&ret1, &ret2, result)?;
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
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

        let (p1, p2) = self.split_vector(point)?;
        let (o1, o2) = self.split_vector(other)?;

        let mut tan1 = DVector::zeros(self.dim1);
        let mut tan2 = DVector::zeros(self.dim2);
        
        self.manifold1.inverse_retract(&p1, &o1, &mut tan1, workspace)?;
        self.manifold2.inverse_retract(&p2, &o2, &mut tan2, workspace)?;
        
        self.combine_vectors_mut(&tan1, &tan2, result)?;
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
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

        let (p1, p2) = self.split_vector(point)?;
        let (g1, g2) = self.split_vector(euclidean_grad)?;

        let mut rgrad1 = g1.clone();
        let mut rgrad2 = g2.clone();
        
        self.manifold1.euclidean_to_riemannian_gradient(&p1, &g1, &mut rgrad1, workspace)?;
        self.manifold2.euclidean_to_riemannian_gradient(&p2, &g2, &mut rgrad2, workspace)?;
        
        self.combine_vectors_mut(&rgrad1, &rgrad2, result)?;
        Ok(())
    }

    fn random_point(&self) -> Self::Point {
        let p1 = self.manifold1.random_point();
        let p2 = self.manifold2.random_point();
        self.combine_vectors(&p1, &p2).unwrap()
    }

    fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector, workspace: &mut Workspace<T>) -> Result<()> {
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

        let (p1, p2) = self.split_vector(point)?;

        let mut tan1 = DVector::zeros(self.dim1);
        let mut tan2 = DVector::zeros(self.dim2);
        
        self.manifold1.random_tangent(&p1, &mut tan1, workspace)?;
        self.manifold2.random_tangent(&p2, &mut tan2, workspace)?;
        
        self.combine_vectors_mut(&tan1, &tan2, result)?;
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, workspace: &mut Workspace<T>) -> Result<T> {
        if x.len() != self.total_dim || y.len() != self.total_dim {
            return Err(ManifoldError::dimension_mismatch(
                self.total_dim,
                x.len().max(y.len())
            ));
        }

        let (x1, x2) = self.split_vector(x)?;
        let (y1, y2) = self.split_vector(y)?;

        let d1 = self.manifold1.distance(&x1, &y1, workspace)?;
        let d2 = self.manifold2.distance(&x2, &y2, workspace)?;
        
        Ok(<T as Float>::sqrt(d1 * d1 + d2 * d2))
    }

    fn parallel_transport(
        &self,
        from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
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

        let (f1, f2) = self.split_vector(from)?;
        let (t1, t2) = self.split_vector(to)?;
        let (v1, v2) = self.split_vector(vector)?;

        let mut trans1 = v1.clone();
        let mut trans2 = v2.clone();
        
        self.manifold1.parallel_transport(&f1, &t1, &v1, &mut trans1, workspace)?;
        self.manifold2.parallel_transport(&f2, &t2, &v2, &mut trans2, workspace)?;
        
        self.combine_vectors_mut(&trans1, &trans2, result)?;
        Ok(())
    }

    fn has_exact_exp_log(&self) -> bool {
        self.manifold1.has_exact_exp_log() && self.manifold2.has_exact_exp_log()
    }

    fn is_flat(&self) -> bool {
        self.manifold1.is_flat() && self.manifold2.is_flat()
    }
}

/// Creates a static product manifold with type inference.
///
/// This is a convenience function that infers types automatically.
///
/// # Example
///
/// ```rust
/// use riemannopt_manifolds::{product_static, Sphere, Stiefel};
/// 
/// let sphere = Sphere::<f64>::new(3).unwrap();
/// let stiefel = Stiefel::<f64>::new(4, 2).unwrap();
/// let product = product_static(sphere, stiefel);
/// ```
pub fn product_static<T, M1, M2>(manifold1: M1, manifold2: M2) -> ProductStatic<T, M1, M2>
where
    T: Scalar,
    M1: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
    M2: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
{
    ProductStatic::new(manifold1, manifold2)
}

// Alias for backward compatibility
pub type ProductManifoldStatic<T, M1, M2> = ProductStatic<T, M1, M2>;

// Convenience function for backward compatibility
pub use product_static as product;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sphere;
    use approx::assert_relative_eq;
    use riemannopt_core::memory::workspace::Workspace;

    #[test]
    fn test_product_static_creation() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        let product = ProductStatic::new(sphere, sphere2);
        
        assert_eq!(product.component_dimensions(), (3, 4));
        assert_eq!(product.dimension(), 2 + 3); // S^2 + S^3
    }

    #[test]
    fn test_split_combine() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        let product = ProductStatic::new(sphere, sphere2);
        
        let point = product.random_point();
        
        // Split and recombine
        let (p1, p2) = product.split_point(&point).unwrap();
        assert_eq!(p1.len(), 3);
        assert_eq!(p2.len(), 3);
        
        let recombined = product.combine_points(&p1, &p2).unwrap();
        assert_relative_eq!(point, recombined, epsilon = 1e-14);
    }

    #[test]
    fn test_product_static_operations() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        let product = ProductStatic::new(sphere, sphere2);
        let mut workspace = Workspace::<f64>::new();
        
        // Test point validation
        let point = product.random_point();
        assert!(product.is_point_on_manifold(&point, 1e-6));
        
        // Test tangent projection
        let mut tangent = DVector::zeros(product.total_dim);
        product.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        assert!(product.is_vector_in_tangent_space(&point, &tangent, 1e-6));
        
        // Test retraction
        let scaled_tangent = tangent * 0.1;
        let mut retracted = DVector::zeros(product.total_dim);
        product.retract(&point, &scaled_tangent, &mut retracted, &mut workspace).unwrap();
        assert!(product.is_point_on_manifold(&retracted, 1e-6));
    }

    #[test]
    fn test_product_static_inner_product() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        let product = ProductStatic::new(sphere, sphere2);
        let mut workspace = Workspace::<f64>::new();
        
        let point = product.random_point();
        let mut u = DVector::zeros(product.total_dim);
        let mut v = DVector::zeros(product.total_dim);
        product.random_tangent(&point, &mut u, &mut workspace).unwrap();
        product.random_tangent(&point, &mut v, &mut workspace).unwrap();
        
        let ip_uv = product.inner_product(&point, &u, &v).unwrap();
        let ip_vu = product.inner_product(&point, &v, &u).unwrap();
        
        // Check symmetry
        assert_relative_eq!(ip_uv, ip_vu, epsilon = 1e-10);
    }

    #[test]
    fn test_product_static_distance() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        let product = ProductStatic::new(sphere, sphere2);
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
    fn test_product_static_properties() {
        let sphere1 = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        let product = ProductStatic::new(sphere1, sphere2);
        
        // Neither component is flat
        assert!(!product.is_flat());
        
        // Neither component has exact exp/log
        assert!(!product.has_exact_exp_log());
    }

    #[test]
    fn test_convenience_function() {
        let sphere = Sphere::<f64>::new(3).unwrap();
        let sphere2 = Sphere::<f64>::new(4).unwrap();
        let product = product_static(sphere, sphere2);
        
        assert_eq!(product.dimension(), 2 + 3); // S^2 + S^3
    }
}