//! Static product manifold M1 x M2 with compile-time dispatch
//!
//! This module provides a generic product manifold implementation that uses
//! static dispatch instead of dynamic dispatch for better performance.

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::{Scalar, DVector},
    memory::workspace::Workspace,
};
use nalgebra::Dyn;
use std::marker::PhantomData;

/// A product manifold combining two manifolds M1 x M2 with static dispatch.
///
/// This structure represents the Cartesian product of two Riemannian manifolds,
/// where points are represented as concatenated vectors from both manifolds.
/// Unlike the dynamic version, this uses concrete types for better performance.
///
/// # Type Parameters
///
/// - `T`: The scalar type (f32 or f64)
/// - `M1`: First manifold type
/// - `M2`: Second manifold type
///
/// # Mathematical Properties
///
/// - **Dimension**: dim(M1) + dim(M2)
/// - **Tangent space**: T_(x1,x2) (M1 x M2) = T_x1 M1 x T_x2 M2
/// - **Riemannian metric**: <(u1,u2), (v1,v2)> = <u1,v1>_M1 + <u2,v2>_M2
/// - **Geodesics**: Component-wise geodesics
///
/// # Performance Benefits
///
/// This static version provides:
/// - **Zero-cost abstractions**: No virtual function calls
/// - **Inline optimization**: Compiler can inline manifold operations
/// - **No type conversions**: Works with any scalar type T
/// - **Better cache locality**: Operations are more predictable
///
/// # Examples
///
/// ```rust
/// use riemannopt_manifolds::{ProductManifoldStatic, Sphere, Stiefel};
/// use riemannopt_core::manifold::Manifold;
/// use nalgebra::Dyn;
///
/// // Create a product of sphere and Stiefel manifold
/// let sphere = Sphere::new(3).unwrap();  // S^2 in R^3
/// let stiefel = Stiefel::new(4, 2).unwrap();  // St(4,2)
/// let product = ProductManifoldStatic::new(sphere, stiefel);
///
/// // The dimension is the sum of component dimensions
/// assert_eq!(<ProductManifoldStatic<_, _, _> as Manifold<f64, Dyn>>::dimension(&product), 2 + 5);
///
/// // Generate a random point on the product manifold
/// let point = <ProductManifoldStatic<_, _, _> as Manifold<f64, Dyn>>::random_point(&product);
/// assert!(<ProductManifoldStatic<_, _, _> as Manifold<f64, Dyn>>::is_point_on_manifold(&product, &point, 1e-10));
/// ```
#[derive(Debug, Clone)]
pub struct ProductManifoldStatic<T, M1, M2>
where
    T: Scalar,
    M1: Manifold<T, Dyn>,
    M2: Manifold<T, Dyn>,
{
    /// First component manifold
    pub manifold1: M1,
    /// Second component manifold
    pub manifold2: M2,
    /// Dimension of first manifold's representation
    dim1: usize,
    /// Dimension of second manifold's representation
    dim2: usize,
    _phantom: PhantomData<T>,
}

/// Creates a product manifold with static dispatch.
///
/// This is a convenience function that infers the types automatically.
///
/// # Examples
/// ```rust
/// use riemannopt_manifolds::{product, Sphere, Stiefel};
/// 
/// let sphere = Sphere::new(3).unwrap();
/// let stiefel = Stiefel::new(4, 2).unwrap();
/// let product = product::<f64, _, _>(sphere, stiefel);
/// ```
pub fn product<T, M1, M2>(manifold1: M1, manifold2: M2) -> ProductManifoldStatic<T, M1, M2>
where
    T: Scalar,
    M1: Manifold<T, Dyn>,
    M2: Manifold<T, Dyn>,
{
    ProductManifoldStatic::new(manifold1, manifold2)
}

impl<T, M1, M2> ProductManifoldStatic<T, M1, M2>
where
    T: Scalar,
    M1: Manifold<T, Dyn>,
    M2: Manifold<T, Dyn>,
{
    /// Creates a new product manifold M1 x M2.
    ///
    /// # Arguments
    /// * `manifold1` - First component manifold
    /// * `manifold2` - Second component manifold
    ///
    /// # Returns
    /// A product manifold with combined dimension
    pub fn new(manifold1: M1, manifold2: M2) -> Self {
        // Get actual ambient dimensions by testing random points
        let test_point1 = manifold1.random_point();
        let test_point2 = manifold2.random_point();
        let dim1 = test_point1.len();
        let dim2 = test_point2.len();
        
        Self {
            manifold1,
            manifold2,
            dim1,
            dim2,
            _phantom: PhantomData,
        }
    }

    /// Returns the dimensions of both component manifolds.
    pub fn component_dimensions(&self) -> (usize, usize) {
        (self.dim1, self.dim2)
    }

    /// Returns references to the component manifolds.
    pub fn components(&self) -> (&M1, &M2) {
        (&self.manifold1, &self.manifold2)
    }

    /// Splits a product space vector into components.
    fn split_vector(&self, vector: &DVector<T>) -> Result<(DVector<T>, DVector<T>)> {
        if vector.len() != self.dim1 + self.dim2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("Expected dimension {}", self.dim1 + self.dim2),
                format!("Got dimension {}", vector.len()),
            ));
        }

        let component1 = vector.rows(0, self.dim1).into_owned();
        let component2 = vector.rows(self.dim1, self.dim2).into_owned();

        Ok((component1, component2))
    }

    /// Combines component vectors into a product space vector.
    fn combine_vectors(&self, component1: &DVector<T>, component2: &DVector<T>) -> Result<DVector<T>> {
        if component1.len() != self.dim1 {
            return Err(ManifoldError::dimension_mismatch(
                format!("First component expected dimension {}", self.dim1),
                format!("Got dimension {}", component1.len()),
            ));
        }
        if component2.len() != self.dim2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("Second component expected dimension {}", self.dim2),
                format!("Got dimension {}", component2.len()),
            ));
        }

        let mut combined = DVector::<T>::zeros(self.dim1 + self.dim2);
        combined.rows_mut(0, self.dim1).copy_from(component1);
        combined.rows_mut(self.dim1, self.dim2).copy_from(component2);

        Ok(combined)
    }

    /// Splits a product space vector into components using workspace buffers.
    fn split_vector_with_workspace(
        &self, 
        vector: &DVector<T>,
        comp1: &mut DVector<T>,
        comp2: &mut DVector<T>,
    ) -> Result<()> {
        if vector.len() != self.dim1 + self.dim2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("Expected dimension {}", self.dim1 + self.dim2),
                format!("Got dimension {}", vector.len()),
            ));
        }

        // Ensure output vectors have correct size
        if comp1.len() != self.dim1 {
            *comp1 = DVector::zeros(self.dim1);
        }
        if comp2.len() != self.dim2 {
            *comp2 = DVector::zeros(self.dim2);
        }

        // Copy data
        comp1.copy_from(&vector.rows(0, self.dim1));
        comp2.copy_from(&vector.rows(self.dim1, self.dim2));

        Ok(())
    }

    /// Combines component vectors into result using workspace.
    fn combine_vectors_with_workspace(
        &self,
        comp1: &DVector<T>,
        comp2: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        if comp1.len() != self.dim1 {
            return Err(ManifoldError::dimension_mismatch(
                format!("First component expected dimension {}", self.dim1),
                format!("Got dimension {}", comp1.len()),
            ));
        }
        if comp2.len() != self.dim2 {
            return Err(ManifoldError::dimension_mismatch(
                format!("Second component expected dimension {}", self.dim2),
                format!("Got dimension {}", comp2.len()),
            ));
        }

        // Ensure result has correct size
        if result.len() != self.dim1 + self.dim2 {
            *result = DVector::zeros(self.dim1 + self.dim2);
        }

        result.rows_mut(0, self.dim1).copy_from(comp1);
        result.rows_mut(self.dim1, self.dim2).copy_from(comp2);

        Ok(())
    }
}

impl<T, M1, M2> Manifold<T, Dyn> for ProductManifoldStatic<T, M1, M2>
where
    T: Scalar,
    M1: Manifold<T, Dyn>,
    M2: Manifold<T, Dyn>,
{
    fn name(&self) -> &str {
        "ProductStatic"
    }

    fn dimension(&self) -> usize {
        self.manifold1.dimension() + self.manifold2.dimension()
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tolerance: T) -> bool {
        match self.split_vector(point) {
            Ok((p1, p2)) => {
                self.manifold1.is_point_on_manifold(&p1, tolerance) &&
                self.manifold2.is_point_on_manifold(&p2, tolerance)
            }
            Err(_) => false,
        }
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        tolerance: T,
    ) -> bool {
        match (self.split_vector(point), self.split_vector(vector)) {
            (Ok((p1, p2)), Ok((v1, v2))) => {
                self.manifold1.is_vector_in_tangent_space(&p1, &v1, tolerance) &&
                self.manifold2.is_vector_in_tangent_space(&p2, &v2, tolerance)
            }
            _ => false,
        }
    }

    fn project_point(&self, point: &DVector<T>) -> DVector<T> {
        let (p1, p2) = match self.split_vector(point) {
            Ok(split) => split,
            Err(_) => {
                // Handle dimension mismatch by creating zero vectors
                (DVector::zeros(self.dim1), DVector::zeros(self.dim2))
            }
        };

        let proj1 = self.manifold1.project_point(&p1);
        let proj2 = self.manifold2.project_point(&p2);

        self.combine_vectors(&proj1, &proj2).unwrap()
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        let (p1, p2) = self.split_vector(point)?;
        let (v1, v2) = self.split_vector(vector)?;

        let proj1 = self.manifold1.project_tangent(&p1, &v1)?;
        let proj2 = self.manifold2.project_tangent(&p2, &v2)?;

        self.combine_vectors(&proj1, &proj2)
    }

    fn inner_product(
        &self,
        point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        let (p1, p2) = self.split_vector(point)?;
        let (u1, u2) = self.split_vector(u)?;
        let (v1, v2) = self.split_vector(v)?;

        let inner1 = self.manifold1.inner_product(&p1, &u1, &v1)?;
        let inner2 = self.manifold2.inner_product(&p2, &u2, &v2)?;

        Ok(inner1 + inner2)
    }

    fn retract(&self, point: &DVector<T>, tangent: &DVector<T>) -> Result<DVector<T>> {
        let (p1, p2) = self.split_vector(point)?;
        let (t1, t2) = self.split_vector(tangent)?;

        let ret1 = self.manifold1.retract(&p1, &t1)?;
        let ret2 = self.manifold2.retract(&p2, &t2)?;

        self.combine_vectors(&ret1, &ret2)
    }

    fn inverse_retract(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
    ) -> Result<DVector<T>> {
        let (p1, p2) = self.split_vector(point)?;
        let (o1, o2) = self.split_vector(other)?;

        let inv1 = self.manifold1.inverse_retract(&p1, &o1)?;
        let inv2 = self.manifold2.inverse_retract(&p2, &o2)?;

        self.combine_vectors(&inv1, &inv2)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        grad: &DVector<T>,
    ) -> Result<DVector<T>> {
        let (p1, p2) = self.split_vector(point)?;
        let (g1, g2) = self.split_vector(grad)?;

        let riem1 = self.manifold1.euclidean_to_riemannian_gradient(&p1, &g1)?;
        let riem2 = self.manifold2.euclidean_to_riemannian_gradient(&p2, &g2)?;

        self.combine_vectors(&riem1, &riem2)
    }

    fn parallel_transport(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        let (from1, from2) = self.split_vector(from)?;
        let (to1, to2) = self.split_vector(to)?;
        let (v1, v2) = self.split_vector(vector)?;

        let trans1 = self.manifold1.parallel_transport(&from1, &to1, &v1)?;
        let trans2 = self.manifold2.parallel_transport(&from2, &to2, &v2)?;

        self.combine_vectors(&trans1, &trans2)
    }

    fn random_point(&self) -> DVector<T> {
        let p1 = self.manifold1.random_point();
        let p2 = self.manifold2.random_point();
        self.combine_vectors(&p1, &p2).unwrap()
    }

    fn random_tangent(&self, point: &DVector<T>) -> Result<DVector<T>> {
        let (p1, p2) = self.split_vector(point)?;

        let t1 = self.manifold1.random_tangent(&p1)?;
        let t2 = self.manifold2.random_tangent(&p2)?;

        self.combine_vectors(&t1, &t2)
    }

    fn distance(&self, x: &DVector<T>, y: &DVector<T>) -> Result<T> {
        let (x1, x2) = self.split_vector(x)?;
        let (y1, y2) = self.split_vector(y)?;

        let d1 = self.manifold1.distance(&x1, &y1)?;
        let d2 = self.manifold2.distance(&x2, &y2)?;

        // Euclidean distance in product space
        Ok(num_traits::Float::sqrt(d1 * d1 + d2 * d2))
    }

    // ========================================================================
    // Workspace-based methods for zero-allocation operations
    // ========================================================================

    fn project_tangent_with_workspace(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Get temporary buffers
        let mut p1 = workspace.acquire_temp_vector(self.dim1);
        let mut p2 = workspace.acquire_temp_vector(self.dim2);
        let mut v1 = workspace.acquire_temp_vector(self.dim1);
        let mut v2 = workspace.acquire_temp_vector(self.dim2);
        let mut r1 = workspace.acquire_temp_vector(self.dim1);
        let mut r2 = workspace.acquire_temp_vector(self.dim2);

        // Split inputs
        self.split_vector_with_workspace(point, &mut p1, &mut p2)?;
        self.split_vector_with_workspace(vector, &mut v1, &mut v2)?;

        // Project on each manifold
        self.manifold1.project_tangent_with_workspace(&p1, &v1, &mut r1, workspace)?;
        self.manifold2.project_tangent_with_workspace(&p2, &v2, &mut r2, workspace)?;

        // Combine results
        self.combine_vectors_with_workspace(&r1, &r2, result)?;

        Ok(())
    }

    fn retract_with_workspace(
        &self,
        point: &DVector<T>,
        tangent: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Get temporary buffers
        let mut p1 = workspace.acquire_temp_vector(self.dim1);
        let mut p2 = workspace.acquire_temp_vector(self.dim2);
        let mut t1 = workspace.acquire_temp_vector(self.dim1);
        let mut t2 = workspace.acquire_temp_vector(self.dim2);
        let mut r1 = workspace.acquire_temp_vector(self.dim1);
        let mut r2 = workspace.acquire_temp_vector(self.dim2);

        // Split inputs
        self.split_vector_with_workspace(point, &mut p1, &mut p2)?;
        self.split_vector_with_workspace(tangent, &mut t1, &mut t2)?;

        // Retract on each manifold
        self.manifold1.retract_with_workspace(&p1, &t1, &mut r1, workspace)?;
        self.manifold2.retract_with_workspace(&p2, &t2, &mut r2, workspace)?;

        // Combine results
        self.combine_vectors_with_workspace(&r1, &r2, result)?;

        Ok(())
    }

    fn inverse_retract_with_workspace(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Get temporary buffers
        let mut p1 = workspace.acquire_temp_vector(self.dim1);
        let mut p2 = workspace.acquire_temp_vector(self.dim2);
        let mut o1 = workspace.acquire_temp_vector(self.dim1);
        let mut o2 = workspace.acquire_temp_vector(self.dim2);
        let mut r1 = workspace.acquire_temp_vector(self.dim1);
        let mut r2 = workspace.acquire_temp_vector(self.dim2);

        // Split inputs
        self.split_vector_with_workspace(point, &mut p1, &mut p2)?;
        self.split_vector_with_workspace(other, &mut o1, &mut o2)?;

        // Inverse retract on each manifold
        self.manifold1.inverse_retract_with_workspace(&p1, &o1, &mut r1, workspace)?;
        self.manifold2.inverse_retract_with_workspace(&p2, &o2, &mut r2, workspace)?;

        // Combine results
        self.combine_vectors_with_workspace(&r1, &r2, result)?;

        Ok(())
    }

    fn euclidean_to_riemannian_gradient_with_workspace(
        &self,
        point: &DVector<T>,
        euclidean_grad: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Get temporary buffers
        let mut p1 = workspace.acquire_temp_vector(self.dim1);
        let mut p2 = workspace.acquire_temp_vector(self.dim2);
        let mut g1 = workspace.acquire_temp_vector(self.dim1);
        let mut g2 = workspace.acquire_temp_vector(self.dim2);
        let mut r1 = workspace.acquire_temp_vector(self.dim1);
        let mut r2 = workspace.acquire_temp_vector(self.dim2);

        // Split inputs
        self.split_vector_with_workspace(point, &mut p1, &mut p2)?;
        self.split_vector_with_workspace(euclidean_grad, &mut g1, &mut g2)?;

        // Convert gradients on each manifold
        self.manifold1.euclidean_to_riemannian_gradient_with_workspace(&p1, &g1, &mut r1, workspace)?;
        self.manifold2.euclidean_to_riemannian_gradient_with_workspace(&p2, &g2, &mut r2, workspace)?;

        // Combine results
        self.combine_vectors_with_workspace(&r1, &r2, result)?;

        Ok(())
    }

    fn parallel_transport_with_workspace(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Get temporary buffers
        let mut f1 = workspace.acquire_temp_vector(self.dim1);
        let mut f2 = workspace.acquire_temp_vector(self.dim2);
        let mut t1 = workspace.acquire_temp_vector(self.dim1);
        let mut t2 = workspace.acquire_temp_vector(self.dim2);
        let mut v1 = workspace.acquire_temp_vector(self.dim1);
        let mut v2 = workspace.acquire_temp_vector(self.dim2);
        let mut r1 = workspace.acquire_temp_vector(self.dim1);
        let mut r2 = workspace.acquire_temp_vector(self.dim2);

        // Split inputs
        self.split_vector_with_workspace(from, &mut f1, &mut f2)?;
        self.split_vector_with_workspace(to, &mut t1, &mut t2)?;
        self.split_vector_with_workspace(vector, &mut v1, &mut v2)?;

        // Transport on each manifold
        self.manifold1.parallel_transport_with_workspace(&f1, &t1, &v1, &mut r1, workspace)?;
        self.manifold2.parallel_transport_with_workspace(&f2, &t2, &v2, &mut r2, workspace)?;

        // Combine results
        self.combine_vectors_with_workspace(&r1, &r2, result)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Sphere, Stiefel};

    #[test]
    fn test_product_static_creation() {
        let sphere = Sphere::new(3).unwrap();
        let stiefel = Stiefel::new(4, 2).unwrap();
        let product = ProductManifoldStatic::<f64, _, _>::new(sphere, stiefel);
        
        assert_eq!(product.dimension(), 2 + 5); // S^2 has dim 2, St(4,2) has dim 5
        assert_eq!(product.component_dimensions(), (3, 8)); // ambient dimensions
    }

    #[test]
    fn test_product_static_operations() {
        let sphere = Sphere::new(3).unwrap();
        let stiefel = Stiefel::new(3, 2).unwrap();
        let product = ProductManifoldStatic::<f64, _, _>::new(sphere, stiefel);
        
        // Test random point
        let point = product.random_point();
        assert!(product.is_point_on_manifold(&point, 1e-10));
        
        // Test tangent projection
        let tangent = product.random_tangent(&point).unwrap();
        let projected = product.project_tangent(&point, &tangent).unwrap();
        assert!(product.is_vector_in_tangent_space(&point, &projected, 1e-10));
    }

    #[test]
    fn test_product_static_workspace() {
        // Test with Sphere and Stiefel manifolds
        let sphere = Sphere::new(3).unwrap();
        let stiefel = Stiefel::new(4, 2).unwrap();
        let product = ProductManifoldStatic::<f64, _, _>::new(sphere, stiefel);
        
        let point = product.random_point();
        assert!(product.is_point_on_manifold(&point, 1e-10), "Initial point not on manifold");
        
        // Scale down tangent vector to avoid numerical issues
        let mut tangent = product.random_tangent(&point).unwrap();
        tangent *= 0.01; // Small step
        assert!(product.is_vector_in_tangent_space(&point, &tangent, 1e-10), "Tangent not in tangent space");
        
        let mut workspace = Workspace::<f64>::new();
        let mut result = DVector::zeros(point.len());
        
        // Test workspace-based retraction
        product.retract_with_workspace(&point, &tangent, &mut result, &mut workspace).unwrap();
        assert!(product.is_point_on_manifold(&result, 1e-9), "Result not on manifold");
        
        // Test other workspace methods
        let mut proj_result = DVector::zeros(tangent.len());
        product.project_tangent_with_workspace(&point, &tangent, &mut proj_result, &mut workspace).unwrap();
        assert!(product.is_vector_in_tangent_space(&point, &proj_result, 1e-9));
    }

    #[test]
    fn test_product_helper_function() {
        let sphere = Sphere::new(3).unwrap();
        let stiefel = Stiefel::new(4, 2).unwrap();
        
        // Test the helper function
        let prod: ProductManifoldStatic<f64, _, _> = product(sphere, stiefel);
        
        assert_eq!(prod.dimension(), 2 + 5); // S^2 has dim 2, St(4,2) has dim 5
        
        // Test operations
        let point = prod.random_point();
        assert!(prod.is_point_on_manifold(&point, 1e-10));
    }
}