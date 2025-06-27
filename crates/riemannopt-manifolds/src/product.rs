//! Product manifold M1 x M2 x ... x Mn
//!
//! The product manifold allows combining multiple manifolds into a single
//! manifold structure. This is fundamental for optimization problems that
//! involve variables living on different geometric spaces simultaneously.
//! Common applications include:
//! - Multi-task learning with heterogeneous constraints
//! - Neural networks with mixed geometric layers
//! - Coupled optimization problems
//! - Decomposition methods
//! - Multi-modal representation learning

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::Scalar,
    memory::Workspace,
};
use nalgebra::{DVector, Dyn};


/// A product manifold combining two manifolds M1 x M2.
///
/// This structure represents the Cartesian product of two Riemannian manifolds,
/// where points are represented as concatenated vectors from both manifolds.
/// The Riemannian structure is inherited naturally from the component manifolds.
///
/// # Mathematical Properties
///
/// - **Dimension**: dim(M1) + dim(M2)
/// - **Tangent space**: T_(x1,x2) (M1 x M2) = T_x1 M1 x T_x2 M2
/// - **Riemannian metric**: <(u1,u2), (v1,v2)> = <u1,v1>_M1 + <u2,v2>_M2
/// - **Geodesics**: Component-wise geodesics
///
/// # Implementation Notes
///
/// This implementation uses a specialized approach that maintains type safety
/// while allowing runtime composition of different manifold types. The manifolds
/// are stored as concrete types that implement the Manifold trait for f64.
///
/// # Applications
///
/// - **Neural networks**: Mixing Euclidean and non-Euclidean layers
/// - **Multi-task learning**: Different constraints for different tasks
/// - **Decomposition**: Separate optimization of different components
/// - **Robotics**: Position and orientation constraints
#[derive(Debug)]
pub struct ProductManifold {
    /// First component manifold (stored as trait object)
    manifold1: Box<dyn Manifold<f64, Dyn>>,
    /// Second component manifold (stored as trait object)
    manifold2: Box<dyn Manifold<f64, Dyn>>,
    /// Dimension of first manifold
    dim1: usize,
    /// Dimension of second manifold
    dim2: usize,
}

impl ProductManifold {
    /// Helper function to convert DVector<T> to DVector<f64>
    fn to_f64_vector<T: Scalar>(vector: &DVector<T>) -> DVector<f64> {
        let mut result = DVector::<f64>::zeros(vector.len());
        for (i, val) in vector.iter().enumerate() {
            result[i] = <T as Scalar>::to_f64(*val);
        }
        result
    }

    /// Helper function to convert DVector<f64> to DVector<T>
    fn from_f64_vector<T: Scalar>(vector: &DVector<f64>) -> DVector<T> {
        let mut result = DVector::<T>::zeros(vector.len());
        for (i, val) in vector.iter().enumerate() {
            result[i] = <T as Scalar>::from_f64(*val);
        }
        result
    }

    /// Creates a new product manifold M1 x M2.
    ///
    /// # Arguments
    /// * `manifold1` - First component manifold
    /// * `manifold2` - Second component manifold
    ///
    /// # Returns
    /// A product manifold with combined dimension
    ///
    /// # Examples
    /// ```
    /// use riemannopt_manifolds::{ProductManifold, Sphere, SPD};
    /// 
    /// let sphere = Sphere::new(3).unwrap();  // S^2
    /// let spd = SPD::new(2).unwrap();        // SPD(2)
    /// let product = ProductManifold::new(Box::new(sphere), Box::new(spd));
    /// ```
    pub fn new(
        manifold1: Box<dyn Manifold<f64, Dyn>>, 
        manifold2: Box<dyn Manifold<f64, Dyn>>
    ) -> Self {
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
        }
    }

    /// Returns the dimensions of both component manifolds.
    pub fn component_dimensions(&self) -> (usize, usize) {
        (self.dim1, self.dim2)
    }

    /// Returns references to the component manifolds.
    pub fn components(&self) -> (&dyn Manifold<f64, Dyn>, &dyn Manifold<f64, Dyn>) {
        (self.manifold1.as_ref(), self.manifold2.as_ref())
    }

    /// Splits a product space vector into components.
    ///
    /// # Arguments
    /// * `vector` - Combined vector from product space
    ///
    /// # Returns
    /// Tuple of (component1, component2) vectors
    ///
    /// # Errors
    /// Returns error if vector dimension doesn't match expected product dimension
    fn split_vector<T>(&self, vector: &DVector<T>) -> Result<(DVector<T>, DVector<T>)>
    where
        T: Scalar,
    {
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
    ///
    /// # Arguments
    /// * `component1` - Vector from first manifold
    /// * `component2` - Vector from second manifold
    ///
    /// # Returns
    /// Combined vector in product space
    ///
    /// # Errors
    /// Returns error if component dimensions don't match expected manifold dimensions
    fn combine_vectors<T>(&self, component1: &DVector<T>, component2: &DVector<T>) -> Result<DVector<T>>
    where
        T: Scalar,
    {
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

}

impl<T> Manifold<T, Dyn> for ProductManifold
where
    T: Scalar,
{
    fn name(&self) -> &str {
        "Product"
    }

    fn dimension(&self) -> usize {
        self.dim1 + self.dim2
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tolerance: T) -> bool {
        // Convert to f64 for trait object interface
        let tolerance_f64 = tolerance.to_f64();
        let point_f64 = Self::to_f64_vector(point);
        
        match self.split_vector(&point_f64) {
            Ok((comp1, comp2)) => {
                self.manifold1.is_point_on_manifold(&comp1, tolerance_f64) &&
                self.manifold2.is_point_on_manifold(&comp2, tolerance_f64)
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
        let tolerance_f64 = tolerance.to_f64();
        let point_f64 = Self::to_f64_vector(point);
        let vector_f64 = Self::to_f64_vector(vector);

        match (self.split_vector(&point_f64), self.split_vector(&vector_f64)) {
            (Ok((point1, point2)), Ok((vec1, vec2))) => {
                self.manifold1.is_vector_in_tangent_space(&point1, &vec1, tolerance_f64) &&
                self.manifold2.is_vector_in_tangent_space(&point2, &vec2, tolerance_f64)
            }
            _ => false,
        }
    }

    fn project_point(&self, point: &DVector<T>, result: &mut DVector<T>, _workspace: &mut Workspace<T>) {
        // Ensure result has correct size
        let expected_size = self.dim1 + self.dim2;
        if result.len() != expected_size {
            result.resize_vertically_mut(expected_size, T::zero());
        }
        
        let point_f64 = Self::to_f64_vector(point);

        let (comp1, comp2) = if point.len() != self.dim1 + self.dim2 {
            // Handle dimension mismatch gracefully
            let mut padded_f64 = DVector::zeros(self.dim1 + self.dim2);
            let copy_len = point.len().min(self.dim1 + self.dim2);
            
            for i in 0..copy_len {
                padded_f64[i] = point_f64[i];
            }
            
            self.split_vector(&padded_f64).expect("Padded vector should have correct dimensions")
        } else {
            self.split_vector(&point_f64).expect("Vector should have correct dimensions")
        };

        let mut proj1 = DVector::zeros(self.dim1);
        let mut proj2 = DVector::zeros(self.dim2);
        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        self.manifold1.project_point(&comp1, &mut proj1, &mut workspace1);
        self.manifold2.project_point(&comp2, &mut proj2, &mut workspace2);

        let combined_f64 = self.combine_vectors(&proj1, &proj2)
            .expect("Projected components should have correct dimensions");
        
        let converted = Self::from_f64_vector(&combined_f64);
        result.copy_from(&converted);
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.dim1 + self.dim2;
        if result.len() != expected_size {
            result.resize_vertically_mut(expected_size, T::zero());
        }
        
        let point_f64 = Self::to_f64_vector(point);
        let vector_f64 = Self::to_f64_vector(vector);

        let (point1, point2) = self.split_vector(&point_f64)?;
        let (vec1, vec2) = self.split_vector(&vector_f64)?;

        let mut proj1 = DVector::zeros(self.dim1);
        let mut proj2 = DVector::zeros(self.dim2);
        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        self.manifold1.project_tangent(&point1, &vec1, &mut proj1, &mut workspace1)?;
        self.manifold2.project_tangent(&point2, &vec2, &mut proj2, &mut workspace2)?;

        let combined_f64 = self.combine_vectors(&proj1, &proj2)?;
        let converted = Self::from_f64_vector(&combined_f64);
        result.copy_from(&converted);
        Ok(())
    }

    fn inner_product(
        &self,
        point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        let point_f64 = Self::to_f64_vector(point);
        let u_f64 = Self::to_f64_vector(u);
        let v_f64 = Self::to_f64_vector(v);

        let (point1, point2) = self.split_vector(&point_f64)?;
        let (u1, u2) = self.split_vector(&u_f64)?;
        let (v1, v2) = self.split_vector(&v_f64)?;

        let inner1 = self.manifold1.inner_product(&point1, &u1, &v1)?;
        let inner2 = self.manifold2.inner_product(&point2, &u2, &v2)?;

        Ok(<T as Scalar>::from_f64(inner1 + inner2))
    }

    fn retract(&self, point: &DVector<T>, tangent: &DVector<T>, result: &mut DVector<T>, _workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.dim1 + self.dim2;
        if result.len() != expected_size {
            result.resize_vertically_mut(expected_size, T::zero());
        }
        
        let point_f64 = Self::to_f64_vector(point);
        let tangent_f64 = Self::to_f64_vector(tangent);

        let (point1, point2) = self.split_vector(&point_f64)?;
        let (tangent1, tangent2) = self.split_vector(&tangent_f64)?;

        let mut retract1 = DVector::zeros(self.dim1);
        let mut retract2 = DVector::zeros(self.dim2);
        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        self.manifold1.retract(&point1, &tangent1, &mut retract1, &mut workspace1)?;
        self.manifold2.retract(&point2, &tangent2, &mut retract2, &mut workspace2)?;

        let combined_f64 = self.combine_vectors(&retract1, &retract2)?;
        let converted = Self::from_f64_vector(&combined_f64);
        result.copy_from(&converted);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
        result: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.dim1 + self.dim2;
        if result.len() != expected_size {
            result.resize_vertically_mut(expected_size, T::zero());
        }
        
        let point_f64 = Self::to_f64_vector(point);
        let other_f64 = Self::to_f64_vector(other);

        let (point1, point2) = self.split_vector(&point_f64)?;
        let (other1, other2) = self.split_vector(&other_f64)?;

        let mut inv_retract1 = DVector::zeros(self.dim1);
        let mut inv_retract2 = DVector::zeros(self.dim2);
        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        self.manifold1.inverse_retract(&point1, &other1, &mut inv_retract1, &mut workspace1)?;
        self.manifold2.inverse_retract(&point2, &other2, &mut inv_retract2, &mut workspace2)?;

        let combined_f64 = self.combine_vectors(&inv_retract1, &inv_retract2)?;
        let converted = Self::from_f64_vector(&combined_f64);
        result.copy_from(&converted);
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        grad: &DVector<T>,
        result: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.dim1 + self.dim2;
        if result.len() != expected_size {
            result.resize_vertically_mut(expected_size, T::zero());
        }
        
        let point_f64 = Self::to_f64_vector(point);
        let grad_f64 = Self::to_f64_vector(grad);

        let (point1, point2) = self.split_vector(&point_f64)?;
        let (grad1, grad2) = self.split_vector(&grad_f64)?;

        let mut riem_grad1 = DVector::zeros(self.dim1);
        let mut riem_grad2 = DVector::zeros(self.dim2);
        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        self.manifold1.euclidean_to_riemannian_gradient(&point1, &grad1, &mut riem_grad1, &mut workspace1)?;
        self.manifold2.euclidean_to_riemannian_gradient(&point2, &grad2, &mut riem_grad2, &mut workspace2)?;

        let combined_f64 = self.combine_vectors(&riem_grad1, &riem_grad2)?;
        let converted = Self::from_f64_vector(&combined_f64);
        result.copy_from(&converted);
        Ok(())
    }

    fn random_point(&self) -> DVector<T> {
        let point1 = self.manifold1.random_point();
        let point2 = self.manifold2.random_point();

        let combined_f64 = self.combine_vectors(&point1, &point2)
            .expect("Random points should have correct dimensions");
        
        Self::from_f64_vector(&combined_f64)
    }

    fn random_tangent(&self, point: &DVector<T>, result: &mut DVector<T>, _workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.dim1 + self.dim2;
        if result.len() != expected_size {
            result.resize_vertically_mut(expected_size, T::zero());
        }
        
        let point_f64 = Self::to_f64_vector(point);

        let (point1, point2) = self.split_vector(&point_f64)?;

        let mut tangent1 = DVector::zeros(self.dim1);
        let mut tangent2 = DVector::zeros(self.dim2);
        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        self.manifold1.random_tangent(&point1, &mut tangent1, &mut workspace1)?;
        self.manifold2.random_tangent(&point2, &mut tangent2, &mut workspace2)?;

        let combined_f64 = self.combine_vectors(&tangent1, &tangent2)?;
        let converted = Self::from_f64_vector(&combined_f64);
        result.copy_from(&converted);
        Ok(())
    }

    fn has_exact_exp_log(&self) -> bool {
        // Product has exact exp/log if both components do
        self.manifold1.has_exact_exp_log() && self.manifold2.has_exact_exp_log()
    }

    fn parallel_transport(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        let expected_size = self.dim1 + self.dim2;
        if result.len() != expected_size {
            result.resize_vertically_mut(expected_size, T::zero());
        }
        
        let from_f64 = Self::to_f64_vector(from);
        let to_f64 = Self::to_f64_vector(to);
        let vector_f64 = Self::to_f64_vector(vector);

        let (from1, from2) = self.split_vector(&from_f64)?;
        let (to1, to2) = self.split_vector(&to_f64)?;
        let (vector1, vector2) = self.split_vector(&vector_f64)?;

        let mut transport1 = DVector::zeros(self.dim1);
        let mut transport2 = DVector::zeros(self.dim2);
        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        self.manifold1.parallel_transport(&from1, &to1, &vector1, &mut transport1, &mut workspace1)?;
        self.manifold2.parallel_transport(&from2, &to2, &vector2, &mut transport2, &mut workspace2)?;

        let combined_f64 = self.combine_vectors(&transport1, &transport2)?;
        let converted = Self::from_f64_vector(&combined_f64);
        result.copy_from(&converted);
        Ok(())
    }

    fn distance(&self, point1: &DVector<T>, point2: &DVector<T>, _workspace: &mut Workspace<T>) -> Result<T> {
        let point1_f64 = Self::to_f64_vector(point1);
        let point2_f64 = Self::to_f64_vector(point2);

        let (p1_comp1, p1_comp2) = self.split_vector(&point1_f64)?;
        let (p2_comp1, p2_comp2) = self.split_vector(&point2_f64)?;

        let mut workspace1 = Workspace::new();
        let mut workspace2 = Workspace::new();
        let dist1 = self.manifold1.distance(&p1_comp1, &p2_comp1, &mut workspace1)?;
        let dist2 = self.manifold2.distance(&p1_comp2, &p2_comp2, &mut workspace2)?;

        // Use Euclidean norm of component distances
        let total_dist = (dist1 * dist1 + dist2 * dist2).sqrt();
        
        Ok(<T as Scalar>::from_f64(total_dist))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Sphere, SPD};
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_product_creation() {
        let sphere = Sphere::new(3).unwrap();  // S^2, dimension 2
        let spd = SPD::new(2).unwrap();        // SPD(2), dimension 3
        
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));
        
        assert_eq!(<ProductManifold as Manifold<f64, Dyn>>::dimension(&product), 6); // 3 + 3 = 6 (ambient dimensions)
        assert_eq!(product.component_dimensions(), (3, 3));
    }

    #[test]
    fn test_vector_split_combine() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        // Test splitting and combining
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (comp1, comp2) = product.split_vector(&vector).unwrap();
        
        assert_eq!(comp1, DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(comp2, DVector::from_vec(vec![4.0, 5.0, 6.0]));
        
        let recombined = product.combine_vectors(&comp1, &comp2).unwrap();
        assert_eq!(recombined, vector);
    }

    #[test]
    fn test_point_on_manifold() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        
        // Generate valid points on each manifold before moving into product
        let sphere_point = sphere.random_point();
        let spd_point = spd.random_point();
        
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));
        let combined_point = product.combine_vectors(&sphere_point, &spd_point).unwrap();
        
        // Should be on product manifold
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_point_on_manifold(&product, &combined_point, 1e-12));
    }

    #[test]
    fn test_tangent_space() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        let point = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let ambient_dim = <ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product);
        let mut tangent = DVector::zeros(ambient_dim);
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::random_tangent(&product, &point, &mut tangent, &mut workspace).unwrap();
        
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_vector_in_tangent_space(&product, &point, &tangent, 1e-12));
    }

    #[test]
    fn test_projections() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        // Test point projection
        let bad_point = DVector::from_vec(vec![5.0, 5.0, 1.0, 2.0, -1.0, 3.0]); // Invalid for both manifolds
        let ambient_dim = <ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product);
        let mut projected_point = DVector::zeros(ambient_dim);
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::project_point(&product, &bad_point, &mut projected_point, &mut workspace);
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_point_on_manifold(&product, &projected_point, 1e-12));

        // Test tangent projection
        let point = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let bad_tangent = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut projected_tangent = DVector::zeros(<ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product));
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::project_tangent(&product, &point, &bad_tangent, &mut projected_tangent, &mut workspace).unwrap();
        
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_vector_in_tangent_space(&product, &point, &projected_tangent, 1e-12));
    }

    #[test]
    fn test_inner_product() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        let point = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let mut u = DVector::zeros(<ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product));
        let mut v = DVector::zeros(<ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product));
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::random_tangent(&product, &point, &mut u, &mut workspace).unwrap();
        <ProductManifold as Manifold<f64, Dyn>>::random_tangent(&product, &point, &mut v, &mut workspace).unwrap();

        // Inner product should work
        let inner = <ProductManifold as Manifold<f64, Dyn>>::inner_product(&product, &point, &u, &v).unwrap();
        assert!(inner.is_finite());

        // Inner product with itself should be positive
        let self_inner = <ProductManifold as Manifold<f64, Dyn>>::inner_product(&product, &point, &u, &u).unwrap();
        assert!(self_inner >= 0.0);
    }

    #[test]
    fn test_retraction() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        let point = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let mut tangent = DVector::zeros(<ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product));
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::random_tangent(&product, &point, &mut tangent, &mut workspace).unwrap();

        let mut retracted = DVector::zeros(<ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product));
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::retract(&product, &point, &tangent, &mut retracted, &mut workspace).unwrap();
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_point_on_manifold(&product, &retracted, 1e-12));

        // Test centering property: R(x, 0) = x
        let zero_tangent = DVector::zeros(6);
        let mut centered = DVector::zeros(<ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product));
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::retract(&product, &point, &zero_tangent, &mut centered, &mut workspace).unwrap();
        assert_relative_eq!(&centered, &point, epsilon = 1e-12);
    }

    #[test]
    fn test_distance_properties() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        let point1 = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let point2 = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);

        // Distance should be non-negative
        let mut workspace = Workspace::new();
        let dist = <ProductManifold as Manifold<f64, Dyn>>::distance(&product, &point1, &point2, &mut workspace).unwrap();
        assert!(dist >= 0.0);

        // Distance to self should be zero
        let mut workspace = Workspace::new();
        let self_dist = <ProductManifold as Manifold<f64, Dyn>>::distance(&product, &point1, &point1, &mut workspace).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-7);

        // Distance should be symmetric
        let mut workspace = Workspace::new();
        let dist_rev = <ProductManifold as Manifold<f64, Dyn>>::distance(&product, &point2, &point1, &mut workspace).unwrap();
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-8);
    }

    #[test]
    fn test_exp_log_properties() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        // Product should have exact exp/log if both sphere and SPD do
        assert!(<ProductManifold as Manifold<f64, Dyn>>::has_exact_exp_log(&product)); // Both sphere and SPD have exact exp/log
    }

    #[test]
    fn test_parallel_transport() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        let from = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let to = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let ambient_dim = <ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product);
        let mut vector = DVector::zeros(ambient_dim);
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::random_tangent(&product, &from, &mut vector, &mut workspace).unwrap();

        let mut transported = DVector::zeros(ambient_dim);
        <ProductManifold as Manifold<f64, Dyn>>::parallel_transport(&product, &from, &to, &vector, &mut transported, &mut workspace).unwrap();
        
        // Transported vector should be in tangent space at destination
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_vector_in_tangent_space(&product, &to, &transported, 1e-12));
    }

    #[test]
    fn test_gradient_conversion() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        let point = <ProductManifold as Manifold<f64, Dyn>>::random_point(&product);
        let euclidean_grad = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let ambient_dim = <ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product);
        let mut riemannian_grad = DVector::zeros(ambient_dim);
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::euclidean_to_riemannian_gradient(&product, &point, &euclidean_grad, &mut riemannian_grad, &mut workspace).unwrap();

        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_vector_in_tangent_space(&product, &point, &riemannian_grad, 1e-12));
    }

    #[test]
    fn test_different_manifold_combinations() {
        // Test Sphere x Sphere
        let sphere1 = Sphere::new(3).unwrap();
        let sphere2 = Sphere::new(4).unwrap();
        let sphere_product = ProductManifold::new(Box::new(sphere1), Box::new(sphere2));
        
        assert_eq!(<ProductManifold as Manifold<f64, Dyn>>::dimension(&sphere_product), 7); // 3 + 4 = 7 (ambient dimensions)
        
        let point = <ProductManifold as Manifold<f64, Dyn>>::random_point(&sphere_product);
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_point_on_manifold(&sphere_product, &point, 1e-12));
    }

    #[test]
    fn test_dimension_mismatch_handling() {
        let sphere = Sphere::new(3).unwrap();
        let spd = SPD::new(2).unwrap();
        let product = ProductManifold::new(Box::new(sphere), Box::new(spd));

        // Test with wrong dimension
        let wrong_dim_vector = DVector::from_vec(vec![1.0, 2.0, 3.0]); // Should be length 6
        let result = product.split_vector(&wrong_dim_vector);
        assert!(result.is_err());

        // Test projection with wrong dimension
        let ambient_dim = <ProductManifold as Manifold<f64, Dyn>>::ambient_dimension(&product);
        let mut projected = DVector::zeros(ambient_dim);
        let mut workspace = Workspace::new();
        <ProductManifold as Manifold<f64, Dyn>>::project_point(&product, &wrong_dim_vector, &mut projected, &mut workspace);
        assert_eq!(projected.len(), 6); // Should be corrected to proper dimension
        assert!(<ProductManifold as Manifold<f64, Dyn>>::is_point_on_manifold(&product, &projected, 1e-12));
    }
}