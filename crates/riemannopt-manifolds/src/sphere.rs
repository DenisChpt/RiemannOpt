//! Sphere manifold S^{n-1} = {x in R^n : ||x|| = 1}
//!
//! The unit sphere is one of the most fundamental manifolds in optimization.
//! It naturally appears in:
//! - Principal Component Analysis (PCA)
//! - Independent Component Analysis (ICA)
//! - Sparse coding with unit norm constraints
//! - Eigenvalue problems
//! - Neural network weight normalization

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point},
    types::{DVector, Scalar},
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OVector, U1};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

/// The unit sphere S^{n-1} in R^n.
///
/// This manifold represents all unit vectors in n-dimensional Euclidean space.
/// Points on the sphere satisfy ||x|| = 1, and the tangent space at x consists
/// of all vectors orthogonal to x.
///
/// # Mathematical Properties
///
/// - **Dimension**: n-1 (for sphere in R^n)
/// - **Tangent space**: T_x S^{n-1} = {v in R^n : x^T v = 0}
/// - **Riemannian metric**: Inherited from Euclidean space (canonical metric)
/// - **Exponential map**: exp_x(v) = cos(||v||) x + sin(||v||) v/||v||
/// - **Logarithmic map**: log_x(y) = θ (y - cos(θ)x) / sin(θ), θ = arccos(x^T y)
///
/// # Applications
///
/// - **PCA**: Finding principal directions on the sphere
/// - **Matrix completion**: Unit norm constraints
/// - **Neural networks**: Weight normalization layers
/// - **Robotics**: Unit quaternions for rotations (S^3)
#[derive(Debug, Clone)]
pub struct Sphere {
    /// Ambient dimension (n)
    ambient_dim: usize,
}

impl Sphere {
    /// Creates a new sphere S^{n-1} embedded in R^n.
    ///
    /// # Arguments
    /// * `ambient_dim` - The dimension of the ambient space (n)
    ///
    /// # Returns
    /// A sphere manifold with intrinsic dimension n-1
    ///
    /// # Errors
    /// Returns an error if `ambient_dim` < 2
    pub fn new(ambient_dim: usize) -> Result<Self> {
        if ambient_dim < 2 {
            return Err(ManifoldError::invalid_point(
                "Sphere requires ambient dimension >= 2",
            ));
        }
        Ok(Self { ambient_dim })
    }

    /// Returns the ambient dimension (n)
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dim
    }

    /// Computes the exponential map at point x in direction v.
    ///
    /// The exponential map on the sphere has the closed form:
    /// exp_x(v) = cos(||v||) x + sin(||v||) v/||v||
    ///
    /// This moves along a great circle from x in direction v.
    pub fn exp_map<T, D>(&self, point: &Point<T, D>, tangent: &OVector<T, D>) -> Result<Point<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let tangent_norm = tangent.norm();
        
        if tangent_norm < T::epsilon() {
            // exp_x(0) = x
            return Ok(point.clone());
        }

        let cos_norm = <T as Float>::cos(tangent_norm);
        let sin_norm = <T as Float>::sin(tangent_norm);
        let normalized_tangent = tangent / tangent_norm;

        Ok(point * cos_norm + normalized_tangent * sin_norm)
    }

    /// Computes the logarithmic map from point x to point y.
    ///
    /// The logarithmic map on the sphere is:
    /// log_x(y) = θ (y - cos(θ)x) / sin(θ)
    /// where θ = arccos(clamp(x^T y, -1, 1))
    pub fn log_map<T, D>(&self, point: &Point<T, D>, other: &Point<T, D>) -> Result<OVector<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let inner_product = point.dot(other);
        
        // Clamp to avoid numerical issues with arccos
        let cos_theta = <T as Float>::max(
            <T as Float>::min(inner_product, T::one()),
            -T::one(),
        );
        
        let theta = <T as Float>::acos(cos_theta);
        
        if theta < T::epsilon() {
            // Points are very close, return zero vector
            return Ok(OVector::zeros_generic(point.shape_generic().0, U1));
        }
        
        let sin_theta = <T as Float>::sin(theta);
        
        if sin_theta < T::epsilon() {
            // Points are antipodal, log map is not unique
            // Return any tangent vector of length π
            let mut tangent = self.random_tangent_vector(point)?;
            let current_norm = tangent.norm();
            if current_norm > T::epsilon() {
                tangent = tangent * (<T as Scalar>::from_f64(std::f64::consts::PI) / current_norm);
            }
            return Ok(tangent);
        }
        
        let log_vector = (other - point * cos_theta) * (theta / sin_theta);
        Ok(log_vector)
    }

    /// Generates a random tangent vector at the given point.
    fn random_tangent_vector<T, D>(&self, point: &Point<T, D>) -> Result<OVector<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let mut rng = rand::thread_rng();
        let dim = point.len();
        
        // Generate random vector
        let mut random_vec = OVector::zeros_generic(point.shape_generic().0, U1);
        for i in 0..dim {
            let val: f64 = StandardNormal.sample(&mut rng);
            random_vec[i] = <T as Scalar>::from_f64(val);
        }
        
        // Project to tangent space: v - <v,x>x
        let inner = point.dot(&random_vec);
        let tangent = random_vec - point * inner;
        Ok(tangent)
    }
}

impl<T> Manifold<T, Dyn> for Sphere
where
    T: Scalar,
{
    fn name(&self) -> &str {
        "Sphere"
    }

    fn dimension(&self) -> usize {
        self.ambient_dim - 1
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tolerance: T) -> bool {
        if point.len() != self.ambient_dim {
            return false;
        }
        
        let norm_squared = point.norm_squared();
        <T as Float>::abs(norm_squared - T::one()) < tolerance
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        tolerance: T,
    ) -> bool {
        if point.len() != self.ambient_dim || vector.len() != self.ambient_dim {
            return false;
        }
        
        // Check if v ⊥ x: <v, x> = 0
        let inner_product = point.dot(vector);
        <T as Float>::abs(inner_product) < tolerance
    }

    fn project_point(&self, point: &DVector<T>) -> DVector<T> {
        let norm = point.norm();
        if norm < T::epsilon() {
            // Handle zero vector by creating a standard basis vector
            let mut result = DVector::zeros(self.ambient_dim);
            result[0] = T::one();
            result
        } else {
            point / norm
        }
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Project to tangent space: v - <v,x>x
        let inner = point.dot(vector);
        Ok(vector - point * inner)
    }

    fn inner_product(
        &self,
        _point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        // Use Euclidean inner product (canonical metric)
        Ok(u.dot(v))
    }

    fn retract(&self, point: &DVector<T>, tangent: &DVector<T>) -> Result<DVector<T>> {
        // Use exponential map as retraction (exact on sphere)
        self.exp_map(point, tangent)
    }

    fn inverse_retract(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Use logarithmic map as inverse retraction
        self.log_map(point, other)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        grad: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Project Euclidean gradient to tangent space
        self.project_tangent(point, grad)
    }

    fn random_point(&self) -> DVector<T> {
        let mut rng = rand::thread_rng();
        let mut point = DVector::zeros(self.ambient_dim);
        
        // Sample from standard normal and normalize
        for i in 0..self.ambient_dim {
            let val: f64 = StandardNormal.sample(&mut rng);
            point[i] = <T as Scalar>::from_f64(val);
        }
        
        self.project_point(&point)
    }

    fn random_tangent(&self, point: &DVector<T>) -> Result<DVector<T>> {
        self.random_tangent_vector(point)
    }

    fn has_exact_exp_log(&self) -> bool {
        true // Sphere has closed-form exponential and logarithmic maps
    }

    fn parallel_transport(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Parallel transport on sphere using the connection
        // Formula: P_{x->y}(v) = v - <v,y>y - <v,x>x + <x,y><v,x>y
        let inner_vx = vector.dot(from);
        let inner_vy = vector.dot(to);
        let inner_xy = from.dot(to);
        
        let transported = vector 
            - to * inner_vy 
            - from * inner_vx 
            + to * (inner_xy * inner_vx);
            
        Ok(transported)
    }

    fn distance(&self, point1: &DVector<T>, point2: &DVector<T>) -> Result<T> {
        let inner_product = point1.dot(point2);
        let cos_theta = <T as Float>::max(
            <T as Float>::min(inner_product, T::one()),
            -T::one(),
        );
        Ok(<T as Float>::acos(cos_theta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_sphere_creation() {
        let sphere = Sphere::new(3).unwrap();
        assert_eq!(<Sphere as Manifold<f64, Dyn>>::dimension(&sphere), 2);
        assert_eq!(sphere.ambient_dimension(), 3);
        
        // Test invalid dimension
        assert!(Sphere::new(1).is_err());
    }

    #[test]
    fn test_point_on_manifold() {
        let sphere = Sphere::new(3).unwrap();
        
        let on_sphere = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        assert!(sphere.is_point_on_manifold(&on_sphere, 1e-10));
        
        let not_on_sphere = DVector::from_vec(vec![2.0, 0.0, 0.0]);
        assert!(!sphere.is_point_on_manifold(&not_on_sphere, 1e-10));
    }

    #[test]
    fn test_tangent_space() {
        let sphere = Sphere::new(3).unwrap();
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        
        let tangent = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        assert!(sphere.is_vector_in_tangent_space(&point, &tangent, 1e-10));
        
        let not_tangent = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        assert!(!sphere.is_vector_in_tangent_space(&point, &not_tangent, 1e-10));
    }

    #[test]
    fn test_projection() {
        let sphere = Sphere::new(3).unwrap();
        
        let point = DVector::from_vec(vec![2.0, 0.0, 0.0]);
        let projected = sphere.project_point(&point);
        assert_relative_eq!(projected.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(projected[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_projection() {
        let sphere = Sphere::new(3).unwrap();
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let vector = DVector::from_vec(vec![0.5, 1.0, 0.0]);
        
        let projected = sphere.project_tangent(&point, &vector).unwrap();
        assert_relative_eq!(point.dot(&projected), 0.0, epsilon = 1e-10);
        assert_relative_eq!(projected[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(projected[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_log_maps() {
        let sphere = Sphere::new(3).unwrap();
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let tangent = DVector::from_vec(vec![0.0, 0.5, 0.0]);
        
        // Test exp then log
        let exp_result = sphere.exp_map(&point, &tangent).unwrap();
        assert_relative_eq!(exp_result.norm(), 1.0, epsilon = 1e-10);
        
        let log_result = sphere.log_map(&point, &exp_result).unwrap();
        assert_relative_eq!(&log_result, &tangent, epsilon = 1e-10);
    }

    #[test]
    fn test_retraction_properties() {
        let sphere = Sphere::new(3).unwrap();
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let zero_tangent = DVector::zeros(3);
        
        // Test centering: R(x, 0) = x
        let retracted = sphere.retract(&point, &zero_tangent).unwrap();
        assert_relative_eq!(&retracted, &point, epsilon = 1e-10);
    }

    #[test]
    fn test_random_generation() {
        let sphere = Sphere::new(3).unwrap();
        
        // Test random point
        let random_point = sphere.random_point();
        assert!(sphere.is_point_on_manifold(&random_point, 1e-10));
        
        // Test random tangent
        let tangent = sphere.random_tangent(&random_point).unwrap();
        assert!(sphere.is_vector_in_tangent_space(&random_point, &tangent, 1e-10));
    }

    #[test]
    fn test_distance() {
        let sphere = Sphere::new(3).unwrap();
        let point1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let point2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        
        let distance = sphere.distance(&point1, &point2).unwrap();
        assert_relative_eq!(distance, std::f64::consts::PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_transport() {
        let sphere = Sphere::new(3).unwrap();
        let from = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let to = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let vector = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        
        let transported = sphere.parallel_transport(&from, &to, &vector).unwrap();
        
        // Check it's still in tangent space at destination
        assert!(sphere.is_vector_in_tangent_space(&to, &transported, 1e-10));
        
        // For this specific case, should preserve the vector
        assert_relative_eq!(&transported, &vector, epsilon = 1e-10);
    }
}