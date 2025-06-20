//! # Unit Sphere Manifold S^{n-1}
//!
//! The unit sphere S^{n-1} = {x ∈ ℝⁿ : ‖x‖ = 1} is the set of all unit vectors
//! in n-dimensional Euclidean space. It is one of the most fundamental and
//! well-studied manifolds in differential geometry and optimization.
//!
//! ## Mathematical Structure
//!
//! The unit sphere is a (n-1)-dimensional manifold embedded in ℝⁿ with:
//!
//! - **Intrinsic dimension**: n-1
//! - **Ambient dimension**: n  
//! - **Sectional curvature**: +1 (constant positive curvature)
//! - **Diameter**: π (maximum geodesic distance)
//! - **Volume**: 2π^{n/2}/Γ(n/2) (surface area)
//!
//! ## Geometric Properties
//!
//! ### Tangent Space
//! The tangent space at point x ∈ S^{n-1} is:
//! ```text
//! T_x S^{n-1} = {v ∈ ℝⁿ : ⟨v, x⟩ = 0}
//! ```
//! This is the orthogonal complement of x in ℝⁿ.
//!
//! ### Exponential Map
//! The exponential map has the closed form:
//! ```text
//! exp_x(v) = cos(‖v‖) x + sin(‖v‖) v/‖v‖
//! ```
//! This follows great circles on the sphere.
//!
//! ### Logarithmic Map
//! The logarithmic map (inverse of exponential) is:
//! ```text
//! log_x(y) = θ (y - cos(θ)x) / sin(θ)
//! ```
//! where θ = arccos(⟨x, y⟩) is the geodesic distance.
//!
//! ### Riemannian Distance
//! The geodesic distance between points x, y ∈ S^{n-1} is:
//! ```text
//! d(x, y) = arccos(⟨x, y⟩)
//! ```
//!
//! ## Optimization Applications
//!
//! The sphere manifold appears naturally in many optimization problems:
//!
//! ### 1. Principal Component Analysis (PCA)
//! Finding the first principal component:
//! ```text
//! max_{x ∈ S^{n-1}} x^T Σ x
//! ```
//! where Σ is the covariance matrix.
//!
//! ### 2. Sparse Dictionary Learning
//! Learning unit-norm dictionary atoms:
//! ```text
//! min_{D, X} ‖Y - DX‖_F^2 + λ‖X‖_1
//! s.t. ‖d_i‖ = 1 for all columns d_i of D
//! ```
//!
//! ### 3. Spherical Regression
//! Regression with directional data:
//! ```text
//! min_{β ∈ S^{p-1}} Σᵢ L(yᵢ, ⟨xᵢ, β⟩)
//! ```
//!
//! ### 4. Neural Network Weight Normalization
//! Constraining weights to unit norm:
//! ```text
//! w_normalized = w / ‖w‖
//! ```
//!
//! ## Implementation Features
//!
//! This implementation provides:
//! - Exact exponential and logarithmic maps
//! - Efficient projection operations
//! - Parallel transport along geodesics
//! - Uniform random sampling
//! - Numerical stability for edge cases
//!
//! ## Example Usage
//!
//! ```rust
//! use riemannopt_manifolds::Sphere;
//! use riemannopt_core::manifold::Manifold;
//! use nalgebra::{DVector, Dyn};
//!
//! // Create 3D sphere (2-dimensional manifold)
//! let sphere = Sphere::new(3).unwrap();
//!
//! // Generate random point
//! let x: DVector<f64> = <Sphere as Manifold<f64, Dyn>>::random_point(&sphere);
//! assert!((x.norm() - 1.0f64).abs() < 1e-10f64);
//!
//! // Project arbitrary vector to sphere
//! let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);
//! let projected: DVector<f64> = <Sphere as Manifold<f64, Dyn>>::project_point(&sphere, &y);
//! assert!((projected.norm() - 1.0f64).abs() < 1e-10f64);
//!
//! // Work with tangent vectors
//! let tangent = DVector::from_vec(vec![0.0, 1.0, 0.0]);
//! let tangent_proj: DVector<f64> = <Sphere as Manifold<f64, Dyn>>::project_tangent(&sphere, &x, &tangent).unwrap();
//! assert!(x.dot(&tangent_proj).abs() < 1e-10f64);
//! ```

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point},
    types::{DVector, Scalar},
    memory::workspace::Workspace,
    compute::{get_dispatcher, SimdBackend, specialized::small_dim::*},
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OVector, U1, Vector2, Vector3};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::iter::Sum;


/// The unit sphere S^{n-1} = {x ∈ ℝⁿ : ‖x‖ = 1}.
///
/// The unit sphere is a fundamental Riemannian manifold consisting of all unit vectors
/// in n-dimensional Euclidean space. It is a (n-1)-dimensional manifold embedded in ℝⁿ.
///
/// # Mathematical Definition
///
/// The unit sphere S^{n-1} is defined as:
/// ```text
/// S^{n-1} = {x ∈ ℝⁿ : ‖x‖₂ = 1}
/// ```
///
/// Key geometric properties:
/// - **Intrinsic dimension**: n-1
/// - **Ambient dimension**: n
/// - **Sectional curvature**: K = +1 (constant positive curvature)
/// - **Diameter**: π (maximum geodesic distance)
/// - **Injectivity radius**: π (radius of geodesic balls without cut locus)
///
/// # Tangent Space Structure
///
/// The tangent space at point x ∈ S^{n-1} is the orthogonal complement:
/// ```text
/// T_x S^{n-1} = {v ∈ ℝⁿ : ⟨v, x⟩ = 0}
/// ```
///
/// The Riemannian metric is the restriction of the Euclidean inner product:
/// ```text
/// g_x(u, v) = ⟨u, v⟩_ℝⁿ  for u, v ∈ T_x S^{n-1}
/// ```
///
/// # Geodesics and Exponential Map
///
/// Geodesics on the sphere are great circles. The exponential map has the closed form:
/// ```text
/// exp_x(v) = cos(‖v‖) x + sin(‖v‖) v/‖v‖
/// ```
///
/// This provides exact geodesics without numerical integration.
///
/// # Optimization Context
///
/// The sphere manifold is crucial in constrained optimization problems where variables
/// must have unit norm. Common applications include:
///
/// ## Principal Component Analysis
/// Maximize variance: max_{x ∈ S^{n-1}} x^T Σ x
///
/// ## Dictionary Learning  
/// Learn normalized dictionary atoms for sparse coding
///
/// ## Directional Statistics
/// Analyze data on spheres (e.g., wind directions, protein conformations)
///
/// ## Neural Networks
/// Weight normalization and spherical embeddings
///
/// # Implementation Notes
///
/// This implementation provides:
/// - **Exact algorithms**: Closed-form exponential/logarithmic maps
/// - **Numerical stability**: Careful handling of antipodal points and near-zero vectors
/// - **Efficient projections**: Fast normalization-based projection
/// - **Random sampling**: Uniform distribution via Gaussian normalization
///
/// # Examples
///
/// ## Basic Usage
/// ```rust
/// use riemannopt_manifolds::Sphere;
/// use riemannopt_core::manifold::Manifold;
/// use nalgebra::{DVector, Dyn};
///
/// // Create unit sphere in ℝ³ (2D manifold)
/// let sphere = Sphere::new(3).unwrap();
/// assert_eq!(<Sphere as Manifold<f64, Dyn>>::dimension(&sphere), 2);
/// assert_eq!(sphere.ambient_dimension(), 3);
///
/// // Generate random unit vector
/// let x: DVector<f64> = <Sphere as Manifold<f64, Dyn>>::random_point(&sphere);
/// assert!((x.norm() - 1.0f64).abs() < 1e-12f64);
/// ```
///
/// ## Projection Operations
/// ```rust
/// # use riemannopt_manifolds::Sphere;
/// # use riemannopt_core::manifold::Manifold;
/// # use nalgebra::{DVector, Dyn};
/// # let sphere = Sphere::new(3).unwrap();
///
/// // Project arbitrary point to sphere
/// let point = DVector::from_vec(vec![3.0, 4.0, 0.0]);
/// let projected: DVector<f64> = <Sphere as Manifold<f64, Dyn>>::project_point(&sphere, &point);
/// assert!((projected.norm() - 1.0f64).abs() < 1e-12f64);
/// assert!((projected - &point / point.norm()).norm() < 1e-12f64);
///
/// // Project vector to tangent space
/// let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
/// let v = DVector::from_vec(vec![0.5, 1.0, 2.0]);
/// let v_tangent: DVector<f64> = <Sphere as Manifold<f64, Dyn>>::project_tangent(&sphere, &x, &v).unwrap();
/// assert!(x.dot(&v_tangent).abs() < 1e-12f64); // Orthogonal to x
/// ```
///
/// ## Geodesics
/// ```rust
/// # use riemannopt_manifolds::Sphere;
/// # use nalgebra::DVector;
/// # let sphere = Sphere::new(3).unwrap();
///
/// let x = DVector::<f64>::from_vec(vec![1.0, 0.0, 0.0]);
/// let v = DVector::<f64>::from_vec(vec![0.0, 1.0, 0.0]);  // Tangent vector
///
/// // Move along geodesic
/// let y: DVector<f64> = sphere.exp_map(&x, &v).unwrap();
/// assert!((y.norm() - 1.0f64).abs() < 1e-12f64);
///
/// // Inverse operation
/// let v_recovered: DVector<f64> = sphere.log_map(&x, &y).unwrap();
/// assert!((v - v_recovered).norm() < 1e-12f64);
/// ```
#[derive(Debug, Clone)]
pub struct Sphere {
    /// Ambient dimension n (sphere S^{n-1} embedded in ℝⁿ)
    ambient_dim: usize,
}

impl Sphere {
    /// Creates a new unit sphere S^{n-1} embedded in ℝⁿ.
    ///
    /// # Arguments
    /// * `ambient_dim` - The dimension n of the ambient Euclidean space ℝⁿ
    ///
    /// # Returns
    /// A sphere manifold with:
    /// - Intrinsic dimension: n-1
    /// - Ambient dimension: n
    /// - Constant sectional curvature: +1
    ///
    /// # Errors
    /// Returns `ManifoldError::InvalidPoint` if `ambient_dim < 2`, since
    /// a sphere requires at least 2 ambient dimensions to be well-defined.
    ///
    /// # Mathematical Background
    /// The sphere S^{n-1} has intrinsic dimension n-1 because locally it looks like
    /// (n-1)-dimensional Euclidean space. Common examples:
    /// - S¹ (circle): 1D manifold in ℝ²
    /// - S² (sphere): 2D manifold in ℝ³  
    /// - S³ (3-sphere): 3D manifold in ℝ⁴ (used for quaternions)
    ///
    /// # Examples
    /// ```rust
    /// use riemannopt_manifolds::Sphere;
    /// use riemannopt_core::manifold::Manifold;
    /// use nalgebra::Dyn;
    ///
    /// // Standard 2D sphere (surface of ball in 3D)
    /// let sphere = Sphere::new(3).unwrap();
    /// assert_eq!(<Sphere as Manifold<f64, Dyn>>::dimension(&sphere), 2);
    /// assert_eq!(sphere.ambient_dimension(), 3);
    ///
    /// // Circle (1D manifold in 2D)
    /// let circle = Sphere::new(2).unwrap();
    /// assert_eq!(<Sphere as Manifold<f64, Dyn>>::dimension(&circle), 1);
    ///
    /// // Higher dimensional sphere
    /// let sphere_4d = Sphere::new(5).unwrap();
    /// assert_eq!(<Sphere as Manifold<f64, Dyn>>::dimension(&sphere_4d), 4);
    ///
    /// // Error case: too small dimension
    /// assert!(Sphere::new(1).is_err());
    /// ```
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

    /// Computes the exponential map exp_x(v) at point x in direction v.
    ///
    /// The exponential map on the sphere follows geodesics (great circles) and has
    /// the exact closed form:
    /// ```text
    /// exp_x(v) = cos(‖v‖) x + sin(‖v‖) v/‖v‖
    /// ```
    ///
    /// # Mathematical Properties
    /// - **Domain**: T_x S^{n-1} (tangent space at x)
    /// - **Codomain**: S^{n-1} (the sphere)
    /// - **Geodesic property**: exp_x(tv) traces a geodesic for t ∈ ℝ
    /// - **Distance**: ‖v‖ equals the geodesic distance from x to exp_x(v)
    /// - **Periodicity**: exp_x(v + 2πv/‖v‖) = exp_x(v)
    ///
    /// # Special Cases
    /// - **Identity**: exp_x(0) = x
    /// - **Antipodal**: exp_x(πv/‖v‖) = -x (opposite point)
    /// - **Small v**: exp_x(v) ≈ x + v (first-order approximation)
    ///
    /// # Geometric Interpretation
    /// Starting at point x, move distance ‖v‖ along the great circle in direction v.
    /// The great circle lies in the 2D plane spanned by {x, v}.
    ///
    /// # Arguments
    /// * `point` - Base point x ∈ S^{n-1}
    /// * `tangent` - Tangent vector v ∈ T_x S^{n-1}
    ///
    /// # Returns
    /// Point exp_x(v) ∈ S^{n-1} reached by following the geodesic
    ///
    /// # Examples
    /// ```rust
    /// use riemannopt_manifolds::Sphere;
    /// use nalgebra::DVector;
    ///
    /// let sphere = Sphere::new(3).unwrap();
    /// let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    /// let v = DVector::from_vec(vec![0.0, std::f64::consts::PI/2.0, 0.0]);
    ///
    /// let y = sphere.exp_map(&x, &v).unwrap();
    /// // Should reach (0, 1, 0) after π/2 rotation
    /// assert!((y[0] - 0.0).abs() < 1e-12);
    /// assert!((y[1] - 1.0).abs() < 1e-12);
    /// assert!((y[2] - 0.0).abs() < 1e-12);
    /// ```
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

    /// Computes the logarithmic map log_x(y) from point x to point y.
    ///
    /// The logarithmic map is the inverse of the exponential map, giving the
    /// tangent vector v ∈ T_x S^{n-1} such that exp_x(v) = y.
    ///
    /// # Mathematical Formula
    /// ```text
    /// log_x(y) = θ (y - cos(θ)x) / sin(θ)
    /// ```
    /// where θ = arccos(⟨x, y⟩) is the geodesic distance.
    ///
    /// # Mathematical Properties
    /// - **Domain**: S^{n-1} \ {-x} (sphere minus antipodal point)
    /// - **Codomain**: T_x S^{n-1} (tangent space at x)
    /// - **Inverse**: exp_x(log_x(y)) = y for y ≠ -x
    /// - **Distance**: ‖log_x(y)‖ = d(x, y) (geodesic distance)
    /// - **Direction**: log_x(y)/‖log_x(y)‖ points toward y
    ///
    /// # Special Cases
    /// - **Identity**: log_x(x) = 0
    /// - **Antipodal**: log_x(-x) is not unique (any tangent vector of length π)
    /// - **Close points**: log_x(y) ≈ y - x when ‖y - x‖ is small
    ///
    /// # Numerical Considerations
    /// - Input inner product ⟨x, y⟩ is clamped to [-1, 1] to avoid numerical errors
    /// - For antipodal points (sin(θ) ≈ 0), returns an arbitrary tangent vector of length π
    /// - For nearby points (θ ≈ 0), returns zero vector
    ///
    /// # Arguments
    /// * `point` - Base point x ∈ S^{n-1}
    /// * `other` - Target point y ∈ S^{n-1}
    ///
    /// # Returns
    /// Tangent vector log_x(y) ∈ T_x S^{n-1}
    ///
    /// # Examples
    /// ```rust
    /// use riemannopt_manifolds::Sphere;
    /// use nalgebra::DVector;
    ///
    /// let sphere = Sphere::new(3).unwrap();
    /// let x = DVector::<f64>::from_vec(vec![1.0, 0.0, 0.0]);
    /// let y = DVector::<f64>::from_vec(vec![0.0, 1.0, 0.0]);
    ///
    /// let log_xy: DVector<f64> = sphere.log_map(&x, &y).unwrap();
    /// // Distance should be π/2
    /// assert!((log_xy.norm() - std::f64::consts::PI/2.0).abs() < 1e-12f64);
    /// // Should be orthogonal to x
    /// let dot_product: f64 = x.dot(&log_xy);
    /// assert!(dot_product.abs() < 1e-12f64);
    /// ```
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
    T: Scalar + Sum,
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
        
        // Use SIMD dispatcher for efficient norm computation
        let dispatcher = get_dispatcher::<T>();
        let norm_squared = dispatcher.norm_squared(point);
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
        let dispatcher = get_dispatcher::<T>();
        let inner_product = dispatcher.dot_product(point, vector);
        <T as Float>::abs(inner_product) < tolerance
    }

    fn project_point(&self, point: &DVector<T>) -> DVector<T> {
        // Dispatch to specialized implementations for small dimensions
        match self.ambient_dim {
            2 => {
                // Use specialized 2D operations
                let ops = Ops2D;
                let slice = point.as_slice();
                let norm = ops.norm_small(slice);
                
                if norm < T::epsilon() {
                    // Handle zero vector
                    DVector::from_vec(vec![T::one(), T::zero()])
                } else {
                    let inv_norm = T::one() / norm;
                    DVector::from_vec(vec![slice[0] * inv_norm, slice[1] * inv_norm])
                }
            }
            3 => {
                // Use specialized 3D operations
                let ops = Ops3D;
                let slice = point.as_slice();
                let norm = ops.norm_small(slice);
                
                if norm < T::epsilon() {
                    // Handle zero vector
                    DVector::from_vec(vec![T::one(), T::zero(), T::zero()])
                } else {
                    let inv_norm = T::one() / norm;
                    DVector::from_vec(vec![
                        slice[0] * inv_norm,
                        slice[1] * inv_norm,
                        slice[2] * inv_norm,
                    ])
                }
            }
            4 => {
                // Use specialized 4D operations
                let ops = Ops4D;
                let slice = point.as_slice();
                let norm = ops.norm_small(slice);
                
                if norm < T::epsilon() {
                    // Handle zero vector
                    DVector::from_vec(vec![T::one(), T::zero(), T::zero(), T::zero()])
                } else {
                    let inv_norm = T::one() / norm;
                    DVector::from_vec(vec![
                        slice[0] * inv_norm,
                        slice[1] * inv_norm,
                        slice[2] * inv_norm,
                        slice[3] * inv_norm,
                    ])
                }
            }
            _ => {
                // Generic implementation for higher dimensions
                let dispatcher = get_dispatcher::<T>();
                let norm = dispatcher.norm(point);
                
                if norm < T::epsilon() {
                    // Handle zero vector by creating a standard basis vector
                    // Choose e₁ = (1, 0, ..., 0) as canonical representative
                    let mut result = DVector::zeros(self.ambient_dim);
                    result[0] = T::one();
                    result
                } else {
                    // Standard projection: Π(x) = x / ‖x‖
                    // The dispatcher will handle SIMD/parallel optimizations internally
                    let mut result = point.clone();
                    dispatcher.scale(&mut result, T::one() / norm);
                    result
                }
            }
        }
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        // Dispatch to specialized implementations for small dimensions
        match self.ambient_dim {
            2 => {
                // Use specialized 2D implementation
                let p = Vector2::new(point[0], point[1]);
                let mut v = Vector2::new(vector[0], vector[1]);
                project_tangent_sphere_2d(&p, &mut v);
                Ok(DVector::from_vec(vec![v[0], v[1]]))
            }
            3 => {
                // Use specialized 3D implementation
                let p = Vector3::new(point[0], point[1], point[2]);
                let mut v = Vector3::new(vector[0], vector[1], vector[2]);
                project_tangent_sphere_3d(&p, &mut v);
                Ok(DVector::from_vec(vec![v[0], v[1], v[2]]))
            }
            _ => {
                // Generic implementation for higher dimensions
                // Orthogonal projection to tangent space T_x S^{n-1}
                // Formula: P_x(v) = v - ⟨v, x⟩x
                // This removes the component of v in the normal direction x
                let dispatcher = get_dispatcher::<T>();
                let inner = dispatcher.dot_product(point, vector);
                
                let mut result = vector.clone();
                dispatcher.axpy(-inner, point, &mut result);
                Ok(result)
            }
        }
    }

    fn inner_product(
        &self,
        _point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        // Canonical Riemannian metric: restriction of Euclidean inner product
        // g_x(u, v) = ⟨u, v⟩_ℝⁿ for u, v ∈ T_x S^{n-1}
        
        // Use SIMD dispatcher which handles parallel/SIMD optimizations internally
        let dispatcher = get_dispatcher::<T>();
        Ok(dispatcher.dot_product(u, v))
    }

    fn retract(&self, point: &DVector<T>, tangent: &DVector<T>) -> Result<DVector<T>> {
        // Dispatch to specialized implementations for small dimensions
        match self.ambient_dim {
            2 => {
                // Use specialized 2D implementation
                let p = Vector2::new(point[0], point[1]);
                let v = Vector2::new(tangent[0], tangent[1]);
                let result = retract_sphere_2d(&p, &v, T::one());
                Ok(DVector::from_vec(vec![result[0], result[1]]))
            }
            3 => {
                // Use specialized 3D implementation
                let p = Vector3::new(point[0], point[1], point[2]);
                let v = Vector3::new(tangent[0], tangent[1], tangent[2]);
                let result = retract_sphere_3d(&p, &v, T::one());
                Ok(DVector::from_vec(vec![result[0], result[1], result[2]]))
            }
            _ => {
                // Use exponential map as retraction (exact on sphere)
                // R_x(v) = exp_x(v) = cos(‖v‖)x + sin(‖v‖)v/‖v‖
                // This is the optimal retraction for the sphere
                self.exp_map(point, tangent)
            }
        }
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
        // For embedded manifolds with canonical metric,
        // Riemannian gradient = projection of Euclidean gradient
        // grad_ℳ f(x) = P_x(∇f(x))
        self.project_tangent(point, grad)
    }

    fn random_point(&self) -> DVector<T> {
        let mut rng = rand::thread_rng();
        let mut point = DVector::zeros(self.ambient_dim);
        
        // Sample from multivariate standard normal N(0, I)
        // and normalize to get uniform distribution on sphere
        // This is the Muller method for sampling on spheres
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
        // Parallel transport on sphere preserves angles and lengths
        // Formula for sphere: P_{x→y}(v) = v - ⟨v,y⟩y - ⟨v,x⟩x + ⟨x,y⟩⟨v,x⟩y
        // This transports v ∈ T_x S^{n-1} to T_y S^{n-1}
        let dispatcher = get_dispatcher::<T>();
        let inner_vx = dispatcher.dot_product(vector, from);
        let inner_vy = dispatcher.dot_product(vector, to);
        let inner_xy = dispatcher.dot_product(from, to);
        
        let mut transported = vector.clone();
        dispatcher.axpy(-inner_vy, to, &mut transported);
        dispatcher.axpy(-inner_vx, from, &mut transported);
        dispatcher.axpy(inner_xy * inner_vx, to, &mut transported);
            
        Ok(transported)
    }

    fn distance(&self, point1: &DVector<T>, point2: &DVector<T>) -> Result<T> {
        // Geodesic distance on sphere: d(x, y) = arccos(⟨x, y⟩)
        // This is the length of the shorter great circle arc connecting x and y
        let dispatcher = get_dispatcher::<T>();
        let inner_product = dispatcher.dot_product(point1, point2);
        let cos_theta = <T as Float>::max(
            <T as Float>::min(inner_product, T::one()),
            -T::one(),
        );
        Ok(<T as Float>::acos(cos_theta))
    }

    // ========================================================================
    // Workspace-based methods for zero-allocation operations
    // ========================================================================

    fn project_tangent_with_workspace(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Check dimensions
        if point.len() != self.ambient_dim || vector.len() != self.ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                format!("Expected dimension {}", self.ambient_dim),
                format!("Got point: {}, vector: {}", point.len(), vector.len()),
            ));
        }

        // Check if point is on manifold
        let dispatcher = get_dispatcher::<T>();
        let norm = dispatcher.norm(point);
        if <T as Float>::abs(norm - T::one()) > <T as Scalar>::from_f64(1e-6) {
            return Err(ManifoldError::invalid_point(
                "Point is not on the unit sphere",
            ));
        }

        // Ensure result has correct size
        if result.len() != self.ambient_dim {
            *result = DVector::zeros(self.ambient_dim);
        }

        // Compute projection: v - <v,x>x
        let inner_prod = dispatcher.dot_product(vector, point);
        
        // result = vector - inner_prod * point
        result.copy_from(vector);
        dispatcher.axpy(-inner_prod, point, result);

        Ok(())
    }

    fn retract_with_workspace(
        &self,
        point: &DVector<T>,
        tangent: &DVector<T>,
        result: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Check dimensions
        if point.len() != self.ambient_dim || tangent.len() != self.ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                format!("Expected dimension {}", self.ambient_dim),
                format!("Got point: {}, tangent: {}", point.len(), tangent.len()),
            ));
        }

        // Check if point is on manifold
        let dispatcher = get_dispatcher::<T>();
        let norm = dispatcher.norm(point);
        if <T as Float>::abs(norm - T::one()) > <T as Scalar>::from_f64(1e-6) {
            return Err(ManifoldError::invalid_point(
                "Point is not on the unit sphere",
            ));
        }

        // Ensure result has correct size
        if result.len() != self.ambient_dim {
            *result = DVector::zeros(self.ambient_dim);
        }

        let tangent_norm = dispatcher.norm(tangent);
        
        if tangent_norm < T::epsilon() {
            // If tangent is zero, return the point itself
            result.copy_from(point);
        } else {
            // Exponential map formula: cos(||v||) * x + sin(||v||) * v/||v||
            let cos_norm = <T as Float>::cos(tangent_norm);
            let sin_norm = <T as Float>::sin(tangent_norm);
            let normalized_tangent_factor = sin_norm / tangent_norm;
            
            for i in 0..self.ambient_dim {
                result[i] = cos_norm * point[i] + normalized_tangent_factor * tangent[i];
            }
        }

        Ok(())
    }

    fn inverse_retract_with_workspace(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
        result: &mut DVector<T>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Check dimensions
        if point.len() != self.ambient_dim || other.len() != self.ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                format!("Expected dimension {}", self.ambient_dim),
                format!("Got point: {}, other: {}", point.len(), other.len()),
            ));
        }

        // Check if points are on manifold
        let dispatcher = get_dispatcher::<T>();
        let norm1 = dispatcher.norm(point);
        let norm2 = dispatcher.norm(other);
        if <T as Float>::abs(norm1 - T::one()) > <T as Scalar>::from_f64(1e-6) || <T as Float>::abs(norm2 - T::one()) > <T as Scalar>::from_f64(1e-6) {
            return Err(ManifoldError::invalid_point(
                "Points are not on the unit sphere",
            ));
        }

        // Ensure result has correct size
        if result.len() != self.ambient_dim {
            *result = DVector::zeros(self.ambient_dim);
        }

        // Use logarithmic map
        let inner_product = dispatcher.dot_product(point, other);
        let cos_theta = <T as Float>::max(
            <T as Float>::min(inner_product, T::one()),
            -T::one(),
        );

        if <T as Float>::abs(cos_theta - T::one()) < T::epsilon() {
            // Points are identical
            result.fill(T::zero());
        } else if <T as Float>::abs(cos_theta + T::one()) < T::epsilon() {
            // Points are antipodal - log is undefined
            return Err(ManifoldError::numerical_error(
                "Logarithm undefined: points are antipodal on sphere",
            ));
        } else {
            let theta = <T as Float>::acos(cos_theta);
            let sin_theta = <T as Float>::sin(theta);
            let factor = theta / sin_theta;

            for i in 0..self.ambient_dim {
                result[i] = factor * (other[i] - cos_theta * point[i]);
            }
        }

        Ok(())
    }

    fn euclidean_to_riemannian_gradient_with_workspace(
        &self,
        point: &DVector<T>,
        euclidean_grad: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // For sphere, this is just projection to tangent space
        self.project_tangent_with_workspace(point, euclidean_grad, result, workspace)
    }

    fn parallel_transport_with_workspace(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Check dimensions
        if from.len() != self.ambient_dim || to.len() != self.ambient_dim || vector.len() != self.ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                format!("Expected dimension {}", self.ambient_dim),
                format!("Got from: {}, to: {}, vector: {}", from.len(), to.len(), vector.len()),
            ));
        }

        // Check if points are on manifold
        let dispatcher = get_dispatcher::<T>();
        let norm1 = dispatcher.norm(from);
        let norm2 = dispatcher.norm(to);
        if <T as Float>::abs(norm1 - T::one()) > <T as Scalar>::from_f64(1e-6) || <T as Float>::abs(norm2 - T::one()) > <T as Scalar>::from_f64(1e-6) {
            return Err(ManifoldError::invalid_point(
                "Points are not on the unit sphere",
            ));
        }

        // Ensure result has correct size
        if result.len() != self.ambient_dim {
            *result = DVector::zeros(self.ambient_dim);
        }

        // If points are identical, return the vector unchanged
        let mut diff = workspace.acquire_temp_vector(self.ambient_dim);
        diff.copy_from(to);
        dispatcher.axpy(-T::one(), from, &mut diff);
        let diff_norm = dispatcher.norm(&diff);
        if diff_norm < T::epsilon() {
            result.copy_from(vector);
            return Ok(());
        }

        // Use Schild's ladder approximation for simplicity
        // This could be improved with exact parallel transport formula
        let mut temp = workspace.acquire_temp_vector(self.ambient_dim);
        
        // Project vector to tangent space at 'to'
        self.project_tangent_with_workspace(to, vector, &mut temp, workspace)?;
        result.copy_from(&temp);

        Ok(())
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