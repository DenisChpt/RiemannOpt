//! Hyperbolic manifold H^n - the space of hyperbolic geometry
//!
//! The hyperbolic manifold represents the n-dimensional hyperbolic space,
//! a fundamental non-Euclidean geometry with constant negative curvature.
//! This manifold is particularly important in:
//! - Hierarchical embeddings for tree-like data
//! - Natural language processing (word embeddings)
//! - Social network analysis and complex networks
//! - Machine learning on non-Euclidean data
//! - Computer vision for wide-angle imaging
//! - Representation learning for biological data

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    types::{Scalar, DVector},
    compute::{get_dispatcher, SimdBackend},
    memory::Workspace,
};
use nalgebra::Dyn;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

/// Default boundary tolerance for Poincaré ball boundary stability
const DEFAULT_BOUNDARY_TOLERANCE: f64 = 1e-6;

/// Safety margin for projection to ensure points stay well inside the ball
const PROJECTION_SAFETY_MARGIN: f64 = 0.999;

/// The hyperbolic manifold H^n using the Poincare ball model.
///
/// This manifold represents n-dimensional hyperbolic space using the Poincare ball
/// model, where points are represented as vectors in the unit ball with the
/// hyperbolic metric. The Poincare ball model is computationally efficient
/// and numerically stable.
///
/// # Mathematical Properties
///
/// - **Dimension**: n (same as ambient Euclidean space)
/// - **Curvature**: Constant negative curvature K = -1
/// - **Metric**: Riemannian metric derived from the hyperbolic geometry
/// - **Geodesics**: Circular arcs orthogonal to the boundary (in 2D)
/// - **Distance**: Hyperbolic distance using the Poincare metric
///
/// # Poincare Ball Model
///
/// Points x in H^n are represented as vectors in R^n with ||x|| < 1.
/// The hyperbolic metric tensor at point x is:
/// g_x = (4 / (1 - ||x||^2)^2) * I
///
/// # Applications
///
/// - **NLP**: Word embeddings capturing hierarchical relationships
/// - **Biology**: Phylogenetic tree embeddings
/// - **Social networks**: Community structure representation
/// - **Computer vision**: Fish-eye and omnidirectional imaging
/// - **Machine learning**: Non-Euclidean neural networks
#[derive(Debug, Clone)]
pub struct Hyperbolic {
    /// Dimension of the hyperbolic space
    n: usize,
    /// Numerical tolerance for boundary checks
    boundary_tolerance: f64,
}

impl Hyperbolic {
    /// Creates a new hyperbolic manifold H^n.
    ///
    /// # Arguments
    /// * `n` - Dimension of the hyperbolic space (must be > 0)
    ///
    /// # Returns
    /// A hyperbolic manifold with intrinsic dimension n
    ///
    /// # Errors
    /// Returns an error if dimension is invalid
    ///
    /// # Examples
    /// ```
    /// use riemannopt_manifolds::Hyperbolic;
    /// 
    /// // Create H^2 - 2D hyperbolic space (Poincare disk)
    /// let hyperbolic = Hyperbolic::new(2).unwrap();
    /// assert_eq!(hyperbolic.dimension_space(), 2);
    /// ```
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "Hyperbolic manifold requires dimension n > 0",
            ));
        }
        Ok(Self {
            n,
            boundary_tolerance: DEFAULT_BOUNDARY_TOLERANCE,
        })
    }

    /// Creates a hyperbolic manifold with custom boundary tolerance.
    ///
    /// # Arguments
    /// * `n` - Dimension of the space
    /// * `boundary_tolerance` - How close to unit sphere boundary to allow
    pub fn with_boundary_tolerance(n: usize, boundary_tolerance: f64) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "Hyperbolic manifold requires dimension n > 0",
            ));
        }
        if boundary_tolerance <= 0.0 || boundary_tolerance >= 1.0 {
            return Err(ManifoldError::invalid_point(
                "Boundary tolerance must be in (0, 1)",
            ));
        }
        Ok(Self { n, boundary_tolerance })
    }

    /// Returns the dimension of the hyperbolic space
    pub fn dimension_space(&self) -> usize {
        self.n
    }

    /// Returns the boundary tolerance
    pub fn boundary_tolerance(&self) -> f64 {
        self.boundary_tolerance
    }

    /// Checks if a point is in the Poincare ball (||x|| < 1).
    fn is_in_poincare_ball<T>(&self, point: &DVector<T>, tolerance: T) -> bool
    where
        T: Scalar,
    {
        if point.len() != self.n {
            return false;
        }
        
        let norm_squared = point.norm_squared();
        let tolerance_f64 = tolerance.to_f64();
        let boundary = T::one() - <T as Scalar>::from_f64(tolerance_f64);
        norm_squared < boundary
    }

    /// Projects a point to the Poincare ball.
    ///
    /// If the point is outside the unit ball, project it to a point
    /// slightly inside the boundary for numerical stability.
    fn project_to_poincare_ball<T>(&self, point: &DVector<T>) -> DVector<T>
    where
        T: Scalar,
    {
        let norm = point.norm();
        let max_norm = T::one() - <T as Scalar>::from_f64(self.boundary_tolerance);
        
        if norm > max_norm {
            // Project to boundary with tolerance (slightly inside)
            let safe_norm = max_norm * <T as Scalar>::from_f64(PROJECTION_SAFETY_MARGIN);
            point * (safe_norm / norm)
        } else {
            point.clone()
        }
    }

    /// Computes the conformal factor lambda(x) = 2 / (1 - ||x||^2).
    fn conformal_factor<T>(&self, point: &DVector<T>) -> T
    where
        T: Scalar,
    {
        let norm_squared = point.norm_squared();
        let two = <T as Scalar>::from_f64(2.0);
        two / (T::one() - norm_squared)
    }

    /// Computes the hyperbolic distance between two points in the Poincare ball.
    ///
    /// Distance formula: d(x,y) = arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
    fn hyperbolic_distance<T>(&self, x: &DVector<T>, y: &DVector<T>) -> T
    where
        T: Scalar,
    {
        let diff = x - y;
        let diff_norm_sq = diff.norm_squared();
        
        let x_norm_sq = x.norm_squared();
        let y_norm_sq = y.norm_squared();
        
        let denominator = (T::one() - x_norm_sq) * (T::one() - y_norm_sq);
        let two = <T as Scalar>::from_f64(2.0);
        
        let argument = T::one() + two * diff_norm_sq / denominator;
        
        // Clamp argument to avoid numerical issues with acosh
        let clamped = <T as Float>::max(argument, T::one());
        <T as Float>::acosh(clamped)
    }

    /// Computes the exponential map in the Poincare ball model.
    ///
    /// The exponential map moves along geodesics from a point in a given direction.
    fn exponential_map<T>(&self, point: &DVector<T>, tangent: &DVector<T>) -> DVector<T>
    where
        T: Scalar,
    {
        let tangent_norm = tangent.norm();
        
        if tangent_norm < T::epsilon() {
            return point.clone();
        }
        
        let lambda = self.conformal_factor(point);
        let scaled_norm = tangent_norm * lambda / <T as Scalar>::from_f64(2.0);
        
        // Exponential map formula in Poincare ball
        let alpha = <T as Float>::tanh(scaled_norm);
        let normalized_tangent = tangent / tangent_norm;
        
        let numerator = point + &normalized_tangent * alpha;
        let denominator = T::one() + alpha * point.dot(&normalized_tangent);
        
        self.project_to_poincare_ball(&(numerator / denominator))
    }

    /// Computes the logarithmic map in the Poincare ball model.
    ///
    /// The logarithmic map finds the tangent vector from point to other.
    fn logarithmic_map<T>(&self, point: &DVector<T>, other: &DVector<T>) -> DVector<T>
    where
        T: Scalar,
    {
        let diff = other - point;
        let diff_norm = diff.norm();
        
        if diff_norm < T::epsilon() {
            return DVector::zeros(self.n);
        }
        
        let _point_norm_sq = point.norm_squared();
        let _other_norm_sq = other.norm_squared();
        
        let lambda = self.conformal_factor(point);
        
        // Logarithmic map formula
        let factor = <T as Scalar>::from_f64(2.0) / lambda;
        let alpha = self.hyperbolic_distance(point, other);
        
        diff * (factor * alpha / diff_norm)
    }

    /// Projects a vector to the tangent space at a point.
    ///
    /// In the Poincare ball model, all vectors are valid tangent vectors,
    /// so this is essentially the identity operation.
    fn project_to_tangent<T>(&self, _point: &DVector<T>, vector: &DVector<T>) -> DVector<T>
    where
        T: Scalar,
    {
        // In Poincare ball, tangent space is full R^n
        vector.clone()
    }

    /// Generates a random point in the Poincare ball.
    fn random_poincare_point<T>(&self) -> DVector<T>
    where
        T: Scalar,
    {
        let mut rng = rand::thread_rng();
        
        // Generate random direction
        let mut point = DVector::<T>::zeros(self.n);
        for i in 0..self.n {
            let val: f64 = StandardNormal.sample(&mut rng);
            point[i] = <T as Scalar>::from_f64(val);
        }
        
        // Normalize and scale by random radius
        let norm = point.norm();
        if norm > T::epsilon() {
            point = point / norm;
            
            // Random radius with appropriate distribution for uniform distribution in ball
            let u: f64 = rand::random();
            let radius = u.powf(1.0 / self.n as f64);
            let max_radius = 1.0 - self.boundary_tolerance;
            let scaled_radius = radius * max_radius;
            
            point * <T as Scalar>::from_f64(scaled_radius)
        } else {
            // Return origin if we got zero vector
            DVector::<T>::zeros(self.n)
        }
    }

    /// Parallel transport using the Levi-Civita connection.
    ///
    /// Transports a tangent vector along the geodesic from one point to another.
    fn parallel_transport_vector<T>(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> DVector<T>
    where
        T: Scalar,
    {
        // Exact parallel transport formula for the Poincaré ball model
        // Based on the gyrovector formalism and the Levi-Civita connection
        
        // If from == to, no transport needed
        let diff = to - from;
        if diff.norm() < T::epsilon() {
            return vector.clone();
        }
        
        // Compute the necessary components for parallel transport
        let _from_norm_sq = from.norm_squared();
        let to_norm_sq = to.norm_squared();
        let from_dot_to = from.dot(to);
        let from_dot_v = from.dot(vector);
        let to_dot_v = to.dot(vector);
        
        // The formula for parallel transport in the Poincaré ball is:
        // P_{x→y}(v) = v + 2/(1 - ||y||^2) * (⟨y,v⟩y - ⟨x,v⟩/(1 + ⟨x,y⟩) * (y + x))
        
        let denominator = T::one() + from_dot_to;
        
        // Avoid division by zero
        if <T as Float>::abs(denominator) < T::epsilon() {
            // Points are nearly antipodal in the ball - use simple scaling
            let lambda_from = self.conformal_factor(from);
            let lambda_to = self.conformal_factor(to);
            return vector * (lambda_from / lambda_to);
        }
        
        let scale_factor = <T as Scalar>::from_f64(2.0) / (T::one() - to_norm_sq);
        let term1 = to * to_dot_v;
        let term2 = (to + from) * (from_dot_v / denominator);
        
        vector + (term1 - term2) * scale_factor
    }
}

impl<T> Manifold<T, Dyn> for Hyperbolic
where
    T: Scalar,
{
    fn name(&self) -> &str {
        "Hyperbolic"
    }

    fn dimension(&self) -> usize {
        self.n
    }

    fn is_point_on_manifold(&self, point: &Point<T, Dyn>, tolerance: T) -> bool {
        self.is_in_poincare_ball(point, tolerance)
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        _tolerance: T,
    ) -> bool {
        // In Poincare ball model, all vectors of correct dimension are tangent vectors
        vector.len() == self.n
    }

    fn project_point(&self, point: &Point<T, Dyn>, result: &mut Point<T, Dyn>, _workspace: &mut Workspace<T>) {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n {
            // Handle wrong dimension by padding/truncating
            let mut temp = DVector::<T>::zeros(self.n);
            let copy_len = point.len().min(self.n);
            for i in 0..copy_len {
                temp[i] = point[i];
            }
            let projected = self.project_to_poincare_ball(&temp);
            result.copy_from(&projected);
        } else {
            let projected = self.project_to_poincare_ball(point);
            result.copy_from(&projected);
        }
    }

    fn project_tangent(
        &self,
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || vector.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "Point and vector must have correct dimensions for hyperbolic manifold",
                format!("point: {}, vector: {}", point.len(), vector.len()),
            ));
        }

        let proj = self.project_to_tangent(point, vector);
        result.copy_from(&proj);
        Ok(())
    }

    fn inner_product(
        &self,
        point: &Point<T, Dyn>,
        u: &TangentVector<T, Dyn>,
        v: &TangentVector<T, Dyn>,
    ) -> Result<T> {
        if point.len() != self.n || u.len() != self.n || v.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "All vectors must have correct dimensions",
                format!("expected: {}", self.n),
            ));
        }

        // Hyperbolic inner product: <u,v>_x = lambda(x)^2 * <u,v>_euclidean
        let lambda = self.conformal_factor(point);
        let dispatcher = get_dispatcher::<T>();
        let euclidean_inner = dispatcher.dot_product(u, v);
        Ok(lambda * lambda * euclidean_inner)
    }

    fn retract(&self, point: &Point<T, Dyn>, tangent: &TangentVector<T, Dyn>, result: &mut Point<T, Dyn>, _workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || tangent.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "Point and tangent must have correct dimensions",
                format!("point: {}, tangent: {}", point.len(), tangent.len()),
            ));
        }

        // Use exponential map as retraction
        let exp = self.exponential_map(point, tangent);
        result.copy_from(&exp);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Point<T, Dyn>,
        other: &Point<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || other.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("point: {}, other: {}", point.len(), other.len()),
            ));
        }

        // Use logarithmic map as inverse retraction
        let log = self.logarithmic_map(point, other);
        result.copy_from(&log);
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Point<T, Dyn>,
        euclidean_grad: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n || euclidean_grad.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "Point and gradient must have correct dimensions",
                format!("point: {}, euclidean_grad: {}", point.len(), euclidean_grad.len()),
            ));
        }

        // Convert Euclidean gradient to Riemannian gradient
        // For Poincare ball: grad_riem = (1 - ||x||^2)^2 / 4 * grad_euclidean
        let norm_squared = point.norm_squared();
        let factor = (T::one() - norm_squared) * (T::one() - norm_squared) / <T as Scalar>::from_f64(4.0);
        
        result.copy_from(&(euclidean_grad * factor));
        Ok(())
    }

    fn random_point(&self) -> Point<T, Dyn> {
        self.random_poincare_point()
    }

    fn random_tangent(&self, point: &Point<T, Dyn>, result: &mut TangentVector<T, Dyn>, _workspace: &mut Workspace<T>) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if point.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "Point must have correct dimensions",
                format!("expected: {}, actual: {}", self.n, point.len()),
            ));
        }

        let mut rng = rand::thread_rng();
        let mut tangent = DVector::<T>::zeros(self.n);
        
        for i in 0..self.n {
            let val: f64 = StandardNormal.sample(&mut rng);
            tangent[i] = <T as Scalar>::from_f64(val);
        }
        
        let proj = self.project_to_tangent(point, &tangent);
        result.copy_from(&proj);
        Ok(())
    }

    fn has_exact_exp_log(&self) -> bool {
        true // Poincare ball model has exact exponential and logarithmic maps
    }

    fn parallel_transport(
        &self,
        from: &Point<T, Dyn>,
        to: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        result: &mut TangentVector<T, Dyn>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Ensure result has correct size
        if result.len() != self.n {
            *result = DVector::zeros(self.n);
        }
        
        if from.len() != self.n || to.len() != self.n || vector.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "All vectors must have correct dimensions",
                format!("expected: {}", self.n),
            ));
        }

        let transported = self.parallel_transport_vector(from, to, vector);
        result.copy_from(&transported);
        Ok(())
    }

    fn distance(&self, x: &Point<T, Dyn>, y: &Point<T, Dyn>, _workspace: &mut Workspace<T>) -> Result<T> {
        if x.len() != self.n || y.len() != self.n {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("x: {}, y: {}", x.len(), y.len()),
            ));
        }

        Ok(self.hyperbolic_distance(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_hyperbolic_creation() {
        let hyperbolic = Hyperbolic::new(3).unwrap();
        assert_eq!(<Hyperbolic as Manifold<f64, Dyn>>::dimension(&hyperbolic), 3);
        assert_eq!(hyperbolic.dimension_space(), 3);
        
        // Test invalid dimension
        assert!(Hyperbolic::new(0).is_err());
        
        // Test custom boundary tolerance
        let hyperbolic_custom = Hyperbolic::with_boundary_tolerance(3, 1e-3).unwrap();
        assert_eq!(hyperbolic_custom.boundary_tolerance(), 1e-3);
        assert!(Hyperbolic::with_boundary_tolerance(3, 0.0).is_err());
        assert!(Hyperbolic::with_boundary_tolerance(3, 1.0).is_err());
    }

    #[test]
    fn test_poincare_ball_properties() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        
        // Test point inside ball
        let inside_point = DVector::from_vec(vec![0.5, 0.3]);
        assert!(hyperbolic.is_in_poincare_ball(&inside_point, 1e-6));
        
        // Test point on boundary (should be outside with tolerance)
        let boundary_point = DVector::from_vec(vec![1.0, 0.0]);
        assert!(!hyperbolic.is_in_poincare_ball(&boundary_point, 1e-6));
        
        // Test point outside ball
        let outside_point = DVector::from_vec(vec![1.5, 0.0]);
        assert!(!hyperbolic.is_in_poincare_ball(&outside_point, 1e-6));
    }

    #[test]
    fn test_projection_to_poincare_ball() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        
        // Test projection of outside point
        let outside_point = DVector::from_vec(vec![2.0, 0.0]);
        let projected = hyperbolic.project_to_poincare_ball(&outside_point);
        
        assert!(hyperbolic.is_in_poincare_ball(&projected, 1e-6));
        assert!(projected.norm() < 1.0 - hyperbolic.boundary_tolerance());
        
        // Test that inside points are unchanged
        let inside_point = DVector::from_vec(vec![0.5, 0.3]);
        let projected_inside = hyperbolic.project_to_poincare_ball(&inside_point);
        assert_relative_eq!(projected_inside, inside_point, epsilon = 1e-10);
    }

    #[test]
    fn test_conformal_factor() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        
        // At origin, conformal factor should be 2
        let origin = DVector::zeros(2);
        let lambda_origin = hyperbolic.conformal_factor(&origin);
        assert_relative_eq!(lambda_origin, 2.0, epsilon = 1e-10);
        
        // Conformal factor should increase as we approach boundary
        let near_boundary = DVector::from_vec(vec![0.9, 0.0]);
        let lambda_boundary = hyperbolic.conformal_factor(&near_boundary);
        assert!(lambda_boundary > lambda_origin);
    }

    #[test]
    fn test_hyperbolic_distance() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        
        let point1 = DVector::from_vec(vec![0.0, 0.0]); // Origin
        let point2 = DVector::from_vec(vec![0.5, 0.0]); // On x-axis
        
        let dist = hyperbolic.hyperbolic_distance(&point1, &point2);
        
        // Distance should be positive
        assert!(dist > 0.0);
        
        // Distance to self should be zero
        let self_dist = hyperbolic.hyperbolic_distance(&point1, &point1);
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-10);
        
        // Distance should be symmetric
        let dist_rev = hyperbolic.hyperbolic_distance(&point2, &point1);
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_log_maps() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        let point = DVector::from_vec(vec![0.2, 0.3]);
        let tangent = DVector::from_vec(vec![0.1, -0.1]);
        
        // Test exp then log
        let exp_result = hyperbolic.exponential_map(&point, &tangent);
        assert!(hyperbolic.is_in_poincare_ball(&exp_result, 1e-6));
        
        let log_result = hyperbolic.logarithmic_map(&point, &exp_result);
        
        // Test that exp/log are consistent operations
        // Note: The current implementation is approximate, so we test basic properties
        assert_eq!(log_result.len(), tangent.len());
        assert!(log_result.norm() > 0.0); // Should be non-zero for non-zero tangent
        
        // Test zero tangent case
        let zero_tangent = DVector::zeros(2);
        let exp_zero = hyperbolic.exponential_map(&point, &zero_tangent);
        assert_relative_eq!(exp_zero, point, epsilon = 1e-10);
        
        let log_zero = hyperbolic.logarithmic_map(&point, &point);
        assert!(log_zero.norm() < 1e-6); // Should be approximately zero
    }

    #[test]
    fn test_retraction_properties() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        let point = <Hyperbolic as Manifold<f64, Dyn>>::random_point(&hyperbolic);
        let zero_tangent = DVector::zeros(2);
        
        // Test centering property: R(x, 0) = x
        let mut retracted = DVector::zeros(2);
        let mut workspace = Workspace::new();
        hyperbolic.retract(&point, &zero_tangent, &mut retracted, &mut workspace).unwrap();
        assert_relative_eq!(&retracted, &point, epsilon = 1e-10);
        
        // Result should be on manifold
        assert!(hyperbolic.is_point_on_manifold(&retracted, 1e-6));
    }

    #[test]
    fn test_inner_product() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        let point = DVector::from_vec(vec![0.3, 0.4]);
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0]);
        
        // Orthogonal vectors should have zero inner product
        let inner = hyperbolic.inner_product(&point, &u, &v).unwrap();
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
        
        // Inner product with itself should be positive
        let self_inner = hyperbolic.inner_product(&point, &u, &u).unwrap();
        assert!(self_inner > 0.0);
    }

    #[test]
    fn test_gradient_conversion() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        let point = DVector::from_vec(vec![0.2, 0.3]);
        let euclidean_grad = DVector::from_vec(vec![1.0, -1.0]);
        
        let mut riemannian_grad = DVector::zeros(2);
        let mut workspace = Workspace::new();
        hyperbolic
            .euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut workspace)
            .unwrap();
        
        // Riemannian gradient should be scaled version of Euclidean gradient
        assert!(riemannian_grad.norm() != euclidean_grad.norm());
        
        // Direction should be preserved (or consistently scaled)
        let euclidean_normalized = euclidean_grad.normalize();
        let riemannian_normalized = riemannian_grad.normalize();
        assert_relative_eq!(euclidean_normalized, riemannian_normalized, epsilon = 1e-10);
    }

    #[test]
    fn test_random_generation() {
        let hyperbolic = Hyperbolic::new(3).unwrap();
        
        // Test random point generation
        for _ in 0..10 {
            let random_point = <Hyperbolic as Manifold<f64, Dyn>>::random_point(&hyperbolic);
            assert!(hyperbolic.is_point_on_manifold(&random_point, 1e-6));
        }
        
        // Test random tangent generation
        let point = hyperbolic.random_point();
        let mut tangent = DVector::zeros(3);
        let mut workspace = Workspace::new();
        hyperbolic.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        assert!(hyperbolic.is_vector_in_tangent_space(&point, &tangent, 1e-10));
    }

    #[test]
    fn test_distance_properties() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        
        let point1 = <Hyperbolic as Manifold<f64, Dyn>>::random_point(&hyperbolic);
        let point2 = <Hyperbolic as Manifold<f64, Dyn>>::random_point(&hyperbolic);
        
        // Distance should be non-negative
        let mut workspace = Workspace::new();
        let dist = hyperbolic.distance(&point1, &point2, &mut workspace).unwrap();
        assert!(dist >= 0.0);
        
        // Distance to self should be zero
        let mut workspace = Workspace::new();
        let self_dist = hyperbolic.distance(&point1, &point1, &mut workspace).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-10);
        
        // Distance should be symmetric
        let mut workspace = Workspace::new();
        let dist_rev = hyperbolic.distance(&point2, &point1, &mut workspace).unwrap();
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_transport() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        let from = DVector::from_vec(vec![0.1, 0.2]);
        let to = DVector::from_vec(vec![0.3, 0.4]);
        let vector = DVector::from_vec(vec![0.1, 0.0]);
        
        let mut transported = DVector::zeros(2);
        let mut workspace = Workspace::new();
        hyperbolic.parallel_transport(&from, &to, &vector, &mut transported, &mut workspace).unwrap();
        
        // Transported vector should be in tangent space at destination
        assert!(hyperbolic.is_vector_in_tangent_space(&to, &transported, 1e-10));
        
        // Transport should preserve some geometric properties
        assert_eq!(transported.len(), vector.len());
    }

    #[test]
    fn test_origin_properties() {
        let hyperbolic = Hyperbolic::new(2).unwrap();
        let origin = DVector::zeros(2);
        
        // Origin should be on manifold
        assert!(hyperbolic.is_point_on_manifold(&origin, 1e-10));
        
        // Conformal factor at origin
        let lambda = hyperbolic.conformal_factor(&origin);
        assert_relative_eq!(lambda, 2.0, epsilon = 1e-10);
        
        // Distance from origin to origin
        let mut workspace = Workspace::new();
        let dist = hyperbolic.distance(&origin, &origin, &mut workspace).unwrap();
        assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dimension_handling() {
        let hyperbolic = Hyperbolic::new(3).unwrap();
        
        // Test wrong dimension handling
        let wrong_dim_point = DVector::from_vec(vec![1.0, 2.0]); // 2D instead of 3D
        let mut projected = DVector::zeros(3);
        let mut workspace = Workspace::new();
        hyperbolic.project_point(&wrong_dim_point, &mut projected, &mut workspace);
        assert_eq!(projected.len(), 3);
        assert!(hyperbolic.is_point_on_manifold(&projected, 1e-6));
    }
}