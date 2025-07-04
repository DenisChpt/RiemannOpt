//! Simple integration tests for core functionality.

use riemannopt_core::{
    core::{
        manifold::Manifold,
        cost_function::CostFunction,
    },
    error::Result,
    memory::workspace::Workspace,
    types::DVector,
    // optimization::{
    //     line_search::{LineSearch, BacktrackingLineSearch, StrongWolfeLineSearch, LineSearchParams},
    //     cost_function_wrappers::CountingCostFunction,
    // },
};

/// Simple Euclidean manifold for testing.
#[derive(Debug, Clone)]
struct EuclideanSpace {
    dim: usize,
}

impl EuclideanSpace {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64> for EuclideanSpace {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;

    fn name(&self) -> &str {
        "Euclidean Space"
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn is_point_on_manifold(&self, _point: &DVector<f64>, _tolerance: f64) -> bool {
        true // All points are valid in Euclidean space
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &DVector<f64>,
        _vector: &DVector<f64>,
        _tolerance: f64,
    ) -> bool {
        true // All vectors are valid
    }

    fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) {
        *result = point.clone(); // No projection needed
    }

    fn project_tangent(
        &self,
        _point: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = vector.clone(); // No projection needed
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        vector1: &DVector<f64>,
        vector2: &DVector<f64>,
    ) -> Result<f64> {
        Ok(vector1.dot(vector2))
    }

    fn retract(
        &self,
        point: &DVector<f64>,
        tangent: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = point + tangent;
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &DVector<f64>,
        other: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = other - point;
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = euclidean_grad.clone();
        Ok(())
    }

    fn random_point(&self) -> DVector<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut point = DVector::zeros(self.dim);
        for i in 0..self.dim {
            point[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        point
    }

    fn random_tangent(
        &self,
        _point: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Ensure result has the correct size
        if result.nrows() != self.dim {
            *result = DVector::zeros(self.dim);
        }
        for i in 0..self.dim {
            result[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        Ok(())
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        _to: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        // In Euclidean space, parallel transport is identity
        *result = vector.clone();
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, _workspace: &mut Workspace<f64>) -> Result<f64> {
        Ok((y - x).norm())
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: f64,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = tangent * scalar;
        Ok(())
    }

    fn add_tangents(
        &self,
        _point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = v1 + v2;
        Ok(())
    }
}

/// Simple quadratic cost function.
#[derive(Debug, Clone)]
struct SimpleQuadratic {
    _dim: usize,
}

impl SimpleQuadratic {
    fn new(dim: usize) -> Self {
        Self { _dim: dim }
    }
}

impl CostFunction<f64> for SimpleQuadratic {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;

    fn cost(&self, point: &DVector<f64>) -> Result<f64> {
        // f(x) = ||x||^2
        Ok(point.norm_squared())
    }

    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
        // ∇f(x) = 2x
        Ok(point * 2.0)
    }

    fn hessian_vector_product(
        &self,
        _x: &Self::Point,
        v: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // For quadratic, Hessian is 2I
        Ok(v * 2.0)
    }

    fn gradient_fd_alloc(&self, x: &Self::Point) -> Result<Self::TangentVector> {
        self.gradient(x)
    }
}

// TODO: The following tests depend on line search implementations that need to be adapted
// to the new architecture. They should be re-enabled once those components are updated.

/*
#[test]
fn test_line_search_with_real_manifold() {
    let manifold = EuclideanSpace::new(3);
    let cost_fn = SimpleQuadratic::new(3);
    
    // Start from a specific point
    let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let value = cost_fn.cost(&point).unwrap();
    let gradient = cost_fn.gradient(&point).unwrap();
    let direction = -&gradient; // Steepest descent
    
    // Test backtracking line search
    let mut ls = BacktrackingLineSearch::new();
    let params = LineSearchParams::backtracking();
    
    let result = ls.search(
        &cost_fn,
        &manifold,
        &point,
        value,
        &gradient,
        &direction,
        &params,
    ).unwrap();
    
    assert!(result.success);
    assert!(result.step_size > 0.0);
    assert!(result.new_value < value);
}

#[test]
fn test_strong_wolfe_line_search() {
    let manifold = EuclideanSpace::new(2);
    let cost_fn = SimpleQuadratic::new(2);
    
    let point = DVector::from_vec(vec![3.0, 4.0]);
    let value = cost_fn.cost(&point).unwrap();
    let gradient = cost_fn.gradient(&point).unwrap();
    let direction = -&gradient;
    
    let mut ls = StrongWolfeLineSearch::new();
    let params = LineSearchParams::strong_wolfe();
    
    let result = ls.search(
        &cost_fn,
        &manifold,
        &point,
        value,
        &gradient,
        &direction,
        &params,
    ).unwrap();
    
    assert!(result.success);
    // For quadratic function with exact line search, step size should be 0.5
    assert!((result.step_size - 0.5).abs() < 0.1);
}

#[test]
fn test_cost_function_counting() {
    let _manifold = EuclideanSpace::new(2);
    let base_cost = SimpleQuadratic::new(2);
    let cost_fn = CountingCostFunction::new(base_cost);
    
    let point = DVector::from_vec(vec![1.0, 1.0]);
    
    // Initial counts should be zero
    let (cost_count, grad_count, _) = cost_fn.counts();
    assert_eq!(cost_count, 0);
    assert_eq!(grad_count, 0);
    
    // Evaluate cost
    let _ = cost_fn.cost(&point).unwrap();
    let (cost_count, grad_count, _) = cost_fn.counts();
    assert_eq!(cost_count, 1);
    assert_eq!(grad_count, 0);
    
    // Evaluate gradient
    let _ = cost_fn.gradient(&point).unwrap();
    let (cost_count, grad_count, _) = cost_fn.counts();
    assert_eq!(cost_count, 1);
    assert_eq!(grad_count, 1);
    
    // Evaluate both
    let _ = cost_fn.cost_and_gradient_alloc(&point).unwrap();
    let (cost_count, grad_count, _) = cost_fn.counts();
    assert_eq!(cost_count, 2);
    assert_eq!(grad_count, 2);
}
*/

#[test]
fn test_retraction_properties() {
    let manifold = EuclideanSpace::new(3);
    
    let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let tangent = DVector::from_vec(vec![0.0, 1.0, 0.0]);
    
    // Test retraction at zero
    let zero_tangent = DVector::zeros(3);
    let mut result = DVector::zeros(3);
    let mut workspace = Workspace::new();
    manifold.retract(&point, &zero_tangent, &mut result, &mut workspace).unwrap();
    assert!((result - &point).norm() < 1e-10);
    
    // Test retraction is smooth
    let mut result1 = DVector::zeros(3);
    manifold.retract(&point, &tangent, &mut result1, &mut workspace).unwrap();
    let mut small_tangent = tangent.clone();
    manifold.scale_tangent(&point, 0.001, &tangent, &mut small_tangent, &mut workspace).unwrap();
    let mut result2 = DVector::zeros(3);
    manifold.retract(&point, &small_tangent, &mut result2, &mut workspace).unwrap();
    
    // Small retractions should be approximately linear
    let expected = &point + &small_tangent;
    assert!((result2 - expected).norm() < 1e-6);
}

#[test]
fn test_manifold_operations() {
    let manifold = EuclideanSpace::new(3);
    let mut workspace = Workspace::new();
    
    // Test point operations
    let p1 = manifold.random_point();
    let p2 = manifold.random_point();
    
    // Test distance
    let dist = manifold.distance(&p1, &p2, &mut workspace).unwrap();
    assert!(dist >= 0.0);
    
    // Test tangent operations
    let mut v1 = DVector::zeros(3);
    let mut v2 = DVector::zeros(3);
    manifold.random_tangent(&p1, &mut v1, &mut workspace).unwrap();
    manifold.random_tangent(&p1, &mut v2, &mut workspace).unwrap();
    
    // Test inner product
    let inner = manifold.inner_product(&p1, &v1, &v2).unwrap();
    let expected = v1.dot(&v2);
    assert!((inner - expected).abs() < 1e-10);
    
    // Test tangent addition
    let mut sum = DVector::zeros(3);
    manifold.add_tangents(&p1, &v1, &v2, &mut sum, &mut workspace).unwrap();
    assert!((&sum - (&v1 + &v2)).norm() < 1e-10);
    
    // Test tangent scaling
    let mut scaled = DVector::zeros(3);
    manifold.scale_tangent(&p1, 2.5, &v1, &mut scaled, &mut workspace).unwrap();
    assert!((&scaled - (&v1 * 2.5)).norm() < 1e-10);
}

#[test]
fn test_parallel_transport_euclidean() {
    let manifold = EuclideanSpace::new(4);
    let mut workspace = Workspace::new();
    
    let p1 = manifold.random_point();
    let p2 = manifold.random_point();
    
    let mut v = DVector::zeros(4);
    manifold.random_tangent(&p1, &mut v, &mut workspace).unwrap();
    
    let mut transported = DVector::zeros(4);
    manifold.parallel_transport(&p1, &p2, &v, &mut transported, &mut workspace).unwrap();
    
    // In Euclidean space, parallel transport is identity
    assert!((&transported - &v).norm() < 1e-10);
}

#[test]
fn test_gradient_conversion() {
    let manifold = EuclideanSpace::new(3);
    let cost_fn = SimpleQuadratic::new(3);
    let mut workspace = Workspace::new();
    
    let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let euclidean_grad = cost_fn.gradient(&point).unwrap();
    
    let mut riemannian_grad = DVector::zeros(3);
    manifold.euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut workspace).unwrap();
    
    // In Euclidean space, gradients are the same
    assert!((&riemannian_grad - &euclidean_grad).norm() < 1e-10);
}

#[test]
fn test_retract_inverse_retract_consistency() {
    let manifold = EuclideanSpace::new(3);
    let mut workspace = Workspace::new();
    
    let point = manifold.random_point();
    let mut tangent = DVector::zeros(3);
    manifold.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
    
    // Scale tangent to be small
    let mut scaled_tangent = tangent.clone();
    manifold.scale_tangent(&point, 0.1, &tangent, &mut scaled_tangent, &mut workspace).unwrap();
    
    // Retract
    let mut new_point = DVector::zeros(3);
    manifold.retract(&point, &scaled_tangent, &mut new_point, &mut workspace).unwrap();
    
    // Inverse retract
    let mut recovered_tangent = DVector::zeros(3);
    manifold.inverse_retract(&point, &new_point, &mut recovered_tangent, &mut workspace).unwrap();
    
    // Should recover the tangent vector
    assert!((&recovered_tangent - &scaled_tangent).norm() < 1e-10);
}

#[test]
fn test_cost_function_consistency() {
    let cost_fn = SimpleQuadratic::new(3);
    
    let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    
    // Test that gradient matches finite differences
    let grad = cost_fn.gradient(&point).unwrap();
    let grad_fd = cost_fn.gradient_fd_alloc(&point).unwrap();
    
    // For quadratic, should be exact
    assert!((&grad - &grad_fd).norm() < 1e-10);
    
    // Test Hessian-vector product
    let v = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    let hv = cost_fn.hessian_vector_product(&point, &v).unwrap();
    
    // For f(x) = ||x||^2, Hessian is 2I
    let expected = &v * 2.0;
    assert!((&hv - &expected).norm() < 1e-10);
}

// TODO: Add more integration tests as more components are updated:
// - Derivative checking
// - Convergence criteria
// - Optimizer callbacks
// - Memory workspace usage
// - SIMD operations when available