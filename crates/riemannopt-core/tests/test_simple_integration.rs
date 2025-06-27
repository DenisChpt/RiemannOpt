//! Simple integration tests for core functionality.

use nalgebra::{DVector, Dyn};
use riemannopt_core::prelude::*;
use riemannopt_core::optimization::OptimizerStateLegacy;
use riemannopt_core::memory::workspace::Workspace;

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

impl Manifold<f64, Dyn> for EuclideanSpace {
    fn name(&self) -> &str {
        "Euclidean Space"
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn is_point_on_manifold(&self, _point: &Point<f64, Dyn>, _tolerance: f64) -> bool {
        true // All points are valid in Euclidean space
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &Point<f64, Dyn>,
        _vector: &Point<f64, Dyn>,
        _tolerance: f64,
    ) -> bool {
        true // All vectors are valid
    }

    fn project_point(&self, point: &Point<f64, Dyn>, result: &mut Point<f64, Dyn>, _workspace: &mut Workspace<f64>) {
        result.copy_from(point); // No projection needed
    }

    fn project_tangent(
        &self,
        _point: &Point<f64, Dyn>,
        vector: &Point<f64, Dyn>,
        result: &mut Point<f64, Dyn>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        result.copy_from(vector); // No projection needed
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Point<f64, Dyn>,
        vector1: &Point<f64, Dyn>,
        vector2: &Point<f64, Dyn>,
    ) -> Result<f64> {
        Ok(vector1.dot(vector2))
    }

    fn retract(
        &self,
        point: &Point<f64, Dyn>,
        tangent: &Point<f64, Dyn>,
        result: &mut Point<f64, Dyn>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        result.copy_from(&(point + tangent));
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Point<f64, Dyn>,
        other: &Point<f64, Dyn>,
        result: &mut Point<f64, Dyn>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        result.copy_from(&(other - point));
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &Point<f64, Dyn>,
        euclidean_grad: &Point<f64, Dyn>,
        result: &mut Point<f64, Dyn>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        result.copy_from(euclidean_grad);
        Ok(())
    }

    fn random_point(&self) -> Point<f64, Dyn> {
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
        _point: &Point<f64, Dyn>,
        result: &mut Point<f64, Dyn>,
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

impl CostFunction<f64, Dyn> for SimpleQuadratic {
    fn cost(&self, point: &Point<f64, Dyn>) -> Result<f64> {
        // f(x) = ||x||^2
        Ok(point.norm_squared())
    }

    fn gradient(&self, point: &Point<f64, Dyn>) -> Result<Point<f64, Dyn>> {
        // ∇f(x) = 2x
        Ok(point * 2.0)
    }
}

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
    let small_tangent = &tangent * 0.001;
    let mut result2 = DVector::zeros(3);
    manifold.retract(&point, &small_tangent, &mut result2, &mut workspace).unwrap();
    
    // Small retractions should be approximately linear
    let expected = &point + &small_tangent;
    assert!((result2 - expected).norm() < 1e-6);
}

#[test] 
fn test_optimizer_state_tracking() {
    // Create optimizer state
    let point = DVector::from_vec(vec![1.0, 2.0]);
    let gradient = DVector::from_vec(vec![0.1, 0.2]);
    
    let state = OptimizerStateLegacy {
        point: point.clone(),
        value: 5.0,
        gradient: Some(gradient.clone()),
        gradient_norm: Some(gradient.norm()),
        previous_point: None,
        previous_value: None,
        iteration: 1,
        function_evaluations: 2,
        gradient_evaluations: 1,
        start_time: std::time::Instant::now(),
    };
    
    // Test state accessors
    assert_eq!(state.iteration, 1);
    assert_eq!(state.value, 5.0);
    assert!(state.gradient_norm.is_some());
}

#[test]
fn test_convergence_checking() {
    let stopping = StoppingCriterion::default()
        .with_max_iterations(100)
        .with_gradient_tolerance(1e-6);
    
    let point = DVector::from_vec(vec![0.0, 0.0]);
    let state = OptimizerStateLegacy {
        point: point,
        value: 0.0001,
        gradient: None,
        gradient_norm: Some(1e-8),
        previous_point: None,
        previous_value: Some(0.001),
        iteration: 50,
        function_evaluations: 100,
        gradient_evaluations: 50,
        start_time: std::time::Instant::now(),
    };
    
    // Should converge due to small gradient
    let reason = ConvergenceChecker::check(&state, &EuclideanSpace::new(2), &stopping).unwrap();
    assert!(reason.is_some());
}

#[test]
fn test_derivative_checking() {
    let _manifold = EuclideanSpace::new(2);
    let cost_fn = SimpleQuadratic::new(2);
    
    let point = DVector::from_vec(vec![1.0, 1.0]);
    
    // Check gradient via finite differences using DerivativeChecker
    let tolerance = 1e-6;
    let (is_correct, max_error) = DerivativeChecker::check_gradient(&cost_fn, &point, tolerance).unwrap();
    
    // For exact gradient, should be correct
    assert!(is_correct);
    assert!(max_error < 1e-6);
}