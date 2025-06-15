//! Core optimizer traits and types for Riemannian optimization.
//!
//! This module provides the foundational traits and types for implementing
//! optimization algorithms on Riemannian manifolds. It defines the interface
//! that all optimizers must implement, along with common structures for
//! optimization results and stopping criteria.
//!
//! # Key Components
//!
//! - **Optimizer trait**: Core interface for all optimization algorithms
//! - **OptimizationResult**: Encapsulates the result of an optimization run
//! - **StoppingCriterion**: Various conditions for terminating optimization
//! - **ConvergenceChecker**: Logic for checking convergence conditions

use crate::{
    error::Result,
    manifold::{Manifold, Point, TangentVector},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Result of an optimization run.
///
/// Contains the final point, objective value, and metadata about the
/// optimization process.
#[derive(Debug, Clone)]
pub struct OptimizationResult<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// The optimal point found by the optimizer
    pub point: Point<T, D>,

    /// The objective value at the optimal point
    pub value: T,

    /// The gradient norm at the optimal point (if available)
    pub gradient_norm: Option<T>,

    /// Number of iterations performed
    pub iterations: usize,

    /// Number of function evaluations
    pub function_evaluations: usize,

    /// Number of gradient evaluations
    pub gradient_evaluations: usize,

    /// Total optimization time
    pub duration: Duration,

    /// Reason for termination
    pub termination_reason: TerminationReason,

    /// Whether the optimization converged successfully
    pub converged: bool,
}

impl<T, D> OptimizationResult<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new optimization result.
    pub fn new(
        point: Point<T, D>,
        value: T,
        iterations: usize,
        duration: Duration,
        termination_reason: TerminationReason,
    ) -> Self {
        let converged = matches!(
            termination_reason,
            TerminationReason::Converged | TerminationReason::TargetReached
        );

        Self {
            point,
            value,
            gradient_norm: None,
            iterations,
            function_evaluations: 0,
            gradient_evaluations: 0,
            duration,
            termination_reason,
            converged,
        }
    }

    /// Sets the gradient norm at the optimal point.
    pub fn with_gradient_norm(mut self, norm: T) -> Self {
        self.gradient_norm = Some(norm);
        self
    }

    /// Sets the function evaluation count.
    pub fn with_function_evaluations(mut self, count: usize) -> Self {
        self.function_evaluations = count;
        self
    }

    /// Sets the gradient evaluation count.
    pub fn with_gradient_evaluations(mut self, count: usize) -> Self {
        self.gradient_evaluations = count;
        self
    }
}

/// Reason for termination of the optimization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// Converged to a stationary point (gradient norm below tolerance)
    Converged,
    /// Reached target objective value
    TargetReached,
    /// Maximum iterations reached
    MaxIterations,
    /// Maximum time exceeded
    MaxTime,
    /// Maximum function evaluations reached
    MaxFunctionEvaluations,
    /// Line search failed to find acceptable step
    LineSearchFailed,
    /// Numerical error occurred
    NumericalError,
    /// User requested termination
    UserTerminated,
}

/// Stopping criteria for optimization algorithms.
#[derive(Debug, Clone)]
pub struct StoppingCriterion<T>
where
    T: Scalar,
{
    /// Maximum number of iterations
    pub max_iterations: Option<usize>,

    /// Maximum optimization time
    pub max_time: Option<Duration>,

    /// Maximum number of function evaluations
    pub max_function_evaluations: Option<usize>,

    /// Gradient norm tolerance for convergence
    pub gradient_tolerance: Option<T>,

    /// Function value change tolerance
    pub function_tolerance: Option<T>,

    /// Point change tolerance
    pub point_tolerance: Option<T>,

    /// Target objective value (stop when reached)
    pub target_value: Option<T>,
}

impl<T> Default for StoppingCriterion<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            max_iterations: Some(1000),
            max_time: None,
            max_function_evaluations: None,
            gradient_tolerance: Some(<T as Scalar>::from_f64(1e-6)),
            function_tolerance: Some(<T as Scalar>::from_f64(1e-9)),
            point_tolerance: Some(<T as Scalar>::from_f64(1e-9)),
            target_value: None,
        }
    }
}

impl<T> StoppingCriterion<T>
where
    T: Scalar,
{
    /// Creates a new stopping criterion with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = Some(max_iter);
        self
    }

    /// Sets the maximum optimization time.
    pub fn with_max_time(mut self, max_time: Duration) -> Self {
        self.max_time = Some(max_time);
        self
    }

    /// Sets the gradient tolerance.
    pub fn with_gradient_tolerance(mut self, tol: T) -> Self {
        self.gradient_tolerance = Some(tol);
        self
    }

    /// Sets the function value change tolerance.
    pub fn with_function_tolerance(mut self, tol: T) -> Self {
        self.function_tolerance = Some(tol);
        self
    }

    /// Sets the point change tolerance.
    pub fn with_point_tolerance(mut self, tol: T) -> Self {
        self.point_tolerance = Some(tol);
        self
    }

    /// Sets the target objective value.
    pub fn with_target_value(mut self, target: T) -> Self {
        self.target_value = Some(target);
        self
    }
}

/// State information for optimization algorithms.
pub struct OptimizerState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Current point on the manifold
    pub point: Point<T, D>,

    /// Current objective value
    pub value: T,

    /// Current gradient (if available)
    pub gradient: Option<TangentVector<T, D>>,

    /// Gradient norm
    pub gradient_norm: Option<T>,

    /// Previous point
    pub previous_point: Option<Point<T, D>>,

    /// Previous objective value
    pub previous_value: Option<T>,

    /// Current iteration number
    pub iteration: usize,

    /// Number of function evaluations so far
    pub function_evaluations: usize,

    /// Number of gradient evaluations so far
    pub gradient_evaluations: usize,

    /// Start time of optimization
    pub start_time: Instant,
}

impl<T, D> OptimizerState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new optimizer state.
    pub fn new(point: Point<T, D>, value: T) -> Self {
        Self {
            point,
            value,
            gradient: None,
            gradient_norm: None,
            previous_point: None,
            previous_value: None,
            iteration: 0,
            function_evaluations: 1,
            gradient_evaluations: 0,
            start_time: Instant::now(),
        }
    }

    /// Updates the state with a new point and value.
    pub fn update(&mut self, point: Point<T, D>, value: T) {
        self.previous_point = Some(std::mem::replace(&mut self.point, point));
        self.previous_value = Some(self.value);
        self.value = value;
        self.iteration += 1;
        self.function_evaluations += 1;
    }

    /// Sets the current gradient.
    pub fn set_gradient(&mut self, gradient: TangentVector<T, D>, norm: T) {
        self.gradient = Some(gradient);
        self.gradient_norm = Some(norm);
        self.gradient_evaluations += 1;
    }

    /// Computes the change in objective value.
    pub fn value_change(&self) -> Option<T> {
        self.previous_value
            .map(|prev| <T as num_traits::Float>::abs(self.value - prev))
    }

    /// Computes the distance between current and previous points.
    pub fn point_change(&self, manifold: &impl Manifold<T, D>) -> Result<Option<T>> {
        match &self.previous_point {
            Some(prev) => manifold.distance(&self.point, prev).map(Some),
            None => Ok(None),
        }
    }

    /// Gets the elapsed time since optimization started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Trait for optimization algorithms on Riemannian manifolds.
///
/// This trait defines the interface that all optimizers must implement.
/// It provides methods for setting up the optimization problem and
/// running the optimization algorithm.
pub trait Optimizer<T, D>: Debug
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// The type of the cost function
    type CostFunction;

    /// The type of the manifold
    type Manifold: Manifold<T, D>;

    /// Returns the name of the optimizer.
    fn name(&self) -> &str;

    /// Optimizes the given cost function on the manifold.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to minimize
    /// * `manifold` - The manifold on which to optimize
    /// * `initial_point` - Starting point for optimization
    /// * `stopping_criterion` - Conditions for terminating optimization
    ///
    /// # Returns
    ///
    /// An `OptimizationResult` containing the optimal point and metadata.
    fn optimize(
        &mut self,
        cost_fn: &Self::CostFunction,
        manifold: &Self::Manifold,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>;

    /// Performs a single optimization step.
    ///
    /// This method is useful for implementing custom optimization loops
    /// or for debugging purposes.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function
    /// * `manifold` - The manifold
    /// * `state` - Current optimizer state
    ///
    /// # Returns
    ///
    /// Updated state after one iteration.
    fn step(
        &mut self,
        cost_fn: &Self::CostFunction,
        manifold: &Self::Manifold,
        state: &mut OptimizerState<T, D>,
    ) -> Result<()>;
}

/// Convergence checker for optimization algorithms.
pub struct ConvergenceChecker;

impl ConvergenceChecker {
    /// Checks if any stopping criterion has been met.
    ///
    /// # Arguments
    ///
    /// * `state` - Current optimizer state
    /// * `manifold` - The manifold
    /// * `criterion` - Stopping criteria to check
    ///
    /// # Returns
    ///
    /// The termination reason if any criterion is met, otherwise None.
    pub fn check<T, D>(
        state: &OptimizerState<T, D>,
        manifold: &impl Manifold<T, D>,
        criterion: &StoppingCriterion<T>,
    ) -> Result<Option<TerminationReason>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        // Check iteration limit
        if let Some(max_iter) = criterion.max_iterations {
            if state.iteration >= max_iter {
                return Ok(Some(TerminationReason::MaxIterations));
            }
        }

        // Check time limit
        if let Some(max_time) = criterion.max_time {
            if state.elapsed() >= max_time {
                return Ok(Some(TerminationReason::MaxTime));
            }
        }

        // Check function evaluation limit
        if let Some(max_evals) = criterion.max_function_evaluations {
            if state.function_evaluations >= max_evals {
                return Ok(Some(TerminationReason::MaxFunctionEvaluations));
            }
        }

        // Check gradient norm
        if let (Some(grad_norm), Some(grad_tol)) =
            (state.gradient_norm, criterion.gradient_tolerance)
        {
            if grad_norm < grad_tol {
                return Ok(Some(TerminationReason::Converged));
            }
        }

        // Check function value change
        if let (Some(val_change), Some(val_tol)) =
            (state.value_change(), criterion.function_tolerance)
        {
            if val_change < val_tol && state.iteration > 0 {
                return Ok(Some(TerminationReason::Converged));
            }
        }

        // Check point change
        if let Some(point_tol) = criterion.point_tolerance {
            if let Some(point_change) = state.point_change(manifold)? {
                if point_change < point_tol && state.iteration > 0 {
                    return Ok(Some(TerminationReason::Converged));
                }
            }
        }

        // Check target value
        if let Some(target) = criterion.target_value {
            if state.value <= target {
                return Ok(Some(TerminationReason::TargetReached));
            }
        }

        Ok(None)
    }

    /// Checks if the optimization has converged based on multiple criteria.
    ///
    /// This is a more sophisticated convergence check that requires multiple
    /// criteria to be satisfied simultaneously.
    pub fn check_strict<T, D>(
        state: &OptimizerState<T, D>,
        manifold: &impl Manifold<T, D>,
        criterion: &StoppingCriterion<T>,
        require_gradient: bool,
        require_value_and_point: bool,
    ) -> Result<Option<TerminationReason>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        // First check non-convergence termination conditions
        let basic_check = Self::check(state, manifold, criterion)?;
        if let Some(reason) = basic_check {
            if !matches!(reason, TerminationReason::Converged) {
                return Ok(Some(reason));
            }
        }

        let mut converged = true;

        // Check gradient criterion if required
        if require_gradient {
            if let (Some(grad_norm), Some(grad_tol)) =
                (state.gradient_norm, criterion.gradient_tolerance)
            {
                converged &= grad_norm < grad_tol;
            } else {
                converged = false;
            }
        }

        // Check value and point change if required
        if require_value_and_point && state.iteration > 0 {
            let value_ok = if let (Some(val_change), Some(val_tol)) =
                (state.value_change(), criterion.function_tolerance)
            {
                val_change < val_tol
            } else {
                false
            };

            let point_ok = if let Some(point_tol) = criterion.point_tolerance {
                if let Some(point_change) = state.point_change(manifold)? {
                    point_change < point_tol
                } else {
                    false
                }
            } else {
                false
            };

            converged &= value_ok && point_ok;
        }

        if converged && state.iteration > 0 {
            Ok(Some(TerminationReason::Converged))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_manifolds::MinimalTestManifold;
    use crate::types::DVector;
    use nalgebra::Dyn;
    use std::time::Duration;

    #[test]
    fn test_optimization_result() {
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = OptimizationResult::new(
            point.clone(),
            0.5,
            100,
            Duration::from_secs(1),
            TerminationReason::Converged,
        );

        assert_eq!(result.point, point);
        assert_eq!(result.value, 0.5);
        assert_eq!(result.iterations, 100);
        assert!(result.converged);
        assert_eq!(result.termination_reason, TerminationReason::Converged);
    }

    #[test]
    fn test_stopping_criterion() {
        let criterion = StoppingCriterion::<f64>::new()
            .with_max_iterations(500)
            .with_gradient_tolerance(1e-8)
            .with_function_tolerance(1e-10);

        assert_eq!(criterion.max_iterations, Some(500));
        assert_eq!(criterion.gradient_tolerance, Some(1e-8));
        assert_eq!(criterion.function_tolerance, Some(1e-10));
    }

    #[test]
    fn test_optimizer_state() {
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let mut state = OptimizerState::new(point.clone(), 1.0);

        assert_eq!(state.iteration, 0);
        assert_eq!(state.function_evaluations, 1);
        assert_eq!(state.gradient_evaluations, 0);

        let new_point = DVector::from_vec(vec![0.9, 0.1, 0.0]);
        state.update(new_point.clone(), 0.9);

        assert_eq!(state.iteration, 1);
        assert_eq!(state.point, new_point);
        assert_eq!(state.value, 0.9);
        assert_eq!(state.previous_value, Some(1.0));
        assert!((state.value_change().unwrap() - 0.1f64).abs() < 1e-10);
    }

    #[test]
    fn test_convergence_checker_iterations() {
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let mut state = OptimizerState::<f64, Dyn>::new(point, 1.0);
        state.iteration = 1000;

        let criterion = StoppingCriterion::new().with_max_iterations(1000);

        let manifold = MinimalTestManifold::new(3);
        let result = ConvergenceChecker::check(&state, &manifold, &criterion).unwrap();
        assert_eq!(result, Some(TerminationReason::MaxIterations));
    }

    #[test]
    fn test_convergence_checker_gradient() {
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let mut state = OptimizerState::<f64, Dyn>::new(point.clone(), 1.0);
        let gradient = DVector::from_vec(vec![1e-7, 0.0, 0.0]);
        state.set_gradient(gradient, 1e-7);

        let criterion = StoppingCriterion::new().with_gradient_tolerance(1e-6);

        #[derive(Debug)]
        struct MockManifold;
        impl Manifold<f64, Dyn> for MockManifold {
            fn name(&self) -> &str {
                "Mock"
            }
            fn dimension(&self) -> usize {
                3
            }
            fn is_point_on_manifold(&self, _: &DVector<f64>, _: f64) -> bool {
                true
            }
            fn is_vector_in_tangent_space(
                &self,
                _: &DVector<f64>,
                _: &DVector<f64>,
                _: f64,
            ) -> bool {
                true
            }
            fn project_point(&self, p: &DVector<f64>) -> DVector<f64> {
                p.clone()
            }
            fn project_tangent(&self, _: &DVector<f64>, v: &DVector<f64>) -> Result<DVector<f64>> {
                Ok(v.clone())
            }
            fn inner_product(
                &self,
                _: &DVector<f64>,
                u: &DVector<f64>,
                v: &DVector<f64>,
            ) -> Result<f64> {
                Ok(u.dot(v))
            }
            fn retract(&self, p: &DVector<f64>, v: &DVector<f64>) -> Result<DVector<f64>> {
                Ok(p + v)
            }
            fn inverse_retract(&self, p: &DVector<f64>, q: &DVector<f64>) -> Result<DVector<f64>> {
                Ok(q - p)
            }
            fn euclidean_to_riemannian_gradient(
                &self,
                _: &DVector<f64>,
                g: &DVector<f64>,
            ) -> Result<DVector<f64>> {
                Ok(g.clone())
            }
            fn random_point(&self) -> DVector<f64> {
                DVector::zeros(3)
            }
            fn random_tangent(&self, _: &DVector<f64>) -> Result<DVector<f64>> {
                Ok(DVector::zeros(3))
            }
        }

        let manifold = MockManifold;
        let result = ConvergenceChecker::check(&state, &manifold, &criterion).unwrap();
        assert_eq!(result, Some(TerminationReason::Converged));
    }
}
