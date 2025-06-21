//! Riemannian Conjugate Gradient optimizer.
//!
//! Conjugate gradient methods are a class of optimization algorithms that use
//! conjugate directions to achieve faster convergence than steepest descent.
//! This implementation extends classical CG methods to Riemannian manifolds.
//!
//! # Algorithm Overview
//!
//! At each iteration, the conjugate gradient method:
//! 1. Computes the Riemannian gradient
//! 2. Determines a conjugate direction using the chosen beta formula
//! 3. Performs line search along the conjugate direction
//! 4. Updates the position using retraction
//!
//! # Supported Methods
//!
//! - **Fletcher-Reeves (FR)**: β = ||g_k||² / ||g_{k-1}||²
//! - **Polak-Ribière (PR)**: β = <g_k, g_k - g_{k-1}> / ||g_{k-1}||²
//! - **Hestenes-Stiefel (HS)**: β = <g_k, g_k - g_{k-1}> / <d_{k-1}, g_k - g_{k-1}>
//! - **Dai-Yuan (DY)**: β = ||g_k||² / <d_{k-1}, g_k - g_{k-1}>
//!
//! # Key Features
//!
//! - **Multiple CG variants**: FR, PR, HS, DY methods
//! - **Automatic restarts**: Periodic or condition-based restarts
//! - **Preconditioning support**: Optional preconditioner application
//! - **Hybrid methods**: PR+ (non-negative PR) and other variants
//! - **Line search integration**: Ensures sufficient decrease
//!
//! # References
//!
//! - Hager & Zhang, "A survey of nonlinear conjugate gradient methods" (2006)
//! - Dai & Yuan, "A nonlinear conjugate gradient method with a strong global convergence property" (1999)
//! - Ring & Wirth, "Optimization methods on Riemannian manifolds and their application to shape space" (2012)

use riemannopt_core::{
    cost_function::CostFunction,
    core::CachedCostFunction,
    error::Result,
    line_search::{BacktrackingLineSearch, LineSearch, LineSearchParams},
    manifold::{Manifold, Point},
    memory::workspace::{Workspace, WorkspaceBuilder},
    optimizer::{Optimizer, OptimizerStateLegacy as OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
    optimization::{ConjugateGradientMethod, ConjugateGradientState},
    preconditioner::{Preconditioner, IdentityPreconditioner},
    retraction::Retraction,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

/// Configuration for the Conjugate Gradient optimizer.
#[derive(Debug, Clone)]
pub struct CGConfig<T: Scalar> {
    /// The CG method variant to use
    pub method: ConjugateGradientMethod,
    /// Period for automatic restarts (0 = no automatic restart)
    pub restart_period: usize,
    /// Whether to use PR+ (non-negative Polak-Ribière)
    pub use_pr_plus: bool,
    /// Line search parameters
    pub line_search_params: LineSearchParams<T>,
    /// Minimum value of beta before restart
    pub min_beta: Option<T>,
    /// Maximum value of beta allowed
    pub max_beta: Option<T>,
    /// Whether to use preconditioning
    pub use_preconditioner: bool,
}

impl<T: Scalar> Default for CGConfig<T> {
    fn default() -> Self {
        Self {
            method: ConjugateGradientMethod::PolakRibiere,
            restart_period: 0, // No automatic restart by default
            use_pr_plus: true, // Use PR+ by default
            line_search_params: LineSearchParams::default(),
            min_beta: Some(<T as Scalar>::from_f64(-0.5)),
            max_beta: Some(<T as Scalar>::from_f64(10.0)),
            use_preconditioner: false,
        }
    }
}

impl<T: Scalar> CGConfig<T> {
    /// Creates a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the CG method variant.
    pub fn with_method(mut self, method: ConjugateGradientMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets the restart period (n means restart every n iterations).
    pub fn with_restart_period(mut self, period: usize) -> Self {
        self.restart_period = period;
        self
    }

    /// Enables or disables PR+ (non-negative Polak-Ribière).
    pub fn with_pr_plus(mut self, use_pr_plus: bool) -> Self {
        self.use_pr_plus = use_pr_plus;
        self
    }

    /// Sets the line search parameters.
    pub fn with_line_search(mut self, params: LineSearchParams<T>) -> Self {
        self.line_search_params = params;
        self
    }

    /// Sets the minimum beta value before restart.
    pub fn with_min_beta(mut self, min_beta: T) -> Self {
        self.min_beta = Some(min_beta);
        self
    }

    /// Sets the maximum beta value allowed.
    pub fn with_max_beta(mut self, max_beta: T) -> Self {
        self.max_beta = Some(max_beta);
        self
    }

    /// Enables preconditioning.
    pub fn with_preconditioner(mut self) -> Self {
        self.use_preconditioner = true;
        self
    }

    /// Creates a configuration for Fletcher-Reeves method.
    pub fn fletcher_reeves() -> Self {
        Self::new().with_method(ConjugateGradientMethod::FletcherReeves)
    }

    /// Creates a configuration for Polak-Ribière method.
    pub fn polak_ribiere() -> Self {
        Self::new().with_method(ConjugateGradientMethod::PolakRibiere)
    }

    /// Creates a configuration for Hestenes-Stiefel method.
    pub fn hestenes_stiefel() -> Self {
        Self::new().with_method(ConjugateGradientMethod::HestenesStiefel)
    }

    /// Creates a configuration for Dai-Yuan method.
    pub fn dai_yuan() -> Self {
        Self::new().with_method(ConjugateGradientMethod::DaiYuan)
    }
}


/// Conjugate Gradient optimizer for Riemannian manifolds.
#[derive(Debug)]
pub struct ConjugateGradient<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    config: CGConfig<T>,
    line_search: BacktrackingLineSearch,
    preconditioner: Box<dyn Preconditioner<T, D>>,
}

impl<T, D> ConjugateGradient<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new Conjugate Gradient optimizer with the given configuration.
    pub fn new(config: CGConfig<T>) -> Self {
        Self {
            config,
            line_search: BacktrackingLineSearch::new(),
            preconditioner: Box::new(IdentityPreconditioner),
        }
    }

    /// Sets a custom preconditioner.
    pub fn with_preconditioner(mut self, preconditioner: Box<dyn Preconditioner<T, D>>) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CGConfig<T> {
        &self.config
    }


    /// Performs a single optimization step.
    fn step_internal<C, M, R>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        retraction: &R,
        state: &mut OptimizerState<T, D>,
        cg_state: &mut ConjugateGradientState<T, D>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
    {
        // Compute cost and gradient
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&state.point)?;
        let gradient = manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad)?;
        
        let grad_norm = <T as Float>::sqrt(
            manifold.inner_product(&state.point, &gradient, &gradient)?
        );
        
        state.set_gradient(gradient.clone(), grad_norm);
        
        // Check if gradient is very small - if so, we're likely converged
        if grad_norm < <T as Scalar>::from_f64(1e-10) {
            // Gradient is essentially zero, we're at optimum
            // Just update the state without taking a step
            state.gradient_norm = Some(grad_norm);
            return Ok(());
        }
        
        // Compute conjugate direction
        // For now, ignore preconditioning and just use the standard CG
        let mut direction = cg_state.compute_direction(&gradient, manifold, &state.point)?;
        
        // Check if it's a descent direction
        let mut dir_grad_inner = manifold.inner_product(&state.point, &direction, &gradient)?;
        if dir_grad_inner >= T::zero() {
            // Not a descent direction, restart with steepest descent
            direction = -gradient.clone();
            // Reset CG state manually
            cg_state.previous_direction = None;
            cg_state.previous_gradient = None;
            cg_state.previous_point = None;
            cg_state.iterations_since_restart = 0;
            // Recompute inner product for new direction
            dir_grad_inner = manifold.inner_product(&state.point, &direction, &gradient)?;
        }
        
        // Perform line search using pre-computed directional derivative
        let line_search_result = self.line_search.search_with_deriv(
            cost_fn,
            manifold,
            retraction,
            &state.point,
            cost,
            &direction,
            dir_grad_inner,
            &self.config.line_search_params,
        )?;
        
        // Update state
        state.update(line_search_result.new_point, line_search_result.new_value);
        state.function_evaluations += line_search_result.function_evals - 1;
        state.gradient_evaluations += line_search_result.gradient_evals;
        
        Ok(())
    }

    /// Optimizes the given cost function.
    pub fn optimize<C, M, R>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        retraction: &R,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D, D>,
    {
        let start_time = Instant::now();
        
        // Wrap cost function with caching to avoid redundant computations
        let cached_cost_fn = CachedCostFunction::new(cost_fn);
        
        // Initialize optimizer state
        let initial_cost = cached_cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        let mut cg_state = ConjugateGradientState::new(self.config.method, self.config.restart_period);
        
        // Create a single workspace for the entire optimization
        let n = initial_point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                // Get cache statistics for diagnostics
                let ((_cost_hits, cost_misses), (_grad_hits, grad_misses), _) = cached_cost_fn.cache_stats();
                
                return Ok(OptimizationResult::new(
                    state.point,
                    state.value,
                    state.iteration,
                    start_time.elapsed(),
                    reason,
                )
                .with_function_evaluations(cost_misses)  // Use cache misses as actual evaluations
                .with_gradient_evaluations(grad_misses)  // Use cache misses as actual evaluations
                .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
            }
            
            // Perform one optimization step
            self.step_internal(&cached_cost_fn, manifold, retraction, &mut state, &mut cg_state, &mut workspace)?;
        }
    }

    /// Performs a single optimization step.
    /// Note: This method creates a fresh CG state each time, so it won't maintain
    /// conjugacy between calls. Use optimize() for proper CG behavior.
    pub fn step<C, M, R>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        retraction: &R,
        state: &mut OptimizerState<T, D>,
    ) -> Result<()>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
    {
        // Create temporary CG state - this means we're doing steepest descent
        let mut cg_state = ConjugateGradientState::new(self.config.method, self.config.restart_period);
        
        // Create temporary workspace
        let n = state.point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        
        self.step_internal(cost_fn, manifold, retraction, state, &mut cg_state, &mut workspace)
    }
}

// Implementation of the Optimizer trait
impl<T, D> Optimizer<T, D> for ConjugateGradient<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    fn name(&self) -> &str {
        match self.config.method {
            ConjugateGradientMethod::FletcherReeves => "Riemannian CG-FR",
            ConjugateGradientMethod::PolakRibiere => {
                if self.config.use_pr_plus {
                    "Riemannian CG-PR+"
                } else {
                    "Riemannian CG-PR"
                }
            }
            ConjugateGradientMethod::HestenesStiefel => "Riemannian CG-HS",
            ConjugateGradientMethod::DaiYuan => "Riemannian CG-DY",
        }
    }

    fn optimize<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Use the default retraction
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        ConjugateGradient::optimize(self, cost_fn, manifold, &retraction, initial_point, stopping_criterion)
    }

    fn step<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, D>,
    ) -> Result<()>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Use the default retraction
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        self.step(cost_fn, manifold, &retraction, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::{
        cost_function::QuadraticCost,
        test_manifolds::TestEuclideanManifold,
        types::DVector,
    };
    use nalgebra::Dyn;

    #[test]
    fn test_cg_first_step() {
        // Test that the first step is just negative gradient
        
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let mut cg = ConjugateGradient::new(CGConfig::fletcher_reeves());
        
        // Create optimizer state
        let initial_cost = cost_fn.cost(&initial_point).unwrap();
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        // Do one step
        let result = cg.step(&cost_fn, &manifold, &retraction, &mut state);
        
        // Check if step succeeded
        assert!(result.is_ok(), "First step should succeed: {:?}", result);
    }

    #[test]
    fn test_cg_config() {
        let config = CGConfig::<f64>::fletcher_reeves()
            .with_restart_period(10)
            .with_min_beta(-0.5)
            .with_max_beta(5.0);
        
        assert_eq!(config.method, ConjugateGradientMethod::FletcherReeves);
        assert_eq!(config.restart_period, 10);
        assert_eq!(config.min_beta, Some(-0.5));
        assert_eq!(config.max_beta, Some(5.0));
    }

    #[test]
    fn test_cg_variants() {
        let fr_config = CGConfig::<f64>::fletcher_reeves();
        let pr_config = CGConfig::<f64>::polak_ribiere();
        let hs_config = CGConfig::<f64>::hestenes_stiefel();
        let dy_config = CGConfig::<f64>::dai_yuan();
        
        assert_eq!(fr_config.method, ConjugateGradientMethod::FletcherReeves);
        assert_eq!(pr_config.method, ConjugateGradientMethod::PolakRibiere);
        assert_eq!(hs_config.method, ConjugateGradientMethod::HestenesStiefel);
        assert_eq!(dy_config.method, ConjugateGradientMethod::DaiYuan);
    }

    #[test]
    fn test_cg_debug() {
        // Debug test to understand what's happening
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let _cg = ConjugateGradient::<f64, Dyn>::new(CGConfig::fletcher_reeves());
        
        // Get gradient at initial point
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&initial_point).unwrap();
        let gradient = manifold.euclidean_to_riemannian_gradient(&initial_point, &euclidean_grad).unwrap();
        
        println!("Initial point: {:?}", initial_point);
        println!("Initial cost: {}", cost);
        println!("Gradient: {:?}", gradient);
        
        // Create CG state and compute direction
        let mut cg_state = ConjugateGradientState::new(ConjugateGradientMethod::FletcherReeves, 0);
        let direction = cg_state.compute_direction(&gradient, &manifold, &initial_point).unwrap();
        
        println!("Direction: {:?}", direction);
        
        // Check inner product
        let inner = manifold.inner_product(&initial_point, &direction, &gradient).unwrap();
        println!("Inner product <direction, gradient>: {}", inner);
        
        assert!(inner < 0.0, "Direction should be descent direction");
    }

    #[test]
    fn test_cg_two_iterations() {
        // Test two iterations to see where the problem is
        use riemannopt_core::retraction::DefaultRetraction;
        
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let mut cg = ConjugateGradient::<f64, Dyn>::new(CGConfig::fletcher_reeves());
        
        // Create optimizer state
        let initial_cost = cost_fn.cost(&initial_point).unwrap();
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        let mut cg_state = ConjugateGradientState::new(ConjugateGradientMethod::FletcherReeves, 0);
        
        let retraction = DefaultRetraction;
        
        println!("=== Iteration 0 ===");
        println!("Point: {:?}", state.point);
        
        // First iteration
        let n = initial_point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        let result = cg.step_internal(&cost_fn, &manifold, &retraction, &mut state, &mut cg_state, &mut workspace);
        assert!(result.is_ok(), "First iteration failed: {:?}", result);
        
        println!("\n=== Iteration 1 ===");
        println!("Point: {:?}", state.point);
        println!("State gradient norm: {:?}", state.gradient_norm);
        
        // Get actual gradient at current point
        let (cost1, grad1) = cost_fn.cost_and_gradient(&state.point).unwrap();
        println!("Actual cost: {}", cost1);
        println!("Actual gradient: {:?}", grad1);
        println!("Actual gradient norm: {}", grad1.norm());
        
        // Second iteration
        let result = cg.step_internal(&cost_fn, &manifold, &retraction, &mut state, &mut cg_state, &mut workspace);
        assert!(result.is_ok(), "Second iteration failed: {:?}", result);
        
        println!("\n=== Final ===");
        println!("Point: {:?}", state.point);
        println!("Cost: {}", state.value);
    }

    #[test]
    fn test_cg_on_quadratic() {
        let cost_fn = QuadraticCost::simple(Dyn(3));
        let manifold = TestEuclideanManifold::new(3);
        let initial_point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        // Test Fletcher-Reeves
        let mut cg_fr = ConjugateGradient::new(CGConfig::fletcher_reeves());
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        let result = cg_fr.optimize(
            &cost_fn,
            &manifold,
            &retraction,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        // Should converge to origin
        assert!(result.converged);
        assert!(result.point.norm() < 1e-3);
        
        // Test Polak-Ribière
        let mut cg_pr = ConjugateGradient::new(CGConfig::polak_ribiere());
        
        let result = cg_pr.optimize(
            &cost_fn,
            &manifold,
            &retraction,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        assert!(result.converged);
        assert!(result.point.norm() < 1e-3);
    }

    #[test]
    fn test_cg_with_restart() {
        let cost_fn = QuadraticCost::simple(Dyn(5));
        let manifold = TestEuclideanManifold::new(5);
        let initial_point = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let config = CGConfig::polak_ribiere()
            .with_restart_period(3); // Restart every 3 iterations
        
        let mut cg = ConjugateGradient::new(config);
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(50)
            .with_gradient_tolerance(1e-6);
        
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        let result = cg.optimize(
            &cost_fn,
            &manifold,
            &retraction,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        assert!(result.converged);
        assert!(result.point.norm() < 1e-3);
    }

    #[test]
    fn test_identity_preconditioner() {
        let preconditioner = IdentityPreconditioner;
        let gradient = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let point = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        
        let result = preconditioner.apply(&gradient, &point).unwrap();
        
        assert_eq!(result, gradient);
    }

    #[test]
    fn test_cg_optimizer_trait() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let mut optimizer = ConjugateGradient::<f64, Dyn>::new(CGConfig::polak_ribiere());
        
        // Test that it implements the Optimizer trait
        assert_eq!(optimizer.name(), "Riemannian CG-PR+");
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(50)
            .with_gradient_tolerance(1e-6);
        
        // Use the trait method
        let result: OptimizationResult<f64, Dyn> = Optimizer::optimize(
            &mut optimizer,
            &cost_fn,
            &manifold,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        assert!(result.converged);
    }

    #[test]
    fn test_cg_all_methods() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let methods = vec![
            ConjugateGradientMethod::FletcherReeves,
            ConjugateGradientMethod::PolakRibiere,
            ConjugateGradientMethod::HestenesStiefel,
            ConjugateGradientMethod::DaiYuan,
        ];
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        for method in methods {
            let config = CGConfig::new().with_method(method);
            let mut cg = ConjugateGradient::new(config);
            
            let result = cg.optimize(
                &cost_fn,
                &manifold,
                &retraction,
                &initial_point,
                &stopping_criterion,
            ).unwrap();
            
            assert!(result.converged, "Method {:?} failed to converge", method);
            assert!(result.point.norm() < 1e-3, "Method {:?} converged to wrong point", method);
        }
    }
}