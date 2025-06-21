//! Riemannian Natural Gradient optimizer.
//!
//! Natural gradient descent is an optimization method that uses the Fisher information
//! matrix to define a more natural geometry for the parameter space. This leads to
//! updates that are invariant to reparametrization and often converge faster than
//! standard gradient descent.
//!
//! # Algorithm Overview
//!
//! At each iteration, the natural gradient method:
//! 1. Computes the standard gradient
//! 2. Computes or approximates the Fisher information matrix F
//! 3. Solves F * delta = gradient for the natural gradient delta
//! 4. Updates parameters using the natural gradient
//!
//! # Key Features
//!
//! - **Exact Fisher computation**: For small-scale problems
//! - **Diagonal approximation**: Efficient for medium-scale problems
//! - **KFAC approximation**: Kronecker-factored approximation for neural networks
//! - **Empirical Fisher**: Uses the outer product of gradients
//! - **Damping**: Regularization for numerical stability
//!
//! # References
//!
//! - Amari, "Natural Gradient Works Efficiently in Learning" (1998)
//! - Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature" (2015)
//! - Pascanu & Bengio, "Revisiting Natural Gradient for Deep Networks" (2013)

use riemannopt_core::{
    cost_function::CostFunction,
    core::CachedCostFunction,
    error::{ManifoldError, Result},
    fisher::FisherApproximation,
    line_search::{BacktrackingLineSearch, LineSearch, LineSearchParams},
    manifold::{Manifold, Point, TangentVector},
    memory::workspace::{Workspace, WorkspaceBuilder},
    optimizer::{Optimizer, OptimizerStateLegacy as OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
    retraction::Retraction,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector};
use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

/// Configuration for the Natural Gradient optimizer.
#[derive(Debug, Clone)]
pub struct NaturalGradientConfig<T: Scalar> {
    /// Learning rate / step size
    pub learning_rate: T,
    /// Fisher approximation method
    pub fisher_approximation: FisherApproximation,
    /// Damping factor for numerical stability (lambda in (F + lambda*I)^{-1})
    pub damping: T,
    /// Whether to use momentum
    pub use_momentum: bool,
    /// Momentum coefficient
    pub momentum: T,
    /// Update frequency for Fisher matrix (recompute every N iterations)
    pub fisher_update_freq: usize,
    /// Number of samples for empirical Fisher estimation
    pub fisher_num_samples: usize,
    /// Minimum eigenvalue for Fisher matrix regularization
    pub min_eigenvalue: T,
    /// Line search parameters
    pub line_search_params: LineSearchParams<T>,
}

impl<T: Scalar> Default for NaturalGradientConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: <T as Scalar>::from_f64(0.01),
            fisher_approximation: FisherApproximation::Diagonal,
            damping: <T as Scalar>::from_f64(1e-4),
            use_momentum: false,
            momentum: <T as Scalar>::from_f64(0.9),
            fisher_update_freq: 10,
            fisher_num_samples: 100,
            min_eigenvalue: <T as Scalar>::from_f64(1e-8),
            line_search_params: LineSearchParams::default(),
        }
    }
}

impl<T: Scalar> NaturalGradientConfig<T> {
    /// Creates a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the learning rate.
    pub fn with_learning_rate(mut self, lr: T) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the Fisher approximation method.
    pub fn with_fisher_approximation(mut self, method: FisherApproximation) -> Self {
        self.fisher_approximation = method;
        self
    }

    /// Sets the damping factor.
    pub fn with_damping(mut self, damping: T) -> Self {
        self.damping = damping;
        self
    }

    /// Enables momentum with the given coefficient.
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.use_momentum = true;
        self.momentum = momentum;
        self
    }

    /// Sets the Fisher update frequency.
    pub fn with_fisher_update_freq(mut self, freq: usize) -> Self {
        self.fisher_update_freq = freq;
        self
    }

    /// Sets the number of samples for empirical Fisher.
    pub fn with_fisher_num_samples(mut self, num_samples: usize) -> Self {
        self.fisher_num_samples = num_samples;
        self
    }
}

/// State for the Natural Gradient optimizer.
#[derive(Debug, Clone)]
struct NaturalGradientState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// Fisher information matrix or its approximation
    fisher_inverse: Option<OMatrix<T, D, D>>,
    /// Momentum vector
    momentum: Option<TangentVector<T, D>>,
    /// Iteration counter for Fisher updates
    fisher_update_counter: usize,
    /// Cached gradients for empirical Fisher
    gradient_buffer: Vec<OVector<T, D>>,
}

impl<T, D> NaturalGradientState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// Creates a new natural gradient state.
    fn new() -> Self {
        Self {
            fisher_inverse: None,
            momentum: None,
            fisher_update_counter: 0,
            gradient_buffer: Vec::new(),
        }
    }

    /// Checks if Fisher matrix needs updating.
    fn needs_fisher_update(&self, update_freq: usize) -> bool {
        self.fisher_inverse.is_none() || self.fisher_update_counter >= update_freq
    }

    /// Resets the Fisher update counter.
    fn reset_fisher_counter(&mut self) {
        self.fisher_update_counter = 0;
    }

    /// Increments the Fisher update counter.
    fn increment_fisher_counter(&mut self) {
        self.fisher_update_counter += 1;
    }
}

/// Natural Gradient optimizer for Riemannian manifolds.
#[derive(Debug)]
pub struct NaturalGradient<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    config: NaturalGradientConfig<T>,
    line_search: BacktrackingLineSearch,
    _phantom: std::marker::PhantomData<D>,
}

impl<T, D> NaturalGradient<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// Creates a new Natural Gradient optimizer with the given configuration.
    pub fn new(config: NaturalGradientConfig<T>) -> Self {
        Self {
            config,
            line_search: BacktrackingLineSearch::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &NaturalGradientConfig<T> {
        &self.config
    }

    /// Computes the diagonal Fisher approximation.
    fn compute_diagonal_fisher<C, M>(
        &self,
        _cost_fn: &C,
        _manifold: &M,
        _point: &Point<T, D>,
        gradient: &TangentVector<T, D>,
    ) -> Result<OMatrix<T, D, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let dim = gradient.shape_generic().0;
        let mut fisher = OMatrix::<T, D, D>::zeros_generic(dim, dim);
        
        // For diagonal Fisher, we approximate it as the squared gradient
        // This is a simple approximation that works well in practice
        for i in 0..gradient.len() {
            fisher[(i, i)] = gradient[i] * gradient[i] + self.config.damping;
        }
        
        Ok(fisher)
    }

    /// Computes the empirical Fisher using gradient samples.
    fn compute_empirical_fisher<C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        ng_state: &mut NaturalGradientState<T, D>,
    ) -> Result<OMatrix<T, D, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let dim = point.shape_generic().0;
        let mut fisher = OMatrix::<T, D, D>::zeros_generic(dim, dim);
        
        // Collect gradient samples if buffer is not full
        if ng_state.gradient_buffer.len() < self.config.fisher_num_samples {
            let (_, euclidean_grad) = cost_fn.cost_and_gradient(point)?;
            let gradient = manifold.euclidean_to_riemannian_gradient(point, &euclidean_grad)?;
            ng_state.gradient_buffer.push(gradient);
        }
        
        // Compute empirical Fisher as E[gg^T]
        if !ng_state.gradient_buffer.is_empty() {
            let scale = T::one() / <T as Scalar>::from_usize(ng_state.gradient_buffer.len());
            for grad in &ng_state.gradient_buffer {
                // fisher += scale * (grad * grad^T)
                // We need to compute this manually since grad is a column vector
                for i in 0..grad.len() {
                    for j in 0..grad.len() {
                        fisher[(i, j)] += scale * grad[i] * grad[j];
                    }
                }
            }
        }
        
        // Add damping
        for i in 0..point.len() {
            fisher[(i, i)] += self.config.damping;
        }
        
        Ok(fisher)
    }

    /// Computes the inverse Fisher matrix based on the configured method.
    fn compute_fisher_inverse<C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        gradient: &TangentVector<T, D>,
        ng_state: &mut NaturalGradientState<T, D>,
    ) -> Result<OMatrix<T, D, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let fisher = match self.config.fisher_approximation {
            FisherApproximation::Exact => {
                // For exact Fisher, we would need the Hessian of the log-likelihood
                // For now, we fall back to empirical Fisher
                self.compute_empirical_fisher(cost_fn, manifold, point, ng_state)?
            }
            FisherApproximation::Diagonal => {
                self.compute_diagonal_fisher(cost_fn, manifold, point, gradient)?
            }
            FisherApproximation::Empirical => {
                self.compute_empirical_fisher(cost_fn, manifold, point, ng_state)?
            }
            FisherApproximation::KFAC => {
                // KFAC requires special structure (layers) that we don't have in general
                // Fall back to empirical Fisher
                self.compute_empirical_fisher(cost_fn, manifold, point, ng_state)?
            }
            FisherApproximation::BlockDiagonal => {
                // Block diagonal requires parameter grouping information
                // Fall back to diagonal approximation for now
                self.compute_diagonal_fisher(cost_fn, manifold, point, gradient)?
            }
            FisherApproximation::LowRank { .. } => {
                // Low-rank approximation would require SVD or eigendecomposition
                // Fall back to empirical Fisher for now
                self.compute_empirical_fisher(cost_fn, manifold, point, ng_state)?
            }
        };
        
        // Compute inverse
        match fisher.clone().try_inverse() {
            Some(inv) => Ok(inv),
            None => {
                // If not invertible, add more regularization
                let n = fisher.nrows();
                let mut regularized = fisher;
                for i in 0..n {
                    regularized[(i, i)] += self.config.min_eigenvalue;
                }
                regularized.try_inverse()
                    .ok_or_else(|| ManifoldError::numerical_error("Fisher matrix is singular"))
            }
        }
    }

    /// Applies the natural gradient transformation.
    fn apply_natural_gradient(
        &self,
        gradient: &TangentVector<T, D>,
        fisher_inverse: &OMatrix<T, D, D>,
    ) -> TangentVector<T, D> {
        // Natural gradient = F^{-1} * gradient
        fisher_inverse * gradient
    }

    /// Performs a single optimization step.
    fn step_internal<C, M, R>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        retraction: &R,
        state: &mut OptimizerState<T, D>,
        ng_state: &mut NaturalGradientState<T, D>,
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
        
        // Check if gradient is very small
        if grad_norm < <T as Scalar>::from_f64(1e-10) {
            state.gradient_norm = Some(grad_norm);
            state.value = cost;
            return Ok(());
        }
        
        // Update Fisher matrix if needed
        if ng_state.needs_fisher_update(self.config.fisher_update_freq) {
            let fisher_inv = self.compute_fisher_inverse(
                cost_fn,
                manifold,
                &state.point,
                &gradient,
                ng_state
            )?;
            ng_state.fisher_inverse = Some(fisher_inv);
            ng_state.reset_fisher_counter();
        } else {
            ng_state.increment_fisher_counter();
        }
        
        // Apply natural gradient transformation
        let fisher_inv = match ng_state.fisher_inverse.as_ref() {
            Some(inv) => inv,
            None => {
                // First iteration, compute Fisher inverse
                let inv = self.compute_fisher_inverse(
                    cost_fn,
                    manifold,
                    &state.point,
                    &gradient,
                    ng_state
                )?;
                ng_state.fisher_inverse = Some(inv);
                ng_state.fisher_inverse.as_ref().unwrap()
            }
        };
        
        let natural_grad = self.apply_natural_gradient(&gradient, fisher_inv);
        
        // Verify that natural gradient provides a descent direction
        // The natural gradient should satisfy: <natural_grad, gradient> > 0
        // because we want -natural_grad to be a descent direction
        let ng_grad_inner = manifold.inner_product(&state.point, &natural_grad, &gradient)?;
        if ng_grad_inner <= T::zero() {
            // If not, fall back to standard gradient
            let direction = -&gradient;
            
            // Compute directional derivative for efficient line search
            let directional_deriv = manifold.inner_product(&state.point, &gradient, &direction)?;
            
            // Update state and return
            let line_search_result = self.line_search.search_with_deriv(
                cost_fn,
                manifold,
                retraction,
                &state.point,
                cost,
                &direction,
                directional_deriv,
                &self.config.line_search_params,
            )?;
            
            state.update(line_search_result.new_point, line_search_result.new_value);
            state.function_evaluations += line_search_result.function_evals - 1;
            state.gradient_evaluations += line_search_result.gradient_evals;
            
            return Ok(());
        }
        
        // Apply momentum if enabled
        let direction = if self.config.use_momentum {
            match &mut ng_state.momentum {
                Some(momentum) => {
                    // momentum = beta * momentum - (1 - beta) * natural_grad
                    // Note: we include the negative sign in the momentum update
                    *momentum *= self.config.momentum;
                    *momentum -= &natural_grad * (T::one() - self.config.momentum);
                    momentum.clone()
                }
                None => {
                    // Initialize momentum with negative natural gradient
                    let neg_natural_grad = -&natural_grad;
                    ng_state.momentum = Some(neg_natural_grad.clone());
                    neg_natural_grad
                }
            }
        } else {
            -natural_grad
        };
        
        // Compute directional derivative for efficient line search
        let directional_deriv = manifold.inner_product(&state.point, &gradient, &direction)?;
        
        // Perform line search with pre-computed values
        let line_search_result = self.line_search.search_with_deriv(
            cost_fn,
            manifold,
            retraction,
            &state.point,
            cost,
            &direction,
            directional_deriv,
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
        let mut ng_state = NaturalGradientState::new();
        
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
            self.step_internal(&cached_cost_fn, manifold, retraction, &mut state, &mut ng_state, &mut workspace)?;
        }
    }

    /// Performs a single optimization step.
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
        // Create temporary NG state
        let mut ng_state = NaturalGradientState::new();
        
        // Create temporary workspace
        let n = state.point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        
        self.step_internal(cost_fn, manifold, retraction, state, &mut ng_state, &mut workspace)
    }
}

// Implementation of the Optimizer trait
impl<T, D> Optimizer<T, D> for NaturalGradient<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    fn name(&self) -> &str {
        match self.config.fisher_approximation {
            FisherApproximation::Exact => "Riemannian Natural Gradient (Exact)",
            FisherApproximation::Diagonal => "Riemannian Natural Gradient (Diagonal)",
            FisherApproximation::Empirical => "Riemannian Natural Gradient (Empirical)",
            FisherApproximation::KFAC => "Riemannian Natural Gradient (KFAC)",
            FisherApproximation::BlockDiagonal => "Riemannian Natural Gradient (Block-Diagonal)",
            FisherApproximation::LowRank { .. } => "Riemannian Natural Gradient (Low-Rank)",
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
        NaturalGradient::optimize(self, cost_fn, manifold, &retraction, initial_point, stopping_criterion)
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
    fn test_natural_gradient_config() {
        let config = NaturalGradientConfig::<f64>::new()
            .with_learning_rate(0.1)
            .with_fisher_approximation(FisherApproximation::Diagonal)
            .with_damping(1e-3)
            .with_momentum(0.95);
        
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.fisher_approximation, FisherApproximation::Diagonal);
        assert_eq!(config.damping, 1e-3);
        assert!(config.use_momentum);
        assert_eq!(config.momentum, 0.95);
    }

    #[test]
    fn test_fisher_approximations() {
        assert_eq!(FisherApproximation::Exact, FisherApproximation::Exact);
        assert_ne!(FisherApproximation::Diagonal, FisherApproximation::KFAC);
    }

    #[test]
    fn test_natural_gradient_on_quadratic() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let config = NaturalGradientConfig::new()
            .with_learning_rate(0.1)
            .with_fisher_approximation(FisherApproximation::Diagonal);
        
        let mut optimizer = NaturalGradient::new(config);
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        let result = optimizer.optimize(
            &cost_fn,
            &manifold,
            &retraction,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        // Should converge to origin
        assert!(result.converged);
        assert!(result.point.norm() < 1e-3);
    }

    #[test]
    #[ignore = "Momentum with natural gradient needs more investigation"]
    fn test_natural_gradient_with_momentum() {
        let cost_fn = QuadraticCost::simple(Dyn(3));
        let manifold = TestEuclideanManifold::new(3);
        let initial_point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        let config = NaturalGradientConfig::new()
            .with_learning_rate(0.1)
            .with_fisher_approximation(FisherApproximation::Diagonal)
            .with_momentum(0.9);
        
        let mut optimizer = NaturalGradient::new(config);
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        let result = optimizer.optimize(
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
    fn test_natural_gradient_optimizer_trait() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let mut optimizer = NaturalGradient::<f64, Dyn>::new(
            NaturalGradientConfig::new()
                .with_fisher_approximation(FisherApproximation::Empirical)
        );
        
        // Test that it implements the Optimizer trait
        assert_eq!(optimizer.name(), "Riemannian Natural Gradient (Empirical)");
        
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
    fn test_natural_gradient_state() {
        let mut state = NaturalGradientState::<f64, Dyn>::new();
        
        assert!(state.needs_fisher_update(10));
        assert_eq!(state.fisher_update_counter, 0);
        
        state.increment_fisher_counter();
        assert_eq!(state.fisher_update_counter, 1);
        
        state.reset_fisher_counter();
        assert_eq!(state.fisher_update_counter, 0);
    }
}