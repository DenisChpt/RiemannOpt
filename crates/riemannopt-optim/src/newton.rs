//! Riemannian Newton method
//!
//! The Newton method uses second-order information (Hessian) to achieve
//! faster convergence than first-order methods.

use nalgebra::{Dyn, OVector, OMatrix, allocator::Allocator, DefaultAllocator, Dim};
use num_traits::Float;

use riemannopt_core::{
    cost_function::CostFunction,
    core::CachedCostFunction,
    error::{Result, ManifoldError},
    line_search::{LineSearchParams, BacktrackingLineSearch, LineSearch},
    manifold::{Manifold, Point, TangentVector},
    memory::workspace::{Workspace, WorkspaceBuilder},
    optimizer::{Optimizer, OptimizationResult, StoppingCriterion, OptimizerStateLegacy as OptimizerState, ConvergenceChecker, TerminationReason},
    retraction::DefaultRetraction,
    types::Scalar,
};

/// Configuration for the Riemannian Newton method
#[derive(Debug, Clone)]
pub struct NewtonConfig<T: Scalar> {
    /// Line search parameters
    pub line_search_params: LineSearchParams<T>,
    /// Regularization parameter for Hessian (to ensure positive definiteness)
    pub hessian_regularization: T,
    /// Use Gauss-Newton approximation (for least-squares problems)
    pub use_gauss_newton: bool,
    /// Maximum number of CG iterations for solving Newton system
    pub max_cg_iterations: usize,
    /// Tolerance for CG solver
    pub cg_tolerance: T,
}

impl<T: Scalar> Default for NewtonConfig<T> {
    fn default() -> Self {
        Self {
            line_search_params: LineSearchParams::default(),
            hessian_regularization: <T as Scalar>::from_f64(1e-8),
            use_gauss_newton: false,
            max_cg_iterations: 100,
            cg_tolerance: <T as Scalar>::from_f64(1e-6),
        }
    }
}

impl<T: Scalar> NewtonConfig<T> {
    /// Create a new Newton configuration with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Hessian regularization parameter
    pub fn with_regularization(mut self, reg: T) -> Self {
        self.hessian_regularization = reg;
        self
    }

    /// Enable Gauss-Newton approximation
    pub fn with_gauss_newton(mut self) -> Self {
        self.use_gauss_newton = true;
        self
    }

    /// Set CG solver parameters
    pub fn with_cg_params(mut self, max_iter: usize, tol: T) -> Self {
        self.max_cg_iterations = max_iter;
        self.cg_tolerance = tol;
        self
    }
}

/// State for the Newton optimizer.
#[derive(Debug)]
struct NewtonState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// Line search instance (reused across iterations)
    line_search: BacktrackingLineSearch,
    /// Workspace for CG solver
    cg_workspace: CGWorkspace<T, D>,
    /// Cached Hessian (if using exact Hessian)
    #[allow(dead_code)]
    cached_hessian: Option<OMatrix<T, D, D>>,
    /// Iteration counter
    iteration: usize,
}

impl<T, D> NewtonState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// Creates a new Newton state.
    fn new(dim: D, max_cg_iterations: usize) -> Self {
        Self {
            line_search: BacktrackingLineSearch::new(),
            cg_workspace: CGWorkspace::new(dim, max_cg_iterations),
            cached_hessian: None,
            iteration: 0,
        }
    }
}

/// Workspace for CG solver to avoid allocations
#[derive(Debug)]
struct CGWorkspace<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Direction vector
    d: OVector<T, D>,
    /// Residual vector
    r: OVector<T, D>,
    /// Search direction
    p: OVector<T, D>,
    /// Hessian-vector product
    hp: OVector<T, D>,
}

impl<T, D> CGWorkspace<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn new(dim: D, _max_iterations: usize) -> Self {
        Self {
            d: OVector::zeros_generic(dim, nalgebra::U1),
            r: OVector::zeros_generic(dim, nalgebra::U1),
            p: OVector::zeros_generic(dim, nalgebra::U1),
            hp: OVector::zeros_generic(dim, nalgebra::U1),
        }
    }
}

/// Riemannian Newton method optimizer
#[derive(Debug)]
pub struct Newton<T: Scalar> {
    config: NewtonConfig<T>,
}

impl<T: Scalar> Newton<T> {
    /// Create a new Newton optimizer with the given configuration
    pub fn new(config: NewtonConfig<T>) -> Self {
        Self {
            config,
        }
    }
    
    /// Internal step method that has access to Newton-specific state
    fn step_internal<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, Dyn>,
        newton_state: &mut NewtonState<T, Dyn>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        C: CostFunction<T, Dyn>,
        M: Manifold<T, Dyn>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Dyn, Dyn>,
    {
        // Get current gradient
        let grad = state.gradient.as_ref()
            .ok_or_else(|| ManifoldError::numerical_error("Gradient not computed"))?;
        
        // Get Hessian or Hessian-vector product
        let hessian_vec_prod = |v: &TangentVector<T, Dyn>| -> Result<TangentVector<T, Dyn>> {
            if self.config.use_gauss_newton {
                // Gauss-Newton approximation: H = J^T J
                // This requires the cost function to provide Jacobian
                return Err(ManifoldError::not_implemented(
                    "Gauss-Newton approximation not yet implemented"
                ));
            }
            
            // Full Hessian computation
            match cost_fn.hessian(&state.point) {
                Ok(hess) => {
                    // Apply Hessian to vector
                    Ok(&hess * v)
                },
                Err(_) => {
                    // Approximate Hessian-vector product using finite differences
                    let eps = <T as Scalar>::from_f64(1e-8);
                    let x_plus = manifold.retract(&state.point, &(v * eps))?;
                    let (_, grad_plus) = cost_fn.cost_and_gradient(&x_plus)?;
                    let grad_plus_riem = manifold.euclidean_to_riemannian_gradient(&x_plus, &grad_plus)?;
                    
                    // Parallel transport grad_plus back to x
                    let grad_plus_transported = manifold.parallel_transport(&x_plus, &state.point, &grad_plus_riem)?;
                    
                    Ok((grad_plus_transported - grad) / eps)
                }
            }
        };
        
        // Solve Newton system using workspace
        let newton_dir = self.solve_newton_system(
            manifold, 
            &state.point, 
            grad, 
            hessian_vec_prod,
            &mut newton_state.cg_workspace
        )?;
        
        // Compute directional derivative for efficient line search
        let directional_deriv = manifold.inner_product(&state.point, grad, &newton_dir)?;
        
        // Perform line search using stored instance with pre-computed values
        let retraction = DefaultRetraction;
        let ls_result = newton_state.line_search.search_with_deriv(
            cost_fn,
            manifold,
            &retraction,
            &state.point,
            state.value,
            &newton_dir,
            directional_deriv,
            &self.config.line_search_params,
        )?;
        
        // Update point
        let new_point = manifold.retract(&state.point, &(&newton_dir * ls_result.step_size))?;
        
        // Update gradient and cost
        let (new_cost, new_grad) = cost_fn.cost_and_gradient(&new_point)?;
        let new_riem_grad = manifold.euclidean_to_riemannian_gradient(&new_point, &new_grad)?;
        let new_grad_norm = <T as Float>::sqrt(manifold.inner_product(&new_point, &new_riem_grad, &new_riem_grad)?);
        
        // Update state
        state.update(new_point, new_cost);
        state.set_gradient(new_riem_grad, new_grad_norm);
        
        // Update Newton state
        newton_state.iteration += 1;
        
        Ok(())
    }

    /// Solve the Newton system H*d = -g using CG with workspace
    fn solve_newton_system<M: Manifold<T, Dyn>>(
        &self,
        manifold: &M,
        x: &Point<T, Dyn>,
        grad: &TangentVector<T, Dyn>,
        hessian_vector_product: impl Fn(&TangentVector<T, Dyn>) -> Result<TangentVector<T, Dyn>>,
        workspace: &mut CGWorkspace<T, Dyn>,
    ) -> Result<TangentVector<T, Dyn>> {
        // Use workspace vectors to avoid allocations
        workspace.d.fill(T::zero());
        workspace.r.copy_from(&(-grad));
        workspace.p.copy_from(&workspace.r);
        
        for _ in 0..self.config.max_cg_iterations {
            workspace.hp = hessian_vector_product(&workspace.p)?;
            
            // Add regularization: Hp + reg*p
            workspace.hp.axpy(self.config.hessian_regularization, &workspace.p, T::one());
            
            let rr_inner = manifold.inner_product(x, &workspace.r, &workspace.r)?;
            let php_inner = manifold.inner_product(x, &workspace.p, &workspace.hp)?;
            let alpha = rr_inner / php_inner;
            
            workspace.d.axpy(alpha, &workspace.p, T::one());
            
            // Store old r'r before updating r
            let r_old_norm_sq = rr_inner;
            workspace.r.axpy(-alpha, &workspace.hp, T::one());
            
            let r_new_norm_sq = manifold.inner_product(x, &workspace.r, &workspace.r)?;
            let r_new_norm = <T as Float>::sqrt(r_new_norm_sq);
            if r_new_norm < self.config.cg_tolerance {
                break;
            }
            
            let beta = r_new_norm_sq / r_old_norm_sq;
            
            // p = r + beta * p_old
            // We need to store p_old before updating
            workspace.p *= beta;
            workspace.p += &workspace.r;
        }
        
        Ok(workspace.d.clone())
    }
}

impl<T: Scalar> Optimizer<T, Dyn> for Newton<T>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Dyn> + nalgebra::allocator::Allocator<Dyn, Dyn>,
{
    fn name(&self) -> &str {
        "Riemannian Newton"
    }

    fn optimize<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &Point<T, Dyn>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, Dyn>>
    where
        C: CostFunction<T, Dyn>,
        M: Manifold<T, Dyn>,
    {
        // Wrap cost function with caching to avoid redundant computations
        let cached_cost_fn = CachedCostFunction::new(cost_fn);
        
        // Initialize state
        let (initial_cost, initial_grad) = cached_cost_fn.cost_and_gradient(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        
        let riem_grad = manifold.euclidean_to_riemannian_gradient(initial_point, &initial_grad)?;
        let grad_norm = <T as Float>::sqrt(manifold.inner_product(initial_point, &riem_grad, &riem_grad)?);
        state.set_gradient(riem_grad, grad_norm);
        
        // Initialize Newton-specific state
        let dim = initial_point.shape_generic().0;
        let max_cg_iter = self.config.max_cg_iterations;
        let mut newton_state = NewtonState::new(dim, max_cg_iter);
        
        // Create a single workspace for the entire optimization
        let n = initial_point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                let duration = state.start_time.elapsed();
                
                // Get cache statistics for diagnostics
                let ((_cost_hits, cost_misses), (_grad_hits, grad_misses), (_hess_hits, _hess_misses)) = cached_cost_fn.cache_stats();
                
                return Ok(OptimizationResult {
                    point: state.point.clone(),
                    value: state.value,
                    gradient_norm: state.gradient_norm,
                    iterations: state.iteration,
                    function_evaluations: cost_misses,  // Use cache misses as actual evaluations
                    gradient_evaluations: grad_misses,  // Use cache misses as actual evaluations
                    duration,
                    termination_reason: reason,
                    converged: matches!(reason, TerminationReason::Converged | TerminationReason::TargetReached),
                });
            }
            
            // Take optimization step
            self.step_internal(&cached_cost_fn, manifold, &mut state, &mut newton_state, &mut workspace)?;
        }
    }

    fn step<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, Dyn>,
    ) -> Result<()>
    where
        C: CostFunction<T, Dyn>,
        M: Manifold<T, Dyn>,
    {
        // Create temporary Newton state - this is a limitation of the public step interface
        let dim = state.point.shape_generic().0;
        let max_cg_iter = self.config.max_cg_iterations;
        let mut newton_state = NewtonState::new(dim, max_cg_iter);
        
        // Create temporary workspace
        let n = state.point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        
        self.step_internal(cost_fn, manifold, state, &mut newton_state, &mut workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::cost_function::QuadraticCost;
    use riemannopt_core::test_manifolds::TestEuclideanManifold;
    use nalgebra::DVector;

    #[test]
    fn test_newton_creation() {
        let config = NewtonConfig::<f64>::new();
        let _optimizer = Newton::new(config);
    }

    #[test]
    fn test_newton_on_simple_problem() {
        let config = NewtonConfig::new()
            .with_regularization(1e-6);
        let mut optimizer = Newton::new(config);
        
        let manifold = TestEuclideanManifold::new(2);
        let cost_fn = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        
        let criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        let result = optimizer.optimize(&cost_fn, &manifold, &x0, &criterion).unwrap();
        
        assert!(result.iterations < 10); // Newton should converge fast
        assert!(result.gradient_norm.unwrap_or(1.0) < 1e-6);
    }
}