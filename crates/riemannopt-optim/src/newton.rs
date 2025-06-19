//! Riemannian Newton method
//!
//! The Newton method uses second-order information (Hessian) to achieve
//! faster convergence than first-order methods.

use nalgebra::{DVector, Dyn};
use num_traits::Float;

use riemannopt_core::{
    cost_function::CostFunction,
    error::{Result, ManifoldError},
    line_search::{LineSearchParams, BacktrackingLineSearch, LineSearch},
    manifold::{Manifold, Point, TangentVector},
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

    /// Solve the Newton system H*d = -g using CG
    fn solve_newton_system<M: Manifold<T, Dyn>>(
        &self,
        manifold: &M,
        x: &Point<T, Dyn>,
        grad: &TangentVector<T, Dyn>,
        hessian_vector_product: impl Fn(&TangentVector<T, Dyn>) -> Result<TangentVector<T, Dyn>>,
    ) -> Result<TangentVector<T, Dyn>> {
        // Implement CG solver for the Newton system
        let mut d = DVector::zeros(grad.len());
        let mut r = -grad.clone();
        let mut p = r.clone();
        
        for _ in 0..self.config.max_cg_iterations {
            let hp = hessian_vector_product(&p)?;
            
            // Add regularization: Hp + reg*p
            let hp_reg = &hp + &p * self.config.hessian_regularization;
            
            let alpha = manifold.inner_product(x, &r, &r)? / 
                       manifold.inner_product(x, &p, &hp_reg)?;
            
            d = &d + &p * alpha;
            let r_new = &r - &hp_reg * alpha;
            
            let r_new_norm = <T as Float>::sqrt(manifold.inner_product(x, &r_new, &r_new)?);
            if r_new_norm < self.config.cg_tolerance {
                break;
            }
            
            let beta = manifold.inner_product(x, &r_new, &r_new)? / 
                      manifold.inner_product(x, &r, &r)?;
            
            p = &r_new + &p * beta;
            r = r_new;
        }
        
        Ok(d)
    }
}

impl<T: Scalar> Optimizer<T, Dyn> for Newton<T> {
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
        // Initialize state
        let (initial_cost, initial_grad) = cost_fn.cost_and_gradient(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        
        let riem_grad = manifold.euclidean_to_riemannian_gradient(initial_point, &initial_grad)?;
        let grad_norm = <T as Float>::sqrt(manifold.inner_product(initial_point, &riem_grad, &riem_grad)?);
        state.set_gradient(riem_grad, grad_norm);
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                let duration = state.start_time.elapsed();
                return Ok(OptimizationResult {
                    point: state.point.clone(),
                    value: state.value,
                    gradient_norm: state.gradient_norm,
                    iterations: state.iteration,
                    function_evaluations: state.function_evaluations,
                    gradient_evaluations: state.gradient_evaluations,
                    duration,
                    termination_reason: reason,
                    converged: matches!(reason, TerminationReason::Converged | TerminationReason::TargetReached),
                });
            }
            
            // Take optimization step
            self.step(cost_fn, manifold, &mut state)?;
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
        
        // Solve Newton system
        let newton_dir = self.solve_newton_system(manifold, &state.point, grad, hessian_vec_prod)?;
        
        // Perform line search
        let mut line_search = BacktrackingLineSearch::new();
        let retraction = DefaultRetraction;
        let ls_result = line_search.search(
            cost_fn,
            manifold,
            &retraction,
            &state.point,
            state.value,
            grad,
            &newton_dir,
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
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::cost_function::QuadraticCost;
    use riemannopt_core::test_manifolds::TestEuclideanManifold;

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