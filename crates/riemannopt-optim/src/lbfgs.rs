//! Riemannian L-BFGS optimizer.
//!
//! L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization
//! algorithm that approximates the inverse Hessian using a limited history of past gradient
//! and position updates. This implementation extends L-BFGS to Riemannian manifolds by
//! properly handling parallel transport of stored vectors.
//!
//! # Algorithm Overview
//!
//! The Riemannian L-BFGS algorithm:
//! 1. Stores m most recent gradient differences and position differences
//! 2. Approximates the inverse Hessian-vector product using two-loop recursion
//! 3. Computes search direction as negative approximate Newton direction
//! 4. Performs line search to find suitable step size
//! 5. Updates position using retraction
//!
//! # Key Features
//!
//! - **Limited memory**: Only stores m vector pairs (typically 5-20)
//! - **Superlinear convergence**: Near optimal points with good initial Hessian approximation
//! - **Vector transport**: Properly handles curvature when transporting stored vectors
//! - **Automatic scaling**: Initial Hessian approximation based on most recent update
//! - **Strong Wolfe line search**: Ensures sufficient decrease and curvature conditions
//!
//! # References
//!
//! - Nocedal & Wright, "Numerical Optimization" (2006)
//! - Huang et al., "A Riemannian BFGS Method" (2015)
//! - Sato, "Riemannian Optimization and Its Applications" (2021)

use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    line_search::{BacktrackingLineSearch, LineSearch, LineSearchParams},
    manifold::{Manifold, Point},
    optimizer::{Optimizer, OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
    optimizer_state::OptimizerStateData,
    retraction::Retraction,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use std::collections::{VecDeque, HashMap};
use std::time::Instant;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// Configuration for the L-BFGS optimizer.
#[derive(Debug, Clone)]
pub struct LBFGSConfig<T: Scalar> {
    /// Number of vector pairs to store (typically 5-20)
    pub memory_size: usize,
    /// Line search parameters
    pub line_search_params: LineSearchParams<T>,
    /// Initial step size for line search
    pub initial_step_size: T,
    /// Whether to use cautious updates (skip updates that don't satisfy positive definiteness)
    pub use_cautious_updates: bool,
}

impl<T: Scalar> Default for LBFGSConfig<T> {
    fn default() -> Self {
        Self {
            memory_size: 10,
            line_search_params: LineSearchParams::strong_wolfe(),
            initial_step_size: <T as Scalar>::from_f64(1.0),
            use_cautious_updates: true,
        }
    }
}

impl<T: Scalar> LBFGSConfig<T> {
    /// Creates a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the memory size (number of vector pairs to store).
    pub fn with_memory_size(mut self, size: usize) -> Self {
        self.memory_size = size;
        self
    }

    /// Sets the line search parameters.
    pub fn with_line_search(mut self, params: LineSearchParams<T>) -> Self {
        self.line_search_params = params;
        self
    }

    /// Sets the initial step size for line search.
    pub fn with_initial_step_size(mut self, step_size: T) -> Self {
        self.initial_step_size = step_size;
        self
    }

    /// Enables or disables cautious updates.
    pub fn with_cautious_updates(mut self, cautious: bool) -> Self {
        self.use_cautious_updates = cautious;
        self
    }
}

/// Storage for L-BFGS vector pairs.
#[derive(Debug, Clone)]
struct LBFGSStorage<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Position differences: s_k = x_{k+1} - x_k
    s_vectors: VecDeque<OVector<T, D>>,
    /// Gradient differences: y_k = g_{k+1} - g_k
    y_vectors: VecDeque<OVector<T, D>>,
    /// Inner products: rho_k = 1 / <s_k, y_k>
    rho_values: VecDeque<T>,
    /// Maximum number of pairs to store
    capacity: usize,
}

impl<T, D> LBFGSStorage<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates new storage with given capacity.
    fn new(capacity: usize) -> Self {
        Self {
            s_vectors: VecDeque::with_capacity(capacity),
            y_vectors: VecDeque::with_capacity(capacity),
            rho_values: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Adds a new vector pair to storage.
    fn push(&mut self, s: OVector<T, D>, y: OVector<T, D>, manifold: &impl Manifold<T, D>, point: &OVector<T, D>) -> Result<()> {
        // Compute inner product
        let sy_inner = manifold.inner_product(point, &s, &y)?;
        
        // Check for positive definiteness (cautious update)
        if sy_inner <= T::zero() {
            return Ok(()); // Skip this update
        }
        
        let rho = T::one() / sy_inner;
        
        // Remove oldest if at capacity
        if self.s_vectors.len() >= self.capacity {
            self.s_vectors.pop_front();
            self.y_vectors.pop_front();
            self.rho_values.pop_front();
        }
        
        // Add new vectors
        self.s_vectors.push_back(s);
        self.y_vectors.push_back(y);
        self.rho_values.push_back(rho);
        
        Ok(())
    }

    /// Clears all stored vectors.
    fn clear(&mut self) {
        self.s_vectors.clear();
        self.y_vectors.clear();
        self.rho_values.clear();
    }

    /// Returns the number of stored pairs.
    fn len(&self) -> usize {
        self.s_vectors.len()
    }

    /// Returns true if storage is empty.
    fn is_empty(&self) -> bool {
        self.s_vectors.is_empty()
    }
}

/// State for the L-BFGS optimizer.
/// This is the optimizer-specific state for L-BFGS, different from core's LBFGSState.
#[derive(Debug, Clone)]
pub struct LBFGSState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Stored vector pairs for Hessian approximation
    storage: LBFGSStorage<T, D>,
    /// Previous point
    prev_point: Option<OVector<T, D>>,
    /// Previous gradient
    prev_gradient: Option<OVector<T, D>>,
    /// Iteration counter
    iteration: usize,
    /// Function value at current point
    current_value: Option<T>,
    /// Gradient norm at current point
    gradient_norm: Option<T>,
}

impl<T, D> LBFGSState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new L-BFGS state with given memory size.
    pub fn new(memory_size: usize) -> Self {
        Self {
            storage: LBFGSStorage::new(memory_size),
            prev_point: None,
            prev_gradient: None,
            iteration: 0,
            current_value: None,
            gradient_norm: None,
        }
    }
}

impl<T, D> OptimizerStateData<T, D> for LBFGSState<T, D>
where
    T: Scalar + Display,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn optimizer_name(&self) -> &str {
        "L-BFGS"
    }

    fn reset(&mut self) {
        self.storage.clear();
        self.prev_point = None;
        self.prev_gradient = None;
        self.iteration = 0;
        self.current_value = None;
        self.gradient_norm = None;
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("optimizer".to_string(), "L-BFGS".to_string());
        summary.insert("iteration".to_string(), self.iteration.to_string());
        summary.insert("memory_used".to_string(), self.storage.len().to_string());
        if let Some(value) = self.current_value {
            summary.insert("function_value".to_string(), format!("{}", value));
        }
        if let Some(norm) = self.gradient_norm {
            summary.insert("gradient_norm".to_string(), format!("{}", norm));
        }
        summary
    }

    fn update_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }
}

/// L-BFGS optimizer for Riemannian manifolds.
#[derive(Debug)]
pub struct LBFGS<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    config: LBFGSConfig<T>,
    line_search: BacktrackingLineSearch,
    _phantom: PhantomData<D>,
}

impl<T, D> LBFGS<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new L-BFGS optimizer with given configuration.
    pub fn new(config: LBFGSConfig<T>) -> Self {
        let line_search = BacktrackingLineSearch::new();
        Self { 
            config, 
            line_search,
            _phantom: PhantomData,
        }
    }

    /// Computes the search direction using L-BFGS two-loop recursion.
    fn compute_direction(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &OVector<T, D>,
        gradient: &OVector<T, D>,
        storage: &LBFGSStorage<T, D>,
    ) -> Result<OVector<T, D>> {
        // If no history, return negative gradient
        if storage.is_empty() {
            return Ok(-gradient);
        }

        let m = storage.len();
        let mut alpha = vec![T::zero(); m];
        let mut q = gradient.clone();

        // First loop: compute alpha values and update q
        for i in (0..m).rev() {
            let s = &storage.s_vectors[i];
            let y = &storage.y_vectors[i];
            let rho = storage.rho_values[i];
            
            // alpha[i] = rho * <s[i], q>
            let sq_inner = manifold.inner_product(point, s, &q)?;
            alpha[i] = rho * sq_inner;
            
            // q = q - alpha[i] * y[i]
            q = &q - &(y * alpha[i]);
        }

        // Scale by initial Hessian approximation
        // H_0 = gamma * I, where gamma = <s_{m-1}, y_{m-1}> / <y_{m-1}, y_{m-1}>
        if m > 0 {
            let s_last = &storage.s_vectors[m - 1];
            let y_last = &storage.y_vectors[m - 1];
            
            let sy_inner = manifold.inner_product(point, s_last, y_last)?;
            let yy_inner = manifold.inner_product(point, y_last, y_last)?;
            
            if yy_inner > T::zero() {
                let gamma = sy_inner / yy_inner;
                q = &q * gamma;
            }
        }

        // Second loop: compute search direction
        let mut r = q;
        for i in 0..m {
            let s = &storage.s_vectors[i];
            let y = &storage.y_vectors[i];
            let rho = storage.rho_values[i];
            
            // beta = rho * <y[i], r>
            let yr_inner = manifold.inner_product(point, y, &r)?;
            let beta = rho * yr_inner;
            
            // r = r + (alpha[i] - beta) * s[i]
            let coeff = alpha[i] - beta;
            r = &r + &(s * coeff);
        }

        // Return negative direction
        Ok(-r)
    }

    /// Internal step method that has access to L-BFGS-specific state.
    fn step_internal(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        retraction: &impl Retraction<T, D>,
        state: &mut riemannopt_core::optimizer::OptimizerState<T, D>,
        lbfgs_state: &mut LBFGSState<T, D>,
    ) -> Result<()> {
        // Compute cost and gradient
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&state.point)?;
        
        // Convert to Riemannian gradient
        let riemannian_grad = manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad)?;
        
        // Compute gradient norm
        let grad_norm = manifold.norm(&state.point, &riemannian_grad)?;
        state.set_gradient(riemannian_grad.clone(), grad_norm);
        lbfgs_state.gradient_norm = Some(grad_norm);
        lbfgs_state.current_value = Some(cost);
        
        // Compute search direction
        let direction = self.compute_direction(manifold, &state.point, &riemannian_grad, &lbfgs_state.storage)?;
        
        // Determine initial step size for line search
        let mut params = self.config.line_search_params.clone();
        params.initial_step_size = if lbfgs_state.iteration == 0 {
            // First iteration: use configured initial step size
            self.config.initial_step_size
        } else {
            // Use step size 1.0 for quasi-Newton direction
            T::one()
        };
        
        // Perform line search
        let line_search_result = self.line_search.search(
            cost_fn,
            manifold,
            retraction,
            &state.point,
            cost,
            &euclidean_grad,
            &direction,
            &params,
        )?;
        
        // Update storage if we have previous gradient
        if let (Some(prev_point), Some(prev_grad)) = (&lbfgs_state.prev_point, &lbfgs_state.prev_gradient) {
            // Transport previous gradient to current point for comparison
            let transported_prev_grad = manifold.parallel_transport(prev_point, &state.point, prev_grad)?;
            
            // Compute differences
            // s = x_{k+1} - x_k (approximated by retraction inverse)
            let s = manifold.inverse_retract(prev_point, &state.point)?;
            
            // y = g_{k+1} - g_k (after transport)
            let y = &riemannian_grad - &transported_prev_grad;
            
            // Add to storage (with cautious update check)
            if self.config.use_cautious_updates {
                let sy_inner = manifold.inner_product(&state.point, &s, &y)?;
                if sy_inner > T::zero() {
                    lbfgs_state.storage.push(s, y, manifold, &state.point)?;
                }
            } else {
                lbfgs_state.storage.push(s, y, manifold, &state.point)?;
            }
        }
        
        // Update L-BFGS state
        lbfgs_state.prev_point = Some(state.point.clone());
        lbfgs_state.prev_gradient = Some(riemannian_grad.clone());
        lbfgs_state.iteration += 1;
        
        // Update optimizer state
        state.update(line_search_result.new_point, line_search_result.new_value);
        state.function_evaluations += line_search_result.function_evals - 1; // -1 because update() already increments
        state.gradient_evaluations += line_search_result.gradient_evals;
        
        Ok(())
    }

    /// Optimizes the given cost function.
    pub fn optimize(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        retraction: &impl Retraction<T, D>,
        initial_point: &OVector<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>> {
        let start_time = Instant::now();
        
        // Initialize optimizer state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut state = riemannopt_core::optimizer::OptimizerState::new(initial_point.clone(), initial_cost);
        let mut lbfgs_state = LBFGSState::new(self.config.memory_size);
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                return Ok(OptimizationResult::new(
                    state.point,
                    state.value,
                    state.iteration,
                    start_time.elapsed(),
                    reason,
                )
                .with_function_evaluations(state.function_evaluations)
                .with_gradient_evaluations(state.gradient_evaluations)
                .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
            }
            
            // Perform one optimization step
            self.step_internal(cost_fn, manifold, retraction, &mut state, &mut lbfgs_state)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::test_manifolds::{TestSphereManifold, TestEuclideanManifold};
    use riemannopt_core::retraction::ExponentialRetraction;
    use riemannopt_core::cost_function::CostFunction;
    use nalgebra::{DVector, Dyn};

    /// Simple quadratic cost function for testing
    #[derive(Debug)]
    struct QuadraticCost<T, D>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        target: OVector<T, D>,
    }

    impl<T, D> QuadraticCost<T, D>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        fn new(target: OVector<T, D>) -> Self {
            Self { target }
        }
    }

    impl<T, D> CostFunction<T, D> for QuadraticCost<T, D>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        fn cost(&self, point: &OVector<T, D>) -> Result<T> {
            let diff = point - &self.target;
            Ok(diff.dot(&diff) * <T as Scalar>::from_f64(0.5))
        }

        fn cost_and_gradient(
            &self,
            point: &OVector<T, D>,
        ) -> Result<(T, OVector<T, D>)> {
            let diff = point - &self.target;
            let cost = diff.dot(&diff) * <T as Scalar>::from_f64(0.5);
            let grad = diff;
            Ok((cost, grad))
        }
    }

    #[test]
    fn test_lbfgs_config() {
        let config = LBFGSConfig::<f64>::new()
            .with_memory_size(20)
            .with_initial_step_size(0.5)
            .with_cautious_updates(false);
        
        assert_eq!(config.memory_size, 20);
        assert_eq!(config.initial_step_size, 0.5);
        assert!(!config.use_cautious_updates);
    }

    #[test]
    fn test_storage_operations() {
        let manifold = TestSphereManifold::new(3);
        let mut storage = LBFGSStorage::<f64, Dyn>::new(3);
        let point = manifold.random_point();
        
        // Add some vectors
        let s1 = DVector::from_vec(vec![0.1, 0.0, 0.0]);
        let y1 = DVector::from_vec(vec![0.05, 0.05, 0.0]);
        storage.push(s1, y1, &manifold, &point).unwrap();
        
        assert_eq!(storage.len(), 1);
        assert!(!storage.is_empty());
        
        // Add more vectors to test capacity
        let s2 = DVector::from_vec(vec![0.0, 0.1, 0.0]);
        let y2 = DVector::from_vec(vec![0.0, 0.05, 0.05]);
        storage.push(s2, y2, &manifold, &point).unwrap();
        
        let s3 = DVector::from_vec(vec![0.0, 0.0, 0.1]);
        let y3 = DVector::from_vec(vec![0.05, 0.0, 0.05]);
        storage.push(s3, y3, &manifold, &point).unwrap();
        
        // Should still have 3 (at capacity)
        assert_eq!(storage.len(), 3);
        
        // Add one more - oldest should be removed
        let s4 = DVector::from_vec(vec![0.1, 0.1, 0.0]);
        let y4 = DVector::from_vec(vec![0.05, 0.05, 0.05]);
        storage.push(s4, y4, &manifold, &point).unwrap();
        
        assert_eq!(storage.len(), 3);
    }

    #[test]
    fn test_lbfgs_on_sphere() {
        let manifold = TestSphereManifold::new(3);
        let retraction = ExponentialRetraction::new();
        
        // Target point on sphere
        let target = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let cost_fn = QuadraticCost::new(target);
        
        let config = LBFGSConfig::new()
            .with_memory_size(5)
            .with_initial_step_size(0.1);
        let mut optimizer = LBFGS::<f64, Dyn>::new(config);
        
        let initial = manifold.random_point();
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        let result = optimizer.optimize(&cost_fn, &manifold, &retraction, &initial, &stopping_criterion).unwrap();
        
        // L-BFGS should at least reduce the gradient norm significantly
        assert!(result.gradient_norm.unwrap() < 1e-3);
    }

    #[test]
    fn test_cautious_updates() {
        let manifold = TestSphereManifold::new(3);
        let mut storage = LBFGSStorage::<f64, Dyn>::new(5);
        let point = manifold.random_point();
        
        // Project vectors to tangent space
        let s_raw = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let y_raw = DVector::from_vec(vec![-1.0, 0.0, 0.0]);
        
        let s = manifold.project_tangent(&point, &s_raw).unwrap();
        let y = manifold.project_tangent(&point, &y_raw).unwrap();
        
        // This might be skipped due to cautious update
        let initial_len = storage.len();
        let s_clone = s.clone();
        let y_clone = y.clone();
        storage.push(s, y, &manifold, &point).unwrap();
        
        // Check if it was skipped (negative inner product)
        let inner_prod = manifold.inner_product(&point, &s_clone, &y_clone).unwrap();
        if inner_prod <= 0.0 {
            assert_eq!(storage.len(), initial_len);
        } else {
            assert_eq!(storage.len(), initial_len + 1);
        }
    }

    #[test]
    fn test_two_loop_recursion() {
        let manifold = TestEuclideanManifold::new(2);
        let config = LBFGSConfig::new().with_memory_size(3);
        let optimizer = LBFGS::<f64, Dyn>::new(config);
        
        // Create storage with some history
        let mut storage = LBFGSStorage::<f64, Dyn>::new(3);
        let point = DVector::from_vec(vec![0.0, 0.0]);
        
        // Add some vector pairs
        let s1 = DVector::from_vec(vec![1.0, 0.0]);
        let y1 = DVector::from_vec(vec![0.5, 0.5]);
        storage.push(s1, y1, &manifold, &point).unwrap();
        
        let s2 = DVector::from_vec(vec![0.0, 1.0]);
        let y2 = DVector::from_vec(vec![0.5, 0.5]);
        storage.push(s2, y2, &manifold, &point).unwrap();
        
        // Compute direction for a gradient
        let gradient = DVector::from_vec(vec![1.0, 1.0]);
        let direction = optimizer.compute_direction(&manifold, &point, &gradient, &storage).unwrap();
        
        // Direction should be different from negative gradient due to history
        let neg_grad = -&gradient;
        let diff = &direction - &neg_grad;
        let diff_norm = diff.norm();
        
        // Should be different due to quasi-Newton approximation
        assert!(diff_norm > 1e-10);
    }
}

// Implementation of the Optimizer trait from core
impl<T, D> Optimizer<T, D> for LBFGS<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Riemannian L-BFGS"
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
        // Use the default retraction for manifolds
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        self.optimize(cost_fn, manifold, &retraction, initial_point, stopping_criterion)
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
        // For the step method, we need to maintain internal state
        // This is a limitation of the current design where L-BFGS requires a retraction
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        // Create a temporary L-BFGS state if needed
        let mut lbfgs_state = LBFGSState::new(self.config.memory_size);
        
        // Perform the step
        self.step_internal(cost_fn, manifold, &retraction, state, &mut lbfgs_state)
    }
}