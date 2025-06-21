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
    core::CachedCostFunction,
    error::Result,
    line_search::{BacktrackingLineSearch, LineSearch, LineSearchParams},
    manifold::{Manifold, Point},
    optimizer::{Optimizer, OptimizerStateLegacy as OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
    retraction::{Retraction, DefaultRetraction},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use std::collections::VecDeque;
use std::time::Instant;
use std::fmt::Debug;
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
    /// Position differences: s_k = log_{x_k}(x_{k+1}) (using inverse retraction)
    s_vectors: VecDeque<OVector<T, D>>,
    /// Gradient differences: y_k = g_{k+1} - P_{x_k→x_{k+1}}(g_k) (with parallel transport)
    y_vectors: VecDeque<OVector<T, D>>,
    /// Inner products: rho_k = 1 / <s_k, y_k>_g (using Riemannian metric)
    rho_values: VecDeque<T>,
    /// Points where each (s_k, y_k) pair was computed
    /// This allows us to transport gradients backward instead of transporting history forward
    points: VecDeque<OVector<T, D>>,
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
            points: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Adds a new vector pair to storage.
    /// Vectors are stored at their original points without transport.
    fn push(
        &mut self, 
        s: OVector<T, D>, 
        y: OVector<T, D>, 
        manifold: &impl Manifold<T, D>, 
        point: &OVector<T, D>,
    ) -> Result<()> {
        // Compute inner product at the current point
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
            self.points.pop_front();
        }
        
        // Add new vectors and point (no transport needed)
        self.s_vectors.push_back(s);
        self.y_vectors.push_back(y);
        self.rho_values.push_back(rho);
        self.points.push_back(point.clone());
        
        Ok(())
    }

    /// Clears all stored vectors.
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.s_vectors.clear();
        self.y_vectors.clear();
        self.rho_values.clear();
        self.points.clear();
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
    /// Uses backward transport of gradient to avoid transporting history.
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

        // First loop: backward pass through history
        // Transport q backward to each historical point for inner products
        for i in (0..m).rev() {
            let hist_point = &storage.points[i];
            let s = &storage.s_vectors[i];
            let y = &storage.y_vectors[i];
            let rho = storage.rho_values[i];
            
            // Transport q from current point to historical point
            let q_transported = manifold.parallel_transport(point, hist_point, &q)?;
            
            // alpha[i] = rho * <s[i], q>_g at historical point
            let sq_inner = manifold.inner_product(hist_point, s, &q_transported)?;
            alpha[i] = rho * sq_inner;
            
            // Update q (still at current point)
            q = &q - &(manifold.parallel_transport(hist_point, point, &(y * alpha[i]))?);  
        }

        // Scale by initial Hessian approximation
        if m > 0 {
            let last_point = &storage.points[m - 1];
            let _s_last = &storage.s_vectors[m - 1];
            let y_last = &storage.y_vectors[m - 1];
            
            // Use inner products at the historical point where s and y were computed
            let sy_inner = T::one() / storage.rho_values[m - 1]; // We stored 1/sy_inner as rho
            let yy_inner = manifold.inner_product(last_point, y_last, y_last)?;
            
            if yy_inner > T::zero() && sy_inner > T::zero() {
                let gamma = sy_inner / yy_inner;
                q = &q * gamma;
            }
        }

        // Second loop: forward pass through history
        let mut r = q;
        for i in 0..m {
            let hist_point = &storage.points[i];
            let s = &storage.s_vectors[i];
            let y = &storage.y_vectors[i];
            let rho = storage.rho_values[i];
            
            // Transport r from current point to historical point
            let r_transported = manifold.parallel_transport(point, hist_point, &r)?;
            
            // beta = rho * <y[i], r>_g at historical point
            let yr_inner = manifold.inner_product(hist_point, y, &r_transported)?;
            let beta = rho * yr_inner;
            
            // Update r (still at current point)
            let coeff = alpha[i] - beta;
            r = &r + &(manifold.parallel_transport(hist_point, point, &(s * coeff))?);
        }

        // Project to tangent space to ensure feasibility
        let direction = manifold.project_tangent(point, &(-r))?;
        Ok(direction)
    }

    /// Internal step method that has access to L-BFGS-specific state.
    fn step_internal(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        retraction: &impl Retraction<T, D>,
        state: &mut OptimizerState<T, D>,
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
            // Compute s = log_{x_k}(x_{k+1}) using inverse retraction (logarithmic map)
            let s = manifold.inverse_retract(prev_point, &state.point)?;
            
            // Transport previous gradient to current point
            let transported_prev_grad = manifold.parallel_transport(prev_point, &state.point, prev_grad)?;
            
            // Compute y = g_{k+1} - P_{x_k→x_{k+1}}(g_k)
            let y = &riemannian_grad - &transported_prev_grad;
            
            // Add to storage at the previous point (where s and y are computed)
            lbfgs_state.storage.push(s, y, manifold, prev_point)?;
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
    pub fn optimize<C>(
        &mut self,
        cost_fn: &C,
        manifold: &impl Manifold<T, D>,
        initial_point: &OVector<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        C: CostFunction<T, D>,
        DefaultAllocator: Allocator<D, D>,
    {
        let start_time = Instant::now();
        
        // Wrap cost function with caching to avoid redundant computations
        let cached_cost_fn = CachedCostFunction::new(cost_fn);
        
        // Initialize optimizer state
        let initial_cost = cached_cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        let mut lbfgs_state = LBFGSState::new(self.config.memory_size);
        
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
            let retraction = DefaultRetraction;
            self.step_internal(&cached_cost_fn, manifold, &retraction, &mut state, &mut lbfgs_state)?;
        }
    }
}

// Implementation of the Optimizer trait from core
impl<T, D> Optimizer<T, D> for LBFGS<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
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
        LBFGS::optimize(self, cost_fn, manifold, initial_point, stopping_criterion)
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
        // Use Euclidean manifold for simpler testing of storage mechanics
        let manifold = TestEuclideanManifold::new(3);
        let mut storage = LBFGSStorage::<f64, Dyn>::new(3);
        let point = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        
        // Add some vectors with guaranteed positive inner products
        let s1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let y1 = DVector::from_vec(vec![0.5, 0.0, 0.0]);
        let point2 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        storage.push(s1, y1, &manifold, &point).unwrap();
        
        assert_eq!(storage.len(), 1);
        assert!(!storage.is_empty());
        
        // Add more vectors to test capacity
        let s2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let y2 = DVector::from_vec(vec![0.0, 0.5, 0.0]);
        let point3 = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        storage.push(s2, y2, &manifold, &point2).unwrap();
        
        let s3 = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let y3 = DVector::from_vec(vec![0.0, 0.0, 0.5]);
        let point4 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        storage.push(s3, y3, &manifold, &point3).unwrap();
        
        // Should still have 3 (at capacity)
        assert_eq!(storage.len(), 3);
        
        // Add one more - oldest should be removed
        let s4 = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        let y4 = DVector::from_vec(vec![0.5, 0.5, 0.0]);
        let _point5 = DVector::from_vec(vec![2.0, 2.0, 1.0]);
        storage.push(s4, y4, &manifold, &point4).unwrap();
        
        assert_eq!(storage.len(), 3);
    }

    #[test]
    fn test_lbfgs_on_sphere() {
        let manifold = TestSphereManifold::new(3);
        let _retraction = ExponentialRetraction::<f64>::new();
        
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
        
        let result = optimizer.optimize(&cost_fn, &manifold, &initial, &stopping_criterion).unwrap();
        
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
        // let _point2 = manifold.random_point(); // Not needed anymore
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
        let point2 = DVector::from_vec(vec![0.1, 0.1]);
        storage.push(s1, y1, &manifold, &point).unwrap();
        
        let s2 = DVector::from_vec(vec![0.0, 1.0]);
        let y2 = DVector::from_vec(vec![0.5, 0.5]);
        let _point3 = DVector::from_vec(vec![0.2, 0.2]);
        storage.push(s2, y2, &manifold, &point2).unwrap();
        
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