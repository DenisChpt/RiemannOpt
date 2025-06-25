//! Line search algorithms for Riemannian optimization.
//!
//! This module provides various line search strategies that determine the step
//! size along a search direction. Line search is crucial for ensuring convergence
//! and efficiency of optimization algorithms.
//!
//! # Overview
//!
//! A line search finds a step size α such that moving from point x in direction d
//! by αd results in sufficient decrease in the objective function. On manifolds,
//! this movement is performed using retractions.
//!
//! # Wolfe Conditions
//!
//! The strong Wolfe conditions ensure both sufficient decrease and curvature:
//! - Armijo condition: f(R(x, αd)) ≤ f(x) + c₁α⟨∇f(x), d⟩
//! - Curvature condition: |⟨∇f(R(x, αd)), τ(d)⟩| ≤ c₂|⟨∇f(x), d⟩|
//!
//! where R is the retraction, τ is vector transport, and 0 < c₁ < c₂ < 1.

use crate::{
    cost_function::CostFunction,
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use num_traits::Float;
use std::fmt::Debug;

/// Result of a line search.
#[derive(Debug, Clone)]
pub struct LineSearchResult<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// The accepted step size
    pub step_size: T,

    /// The new point after taking the step
    pub new_point: Point<T, D>,

    /// The function value at the new point
    pub new_value: T,

    /// The gradient at the new point (if computed)
    pub new_gradient: Option<TangentVector<T, D>>,

    /// Number of function evaluations performed
    pub function_evals: usize,

    /// Number of gradient evaluations performed
    pub gradient_evals: usize,

    /// Whether the line search succeeded
    pub success: bool,
}

/// Parameters for line search algorithms.
#[derive(Debug, Clone)]
pub struct LineSearchParams<T>
where
    T: Scalar,
{
    /// Initial step size
    pub initial_step_size: T,

    /// Maximum step size allowed
    pub max_step_size: T,

    /// Minimum step size before failure
    pub min_step_size: T,

    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Armijo parameter c₁ (sufficient decrease)
    pub c1: T,

    /// Wolfe parameter c₂ (curvature condition)
    pub c2: T,

    /// Backtracking factor ρ ∈ (0, 1)
    pub rho: T,
}

impl<T> Default for LineSearchParams<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            initial_step_size: T::one(),
            max_step_size: <T as Scalar>::from_f64(10.0),
            min_step_size: <T as Scalar>::from_f64(1e-10),
            max_iterations: 50,
            c1: <T as Scalar>::from_f64(1e-4),
            c2: <T as Scalar>::from_f64(0.9),
            rho: <T as Scalar>::from_f64(0.5),
        }
    }
}

impl<T> LineSearchParams<T>
where
    T: Scalar,
{
    /// Validates the line search parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Step sizes are not positive or incorrectly ordered
    /// - Wolfe constants don't satisfy 0 < c1 < c2 < 1
    /// - Backtracking factor rho is not in (0, 1)
    /// - Maximum iterations is zero
    pub fn validate(&self) -> Result<()> {
        // Validate step sizes
        if self.initial_step_size <= T::zero() {
            return Err(ManifoldError::invalid_parameter(
                "Initial step size must be positive",
            ));
        }
        
        if self.min_step_size <= T::zero() {
            return Err(ManifoldError::invalid_parameter(
                "Minimum step size must be positive",
            ));
        }
        
        if self.max_step_size <= self.min_step_size {
            return Err(ManifoldError::invalid_parameter(
                "Maximum step size must be greater than minimum step size",
            ));
        }
        
        // Validate Wolfe constants
        if self.c1 <= T::zero() || self.c1 >= T::one() {
            return Err(ManifoldError::invalid_parameter(
                "Armijo constant c1 must be in (0, 1)",
            ));
        }
        
        if self.c2 <= self.c1 || self.c2 >= T::one() {
            return Err(ManifoldError::invalid_parameter(
                "Wolfe constant c2 must satisfy c1 < c2 < 1",
            ));
        }
        
        // Validate backtracking factor
        if self.rho <= T::zero() || self.rho >= T::one() {
            return Err(ManifoldError::invalid_parameter(
                "Backtracking factor rho must be in (0, 1)",
            ));
        }
        
        // Validate iterations
        if self.max_iterations == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Maximum iterations must be at least 1",
            ));
        }
        
        Ok(())
    }

    /// Creates parameters for strong Wolfe conditions.
    pub fn strong_wolfe() -> Self {
        Self::default()
    }

    /// Creates parameters for weak Wolfe conditions.
    pub fn weak_wolfe() -> Self {
        Self {
            c2: <T as Scalar>::from_f64(0.9),
            ..Self::default()
        }
    }

    /// Creates parameters for simple backtracking.
    pub fn backtracking() -> Self {
        Self {
            c1: <T as Scalar>::from_f64(0.5),
            c2: <T as Scalar>::from_f64(0.9), // Set to valid value even if not used
            rho: <T as Scalar>::from_f64(0.5),
            max_iterations: 20,
            ..Self::default()
        }
    }
}

/// Context for line search operations, bundling commonly used references.
///
/// This struct reduces the number of parameters passed to line search methods
/// by grouping related references together.
#[derive(Debug)]
pub struct LineSearchContext<'a, T, D, C, M>
where
    T: Scalar,
    D: Dim,
    C: CostFunction<T, D>,
    M: Manifold<T, D>,
    DefaultAllocator: Allocator<D>,
{
    /// The cost function to minimize
    pub cost_fn: &'a C,
    /// The manifold on which we're optimizing
    pub manifold: &'a M,
    /// Phantom data for type parameters
    _phantom: std::marker::PhantomData<(T, D)>,
}

impl<'a, T, D, C, M> LineSearchContext<'a, T, D, C, M>
where
    T: Scalar,
    D: Dim,
    C: CostFunction<T, D>,
    M: Manifold<T, D>,
    DefaultAllocator: Allocator<D>,
{
    /// Create a new line search context.
    pub fn new(cost_fn: &'a C, manifold: &'a M) -> Self {
        Self {
            cost_fn,
            manifold,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Trait for line search algorithms.
pub trait LineSearch<T, D>: Debug
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Performs a line search along a direction.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to minimize
    /// * `manifold` - The manifold on which we're optimizing
    /// * `point` - Current point on the manifold
    /// * `value` - Function value at the current point
    /// * `gradient` - Gradient at the current point
    /// * `direction` - Search direction (typically negative gradient or descent direction)
    /// * `params` - Line search parameters
    ///
    /// # Returns
    ///
    /// A `LineSearchResult` containing the step size and new point.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        value: T,
        gradient: &TangentVector<T, D>,
        direction: &TangentVector<T, D>,
        params: &LineSearchParams<T>,
    ) -> Result<LineSearchResult<T, D>> {
        // Default implementation: compute directional derivative and use efficient method
        let directional_deriv = manifold.inner_product(point, gradient, direction)?;
        self.search_with_deriv(cost_fn, manifold, point, value, direction, directional_deriv, params)
    }
    
    /// Performs a line search along a direction with pre-computed directional derivative.
    ///
    /// This method is more efficient when the directional derivative has already been computed.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to minimize
    /// * `manifold` - The manifold on which we're optimizing
    /// * `point` - Current point on the manifold
    /// * `value` - Function value at the current point
    /// * `direction` - Search direction (typically negative gradient or descent direction)
    /// * `directional_deriv` - Pre-computed directional derivative ⟨∇f(x), d⟩
    /// * `params` - Line search parameters
    ///
    /// # Returns
    ///
    /// A `LineSearchResult` containing the step size and new point.
    #[allow(clippy::too_many_arguments)]
    fn search_with_deriv(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        value: T,
        direction: &TangentVector<T, D>,
        directional_deriv: T,
        params: &LineSearchParams<T>,
    ) -> Result<LineSearchResult<T, D>>;
    

    /// Returns the name of this line search method.
    fn name(&self) -> &str;
    
    /// Performs a line search along a direction using a context.
    ///
    /// This method provides a cleaner API by bundling common parameters into a context.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Line search context containing cost function, manifold, and retraction
    /// * `point` - Current point on the manifold
    /// * `value` - Function value at the current point
    /// * `gradient` - Gradient at the current point
    /// * `direction` - Search direction (typically negative gradient or descent direction)
    /// * `params` - Line search parameters
    ///
    /// # Returns
    ///
    /// A `LineSearchResult` containing the step size and new point.
    fn search_with_context<C, M>(
        &mut self,
        ctx: &LineSearchContext<T, D, C, M>,
        point: &Point<T, D>,
        value: T,
        gradient: &TangentVector<T, D>,
        direction: &TangentVector<T, D>,
        params: &LineSearchParams<T>,
    ) -> Result<LineSearchResult<T, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Default implementation delegates to the old method
        self.search(
            ctx.cost_fn,
            ctx.manifold,
            point,
            value,
            gradient,
            direction,
            params,
        )
    }
}

/// Backtracking line search with Armijo condition.
///
/// This is the simplest line search that ensures sufficient decrease.
/// It starts with a large step and reduces it until the Armijo condition is met.
#[derive(Debug, Clone, Copy)]
pub struct BacktrackingLineSearch;

impl BacktrackingLineSearch {
    /// Creates a new backtracking line search.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BacktrackingLineSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> LineSearch<T, D> for BacktrackingLineSearch
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn search_with_deriv(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        value: T,
        direction: &TangentVector<T, D>,
        directional_deriv: T,
        params: &LineSearchParams<T>,
    ) -> Result<LineSearchResult<T, D>> {
        // Validate parameters
        params.validate()?;

        if directional_deriv >= T::zero() {
            return Err(ManifoldError::numerical_error(
                "Search direction is not a descent direction",
            ));
        }

        let mut step_size = params.initial_step_size;

        // Backtracking loop
        for (iteration, _) in (0..params.max_iterations).enumerate() {
            // Compute new point using retraction
            let scaled_direction = direction * step_size;
            let mut new_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.retract(point, &scaled_direction, &mut new_point)?;

            // Evaluate function at new point
            let new_value = cost_fn.cost(&new_point)?;
            let function_evals = iteration + 1;

            // Check Armijo condition
            let expected_decrease = params.c1 * step_size * directional_deriv;
            if new_value <= value + expected_decrease {
                // Success!
                return Ok(LineSearchResult {
                    step_size,
                    new_point,
                    new_value,
                    new_gradient: None,
                    function_evals,
                    gradient_evals: 0,
                    success: true,
                });
            }

            // Reduce step size
            step_size *= params.rho;

            // Check if step size is too small
            if step_size < params.min_step_size {
                break;
            }
        }

        // Line search failed - provide context
        Err(ManifoldError::numerical_error(
            format!(
                "Backtracking line search failed after {} iterations. Final step size: {:.2e}, min threshold: {:.2e}, directional derivative: {:.2e}",
                params.max_iterations, 
                step_size.to_f64(), 
                params.min_step_size.to_f64(), 
                directional_deriv.to_f64()
            ),
        ))
    }

    fn name(&self) -> &str {
        "Backtracking"
    }
}

/// Line search with strong Wolfe conditions.
///
/// This ensures both sufficient decrease and curvature conditions,
/// which is important for quasi-Newton methods.
#[derive(Debug, Clone)]
pub struct StrongWolfeLineSearch {
    /// Tolerance for the line search
    tolerance: f64,
}

impl StrongWolfeLineSearch {
    /// Creates a new strong Wolfe line search.
    pub fn new() -> Self {
        Self { tolerance: 1e-10 }
    }

    /// Sets the tolerance for the line search.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

impl Default for StrongWolfeLineSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> LineSearch<T, D> for StrongWolfeLineSearch
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    
    fn search_with_deriv(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        value: T,
        direction: &TangentVector<T, D>,
        directional_deriv: T,
        params: &LineSearchParams<T>,
    ) -> Result<LineSearchResult<T, D>> {
        // Validate parameters
        params.validate()?;

        if directional_deriv >= T::zero() {
            return Err(ManifoldError::numerical_error(
                "Search direction is not a descent direction",
            ));
        }

        let mut alpha_low = T::zero();
        let alpha_high = params.max_step_size;
        let mut alpha = params.initial_step_size;

        let mut value_low = value;
        let mut deriv_low = directional_deriv;

        let mut function_evals = 0;
        let mut gradient_evals = 0;

        // Bracketing phase
        for _ in 0..params.max_iterations {
            // Evaluate at current alpha
            let scaled_dir = direction * alpha;
            let mut new_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.retract(point, &scaled_dir, &mut new_point)?;
            let new_value = cost_fn.cost(&new_point)?;
            function_evals += 1;

            // Check Armijo condition
            let armijo_rhs = value + params.c1 * alpha * directional_deriv;

            if new_value > armijo_rhs || (function_evals > 1 && new_value >= value_low) {
                // We've gone too far, zoom between alpha_low and alpha
                // For zoom, we need gradient so we can't avoid computing it
                let gradient = cost_fn.gradient(point)?;
                gradient_evals += 1;
                return self.zoom(
                    cost_fn,
                    manifold,
                    point,
                    value,
                    &gradient,
                    direction,
                    directional_deriv,
                    alpha_low,
                    alpha,
                    value_low,
                    deriv_low,
                    params,
                    function_evals,
                    gradient_evals,
                );
            }

            // Compute gradient at new point
            let new_gradient = cost_fn.gradient(&new_point)?;
            gradient_evals += 1;

            // Transport direction to new point and compute directional derivative
            let mut transported_dir = TangentVector::<T, D>::zeros_generic(direction.shape_generic().0, nalgebra::Const::<1>);
            manifold.parallel_transport(point, &new_point, direction, &mut transported_dir)?;
            let new_deriv = manifold.inner_product(&new_point, &new_gradient, &transported_dir)?;

            // Check curvature condition
            if <T as Float>::abs(new_deriv) <= params.c2 * <T as Float>::abs(directional_deriv) {
                // Strong Wolfe conditions satisfied
                return Ok(LineSearchResult {
                    step_size: alpha,
                    new_point,
                    new_value,
                    new_gradient: Some(new_gradient),
                    function_evals,
                    gradient_evals,
                    success: true,
                });
            }

            if new_deriv >= T::zero() {
                // We've gone past the minimum, zoom between alpha and alpha_low
                let gradient = cost_fn.gradient(point)?;
                gradient_evals += 1;
                return self.zoom(
                    cost_fn,
                    manifold,
                    point,
                    value,
                    &gradient,
                    direction,
                    directional_deriv,
                    alpha,
                    alpha_low,
                    new_value,
                    new_deriv,
                    params,
                    function_evals,
                    gradient_evals,
                );
            }

            // Update bounds
            alpha_low = alpha;
            value_low = new_value;
            deriv_low = new_deriv;

            // Increase alpha
            alpha = <T as Float>::min(alpha * <T as Scalar>::from_f64(2.0), alpha_high);

            if <T as Float>::abs(alpha - alpha_high) < <T as Scalar>::from_f64(self.tolerance) {
                break;
            }
        }

        Err(ManifoldError::numerical_error(
            format!(
                "Strong Wolfe line search failed to bracket after {} iterations. Current alpha: {:.2e}, tolerance: {:.2e}",
                params.max_iterations,
                alpha.to_f64(),
                self.tolerance
            ),
        ))
    }

    fn name(&self) -> &str {
        "StrongWolfe"
    }
}

impl StrongWolfeLineSearch {
    /// Zoom phase of the strong Wolfe line search.
    #[allow(clippy::too_many_arguments)]
    fn zoom<T, D>(
        &self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        value: T,
        _gradient: &TangentVector<T, D>,
        direction: &TangentVector<T, D>,
        init_deriv: T,
        mut alpha_low: T,
        mut alpha_high: T,
        mut value_low: T,
        mut _deriv_low: T,
        params: &LineSearchParams<T>,
        mut function_evals: usize,
        mut gradient_evals: usize,
    ) -> Result<LineSearchResult<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        for _ in 0..params.max_iterations {
            // Bisection (could use cubic interpolation for better convergence)
            let alpha = (alpha_low + alpha_high) * <T as Scalar>::from_f64(0.5);

            // Evaluate at alpha
            let scaled_dir = direction * alpha;
            let mut new_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.retract(point, &scaled_dir, &mut new_point)?;
            let new_value = cost_fn.cost(&new_point)?;
            function_evals += 1;

            // Check Armijo condition
            let armijo_rhs = value + params.c1 * alpha * init_deriv;

            if new_value > armijo_rhs || new_value >= value_low {
                alpha_high = alpha;
            } else {
                // Compute gradient
                let new_gradient = cost_fn.gradient(&new_point)?;
                gradient_evals += 1;

                // Transport direction and compute derivative
                let mut transported_dir = TangentVector::<T, D>::zeros_generic(direction.shape_generic().0, nalgebra::Const::<1>);
                manifold.parallel_transport(point, &new_point, direction, &mut transported_dir)?;
                let new_deriv =
                    manifold.inner_product(&new_point, &new_gradient, &transported_dir)?;

                // Check strong Wolfe curvature condition
                if <T as Float>::abs(new_deriv) <= params.c2 * <T as Float>::abs(init_deriv) {
                    return Ok(LineSearchResult {
                        step_size: alpha,
                        new_point,
                        new_value,
                        new_gradient: Some(new_gradient),
                        function_evals,
                        gradient_evals,
                        success: true,
                    });
                }

                if new_deriv * (alpha_high - alpha_low) >= T::zero() {
                    alpha_high = alpha_low;
                }

                alpha_low = alpha;
                value_low = new_value;
                _deriv_low = new_deriv;
            }

            // Check convergence
            if <T as Float>::abs(alpha_high - alpha_low) < <T as Scalar>::from_f64(self.tolerance) {
                // Return the best point found
                let scaled_dir = direction * alpha_low;
                let mut final_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
                manifold.retract(point, &scaled_dir, &mut final_point)?;

                return Ok(LineSearchResult {
                    step_size: alpha_low,
                    new_point: final_point,
                    new_value: value_low,
                    new_gradient: None,
                    function_evals,
                    gradient_evals,
                    success: true,
                });
            }
        }

        Err(ManifoldError::numerical_error(
            format!(
                "Strong Wolfe zoom phase failed to converge after {} iterations. Final bracket: [{:.2e}, {:.2e}], tolerance: {:.2e}",
                params.max_iterations,
                alpha_low.to_f64(),
                alpha_high.to_f64(),
                self.tolerance
            ),
        ))
    }
}

/// Fixed step size "line search" for simple algorithms.
#[derive(Debug, Clone, Copy)]
pub struct FixedStepSize<T> {
    step_size: T,
}

impl<T: Scalar> FixedStepSize<T> {
    /// Creates a fixed step size line search.
    pub fn new(step_size: T) -> Self {
        Self { step_size }
    }
}

impl<T, D> LineSearch<T, D> for FixedStepSize<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn search_with_deriv(
        &mut self,
        cost_fn: &impl CostFunction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        _value: T,
        direction: &TangentVector<T, D>,
        _directional_deriv: T,
        _params: &LineSearchParams<T>,
    ) -> Result<LineSearchResult<T, D>> {
        let scaled_direction = direction * self.step_size;
        let mut new_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        manifold.retract(point, &scaled_direction, &mut new_point)?;
        let new_value = cost_fn.cost(&new_point)?;

        Ok(LineSearchResult {
            step_size: self.step_size,
            new_point,
            new_value,
            new_gradient: None,
            function_evals: 1,
            gradient_evals: 0,
            success: true,
        })
    }

    fn name(&self) -> &str {
        "FixedStep"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cost_function::QuadraticCost, types::DVector};
    use approx::assert_relative_eq;
    use nalgebra::Dyn;

    // Simple Euclidean manifold for testing
    #[derive(Debug)]
    struct EuclideanManifold {
        dim: usize,
    }

    impl Manifold<f64, Dyn> for EuclideanManifold {
        fn name(&self) -> &str {
            "Euclidean"
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn is_point_on_manifold(&self, _point: &DVector<f64>, _tol: f64) -> bool {
            true
        }
        fn is_vector_in_tangent_space(
            &self,
            _point: &DVector<f64>,
            _vector: &DVector<f64>,
            _tol: f64,
        ) -> bool {
            true
        }
        fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>) {
            result.copy_from(point);
        }
        fn project_tangent(
            &self,
            _point: &DVector<f64>,
            vector: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(vector);
            Ok(())
        }
        fn inner_product(
            &self,
            _point: &DVector<f64>,
            u: &DVector<f64>,
            v: &DVector<f64>,
        ) -> Result<f64> {
            Ok(u.dot(v))
        }
        fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
            result.copy_from(&(point + tangent));
            Ok(())
        }
        fn inverse_retract(
            &self,
            point: &DVector<f64>,
            other: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(&(other - point));
            Ok(())
        }
        fn euclidean_to_riemannian_gradient(
            &self,
            _point: &DVector<f64>,
            euclidean_grad: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(euclidean_grad);
            Ok(())
        }
        fn random_point(&self) -> DVector<f64> {
            DVector::zeros(self.dim)
        }
        fn random_tangent(&self, _point: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
            result.fill(0.0);
            Ok(())
        }
    }

    #[test]
    fn test_backtracking_line_search() {
        let manifold = EuclideanManifold { dim: 2 };
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

        let point = DVector::from_vec(vec![1.0, 1.0]);
        let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
        let direction = -&gradient; // Steepest descent direction

        let mut ls = BacktrackingLineSearch::new();
        let params = LineSearchParams::backtracking();

        let result = ls
            .search(
                &cost,
                &manifold,
                &point,
                value,
                &gradient,
                &direction,
                &params,
            )
            .unwrap();

        assert!(result.success);
        assert!(result.step_size > 0.0);
        assert!(result.new_value < value);
    }

    #[test]
    fn test_backtracking_descent_check() {
        let manifold = EuclideanManifold { dim: 2 };
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

        let point = DVector::from_vec(vec![1.0, 1.0]);
        let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
        let bad_direction = gradient.clone(); // Ascent direction

        let mut ls = BacktrackingLineSearch::new();
        let params = LineSearchParams::backtracking();

        let result = ls.search(
            &cost,
            &manifold,
            &point,
            value,
            &gradient,
            &bad_direction,
            &params,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_strong_wolfe_line_search() {
        let manifold = EuclideanManifold { dim: 2 };
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

        let point = DVector::from_vec(vec![2.0, 3.0]);
        let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
        let direction = -&gradient;

        let mut ls = StrongWolfeLineSearch::new();
        let params = LineSearchParams::strong_wolfe();

        let result = ls
            .search(
                &cost,
                &manifold,
                &point,
                value,
                &gradient,
                &direction,
                &params,
            )
            .unwrap();

        assert!(result.success);
        assert!(result.new_gradient.is_some());
        assert!(result.new_value < value);

        // For quadratic function, optimal step size is 1.0
        assert_relative_eq!(result.step_size, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_fixed_step_size() {
        let manifold = EuclideanManifold { dim: 2 };
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

        let point = DVector::from_vec(vec![1.0, 1.0]);
        let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
        let direction = -&gradient;

        let mut ls = FixedStepSize::new(0.1);
        let params = LineSearchParams::default();

        let result = ls
            .search(
                &cost,
                &manifold,
                &point,
                value,
                &gradient,
                &direction,
                &params,
            )
            .unwrap();

        assert!(result.success);
        assert_eq!(result.step_size, 0.1);
        assert_eq!(result.function_evals, 1);

        // New point should be point + 0.1 * direction
        let expected = &point + &direction * 0.1;
        assert_relative_eq!(result.new_point, expected);
    }

    #[test]
    fn test_line_search_params() {
        let params = LineSearchParams::<f64>::strong_wolfe();
        assert_eq!(params.c1, 1e-4);
        assert_eq!(params.c2, 0.9);

        let params = LineSearchParams::<f64>::backtracking();
        assert_eq!(params.c1, 0.5);
        assert_eq!(params.rho, 0.5);
        assert_eq!(params.max_iterations, 20);
    }
}
