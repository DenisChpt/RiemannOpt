//! # Riemannian Stochastic Gradient Descent (SGD)
//!
//! This module implements the Stochastic Gradient Descent optimizer adapted for
//! Riemannian manifolds. SGD is the most fundamental optimization algorithm,
//! here extended to handle the non-Euclidean geometry of manifolds through
//! retraction operations and proper handling of the Riemannian metric.
//!
//! ## Mathematical Foundation
//!
//! Given a smooth cost function f: ℳ → ℝ on a Riemannian manifold (ℳ, g),
//! Riemannian SGD iteratively minimizes f by following steepest descent directions
//! adapted to the manifold geometry.
//!
//! ### Basic Algorithm
//! For k = 0, 1, 2, ...:
//! ```text
//! 1. Compute Riemannian gradient: ξ_k = grad f(x_k)
//! 2. Choose step size: α_k > 0
//! 3. Update: x_{k+1} = R_{x_k}(-α_k ξ_k)
//! ```
//! where R_{x_k} is a retraction at x_k.
//!
//! ### Momentum Variants
//!
//! #### Classical Momentum
//! ```text
//! v_0 = 0
//! v_{k+1} = β v_k + grad f(x_k)
//! x_{k+1} = R_{x_k}(-α_k v_{k+1})
//! ```
//! where β ∈ [0,1) is the momentum coefficient.
//!
//! #### Nesterov Acceleration
//! ```text
//! v_0 = 0
//! y_k = R_{x_k}(β v_k)                    # Lookahead step
//! v_{k+1} = β v_k + grad f(y_k)              # Update momentum
//! x_{k+1} = R_{x_k}(-α_k v_{k+1})           # Take step
//! ```
//!
//! ## Convergence Theory
//!
//! ### Assumptions
//! - f is continuously differentiable
//! - f is bounded below: inf f(x) > -∞
//! - Riemannian gradient is Lipschitz continuous
//!
//! ### Convergence Rate
//! - **Convex functions**: O(1/√k) convergence in function value
//! - **Strongly convex**: Linear convergence with appropriate step sizes
//! - **Non-convex**: Convergence to stationary points (||grad f|| → 0)
//!
//! ### Step Size Schedules
//! 1. **Constant**: α_k = α (requires line search for convergence guarantees)
//! 2. **Diminishing**: α_k = α/(1 + βk) with ∑ α_k = ∞, ∑ α_k² < ∞
//! 3. **Exponential decay**: α_k = α_0 γ^k with γ ∈ (0,1)
//!
//! ## Implementation Features
//!
//! ### Core Capabilities
//! - **Multiple manifolds**: Works with any manifold implementing the Manifold trait
//! - **Step size scheduling**: Constant, exponential decay, polynomial decay, custom
//! - **Momentum methods**: None, Classical, Nesterov with proper parallel transport
//! - **Gradient clipping**: Prevents exploding gradients in unstable regions
//! - **Line search**: Optional Armijo backtracking for adaptive step sizes
//!
//! ### Numerical Considerations
//! - **Retraction choice**: QR, exponential map, or custom retractions
//! - **Parallel transport**: Momentum vectors transported between tangent spaces
//! - **Gradient projection**: Euclidean → Riemannian gradient conversion
//! - **Numerical stability**: Careful handling of small gradients and step sizes
//!
//! ## Practical Applications
//!
//! ### Machine Learning
//! - **Neural networks**: Training with orthogonal weight constraints (Stiefel)
//! - **Matrix factorization**: Low-rank approximations with manifold constraints
//! - **Dimensionality reduction**: PCA, ICA on appropriate manifolds
//!
//! ### Computer Vision  
//! - **Structure from motion**: Camera pose optimization (rotation groups)
//! - **Shape analysis**: Kendall shape spaces, diffeomorphism groups
//! - **Image registration**: Diffeomorphic registration on appropriate manifolds
//!
//! ### Signal Processing
//! - **Dictionary learning**: Sparse coding with unit norm constraints
//! - **Blind source separation**: ICA on Stiefel manifolds
//! - **Beamforming**: Optimization on complex unit spheres
//!
//! ## Usage Examples
//!
//! ### Basic SGD
//! ```rust
//! # use riemannopt_optim::{SGD, SGDConfig};
//! # use riemannopt_manifolds::Sphere;
//! # use riemannopt_core::{
//! #     manifold::Manifold,
//! #     optimizer::{Optimizer, StoppingCriterion},
//! #     cost_function::CostFunction,
//! #     error::Result,
//! # };
//! # use nalgebra::DVector;
//! # 
//! # fn example() -> Result<()> {
//! # // Setup test cost function
//! # #[derive(Debug)]
//! # struct TestCost;
//! # impl CostFunction<f64, nalgebra::Dyn> for TestCost {
//! #     fn cost(&self, x: &DVector<f64>) -> Result<f64> {
//! #         Ok(x.norm_squared())
//! #     }
//! #     fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>> {
//! #         Ok(x * 2.0)
//! #     }
//! # }
//! # let cost_fn = TestCost;
//! # let x0 = DVector::from_vec(vec![1.0; 10]);
//! # let stopping_criterion = StoppingCriterion::new().with_max_iterations(100);
//! #
//! let manifold = Sphere::new(10).unwrap();
//! let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
//!
//! // Optimize cost function on sphere
//! let result = sgd.optimize(&cost_fn, &manifold, &x0, &stopping_criterion)?;
//! # Ok(())
//! # }
//! # example().unwrap();
//! ```
//!
//! ### Advanced Configuration
//! ```rust
//! use riemannopt_optim::{SGD, SGDConfig, MomentumMethod};
//!
//! let sgd = SGD::new(
//!     SGDConfig::new()
//!         .with_exponential_decay(0.1, 0.95)     // Initial step 0.1, decay 0.95
//!         .with_classical_momentum(0.9)           // Momentum coefficient 0.9
//!         .with_gradient_clip(1.0)                // Clip gradients to norm 1.0
//!         .with_line_search(20)                   // Armijo line search, 20 max iter
//! );
//! ```
//!
//! ## Algorithm Variants
//!
//! This implementation supports several algorithmic variants:
//!
//! ### Vanilla SGD
//! The basic algorithm without momentum, suitable for:
//! - Simple optimization problems
//! - When memory is constrained
//! - As a baseline for comparison
//!
//! ### SGD with Classical Momentum  
//! Accelerates convergence by accumulating gradient history:
//! - Faster convergence on smooth objectives
//! - Helps navigate ravines and plateaus
//! - Requires careful tuning of momentum coefficient
//!
//! ### SGD with Nesterov Acceleration
//! "Lookahead" variant that often converges faster:
//! - Theoretically optimal for strongly convex functions
//! - Better oscillation damping than classical momentum
//! - More complex implementation but often worth it
//!
//! ## Performance Characteristics
//!
//! - **Memory**: O(d) where d is manifold ambient dimension
//! - **Per-iteration cost**: O(d) + cost of retraction and gradient computation
//! - **Convergence**: Highly dependent on problem conditioning and step size
//! - **Parallelization**: Easily parallelizable for batch processing
//!
//! ## References
//!
//! 1. Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*.
//! 2. Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*.
//! 3. Sutskever, I., et al. (2013). "On the importance of initialization and momentum in deep learning."

use riemannopt_core::{
    cost_function::CostFunction,
    core::CachedCostFunction,
    error::Result,
    manifold::{Manifold, Point, TangentVector},
    memory::workspace::{Workspace, WorkspaceBuilder},
    optimizer::{Optimizer, OptimizerStateLegacy as OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker, TerminationReason},
    optimization::callback::{OptimizationCallback, CallbackInfo},
    step_size::StepSizeSchedule,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use std::time::Instant;



/// Momentum method for Riemannian SGD.
///
/// Momentum methods accelerate convergence by incorporating information from
/// previous gradients. On Riemannian manifolds, this requires careful handling
/// of momentum vectors as they move between different tangent spaces.
///
/// # Mathematical Background
///
/// In Euclidean space, momentum methods maintain a velocity vector v_k that
/// accumulates gradient information. On manifolds, these vectors live in 
/// different tangent spaces and must be transported appropriately.
///
/// # Variants
///
/// - **None**: Standard gradient descent without momentum
/// - **Classical**: Heavy ball method with exponential averaging  
/// - **Nesterov**: Accelerated method with lookahead step
#[derive(Debug, Clone)]
pub enum MomentumMethod<T>
where
    T: Scalar,
{
    /// No momentum - pure gradient descent.
    /// 
    /// Update rule: x_{k+1} = R_{x_k}(-α_k grad f(x_k))
    /// 
    /// **Advantages:**
    /// - Simple and robust
    /// - Low memory requirements
    /// - No hyperparameter tuning
    /// 
    /// **Disadvantages:**
    /// - Slow convergence on ill-conditioned problems
    /// - Sensitive to step size choice
    None,
    
    /// Classical momentum (Heavy Ball method).
    /// 
    /// Update rule:
    /// ```text
    /// v_{k+1} = β Τ_{x_{k-1}}^{x_k}(v_k) + grad f(x_k)
    /// x_{k+1} = R_{x_k}(-α_k v_{k+1})
    /// ```
    /// where Τ denotes parallel transport from the previous point to the current point,
    /// and β ∈ [0,1) is the momentum coefficient.
    /// 
    /// **Advantages:**
    /// - Accelerates convergence on smooth objectives
    /// - Helps escape local plateaus
    /// - Dampens oscillations in narrow valleys
    /// 
    /// **Typical values:** β ∈ [0.8, 0.99]
    Classical {
        /// Momentum coefficient β ∈ [0,1). Higher values give more momentum.
        coefficient: T,
    },
    
    /// Nesterov Accelerated Gradient (NAG).
    /// 
    /// Update rule:
    /// ```text
    /// v_{k+1} = β Τ_{x_{k-1}}^{x_k}(v_k) + grad f(x_k)
    /// x_{k+1} = R_{x_k}(-α_k (β v_{k+1} + grad f(x_k)))
    /// ```
    /// where Τ denotes parallel transport from the previous point to the current point.
    /// 
    /// **Advantages:**
    /// - Theoretically optimal convergence rate for strongly convex functions
    /// - Better oscillation control than classical momentum
    /// - Often faster practical convergence
    /// 
    /// **Disadvantages:**
    /// - More complex implementation
    /// - Requires additional gradient evaluation
    /// 
    /// **Typical values:** β ∈ [0.9, 0.99]
    Nesterov {
        /// Momentum coefficient β ∈ [0,1). Should be close to 1 for best acceleration.
        coefficient: T,
    },
}

/// Configuration for the Riemannian SGD optimizer.
///
/// This struct encapsulates all hyperparameters and options for SGD optimization.
/// It provides a builder pattern for easy configuration.
///
/// # Default Configuration
/// 
/// - Step size: Constant 0.01
/// - Momentum: None
/// - Gradient clipping: Disabled
/// - Line search: Disabled
///
/// # Examples
///
/// ```rust
/// use riemannopt_optim::{SGDConfig, MomentumMethod};
/// 
/// // Basic configuration
/// let config = SGDConfig::new().with_constant_step_size(0.01);
/// 
/// // Advanced configuration
/// let config = SGDConfig::new()
///     .with_exponential_decay(0.1, 0.95)
///     .with_classical_momentum(0.9)
///     .with_gradient_clip(1.0)
///     .with_line_search(20);
/// ```
#[derive(Debug, Clone)]
pub struct SGDConfig<T>
where
    T: Scalar,
{
    /// Step size schedule controlling how α_k evolves over iterations.
    /// 
    /// Common choices:
    /// - Constant: Good for well-conditioned problems
    /// - Exponential decay: Ensures convergence guarantees
    /// - Polynomial decay: Theoretical convergence for non-convex problems
    pub step_size: StepSizeSchedule<T>,
    
    /// Momentum method for acceleration.
    /// 
    /// Momentum can significantly improve convergence on smooth objectives
    /// but requires careful tuning of the momentum coefficient.
    pub momentum: MomentumMethod<T>,
    
    /// Gradient clipping threshold to prevent exploding gradients.
    /// 
    /// If Some(threshold), gradients with norm > threshold are scaled to
    /// have norm = threshold. Set to None to disable clipping.
    /// 
    /// **Recommended values:** 1.0 to 10.0 for most problems
    pub gradient_clip: Option<T>,
    
    /// Whether to use line search for adaptive step size selection.
    /// 
    /// When enabled, overrides the step size schedule and uses Armijo
    /// backtracking to find acceptable step sizes automatically.
    /// 
    /// **Trade-off:** More function evaluations vs. better step sizes
    pub use_line_search: bool,
    
    /// Maximum iterations for line search procedures.
    /// 
    /// Controls the computational budget for finding acceptable step sizes.
    /// Higher values allow more thorough search but increase per-iteration cost.
    /// 
    /// **Typical values:** 10-50 iterations
    pub max_line_search_iterations: usize,
}

impl<T> Default for SGDConfig<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            step_size: StepSizeSchedule::Constant(<T as Scalar>::from_f64(0.01)),
            momentum: MomentumMethod::None,
            gradient_clip: None,
            use_line_search: false,
            max_line_search_iterations: 20,
        }
    }
}

impl<T> SGDConfig<T>
where
    T: Scalar,
{
    /// Creates a new SGD configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Sets the step size schedule.
    pub fn with_step_size(mut self, schedule: StepSizeSchedule<T>) -> Self {
        self.step_size = schedule;
        self
    }
    
    /// Sets constant step size.
    pub fn with_constant_step_size(mut self, step_size: T) -> Self {
        self.step_size = StepSizeSchedule::Constant(step_size);
        self
    }
    
    /// Sets exponential decay schedule.
    pub fn with_exponential_decay(mut self, initial: T, decay_rate: T) -> Self {
        self.step_size = StepSizeSchedule::ExponentialDecay { initial, decay_rate };
        self
    }
    
    /// Sets momentum method.
    pub fn with_momentum(mut self, momentum: MomentumMethod<T>) -> Self {
        self.momentum = momentum;
        self
    }
    
    /// Sets classical momentum.
    pub fn with_classical_momentum(mut self, coefficient: T) -> Self {
        self.momentum = MomentumMethod::Classical { coefficient };
        self
    }
    
    /// Sets Nesterov momentum.
    pub fn with_nesterov_momentum(mut self, coefficient: T) -> Self {
        self.momentum = MomentumMethod::Nesterov { coefficient };
        self
    }
    
    /// Sets gradient clipping threshold.
    pub fn with_gradient_clip(mut self, threshold: T) -> Self {
        self.gradient_clip = Some(threshold);
        self
    }
    
    /// Enables line search for adaptive step sizes.
    pub fn with_line_search(mut self, max_iterations: usize) -> Self {
        self.use_line_search = true;
        self.max_line_search_iterations = max_iterations;
        self
    }
}

/// Riemannian SGD optimizer state.
#[derive(Debug)]
struct SGDState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Momentum vector (if using momentum)
    momentum: Option<TangentVector<T, D>>,
    
    /// Previous gradient for momentum calculation  
    previous_gradient: Option<TangentVector<T, D>>,
    
    /// Previous point for parallel transport
    previous_point: Option<Point<T, D>>,
}

impl<T, D> SGDState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn new() -> Self {
        Self {
            momentum: None,
            previous_gradient: None,
            previous_point: None,
        }
    }
}

/// Riemannian Stochastic Gradient Descent optimizer.
///
/// This optimizer implements SGD adapted for Riemannian manifolds, providing
/// the foundation for optimization on curved spaces with proper handling of
/// the manifold geometry through retractions and Riemannian gradients.
///
/// # Mathematical Algorithm
///
/// Given a smooth cost function f: ℳ → ℝ on a Riemannian manifold (ℳ, g),
/// Riemannian SGD performs the following update:
///
/// ```text
/// For k = 0, 1, 2, ...:
///   1. ξ_k ← grad f(x_k)                    // Riemannian gradient
///   2. d_k ← ComputeDirection(ξ_k)          // Apply momentum if enabled
///   3. α_k ← GetStepSize(k)                 // Step size from schedule/line search
///   4. x_{k+1} ← R_{x_k}(α_k d_k)           // Retraction step
/// ```
///
/// where:
/// - grad f(x) is the Riemannian gradient (unique tangent vector satisfying g_x(grad f(x), v) = Df(x)[v])
/// - R_x: T_xℳ → ℳ is a retraction mapping from tangent space to manifold
/// - d_k is the search direction (negative gradient plus momentum)
///
/// # Convergence Guarantees
///
/// Under standard assumptions (Lipschitz gradients, bounded below objective):
///
/// ## Convex Case
/// With diminishing step sizes ∑ α_k = ∞, ∑ α_k² < ∞:
/// ```text
/// min_{0≤j≤k} f(x_j) - f* = O(1/√k)
/// ```
///
/// ## Strongly Convex Case  
/// With constant or appropriately chosen step sizes:
/// ```text
/// f(x_k) - f* = O(ρ^k)
/// ```
/// for some ρ ∈ (0,1) (linear convergence).
///
/// ## Non-convex Case
/// With diminishing step sizes:
/// ```text
/// lim inf_{k→∞} ||∇f(x_k)|| = 0
/// ```
/// (convergence to stationary points).
///
/// # Implementation Features
///
/// ## Core Capabilities
/// - **Universal manifold support**: Works with any manifold implementing the Manifold trait
/// - **Flexible step sizing**: Constant, decay schedules, or adaptive line search
/// - **Momentum acceleration**: Classical and Nesterov variants with proper parallel transport
/// - **Numerical robustness**: Gradient clipping and careful handling of edge cases
///
/// ## Advanced Features
/// - **Line search integration**: Armijo backtracking for automatic step size selection
/// - **Parallel transport**: Proper handling of momentum vectors between tangent spaces
/// - **Gradient conversion**: Automatic Euclidean → Riemannian gradient transformation
/// - **Early stopping**: Integration with convergence criteria and stopping conditions
///
/// # Usage Patterns
///
/// ## Basic Optimization
/// ```rust
/// # use riemannopt_optim::{SGD, SGDConfig};
/// # use riemannopt_manifolds::Sphere;
/// # use riemannopt_core::{
/// #     optimizer::{Optimizer, StoppingCriterion},
/// #     cost_function::CostFunction,
/// #     error::Result,
/// # };
/// # use nalgebra::DVector;
/// # 
/// # fn example() -> Result<()> {
/// # // Setup test cost function
/// # #[derive(Debug)]
/// # struct TestCost;
/// # impl CostFunction<f64, nalgebra::Dyn> for TestCost {
/// #     fn cost(&self, x: &DVector<f64>) -> Result<f64> {
/// #         Ok(x.norm_squared())
/// #     }
/// #     fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>> {
/// #         Ok(x * 2.0)
/// #     }
/// # }
/// # let cost_fn = TestCost;
/// # let x0 = DVector::from_vec(vec![1.0; 10]);
/// # let stopping_criterion = StoppingCriterion::new().with_max_iterations(100);
/// #
/// let manifold = Sphere::new(10).unwrap();
/// let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
///
/// let result = sgd.optimize(&cost_fn, &manifold, &x0, &stopping_criterion)?;
/// println!("Converged: {}, Final cost: {}", result.converged, result.value);
/// # Ok(())
/// # }
/// # example().unwrap();
/// ```
///
/// ## Advanced Configuration
/// ```rust
/// use riemannopt_optim::{SGD, SGDConfig, MomentumMethod};
///
/// let sgd = SGD::new(
///     SGDConfig::new()
///         .with_exponential_decay(0.1, 0.95)     // Start at 0.1, decay by 0.95 each iter
///         .with_classical_momentum(0.9)           // Momentum coefficient 0.9
///         .with_gradient_clip(1.0)                // Clip gradients to unit norm
///         .with_line_search(20)                   // Armijo line search, max 20 iters
/// );
/// ```
///
/// ## Custom Optimization Loop
/// ```rust
/// # use riemannopt_optim::{SGD, SGDConfig};
/// # use riemannopt_manifolds::Sphere;
/// # use riemannopt_core::{
/// #     optimizer::{OptimizerStateLegacy as OptimizerState, StoppingCriterion, ConvergenceChecker},
/// #     cost_function::CostFunction,
/// #     error::Result,
/// # };
/// # use nalgebra::DVector;
/// # 
/// # fn example() -> Result<()> {
/// # // Setup test cost function
/// # #[derive(Debug)]
/// # struct TestCost;
/// # impl CostFunction<f64, nalgebra::Dyn> for TestCost {
/// #     fn cost(&self, x: &DVector<f64>) -> Result<f64> {
/// #         Ok(x.norm_squared())
/// #     }
/// #     fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>> {
/// #         Ok(x * 2.0)
/// #     }
/// # }
/// # let cost_fn = TestCost;
/// # let manifold = Sphere::new(10).unwrap();
/// # let x0 = DVector::from_vec(vec![1.0; 10]);
/// # let initial_cost = 10.0;
/// # let stopping_criterion = StoppingCriterion::new().with_max_iterations(10);
/// #
/// let mut sgd = SGD::new(SGDConfig::default());
/// let mut state = OptimizerState::new(x0.clone(), initial_cost);
///
/// while ConvergenceChecker::check(&state, &manifold, &stopping_criterion)?.is_none() {
///     sgd.step(&cost_fn, &manifold, &mut state)?;
///     println!("Iteration {}: cost = {}", state.iteration, state.value);
/// }
/// # Ok(())
/// # }
/// # example().unwrap();
/// ```
///
/// # Performance Considerations
///
/// ## Computational Complexity
/// - **Per iteration**: O(d) + O(retraction) + O(gradient evaluation)
/// - **Memory**: O(d) for momentum vector (if enabled)
/// - **Function evaluations**: 1 per iteration (+ line search if enabled)
///
/// ## Hyperparameter Guidelines
///
/// ### Step Size
/// - Start with 0.01-0.1 for most problems
/// - Use line search if unsure about appropriate range
/// - Decay for convergence guarantees in non-convex settings
///
/// ### Momentum
/// - Classical: β ∈ [0.8, 0.95] for most problems
/// - Nesterov: β ∈ [0.9, 0.99] for smooth objectives
/// - Disable for very noisy gradients
///
/// ### Gradient Clipping
/// - Threshold 1.0-10.0 for most problems
/// - Essential for training deep networks on manifolds
/// - Monitor gradient norms to set appropriate thresholds
///
/// # Relationship to Other Optimizers
///
/// SGD serves as the foundation for more advanced optimizers:
/// - **Adam**: Adds adaptive per-parameter step sizes
/// - **L-BFGS**: Uses second-order information (limited memory)
/// - **Natural Gradient**: Uses the Fisher information metric
/// - **Trust Region**: Constrains step size based on model validity
///
/// # Examples by Application Domain
///
/// ## Machine Learning: Neural Network Training
/// ```rust
/// # use riemannopt_optim::{SGD, SGDConfig};
/// # use riemannopt_manifolds::Stiefel;
/// // Orthogonal weight constraints on Stiefel manifold
/// let stiefel = Stiefel::new(784, 128).unwrap();
/// let sgd = SGD::new(
///     SGDConfig::new()
///         .with_constant_step_size(0.001)
///         .with_classical_momentum(0.9)
///         .with_gradient_clip(1.0)
/// );
/// ```
///
/// ## Computer Vision: Structure from Motion
/// ```rust
/// # use riemannopt_optim::{SGD, SGDConfig};
/// # // Note: SpecialOrthogonal manifold is not yet implemented
/// # struct SpecialOrthogonal;
/// # impl SpecialOrthogonal {
/// #     fn new(_n: usize) -> Result<Self, &'static str> { Ok(SpecialOrthogonal) }
/// # }
/// // Camera rotation optimization on SO(3)
/// let so3 = SpecialOrthogonal::new(3).unwrap();
/// let sgd = SGD::new(
///     SGDConfig::new()
///         .with_exponential_decay(0.01, 0.99)
///         .with_line_search(10)
/// );
/// ```
///
/// ## Signal Processing: Dictionary Learning
/// ```rust
/// # use riemannopt_optim::{SGD, SGDConfig};
/// # use riemannopt_manifolds::Sphere;
/// # use riemannopt_core::step_size::StepSizeSchedule;
/// // Unit norm dictionary atoms on sphere
/// let sphere = Sphere::new(256).unwrap();
/// let sgd = SGD::new(
///     SGDConfig::new()
///         .with_step_size(StepSizeSchedule::PolynomialDecay {
///             initial: 0.1,
///             decay_rate: 0.1,
///             power: 0.5,
///         })
///         .with_nesterov_momentum(0.95)
/// );
/// ```
#[derive(Debug)]
pub struct SGD<T>
where
    T: Scalar,
{
    config: SGDConfig<T>,
}

impl<T> SGD<T>
where
    T: Scalar,
{
    /// Creates a new SGD optimizer with the given configuration.
    pub fn new(config: SGDConfig<T>) -> Self {
        Self { config }
    }
    
    /// Creates an SGD optimizer with default configuration.
    /// Creates a new SGD optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(SGDConfig::default())
    }
    
    /// Returns the optimizer configuration.
    pub fn config(&self) -> &SGDConfig<T> {
        &self.config
    }
    
    /// Returns the optimizer name.
    pub fn name(&self) -> &str {
        "Riemannian SGD"
    }
    
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
    pub fn optimize<D, C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        D: Dim,
        DefaultAllocator: Allocator<D> + Allocator<D, D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let start_time = Instant::now();
        
        // Wrap cost function with caching to avoid redundant computations
        let cached_cost_fn = CachedCostFunction::new(cost_fn);
        
        // Initialize optimization state
        let initial_cost = cached_cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        
        // Initialize SGD-specific state
        let mut sgd_state = SGDState::new();
        
        // Create a single workspace for the entire optimization
        let n = initial_point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .with_momentum_buffers(n)
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
            self.step_internal(&cached_cost_fn, manifold, &mut state, &mut sgd_state, &mut workspace)?;
        }
    }

    /// Optimizes a cost function on a manifold with an optional callback.
    ///
    /// This method extends `optimize` by allowing a callback function to be called
    /// at each iteration, enabling monitoring, logging, or early stopping.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to minimize
    /// * `manifold` - The manifold constraint
    /// * `initial_point` - Starting point on the manifold
    /// * `stopping_criterion` - Conditions for terminating optimization
    /// * `callback` - Optional callback called at each iteration
    ///
    /// # Returns
    ///
    /// An `OptimizationResult` containing the optimal point and metadata.
    pub fn optimize_with_callback<D, C, M, CB>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
        mut callback: Option<&mut CB>,
    ) -> Result<OptimizationResult<T, D>>
    where
        D: Dim,
        DefaultAllocator: Allocator<D> + Allocator<D, D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
        CB: OptimizationCallback<T, D>,
    {
        let start_time = Instant::now();
        
        // Call callback at start if provided
        if let Some(cb) = callback.as_mut() {
            cb.on_optimization_start()?;
        }
        
        // Wrap cost function with caching to avoid redundant computations
        let cached_cost_fn = CachedCostFunction::new(cost_fn);
        
        // Initialize optimization state
        let initial_cost = cached_cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        
        // Initialize SGD-specific state
        let mut sgd_state = SGDState::new();
        
        // Create a single workspace for the entire optimization
        let n = initial_point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .with_momentum_buffers(n)
            .build();
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                // Get cache statistics for diagnostics
                let ((_cost_hits, cost_misses), (_grad_hits, grad_misses), _) = cached_cost_fn.cache_stats();
                
                // Call final callback if provided
                if let Some(cb) = callback {
                    let info = CallbackInfo {
                        state: state.clone(),
                        elapsed: start_time.elapsed(),
                        converged: true,
                    };
                    cb.on_optimization_end(&info)?;
                }
                
                return Ok(OptimizationResult::new(
                    state.point,
                    state.value,
                    state.iteration,
                    start_time.elapsed(),
                    reason,
                )
                .with_function_evaluations(cost_misses)
                .with_gradient_evaluations(grad_misses)
                .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
            }
            
            // Perform one optimization step
            self.step_internal(&cached_cost_fn, manifold, &mut state, &mut sgd_state, &mut workspace)?;
            
            // Call iteration callback if provided
            if let Some(cb) = callback.as_mut() {
                let info = CallbackInfo {
                    state: state.clone(),
                    elapsed: start_time.elapsed(),
                    converged: false,
                };
                let should_continue = cb.on_iteration_end(&info)?;
                
                // Check if callback requested early stopping
                if !should_continue {
                    let ((_cost_hits, cost_misses), (_grad_hits, grad_misses), _) = cached_cost_fn.cache_stats();
                    
                    return Ok(OptimizationResult::new(
                        state.point,
                        state.value,
                        state.iteration,
                        start_time.elapsed(),
                        TerminationReason::CallbackRequest,
                    )
                    .with_function_evaluations(cost_misses)
                    .with_gradient_evaluations(grad_misses)
                    .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
                }
            }
        }
    }

    /// Performs a single optimization step.
    ///
    /// **Note**: This method creates temporary SGD state and workspace on each call,
    /// which impacts performance. It's intended for debugging or custom optimization loops.
    /// For production use, prefer `optimize()` which maintains state across iterations
    /// and reuses workspace for better performance.
    ///
    /// # Performance Impact
    ///
    /// - Creates new SGD state (momentum buffers) on each call
    /// - Allocates new workspace
    /// - Cannot maintain momentum between calls
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
    pub fn step<D, C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, D>,
    ) -> Result<()>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // For the public interface, we need to maintain internal SGD state
        // This is a limitation of the current design - ideally the state would be generic
        let mut sgd_state = SGDState::new();
        let n = state.point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .with_momentum_buffers(n)
            .build();
        self.step_internal(cost_fn, manifold, state, &mut sgd_state, &mut workspace)
    }
    
    /// Clips the gradient if gradient clipping is enabled.
    ///
    /// Gradient clipping prevents exploding gradients by scaling down gradients
    /// that exceed a specified threshold. This is crucial for numerical stability
    /// in deep learning and optimization with poor conditioning.
    ///
    /// # Algorithm
    /// If ||ξ|| > threshold, then ξ ← ξ * (threshold / ||ξ||)
    ///
    /// # Arguments
    /// * `gradient` - Tangent vector to clip (modified in-place)
    ///
    /// # Mathematical Properties
    /// - Preserves gradient direction: clipped gradient ∝ original gradient
    /// - Bounded norm: ||ξ_clipped|| ≤ threshold
    /// - Identity operation when ||ξ|| ≤ threshold
    fn clip_gradient<D>(&self, gradient: &mut TangentVector<T, D>)
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        if let Some(threshold) = self.config.gradient_clip {
            let norm = gradient.norm();
            if norm > threshold {
                gradient.scale_mut(threshold / norm);
            }
        }
    }
    
    /// Computes the search direction including momentum if enabled.
    ///
    /// The search direction determines the tangent vector along which to move.
    /// Without momentum, this is simply the negative gradient. With momentum,
    /// it incorporates information from previous iterations.
    ///
    /// # Momentum Implementation Details
    ///
    /// ## Classical Momentum
    /// ```text
    /// v_k = β * transport(v_{k-1}) + grad f(x_k)
    /// direction = -v_k
    /// ```
    ///
    /// ## Nesterov Momentum
    /// ```text
    /// v_k = β * transport(v_{k-1}) + grad f(x_k)
    /// direction = -(β * v_k + grad f(x_k))
    /// ```
    ///
    /// # Parallel Transport
    /// 
    /// Momentum vectors must be transported between tangent spaces as the
    /// optimization progresses. This implementation uses proper parallel
    /// transport to move the momentum vector from the previous point to
    /// the current point, preserving its geometric properties.
    ///
    /// # Arguments
    /// * `gradient` - Current Riemannian gradient ξ_k ∈ T_{x_k}ℳ
    /// * `sgd_state` - Mutable SGD state containing momentum history
    /// * `manifold` - Manifold for geometric operations
    /// * `current_point` - Current point x_k ∈ ℳ
    ///
    /// # Returns
    /// Search direction d_k ∈ T_{x_k}ℳ (typically negative for minimization)
    fn compute_search_direction<D, M>(
        &self,
        gradient: &TangentVector<T, D>,
        sgd_state: &mut SGDState<T, D>,
        manifold: &M,
        current_point: &Point<T, D>,
    ) -> Result<TangentVector<T, D>>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        M: Manifold<T, D>,
    {
        match &self.config.momentum {
            MomentumMethod::None => {
                // Simple gradient descent direction
                Ok(-gradient)
            }
            MomentumMethod::Classical { coefficient } => {
                // Classical momentum: v_k = beta*v_{k-1} + grad_k
                let direction = if let Some(ref prev_momentum) = sgd_state.momentum {
                    // Transport previous momentum to current point using parallel transport
                    let transported_momentum = if let Some(ref prev_point) = sgd_state.previous_point {
                        // Use parallel transport (workspace when available)
                        // TODO: Use workspace when manifold types are aligned
                        manifold.parallel_transport(prev_point, current_point, prev_momentum)?
                    } else {
                        // First iteration: no previous point, just use the momentum as-is
                        prev_momentum.clone()
                    };
                    
                    transported_momentum * *coefficient + gradient
                } else {
                    gradient.clone()
                };
                
                sgd_state.momentum = Some(direction.clone());
                sgd_state.previous_gradient = Some(gradient.clone());
                
                Ok(-direction)
            }
            MomentumMethod::Nesterov { coefficient } => {
                // Nesterov momentum: lookahead then gradient step
                let direction = if let Some(ref prev_momentum) = sgd_state.momentum {
                    // Transport previous momentum to current point using parallel transport
                    let transported_momentum = if let Some(ref prev_point) = sgd_state.previous_point {
                        // Use parallel transport (workspace when available)
                        // TODO: Use workspace when manifold types are aligned
                        manifold.parallel_transport(prev_point, current_point, prev_momentum)?
                    } else {
                        // First iteration: no previous point, just use the momentum as-is
                        prev_momentum.clone()
                    };
                    
                    // v_k = beta*v_{k-1} + grad_k
                    let new_momentum = transported_momentum * *coefficient + gradient;
                    
                    // Nesterov update: use beta*v_k + grad_k as direction
                    new_momentum * *coefficient + gradient
                } else {
                    gradient.clone()
                };
                
                sgd_state.momentum = Some(direction.clone() - gradient);
                sgd_state.previous_gradient = Some(gradient.clone());
                
                Ok(-direction)
            }
        }
    }

    /// Internal step method that has access to SGD-specific state.
    fn step_internal<D, C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, D>,
        sgd_state: &mut SGDState<T, D>,
_workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Compute cost and Euclidean gradient
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&state.point)?;
        
        // Convert to Riemannian gradient (using workspace when available)
        // TODO: Use workspace when manifold types are aligned
        let mut riemannian_grad = manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad)?;
        
        // Apply gradient clipping if enabled
        self.clip_gradient(&mut riemannian_grad);
        
        let grad_norm = riemannian_grad.norm();
        state.set_gradient(riemannian_grad.clone(), grad_norm);
        
        // Compute search direction (including momentum)
        let search_direction = self.compute_search_direction(
            &riemannian_grad,
            sgd_state,
            manifold,
            &state.point,
        )?;
        
        // Determine step size
        let step_size = if self.config.use_line_search {
            // Use line search to find appropriate step size
            self.line_search_step_size(cost_fn, manifold, &state.point, &search_direction, cost, _workspace)?
        } else {
            // Use scheduled step size
            self.config.step_size.get_step_size(state.iteration)
        };
        
        // Take the step using retraction (workspace when available)
        let tangent_step = search_direction * step_size;
        // TODO: Use workspace when manifold types are aligned
        let new_point = manifold.retract(&state.point, &tangent_step)?;
        
        // Evaluate cost at new point
        let new_cost = cost_fn.cost(&new_point)?;
        
        // Store the current point as previous point for next iteration
        sgd_state.previous_point = Some(state.point.clone());
        
        // Update state
        state.update(new_point, new_cost);
        
        Ok(())
    }
    
    /// Performs backtracking line search to find an appropriate step size.
    ///
    /// Implements the Armijo condition for sufficient decrease:
    /// f(R_x(α d)) ≤ f(x) + c_1 α ⟨grad f(x), d⟩
    ///
    /// where c_1 ∈ (0, 1) is the Armijo parameter (typically 10^-4).
    ///
    /// # Algorithm
    /// 1. Start with initial step size α_0
    /// 2. Check Armijo condition
    /// 3. If failed, reduce α ← ρ * α (typically ρ = 0.5)
    /// 4. Repeat until condition satisfied or max iterations reached
    ///
    /// # Benefits
    /// - Automatic step size adaptation
    /// - Convergence guarantees without manual tuning
    /// - Robust to poor initial step size estimates
    ///
    /// # Cost
    /// - Additional function evaluations (typically 1-5 per iteration)
    /// - More complex implementation
    ///
    /// # Arguments
    /// * `cost_fn` - Cost function for evaluation
    /// * `manifold` - Manifold for retraction operations
    /// * `point` - Current point x_k
    /// * `direction` - Search direction d_k
    /// * `current_cost` - Current function value f(x_k)
    ///
    /// # Returns
    /// Step size α_k > 0 satisfying Armijo condition
    fn line_search_step_size<D, C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        direction: &TangentVector<T, D>,
        current_cost: T,
_workspace: &mut Workspace<T>,
    ) -> Result<T>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Simple backtracking line search (Armijo condition)
        let c1 = <T as Scalar>::from_f64(1e-4); // Armijo parameter
        let initial_step = self.config.step_size.get_step_size(0);
        let shrink_factor = <T as Scalar>::from_f64(0.5);
        
        let direction_norm = direction.norm();
        if direction_norm < T::epsilon() {
            return Ok(T::zero());
        }
        
        let mut step_size = initial_step;
        
        for _ in 0..self.config.max_line_search_iterations {
            let tangent_step = direction * step_size;
            // TODO: Use workspace when manifold types are aligned
            let new_point = manifold.retract(point, &tangent_step)?;
            let new_cost = cost_fn.cost(&new_point)?;
            
            // Check Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad_f(x)^T*d
            let expected_decrease = c1 * step_size * direction_norm * direction_norm;
            
            if new_cost <= current_cost - expected_decrease {
                return Ok(step_size);
            }
            
            step_size *= shrink_factor;
            
            if step_size < T::epsilon() {
                break;
            }
        }
        
        // If line search fails, return a small step size
        Ok(<T as Scalar>::from_f64(1e-8))
    }
}

// Implementation of the Optimizer trait from core
impl<T, D> Optimizer<T, D> for SGD<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    fn name(&self) -> &str {
        "Riemannian SGD"
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
        // Call the concrete optimize method (not a recursive call)
        SGD::optimize(self, cost_fn, manifold, initial_point, stopping_criterion)
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
        // Call the concrete step method (not a recursive call)
        SGD::step(self, cost_fn, manifold, state)
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
    use approx::assert_relative_eq;

    #[test]
    fn test_sgd_creation() {
        let config = SGDConfig::<f64>::new()
            .with_constant_step_size(0.01)
            .with_classical_momentum(0.9);
        
        let sgd = SGD::new(config);
        assert_eq!(sgd.name(), "Riemannian SGD");
    }
    
    #[test]
    fn test_step_size_schedules() {
        // Test constant step size
        let constant = StepSizeSchedule::Constant(0.1);
        assert_relative_eq!(constant.get_step_size(0), 0.1, epsilon = 1e-10);
        assert_relative_eq!(constant.get_step_size(100), 0.1, epsilon = 1e-10);
        
        // Test exponential decay
        let exp_decay = StepSizeSchedule::ExponentialDecay {
            initial: 1.0,
            decay_rate: 0.9,
        };
        assert_relative_eq!(exp_decay.get_step_size(0), 1.0, epsilon = 1e-10);
        assert!(exp_decay.get_step_size(10) < 1.0);
        
        // Test polynomial decay
        let poly_decay = StepSizeSchedule::PolynomialDecay {
            initial: 1.0,
            decay_rate: 1.0,
            power: 1.0,
        };
        assert_relative_eq!(poly_decay.get_step_size(0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(poly_decay.get_step_size(1), 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gradient_clipping() {
        let config = SGDConfig::<f64>::new().with_gradient_clip(1.0);
        let sgd = SGD::new(config);
        
        let mut gradient = DVector::from_vec(vec![2.0, 0.0, 0.0]);
        sgd.clip_gradient(&mut gradient);
        
        // Gradient should be clipped to norm 1.0
        assert_relative_eq!(gradient.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[0], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_sgd_optimization_simple() {
        let cost_fn = QuadraticCost::simple(Dyn(3));
        let manifold = TestEuclideanManifold::new(3);
        let initial_point = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        
        let mut sgd = SGD::new(SGDConfig::<f64>::new().with_constant_step_size(0.1));
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        let result = sgd.optimize(&cost_fn, &manifold, &initial_point, &stopping_criterion).unwrap();
        
        // Should converge to origin
        assert!(result.converged);
        assert!(result.point.norm() < 1e-3);
    }
    
    #[test]
    fn test_momentum_methods() {
        let config_none = SGDConfig::<f64>::new();
        let config_classical = SGDConfig::<f64>::new().with_classical_momentum(0.9);
        let config_nesterov = SGDConfig::<f64>::new().with_nesterov_momentum(0.9);
        
        let sgd_none = SGD::new(config_none);
        let sgd_classical = SGD::new(config_classical);
        let sgd_nesterov = SGD::new(config_nesterov);
        
        // Just test that they can be created without panic
        assert_eq!(sgd_none.name(), "Riemannian SGD");
        assert_eq!(sgd_classical.name(), "Riemannian SGD");
        assert_eq!(sgd_nesterov.name(), "Riemannian SGD");
    }
    
    #[test]
    fn test_line_search_configuration() {
        let config = SGDConfig::<f64>::new().with_line_search(50);
        let sgd = SGD::new(config);
        
        assert!(sgd.config().use_line_search);
        assert_eq!(sgd.config().max_line_search_iterations, 50);
    }
    
    #[test]
    fn test_sgd_with_momentum() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![2.0, 2.0]);
        
        let mut sgd = SGD::new(
            SGDConfig::<f64>::new()
                .with_constant_step_size(0.01)
                .with_classical_momentum(0.9)
        );
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(500)
            .with_gradient_tolerance(1e-6);
        
        let result = sgd.optimize(&cost_fn, &manifold, &initial_point, &stopping_criterion).unwrap();
        
        // Should converge, potentially faster than without momentum
        assert!(result.point.norm() < 1e-3);
        assert!(result.iterations <= 500);
    }
}