//! Step size scheduling strategies for Riemannian optimization algorithms.
//!
//! This module implements mathematically principled step size schedules that
//! adapt the learning rate throughout optimization. These schedules are crucial
//! for balancing exploration (large steps) with convergence precision (small steps),
//! and many provide theoretical convergence guarantees.
//!
//! # Mathematical Foundation
//!
//! ## Step Size Theory
//!
//! The choice of step size αₖ at iteration k significantly affects:
//! - **Convergence rate**: How quickly the algorithm approaches optimal solutions
//! - **Final accuracy**: The precision of the final solution
//! - **Stability**: Whether the algorithm diverges or oscillates
//!
//! ## Theoretical Requirements
//!
//! For many optimization algorithms, convergence theory requires:
//! - **Summable**: Σₖ αₖ² < ∞ (prevents overshooting)
//! - **Non-summable**: Σₖ αₖ = ∞ (ensures sufficient progress)
//!
//! Common schedules satisfying these conditions:
//! - αₖ = α₀/√k (square root decay)
//! - αₖ = α₀/(1 + βk) (polynomial decay with power 1)
//!
//! # Schedule Categories
//!
//! ## Deterministic Schedules
//! - **Constant**: Fixed step size (for convex problems with line search)
//! - **Polynomial decay**: αₖ = α₀/(1 + βk)ᵖ with theoretical guarantees
//! - **Exponential decay**: αₖ = α₀γᵏ for fast initial progress
//!
//! ## Adaptive Schedules
//! - **Square root decay**: Standard for stochastic optimization
//! - **Custom functions**: Problem-specific adaptive strategies
//!
//! # Convergence Analysis
//!
//! ## Convex Functions
//! For convex f and αₖ = α₀/√k:
//! - **Convergence rate**: O(log k/√k) for strongly convex functions
//! - **Regret bounds**: O(√k) for online optimization
//!
//! ## Non-Convex Functions  
//! For smooth non-convex f and diminishing step sizes:
//! - **Stationary points**: Convergence to critical points
//! - **Rate**: Depends on step size schedule and problem structure
//!
//! # Implementation Guidelines
//!
//! ## Parameter Selection
//! - **Initial step size α₀**: Problem-dependent, often 0.01-1.0
//! - **Decay rate**: Balance between early progress and final precision
//! - **Power parameter**: Higher values lead to faster decay
//!
//! ## Practical Considerations
//! - Monitor convergence and adjust schedules if needed
//! - Consider problem conditioning when setting parameters
//! - Use validation performance for schedule selection
//!
//! # Examples
//!
//! ## Stochastic Gradient Descent
//! ```rust,no_run
//! # use riemannopt_core::prelude::*;
//! // Theoretical guarantee: O(1/√k) convergence
//! let schedule = StepSizeSchedule::sqrt_decay(0.1);
//! ```
//!
//! ## Batch Optimization
//! ```rust,no_run
//! # use riemannopt_core::prelude::*;
//! // Fast early progress, fine-tuning later
//! let schedule = StepSizeSchedule::exponential_decay(1.0, 0.95);
//! ```
//!
//! ## Adaptive Learning
//! ```rust,no_run
//! # use riemannopt_core::prelude::*;
//! // Balanced decay for general problems
//! let schedule = StepSizeSchedule::polynomial_decay(0.1, 0.01, 1.0);
//! ```

use crate::types::Scalar;
use num_traits::Float;
use std::fmt::Debug;

/// Mathematical step size scheduling strategies for optimization algorithms.
///
/// These schedules implement various mathematically motivated approaches for
/// adapting the step size throughout optimization. Each schedule has different
/// theoretical properties and is suited to different problem classes.
///
/// # Schedule Types
///
/// ## Constant Schedule
/// - **Use case**: Convex problems with line search, theoretical analysis
/// - **Properties**: No adaptation, requires careful initial selection
/// - **Theory**: Convergence depends on problem-specific step size bounds
///
/// ## Exponential Decay
/// - **Formula**: αₖ = α₀ · γᵏ where 0 < γ < 1
/// - **Use case**: Fast initial progress followed by refinement
/// - **Properties**: Geometric decrease, finite total step sum
/// - **Theory**: May not satisfy non-summable condition for some problems
///
/// ## Polynomial Decay
/// - **Formula**: αₖ = α₀ / (1 + βk)ᵖ where p > 0
/// - **Use case**: General optimization with convergence guarantees
/// - **Properties**: Flexible decay rate through power parameter
/// - **Theory**: Satisfies theoretical conditions when p ∈ (0.5, 1]
///
/// ## Square Root Decay
/// - **Formula**: αₖ = α₀ / √(1 + k)
/// - **Use case**: Stochastic optimization, online learning
/// - **Properties**: Standard choice for SGD-type algorithms
/// - **Theory**: Optimal convergence rates for many stochastic problems
///
/// ## Custom Schedules
/// - **Use case**: Problem-specific adaptive strategies
/// - **Properties**: Maximum flexibility for specialized applications
/// - **Implementation**: Requires external function definition
#[derive(Debug, Clone, Copy)]
pub enum StepSizeSchedule<T: Scalar> {
    /// Fixed step size αₖ = α₀ for all iterations
    /// Suitable for convex problems with line search or theoretical analysis
    Constant(T),
    
    /// Exponential decay: αₖ = α₀ · γᵏ where 0 < γ < 1
    /// Provides fast initial progress with geometric step reduction
    ExponentialDecay {
        /// Initial step size α₀
        initial: T,
        /// Decay factor γ ∈ (0, 1), typically 0.9-0.99
        decay_rate: T,
    },
    
    /// Polynomial decay: αₖ = α₀ / (1 + βk)ᵖ where β > 0, p > 0
    /// Flexible decay with theoretical convergence guarantees
    PolynomialDecay {
        /// Initial step size α₀
        initial: T,
        /// Decay coefficient β > 0, controls decay speed
        decay_rate: T,
        /// Decay power p > 0, typically 0.5-1.0 for convergence
        power: T,
    },
    
    /// Square root decay: αₖ = α₀ / √(1 + k)
    /// Standard choice for stochastic optimization with optimal convergence rates
    SquareRootDecay {
        /// Initial step size α₀
        initial: T,
    },
    
    /// Custom schedule implemented via external function
    /// Enables problem-specific adaptive strategies
    Custom,
}

impl<T: Scalar> StepSizeSchedule<T> {
    /// Computes the step size αₖ for iteration k according to the schedule.
    ///
    /// This method implements the mathematical formula for each schedule type,
    /// ensuring consistent behavior across different optimization algorithms.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number k ≥ 0
    ///
    /// # Returns
    ///
    /// Step size αₖ > 0 for the specified iteration.
    ///
    /// # Mathematical Formulas
    ///
    /// - **Constant**: αₖ = α₀
    /// - **Exponential**: αₖ = α₀ · γᵏ
    /// - **Polynomial**: αₖ = α₀ / (1 + βk)ᵖ
    /// - **Square root**: αₖ = α₀ / √(1 + k)
    ///
    /// # Notes
    ///
    /// For iteration 0, most schedules return the initial step size α₀.
    /// Custom schedules return a default value and require external handling.
    pub fn get_step_size(&self, iteration: usize) -> T {
        let k = <T as Scalar>::from_usize(iteration);
        
        match self {
            Self::Constant(alpha) => *alpha,
            
            Self::ExponentialDecay { initial, decay_rate } => {
                *initial * <T as Float>::powf(*decay_rate, k)
            }
            
            Self::PolynomialDecay { initial, decay_rate, power } => {
                *initial / <T as Float>::powf(T::one() + *decay_rate * k, *power)
            }
            
            Self::SquareRootDecay { initial } => {
                *initial / <T as Float>::sqrt(T::one() + k)
            }
            
            Self::Custom => {
                // Custom schedules require external implementation
                // This default value indicates the schedule needs external handling
                <T as Scalar>::from_f64(0.01)
            }
        }
    }
    
    /// Creates a constant step size schedule αₖ = α₀.
    ///
    /// Use for convex optimization with line search or when theoretical
    /// analysis requires fixed step sizes.
    ///
    /// # Arguments
    ///
    /// * `step_size` - Fixed step size α₀ > 0 for all iterations
    ///
    /// # Theory
    ///
    /// For strongly convex functions with Lipschitz gradients,
    /// α₀ = 1/L provides O(1/k) convergence where L is the Lipschitz constant.
    pub fn constant(step_size: T) -> Self {
        Self::Constant(step_size)
    }
    
    /// Creates an exponential decay schedule αₖ = α₀ · γᵏ.
    ///
    /// Provides aggressive initial progress with geometric step reduction.
    /// Suitable for problems where fast early convergence is desired.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial step size α₀ > 0
    /// * `decay_rate` - Decay factor γ ∈ (0, 1), typically 0.9-0.99
    ///
    /// # Theory
    ///
    /// The total step sum Σₖ αₖ = α₀/(1-γ) is finite, which may limit
    /// convergence to global optima for some non-convex problems.
    pub fn exponential_decay(initial: T, decay_rate: T) -> Self {
        Self::ExponentialDecay { initial, decay_rate }
    }
    
    /// Creates a polynomial decay schedule αₖ = α₀ / (1 + βk)ᵖ.
    ///
    /// Flexible schedule with strong theoretical foundations. Choice of
    /// power p controls convergence properties.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial step size α₀ > 0
    /// * `decay_rate` - Decay coefficient β > 0, controls decay speed
    /// * `power` - Decay power p > 0, typically 0.5-1.0
    ///
    /// # Theory
    ///
    /// For p ∈ (0.5, 1]:
    /// - Σₖ αₖ = ∞ (non-summable condition satisfied)
    /// - Σₖ αₖ² < ∞ (summable squares condition satisfied)
    /// - Guarantees convergence for many optimization problems
    pub fn polynomial_decay(initial: T, decay_rate: T, power: T) -> Self {
        Self::PolynomialDecay { initial, decay_rate, power }
    }
    
    /// Creates a square root decay schedule αₖ = α₀ / √(1 + k).
    ///
    /// The standard choice for stochastic optimization providing optimal
    /// convergence rates for many problems.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial step size α₀ > 0
    ///
    /// # Theory
    ///
    /// This schedule:
    /// - Satisfies both Σₖ αₖ = ∞ and Σₖ αₖ² < ∞
    /// - Provides O(log k/√k) convergence for strongly convex functions
    /// - Achieves optimal regret bounds O(√k) for online optimization
    /// - Standard choice for SGD, AdaGrad precursors, and online learning
    pub fn sqrt_decay(initial: T) -> Self {
        Self::SquareRootDecay { initial }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constant_schedule() {
        let schedule = StepSizeSchedule::constant(0.1);
        assert_eq!(schedule.get_step_size(0), 0.1);
        assert_eq!(schedule.get_step_size(100), 0.1);
        assert_eq!(schedule.get_step_size(1000), 0.1);
    }
    
    #[test]
    fn test_exponential_decay() {
        let schedule = StepSizeSchedule::exponential_decay(1.0, 0.9);
        let step0 = schedule.get_step_size(0);
        let step1 = schedule.get_step_size(1);
        let step10 = schedule.get_step_size(10);
        
        assert!((step0 - 1.0).abs() < 1e-10);
        assert!((step1 - 0.9).abs() < 1e-10);
        assert!(step10 < step1);
        assert!(step10 < 0.5); // 0.9^10 ≈ 0.349
    }
    
    #[test]
    fn test_polynomial_decay() {
        let schedule = StepSizeSchedule::polynomial_decay(1.0, 0.1, 2.0);
        let step0 = schedule.get_step_size(0);
        let step10 = schedule.get_step_size(10);
        
        assert!((step0 - 1.0).abs() < 1e-10);
        // At t=10: 1.0 / (1 + 0.1*10)^2 = 1.0 / 4 = 0.25
        assert!((step10 - 0.25).abs() < 1e-10);
    }
    
    #[test]
    fn test_sqrt_decay() {
        let schedule = StepSizeSchedule::sqrt_decay(1.0);
        let step0 = schedule.get_step_size(0);
        let step3 = schedule.get_step_size(3);
        
        assert!((step0 - 1.0).abs() < 1e-10);
        // At t=3: 1.0 / sqrt(4) = 0.5
        assert!((step3 - 0.5).abs() < 1e-10);
    }
}