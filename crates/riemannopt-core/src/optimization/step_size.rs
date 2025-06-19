//! Step size scheduling strategies for optimization algorithms.
//!
//! This module provides various learning rate schedules that can be used
//! by different optimizers to adjust the step size during training.

use crate::types::Scalar;
use num_traits::Float;
use std::fmt::Debug;

/// Step size scheduling strategy.
#[derive(Debug, Clone, Copy)]
pub enum StepSizeSchedule<T: Scalar> {
    /// Constant step size
    Constant(T),
    /// Exponential decay: α(t) = α₀ * γ^t
    ExponentialDecay {
        initial: T,
        decay_rate: T,
    },
    /// Polynomial decay: α(t) = α₀ / (1 + βt)^p
    PolynomialDecay {
        initial: T,
        decay_rate: T,
        power: T,
    },
    /// Square root decay: α(t) = α₀ / √(1 + t)
    SquareRootDecay {
        initial: T,
    },
    /// Custom schedule using a function
    Custom,
}

impl<T: Scalar> StepSizeSchedule<T> {
    /// Computes the step size at iteration t.
    pub fn get_step_size(&self, iteration: usize) -> T {
        let t = <T as Scalar>::from_usize(iteration);
        
        match self {
            Self::Constant(alpha) => *alpha,
            
            Self::ExponentialDecay { initial, decay_rate } => {
                *initial * <T as Float>::powf(*decay_rate, t)
            }
            
            Self::PolynomialDecay { initial, decay_rate, power } => {
                *initial / <T as Float>::powf(T::one() + *decay_rate * t, *power)
            }
            
            Self::SquareRootDecay { initial } => {
                *initial / <T as Float>::sqrt(T::one() + t)
            }
            
            Self::Custom => {
                // For custom schedules, return a default value
                // In practice, this would be handled by the optimizer
                <T as Scalar>::from_f64(0.01)
            }
        }
    }
    
    /// Creates a constant step size schedule.
    pub fn constant(step_size: T) -> Self {
        Self::Constant(step_size)
    }
    
    /// Creates an exponential decay schedule.
    pub fn exponential_decay(initial: T, decay_rate: T) -> Self {
        Self::ExponentialDecay { initial, decay_rate }
    }
    
    /// Creates a polynomial decay schedule.
    pub fn polynomial_decay(initial: T, decay_rate: T, power: T) -> Self {
        Self::PolynomialDecay { initial, decay_rate, power }
    }
    
    /// Creates a square root decay schedule.
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