//! RiemannOpt Optimization - Optimization algorithms for Riemannian manifolds.
//!
//! This crate provides concrete implementations of optimization algorithms
//! adapted for Riemannian manifolds, including first-order and second-order methods.
//!
//! # Available Optimizers
//!
//! - **SGD**: Stochastic gradient descent with momentum and adaptive step sizes
//! - **Adam**: Adaptive moment estimation for Riemannian manifolds
//! - **L-BFGS**: Limited memory Broyden-Fletcher-Goldfarb-Shanno
//! - **Trust Region**: Trust region methods with various subproblem solvers
//!
//! # Examples
//!
//! ```rust
//! use riemannopt_optim::{SGD, SGDConfig};
//! use riemannopt_core::optimizer::StoppingCriterion;
//! 
//! // Create SGD optimizer with momentum
//! let mut optimizer = SGD::new(
//!     SGDConfig::new()
//!         .with_constant_step_size(0.01)
//!         .with_classical_momentum(0.9)
//! );
//! 
//! // Set up stopping criteria
//! let stopping_criterion = StoppingCriterion::new()
//!     .with_max_iterations(1000)
//!     .with_gradient_tolerance(1e-6);
//! 
//! // Run optimization (cost_fn, manifold, initial_point defined elsewhere)
//! // let result = optimizer.optimize(&cost_fn, &manifold, &initial_point, &stopping_criterion)?;
//! ```

pub mod sgd;
pub mod adam;
pub mod lbfgs;
pub mod trust_region;
pub mod conjugate_gradient;
pub mod natural_gradient;
pub mod parallel_sgd;
pub mod newton;

// Re-export main optimizers for convenience
pub use sgd::{SGD, SGDConfig, MomentumMethod};
pub use adam::{Adam, AdamConfig};
pub use lbfgs::{LBFGS, LBFGSConfig};
pub use trust_region::{TrustRegion, TrustRegionConfig};
pub use conjugate_gradient::{ConjugateGradient, CGConfig};
pub use natural_gradient::{NaturalGradient, NaturalGradientConfig};
pub use parallel_sgd::ParallelSGDUtils;
pub use newton::{Newton, NewtonConfig};

// Re-export commonly used items from core
pub use riemannopt_core::{
    step_size::StepSizeSchedule,
    preconditioner::{Preconditioner, IdentityPreconditioner},
    fisher::FisherApproximation,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exports() {
        // Test that we can create optimizers from re-exports
        let _config = SGDConfig::<f64>::new();
        let _schedule = StepSizeSchedule::Constant(0.01_f64);
        let _momentum = MomentumMethod::Classical { coefficient: 0.9_f64 };
    }
}