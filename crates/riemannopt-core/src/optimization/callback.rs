//! Callback support for optimization algorithms.
//!
//! This module provides traits and types for implementing callbacks that can
//! monitor and control the optimization process.

use crate::optimization::OptimizerState;
use crate::error::Result;
use crate::types::Scalar;
use std::time::Duration;
use std::fmt::Debug;

/// Information passed to callbacks during optimization.
#[derive(Clone, Debug)]
pub struct CallbackInfo<T: Scalar, P, TV>
where
    P: Clone + Debug,
    TV: Clone + Debug,
{
    /// Current optimization state
    pub state: OptimizerState<T, P, TV>,
    
    /// Elapsed time since optimization start
    pub elapsed: Duration,
    
    /// Whether convergence has been achieved
    pub converged: bool,
}

/// Trait for optimization callbacks.
///
/// Callbacks allow monitoring and controlling the optimization process.
/// They can be used for logging, visualization, early stopping, etc.
pub trait OptimizationCallback<T: Scalar, P, TV>: Send
where
    P: Clone + Debug,
    TV: Clone + Debug,
{
    /// Called at the start of optimization.
    fn on_optimization_start(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Called at the end of each iteration.
    /// 
    /// Returns `true` to continue optimization, `false` to stop early.
    fn on_iteration_end(&mut self, info: &CallbackInfo<T, P, TV>) -> Result<bool> {
        let _ = info; // Unused by default
        Ok(true)
    }
    
    /// Called at the end of optimization.
    fn on_optimization_end(&mut self, info: &CallbackInfo<T, P, TV>) -> Result<()> {
        let _ = info; // Unused by default
        Ok(())
    }
}

/// A no-op callback that does nothing.
pub struct NoOpCallback;

impl<T: Scalar, P: Clone + Debug, TV: Clone + Debug> OptimizationCallback<T, P, TV> for NoOpCallback
{
    // Use default implementations
}

/// A callback that prints progress to stdout.
pub struct PrintProgressCallback {
    print_every: usize,
}

impl PrintProgressCallback {
    /// Create a new progress printing callback.
    pub fn new(print_every: usize) -> Self {
        Self { print_every }
    }
}

impl<T: Scalar, P: Clone + Debug, TV: Clone + Debug> OptimizationCallback<T, P, TV> for PrintProgressCallback
where
    T: std::fmt::Display,
{
    fn on_optimization_start(&mut self) -> Result<()> {
        println!("Starting optimization...");
        Ok(())
    }
    
    fn on_iteration_end(&mut self, info: &CallbackInfo<T, P, TV>) -> Result<bool> {
        if info.state.iteration % self.print_every == 0 {
            println!(
                "Iteration {}: cost = {}, gradient norm = {:?}",
                info.state.iteration,
                info.state.value,
                info.state.gradient_norm
            );
        }
        Ok(true)
    }
    
    fn on_optimization_end(&mut self, info: &CallbackInfo<T, P, TV>) -> Result<()> {
        println!(
            "Optimization complete after {} iterations. Final cost: {}",
            info.state.iteration,
            info.state.value
        );
        Ok(())
    }
}