//! Callback support for optimization algorithms.
//!
//! This module provides traits and types for implementing callbacks that can
//! monitor and control the optimization process.

use crate::optimization::OptimizerStateLegacy as OptimizerState;
use crate::core::error::Result;
use crate::core::types::Scalar;
use nalgebra::{Dim, DefaultAllocator};
use nalgebra::allocator::Allocator;
use std::time::Duration;

/// Information passed to callbacks during optimization.
#[derive(Clone, Debug)]
pub struct CallbackInfo<T: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Current optimization state
    pub state: OptimizerState<T, D>,
    
    /// Elapsed time since optimization start
    pub elapsed: Duration,
    
    /// Whether convergence has been achieved
    pub converged: bool,
}

/// Trait for optimization callbacks.
///
/// Callbacks allow monitoring and controlling the optimization process.
/// They can be used for logging, visualization, early stopping, etc.
pub trait OptimizationCallback<T: Scalar, D: Dim>: Send
where
    DefaultAllocator: Allocator<D>,
{
    /// Called at the start of optimization.
    fn on_optimization_start(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Called at the end of each iteration.
    /// 
    /// Returns `true` to continue optimization, `false` to stop early.
    fn on_iteration_end(&mut self, info: &CallbackInfo<T, D>) -> Result<bool> {
        let _ = info; // Unused by default
        Ok(true)
    }
    
    /// Called at the end of optimization.
    fn on_optimization_end(&mut self, info: &CallbackInfo<T, D>) -> Result<()> {
        let _ = info; // Unused by default
        Ok(())
    }
}

/// A no-op callback that does nothing.
pub struct NoOpCallback;

impl<T: Scalar, D: Dim> OptimizationCallback<T, D> for NoOpCallback
where
    DefaultAllocator: Allocator<D>,
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

impl<T: Scalar, D: Dim> OptimizationCallback<T, D> for PrintProgressCallback
where
    DefaultAllocator: Allocator<D>,
    T: std::fmt::Display,
{
    fn on_optimization_start(&mut self) -> Result<()> {
        println!("Starting optimization...");
        Ok(())
    }
    
    fn on_iteration_end(&mut self, info: &CallbackInfo<T, D>) -> Result<bool> {
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
    
    fn on_optimization_end(&mut self, info: &CallbackInfo<T, D>) -> Result<()> {
        println!(
            "Optimization complete after {} iterations. Final cost: {}",
            info.state.iteration,
            info.state.value
        );
        Ok(())
    }
}