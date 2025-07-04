//! State management for optimization algorithms.
//!
//! This module provides traits and structures for managing the internal state
//! of optimization algorithms. Different algorithms require different state
//! information (e.g., momentum for SGD, moment estimates for Adam), and this
//! module provides a flexible framework for handling these requirements.

use crate::{
    memory::Workspace,
    types::Scalar,
};
use std::collections::HashMap;
use std::fmt::Debug;

/// Trait for optimizer-specific state.
///
/// Each optimization algorithm can define its own state structure that
/// implements this trait. The state contains algorithm-specific information
/// that persists between iterations.
pub trait OptimizerStateData<T, TV>: Debug
where
    T: Scalar,
{
    /// Clone the state data into a boxed trait object.
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, TV>>;
    /// Returns the name of the optimizer this state is for.
    fn optimizer_name(&self) -> &str;

    /// Resets the state to its initial values.
    fn reset(&mut self);

    /// Returns a summary of the current state as key-value pairs.
    fn summary(&self) -> HashMap<String, String>;

    /// Updates any iteration-dependent parameters (e.g., learning rate decay).
    fn update_iteration(&mut self, iteration: usize);
}

/// General optimizer state that includes workspace and algorithm-specific data.
#[derive(Debug)]
pub struct OptimizerStateWithData<T, P, TV>
where
    T: Scalar,
{
    /// Pre-allocated workspace for computations
    pub workspace: Workspace<T>,
    
    /// Algorithm-specific state data
    pub data: Box<dyn OptimizerStateData<T, TV>>,
    
    /// Current iteration number
    pub iteration: usize,
    
    /// Current best point (if tracking)
    pub best_point: Option<P>,
    
    /// Current best value (if tracking)
    pub best_value: Option<T>,
}

impl<T, P, TV> OptimizerStateWithData<T, P, TV>
where
    T: Scalar,
    P: Clone,
    TV: Clone,
{
    /// Create a new optimizer state with given workspace and algorithm-specific data.
    pub fn new(workspace: Workspace<T>, data: Box<dyn OptimizerStateData<T, TV>>) -> Self {
        Self {
            workspace,
            data,
            iteration: 0,
            best_point: None,
            best_value: None,
        }
    }
    
    /// Create a new optimizer state with a default workspace of given size.
    pub fn with_size(n: usize, data: Box<dyn OptimizerStateData<T, TV>>) -> Self {
        Self::new(Workspace::with_size(n), data)
    }
    
    /// Get a reference to the workspace.
    pub fn workspace(&self) -> &Workspace<T> {
        &self.workspace
    }
    
    /// Get a mutable reference to the workspace.
    pub fn workspace_mut(&mut self) -> &mut Workspace<T> {
        &mut self.workspace
    }
    
    /// Update the iteration count and any iteration-dependent parameters.
    pub fn update_iteration(&mut self) {
        self.iteration += 1;
        self.data.update_iteration(self.iteration);
    }
    
    /// Update the best point and value if the new value is better.
    pub fn update_best(&mut self, point: P, value: T) {
        if self.best_value.is_none() || value < self.best_value.unwrap() {
            self.best_point = Some(point);
            self.best_value = Some(value);
        }
    }
    
    /// Reset the state to initial values.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.best_point = None;
        self.best_value = None;
        self.data.reset();
        self.workspace.clear();
    }
    
    /// Get a mutable reference to the algorithm-specific data.
    pub fn data_mut(&mut self) -> &mut dyn OptimizerStateData<T, TV> {
        &mut *self.data
    }
}

impl<T, P, TV> Clone for OptimizerStateWithData<T, P, TV>
where
    T: Scalar,
    P: Clone,
    TV: Clone,
{
    fn clone(&self) -> Self {
        Self {
            workspace: self.workspace.clone(),
            data: self.data.clone_box(),
            iteration: self.iteration,
            best_point: self.best_point.clone(),
            best_value: self.best_value,
        }
    }
}

// Concrete state implementations have been moved to their respective optimizer modules
// in the riemannopt-optim crate to maintain proper separation of concerns.
// Only the trait and generic structure remain here.

// AdamState and AdamStateBuilder have been moved to crates/riemannopt-optim/src/adam.rs

// LBFGSState has been moved to crates/riemannopt-optim/src/lbfgs.rs

// ConjugateGradientState and ConjugateGradientMethod have been moved to 
// crates/riemannopt-optim/src/conjugate_gradient.rs

#[cfg(test)]
mod tests {
    use super::*;
    
    // Tests for the concrete state implementations have been moved to their respective
    // optimizer modules in the riemannopt-optim crate.
    // Here we only test the generic infrastructure.
    
    #[test]
    fn test_optimizer_state_with_data_infrastructure() {
        // This test verifies the generic infrastructure without depending on concrete states
        struct DummyState;
        
        impl<T: Scalar> OptimizerStateData<T, ()> for DummyState {
            fn clone_box(&self) -> Box<dyn OptimizerStateData<T, ()>> {
                Box::new(DummyState)
            }
            
            fn optimizer_name(&self) -> &str {
                "Dummy"
            }
            
            fn reset(&mut self) {}
            
            fn summary(&self) -> HashMap<String, String> {
                HashMap::new()
            }
            
            fn update_iteration(&mut self, _iteration: usize) {}
        }
        
        let dummy_data: Box<dyn OptimizerStateData<f64, ()>> = Box::new(DummyState);
        let mut state = OptimizerStateWithData::<f64, (), ()>::new(
            Workspace::with_size(10), 
            dummy_data
        );
        
        // Test iteration updates
        assert_eq!(state.iteration, 0);
        state.update_iteration();
        assert_eq!(state.iteration, 1);
        
        // Test reset
        state.reset();
        assert_eq!(state.iteration, 0);
    }
}
