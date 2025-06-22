//! State management for optimization algorithms.
//!
//! This module provides traits and structures for managing the internal state
//! of optimization algorithms. Different algorithms require different state
//! information (e.g., momentum for SGD, moment estimates for Adam), and this
//! module provides a flexible framework for handling these requirements.

use crate::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    memory::Workspace,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Trait for optimizer-specific state.
///
/// Each optimization algorithm can define its own state structure that
/// implements this trait. The state contains algorithm-specific information
/// that persists between iterations.
pub trait OptimizerStateData<T, D>: Debug
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Clone the state data into a boxed trait object.
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, D>>;
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
pub struct OptimizerState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Pre-allocated workspace for computations
    pub workspace: Workspace<T>,
    
    /// Algorithm-specific state data
    pub data: Box<dyn OptimizerStateData<T, D>>,
    
    /// Current iteration number
    pub iteration: usize,
    
    /// Current best point (if tracking)
    pub best_point: Option<Point<T, D>>,
    
    /// Current best value (if tracking)
    pub best_value: Option<T>,
}

impl<T, D> OptimizerState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Create a new optimizer state with given workspace and algorithm-specific data.
    pub fn new(workspace: Workspace<T>, data: Box<dyn OptimizerStateData<T, D>>) -> Self {
        Self {
            workspace,
            data,
            iteration: 0,
            best_point: None,
            best_value: None,
        }
    }
    
    /// Create a new optimizer state with a default workspace of given size.
    pub fn with_size(n: usize, data: Box<dyn OptimizerStateData<T, D>>) -> Self {
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
    pub fn update_best(&mut self, point: Point<T, D>, value: T) {
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
}

impl<T, D> Clone for OptimizerState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
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

/// State for gradient descent with momentum.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: Serialize, D: Serialize, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Serialize",
        deserialize = "T: Deserialize<'de>, D: Deserialize<'de>, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Deserialize<'de>"
    ))
)]
pub struct MomentumState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Momentum vector
    pub momentum: Option<TangentVector<T, D>>,

    /// Momentum coefficient (typically 0.9)
    pub beta: T,

    /// Whether to use Nesterov acceleration
    pub nesterov: bool,
}

impl<T, D> MomentumState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new momentum state.
    pub fn new(beta: T, nesterov: bool) -> Self {
        Self {
            momentum: None,
            beta,
            nesterov,
        }
    }

    /// Updates the momentum vector.
    pub fn update_momentum(&mut self, gradient: &TangentVector<T, D>) {
        match &mut self.momentum {
            Some(m) => {
                // m = beta * m + (1 - beta) * gradient
                *m *= self.beta;
                *m += gradient * (T::one() - self.beta);
            }
            None => {
                self.momentum = Some(gradient.clone());
            }
        }
    }

    /// Gets the search direction based on the current momentum.
    pub fn get_direction(&self, gradient: &TangentVector<T, D>) -> TangentVector<T, D> {
        match (&self.momentum, self.nesterov) {
            (Some(m), true) => {
                // Nesterov: direction = gradient + beta * momentum
                gradient + m * self.beta
            }
            (Some(m), false) => {
                // Classical momentum: use momentum directly
                m.clone()
            }
            (None, _) => gradient.clone(),
        }
    }

    /// Gets a reference to the search direction to avoid cloning when possible.
    pub fn get_direction_ref<'a>(&'a self, gradient: &'a TangentVector<T, D>) -> Option<&'a TangentVector<T, D>> {
        match (&self.momentum, self.nesterov) {
            (Some(m), false) => Some(m),
            (None, _) => Some(gradient),
            _ => None, // Nesterov requires computation
        }
    }
}

impl<T, D> OptimizerStateData<T, D> for MomentumState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, D>> {
        Box::new(self.clone())
    }
    
    fn optimizer_name(&self) -> &str {
        if self.nesterov {
            "Nesterov Momentum"
        } else {
            "Classical Momentum"
        }
    }

    fn reset(&mut self) {
        self.momentum = None;
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("beta".to_string(), format!("{}", self.beta));
        summary.insert("nesterov".to_string(), self.nesterov.to_string());
        summary.insert(
            "has_momentum".to_string(),
            self.momentum.is_some().to_string(),
        );
        summary
    }

    fn update_iteration(&mut self, _iteration: usize) {
        // Momentum doesn't change with iteration by default
    }
}

/// State for Adam optimizer.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: Serialize, D: Serialize, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Serialize",
        deserialize = "T: Deserialize<'de>, D: Deserialize<'de>, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Deserialize<'de>"
    ))
)]
pub struct AdamState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// First moment estimate (mean of gradients)
    pub m: Option<TangentVector<T, D>>,

    /// Second moment estimate (mean of squared gradients)
    pub v: Option<TangentVector<T, D>>,

    /// Exponential decay rate for first moment
    pub beta1: T,

    /// Exponential decay rate for second moment
    pub beta2: T,

    /// Small constant for numerical stability
    pub epsilon: T,

    /// Current time step (for bias correction)
    pub t: usize,

    /// Whether to use AMSGrad variant
    pub amsgrad: bool,

    /// Maximum second moment (for AMSGrad)
    pub v_max: Option<TangentVector<T, D>>,
}

impl<T, D> AdamState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new Adam state.
    pub fn new(beta1: T, beta2: T, epsilon: T, amsgrad: bool) -> Self {
        Self {
            m: None,
            v: None,
            beta1,
            beta2,
            epsilon,
            t: 0,
            amsgrad,
            v_max: None,
        }
    }

    /// Updates the moment estimates with a new gradient.
    pub fn update_moments(&mut self, gradient: &TangentVector<T, D>) {
        self.t += 1;

        // Update first moment
        match &mut self.m {
            Some(m) => {
                // m = beta1 * m + (1 - beta1) * gradient
                *m *= self.beta1;
                *m += gradient * (T::one() - self.beta1);
            }
            None => {
                self.m = Some(gradient * (T::one() - self.beta1));
            }
        }

        // Update second moment
        let grad_squared = gradient.component_mul(gradient);
        match &mut self.v {
            Some(v) => {
                // v = beta2 * v + (1 - beta2) * gradient^2
                *v *= self.beta2;
                *v += &grad_squared * (T::one() - self.beta2);

                // Update v_max for AMSGrad
                if self.amsgrad {
                    match &mut self.v_max {
                        Some(v_max) => {
                            v_max.iter_mut().zip(v.iter()).for_each(|(vmax, vi)| {
                                *vmax = <T as num_traits::Float>::max(*vmax, *vi)
                            });
                        }
                        None => {
                            self.v_max = Some(v.clone());
                        }
                    }
                }
            }
            None => {
                let v = grad_squared * (T::one() - self.beta2);
                if self.amsgrad {
                    self.v_max = Some(v.clone());
                }
                self.v = Some(v);
            }
        }
    }

    /// Gets the Adam update direction with bias correction.
    pub fn get_direction(&self) -> Option<TangentVector<T, D>> {
        match (&self.m, &self.v) {
            (Some(m), Some(v)) => {
                // Bias correction
                let t_scalar = <T as Scalar>::from_usize(self.t);
                let bias_correction1 =
                    T::one() - <T as num_traits::Float>::powf(self.beta1, t_scalar);
                let bias_correction2 =
                    T::one() - <T as num_traits::Float>::powf(self.beta2, t_scalar);

                // Prevent division by zero in bias correction
                if bias_correction1 <= T::epsilon() || bias_correction2 <= T::epsilon() {
                    return None;
                }

                // Compute bias-corrected moments
                let m_hat = m / bias_correction1;
                let v_to_use = if self.amsgrad {
                    self.v_max.as_ref().unwrap_or(v)
                } else {
                    v
                };

                // Compute update: m_hat / (sqrt(v_hat) + epsilon)
                let mut direction = m_hat;
                
                // Use safe iteration instead of indexing
                if direction.len() != v_to_use.len() {
                    return None; // Dimension mismatch
                }
                
                for (d, &v_i) in direction.iter_mut().zip(v_to_use.iter()) {
                    let v_hat_i = v_i / bias_correction2;
                    // Ensure v_hat_i is non-negative before taking square root
                    let v_hat_i = <T as num_traits::Float>::max(v_hat_i, T::zero());
                    *d /= <T as num_traits::Float>::sqrt(v_hat_i) + self.epsilon;
                }

                Some(direction)
            }
            _ => None,
        }
    }
}

impl<T, D> OptimizerStateData<T, D> for AdamState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, D>> {
        Box::new(self.clone())
    }
    
    fn optimizer_name(&self) -> &str {
        if self.amsgrad {
            "AMSGrad"
        } else {
            "Adam"
        }
    }

    fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.v_max = None;
        self.t = 0;
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("beta1".to_string(), format!("{}", self.beta1));
        summary.insert("beta2".to_string(), format!("{}", self.beta2));
        summary.insert("epsilon".to_string(), format!("{}", self.epsilon));
        summary.insert("t".to_string(), self.t.to_string());
        summary.insert("amsgrad".to_string(), self.amsgrad.to_string());
        summary
    }

    fn update_iteration(&mut self, _iteration: usize) {
        // Time step is updated in update_moments
    }
}

/// Builder for `AdamState` with a fluent API.
///
/// # Example
///
/// ```rust,ignore
/// let state = AdamStateBuilder::new()
///     .beta1(0.9)
///     .beta2(0.999)
///     .epsilon(1e-8)
///     .amsgrad(true)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct AdamStateBuilder<T: Scalar> {
    beta1: T,
    beta2: T,
    epsilon: T,
    amsgrad: bool,
}

impl<T: Scalar> AdamStateBuilder<T> {
    /// Creates a new builder with default values.
    pub fn new() -> Self {
        Self {
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            amsgrad: false,
        }
    }
    
    /// Sets the exponential decay rate for the first moment.
    pub fn beta1(mut self, beta1: T) -> Self {
        self.beta1 = beta1;
        self
    }
    
    /// Sets the exponential decay rate for the second moment.
    pub fn beta2(mut self, beta2: T) -> Self {
        self.beta2 = beta2;
        self
    }
    
    /// Sets the epsilon value for numerical stability.
    pub fn epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    /// Enables or disables the AMSGrad variant.
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
    
    /// Builds the `AdamState`.
    pub fn build<D>(self) -> AdamState<T, D>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        AdamState::new(self.beta1, self.beta2, self.epsilon, self.amsgrad)
    }
}

impl<T: Scalar> Default for AdamStateBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// State for L-BFGS optimizer.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: Serialize, D: Serialize, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Serialize",
        deserialize = "T: Deserialize<'de>, D: Deserialize<'de>, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Deserialize<'de>"
    ))
)]
pub struct LBFGSState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Memory size (number of vector pairs to store)
    pub memory_size: usize,

    /// Stored position differences (s_k = x_{k+1} - x_k)
    pub s_history: Vec<TangentVector<T, D>>,

    /// Stored gradient differences (y_k = g_{k+1} - g_k)
    pub y_history: Vec<TangentVector<T, D>>,

    /// Inner products rho_k = 1 / (y_k^T s_k)
    pub rho_history: Vec<T>,

    /// Previous point
    pub previous_point: Option<Point<T, D>>,

    /// Previous gradient
    pub previous_gradient: Option<TangentVector<T, D>>,
}

impl<T, D> LBFGSState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new L-BFGS state.
    pub fn new(memory_size: usize) -> Self {
        Self {
            memory_size,
            s_history: Vec::with_capacity(memory_size),
            y_history: Vec::with_capacity(memory_size),
            rho_history: Vec::with_capacity(memory_size),
            previous_point: None,
            previous_gradient: None,
        }
    }

    /// Updates the history with new position and gradient information.
    pub fn update_history(
        &mut self,
        point: &Point<T, D>,
        gradient: &TangentVector<T, D>,
        manifold: &impl Manifold<T, D>,
    ) -> Result<()> {
        if let (Some(prev_point), Some(prev_grad)) = (&self.previous_point, &self.previous_gradient)
        {
            // Compute s_k = transport(x_k, x_{k+1}) of (x_{k+1} - x_k)
            let mut s = TangentVector::<T, D>::zeros_generic(gradient.shape_generic().0, nalgebra::U1);
            manifold.inverse_retract(prev_point, point, &mut s)?;

            // Compute y_k = g_{k+1} - transport(g_k)
            let mut transported_grad = TangentVector::<T, D>::zeros_generic(gradient.shape_generic().0, nalgebra::U1);
            manifold.parallel_transport(prev_point, point, prev_grad, &mut transported_grad)?;
            let y = gradient - &transported_grad;

            // Compute rho_k = 1 / (y_k^T s_k)
            let sy_inner = manifold.inner_product(point, &s, &y)?;

            if sy_inner > T::epsilon() {
                let rho = T::one() / sy_inner;

                // Add to history (with circular buffer behavior)
                if self.s_history.len() >= self.memory_size {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                    self.rho_history.remove(0);
                }

                self.s_history.push(s);
                self.y_history.push(y);
                self.rho_history.push(rho);
            }
        }

        self.previous_point = Some(point.clone());
        self.previous_gradient = Some(gradient.clone());

        Ok(())
    }

    /// Applies the L-BFGS two-loop recursion to compute search direction.
    pub fn compute_direction(
        &self,
        gradient: &TangentVector<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
    ) -> Result<TangentVector<T, D>> {
        let mut q = gradient.clone();
        let mut alphas = Vec::with_capacity(self.s_history.len());

        // First loop (backward)
        for i in (0..self.s_history.len()).rev() {
            let alpha =
                self.rho_history[i] * manifold.inner_product(point, &self.s_history[i], &q)?;
            q -= &self.y_history[i] * alpha;
            alphas.push(alpha);
        }

        // Scale by initial Hessian approximation
        let mut r = if !self.s_history.is_empty() {
            let last_idx = self.s_history.len() - 1;
            let yy = manifold.inner_product(
                point,
                &self.y_history[last_idx],
                &self.y_history[last_idx],
            )?;
            let sy = T::one() / self.rho_history[last_idx];
            q * (sy / yy)
        } else {
            q
        };

        // Second loop (forward)
        alphas.reverse();
        for (i, alpha) in alphas.iter().enumerate().take(self.s_history.len()) {
            let beta =
                self.rho_history[i] * manifold.inner_product(point, &self.y_history[i], &r)?;
            r += &self.s_history[i] * (*alpha - beta);
        }

        Ok(-r)
    }
}

impl<T, D> OptimizerStateData<T, D> for LBFGSState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, D>> {
        Box::new(self.clone())
    }
    
    fn optimizer_name(&self) -> &str {
        "L-BFGS"
    }

    fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
        self.previous_point = None;
        self.previous_gradient = None;
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("memory_size".to_string(), self.memory_size.to_string());
        summary.insert("stored_pairs".to_string(), self.s_history.len().to_string());
        summary
    }

    fn update_iteration(&mut self, _iteration: usize) {
        // L-BFGS doesn't have iteration-dependent parameters
    }
}

/// State for conjugate gradient methods.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: Serialize, D: Serialize, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Serialize",
        deserialize = "T: Deserialize<'de>, D: Deserialize<'de>, DefaultAllocator: Allocator<D>, <DefaultAllocator as Allocator<D>>::Buffer<T>: Deserialize<'de>"
    ))
)]
pub struct ConjugateGradientState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Previous search direction
    pub previous_direction: Option<TangentVector<T, D>>,

    /// Previous gradient
    pub previous_gradient: Option<TangentVector<T, D>>,

    /// Previous point
    pub previous_point: Option<Point<T, D>>,

    /// Method for computing beta (FR, PR, HS, DY)
    pub method: ConjugateGradientMethod,

    /// Number of iterations since last restart
    pub iterations_since_restart: usize,

    /// Restart period (0 means no periodic restart)
    pub restart_period: usize,
}

/// Conjugate gradient method variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ConjugateGradientMethod {
    /// Fletcher-Reeves
    FletcherReeves,
    /// Polak-Ribière
    PolakRibiere,
    /// Hestenes-Stiefel
    HestenesStiefel,
    /// Dai-Yuan
    DaiYuan,
}

impl<T, D> ConjugateGradientState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new conjugate gradient state.
    pub fn new(method: ConjugateGradientMethod, restart_period: usize) -> Self {
        Self {
            previous_direction: None,
            previous_gradient: None,
            previous_point: None,
            method,
            iterations_since_restart: 0,
            restart_period,
        }
    }

    /// Computes the conjugate gradient direction.
    pub fn compute_direction(
        &mut self,
        gradient: &TangentVector<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
    ) -> Result<TangentVector<T, D>> {
        self.iterations_since_restart += 1;

        // Check if we should restart
        let should_restart =
            self.restart_period > 0 && self.iterations_since_restart >= self.restart_period;

        if should_restart || self.previous_direction.is_none() {
            // Restart with steepest descent
            self.iterations_since_restart = 0;
            let neg_gradient = -gradient;
            self.previous_direction = Some(neg_gradient.clone());
            self.previous_gradient = Some(gradient.clone());
            self.previous_point = Some(point.clone());
            return Ok(neg_gradient);
        }

        // Transport previous direction and gradient to current point
        // These are guaranteed to be Some because we checked above
        let prev_point = self.previous_point.as_ref()
            .ok_or_else(|| ManifoldError::invalid_parameter("Missing previous point in CG state"))?;
        let prev_direction = self.previous_direction.as_ref()
            .ok_or_else(|| ManifoldError::invalid_parameter("Missing previous direction in CG state"))?;
        let prev_gradient = self.previous_gradient.as_ref()
            .ok_or_else(|| ManifoldError::invalid_parameter("Missing previous gradient in CG state"))?;
            
        let mut transported_dir = TangentVector::<T, D>::zeros_generic(gradient.shape_generic().0, nalgebra::U1);
        manifold.parallel_transport(
            prev_point,
            point,
            prev_direction,
            &mut transported_dir,
        )?;
        let mut transported_grad = TangentVector::<T, D>::zeros_generic(gradient.shape_generic().0, nalgebra::U1);
        manifold.parallel_transport(
            prev_point,
            point,
            prev_gradient,
            &mut transported_grad,
        )?;

        // Compute beta according to the chosen method
        let beta = match self.method {
            ConjugateGradientMethod::FletcherReeves => {
                let gg_new = manifold.inner_product(point, gradient, gradient)?;
                let gg_old = manifold.inner_product(point, &transported_grad, &transported_grad)?;
                gg_new / gg_old
            }
            ConjugateGradientMethod::PolakRibiere => {
                let g_diff = gradient - &transported_grad;
                let gd_inner = manifold.inner_product(point, gradient, &g_diff)?;
                let gg_old = manifold.inner_product(point, &transported_grad, &transported_grad)?;
                gd_inner / gg_old
            }
            ConjugateGradientMethod::HestenesStiefel => {
                let g_diff = gradient - &transported_grad;
                let gd_inner = manifold.inner_product(point, gradient, &g_diff)?;
                let dd_inner = manifold.inner_product(point, &transported_dir, &g_diff)?;
                -gd_inner / dd_inner
            }
            ConjugateGradientMethod::DaiYuan => {
                let g_diff = gradient - &transported_grad;
                let gg_new = manifold.inner_product(point, gradient, gradient)?;
                let dg_inner = manifold.inner_product(point, &transported_dir, &g_diff)?;
                -gg_new / dg_inner
            }
        };

        // Ensure beta is non-negative for some methods
        let beta = match self.method {
            ConjugateGradientMethod::PolakRibiere => <T as num_traits::Float>::max(beta, T::zero()),
            _ => beta,
        };

        // Compute new direction: d = -g + beta * d_prev
        let direction = -gradient + &transported_dir * beta;

        // Update state - only clone once at the end
        self.previous_direction = Some(direction.clone());
        self.previous_gradient = Some(gradient.clone());
        self.previous_point = Some(point.clone());

        Ok(direction)
    }
}

impl<T, D> OptimizerStateData<T, D> for ConjugateGradientState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, D>> {
        Box::new(self.clone())
    }
    
    fn optimizer_name(&self) -> &str {
        match self.method {
            ConjugateGradientMethod::FletcherReeves => "CG-FR",
            ConjugateGradientMethod::PolakRibiere => "CG-PR",
            ConjugateGradientMethod::HestenesStiefel => "CG-HS",
            ConjugateGradientMethod::DaiYuan => "CG-DY",
        }
    }

    fn reset(&mut self) {
        self.previous_direction = None;
        self.previous_gradient = None;
        self.previous_point = None;
        self.iterations_since_restart = 0;
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("method".to_string(), format!("{:?}", self.method));
        summary.insert(
            "restart_period".to_string(),
            self.restart_period.to_string(),
        );
        summary.insert(
            "iterations_since_restart".to_string(),
            self.iterations_since_restart.to_string(),
        );
        summary
    }

    fn update_iteration(&mut self, _iteration: usize) {
        // Iteration counting is handled in compute_direction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVector;
    use nalgebra::Dyn;

    #[test]
    fn test_momentum_state() {
        let mut state = MomentumState::<f64, Dyn>::new(0.9, false);
        assert_eq!(state.optimizer_name(), "Classical Momentum");

        let gradient = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        state.update_momentum(&gradient);

        assert!(state.momentum.is_some());
        let summary = state.summary();
        assert_eq!(summary.get("beta").unwrap(), "0.9");
        assert_eq!(summary.get("has_momentum").unwrap(), "true");

        state.reset();
        assert!(state.momentum.is_none());
    }

    #[test]
    fn test_nesterov_momentum() {
        let mut state = MomentumState::<f64, Dyn>::new(0.9, true);
        assert_eq!(state.optimizer_name(), "Nesterov Momentum");

        let gradient = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        state.update_momentum(&gradient);

        let direction = state.get_direction(&gradient);
        // For first iteration with Nesterov, direction = gradient + beta * gradient = 1.9 * gradient
        assert_eq!(direction, gradient * 1.9);
    }

    #[test]
    fn test_adam_state() {
        let mut state = AdamState::<f64, Dyn>::new(0.9, 0.999, 1e-8, false);
        assert_eq!(state.optimizer_name(), "Adam");

        let gradient = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        state.update_moments(&gradient);

        assert_eq!(state.t, 1);
        assert!(state.m.is_some());
        assert!(state.v.is_some());

        let direction = state.get_direction();
        assert!(direction.is_some());

        let summary = state.summary();
        assert_eq!(summary.get("t").unwrap(), "1");
        assert_eq!(summary.get("amsgrad").unwrap(), "false");
    }

    #[test]
    fn test_amsgrad_state() {
        let mut state = AdamState::<f64, Dyn>::new(0.9, 0.999, 1e-8, true);
        assert_eq!(state.optimizer_name(), "AMSGrad");

        let gradient = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        state.update_moments(&gradient);

        assert!(state.v_max.is_some());
    }

    #[test]
    fn test_lbfgs_state() {
        let state = LBFGSState::<f64, Dyn>::new(5);
        assert_eq!(state.optimizer_name(), "L-BFGS");
        assert_eq!(state.memory_size, 5);

        let summary = state.summary();
        assert_eq!(summary.get("memory_size").unwrap(), "5");
        assert_eq!(summary.get("stored_pairs").unwrap(), "0");
    }

    #[test]
    fn test_conjugate_gradient_state() {
        let mut state =
            ConjugateGradientState::<f64, Dyn>::new(ConjugateGradientMethod::FletcherReeves, 10);
        assert_eq!(state.optimizer_name(), "CG-FR");

        let summary = state.summary();
        assert_eq!(summary.get("method").unwrap(), "FletcherReeves");
        assert_eq!(summary.get("restart_period").unwrap(), "10");

        state.reset();
        assert_eq!(state.iterations_since_restart, 0);
    }

    #[test]
    #[cfg(feature = "serde")]
    #[ignore] // Temporarily ignore due to nalgebra serde bounds complexity
    fn test_state_serialization() {
        // Test that states can be serialized and deserialized
        let state = MomentumState::<f64, Dyn>::new(0.9, true);
        let serialized = serde_json::to_string(&state).unwrap();
        let deserialized: MomentumState<f64, Dyn> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.beta, state.beta);
        assert_eq!(deserialized.nesterov, state.nesterov);
    }
}
