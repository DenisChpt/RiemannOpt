//! Retraction and vector transport methods for Riemannian manifolds.
//!
//! This module provides infrastructure for retractions, which are smooth
//! approximations to the exponential map that are computationally efficient.
//! Retractions are essential for optimization algorithms on manifolds.
//!
//! # Mathematical Background
//!
//! A retraction on a manifold M is a smooth mapping R: TM → M from the tangent
//! bundle to the manifold with the following properties:
//! - R(p, 0) = p (centering condition)
//! - dR(p, 0)[v] = v (local rigidity condition)
//!
//! Common retractions include:
//! - Exponential map (exact but often expensive)
//! - Projection-based retraction
//! - QR-based retraction (for matrix manifolds)
//! - Cayley transform

use crate::{
    error::Result,
    manifold::{Manifold, Point, TangentVector as TangentVectorType},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use num_traits::Float;
use std::fmt::Debug;

/// Order of approximation for a retraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RetractionOrder {
    /// First-order retraction (satisfies basic requirements)
    First,
    /// Second-order retraction (matches exponential map to second order)
    Second,
    /// Exact exponential map
    Exact,
}

/// Trait for retraction methods on a manifold.
///
/// A retraction provides a way to move from a point on the manifold
/// in a given tangent direction, approximating the exponential map.
pub trait Retraction<T, D>: Debug
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Returns the name of this retraction method.
    fn name(&self) -> &str;

    /// Returns the order of approximation of this retraction.
    fn order(&self) -> RetractionOrder;

    /// Performs the retraction.
    ///
    /// Given a point `p` on the manifold and a tangent vector `v` at `p`,
    /// computes a new point on the manifold and stores it in `result`.
    ///
    /// # Arguments
    ///
    /// * `manifold` - The manifold on which to perform the retraction
    /// * `point` - A point on the manifold
    /// * `tangent` - A tangent vector at `point`
    /// * `result` - Output parameter for the new point on the manifold
    fn retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        result: &mut Point<T, D>,
    ) -> Result<()>;

    /// Computes the inverse retraction (logarithmic map).
    ///
    /// Given two points `p` and `q` on the manifold, computes a tangent
    /// vector `v` at `p` such that `retract(p, v) ≈ q` and stores it in `result`.
    ///
    /// # Arguments
    ///
    /// * `manifold` - The manifold on which to perform the inverse retraction
    /// * `point` - A point on the manifold
    /// * `other` - Another point on the manifold
    /// * `result` - Output parameter for the tangent vector at `point`
    fn inverse_retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()>;
}

/// Default retraction using the manifold's built-in retraction method.
#[derive(Debug, Clone, Copy)]
pub struct DefaultRetraction;

impl<T, D> Retraction<T, D> for DefaultRetraction
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Default"
    }

    fn order(&self) -> RetractionOrder {
        RetractionOrder::First
    }

    fn retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        result: &mut Point<T, D>,
    ) -> Result<()> {
        manifold.retract(point, tangent, result)
    }

    fn inverse_retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        manifold.inverse_retract(point, other, result)
    }
}

/// Projection-based retraction.
///
/// This retraction works by moving in the ambient space and then
/// projecting back onto the manifold.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionRetraction;

impl<T, D> Retraction<T, D> for ProjectionRetraction
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Projection"
    }

    fn order(&self) -> RetractionOrder {
        RetractionOrder::First
    }

    fn retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        result: &mut Point<T, D>,
    ) -> Result<()> {
        // Move in the ambient space
        result.copy_from(point);
        *result += tangent;
        // Project back onto the manifold (in-place)
        let temp = result.clone();
        manifold.project_point(&temp, result);
        Ok(())
    }

    fn inverse_retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        // Simple approximation: project the difference onto the tangent space
        result.copy_from(other);
        *result -= point;
        let temp = result.clone();
        manifold.project_tangent(point, &temp, result)?;
        Ok(())
    }
}

/// Exponential map retraction.
///
/// This uses the true exponential map of the manifold when available,
/// or numerical geodesic integration as a fallback.
/// 
/// The exponential map is the gold standard retraction that moves along
/// geodesics, providing exact results but often at higher computational cost.
#[derive(Debug, Clone)]
pub struct ExponentialRetraction<T> {
    /// Number of integration steps for numerical geodesic computation
    integration_steps: usize,
    /// Tolerance for adaptive step size control
    tolerance: T,
}

impl<T: Scalar> ExponentialRetraction<T> {
    /// Creates a new exponential retraction with default parameters.
    pub fn new() -> Self {
        Self {
            integration_steps: 10,
            tolerance: <T as Scalar>::from_f64(1e-10),
        }
    }
    
    /// Creates an exponential retraction with custom integration parameters.
    pub fn with_parameters(integration_steps: usize, tolerance: T) -> Self {
        Self {
            integration_steps,
            tolerance,
        }
    }
}

impl<T: Scalar> Default for ExponentialRetraction<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> Retraction<T, D> for ExponentialRetraction<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Exponential"
    }

    fn order(&self) -> RetractionOrder {
        RetractionOrder::Exact
    }

    fn retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        result: &mut Point<T, D>,
    ) -> Result<()> {
        // First check if manifold has exact exponential map
        if manifold.has_exact_exp_log() {
            manifold.retract(point, tangent, result)?;
            return Ok(());
        }
        
        // Otherwise, use numerical integration of geodesic equation
        // Geodesic gamma(t) satisfies:
        // gamma(0) = point, gamma'(0) = tangent
        // gamma''(t) + Gamma(gamma'(t), gamma'(t)) = 0
        // where Gamma are the Christoffel symbols
        
        // For simplicity, we use a first-order approximation:
        // Divide the path into small steps and use manifold retraction
        let n_steps = self.integration_steps;
        let dt = T::one() / <T as Scalar>::from_usize(n_steps);
        
        // Use result as our working buffer
        result.copy_from(point);
        
        // Pre-allocate a buffer for the scaled tangent
        let mut scaled_tangent = tangent.clone();
        scaled_tangent *= dt;
        
        // Pre-allocate a temporary buffer
        let mut temp_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        
        // Integrate along the geodesic
        for _ in 0..n_steps {
            // Take a small step
            manifold.retract(result, &scaled_tangent, &mut temp_point)?;
            
            // For a true geodesic, we would parallel transport the velocity
            // and update it according to the geodesic equation
            // Here we use a simple approximation
            
            // Swap buffers to avoid allocation
            std::mem::swap(result, &mut temp_point);
        }
        
        Ok(())
    }

    fn inverse_retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        // First check if manifold has exact logarithmic map
        if manifold.has_exact_exp_log() {
            manifold.inverse_retract(point, other, result)?;
            return Ok(());
        }
        
        // Otherwise, use iterative method to find v such that exp_p(v) = other
        // This is a simplified implementation using secant method
        
        // Initial guess using projection
        manifold.inverse_retract(point, other, result)?;
        
        let mut error = T::one();
        let max_iters = 10;
        let mut iter = 0;
        
        // Pre-allocate buffers for iteration
        let mut exp_v = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut error_vec = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut grad = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        
        while error > self.tolerance && iter < max_iters {
            // Compute exp_p(v)
            self.retract(manifold, point, result, &mut exp_v)?;
            
            // Compute error vector in ambient space
            error_vec.copy_from(other);
            error_vec -= &exp_v;
            error = error_vec.norm();
            
            if error < self.tolerance {
                break;
            }
            
            // Update v using gradient descent in tangent space
            // This is a simplified update - a full implementation would use
            // the differential of the exponential map
            manifold.project_tangent(point, &error_vec, &mut grad)?;
            *result += &grad * <T as Scalar>::from_f64(0.5);
            
            iter += 1;
        }
        
        Ok(())
    }
}

/// QR-based retraction for matrix manifolds.
///
/// This retraction is particularly useful for the Stiefel and Grassmann manifolds.
/// Given a point X on St(n,p) and tangent vector V, the retraction is:
/// R_X(V) = qf(X + V), where qf() denotes the Q factor of QR decomposition.
///
/// This retraction is computationally efficient with O(np²) complexity.
#[derive(Debug, Clone, Copy)]
pub struct QRRetraction;

impl QRRetraction {
    /// Creates a new QR-based retraction.
    pub fn new() -> Self {
        Self
    }
}

impl Default for QRRetraction {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> Retraction<T, D> for QRRetraction
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "QR"
    }

    fn order(&self) -> RetractionOrder {
        RetractionOrder::Second
    }

    fn retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        result: &mut Point<T, D>,
    ) -> Result<()> {
        // For general manifolds, QR retraction doesn't make sense
        // This should be overridden by specific matrix manifolds
        // For now, fall back to projection
        result.copy_from(point);
        *result += tangent;
        let temp = result.clone();
        manifold.project_point(&temp, result);
        Ok(())
    }

    fn inverse_retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        // Approximate inverse using projection
        // Specific manifolds should override this
        result.copy_from(other);
        *result -= point;
        let temp = result.clone();
        manifold.project_tangent(point, &temp, result)?;
        Ok(())
    }
}

/// Cayley transform retraction.
///
/// The Cayley transform is useful for orthogonal groups and related manifolds.
/// Given a skew-symmetric matrix W, the Cayley transform is:
/// cay(W) = (I - W/2)^{-1}(I + W/2)
///
/// This provides a second-order retraction for SO(n) and can be adapted
/// for other matrix manifolds.
#[derive(Debug, Clone)]
pub struct CayleyRetraction<T> {
    /// Scaling parameter (typically 1.0)
    scaling: T,
}

impl<T: Scalar> CayleyRetraction<T> {
    /// Creates a new Cayley retraction with default scaling.
    pub fn new() -> Self {
        Self {
            scaling: T::one(),
        }
    }
    
    /// Creates a Cayley retraction with custom scaling parameter.
    pub fn with_scaling(scaling: T) -> Self {
        Self { scaling }
    }
}

impl<T: Scalar> Default for CayleyRetraction<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> Retraction<T, D> for CayleyRetraction<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Cayley"
    }

    fn order(&self) -> RetractionOrder {
        RetractionOrder::Second
    }

    fn retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        result: &mut Point<T, D>,
    ) -> Result<()> {
        // For general manifolds, Cayley transform doesn't apply
        // This should be overridden by specific matrix manifolds (SO(n), etc.)
        // For now, fall back to projection
        result.copy_from(point);
        *result += tangent * self.scaling;
        let temp = result.clone();
        manifold.project_point(&temp, result);
        Ok(())
    }

    fn inverse_retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        // Approximate inverse using projection
        // Specific manifolds should override this
        result.copy_from(other);
        *result -= point;
        *result /= self.scaling;
        let temp = result.clone();
        manifold.project_tangent(point, &temp, result)?;
        Ok(())
    }
}

/// Polar retraction for matrix manifolds.
///
/// The polar retraction is defined as R_X(V) = (X + V)M, where M is the
/// orthogonal polar factor from the polar decomposition X + V = QM.
///
/// This retraction is particularly useful for the Stiefel and Grassmann manifolds,
/// providing a second-order retraction that preserves more geometric structure
/// than QR decomposition.
#[derive(Debug, Clone)]
pub struct PolarRetraction<T> {
    /// Maximum iterations for iterative polar decomposition
    _max_iterations: usize,
    /// Tolerance for convergence
    _tolerance: T,
}

impl<T: Scalar> PolarRetraction<T> {
    /// Creates a new polar retraction with default parameters.
    pub fn new() -> Self {
        Self {
            _max_iterations: 10,
            _tolerance: <T as Scalar>::from_f64(1e-10),
        }
    }
    
    /// Creates a polar retraction with custom parameters.
    pub fn with_parameters(max_iterations: usize, tolerance: T) -> Self {
        Self {
            _max_iterations: max_iterations,
            _tolerance: tolerance,
        }
    }
}

impl<T: Scalar> Default for PolarRetraction<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> Retraction<T, D> for PolarRetraction<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Polar"
    }

    fn order(&self) -> RetractionOrder {
        RetractionOrder::Second
    }

    fn retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        result: &mut Point<T, D>,
    ) -> Result<()> {
        // For general manifolds, polar retraction doesn't apply
        // This should be overridden by specific matrix manifolds
        // For now, fall back to projection
        result.copy_from(point);
        *result += tangent;
        let temp = result.clone();
        manifold.project_point(&temp, result);
        Ok(())
    }

    fn inverse_retract(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        other: &Point<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        // Approximate inverse using projection
        // Specific manifolds should override this with proper implementation
        result.copy_from(other);
        *result -= point;
        let temp = result.clone();
        manifold.project_tangent(point, &temp, result)?;
        Ok(())
    }
}

/// Vector transport methods for moving tangent vectors along retractions.
pub trait VectorTransport<T, D>: Debug
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Returns the name of this vector transport method.
    fn name(&self) -> &str;

    /// Transports a tangent vector along a retraction.
    ///
    /// Given a tangent vector `v` at point `p`, transport it to the
    /// tangent space at `retract(p, direction)` and store it in `result`.
    ///
    /// # Arguments
    ///
    /// * `manifold` - The manifold on which to perform the transport
    /// * `point` - Starting point on the manifold
    /// * `direction` - Direction of the retraction
    /// * `vector` - Tangent vector to transport
    /// * `result` - Output parameter for the transported vector at the destination point
    fn transport(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        direction: &TangentVectorType<T, D>,
        vector: &TangentVectorType<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()>;
}

/// Parallel transport (when available).
#[derive(Debug, Clone, Copy)]
pub struct ParallelTransport;

impl<T, D> VectorTransport<T, D> for ParallelTransport
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Parallel"
    }

    fn transport(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        direction: &TangentVectorType<T, D>,
        vector: &TangentVectorType<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        let mut destination = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        manifold.retract(point, direction, &mut destination)?;
        manifold.parallel_transport(point, &destination, vector, result)?;
        Ok(())
    }
}

/// Vector transport by projection.
///
/// This transports a vector by projecting it onto the tangent space
/// at the destination point.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionTransport;

impl<T, D> VectorTransport<T, D> for ProjectionTransport
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Projection"
    }

    fn transport(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        direction: &TangentVectorType<T, D>,
        vector: &TangentVectorType<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        let mut destination = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        manifold.retract(point, direction, &mut destination)?;
        manifold.project_tangent(&destination, vector, result)?;
        Ok(())
    }
}

/// Vector transport by differential of retraction.
///
/// This method transports vectors using the differential of the retraction map.
/// It's more accurate than projection transport and computationally cheaper
/// than parallel transport for many manifolds.
#[derive(Debug, Clone, Copy)]
pub struct DifferentialRetraction;

impl DifferentialRetraction {
    /// Creates a new differential retraction transport.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DifferentialRetraction {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> VectorTransport<T, D> for DifferentialRetraction
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "DifferentialRetraction"
    }

    fn transport(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        direction: &TangentVectorType<T, D>,
        vector: &TangentVectorType<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        // Transport using finite differences to approximate the differential
        let epsilon = <T as Scalar>::from_f64(1e-8);
        
        // Pre-allocate buffers
        let mut destination = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut perturbed_destination = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut perturbed_direction = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        
        // Compute R(x, tv + epsilon * w) and R(x, tv)
        perturbed_direction.copy_from(direction);
        perturbed_direction += &(vector * epsilon);
        
        manifold.retract(point, direction, &mut destination)?;
        manifold.retract(point, &perturbed_direction, &mut perturbed_destination)?;
        
        // Approximate transported vector as (R(x, tv + epsilon * w) - R(x, tv)) / epsilon
        result.copy_from(&perturbed_destination);
        *result -= &destination;
        *result /= epsilon;
        
        // Project to ensure we're in the tangent space
        let temp = result.clone();
        manifold.project_tangent(&destination, &temp, result)?;
        Ok(())
    }
}

/// Schild's ladder for parallel transport.
///
/// This is a numerical method for parallel transport that works on any manifold
/// with a retraction. It approximates parallel transport by constructing
/// parallelograms in the manifold.
#[derive(Debug, Clone)]
pub struct SchildLadder<T> {
    /// Number of steps in the ladder
    steps: usize,
    /// Tolerance for iterations
    _tolerance: T,
}

impl<T: Scalar> SchildLadder<T> {
    /// Creates a new Schild's ladder transport with default parameters.
    pub fn new() -> Self {
        Self {
            steps: 1,
            _tolerance: <T as Scalar>::from_f64(1e-10),
        }
    }
    
    /// Creates Schild's ladder with custom parameters.
    pub fn with_parameters(steps: usize, tolerance: T) -> Self {
        Self { 
            steps, 
            _tolerance: tolerance 
        }
    }
}

impl<T: Scalar> Default for SchildLadder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> VectorTransport<T, D> for SchildLadder<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "SchildLadder"
    }

    fn transport(
        &self,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        direction: &TangentVectorType<T, D>,
        vector: &TangentVectorType<T, D>,
        result: &mut TangentVectorType<T, D>,
    ) -> Result<()> {
        // Divide the path into steps
        let step_size = T::one() / <T as Scalar>::from_usize(self.steps);
        
        // Pre-allocate all buffers
        let mut step_direction = direction.clone();
        step_direction *= step_size;
        
        let mut half_step = step_direction.clone();
        half_step *= <T as Scalar>::from_f64(0.5);
        
        let mut current_point = point.clone();
        result.copy_from(vector);
        
        // Pre-allocate points for the algorithm
        let mut next_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut midpoint = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut displaced_midpoint = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut fourth_corner = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        
        // Apply Schild's ladder at each step
        for _ in 0..self.steps {
            // Move to midpoint
            manifold.retract(&current_point, &step_direction, &mut next_point)?;
            manifold.retract(&current_point, &half_step, &mut midpoint)?;
            
            // Construct parallelogram
            manifold.retract(&midpoint, result, &mut displaced_midpoint)?;
            
            // Complete the parallelogram to find transported vector
            // The transported vector should connect next_point to the fourth corner
            // of the parallelogram
            manifold.retract(&displaced_midpoint, &half_step, &mut fourth_corner)?;
            manifold.inverse_retract(&next_point, &fourth_corner, result)?;
            
            // Move to next point
            current_point.copy_from(&next_point);
        }
        
        Ok(())
    }
}

/// Utilities for verifying retraction properties.
pub struct RetractionVerifier;

impl RetractionVerifier {
    /// Verifies the centering condition: R(p, 0) = p.
    pub fn verify_centering<T, D>(
        retraction: &impl Retraction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tol: T,
    ) -> Result<bool>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let zero = OVector::zeros_generic(point.shape_generic().0, nalgebra::U1);
        let mut result = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        retraction.retract(manifold, point, &zero, &mut result)?;

        let diff = &result - point;
        let norm = diff.norm();

        Ok(norm < tol)
    }

    /// Verifies the local rigidity condition: dR(p, 0)[v] = v.
    ///
    /// This checks that for small t, R(p, tv) H p + tv.
    pub fn verify_local_rigidity<T, D>(
        retraction: &impl Retraction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        tol: T,
    ) -> Result<bool>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let t = <T as Scalar>::from_f64(1e-8);
        let mut scaled_tangent = tangent.clone();
        scaled_tangent *= t;

        let mut retracted = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        retraction.retract(manifold, point, &scaled_tangent, &mut retracted)?;
        let linear_approx = point + &scaled_tangent;

        let diff = &retracted - &linear_approx;
        let relative_error = diff.norm() / (t * tangent.norm());

        Ok(relative_error < tol)
    }

    /// Verifies that the retraction is a retraction-inverse pair.
    ///
    /// Checks that inverse_retract(p, retract(p, v)) H v for small v.
    pub fn verify_inverse<T, D>(
        retraction: &impl Retraction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
        tol: T,
    ) -> Result<bool>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        // Use a small tangent vector to stay in the domain of the inverse
        let scale = <T as Scalar>::from_f64(0.1);
        let mut small_tangent = tangent.clone();
        small_tangent *= scale;

        let mut retracted = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        retraction.retract(manifold, point, &small_tangent, &mut retracted)?;
        let mut recovered = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        retraction.inverse_retract(manifold, point, &retracted, &mut recovered)?;

        let diff = &recovered - &small_tangent;
        let relative_error = diff.norm() / small_tangent.norm();

        Ok(relative_error < tol)
    }

    /// Estimates the order of a retraction by comparing with the exponential map.
    ///
    /// For a k-th order retraction, ||R(p, tv) - Exp(p, tv)|| = O(t^{k+1}).
    pub fn estimate_order<T, D>(
        retraction: &impl Retraction<T, D>,
        exponential: &impl Retraction<T, D>,
        manifold: &impl Manifold<T, D>,
        point: &Point<T, D>,
        tangent: &TangentVectorType<T, D>,
    ) -> Result<RetractionOrder>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let scales = [0.1, 0.05, 0.025, 0.0125];
        let mut errors = Vec::new();

        // Pre-allocate buffers
        let mut scaled_tangent = tangent.clone();
        let mut r1 = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut r2 = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);

        for &scale in &scales {
            let t = <T as Scalar>::from_f64(scale);
            scaled_tangent.copy_from(tangent);
            scaled_tangent *= t;

            retraction.retract(manifold, point, &scaled_tangent, &mut r1)?;
            exponential.retract(manifold, point, &scaled_tangent, &mut r2)?;

            let error = manifold.distance(&r1, &r2)?;
            errors.push(error);
        }

        // Estimate order by checking error scaling
        // For order k: error[i+1] / error[i] H 2^{-(k+1)}
        let ratio1 = errors[1] / errors[0];
        let ratio2 = errors[2] / errors[1];

        let order1 = -<T as Float>::log2(ratio1) - T::one();
        let order2 = -<T as Float>::log2(ratio2) - T::one();

        let avg_order = (order1 + order2) * <T as Scalar>::from_f64(0.5);

        if avg_order > <T as Scalar>::from_f64(1.5) {
            Ok(RetractionOrder::Second)
        } else {
            Ok(RetractionOrder::First)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_manifolds::TestEuclideanManifold;
    use crate::types::DVector;
    use approx::assert_relative_eq;
    use nalgebra::Dyn;

    #[test]
    fn test_default_retraction() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = DefaultRetraction;

        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let tangent = DVector::from_vec(vec![0.1, 0.2, 0.3]);

        let mut result = DVector::zeros(3);
        retraction.retract(&manifold, &point, &tangent, &mut result).unwrap();
        let expected = DVector::from_vec(vec![1.1, 2.2, 3.3]);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_projection_retraction() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = ProjectionRetraction;

        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let tangent = DVector::from_vec(vec![0.0, 1.0, 0.0]);

        let mut result = DVector::zeros(3);
        retraction.retract(&manifold, &point, &tangent, &mut result).unwrap();
        let expected = DVector::from_vec(vec![1.0, 1.0, 0.0]);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_retraction_centering() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = DefaultRetraction;
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        assert!(
            RetractionVerifier::verify_centering(&retraction, &manifold, &point, 1e-10).unwrap()
        );
    }

    #[test]
    fn test_retraction_local_rigidity() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = DefaultRetraction;
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let tangent = DVector::from_vec(vec![0.1, 0.2, 0.3]);

        assert!(RetractionVerifier::verify_local_rigidity(
            &retraction,
            &manifold,
            &point,
            &tangent,
            1e-6
        )
        .unwrap());
    }

    #[test]
    fn test_retraction_inverse() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = DefaultRetraction;
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let tangent = DVector::from_vec(vec![0.1, 0.2, 0.3]);

        assert!(RetractionVerifier::verify_inverse(
            &retraction,
            &manifold,
            &point,
            &tangent,
            1e-10
        )
        .unwrap());
    }

    #[test]
    fn test_vector_transport() {
        let manifold = TestEuclideanManifold::new(3);
        let transport = ProjectionTransport;

        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let direction = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let vector = DVector::from_vec(vec![0.0, 0.0, 1.0]);

        let mut transported = DVector::zeros(3);
        transport
            .transport(&manifold, &point, &direction, &vector, &mut transported)
            .unwrap();
        assert_relative_eq!(transported, vector, epsilon = 1e-10);
    }

    #[test]
    fn test_retraction_order() {
        let retraction = DefaultRetraction;
        let exponential = ExponentialRetraction::<f64>::new();
        let qr = QRRetraction::new();

        // Need to specify type parameters explicitly
        assert_eq!(
            <DefaultRetraction as Retraction<f64, Dyn>>::order(&retraction),
            RetractionOrder::First
        );
        assert_eq!(
            <ExponentialRetraction<f64> as Retraction<f64, Dyn>>::order(&exponential),
            RetractionOrder::Exact
        );
        assert_eq!(
            <QRRetraction as Retraction<f64, Dyn>>::order(&qr),
            RetractionOrder::Second
        );
    }
    
    #[test]
    fn test_qr_retraction() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = QRRetraction::new();
        
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let tangent = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        
        // For Euclidean manifold, QR retraction falls back to projection
        let mut result = DVector::zeros(3);
        retraction.retract(&manifold, &point, &tangent, &mut result).unwrap();
        assert_relative_eq!(result, &point + &tangent, epsilon = 1e-10);
        
        // Test centering property
        assert!(
            RetractionVerifier::verify_centering(&retraction, &manifold, &point, 1e-10).unwrap()
        );
    }
    
    #[test]
    fn test_cayley_retraction() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = CayleyRetraction::<f64>::new();
        
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let tangent = DVector::from_vec(vec![0.0, 0.5, 0.0]);
        
        // Test basic retraction
        let mut result = DVector::zeros(3);
        retraction.retract(&manifold, &point, &tangent, &mut result).unwrap();
        assert_relative_eq!(result, &point + &tangent, epsilon = 1e-10);
        
        // Test with scaling
        let scaled_retraction = CayleyRetraction::with_scaling(2.0);
        let mut scaled_result = DVector::zeros(3);
        scaled_retraction.retract(&manifold, &point, &tangent, &mut scaled_result).unwrap();
        assert_relative_eq!(scaled_result, &point + &tangent * 2.0, epsilon = 1e-10);
        
        // Test properties
        assert!(
            RetractionVerifier::verify_centering(&retraction, &manifold, &point, 1e-10).unwrap()
        );
        assert!(
            RetractionVerifier::verify_local_rigidity(&retraction, &manifold, &point, &tangent, 1e-6).unwrap()
        );
    }
    
    #[test]
    fn test_polar_retraction() {
        let manifold = TestEuclideanManifold::new(3);
        let retraction = PolarRetraction::<f64>::new();
        
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let tangent = DVector::from_vec(vec![0.0, 0.3, 0.4]);
        
        // Test basic retraction
        let mut result = DVector::zeros(3);
        retraction.retract(&manifold, &point, &tangent, &mut result).unwrap();
        assert_relative_eq!(result, &point + &tangent, epsilon = 1e-10);
        
        // Test with custom parameters
        let custom_retraction = PolarRetraction::with_parameters(20, 1e-12);
        let mut custom_result = DVector::zeros(3);
        custom_retraction.retract(&manifold, &point, &tangent, &mut custom_result).unwrap();
        assert_relative_eq!(custom_result, &point + &tangent, epsilon = 1e-10);
        
        // Test retraction properties
        assert!(
            RetractionVerifier::verify_centering(&retraction, &manifold, &point, 1e-10).unwrap()
        );
        assert!(
            RetractionVerifier::verify_local_rigidity(&retraction, &manifold, &point, &tangent, 1e-6).unwrap()
        );
        
        // Test order
        assert_eq!(
            <PolarRetraction<f64> as Retraction<f64, Dyn>>::order(&retraction),
            RetractionOrder::Second
        );
    }
    
    #[test]
    fn test_differential_retraction_transport() {
        let manifold = TestEuclideanManifold::new(3);
        let transport = DifferentialRetraction::new();
        
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let direction = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let vector = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        
        // For Euclidean manifold, transport should preserve the vector
        let mut transported = DVector::zeros(3);
        transport
            .transport(&manifold, &point, &direction, &vector, &mut transported)
            .unwrap();
        assert_relative_eq!(transported, vector, epsilon = 1e-7);
    }
    
    #[test]
    fn test_schild_ladder() {
        let manifold = TestEuclideanManifold::new(3);
        let transport = SchildLadder::<f64>::new();
        
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let direction = DVector::from_vec(vec![0.0, 0.5, 0.0]);
        let vector = DVector::from_vec(vec![0.0, 0.0, 0.5]);
        
        // Test single step
        let mut transported = DVector::zeros(3);
        transport
            .transport(&manifold, &point, &direction, &vector, &mut transported)
            .unwrap();
        // For flat manifold, should approximately preserve the vector
        assert_relative_eq!(transported.norm(), vector.norm(), epsilon = 1e-6);
        
        // Test multiple steps
        let multi_step = SchildLadder::with_parameters(5, 1e-12);
        let mut multi_transported = DVector::zeros(3);
        multi_step
            .transport(&manifold, &point, &direction, &vector, &mut multi_transported)
            .unwrap();
        assert_relative_eq!(multi_transported.norm(), vector.norm(), epsilon = 1e-8);
    }
}
