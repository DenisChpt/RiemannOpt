//! Preconditioning strategies for optimization algorithms.
//!
//! This module provides traits and implementations for preconditioning,
//! which transforms the gradient to improve convergence properties.

use crate::{
    error::Result,
    manifold::Point,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use std::fmt::Debug;

/// Type alias for tangent vectors.
pub type TangentVector<T, D> = OVector<T, D>;

/// Preconditioner trait for gradient transformations.
///
/// A preconditioner transforms the gradient to improve the conditioning
/// of the optimization problem. The preconditioned gradient should have
/// better convergence properties than the original gradient.
pub trait Preconditioner<T, D>: Debug
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Applies the preconditioner to a gradient vector.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient to precondition
    /// * `point` - The current point on the manifold
    ///
    /// # Returns
    ///
    /// The preconditioned gradient vector
    fn apply(
        &self,
        gradient: &TangentVector<T, D>,
        point: &Point<T, D>,
    ) -> Result<TangentVector<T, D>>;
    
    /// Returns the name of this preconditioner.
    fn name(&self) -> &str {
        "Generic Preconditioner"
    }
}

/// Identity preconditioner (no preconditioning).
///
/// This preconditioner returns the gradient unchanged. It's useful
/// as a default when no preconditioning is desired.
#[derive(Debug, Clone, Copy)]
pub struct IdentityPreconditioner;

impl<T, D> Preconditioner<T, D> for IdentityPreconditioner
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn apply(
        &self,
        gradient: &TangentVector<T, D>,
        _point: &Point<T, D>,
    ) -> Result<TangentVector<T, D>> {
        Ok(gradient.clone())
    }
    
    fn name(&self) -> &str {
        "Identity"
    }
}

/// Diagonal preconditioner.
///
/// Scales each component of the gradient by a diagonal matrix.
/// This is efficient and often effective for many problems.
#[derive(Debug, Clone)]
pub struct DiagonalPreconditioner<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Diagonal scaling factors
    diagonal: OVector<T, D>,
}

impl<T, D> DiagonalPreconditioner<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new diagonal preconditioner with the given diagonal.
    pub fn new(diagonal: OVector<T, D>) -> Self {
        Self { diagonal }
    }
    
    /// Creates a diagonal preconditioner with uniform scaling.
    pub fn uniform(dim: D, scale: T) -> Self {
        let diagonal = OVector::from_element_generic(dim, nalgebra::U1, scale);
        Self { diagonal }
    }
}

impl<T, D> Preconditioner<T, D> for DiagonalPreconditioner<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn apply(
        &self,
        gradient: &TangentVector<T, D>,
        _point: &Point<T, D>,
    ) -> Result<TangentVector<T, D>> {
        Ok(gradient.component_mul(&self.diagonal))
    }
    
    fn name(&self) -> &str {
        "Diagonal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVector;
    use nalgebra::Dyn;
    
    #[test]
    fn test_identity_preconditioner() {
        let preconditioner = IdentityPreconditioner;
        let gradient = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let point = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        
        let result = preconditioner.apply(&gradient, &point).unwrap();
        assert_eq!(result, gradient);
        assert_eq!(<IdentityPreconditioner as Preconditioner<f64, Dyn>>::name(&preconditioner), "Identity");
    }
    
    #[test]
    fn test_diagonal_preconditioner() {
        let diagonal = DVector::from_vec(vec![0.5, 2.0, 1.0]);
        let preconditioner = DiagonalPreconditioner::new(diagonal);
        let gradient = DVector::from_vec(vec![4.0, 3.0, 2.0]);
        let point = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        
        let result = preconditioner.apply(&gradient, &point).unwrap();
        let expected = DVector::from_vec(vec![2.0, 6.0, 2.0]);
        assert_eq!(result, expected);
        assert_eq!(<DiagonalPreconditioner<f64, Dyn> as Preconditioner<f64, Dyn>>::name(&preconditioner), "Diagonal");
    }
    
    #[test]
    fn test_uniform_diagonal_preconditioner() {
        let preconditioner = DiagonalPreconditioner::<f64, Dyn>::uniform(Dyn(3), 0.5);
        let gradient = DVector::from_vec(vec![2.0, 4.0, 6.0]);
        let point = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        
        let result = preconditioner.apply(&gradient, &point).unwrap();
        let expected = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(result, expected);
    }
}