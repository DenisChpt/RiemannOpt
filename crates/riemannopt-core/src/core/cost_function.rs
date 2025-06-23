//! Cost function interface for optimization algorithms.
//!
//! This module provides traits and utilities for defining cost functions
//! on Riemannian manifolds. It supports various evaluation modes including
//! value-only, value with gradient, and second-order information via Hessian.
//!
//! # Design Philosophy
//!
//! The cost function interface is designed to be flexible and efficient:
//! - Support for different evaluation modes to avoid redundant computations
//! - Automatic finite difference approximations when derivatives aren't available
//! - Type-safe handling of manifold constraints
//! - Support for both in-place and allocating APIs

use crate::{
    error::{ManifoldError, Result},
    manifold::{Point, TangentVector},
    types::Scalar,
    memory::workspace::Workspace,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector};
use num_traits::Float;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Trait for cost functions on Riemannian manifolds.
///
/// This is the main trait that optimization algorithms use to evaluate
/// the objective function and its derivatives.
pub trait CostFunction<T, D>: Debug
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Evaluates the cost function at a point.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    ///
    /// # Returns
    ///
    /// The cost function value at the point.
    fn cost(&self, point: &Point<T, D>) -> Result<T>;

    /// Evaluates the cost and Euclidean gradient at a point.
    ///
    /// The gradient returned is in the ambient space (Euclidean gradient).
    /// To get the Riemannian gradient, use the manifold's
    /// `euclidean_to_riemannian_gradient` method.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    ///
    /// # Returns
    ///
    /// A tuple of (cost, euclidean_gradient).
    ///
    /// # Default Implementation
    ///
    /// Uses finite differences to approximate the gradient if not overridden.
    fn cost_and_gradient(&self, point: &Point<T, D>) -> Result<(T, TangentVector<T, D>)> {
        // Default: use finite differences
        let cost = self.cost(point)?;
        let gradient = self.gradient_fd(point)?;
        Ok((cost, gradient))
    }
    
    /// Evaluates the cost and Euclidean gradient using workspace.
    ///
    /// This is the primary method that avoids allocations.
    /// Implementations should override this method for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `workspace` - Pre-allocated workspace
    /// * `gradient` - Output buffer for the gradient
    ///
    /// # Returns
    ///
    /// The cost value. The gradient is written to the output buffer.
    fn cost_and_gradient_with_workspace(
        &self,
        point: &Point<T, D>,
        workspace: &mut Workspace<T>,
        gradient: &mut TangentVector<T, D>,
    ) -> Result<T> {
        // Default implementation: compute gradient using finite differences
        let cost = self.cost(point)?;
        self.gradient_fd_with_workspace(point, workspace, gradient)?;
        Ok(cost)
    }

    /// Computes only the Euclidean gradient at a point.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    ///
    /// # Returns
    ///
    /// The Euclidean gradient at the point.
    ///
    /// # Default Implementation
    ///
    /// Calls `cost_and_gradient` and discards the cost value.
    fn gradient(&self, point: &Point<T, D>) -> Result<TangentVector<T, D>> {
        self.cost_and_gradient(point).map(|(_, grad)| grad)
    }

    /// Evaluates the Hessian matrix at a point.
    ///
    /// The Hessian is the matrix of second partial derivatives in the
    /// ambient space. For Riemannian optimization, this typically needs
    /// to be projected to get the Riemannian Hessian.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    ///
    /// # Returns
    ///
    /// The Hessian matrix at the point.
    ///
    /// # Default Implementation
    ///
    /// Returns `NotImplemented` error. Override for second-order methods.
    fn hessian(&self, _point: &Point<T, D>) -> Result<OMatrix<T, D, D>>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        Err(ManifoldError::not_implemented(
            "Hessian computation not implemented for this cost function",
        ))
    }

    /// Computes a Hessian-vector product.
    ///
    /// This computes H*v where H is the Hessian at the point and v is
    /// a tangent vector. This can often be computed more efficiently
    /// than forming the full Hessian matrix.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `vector` - A tangent vector
    ///
    /// # Returns
    ///
    /// The product H*v.
    ///
    /// # Default Implementation
    ///
    /// Uses finite differences on the gradient.
    fn hessian_vector_product(
        &self,
        point: &Point<T, D>,
        vector: &TangentVector<T, D>,
    ) -> Result<TangentVector<T, D>> {
        // Use finite differences on the gradient
        let eps = <T as Float>::sqrt(T::epsilon());
        let norm = vector.norm();

        if norm < T::epsilon() {
            return Ok(TangentVector::zeros_generic(
                point.shape_generic().0,
                nalgebra::U1,
            ));
        }

        let t = eps / norm;
        let perturbed = point + vector * t;

        let grad1 = self.gradient(point)?;
        let grad2 = self.gradient(&perturbed)?;

        Ok((grad2 - grad1) / t)
    }

    /// Computes the gradient using finite differences.
    ///
    /// This is a convenience method that allocates memory.
    /// For performance-critical code, use `gradient_fd_with_workspace` instead.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    ///
    /// # Returns
    ///
    /// An approximation of the gradient.
    fn gradient_fd(&self, point: &Point<T, D>) -> Result<TangentVector<T, D>> {
        let n = point.len();
        let mut gradient = TangentVector::zeros_generic(point.shape_generic().0, nalgebra::U1);
        let h = <T as Float>::sqrt(T::epsilon());

        // Use iterator with mutable access to avoid direct indexing
        for i in 0..n {
            let mut e_i = TangentVector::zeros_generic(point.shape_generic().0, nalgebra::U1);
            // Safe indexing with bounds check
            if let Some(elem) = e_i.get_mut(i) {
                *elem = T::one();
            } else {
                return Err(ManifoldError::invalid_parameter(
                    format!("Index {} out of bounds for dimension {}", i, n),
                ));
            }

            // Central difference
            let point_plus = point + &e_i * h;
            let point_minus = point - &e_i * h;

            let f_plus = self.cost(&point_plus)?;
            let f_minus = self.cost(&point_minus)?;

            // Safe indexing for gradient
            if let Some(grad_elem) = gradient.get_mut(i) {
                *grad_elem = (f_plus - f_minus) / (h + h);
            } else {
                return Err(ManifoldError::invalid_parameter(
                    format!("Index {} out of bounds for gradient", i),
                ));
            }
        }

        Ok(gradient)
    }
    
    /// Compute gradient using finite differences with pre-allocated workspace.
    ///
    /// This method avoids allocations by using pre-allocated buffers from the workspace.
    /// It's especially beneficial when computing gradients repeatedly during optimization.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the manifold
    /// * `workspace` - Pre-allocated workspace containing necessary buffers
    /// * `gradient` - Output buffer for the gradient
    ///
    /// # Errors
    ///
    /// Returns an error if the computation fails.
    fn gradient_fd_with_workspace(
        &self,
        point: &Point<T, D>,
        _workspace: &mut Workspace<T>,
        gradient: &mut TangentVector<T, D>,
    ) -> Result<()> {
        // Default implementation: For now, just compute gradient normally
        // Specific implementations can override this to use workspace effectively
        let grad = self.gradient_fd(point)?;
        gradient.copy_from(&grad);
        Ok(())
    }
    
    /// Compute gradient using finite differences with parallel execution.
    ///
    /// This method evaluates the gradient in parallel when the dimension is large enough.
    /// It uses the same central difference formula as `gradient_fd` but distributes
    /// the work across multiple threads.
    ///
    /// # Arguments
    ///
    /// * `point` - The point at which to compute the gradient
    /// * `config` - Parallel execution configuration
    ///
    /// # Returns
    ///
    /// An approximation of the gradient computed in parallel.
    fn gradient_fd_parallel(
        &self,
        point: &Point<T, D>,
        _config: &crate::compute::cpu::parallel::ParallelConfig,
    ) -> Result<TangentVector<T, D>> 
    where 
        Self: Sync,
    {
        // Default implementation: use sequential version
        // For parallel execution with dynamic vectors, implementations should use
        // the gradient_fd_parallel_dvec function from cost_function_simd module
        self.gradient_fd(point)
    }
}

/// A simple quadratic cost function for testing.
///
/// Computes f(x) = 0.5 * x^T * A * x + b^T * x + c
#[derive(Debug, Clone)]
pub struct QuadraticCost<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// The quadratic form matrix (should be symmetric)
    pub a: OMatrix<T, D, D>,
    /// The linear term
    pub b: OVector<T, D>,
    /// The constant term
    pub c: T,
}

impl<T, D> QuadraticCost<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Creates a new quadratic cost function.
    pub fn new(a: OMatrix<T, D, D>, b: OVector<T, D>, c: T) -> Self {
        Self { a, b, c }
    }

    /// Creates a simple quadratic with identity matrix: f(x) = 0.5 * ||x||^2
    pub fn simple(dim: D) -> Self {
        Self {
            a: OMatrix::identity_generic(dim, dim),
            b: OVector::zeros_generic(dim, nalgebra::U1),
            c: T::zero(),
        }
    }
}

impl<T, D> CostFunction<T, D> for QuadraticCost<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    fn cost(&self, point: &Point<T, D>) -> Result<T> {
        let ax = &self.a * point;
        let quad_term = point.dot(&ax) * <T as Scalar>::from_f64(0.5);
        let linear_term = self.b.dot(point);
        Ok(quad_term + linear_term + self.c)
    }

    fn cost_and_gradient(&self, point: &Point<T, D>) -> Result<(T, TangentVector<T, D>)> {
        let ax = &self.a * point;
        let cost = point.dot(&ax) * <T as Scalar>::from_f64(0.5) + self.b.dot(point) + self.c;
        let gradient = ax + &self.b;
        Ok((cost, gradient))
    }
    
    fn cost_and_gradient_with_workspace(
        &self,
        point: &Point<T, D>,
        _workspace: &mut Workspace<T>,
        gradient: &mut TangentVector<T, D>,
    ) -> Result<T> {
        // For QuadraticCost, we don't need workspace buffers
        // Compute Ax directly into gradient
        gradient.copy_from(&self.b);
        gradient.gemv(T::one(), &self.a, point, T::one());
        
        let cost = point.dot(gradient) * <T as Scalar>::from_f64(0.5) + self.c;
        Ok(cost)
    }

    fn gradient(&self, point: &Point<T, D>) -> Result<TangentVector<T, D>> {
        Ok(&self.a * point + &self.b)
    }

    fn hessian(&self, _point: &Point<T, D>) -> Result<OMatrix<T, D, D>>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        Ok(self.a.clone())
    }

    fn hessian_vector_product(
        &self,
        _point: &Point<T, D>,
        vector: &TangentVector<T, D>,
    ) -> Result<TangentVector<T, D>> {
        Ok(&self.a * vector)
    }
}

/// Wrapper to count function evaluations for testing and debugging.
#[derive(Debug)]
pub struct CountingCostFunction<F, T, D>
where
    F: CostFunction<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// The underlying cost function
    pub inner: F,
    /// Number of cost evaluations
    pub cost_count: AtomicUsize,
    /// Number of gradient evaluations
    pub gradient_count: AtomicUsize,
    /// Number of Hessian evaluations
    pub hessian_count: AtomicUsize,
    _phantom: std::marker::PhantomData<(T, D)>,
}

impl<F, T, D> CountingCostFunction<F, T, D>
where
    F: CostFunction<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new counting wrapper around a cost function.
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            cost_count: AtomicUsize::new(0),
            gradient_count: AtomicUsize::new(0),
            hessian_count: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Resets all counters to zero.
    pub fn reset_counts(&self) {
        self.cost_count.store(0, Ordering::Relaxed);
        self.gradient_count.store(0, Ordering::Relaxed);
        self.hessian_count.store(0, Ordering::Relaxed);
    }

    /// Returns the current evaluation counts.
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.cost_count.load(Ordering::Relaxed),
            self.gradient_count.load(Ordering::Relaxed),
            self.hessian_count.load(Ordering::Relaxed),
        )
    }
}

impl<F, T, D> CostFunction<T, D> for CountingCostFunction<F, T, D>
where
    F: CostFunction<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn cost(&self, point: &Point<T, D>) -> Result<T> {
        self.cost_count.fetch_add(1, Ordering::Relaxed);
        self.inner.cost(point)
    }

    fn cost_and_gradient(&self, point: &Point<T, D>) -> Result<(T, TangentVector<T, D>)> {
        self.cost_count.fetch_add(1, Ordering::Relaxed);
        self.gradient_count.fetch_add(1, Ordering::Relaxed);
        self.inner.cost_and_gradient(point)
    }

    fn gradient(&self, point: &Point<T, D>) -> Result<TangentVector<T, D>> {
        self.gradient_count.fetch_add(1, Ordering::Relaxed);
        self.inner.gradient(point)
    }

    fn hessian(&self, point: &Point<T, D>) -> Result<OMatrix<T, D, D>>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        self.hessian_count.fetch_add(1, Ordering::Relaxed);
        self.inner.hessian(point)
    }

    fn hessian_vector_product(
        &self,
        point: &Point<T, D>,
        vector: &TangentVector<T, D>,
    ) -> Result<TangentVector<T, D>> {
        // Note: we don't count this separately as it may use gradient evaluations
        self.inner.hessian_vector_product(point, vector)
    }
}

/// Extension trait for parallel gradient computation on dynamic vectors.
pub trait CostFunctionParallel<T: Scalar>: CostFunction<T, nalgebra::Dyn> {
    /// Compute gradient in parallel for dynamic vectors.
    fn gradient_fd_parallel_dvec(
        &self,
        point: &nalgebra::DVector<T>,
        config: &crate::compute::cpu::parallel::ParallelConfig,
    ) -> Result<nalgebra::DVector<T>>
    where
        Self: Sync,
    {
        use super::cost_function_simd::gradient_fd_simd_parallel;
        
        let cost_fn = |p: &nalgebra::DVector<T>| -> Result<T> {
            self.cost(p)
        };
        
        gradient_fd_simd_parallel(&cost_fn, point, config)
    }
}

/// Helper function to compute gradient using finite differences with workspace for dynamic vectors.
///
/// This function provides an efficient implementation for dynamic-dimensional problems
/// by reusing pre-allocated buffers from the workspace.
pub fn gradient_fd_dvec_with_workspace<T, F>(
    cost_fn: &F,
    point: &nalgebra::DVector<T>,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar + Float,
    F: Fn(&nalgebra::DVector<T>) -> Result<T>,
{
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Get pre-allocated buffers from workspace
    let (gradient, e_i, point_plus, point_minus) = workspace.get_gradient_buffers_mut()
        .ok_or_else(|| ManifoldError::invalid_parameter(
            "Workspace missing required gradient buffers".to_string()
        ))?;
        
    // Verify dimensions
    if gradient.len() != n || e_i.len() != n || point_plus.len() != n || point_minus.len() != n {
        return Err(ManifoldError::invalid_parameter(
            format!("Workspace buffers have incorrect dimensions for point of size {}", n),
        ));
    }
    
    // Clear gradient
    gradient.fill(T::zero());
    
    for i in 0..n {
        // Clear and set unit vector
        e_i.fill(T::zero());
        e_i[i] = T::one();
        
        // Compute point + h*e_i and point - h*e_i
        point_plus.copy_from(point);
        point_minus.copy_from(point);
        point_plus[i] += h;
        point_minus[i] -= h;
        
        // Central difference
        let f_plus = cost_fn(point_plus)?;
        let f_minus = cost_fn(point_minus)?;
        
        // Update gradient
        gradient[i] = (f_plus - f_minus) / (h + h);
    }
    
    Ok(())
}

// Blanket implementation for all cost functions with dynamic dimension
impl<T: Scalar + 'static, F: CostFunction<T, nalgebra::Dyn>> CostFunctionParallel<T> for F {}

/// Utilities for checking gradient and Hessian implementations.
pub struct DerivativeChecker;

impl DerivativeChecker {
    /// Checks if the gradient implementation matches finite differences.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to check
    /// * `point` - Point at which to check the gradient
    /// * `tol` - Tolerance for the check
    ///
    /// # Returns
    ///
    /// A tuple of (passes, max_error) where passes indicates if the
    /// gradient is correct within tolerance, and max_error is the
    /// maximum component-wise error.
    pub fn check_gradient<T, D>(
        cost_fn: &impl CostFunction<T, D>,
        point: &Point<T, D>,
        tol: T,
    ) -> Result<(bool, T)>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let analytical_grad = cost_fn.gradient(point)?;
        let fd_grad = cost_fn.gradient_fd(point)?;

        let diff = &analytical_grad - &fd_grad;
        let max_error = diff
            .iter()
            .map(|x| <T as Float>::abs(*x))
            .fold(T::zero(), |a, b| <T as Float>::max(a, b));

        Ok((max_error < tol, max_error))
    }

    /// Checks if the Hessian implementation matches finite differences.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to check
    /// * `point` - Point at which to check the Hessian
    /// * `tol` - Tolerance for the check
    ///
    /// # Returns
    ///
    /// A tuple of (passes, max_error).
    pub fn check_hessian<T, D>(
        cost_fn: &impl CostFunction<T, D>,
        point: &Point<T, D>,
        tol: T,
    ) -> Result<(bool, T)>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        let hessian = cost_fn.hessian(point)?;
        let n = point.len();
        let h = <T as Float>::sqrt(T::epsilon());

        let mut max_error = T::zero();

        // Check Hessian using finite differences on the gradient
        for i in 0..n {
            let mut e_i = OVector::zeros_generic(point.shape_generic().0, nalgebra::U1);
            e_i[i] = T::one();

            let point_plus = point + &e_i * h;
            let point_minus = point - &e_i * h;

            let grad_plus = cost_fn.gradient(&point_plus)?;
            let grad_minus = cost_fn.gradient(&point_minus)?;

            let hessian_col_fd = (grad_plus - grad_minus) / (h + h);

            for j in 0..n {
                let error = <T as Float>::abs(hessian[(j, i)] - hessian_col_fd[j]);
                max_error = <T as Float>::max(max_error, error);
            }
        }

        Ok((max_error < tol, max_error))
    }

    /// Checks if the Hessian is symmetric.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to check
    /// * `point` - Point at which to check the Hessian
    /// * `tol` - Tolerance for symmetry check
    ///
    /// # Returns
    ///
    /// A tuple of (is_symmetric, max_asymmetry).
    pub fn check_hessian_symmetry<T, D>(
        cost_fn: &impl CostFunction<T, D>,
        point: &Point<T, D>,
        tol: T,
    ) -> Result<(bool, T)>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        let hessian = cost_fn.hessian(point)?;
        let n = hessian.nrows();

        let mut max_asymmetry = T::zero();

        for i in 0..n {
            for j in i + 1..n {
                let asymmetry = <T as Float>::abs(hessian[(i, j)] - hessian[(j, i)]);
                max_asymmetry = <T as Float>::max(max_asymmetry, asymmetry);
            }
        }

        Ok((max_asymmetry < tol, max_asymmetry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVector;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, Dyn};

    #[test]
    fn test_quadratic_cost() {
        // f(x) = 0.5 * x^T * x = 0.5 * ||x||^2
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        // Cost should be 0.5 * (1 + 4 + 9) = 7
        let value = cost.cost(&point).unwrap();
        assert_relative_eq!(value, 7.0);

        // Gradient should be x
        let gradient = cost.gradient(&point).unwrap();
        assert_relative_eq!(gradient, point);

        // Hessian should be identity
        let hessian = cost.hessian(&point).unwrap();
        assert_eq!(hessian, DMatrix::identity(3, 3));
    }

    #[test]
    fn test_quadratic_cost_general() {
        // f(x) = x1^2 + x2^2 + x1*x2 + 2*x1 + 3*x2 + 5
        let mut a = DMatrix::zeros(2, 2);
        a[(0, 0)] = 2.0; // d²f/dx1² = 2
        a[(1, 1)] = 2.0; // d²f/dx2² = 2
        a[(0, 1)] = 1.0; // d²f/dx1dx2 = 1
        a[(1, 0)] = 1.0; // Symmetric

        let b = DVector::from_vec(vec![2.0, 3.0]);
        let c = 5.0;

        let cost = QuadraticCost::new(a.clone(), b.clone(), c);
        let point = DVector::from_vec(vec![1.0, -1.0]);

        // f(1, -1) = 1 + 1 - 1 + 2 - 3 + 5 = 5
        let value = cost.cost(&point).unwrap();
        assert_relative_eq!(value, 5.0);

        // grad f = [2*x1 + x2 + 2, 2*x2 + x1 + 3] = [2 - 1 + 2, -2 + 1 + 3] = [3, 2]
        let gradient = cost.gradient(&point).unwrap();
        assert_relative_eq!(gradient[0], 3.0);
        assert_relative_eq!(gradient[1], 2.0);
    }

    #[test]
    fn test_cost_and_gradient() {
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let (value, gradient) = cost.cost_and_gradient(&point).unwrap();
        assert_relative_eq!(value, 7.0);
        assert_relative_eq!(gradient, point);
    }

    #[test]
    fn test_hessian_vector_product() {
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let vector = DVector::from_vec(vec![0.1, 0.2, 0.3]);

        // For identity Hessian, Hv = v
        let hv = cost.hessian_vector_product(&point, &vector).unwrap();
        assert_relative_eq!(hv, vector);
    }

    #[test]
    fn test_finite_difference_gradient() {
        // Test on a simple function: f(x) = x1^2 + 2*x2^2
        struct SimpleCost;

        impl CostFunction<f64, Dyn> for SimpleCost {
            fn cost(&self, point: &DVector<f64>) -> Result<f64> {
                Ok(point[0] * point[0] + 2.0 * point[1] * point[1])
            }
        }

        impl Debug for SimpleCost {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "SimpleCost")
            }
        }

        let cost = SimpleCost;
        let point = DVector::from_vec(vec![1.0, 2.0]);

        let fd_grad = cost.gradient_fd(&point).unwrap();
        // Analytical gradient: [2*x1, 4*x2] = [2, 8]
        assert_relative_eq!(fd_grad[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(fd_grad[1], 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_counting_cost_function() {
        let inner = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cost = CountingCostFunction::new(inner);
        let point = DVector::from_vec(vec![1.0, 1.0]);

        // Initial counts should be zero
        assert_eq!(cost.counts(), (0, 0, 0));

        // Evaluate cost
        let _ = cost.cost(&point).unwrap();
        assert_eq!(cost.counts(), (1, 0, 0));

        // Evaluate gradient
        let _ = cost.gradient(&point).unwrap();
        assert_eq!(cost.counts(), (1, 1, 0));

        // Evaluate cost and gradient
        let _ = cost.cost_and_gradient(&point).unwrap();
        assert_eq!(cost.counts(), (2, 2, 0));

        // Evaluate Hessian
        let _ = cost.hessian(&point).unwrap();
        assert_eq!(cost.counts(), (2, 2, 1));

        // Reset counts
        cost.reset_counts();
        assert_eq!(cost.counts(), (0, 0, 0));
    }

    #[test]
    fn test_derivative_checker_gradient() {
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let (passes, error) = DerivativeChecker::check_gradient(&cost, &point, 1e-6).unwrap();
        assert!(passes);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_derivative_checker_hessian() {
        let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let point = DVector::from_vec(vec![1.0, 2.0]);

        let (passes, error) = DerivativeChecker::check_hessian(&cost, &point, 1e-6).unwrap();
        assert!(passes);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_derivative_checker_symmetry() {
        // Create an asymmetric "Hessian" to test the checker
        let mut a = DMatrix::zeros(2, 2);
        a[(0, 0)] = 1.0;
        a[(1, 1)] = 1.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 2.0; // Symmetric

        let cost = QuadraticCost::new(a, DVector::zeros(2), 0.0);
        let point = DVector::from_vec(vec![1.0, 1.0]);

        let (is_symmetric, asymmetry) =
            DerivativeChecker::check_hessian_symmetry(&cost, &point, 1e-10).unwrap();
        assert!(is_symmetric);
        assert!(asymmetry < 1e-10);
    }
    
    #[test]
    fn test_gradient_fd_parallel() {
        use crate::compute::cpu::parallel::ParallelConfig;
        
        let n = 200; // Large enough to trigger parallel execution
        let a = nalgebra::DMatrix::<f64>::identity(n, n) * 2.0;
        let b = nalgebra::DVector::zeros(n);
        let c = 0.0;
        
        let quadratic = QuadraticCost::new(a.clone(), b.clone(), c);
        let x = nalgebra::DVector::from_element(n, 1.0);
        
        // Sequential gradient
        let grad_seq = quadratic.gradient(&x).unwrap();
        
        // Parallel gradient using the extension trait
        let config = ParallelConfig::default();
        let grad_par = quadratic.gradient_fd_parallel_dvec(&x, &config).unwrap();
        
        // Compare results
        for i in 0..n {
            assert_relative_eq!(grad_par[i], grad_seq[i], epsilon = 1e-4);
        }
    }
    
    #[test]
    fn test_parallel_config() {
        use crate::compute::cpu::parallel::ParallelConfig;
        
        let config = ParallelConfig::new()
            .with_min_dimension(50)
            .with_num_threads(4)
            .with_chunk_size(10);
        
        assert!(config.should_parallelize(100));
        assert!(!config.should_parallelize(30));
        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.chunk_size, Some(10));
    }
    
    #[test]
    fn test_gradient_fd_parallel_small_dimension() {
        use crate::compute::cpu::parallel::ParallelConfig;
        
        // Test that small dimensions fall back to sequential
        let n = 5;
        let a = nalgebra::DMatrix::<f64>::identity(n, n);
        let b = nalgebra::DVector::zeros(n);
        let c = 1.0;
        
        let quadratic = QuadraticCost::new(a, b, c);
        let x = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let config = ParallelConfig::default();
        let grad_par = quadratic.gradient_fd_parallel_dvec(&x, &config).unwrap();
        let grad_seq = quadratic.gradient_fd(&x).unwrap();
        
        // Should be identical since it falls back to sequential
        for i in 0..n {
            assert_relative_eq!(grad_par[i], grad_seq[i], epsilon = 1e-10);
        }
    }
}
