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
	linalg::{self, LinAlgBackend, MatrixOps, VectorOps, VectorView},
	types::Scalar,
};
use num_traits::Float;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Trait for cost functions on Riemannian manifolds.
///
/// This is the main trait that optimization algorithms use to evaluate
/// the objective function and its derivatives.
pub trait CostFunction<T: Scalar>: Debug {
	type Point;
	type TangentVector;
	/// Evaluates the cost function at a point.
	///
	/// # Arguments
	///
	/// * `point` - A point on the manifold
	///
	/// # Returns
	///
	/// The cost function value at the point.
	fn cost(&self, point: &Self::Point) -> Result<T>;

	/// Evaluates the cost and Euclidean gradient at a point (allocating version).
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
	fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(T, Self::TangentVector)> {
		// Default: use finite differences
		let cost = self.cost(point)?;
		let gradient = self.gradient_fd_alloc(point)?;
		Ok((cost, gradient))
	}

	/// Evaluates the cost and Euclidean gradient in-place.
	///
	/// This is the primary method that avoids allocations.
	/// Implementations should override this method for optimal performance.
	///
	/// # Arguments
	///
	/// * `point` - A point on the manifold
	/// * `gradient` - Output buffer for the gradient
	///
	/// # Returns
	///
	/// The cost value. The gradient is written to the output buffer.
	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<T> {
		// Default implementation: compute gradient using finite differences
		let cost = self.cost(point)?;
		self.gradient_fd(point, gradient)?;
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
	/// Calls `cost_and_gradient_alloc` and discards the cost value.
	fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		self.cost_and_gradient_alloc(point).map(|(_, grad)| grad)
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
	fn hessian(&self, _point: &Self::Point) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
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
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector>;

	/// Computes the gradient using finite differences (allocating version).
	///
	/// This is a convenience method that allocates memory.
	/// For performance-critical code, use `gradient_fd` instead.
	///
	/// # Arguments
	///
	/// * `point` - A point on the manifold
	///
	/// # Returns
	///
	/// An approximation of the gradient.
	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector>;

	/// Compute gradient using finite differences in-place.
	///
	/// This method writes the result directly into the provided gradient buffer.
	///
	/// # Arguments
	///
	/// * `point` - A point on the manifold
	/// * `gradient` - Output buffer for the gradient
	///
	/// # Errors
	///
	/// Returns an error if the computation fails.
	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		// For generic dimensions, we can't easily use workspace DVector buffers
		// Default implementation: compute gradient using allocations
		let grad = self.gradient_fd_alloc(point)?;
		*gradient = grad;
		Ok(())
	}
}

/// A simple quadratic cost function for testing.
///
/// Computes f(x) = 0.5 * x^T * A * x + b^T * x + c
#[derive(Debug, Clone)]
pub struct QuadraticCost<T>
where
	T: Scalar,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// The quadratic form matrix (should be symmetric)
	pub a: linalg::Mat<T>,
	/// The linear term
	pub b: linalg::Vec<T>,
	/// The constant term
	pub c: T,
}

impl<T> QuadraticCost<T>
where
	T: Scalar,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	/// Creates a new quadratic cost function.
	pub fn new(a: linalg::Mat<T>, b: linalg::Vec<T>, c: T) -> Self {
		Self { a, b, c }
	}

	/// Creates a simple quadratic with identity matrix: f(x) = 0.5 * ||x||^2
	pub fn simple(dim: usize) -> Self {
		Self {
			a: linalg::Mat::<T>::identity(dim),
			b: linalg::Vec::<T>::zeros(dim),
			c: T::zero(),
		}
	}
}

impl<T> CostFunction<T> for QuadraticCost<T>
where
	T: Scalar,
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;

	fn cost(&self, point: &Self::Point) -> Result<T> {
		let ax = self.a.mat_vec(point);
		let quad_term = VectorView::dot(point, &ax) * Scalar::from_f64(0.5);
		let linear_term = VectorView::dot(&self.b, point);
		Ok(quad_term + linear_term + self.c)
	}

	fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(T, Self::TangentVector)> {
		let ax = self.a.mat_vec(point);
		let cost = VectorView::dot(point, &ax) * Scalar::from_f64(0.5)
			+ VectorView::dot(&self.b, point)
			+ self.c;
		let gradient = VectorOps::add(&ax, &self.b);
		Ok((cost, gradient))
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<T> {
		// gradient = A*x + b
		let ax = self.a.mat_vec(point);
		VectorOps::copy_from(gradient, &ax);
		VectorOps::add_assign(gradient, &self.b);
		let cost = VectorView::dot(point, gradient) * Scalar::from_f64(0.5) + self.c;
		Ok(cost)
	}

	fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let ax = self.a.mat_vec(point);
		Ok(VectorOps::add(&ax, &self.b))
	}

	fn hessian(&self, _point: &Self::Point) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		Ok(self.a.clone())
	}

	fn hessian_vector_product(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		Ok(self.a.mat_vec(vector))
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let n = VectorView::len(point);
		let mut gradient = linalg::Vec::<T>::zeros(n);
		let h = T::fd_epsilon();
		let mut perturbed = point.clone();

		for i in 0..n {
			let orig = VectorView::get(&perturbed, i);

			*VectorOps::get_mut(&mut perturbed, i) = orig + h;
			let f_plus = self.cost(&perturbed)?;

			*VectorOps::get_mut(&mut perturbed, i) = orig - h;
			let f_minus = self.cost(&perturbed)?;

			*VectorOps::get_mut(&mut perturbed, i) = orig;
			*VectorOps::get_mut(&mut gradient, i) = (f_plus - f_minus) / (h + h);
		}

		Ok(gradient)
	}
}

/// Wrapper to count function evaluations for testing and debugging.
#[derive(Debug)]
pub struct CountingCostFunction<F, T>
where
	F: CostFunction<T>,
	T: Scalar,
{
	/// The underlying cost function
	pub inner: F,
	/// Number of cost evaluations
	pub cost_count: AtomicUsize,
	/// Number of gradient evaluations
	pub gradient_count: AtomicUsize,
	/// Number of Hessian evaluations
	pub hessian_count: AtomicUsize,
	_phantom: std::marker::PhantomData<T>,
}

impl<F, T> CountingCostFunction<F, T>
where
	F: CostFunction<T>,
	T: Scalar,
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

impl<F, T> CostFunction<T> for CountingCostFunction<F, T>
where
	F: CostFunction<T>,
	T: Scalar,
{
	type Point = F::Point;
	type TangentVector = F::TangentVector;
	fn cost(&self, point: &Self::Point) -> Result<T> {
		self.cost_count.fetch_add(1, Ordering::Relaxed);
		self.inner.cost(point)
	}

	fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(T, Self::TangentVector)> {
		self.cost_count.fetch_add(1, Ordering::Relaxed);
		self.gradient_count.fetch_add(1, Ordering::Relaxed);
		self.inner.cost_and_gradient_alloc(point)
	}

	fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		self.gradient_count.fetch_add(1, Ordering::Relaxed);
		self.inner.gradient(point)
	}

	fn hessian(&self, point: &Self::Point) -> Result<linalg::Mat<T>>
	where
		linalg::DefaultBackend: LinAlgBackend<T>,
	{
		self.hessian_count.fetch_add(1, Ordering::Relaxed);
		self.inner.hessian(point)
	}

	fn hessian_vector_product(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// Note: we don't count this separately as it may use gradient evaluations
		self.inner.hessian_vector_product(point, vector)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		self.gradient_count.fetch_add(1, Ordering::Relaxed);
		self.inner.gradient_fd_alloc(point)
	}

	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		self.gradient_count.fetch_add(1, Ordering::Relaxed);
		self.inner.gradient_fd(point, gradient)
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<T> {
		self.cost_count.fetch_add(1, Ordering::Relaxed);
		self.gradient_count.fetch_add(1, Ordering::Relaxed);
		self.inner.cost_and_gradient(point, gradient)
	}
}

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
	pub fn check_gradient<T, C>(cost_fn: &C, point: &C::Point, tol: T) -> Result<(bool, T)>
	where
		T: Scalar,
		C: CostFunction<T>,
		C::TangentVector: VectorOps<T>,
	{
		let analytical_grad = cost_fn.gradient(point)?;
		let fd_grad = cost_fn.gradient_fd_alloc(point)?;

		let diff = VectorOps::sub(&analytical_grad, &fd_grad);
		let max_error = VectorView::iter(&diff)
			.map(|x| <T as Float>::abs(x))
			.fold(T::zero(), |a, b| <T as Float>::max(a, b));

		Ok((max_error < tol, max_error))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::linalg::{self, MatrixView, VectorView};

	#[test]
	fn test_quadratic_cost() {
		let cost = QuadraticCost::<f64>::simple(3);
		let point = linalg::Vec::<f64>::from_slice(&[1.0, 2.0, 3.0]);

		let value = cost.cost(&point).unwrap();
		assert!((value - 7.0).abs() < 1e-14);

		let gradient = cost.gradient(&point).unwrap();
		for i in 0..3 {
			assert!((VectorView::get(&gradient, i) - VectorView::get(&point, i)).abs() < 1e-14);
		}

		let hessian = cost.hessian(&point).unwrap();
		for i in 0..3 {
			for j in 0..3 {
				let expected = if i == j { 1.0 } else { 0.0 };
				assert!((hessian.get(i, j) - expected).abs() < 1e-14);
			}
		}
	}

	#[test]
	fn test_quadratic_cost_general() {
		let a = linalg::Mat::<f64>::from_fn(2, 2, |i, j| [[2.0, 1.0], [1.0, 2.0]][i][j]);
		let b = linalg::Vec::<f64>::from_slice(&[2.0, 3.0]);
		let c = 5.0;

		let cost = QuadraticCost::new(a, b, c);
		let point = linalg::Vec::<f64>::from_slice(&[1.0, -1.0]);

		let value = cost.cost(&point).unwrap();
		assert!((value - 5.0).abs() < 1e-14);

		let gradient = cost.gradient(&point).unwrap();
		let g0: f64 = VectorView::get(&gradient, 0);
		let g1: f64 = VectorView::get(&gradient, 1);
		assert!((g0 - 3.0).abs() < 1e-14);
		assert!((g1 - 2.0).abs() < 1e-14);
	}

	#[test]
	fn test_cost_and_gradient_alloc() {
		let cost = QuadraticCost::<f64>::simple(3);
		let point = linalg::Vec::<f64>::from_slice(&[1.0, 2.0, 3.0]);

		let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
		assert!((value - 7.0).abs() < 1e-14);
		for i in 0..3 {
			assert!((VectorView::get(&gradient, i) - VectorView::get(&point, i)).abs() < 1e-14);
		}
	}

	#[test]
	fn test_hessian_vector_product() {
		let cost = QuadraticCost::<f64>::simple(3);
		let point = linalg::Vec::<f64>::from_slice(&[1.0, 2.0, 3.0]);
		let vector = linalg::Vec::<f64>::from_slice(&[0.1, 0.2, 0.3]);

		let hv = cost.hessian_vector_product(&point, &vector).unwrap();
		for i in 0..3 {
			assert!((VectorView::get(&hv, i) - VectorView::get(&vector, i)).abs() < 1e-14);
		}
	}

	#[test]
	fn test_finite_difference_gradient() {
		let cost = QuadraticCost::<f64>::simple(2);
		let point = linalg::Vec::<f64>::from_slice(&[1.0, 2.0]);

		let fd_grad = cost.gradient_fd_alloc(&point).unwrap();
		assert!((VectorView::get(&fd_grad, 0) - 1.0).abs() < 1e-6);
		assert!((VectorView::get(&fd_grad, 1) - 2.0).abs() < 1e-6);
	}

	#[test]
	fn test_counting_cost_function() {
		let inner = QuadraticCost::<f64>::simple(2);
		let cost = CountingCostFunction::new(inner);
		let point = linalg::Vec::<f64>::from_slice(&[1.0, 1.0]);

		assert_eq!(cost.counts(), (0, 0, 0));
		let _ = cost.cost(&point).unwrap();
		assert_eq!(cost.counts(), (1, 0, 0));
		let _ = cost.gradient(&point).unwrap();
		assert_eq!(cost.counts(), (1, 1, 0));
		let _ = cost.cost_and_gradient_alloc(&point).unwrap();
		assert_eq!(cost.counts(), (2, 2, 0));
		let _ = cost.hessian(&point).unwrap();
		assert_eq!(cost.counts(), (2, 2, 1));
		cost.reset_counts();
		assert_eq!(cost.counts(), (0, 0, 0));
	}

	#[test]
	fn test_derivative_checker_gradient() {
		let cost = QuadraticCost::<f64>::simple(3);
		let point = linalg::Vec::<f64>::from_slice(&[1.0, 2.0, 3.0]);

		let (passes, error) = DerivativeChecker::check_gradient(&cost, &point, 1e-6).unwrap();
		assert!(passes);
		assert!(error < 1e-10);
	}
}
