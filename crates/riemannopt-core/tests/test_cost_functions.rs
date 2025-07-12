//! Comprehensive tests for cost functions in RiemannOpt.
//!
//! This module tests various cost function implementations, including:
//! - Basic cost function evaluation
//! - Gradient computation (analytical and finite differences)
//! - Counting functionality
//! - Different types of cost functions on various manifolds

use riemannopt_core::{
    core::{
        cost_function::{CostFunction, CountingCostFunction},
    },
    error::Result,
    types::{Scalar, DVector},
    memory::workspace::Workspace,
};
use approx::assert_relative_eq;
use std::fmt::Debug;
use std::sync::atomic::Ordering;

/// Simple quadratic cost function: f(x) = 0.5 * ||x - target||^2
#[derive(Debug, Clone)]
struct QuadraticCost<T: Scalar> {
    target: DVector<T>,
}

impl<T: Scalar> QuadraticCost<T> {
    fn new(target: DVector<T>) -> Self {
        Self { target }
    }
}

impl<T: Scalar> CostFunction<T> for QuadraticCost<T> {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;
    
    fn cost(&self, point: &Self::Point) -> Result<T> {
        let diff = point - &self.target;
        let cost = <T as Scalar>::from_f64(0.5) * diff.dot(&diff);
        Ok(cost)
    }
    
    fn cost_and_gradient(
        &self,
        point: &Self::Point,
        _workspace: &mut Workspace<T>,
        gradient: &mut Self::TangentVector,
    ) -> Result<T> {
        let diff = point - &self.target;
        let cost = <T as Scalar>::from_f64(0.5) * diff.dot(&diff);
        *gradient = diff;
        Ok(cost)
    }
    
    fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        Ok(point - &self.target)
    }
    
    fn hessian_vector_product(
        &self,
        _point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // For quadratic f(x) = 0.5||x-t||^2, Hessian is identity
        Ok(vector.clone())
    }
    
    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let epsilon = <T as num_traits::Float>::sqrt(<T as num_traits::Float>::epsilon());
        let mut gradient = DVector::zeros(point.len());
        
        for i in 0..point.len() {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            
            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;
            
            let cost_plus = self.cost(&point_plus)?;
            let cost_minus = self.cost(&point_minus)?;
            
            gradient[i] = (cost_plus - cost_minus) / (<T as Scalar>::from_f64(2.0) * epsilon);
        }
        
        Ok(gradient)
    }
}

/// Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
#[derive(Debug, Clone)]
struct RosenbrockCost<T: Scalar> {
    a: T,
    b: T,
}

impl<T: Scalar> RosenbrockCost<T> {
    fn new(a: T, b: T) -> Self {
        Self { a, b }
    }
}

impl<T: Scalar> CostFunction<T> for RosenbrockCost<T> {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;
    
    fn cost(&self, point: &Self::Point) -> Result<T> {
        assert!(point.len() >= 2, "Rosenbrock requires at least 2D");
        let x = point[0];
        let y = point[1];
        
        let term1 = self.a - x;
        let term2 = y - x * x;
        Ok(term1 * term1 + self.b * term2 * term2)
    }
    
    fn cost_and_gradient(
        &self,
        point: &Self::Point,
        _workspace: &mut Workspace<T>,
        gradient: &mut Self::TangentVector,
    ) -> Result<T> {
        assert!(point.len() >= 2, "Rosenbrock requires at least 2D");
        let x = point[0];
        let y = point[1];
        
        let term1 = self.a - x;
        let term2 = y - x * x;
        let cost = term1 * term1 + self.b * term2 * term2;
        
        // Gradient computation
        let two = <T as Scalar>::from_f64(2.0);
        let four = <T as Scalar>::from_f64(4.0);
        
        gradient.fill(T::zero());
        gradient[0] = -two * term1 - four * self.b * x * term2;
        gradient[1] = two * self.b * term2;
        
        Ok(cost)
    }
    
    fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let mut gradient = DVector::zeros(point.len());
        self.cost_and_gradient(point, &mut Workspace::new(), &mut gradient)?;
        Ok(gradient)
    }
    
    fn hessian_vector_product(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        assert!(point.len() >= 2, "Rosenbrock requires at least 2D");
        let x = point[0];
        let y = point[1];
        
        let two = <T as Scalar>::from_f64(2.0);
        let four = <T as Scalar>::from_f64(4.0);
        let twelve = <T as Scalar>::from_f64(12.0);
        
        // Hessian elements for 2D Rosenbrock
        let h11 = two + twelve * self.b * x * x - four * self.b * (y - x * x);
        let h12 = -four * self.b * x;
        let h21 = h12;
        let h22 = two * self.b;
        
        let mut result = DVector::zeros(point.len());
        result[0] = h11 * vector[0] + h12 * vector[1];
        result[1] = h21 * vector[0] + h22 * vector[1];
        
        Ok(result)
    }
    
    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let epsilon = <T as num_traits::Float>::sqrt(<T as num_traits::Float>::epsilon());
        let mut gradient = DVector::zeros(point.len());
        
        for i in 0..point.len() {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            
            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;
            
            let cost_plus = self.cost(&point_plus)?;
            let cost_minus = self.cost(&point_minus)?;
            
            gradient[i] = (cost_plus - cost_minus) / (<T as Scalar>::from_f64(2.0) * epsilon);
        }
        
        Ok(gradient)
    }
}

/// Sphere-constrained least squares: minimize ||Ax - b||^2 subject to ||x||=1
#[derive(Debug, Clone)]
struct SphereLeastSquares<T: Scalar> {
    matrix_a: nalgebra::DMatrix<T>,
    vector_b: DVector<T>,
}

impl<T: Scalar> SphereLeastSquares<T> {
    fn new(matrix_a: nalgebra::DMatrix<T>, vector_b: DVector<T>) -> Self {
        assert_eq!(matrix_a.nrows(), vector_b.len());
        Self { matrix_a, vector_b }
    }
}

impl<T: Scalar> CostFunction<T> for SphereLeastSquares<T> {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;
    
    fn cost(&self, point: &Self::Point) -> Result<T> {
        let residual = &self.matrix_a * point - &self.vector_b;
        let cost = <T as Scalar>::from_f64(0.5) * residual.dot(&residual);
        Ok(cost)
    }
    
    fn cost_and_gradient(
        &self,
        point: &Self::Point,
        _workspace: &mut Workspace<T>,
        gradient: &mut Self::TangentVector,
    ) -> Result<T> {
        let residual = &self.matrix_a * point - &self.vector_b;
        let cost = <T as Scalar>::from_f64(0.5) * residual.dot(&residual);
        
        // Euclidean gradient: A^T(Ax - b)
        *gradient = self.matrix_a.transpose() * residual;
        
        Ok(cost)
    }
    
    fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let residual = &self.matrix_a * point - &self.vector_b;
        Ok(self.matrix_a.transpose() * residual)
    }
    
    fn hessian_vector_product(
        &self,
        _point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // For f(x) = 0.5||Ax-b||^2, Hessian is A^T A
        Ok(self.matrix_a.transpose() * (&self.matrix_a * vector))
    }
    
    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let epsilon = <T as num_traits::Float>::sqrt(<T as num_traits::Float>::epsilon());
        let mut gradient = DVector::zeros(point.len());
        
        for i in 0..point.len() {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            
            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;
            
            let cost_plus = self.cost(&point_plus)?;
            let cost_minus = self.cost(&point_minus)?;
            
            gradient[i] = (cost_plus - cost_minus) / (<T as Scalar>::from_f64(2.0) * epsilon);
        }
        
        Ok(gradient)
    }
}

#[cfg(test)]
mod basic_tests {
    use super::*;
    
    #[test]
    fn test_quadratic_cost() {
        let target = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let cost_fn = QuadraticCost::new(target.clone());
        
        // Test at target (minimum)
        let cost = cost_fn.cost(&target).unwrap();
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
        
        // Test at another point
        let point = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let cost = cost_fn.cost(&point).unwrap();
        assert_relative_eq!(cost, 7.0, epsilon = 1e-10); // 0.5 * (1 + 4 + 9)
        
        // Test gradient
        let mut workspace = Workspace::new();
        let mut gradient = DVector::zeros(3);
        let cost = cost_fn.cost_and_gradient(&point, &mut workspace, &mut gradient).unwrap();
        
        assert_relative_eq!(cost, 7.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[1], -2.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[2], -3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_rosenbrock() {
        let cost_fn = RosenbrockCost::new(1.0_f64, 100.0);
        
        // Test at minimum (1, 1)
        let minimum = DVector::from_vec(vec![1.0, 1.0]);
        let cost = cost_fn.cost(&minimum).unwrap();
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
        
        // Test gradient at minimum
        let mut workspace = Workspace::new();
        let mut gradient = DVector::zeros(2);
        cost_fn.cost_and_gradient(&minimum, &mut workspace, &mut gradient).unwrap();
        
        assert_relative_eq!(gradient.norm(), 0.0, epsilon = 1e-10);
        
        // Test at another point
        let point = DVector::from_vec(vec![0.0, 0.0]);
        let cost = cost_fn.cost(&point).unwrap();
        assert_relative_eq!(cost, 1.0, epsilon = 1e-10); // (1-0)² + 100(0-0²)²
        
        // Test gradient
        cost_fn.cost_and_gradient(&point, &mut workspace, &mut gradient).unwrap();
        assert_relative_eq!(gradient[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[1], 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_sphere_least_squares() {
        // Simple 2x2 system
        let a = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let cost_fn = SphereLeastSquares::new(a, b);
        
        // Test at a point on the sphere
        let point = DVector::from_vec(vec![1.0/2.0_f64.sqrt(), 1.0/2.0_f64.sqrt()]);
        let cost = cost_fn.cost(&point).unwrap();
        
        // Residual = [1/√2 - 1, 1/√2 - 1] = [1/√2 - 1, 1/√2 - 1]
        let expected_cost = 0.5 * 2.0 * (1.0/2.0_f64.sqrt() - 1.0).powi(2);
        assert_relative_eq!(cost, expected_cost, epsilon = 1e-10);
        
        // Test gradient
        let mut workspace = Workspace::new();
        let mut gradient = DVector::zeros(2);
        cost_fn.cost_and_gradient(&point, &mut workspace, &mut gradient).unwrap();
        
        // Gradient should be A^T(Ax - b)
        let expected_grad = DVector::from_vec(vec![
            1.0/2.0_f64.sqrt() - 1.0,
            1.0/2.0_f64.sqrt() - 1.0
        ]);
        assert_relative_eq!(gradient[0], expected_grad[0], epsilon = 1e-10);
        assert_relative_eq!(gradient[1], expected_grad[1], epsilon = 1e-10);
    }
}

#[cfg(test)]
mod finite_difference_tests {
    use super::*;
    
    #[test]
    fn test_finite_difference_gradient() {
        let target = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let quadratic = QuadraticCost::new(target);
        
        let point = DVector::from_vec(vec![0.5, 1.5, 2.5]);
        
        // Get analytical gradient
        let analytical_grad = quadratic.gradient(&point).unwrap();
        
        // Get finite difference gradient
        let fd_grad = quadratic.gradient_fd_alloc(&point).unwrap();
        
        // Compare - finite differences should be close but not exact
        for i in 0..3 {
            assert_relative_eq!(fd_grad[i], analytical_grad[i], epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_gradient_fd_method() {
        let cost_fn = RosenbrockCost::new(1.0_f64, 100.0);
        let point = DVector::from_vec(vec![0.5, 0.25]);
        
        // Get analytical gradient
        let analytical_grad = cost_fn.gradient(&point).unwrap();
        
        // Get finite difference gradient
        let fd_grad = cost_fn.gradient_fd_alloc(&point).unwrap();
        
        // Compare
        assert_relative_eq!(fd_grad[0], analytical_grad[0], epsilon = 1e-6);
        assert_relative_eq!(fd_grad[1], analytical_grad[1], epsilon = 1e-6);
    }
}

#[cfg(test)]
mod counting_tests {
    use super::*;
    
    #[test]
    fn test_counting_cost_function() {
        let target = DVector::from_vec(vec![1.0_f64, 2.0]);
        let base_cost = QuadraticCost::new(target);
        let counting_cost = CountingCostFunction::new(base_cost);
        
        let point = DVector::from_vec(vec![0.0, 0.0]);
        let mut workspace = Workspace::new();
        let mut gradient = DVector::zeros(2);
        
        // Initial counts should be zero
        assert_eq!(counting_cost.cost_count.load(Ordering::Relaxed), 0);
        assert_eq!(counting_cost.gradient_count.load(Ordering::Relaxed), 0);
        
        // Call cost
        let _ = counting_cost.cost(&point).unwrap();
        assert_eq!(counting_cost.cost_count.load(Ordering::Relaxed), 1);
        assert_eq!(counting_cost.gradient_count.load(Ordering::Relaxed), 0);
        
        // Call gradient
        let _ = counting_cost.gradient(&point).unwrap();
        assert_eq!(counting_cost.cost_count.load(Ordering::Relaxed), 1);
        assert_eq!(counting_cost.gradient_count.load(Ordering::Relaxed), 1);
        
        // Call cost_and_gradient
        let _ = counting_cost.cost_and_gradient(&point, &mut workspace, &mut gradient).unwrap();
        // Note: cost_and_gradient increments both cost_count and gradient_count
        assert_eq!(counting_cost.cost_count.load(Ordering::Relaxed), 2);
        assert_eq!(counting_cost.gradient_count.load(Ordering::Relaxed), 2);
        
        // Total evaluations
        let total = counting_cost.cost_count.load(Ordering::Relaxed) + 
                   counting_cost.gradient_count.load(Ordering::Relaxed);
        assert_eq!(total, 4);
        
        // Reset counts
        counting_cost.reset_counts();
        assert_eq!(counting_cost.cost_count.load(Ordering::Relaxed), 0);
        assert_eq!(counting_cost.gradient_count.load(Ordering::Relaxed), 0);
    }
}

#[cfg(test)]
mod hessian_tests {
    use super::*;
    
    #[test]
    fn test_hessian_vector_product() {
        // Test quadratic function
        let target = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let quadratic = QuadraticCost::new(target);
        let point = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let vector = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        
        let hvp = quadratic.hessian_vector_product(&point, &vector).unwrap();
        // For quadratic with identity Hessian, Hv = v
        assert_relative_eq!(hvp[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(hvp[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(hvp[2], 0.0, epsilon = 1e-10);
        
        // Test Rosenbrock
        let rosenbrock = RosenbrockCost::new(1.0_f64, 100.0);
        let point = DVector::from_vec(vec![0.5, 0.25]);
        let vector = DVector::from_vec(vec![1.0, 0.0]);
        
        let hvp = rosenbrock.hessian_vector_product(&point, &vector).unwrap();
        // Verify non-zero result
        assert!(hvp.norm() > 0.0);
    }
}

#[cfg(test)]
mod allocating_interface_tests {
    use super::*;
    
    #[test]
    fn test_allocating_interface() {
        let cost_fn = RosenbrockCost::new(1.0_f64, 100.0);
        let point = DVector::from_vec(vec![0.5_f64, 0.25]);
        
        // Test gradient
        let gradient = cost_fn.gradient(&point).unwrap();
        
        // Verify gradient values
        let x = 0.5;
        let y = 0.25;
        let expected_gx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        let expected_gy = 200.0 * (y - x * x);
        
        assert_relative_eq!(gradient[0], expected_gx, epsilon = 1e-10);
        assert_relative_eq!(gradient[1], expected_gy, epsilon = 1e-10);
        
        // Test cost_and_gradient_alloc
        let (cost, grad) = cost_fn.cost_and_gradient_alloc(&point).unwrap();
        
        let expected_cost = (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);
        assert_relative_eq!(cost, expected_cost, epsilon = 1e-10);
        assert_relative_eq!(grad[0], expected_gx, epsilon = 1e-10);
        assert_relative_eq!(grad[1], expected_gy, epsilon = 1e-10);
    }
}