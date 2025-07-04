//! Integration tests for the SGD optimizer

use riemannopt_core::{
    core::cost_function::CostFunction,
    error::Result,
    memory::workspace::Workspace,
    optimization::{
        optimizer::{Optimizer, StoppingCriterion},
    },
    types::{DVector, Scalar},
};
use riemannopt_manifolds::Sphere;
use riemannopt_optim::{SGD, SGDConfig};

/// Simple quadratic cost function on the sphere: f(x) = x^T A x
#[derive(Debug)]
struct QuadraticOnSphere<T: Scalar> {
    matrix: DVector<T>,
}

impl<T: Scalar> QuadraticOnSphere<T> {
    fn new(dim: usize) -> Self {
        // Create a diagonal matrix with eigenvalues 1, 2, ..., dim
        let mut matrix = DVector::zeros(dim);
        for i in 0..dim {
            matrix[i] = <T as Scalar>::from_f64((i + 1) as f64);
        }
        Self { matrix }
    }
}

impl<T: Scalar> CostFunction<T> for QuadraticOnSphere<T> {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;

    fn cost(&self, x: &Self::Point) -> Result<T> {
        // f(x) = sum_i matrix[i] * x[i]^2
        let mut cost = T::zero();
        for i in 0..x.len() {
            cost = cost + self.matrix[i] * x[i] * x[i];
        }
        Ok(cost)
    }

    fn cost_and_gradient(
        &self,
        x: &Self::Point,
        _workspace: &mut Workspace<T>,
        gradient: &mut Self::TangentVector,
    ) -> Result<T> {
        // grad f(x) = 2 * diag(matrix) * x
        let mut cost = T::zero();
        for i in 0..x.len() {
            cost = cost + self.matrix[i] * x[i] * x[i];
            gradient[i] = <T as Scalar>::from_f64(2.0) * self.matrix[i] * x[i];
        }
        Ok(cost)
    }
    
    fn hessian_vector_product(
        &self,
        _x: &Self::Point,
        v: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // Hv = 2 * diag(matrix) * v
        let mut result = v.clone();
        for i in 0..v.len() {
            result[i] = <T as Scalar>::from_f64(2.0) * self.matrix[i] * v[i];
        }
        Ok(result)
    }
    
    fn gradient_fd_alloc(&self, x: &Self::Point) -> Result<Self::TangentVector> {
        // Fallback to gradient computation
        let mut gradient = DVector::zeros(x.len());
        self.cost_and_gradient(x, &mut Workspace::new(), &mut gradient)?;
        Ok(gradient)
    }
}

#[test]
fn test_sgd_on_sphere() -> Result<()> {
    let dim = 10;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point on sphere
    let mut initial_point = DVector::zeros(dim);
    initial_point[0] = 1.0; // Standard basis vector
    
    // Create optimizer
    let mut optimizer = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
    
    // Set stopping criterion
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(100)
        .with_gradient_tolerance(1e-6);
    
    // Optimize
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    // The minimum on the sphere for this quadratic form should be at the eigenvector
    // corresponding to the smallest eigenvalue, which is the first coordinate (eigenvalue = 1)
    println!("Final point: {:?}", result.point);
    println!("Final value: {}", result.value);
    println!("Final gradient norm: {:?}", result.gradient_norm);
    println!("Iterations: {}", result.iterations);
    
    assert!(result.point[0].abs() > 0.9, "Solution should be close to first basis vector (smallest eigenvalue)");
    assert!(result.value < 1.1, "Minimum value should be close to 1");
    assert!(result.gradient_norm.unwrap_or(1.0) < 1e-4, "Gradient norm should be small at convergence");
    
    Ok(())
}

#[test]
fn test_sgd_with_momentum() -> Result<()> {
    let dim = 10;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point on sphere
    let mut initial_point = DVector::zeros(dim);
    initial_point[0] = 1.0;
    
    // Create optimizer with momentum
    let mut optimizer = SGD::new(
        SGDConfig::new()
            .with_constant_step_size(0.01)
            .with_classical_momentum(0.9)
    );
    
    // Set stopping criterion
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(100)
        .with_gradient_tolerance(1e-6);
    
    // Optimize
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    // Check convergence
    assert!(result.gradient_norm.unwrap_or(1.0) < 1e-5, "Gradient norm should be small at convergence");
    assert!(result.iterations < 100, "Should converge before max iterations");
    
    Ok(())
}

#[test]
fn test_sgd_gradient_clipping() -> Result<()> {
    let dim = 5;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point
    let mut initial_point = DVector::zeros(dim);
    initial_point[0] = 1.0;
    
    // Create optimizer with gradient clipping
    let mut optimizer = SGD::new(
        SGDConfig::new()
            .with_constant_step_size(0.1)
            .with_gradient_clip(1.0)
    );
    
    // Just run a few iterations to ensure gradient clipping doesn't break
    let stopping_criterion = StoppingCriterion::new().with_max_iterations(10);
    
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    // Basic sanity checks
    assert!(result.value.is_finite());
    assert!(result.gradient_norm.unwrap_or(0.0).is_finite());
    
    Ok(())
}