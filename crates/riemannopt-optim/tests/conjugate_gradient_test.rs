//! Integration tests for the Conjugate Gradient optimizer

use riemannopt_core::{
    core::cost_function::CostFunction,
    error::Result,
    memory::workspace::Workspace,
    optimization::{
        optimizer::StoppingCriterion,
    },
    types::{DVector, Scalar},
};
use riemannopt_manifolds::Sphere;
use riemannopt_optim::{ConjugateGradient, CGConfig};
use num_traits::Float;

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
    
    fn gradient_fd_alloc(&self, x: &Self::Point) -> Result<Self::TangentVector> {
        // Fallback to gradient computation
        let mut gradient = DVector::zeros(x.len());
        self.cost_and_gradient(x, &mut Workspace::new(), &mut gradient)?;
        Ok(gradient)
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
}

#[test]
fn test_cg_fletcher_reeves() -> Result<()> {
    let dim = 10;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point on sphere - start from a non-critical point
    let mut initial_point = DVector::zeros(dim);
    initial_point[0] = 0.7;
    initial_point[dim-1] = 0.3;
    initial_point /= initial_point.norm(); // Normalize to be on sphere
    
    // Create optimizer with Fletcher-Reeves method
    let mut optimizer = ConjugateGradient::new(
        CGConfig::fletcher_reeves()
            .with_restart_period(10)
    );
    
    // Set stopping criterion
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(100)
        .with_gradient_tolerance(1e-6);
    
    // Optimize
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    println!("CG-FR Results:");
    println!("Final point: {:?}", result.point);
    println!("Final value: {}", result.value);
    println!("Final gradient norm: {:?}", result.gradient_norm);
    println!("Iterations: {}", result.iterations);
    
    // Check convergence
    assert!(result.gradient_norm.unwrap_or(1.0) < 1e-4, "Gradient norm should be small at convergence");
    assert!(result.value < 1.1, "Minimum value should be close to 1");
    
    Ok(())
}

#[test]
fn test_cg_polak_ribiere() -> Result<()> {
    let dim = 8;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point
    let mut initial_point = DVector::zeros(dim);
    initial_point[0] = 0.5;
    initial_point[dim/2] = 0.5;
    initial_point /= initial_point.norm();
    
    // Create optimizer with Polak-Ribière method
    let mut optimizer = ConjugateGradient::new(
        CGConfig::polak_ribiere()
            .with_pr_plus(true)  // Use PR+ variant
            .with_min_beta(-0.5_f64)
            .with_max_beta(5.0_f64)
    );
    
    // Set stopping criterion
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(80)
        .with_gradient_tolerance(1e-6);
    
    // Optimize
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    println!("\nCG-PR+ Results:");
    println!("Iterations: {}", result.iterations);
    println!("Final gradient norm: {:?}", result.gradient_norm);
    
    // Check convergence
    assert!(result.gradient_norm.unwrap_or(1.0) < 1e-4, "Should converge to small gradient norm");
    
    Ok(())
}

#[test]
fn test_cg_hestenes_stiefel() -> Result<()> {
    let dim = 5;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point
    let mut initial_point = DVector::zeros(dim);
    initial_point[0] = 0.6;
    initial_point[dim-1] = 0.4;
    initial_point /= initial_point.norm();
    
    // Create optimizer with Hestenes-Stiefel method
    let mut optimizer = ConjugateGradient::new(
        CGConfig::hestenes_stiefel()
    );
    
    // Set stopping criterion
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(50)
        .with_gradient_tolerance(1e-5);
    
    // Optimize
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    println!("\nCG-HS Results:");
    println!("Iterations: {}", result.iterations);
    println!("Final value: {}", result.value);
    
    // Basic sanity checks
    assert!(result.value.is_finite());
    assert!(result.gradient_norm.unwrap_or(0.0).is_finite());
    
    Ok(())
}

#[test]
fn test_cg_dai_yuan() -> Result<()> {
    let dim = 6;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point
    let mut initial_point = DVector::zeros(dim);
    for i in 0..dim {
        initial_point[i] = <f64 as Float>::sin((i as f64) * 0.5);
    }
    initial_point /= initial_point.norm();
    
    // Create optimizer with Dai-Yuan method
    let mut optimizer = ConjugateGradient::new(
        CGConfig::dai_yuan()
            .with_restart_period(15)
    );
    
    // Set stopping criterion
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(60)
        .with_gradient_tolerance(1e-5);
    
    // Optimize
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    println!("\nCG-DY Results:");
    println!("Iterations: {}", result.iterations);
    println!("Final value: {}", result.value);
    
    // Check that it converged
    assert!(result.iterations > 0);
    assert!(result.value < initial_point.dot(&cost_fn.matrix.component_mul(&initial_point)));
    
    Ok(())
}

#[test]
fn test_cg_with_line_search() -> Result<()> {
    let dim = 5;
    let sphere = Sphere::<f64>::new(dim)?;
    let cost_fn = QuadraticOnSphere::new(dim);
    
    // Initial point
    let mut initial_point = DVector::zeros(dim);
    initial_point[0] = 1.0; // Start at a critical point
    
    // Create optimizer with line search enabled
    let mut optimizer = ConjugateGradient::new(
        CGConfig::polak_ribiere()
            .with_line_search()
    );
    
    // Set stopping criterion
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(30)
        .with_gradient_tolerance(1e-6);
    
    // Optimize - should handle critical point well
    let result = optimizer.optimize(&cost_fn, &sphere, &initial_point, &stopping_criterion)?;
    
    // Should converge immediately if starting at minimum
    assert!(result.iterations <= 2, "Should converge quickly from critical point");
    
    Ok(())
}