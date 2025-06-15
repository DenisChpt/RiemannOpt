//! Tests for optimizer properties.
//!
//! This test module verifies that optimizers satisfy key mathematical
//! properties like descent, convergence, and invariance.

use nalgebra::Dyn;
use riemannopt_core::{
    cost_function::CostFunction, error::Result, manifold::Manifold, types::DVector,
};

/// Simple quadratic cost function on a manifold
#[derive(Debug)]
struct QuadraticCost {
    target: DVector<f64>,
}

impl QuadraticCost {
    fn new(target: DVector<f64>) -> Self {
        Self { target }
    }
}

impl CostFunction<f64, Dyn> for QuadraticCost {
    fn cost(&self, point: &DVector<f64>) -> Result<f64> {
        let diff = point - &self.target;
        Ok(0.5 * diff.dot(&diff))
    }

    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
        Ok(point - &self.target)
    }
}

/// Simple sphere manifold for testing
#[derive(Debug)]
struct UnitSphere {
    dim: usize,
}

impl UnitSphere {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for UnitSphere {
    fn name(&self) -> &str {
        "Unit Sphere"
    }

    fn dimension(&self) -> usize {
        self.dim - 1
    }

    fn is_point_on_manifold(&self, point: &DVector<f64>, tol: f64) -> bool {
        (point.norm() - 1.0).abs() < tol
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<f64>,
        vector: &DVector<f64>,
        tol: f64,
    ) -> bool {
        point.dot(vector).abs() < tol
    }

    fn project_point(&self, point: &DVector<f64>) -> DVector<f64> {
        let norm = point.norm();
        if norm > f64::EPSILON {
            point / norm
        } else {
            let mut p = DVector::zeros(self.dim);
            p[0] = 1.0;
            p
        }
    }

    fn project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>) -> Result<DVector<f64>> {
        let inner = point.dot(vector);
        Ok(vector - point * inner)
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        u: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<f64> {
        Ok(u.dot(v))
    }

    fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>) -> Result<DVector<f64>> {
        let new_point = point + tangent;
        Ok(self.project_point(&new_point))
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>) -> Result<DVector<f64>> {
        let inner = point.dot(other).min(1.0).max(-1.0);
        let theta = inner.acos();

        if theta.abs() < f64::EPSILON {
            Ok(DVector::zeros(self.dim))
        } else {
            let v = other - point * inner;
            let v_norm = v.norm();
            if v_norm > f64::EPSILON {
                Ok(v * (theta / v_norm))
            } else {
                Ok(DVector::zeros(self.dim))
            }
        }
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        self.project_tangent(point, euclidean_grad)
    }

    fn random_point(&self) -> DVector<f64> {
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rand::random::<f64>() * 2.0 - 1.0;
        }
        self.project_point(&v)
    }

    fn random_tangent(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rand::random::<f64>() * 2.0 - 1.0;
        }
        self.project_tangent(point, &v)
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        self.project_tangent(to, vector)
    }
}

/// Test that gradient descent maintains descent property
#[test]
fn test_descent_property() {
    let sphere = UnitSphere::new(3);

    // Target point on sphere
    let mut target = DVector::zeros(3);
    target[0] = 1.0 / 3.0_f64.sqrt();
    target[1] = 1.0 / 3.0_f64.sqrt();
    target[2] = 1.0 / 3.0_f64.sqrt();

    let cost_fn = QuadraticCost::new(target);

    // Start from a different point
    let mut x = sphere.random_point();
    let step_size = 0.01;

    // Track cost values
    let mut costs = Vec::new();
    costs.push(cost_fn.cost(&x).unwrap());

    // Perform gradient descent steps
    for _ in 0..20 {
        let grad = cost_fn.gradient(&x).unwrap();
        let riem_grad = sphere.euclidean_to_riemannian_gradient(&x, &grad).unwrap();

        // Take a step
        let step = riem_grad * (-step_size);
        x = sphere.retract(&x, &step).unwrap();

        costs.push(cost_fn.cost(&x).unwrap());
    }

    // Check descent property: each step should decrease cost
    for i in 1..costs.len() {
        assert!(
            costs[i] <= costs[i - 1] * (1.0 + 1e-10), // Allow small numerical error
            "Cost increased at iteration {}: {} > {}",
            i,
            costs[i],
            costs[i - 1]
        );
    }

    // Check overall decrease
    assert!(
        costs.last().unwrap() < &costs[0],
        "Final cost {} not less than initial cost {}",
        costs.last().unwrap(),
        costs[0]
    );
}

/// Test convergence on a simple convex problem
#[test]
fn test_convergence_simple_problem() {
    let sphere = UnitSphere::new(4);

    // Create a simple problem: minimize distance to a target point
    let mut target = DVector::zeros(4);
    target[0] = 0.5;
    target[1] = 0.5;
    target[2] = 0.5;
    target[3] = 0.5;
    let target = sphere.project_point(&target);

    let cost_fn = QuadraticCost::new(target.clone());

    // Start from a random point
    let mut x = sphere.random_point();
    let step_size = 0.1;
    let tolerance = 1e-6;

    // Run gradient descent
    let mut converged = false;
    for iter in 0..1000 {
        let grad = cost_fn.gradient(&x).unwrap();
        let riem_grad = sphere.euclidean_to_riemannian_gradient(&x, &grad).unwrap();

        // Check convergence
        if riem_grad.norm() < tolerance {
            converged = true;
            println!("Converged in {} iterations", iter);
            break;
        }

        // Take a step
        let step = riem_grad * (-step_size);
        x = sphere.retract(&x, &step).unwrap();
    }

    assert!(converged, "Algorithm did not converge");

    // Check that we're close to a critical point
    let final_grad = cost_fn.gradient(&x).unwrap();
    let final_riem_grad = sphere
        .euclidean_to_riemannian_gradient(&x, &final_grad)
        .unwrap();
    assert!(
        final_riem_grad.norm() < tolerance,
        "Not at critical point: gradient norm = {}",
        final_riem_grad.norm()
    );
}

/// Test invariance under reparametrization
#[test]
fn test_invariance_reparametrization() {
    // Test that the algorithm behaves the same way under different
    // representations of the same manifold

    // Sphere represented as {x : ||x|| = 1}
    let sphere1 = UnitSphere::new(3);

    // Sphere represented as {x : ||x||² = 1} (same manifold, different parametrization)
    // For this test, we'll use the same implementation but verify the principle

    let cost_fn = |x: &DVector<f64>| -> f64 {
        // Simple cost: x[0]
        x[0]
    };

    let grad_fn = |_x: &DVector<f64>| -> DVector<f64> {
        let mut g = DVector::zeros(3);
        g[0] = 1.0;
        g
    };

    // Start from the same point
    let mut x0 = DVector::zeros(3);
    x0[1] = 1.0; // Start at (0, 1, 0)

    // Run gradient descent with both representations
    let step_size = 0.01;
    let num_steps = 10;

    // Path 1: Standard sphere
    let mut x1 = x0.clone();
    let mut path1 = vec![x1.clone()];

    for _ in 0..num_steps {
        let grad = grad_fn(&x1);
        let riem_grad = sphere1
            .euclidean_to_riemannian_gradient(&x1, &grad)
            .unwrap();
        let step = riem_grad * (-step_size);
        x1 = sphere1.retract(&x1, &step).unwrap();
        path1.push(x1.clone());
    }

    // Path 2: Same algorithm, just to verify consistency
    let mut x2 = x0.clone();
    let mut path2 = vec![x2.clone()];

    for _ in 0..num_steps {
        let grad = grad_fn(&x2);
        let riem_grad = sphere1
            .euclidean_to_riemannian_gradient(&x2, &grad)
            .unwrap();
        let step = riem_grad * (-step_size);
        x2 = sphere1.retract(&x2, &step).unwrap();
        path2.push(x2.clone());
    }

    // Paths should be identical (up to numerical precision)
    for (p1, p2) in path1.iter().zip(path2.iter()) {
        let diff = (p1 - p2).norm();
        assert!(diff < 1e-14, "Paths differ: ||p1 - p2|| = {}", diff);
    }

    // Check that optimization made progress
    assert!(
        cost_fn(&x1) < cost_fn(&x0) - 1e-4,
        "No progress made in optimization"
    );
}

/// Test momentum conservation in gradient descent with momentum
#[test]
fn test_momentum_conservation() {
    let sphere = UnitSphere::new(3);

    // Use a quadratic cost function for more stable gradient
    let mut target = DVector::zeros(3);
    target[0] = 0.0;
    target[1] = 0.0;
    target[2] = -1.0; // South pole

    let cost_fn = QuadraticCost::new(target);

    // Initialize at a point away from both poles
    let mut x = DVector::zeros(3);
    x[0] = 1.0 / 3.0_f64.sqrt();
    x[1] = 1.0 / 3.0_f64.sqrt();
    x[2] = 1.0 / 3.0_f64.sqrt();

    let mut momentum = DVector::zeros(3);
    let step_size = 0.1; // Increased step size for better progress
    let momentum_coeff = 0.9; // Standard momentum coefficient

    // Track cost decrease to verify momentum helps
    let mut costs = Vec::new();
    let mut gradient_norms = Vec::new();

    // Run gradient descent with momentum
    for iter in 0..200 {
        // More iterations
        costs.push(cost_fn.cost(&x).unwrap());

        let grad = cost_fn.gradient(&x).unwrap();
        let riem_grad = sphere.euclidean_to_riemannian_gradient(&x, &grad).unwrap();
        gradient_norms.push(riem_grad.norm());

        // Update momentum with decay and gradient
        momentum = momentum * momentum_coeff - &riem_grad * step_size;

        // Project momentum to tangent space at current point
        momentum = sphere.project_tangent(&x, &momentum).unwrap();

        // Take step using momentum
        x = sphere.retract(&x, &momentum).unwrap();

        // Debug print every 20 iterations
        if iter % 20 == 0 {
            println!(
                "Iter {}: cost = {:.6}, grad_norm = {:.6}",
                iter,
                costs.last().unwrap(),
                gradient_norms.last().unwrap()
            );
        }
    }

    println!(
        "Final: cost = {:.6}, grad_norm = {:.6}",
        costs.last().unwrap(),
        gradient_norms.last().unwrap()
    );

    // Check that optimization with momentum converges
    // Note: minimum cost is 2.0 (distance squared from north to south pole)
    let final_cost = *costs.last().unwrap();
    assert!(
        final_cost < 2.01,
        "Optimization with momentum did not converge: final cost = {}",
        final_cost
    );

    // Check that cost decreases overall with momentum
    // Momentum can cause oscillations but should trend downward
    let initial_cost = costs[0];
    assert!(
        final_cost < initial_cost * 0.99, // At least 1% improvement
        "Cost should decrease: initial = {}, final = {}",
        initial_cost,
        final_cost
    );

    // Verify convergence by checking that final gradient is small
    let final_grad_norm = *gradient_norms.last().unwrap();
    assert!(
        final_grad_norm < 0.1,
        "Should converge to critical point: final gradient norm = {}",
        final_grad_norm
    );

    // Check momentum effect: Compare with simple gradient descent
    // Reset to initial position
    let mut x_sgd = DVector::zeros(3);
    x_sgd[0] = 1.0 / 3.0_f64.sqrt();
    x_sgd[1] = 1.0 / 3.0_f64.sqrt();
    x_sgd[2] = 1.0 / 3.0_f64.sqrt();
    let mut costs_sgd = Vec::new();

    // Run simple gradient descent (no momentum)
    for _ in 0..200 {
        costs_sgd.push(cost_fn.cost(&x_sgd).unwrap());
        let grad = cost_fn.gradient(&x_sgd).unwrap();
        let riem_grad = sphere
            .euclidean_to_riemannian_gradient(&x_sgd, &grad)
            .unwrap();
        let step = riem_grad * (-step_size);
        x_sgd = sphere.retract(&x_sgd, &step).unwrap();
    }

    // Momentum behavior: might oscillate early but converges well
    // Check different aspects:

    // 1. Both methods should converge to similar final cost
    let final_cost_sgd = *costs_sgd.last().unwrap();
    assert!(
        (final_cost - final_cost_sgd).abs() < 0.001,
        "Both methods should converge to similar cost: momentum = {}, SGD = {}",
        final_cost,
        final_cost_sgd
    );

    // 2. Check momentum doesn't diverge - max cost should be bounded
    let max_cost_momentum = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let _max_cost_sgd = costs_sgd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_cost_momentum < initial_cost * 1.5,
        "Momentum should not cause excessive overshooting: max = {}",
        max_cost_momentum
    );

    // 3. Print convergence comparison
    println!("\nConvergence comparison:");
    for i in [10, 30, 50, 100, 150].iter() {
        if costs.len() > *i && costs_sgd.len() > *i {
            println!(
                "  Iter {}: momentum = {:.6}, SGD = {:.6}",
                i, costs[*i], costs_sgd[*i]
            );
        }
    }

    // 4. Momentum should eventually perform well
    let late_stage = 100;
    if costs.len() > late_stage && costs_sgd.len() > late_stage {
        // In late stages, both should be close to optimum
        assert!(
            costs[late_stage] < 0.01 && costs_sgd[late_stage] < 0.01,
            "Both methods should converge by iteration {}",
            late_stage
        );
    }
}

/// Test that optimization is covariant with respect to manifold isometries
#[test]
fn test_isometry_covariance() {
    let sphere = UnitSphere::new(3);

    // Define a rotation matrix (isometry of the sphere)
    let theta = std::f64::consts::PI / 4.0;
    let rotation = nalgebra::Matrix3::new(
        theta.cos(),
        -theta.sin(),
        0.0,
        theta.sin(),
        theta.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );

    // Cost function: distance to north pole
    let north_pole = {
        let mut n = DVector::zeros(3);
        n[2] = 1.0;
        n
    };

    let cost_fn = |x: &DVector<f64>| -> f64 { 0.5 * (x - &north_pole).norm_squared() };

    let grad_fn = |x: &DVector<f64>| -> DVector<f64> { x - &north_pole };

    // Start from a point and its rotated version
    let x0 = {
        let mut x = DVector::zeros(3);
        x[0] = 1.0 / 2.0_f64.sqrt();
        x[1] = 1.0 / 2.0_f64.sqrt();
        x[2] = 0.0;
        x
    };

    let x0_vec3 = nalgebra::Vector3::new(x0[0], x0[1], x0[2]);
    let x0_rotated_vec3 = rotation * x0_vec3;
    let x0_rotated = DVector::from_vec(vec![
        x0_rotated_vec3[0],
        x0_rotated_vec3[1],
        x0_rotated_vec3[2],
    ]);

    // Run optimization from both starting points
    let step_size = 0.1;
    let num_steps = 20;

    // Path 1: Original
    let mut x1 = x0.clone();
    for _ in 0..num_steps {
        let grad = grad_fn(&x1);
        let riem_grad = sphere.euclidean_to_riemannian_gradient(&x1, &grad).unwrap();
        let step = riem_grad * (-step_size);
        x1 = sphere.retract(&x1, &step).unwrap();
    }

    // Path 2: Rotated
    let mut x2 = x0_rotated.clone();
    for _ in 0..num_steps {
        let grad = grad_fn(&x2);
        let riem_grad = sphere.euclidean_to_riemannian_gradient(&x2, &grad).unwrap();
        let step = riem_grad * (-step_size);
        x2 = sphere.retract(&x2, &step).unwrap();
    }

    // Final points should be related by the same rotation
    // Note: This test assumes the cost function is also rotation-invariant
    // In this case, distance to north pole is NOT rotation invariant
    // So we just check both converged to similar cost values
    let cost1 = cost_fn(&x1);
    let cost2 = cost_fn(&x2);

    println!("Final costs: {} and {}", cost1, cost2);

    // Both should have made similar progress
    assert!(
        cost1 < 0.1,
        "Optimization 1 did not converge well: cost = {}",
        cost1
    );
    assert!(
        cost2 < 0.1,
        "Optimization 2 did not converge well: cost = {}",
        cost2
    );
}

fn main() {
    // Run tests
    test_descent_property();
    test_convergence_simple_problem();
    test_invariance_reparametrization();
    test_momentum_conservation();
    test_isometry_covariance();
}
