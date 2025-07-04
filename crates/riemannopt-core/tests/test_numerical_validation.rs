//! Tests for numerical validation utilities.
//!
//! This test module verifies gradient checking, retraction convergence,
//! metric compatibility, and numerical stability.

use riemannopt_core::{
    core::{
        manifold::Manifold,
        cost_function::CostFunction,
    },
    error::Result,
    memory::workspace::Workspace,
    types::DVector,
};

/// Simple sphere manifold for testing
#[derive(Debug)]
struct Sphere {
    dim: usize,
}

impl Sphere {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64> for Sphere {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;

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

    fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) {
        let norm = point.norm();
        if norm > f64::EPSILON {
            *result = point / norm;
        } else {
            result.fill(0.0);
            result[0] = 1.0;
        }
    }

    fn project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
        let inner = point.dot(vector);
        *result = vector - point * inner;
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        u: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<f64> {
        Ok(u.dot(v))
    }

    fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
        // Use exact exponential map for sphere
        let norm_v = tangent.norm();
        if norm_v < f64::EPSILON {
            *result = point.clone();
        } else {
            let cos_norm = norm_v.cos();
            let sin_norm = norm_v.sin();
            *result = point * cos_norm + tangent * (sin_norm / norm_v);
        }
        Ok(())
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
        let inner = point.dot(other).min(1.0).max(-1.0);
        let theta = inner.acos();

        if theta.abs() < f64::EPSILON {
            result.fill(0.0);
        } else {
            let v = other - point * inner;
            let v_norm = v.norm();
            if v_norm > f64::EPSILON {
                *result = v * (theta / v_norm);
            } else {
                result.fill(0.0);
            }
        }
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
        result: &mut DVector<f64>,
        workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        self.project_tangent(point, euclidean_grad, result, workspace)
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
        workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        // Simple projection-based transport for testing
        self.project_tangent(to, vector, result, workspace)
    }

    fn random_point(&self) -> DVector<f64> {
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rand::random::<f64>() * 2.0 - 1.0;
        }
        let mut result = DVector::zeros(self.dim);
        let mut workspace = Workspace::new();
        self.project_point(&v, &mut result, &mut workspace);
        result
    }

    fn random_tangent(&self, point: &DVector<f64>, result: &mut DVector<f64>, workspace: &mut Workspace<f64>) -> Result<()> {
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rand::random::<f64>() * 2.0 - 1.0;
        }
        self.project_tangent(point, &v, result, workspace)
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, workspace: &mut Workspace<f64>) -> Result<f64> {
        let mut tangent = DVector::zeros(x.len());
        self.inverse_retract(x, y, &mut tangent, workspace)?;
        self.norm(x, &tangent)
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: f64,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = tangent * scalar;
        Ok(())
    }

    fn add_tangents(
        &self,
        _point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = v1 + v2;
        Ok(())
    }
}

/// Test cost function for gradient checking
struct TestCostFunction<F, G> 
where 
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    f: F,
    grad_f: G,
}

impl<F, G> std::fmt::Debug for TestCostFunction<F, G> 
where 
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestCostFunction").finish()
    }
}

impl<F, G> CostFunction<f64> for TestCostFunction<F, G>
where 
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;

    fn cost(&self, x: &Self::Point) -> Result<f64> {
        Ok((self.f)(x))
    }

    fn gradient(&self, x: &Self::Point) -> Result<Self::TangentVector> {
        Ok((self.grad_f)(x))
    }

    fn hessian_vector_product(
        &self,
        _x: &Self::Point,
        _v: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // Not implemented for test
        unimplemented!("Hessian not needed for gradient checking")
    }

    fn gradient_fd_alloc(&self, x: &Self::Point) -> Result<Self::TangentVector> {
        self.gradient(x)
    }
}

/// Check gradient using finite differences
fn check_gradient_finite_diff<M: Manifold<f64>>(
    manifold: &M,
    cost_fn: &impl CostFunction<f64, Point = M::Point, TangentVector = M::TangentVector>,
    point: &M::Point,
    step_size: f64,
    tolerance: f64,
) -> Result<bool> {
    let mut workspace = Workspace::new();
    
    // Get analytic gradient
    let euclidean_grad = cost_fn.gradient(point)?;
    let mut riemannian_grad = euclidean_grad.clone();
    manifold.euclidean_to_riemannian_gradient(point, &euclidean_grad, &mut riemannian_grad, &mut workspace)?;
    
    // Get a random tangent vector for directional derivative
    let mut direction = riemannian_grad.clone();
    manifold.random_tangent(point, &mut direction, &mut workspace)?;
    
    // Normalize direction
    let dir_norm = manifold.norm(point, &direction)?;
    let mut normalized_dir = direction.clone();
    manifold.scale_tangent(point, 1.0 / dir_norm, &direction, &mut normalized_dir, &mut workspace)?;
    direction = normalized_dir;
    
    // Compute directional derivative analytically
    let analytic_deriv = manifold.inner_product(point, &riemannian_grad, &direction)?;
    
    // Compute finite difference approximation
    let mut point_plus = point.clone();
    let mut scaled_dir = direction.clone();
    manifold.scale_tangent(point, step_size, &direction, &mut scaled_dir, &mut workspace)?;
    manifold.retract(point, &scaled_dir, &mut point_plus, &mut workspace)?;
    
    let mut point_minus = point.clone();
    manifold.scale_tangent(point, -step_size, &direction, &mut scaled_dir, &mut workspace)?;
    manifold.retract(point, &scaled_dir, &mut point_minus, &mut workspace)?;
    
    let f_plus = cost_fn.cost(&point_plus)?;
    let f_minus = cost_fn.cost(&point_minus)?;
    let fd_deriv = (f_plus - f_minus) / (2.0 * step_size);
    
    // Check relative error
    let error = (analytic_deriv - fd_deriv).abs();
    let relative_error = if analytic_deriv.abs() > 1e-10 {
        error / analytic_deriv.abs()
    } else {
        error
    };
    
    Ok(relative_error < tolerance)
}

#[test]
fn test_gradient_checking_sphere() {
    let sphere = Sphere::new(3);
    
    // Test function: f(x) = x[0], constrained to sphere
    let cost_fn = TestCostFunction {
        f: |x: &DVector<f64>| x[0],
        grad_f: |_x: &DVector<f64>| {
            let mut e0 = DVector::zeros(3);
            e0[0] = 1.0;
            e0
        },
    };

    let point = sphere.random_point();
    let passed = check_gradient_finite_diff(&sphere, &cost_fn, &point, 1e-7, 1e-5).unwrap();
    
    assert!(passed, "Gradient check failed for linear function on sphere");
}

#[test]
fn test_quadratic_gradient_checking() {
    let sphere = Sphere::new(4);
    
    // Test function: f(x) = 0.5 * ||x||^2 (which equals 0.5 on sphere)
    let cost_fn = TestCostFunction {
        f: |x: &DVector<f64>| 0.5 * x.dot(x),
        grad_f: |x: &DVector<f64>| x.clone(),
    };

    let point = sphere.random_point();
    let passed = check_gradient_finite_diff(&sphere, &cost_fn, &point, 1e-8, 1e-5).unwrap();
    
    assert!(passed, "Gradient check failed for quadratic function on sphere");
}

#[test]
fn test_retraction_order() {
    // Test that retraction has correct order of approximation
    let sphere = Sphere::new(3);
    let mut workspace = Workspace::new();
    
    let point = sphere.random_point();
    let mut tangent = DVector::zeros(3);
    sphere.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
    
    // Scale tangent to have small norm
    let mut scaled = tangent.clone();
    sphere.scale_tangent(&point, 0.1, &tangent, &mut scaled, &mut workspace).unwrap();
    tangent = scaled;
    
    let step_sizes = vec![1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4];
    let mut errors = Vec::new();
    
    for &h in &step_sizes {
        let mut scaled_tangent = tangent.clone();
        sphere.scale_tangent(&point, h, &tangent, &mut scaled_tangent, &mut workspace).unwrap();
        
        // Compute retraction
        let mut retracted = point.clone();
        sphere.retract(&point, &scaled_tangent, &mut retracted, &mut workspace).unwrap();
        
        // For sphere, we can compute exact exponential map
        let norm_v = scaled_tangent.norm();
        let exact = if norm_v < f64::EPSILON {
            point.clone()
        } else {
            &point * norm_v.cos() + &scaled_tangent * (norm_v.sin() / norm_v)
        };
        
        // Measure error
        let error = (&retracted - &exact).norm();
        errors.push(error);
    }
    
    // Check convergence order (should be at least 2 for good retractions)
    let mut min_order = f64::INFINITY;
    for i in 1..errors.len() {
        if errors[i] > 1e-15 && errors[i-1] > 1e-15 {
            let ratio = errors[i-1] / errors[i];
            let h_ratio = step_sizes[i-1] / step_sizes[i];
            let order = ratio.log2() / h_ratio.log2();
            min_order = min_order.min(order);
        }
    }
    
    // For exponential map retraction, this should be exact (infinite order)
    // but we'll check for at least quadratic
    assert!(min_order > 1.9, "Retraction order too low: {}", min_order);
}

#[test]
fn test_projection_retraction_convergence() {
    // Test projection retraction vs exponential map
    let sphere = Sphere::new(3);
    let mut workspace = Workspace::new();
    
    let point = sphere.random_point();
    let mut tangent = DVector::zeros(3);
    sphere.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
    let mut scaled = tangent.clone();
    sphere.scale_tangent(&point, 0.1, &tangent, &mut scaled, &mut workspace).unwrap();
    tangent = scaled;
    
    let step_sizes = vec![0.1, 0.05, 0.025, 0.0125];
    let mut errors = Vec::new();
    
    for &h in &step_sizes {
        let mut scaled_tangent = tangent.clone();
        sphere.scale_tangent(&point, h, &tangent, &mut scaled_tangent, &mut workspace).unwrap();
        
        // Projection retraction: normalize(point + tangent)
        let projected = (&point + &scaled_tangent).normalize();
        
        // Exact exponential map
        let mut exact = point.clone();
        sphere.retract(&point, &scaled_tangent, &mut exact, &mut workspace).unwrap();
        
        let error = (&projected - &exact).norm();
        errors.push(error);
    }
    
    // Check that projection retraction has order 2
    for i in 1..errors.len() {
        if errors[i] > 1e-15 && errors[i-1] > 1e-15 {
            let ratio = errors[i-1] / errors[i];
            // For order 2: halving h should give 1/4 error
            assert!(ratio > 3.0 && ratio < 9.0, 
                "Projection retraction not quadratic: ratio = {}", ratio);
        }
    }
}

#[test]
fn test_tangent_space_orthogonality() {
    let sphere = Sphere::new(5);
    let mut workspace = Workspace::new();
    
    for _ in 0..10 {
        let point = sphere.random_point();
        
        // Generate random tangent vectors
        let mut v1 = DVector::zeros(5);
        let mut v2 = DVector::zeros(5);
        sphere.random_tangent(&point, &mut v1, &mut workspace).unwrap();
        sphere.random_tangent(&point, &mut v2, &mut workspace).unwrap();
        
        // Check orthogonality to point
        let inner1 = point.dot(&v1);
        let inner2 = point.dot(&v2);
        
        assert!(inner1.abs() < 1e-14, "Tangent vector not orthogonal to point: {}", inner1);
        assert!(inner2.abs() < 1e-14, "Tangent vector not orthogonal to point: {}", inner2);
    }
}

#[test]
fn test_retraction_inverse_consistency() {
    let sphere = Sphere::new(4);
    let mut workspace = Workspace::new();
    
    for _ in 0..5 {
        let point = sphere.random_point();
        let mut tangent = DVector::zeros(4);
        sphere.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        
        // Scale to moderate size
        let mut scaled = tangent.clone();
        sphere.scale_tangent(&point, 0.5, &tangent, &mut scaled, &mut workspace).unwrap();
        tangent = scaled;
        
        // Retract then inverse retract
        let mut other = point.clone();
        sphere.retract(&point, &tangent, &mut other, &mut workspace).unwrap();
        
        let mut recovered_tangent = tangent.clone();
        sphere.inverse_retract(&point, &other, &mut recovered_tangent, &mut workspace).unwrap();
        
        // Check recovery
        let error = (&tangent - &recovered_tangent).norm();
        let relative_error = error / tangent.norm();
        
        assert!(relative_error < 1e-12, 
            "Retraction not invertible: relative error = {}", relative_error);
    }
}

#[test]
fn test_parallel_transport_metric_preservation() {
    let sphere = Sphere::new(3);
    let mut workspace = Workspace::new();
    
    // For sphere with projection-based transport, metric is preserved
    let from = sphere.random_point();
    let to = sphere.random_point();
    
    // Generate two tangent vectors at 'from'
    let mut v1 = DVector::zeros(3);
    let mut v2 = DVector::zeros(3);
    sphere.random_tangent(&from, &mut v1, &mut workspace).unwrap();
    sphere.random_tangent(&from, &mut v2, &mut workspace).unwrap();
    
    // Compute inner product before transport
    let inner_before = sphere.inner_product(&from, &v1, &v2).unwrap();
    
    // Transport both vectors
    let mut v1_transported = v1.clone();
    let mut v2_transported = v2.clone();
    sphere.parallel_transport(&from, &to, &v1, &mut v1_transported, &mut workspace).unwrap();
    sphere.parallel_transport(&from, &to, &v2, &mut v2_transported, &mut workspace).unwrap();
    
    // Compute inner product after transport
    let inner_after = sphere.inner_product(&to, &v1_transported, &v2_transported).unwrap();
    
    // For projection transport on sphere, this is only approximate
    // but should be close for nearby points
    let distance = sphere.distance(&from, &to, &mut workspace).unwrap();
    if distance < 0.5 {
        let relative_change = (inner_after - inner_before).abs() / (inner_before.abs() + 1e-10);
        assert!(relative_change < 0.1, 
            "Parallel transport doesn't preserve metric well: relative change = {}", relative_change);
    }
}

/// Test manifold with custom metric
#[derive(Debug)]
struct WeightedEuclidean {
    dim: usize,
    weights: Vec<f64>,
}

impl WeightedEuclidean {
    fn new(dim: usize, weights: Vec<f64>) -> Self {
        assert_eq!(dim, weights.len());
        Self { dim, weights }
    }
}

impl Manifold<f64> for WeightedEuclidean {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;

    fn name(&self) -> &str {
        "Weighted Euclidean"
    }
    
    fn dimension(&self) -> usize {
        self.dim
    }
    
    fn is_point_on_manifold(&self, _point: &DVector<f64>, _tol: f64) -> bool {
        true
    }
    
    fn is_vector_in_tangent_space(
        &self,
        _point: &DVector<f64>,
        _vector: &DVector<f64>,
        _tol: f64,
    ) -> bool {
        true
    }
    
    fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) {
        *result = point.clone();
    }
    
    fn project_tangent(
        &self,
        _point: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = vector.clone();
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        u: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<f64> {
        let mut result = 0.0;
        for i in 0..self.dim {
            result += self.weights[i] * u[i] * v[i];
        }
        Ok(result)
    }

    fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
        *result = point + tangent;
        Ok(())
    }
    
    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
        *result = other - point;
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &DVector<f64>,
        grad: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = grad.clone();
        for i in 0..self.dim {
            result[i] /= self.weights[i];
        }
        Ok(())
    }

    fn parallel_transport(
        &self,
        _from: &Self::Point,
        _to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        // For flat manifolds, parallel transport is identity
        result.copy_from(vector);
        Ok(())
    }

    fn random_point(&self) -> DVector<f64> {
        DVector::from_fn(self.dim, |_, _| rand::random::<f64>())
    }
    
    fn random_tangent(&self, _point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
        *result = DVector::from_fn(self.dim, |_, _| {
            rand::random::<f64>() * 2.0 - 1.0
        });
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, _workspace: &mut Workspace<f64>) -> Result<f64> {
        let diff = y - x;
        let mut norm_sq = 0.0;
        for i in 0..self.dim {
            norm_sq += self.weights[i] * diff[i] * diff[i];
        }
        Ok(norm_sq.sqrt())
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: f64,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = tangent * scalar;
        Ok(())
    }

    fn add_tangents(
        &self,
        _point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = v1 + v2;
        Ok(())
    }
}

#[test]
fn test_gradient_weighted_metric() {
    let weights = vec![1.0, 2.0, 0.5];
    let manifold = WeightedEuclidean::new(3, weights);
    
    // f(x) = sum(x_i)
    let cost_fn = TestCostFunction {
        f: |x: &DVector<f64>| x.iter().sum(),
        grad_f: |_x: &DVector<f64>| DVector::from_vec(vec![1.0, 1.0, 1.0]),
    };

    let point = manifold.random_point();
    let passed = check_gradient_finite_diff(&manifold, &cost_fn, &point, 1e-8, 1e-6).unwrap();
    
    assert!(passed, "Gradient check failed for weighted metric");
}

#[test]
fn test_gradient_norm_preservation() {
    // Test that Riemannian gradient has correct norm
    let sphere = Sphere::new(4);
    let mut workspace = Workspace::new();
    
    // Linear functional: f(x) = <a, x>
    let a = DVector::from_vec(vec![1.0, 2.0, -1.0, 0.5]);
    let cost_fn = TestCostFunction {
        f: |x: &DVector<f64>| a.dot(x),
        grad_f: |_x: &DVector<f64>| a.clone(),
    };
    
    let point = sphere.random_point();
    
    // Get Riemannian gradient
    let euclidean_grad = cost_fn.gradient(&point).unwrap();
    let mut riemannian_grad = euclidean_grad.clone();
    sphere.euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut workspace).unwrap();
    
    // Check that gradient is in tangent space
    let inner = point.dot(&riemannian_grad);
    assert!(inner.abs() < 1e-14, "Riemannian gradient not in tangent space");
    
    // Check gradient norm matches directional derivatives
    let grad_norm = sphere.norm(&point, &riemannian_grad).unwrap();
    
    // Take directional derivative in direction of gradient
    if grad_norm > 1e-10 {
        let mut direction = riemannian_grad.clone();
        let mut normalized = direction.clone();
        sphere.scale_tangent(&point, 1.0 / grad_norm, &direction, &mut normalized, &mut workspace).unwrap();
        direction = normalized;
        
        let directional_deriv = sphere.inner_product(&point, &riemannian_grad, &direction).unwrap();
        
        assert!((directional_deriv - grad_norm).abs() < 1e-12,
            "Gradient norm doesn't match directional derivative");
    }
}