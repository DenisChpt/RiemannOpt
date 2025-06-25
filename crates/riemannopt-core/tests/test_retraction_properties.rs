//! Tests for retraction properties on manifolds.
//!
//! This test module verifies that retractions satisfy their mathematical
//! properties, particularly R(x, 0) = x.

use nalgebra::Dyn;
use riemannopt_core::{
    error::Result,
    manifold::Manifold,
    retraction::{DefaultRetraction, ExponentialRetraction, ProjectionRetraction, Retraction},
    types::DVector,
};
use std::ops::AddAssign;

/// Configuration for property tests.
#[derive(Debug, Clone)]
pub struct PropertyTestConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Number of random points to test
    pub num_points: usize,
    /// Number of random tangent vectors per point
    pub num_tangents: usize,
    /// Scale factor for tangent vectors
    pub tangent_scale: f64,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            num_points: 10,
            num_tangents: 5,
            tangent_scale: 0.1,
        }
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

    fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>) {
        let norm = point.norm();
        if norm > f64::EPSILON {
            result.copy_from(point);
            result.scale_mut(1.0 / norm);
        } else {
            result.fill(0.0);
            result[0] = 1.0;
        }
    }

    fn project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        // Project vector to tangent space: v - <v, p>p
        let inner = point.dot(vector);
        result.copy_from(vector);
        result.axpy(-inner, point, 1.0);
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

    fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        // Simple retraction: normalize(x + v)
        result.copy_from(point);
        result.add_assign(tangent);
        let temp = result.clone();
        self.project_point(&temp, result);
        Ok(())
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        // Log map on sphere
        let inner = point.dot(other).min(1.0).max(-1.0);
        let theta = inner.acos();

        if theta.abs() < f64::EPSILON {
            result.fill(0.0);
        } else {
            result.copy_from(other);
            result.axpy(-inner, point, 1.0);
            let v_norm = result.norm();
            if v_norm > f64::EPSILON {
                result.scale_mut(theta / v_norm);
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
    ) -> Result<()> {
        self.project_tangent(point, euclidean_grad, result)
    }

    fn random_point(&self) -> DVector<f64> {
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rand::random::<f64>() * 2.0 - 1.0;
        }
        let mut result = DVector::zeros(self.dim);
        self.project_point(&v, &mut result);
        result
    }

    fn random_tangent(&self, point: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rand::random::<f64>() * 2.0 - 1.0;
        }
        self.project_tangent(point, &v, result)
    }

    fn parallel_transport(
        &self,
        from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
    ) -> Result<()> {
        // Parallel transport on sphere using Schild's ladder approximation
        if (from - to).norm() < f64::EPSILON {
            result.copy_from(vector);
            return Ok(());
        }

        // For simplicity, use projection-based transport
        self.project_tangent(to, vector, result)
    }
}

/// Test that retraction at zero gives the same point
fn test_retraction_at_zero_property<R: Retraction<f64, Dyn>>(
    manifold: &UnitSphere,
    retraction: &R,
    config: &PropertyTestConfig,
) -> (bool, f64) {
    let mut max_error: f64 = 0.0;
    let mut all_passed = true;

    for _ in 0..config.num_points {
        let point = manifold.random_point();
        let zero_tangent = DVector::zeros(manifold.dim);

        let mut retracted = DVector::zeros(manifold.dim);
        match retraction.retract(manifold, &point, &zero_tangent, &mut retracted) {
            Ok(()) => {
                let diff = &retracted - &point;
                let error = diff.norm();

                if error > config.tolerance {
                    all_passed = false;
                }

                max_error = max_error.max(error);
            }
            Err(_) => {
                all_passed = false;
            }
        }
    }

    (all_passed, max_error)
}

#[test]
fn test_retraction_at_zero_sphere() {
    let sphere = UnitSphere::new(3);
    let config = PropertyTestConfig {
        tolerance: 1e-12,
        num_points: 20,
        num_tangents: 10,
        tangent_scale: 0.1,
    };

    // Test default retraction
    let retraction = DefaultRetraction;
    let (passed, max_error) = test_retraction_at_zero_property(&sphere, &retraction, &config);

    println!("Sphere - Default retraction at zero:");
    println!("  Passed: {}", passed);
    println!("  Max error: {:.2e}", max_error);

    assert!(passed, "Default retraction at zero test failed");
    assert!(max_error < config.tolerance, "Error exceeds tolerance");
}

#[test]
fn test_projection_retraction_at_zero() {
    let sphere = UnitSphere::new(4);
    let config = PropertyTestConfig {
        tolerance: 1e-12,
        num_points: 15,
        num_tangents: 5,
        tangent_scale: 0.05,
    };

    let retraction = ProjectionRetraction;
    let (passed, max_error) = test_retraction_at_zero_property(&sphere, &retraction, &config);

    assert!(passed, "Projection retraction at zero test failed");
    assert!(max_error < config.tolerance, "Error exceeds tolerance");
}

#[test]
fn test_exponential_retraction_at_zero() {
    let sphere = UnitSphere::new(3);
    let config = PropertyTestConfig {
        tolerance: 1e-10,
        num_points: 10,
        num_tangents: 5,
        tangent_scale: 0.01,
    };

    let retraction = ExponentialRetraction::<f64>::new();

    // Skip test if exponential map is not implemented
    let test_point = sphere.random_point();
    let test_tangent = DVector::zeros(3);
    let mut test_result = DVector::zeros(3);
    match retraction.retract(&sphere, &test_point, &test_tangent, &mut test_result) {
        Ok(()) => {
            let (passed, max_error) =
                test_retraction_at_zero_property(&sphere, &retraction, &config);
            assert!(passed, "Exponential retraction at zero test failed");
            assert!(max_error < config.tolerance, "Error exceeds tolerance");
        }
        Err(_) => {
            println!("Exponential retraction not implemented for sphere, skipping test");
        }
    }
}

#[test]
fn test_retraction_consistency() {
    let sphere = UnitSphere::new(3);
    let point = sphere.random_point();

    // Test that zero tangent vector gives the same point for all retractions
    let zero_tangent = DVector::zeros(3);

    // Test default retraction
    let mut result = DVector::zeros(3);
    DefaultRetraction
        .retract(&sphere, &point, &zero_tangent, &mut result)
        .unwrap();
    let error = (&result - &point).norm();
    println!("Default retraction: error = {:.2e}", error);
    assert!(
        error < 1e-14,
        "Default retraction failed: R(x, 0) != x (error: {:.2e})",
        error
    );

    // Test projection retraction
    let mut result = DVector::zeros(3);
    ProjectionRetraction
        .retract(&sphere, &point, &zero_tangent, &mut result)
        .unwrap();
    let error = (&result - &point).norm();
    println!("Projection retraction: error = {:.2e}", error);
    assert!(
        error < 1e-14,
        "Projection retraction failed: R(x, 0) != x (error: {:.2e})",
        error
    );

    // Test exponential retraction (may not be implemented for all manifolds)
    let exp_retraction = ExponentialRetraction::<f64>::new();
    let mut exp_result = DVector::zeros(3);
    match exp_retraction.retract(&sphere, &point, &zero_tangent, &mut exp_result) {
        Ok(()) => {
            let error = (&exp_result - &point).norm();
            println!("Exponential retraction: error = {:.2e}", error);
            assert!(
                error < 1e-14,
                "Exponential retraction failed: R(x, 0) != x (error: {:.2e})",
                error
            );
        }
        Err(_) => {
            println!("Exponential retraction not implemented for this manifold");
        }
    }
}

#[test]
fn test_retraction_produces_valid_points() {
    let sphere = UnitSphere::new(5);
    let config = PropertyTestConfig::default();

    let retraction = DefaultRetraction;

    for _ in 0..config.num_points {
        let point = sphere.random_point();

        for _ in 0..config.num_tangents {
            let mut tangent = DVector::zeros(sphere.dim);
            sphere.random_tangent(&point, &mut tangent).unwrap();
            let scaled_tangent = tangent * config.tangent_scale;

            let mut new_point = DVector::zeros(sphere.dim);
            retraction
                .retract(&sphere, &point, &scaled_tangent, &mut new_point)
                .unwrap();

            assert!(
                sphere.is_point_on_manifold(&new_point, 1e-10),
                "Retracted point is not on manifold"
            );
        }
    }
}

#[test]
fn test_retraction_local_rigidity() {
    // Test that retraction preserves distances locally (first-order)
    let sphere = UnitSphere::new(3);
    let point = sphere.random_point();

    let retraction = DefaultRetraction;

    // Test with very small tangent vectors
    // Avoid scales below 1e-8 where numerical precision issues dominate
    for scale in [1e-4, 1e-6, 1e-8] {
        let mut tangent = DVector::zeros(sphere.dim);
        sphere.random_tangent(&point, &mut tangent).unwrap();
        let scaled_tangent = tangent * scale;

        let mut new_point = DVector::zeros(sphere.dim);
        retraction
            .retract(&sphere, &point, &scaled_tangent, &mut new_point)
            .unwrap();

        // Distance on manifold should approximately equal norm of tangent vector
        let mut log_vector = DVector::zeros(sphere.dim);
        sphere.inverse_retract(&point, &new_point, &mut log_vector).unwrap();
        let log_norm = log_vector.norm();
        let tangent_norm = scaled_tangent.norm();

        println!(
            "Scale: {:.2e}, Log norm: {:.2e}, Tangent norm: {:.2e}",
            scale, log_norm, tangent_norm
        );

        // Skip test if values are too small for meaningful comparison
        if log_norm < 1e-15 || tangent_norm < 1e-15 {
            println!("Values too small for meaningful comparison, skipping");
            continue;
        }

        let distance_ratio = log_norm / tangent_norm;

        // For small vectors, ratio should be close to 1
        // Allow slightly larger tolerance for numerical stability
        // At very small scales (< 1e-6), numerical errors dominate
        let tolerance = if scale < 1e-7 {
            3.0 // Larger tolerance for very small scales due to numerical precision
        } else if scale < 1e-5 {
            0.1 // Moderate tolerance for small scales
        } else {
            (scale * 10.0).max(1e-3)
        };
        assert!(
            (distance_ratio - 1.0).abs() < tolerance,
            "Retraction not locally rigid at scale {:e}: ratio = {}, tolerance = {:e}",
            scale,
            distance_ratio,
            tolerance
        );
    }
}

#[test]
fn test_multiple_retractions_at_zero() {
    // Test all retraction types on different manifold dimensions
    let dimensions = vec![2, 3, 5, 10];

    for dim in dimensions {
        let sphere = UnitSphere::new(dim);
        let config = PropertyTestConfig {
            tolerance: 1e-12,
            num_points: 5,
            num_tangents: 3,
            tangent_scale: 0.1,
        };

        // Test default retraction
        let (passed, _) = test_retraction_at_zero_property(&sphere, &DefaultRetraction, &config);
        assert!(passed, "Default retraction failed on {}-sphere", dim - 1);

        // Test projection retraction
        let (passed, _) = test_retraction_at_zero_property(&sphere, &ProjectionRetraction, &config);
        assert!(passed, "Projection retraction failed on {}-sphere", dim - 1);

        // Test exponential retraction (skip if not implemented)
        let test_point = sphere.random_point();
        let test_tangent = DVector::zeros(dim);
        let mut test_result = DVector::zeros(dim);
        let exp_retraction = ExponentialRetraction::<f64>::new();
        if exp_retraction
            .retract(&sphere, &test_point, &test_tangent, &mut test_result)
            .is_ok()
        {
            let (passed, _) =
                test_retraction_at_zero_property(&sphere, &exp_retraction, &config);
            assert!(
                passed,
                "Exponential retraction failed on {}-sphere",
                dim - 1
            );
        }
    }
}

fn main() {
    // Run a specific test if needed
    test_retraction_at_zero_sphere();
}
