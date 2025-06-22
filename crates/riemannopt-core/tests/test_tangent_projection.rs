//! Tests for tangent space projection idempotency.
//!
//! This test module verifies that projecting to the tangent space
//! is an idempotent operation: P(P(v)) = P(v).

use nalgebra::Dyn;
use riemannopt_core::{error::Result, manifold::Manifold, types::DVector};

/// Sphere manifold for testing projections
#[derive(Debug)]
struct TestSphere {
    dim: usize,
}

impl TestSphere {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for TestSphere {
    fn name(&self) -> &str {
        "Test Sphere"
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
            result.copy_from(&(point / norm));
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
        result.copy_from(point);
        *result += tangent;
        let temp = result.clone();
        self.project_point(&temp, result);
        Ok(())
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
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
}

#[test]
fn test_projection_idempotent_sphere() {
    let sphere = TestSphere::new(3);
    let tolerance = 1e-14;

    for _ in 0..20 {
        let point = sphere.random_point();

        // Test with random vectors (not necessarily in tangent space)
        for _ in 0..10 {
            let mut random_vec = DVector::zeros(3);
            for i in 0..3 {
                random_vec[i] = rand::random::<f64>() * 2.0 - 1.0;
            }

            // Project once
            let mut projected_once = DVector::zeros(3);
            sphere.project_tangent(&point, &random_vec, &mut projected_once).unwrap();

            // Project twice
            let mut projected_twice = DVector::zeros(3);
            sphere.project_tangent(&point, &projected_once, &mut projected_twice).unwrap();

            // Check idempotency
            let diff = &projected_twice - &projected_once;
            let error = diff.norm();

            assert!(
                error < tolerance,
                "Projection not idempotent: ||P(P(v)) - P(v)|| = {} > {}",
                error,
                tolerance
            );

            // Verify projected vector is in tangent space
            assert!(
                sphere.is_vector_in_tangent_space(&point, &projected_once, tolerance),
                "Projected vector not in tangent space"
            );
        }
    }
}

#[test]
fn test_projection_preserves_tangent_vectors() {
    let sphere = TestSphere::new(4);
    let tolerance = 1e-14;

    for _ in 0..20 {
        let point = sphere.random_point();

        // Generate a vector already in tangent space
        let mut tangent = DVector::zeros(4);
        sphere.random_tangent(&point, &mut tangent).unwrap();

        // Project it
        let mut projected = DVector::zeros(4);
        sphere.project_tangent(&point, &tangent, &mut projected).unwrap();

        // Should be unchanged
        let diff = &projected - &tangent;
        let error = diff.norm();

        assert!(
            error < tolerance,
            "Projection changed tangent vector: ||P(v) - v|| = {} for v in T_pM",
            error
        );
    }
}

#[test]
fn test_projection_orthogonality() {
    let sphere = TestSphere::new(3);
    let tolerance = 1e-14;

    for _ in 0..20 {
        let point = sphere.random_point();

        // Generate random vector
        let mut v = DVector::zeros(3);
        for i in 0..3 {
            v[i] = rand::random::<f64>() * 2.0 - 1.0;
        }

        // Project it
        let mut projected = DVector::zeros(3);
        sphere.project_tangent(&point, &v, &mut projected).unwrap();

        // The difference (v - P(v)) should be parallel to the point
        let diff = &v - &projected;

        // Check if diff is parallel to point
        let cross_prod_norm = if sphere.dim == 3 {
            // For 3D, use cross product
            let cross = DVector::from_vec(vec![
                diff[1] * point[2] - diff[2] * point[1],
                diff[2] * point[0] - diff[0] * point[2],
                diff[0] * point[1] - diff[1] * point[0],
            ]);
            cross.norm()
        } else {
            // For general dimension, check if diff = λ * point for some λ
            let lambda = if point.norm() > tolerance {
                diff.dot(&point) / point.dot(&point)
            } else {
                0.0
            };
            (&diff - &point * lambda).norm()
        };

        assert!(
            cross_prod_norm < tolerance || diff.norm() < tolerance,
            "Projection residual not parallel to normal: error = {}",
            cross_prod_norm
        );
    }
}

#[test]
fn test_projection_linearity() {
    let sphere = TestSphere::new(5);
    let tolerance = 1e-14;

    let point = sphere.random_point();

    // Generate two random vectors
    let mut u = DVector::zeros(5);
    let mut v = DVector::zeros(5);
    for i in 0..5 {
        u[i] = rand::random::<f64>() * 2.0 - 1.0;
        v[i] = rand::random::<f64>() * 2.0 - 1.0;
    }

    let alpha = 2.5;
    let beta = -1.3;

    // P(αu + βv) = αP(u) + βP(v)
    let combined = u.clone() * alpha + v.clone() * beta;
    let mut proj_combined = DVector::zeros(5);
    sphere.project_tangent(&point, &combined, &mut proj_combined).unwrap();

    let mut proj_u = DVector::zeros(5);
    sphere.project_tangent(&point, &u, &mut proj_u).unwrap();
    let mut proj_v = DVector::zeros(5);
    sphere.project_tangent(&point, &v, &mut proj_v).unwrap();
    let linear_combination = proj_u * alpha + proj_v * beta;

    let diff = &proj_combined - &linear_combination;
    let error = diff.norm();

    assert!(
        error < tolerance,
        "Projection not linear: ||P(αu + βv) - (αP(u) + βP(v))|| = {}",
        error
    );
}

#[test]
fn test_projection_dimension_reduction() {
    // On an n-sphere, the tangent space has dimension n-1
    let sphere = TestSphere::new(4); // 3-sphere in R^4

    // Use a fixed point to make the test deterministic
    let mut point = DVector::zeros(4);
    point[0] = 1.0; // Point at north pole

    // Generate n linearly independent vectors
    let mut vectors = Vec::new();
    for i in 0..4 {
        let mut v = DVector::zeros(4);
        v[i] = 1.0;
        vectors.push(v);
    }

    // Project them all
    let mut projected: Vec<DVector<f64>> = Vec::new();
    for v in &vectors {
        let mut proj = DVector::zeros(4);
        sphere.project_tangent(&point, v, &mut proj).unwrap();
        projected.push(proj);
    }

    // Count non-zero vectors
    let non_zero_count = projected.iter().filter(|v| v.norm() > 1e-10).count();

    // The first vector [1,0,0,0] is parallel to the point, so it projects to zero
    // The other 3 vectors are in the tangent space
    assert_eq!(
        non_zero_count, 3,
        "Expected 3 non-zero vectors in tangent space, got {}",
        non_zero_count
    );
}

#[test]
fn test_projection_multiple_iterations() {
    let sphere = TestSphere::new(3);
    let tolerance = 1e-14;

    let point = sphere.random_point();
    let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);

    // Project multiple times
    let mut p1 = DVector::zeros(3);
    sphere.project_tangent(&point, &v, &mut p1).unwrap();
    let mut p2 = DVector::zeros(3);
    sphere.project_tangent(&point, &p1, &mut p2).unwrap();
    let mut p3 = DVector::zeros(3);
    sphere.project_tangent(&point, &p2, &mut p3).unwrap();
    let mut p4 = DVector::zeros(3);
    sphere.project_tangent(&point, &p3, &mut p4).unwrap();

    // All should be equal after first projection
    assert!((p1 - &p2).norm() < tolerance, "P != P²");
    assert!((p2 - &p3).norm() < tolerance, "P² != P³");
    assert!((p3 - &p4).norm() < tolerance, "P³ != P⁴");
}

#[test]
fn test_projection_zero_vector() {
    let sphere = TestSphere::new(3);

    let point = sphere.random_point();
    let zero = DVector::zeros(3);

    let mut projected = DVector::zeros(3);
    sphere.project_tangent(&point, &zero, &mut projected).unwrap();

    assert!(
        projected.norm() < 1e-15,
        "Projection of zero vector is not zero: ||P(0)|| = {}",
        projected.norm()
    );
}

#[test]
fn test_projection_normal_vector() {
    let sphere = TestSphere::new(3);
    let tolerance = 1e-14;

    let point = sphere.random_point();

    // The point itself is normal to the tangent space
    let mut projected = DVector::zeros(3);
    sphere.project_tangent(&point, &point, &mut projected).unwrap();

    assert!(
        projected.norm() < tolerance,
        "Projection of normal vector is not zero: ||P(n)|| = {}",
        projected.norm()
    );
}

