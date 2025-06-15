//! Tests for Riemannian metric properties.
//!
//! This test module verifies that metrics satisfy positive definiteness
//! and other required mathematical properties.

use nalgebra::Dyn;
use riemannopt_core::{error::Result, manifold::Manifold, types::DVector};

/// Test manifold with custom metric
#[derive(Debug)]
struct ManifoldWithMetric {
    dim: usize,
    /// Metric matrix (must be positive definite)
    metric_matrix: nalgebra::DMatrix<f64>,
}

impl ManifoldWithMetric {
    fn new_euclidean(dim: usize) -> Self {
        Self {
            dim,
            metric_matrix: nalgebra::DMatrix::identity(dim, dim),
        }
    }

    fn new_weighted(dim: usize, weights: Vec<f64>) -> Self {
        assert_eq!(weights.len(), dim);
        let mut metric = nalgebra::DMatrix::zeros(dim, dim);
        for (i, &w) in weights.iter().enumerate() {
            metric[(i, i)] = w;
        }
        Self {
            dim,
            metric_matrix: metric,
        }
    }
}

impl Manifold<f64, Dyn> for ManifoldWithMetric {
    fn name(&self) -> &str {
        "Test Manifold with Metric"
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

    fn project_point(&self, point: &DVector<f64>) -> DVector<f64> {
        point.clone()
    }

    fn project_tangent(
        &self,
        _point: &DVector<f64>,
        vector: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        Ok(vector.clone())
    }

    fn inner_product(
        &self,
        _point: &DVector<f64>,
        u: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<f64> {
        // <u, v>_g = u^T * G * v
        let gu = &self.metric_matrix * u;
        Ok(v.dot(&gu))
    }

    fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>) -> Result<DVector<f64>> {
        Ok(point + tangent)
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>) -> Result<DVector<f64>> {
        Ok(other - point)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        // Riemannian gradient = G^{-1} * Euclidean gradient
        let g_inv = self.metric_matrix.clone().try_inverse().ok_or_else(|| {
            riemannopt_core::error::ManifoldError::numerical_error("Metric not invertible")
        })?;
        Ok(&g_inv * euclidean_grad)
    }

    fn random_point(&self) -> DVector<f64> {
        DVector::from_fn(self.dim, |_, _| rand::random::<f64>() * 2.0 - 1.0)
    }

    fn random_tangent(&self, _point: &DVector<f64>) -> Result<DVector<f64>> {
        Ok(DVector::from_fn(self.dim, |_, _| {
            rand::random::<f64>() * 2.0 - 1.0
        }))
    }
}

#[test]
fn test_euclidean_metric_positive_definite() {
    let manifold = ManifoldWithMetric::new_euclidean(3);
    let point = manifold.random_point();

    // Test with multiple random vectors
    for _ in 0..20 {
        let v = manifold.random_tangent(&point).unwrap();
        let inner = manifold.inner_product(&point, &v, &v).unwrap();

        // For non-zero vectors, inner product should be positive
        if v.norm() > 1e-10 {
            assert!(
                inner > 0.0,
                "Metric not positive definite: <v,v> = {} for ||v|| = {}",
                inner,
                v.norm()
            );
        }
    }
}

#[test]
fn test_weighted_metric_positive_definite() {
    // Test with positive weights
    let weights = vec![1.0, 2.0, 0.5, 3.0];
    let manifold = ManifoldWithMetric::new_weighted(4, weights);
    let point = manifold.random_point();

    for _ in 0..20 {
        let v = manifold.random_tangent(&point).unwrap();
        let inner = manifold.inner_product(&point, &v, &v).unwrap();

        if v.norm() > 1e-10 {
            assert!(
                inner > 0.0,
                "Weighted metric not positive definite: <v,v> = {}",
                inner
            );
        }
    }
}

#[test]
fn test_metric_symmetry() {
    let manifold = ManifoldWithMetric::new_euclidean(3);
    let point = manifold.random_point();

    // Test that <u,v> = <v,u>
    for _ in 0..10 {
        let u = manifold.random_tangent(&point).unwrap();
        let v = manifold.random_tangent(&point).unwrap();

        let inner_uv = manifold.inner_product(&point, &u, &v).unwrap();
        let inner_vu = manifold.inner_product(&point, &v, &u).unwrap();

        assert!(
            (inner_uv - inner_vu).abs() < 1e-14,
            "Metric not symmetric: <u,v> = {}, <v,u> = {}",
            inner_uv,
            inner_vu
        );
    }
}

#[test]
fn test_metric_bilinearity() {
    let manifold = ManifoldWithMetric::new_euclidean(3);
    let point = manifold.random_point();

    let u = manifold.random_tangent(&point).unwrap();
    let v = manifold.random_tangent(&point).unwrap();
    let w = manifold.random_tangent(&point).unwrap();

    let alpha = 2.5;
    let beta = -1.3;

    // Test linearity in first argument: <αu + βv, w> = α<u,w> + β<v,w>
    let lhs = manifold
        .inner_product(&point, &(u.clone() * alpha + v.clone() * beta), &w)
        .unwrap();
    let rhs = alpha * manifold.inner_product(&point, &u, &w).unwrap()
        + beta * manifold.inner_product(&point, &v, &w).unwrap();

    assert!(
        (lhs - rhs).abs() < 1e-14,
        "Metric not linear in first argument: {} != {}",
        lhs,
        rhs
    );

    // Test linearity in second argument: <u, αv + βw> = α<u,v> + β<u,w>
    let lhs = manifold
        .inner_product(&point, &u, &(v.clone() * alpha + w.clone() * beta))
        .unwrap();
    let rhs = alpha * manifold.inner_product(&point, &u, &v).unwrap()
        + beta * manifold.inner_product(&point, &u, &w).unwrap();

    assert!(
        (lhs - rhs).abs() < 1e-14,
        "Metric not linear in second argument: {} != {}",
        lhs,
        rhs
    );
}

#[test]
fn test_cauchy_schwarz_inequality() {
    let manifold = ManifoldWithMetric::new_euclidean(5);
    let point = manifold.random_point();

    // Test Cauchy-Schwarz: |<u,v>| ≤ ||u|| ||v||
    for _ in 0..20 {
        let u = manifold.random_tangent(&point).unwrap();
        let v = manifold.random_tangent(&point).unwrap();

        let inner_uv = manifold.inner_product(&point, &u, &v).unwrap();
        let norm_u = manifold.inner_product(&point, &u, &u).unwrap().sqrt();
        let norm_v = manifold.inner_product(&point, &v, &v).unwrap().sqrt();

        assert!(
            inner_uv.abs() <= norm_u * norm_v + 1e-14,
            "Cauchy-Schwarz violated: |<u,v>| = {} > ||u|| ||v|| = {}",
            inner_uv.abs(),
            norm_u * norm_v
        );
    }
}

#[test]
fn test_triangle_inequality() {
    let manifold = ManifoldWithMetric::new_weighted(3, vec![1.0, 1.5, 2.0]);
    let point = manifold.random_point();

    // Test triangle inequality: ||u + v|| ≤ ||u|| + ||v||
    for _ in 0..20 {
        let u = manifold.random_tangent(&point).unwrap();
        let v = manifold.random_tangent(&point).unwrap();
        let sum = &u + &v;

        let norm_sum = manifold.inner_product(&point, &sum, &sum).unwrap().sqrt();
        let norm_u = manifold.inner_product(&point, &u, &u).unwrap().sqrt();
        let norm_v = manifold.inner_product(&point, &v, &v).unwrap().sqrt();

        assert!(
            norm_sum <= norm_u + norm_v + 1e-14,
            "Triangle inequality violated: ||u+v|| = {} > ||u|| + ||v|| = {}",
            norm_sum,
            norm_u + norm_v
        );
    }
}

#[test]
fn test_zero_vector_norm() {
    let manifold = ManifoldWithMetric::new_euclidean(4);
    let point = manifold.random_point();

    // Zero vector should have zero norm
    let zero = DVector::zeros(4);
    let inner = manifold.inner_product(&point, &zero, &zero).unwrap();

    assert!(
        inner.abs() < 1e-15,
        "Zero vector has non-zero norm: {}",
        inner
    );
}

#[test]
fn test_metric_scaling() {
    // Test that scaling a vector scales the norm quadratically
    let manifold = ManifoldWithMetric::new_euclidean(3);
    let point = manifold.random_point();
    let v = manifold.random_tangent(&point).unwrap();

    let norm_v = manifold.inner_product(&point, &v, &v).unwrap();

    for scale in [2.0, -3.0, 0.5, 10.0] {
        let scaled_v = &v * scale;
        let norm_scaled = manifold
            .inner_product(&point, &scaled_v, &scaled_v)
            .unwrap();

        assert!(
            (norm_scaled - scale * scale * norm_v).abs() < 1e-13,
            "Metric scaling incorrect: ||{}v||^2 = {}, expected {}",
            scale,
            norm_scaled,
            scale * scale * norm_v
        );
    }
}

#[test]
fn test_orthogonality() {
    // Test that orthogonal vectors have zero inner product
    let manifold = ManifoldWithMetric::new_euclidean(3);
    let point = manifold.random_point();

    // Create orthogonal vectors
    let v1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let v2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
    let v3 = DVector::from_vec(vec![0.0, 0.0, 1.0]);

    let inner12 = manifold.inner_product(&point, &v1, &v2).unwrap();
    let inner13 = manifold.inner_product(&point, &v1, &v3).unwrap();
    let inner23 = manifold.inner_product(&point, &v2, &v3).unwrap();

    assert!(
        inner12.abs() < 1e-15,
        "Orthogonal vectors not orthogonal: <e1,e2> = {}",
        inner12
    );
    assert!(
        inner13.abs() < 1e-15,
        "Orthogonal vectors not orthogonal: <e1,e3> = {}",
        inner13
    );
    assert!(
        inner23.abs() < 1e-15,
        "Orthogonal vectors not orthogonal: <e2,e3> = {}",
        inner23
    );
}

#[test]
fn test_metric_eigenvalues() {
    // Test that all eigenvalues of the metric are positive
    let weights = vec![0.5, 1.0, 2.0, 3.0, 1.5];
    let manifold = ManifoldWithMetric::new_weighted(5, weights.clone());

    // For diagonal metric, eigenvalues are the diagonal elements
    for (i, &w) in weights.iter().enumerate() {
        assert!(w > 0.0, "Metric eigenvalue {} is not positive: {}", i, w);
    }

    // Verify by computing condition number estimate
    let point = manifold.random_point();
    let mut min_ratio = f64::INFINITY;
    let mut max_ratio: f64 = 0.0;

    for _ in 0..50 {
        let v = manifold.random_tangent(&point).unwrap();
        let v_norm = v.norm();
        if v_norm > 1e-10 {
            let inner = manifold.inner_product(&point, &v, &v).unwrap();
            let ratio = inner / (v_norm * v_norm);
            min_ratio = min_ratio.min(ratio);
            max_ratio = max_ratio.max(ratio);
        }
    }

    let condition_number = max_ratio / min_ratio;
    println!("Estimated condition number: {:.2}", condition_number);

    // Check that condition number is reasonable (all eigenvalues positive)
    assert!(min_ratio > 0.0, "Metric has non-positive eigenvalue");
    assert!(condition_number < 100.0, "Metric is poorly conditioned");
}

fn main() {
    // Run specific test if needed
    test_euclidean_metric_positive_definite();
}
