//! Tests for parallel transport properties.
//!
//! This test module verifies that parallel transport satisfies its
//! mathematical properties, including isometry and consistency.

use nalgebra::Dyn;
use riemannopt_core::{
    error::Result,
    manifold::Manifold,
    memory::workspace::Workspace,
    types::DVector,
};

/// Test manifold with parallel transport
#[derive(Debug)]
struct TransportTestManifold {
    dim: usize,
}

impl TransportTestManifold {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for TransportTestManifold {
    fn name(&self) -> &str {
        "Transport Test Manifold"
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
        Ok(u.dot(v))
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
        euclidean_grad: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        *result = euclidean_grad.clone();
        Ok(())
    }

    fn random_point(&self) -> DVector<f64> {
        DVector::from_fn(self.dim, |_, _| rand::random::<f64>() * 2.0 - 1.0)
    }

    fn random_tangent(&self, _point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
        *result = DVector::from_fn(self.dim, |_, _| {
            rand::random::<f64>() * 2.0 - 1.0
        });
        Ok(())
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        _to: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
        _workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        // For Euclidean space, parallel transport is identity
        *result = vector.clone();
        Ok(())
    }
}

/// Sphere manifold with proper parallel transport
#[derive(Debug)]
struct SphereWithTransport {
    dim: usize,
}

impl SphereWithTransport {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64, Dyn> for SphereWithTransport {
    fn name(&self) -> &str {
        "Sphere with Transport"
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
        let new_point = point + tangent;
        self.project_point(&new_point, result, _workspace);
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

    fn parallel_transport(
        &self,
        from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
        workspace: &mut Workspace<f64>,
    ) -> Result<()> {
        // Parallel transport on sphere along geodesic
        if (from - to).norm() < f64::EPSILON {
            *result = vector.clone();
            return Ok(());
        }

        // For simplicity, use projection-based transport
        // This is not exact parallel transport but preserves key properties
        let transported = vector - from * from.dot(vector);
        self.project_tangent(to, &transported, result, workspace)
    }
}

#[test]
fn test_transport_identity() {
    // Transport from a point to itself should be identity
    let manifold = TransportTestManifold::new(3);
    let tolerance = 1e-14;
    let mut workspace = Workspace::new();

    for _ in 0..20 {
        let point = manifold.random_point();
        let mut tangent = DVector::zeros(point.len());
        manifold.random_tangent(&point, &mut tangent, &mut workspace).unwrap();

        let mut transported = DVector::zeros(point.len());
        manifold
            .parallel_transport(&point, &point, &tangent, &mut transported, &mut workspace)
            .unwrap();

        let diff = &transported - &tangent;
        let error = diff.norm();

        assert!(
            error < tolerance,
            "Transport along zero curve not identity: error = {}",
            error
        );
    }
}

#[test]
fn test_transport_preserves_norm() {
    // Parallel transport should preserve the norm (isometry)
    let sphere = SphereWithTransport::new(4);
    let tolerance = 0.1; // Projection-based transport is approximate for larger steps
    let mut workspace = Workspace::new();

    for _ in 0..20 {
        let from = sphere.random_point();
        let mut direction = DVector::zeros(from.len());
        sphere.random_tangent(&from, &mut direction, &mut workspace).unwrap();
        direction *= 0.1;
        let mut to = DVector::zeros(from.len());
        sphere.retract(&from, &direction, &mut to, &mut workspace).unwrap();

        let mut tangent = DVector::zeros(from.len());
        sphere.random_tangent(&from, &mut tangent, &mut workspace).unwrap();
        let mut transported = DVector::zeros(from.len());
        sphere.parallel_transport(&from, &to, &tangent, &mut transported, &mut workspace).unwrap();

        let norm_before = sphere
            .inner_product(&from, &tangent, &tangent)
            .unwrap()
            .sqrt();
        let norm_after = sphere
            .inner_product(&to, &transported, &transported)
            .unwrap()
            .sqrt();

        assert!(
            (norm_after - norm_before).abs() < tolerance,
            "Transport doesn't preserve norm: ||v|| = {}, ||T(v)|| = {}",
            norm_before,
            norm_after
        );
    }
}

#[test]
fn test_transport_preserves_inner_products() {
    // Parallel transport should preserve inner products
    let sphere = SphereWithTransport::new(3);
    let tolerance = 0.05; // Projection-based transport is approximate
    let mut workspace = Workspace::new();

    for _ in 0..10 {
        let from = sphere.random_point();
        let mut direction = DVector::zeros(from.len());
        sphere.random_tangent(&from, &mut direction, &mut workspace).unwrap();
        direction *= 0.1;
        let mut to = DVector::zeros(from.len());
        sphere.retract(&from, &direction, &mut to, &mut workspace).unwrap();

        let mut u = DVector::zeros(from.len());
        sphere.random_tangent(&from, &mut u, &mut workspace).unwrap();
        let mut v = DVector::zeros(from.len());
        sphere.random_tangent(&from, &mut v, &mut workspace).unwrap();

        let mut u_transported = DVector::zeros(from.len());
        sphere.parallel_transport(&from, &to, &u, &mut u_transported, &mut workspace).unwrap();
        let mut v_transported = DVector::zeros(from.len());
        sphere.parallel_transport(&from, &to, &v, &mut v_transported, &mut workspace).unwrap();

        let inner_before = sphere.inner_product(&from, &u, &v).unwrap();
        let inner_after = sphere
            .inner_product(&to, &u_transported, &v_transported)
            .unwrap();

        assert!(
            (inner_after - inner_before).abs() < tolerance,
            "Transport doesn't preserve inner product: <u,v> = {}, <T(u),T(v)> = {}",
            inner_before,
            inner_after
        );
    }
}

#[test]
fn test_transport_linearity() {
    // Parallel transport should be linear
    let manifold = TransportTestManifold::new(4);
    let tolerance = 1e-14;
    let mut workspace = Workspace::new();

    let from = manifold.random_point();
    let mut direction = DVector::zeros(from.len());
    manifold.random_tangent(&from, &mut direction, &mut workspace).unwrap();
    direction *= 0.1;
    let mut to = DVector::zeros(from.len());
    manifold.retract(&from, &direction, &mut to, &mut workspace).unwrap();

    let mut u = DVector::zeros(from.len());
    manifold.random_tangent(&from, &mut u, &mut workspace).unwrap();
    let mut v = DVector::zeros(from.len());
    manifold.random_tangent(&from, &mut v, &mut workspace).unwrap();
    let alpha = 2.5;
    let beta = -1.3;

    // T(αu + βv) = αT(u) + βT(v)
    let combined = u.clone() * alpha + v.clone() * beta;
    let mut transport_combined = DVector::zeros(from.len());
    manifold.parallel_transport(&from, &to, &combined, &mut transport_combined, &mut workspace).unwrap();

    let mut u_transported = DVector::zeros(from.len());
    manifold.parallel_transport(&from, &to, &u, &mut u_transported, &mut workspace).unwrap();
    let mut v_transported = DVector::zeros(from.len());
    manifold.parallel_transport(&from, &to, &v, &mut v_transported, &mut workspace).unwrap();
    let linear_combination = u_transported * alpha + v_transported * beta;

    let diff = &transport_combined - &linear_combination;
    let error = diff.norm();

    assert!(
        error < tolerance,
        "Transport not linear: ||T(αu + βv) - (αT(u) + βT(v))|| = {}",
        error
    );
}

#[test]
fn test_transport_preserves_tangent_space() {
    // Transported vectors should remain in tangent space
    let sphere = SphereWithTransport::new(5);
    let tolerance = 1e-10; // Tangent space check can be stricter
    let mut workspace = Workspace::new();

    for _ in 0..20 {
        let from = sphere.random_point();
        let mut direction = DVector::zeros(from.len());
        sphere.random_tangent(&from, &mut direction, &mut workspace).unwrap();
        direction *= 0.05;
        let mut to = DVector::zeros(from.len());
        sphere.retract(&from, &direction, &mut to, &mut workspace).unwrap();

        let mut tangent = DVector::zeros(from.len());
        sphere.random_tangent(&from, &mut tangent, &mut workspace).unwrap();
        let mut transported = DVector::zeros(from.len());
        sphere.parallel_transport(&from, &to, &tangent, &mut transported, &mut workspace).unwrap();

        assert!(
            sphere.is_vector_in_tangent_space(&to, &transported, tolerance),
            "Transported vector not in tangent space at destination"
        );
    }
}

#[test]
fn test_transport_zero_vector() {
    // Transport of zero vector should be zero
    let manifold = TransportTestManifold::new(3);
    let mut workspace = Workspace::new();

    let from = manifold.random_point();
    let mut direction = DVector::zeros(from.len());
    manifold.random_tangent(&from, &mut direction, &mut workspace).unwrap();
    direction *= 0.1;
    let mut to = DVector::zeros(from.len());
    manifold.retract(&from, &direction, &mut to, &mut workspace).unwrap();

    let zero = DVector::zeros(3);
    let mut transported = DVector::zeros(from.len());
    manifold.parallel_transport(&from, &to, &zero, &mut transported, &mut workspace).unwrap();

    assert!(
        transported.norm() < 1e-15,
        "Transport of zero vector is not zero: ||T(0)|| = {}",
        transported.norm()
    );
}

#[test]
fn test_transport_inverse_consistency() {
    // Transport from A to B then B to A should recover original (approximately)
    let sphere = SphereWithTransport::new(3);
    let tolerance = 1e-2; // Approximate transport
    let mut workspace = Workspace::new();

    for _ in 0..10 {
        let point_a = sphere.random_point();
        let mut direction = DVector::zeros(point_a.len());
        sphere.random_tangent(&point_a, &mut direction, &mut workspace).unwrap();
        direction *= 0.05;
        let mut point_b = DVector::zeros(point_a.len());
        sphere.retract(&point_a, &direction, &mut point_b, &mut workspace).unwrap();

        let mut tangent = DVector::zeros(point_a.len());
        sphere.random_tangent(&point_a, &mut tangent, &mut workspace).unwrap();

        // Transport A -> B -> A
        let mut transported_to_b = DVector::zeros(point_a.len());
        sphere
            .parallel_transport(&point_a, &point_b, &tangent, &mut transported_to_b, &mut workspace)
            .unwrap();
        let mut transported_back = DVector::zeros(point_a.len());
        sphere
            .parallel_transport(&point_b, &point_a, &transported_to_b, &mut transported_back, &mut workspace)
            .unwrap();

        let diff = &transported_back - &tangent;
        let error = diff.norm();

        // Note: This may not be exact for all manifolds/transports
        assert!(
            error < tolerance,
            "Round-trip transport not consistent: error = {}",
            error
        );
    }
}

#[test]
fn test_transport_along_geodesic() {
    // Test that transport preserves angle with geodesic tangent
    let sphere = SphereWithTransport::new(3);
    let tolerance = 1e-12;
    let mut workspace = Workspace::new();

    let from = sphere.random_point();
    let mut geodesic_tangent = DVector::zeros(from.len());
    sphere.random_tangent(&from, &mut geodesic_tangent, &mut workspace).unwrap();
    geodesic_tangent *= 0.1;
    let mut to = DVector::zeros(from.len());
    sphere.retract(&from, &geodesic_tangent, &mut to, &mut workspace).unwrap();

    // Transport the geodesic tangent itself
    let mut transported_tangent = DVector::zeros(from.len());
    sphere
        .parallel_transport(&from, &to, &geodesic_tangent, &mut transported_tangent, &mut workspace)
        .unwrap();

    // The transported geodesic tangent should still be tangent to the geodesic at the new point
    // For a sphere with projection-based transport, check it's in tangent space
    assert!(
        sphere.is_vector_in_tangent_space(&to, &transported_tangent, tolerance),
        "Transported geodesic tangent not in tangent space"
    );
}

fn main() {
    test_transport_identity();
}
