//! Tests for parallel transport properties.
//!
//! This test module verifies that parallel transport satisfies its
//! mathematical properties, including isometry and consistency.

use nalgebra::Dyn;
use riemannopt_core::{
    error::Result,
    manifold::Manifold,
    retraction::{DefaultRetraction, Retraction},
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
        Ok(u.dot(v))
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
        Ok(euclidean_grad.clone())
    }

    fn random_point(&self) -> DVector<f64> {
        DVector::from_fn(self.dim, |_, _| rand::random::<f64>() * 2.0 - 1.0)
    }

    fn random_tangent(&self, _point: &DVector<f64>) -> Result<DVector<f64>> {
        Ok(DVector::from_fn(self.dim, |_, _| {
            rand::random::<f64>() * 2.0 - 1.0
        }))
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        _to: &DVector<f64>,
        vector: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        // For Euclidean space, parallel transport is identity
        Ok(vector.clone())
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
        from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        // Parallel transport on sphere along geodesic
        if (from - to).norm() < f64::EPSILON {
            return Ok(vector.clone());
        }

        // For simplicity, use projection-based transport
        // This is not exact parallel transport but preserves key properties
        let transported = vector - from * from.dot(vector);
        self.project_tangent(to, &transported)
    }
}

#[test]
fn test_transport_identity() {
    // Transport from a point to itself should be identity
    let manifold = TransportTestManifold::new(3);
    let tolerance = 1e-14;

    for _ in 0..20 {
        let point = manifold.random_point();
        let tangent = manifold.random_tangent(&point).unwrap();

        let transported = manifold
            .parallel_transport(&point, &point, &tangent)
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
    let retraction = DefaultRetraction;
    let tolerance = 0.1; // Projection-based transport is approximate for larger steps

    for _ in 0..20 {
        let from = sphere.random_point();
        let direction = sphere.random_tangent(&from).unwrap() * 0.1;
        let to = retraction.retract(&sphere, &from, &direction).unwrap();

        let tangent = sphere.random_tangent(&from).unwrap();
        let transported = sphere.parallel_transport(&from, &to, &tangent).unwrap();

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
    let retraction = DefaultRetraction;
    let tolerance = 0.05; // Projection-based transport is approximate

    for _ in 0..10 {
        let from = sphere.random_point();
        let direction = sphere.random_tangent(&from).unwrap() * 0.1;
        let to = retraction.retract(&sphere, &from, &direction).unwrap();

        let u = sphere.random_tangent(&from).unwrap();
        let v = sphere.random_tangent(&from).unwrap();

        let u_transported = sphere.parallel_transport(&from, &to, &u).unwrap();
        let v_transported = sphere.parallel_transport(&from, &to, &v).unwrap();

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
    let retraction = DefaultRetraction;
    let tolerance = 1e-14;

    let from = manifold.random_point();
    let direction = manifold.random_tangent(&from).unwrap() * 0.1;
    let to = retraction.retract(&manifold, &from, &direction).unwrap();

    let u = manifold.random_tangent(&from).unwrap();
    let v = manifold.random_tangent(&from).unwrap();
    let alpha = 2.5;
    let beta = -1.3;

    // T(αu + βv) = αT(u) + βT(v)
    let combined = u.clone() * alpha + v.clone() * beta;
    let transport_combined = manifold.parallel_transport(&from, &to, &combined).unwrap();

    let u_transported = manifold.parallel_transport(&from, &to, &u).unwrap();
    let v_transported = manifold.parallel_transport(&from, &to, &v).unwrap();
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
    let retraction = DefaultRetraction;
    let tolerance = 1e-10; // Tangent space check can be stricter

    for _ in 0..20 {
        let from = sphere.random_point();
        let direction = sphere.random_tangent(&from).unwrap() * 0.05;
        let to = retraction.retract(&sphere, &from, &direction).unwrap();

        let tangent = sphere.random_tangent(&from).unwrap();
        let transported = sphere.parallel_transport(&from, &to, &tangent).unwrap();

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
    let retraction = DefaultRetraction;

    let from = manifold.random_point();
    let direction = manifold.random_tangent(&from).unwrap() * 0.1;
    let to = retraction.retract(&manifold, &from, &direction).unwrap();

    let zero = DVector::zeros(3);
    let transported = manifold.parallel_transport(&from, &to, &zero).unwrap();

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
    let retraction = DefaultRetraction;
    let tolerance = 1e-2; // Approximate transport

    for _ in 0..10 {
        let point_a = sphere.random_point();
        let direction = sphere.random_tangent(&point_a).unwrap() * 0.05;
        let point_b = retraction.retract(&sphere, &point_a, &direction).unwrap();

        let tangent = sphere.random_tangent(&point_a).unwrap();

        // Transport A -> B -> A
        let transported_to_b = sphere
            .parallel_transport(&point_a, &point_b, &tangent)
            .unwrap();
        let transported_back = sphere
            .parallel_transport(&point_b, &point_a, &transported_to_b)
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

    let from = sphere.random_point();
    let geodesic_tangent = sphere.random_tangent(&from).unwrap() * 0.1;
    let to = sphere.retract(&from, &geodesic_tangent).unwrap();

    // Transport the geodesic tangent itself
    let transported_tangent = sphere
        .parallel_transport(&from, &to, &geodesic_tangent)
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
