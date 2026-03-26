//! Integration tests for the Sphere manifold

use approx::assert_relative_eq;
use riemannopt_core::linalg::{self, VectorOps, VectorView};
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::Sphere;

#[test]
fn test_sphere_basic_properties() {
	let sphere = Sphere::<f64>::new(3).unwrap();

	// Test dimensions
	assert_eq!(sphere.dimension(), 2);
	assert_eq!(sphere.ambient_dimension(), 3);
	assert_eq!(sphere.name(), "Sphere");
}

#[test]
fn test_sphere_projection() {
	let sphere = Sphere::<f64>::new(3).unwrap();

	// Test projection of a non-unit vector
	let x: linalg::Vec<f64> = VectorOps::from_slice(&[3.0, 4.0, 0.0]);
	let mut proj: linalg::Vec<f64> = VectorOps::zeros(3);
	sphere.project_point(&x, &mut proj);

	// Check result is on sphere
	assert_relative_eq!(proj.norm(), 1.0, epsilon = 1e-14);
	assert!((proj.get(0) - 0.6).abs() < 1e-14);
	assert!((proj.get(1) - 0.8).abs() < 1e-14);
}

#[test]
fn test_sphere_tangent_projection() {
	let sphere = Sphere::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.5, 1.0, 2.0]);
	let mut v_tangent: linalg::Vec<f64> = VectorOps::zeros(3);

	sphere
		.project_tangent(&x, &v, &mut v_tangent, &mut ())
		.unwrap();

	// Check orthogonality
	assert_relative_eq!(x.dot(&v_tangent), 0.0, epsilon = 1e-14);

	// Check projection formula: v_tangent = v - <v,x>x
	let xdv = x.dot(&v);
	let mut expected = x.clone();
	expected.scale_mut(xdv);
	let expected = VectorOps::sub(&v, &expected);
	let diff = VectorOps::sub(&v_tangent, &expected);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);
}

#[test]
fn test_sphere_retraction() {
	let sphere = Sphere::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 0.5, 0.0]);
	let mut y: linalg::Vec<f64> = VectorOps::zeros(3);

	// Test retraction
	sphere.retract(&x, &v, &mut y, &mut ()).unwrap();

	// Result should be on sphere
	assert_relative_eq!(y.norm(), 1.0, epsilon = 1e-14);

	// Test zero retraction returns same point
	let zero: linalg::Vec<f64> = VectorOps::zeros(3);
	let mut x_recovered: linalg::Vec<f64> = VectorOps::zeros(3);
	sphere
		.retract(&x, &zero, &mut x_recovered, &mut ())
		.unwrap();
	let diff = VectorOps::sub(&x, &x_recovered);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);
}

#[test]
fn test_sphere_inner_product() {
	let sphere = Sphere::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let u: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 1.0, 0.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 0.0, 1.0]);

	// Test inner product (should be standard Euclidean)
	let inner_uv = sphere.inner_product(&x, &u, &v, &mut ()).unwrap();
	assert_relative_eq!(inner_uv, 0.0, epsilon = 1e-14);

	let inner_uu = sphere.inner_product(&x, &u, &u, &mut ()).unwrap();
	assert_relative_eq!(inner_uu, 1.0, epsilon = 1e-14);
}

#[test]
fn test_sphere_random_point() {
	let sphere = Sphere::<f64>::new(10).unwrap();

	for _ in 0..10 {
		let x = sphere.random_point();

		// Check point is on sphere
		assert_relative_eq!(x.norm(), 1.0, epsilon = 1e-14);
	}
}

#[test]
fn test_sphere_parallel_transport() {
	let sphere = Sphere::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let y: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 1.0, 0.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 0.0, 1.0]);

	let transported = sphere.parallel_transport(&x, &y, &v, &mut ()).unwrap();

	// Check transported vector is in tangent space at y
	assert_relative_eq!(y.dot(&transported), 0.0, epsilon = 1e-14);

	// Check norm preservation
	assert_relative_eq!(transported.norm(), v.norm(), epsilon = 1e-14);
}
