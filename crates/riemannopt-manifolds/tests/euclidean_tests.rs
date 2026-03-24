//! Integration tests for the Euclidean manifold

use approx::assert_relative_eq;
use riemannopt_core::linalg::{self, VectorOps};
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::Euclidean;

#[test]
fn test_euclidean_basic_properties() {
	let euclidean = Euclidean::<f64>::new(5).unwrap();

	assert_eq!(euclidean.dimension(), 5);
	assert_eq!(euclidean.name(), "Euclidean");
	assert_eq!(euclidean.dim(), 5);
}

#[test]
fn test_euclidean_creation() {
	// Valid cases
	let e1 = Euclidean::<f64>::new(1).unwrap();
	assert_eq!(e1.dimension(), 1);

	let e100 = Euclidean::<f64>::new(100).unwrap();
	assert_eq!(e100.dimension(), 100);

	// Invalid case
	assert!(Euclidean::<f64>::new(0).is_err());
}

#[test]
fn test_euclidean_projection() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0]);
	let mut proj: linalg::Vec<f64> = VectorOps::zeros(3);
	euclidean.project_point(&x, &mut proj);

	// Projection is identity in Euclidean space
	let diff = VectorOps::sub(&proj, &x);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);
}

#[test]
fn test_euclidean_tangent_projection() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.5, -1.0, 2.0]);
	let mut v_tangent: linalg::Vec<f64> = VectorOps::zeros(3);

	euclidean.project_tangent(&x, &v, &mut v_tangent).unwrap();

	// Tangent projection is identity in Euclidean space
	let diff = VectorOps::sub(&v_tangent, &v);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);
}

#[test]
fn test_euclidean_retraction() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.1, 0.2, 0.3]);
	let mut y: linalg::Vec<f64> = VectorOps::zeros(3);

	euclidean.retract(&x, &v, &mut y).unwrap();

	// Retraction is addition in Euclidean space
	let expected = VectorOps::add(&x, &v);
	let diff = VectorOps::sub(&y, &expected);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);

	// Zero retraction returns same point
	let zero: linalg::Vec<f64> = VectorOps::zeros(3);
	let mut x_recovered: linalg::Vec<f64> = VectorOps::zeros(3);
	euclidean.retract(&x, &zero, &mut x_recovered).unwrap();
	let diff = VectorOps::sub(&x, &x_recovered);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);
}

#[test]
fn test_euclidean_inner_product() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let u: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[4.0, 5.0, 6.0]);

	// Test inner product is standard dot product
	let inner_uv = euclidean.inner_product(&x, &u, &v).unwrap();
	assert_relative_eq!(inner_uv, u.dot(&v), epsilon = 1e-14);

	// Test symmetry
	let inner_vu = euclidean.inner_product(&x, &v, &u).unwrap();
	assert_relative_eq!(inner_uv, inner_vu, epsilon = 1e-14);

	// Test positive definiteness
	let inner_uu = euclidean.inner_product(&x, &u, &u).unwrap();
	assert!(inner_uu > 0.0);

	// Test orthogonality
	let e1: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let e2: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 1.0, 0.0]);
	let inner_e1e2 = euclidean.inner_product(&x, &e1, &e2).unwrap();
	assert_relative_eq!(inner_e1e2, 0.0, epsilon = 1e-14);
}

#[test]
fn test_euclidean_random_point() {
	let euclidean = Euclidean::<f64>::new(10).unwrap();

	for _ in 0..5 {
		let mut x: linalg::Vec<f64> = VectorOps::zeros(10);
		euclidean.random_point(&mut x).unwrap();

		// Point should have correct dimension and be on manifold
		assert_eq!(x.len(), 10);
		assert!(euclidean.is_point_on_manifold(&x, 1e-10));
	}
}

#[test]
fn test_euclidean_parallel_transport() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0]);
	let y: linalg::Vec<f64> = VectorOps::from_slice(&[4.0, 5.0, 6.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[0.1, 0.2, 0.3]);

	let mut transported: linalg::Vec<f64> = VectorOps::zeros(3);
	euclidean
		.parallel_transport(&x, &y, &v, &mut transported)
		.unwrap();

	// Parallel transport is identity in flat space
	let diff = VectorOps::sub(&transported, &v);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);

	// Norm preservation
	assert_relative_eq!(transported.norm(), v.norm(), epsilon = 1e-14);
}

#[test]
fn test_euclidean_euclidean_to_riemannian_gradient() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0]);
	let egrad: linalg::Vec<f64> = VectorOps::from_slice(&[0.5, -0.3, 0.7]);
	let mut rgrad: linalg::Vec<f64> = VectorOps::zeros(3);

	euclidean
		.euclidean_to_riemannian_gradient(&x, &egrad, &mut rgrad)
		.unwrap();

	// In Euclidean space, gradients are the same
	let diff = VectorOps::sub(&rgrad, &egrad);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);
}

#[test]
fn test_euclidean_is_point_on_manifold() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let good: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0, 3.0]);
	assert!(euclidean.is_point_on_manifold(&good, 1e-10));

	// Wrong dimension
	let bad: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 2.0]);
	assert!(!euclidean.is_point_on_manifold(&bad, 1e-10));
}

#[test]
fn test_euclidean_distance() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let y: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 1.0, 0.0]);

	let dist = euclidean.distance(&x, &y).unwrap();
	assert_relative_eq!(dist, std::f64::consts::SQRT_2, epsilon = 1e-14);

	// Distance to self is zero
	let dist_self = euclidean.distance(&x, &x).unwrap();
	assert_relative_eq!(dist_self, 0.0, epsilon = 1e-14);
}

#[test]
fn test_euclidean_is_flat() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();
	// Euclidean space does not declare is_flat as a standalone method,
	// but we can verify the flat-space behavior: parallel transport = identity
	let x: linalg::Vec<f64> = VectorOps::from_slice(&[0.0, 0.0, 0.0]);
	let y: linalg::Vec<f64> = VectorOps::from_slice(&[10.0, 10.0, 10.0]);
	let v: linalg::Vec<f64> = VectorOps::from_slice(&[1.0, 0.0, 0.0]);
	let mut transported: linalg::Vec<f64> = VectorOps::zeros(3);
	euclidean
		.parallel_transport(&x, &y, &v, &mut transported)
		.unwrap();
	let diff = VectorOps::sub(&transported, &v);
	assert_relative_eq!(diff.norm(), 0.0, epsilon = 1e-14);
}
