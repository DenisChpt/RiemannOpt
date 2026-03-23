//! Integration tests for the Euclidean manifold

use approx::assert_relative_eq;
use nalgebra::DVector;
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

	let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
	let mut proj = DVector::zeros(3);
	euclidean.project_point(&x, &mut proj);

	// Projection is identity in Euclidean space
	assert_relative_eq!(proj, x, epsilon = 1e-14);
}

#[test]
fn test_euclidean_tangent_projection() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
	let v = DVector::from_vec(vec![0.5, -1.0, 2.0]);
	let mut v_tangent = DVector::zeros(3);

	euclidean.project_tangent(&x, &v, &mut v_tangent).unwrap();

	// Tangent projection is identity in Euclidean space
	assert_relative_eq!(v_tangent, v, epsilon = 1e-14);
}

#[test]
fn test_euclidean_retraction() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
	let v = DVector::from_vec(vec![0.1, 0.2, 0.3]);
	let mut y = DVector::zeros(3);

	euclidean.retract(&x, &v, &mut y).unwrap();

	// Retraction is addition in Euclidean space
	let expected = &x + &v;
	assert_relative_eq!(y, expected, epsilon = 1e-14);

	// Zero retraction returns same point
	let zero = DVector::zeros(3);
	let mut x_recovered = DVector::zeros(3);
	euclidean.retract(&x, &zero, &mut x_recovered).unwrap();
	assert_relative_eq!(x, x_recovered, epsilon = 1e-14);
}

#[test]
fn test_euclidean_inner_product() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
	let u = DVector::from_vec(vec![1.0, 2.0, 3.0]);
	let v = DVector::from_vec(vec![4.0, 5.0, 6.0]);

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
	let e1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
	let e2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
	let inner_e1e2 = euclidean.inner_product(&x, &e1, &e2).unwrap();
	assert_relative_eq!(inner_e1e2, 0.0, epsilon = 1e-14);
}

#[test]
fn test_euclidean_random_point() {
	let euclidean = Euclidean::<f64>::new(10).unwrap();

	for _ in 0..5 {
		let mut x = DVector::zeros(10);
		euclidean.random_point(&mut x).unwrap();

		// Point should have correct dimension and be on manifold
		assert_eq!(x.len(), 10);
		assert!(euclidean.is_point_on_manifold(&x, 1e-10));
	}
}

#[test]
fn test_euclidean_parallel_transport() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
	let y = DVector::from_vec(vec![4.0, 5.0, 6.0]);
	let v = DVector::from_vec(vec![0.1, 0.2, 0.3]);

	let mut transported = DVector::zeros(3);
	euclidean
		.parallel_transport(&x, &y, &v, &mut transported)
		.unwrap();

	// Parallel transport is identity in flat space
	assert_relative_eq!(transported, v, epsilon = 1e-14);

	// Norm preservation
	assert_relative_eq!(transported.norm(), v.norm(), epsilon = 1e-14);
}

#[test]
fn test_euclidean_euclidean_to_riemannian_gradient() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
	let egrad = DVector::from_vec(vec![0.5, -0.3, 0.7]);
	let mut rgrad = DVector::zeros(3);

	euclidean
		.euclidean_to_riemannian_gradient(&x, &egrad, &mut rgrad)
		.unwrap();

	// In Euclidean space, gradients are the same
	assert_relative_eq!(rgrad, egrad, epsilon = 1e-14);
}

#[test]
fn test_euclidean_is_point_on_manifold() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let good = DVector::from_vec(vec![1.0, 2.0, 3.0]);
	assert!(euclidean.is_point_on_manifold(&good, 1e-10));

	// Wrong dimension
	let bad = DVector::from_vec(vec![1.0, 2.0]);
	assert!(!euclidean.is_point_on_manifold(&bad, 1e-10));
}

#[test]
fn test_euclidean_distance() {
	let euclidean = Euclidean::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
	let y = DVector::from_vec(vec![0.0, 1.0, 0.0]);

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
	let x = DVector::from_vec(vec![0.0, 0.0, 0.0]);
	let y = DVector::from_vec(vec![10.0, 10.0, 10.0]);
	let v = DVector::from_vec(vec![1.0, 0.0, 0.0]);
	let mut transported = DVector::zeros(3);
	euclidean
		.parallel_transport(&x, &y, &v, &mut transported)
		.unwrap();
	assert_relative_eq!(transported, v, epsilon = 1e-14);
}
