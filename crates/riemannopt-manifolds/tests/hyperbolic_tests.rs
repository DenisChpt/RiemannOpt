//! Integration tests for the Hyperbolic manifold

use approx::assert_relative_eq;
use nalgebra::DVector;
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::Hyperbolic;

#[test]
fn test_hyperbolic_basic_properties() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	assert_eq!(hyp.dimension(), 3);
	assert_eq!(hyp.name(), "Hyperbolic");
	assert_eq!(hyp.ambient_dim(), 3);
}

#[test]
fn test_hyperbolic_creation() {
	// Valid cases
	let h1 = Hyperbolic::<f64>::new(1).unwrap();
	assert_eq!(h1.dimension(), 1);

	let h10 = Hyperbolic::<f64>::new(10).unwrap();
	assert_eq!(h10.dimension(), 10);

	// Invalid case
	assert!(Hyperbolic::<f64>::new(0).is_err());
}

#[test]
fn test_hyperbolic_projection() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	// A point already inside the ball
	let x = DVector::from_vec(vec![0.1, 0.2, 0.3]);
	let mut proj = DVector::zeros(3);
	hyp.project_point(&x, &mut proj);
	assert!(hyp.is_point_on_manifold(&proj, 1e-6));

	// A point outside the ball should be projected inside
	let outside = DVector::from_vec(vec![10.0, 0.0, 0.0]);
	let mut proj_out = DVector::zeros(3);
	hyp.project_point(&outside, &mut proj_out);
	assert!(
		proj_out.norm() < 1.0,
		"Projected point should be inside ball"
	);
	assert!(hyp.is_point_on_manifold(&proj_out, 1e-6));
}

#[test]
fn test_hyperbolic_tangent_projection() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	// In the Poincare ball, all vectors are valid tangent vectors
	let x = DVector::from_vec(vec![0.1, 0.2, 0.3]);
	let v = DVector::from_vec(vec![1.0, -2.0, 0.5]);
	let mut v_tangent = DVector::zeros(3);

	hyp.project_tangent(&x, &v, &mut v_tangent).unwrap();

	// Tangent projection is identity in the Poincare ball
	assert_relative_eq!(v_tangent, v, epsilon = 1e-14);
}

#[test]
fn test_hyperbolic_retraction() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![0.1, 0.2, 0.0]);
	let v = DVector::from_vec(vec![0.05, -0.03, 0.01]);
	let mut y = DVector::zeros(3);

	hyp.retract(&x, &v, &mut y).unwrap();

	// Result should be inside the ball
	assert!(y.norm() < 1.0, "Retracted point should be in Poincare ball");
	assert!(hyp.is_point_on_manifold(&y, 1e-6));

	// Zero retraction returns same point
	let zero = DVector::zeros(3);
	let mut x_recovered = DVector::zeros(3);
	hyp.retract(&x, &zero, &mut x_recovered).unwrap();
	assert_relative_eq!(x, x_recovered, epsilon = 1e-14);
}

#[test]
fn test_hyperbolic_inner_product() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![0.1, 0.2, 0.0]);
	let u = DVector::from_vec(vec![1.0, 0.0, 0.0]);
	let v = DVector::from_vec(vec![0.0, 1.0, 0.0]);

	// Test symmetry
	let inner_uv = hyp.inner_product(&x, &u, &v).unwrap();
	let inner_vu = hyp.inner_product(&x, &v, &u).unwrap();
	assert_relative_eq!(inner_uv, inner_vu, epsilon = 1e-14);

	// Test positive definiteness
	let inner_uu = hyp.inner_product(&x, &u, &u).unwrap();
	assert!(inner_uu > 0.0);

	// The hyperbolic inner product should differ from Euclidean
	// <u,v>_x = lambda(x)^2 * <u,v>_E
	let lambda = 2.0 / (1.0 - x.norm_squared());
	let expected_uu = lambda * lambda * u.dot(&u);
	assert_relative_eq!(inner_uu, expected_uu, epsilon = 1e-12);
}

#[test]
fn test_hyperbolic_random_point() {
	let hyp = Hyperbolic::<f64>::new(5).unwrap();

	for _ in 0..10 {
		let mut x = DVector::zeros(5);
		hyp.random_point(&mut x).unwrap();

		// Point should be inside the ball
		assert!(x.norm() < 1.0, "Random point should be in Poincare ball");
		assert!(hyp.is_point_on_manifold(&x, 1e-6));
	}
}

#[test]
fn test_hyperbolic_parallel_transport() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![0.1, 0.2, 0.0]);
	let y = DVector::from_vec(vec![0.3, -0.1, 0.05]);
	let v = DVector::from_vec(vec![0.5, 0.0, 0.1]);

	let transported = hyp.parallel_transport(&x, &y, &v).unwrap();

	// Transported vector should have correct dimension
	assert_eq!(transported.len(), 3);

	// Vector should be non-zero (for non-zero input)
	assert!(transported.norm() > 0.0);
}

#[test]
fn test_hyperbolic_euclidean_to_riemannian_gradient() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![0.1, 0.2, 0.0]);
	let egrad = DVector::from_vec(vec![1.0, -0.5, 0.3]);
	let mut rgrad = DVector::zeros(3);

	hyp.euclidean_to_riemannian_gradient(&x, &egrad, &mut rgrad)
		.unwrap();

	// Riemannian gradient should be a scaled version of Euclidean gradient
	// grad_riem = (1 - K||x||^2)^2 / 4 * grad_euc, with K = -1
	let norm_sq = x.norm_squared();
	let factor = (1.0 + norm_sq) * (1.0 + norm_sq) / 4.0;
	let expected = &egrad * factor;
	assert_relative_eq!(rgrad, expected, epsilon = 1e-12);
}

#[test]
fn test_hyperbolic_is_point_on_manifold() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	// Point inside ball
	let inside = DVector::from_vec(vec![0.1, 0.2, 0.3]);
	assert!(hyp.is_point_on_manifold(&inside, 1e-6));

	// Point on boundary (should fail)
	let boundary = DVector::from_vec(vec![1.0, 0.0, 0.0]);
	assert!(!hyp.is_point_on_manifold(&boundary, 1e-6));

	// Point outside ball
	let outside = DVector::from_vec(vec![2.0, 0.0, 0.0]);
	assert!(!hyp.is_point_on_manifold(&outside, 1e-6));

	// Origin is in the ball
	let origin = DVector::zeros(3);
	assert!(hyp.is_point_on_manifold(&origin, 1e-6));
}

#[test]
fn test_hyperbolic_distance() {
	let hyp = Hyperbolic::<f64>::new(3).unwrap();

	let x = DVector::from_vec(vec![0.1, 0.0, 0.0]);
	let y = DVector::from_vec(vec![-0.1, 0.0, 0.0]);

	let dist = hyp.distance(&x, &y).unwrap();
	assert!(dist > 0.0);

	// Distance to self is zero
	let dist_self = hyp.distance(&x, &x).unwrap();
	assert_relative_eq!(dist_self, 0.0, epsilon = 1e-14);

	// Symmetry
	let dist_yx = hyp.distance(&y, &x).unwrap();
	assert_relative_eq!(dist, dist_yx, epsilon = 1e-12);
}
