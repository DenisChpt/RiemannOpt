//! Integration tests for the Grassmann manifold

use approx::assert_relative_eq;
use riemannopt_core::linalg::{self, MatrixOps};
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::Grassmann;

#[test]
fn test_grassmann_basic_properties() {
	let gr = Grassmann::<f64>::new(5, 2).unwrap();

	// dim(Gr(n,p)) = p(n-p)
	assert_eq!(gr.dimension(), 2 * (5 - 2)); // 6
	assert_eq!(gr.name(), "Grassmann");
}

#[test]
fn test_grassmann_creation() {
	// Valid cases
	let gr52 = Grassmann::<f64>::new(5, 2).unwrap();
	assert_eq!(gr52.ambient_dim(), 5);
	assert_eq!(gr52.subspace_dim(), 2);
	assert_eq!(gr52.dimension(), 6);

	let gr41 = Grassmann::<f64>::new(4, 1).unwrap();
	assert_eq!(gr41.dimension(), 3); // projective space

	// Invalid cases
	assert!(Grassmann::<f64>::new(3, 3).is_err()); // p must be < n
	assert!(Grassmann::<f64>::new(3, 0).is_err()); // p must be > 0
	assert!(Grassmann::<f64>::new(2, 5).is_err()); // p >= n
}

#[test]
fn test_grassmann_projection() {
	let gr = Grassmann::<f64>::new(4, 2).unwrap();

	// Create a non-orthonormal matrix
	let a = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(4, 2, |i, j| {
		((i + 1) as f64) * 0.3 + ((j + 1) as f64) * 0.2
	});

	let mut proj = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	gr.project_point(&a, &mut proj);

	// Check Y^T Y = I
	let yty = MatrixOps::mat_mul(&MatrixOps::transpose(&proj), &proj);
	let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(2);
	let diff = MatrixOps::sub(&yty, &eye);
	assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-12);
}

#[test]
fn test_grassmann_tangent_projection() {
	let gr = Grassmann::<f64>::new(5, 2).unwrap();

	// Create a point on the manifold
	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_point(&mut y).unwrap();

	// Create an arbitrary matrix
	let z = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(5, 2, |i, j| {
		((i as f64) - 2.0) * 0.1 + ((j as f64) - 0.5) * 0.3
	});

	let mut z_tangent = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.project_tangent(&y, &z, &mut z_tangent).unwrap();

	// Check horizontality: Y^T Z = 0
	let ytz = MatrixOps::mat_mul(&MatrixOps::transpose(&y), &z_tangent);
	assert_relative_eq!(MatrixOps::norm(&ytz), 0.0, epsilon = 1e-12);
}

#[test]
fn test_grassmann_retraction() {
	let gr = Grassmann::<f64>::new(4, 2).unwrap();

	// Create a point on manifold
	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	gr.random_point(&mut y).unwrap();

	// Create a tangent vector
	let mut v = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	gr.random_tangent(&y, &mut v).unwrap();
	v.scale_mut(0.1);

	let mut result = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	gr.retract(&y, &v, &mut result).unwrap();

	// Result should be on manifold: Y^T Y = I
	let rtr = MatrixOps::mat_mul(&MatrixOps::transpose(&result), &result);
	let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(2);
	let diff = MatrixOps::sub(&rtr, &eye);
	assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-12);

	// Zero retraction returns same point
	let zero = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	let mut y_recovered = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	gr.retract(&y, &zero, &mut y_recovered).unwrap();
	let diff = MatrixOps::sub(&y, &y_recovered);
	assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-12);
}

#[test]
fn test_grassmann_inner_product() {
	let gr = Grassmann::<f64>::new(5, 2).unwrap();

	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_point(&mut y).unwrap();

	let mut u = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_tangent(&y, &mut u).unwrap();

	let mut v = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_tangent(&y, &mut v).unwrap();

	// Test symmetry
	let inner_uv = gr.inner_product(&y, &u, &v).unwrap();
	let inner_vu = gr.inner_product(&y, &v, &u).unwrap();
	assert_relative_eq!(inner_uv, inner_vu, epsilon = 1e-12);

	// Test positive definiteness
	let inner_uu = gr.inner_product(&y, &u, &u).unwrap();
	assert!(inner_uu >= 0.0);
}

#[test]
fn test_grassmann_random_point() {
	let gr = Grassmann::<f64>::new(6, 3).unwrap();

	for _ in 0..5 {
		let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(6, 3);
		gr.random_point(&mut y).unwrap();

		// Check Y^T Y = I
		let yty = MatrixOps::mat_mul(&MatrixOps::transpose(&y), &y);
		let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(3);
		let diff = MatrixOps::sub(&yty, &eye);
		assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-12);
	}
}

#[test]
fn test_grassmann_parallel_transport() {
	let gr = Grassmann::<f64>::new(5, 2).unwrap();

	let mut y1 = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_point(&mut y1).unwrap();

	let mut y2 = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_point(&mut y2).unwrap();

	let mut v = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_tangent(&y1, &mut v).unwrap();

	let mut transported = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.parallel_transport(&y1, &y2, &v, &mut transported)
		.unwrap();

	// Check transported vector is in tangent space at y2: Y2^T transported = 0
	let y2t_transported = MatrixOps::mat_mul(&MatrixOps::transpose(&y2), &transported);
	assert_relative_eq!(MatrixOps::norm(&y2t_transported), 0.0, epsilon = 1e-10);

	// Check norm is reasonable (should be preserved or close)
	assert!(MatrixOps::norm(&transported) > 0.0);
}

#[test]
fn test_grassmann_euclidean_to_riemannian_gradient() {
	let gr = Grassmann::<f64>::new(5, 2).unwrap();

	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.random_point(&mut y).unwrap();

	// Create an arbitrary euclidean gradient
	let egrad = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(5, 2, |i, j| {
		((i + 1) as f64) * 0.1 - ((j + 1) as f64) * 0.05
	});

	let mut rgrad = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(5, 2);
	gr.euclidean_to_riemannian_gradient(&y, &egrad, &mut rgrad)
		.unwrap();

	// Result should be in tangent space: Y^T rgrad = 0
	let ytr = MatrixOps::mat_mul(&MatrixOps::transpose(&y), &rgrad);
	assert_relative_eq!(MatrixOps::norm(&ytr), 0.0, epsilon = 1e-12);
}

#[test]
fn test_grassmann_is_point_on_manifold() {
	let gr = Grassmann::<f64>::new(4, 2).unwrap();

	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	gr.random_point(&mut y).unwrap();
	assert!(gr.is_point_on_manifold(&y, 1e-10));

	// A non-orthonormal matrix should not be on manifold
	let bad = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(4, 2, |i, j| (i + j) as f64);
	assert!(!gr.is_point_on_manifold(&bad, 1e-10));
}
