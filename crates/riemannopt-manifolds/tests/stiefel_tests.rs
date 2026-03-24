//! Integration tests for the Stiefel manifold

use approx::assert_relative_eq;
use riemannopt_core::linalg::{self, MatrixOps};
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::Stiefel;

#[test]
fn test_stiefel_basic_properties() {
	let stiefel = Stiefel::<f64>::new(5, 3).unwrap();

	// Test dimensions
	assert_eq!(stiefel.dimension(), 5 * 3 - 3 * 4 / 2); // 15 - 6 = 9
	assert_eq!(stiefel.name(), "Stiefel");
}

#[test]
fn test_stiefel_creation() {
	// Valid Stiefel manifolds
	let st32 = Stiefel::<f64>::new(3, 2).unwrap();
	assert_eq!(st32.rows(), 3);
	assert_eq!(st32.cols(), 2);
	assert_eq!(st32.dimension(), 3 * 2 - 2 * 3 / 2); // 6 - 3 = 3

	let st55 = Stiefel::<f64>::new(5, 5).unwrap();
	assert_eq!(st55.dimension(), 5 * 5 - 5 * 6 / 2); // 25 - 15 = 10

	// Invalid cases
	assert!(Stiefel::<f64>::new(2, 3).is_err()); // n < p
	assert!(Stiefel::<f64>::new(5, 0).is_err()); // p = 0
}

#[test]
fn test_stiefel_projection() {
	let stiefel = Stiefel::<f64>::new(4, 2).unwrap();

	// Create a random matrix
	let a = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(4, 2, |i, j| {
		((i + 1) as f64) * 0.3 + ((j + 1) as f64) * 0.2
	});

	let mut proj = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	stiefel.project_point(&a, &mut proj);

	// Check that X^T X = I
	let xtx = MatrixOps::mat_mul(&MatrixOps::transpose(&proj), &proj);
	let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(2);
	let diff = MatrixOps::sub(&xtx, &eye);
	assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-14);
}

#[test]
fn test_stiefel_tangent_projection() {
	let stiefel = Stiefel::<f64>::new(4, 2).unwrap();

	// Create an orthonormal point on the manifold
	let mut x = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	stiefel.random_point(&mut x).unwrap();

	// Create a random tangent vector
	let z = <linalg::Mat<f64> as MatrixOps<f64>>::from_fn(4, 2, |i, j| {
		((i as f64) - 2.0) * 0.1 + ((j as f64) - 0.5) * 0.2
	});

	let mut z_tangent = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	stiefel.project_tangent(&x, &z, &mut z_tangent).unwrap();

	// Check tangent space constraint: X^T Z + Z^T X = 0
	let xtz = MatrixOps::mat_mul(&MatrixOps::transpose(&x), &z_tangent);
	let skew_check = MatrixOps::add(&xtz, &MatrixOps::transpose(&xtz));
	assert_relative_eq!(MatrixOps::norm(&skew_check), 0.0, epsilon = 1e-14);
}

#[test]
fn test_stiefel_retraction() {
	let stiefel = Stiefel::<f64>::new(3, 2).unwrap();

	// Create an orthonormal point
	let mut x = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	stiefel.random_point(&mut x).unwrap();

	// Create a tangent vector
	let mut v = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	stiefel.random_tangent(&x, &mut v).unwrap();

	// Scale it down for better numerical behavior
	v.scale_mut(0.1);

	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	stiefel.retract(&x, &v, &mut y).unwrap();

	// Check that result is on manifold: Y^T Y = I
	let yty = MatrixOps::mat_mul(&MatrixOps::transpose(&y), &y);
	let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(2);
	let diff = MatrixOps::sub(&yty, &eye);
	assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-14);

	// Test zero retraction
	let zero = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	let mut x_recovered = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	stiefel.retract(&x, &zero, &mut x_recovered).unwrap();
	let diff = MatrixOps::sub(&x, &x_recovered);
	assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-14);
}

#[test]
fn test_stiefel_inner_product() {
	let stiefel = Stiefel::<f64>::new(3, 2).unwrap();

	let mut x = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	stiefel.random_point(&mut x).unwrap();

	let mut u = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	stiefel.random_tangent(&x, &mut u).unwrap();

	let mut v = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
	stiefel.random_tangent(&x, &mut v).unwrap();

	// Test inner product (should be Frobenius inner product)
	let inner_uv = stiefel.inner_product(&x, &u, &v).unwrap();
	let expected = MatrixOps::trace(&MatrixOps::mat_mul(&MatrixOps::transpose(&u), &v));
	assert_relative_eq!(inner_uv, expected, epsilon = 1e-14);

	// Test symmetry
	let inner_vu = stiefel.inner_product(&x, &v, &u).unwrap();
	assert_relative_eq!(inner_uv, inner_vu, epsilon = 1e-14);

	// Test positive definiteness
	let inner_uu = stiefel.inner_product(&x, &u, &u).unwrap();
	assert!(inner_uu >= 0.0);
}

#[test]
fn test_stiefel_parallel_transport() {
	let stiefel = Stiefel::<f64>::new(4, 2).unwrap();

	let mut x = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	stiefel.random_point(&mut x).unwrap();

	let mut y = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	stiefel.random_point(&mut y).unwrap();

	let mut v = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	stiefel.random_tangent(&x, &mut v).unwrap();

	let mut transported = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(4, 2);
	stiefel
		.parallel_transport(&x, &y, &v, &mut transported)
		.unwrap();

	// Check transported vector is in tangent space at y
	let ytt = MatrixOps::mat_mul(&MatrixOps::transpose(&y), &transported);
	let skew_check = MatrixOps::add(&ytt, &MatrixOps::transpose(&ytt));
	assert_relative_eq!(MatrixOps::norm(&skew_check), 0.0, epsilon = 1e-12);

	// Norm preservation depends on the transport implementation
	// For this manifold, we just check the result is reasonable
	assert!(MatrixOps::norm(&transported) > 0.0);
}

#[test]
fn test_stiefel_random_point() {
	let stiefel = Stiefel::<f64>::new(6, 3).unwrap();

	for _ in 0..5 {
		let mut x = <linalg::Mat<f64> as MatrixOps<f64>>::zeros(6, 3);
		stiefel.random_point(&mut x).unwrap();

		// Check point is on manifold: X^T X = I
		let xtx = MatrixOps::mat_mul(&MatrixOps::transpose(&x), &x);
		let eye = <linalg::Mat<f64> as MatrixOps<f64>>::identity(3);
		let diff = MatrixOps::sub(&xtx, &eye);
		assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-14);
	}
}

#[test]
fn test_stiefel_special_cases() {
	// St(n,1) should behave like sphere
	let st31 = Stiefel::<f64>::new(3, 1).unwrap();
	assert_eq!(st31.dimension(), 2); // Same as S^2

	// St(n,n) is orthogonal group
	let st33 = Stiefel::<f64>::new(3, 3).unwrap();
	assert_eq!(st33.dimension(), 3); // Dimension of SO(3)
}
