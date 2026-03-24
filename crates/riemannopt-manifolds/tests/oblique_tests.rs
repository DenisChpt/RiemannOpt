//! Integration tests for the Oblique manifold

use approx::assert_relative_eq;
use riemannopt_core::linalg::{self, MatrixOps, VectorOps};
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::Oblique;

/// Helper to call trait methods with f64 type parameter.
fn as_manifold(
	ob: &Oblique,
) -> &dyn Manifold<f64, Point = linalg::Mat<f64>, TangentVector = linalg::Mat<f64>> {
	ob
}

#[test]
fn test_oblique_basic_properties() {
	let oblique = Oblique::new(4, 3).unwrap();
	let m = as_manifold(&oblique);

	// dim(OB(n,p)) = p(n-1)
	assert_eq!(m.dimension(), 3 * (4 - 1)); // 9
	assert_eq!(m.name(), "Oblique");
	assert_eq!(oblique.ambient_dim(), (4, 3));
	assert_eq!(oblique.manifold_dim(), 9);
}

#[test]
fn test_oblique_creation() {
	// Valid cases
	let ob21 = Oblique::new(2, 1).unwrap();
	assert_eq!(as_manifold(&ob21).dimension(), 1);

	let ob53 = Oblique::new(5, 3).unwrap();
	assert_eq!(as_manifold(&ob53).dimension(), 12);

	// Invalid cases
	assert!(Oblique::new(0, 3).is_err());
	assert!(Oblique::new(3, 0).is_err());
	assert!(Oblique::new(0, 0).is_err());
}

#[test]
fn test_oblique_projection() {
	let oblique = Oblique::new(4, 3).unwrap();
	let m = as_manifold(&oblique);

	// Create a matrix with non-unit columns
	let a: linalg::Mat<f64> =
		MatrixOps::from_fn(4, 3, |i, j| ((i + 1) as f64) * 0.5 + ((j + 1) as f64) * 0.3);

	let mut proj: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.project_point(&a, &mut proj);

	// Check each column has unit norm
	for j in 0..3 {
		let col: linalg::Vec<f64> = MatrixOps::column(&proj, j);
		assert_relative_eq!(col.norm(), 1.0, epsilon = 1e-14);
	}
}

#[test]
fn test_oblique_tangent_projection() {
	let oblique = Oblique::new(4, 3).unwrap();
	let m = as_manifold(&oblique);

	// Create a point on the manifold
	let mut x: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.random_point(&mut x).unwrap();

	// Create an arbitrary matrix
	let v: linalg::Mat<f64> = MatrixOps::from_fn(4, 3, |i, j| {
		((i as f64) - 1.5) * 0.2 + ((j as f64) - 1.0) * 0.1
	});

	let mut v_tangent: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.project_tangent(&x, &v, &mut v_tangent).unwrap();

	// Check orthogonality: x_j^T v_j = 0 for each column
	for j in 0..3 {
		let x_col: linalg::Vec<f64> = MatrixOps::column(&x, j);
		let vt_col: linalg::Vec<f64> = MatrixOps::column(&v_tangent, j);
		let inner = x_col.dot(&vt_col);
		assert_relative_eq!(inner, 0.0, epsilon = 1e-14);
	}
}

#[test]
fn test_oblique_retraction() {
	let oblique = Oblique::new(4, 3).unwrap();
	let m = as_manifold(&oblique);

	// Create a point on manifold
	let mut x: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.random_point(&mut x).unwrap();

	// Create a tangent vector
	let mut v: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.random_tangent(&x, &mut v).unwrap();
	v.scale_mut(0.1);

	let mut y: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.retract(&x, &v, &mut y).unwrap();

	// Result should have unit-norm columns
	for j in 0..3 {
		let col: linalg::Vec<f64> = MatrixOps::column(&y, j);
		assert_relative_eq!(col.norm(), 1.0, epsilon = 1e-14);
	}

	// Zero retraction returns same point
	let zero: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	let mut x_recovered: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.retract(&x, &zero, &mut x_recovered).unwrap();
	let diff = MatrixOps::sub(&x, &x_recovered);
	assert_relative_eq!(MatrixOps::norm(&diff), 0.0, epsilon = 1e-14);
}

#[test]
fn test_oblique_inner_product() {
	let oblique = Oblique::new(4, 3).unwrap();
	let m = as_manifold(&oblique);

	let mut x: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.random_point(&mut x).unwrap();

	let mut u: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.random_tangent(&x, &mut u).unwrap();

	let mut v: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.random_tangent(&x, &mut v).unwrap();

	// Test symmetry
	let inner_uv = m.inner_product(&x, &u, &v).unwrap();
	let inner_vu = m.inner_product(&x, &v, &u).unwrap();
	assert_relative_eq!(inner_uv, inner_vu, epsilon = 1e-14);

	// Test positive definiteness
	let inner_uu = m.inner_product(&x, &u, &u).unwrap();
	assert!(inner_uu >= 0.0);
}

#[test]
fn test_oblique_random_point() {
	let oblique = Oblique::new(5, 4).unwrap();
	let m = as_manifold(&oblique);

	for _ in 0..5 {
		let mut x: linalg::Mat<f64> = MatrixOps::zeros(5, 4);
		m.random_point(&mut x).unwrap();

		// Check each column has unit norm
		for j in 0..4 {
			let col: linalg::Vec<f64> = MatrixOps::column(&x, j);
			assert_relative_eq!(col.norm(), 1.0, epsilon = 1e-14);
		}

		assert!(m.is_point_on_manifold(&x, 1e-10));
	}
}

#[test]
fn test_oblique_parallel_transport() {
	let oblique = Oblique::new(4, 2).unwrap();

	let mut x: linalg::Mat<f64> = MatrixOps::zeros(4, 2);
	let mut y: linalg::Mat<f64> = MatrixOps::zeros(4, 2);
	let mut v: linalg::Mat<f64> = MatrixOps::zeros(4, 2);

	let m = as_manifold(&oblique);
	m.random_point(&mut x).unwrap();
	m.random_point(&mut y).unwrap();
	m.random_tangent(&x, &mut v).unwrap();

	let mut transported: linalg::Mat<f64> = MatrixOps::zeros(4, 2);
	m.parallel_transport(&x, &y, &v, &mut transported).unwrap();

	// Transported vector should be in tangent space at y:
	// y_j^T transported_j = 0 for each column
	for j in 0..2 {
		let y_col: linalg::Vec<f64> = MatrixOps::column(&y, j);
		let t_col: linalg::Vec<f64> = MatrixOps::column(&transported, j);
		let inner = y_col.dot(&t_col);
		assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
	}

	// Non-zero vector should transport to non-zero
	assert!(MatrixOps::norm(&transported) > 0.0);
}

#[test]
fn test_oblique_euclidean_to_riemannian_gradient() {
	let oblique = Oblique::new(4, 3).unwrap();
	let m = as_manifold(&oblique);

	let mut x: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.random_point(&mut x).unwrap();

	let egrad: linalg::Mat<f64> = MatrixOps::from_fn(4, 3, |i, j| {
		((i + 1) as f64) * 0.1 - ((j + 1) as f64) * 0.05
	});

	let mut rgrad: linalg::Mat<f64> = MatrixOps::zeros(4, 3);
	m.euclidean_to_riemannian_gradient(&x, &egrad, &mut rgrad)
		.unwrap();

	// Result should be in tangent space: x_j^T rgrad_j = 0
	for j in 0..3 {
		let x_col: linalg::Vec<f64> = MatrixOps::column(&x, j);
		let r_col: linalg::Vec<f64> = MatrixOps::column(&rgrad, j);
		let inner = x_col.dot(&r_col);
		assert_relative_eq!(inner, 0.0, epsilon = 1e-14);
	}
}

#[test]
fn test_oblique_is_point_on_manifold() {
	let oblique = Oblique::new(3, 2).unwrap();
	let m = as_manifold(&oblique);

	// Good point: columns have unit norm
	let mut x: linalg::Mat<f64> = MatrixOps::zeros(3, 2);
	m.random_point(&mut x).unwrap();
	assert!(m.is_point_on_manifold(&x, 1e-10));

	// Bad point: columns do not have unit norm
	let bad: linalg::Mat<f64> = MatrixOps::from_fn(3, 2, |i, j| (i + j + 1) as f64);
	assert!(!m.is_point_on_manifold(&bad, 1e-10));
}

#[test]
fn test_oblique_distance() {
	let oblique = Oblique::new(3, 2).unwrap();
	let m = as_manifold(&oblique);

	let mut x: linalg::Mat<f64> = MatrixOps::zeros(3, 2);
	m.random_point(&mut x).unwrap();

	let mut y: linalg::Mat<f64> = MatrixOps::zeros(3, 2);
	m.random_point(&mut y).unwrap();

	let dist: f64 = m.distance(&x, &y).unwrap();
	assert!(dist >= 0.0);

	// Distance to self is zero (relaxed epsilon due to arccos numerical precision)
	let dist_self: f64 = m.distance(&x, &x).unwrap();
	assert_relative_eq!(dist_self, 0.0, epsilon = 1e-7);

	// Symmetry
	let dist_yx: f64 = m.distance(&y, &x).unwrap();
	assert_relative_eq!(dist, dist_yx, epsilon = 1e-12);
}
