//! Integration tests for the SPD manifold

use approx::assert_relative_eq;
use nalgebra::DMatrix;
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::SPD;

#[test]
fn test_spd_basic_properties() {
	let spd = SPD::<f64>::new(3).unwrap();

	// dim(S++(n)) = n(n+1)/2
	assert_eq!(spd.dimension(), 3 * 4 / 2); // 6
	assert_eq!(spd.name(), "SPD");
	assert_eq!(spd.matrix_dim(), 3);
}

#[test]
fn test_spd_creation() {
	// Valid cases
	let spd1 = SPD::<f64>::new(1).unwrap();
	assert_eq!(spd1.dimension(), 1);

	let spd4 = SPD::<f64>::new(4).unwrap();
	assert_eq!(spd4.dimension(), 10);

	// Invalid case
	assert!(SPD::<f64>::new(0).is_err());
}

#[test]
fn test_spd_projection() {
	let spd = SPD::<f64>::new(3).unwrap();

	// Create a non-SPD matrix (not symmetric, not positive definite)
	let a = DMatrix::from_fn(3, 3, |i, j| ((i + 1) as f64) * 0.3 + ((j + 1) as f64) * 0.2);

	let mut proj = DMatrix::zeros(3, 3);
	spd.project_point(&a, &mut proj);

	// Result should be on manifold
	assert!(spd.is_point_on_manifold(&proj, 1e-8));

	// Result should be symmetric
	assert_relative_eq!(
		&proj - proj.transpose(),
		DMatrix::zeros(3, 3),
		epsilon = 1e-14
	);

	// Result should be positive definite
	let eigen = proj.clone().symmetric_eigen();
	for &eval in eigen.eigenvalues.iter() {
		assert!(eval > 0.0);
	}
}

#[test]
fn test_spd_tangent_projection() {
	let spd = SPD::<f64>::new(3).unwrap();

	// Create an SPD point
	let mut p = DMatrix::zeros(3, 3);
	spd.random_point(&mut p).unwrap();

	// Create an arbitrary (non-symmetric) matrix
	let v = DMatrix::from_fn(3, 3, |i, j| {
		((i as f64) - 1.0) * 0.2 + ((j as f64) - 1.0) * 0.1
	});

	let mut v_tangent = DMatrix::zeros(3, 3);
	spd.project_tangent(&p, &v, &mut v_tangent).unwrap();

	// Tangent vector should be symmetric
	let symmetry_error = (&v_tangent - v_tangent.transpose()).norm();
	assert_relative_eq!(symmetry_error, 0.0, epsilon = 1e-14);
}

#[test]
fn test_spd_retraction() {
	let spd = SPD::<f64>::new(3).unwrap();

	// Create an SPD point
	let mut p = DMatrix::zeros(3, 3);
	spd.random_point(&mut p).unwrap();

	// Create a symmetric tangent vector (scaled small)
	let mut v = DMatrix::zeros(3, 3);
	spd.random_tangent(&p, &mut v).unwrap();
	v *= 0.01; // small step

	let mut q = DMatrix::zeros(3, 3);
	spd.retract(&p, &v, &mut q).unwrap();

	// Result should be on manifold (symmetric, positive definite)
	assert!(spd.is_point_on_manifold(&q, 1e-8));

	// Zero retraction should return same point
	let zero = DMatrix::zeros(3, 3);
	let mut p_recovered = DMatrix::zeros(3, 3);
	spd.retract(&p, &zero, &mut p_recovered).unwrap();
	assert_relative_eq!(p, p_recovered, epsilon = 1e-10);
}

#[test]
fn test_spd_inner_product() {
	let spd = SPD::<f64>::new(3).unwrap();

	let mut p = DMatrix::zeros(3, 3);
	spd.random_point(&mut p).unwrap();

	let mut u = DMatrix::zeros(3, 3);
	spd.random_tangent(&p, &mut u).unwrap();

	let mut v = DMatrix::zeros(3, 3);
	spd.random_tangent(&p, &mut v).unwrap();

	// Test symmetry
	let inner_uv = spd.inner_product(&p, &u, &v).unwrap();
	let inner_vu = spd.inner_product(&p, &v, &u).unwrap();
	assert_relative_eq!(inner_uv, inner_vu, epsilon = 1e-10);

	// Test positive definiteness
	let inner_uu = spd.inner_product(&p, &u, &u).unwrap();
	assert!(inner_uu >= 0.0);
}

#[test]
fn test_spd_random_point() {
	let spd = SPD::<f64>::new(4).unwrap();

	for _ in 0..5 {
		let mut p = DMatrix::zeros(4, 4);
		spd.random_point(&mut p).unwrap();

		// Check point is on manifold
		assert!(spd.is_point_on_manifold(&p, 1e-8));

		// Check symmetry
		let symmetry_error = (&p - p.transpose()).norm();
		assert_relative_eq!(symmetry_error, 0.0, epsilon = 1e-14);

		// Check positive definiteness
		let eigen = p.clone().symmetric_eigen();
		for &eval in eigen.eigenvalues.iter() {
			assert!(eval > 0.0);
		}
	}
}

#[test]
fn test_spd_euclidean_to_riemannian_gradient() {
	let spd = SPD::<f64>::new(3).unwrap();

	let mut p = DMatrix::zeros(3, 3);
	spd.random_point(&mut p).unwrap();

	// Create a symmetric euclidean gradient
	let egrad_raw = DMatrix::from_fn(3, 3, |i, j| {
		((i + 1) as f64) * 0.1 + ((j + 1) as f64) * 0.05
	});
	let egrad = (&egrad_raw + egrad_raw.transpose()) * 0.5;

	let mut rgrad = DMatrix::zeros(3, 3);
	spd.euclidean_to_riemannian_gradient(&p, &egrad, &mut rgrad)
		.unwrap();

	// Result should be symmetric (in tangent space)
	let symmetry_error = (&rgrad - rgrad.transpose()).norm();
	assert_relative_eq!(symmetry_error, 0.0, epsilon = 1e-10);
}

#[test]
fn test_spd_is_point_on_manifold() {
	let spd = SPD::<f64>::new(3).unwrap();

	// Identity is SPD
	let eye = DMatrix::identity(3, 3);
	assert!(spd.is_point_on_manifold(&eye, 1e-10));

	// Non-symmetric matrix is not on manifold
	let non_sym = DMatrix::from_fn(3, 3, |i, j| (i + j + 1) as f64);
	assert!(!spd.is_point_on_manifold(&non_sym, 1e-10));

	// Negative definite matrix is not on manifold
	let neg = -DMatrix::identity(3, 3);
	assert!(!spd.is_point_on_manifold(&neg, 1e-10));
}

#[test]
fn test_spd_parallel_transport() {
	let spd = SPD::<f64>::new(3).unwrap();

	// Use identity as a well-conditioned SPD point
	let p = DMatrix::<f64>::identity(3, 3);

	let mut v = DMatrix::zeros(3, 3);
	spd.random_tangent(&p, &mut v).unwrap();

	// Transport from P to P (self-transport) via trait method
	let mut transported = DMatrix::zeros(3, 3);
	Manifold::<f64>::parallel_transport(&spd, &p, &p, &v, &mut transported).unwrap();

	// Transported vector should be symmetric (tangent space constraint)
	let symmetry_error = (&transported - transported.transpose()).norm();
	assert_relative_eq!(symmetry_error, 0.0, epsilon = 1e-8);

	// For identity matrix self-transport, E = I, so transport preserves vector
	let error = (&transported - &v).norm();
	assert_relative_eq!(error, 0.0, epsilon = 1e-8);
}
