//! Integration tests for the Stiefel manifold

use riemannopt_manifolds::Stiefel;
use riemannopt_core::manifold::Manifold;
use nalgebra::DMatrix;
use approx::assert_relative_eq;

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
    let a = DMatrix::from_fn(4, 2, |i, j| {
        ((i + 1) as f64) * 0.3 + ((j + 1) as f64) * 0.2
    });

    let mut proj = DMatrix::zeros(4, 2);
    stiefel.project_point(&a, &mut proj);

    // Check that X^T X = I
    let xtx = proj.transpose() * &proj;
    let eye = DMatrix::identity(2, 2);
    assert_relative_eq!(xtx, eye, epsilon = 1e-14);
}

#[test]
fn test_stiefel_tangent_projection() {
    let stiefel = Stiefel::<f64>::new(4, 2).unwrap();

    // Create an orthonormal point on the manifold
    let mut x = DMatrix::zeros(4, 2);
    stiefel.random_point(&mut x).unwrap();

    // Create a random tangent vector
    let z = DMatrix::from_fn(4, 2, |i, j| {
        ((i as f64) - 2.0) * 0.1 + ((j as f64) - 0.5) * 0.2
    });

    let mut z_tangent = DMatrix::zeros(4, 2);
    stiefel.project_tangent(&x, &z, &mut z_tangent).unwrap();

    // Check tangent space constraint: X^T Z + Z^T X = 0
    let xtz = x.transpose() * &z_tangent;
    let skew_check = &xtz + &xtz.transpose();
    assert_relative_eq!(skew_check.norm(), 0.0, epsilon = 1e-14);
}

#[test]
fn test_stiefel_retraction() {
    let stiefel = Stiefel::<f64>::new(3, 2).unwrap();

    // Create an orthonormal point
    let mut x = DMatrix::zeros(3, 2);
    stiefel.random_point(&mut x).unwrap();

    // Create a tangent vector
    let mut v = DMatrix::zeros(3, 2);
    stiefel.random_tangent(&x, &mut v).unwrap();

    // Scale it down for better numerical behavior
    v *= 0.1;

    let mut y = DMatrix::zeros(3, 2);
    stiefel.retract(&x, &v, &mut y).unwrap();

    // Check that result is on manifold: Y^T Y = I
    let yty = y.transpose() * &y;
    let eye = DMatrix::identity(2, 2);
    assert_relative_eq!(yty, eye, epsilon = 1e-14);

    // Test zero retraction
    let zero = DMatrix::zeros(3, 2);
    let mut x_recovered = DMatrix::zeros(3, 2);
    stiefel.retract(&x, &zero, &mut x_recovered).unwrap();
    assert_relative_eq!(x, x_recovered, epsilon = 1e-14);
}

#[test]
fn test_stiefel_inner_product() {
    let stiefel = Stiefel::<f64>::new(3, 2).unwrap();

    let mut x = DMatrix::zeros(3, 2);
    stiefel.random_point(&mut x).unwrap();

    let mut u = DMatrix::zeros(3, 2);
    stiefel.random_tangent(&x, &mut u).unwrap();

    let mut v = DMatrix::zeros(3, 2);
    stiefel.random_tangent(&x, &mut v).unwrap();

    // Test inner product (should be Frobenius inner product)
    let inner_uv = stiefel.inner_product(&x, &u, &v).unwrap();
    let expected = (u.transpose() * &v).trace();
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

    let mut x = DMatrix::zeros(4, 2);
    stiefel.random_point(&mut x).unwrap();

    let mut y = DMatrix::zeros(4, 2);
    stiefel.random_point(&mut y).unwrap();

    let mut v = DMatrix::zeros(4, 2);
    stiefel.random_tangent(&x, &mut v).unwrap();

    let transported = stiefel.parallel_transport(&x, &y, &v).unwrap();

    // Check transported vector is in tangent space at y
    let ytt = y.transpose() * &transported;
    let skew_check = &ytt + &ytt.transpose();
    assert_relative_eq!(skew_check.norm(), 0.0, epsilon = 1e-12);

    // Norm preservation depends on the transport implementation
    // For this manifold, we just check the result is reasonable
    assert!(transported.norm() > 0.0);
}

#[test]
fn test_stiefel_random_point() {
    let stiefel = Stiefel::<f64>::new(6, 3).unwrap();

    for _ in 0..5 {
        let mut x = DMatrix::zeros(6, 3);
        stiefel.random_point(&mut x).unwrap();

        // Check point is on manifold: X^T X = I
        let xtx = x.transpose() * &x;
        let eye = DMatrix::identity(3, 3);
        assert_relative_eq!(xtx, eye, epsilon = 1e-14);
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
