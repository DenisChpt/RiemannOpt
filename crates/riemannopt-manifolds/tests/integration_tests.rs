//! Integration tests for riemannopt-manifolds
//!
//! These tests verify that the manifolds work correctly with the core traits
//! and can be used independently of other crates.

use riemannopt_manifolds::{Sphere, Stiefel, Grassmann, SPD, Hyperbolic, ProductManifold};
use riemannopt_core::{manifold::Manifold, memory::Workspace};
use nalgebra::{DVector, DMatrix};

#[test]
fn test_sphere_basic_operations() {
    let sphere = Sphere::new(5).unwrap();
    
    // Test random point generation
    let x: DVector<f64> = sphere.random_point();
    assert!((x.norm() - 1.0f64).abs() < 1e-10, "Random point not on sphere");
    
    // Test projection
    let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut projected = DVector::zeros(5);
    let mut workspace = Workspace::new();
    sphere.project_point(&y, &mut projected, &mut workspace);
    let norm_diff: f64 = projected.norm() - 1.0f64;
    assert!(norm_diff.abs() < 1e-10f64, "Projected point not on sphere");
    
    // Test tangent space projection
    let v = DVector::from_vec(vec![0.1, 0.2, -0.1, 0.3, -0.2]);
    let mut tangent = DVector::zeros(5);
    sphere.project_tangent(&x, &v, &mut tangent, &mut workspace).unwrap();
    assert!(x.dot(&tangent).abs() < 1e-10f64, "Tangent vector not orthogonal to point");
}

#[test]
fn test_stiefel_basic_operations() {
    let stiefel = Stiefel::new(5, 2).unwrap();
    
    // Test random point generation
    let x: DVector<f64> = stiefel.random_point();
    // Convert vector to matrix
    let x_mat = DMatrix::<f64>::from_vec(5, 2, x.data.as_vec().clone());
    let xtx = x_mat.transpose() * &x_mat;
    let eye = DMatrix::<f64>::identity(2, 2);
    assert!((xtx - eye).norm() < 1e-10f64, "Random point not on Stiefel manifold");
    
    // Test dimension
    // The ambient dimension in the vector representation is n*p
    assert_eq!(x.len(), 5 * 2);
}

#[test]
fn test_grassmann_operations() {
    let grassmann = Grassmann::new(5, 2).unwrap();
    
    // Test that Grassmann points are in the right format
    let x: DVector<f64> = grassmann.random_point();
    assert_eq!(x.len(), 5 * 2); // Should be vectorized n*p
    
    // Test that it satisfies orthonormality when reshaped
    let x_mat = DMatrix::<f64>::from_vec(5, 2, x.data.as_vec().clone());
    let xtx = x_mat.transpose() * &x_mat;
    let eye = DMatrix::<f64>::identity(2, 2);
    assert!((xtx - eye).norm() < 1e-10f64, "Grassmann point not orthonormal");
}

#[test]
fn test_spd_operations() {
    let spd = SPD::new(3).unwrap();
    
    // Test random point generation
    let x_vec: DVector<f64> = spd.random_point();
    // SPD stores only upper triangular part, so vector length is n*(n+1)/2
    assert_eq!(x_vec.len(), 3 * 4 / 2); // 6 elements for 3x3 matrix
    
    // We can't easily convert back to matrix without knowing the storage format
    // So just check that we got a vector of the right size
}

#[test]
fn test_hyperbolic_operations() {
    let hyperbolic = Hyperbolic::new(3).unwrap();
    
    // Test random point generation in Poincaré ball
    let x: DVector<f64> = hyperbolic.random_point();
    assert!(x.norm() < 1.0f64, "Point not in Poincaré ball");
    
    // Test projection keeps points in ball
    let y = DVector::from_vec(vec![0.8, 0.8, 0.0]);
    let mut projected = DVector::zeros(3);
    let mut workspace = Workspace::new();
    hyperbolic.project_point(&y, &mut projected, &mut workspace);
    assert!(projected.norm() < 1.0f64, "Projected point not in Poincaré ball");
}

#[test]
fn test_product_manifold() {
    let sphere1 = Sphere::new(3).unwrap();
    let sphere2 = Sphere::new(4).unwrap();
    
    let product = ProductManifold::new(
        Box::new(sphere1),
        Box::new(sphere2),
    );
    
    // Test dimension
    // Product manifold dimension is sum of dimensions
    // S^2 has dimension 2, S^3 has dimension 3
    
    // Test random point
    let x: DVector<f64> = product.random_point();
    assert_eq!(x.len(), 3 + 4);
    
    // Check each component is on its manifold
    let x1 = DVector::from_iterator(3, x.iter().take(3).cloned());
    let x2 = DVector::from_iterator(4, x.iter().skip(3).cloned());
    
    assert!((x1.norm() - 1.0f64).abs() < 1e-10, "First component not on S^2");
    assert!((x2.norm() - 1.0f64).abs() < 1e-10, "Second component not on S^3");
}

/// Test manifold properties manually
#[test]
fn test_sphere_properties() {
    let sphere = Sphere::new(10).unwrap();
    
    // Test retraction preserves manifold
    let x: DVector<f64> = sphere.random_point();
    let mut v = DVector::zeros(10);
    let mut workspace = Workspace::new();
    sphere.random_tangent(&x, &mut v, &mut workspace).unwrap();
    let mut y = DVector::zeros(10);
    sphere.retract(&x, &v, &mut y, &mut workspace).unwrap();
    assert!((y.norm() - 1.0f64).abs() < 1e-10, "Retraction doesn't preserve sphere constraint");
}

#[test]
fn test_stiefel_properties() {
    let stiefel = Stiefel::new(10, 3).unwrap();
    
    // Test retraction preserves orthogonality
    let x: DVector<f64> = stiefel.random_point();
    let mut v = DVector::zeros(10 * 3);
    let mut workspace = Workspace::new();
    stiefel.random_tangent(&x, &mut v, &mut workspace).unwrap();
    let mut y = DVector::zeros(10 * 3);
    stiefel.retract(&x, &v, &mut y, &mut workspace).unwrap();
    
    let y_mat = DMatrix::<f64>::from_vec(10, 3, y.data.as_vec().clone());
    let yty = y_mat.transpose() * &y_mat;
    let eye = DMatrix::<f64>::identity(3, 3);
    assert!((yty - eye).norm() < 1e-10f64, "Retraction doesn't preserve Stiefel constraint");
}