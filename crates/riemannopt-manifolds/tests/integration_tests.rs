//! Integration tests for riemannopt-manifolds
//!
//! These tests verify that the manifolds work correctly with the core traits
//! and can be used independently of other crates.

use riemannopt_manifolds::{Sphere, Stiefel, Grassmann, SPD, Hyperbolic};
use riemannopt_core::{manifold::Manifold, memory::workspace::Workspace};
use nalgebra::{DVector, DMatrix};

#[test]
fn test_sphere_basic_operations() {
    let sphere = Sphere::<f64>::new(5).unwrap();
    let mut workspace = Workspace::<f64>::new();
    
    // Test random point generation
    let x = sphere.random_point();
    assert!((x.norm() - 1.0).abs() < 1e-10, "Random point not on sphere");
    
    // Test projection
    let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut projected = DVector::zeros(5);
    sphere.project_point(&y, &mut projected, &mut workspace);
    assert!((projected.norm() - 1.0).abs() < 1e-10, "Projected point not on sphere");
    
    // Test tangent space projection
    let v = DVector::from_vec(vec![0.1, 0.2, -0.1, 0.3, -0.2]);
    let mut tangent = DVector::zeros(5);
    sphere.project_tangent(&x, &v, &mut tangent, &mut workspace).unwrap();
    assert!(x.dot(&tangent).abs() < 1e-10, "Tangent vector not orthogonal to point");
}

#[test]
fn test_stiefel_basic_operations() {
    let stiefel = Stiefel::<f64>::new(5, 2).unwrap();
    let mut workspace = Workspace::<f64>::new();
    
    // Test random point generation
    let x = stiefel.random_point();
    let xtx = x.transpose() * &x;
    let identity = DMatrix::<f64>::identity(2, 2);
    assert!((xtx - identity).norm() < 1e-10, "X^T X != I");
    
    // Test tangent space projection
    let v = DMatrix::from_fn(5, 2, |i, j| 0.1 * (i as f64 - j as f64));
    let mut tangent = v.clone();
    stiefel.project_tangent(&x, &v, &mut tangent, &mut workspace).unwrap();
    
    // Check tangent space constraint: X^T Z + Z^T X = 0
    let xtz = x.transpose() * &tangent;
    let skew_check = &xtz + &xtz.transpose();
    assert!(skew_check.norm() < 1e-10, "Tangent not in tangent space");
}

#[test]
fn test_grassmann_basic_operations() {
    let grassmann = Grassmann::<f64>::new(6, 3).unwrap();
    let mut workspace = Workspace::<f64>::new();
    
    // Test random point generation
    let y = grassmann.random_point();
    let yty = y.transpose() * &y;
    let identity = DMatrix::<f64>::identity(3, 3);
    assert!((yty - identity).norm() < 1e-10, "Y^T Y != I");
    
    // Test retraction
    let mut delta = DMatrix::zeros(6, 3);
    grassmann.random_tangent(&y, &mut delta, &mut workspace).unwrap();
    delta *= 0.1;
    
    let mut y_new = DMatrix::zeros(6, 3);
    grassmann.retract(&y, &delta, &mut y_new, &mut workspace).unwrap();
    assert!(grassmann.is_point_on_manifold(&y_new, 1e-10));
}

#[test]
fn test_spd_basic_operations() {
    let spd = SPD::<f64>::new(3).unwrap();
    let mut workspace = Workspace::<f64>::new();
    
    // Test random point generation
    let p = spd.random_point();
    let eigenvalues = p.clone().symmetric_eigen().eigenvalues;
    assert!(eigenvalues.iter().all(|&lambda| lambda > 0.0), "Not positive definite");
    
    // Test projection of non-SPD matrix
    let non_spd = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        2.0, -1.0, 0.0,
        3.0, 0.0, 1.0
    ]);
    let mut projected = non_spd.clone();
    spd.project_point(&non_spd, &mut projected, &mut workspace);
    let proj_eigenvalues = projected.clone().symmetric_eigen().eigenvalues;
    assert!(proj_eigenvalues.iter().all(|&lambda| lambda > 1e-10), "Projection not SPD");
}

#[test]
fn test_hyperbolic_basic_operations() {
    let hyperbolic = Hyperbolic::<f64>::new(3).unwrap();
    let mut workspace = Workspace::<f64>::new();
    
    // Test random point generation
    let x = hyperbolic.random_point();
    assert!(x.norm() < 1.0 - 1e-6, "Point not in Poincaré ball");
    
    // Test retraction
    let v = DVector::from_vec(vec![0.1, 0.05, -0.1]);
    let mut x_new = DVector::zeros(3);
    hyperbolic.retract(&x, &v, &mut x_new, &mut workspace).unwrap();
    assert!(x_new.norm() < 1.0 - 1e-6, "Retracted point not in ball");
}

// TODO: This test is commented out because Product manifold requires all component
// manifolds to have the same Point and TangentVector types. Sphere uses DVector
// while SPD uses DMatrix, making them incompatible.
// #[test]
// fn test_product_manifold() {
//     let sphere = Sphere::<f64>::new(3).unwrap();
//     let spd = SPD::<f64>::new(2).unwrap();
//     let product = Product::new(vec![Box::new(sphere), Box::new(spd)]);
//     let mut workspace = Workspace::<f64>::new();
//     
//     // Test dimensions
//     assert_eq!(product.dimension(), 2 + 3); // S^2 + SPD(2)
//     
//     // Test random point
//     let x = product.random_point();
//     assert!(product.is_point_on_manifold(&x, 1e-10));
//     
//     // Test tangent space
//     let mut v = DVector::zeros(product.dimension());
//     product.random_tangent(&x, &mut v, &mut workspace).unwrap();
//     assert!(product.is_vector_in_tangent_space(&x, &v, 1e-10));
// }

#[test]
fn test_manifold_properties() {
    // Test dimension calculations
    assert_eq!(Sphere::<f64>::new(10).unwrap().dimension(), 9);
    assert_eq!(Stiefel::<f64>::new(5, 2).unwrap().dimension(), 6); // 5*2 - 2*3/2
    assert_eq!(Grassmann::<f64>::new(5, 2).unwrap().dimension(), 6); // 2*(5-2)
    assert_eq!(SPD::<f64>::new(3).unwrap().dimension(), 6); // 3*(3+1)/2
    assert_eq!(Hyperbolic::<f64>::new(4).unwrap().dimension(), 4);
}

#[test]
fn test_distance_functions() {
    let mut workspace = Workspace::<f64>::new();
    
    // Sphere distance
    let sphere = Sphere::<f64>::new(3).unwrap();
    let x1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let x2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
    let dist = sphere.distance(&x1, &x2, &mut workspace).unwrap();
    assert!((dist - std::f64::consts::PI / 2.0).abs() < 1e-10);
    
    // SPD distance should be symmetric
    let spd = SPD::<f64>::new(2).unwrap();
    let p1 = spd.random_point();
    let p2 = spd.random_point();
    let d12 = <SPD<f64> as Manifold<f64>>::distance(&spd, &p1, &p2, &mut workspace).unwrap();
    let d21 = <SPD<f64> as Manifold<f64>>::distance(&spd, &p2, &p1, &mut workspace).unwrap();
    assert!((d12 - d21).abs() < 1e-10, "Distance not symmetric");
}

#[test]
fn test_parallel_transport() {
    let mut workspace = Workspace::<f64>::new();
    
    // Test on sphere
    let sphere = Sphere::<f64>::new(5).unwrap();
    let x = sphere.random_point();
    let y = sphere.random_point();
    let mut v = DVector::zeros(5);
    sphere.random_tangent(&x, &mut v, &mut workspace).unwrap();
    
    let mut v_transported = DVector::zeros(5);
    <Sphere<f64> as Manifold<f64>>::parallel_transport(&sphere, &x, &y, &v, &mut v_transported, &mut workspace).unwrap();
    
    // Transported vector should be in tangent space at y
    assert!(y.dot(&v_transported).abs() < 1e-10, "Transported vector not in tangent space");
    
    // Norm should be preserved (for sphere with canonical metric)
    assert!((v.norm() - v_transported.norm()).abs() < 1e-10, "Norm not preserved");
}