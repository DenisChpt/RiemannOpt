//! Debug test for Conjugate Gradient tangent space issues

use riemannopt_core::{
    core::manifold::Manifold,
    error::Result,
    memory::workspace::Workspace,
    types::DVector,
};
use riemannopt_manifolds::Sphere;

#[test]
fn test_sphere_tangent_operations() -> Result<()> {
    let dim = 3;
    let sphere = Sphere::<f64>::new(dim)?;
    
    // Point on sphere
    let mut point = DVector::zeros(dim);
    point[0] = 1.0/f64::sqrt(3.0);
    point[1] = 1.0/f64::sqrt(3.0);
    point[2] = 1.0/f64::sqrt(3.0);
    
    // Create a tangent vector
    let mut v1 = DVector::zeros(dim);
    v1[0] = 1.0;
    v1[1] = -0.5;
    v1[2] = -0.5;
    
    // Project to tangent space
    let mut tangent_v1 = v1.clone();
    sphere.project_tangent(&point, &v1, &mut tangent_v1, &mut Workspace::new())?;
    
    println!("Point: {:?}", point);
    println!("Original v1: {:?}", v1);
    println!("Tangent v1: {:?}", tangent_v1);
    println!("x^T v1 = {}", point.dot(&tangent_v1));
    
    // Scale tangent vector
    let mut scaled_v1 = tangent_v1.clone();
    sphere.scale_tangent(&point, -1.0, &tangent_v1, &mut scaled_v1, &mut Workspace::new())?;
    
    println!("\nScaled v1 (-1.0): {:?}", scaled_v1);
    println!("x^T scaled_v1 = {}", point.dot(&scaled_v1));
    
    // Create another tangent vector
    let mut v2 = DVector::zeros(dim);
    v2[0] = -0.5;
    v2[1] = 1.0;
    v2[2] = -0.5;
    
    let mut tangent_v2 = v2.clone();
    sphere.project_tangent(&point, &v2, &mut tangent_v2, &mut Workspace::new())?;
    
    println!("\nTangent v2: {:?}", tangent_v2);
    println!("x^T v2 = {}", point.dot(&tangent_v2));
    
    // Add tangent vectors
    let mut sum = tangent_v1.clone();
    sphere.add_tangents(&point, &tangent_v1, &tangent_v2, &mut sum, &mut Workspace::new())?;
    
    println!("\nSum v1 + v2: {:?}", sum);
    println!("x^T sum = {}", point.dot(&sum));
    
    // Test gradient to Riemannian gradient conversion
    let euclidean_grad = DVector::from_vec(vec![2.0, 2.0, 2.0]);
    let mut riemannian_grad = euclidean_grad.clone();
    sphere.euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut riemannian_grad, &mut Workspace::new())?;
    
    println!("\nEuclidean gradient: {:?}", euclidean_grad);
    println!("Riemannian gradient: {:?}", riemannian_grad);
    println!("x^T riemannian_grad = {}", point.dot(&riemannian_grad));
    
    // Verify tangent space constraint
    assert!(point.dot(&tangent_v1).abs() < 1e-10, "v1 not in tangent space");
    assert!(point.dot(&tangent_v2).abs() < 1e-10, "v2 not in tangent space");
    assert!(point.dot(&sum).abs() < 1e-10, "sum not in tangent space");
    assert!(point.dot(&riemannian_grad).abs() < 1e-10, "Riemannian gradient not in tangent space");
    
    Ok(())
}