//! Test simple des manifolds avec les types associés

use riemannopt_manifolds::{Sphere, Stiefel};
use riemannopt_core::{Manifold, Workspace};
use nalgebra::{DVector, DMatrix};

fn main() {
    println!("=== Test de la sphère ===");
    
    // Créer une sphère de dimension 3
    let sphere = Sphere::new(3);
    println!("Dimension de la sphère : {}", sphere.dimension());
    
    // Créer un point sur la sphère
    let mut point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let mut workspace = Workspace::new();
    
    // Projeter le point sur la sphère
    let mut projected = point.clone();
    sphere.project_point(&point, &mut projected, &mut workspace);
    println!("Point projeté : {:?}", projected);
    
    // Vérifier que le point est sur la variété
    let is_on_manifold = sphere.is_point_on_manifold(&projected, 1e-10);
    println!("Le point est-il sur la sphère ? {}", is_on_manifold);
    
    println!("\n=== Test de Stiefel ===");
    
    // Créer une variété de Stiefel St(4, 2)
    let stiefel = Stiefel::new(4, 2);
    println!("Dimension de Stiefel : {}", stiefel.dimension());
    
    // Créer une matrice 4x2 avec colonnes orthonormées
    let mut mat = DMatrix::from_column_slice(4, 2, &[
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    ]);
    
    // Projeter sur Stiefel
    let mut proj_mat = mat.clone();
    stiefel.project_point(&mat, &mut proj_mat, &mut workspace);
    
    // Vérifier que c'est sur la variété
    let is_on_stiefel = stiefel.is_point_on_manifold(&proj_mat, 1e-10);
    println!("La matrice est-elle sur Stiefel ? {}", is_on_stiefel);
    
    // Vérifier l'orthonormalité
    let gram = proj_mat.transpose() * &proj_mat;
    println!("Matrice de Gram (devrait être identité) :");
    println!("{:?}", gram);
}