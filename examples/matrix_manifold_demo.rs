//! Example demonstrating the use of matrix-based manifolds.
//!
//! This example shows how matrix manifolds work directly with matrix
//! operations in the new unified API.

use nalgebra::DMatrix;
use riemannopt_manifolds::{Stiefel, SPD, Grassmann, Oblique};
use riemannopt_core::manifold::Manifold;
use riemannopt_core::memory::workspace::Workspace;

fn main() {
    println!("=== Matrix Manifold Demo ===\n");

    // Example 1: Stiefel manifold with direct matrix operations
    println!("1. Stiefel Manifold St(5,3)");
    let stiefel = Stiefel::<f64>::new(5, 3).unwrap();
    let mut workspace = Workspace::<f64>::new();

    // Generate a random point on the manifold
    let x = stiefel.random_point();
    println!("Random point X on St(5,3):");
    println!("{:.4}", x);
    println!("X^T X = I_3? {}", stiefel.is_point_on_manifold(&x, 1e-10));

    // Generate a random tangent vector
    let mut v = DMatrix::zeros(5, 3);
    stiefel.random_tangent(&x, &mut v, &mut workspace).unwrap();
    println!("\nRandom tangent vector V:");
    println!("{:.4}", v);
    println!("V in tangent space? {}", stiefel.is_vector_in_tangent_space(&x, &v, 1e-10));

    // Example 2: SPD manifold
    println!("\n\n2. Symmetric Positive Definite Manifold S⁺⁺(3)");
    let spd = SPD::<f64>::new(3).unwrap();

    let p = spd.random_point();
    println!("Random SPD matrix P:");
    println!("{:.4}", p);

    // Compute Riemannian gradient from Euclidean gradient
    let eucl_grad = DMatrix::from_fn(3, 3, |i, j| (i + j) as f64);
    let mut riem_grad = eucl_grad.clone();
    spd.euclidean_to_riemannian_gradient(&p, &eucl_grad, &mut riem_grad, &mut workspace).unwrap();
    println!("\nEuclidean gradient:");
    println!("{:.4}", eucl_grad);
    println!("Riemannian gradient:");
    println!("{:.4}", riem_grad);

    // Example 3: Grassmann manifold
    println!("\n\n3. Grassmann Manifold Gr(4,2)");
    let grassmann = Grassmann::<f64>::new(4, 2).unwrap();

    let y = grassmann.random_point();
    println!("Random point Y on Gr(4,2) (orthonormal basis):");
    println!("{:.4}", y);

    // Retraction example
    let mut delta = DMatrix::zeros(4, 2);
    grassmann.random_tangent(&y, &mut delta, &mut workspace).unwrap();
    delta *= 0.1; // Small step

    let mut y_new = DMatrix::zeros(4, 2);
    grassmann.retract(&y, &delta, &mut y_new, &mut workspace).unwrap();
    println!("\nAfter retraction with step 0.1:");
    println!("{:.4}", y_new);
    println!("Still on manifold? {}", grassmann.is_point_on_manifold(&y_new, 1e-10));

    // Example 4: Oblique manifold
    println!("\n\n4. Oblique Manifold OB(3,4)");
    let oblique = Oblique::new(3, 4).unwrap();

    let z = oblique.random_point();
    println!("Random point Z on OB(3,4) (unit-norm columns):");
    println!("{:.4}", z);

    // Verify column norms
    print!("Column norms: ");
    for j in 0..4 {
        print!("{:.4} ", z.column(j).norm());
    }
    println!();

    // Distance between two points
    let z2 = oblique.random_point();
    let dist = oblique.distance(&z, &z2, &mut workspace).unwrap();
    println!("\nDistance to another random point: {:.4}", dist);

    println!("\n=== Demo Complete ===");
}