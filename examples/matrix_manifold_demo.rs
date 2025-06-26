//! Example demonstrating the use of MatrixManifold trait.
//!
//! This example shows how matrix manifolds can be used directly with matrix
//! operations, avoiding the overhead of vectorization.

use nalgebra::DMatrix;
use riemannopt_manifolds::{StiefelMatrix, SPDMatrix, GrassmannMatrix, ObliqueMatrix, MatrixManifold};
use riemannopt_core::memory::Workspace;

fn main() {
    println!("=== MatrixManifold Demo ===\n");

    // Example 1: Stiefel manifold with direct matrix operations
    println!("1. Stiefel Manifold St(5,3)");
    let stiefel = StiefelMatrix::new(5, 3).unwrap();
    let mut workspace = Workspace::new();

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

    // Perform retraction
    let mut y = DMatrix::zeros(5, 3);
    stiefel.retract(&x, &v, &mut y, &mut workspace).unwrap();
    println!("\nRetraction R_X(V):");
    println!("{:.4}", y);
    println!("Result on manifold? {}", stiefel.is_point_on_manifold(&y, 1e-10));

    // Compute distance
    let dist = stiefel.distance(&x, &y, &mut workspace).unwrap();
    println!("\nDistance d(X, Y) = {:.6}", dist);

    // Example 2: SPD manifold with direct matrix operations
    println!("\n\n2. SPD Manifold SPD(3)");
    let spd = SPDMatrix::new(3).unwrap();

    // Generate a random SPD matrix
    let p = spd.random_point();
    println!("Random SPD matrix P:");
    println!("{:.4}", p);
    println!("P is SPD? {}", spd.is_point_on_manifold(&p, 1e-10));

    // Generate a random tangent vector (symmetric matrix)
    let mut u = DMatrix::zeros(3, 3);
    spd.random_tangent(&p, &mut u, &mut workspace).unwrap();
    println!("\nRandom tangent vector U:");
    println!("{:.4}", u);

    // Compute inner product with affine-invariant metric
    let inner = spd.inner_product(&p, &u, &u).unwrap();
    println!("\nInner product <U,U>_P = {:.6}", inner);

    // Perform retraction
    let mut q = DMatrix::zeros(3, 3);
    spd.retract(&p, &u, &mut q, &mut workspace).unwrap();
    println!("\nRetraction R_P(U):");
    println!("{:.4}", q);
    println!("Result is SPD? {}", spd.is_point_on_manifold(&q, 1e-10));

    // Example 3: Grassmann manifold with direct matrix operations
    println!("\n\n3. Grassmann Manifold Gr(4,2)");
    let grassmann = GrassmannMatrix::new(4, 2).unwrap();
    
    // Generate a random point on the manifold
    let g = grassmann.random_point();
    println!("Random point G on Gr(4,2):");
    println!("{:.4}", g);
    println!("G^T G = I_2? {}", grassmann.is_point_on_manifold(&g, 1e-10));
    
    // Generate a horizontal tangent vector
    let mut h = DMatrix::zeros(4, 2);
    grassmann.random_tangent(&g, &mut h, &mut workspace).unwrap();
    println!("\nRandom tangent vector H:");
    println!("{:.4}", h);
    println!("G^T H = 0? {:.6}", (g.transpose() * &h).norm());
    
    // Example 4: Oblique manifold with direct matrix operations
    println!("\n\n4. Oblique Manifold OB(3,4)");
    let oblique = ObliqueMatrix::new(3, 4).unwrap();
    
    // Generate a random point
    let o: DMatrix<f64> = oblique.random_point();
    println!("Random point O on OB(3,4):");
    println!("{:.4}", o);
    
    // Check column norms
    print!("Column norms: ");
    for j in 0..4 {
        print!("{:.4} ", o.column(j).norm());
    }
    println!();
    
    // Example 5: Using MatrixManifold with optimization (conceptual)
    println!("\n\n5. Optimization Example (Conceptual)");
    
    // Suppose we have a cost function f: St(n,p) -> R
    let cost_fn = |x: &DMatrix<f64>| -> f64 {
        // Example: minimize ||X - A||_F^2 for some target A
        let a = DMatrix::from_fn(5, 3, |i, j| (i + j) as f64);
        (x - a).norm_squared()
    };

    // Compute Euclidean gradient
    let euclidean_grad = |x: &DMatrix<f64>| -> DMatrix<f64> {
        let a = DMatrix::from_fn(5, 3, |i, j| (i + j) as f64);
        2.0 * (x - a)
    };

    let x = stiefel.random_point();
    let grad_e = euclidean_grad(&x);
    
    // Convert to Riemannian gradient
    let mut grad_r = DMatrix::zeros(5, 3);
    stiefel.euclidean_to_riemannian_gradient(&x, &grad_e, &mut grad_r, &mut workspace).unwrap();
    
    println!("At point X:");
    println!("Cost = {:.6}", cost_fn(&x));
    println!("Euclidean gradient norm = {:.6}", grad_e.norm());
    println!("Riemannian gradient norm = {:.6}", grad_r.norm());

    // Take a gradient descent step
    let step_size = 0.1;
    let tangent = -step_size * &grad_r;
    let mut x_new = DMatrix::zeros(5, 3);
    stiefel.retract(&x, &tangent, &mut x_new, &mut workspace).unwrap();
    
    println!("\nAfter gradient step:");
    println!("New cost = {:.6}", cost_fn(&x_new));
    println!("Cost decreased: {}", cost_fn(&x_new) < cost_fn(&x));

    println!("\n=== Demo Complete ===");
}