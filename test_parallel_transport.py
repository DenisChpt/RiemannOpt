#!/usr/bin/env python3
"""Test the parallel transport implementations for each manifold."""

import numpy as np
import riemannopt as ro


def test_spd_parallel_transport():
    """Test SPD manifold parallel transport with affine-invariant metric."""
    print("\n=== Testing SPD Parallel Transport ===")
    
    # Create SPD manifold
    spd = ro.SPD(3)
    
    # Create two SPD matrices
    X = np.eye(3) * 2.0  # 2*I
    Y = np.diag([3.0, 2.0, 1.0])  # Different SPD matrix
    
    # Create a tangent vector at X (symmetric matrix)
    V = np.array([[0.0, 0.5, 0.0],
                  [0.5, 0.0, 0.5],
                  [0.0, 0.5, 0.0]])
    
    # Convert to vectorized form
    x_vec = spd.matrix_to_vector(X)
    y_vec = spd.matrix_to_vector(Y)
    v_vec = spd.matrix_to_vector(V)
    
    # Test parallel transport
    transported = spd.parallel_transport(x_vec, y_vec, v_vec)
    transported_matrix = spd.vector_to_matrix(transported)
    
    print(f"Original tangent vector at X:\n{V}")
    print(f"\nTransported tangent vector at Y:\n{transported_matrix}")
    
    # Check that transported vector is symmetric
    assert np.allclose(transported_matrix, transported_matrix.T), "Transported vector should be symmetric"
    
    # Check that it's in tangent space at Y (any symmetric matrix is in tangent space)
    assert spd.is_vector_in_tangent_space(y_vec, transported, 1e-10), "Transported vector should be in tangent space at Y"
    
    print("✓ SPD parallel transport test passed")


def test_stiefel_parallel_transport():
    """Test Stiefel manifold parallel transport."""
    print("\n=== Testing Stiefel Parallel Transport ===")
    
    # Create Stiefel manifold St(5,2)
    stiefel = ro.Stiefel(5, 2)
    
    # Create two points on Stiefel manifold
    X = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 0.0]])
    
    # Create another orthonormal matrix
    Y_raw = np.array([[0.6, -0.8],
                      [0.8,  0.6],
                      [0.0,  0.0],
                      [0.0,  0.0],
                      [0.0,  0.0]])
    Y, _ = np.linalg.qr(Y_raw)  # Ensure orthonormality
    
    # Create tangent vector at X
    V = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0]])
    
    # Check V is in tangent space at X
    assert np.allclose(X.T @ V + V.T @ X, 0), "V should be in tangent space at X"
    
    # Convert to vectorized form
    x_vec = X.flatten('F')  # Column-major order
    y_vec = Y.flatten('F')
    v_vec = V.flatten('F')
    
    # Test parallel transport
    transported = stiefel.parallel_transport(x_vec, y_vec, v_vec)
    transported_matrix = transported.reshape((5, 2), order='F')
    
    print(f"Original tangent vector at X:\n{V}")
    print(f"\nTransported tangent vector at Y:\n{transported_matrix}")
    
    # Check that transported vector is in tangent space at Y
    tangent_check = Y.T @ transported_matrix + transported_matrix.T @ Y
    print(f"\nTangent space check (should be ~0):\n{tangent_check}")
    assert np.allclose(tangent_check, 0, atol=1e-10), "Transported vector should be in tangent space at Y"
    
    print("✓ Stiefel parallel transport test passed")


def test_sphere_parallel_transport():
    """Test sphere parallel transport (should preserve angles and lengths)."""
    print("\n=== Testing Sphere Parallel Transport ===")
    
    # Create sphere S^2 in R^3
    sphere = ro.Sphere(3)
    
    # Create two points on sphere
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    
    # Create tangent vector at x
    v = np.array([0.0, 0.0, 1.0])  # Points in z direction
    
    # Check it's tangent
    assert abs(np.dot(x, v)) < 1e-10, "v should be tangent to x"
    
    # Test parallel transport
    transported = sphere.parallel_transport(x, y, v)
    
    print(f"Original tangent vector at x: {v}")
    print(f"Transported tangent vector at y: {transported}")
    
    # Check transported vector is tangent at y
    assert abs(np.dot(y, transported)) < 1e-10, "Transported vector should be tangent to y"
    
    # Check length is preserved
    assert abs(np.linalg.norm(transported) - np.linalg.norm(v)) < 1e-10, "Length should be preserved"
    
    print("✓ Sphere parallel transport test passed")


def test_hyperbolic_parallel_transport():
    """Test hyperbolic manifold parallel transport."""
    print("\n=== Testing Hyperbolic Parallel Transport ===")
    
    # Create hyperbolic manifold H^2 (Poincaré disk)
    hyperbolic = ro.Hyperbolic(2)
    
    # Create two points in Poincaré disk
    x = np.array([0.3, 0.0])
    y = np.array([0.0, 0.4])
    
    # Create tangent vector at x
    v = np.array([0.0, 0.5])
    
    # Test parallel transport
    transported = hyperbolic.parallel_transport(x, y, v)
    
    print(f"Original tangent vector at x: {v}")
    print(f"Transported tangent vector at y: {transported}")
    
    # In Poincaré ball model, all vectors are valid tangent vectors
    # Check that result has correct dimension
    assert len(transported) == 2, "Transported vector should have correct dimension"
    
    # For small displacements, parallel transport should be approximately the identity
    x_close = np.array([0.31, 0.01])
    transported_close = hyperbolic.parallel_transport(x, x_close, v)
    assert np.allclose(transported_close, v, atol=0.1), "For small displacements, transport should be approximately identity"
    
    print("✓ Hyperbolic parallel transport test passed")


def test_grassmann_parallel_transport():
    """Test Grassmann manifold parallel transport."""
    print("\n=== Testing Grassmann Parallel Transport ===")
    
    # Create Grassmann manifold Gr(4,2)
    grassmann = ro.Grassmann(4, 2)
    
    # Create two points (orthonormal matrices representing subspaces)
    X = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0],
                  [0.0, 0.0]])
    
    Y = np.array([[0.6, -0.8],
                  [0.8,  0.6],
                  [0.0,  0.0],
                  [0.0,  0.0]])
    
    # Create horizontal tangent vector at X
    V = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]])
    
    # Check V is horizontal (X^T V = 0)
    assert np.allclose(X.T @ V, 0), "V should be in horizontal tangent space"
    
    # Convert to vectorized form
    x_vec = X.flatten('F')
    y_vec = Y.flatten('F')
    v_vec = V.flatten('F')
    
    # Test parallel transport
    transported = grassmann.parallel_transport(x_vec, y_vec, v_vec)
    transported_matrix = transported.reshape((4, 2), order='F')
    
    print(f"Original tangent vector at X:\n{V}")
    print(f"\nTransported tangent vector at Y:\n{transported_matrix}")
    
    # Check that transported vector is horizontal at Y
    horizontal_check = Y.T @ transported_matrix
    print(f"\nHorizontal check (should be ~0):\n{horizontal_check}")
    assert np.allclose(horizontal_check, 0, atol=1e-10), "Transported vector should be horizontal at Y"
    
    print("✓ Grassmann parallel transport test passed")


if __name__ == "__main__":
    print("Testing Parallel Transport Implementations")
    print("="*50)
    
    test_sphere_parallel_transport()
    test_stiefel_parallel_transport()
    test_grassmann_parallel_transport()
    test_spd_parallel_transport()
    test_hyperbolic_parallel_transport()
    
    print("\n" + "="*50)
    print("All parallel transport tests passed! ✓")