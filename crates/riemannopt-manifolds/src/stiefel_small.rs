//! # Specialized Implementations for Small Stiefel Manifolds
//!
//! This module provides highly optimized, hand-coded implementations for common small
//! Stiefel manifold configurations, specifically St(3,2) and St(4,2). These specialized
//! implementations offer significant performance improvements over the generic Stiefel
//! manifold operations by eliminating dynamic memory allocation and leveraging compile-time
//! knowledge of matrix dimensions.
//!
//! ## Performance Benefits
//!
//! 1. **No Dynamic Allocation**: All operations work with fixed-size arrays
//! 2. **Unrolled Loops**: Matrix multiplications are fully unrolled
//! 3. **Cache Efficiency**: Small, fixed-size data structures fit in CPU cache
//! 4. **Compiler Optimization**: Known dimensions enable better optimization
//!
//! ## Supported Configurations
//!
//! - **St(3,2)**: 3×2 matrices with orthonormal columns (6 elements)
//! - **St(4,2)**: 4×2 matrices with orthonormal columns (8 elements)
//!
//! ## Memory Layout
//!
//! Matrices are stored in column-major order as contiguous arrays:
//! - For a 3×2 matrix X = [x₁ x₂], the array is [x₁₁, x₂₁, x₃₁, x₁₂, x₂₂, x₃₂]
//! - This layout matches BLAS conventions for optimal performance
//!
//! ## Use Cases
//!
//! These specialized functions are particularly useful in:
//! - Real-time systems with strict performance requirements
//! - Embedded systems with limited memory
//! - Inner loops of optimization algorithms
//! - GPU kernel implementations (as templates)

use riemannopt_core::types::Scalar;

/// Specialized projection to tangent space for St(3,2).
/// 
/// Computes the orthogonal projection of a matrix V onto the tangent space
/// of the Stiefel manifold at point X.
/// 
/// # Mathematical Formula
/// 
/// For X ∈ St(3,2) and V ∈ ℝ³ˣ², the projection is:
/// ```text
/// P_X(V) = V - X sym(X^T V)
/// ```
/// where sym(A) = (A + A^T)/2 is the symmetric part.
/// 
/// # Algorithm Details
/// 
/// 1. Compute X^T V (2×2 matrix multiplication)
/// 2. Symmetrize: S = (X^T V + V^T X)/2
/// 3. Project: result = V - X S
/// 
/// # Performance
/// 
/// - **Operations**: 36 multiplications, 30 additions
/// - **Memory**: Stack-allocated temporaries only
/// - **Vectorization**: Compiler may auto-vectorize the loops
#[inline(always)]
pub fn project_tangent_stiefel_3_2<T: Scalar>(
    x: &[T], // 3x2 matrix as column-major array of length 6
    v: &[T], // 3x2 matrix as column-major array of length 6
    result: &mut [T], // Output: 3x2 matrix as column-major array
) {
    debug_assert_eq!(x.len(), 6);
    debug_assert_eq!(v.len(), 6);
    debug_assert_eq!(result.len(), 6);
    
    // Compute X^T V (2x2 matrix)
    // X is 3x2, V is 3x2, so X^T V is 2x2
    let xtv_00 = x[0] * v[0] + x[1] * v[1] + x[2] * v[2]; // Column 0, row 0
    let xtv_10 = x[3] * v[0] + x[4] * v[1] + x[5] * v[2]; // Column 0, row 1
    let xtv_01 = x[0] * v[3] + x[1] * v[4] + x[2] * v[5]; // Column 1, row 0
    let xtv_11 = x[3] * v[3] + x[4] * v[4] + x[5] * v[5]; // Column 1, row 1
    
    // Compute symmetric part: (X^T V + V^T X)/2
    let half = <T as Scalar>::from_f64(0.5);
    let sym_00 = xtv_00; // Diagonal elements are already symmetric
    let sym_11 = xtv_11;
    let sym_01 = (xtv_01 + xtv_10) * half; // Off-diagonal
    let sym_10 = sym_01; // Symmetric
    
    // Compute result = V - X * symmetric
    // Column 0 of result
    result[0] = v[0] - (x[0] * sym_00 + x[3] * sym_10);
    result[1] = v[1] - (x[1] * sym_00 + x[4] * sym_10);
    result[2] = v[2] - (x[2] * sym_00 + x[5] * sym_10);
    
    // Column 1 of result
    result[3] = v[3] - (x[0] * sym_01 + x[3] * sym_11);
    result[4] = v[4] - (x[1] * sym_01 + x[4] * sym_11);
    result[5] = v[5] - (x[2] * sym_01 + x[5] * sym_11);
}

/// Specialized projection to tangent space for St(4,2).
/// 
/// Computes the orthogonal projection of a matrix V onto the tangent space
/// of the Stiefel manifold at point X.
/// 
/// # Mathematical Formula
/// 
/// For X ∈ St(4,2) and V ∈ ℝ⁴ˣ², the projection follows the same formula
/// as St(3,2) but with different dimensions:
/// ```text
/// P_X(V) = V - X sym(X^T V)
/// ```
/// 
/// # Performance
/// 
/// - **Operations**: 48 multiplications, 40 additions  
/// - **Memory**: Stack-allocated temporaries only
/// - **Vectorization**: Well-suited for SSE/AVX instructions
#[inline(always)]
pub fn project_tangent_stiefel_4_2<T: Scalar>(
    x: &[T], // 4x2 matrix as column-major array of length 8
    v: &[T], // 4x2 matrix as column-major array of length 8
    result: &mut [T], // Output: 4x2 matrix as column-major array
) {
    debug_assert_eq!(x.len(), 8);
    debug_assert_eq!(v.len(), 8);
    debug_assert_eq!(result.len(), 8);
    
    // Compute X^T V (2x2 matrix)
    let xtv_00 = x[0] * v[0] + x[1] * v[1] + x[2] * v[2] + x[3] * v[3];
    let xtv_10 = x[4] * v[0] + x[5] * v[1] + x[6] * v[2] + x[7] * v[3];
    let xtv_01 = x[0] * v[4] + x[1] * v[5] + x[2] * v[6] + x[3] * v[7];
    let xtv_11 = x[4] * v[4] + x[5] * v[5] + x[6] * v[6] + x[7] * v[7];
    
    // Compute symmetric part
    let half = <T as Scalar>::from_f64(0.5);
    let sym_00 = xtv_00;
    let sym_11 = xtv_11;
    let sym_01 = (xtv_01 + xtv_10) * half;
    let sym_10 = sym_01;
    
    // Compute result = V - X * symmetric
    // Column 0
    result[0] = v[0] - (x[0] * sym_00 + x[4] * sym_10);
    result[1] = v[1] - (x[1] * sym_00 + x[5] * sym_10);
    result[2] = v[2] - (x[2] * sym_00 + x[6] * sym_10);
    result[3] = v[3] - (x[3] * sym_00 + x[7] * sym_10);
    
    // Column 1
    result[4] = v[4] - (x[0] * sym_01 + x[4] * sym_11);
    result[5] = v[5] - (x[1] * sym_01 + x[5] * sym_11);
    result[6] = v[6] - (x[2] * sym_01 + x[6] * sym_11);
    result[7] = v[7] - (x[3] * sym_01 + x[7] * sym_11);
}

/// Check if specialized Stiefel operations are available for given dimensions.
/// 
/// # Arguments
/// 
/// * `n` - Number of rows in the Stiefel matrix
/// * `p` - Number of columns in the Stiefel matrix
/// 
/// # Returns
/// 
/// `true` if specialized implementations exist for St(n,p), `false` otherwise.
/// 
/// # Example
/// 
/// ```rust
/// use riemannopt_manifolds::stiefel_small::can_use_specialized_stiefel;
/// 
/// assert!(can_use_specialized_stiefel(3, 2));  // St(3,2) is specialized
/// assert!(can_use_specialized_stiefel(4, 2));  // St(4,2) is specialized
/// assert!(!can_use_specialized_stiefel(5, 2)); // St(5,2) uses generic code
/// ```
#[inline(always)]
pub fn can_use_specialized_stiefel(n: usize, p: usize) -> bool {
    (n == 3 && p == 2) || (n == 4 && p == 2)
}

#[cfg(test)]
mod tests {
    use super::*;
        
    #[test]
    fn test_project_tangent_stiefel_3_2() {
        // Create an orthonormal matrix X in St(3,2)
        let x: Vec<f64> = vec![
            1.0, 0.0, 0.0,  // Column 0
            0.0, 1.0, 0.0,  // Column 1
        ];
        
        // Create a test vector
        let v: Vec<f64> = vec![
            0.5, 0.5, 0.5,  // Column 0
            0.5, 0.5, 0.5,  // Column 1
        ];
        
        let mut result = vec![0.0f64; 6];
        project_tangent_stiefel_3_2(&x, &v, &mut result);
        
        // Check that X^T * result + result^T * X = 0 (approximately)
        // This verifies the tangent space constraint
        
        // X^T * result
        let xtr_00 = x[0] * result[0] + x[1] * result[1] + x[2] * result[2];
        let xtr_10 = x[3] * result[0] + x[4] * result[1] + x[5] * result[2];
        let xtr_01 = x[0] * result[3] + x[1] * result[4] + x[2] * result[5];
        let xtr_11 = x[3] * result[3] + x[4] * result[4] + x[5] * result[5];
        
        // Check skew-symmetry
        assert!((xtr_00 + xtr_00).abs() < 1e-10);
        assert!((xtr_11 + xtr_11).abs() < 1e-10);
        assert!((xtr_01 + xtr_10).abs() < 1e-10);
    }
}