//! Specialized implementations for small Stiefel manifolds
//!
//! This module provides optimized implementations for common small
//! Stiefel manifold configurations like St(3,2), St(4,2), etc.

use riemannopt_core::types::Scalar;

/// Specialized projection to tangent space for St(3,2)
/// 
/// For X ∈ St(3,2) and V ∈ ℝ³ˣ², the projection is:
/// P_X(V) = V - X(X^T V + V^T X)/2
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

/// Specialized projection to tangent space for St(4,2)
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

/// Check if we can use specialized Stiefel operations
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
        assert!((xtr_00 + xtr_00<T as Float>::abs()) < 1e-10);
        assert!((xtr_11 + xtr_11<T as Float>::abs()) < 1e-10);
        assert!((xtr_01 + xtr_10<T as Float>::abs()) < 1e-10);
    }
}