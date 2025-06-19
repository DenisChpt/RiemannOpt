//! Numerical stability utilities for Riemannian optimization
//!
//! This module provides functions and constants to ensure numerical stability
//! across different manifold operations and optimizers.

use nalgebra::{DVector, DMatrix, RealField};
use num_traits::{Float, FromPrimitive};
use crate::types::Scalar;

/// Machine epsilon scaled for safe comparisons
pub fn safe_epsilon<T: RealField>() -> T {
    T::from_f64(1e-10).unwrap_or_else(|| T::default_epsilon() * T::from_f64(100.0).unwrap())
}

/// Tolerance for checking orthogonality constraints
pub fn orthogonality_tolerance<T: RealField>() -> T {
    T::from_f64(1e-8).unwrap_or_else(|| T::default_epsilon() * T::from_f64(1000.0).unwrap())
}

/// Tolerance for checking positive definiteness
pub fn positive_definite_tolerance<T: RealField>() -> T {
    T::from_f64(1e-10).unwrap_or_else(|| T::default_epsilon() * T::from_f64(100.0).unwrap())
}

/// Safe division that avoids division by zero
pub fn safe_divide<T: RealField + Copy>(numerator: T, denominator: T) -> T {
    let eps = safe_epsilon::<T>();
    if denominator.abs() < eps {
        if numerator.abs() < eps {
            T::one() // 0/0 case, return 1
        } else if numerator > T::zero() {
            T::from_f64(1e10).unwrap() // Large positive value
        } else {
            T::from_f64(-1e10).unwrap() // Large negative value
        }
    } else {
        numerator / denominator
    }
}

/// Safe square root that handles small negative values due to rounding
pub fn safe_sqrt<T: Float>(x: T) -> T {
    if x < T::zero() && x > -T::epsilon() {
        T::zero()
    } else if x < T::zero() {
        T::nan() // Return NaN for actual negative values
    } else {
        x.sqrt()
    }
}

/// Stabilized norm computation for vectors
pub fn stable_norm<T: RealField + Copy>(v: &DVector<T>) -> T {
    // Use two-pass algorithm for better numerical stability
    let max_elem = v.iter().map(|x| x.abs()).fold(T::zero(), |a, b| a.max(b));
    
    if max_elem < safe_epsilon::<T>() {
        return T::zero();
    }
    
    // Scale by max element to avoid overflow/underflow
    let scaled_norm = v.iter()
        .map(|x| {
            let scaled = *x / max_elem;
            scaled * scaled
        })
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();
    
    max_elem * scaled_norm
}

/// Numerically stable matrix inverse using SVD
pub fn stable_inverse<T: RealField + Copy>(matrix: &DMatrix<T>) -> Result<DMatrix<T>, &'static str> {
    let svd = matrix.clone().svd(true, true);
    let tolerance = T::from_f64(1e-10).unwrap() * svd.singular_values.max();
    
    // Check condition number
    let min_sv = svd.singular_values.min();
    if min_sv < tolerance {
        return Err("Matrix is numerically singular");
    }
    
    // Compute pseudo-inverse
    let u = svd.u.ok_or("SVD failed to compute U")?;
    let vt = svd.v_t.ok_or("SVD failed to compute V^T")?;
    
    let mut inv_s = DMatrix::zeros(matrix.ncols(), matrix.nrows());
    for i in 0..svd.singular_values.len() {
        if svd.singular_values[i] > tolerance {
            inv_s[(i, i)] = T::one() / svd.singular_values[i].clone();
        }
    }
    
    Ok(vt.transpose() * inv_s * u.transpose())
}

/// Stabilized Gram-Schmidt orthogonalization
pub fn stable_gram_schmidt<T: RealField + Copy>(vectors: &DMatrix<T>) -> DMatrix<T> {
    let (m, n) = (vectors.nrows(), vectors.ncols());
    let mut q = DMatrix::zeros(m, n);
    
    for j in 0..n {
        // Copy column
        let mut v = vectors.column(j).into_owned();
        
        // Orthogonalize against previous columns (two passes for stability)
        for _ in 0..2 {
            for i in 0..j {
                let qi = q.column(i);
                let proj = v.dot(&qi);
                v.axpy(-proj, &qi.clone_owned(), T::one());
            }
        }
        
        // Normalize
        let norm = stable_norm(&v);
        if norm > safe_epsilon::<T>() {
            q.set_column(j, &(v / norm));
        } else {
            // Column is linearly dependent, set to zero
            q.set_column(j, &DVector::zeros(m));
        }
    }
    
    q
}

/// Check if a matrix is finite (no NaN or Inf values)
pub fn is_finite_matrix<T: RealField + Copy>(matrix: &DMatrix<T>) -> bool {
    matrix.iter().all(|x| x.is_finite())
}

/// Clip gradient values to prevent explosion
pub fn clip_gradient<T: RealField + Copy>(gradient: &mut DVector<T>, max_norm: T) {
    let norm = stable_norm(gradient);
    if norm > max_norm && norm > T::zero() {
        gradient.scale_mut(max_norm / norm);
    }
}

/// Safe logarithm that handles edge cases
pub fn safe_log<T: Float + FromPrimitive>(x: T) -> T {
    if x <= T::zero() {
        T::from_f64(-50.0).unwrap_or(T::neg_infinity()) // Large negative value instead of -inf
    } else {
        x.ln()
    }
}

/// Regularize a matrix to ensure positive definiteness
pub fn regularize_spd<T: RealField + Copy>(matrix: &mut DMatrix<T>, reg_param: T) {
    let n = matrix.nrows();
    for i in 0..n {
        matrix[(i, i)] += reg_param;
    }
}

/// Check if optimization should stop due to numerical issues
pub fn check_numerical_issues<T: Scalar>(
    gradient_norm: T,
    cost: T,
    iteration: usize,
) -> Option<String> {
    if !gradient_norm.is_finite() {
        return Some(format!("Gradient norm is not finite at iteration {}", iteration));
    }
    
    if !cost.is_finite() {
        return Some(format!("Cost function is not finite at iteration {}", iteration));
    }
    
    if gradient_norm > <T as Scalar>::from_f64(1e10) {
        return Some(format!(
            "Gradient explosion detected at iteration {}: norm = {:e}",
            iteration,
            gradient_norm.to_f64()
        ));
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_safe_divide() {
        assert_eq!(safe_divide(2.0, 0.0), 1e10);
        assert_eq!(safe_divide(-2.0, 0.0), -1e10);
        assert_eq!(safe_divide(0.0, 0.0), 1.0);
        assert_eq!(safe_divide(6.0, 2.0), 3.0);
    }
    
    #[test]
    fn test_safe_sqrt() {
        assert_eq!(safe_sqrt(4.0), 2.0);
        assert_eq!(safe_sqrt(0.0), 0.0);
        assert_eq!(safe_sqrt(-1e-16), 0.0); // Small negative due to rounding
        assert!(safe_sqrt(-1.0).is_nan());
    }
    
    #[test]
    fn test_stable_norm() {
        let v = DVector::from_vec(vec![3.0, 4.0]);
        assert_relative_eq!(stable_norm(&v), 5.0);
        
        // Test with very large values
        let v_large = DVector::from_vec(vec![1e100, 1e100]);
        assert_relative_eq!(stable_norm(&v_large), 1e100 * 2f64.sqrt());
        
        // Test with very small values
        let v_small = DVector::from_vec(vec![1e-100, 1e-100]);
        assert_relative_eq!(stable_norm(&v_small), 1e-100 * 2f64.sqrt());
    }
}