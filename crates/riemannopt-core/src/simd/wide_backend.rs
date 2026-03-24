//! SIMD backend using the `wide` crate for portable SIMD operations.

use super::dispatch::SimdBackend;
use super::ops::{SimdMatrixOps, SimdOps, SimdVector, SimdVectorOps};
use nalgebra::{DMatrix, DVector};
use std::marker::PhantomData;

/// Backend that uses the `wide` crate for SIMD operations.
pub struct WideBackend<T> {
	_phantom: PhantomData<T>,
}

impl<T> WideBackend<T> {
	pub fn new() -> Self {
		Self {
			_phantom: PhantomData,
		}
	}
}

impl<T> Default for WideBackend<T> {
	fn default() -> Self {
		Self::new()
	}
}

impl<T: SimdOps> SimdBackend<T> for WideBackend<T> {
	fn dot_product(&self, a: &DVector<T>, b: &DVector<T>) -> T {
		SimdVectorOps::dot_product(a.as_view(), b.as_view())
	}

	fn norm(&self, v: &DVector<T>) -> T {
		SimdVectorOps::norm(v.as_view())
	}

	fn norm_squared(&self, v: &DVector<T>) -> T {
		// More efficient than computing norm and squaring
		let n = v.len();
		let simd_width = T::SIMD_WIDTH;
		let simd_end = n - (n % simd_width);

		let v_slice = v.as_slice();

		// SIMD part
		let mut sum = T::SimdVector::splat(T::zero());
		for i in (0..simd_end).step_by(simd_width) {
			let vv = T::SimdVector::from_slice(&v_slice[i..]);
			sum = vv.mul_add(vv, sum);
		}

		let mut result = sum.horizontal_sum();

		// Scalar remainder
		for i in simd_end..n {
			result = result + v_slice[i] * v_slice[i];
		}

		result
	}

	fn add(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
		SimdVectorOps::add(a, b, result)
	}

	fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) {
		assert_eq!(x.len(), y.len(), "Vectors must have same length");

		let n = x.len();
		let simd_width = T::SIMD_WIDTH;
		let simd_end = n - (n % simd_width);

		let x_slice = x.as_slice();
		let y_slice = y.as_mut_slice();
		let alpha_vec = T::SimdVector::splat(alpha);

		// SIMD part
		for i in (0..simd_end).step_by(simd_width) {
			let vx = T::SimdVector::from_slice(&x_slice[i..]);
			let vy = T::SimdVector::from_slice(&y_slice[i..]);
			let result = vx.mul_add(alpha_vec, vy);
			result.store_to_slice(&mut y_slice[i..]);
		}

		// Scalar remainder
		for i in simd_end..n {
			y_slice[i] = y_slice[i] + alpha * x_slice[i];
		}
	}

	fn scale(&self, v: &mut DVector<T>, scalar: T) {
		SimdVectorOps::scale(v, scalar)
	}

	fn normalize(&self, v: &mut DVector<T>) -> T {
		SimdVectorOps::normalize(v)
	}

	fn gemv(&self, a: &DMatrix<T>, x: &DVector<T>, y: &mut DVector<T>, alpha: T, beta: T) {
		SimdMatrixOps::gemv(a, x, y, alpha, beta)
	}

	fn frobenius_norm(&self, a: &DMatrix<T>) -> T {
		SimdMatrixOps::frobenius_norm(a)
	}

	fn hadamard_product(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
		assert_eq!(a.len(), b.len(), "Vectors must have same length");
		assert_eq!(a.len(), result.len(), "Result must have same length");

		let n = a.len();
		let simd_width = T::SIMD_WIDTH;
		let simd_end = n - (n % simd_width);

		let a_slice = a.as_slice();
		let b_slice = b.as_slice();
		let result_slice = result.as_mut_slice();

		// SIMD part
		for i in (0..simd_end).step_by(simd_width) {
			let va = T::SimdVector::from_slice(&a_slice[i..]);
			let vb = T::SimdVector::from_slice(&b_slice[i..]);
			let prod = va.mul(vb);
			prod.store_to_slice(&mut result_slice[i..]);
		}

		// Scalar remainder
		for i in simd_end..n {
			result_slice[i] = a_slice[i] * b_slice[i];
		}
	}

	fn is_efficient_for_size(&self, size: usize) -> bool {
		// Wide backend is efficient for vectors larger than SIMD width
		size >= T::SIMD_WIDTH * 2
	}

	fn max_abs_diff(&self, a: &DVector<T>, b: &DVector<T>) -> T {
		assert_eq!(a.len(), b.len(), "Vectors must have same length");

		let n = a.len();
		let simd_width = T::SIMD_WIDTH;
		let simd_end = n - (n % simd_width);

		let a_slice = a.as_slice();
		let b_slice = b.as_slice();

		// SIMD part - compute absolute differences and track maximum
		let mut max_vec = T::SimdVector::splat(T::zero());
		for i in (0..simd_end).step_by(simd_width) {
			let va = T::SimdVector::from_slice(&a_slice[i..]);
			let vb = T::SimdVector::from_slice(&b_slice[i..]);
			let diff = va.add(vb.mul(T::SimdVector::splat(-T::one()))); // va - vb

			// Compute absolute value by selecting max of diff and -diff
			let neg_diff = diff.mul(T::SimdVector::splat(-T::one()));
			let abs_diff = diff.max(neg_diff);

			// Update maximum
			max_vec = max_vec.max(abs_diff);
		}

		// Get the maximum from the SIMD vector
		let mut max_diff = max_vec.horizontal_max();

		// Scalar remainder
		for i in simd_end..n {
			let diff = a_slice[i] - b_slice[i];
			let abs_diff = <T as num_traits::Signed>::abs(&diff);
			if abs_diff > max_diff {
				max_diff = abs_diff;
			}
		}

		max_diff
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use approx::assert_relative_eq;

	#[test]
	fn test_wide_backend_operations() {
		let backend = WideBackend::<f32>::new();

		let a = DVector::from_vec(vec![1.0; 100]);
		let b = DVector::from_vec(vec![2.0; 100]);

		// Test dot product
		let dot = backend.dot_product(&a, &b);
		assert_relative_eq!(dot, 200.0, epsilon = 1e-6);

		// Test norm squared
		let norm_sq = backend.norm_squared(&a);
		assert_relative_eq!(norm_sq, 100.0, epsilon = 1e-6);

		// Test axpy
		let mut y = DVector::from_vec(vec![1.0; 100]);
		backend.axpy(0.5, &a, &mut y);
		for i in 0..100 {
			assert_relative_eq!(y[i], 1.5, epsilon = 1e-6);
		}

		// Test hadamard product
		let mut result = DVector::zeros(100);
		backend.hadamard_product(&a, &b, &mut result);
		for i in 0..100 {
			assert_relative_eq!(result[i], 2.0, epsilon = 1e-6);
		}
	}

	#[test]
	fn test_efficiency_check() {
		let backend = WideBackend::<f64>::new();

		// Should be inefficient for small sizes
		assert!(!backend.is_efficient_for_size(3));

		// Should be efficient for larger sizes
		assert!(backend.is_efficient_for_size(100));
	}

	#[test]
	fn test_scale_vector() {
		let backend = WideBackend::<f32>::new();
		let mut x = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

		backend.scale(&mut x, 2.5);

		assert_relative_eq!(x[0], 5.0, epsilon = 1e-6);
		assert_relative_eq!(x[1], 7.5, epsilon = 1e-6);
		assert_relative_eq!(x[2], 10.0, epsilon = 1e-6);
		assert_relative_eq!(x[3], 12.5, epsilon = 1e-6);
	}

	#[test]
	fn test_add_vectors() {
		let backend = WideBackend::<f64>::new();
		let a = DVector::from_vec(vec![1.0, 2.0, 3.0]);
		let b = DVector::from_vec(vec![4.0, 5.0, 6.0]);
		let mut result = DVector::zeros(3);

		backend.add(&a, &b, &mut result);

		assert_relative_eq!(result[0], 5.0, epsilon = 1e-10);
		assert_relative_eq!(result[1], 7.0, epsilon = 1e-10);
		assert_relative_eq!(result[2], 9.0, epsilon = 1e-10);
	}

	#[test]
	fn test_max_abs_diff() {
		let backend = WideBackend::<f64>::new();
		let a = DVector::from_vec(vec![1.0, -5.0, 3.0, 7.0]);
		let b = DVector::from_vec(vec![2.0, -2.0, 3.0, 4.0]);

		// Max abs diff should be |(-5) - (-2)| = 3
		let max_diff = backend.max_abs_diff(&a, &b);
		assert_relative_eq!(max_diff, 3.0, epsilon = 1e-10);
	}

	#[test]
	fn test_frobenius_norm() {
		let backend = WideBackend::<f32>::new();
		let matrix = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

		// Frobenius norm = sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) = sqrt(91)
		let expected = (91.0_f32).sqrt();
		let result = backend.frobenius_norm(&matrix);
		assert_relative_eq!(result, expected, epsilon = 1e-6);
	}

	#[test]
	fn test_large_vectors() {
		// Test with larger vectors to ensure wide operations work correctly
		let backend = WideBackend::<f64>::new();
		let size = 1000;

		let a = DVector::from_vec(vec![1.5; size]);
		let b = DVector::from_vec(vec![2.5; size]);

		// Test dot product
		let dot = backend.dot_product(&a, &b);
		assert_relative_eq!(dot, 3.75 * size as f64, epsilon = 1e-8);

		// Test norm squared
		let norm_sq = backend.norm_squared(&a);
		assert_relative_eq!(norm_sq, 2.25 * size as f64, epsilon = 1e-8);
	}

	#[test]
	fn test_edge_cases() {
		let backend = WideBackend::<f32>::new();

		// Test with single element
		let single = DVector::from_vec(vec![42.0]);
		assert_relative_eq!(backend.norm_squared(&single), 1764.0, epsilon = 1e-6);
		assert_relative_eq!(backend.norm(&single), 42.0, epsilon = 1e-6);

		// Test normalize with zero vector
		let mut zero_vec = DVector::from_vec(vec![0.0, 0.0, 0.0]);
		let norm = backend.normalize(&mut zero_vec);
		assert_relative_eq!(norm, 0.0, epsilon = 1e-6);

		// Test normalize with non-zero vector
		let mut vec = DVector::from_vec(vec![3.0, 4.0]);
		let norm = backend.normalize(&mut vec);
		assert_relative_eq!(norm, 5.0, epsilon = 1e-6);
		assert_relative_eq!(vec[0], 0.6, epsilon = 1e-6);
		assert_relative_eq!(vec[1], 0.8, epsilon = 1e-6);
	}
}
