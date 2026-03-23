//! SIMD-accelerated operations for CPU parallel computing.
//!
//! This module provides SIMD (Single Instruction, Multiple Data) optimized
//! implementations of common operations using the `wide` crate.
//!
//! # Features
//!
//! - Vectorized dot products and norms
//! - Element-wise vector operations
//! - Matrix-vector multiplication
//! - Optimized manifold projections
//!
//! # Example
//!
//! ```rust
//! use riemannopt_core::compute::cpu::{SimdDispatcher, get_dispatcher, SimdBackend};
//! use nalgebra::DVector;
//!
//! let a = DVector::from_vec(vec![1.0_f32; 100]);
//! let b = DVector::from_vec(vec![2.0_f32; 100]);
//!
//! // SIMD-accelerated dot product using dispatcher
//! let dispatcher = get_dispatcher::<f32>();
//! let dot = dispatcher.dot_product(&a, &b);
//! assert_eq!(dot, 200.0);
//! ```

use crate::types::Scalar;
use nalgebra::{DMatrix, DVector, DVectorView};
use wide::{f32x8, f64x4};

/// SIMD operations for different scalar types
pub trait SimdOps: Scalar {
	/// SIMD vector type for this scalar
	type SimdVector: SimdVector<Scalar = Self>;

	/// Number of elements in SIMD vector
	const SIMD_WIDTH: usize;
}

/// Trait for SIMD vector operations
pub trait SimdVector: Copy {
	type Scalar: Scalar;

	/// Create from slice
	fn from_slice(slice: &[Self::Scalar]) -> Self;

	/// Store to slice
	fn store_to_slice(self, slice: &mut [Self::Scalar]);

	/// Splat a single value
	fn splat(value: Self::Scalar) -> Self;

	/// Add two vectors
	fn add(self, other: Self) -> Self;

	/// Multiply two vectors
	fn mul(self, other: Self) -> Self;

	/// Fused multiply-add
	fn mul_add(self, mul: Self, add: Self) -> Self;

	/// Horizontal sum
	fn horizontal_sum(self) -> Self::Scalar;

	/// Square root
	fn sqrt(self) -> Self;

	/// Reciprocal
	fn recip(self) -> Self;

	/// Element-wise maximum
	fn max(self, other: Self) -> Self;

	/// Horizontal maximum (maximum of all elements)
	fn horizontal_max(self) -> Self::Scalar;

	/// Absolute value
	fn abs(self) -> Self;
}

// Implement for f32x8
impl SimdVector for f32x8 {
	type Scalar = f32;

	fn from_slice(slice: &[f32]) -> Self {
		f32x8::from([
			slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
		])
	}

	fn store_to_slice(self, slice: &mut [f32]) {
		let arr = self.to_array();
		slice[..8].copy_from_slice(&arr);
	}

	fn splat(value: f32) -> Self {
		f32x8::splat(value)
	}

	fn add(self, other: Self) -> Self {
		self + other
	}

	fn mul(self, other: Self) -> Self {
		self * other
	}

	fn mul_add(self, mul: Self, add: Self) -> Self {
		self.mul_add(mul, add)
	}

	fn horizontal_sum(self) -> f32 {
		let arr = self.to_array();
		arr.iter().sum()
	}

	fn sqrt(self) -> Self {
		self.sqrt()
	}

	fn recip(self) -> Self {
		f32x8::splat(1.0) / self
	}

	fn max(self, other: Self) -> Self {
		self.max(other)
	}

	fn horizontal_max(self) -> f32 {
		let arr = self.to_array();
		arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
	}

	fn abs(self) -> Self {
		self.abs()
	}
}

// Implement for f64x4
impl SimdVector for f64x4 {
	type Scalar = f64;

	fn from_slice(slice: &[f64]) -> Self {
		f64x4::from([slice[0], slice[1], slice[2], slice[3]])
	}

	fn store_to_slice(self, slice: &mut [f64]) {
		let arr = self.to_array();
		slice[..4].copy_from_slice(&arr);
	}

	fn splat(value: f64) -> Self {
		f64x4::splat(value)
	}

	fn add(self, other: Self) -> Self {
		self + other
	}

	fn mul(self, other: Self) -> Self {
		self * other
	}

	fn mul_add(self, mul: Self, add: Self) -> Self {
		self.mul_add(mul, add)
	}

	fn horizontal_sum(self) -> f64 {
		let arr = self.to_array();
		arr.iter().sum()
	}

	fn sqrt(self) -> Self {
		self.sqrt()
	}

	fn recip(self) -> Self {
		f64x4::splat(1.0) / self
	}

	fn max(self, other: Self) -> Self {
		self.max(other)
	}

	fn horizontal_max(self) -> f64 {
		let arr = self.to_array();
		arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
	}

	fn abs(self) -> Self {
		self.abs()
	}
}

impl SimdOps for f32 {
	type SimdVector = f32x8;
	const SIMD_WIDTH: usize = 8;
}

impl SimdOps for f64 {
	type SimdVector = f64x4;
	const SIMD_WIDTH: usize = 4;
}

/// SIMD-accelerated vector operations (internal use only)
pub(crate) struct SimdVectorOps;

impl SimdVectorOps {
	/// Compute dot product using SIMD
	pub fn dot_product<T: SimdOps>(a: DVectorView<T>, b: DVectorView<T>) -> T {
		assert_eq!(a.len(), b.len(), "Vectors must have same length");

		let n = a.len();
		let simd_width = T::SIMD_WIDTH;
		let simd_end = n - (n % simd_width);

		let a_slice = a.as_slice();
		let b_slice = b.as_slice();

		// SIMD part
		let mut sum = T::SimdVector::splat(T::zero());
		for i in (0..simd_end).step_by(simd_width) {
			let va = T::SimdVector::from_slice(&a_slice[i..]);
			let vb = T::SimdVector::from_slice(&b_slice[i..]);
			// mul_add(a, b, c) = a * b + c, so we want va * vb + sum
			sum = va.mul_add(vb, sum);
		}

		let mut result = sum.horizontal_sum();

		// Scalar remainder
		for i in simd_end..n {
			result += a_slice[i] * b_slice[i];
		}

		result
	}

	/// Compute vector norm using SIMD
	pub fn norm<T: SimdOps>(v: DVectorView<T>) -> T {
		use num_traits::Float;

		let n = v.len();
		let w = T::SIMD_WIDTH;
		let simd_end = n - (n % w);
		let slice = v.as_slice();

		// Find maximum absolute value
		let mut max_abs = T::zero();
		for i in (0..simd_end).step_by(w) {
			let vv = T::SimdVector::from_slice(&slice[i..]);
			let abs_vv = vv.abs();
			let chunk_m = abs_vv.horizontal_max();
			max_abs = <T as Float>::max(max_abs, chunk_m);
		}
		// scalar remainder
		for &x in &slice[simd_end..] {
			max_abs = <T as Float>::max(max_abs, <T as Float>::abs(x));
		}
		// if max_abs is zero, return zero norm
		if max_abs == T::zero() {
			return T::zero();
		}

		// normalize the vector and compute the norm
		let inv_max = T::one() / max_abs;
		let inv_vv = T::SimdVector::splat(inv_max);

		let mut sum_vec = T::SimdVector::splat(T::zero());
		for i in (0..simd_end).step_by(w) {
			let y = T::SimdVector::from_slice(&slice[i..]).mul(inv_vv);
			sum_vec = y.mul_add(y, sum_vec);
		}
		let mut sum = sum_vec.horizontal_sum();
		// scalar remainder
		for &x in &slice[simd_end..] {
			let y = x * inv_max;
			sum += y * y;
		}

		// Return the norm
		max_abs * Float::sqrt(sum)
	}

	/// Element-wise vector addition using SIMD
	pub(crate) fn add<T: SimdOps>(a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
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
			let sum = va.add(vb);
			sum.store_to_slice(&mut result_slice[i..]);
		}

		// Scalar remainder
		for i in simd_end..n {
			result_slice[i] = a_slice[i] + b_slice[i];
		}
	}

	/// Scale vector using SIMD
	pub(crate) fn scale<T: SimdOps>(v: &mut DVector<T>, scalar: T) {
		let n = v.len();
		let simd_width = T::SIMD_WIDTH;
		let simd_end = n - (n % simd_width);

		let v_slice = v.as_mut_slice();
		let scalar_vec = T::SimdVector::splat(scalar);

		// SIMD part
		for i in (0..simd_end).step_by(simd_width) {
			let vv = T::SimdVector::from_slice(&v_slice[i..]);
			let scaled = vv.mul(scalar_vec);
			scaled.store_to_slice(&mut v_slice[i..]);
		}

		// Scalar remainder
		for i in simd_end..n {
			v_slice[i] = v_slice[i] * scalar;
		}
	}

	/// Normalize vector in-place using SIMD
	pub fn normalize<T: SimdOps>(v: &mut DVector<T>) -> T {
		let norm = Self::norm(v.as_view());
		if norm > T::zero() {
			let inv_norm = T::one() / norm;
			Self::scale(v, inv_norm);
		}
		norm
	}
}

/// SIMD-accelerated matrix operations (internal use only)
pub(crate) struct SimdMatrixOps;

impl SimdMatrixOps {
	/// Matrix-vector multiplication using SIMD
	///
	/// For best performance with large matrices, use nalgebra's built-in gemv
	/// which is already optimized. This method is kept for API compatibility.
	pub fn gemv<T: SimdOps>(a: &DMatrix<T>, x: &DVector<T>, y: &mut DVector<T>, alpha: T, beta: T) {
		assert_eq!(a.ncols(), x.len(), "Dimension mismatch");
		assert_eq!(a.nrows(), y.len(), "Dimension mismatch");

		// Use nalgebra's optimized gemv implementation
		// y = alpha * A * x + beta * y
		y.gemv(alpha, a, x, beta);
	}

	/// Frobenius norm using SIMD
	pub fn frobenius_norm<T: SimdOps>(a: &DMatrix<T>) -> T {
		let data = a.as_slice();
		let n = data.len();
		let simd_width = T::SIMD_WIDTH;
		let simd_end = n - (n % simd_width);

		// SIMD part
		let mut sum = T::SimdVector::splat(T::zero());
		for i in (0..simd_end).step_by(simd_width) {
			let v = T::SimdVector::from_slice(&data[i..]);
			sum = v.mul_add(v, sum);
		}

		let mut result = sum.horizontal_sum();

		// Scalar remainder
		for i in simd_end..n {
			result = result + data[i] * data[i];
		}

		<T as num_traits::Float>::sqrt(result)
	}
}

/// SIMD-accelerated manifold operations (internal use only)
pub(crate) mod simd_manifolds {
	use super::*;
	use crate::compute::cpu::{get_dispatcher, ScalarDispatch, SimdBackend};

	/// SIMD sphere projection
	#[allow(dead_code)]
	pub fn project_sphere_simd<T: SimdOps + ScalarDispatch>(point: &mut DVector<T>) {
		let dispatcher = get_dispatcher::<T>();
		let _norm = dispatcher.normalize(point);
	}

	/// SIMD orthogonalization for Stiefel using Modified Gram-Schmidt
	#[allow(dead_code)]
	pub fn orthogonalize_simd<T: SimdOps + ScalarDispatch>(matrix: &mut DMatrix<T>) {
		let p = matrix.ncols();
		let dispatcher = get_dispatcher::<T>();

		for j in 0..p {
			// Get a mutable view of column j
			let col_j_norm = {
				let col_j = matrix.column(j);
				let col_j_vec = DVector::from_iterator(col_j.len(), col_j.iter().cloned());
				dispatcher.norm(&col_j_vec)
			};

			// Normalize column j in-place
			if col_j_norm > T::zero() {
				let inv_norm = T::one() / col_j_norm;
				matrix.column_mut(j).scale_mut(inv_norm);
			}

			// Orthogonalize remaining columns against column j
			for k in (j + 1)..p {
				// Compute dot product between columns j and k
				let dot = {
					let col_j = matrix.column(j);
					let col_k = matrix.column(k);
					let col_j_vec = DVector::from_iterator(col_j.len(), col_j.iter().cloned());
					let col_k_vec = DVector::from_iterator(col_k.len(), col_k.iter().cloned());
					dispatcher.dot_product(&col_j_vec, &col_k_vec)
				};

				// Update column k: col_k -= dot * col_j
				let col_j_clone = matrix.column(j).clone_owned();
				matrix.column_mut(k).axpy(-dot, &col_j_clone, T::one());
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::compute::cpu::{get_dispatcher, SimdBackend};
	use approx::assert_relative_eq;

	#[test]
	fn test_simd_dot_product() {
		let a = DVector::from_vec(vec![1.0_f32; 100]);
		let b = DVector::from_vec(vec![2.0_f32; 100]);

		let dispatcher = get_dispatcher::<f32>();
		let result = dispatcher.dot_product(&a, &b);
		assert_relative_eq!(result, 200.0, epsilon = 1e-6);
	}

	#[test]
	fn test_simd_norm() {
		let v = DVector::from_vec(vec![3.0_f64, 4.0, 0.0, 0.0]);
		let dispatcher = get_dispatcher::<f64>();
		let norm = dispatcher.norm(&v);
		assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
	}

	#[test]
	fn test_simd_add() {
		let a = DVector::from_vec(vec![1.0_f32; 17]); // Non-multiple of SIMD width
		let b = DVector::from_vec(vec![2.0_f32; 17]);
		let mut result = DVector::zeros(17);

		let dispatcher = get_dispatcher::<f32>();
		dispatcher.add(&a, &b, &mut result);

		for i in 0..17 {
			assert_relative_eq!(result[i], 3.0, epsilon = 1e-6);
		}
	}

	#[test]
	fn test_simd_normalize() {
		let mut v = DVector::from_vec(vec![3.0_f64, 4.0]);
		let dispatcher = get_dispatcher::<f64>();
		let norm = dispatcher.normalize(&mut v);

		assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
		assert_relative_eq!(v[0], 0.6, epsilon = 1e-10);
		assert_relative_eq!(v[1], 0.8, epsilon = 1e-10);
	}

	#[test]
	fn test_simd_scale() {
		let mut a = DVector::from_vec(vec![2.0_f32, 3.0, 4.0]);
		let dispatcher = get_dispatcher::<f32>();
		dispatcher.scale(&mut a, 2.5);

		assert_relative_eq!(a[0], 5.0, epsilon = 1e-6);
		assert_relative_eq!(a[1], 7.5, epsilon = 1e-6);
		assert_relative_eq!(a[2], 10.0, epsilon = 1e-6);
	}

	#[test]
	fn test_simd_axpy() {
		let x = DVector::from_vec(vec![1.0_f64, 2.0, 3.0]);
		let mut y = DVector::from_vec(vec![4.0_f64, 5.0, 6.0]);
		let dispatcher = get_dispatcher::<f64>();

		// y = 2.0 * x + y
		dispatcher.axpy(2.0, &x, &mut y);

		assert_relative_eq!(y[0], 6.0, epsilon = 1e-10); // 2*1 + 4
		assert_relative_eq!(y[1], 9.0, epsilon = 1e-10); // 2*2 + 5
		assert_relative_eq!(y[2], 12.0, epsilon = 1e-10); // 2*3 + 6
	}

	#[test]
	fn test_simd_hadamard_product() {
		let a = DVector::from_vec(vec![2.0_f32, 3.0, 4.0]);
		let b = DVector::from_vec(vec![5.0_f32, 6.0, 7.0]);
		let mut result = DVector::zeros(3);
		let dispatcher = get_dispatcher::<f32>();

		dispatcher.hadamard_product(&a, &b, &mut result);

		assert_relative_eq!(result[0], 10.0, epsilon = 1e-6);
		assert_relative_eq!(result[1], 18.0, epsilon = 1e-6);
		assert_relative_eq!(result[2], 28.0, epsilon = 1e-6);
	}

	#[test]
	fn test_simd_frobenius_norm() {
		let matrix = DMatrix::from_row_slice(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

		// Frobenius norm = sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) = sqrt(91)
		let expected = (91.0_f64).sqrt();
		let dispatcher = get_dispatcher::<f64>();
		let result = dispatcher.frobenius_norm(&matrix);
		assert_relative_eq!(result, expected, epsilon = 1e-10);
	}

	#[test]
	fn test_normalize_simd() {
		let mut point = DVector::from_vec(vec![3.0_f32, 4.0, 0.0]);
		let dispatcher = get_dispatcher::<f32>();
		let original_norm = dispatcher.normalize(&mut point);

		// Check that the original norm was 5
		assert_relative_eq!(original_norm, 5.0, epsilon = 1e-6);

		// After normalization, norm should be 1
		let norm = point.norm();
		assert_relative_eq!(norm, 1.0, epsilon = 1e-6);

		// Check individual components
		assert_relative_eq!(point[0], 0.6, epsilon = 1e-6);
		assert_relative_eq!(point[1], 0.8, epsilon = 1e-6);
		assert_relative_eq!(point[2], 0.0, epsilon = 1e-6);
	}

	#[test]
	fn test_simd_gemv() {
		let matrix = DMatrix::from_row_slice(2, 3, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
		let x = DVector::from_vec(vec![1.0_f32, 2.0, 3.0]);
		let mut y = DVector::zeros(2);

		let dispatcher = get_dispatcher::<f32>();
		dispatcher.gemv(&matrix, &x, &mut y, 1.0, 0.0);

		// y = matrix * x = [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
		assert_relative_eq!(y[0], 14.0, epsilon = 1e-6);
		assert_relative_eq!(y[1], 32.0, epsilon = 1e-6);

		// Test with beta != 0
		let mut y2 = DVector::from_vec(vec![10.0_f32, 20.0]);
		dispatcher.gemv(&matrix, &x, &mut y2, 2.0, 0.5);

		// y2 = 2.0 * matrix * x + 0.5 * y2 = [2*14 + 0.5*10, 2*32 + 0.5*20] = [33, 74]
		assert_relative_eq!(y2[0], 33.0, epsilon = 1e-6);
		assert_relative_eq!(y2[1], 74.0, epsilon = 1e-6);
	}

	#[test]
	fn test_simd_operations_with_various_sizes() {
		// Test with different vector sizes to ensure SIMD fallback works correctly
		let sizes = vec![
			1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
		];

		for size in sizes {
			let a = DVector::from_vec(vec![1.0_f32; size]);
			let b = DVector::from_vec(vec![2.0_f32; size]);

			let dispatcher = get_dispatcher::<f32>();

			// Test dot product
			let dot = dispatcher.dot_product(&a, &b);
			assert_relative_eq!(dot, 2.0 * size as f32, epsilon = 1e-5);

			// Test norm
			let norm = dispatcher.norm(&a);
			assert_relative_eq!(norm, (size as f32).sqrt(), epsilon = 1e-5);

			// Test add
			let mut result = DVector::zeros(size);
			dispatcher.add(&a, &b, &mut result);
			for i in 0..size {
				assert_relative_eq!(result[i], 3.0, epsilon = 1e-6);
			}
		}
	}
}
