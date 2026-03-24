//! Type definitions and aliases for Riemannian optimization.
//!
//! This module provides common type aliases, traits for numeric types,
//! and constants used throughout the library.

use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::linalg::RealScalar;

/// Trait for scalar types used in optimization (f32 or f64).
///
/// This trait combines all the necessary numeric traits required
/// for Riemannian optimization algorithms. It extends [`RealScalar`]
/// (the backend-agnostic scalar trait) with additional constants and
/// conversion methods.
pub trait Scalar:
	RealScalar
	+ Float
	+ FromPrimitive
	+ Display
	+ Debug
	+ Default
	+ Copy
	+ Send
	+ Sync
	+ 'static
{
	/// Machine epsilon for this scalar type.
	const EPSILON: Self;

	/// Default tolerance for convergence checks.
	const DEFAULT_TOLERANCE: Self;

	/// Default tolerance for gradient norm convergence.
	const DEFAULT_GRADIENT_TOLERANCE: Self;

	/// Tolerance for checking if a point is on the manifold.
	const MANIFOLD_TOLERANCE: Self;

	/// Tolerance for checking orthogonality.
	const ORTHOGONALITY_TOLERANCE: Self;

	/// Maximum value for line search step size.
	const MAX_STEP_SIZE: Self;

	/// Minimum value for line search step size.
	const MIN_STEP_SIZE: Self;

	/// Optimal step size for central-difference finite differences.
	///
	/// For a function with machine-epsilon `ε`, the optimal step is `ε^(1/3)`
	/// for central differences (balancing truncation and rounding errors).
	/// We use `√ε` as a practical compromise that works well for both `f32`
	/// and `f64`.
	const FD_EPSILON: Self;

	/// Returns [`Self::FD_EPSILON`] — convenience method for generic code.
	#[inline]
	fn fd_epsilon() -> Self {
		Self::FD_EPSILON
	}

	/// Convert from f64 (for constants).
	///
	/// # Panics
	///
	/// Panics if the conversion fails. Use `try_from_f64` for a non-panicking version.
	fn from_f64(v: f64) -> Self {
		<Self as FromPrimitive>::from_f64(v).expect("Failed to convert from f64")
	}

	/// Try to convert from f64.
	///
	/// Returns None if the conversion fails.
	fn try_from_f64(v: f64) -> Option<Self> {
		<Self as FromPrimitive>::from_f64(v)
	}

	/// Convert to f64 (for logging/display).
	///
	/// # Panics
	///
	/// Panics if the conversion fails. Use `try_to_f64` for a non-panicking version.
	fn to_f64(self) -> f64 {
		num_traits::cast(self).expect("Failed to convert to f64")
	}

	/// Try to convert to f64.
	///
	/// Returns None if the conversion fails.
	fn try_to_f64(self) -> Option<f64> {
		num_traits::cast(self)
	}

	/// Convert from usize (for iteration counts).
	///
	/// # Panics
	///
	/// Panics if the conversion fails. Use `try_from_usize` for a non-panicking version.
	fn from_usize(v: usize) -> Self {
		<Self as FromPrimitive>::from_usize(v).expect("Failed to convert from usize")
	}

	/// Try to convert from usize.
	///
	/// Returns None if the conversion fails.
	fn try_from_usize(v: usize) -> Option<Self> {
		<Self as FromPrimitive>::from_usize(v)
	}
}

impl Scalar for f32 {
	const EPSILON: Self = f32::EPSILON;
	const DEFAULT_TOLERANCE: Self = 1e-4;
	const DEFAULT_GRADIENT_TOLERANCE: Self = 1e-5;
	const MANIFOLD_TOLERANCE: Self = 1e-6;
	const ORTHOGONALITY_TOLERANCE: Self = 1e-6;
	const MAX_STEP_SIZE: Self = 1e3;
	const MIN_STEP_SIZE: Self = 1e-10;
	// √(f32::EPSILON) ≈ 3.45e-4
	const FD_EPSILON: Self = 3.4526698e-4;
}

impl Scalar for f64 {
	const EPSILON: Self = f64::EPSILON;
	const DEFAULT_TOLERANCE: Self = 1e-6;
	const DEFAULT_GRADIENT_TOLERANCE: Self = 1e-8;
	const MANIFOLD_TOLERANCE: Self = 1e-12;
	const ORTHOGONALITY_TOLERANCE: Self = 1e-12;
	const MAX_STEP_SIZE: Self = 1e6;
	const MIN_STEP_SIZE: Self = 1e-16;
	// √(f64::EPSILON) ≈ 1.49e-8
	const FD_EPSILON: Self = 1.4901161193847656e-8;
}

/// Numerical constants for different precision levels.
pub mod constants {
	use super::Scalar;

	/// Get machine epsilon for the given scalar type.
	pub fn epsilon<T: Scalar>() -> T {
		T::EPSILON
	}

	/// Get default convergence tolerance.
	pub fn default_tolerance<T: Scalar>() -> T {
		T::DEFAULT_TOLERANCE
	}

	/// Get default gradient convergence tolerance.
	pub fn gradient_tolerance<T: Scalar>() -> T {
		T::DEFAULT_GRADIENT_TOLERANCE
	}

	/// Get manifold membership tolerance.
	pub fn manifold_tolerance<T: Scalar>() -> T {
		T::MANIFOLD_TOLERANCE
	}

	/// Get orthogonality checking tolerance.
	pub fn orthogonality_tolerance<T: Scalar>() -> T {
		T::ORTHOGONALITY_TOLERANCE
	}

	/// Get the optimal finite-difference step size.
	pub fn fd_epsilon<T: Scalar>() -> T {
		T::FD_EPSILON
	}

	/// Get maximum step size for line search.
	pub fn max_step_size<T: Scalar>() -> T {
		T::MAX_STEP_SIZE
	}

	/// Get minimum step size for line search.
	pub fn min_step_size<T: Scalar>() -> T {
		T::MIN_STEP_SIZE
	}

	/// Golden ratio constant.
	pub fn golden_ratio<T: Scalar>() -> T {
		<T as Scalar>::from_f64(1.618033988749895)
	}

	/// Square root of 2.
	pub fn sqrt_2<T: Scalar>() -> T {
		<T as Scalar>::from_f64(std::f64::consts::SQRT_2)
	}

	/// Pi constant.
	pub fn pi<T: Scalar>() -> T {
		<T as Scalar>::from_f64(std::f64::consts::PI)
	}

	/// Euler's number (e).
	pub fn e<T: Scalar>() -> T {
		<T as Scalar>::from_f64(std::f64::consts::E)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use approx::assert_relative_eq;

	#[test]
	fn test_scalar_trait_f32() {
		assert_eq!(f32::EPSILON, std::f32::EPSILON);
		assert!(f32::DEFAULT_TOLERANCE > 0.0);
		assert!(f32::DEFAULT_GRADIENT_TOLERANCE > 0.0);
		assert!(f32::MANIFOLD_TOLERANCE > 0.0);
		assert!(f32::MIN_STEP_SIZE < f32::MAX_STEP_SIZE);
	}

	#[test]
	fn test_scalar_trait_f64() {
		assert_eq!(f64::EPSILON, std::f64::EPSILON);
		assert!(f64::DEFAULT_TOLERANCE > 0.0);
		assert!(f64::DEFAULT_GRADIENT_TOLERANCE > 0.0);
		assert!(f64::MANIFOLD_TOLERANCE > 0.0);
		assert!(f64::MIN_STEP_SIZE < f64::MAX_STEP_SIZE);
	}

	#[test]
	fn test_scalar_conversions() {
		let val_f64 = 3.14159;
		let val_f32 = <f32 as Scalar>::from_f64(val_f64);
		assert_relative_eq!(val_f32 as f64, val_f64, epsilon = 1e-6);

		let back_f64 = val_f32.to_f64();
		assert_relative_eq!(back_f64, val_f32 as f64);
	}

	#[test]
	fn test_constants() {
		// Test f32 constants
		assert!(constants::epsilon::<f32>() > 0.0);
		assert!(constants::default_tolerance::<f32>() > constants::epsilon::<f32>());
		assert_relative_eq!(
			constants::golden_ratio::<f32>(),
			1.618033988749895_f32,
			epsilon = 1e-6
		);
		assert_relative_eq!(constants::pi::<f32>(), std::f32::consts::PI, epsilon = 1e-6);

		// Test f64 constants
		assert!(constants::epsilon::<f64>() > 0.0);
		assert!(constants::default_tolerance::<f64>() > constants::epsilon::<f64>());
		assert_relative_eq!(
			constants::golden_ratio::<f64>(),
			1.618033988749895_f64,
			epsilon = 1e-12
		);
		assert_relative_eq!(
			constants::pi::<f64>(),
			std::f64::consts::PI,
			epsilon = 1e-12
		);
	}

	#[test]
	fn test_tolerance_ordering() {
		// For f32
		assert!(f32::EPSILON < f32::MANIFOLD_TOLERANCE);
		assert!(f32::MANIFOLD_TOLERANCE < f32::DEFAULT_GRADIENT_TOLERANCE);
		assert!(f32::DEFAULT_GRADIENT_TOLERANCE < f32::DEFAULT_TOLERANCE);

		// For f64
		assert!(f64::EPSILON < f64::MANIFOLD_TOLERANCE);
		assert!(f64::MANIFOLD_TOLERANCE < f64::DEFAULT_GRADIENT_TOLERANCE);
		assert!(f64::DEFAULT_GRADIENT_TOLERANCE < f64::DEFAULT_TOLERANCE);
	}
}
