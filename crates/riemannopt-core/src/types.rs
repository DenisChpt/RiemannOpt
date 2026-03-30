//! Type definitions and aliases for Riemannian optimization.
//!
//! This module provides the [`Scalar`] trait — the single numeric abstraction
//! used throughout the library. All domain-specific constants (tolerances,
//! step sizes, mathematical constants) live as associated constants on
//! `Scalar`, guaranteed to be in `.rodata` and resolved at compile time.

use std::fmt::{Debug, Display};

use crate::linalg::RealScalar;

/// Trait for scalar types used in optimization (f32 or f64).
///
/// Extends [`RealScalar`] (backend-agnostic numeric ops) with
/// domain-specific constants and convenience conversions.
/// Conversions delegate to [`RealScalar`] — no duplication with
/// `num_traits::FromPrimitive`.
pub trait Scalar: RealScalar + Display + Debug + 'static {
	// ========================================================================
	// Machine & tolerance constants
	// ========================================================================

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

	// ========================================================================
	// Finite-difference step sizes
	// ========================================================================

	/// Optimal step for forward finite differences: h ≈ ε^(1/2).
	///
	/// For forward differences (f(x+h) − f(x)) / h, the error is
	/// O(h) + O(ε/h), minimized at h = √ε.
	const FD_FORWARD_STEP: Self;

	/// Optimal step for central finite differences: h ≈ ε^(1/3).
	///
	/// For central differences (f(x+h) − f(x−h)) / 2h, the error is
	/// O(h²) + O(ε/h), minimized at h = ε^(1/3).
	const FD_CENTRAL_STEP: Self;

	// ========================================================================
	// Mathematical constants (guaranteed .rodata, zero-cost)
	// ========================================================================

	/// π
	const PI: Self;

	/// √2
	const SQRT_2: Self;

	/// Euler's number e
	const E: Self;

	/// Golden ratio φ = (1 + √5) / 2
	const GOLDEN_RATIO: Self;

	/// Threshold for switching to Taylor series in exp/log maps.
	///
	/// Below this angle, sin(θ)/θ and θ/sin(θ) suffer catastrophic
	/// cancellation. Pre-computed as max(1e-5, 50·√ε).
	///
	/// - f64: ≈ 1e-5
	/// - f32: ≈ 1.7e-2
	const SMALL_ANGLE_THRESHOLD: Self;

	// ========================================================================
	// Convenience conversions (delegate to RealScalar)
	// ========================================================================

	/// Convert from f64 (infallible cast via `as`).
	#[inline]
	fn from_f64(v: f64) -> Self {
		Self::from_f64_const(v)
	}

	/// Convert to f64 (potentially lossy for f32 → f64).
	#[inline]
	fn to_f64(self) -> f64 {
		self.to_f64_lossy()
	}

	/// Returns [`Self::FD_FORWARD_STEP`] — convenience for generic code.
	#[inline]
	fn fd_epsilon() -> Self {
		Self::FD_FORWARD_STEP
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
	const FD_FORWARD_STEP: Self = 3.4526698e-4;
	// (f32::EPSILON)^(1/3) ≈ 4.92e-3
	const FD_CENTRAL_STEP: Self = 4.9215667e-3;

	const PI: Self = std::f32::consts::PI;
	const SQRT_2: Self = std::f32::consts::SQRT_2;
	const E: Self = std::f32::consts::E;
	const GOLDEN_RATIO: Self = 1.618_033_9;
	// max(1e-5, 50·√(f32::EPSILON)) = max(1e-5, 50·3.45e-4) ≈ 1.73e-2
	const SMALL_ANGLE_THRESHOLD: Self = 1.7263349e-2;
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
	const FD_FORWARD_STEP: Self = 1.4901161193847656e-8;
	// (f64::EPSILON)^(1/3) ≈ 6.06e-6
	const FD_CENTRAL_STEP: Self = 6.055454452393343e-6;

	const PI: Self = std::f64::consts::PI;
	const SQRT_2: Self = std::f64::consts::SQRT_2;
	const E: Self = std::f64::consts::E;
	const GOLDEN_RATIO: Self = 1.618_033_988_749_895;
	// max(1e-5, 50·√(f64::EPSILON)) = max(1e-5, 50·1.49e-8) ≈ 1e-5
	const SMALL_ANGLE_THRESHOLD: Self = 1e-5;
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
	fn test_fd_steps_ordering() {
		// Central step > forward step (ε^(1/3) > ε^(1/2) for ε < 1)
		assert!(f32::FD_CENTRAL_STEP > f32::FD_FORWARD_STEP);
		assert!(f64::FD_CENTRAL_STEP > f64::FD_FORWARD_STEP);
	}

	#[test]
	fn test_math_constants() {
		assert_relative_eq!(f32::PI, std::f32::consts::PI);
		assert_relative_eq!(f64::PI, std::f64::consts::PI);
		assert_relative_eq!(f32::SQRT_2, std::f32::consts::SQRT_2);
		assert_relative_eq!(f64::SQRT_2, std::f64::consts::SQRT_2);
		assert_relative_eq!(f64::GOLDEN_RATIO, 1.618_033_988_749_895, epsilon = 1e-12);
	}

	#[test]
	fn test_conversions_delegate_to_realscalar() {
		// from_f64 delegates to from_f64_const (infallible cast)
		let v: f32 = Scalar::from_f64(3.14);
		assert_relative_eq!(v, 3.14_f32, epsilon = 1e-6);

		// to_f64 delegates to to_f64_lossy
		let back: f64 = v.to_f64();
		assert_relative_eq!(back, 3.14, epsilon = 1e-6);
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
