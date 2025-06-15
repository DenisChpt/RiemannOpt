//! Type definitions and aliases for Riemannian optimization.
//!
//! This module provides common type aliases, traits for numeric types,
//! and constants used throughout the library.

use nalgebra::{Const, Dim, DimName, Dyn, OMatrix, OVector, RealField, Scalar as NalgebraScalar};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

/// Trait for scalar types used in optimization (f32 or f64).
///
/// This trait combines all the necessary numeric traits required
/// for Riemannian optimization algorithms.
pub trait Scalar:
    NalgebraScalar
    + RealField
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
}

impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const DEFAULT_TOLERANCE: Self = 1e-6;
    const DEFAULT_GRADIENT_TOLERANCE: Self = 1e-8;
    const MANIFOLD_TOLERANCE: Self = 1e-12;
    const ORTHOGONALITY_TOLERANCE: Self = 1e-12;
    const MAX_STEP_SIZE: Self = 1e6;
    const MIN_STEP_SIZE: Self = 1e-16;
}

/// Type alias for a dynamically-sized matrix.
pub type DMatrix<T> = OMatrix<T, Dyn, Dyn>;

/// Type alias for a statically-sized matrix.
pub type SMatrix<T, const R: usize, const C: usize> = OMatrix<T, Const<R>, Const<C>>;

/// Type alias for a square matrix with dynamic size.
pub type DSquareMatrix<T> = OMatrix<T, Dyn, Dyn>;

/// Type alias for a square matrix with static size.
pub type SSquareMatrix<T, const N: usize> = OMatrix<T, Const<N>, Const<N>>;

/// Type alias for a dynamically-sized vector.
pub type DVector<T> = OVector<T, Dyn>;

/// Type alias for a statically-sized vector.
pub type SVector<T, const N: usize> = OVector<T, Const<N>>;

/// Type alias for a general matrix with potentially different row and column dimensions.
pub type Matrix<T, R, C> = OMatrix<T, R, C>;

/// Type alias for a general vector.
pub type Vector<T, D> = OVector<T, D>;

/// Dimension trait for compile-time dimension checking.
pub trait Dimension: Dim + DimName + Copy + Debug + Send + Sync + 'static {}

impl<D> Dimension for D where D: Dim + DimName + Copy + Debug + Send + Sync + 'static {}

/// Type alias for dimensions that can be either static or dynamic.
pub type DimOrDynamic<const N: usize> = Const<N>;

/// Helper trait for operations that work with both static and dynamic dimensions.
pub trait DimensionOps: Dim {
    /// Get the value of the dimension if it's known at compile time.
    fn try_to_usize(&self) -> Option<usize>;

    /// Check if this dimension equals another dimension.
    fn is_equal(&self, other: &Self) -> bool;
}

impl DimensionOps for Dyn {
    fn try_to_usize(&self) -> Option<usize> {
        Some(self.value())
    }

    fn is_equal(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl<const N: usize> DimensionOps for Const<N> {
    fn try_to_usize(&self) -> Option<usize> {
        Some(N)
    }

    fn is_equal(&self, _other: &Self) -> bool {
        true // Const<N> dimensions are always equal if they have the same N
    }
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
    fn test_matrix_type_aliases() {
        // Test dynamic matrix
        let _dm: DMatrix<f64> = DMatrix::zeros(3, 4);

        // Test static matrix
        let _sm: SMatrix<f64, 3, 4> = SMatrix::zeros();

        // Test square matrices
        let _dsq: DSquareMatrix<f64> = DSquareMatrix::identity(5, 5);
        let _ssq: SSquareMatrix<f64, 5> = SSquareMatrix::identity();

        // Test vectors
        let _dv: DVector<f64> = DVector::zeros(10);
        let _sv: SVector<f64, 10> = SVector::zeros();
    }

    #[test]
    fn test_dimension_ops() {
        let dyn_dim = Dyn(5);
        assert_eq!(dyn_dim.try_to_usize(), Some(5));

        let const_dim = Const::<5>;
        assert_eq!(const_dim.try_to_usize(), Some(5));

        assert!(dyn_dim.is_equal(&Dyn(5)));
        assert!(!dyn_dim.is_equal(&Dyn(6)));
        assert!(const_dim.is_equal(&Const::<5>));
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
