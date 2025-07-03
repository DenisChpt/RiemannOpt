//! Test utilities for property-based testing of manifolds.
//!
//! This module provides utilities for testing mathematical properties
//! of Riemannian manifolds, including retractions, metrics, and other
//! geometric operations.

#[allow(unused_imports)]
use crate::{
    error::Result,
    core::manifold::Manifold,
    memory::workspace::Workspace,
    manifold_ops::retraction::Retraction,
    types::Scalar,
};
// Removed unused imports

// Re-export test manifolds for use in other crates
#[cfg(any(test, feature = "test-utils"))]
pub use crate::test_manifolds::MinimalTestManifold;

/// Configuration for property-based tests
#[derive(Debug, Clone)]
pub struct PropertyTestConfig<T> {
    /// Number of random points to test
    pub num_points: usize,
    /// Number of random vectors to test per point
    pub num_vectors: usize,
    /// Tolerance for numerical comparisons
    pub tolerance: T,
    /// Step size for numerical derivatives
    pub step_size: T,
}

impl<T: Scalar> Default for PropertyTestConfig<T> {
    fn default() -> Self {
        Self {
            num_points: 10,
            num_vectors: 5,
            tolerance: <T as Scalar>::from_f64(1e-10),
            step_size: <T as Scalar>::from_f64(1e-8),
        }
    }
}

/// Result of a property test
#[derive(Debug)]
pub struct PropertyTestResult<T> {
    /// Whether all tests passed
    pub passed: bool,
    /// Maximum error observed
    pub max_error: T,
    /// Number of tests performed
    pub num_tests: usize,
    /// List of failure messages
    pub errors: Vec<String>,
}

/// Property tests for manifold implementations
pub struct ManifoldPropertyTests;

// TODO: Update all property tests to work with associated types
// The tests below are temporarily commented out until the migration is complete

/*
impl ManifoldPropertyTests {
    /// Tests that retraction at zero returns the same point
    pub fn test_retraction_at_zero<T, D, M, R>(
        manifold: &M,
        retraction: &R,
        config: &PropertyTestConfig<T>,
    ) -> PropertyTestResult<T>
    where
        T: Scalar,
        D: Dim,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        // Implementation removed - needs update for associated types
        PropertyTestResult {
            passed: true,
            max_error: T::zero(),
            num_tests: 0,
            errors: vec!["Test not implemented for associated types".to_string()],
        }
    }

    // Other test methods would follow the same pattern...
}
*/

/// Utility functions for testing
pub mod test_helpers {
    use super::*;

    /// Generates a random unit vector
    pub fn random_unit_vector<T: Scalar>(n: usize) -> crate::types::DVector<T> {
        let mut v = crate::types::DVector::zeros(n);
        for i in 0..n {
            v[i] = <T as Scalar>::from_f64(((i + 1) as f64).sin());
        }
        let norm = v.norm();
        if norm > T::epsilon() {
            v /= norm;
        }
        v
    }

    /// Computes relative error between two values
    pub fn relative_error<T: Scalar>(actual: T, expected: T) -> T {
        if <T as num_traits::Float>::abs(expected) < T::epsilon() {
            <T as num_traits::Float>::abs(actual)
        } else {
            <T as num_traits::Float>::abs(actual - expected) / <T as num_traits::Float>::abs(expected)
        }
    }

    /// Checks if two values are approximately equal
    pub fn approx_equal<T: Scalar>(a: T, b: T, tolerance: T) -> bool {
        <T as num_traits::Float>::abs(a - b) < tolerance
    }
}