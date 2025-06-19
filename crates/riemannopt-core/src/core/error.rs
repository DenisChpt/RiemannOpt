//! Error types for Riemannian manifold operations.
//!
//! This module defines the core error types used throughout the library
//! for manifold-specific operations and numerical computations.

use thiserror::Error;

/// Errors that can occur during manifold operations.
#[derive(Debug, Clone, Error)]
pub enum ManifoldError {
    /// Point is not on the manifold.
    ///
    /// This error occurs when a point fails to satisfy the manifold constraints
    /// within numerical tolerance.
    #[error("Point is not on the manifold: {reason}")]
    InvalidPoint {
        /// Description of why the point is invalid
        reason: String,
    },

    /// Vector is not in the tangent space.
    ///
    /// This error occurs when a vector does not belong to the tangent space
    /// at a given point on the manifold.
    #[error("Vector is not in the tangent space: {reason}")]
    InvalidTangent {
        /// Description of why the tangent vector is invalid
        reason: String,
    },

    /// Dimension mismatch between tensors.
    ///
    /// This error occurs when operations involve tensors with incompatible dimensions.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimensions
        expected: String,
        /// Actual dimensions
        actual: String,
    },

    /// Numerical instability detected.
    ///
    /// This error occurs when numerical operations become unstable,
    /// such as division by near-zero values or loss of precision.
    #[error("Numerical instability detected: {reason}")]
    NumericalError {
        /// Description of the numerical issue
        reason: String,
    },

    /// Method or feature not implemented.
    ///
    /// This error is used for optional methods that are not implemented
    /// for a particular manifold.
    #[error("Feature not implemented: {feature}")]
    NotImplemented {
        /// Name of the unimplemented feature
        feature: String,
    },
}

impl ManifoldError {
    /// Create an InvalidPoint error with a custom reason.
    pub fn invalid_point<S: Into<String>>(reason: S) -> Self {
        Self::InvalidPoint {
            reason: reason.into(),
        }
    }

    /// Create an InvalidTangent error with a custom reason.
    pub fn invalid_tangent<S: Into<String>>(reason: S) -> Self {
        Self::InvalidTangent {
            reason: reason.into(),
        }
    }

    /// Create a DimensionMismatch error.
    pub fn dimension_mismatch<S1, S2>(expected: S1, actual: S2) -> Self 
    where
        S1: std::fmt::Display,
        S2: std::fmt::Display,
    {
        Self::DimensionMismatch {
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }

    /// Create a NumericalError with a custom reason.
    pub fn numerical_error<S: Into<String>>(reason: S) -> Self {
        Self::NumericalError {
            reason: reason.into(),
        }
    }

    /// Create a NotImplemented error for a specific feature.
    pub fn not_implemented<S: Into<String>>(feature: S) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }
    
    /// Create an InvalidParameter error (maps to InvalidPoint for simplicity).
    pub fn invalid_parameter<S: Into<String>>(reason: S) -> Self {
        Self::InvalidPoint {
            reason: reason.into(),
        }
    }
}

/// Errors that can occur during optimization.
#[derive(Debug, Clone, Error)]
pub enum OptimizerError {
    /// Line search failed to find an acceptable step.
    ///
    /// This error occurs when the line search algorithm cannot find
    /// a step size that satisfies the sufficient decrease conditions.
    #[error("Line search failed: {reason}")]
    LineSearchFailed {
        /// Description of why the line search failed
        reason: String,
        /// Number of iterations attempted
        iterations: usize,
        /// Last step size tried
        last_step_size: f64,
        /// Function value at the starting point
        initial_value: f64,
    },

    /// Maximum number of iterations reached without convergence.
    ///
    /// This error indicates that the optimizer has reached its iteration
    /// limit without satisfying the convergence criteria.
    #[error("Maximum iterations ({max_iterations}) reached without convergence")]
    MaxIterationsReached {
        /// Maximum number of iterations allowed
        max_iterations: usize,
        /// Final function value
        final_value: f64,
        /// Final gradient norm
        final_gradient_norm: f64,
        /// Convergence tolerance that was not met
        tolerance: f64,
    },

    /// Invalid optimizer configuration.
    ///
    /// This error occurs when the optimizer is configured with invalid
    /// parameters (e.g., negative learning rate, invalid momentum).
    #[error("Invalid optimizer configuration: {reason}")]
    InvalidConfiguration {
        /// Description of the configuration error
        reason: String,
        /// Name of the invalid parameter
        parameter: String,
        /// Value that was invalid
        value: String,
    },

    /// Propagated manifold error.
    ///
    /// This error wraps manifold-specific errors that occur during
    /// optimization operations.
    #[error("Manifold operation failed: {0}")]
    ManifoldError(#[from] ManifoldError),
    
    /// Invalid search direction.
    ///
    /// This error occurs when the search direction is not a descent direction.
    #[error("Invalid search direction: not a descent direction")]
    InvalidSearchDirection,
}

impl OptimizerError {
    /// Create a LineSearchFailed error with detailed context.
    pub fn line_search_failed<S: Into<String>>(
        reason: S,
        iterations: usize,
        last_step_size: f64,
        initial_value: f64,
    ) -> Self {
        Self::LineSearchFailed {
            reason: reason.into(),
            iterations,
            last_step_size,
            initial_value,
        }
    }

    /// Create a MaxIterationsReached error with convergence information.
    pub fn max_iterations_reached(
        max_iterations: usize,
        final_value: f64,
        final_gradient_norm: f64,
        tolerance: f64,
    ) -> Self {
        Self::MaxIterationsReached {
            max_iterations,
            final_value,
            final_gradient_norm,
            tolerance,
        }
    }

    /// Create an InvalidConfiguration error.
    pub fn invalid_configuration<S1, S2, S3>(reason: S1, parameter: S2, value: S3) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::InvalidConfiguration {
            reason: reason.into(),
            parameter: parameter.into(),
            value: value.into(),
        }
    }
    
    /// Create an InvalidState error.
    pub fn invalid_state<S: Into<String>>(reason: S) -> Self {
        Self::InvalidConfiguration {
            reason: reason.into(),
            parameter: "state".to_string(),
            value: "invalid".to_string(),
        }
    }
}

/// Result type alias for operations that can produce ManifoldError.
pub type Result<T> = std::result::Result<T, ManifoldError>;

/// Result type alias for optimizer operations.
pub type OptimizerResult<T> = std::result::Result<T, OptimizerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = ManifoldError::invalid_point("determinant is negative");
        assert!(matches!(err, ManifoldError::InvalidPoint { .. }));
        assert_eq!(
            err.to_string(),
            "Point is not on the manifold: determinant is negative"
        );

        let err = ManifoldError::dimension_mismatch("(3, 3)", "(4, 4)");
        assert!(matches!(err, ManifoldError::DimensionMismatch { .. }));
        assert_eq!(
            err.to_string(),
            "Dimension mismatch: expected (3, 3), got (4, 4)"
        );
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            ManifoldError::invalid_point("not unit norm"),
            ManifoldError::invalid_tangent("not orthogonal to point"),
            ManifoldError::dimension_mismatch("square matrix", "rectangular matrix"),
            ManifoldError::numerical_error("eigenvalue computation failed"),
            ManifoldError::not_implemented("parallel transport"),
        ];

        for err in errors {
            // Ensure Display trait is implemented and produces non-empty strings
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn test_optimizer_error_creation() {
        // Test LineSearchFailed
        let err = OptimizerError::line_search_failed("step size too small", 10, 1e-10, 100.0);
        assert!(matches!(err, OptimizerError::LineSearchFailed { .. }));
        assert!(err.to_string().contains("Line search failed"));

        // Test MaxIterationsReached
        let err = OptimizerError::max_iterations_reached(1000, 0.5, 0.001, 1e-6);
        assert!(matches!(err, OptimizerError::MaxIterationsReached { .. }));
        assert!(err.to_string().contains("1000"));

        // Test InvalidConfiguration
        let err =
            OptimizerError::invalid_configuration("must be positive", "learning_rate", "-0.1");
        assert!(matches!(err, OptimizerError::InvalidConfiguration { .. }));
        assert!(err.to_string().contains("Invalid optimizer configuration"));
    }

    #[test]
    fn test_optimizer_error_context() {
        // Test that LineSearchFailed contains all context
        let err =
            OptimizerError::line_search_failed("Wolfe conditions not satisfied", 25, 1e-8, 42.0);

        if let OptimizerError::LineSearchFailed {
            reason,
            iterations,
            last_step_size,
            initial_value,
        } = err
        {
            assert_eq!(reason, "Wolfe conditions not satisfied");
            assert_eq!(iterations, 25);
            assert_eq!(last_step_size, 1e-8);
            assert_eq!(initial_value, 42.0);
        } else {
            panic!("Expected LineSearchFailed variant");
        }

        // Test MaxIterationsReached context
        let err = OptimizerError::max_iterations_reached(500, 1.23, 0.456, 1e-3);

        if let OptimizerError::MaxIterationsReached {
            max_iterations,
            final_value,
            final_gradient_norm,
            tolerance,
        } = err
        {
            assert_eq!(max_iterations, 500);
            assert_eq!(final_value, 1.23);
            assert_eq!(final_gradient_norm, 0.456);
            assert_eq!(tolerance, 1e-3);
        } else {
            panic!("Expected MaxIterationsReached variant");
        }
    }

    #[test]
    fn test_manifold_error_propagation() {
        // Test that ManifoldError can be converted to OptimizerError
        let manifold_err = ManifoldError::invalid_point("not on sphere");
        let optimizer_err: OptimizerError = manifold_err.into();

        assert!(matches!(optimizer_err, OptimizerError::ManifoldError(_)));
        assert!(optimizer_err
            .to_string()
            .contains("Manifold operation failed"));
        assert!(optimizer_err.to_string().contains("not on sphere"));
    }

    #[test]
    fn test_optimizer_error_display() {
        let errors = vec![
            OptimizerError::line_search_failed("step size underflow", 50, 1e-16, 10.0),
            OptimizerError::max_iterations_reached(10000, 0.1, 0.01, 1e-6),
            OptimizerError::invalid_configuration("negative value", "momentum", "-0.5"),
            OptimizerError::ManifoldError(ManifoldError::numerical_error("singular matrix")),
        ];

        for err in errors {
            // Ensure Display trait is implemented and produces non-empty strings
            assert!(!err.to_string().is_empty());
        }
    }
}
