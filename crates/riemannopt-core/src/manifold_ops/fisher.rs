//! Fisher information matrix approximation methods.
//!
//! This module provides enums and utilities for different approaches
//! to approximating the Fisher information matrix in natural gradient
//! and other second-order optimization methods.

use std::fmt;

/// Fisher information approximation method.
///
/// The Fisher information matrix is central to natural gradient methods
/// but can be expensive to compute exactly. This enum represents different
/// approximation strategies with varying trade-offs between accuracy and
/// computational cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FisherApproximation {
    /// Compute exact Fisher information matrix.
    ///
    /// This is the most accurate but also most expensive option,
    /// requiring computation of the full Hessian of the log-likelihood.
    Exact,
    
    /// Use diagonal approximation of Fisher.
    ///
    /// Only computes and stores the diagonal elements of the Fisher matrix,
    /// significantly reducing memory and computation requirements.
    Diagonal,
    
    /// Use empirical Fisher (outer product of gradients).
    ///
    /// Approximates the Fisher matrix using sample gradients:
    /// F ≈ E[gg^T] where g is the gradient.
    Empirical,
    
    /// Use KFAC (Kronecker-Factored Approximate Curvature).
    ///
    /// Approximates the Fisher matrix as a Kronecker product,
    /// particularly effective for neural networks with specific structure.
    KFAC,
    
    /// Use block-diagonal approximation.
    ///
    /// Divides parameters into blocks and only computes Fisher
    /// within each block, ignoring cross-block correlations.
    BlockDiagonal,
    
    /// Use low-rank approximation.
    ///
    /// Approximates the Fisher matrix using a low-rank factorization,
    /// balancing accuracy and computational efficiency.
    LowRank { rank: usize },
}

impl FisherApproximation {
    /// Returns a human-readable name for this approximation method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Exact => "Exact Fisher",
            Self::Diagonal => "Diagonal Fisher",
            Self::Empirical => "Empirical Fisher",
            Self::KFAC => "KFAC",
            Self::BlockDiagonal => "Block-Diagonal Fisher",
            Self::LowRank { .. } => "Low-Rank Fisher",
        }
    }
    
    /// Returns whether this approximation requires gradient samples.
    pub fn requires_samples(&self) -> bool {
        matches!(self, Self::Empirical | Self::KFAC)
    }
    
    /// Returns whether this approximation is diagonal.
    pub fn is_diagonal(&self) -> bool {
        matches!(self, Self::Diagonal)
    }
    
    /// Returns the default approximation method.
    pub fn default_approximation() -> Self {
        Self::Diagonal
    }
}

impl Default for FisherApproximation {
    fn default() -> Self {
        Self::default_approximation()
    }
}

impl fmt::Display for FisherApproximation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LowRank { rank } => write!(f, "Low-Rank Fisher (rank={})", rank),
            _ => write!(f, "{}", self.name()),
        }
    }
}

/// Configuration for Fisher matrix computation.
#[derive(Debug, Clone)]
pub struct FisherConfig {
    /// Approximation method to use
    pub approximation: FisherApproximation,
    /// Damping factor for numerical stability (λ in (F + λI)^{-1})
    pub damping: f64,
    /// Number of samples for empirical estimation
    pub num_samples: usize,
    /// Update frequency (recompute every N iterations)
    pub update_freq: usize,
    /// Minimum eigenvalue for regularization
    pub min_eigenvalue: f64,
}

impl Default for FisherConfig {
    fn default() -> Self {
        Self {
            approximation: FisherApproximation::default(),
            damping: 1e-4,
            num_samples: 100,
            update_freq: 10,
            min_eigenvalue: 1e-8,
        }
    }
}

impl FisherConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Sets the approximation method.
    pub fn with_approximation(mut self, approximation: FisherApproximation) -> Self {
        self.approximation = approximation;
        self
    }
    
    /// Sets the damping factor.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }
    
    /// Sets the number of samples for empirical estimation.
    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
        self
    }
    
    /// Sets the update frequency.
    pub fn with_update_freq(mut self, update_freq: usize) -> Self {
        self.update_freq = update_freq;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fisher_approximation() {
        assert_eq!(FisherApproximation::Exact.name(), "Exact Fisher");
        assert_eq!(FisherApproximation::Diagonal.name(), "Diagonal Fisher");
        assert_eq!(FisherApproximation::KFAC.name(), "KFAC");
        
        assert!(FisherApproximation::Empirical.requires_samples());
        assert!(FisherApproximation::KFAC.requires_samples());
        assert!(!FisherApproximation::Diagonal.requires_samples());
        
        assert!(FisherApproximation::Diagonal.is_diagonal());
        assert!(!FisherApproximation::Exact.is_diagonal());
    }
    
    #[test]
    fn test_fisher_approximation_display() {
        assert_eq!(format!("{}", FisherApproximation::Exact), "Exact Fisher");
        assert_eq!(
            format!("{}", FisherApproximation::LowRank { rank: 10 }),
            "Low-Rank Fisher (rank=10)"
        );
    }
    
    #[test]
    fn test_fisher_config() {
        let config = FisherConfig::new()
            .with_approximation(FisherApproximation::KFAC)
            .with_damping(1e-3)
            .with_num_samples(200)
            .with_update_freq(5);
        
        assert_eq!(config.approximation, FisherApproximation::KFAC);
        assert_eq!(config.damping, 1e-3);
        assert_eq!(config.num_samples, 200);
        assert_eq!(config.update_freq, 5);
    }
}