//! Core traits for backend-agnostic operations.
//!
//! This module defines traits that allow different parts of the library
//! to work with different computation backends transparently.

use crate::{
    compute::backend::ComputeBackend,
    error::Result,
    types::Scalar,
};
use nalgebra::DVector;
use std::sync::Arc;

/// Trait for types that can use a compute backend.
pub trait BackendUser<T: Scalar> {
    /// Gets the current backend.
    fn backend(&self) -> &dyn ComputeBackend<T>;
    
    /// Sets a new backend.
    fn set_backend(&mut self, backend: Arc<dyn ComputeBackend<T>>);
}

/// Trait for operations that can be accelerated by backends.
pub trait BackendOps<T: Scalar> {
    /// Performs a dot product using the backend.
    fn backend_dot(&self, a: &DVector<T>, b: &DVector<T>) -> Result<T> {
        self.backend().dot(a, b)
    }
    
    /// Performs axpy operation using the backend.
    fn backend_axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) -> Result<()> {
        self.backend().axpy(alpha, x, y)
    }
    
    /// Computes norm using the backend.
    fn backend_norm(&self, x: &DVector<T>) -> Result<T> {
        self.backend().norm(x)
    }
    
    /// Normalizes a vector using the backend.
    fn backend_normalize(&self, x: &mut DVector<T>) -> Result<()> {
        self.backend().normalize(x)
    }
    
    /// Gets the backend.
    fn backend(&self) -> &dyn ComputeBackend<T>;
}

/// Extension trait for cost functions to use backends.
pub trait CostFunctionBackend<T: Scalar, const D: usize>: Sized {
    /// Wraps the cost function with a specific backend.
    fn with_backend(self, backend: Arc<dyn ComputeBackend<T>>) -> BackendCostFunction<T, D, Self>;
}

/// A cost function wrapper that uses a specific backend.
#[derive(Debug)]
pub struct BackendCostFunction<T: Scalar, const D: usize, F> {
    inner: F,
    backend: Arc<dyn ComputeBackend<T>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const D: usize, F> BackendCostFunction<T, D, F> {
    /// Creates a new backend-enabled cost function.
    pub fn new(inner: F, backend: Arc<dyn ComputeBackend<T>>) -> Self {
        Self {
            inner,
            backend,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Gets the inner cost function.
    pub fn inner(&self) -> &F {
        &self.inner
    }
    
    /// Gets the backend.
    pub fn backend(&self) -> &dyn ComputeBackend<T> {
        self.backend.as_ref()
    }
}

/// Trait for manifold operations that can use backends.
pub trait ManifoldBackendOps<T: Scalar, const D: usize> {
    /// Computes inner product using the backend.
    fn backend_inner_product(
        &self,
        backend: &dyn ComputeBackend<T>,
        point: &DVector<T>,
        v1: &DVector<T>,
        v2: &DVector<T>,
    ) -> Result<T>;
    
    /// Projects to tangent space using the backend.
    fn backend_project_tangent(
        &self,
        backend: &dyn ComputeBackend<T>,
        point: &DVector<T>,
        vector: &mut DVector<T>,
    ) -> Result<()>;
    
    /// Performs retraction using the backend.
    fn backend_retract(
        &self,
        backend: &dyn ComputeBackend<T>,
        point: &DVector<T>,
        tangent: &DVector<T>,
        t: T,
    ) -> Result<DVector<T>>;
}

/// Backend-aware optimizer state.
pub trait BackendOptimizerState<T: Scalar> {
    /// Gets the backend used by this state.
    fn backend(&self) -> &dyn ComputeBackend<T>;
    
    /// Updates the backend.
    fn set_backend(&mut self, backend: Arc<dyn ComputeBackend<T>>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::backend::CpuBackend;
    
    struct TestBackendUser {
        backend: Arc<dyn ComputeBackend<f64>>,
    }
    
    impl BackendUser<f64> for TestBackendUser {
        fn backend(&self) -> &dyn ComputeBackend<f64> {
            self.backend.as_ref()
        }
        
        fn set_backend(&mut self, backend: Arc<dyn ComputeBackend<f64>>) {
            self.backend = backend;
        }
    }
    
    impl BackendOps<f64> for TestBackendUser {
        fn backend(&self) -> &dyn ComputeBackend<f64> {
            self.backend.as_ref()
        }
    }
    
    #[test]
    fn test_backend_ops() {
        let backend = Arc::new(CpuBackend::new());
        let user = TestBackendUser {
            backend: backend.clone(),
        };
        
        let a = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        
        let dot = user.backend_dot(&a, &b).unwrap();
        assert_eq!(dot, 32.0);
        
        let norm = user.backend_norm(&a).unwrap();
        assert!((norm - 14.0_f64.sqrt()).abs() < 1e-10);
    }
}