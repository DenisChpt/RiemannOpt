//! Backend abstraction for computation.
//!
//! This module defines the trait for different computation backends
//! (CPU, GPU, etc.) and provides a unified interface for selecting
//! the optimal backend at runtime.

use crate::{
    error::{ManifoldError as Error, Result},
    types::Scalar,
};
use nalgebra::{DVector, DMatrix};
use std::fmt::Debug;

/// Trait for computation backends.
///
/// This trait defines all the fundamental operations that a backend
/// must support. To make it dyn-compatible, we use a type-erased approach
/// where each backend is specific to a scalar type.
pub trait ComputeBackend<T: Scalar>: Debug + Send + Sync {
    /// Backend name for identification.
    fn name(&self) -> &str;
    
    /// Check if this backend is available on the current system.
    fn is_available(&self) -> bool;
    
    /// Get the preferred dimension threshold for using this backend.
    fn preferred_dimension_threshold(&self) -> usize;
    
    /// Vector operations
    fn dot(&self, a: &DVector<T>, b: &DVector<T>) -> Result<T>;
    
    fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) -> Result<()>;
    
    fn scal(&self, alpha: T, x: &mut DVector<T>) -> Result<()>;
    
    fn norm(&self, x: &DVector<T>) -> Result<T>;
    
    fn normalize(&self, x: &mut DVector<T>) -> Result<()>;
    
    /// Matrix-vector operations
    fn gemv(
        &self,
        alpha: T,
        a: &DMatrix<T>,
        x: &DVector<T>,
        beta: T,
        y: &mut DVector<T>,
    ) -> Result<()>;
    
    /// Matrix-matrix operations
    fn gemm(
        &self,
        alpha: T,
        a: &DMatrix<T>,
        b: &DMatrix<T>,
        beta: T,
        c: &mut DMatrix<T>,
    ) -> Result<()>;
    
    /// Element-wise operations
    fn element_wise_add(
        &self,
        a: &DVector<T>,
        b: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()>;
    
    fn element_wise_mul(
        &self,
        a: &DVector<T>,
        b: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()>;
    
    /// Batch operations for efficiency
    fn batch_dot(
        &self,
        pairs: &[(DVector<T>, DVector<T>)],
    ) -> Result<Vec<T>>;
    
    fn batch_normalize(
        &self,
        vectors: &mut [DVector<T>],
    ) -> Result<()>;
    
    /// Memory transfer operations (for GPU backends)
    fn can_transfer(&self) -> bool {
        false
    }
    
    fn transfer_to_device(&self, _data: &DVector<T>) -> Result<()> {
        Ok(())
    }
    
    fn transfer_from_device(&self, _data: &mut DVector<T>) -> Result<()> {
        Ok(())
    }
    
    /// Performance hints
    fn prefers_batched_operations(&self) -> bool {
        false
    }
    
    fn optimal_batch_size(&self) -> usize {
        1
    }
}

/// Backend selection strategy.
#[derive(Debug, Clone)]
pub enum BackendSelection {
    /// Always use a specific backend
    Fixed(String),
    /// Automatically select based on problem characteristics
    Auto,
    /// Use CPU for small problems, GPU for large
    Adaptive {
        cpu_threshold: usize,
    },
    /// Custom selection function
    Custom(fn(usize) -> String),
}

/// Backend selector that chooses the optimal backend.
#[derive(Debug)]
pub struct BackendSelector<T: Scalar> {
    backends: Vec<Box<dyn ComputeBackend<T>>>,
    selection: BackendSelection,
    current_backend: String,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> BackendSelector<T> {
    /// Creates a new backend selector.
    pub fn new(selection: BackendSelection) -> Self {
        let backends = Self::discover_backends();
        let current_backend = match &selection {
            BackendSelection::Fixed(name) => name.clone(),
            _ => backends.first()
                .map(|b| b.name().to_string())
                .unwrap_or_else(|| "cpu".to_string()),
        };
        
        Self {
            backends,
            selection,
            current_backend,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Discovers available backends on the system.
    fn discover_backends() -> Vec<Box<dyn ComputeBackend<T>>> {
        let mut backends: Vec<Box<dyn ComputeBackend<T>>> = vec![];
        
        // Always add CPU backend
        backends.push(Box::new(CpuBackend::<T>::new()));
        
        // TODO: Add GPU backend detection here
        // if gpu_available() {
        //     backends.push(Box::new(GpuBackend::new()));
        // }
        
        backends
    }
    
    /// Selects the optimal backend for a given problem size.
    pub fn select_backend(&mut self, dimension: usize) -> &dyn ComputeBackend<T> {
        let backend_name = match &self.selection {
            BackendSelection::Fixed(name) => name.clone(),
            BackendSelection::Auto => self.auto_select(dimension),
            BackendSelection::Adaptive { cpu_threshold } => {
                if dimension < *cpu_threshold {
                    "cpu".to_string()
                } else {
                    self.find_best_gpu_backend()
                        .unwrap_or_else(|| "cpu".to_string())
                }
            },
            BackendSelection::Custom(f) => f(dimension),
        };
        
        self.current_backend = backend_name;
        self.get_backend(&self.current_backend.clone())
            .expect("Selected backend not found")
    }
    
    /// Automatically selects the best backend.
    fn auto_select(&self, dimension: usize) -> String {
        // Simple heuristic: use GPU for large problems if available
        for backend in &self.backends {
            if backend.is_available() && dimension >= backend.preferred_dimension_threshold() {
                return backend.name().to_string();
            }
        }
        
        // Default to CPU
        "cpu".to_string()
    }
    
    /// Finds the best available GPU backend.
    fn find_best_gpu_backend(&self) -> Option<String> {
        self.backends.iter()
            .filter(|b| b.name().contains("gpu") && b.is_available())
            .map(|b| b.name().to_string())
            .next()
    }
    
    /// Gets a backend by name.
    pub fn get_backend(&self, name: &str) -> Option<&dyn ComputeBackend<T>> {
        self.backends.iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }
    
    /// Returns the currently selected backend.
    pub fn current_backend(&self) -> &dyn ComputeBackend<T> {
        self.get_backend(&self.current_backend)
            .expect("Current backend not found")
    }
    
    /// Lists all available backends.
    pub fn available_backends(&self) -> Vec<&str> {
        self.backends.iter()
            .filter(|b| b.is_available())
            .map(|b| b.name())
            .collect()
    }
}

/// CPU backend implementation.
#[derive(Debug)]
pub struct CpuBackend<T: Scalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> CpuBackend<T> {
    /// Creates a new CPU backend.
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> ComputeBackend<T> for CpuBackend<T> {
    fn name(&self) -> &str {
        "cpu"
    }
    
    fn is_available(&self) -> bool {
        true // CPU is always available
    }
    
    fn preferred_dimension_threshold(&self) -> usize {
        0 // CPU can handle any dimension
    }
    
    fn dot(&self, a: &DVector<T>, b: &DVector<T>) -> Result<T> {
        if a.len() != b.len() {
            return Err(Error::DimensionMismatch {
                expected: a.len().to_string(),
                actual: b.len().to_string(),
            });
        }
        
        // For now, use nalgebra directly
        // TODO: Add SIMD optimization when SimdDispatcher is updated
        Ok(a.dot(b))
    }
    
    fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::DimensionMismatch {
                expected: x.len().to_string(),
                actual: y.len().to_string(),
            });
        }
        
        // Use nalgebra
        y.axpy(alpha, x, T::one());
        Ok(())
    }
    
    fn scal(&self, alpha: T, x: &mut DVector<T>) -> Result<()> {
        // Use nalgebra's built-in scaling
        *x *= alpha;
        Ok(())
    }
    
    fn norm(&self, x: &DVector<T>) -> Result<T> {
        // Use nalgebra
        Ok(x.norm())
    }
    
    fn normalize(&self, x: &mut DVector<T>) -> Result<()> {
        let norm = self.norm(x)?;
        if norm > T::zero() {
            self.scal(T::one() / norm, x)?;
        }
        Ok(())
    }
    
    fn gemv(
        &self,
        alpha: T,
        a: &DMatrix<T>,
        x: &DVector<T>,
        beta: T,
        y: &mut DVector<T>,
    ) -> Result<()> {
        if a.ncols() != x.len() {
            return Err(Error::DimensionMismatch {
                expected: a.ncols().to_string(),
                actual: x.len().to_string(),
            });
        }
        if a.nrows() != y.len() {
            return Err(Error::DimensionMismatch {
                expected: a.nrows().to_string(),
                actual: y.len().to_string(),
            });
        }
        
        // y = alpha * A * x + beta * y
        y.gemv(alpha, a, x, beta);
        Ok(())
    }
    
    fn gemm(
        &self,
        alpha: T,
        a: &DMatrix<T>,
        b: &DMatrix<T>,
        beta: T,
        c: &mut DMatrix<T>,
    ) -> Result<()> {
        if a.ncols() != b.nrows() {
            return Err(Error::DimensionMismatch {
                expected: a.ncols().to_string(),
                actual: b.nrows().to_string(),
            });
        }
        if a.nrows() != c.nrows() || b.ncols() != c.ncols() {
            return Err(Error::DimensionMismatch {
                expected: (a.nrows() * b.ncols()).to_string(),
                actual: (c.nrows() * c.ncols()).to_string(),
            });
        }
        
        // C = alpha * A * B + beta * C
        c.gemm(alpha, a, b, beta);
        Ok(())
    }
    
    fn element_wise_add(
        &self,
        a: &DVector<T>,
        b: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(Error::DimensionMismatch {
                expected: a.len().to_string(),
                actual: b.len().to_string(),
            });
        }
        
        // Element-wise operation
        result.copy_from(a);
        *result += b;
        Ok(())
    }
    
    fn element_wise_mul(
        &self,
        a: &DVector<T>,
        b: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(Error::DimensionMismatch {
                expected: a.len().to_string(),
                actual: b.len().to_string(),
            });
        }
        
        // Element-wise multiplication
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
        Ok(())
    }
    
    fn batch_dot(
        &self,
        pairs: &[(DVector<T>, DVector<T>)],
    ) -> Result<Vec<T>> {
        // For small batches, sequential is fine
        if pairs.len() < 4 {
            pairs.iter()
                .map(|(a, b)| self.dot(a, b))
                .collect()
        } else {
            // Use parallel computation for larger batches
            use rayon::prelude::*;
            pairs.par_iter()
                .map(|(a, b)| self.dot(a, b))
                .collect()
        }
    }
    
    fn batch_normalize(
        &self,
        vectors: &mut [DVector<T>],
    ) -> Result<()> {
        // For small batches, sequential is fine
        if vectors.len() < 4 {
            for v in vectors {
                self.normalize(v)?;
            }
        } else {
            // Use parallel computation for larger batches
            use rayon::prelude::*;
            vectors.par_iter_mut()
                .try_for_each(|v| self.normalize(v))?;
        }
        Ok(())
    }
    
    fn prefers_batched_operations(&self) -> bool {
        true // CPU benefits from batching for cache efficiency
    }
    
    fn optimal_batch_size(&self) -> usize {
        // Heuristic based on typical cache sizes
        64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    
    #[test]
    fn test_cpu_backend_operations() {
        let backend = CpuBackend::<f64>::new();
        
        // Test dot product
        let a = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let dot = backend.dot(&a, &b).unwrap();
        assert_eq!(dot, 32.0);
        
        // Test axpy
        let mut y = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        backend.axpy(2.0, &a, &mut y).unwrap();
        assert_eq!(y, DVector::from_vec(vec![3.0, 5.0, 7.0]));
        
        // Test norm
        let norm = backend.norm(&a).unwrap();
        assert!((norm - 14.0_f64.sqrt()).abs() < 1e-10);
        
        // Test normalize
        let mut v = DVector::from_vec(vec![3.0, 4.0]);
        backend.normalize(&mut v).unwrap();
        assert!((v.norm() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_backend_selector() {
        let mut selector = BackendSelector::<f64>::new(BackendSelection::Auto);
        
        // Should have at least CPU backend
        assert!(!selector.available_backends().is_empty());
        assert!(selector.available_backends().contains(&"cpu"));
        
        // Test selection
        let backend = selector.select_backend(100);
        assert_eq!(backend.name(), "cpu");
        
        // Test fixed selection
        let mut fixed_selector = BackendSelector::<f64>::new(BackendSelection::Fixed("cpu".to_string()));
        let backend = fixed_selector.select_backend(1000000);
        assert_eq!(backend.name(), "cpu");
    }
    
    #[test]
    fn test_batch_operations() {
        let backend = CpuBackend::<f64>::new();
        
        // Test batch dot product
        let pairs = vec![
            (DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![3.0, 4.0])),
            (DVector::from_vec(vec![5.0, 6.0]), DVector::from_vec(vec![7.0, 8.0])),
        ];
        let results = backend.batch_dot(&pairs).unwrap();
        assert_eq!(results, vec![11.0, 83.0]);
        
        // Test batch normalize
        let mut vectors = vec![
            DVector::from_vec(vec![3.0, 4.0]),
            DVector::from_vec(vec![5.0, 12.0]),
        ];
        backend.batch_normalize(&mut vectors).unwrap();
        assert!((vectors[0].norm() - 1.0).abs() < 1e-10);
        assert!((vectors[1].norm() - 1.0).abs() < 1e-10);
    }
}