//! Backend adapter for sparse matrix operations.

use crate::{
    compute::backend::{ComputeBackend, CpuBackend},
    compute::specialized::sparse::{CsrMatrix, SparseUtils},
    error::Result,
    types::Scalar,
};
use nalgebra::{DVector, DMatrix};
use std::fmt::Debug;

/// A backend wrapper that automatically uses sparse operations when beneficial.
#[derive(Debug)]
pub struct SparseAwareBackend<T: Scalar> {
    /// The underlying backend
    base_backend: CpuBackend<T>,
    /// Sparsity threshold (0.0 to 1.0)
    sparsity_threshold: f64,
    /// Whether to auto-detect sparsity
    auto_detect: bool,
}

impl<T: Scalar> SparseAwareBackend<T> {
    /// Creates a new sparse-aware backend.
    pub fn new() -> Self {
        Self {
            base_backend: CpuBackend::new(),
            sparsity_threshold: 0.9, // Use sparse if >90% zeros
            auto_detect: true,
        }
    }
    
    /// Sets the sparsity threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.sparsity_threshold = threshold.clamp(0.0, 1.0);
        self
    }
    
    /// Enables or disables auto-detection of sparsity.
    pub fn with_auto_detect(mut self, auto_detect: bool) -> Self {
        self.auto_detect = auto_detect;
        self
    }
}

impl<T: Scalar> ComputeBackend<T> for SparseAwareBackend<T> {
    fn name(&self) -> &str {
        "sparse_aware_cpu"
    }
    
    fn is_available(&self) -> bool {
        true
    }
    
    fn preferred_dimension_threshold(&self) -> usize {
        100 // Sparse operations beneficial for larger problems
    }
    
    // Most operations delegate to base backend
    fn dot(&self, a: &DVector<T>, b: &DVector<T>) -> Result<T> {
        self.base_backend.dot(a, b)
    }
    
    fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) -> Result<()> {
        self.base_backend.axpy(alpha, x, y)
    }
    
    fn scal(&self, alpha: T, x: &mut DVector<T>) -> Result<()> {
        self.base_backend.scal(alpha, x)
    }
    
    fn norm(&self, x: &DVector<T>) -> Result<T> {
        self.base_backend.norm(x)
    }
    
    fn normalize(&self, x: &mut DVector<T>) -> Result<()> {
        self.base_backend.normalize(x)
    }
    
    // Matrix operations check for sparsity
    fn gemv(
        &self,
        alpha: T,
        a: &DMatrix<T>,
        x: &DVector<T>,
        beta: T,
        y: &mut DVector<T>,
    ) -> Result<()> {
        if self.auto_detect && SparseUtils::should_use_sparse(a, self.sparsity_threshold) {
            // Convert to sparse and perform SpMV
            let csr = CsrMatrix::from_dense(a, T::zero());
            csr.spmv_scaled(alpha, x, beta, y)?;
            Ok(())
        } else {
            // Use dense operations
            self.base_backend.gemv(alpha, a, x, beta, y)
        }
    }
    
    fn gemm(
        &self,
        alpha: T,
        a: &DMatrix<T>,
        b: &DMatrix<T>,
        beta: T,
        c: &mut DMatrix<T>,
    ) -> Result<()> {
        // For now, always use dense for matrix-matrix
        // TODO: Implement sparse matrix multiplication
        self.base_backend.gemm(alpha, a, b, beta, c)
    }
    
    fn element_wise_add(
        &self,
        a: &DVector<T>,
        b: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        self.base_backend.element_wise_add(a, b, result)
    }
    
    fn element_wise_mul(
        &self,
        a: &DVector<T>,
        b: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        self.base_backend.element_wise_mul(a, b, result)
    }
    
    fn batch_dot(
        &self,
        pairs: &[(DVector<T>, DVector<T>)],
    ) -> Result<Vec<T>> {
        self.base_backend.batch_dot(pairs)
    }
    
    fn batch_normalize(
        &self,
        vectors: &mut [DVector<T>],
    ) -> Result<()> {
        self.base_backend.batch_normalize(vectors)
    }
    
    fn prefers_batched_operations(&self) -> bool {
        true
    }
    
    fn optimal_batch_size(&self) -> usize {
        32 // Smaller batches for sparse operations
    }
}

/// Extension trait for sparse matrix operations.
pub trait SparseMatrixOps<T: Scalar> {
    /// Performs sparse matrix-vector multiplication.
    fn sparse_gemv(
        &self,
        matrix: &CsrMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
    ) -> Result<()>;
    
    /// Performs transpose sparse matrix-vector multiplication.
    fn sparse_gemv_transpose(
        &self,
        matrix: &CsrMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
    ) -> Result<()>;
}

impl<T: Scalar> SparseMatrixOps<T> for dyn ComputeBackend<T> {
    fn sparse_gemv(
        &self,
        matrix: &CsrMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
    ) -> Result<()> {
        matrix.spmv(x, y)
    }
    
    fn sparse_gemv_transpose(
        &self,
        matrix: &CsrMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
    ) -> Result<()> {
        let transposed = matrix.transpose();
        transposed.spmv(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    
    #[test]
    fn test_sparse_aware_backend() {
        let backend = SparseAwareBackend::<f64>::new()
            .with_threshold(0.5); // Use sparse if >50% zeros
        
        // Create a sparse matrix
        let mut a = DMatrix::zeros(5, 5);
        a[(0, 0)] = 1.0;
        a[(1, 1)] = 2.0;
        a[(2, 2)] = 3.0;
        a[(3, 3)] = 4.0;
        a[(4, 4)] = 5.0;
        
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut y = DVector::zeros(5);
        
        // Should use sparse operations
        backend.gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
        
        let expected = DVector::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);
        assert_eq!(y, expected);
    }
    
    #[test]
    fn test_sparse_extension() {
        let backend: Box<dyn ComputeBackend<f64>> = Box::new(CpuBackend::new());
        
        let mut coo = crate::compute::specialized::sparse::CooMatrix::new(3, 3);
        coo.push(0, 0, 2.0).unwrap();
        coo.push(1, 1, 3.0).unwrap();
        coo.push(2, 2, 4.0).unwrap();
        let csr = coo.to_csr();
        
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = DVector::zeros(3);
        
        backend.sparse_gemv(&csr, &x, &mut y).unwrap();
        
        assert_eq!(y, DVector::from_vec(vec![2.0, 6.0, 12.0]));
    }
}