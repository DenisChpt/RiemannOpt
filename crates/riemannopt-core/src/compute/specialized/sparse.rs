//! Sparse matrix support for optimization problems.
//!
//! This module provides efficient operations for sparse matrices,
//! which are common in large-scale optimization problems.

use crate::{
    error::{ManifoldError as Error, Result},
    types::Scalar,
};
use nalgebra::{DVector, DMatrix};
use num_traits::Float;

/// Compressed Sparse Row (CSR) format matrix.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T: Scalar> {
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
    /// Row pointers (length nrows + 1)
    row_ptr: Vec<usize>,
    /// Column indices (length nnz)
    col_idx: Vec<usize>,
    /// Non-zero values (length nnz)
    values: Vec<T>,
}

impl<T: Scalar> CsrMatrix<T> {
    /// Creates a new CSR matrix from raw data.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        // Validate inputs
        if row_ptr.len() != nrows + 1 {
            return Err(Error::DimensionMismatch {
                expected: format!("row_ptr length {}", nrows + 1),
                actual: format!("row_ptr length {}", row_ptr.len()),
            });
        }
        
        let nnz = row_ptr[nrows];
        if col_idx.len() != nnz {
            return Err(Error::DimensionMismatch {
                expected: format!("col_idx length {}", nnz),
                actual: format!("col_idx length {}", col_idx.len()),
            });
        }
        
        if values.len() != nnz {
            return Err(Error::DimensionMismatch {
                expected: format!("values length {}", nnz),
                actual: format!("values length {}", values.len()),
            });
        }
        
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }
    
    /// Creates a CSR matrix from a dense matrix.
    pub fn from_dense(dense: &DMatrix<T>, tolerance: T) -> Self {
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                let val = dense[(i, j)];
                if Float::abs(val) > tolerance {
                    col_idx.push(j);
                    values.push(val);
                }
            }
            row_ptr.push(col_idx.len());
        }
        
        Self {
            nrows: dense.nrows(),
            ncols: dense.ncols(),
            row_ptr,
            col_idx,
            values,
        }
    }
    
    /// Converts to a dense matrix.
    pub fn to_dense(&self) -> DMatrix<T> {
        let mut dense = DMatrix::zeros(self.nrows, self.ncols);
        
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            for k in start..end {
                let j = self.col_idx[k];
                dense[(i, j)] = self.values[k];
            }
        }
        
        dense
    }
    
    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    
    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    
    /// Returns the number of non-zero elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Returns the sparsity (fraction of zero elements).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        let total_elements = self.nrows * self.ncols;
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }
    
    /// Sparse matrix-vector multiplication: y = A * x
    pub fn spmv(&self, x: &DVector<T>, y: &mut DVector<T>) -> Result<()> {
        if x.len() != self.ncols {
            return Err(Error::DimensionMismatch {
                expected: format!("vector length {}", self.ncols),
                actual: format!("vector length {}", x.len()),
            });
        }
        
        if y.len() != self.nrows {
            return Err(Error::DimensionMismatch {
                expected: format!("result length {}", self.nrows),
                actual: format!("result length {}", y.len()),
            });
        }
        
        // Clear output vector
        y.fill(T::zero());
        
        // Perform SpMV
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            let mut sum = T::zero();
            for k in start..end {
                let j = self.col_idx[k];
                sum = sum + self.values[k] * x[j];
            }
            y[i] = sum;
        }
        
        Ok(())
    }
    
    /// Sparse matrix-vector multiplication with scaling: y = alpha * A * x + beta * y
    pub fn spmv_scaled(
        &self,
        alpha: T,
        x: &DVector<T>,
        beta: T,
        y: &mut DVector<T>,
    ) -> Result<()> {
        if x.len() != self.ncols {
            return Err(Error::DimensionMismatch {
                expected: format!("vector length {}", self.ncols),
                actual: format!("vector length {}", x.len()),
            });
        }
        
        if y.len() != self.nrows {
            return Err(Error::DimensionMismatch {
                expected: format!("result length {}", self.nrows),
                actual: format!("result length {}", y.len()),
            });
        }
        
        // Scale existing y values
        if beta != T::one() {
            if beta == T::zero() {
                y.fill(T::zero());
            } else {
                *y *= beta;
            }
        }
        
        // Add alpha * A * x
        if alpha != T::zero() {
            for i in 0..self.nrows {
                let start = self.row_ptr[i];
                let end = self.row_ptr[i + 1];
                
                let mut sum = T::zero();
                for k in start..end {
                    let j = self.col_idx[k];
                    sum = sum + self.values[k] * x[j];
                }
                y[i] = y[i] + alpha * sum;
            }
        }
        
        Ok(())
    }
    
    /// Transpose of the sparse matrix.
    pub fn transpose(&self) -> Self {
        // Count entries per column
        let mut col_counts = vec![0; self.ncols];
        for &j in &self.col_idx {
            col_counts[j] += 1;
        }
        
        // Build row pointers for transpose
        let mut t_row_ptr = vec![0; self.ncols + 1];
        for j in 0..self.ncols {
            t_row_ptr[j + 1] = t_row_ptr[j] + col_counts[j];
        }
        
        // Fill in values and column indices
        let mut t_col_idx = vec![0; self.nnz()];
        let mut t_values = vec![T::zero(); self.nnz()];
        let mut col_positions = t_row_ptr[..self.ncols].to_vec();
        
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            for k in start..end {
                let j = self.col_idx[k];
                let pos = col_positions[j];
                t_col_idx[pos] = i;
                t_values[pos] = self.values[k];
                col_positions[j] += 1;
            }
        }
        
        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptr: t_row_ptr,
            col_idx: t_col_idx,
            values: t_values,
        }
    }
}

/// Coordinate (COO) format matrix for easier construction.
#[derive(Debug, Clone)]
pub struct CooMatrix<T: Scalar> {
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
    /// Triplets (row, col, value)
    triplets: Vec<(usize, usize, T)>,
}

impl<T: Scalar> CooMatrix<T> {
    /// Creates a new empty COO matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            triplets: Vec::new(),
        }
    }
    
    /// Adds an entry to the matrix.
    pub fn push(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if row >= self.nrows {
            return Err(Error::DimensionMismatch {
                expected: format!("row < {}", self.nrows),
                actual: format!("row = {}", row),
            });
        }
        
        if col >= self.ncols {
            return Err(Error::DimensionMismatch {
                expected: format!("col < {}", self.ncols),
                actual: format!("col = {}", col),
            });
        }
        
        if value != T::zero() {
            self.triplets.push((row, col, value));
        }
        
        Ok(())
    }
    
    /// Converts to CSR format.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        // Sort triplets by (row, col)
        let mut sorted_triplets = self.triplets.clone();
        sorted_triplets.sort_by_key(|&(r, c, _)| (r, c));
        
        // Remove duplicates by summing values
        let mut unique_triplets = Vec::new();
        for (r, c, v) in sorted_triplets {
            if let Some(last) = unique_triplets.last_mut() {
                let (last_r, last_c, ref mut last_v) = last;
                if *last_r == r && *last_c == c {
                    *last_v = *last_v + v;
                    continue;
                }
            }
            unique_triplets.push((r, c, v));
        }
        
        // Build CSR format
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        
        let mut current_row = 0;
        for (r, c, v) in unique_triplets {
            while current_row < r {
                row_ptr.push(col_idx.len());
                current_row += 1;
            }
            col_idx.push(c);
            values.push(v);
        }
        
        while current_row < self.nrows {
            row_ptr.push(col_idx.len());
            current_row += 1;
        }
        
        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr,
            col_idx,
            values,
        }
    }
}

/// Sparse matrix utilities.
pub struct SparseUtils;

impl SparseUtils {
    /// Check if a matrix should be treated as sparse.
    pub fn should_use_sparse<T: Scalar>(matrix: &DMatrix<T>, threshold: f64) -> bool {
        let total_elements = matrix.nrows() * matrix.ncols();
        if total_elements == 0 {
            return false;
        }
        
        let mut nnz = 0;
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if matrix[(i, j)] != T::zero() {
                    nnz += 1;
                }
            }
        }
        
        let sparsity = 1.0 - (nnz as f64 / total_elements as f64);
        sparsity > threshold
    }
    
    /// Create a sparse identity matrix.
    pub fn sparse_identity<T: Scalar>(n: usize) -> CsrMatrix<T> {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        
        for i in 0..n {
            row_ptr.push(i);
            col_idx.push(i);
            values.push(T::one());
        }
        row_ptr.push(n);
        
        CsrMatrix {
            nrows: n,
            ncols: n,
            row_ptr,
            col_idx,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DVector, DMatrix};
    
    #[test]
    fn test_csr_from_dense() {
        let dense = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 2.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 5.0,
        ]);
        
        let csr = CsrMatrix::from_dense(&dense, 1e-10);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 5);
        assert!((csr.sparsity() - 4.0/9.0).abs() < 1e-10);
        
        let reconstructed = csr.to_dense();
        assert_eq!(dense, reconstructed);
    }
    
    #[test]
    fn test_spmv() {
        let dense = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 2.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 5.0,
        ]);
        let csr = CsrMatrix::from_dense(&dense, 1e-10);
        
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = DVector::zeros(3);
        
        csr.spmv(&x, &mut y).unwrap();
        
        let expected = dense * &x;
        assert_eq!(y, expected);
    }
    
    #[test]
    fn test_coo_to_csr() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0).unwrap();
        coo.push(0, 2, 2.0).unwrap();
        coo.push(1, 1, 3.0).unwrap();
        coo.push(2, 0, 4.0).unwrap();
        coo.push(2, 2, 5.0).unwrap();
        
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 5);
        
        let dense = csr.to_dense();
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(0, 2)], 2.0);
        assert_eq!(dense[(1, 1)], 3.0);
        assert_eq!(dense[(2, 0)], 4.0);
        assert_eq!(dense[(2, 2)], 5.0);
    }
    
    #[test]
    fn test_transpose() {
        let dense = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            0.0, 3.0,
            4.0, 0.0,
        ]);
        let csr = CsrMatrix::from_dense(&dense, 1e-10);
        
        let transposed = csr.transpose();
        assert_eq!(transposed.nrows(), 2);
        assert_eq!(transposed.ncols(), 3);
        
        let transposed_dense = transposed.to_dense();
        assert_eq!(transposed_dense, dense.transpose());
    }
    
    #[test]
    fn test_sparse_identity() {
        let id = SparseUtils::sparse_identity::<f64>(4);
        assert_eq!(id.nrows(), 4);
        assert_eq!(id.ncols(), 4);
        assert_eq!(id.nnz(), 4);
        
        let dense = id.to_dense();
        assert_eq!(dense, DMatrix::identity(4, 4));
    }
}