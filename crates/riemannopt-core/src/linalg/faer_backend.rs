//! Faer backend implementation.
//!
//! Implements `VectorOps`, `MatrixOps`, and `DecompositionOps` for
//! `faer::Col<T>` and `faer::Mat<T>`.
//!
//! Enable with feature flag `faer-backend`.

use faer::linalg::solvers::SolveCore;
use faer::Conj;
use faer::{Col, Mat, Side};
use num_traits::Float;

use super::traits::{DecompositionOps, LinAlgBackend, MatrixOps, VectorOps};
use super::types::{CholeskyResult, EigenResult, QrResult, SvdResult};

// ═══════════════════════════════════════════════════════════════════════════
//  Backend marker type
// ═══════════════════════════════════════════════════════════════════════════

/// The faer linear algebra backend.
///
/// Provides near-BLAS performance with pure Rust (no C dependencies).
#[derive(Debug, Clone, Copy)]
pub struct FaerBackend;

impl LinAlgBackend<f32> for FaerBackend {
	type Vector = Col<f32>;
	type Matrix = Mat<f32>;
}

impl LinAlgBackend<f64> for FaerBackend {
	type Vector = Col<f64>;
	type Matrix = Mat<f64>;
}

// ═══════════════════════════════════════════════════════════════════════════
//  VectorOps for faer::Col<T>
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_vector_ops {
	($t:ty) => {
		impl VectorOps<$t> for Col<$t> {
			fn zeros(len: usize) -> Self {
				Col::zeros(len)
			}

			fn from_fn(len: usize, mut f: impl FnMut(usize) -> $t) -> Self {
				Col::from_fn(len, |i| f(i))
			}

			fn from_slice(data: &[$t]) -> Self {
				let mut v = Col::zeros(data.len());
				for (i, &val) in data.iter().enumerate() {
					v[i] = val;
				}
				v
			}

			fn len(&self) -> usize {
				self.nrows()
			}

			fn dot(&self, other: &Self) -> $t {
				let mut sum: $t = 0.0;
				for i in 0..self.nrows() {
					sum += self[i] * other[i];
				}
				sum
			}

			fn get(&self, i: usize) -> $t {
				self[i]
			}

			fn get_mut(&mut self, i: usize) -> &mut $t {
				&mut self[i]
			}

			fn as_slice(&self) -> &[$t] {
				let ptr = self.as_ptr();
				unsafe { std::slice::from_raw_parts(ptr, self.nrows()) }
			}

			fn as_mut_slice(&mut self) -> &mut [$t] {
				let ptr = self.as_ptr_mut();
				let len = self.nrows();
				unsafe { std::slice::from_raw_parts_mut(ptr, len) }
			}

			fn copy_from(&mut self, other: &Self) {
				for i in 0..self.nrows() {
					self[i] = other[i];
				}
			}

			fn fill(&mut self, value: $t) {
				for i in 0..self.nrows() {
					self[i] = value;
				}
			}

			fn axpy(&mut self, alpha: $t, x: &Self, beta: $t) {
				for i in 0..self.nrows() {
					self[i] = beta * self[i] + alpha * x[i];
				}
			}

			fn scale_mut(&mut self, alpha: $t) {
				for i in 0..self.nrows() {
					self[i] *= alpha;
				}
			}

			fn map(&self, mut f: impl FnMut($t) -> $t) -> Self {
				Col::from_fn(self.nrows(), |i| f(self[i]))
			}

			fn iter(&self) -> impl Iterator<Item = $t> + '_ {
				(0..self.nrows()).map(|i| self[i])
			}
		}
	};
}

impl_faer_vector_ops!(f32);
impl_faer_vector_ops!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  MatrixOps for faer::Mat<T>
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_matrix_ops {
	($t:ty) => {
		impl MatrixOps<$t> for Mat<$t> {
			type Col = Col<$t>;

			fn zeros(nrows: usize, ncols: usize) -> Self {
				Mat::zeros(nrows, ncols)
			}

			fn identity(n: usize) -> Self {
				Mat::identity(n, n)
			}

			fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> $t) -> Self {
				Mat::from_fn(nrows, ncols, |i, j| f(i, j))
			}

			fn from_diagonal(diag: &Self::Col) -> Self {
				let n = diag.nrows();
				let mut m = Mat::zeros(n, n);
				for i in 0..n {
					m[(i, i)] = diag[i];
				}
				m
			}

			fn nrows(&self) -> usize {
				(*self).nrows()
			}

			fn ncols(&self) -> usize {
				(*self).ncols()
			}

			fn norm(&self) -> $t {
				let mut sum: $t = 0.0;
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						let v = self[(i, j)];
						sum += v * v;
					}
				}
				<$t as Float>::sqrt(sum)
			}

			fn trace(&self) -> $t {
				let n = self.nrows().min(self.ncols());
				let mut t: $t = 0.0;
				for i in 0..n {
					t += self[(i, i)];
				}
				t
			}

			fn get(&self, i: usize, j: usize) -> $t {
				self[(i, j)]
			}

			fn get_mut(&mut self, i: usize, j: usize) -> &mut $t {
				&mut self[(i, j)]
			}

			fn column(&self, j: usize) -> Self::Col {
				let mut result = Col::zeros(self.nrows());
				for i in 0..self.nrows() {
					result[i] = self[(i, j)];
				}
				result
			}

			fn columns(&self, start: usize, count: usize) -> Self {
				self.subcols(start, count).to_owned()
			}

			fn rows(&self, start: usize, count: usize) -> Self {
				self.subrows(start, count).to_owned()
			}

			fn set_column(&mut self, j: usize, col: &Self::Col) {
				for i in 0..self.nrows() {
					self[(i, j)] = col[i];
				}
			}

			fn set_rows(&mut self, start: usize, src: &Self) {
				for j in 0..self.ncols() {
					for i in 0..src.nrows() {
						self[(start + i, j)] = src[(i, j)];
					}
				}
			}

			fn as_slice(&self) -> &[$t] {
				// faer stores column-major but may have stride > nrows.
				// We can only return a contiguous slice if col_stride == nrows.
				let ptr = self.as_ptr();
				let nrows = (*self).nrows();
				let ncols = (*self).ncols();
				let stride = self.col_stride();
				if ncols <= 1 || stride == nrows as isize {
					let len = if ncols == 0 { 0 } else { stride as usize * (ncols - 1) + nrows };
					unsafe { std::slice::from_raw_parts(ptr, len) }
				} else {
					panic!(
						"faer::Mat as_slice: non-contiguous layout (col_stride={stride} != nrows={nrows})"
					);
				}
			}

			fn as_mut_slice(&mut self) -> &mut [$t] {
				let ptr = self.as_ptr_mut();
				let nrows = (*self).nrows();
				let ncols = (*self).ncols();
				let stride = self.col_stride();
				if ncols <= 1 || stride == nrows as isize {
					let len = if ncols == 0 { 0 } else { stride as usize * (ncols - 1) + nrows };
					unsafe { std::slice::from_raw_parts_mut(ptr, len) }
				} else {
					panic!(
						"faer::Mat as_mut_slice: non-contiguous layout (col_stride={stride} != nrows={nrows})"
					);
				}
			}

			fn column_as_mut_slice(&mut self, j: usize) -> &mut [$t] {
				let nrows = (*self).nrows();
				let stride = self.col_stride();
				let ptr = self.as_ptr_mut();
				unsafe {
					let col_ptr = ptr.offset(j as isize * stride);
					std::slice::from_raw_parts_mut(col_ptr, nrows)
				}
			}

			fn from_column_slice(nrows: usize, ncols: usize, data: &[$t]) -> Self {
				Mat::from_fn(nrows, ncols, |i, j| data[j * nrows + i])
			}

			fn copy_from(&mut self, other: &Self) {
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						self[(i, j)] = other[(i, j)];
					}
				}
			}

			fn fill(&mut self, value: $t) {
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						self[(i, j)] = value;
					}
				}
			}

			fn scale_mut(&mut self, alpha: $t) {
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						self[(i, j)] *= alpha;
					}
				}
			}

			fn transpose(&self) -> Self {
				self.transpose().to_owned()
			}

			fn mat_mul(&self, other: &Self) -> Self {
				self * other
			}

			fn mat_vec(&self, v: &Self::Col) -> Self::Col {
				self * v
			}

			fn add(&self, other: &Self) -> Self {
				let mut result = self.clone();
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						result[(i, j)] += other[(i, j)];
					}
				}
				result
			}

			fn sub(&self, other: &Self) -> Self {
				let mut result = self.clone();
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						result[(i, j)] -= other[(i, j)];
					}
				}
				result
			}

			fn scale_by(&self, alpha: $t) -> Self {
				let mut result = self.clone();
				for j in 0..result.ncols() {
					for i in 0..result.nrows() {
						result[(i, j)] *= alpha;
					}
				}
				result
			}

			fn add_assign(&mut self, other: &Self) {
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						self[(i, j)] += other[(i, j)];
					}
				}
			}

			fn sub_assign(&mut self, other: &Self) {
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						self[(i, j)] -= other[(i, j)];
					}
				}
			}

			fn gemm(&mut self, alpha: $t, a: &Self, b: &Self, beta: $t) {
				let ab = a * b;
				for j in 0..self.ncols() {
					for i in 0..self.nrows() {
						self[(i, j)] = alpha * ab[(i, j)] + beta * self[(i, j)];
					}
				}
			}
		}
	};
}

impl_faer_matrix_ops!(f32);
impl_faer_matrix_ops!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  DecompositionOps for faer::Mat<T>
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_decomposition_ops {
	($t:ty) => {
		impl DecompositionOps<$t> for Mat<$t> {
			fn svd(&self) -> SvdResult<$t, Self> {
				let svd = self.thin_svd().expect("SVD failed");
				let u_mat = svd.U().to_owned();
				let v_mat = svd.V().to_owned();
				let s_diag = svd.S().column_vector();
				let k = s_diag.nrows();

				let mut singular_values = Col::zeros(k);
				for i in 0..k {
					singular_values[i] = s_diag[i];
				}

				let vt = v_mat.transpose().to_owned();

				SvdResult {
					u: Some(u_mat),
					singular_values,
					vt: Some(vt),
				}
			}

			fn qr(&self) -> QrResult<$t, Self> {
				let qr = (*self).qr();
				let q = qr.compute_thin_Q();
				let r = qr.thin_R().to_owned();
				QrResult::new(q, r)
			}

			fn symmetric_eigen(&self) -> EigenResult<$t, Self> {
				let eigen = self
					.self_adjoint_eigen(Side::Lower)
					.expect("Eigendecomposition failed");
				let eigenvectors = eigen.U().to_owned();
				let s_diag = eigen.S().column_vector();
				let n = s_diag.nrows();

				let mut eigenvalues = Col::zeros(n);
				for i in 0..n {
					eigenvalues[i] = s_diag[i];
				}

				EigenResult {
					eigenvalues,
					eigenvectors,
				}
			}

			fn cholesky(&self) -> Option<CholeskyResult<$t, Self>> {
				match self.llt(Side::Lower) {
					Ok(llt) => {
						let l = llt.L().to_owned();
						Some(CholeskyResult::new(l))
					}
					Err(_) => None,
				}
			}

			fn try_inverse(&self) -> Option<Self> {
				let n = self.nrows();
				if n != self.ncols() {
					return None;
				}
				let lu = self.partial_piv_lu();
				let mut result = Mat::identity(n, n);
				lu.solve_in_place_with_conj(Conj::No, result.as_mut());
				Some(result)
			}

			fn cholesky_solve(&self, rhs: &Self) -> Option<Self> {
				match self.llt(Side::Lower) {
					Ok(llt) => {
						let mut result = rhs.clone();
						llt.solve_in_place_with_conj(Conj::No, result.as_mut());
						Some(result)
					}
					Err(_) => None,
				}
			}
		}
	};
}

impl_faer_decomposition_ops!(f32);
impl_faer_decomposition_ops!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_vector_basic() {
		let v = <Col<f64> as VectorOps<f64>>::from_fn(3, |i| (i + 1) as f64);
		assert_eq!(VectorOps::len(&v), 3);
		assert!((VectorOps::get(&v, 0) - 1.0).abs() < 1e-14);
		assert!((VectorOps::norm_squared(&v) - 14.0).abs() < 1e-14);
	}

	#[test]
	fn test_vector_dot() {
		let a = <Col<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let b = <Col<f64> as VectorOps<f64>>::from_slice(&[4.0, 5.0, 6.0]);
		assert!((VectorOps::dot(&a, &b) - 32.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_identity() {
		let m = <Mat<f64> as MatrixOps<f64>>::identity(3);
		assert_eq!(MatrixOps::nrows(&m), 3);
		assert!((MatrixOps::trace(&m) - 3.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_mul() {
		let a = <Mat<f64> as MatrixOps<f64>>::identity(3);
		let v = <Col<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let r = MatrixOps::mat_vec(&a, &v);
		assert!((VectorOps::get(&r, 0) - 1.0).abs() < 1e-14);
		assert!((VectorOps::get(&r, 2) - 3.0).abs() < 1e-14);
	}

	#[test]
	fn test_svd() {
		let m = <Mat<f64> as MatrixOps<f64>>::identity(3);
		let svd = DecompositionOps::svd(&m);
		assert!(svd.u.is_some());
		for i in 0..3 {
			assert!((VectorOps::get(&svd.singular_values, i) - 1.0).abs() < 1e-10);
		}
	}

	#[test]
	fn test_symmetric_eigen() {
		let diag = <Col<f64> as VectorOps<f64>>::from_slice(&[3.0, 1.0, 2.0]);
		let m = <Mat<f64> as MatrixOps<f64>>::from_diagonal(&diag);
		let eigen = DecompositionOps::symmetric_eigen(&m);
		let mut eigs: Vec<f64> = VectorOps::iter(&eigen.eigenvalues).collect();
		eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
		assert!((eigs[0] - 1.0).abs() < 1e-10);
		assert!((eigs[1] - 2.0).abs() < 1e-10);
		assert!((eigs[2] - 3.0).abs() < 1e-10);
	}

	#[test]
	fn test_vector_add_sub() {
		let a = <Col<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let b = <Col<f64> as VectorOps<f64>>::from_slice(&[4.0, 5.0, 6.0]);
		let sum = VectorOps::add(&a, &b);
		assert!((VectorOps::get(&sum, 0) - 5.0).abs() < 1e-14);
		assert!((VectorOps::get(&sum, 2) - 9.0).abs() < 1e-14);
		let diff = VectorOps::sub(&a, &b);
		assert!((VectorOps::get(&diff, 0) - (-3.0)).abs() < 1e-14);
		let neg = VectorOps::neg(&a);
		assert!((VectorOps::get(&neg, 1) - (-2.0)).abs() < 1e-14);
	}

	#[test]
	fn test_vector_component_mul() {
		let a = <Col<f64> as VectorOps<f64>>::from_slice(&[2.0, 3.0, 4.0]);
		let b = <Col<f64> as VectorOps<f64>>::from_slice(&[5.0, 6.0, 7.0]);
		let c = VectorOps::component_mul(&a, &b);
		assert!((VectorOps::get(&c, 0) - 10.0).abs() < 1e-14);
		assert!((VectorOps::get(&c, 1) - 18.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_column_as_mut_slice() {
		let mut m = <Mat<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3) as f64);
		let col = MatrixOps::column_as_mut_slice(&mut m, 1);
		assert!((col[0] - 3.0).abs() < 1e-14);
		col[0] = 99.0;
		assert!((MatrixOps::get(&m, 0, 1) - 99.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_from_column_slice() {
		let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
		let m = <Mat<f64> as MatrixOps<f64>>::from_column_slice(3, 2, &data);
		assert!((MatrixOps::get(&m, 0, 0) - 1.0).abs() < 1e-14);
		assert!((MatrixOps::get(&m, 2, 0) - 3.0).abs() < 1e-14);
		assert!((MatrixOps::get(&m, 0, 1) - 4.0).abs() < 1e-14);
	}

	#[test]
	fn test_cholesky_solve() {
		let a = <Mat<f64> as MatrixOps<f64>>::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
		let b = <Mat<f64> as MatrixOps<f64>>::identity(2);
		let x = DecompositionOps::cholesky_solve(&a, &b).unwrap();
		let ax = MatrixOps::mat_mul(&a, &x);
		for i in 0..2 {
			for j in 0..2 {
				let expected = if i == j { 1.0 } else { 0.0 };
				assert!((MatrixOps::get(&ax, i, j) - expected).abs() < 1e-10);
			}
		}
	}

	#[test]
	fn test_backend_type() {
		type V = <FaerBackend as LinAlgBackend<f64>>::Vector;
		type M = <FaerBackend as LinAlgBackend<f64>>::Matrix;
		let v = V::zeros(5);
		let m = <M as MatrixOps<f64>>::identity(3);
		assert_eq!(VectorOps::len(&v), 5);
		assert_eq!(MatrixOps::nrows(&m), 3);
	}
}
