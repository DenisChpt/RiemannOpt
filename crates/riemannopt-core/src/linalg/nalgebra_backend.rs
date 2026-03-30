//! Nalgebra backend implementation.
//!
//! Implements `VectorView`, `VectorOps`, `MatrixView`, `MatrixOps`, and
//! `DecompositionOps` for `nalgebra::DVector<T>` and `nalgebra::DMatrix<T>`.
//!
//! View types use nalgebra's native `DVectorView` and `DMatrixView`.

use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView, RealField, Scalar as NalgebraScalar};
use num_traits::Float;

use super::traits::{
	DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView,
};
use super::types::{CholeskyResult, EigenResult, QrResult, SvdResult};

// ═══════════════════════════════════════════════════════════════════════════
//  Backend marker type
// ═══════════════════════════════════════════════════════════════════════════

/// The nalgebra linear algebra backend (default).
#[derive(Debug, Clone, Copy)]
pub struct NalgebraBackend;

impl LinAlgBackend<f32> for NalgebraBackend {
	type Vector = DVector<f32>;
	type Matrix = DMatrix<f32>;
}

impl LinAlgBackend<f64> for NalgebraBackend {
	type Vector = DVector<f64>;
	type Matrix = DMatrix<f64>;
}

// ═══════════════════════════════════════════════════════════════════════════
//  VectorView for DVectorView<'_, T>  (borrowed view)
// ═══════════════════════════════════════════════════════════════════════════

impl<'v, T> VectorView<T> for DVectorView<'v, T>
where
	T: RealScalar + NalgebraScalar + RealField,
{
	#[inline]
	fn len(&self) -> usize {
		self.nrows()
	}

	#[inline]
	fn get(&self, i: usize) -> T {
		self[i]
	}

	#[inline]
	fn dot(&self, other: &Self) -> T {
		nalgebra::Matrix::dot(self, other)
	}

	#[inline]
	fn norm(&self) -> T {
		nalgebra::Matrix::norm(self)
	}

	#[inline]
	fn norm_squared(&self) -> T {
		nalgebra::Matrix::norm_squared(self)
	}

	fn iter(&self) -> impl Iterator<Item = T> + '_ {
		nalgebra::Matrix::iter(self).copied()
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  VectorView for DVector<T>  (owned)
// ═══════════════════════════════════════════════════════════════════════════

impl<T> VectorView<T> for DVector<T>
where
	T: RealScalar + NalgebraScalar + RealField,
{
	#[inline]
	fn len(&self) -> usize {
		self.nrows()
	}

	#[inline]
	fn get(&self, i: usize) -> T {
		self[i]
	}

	#[inline]
	fn dot(&self, other: &Self) -> T {
		nalgebra::DVector::dot(self, other)
	}

	#[inline]
	fn norm(&self) -> T {
		nalgebra::Matrix::norm(self)
	}

	#[inline]
	fn norm_squared(&self) -> T {
		nalgebra::Matrix::norm_squared(self)
	}

	fn iter(&self) -> impl Iterator<Item = T> + '_ {
		nalgebra::Matrix::iter(self).copied()
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  VectorOps for DVector<T>
// ═══════════════════════════════════════════════════════════════════════════

impl<T> VectorOps<T> for DVector<T>
where
	T: RealScalar + NalgebraScalar + RealField,
{
	#[inline]
	fn zeros(len: usize) -> Self {
		DVector::zeros(len)
	}

	#[inline]
	fn from_fn(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
		DVector::from_fn(len, |i, _| f(i))
	}

	#[inline]
	fn from_slice(data: &[T]) -> Self {
		DVector::from_column_slice(data)
	}

	#[inline]
	fn get_mut(&mut self, i: usize) -> &mut T {
		&mut self[i]
	}

	#[inline]
	fn as_slice(&self) -> &[T] {
		nalgebra::Matrix::as_slice(self)
	}

	#[inline]
	fn as_mut_slice(&mut self) -> &mut [T] {
		nalgebra::Matrix::as_mut_slice(self)
	}

	#[inline]
	fn copy_from(&mut self, other: &Self) {
		nalgebra::Matrix::copy_from(self, other);
	}

	#[inline]
	fn fill(&mut self, value: T) {
		nalgebra::Matrix::fill(self, value);
	}

	#[inline]
	fn axpy(&mut self, alpha: T, x: &Self, beta: T) {
		*self *= beta;
		nalgebra::Matrix::axpy(self, alpha, x, T::one());
	}

	#[inline]
	fn scale_mut(&mut self, alpha: T) {
		*self *= alpha;
	}

	fn map(&self, mut f: impl FnMut(T) -> T) -> Self {
		nalgebra::Matrix::map(self, |x| f(x))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  MatrixView for DMatrixView<'_, T>  (borrowed view)
// ═══════════════════════════════════════════════════════════════════════════

impl<'v, T> MatrixView<T> for DMatrixView<'v, T>
where
	T: RealScalar + NalgebraScalar + RealField,
{
	type ColView<'a>
		= DVectorView<'a, T>
	where
		Self: 'a;

	#[inline]
	fn nrows(&self) -> usize {
		nalgebra::Matrix::nrows(self)
	}

	#[inline]
	fn ncols(&self) -> usize {
		nalgebra::Matrix::ncols(self)
	}

	#[inline]
	fn get(&self, i: usize, j: usize) -> T {
		self[(i, j)]
	}

	#[inline]
	fn column(&self, j: usize) -> Self::ColView<'_> {
		nalgebra::Matrix::column(self, j)
	}

	fn norm(&self) -> T {
		let mut sum = T::zero();
		for v in nalgebra::Matrix::iter(self) {
			sum = sum + *v * *v;
		}
		Float::sqrt(sum)
	}

	fn trace(&self) -> T {
		let n = nalgebra::Matrix::nrows(self).min(nalgebra::Matrix::ncols(self));
		let mut t = T::zero();
		for i in 0..n {
			t = t + self[(i, i)];
		}
		t
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  MatrixView for DMatrix<T>  (owned)
// ═══════════════════════════════════════════════════════════════════════════

impl<T> MatrixView<T> for DMatrix<T>
where
	T: RealScalar + NalgebraScalar + RealField,
{
	type ColView<'a>
		= DVectorView<'a, T>
	where
		Self: 'a;

	#[inline]
	fn nrows(&self) -> usize {
		nalgebra::Matrix::nrows(self)
	}

	#[inline]
	fn ncols(&self) -> usize {
		nalgebra::Matrix::ncols(self)
	}

	#[inline]
	fn get(&self, i: usize, j: usize) -> T {
		self[(i, j)]
	}

	#[inline]
	fn column(&self, j: usize) -> Self::ColView<'_> {
		nalgebra::Matrix::column(self, j)
	}

	fn norm(&self) -> T {
		let mut sum = T::zero();
		for v in nalgebra::Matrix::iter(self) {
			sum = sum + *v * *v;
		}
		Float::sqrt(sum)
	}

	fn trace(&self) -> T {
		let n = nalgebra::Matrix::nrows(self).min(nalgebra::Matrix::ncols(self));
		let mut t = T::zero();
		for i in 0..n {
			t = t + self[(i, i)];
		}
		t
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  MatrixOps for DMatrix<T>
// ═══════════════════════════════════════════════════════════════════════════

impl<T> MatrixOps<T> for DMatrix<T>
where
	T: RealScalar + NalgebraScalar + RealField,
{
	type Col = DVector<T>;
	type View<'a> = DMatrixView<'a, T>;

	#[inline]
	fn zeros(nrows: usize, ncols: usize) -> Self {
		DMatrix::zeros(nrows, ncols)
	}

	#[inline]
	fn identity(n: usize) -> Self {
		DMatrix::identity(n, n)
	}

	#[inline]
	fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
		DMatrix::from_fn(nrows, ncols, |i, j| f(i, j))
	}

	#[inline]
	fn from_diagonal(diag: &Self::Col) -> Self {
		DMatrix::from_diagonal(diag)
	}

	fn from_column_slice(nrows: usize, ncols: usize, data: &[T]) -> Self {
		DMatrix::from_column_slice(nrows, ncols, data)
	}

	// ── View accessors ───────────────────────────────────────────────

	#[inline]
	fn view_from_column_slice<'a>(nrows: usize, ncols: usize, data: &'a [T]) -> Self::View<'a> {
		DMatrixView::from_slice(data, nrows, ncols)
	}

	#[inline]
	fn as_view(&self) -> Self::View<'_> {
		self.as_view()
	}

	#[inline]
	fn columns(&self, start: usize, count: usize) -> Self::View<'_> {
		nalgebra::Matrix::columns(self, start, count)
	}

	#[inline]
	fn rows(&self, start: usize, count: usize) -> Self::View<'_> {
		nalgebra::Matrix::rows(self, start, count)
	}

	// ── Owned conversions ────────────────────────────────────────────

	#[inline]
	fn column_to_owned(&self, j: usize) -> Self::Col {
		nalgebra::Matrix::column(self, j).clone_owned()
	}

	fn transpose_to_owned(&self) -> Self {
		nalgebra::Matrix::transpose(self)
	}

	fn columns_to_owned(&self, start: usize, count: usize) -> Self {
		nalgebra::Matrix::columns(self, start, count).clone_owned()
	}

	fn rows_to_owned(&self, start: usize, count: usize) -> Self {
		nalgebra::Matrix::rows(self, start, count).clone_owned()
	}

	// ── Element access ───────────────────────────────────────────────

	#[inline]
	fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
		&mut self[(i, j)]
	}

	fn set_column(&mut self, j: usize, col: &Self::Col) {
		self.column_mut(j).copy_from(col);
	}

	fn set_rows(&mut self, start: usize, src: &Self) {
		let count = src.nrows();
		let mut view = nalgebra::Matrix::rows_mut(self, start, count);
		view.copy_from(src);
	}

	#[inline]
	fn as_slice(&self) -> &[T] {
		nalgebra::Matrix::as_slice(self)
	}

	#[inline]
	fn as_mut_slice(&mut self) -> &mut [T] {
		nalgebra::Matrix::as_mut_slice(self)
	}

	// ── In-place mutations ───────────────────────────────────────────

	#[inline]
	fn copy_from(&mut self, other: &Self) {
		nalgebra::Matrix::copy_from(self, other);
	}

	#[inline]
	fn fill(&mut self, value: T) {
		nalgebra::Matrix::fill(self, value);
	}

	#[inline]
	fn scale_mut(&mut self, alpha: T) {
		*self *= alpha;
	}

	// ── Allocating arithmetic ────────────────────────────────────────

	fn mat_mul(&self, other: &Self) -> Self {
		self * other
	}

	fn mat_vec(&self, v: &Self::Col) -> Self::Col {
		self * v
	}

	fn mat_vec_axpy(&self, alpha: T, x: &Self::Col, beta: T, y: &mut Self::Col) {
		y.gemv(alpha, self, x, beta);
	}

	fn add(&self, other: &Self) -> Self {
		self + other
	}

	fn sub(&self, other: &Self) -> Self {
		self - other
	}

	fn scale_by(&self, alpha: T) -> Self {
		self * alpha
	}

	fn add_assign(&mut self, other: &Self) {
		*self += other;
	}

	fn sub_assign(&mut self, other: &Self) {
		*self -= other;
	}

	// ── GEMM (view operands) ─────────────────────────────────────────

	fn gemm(&mut self, alpha: T, a: Self::View<'_>, b: Self::View<'_>, beta: T) {
		nalgebra::Matrix::gemm(self, alpha, &a, &b, beta);
	}

	fn gemm_at(&mut self, alpha: T, a: Self::View<'_>, b: Self::View<'_>, beta: T) {
		nalgebra::Matrix::gemm_tr(self, alpha, &a, &b, beta);
	}

	fn gemm_bt(&mut self, alpha: T, a: Self::View<'_>, b: Self::View<'_>, beta: T) {
		// nalgebra doesn't have a native gemm with B transposed,
		// so we transpose B and use regular gemm.
		let bt = b.transpose();
		nalgebra::Matrix::gemm(self, alpha, &a, &bt, beta);
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  DecompositionOps for DMatrix<T>
// ═══════════════════════════════════════════════════════════════════════════

impl<T> DecompositionOps<T> for DMatrix<T>
where
	T: RealScalar + NalgebraScalar + RealField,
{
	fn svd(&self) -> SvdResult<T, Self> {
		let svd = self.clone().svd(true, true);
		SvdResult {
			u: svd.u,
			singular_values: svd.singular_values,
			vt: svd.v_t,
		}
	}

	fn qr(&self) -> QrResult<T, Self> {
		let qr = self.clone().qr();
		QrResult::new(qr.q(), qr.r())
	}

	fn symmetric_eigen(&self) -> EigenResult<T, Self> {
		let eigen = nalgebra::SymmetricEigen::new(self.clone());
		EigenResult {
			eigenvalues: eigen.eigenvalues,
			eigenvectors: eigen.eigenvectors,
		}
	}

	fn cholesky(&self) -> Option<CholeskyResult<T, Self>> {
		nalgebra::linalg::Cholesky::new(self.clone()).map(|c| CholeskyResult::new(c.l()))
	}

	fn inverse(&self, result: &mut Self) -> bool {
		if let Some(inv) = nalgebra::DMatrix::try_inverse(self.clone()) {
			result.copy_from(&inv);
			true
		} else {
			false
		}
	}

	fn cholesky_solve(&self, rhs: &Self, result: &mut Self) -> bool {
		if let Some(chol) = nalgebra::linalg::Cholesky::new(self.clone()) {
			let sol = chol.solve(rhs);
			result.copy_from(&sol);
			true
		} else {
			false
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
	use super::*;
	use approx::assert_relative_eq;

	#[test]
	fn test_vector_basic_ops() {
		let v = <DVector<f64> as VectorOps<f64>>::zeros(5);
		assert_eq!(VectorView::len(&v), 5);
		assert_relative_eq!(v.dot(&v), 0.0);

		let v = <DVector<f64> as VectorOps<f64>>::from_fn(3, |i| (i + 1) as f64);
		assert_relative_eq!(VectorView::get(&v, 0), 1.0);
		assert_relative_eq!(VectorView::get(&v, 1), 2.0);
		assert_relative_eq!(VectorView::get(&v, 2), 3.0);
		assert_relative_eq!(VectorView::norm_squared(&v), 14.0);
	}

	#[test]
	fn test_vector_axpy() {
		let x = <DVector<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let mut y = <DVector<f64> as VectorOps<f64>>::from_slice(&[10.0, 20.0, 30.0]);
		VectorOps::axpy(&mut y, 2.0, &x, 1.0);
		assert_relative_eq!(VectorView::get(&y, 0), 12.0);
		assert_relative_eq!(VectorView::get(&y, 1), 24.0);
		assert_relative_eq!(VectorView::get(&y, 2), 36.0);
	}

	#[test]
	fn test_matrix_basic_ops() {
		let m = <DMatrix<f64> as MatrixOps<f64>>::identity(3);
		assert_eq!(MatrixView::nrows(&m), 3);
		assert_eq!(MatrixView::ncols(&m), 3);
		assert_relative_eq!(MatrixView::trace(&m), 3.0);
		assert_relative_eq!(MatrixView::get(&m, 0, 0), 1.0);
		assert_relative_eq!(MatrixView::get(&m, 0, 1), 0.0);
	}

	#[test]
	fn test_matrix_multiply() {
		let a = <DMatrix<f64> as MatrixOps<f64>>::identity(3);
		let v = <DVector<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let result = MatrixOps::mat_vec(&a, &v);
		assert_relative_eq!(VectorView::get(&result, 0), 1.0);
		assert_relative_eq!(VectorView::get(&result, 1), 2.0);
		assert_relative_eq!(VectorView::get(&result, 2), 3.0);
	}

	#[test]
	fn test_column_view() {
		let m = <DMatrix<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3 + 1) as f64);
		let col0 = m.column(0);
		assert_relative_eq!(VectorView::get(&col0, 0), 1.0);
		assert_relative_eq!(VectorView::get(&col0, 2), 3.0);
	}

	#[test]
	fn test_gemm_with_views() {
		let a = <DMatrix<f64> as MatrixOps<f64>>::identity(3);
		let b =
			<DMatrix<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3 + 1) as f64);
		let mut c = <DMatrix<f64> as MatrixOps<f64>>::zeros(3, 2);
		MatrixOps::gemm(&mut c, 1.0, a.as_view(), b.as_view(), 0.0);
		assert_relative_eq!(MatrixView::get(&c, 0, 0), 1.0);
		assert_relative_eq!(MatrixView::get(&c, 2, 1), 6.0);
	}

	#[test]
	fn test_svd() {
		let m = <DMatrix<f64> as MatrixOps<f64>>::identity(3);
		let svd = DecompositionOps::svd(&m);
		assert!(svd.u.is_some());
		assert!(svd.vt.is_some());
		for i in 0..3 {
			assert_relative_eq!(
				VectorView::get(&svd.singular_values, i),
				1.0,
				epsilon = 1e-10
			);
		}
	}

	#[test]
	fn test_qr() {
		let m =
			<DMatrix<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| if i == j { 1.0 } else { 0.0 });
		let qr = DecompositionOps::qr(&m);
		let reconstructed = MatrixOps::mat_mul(qr.q(), qr.r());
		for i in 0..3 {
			for j in 0..2 {
				assert_relative_eq!(
					MatrixView::get(&reconstructed, i, j),
					MatrixView::get(&m, i, j),
					epsilon = 1e-10
				);
			}
		}
	}

	#[test]
	fn test_symmetric_eigen() {
		let diag = <DVector<f64> as VectorOps<f64>>::from_slice(&[3.0, 1.0, 2.0]);
		let m = <DMatrix<f64> as MatrixOps<f64>>::from_diagonal(&diag);
		let eigen = DecompositionOps::symmetric_eigen(&m);
		let mut eigs: Vec<f64> = VectorView::iter(&eigen.eigenvalues).collect();
		eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
		assert_relative_eq!(eigs[0], 1.0, epsilon = 1e-10);
		assert_relative_eq!(eigs[1], 2.0, epsilon = 1e-10);
		assert_relative_eq!(eigs[2], 3.0, epsilon = 1e-10);
	}

	#[test]
	fn test_cholesky() {
		let m =
			<DMatrix<f64> as MatrixOps<f64>>::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
		let result = DecompositionOps::cholesky(&m);
		assert!(result.is_some());
		let l = result.unwrap().l().clone();
		let reconstructed = MatrixOps::mat_mul(&l, &MatrixOps::transpose_to_owned(&l));
		for i in 0..2 {
			for j in 0..2 {
				assert_relative_eq!(
					MatrixView::get(&reconstructed, i, j),
					MatrixView::get(&m, i, j),
					epsilon = 1e-10
				);
			}
		}
	}

	#[test]
	fn test_vector_add_sub() {
		let a = <DVector<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let b = <DVector<f64> as VectorOps<f64>>::from_slice(&[4.0, 5.0, 6.0]);
		let sum = VectorOps::add(&a, &b);
		assert_relative_eq!(VectorView::get(&sum, 0), 5.0);
		assert_relative_eq!(VectorView::get(&sum, 2), 9.0);
		let diff = VectorOps::sub(&a, &b);
		assert_relative_eq!(VectorView::get(&diff, 0), -3.0);
		let neg = VectorOps::neg(&a);
		assert_relative_eq!(VectorView::get(&neg, 1), -2.0);
	}

	#[test]
	fn test_vector_component_mul() {
		let a = <DVector<f64> as VectorOps<f64>>::from_slice(&[2.0, 3.0, 4.0]);
		let b = <DVector<f64> as VectorOps<f64>>::from_slice(&[5.0, 6.0, 7.0]);
		let c = VectorOps::component_mul(&a, &b);
		assert_relative_eq!(VectorView::get(&c, 0), 10.0);
		assert_relative_eq!(VectorView::get(&c, 1), 18.0);
		assert_relative_eq!(VectorView::get(&c, 2), 28.0);
	}

	#[test]
	fn test_vector_add_sub_assign() {
		let mut a = <DVector<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let b = <DVector<f64> as VectorOps<f64>>::from_slice(&[4.0, 5.0, 6.0]);
		VectorOps::add_assign(&mut a, &b);
		assert_relative_eq!(VectorView::get(&a, 0), 5.0);
		VectorOps::sub_assign(&mut a, &b);
		assert_relative_eq!(VectorView::get(&a, 0), 1.0);
	}

	#[test]
	fn test_matrix_column_as_mut_slice() {
		let mut m = <DMatrix<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3) as f64);
		let col = MatrixOps::column_as_mut_slice(&mut m, 1);
		assert_relative_eq!(col[0], 3.0);
		col[0] = 99.0;
		assert_relative_eq!(MatrixView::get(&m, 0, 1), 99.0);
	}

	#[test]
	fn test_matrix_from_column_slice() {
		let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
		let m = <DMatrix<f64> as MatrixOps<f64>>::from_column_slice(3, 2, &data);
		assert_relative_eq!(MatrixView::get(&m, 0, 0), 1.0);
		assert_relative_eq!(MatrixView::get(&m, 2, 0), 3.0);
		assert_relative_eq!(MatrixView::get(&m, 0, 1), 4.0);
	}

	#[test]
	fn test_matrix_column_dot() {
		let m = <DMatrix<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j + 1) as f64);
		let d = MatrixView::column_dot(&m, 0, &m, 1);
		assert_relative_eq!(d, 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0);
	}

	#[test]
	fn test_cholesky_solve() {
		let a =
			<DMatrix<f64> as MatrixOps<f64>>::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
		let b = <DMatrix<f64> as MatrixOps<f64>>::identity(2);
		let mut x = <DMatrix<f64> as MatrixOps<f64>>::zeros(2, 2);
		assert!(DecompositionOps::cholesky_solve(&a, &b, &mut x));
		let ax = MatrixOps::mat_mul(&a, &x);
		for i in 0..2 {
			for j in 0..2 {
				let expected = if i == j { 1.0 } else { 0.0 };
				assert_relative_eq!(MatrixView::get(&ax, i, j), expected, epsilon = 1e-10);
			}
		}
	}

	#[test]
	fn test_backend_type() {
		type V = <NalgebraBackend as LinAlgBackend<f64>>::Vector;
		type M = <NalgebraBackend as LinAlgBackend<f64>>::Matrix;
		let v = V::zeros(5);
		let m = <M as MatrixOps<f64>>::identity(3);
		assert_eq!(VectorView::len(&v), 5);
		assert_eq!(MatrixView::nrows(&m), 3);
	}
}
