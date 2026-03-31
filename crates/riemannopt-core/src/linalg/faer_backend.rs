//! Faer backend implementation.
//!
//! Implements `VectorView`, `VectorOps`, `MatrixView`, `MatrixOps`, and
//! `DecompositionOps` for faer types.
//!
//! - Owned types: `faer::Col<T>`, `faer::Mat<T>`
//! - View types:  `faer::ColRef<'_, T>`, `faer::MatRef<'_, T>`
//!
//! Enable with feature flag `faer-backend`.
//!
//! # Performance
//!
//! - All trait methods are `#[inline]` for monomorphisation and LTO.
//! - Dot product uses `faer::linalg::matmul::dot::inner_prod` (SIMD, 4x unrolled).
//! - GEMM variants use faer's native transpose views (stride swap, zero-alloc).
//! - `column()`, `columns()`, `rows()` return borrowed views — no heap allocation.

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Col, ColRef, Conj, Mat, MatRef, Side};

use super::traits::{
	DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView,
};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::factor;
use faer::reborrow::*;
use faer::Par;
// ═══════════════════════════════════════════════════════════════════════════
//  Backend marker type
// ═══════════════════════════════════════════════════════════════════════════

/// The faer linear algebra backend.
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
//  VectorView for faer::ColRef<'_, T>  (borrowed view)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_vector_view_for_colref {
	($t:ty) => {
		impl VectorView<$t> for ColRef<'_, $t> {
			#[inline]
			fn len(&self) -> usize {
				self.nrows()
			}

			#[inline]
			fn is_empty(&self) -> bool {
				self.nrows() == 0
			}

			#[inline]
			fn get(&self, i: usize) -> $t {
				self[i]
			}

			#[inline]
			fn dot(&self, other: &Self) -> $t {
				faer::linalg::matmul::dot::inner_prod(self.transpose(), Conj::No, *other, Conj::No)
			}

			#[inline]
			fn norm(&self) -> $t {
				self.norm_l2()
			}

			#[inline]
			fn norm_squared(&self) -> $t {
				self.squared_norm_l2()
			}

			#[inline]
			fn iter(&self) -> impl Iterator<Item = $t> + '_ {
				(*self).iter().copied()
			}
		}
	};
}

impl_faer_vector_view_for_colref!(f32);
impl_faer_vector_view_for_colref!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  VectorView for faer::Col<T>  (owned)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_vector_view_for_col {
	($t:ty) => {
		impl VectorView<$t> for Col<$t> {
			#[inline]
			fn len(&self) -> usize {
				self.nrows()
			}

			#[inline]
			fn is_empty(&self) -> bool {
				self.nrows() == 0
			}

			#[inline]
			fn get(&self, i: usize) -> $t {
				self[i]
			}

			#[inline]
			fn dot(&self, other: &Self) -> $t {
				faer::linalg::matmul::dot::inner_prod(
					self.as_ref().transpose(),
					Conj::No,
					other.as_ref(),
					Conj::No,
				)
			}

			#[inline]
			fn norm(&self) -> $t {
				self.as_ref().norm_l2()
			}

			#[inline]
			fn norm_squared(&self) -> $t {
				self.as_ref().squared_norm_l2()
			}

			#[inline]
			fn iter(&self) -> impl Iterator<Item = $t> + '_ {
				self.as_ref().iter().copied()
			}
		}
	};
}

impl_faer_vector_view_for_col!(f32);
impl_faer_vector_view_for_col!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  VectorOps for faer::Col<T>  (owned)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_vector_ops {
	($t:ty) => {
		impl VectorOps<$t> for Col<$t> {
			#[inline]
			fn zeros(len: usize) -> Self {
				Col::zeros(len)
			}

			#[inline]
			fn from_fn(len: usize, f: impl FnMut(usize) -> $t) -> Self {
				Col::from_fn(len, f)
			}

			#[inline]
			fn from_slice(data: &[$t]) -> Self {
				ColRef::from_slice(data).to_owned()
			}

			#[inline]
			fn get_mut(&mut self, i: usize) -> &mut $t {
				&mut self[i]
			}

			#[inline]
			fn as_slice(&self) -> &[$t] {
				self.as_ref().try_as_col_major().unwrap().as_slice()
			}

			#[inline]
			fn as_mut_slice(&mut self) -> &mut [$t] {
				self.as_mut().try_as_col_major_mut().unwrap().as_slice_mut()
			}

			#[inline]
			fn copy_from(&mut self, other: &Self) {
				self.as_mut().copy_from(other.as_ref());
			}

			#[inline]
			fn fill(&mut self, value: $t) {
				self.as_mut().fill(value);
			}

			#[inline]
			fn axpy(&mut self, alpha: $t, x: &Self, beta: $t) {
				if beta == 0.0 {
					faer::zip!(self.as_mut(), x.as_ref()).for_each(|faer::unzip!(s, x)| {
						*s = alpha * *x;
					});
				} else if beta == 1.0 {
					faer::zip!(self.as_mut(), x.as_ref()).for_each(|faer::unzip!(s, x)| {
						*s += alpha * *x;
					});
				} else {
					faer::zip!(self.as_mut(), x.as_ref()).for_each(|faer::unzip!(s, x)| {
						*s = beta * *s + alpha * *x;
					});
				}
			}

			#[inline]
			fn scale_mut(&mut self, alpha: $t) {
				*self *= faer::Scale(alpha);
			}

			#[inline]
			fn div_scalar_mut(&mut self, alpha: $t) {
				*self *= faer::Scale(1.0 / alpha);
			}

			#[inline]
			fn add(&self, other: &Self) -> Self {
				self + other
			}

			#[inline]
			fn sub(&self, other: &Self) -> Self {
				self - other
			}

			#[inline]
			fn neg(&self) -> Self {
				-self
			}

			#[inline]
			fn component_mul(&self, other: &Self) -> Self {
				Col::from_fn(self.nrows(), |i| self[i] * other[i])
			}

			#[inline]
			fn add_assign(&mut self, other: &Self) {
				*self += other;
			}

			#[inline]
			fn sub_assign(&mut self, other: &Self) {
				*self -= other;
			}

			#[inline]
			fn component_mul_assign(&mut self, other: &Self) {
				faer::zip!(self.as_mut(), other.as_ref()).for_each(|faer::unzip!(s, o)| {
					*s *= *o;
				});
			}

			#[inline]
			fn component_div_assign(&mut self, other: &Self) {
				faer::zip!(self.as_mut(), other.as_ref()).for_each(|faer::unzip!(s, o)| {
					*s /= *o;
				});
			}

			#[inline]
			fn map_mut(&mut self, mut f: impl FnMut($t) -> $t) {
				faer::zip!(self.as_mut()).for_each(|faer::unzip!(s)| {
					*s = f(*s);
				});
			}

			#[inline]
			fn map(&self, mut f: impl FnMut($t) -> $t) -> Self {
				Col::from_fn(self.nrows(), |i| f(self[i]))
			}
		}
	};
}

impl_faer_vector_ops!(f32);
impl_faer_vector_ops!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  MatrixView for faer::MatRef<'_, T>  (borrowed view)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_matrix_view_for_matref {
	($t:ty) => {
		impl MatrixView<$t> for MatRef<'_, $t> {
			type ColView<'a>
				= ColRef<'a, $t>
			where
				Self: 'a;

			#[inline]
			fn nrows(&self) -> usize {
				(*self).nrows()
			}

			#[inline]
			fn ncols(&self) -> usize {
				(*self).ncols()
			}

			#[inline]
			fn get(&self, i: usize, j: usize) -> $t {
				self[(i, j)]
			}

			#[inline]
			fn column(&self, j: usize) -> Self::ColView<'_> {
				self.col(j)
			}

			#[inline]
			fn norm(&self) -> $t {
				self.norm_l2()
			}

			#[inline]
			fn trace(&self) -> $t {
				self.diagonal().column_vector().sum()
			}

			#[inline]
			fn column_dot(&self, j: usize, other: &Self, k: usize) -> $t {
				faer::linalg::matmul::dot::inner_prod(
					self.col(j).transpose(),
					Conj::No,
					other.col(k),
					Conj::No,
				)
			}

			#[inline]
			fn frobenius_dot(&self, other: &Self) -> $t {
				let mut sum: $t = 0.0;
				faer::zip!(self, other).for_each(|faer::unzip!(a, b)| {
					sum += *a * *b;
				});
				sum
			}
		}
	};
}

impl_faer_matrix_view_for_matref!(f32);
impl_faer_matrix_view_for_matref!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  MatrixView for faer::Mat<T>  (owned)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_matrix_view_for_mat {
	($t:ty) => {
		impl MatrixView<$t> for Mat<$t> {
			type ColView<'a>
				= ColRef<'a, $t>
			where
				Self: 'a;

			#[inline]
			fn nrows(&self) -> usize {
				(*self).nrows()
			}

			#[inline]
			fn ncols(&self) -> usize {
				(*self).ncols()
			}

			#[inline]
			fn get(&self, i: usize, j: usize) -> $t {
				self[(i, j)]
			}

			#[inline]
			fn column(&self, j: usize) -> Self::ColView<'_> {
				self.as_ref().col(j)
			}

			#[inline]
			fn norm(&self) -> $t {
				self.as_ref().norm_l2()
			}

			#[inline]
			fn trace(&self) -> $t {
				self.diagonal().column_vector().sum()
			}

			#[inline]
			fn column_dot(&self, j: usize, other: &Self, k: usize) -> $t {
				faer::linalg::matmul::dot::inner_prod(
					self.as_ref().col(j).transpose(),
					Conj::No,
					other.as_ref().col(k),
					Conj::No,
				)
			}

			#[inline]
			fn frobenius_dot(&self, other: &Self) -> $t {
				let mut sum: $t = 0.0;
				faer::zip!(self, other).for_each(|faer::unzip!(a, b)| {
					sum += *a * *b;
				});
				sum
			}
		}
	};
}

impl_faer_matrix_view_for_mat!(f32);
impl_faer_matrix_view_for_mat!(f64);

// ═══════════════════════════════════════════════════════════════════════════
//  MatrixOps for faer::Mat<T>  (owned)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_faer_matrix_ops {
	($t:ty) => {
		impl MatrixOps<$t> for Mat<$t> {
			type Col = Col<$t>;
			type View<'a> = MatRef<'a, $t>;

			#[inline]
			fn zeros(nrows: usize, ncols: usize) -> Self {
				Mat::zeros(nrows, ncols)
			}

			#[inline]
			fn identity(n: usize) -> Self {
				Mat::identity(n, n)
			}

			#[inline]
			fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> $t) -> Self {
				Mat::from_fn(nrows, ncols, f)
			}

			#[inline]
			fn from_diagonal(diag: &Self::Col) -> Self {
				let n = diag.nrows();
				let mut m = Mat::zeros(n, n);
				m.as_mut()
					.diagonal_mut()
					.column_vector_mut()
					.copy_from(diag.as_ref());
				m
			}

			#[inline]
			fn from_column_slice(nrows: usize, ncols: usize, data: &[$t]) -> Self {
				faer::MatRef::from_column_major_slice(data, nrows, ncols).to_owned()
			}

			// ── View accessors (zero-alloc) ──────────────────────────

			#[inline]
			fn view_from_column_slice<'a>(
				nrows: usize,
				ncols: usize,
				data: &'a [$t],
			) -> Self::View<'a> {
				faer::MatRef::from_column_major_slice(data, nrows, ncols)
			}

			#[inline]
			fn as_view(&self) -> Self::View<'_> {
				self.as_ref()
			}

			#[inline]
			fn columns(&self, start: usize, count: usize) -> Self::View<'_> {
				self.as_ref().subcols(start, count)
			}

			#[inline]
			fn rows(&self, start: usize, count: usize) -> Self::View<'_> {
				self.as_ref().subrows(start, count)
			}

			// ── Owned conversions ────────────────────────────────────

			#[inline]
			fn column_to_owned(&self, j: usize) -> Self::Col {
				self.as_ref().col(j).to_owned()
			}

			#[inline]
			fn transpose_to_owned(&self) -> Self {
				self.transpose().to_owned()
			}

			#[inline]
			fn columns_to_owned(&self, start: usize, count: usize) -> Self {
				self.as_ref().subcols(start, count).to_owned()
			}

			#[inline]
			fn rows_to_owned(&self, start: usize, count: usize) -> Self {
				self.as_ref().subrows(start, count).to_owned()
			}

			// ── Element access ───────────────────────────────────────

			#[inline]
			fn get_mut(&mut self, i: usize, j: usize) -> &mut $t {
				&mut self[(i, j)]
			}

			#[inline]
			fn set_column(&mut self, j: usize, col: &Self::Col) {
				self.as_mut().col_mut(j).copy_from(col.as_ref());
			}

			#[inline]
			fn set_rows(&mut self, start: usize, src: &Self) {
				self.as_mut()
					.subrows_mut(start, src.nrows())
					.copy_from(src.as_ref());
			}

			#[inline]
			fn as_slice(&self) -> &[$t] {
				let ptr = self.as_ptr();
				let nrows = (*self).nrows();
				let ncols = (*self).ncols();
				let stride = self.col_stride();
				if ncols <= 1 || stride == nrows as isize {
					let len = if ncols == 0 {
						0
					} else {
						stride as usize * (ncols - 1) + nrows
					};
					unsafe { std::slice::from_raw_parts(ptr, len) }
				} else {
					panic!(
						"faer::Mat as_slice: non-contiguous layout \
						 (col_stride={stride} != nrows={nrows})"
					);
				}
			}

			#[inline]
			fn as_mut_slice(&mut self) -> &mut [$t] {
				let ptr = self.as_ptr_mut();
				let nrows = (*self).nrows();
				let ncols = (*self).ncols();
				let stride = self.col_stride();
				if ncols <= 1 || stride == nrows as isize {
					let len = if ncols == 0 {
						0
					} else {
						stride as usize * (ncols - 1) + nrows
					};
					unsafe { std::slice::from_raw_parts_mut(ptr, len) }
				} else {
					panic!(
						"faer::Mat as_mut_slice: non-contiguous layout \
						 (col_stride={stride} != nrows={nrows})"
					);
				}
			}

			#[inline]
			fn column_as_mut_slice(&mut self, j: usize) -> &mut [$t] {
				self.col_as_slice_mut(j)
			}

			// ── In-place mutations ───────────────────────────────────

			#[inline]
			fn copy_from(&mut self, other: &Self) {
				self.as_mut().copy_from(other.as_ref());
			}

			#[inline]
			fn fill(&mut self, value: $t) {
				self.as_mut().fill(value);
			}

			#[inline]
			fn scale_mut(&mut self, alpha: $t) {
				*self *= faer::Scale(alpha);
			}

			#[inline]
			fn scale_columns(&mut self, source: &Self, diag: &Self::Col) {
				let p = source.ncols();
				for j in 0..p {
					let d = diag[j];
					let src_col = source.as_ref().col(j);
					let dst_col = self.as_mut().col_mut(j);
					faer::zip!(dst_col, src_col).for_each(|faer::unzip!(dst, src)| {
						*dst = d * *src;
					});
				}
			}

			// ── Allocating arithmetic ────────────────────────────────

			#[inline]
			fn mat_mul(&self, other: &Self) -> Self {
				self * other
			}

			#[inline]
			fn mat_vec(&self, v: &Self::Col) -> Self::Col {
				self * v
			}

			#[inline]
			fn add(&self, other: &Self) -> Self {
				self + other
			}

			#[inline]
			fn sub(&self, other: &Self) -> Self {
				self - other
			}

			#[inline]
			fn scale_by(&self, alpha: $t) -> Self {
				self * faer::Scale(alpha)
			}

			// ── In-place element-wise ────────────────────────────────

			#[inline]
			fn mat_axpy(&mut self, alpha: $t, x: &Self, beta: $t) {
				if beta == 0.0 {
					faer::zip!(self.as_mut(), x.as_ref()).for_each(|faer::unzip!(s, x)| {
						*s = alpha * *x;
					});
				} else if beta == 1.0 {
					faer::zip!(self.as_mut(), x.as_ref()).for_each(|faer::unzip!(s, x)| {
						*s += alpha * *x;
					});
				} else {
					faer::zip!(self.as_mut(), x.as_ref()).for_each(|faer::unzip!(s, x)| {
						*s = beta * *s + alpha * *x;
					});
				}
			}

			// ── In-place Matrix-Vector ───────────────────────────────

			#[inline]
			fn mat_vec_axpy(&self, alpha: $t, x: &Self::Col, beta: $t, y: &mut Self::Col) {
				// y = alpha * A * x + beta * y
				// Use faer's SIMD-optimized matmul with as_mat() for zero-copy Col→MatRef
				let par = faer::get_global_parallelism();
				let accum = if beta == 0.0 {
					faer::Accum::Replace
				} else {
					if beta != 1.0 {
						*y *= faer::Scale(beta);
					}
					faer::Accum::Add
				};
				faer::linalg::matmul::matmul(
					y.as_mut().as_mat_mut(),
					accum,
					self.as_ref(),
					x.as_ref().as_mat(),
					alpha,
					par,
				);
			}

			#[inline]
			fn mat_t_vec_axpy(&self, alpha: $t, x: &Self::Col, beta: $t, y: &mut Self::Col) {
				// y = alpha * A^T * x + beta * y
				// Uses faer's native transpose view (stride swap, zero-alloc).
				let par = faer::get_global_parallelism();
				let accum = if beta == 0.0 {
					faer::Accum::Replace
				} else {
					if beta != 1.0 {
						*y *= faer::Scale(beta);
					}
					faer::Accum::Add
				};
				faer::linalg::matmul::matmul(
					y.as_mut().as_mat_mut(),
					accum,
					self.as_ref().transpose(),
					x.as_ref().as_mat(),
					alpha,
					par,
				);
			}

			#[inline]
			fn mat_vec_into(&self, x: &Self::Col, y: &mut Self::Col) {
				let par = faer::get_global_parallelism();
				faer::linalg::matmul::matmul(
					y.as_mut().as_mat_mut(),
					faer::Accum::Replace,
					self.as_ref(),
					x.as_ref().as_mat(),
					1.0 as $t,
					par,
				);
			}

			#[inline]
			fn ger(&mut self, alpha: $t, x: &Self::Col, y: &Self::Col) {
				// self += alpha * x * y^T  (rank-1 update)
				// Uses faer's native matmul with column-as-matrix views.
				let par = faer::get_global_parallelism();
				faer::linalg::matmul::matmul(
					self.as_mut(),
					faer::Accum::Add,
					x.as_ref().as_mat(),
					y.as_ref().as_mat().transpose(),
					alpha,
					par,
				);
			}

			#[inline]
			fn mat_component_mul_assign(&mut self, other: &Self) {
				faer::zip!(self.as_mut(), other.as_ref()).for_each(|faer::unzip!(s, o)| {
					*s *= *o;
				});
			}

			#[inline]
			fn add_assign(&mut self, other: &Self) {
				*self += other;
			}

			#[inline]
			fn sub_assign(&mut self, other: &Self) {
				*self -= other;
			}

			#[inline]
			fn add_transpose_of(&mut self, alpha: $t, src: &Self) {
				let src_t = src.as_ref().transpose();
				faer::zip!(self.as_mut(), src_t).for_each(|faer::unzip!(d, s)| {
					*d += alpha * *s;
				});
			}

			#[inline]
			fn transpose_into(&self, dst: &mut Self) {
				dst.as_mut().copy_from(self.as_ref().transpose());
			}

			// ── GEMM (view operands) ─────────────────────────────────

			#[inline]
			fn gemm(&mut self, alpha: $t, a: Self::View<'_>, b: Self::View<'_>, beta: $t) {
				let par = faer::get_global_parallelism();
				if beta == 0.0 {
					faer::linalg::matmul::matmul(
						self.as_mut(),
						faer::Accum::Replace,
						a,
						b,
						alpha,
						par,
					);
				} else if beta == 1.0 {
					faer::linalg::matmul::matmul(self.as_mut(), faer::Accum::Add, a, b, alpha, par);
				} else {
					*self *= faer::Scale(beta);
					faer::linalg::matmul::matmul(self.as_mut(), faer::Accum::Add, a, b, alpha, par);
				}
			}

			#[inline]
			fn gemm_at(&mut self, alpha: $t, a: Self::View<'_>, b: Self::View<'_>, beta: $t) {
				let par = faer::get_global_parallelism();
				// Zero-cost transpose: stride swap on the view, no allocation.
				let at = a.transpose();
				if beta == 0.0 {
					faer::linalg::matmul::matmul(
						self.as_mut(),
						faer::Accum::Replace,
						at,
						b,
						alpha,
						par,
					);
				} else if beta == 1.0 {
					faer::linalg::matmul::matmul(
						self.as_mut(),
						faer::Accum::Add,
						at,
						b,
						alpha,
						par,
					);
				} else {
					*self *= faer::Scale(beta);
					faer::linalg::matmul::matmul(
						self.as_mut(),
						faer::Accum::Add,
						at,
						b,
						alpha,
						par,
					);
				}
			}

			#[inline]
			fn gemm_bt(&mut self, alpha: $t, a: Self::View<'_>, b: Self::View<'_>, beta: $t) {
				let par = faer::get_global_parallelism();
				let bt = b.transpose();
				if beta == 0.0 {
					faer::linalg::matmul::matmul(
						self.as_mut(),
						faer::Accum::Replace,
						a,
						bt,
						alpha,
						par,
					);
				} else if beta == 1.0 {
					faer::linalg::matmul::matmul(
						self.as_mut(),
						faer::Accum::Add,
						a,
						bt,
						alpha,
						par,
					);
				} else {
					*self *= faer::Scale(beta);
					faer::linalg::matmul::matmul(
						self.as_mut(),
						faer::Accum::Add,
						a,
						bt,
						alpha,
						par,
					);
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
			type ScratchBuffer = MemBuffer;

			// ── QR helpers ───────────────────────────────────────────

			fn qr_h_factor_shape(nrows: usize, ncols: usize) -> (usize, usize) {
				let k = nrows.min(ncols);
				let block_size = factor::recommended_block_size::<$t>(nrows, ncols);
				(block_size, k)
			}

			fn create_qr_scratch(nrows: usize, ncols: usize) -> MemBuffer {
				let k = nrows.min(ncols);
				let block_size = factor::recommended_block_size::<$t>(nrows, ncols);

				let qr_req = factor::qr_in_place_scratch::<$t>(
					nrows,
					ncols,
					block_size,
					Par::Seq,
					Default::default(),
				);
				let apply_req =
					householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<$t>(
						nrows, block_size, k,
					);

				MemBuffer::new(StackReq::any_of(&[qr_req, apply_req]))
			}

			// ── QR ─────────────────────────────────
			#[inline]
			fn qr(
				&mut self,
				q: &mut Self,
				r: &mut Self,
				h_factor: &mut Self,
				scratch: &mut MemBuffer,
			) {
				let m = self.nrows();
				let n = self.ncols();
				let k = m.min(n);

				debug_assert_eq!((q.nrows(), q.ncols()), (m, k));
				debug_assert_eq!((r.nrows(), r.ncols()), (k, n));
				debug_assert_eq!(h_factor.ncols(), k);

				let mut stack = MemStack::new(scratch);

				// 1. Factorize in-place: R in upper triangle,
				//    Householder vectors in strict lower triangle,
				//    block-reflector coefficients in h_factor.
				factor::qr_in_place(
					self.as_mut(),
					h_factor.as_mut(),
					Par::Seq,
					stack.rb_mut(),
					Default::default(),
				);

				// 2. Extract R (upper triangle of first k rows).
				r.fill(0.0 as $t);
				for j in 0..n {
					let i_end = (j + 1).min(k);
					for i in 0..i_end {
						r[(i, j)] = self[(i, j)];
					}
				}

				// 3. Reconstruct thin Q: identity → apply H₁H₂…Hₖ.
				q.fill(0.0 as $t);
				for i in 0..k {
					q[(i, i)] = 1.0 as $t;
				}

				householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
					self.as_ref().subcols(0, k),
					h_factor.as_ref(),
					Conj::No,
					q.as_mut(),
					Par::Seq,
					stack.rb_mut(),
				);
			}

			// ── SVD ──────────────────────────────────────────────────
			#[inline]
			fn svd(&self, u: &mut Self, s: &mut Col<$t>, vt: &mut Self) {
				let m = self.nrows();
				let n = self.ncols();
				let k = m.min(n);

				debug_assert_eq!((u.nrows(), u.ncols()), (m, k));
				debug_assert_eq!(s.nrows(), k);
				debug_assert_eq!((vt.nrows(), vt.ncols()), (k, n));

				let decomp = self.thin_svd().expect("SVD failed");
				u.as_mut().copy_from(decomp.U());
				s.as_mut().copy_from(decomp.S().column_vector());
				vt.as_mut().copy_from(decomp.V().transpose());
			}

			// ── Symmetric Eigen ──────────────────────────────────────
			#[inline]
			fn symmetric_eigen(&self, eigenvalues: &mut Col<$t>, eigenvectors: &mut Self) {
				let n = self.nrows();
				debug_assert_eq!(self.ncols(), n);
				debug_assert_eq!(eigenvalues.nrows(), n);
				debug_assert_eq!((eigenvectors.nrows(), eigenvectors.ncols()), (n, n));

				let decomp = self
					.self_adjoint_eigen(Side::Lower)
					.expect("Eigendecomposition failed");
				eigenvalues.as_mut().copy_from(decomp.S().column_vector());
				eigenvectors.as_mut().copy_from(decomp.U());
			}

			// ── Cholesky ─────────────────────────────────────────────
			#[inline]
			fn cholesky(&self, l: &mut Self) -> bool {
				let n = self.nrows();
				debug_assert_eq!(self.ncols(), n);
				debug_assert_eq!((l.nrows(), l.ncols()), (n, n));

				match self.llt(Side::Lower) {
					Ok(llt) => {
						l.as_mut().copy_from(llt.L());
						true
					}
					Err(_) => false,
				}
			}

			// ── Inverse ──────────────────────────────────────────────
			#[inline]
			fn inverse(&self, result: &mut Self) -> bool {
				let n = self.nrows();
				if n != self.ncols() {
					return false;
				}
				*result = Mat::identity(n, n);
				match self.llt(Side::Lower) {
					Ok(llt) => {
						llt.solve_in_place(result.as_mut());
						true
					}
					Err(_) => {
						*result = self.partial_piv_lu().inverse();
						true
					}
				}
			}

			// ── Cholesky Solve ───────────────────────────────────────
			#[inline]
			fn cholesky_solve(&self, rhs: &Self, result: &mut Self) -> bool {
				match self.llt(Side::Lower) {
					Ok(llt) => {
						result.as_mut().copy_from(rhs.as_ref());
						llt.solve_in_place(result.as_mut());
						true
					}
					Err(_) => false,
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
		assert_eq!(VectorView::len(&v), 3);
		assert!((VectorView::get(&v, 0) - 1.0).abs() < 1e-14);
		assert!((VectorView::norm_squared(&v) - 14.0).abs() < 1e-14);
	}

	#[test]
	fn test_vector_dot() {
		let a = <Col<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let b = <Col<f64> as VectorOps<f64>>::from_slice(&[4.0, 5.0, 6.0]);
		assert!((VectorView::dot(&a, &b) - 32.0).abs() < 1e-14);
	}

	#[test]
	fn test_colref_view_dot() {
		let a = <Col<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let b = <Col<f64> as VectorOps<f64>>::from_slice(&[4.0, 5.0, 6.0]);
		// Use views directly — zero allocation.
		let ar = a.as_ref();
		let br = b.as_ref();
		assert!((VectorView::dot(&ar, &br) - 32.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_identity() {
		let m = <Mat<f64> as MatrixOps<f64>>::identity(3);
		assert_eq!(MatrixView::nrows(&m), 3);
		assert!((MatrixView::trace(&m) - 3.0).abs() < 1e-14);
	}

	#[test]
	fn test_column_view_zero_alloc() {
		let m = <Mat<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3 + 1) as f64);
		// column() now returns ColRef — no heap allocation.
		let col0 = m.column(0);
		assert!((VectorView::get(&col0, 0) - 1.0).abs() < 1e-14);
		assert!((VectorView::get(&col0, 2) - 3.0).abs() < 1e-14);
		let col1 = m.column(1);
		assert!((VectorView::get(&col1, 0) - 4.0).abs() < 1e-14);
	}

	#[test]
	fn test_columns_view_zero_alloc() {
		let m = <Mat<f64> as MatrixOps<f64>>::from_fn(3, 4, |i, j| (i + j * 3) as f64);
		let sub = m.columns(1, 2);
		assert_eq!(MatrixView::nrows(&sub), 3);
		assert_eq!(MatrixView::ncols(&sub), 2);
		assert!((MatrixView::get(&sub, 0, 0) - 3.0).abs() < 1e-14); // col 1, row 0
	}

	#[test]
	fn test_gemm_with_views() {
		let a = <Mat<f64> as MatrixOps<f64>>::identity(3);
		let b = <Mat<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3 + 1) as f64);
		let mut c = <Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
		// Pass views to gemm — zero-alloc operands.
		c.gemm(1.0, a.as_view(), b.as_view(), 0.0);
		assert!((MatrixView::get(&c, 0, 0) - 1.0).abs() < 1e-14);
		assert!((MatrixView::get(&c, 2, 1) - 6.0).abs() < 1e-14);
	}

	#[test]
	fn test_gemm_at_with_views() {
		let a = <Mat<f64> as MatrixOps<f64>>::identity(3);
		let b = <Mat<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3 + 1) as f64);
		let mut c = <Mat<f64> as MatrixOps<f64>>::zeros(3, 2);
		c.gemm_at(1.0, a.as_view(), b.as_view(), 0.0);
		assert!((MatrixView::get(&c, 0, 0) - 1.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_mul() {
		let a = <Mat<f64> as MatrixOps<f64>>::identity(3);
		let v = <Col<f64> as VectorOps<f64>>::from_slice(&[1.0, 2.0, 3.0]);
		let r = MatrixOps::mat_vec(&a, &v);
		assert!((VectorView::get(&r, 0) - 1.0).abs() < 1e-14);
		assert!((VectorView::get(&r, 2) - 3.0).abs() < 1e-14);
	}

	#[test]
	fn test_svd() {
		let m = <Mat<f64> as MatrixOps<f64>>::identity(3);
		let mut u = Mat::<f64>::zeros(3, 3);
		let mut s = Col::<f64>::zeros(3);
		let mut vt = Mat::<f64>::zeros(3, 3);
		DecompositionOps::svd(&m, &mut u, &mut s, &mut vt);
		for i in 0..3 {
			assert!((VectorView::get(&s, i) - 1.0).abs() < 1e-10);
		}
	}

	#[test]
	fn test_symmetric_eigen() {
		let diag = <Col<f64> as VectorOps<f64>>::from_slice(&[3.0, 1.0, 2.0]);
		let m = <Mat<f64> as MatrixOps<f64>>::from_diagonal(&diag);
		let mut eigenvalues = Col::<f64>::zeros(3);
		let mut eigenvectors = Mat::<f64>::zeros(3, 3);
		DecompositionOps::symmetric_eigen(&m, &mut eigenvalues, &mut eigenvectors);
		let mut eigs: Vec<f64> = VectorView::iter(&eigenvalues).collect();
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
		assert!((VectorView::get(&sum, 0) - 5.0).abs() < 1e-14);
		assert!((VectorView::get(&sum, 2) - 9.0).abs() < 1e-14);
		let diff = VectorOps::sub(&a, &b);
		assert!((VectorView::get(&diff, 0) - (-3.0)).abs() < 1e-14);
		let neg = VectorOps::neg(&a);
		assert!((VectorView::get(&neg, 1) - (-2.0)).abs() < 1e-14);
	}

	#[test]
	fn test_vector_component_mul() {
		let a = <Col<f64> as VectorOps<f64>>::from_slice(&[2.0, 3.0, 4.0]);
		let b = <Col<f64> as VectorOps<f64>>::from_slice(&[5.0, 6.0, 7.0]);
		let c = VectorOps::component_mul(&a, &b);
		assert!((VectorView::get(&c, 0) - 10.0).abs() < 1e-14);
		assert!((VectorView::get(&c, 1) - 18.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_column_as_mut_slice() {
		let mut m = <Mat<f64> as MatrixOps<f64>>::from_fn(3, 2, |i, j| (i + j * 3) as f64);
		let col = MatrixOps::column_as_mut_slice(&mut m, 1);
		assert!((col[0] - 3.0).abs() < 1e-14);
		col[0] = 99.0;
		assert!((MatrixView::get(&m, 0, 1) - 99.0).abs() < 1e-14);
	}

	#[test]
	fn test_matrix_from_column_slice() {
		let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
		let m = <Mat<f64> as MatrixOps<f64>>::from_column_slice(3, 2, &data);
		assert!((MatrixView::get(&m, 0, 0) - 1.0).abs() < 1e-14);
		assert!((MatrixView::get(&m, 2, 0) - 3.0).abs() < 1e-14);
		assert!((MatrixView::get(&m, 0, 1) - 4.0).abs() < 1e-14);
	}

	#[test]
	fn test_cholesky_solve() {
		let a = <Mat<f64> as MatrixOps<f64>>::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
		let b = <Mat<f64> as MatrixOps<f64>>::identity(2);
		let mut x = <Mat<f64> as MatrixOps<f64>>::zeros(2, 2);
		assert!(DecompositionOps::cholesky_solve(&a, &b, &mut x));
		let ax = MatrixOps::mat_mul(&a, &x);
		for i in 0..2 {
			for j in 0..2 {
				let expected = if i == j { 1.0 } else { 0.0 };
				assert!((MatrixView::get(&ax, i, j) - expected).abs() < 1e-10);
			}
		}
	}

	#[test]
	fn test_backend_type() {
		type V = <FaerBackend as LinAlgBackend<f64>>::Vector;
		type M = <FaerBackend as LinAlgBackend<f64>>::Matrix;
		let v = V::zeros(5);
		let m = <M as MatrixOps<f64>>::identity(3);
		assert_eq!(VectorView::len(&v), 5);
		assert_eq!(MatrixView::nrows(&m), 3);
	}
}
