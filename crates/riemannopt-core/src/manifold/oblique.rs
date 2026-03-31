//! # Oblique Manifold OB(n,p)
//!
//! The Oblique manifold OB(n,p) is the product of p unit spheres in ℝⁿ.
//! X ∈ ℝⁿˣᵖ such that ‖x_j‖₂ = 1 for all j=1..p.

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorView},
	manifold::Manifold,
	types::Scalar,
};

/// The Oblique manifold OB(n,p).
///
/// Consists of n×p matrices where each column is a unit vector.
#[derive(Clone)]
pub struct Oblique<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	p: usize,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for Oblique<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Oblique OB({}, {})", self.n, self.p)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Oblique<T, B> {
	pub fn new(n: usize, p: usize) -> Self {
		assert!(
			n >= 1 && p >= 1,
			"Oblique manifold requires n >= 1 and p >= 1"
		);
		Self {
			n,
			p,
			_phantom: PhantomData,
		}
	}
}

impl<T, B> Manifold<T> for Oblique<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Matrix;
	type TangentVector = B::Matrix;
	type Workspace = (); // Columns are processed independently, no large buffers needed

	#[inline]
	fn name(&self) -> &str {
		"Oblique"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.p * (self.n - 1)
	}

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.p {
			return false;
		}
		for j in 0..self.p {
			let col = point.column(j);
			if (col.norm() - T::one()).abs() > tol {
				return false;
			}
		}
		true
	}

	#[inline]
	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		for j in 0..self.p {
			let x_j = point.column(j);
			let v_j = vector.column(j);
			if x_j.dot(&v_j).abs() > tol {
				return false;
			}
		}
		true
	}

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		for j in 0..self.p {
			let norm = point.column(j).norm();
			let res_j = result.column_as_mut_slice(j);
			if norm > T::EPSILON {
				let inv_norm = T::one() / norm;
				for (i, val) in res_j[..self.n].iter_mut().enumerate() {
					*val = point.get(i, j) * inv_norm;
				}
			} else {
				res_j.fill(T::zero());
				res_j[0] = T::one();
			}
		}
	}

	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		for j in 0..self.p {
			let x_j = point.column(j);
			let v_j = vector.column(j);
			let inner = x_j.dot(&v_j);

			let res_j = result.column_as_mut_slice(j);
			for (i, val) in res_j[..self.n].iter_mut().enumerate() {
				*val = vector.get(i, j) - inner * point.get(i, j);
			}
		}
	}

	#[inline]
	fn inner_product(
		&self,
		_p: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> T {
		u.frobenius_dot(v)
	}

	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) {
		// R_x(v) column-wise: (x+v)/||x+v||
		for j in 0..self.p {
			let res_j_slice = result.column_as_mut_slice(j);
			let mut norm_sq = T::zero();
			for (i, dst) in res_j_slice[..self.n].iter_mut().enumerate() {
				let v = point.get(i, j) + tangent.get(i, j);
				*dst = v;
				norm_sq += v * v;
			}
			let norm = norm_sq.sqrt();
			let inv_norm = if norm > T::EPSILON {
				T::one() / norm
			} else {
				T::one()
			};
			for dst in &mut res_j_slice[..self.n] {
				*dst *= inv_norm;
			}
		}
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		egrad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut (),
	) {
		self.project_tangent(point, egrad, result, ws);
	}

	#[inline]
	fn euclidean_to_riemannian_hessian(
		&self,
		point: &Self::Point,
		egrad: &Self::TangentVector,
		ehvp: &Self::TangentVector,
		xi: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		// Hess f(X)[ξ] column-wise: project(ehvp_j) - <x_j, egrad_j> * xi_j
		for j in 0..self.p {
			let x_j = point.column(j);
			let eg_j = egrad.column(j);
			let inner_eg = x_j.dot(&eg_j);

			let ehvp_j = ehvp.column(j);
			let inner_hvp = x_j.dot(&ehvp_j);

			let res_j = result.column_as_mut_slice(j);
			for (i, val) in res_j[..self.n].iter_mut().enumerate() {
				let proj_hvp = ehvp.get(i, j) - inner_hvp * point.get(i, j);
				*val = proj_hvp - inner_eg * xi.get(i, j);
			}
		}
	}

	// ════════════════════════════════════════════════════════════════════════
	// Vector Ops (Algebraic)
	// ════════════════════════════════════════════════════════════════════════

	#[inline]
	fn scale_tangent(&self, scalar: T, v: &mut Self::TangentVector) {
		v.scale_mut(scalar);
	}

	#[inline]
	fn add_tangents(&self, v1: &mut Self::TangentVector, v2: &Self::TangentVector) {
		v1.add_assign(v2);
	}

	#[inline]
	fn axpy_tangent(&self, alpha: T, x: &Self::TangentVector, y: &mut Self::TangentVector) {
		y.mat_axpy(alpha, x, T::one());
	}

	#[inline]
	fn allocate_point(&self) -> Self::Point {
		B::Matrix::zeros(self.n, self.p)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		B::Matrix::zeros(self.n, self.p)
	}

	/// log_x(y) column-wise: θ_j/sin(θ_j) · (y_j − ⟨x_j, y_j⟩ x_j)
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		for j in 0..self.p {
			let x_j = point.column(j);
			let y_j = other.column(j);
			let inner = x_j.dot(&y_j);
			let clamped = inner.min(T::one()).max(-T::one());
			let theta = clamped.acos();

			let res_j = result.column_as_mut_slice(j);
			if theta < T::SMALL_ANGLE_THRESHOLD {
				// Taylor: θ/sin(θ) ≈ 1 + θ²/6
				let scale = T::one() + theta * theta * <T as Scalar>::from_f64(1.0 / 6.0);
				for (i, val) in res_j[..self.n].iter_mut().enumerate() {
					*val = scale * (other.get(i, j) - inner * point.get(i, j));
				}
			} else {
				let scale = theta / theta.sin();
				for (i, val) in res_j[..self.n].iter_mut().enumerate() {
					*val = scale * (other.get(i, j) - inner * point.get(i, j));
				}
			}
		}
	}

	/// Column-wise parallel transport (projection onto tangent space at `to`).
	#[inline]
	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		// Vector transport by projection: project each column onto T_{to_j} S^{n-1}
		for j in 0..self.p {
			let y_j = to.column(j);
			let v_j = vector.column(j);
			let inner = y_j.dot(&v_j);
			let res_j = result.column_as_mut_slice(j);
			for (i, val) in res_j[..self.n].iter_mut().enumerate() {
				*val = vector.get(i, j) - inner * to.get(i, j);
			}
		}
	}

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;
		for j in 0..self.p {
			let res_j = result.column_as_mut_slice(j);
			let mut norm_sq = T::zero();
			for val in &mut res_j[..self.n] {
				let v = <T as Scalar>::from_f64(normal.sample(&mut rng));
				*val = v;
				norm_sq += v * v;
			}
			let inv = T::one() / norm_sq.sqrt();
			for val in &mut res_j[..self.n] {
				*val *= inv;
			}
		}
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for j in 0..self.p {
			let res_j = result.column_as_mut_slice(j);
			let mut inner = T::zero();

			for (i, val) in res_j[..self.n].iter_mut().enumerate() {
				let v = <T as Scalar>::from_f64(normal.sample(&mut rng));
				*val = v;
				inner += point.get(i, j) * v;
			}

			// Projection tangentielle : v_j = v_j - <x_j, v_j> x_j
			for (i, val) in res_j[..self.n].iter_mut().enumerate() {
				*val -= inner * point.get(i, j);
			}
		}

		let norm_sq = result.frobenius_dot(result);
		if norm_sq > T::zero() {
			result.scale_mut(T::one() / norm_sq.sqrt());
		}
	}

	/// d(X, Y) = √(Σ_j arccos²(x_j · y_j))  (product geodesic distance)
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let mut dist_sq = T::zero();
		for j in 0..self.p {
			let inner = x.column(j).dot(&y.column(j));
			let clamped = inner.min(T::one()).max(-T::one());
			let theta = clamped.acos();
			dist_sq += theta * theta;
		}
		dist_sq.sqrt()
	}
}
