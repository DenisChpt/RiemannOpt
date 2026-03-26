//! Shared test utilities and manifolds for benchmarks and tests.

#![cfg(any(test, feature = "test-utils"))]

use crate::{
	cost_function::CostFunction,
	error::Result,
	linalg::{self, MatrixOps, VectorOps, VectorView},
	manifold::Manifold,
};
use rand::prelude::*;

/// Test sphere manifold for optimization problems
#[derive(Debug, Clone)]
pub struct TestSphere {
	dim: usize,
}

impl TestSphere {
	pub fn new(dim: usize) -> Self {
		Self { dim }
	}
}

impl Manifold<f64> for TestSphere {
	type Point = linalg::Vec<f64>;
	type TangentVector = linalg::Vec<f64>;
	type Workspace = ();
	fn name(&self) -> &str {
		"Test Sphere"
	}

	fn dimension(&self) -> usize {
		self.dim - 1
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: f64) -> bool {
		(VectorView::norm(point) - 1.0).abs() < tol
	}

	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: f64,
	) -> bool {
		VectorView::dot(point, vector).abs() < tol
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let norm = VectorView::norm(point);
		if norm > f64::EPSILON {
			VectorOps::copy_from(result, point);
			VectorOps::scale_mut(result, 1.0 / norm);
		} else {
			VectorOps::fill(result, 0.0);
			*VectorOps::get_mut(result, 0) = 1.0;
		}
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		let inner = VectorView::dot(point, vector);
		VectorOps::copy_from(result, vector);
		result.axpy(-inner, point, 1.0);
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> Result<f64> {
		Ok(VectorView::dot(u, v))
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) -> Result<()> {
		// Exponential map on sphere
		let norm_v = VectorView::norm(tangent);
		if norm_v < f64::EPSILON {
			VectorOps::copy_from(result, point);
		} else {
			let cos_norm = norm_v.cos();
			let sin_norm = norm_v.sin();
			// result = point * cos_norm + tangent * (sin_norm / norm_v)
			VectorOps::copy_from(result, point);
			VectorOps::scale_mut(result, cos_norm);
			result.axpy(sin_norm / norm_v, tangent, 1.0);
		}
		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		let inner = VectorView::dot(point, other).min(1.0).max(-1.0);
		let theta = inner.acos();

		if theta.abs() < f64::EPSILON {
			VectorOps::fill(result, 0.0);
		} else {
			// v = other - point * inner
			VectorOps::copy_from(result, other);
			result.axpy(-inner, point, 1.0);
			let v_norm = VectorView::norm(result);
			if v_norm > f64::EPSILON {
				VectorOps::scale_mut(result, theta / v_norm);
			} else {
				VectorOps::fill(result, 0.0);
			}
		}
		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		self.project_tangent(point, euclidean_grad, result, _ws)
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		let mut rng = rand::rng();
		*result = linalg::Vec::<f64>::from_fn(self.dim, |_| rng.random::<f64>() * 2.0 - 1.0);
		let v = result.clone();
		self.project_point(&v, result);
		Ok(())
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		let mut rng = rand::rng();
		let v = linalg::Vec::<f64>::from_fn(self.dim, |_| rng.random::<f64>() * 2.0 - 1.0);
		self.project_tangent(point, &v, result, &mut ())
	}

	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) -> Result<()> {
		self.project_tangent(to, vector, result, _ws)
	}

	fn has_exact_exp_log(&self) -> bool {
		true
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<f64> {
		let mut tangent = linalg::Vec::<f64>::zeros(VectorView::len(x));
		self.inverse_retract(x, y, &mut tangent, &mut ())?;
		self.inner_product(x, &tangent, &tangent, &mut ())
			.map(|v| v.sqrt())
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: f64,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, tangent);
		VectorOps::scale_mut(result, scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		let sum = VectorOps::add(v1, v2);
		self.project_tangent(point, &sum, result, &mut ())
	}

	fn axpy_tangent(
		&self,
		point: &Self::Point,
		alpha: f64,
		x: &Self::TangentVector,
		y: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_temp: &mut Self::TangentVector,
	) -> Result<()> {
		// result = y + alpha * x
		let mut axpy_result = y.clone();
		axpy_result.axpy(alpha, x, 1.0);
		self.project_tangent(point, &axpy_result, result, &mut ())
	}

	fn norm(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_ws: &mut (),
	) -> Result<f64> {
		Ok(VectorView::norm(vector))
	}
}

/// Rayleigh quotient minimization problem
#[derive(Debug)]
pub struct RayleighQuotient {
	matrix: linalg::Mat<f64>,
}

impl RayleighQuotient {
	pub fn new(dim: usize) -> Self {
		let mut rng = rand::rng();
		let m = linalg::Mat::<f64>::from_fn(dim, dim, |_, _| rng.random::<f64>());
		// Make symmetric: matrix = m + m^T
		let mt = MatrixOps::transpose_to_owned(&m);
		let matrix = MatrixOps::add(&m, &mt);
		Self { matrix }
	}
}

impl CostFunction<f64> for RayleighQuotient {
	type Point = linalg::Vec<f64>;
	type TangentVector = linalg::Vec<f64>;
	fn cost(&self, x: &Self::Point) -> Result<f64> {
		let ax = self.matrix.mat_vec(x);
		Ok(VectorView::dot(x, &ax))
	}

	fn cost_and_gradient(
		&self,
		x: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let ax = self.matrix.mat_vec(x);
		let cost = VectorView::dot(x, &ax);
		// gradient = 2 * A * x
		VectorOps::copy_from(gradient, &ax);
		VectorOps::scale_mut(gradient, 2.0);
		Ok(cost)
	}

	fn cost_and_gradient_alloc(&self, x: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		let ax = self.matrix.mat_vec(x);
		let cost = VectorView::dot(x, &ax);
		let mut gradient = ax;
		VectorOps::scale_mut(&mut gradient, 2.0);
		Ok((cost, gradient))
	}

	fn hessian_vector_product(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// For Rayleigh quotient: H*v = 2*A*v
		let mut result = self.matrix.mat_vec(vector);
		VectorOps::scale_mut(&mut result, 2.0);
		Ok(result)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let eps = 1e-8;
		let n = VectorView::len(point);
		let mut gradient = linalg::Vec::<f64>::zeros(n);

		for i in 0..n {
			let mut point_plus = point.clone();
			let mut point_minus = point.clone();
			*VectorOps::get_mut(&mut point_plus, i) += eps;
			*VectorOps::get_mut(&mut point_minus, i) -= eps;

			let f_plus = self.cost(&point_plus)?;
			let f_minus = self.cost(&point_minus)?;
			*VectorOps::get_mut(&mut gradient, i) = (f_plus - f_minus) / (2.0 * eps);
		}

		Ok(gradient)
	}
}

/// Spherical Rosenbrock function
#[derive(Debug)]
pub struct SphericalRosenbrock {
	dim: usize,
}

impl SphericalRosenbrock {
	pub fn new(dim: usize) -> Self {
		Self { dim }
	}
}

impl CostFunction<f64> for SphericalRosenbrock {
	type Point = linalg::Vec<f64>;
	type TangentVector = linalg::Vec<f64>;
	fn cost(&self, x: &Self::Point) -> Result<f64> {
		let mut cost = 0.0;
		for i in 0..self.dim - 1 {
			let xi = VectorView::get(x, i);
			let xi1 = VectorView::get(x, i + 1);
			let a = 1.0 - xi;
			let b = xi1 - xi * xi;
			cost += a * a + 100.0 * b * b;
		}
		Ok(cost)
	}

	fn cost_and_gradient(
		&self,
		x: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let mut cost = 0.0;
		VectorOps::fill(gradient, 0.0);

		for i in 0..self.dim - 1 {
			let xi = VectorView::get(x, i);
			let xi1 = VectorView::get(x, i + 1);
			let a = 1.0 - xi;
			let b = xi1 - xi * xi;
			cost += a * a + 100.0 * b * b;

			*VectorOps::get_mut(gradient, i) += -2.0 * a - 400.0 * xi * b;
			if i > 0 {
				let xi_prev = VectorView::get(x, i - 1);
				*VectorOps::get_mut(gradient, i) += 200.0 * (xi - xi_prev * xi_prev);
			}
		}
		if self.dim > 1 {
			let x_last = VectorView::get(x, self.dim - 1);
			let x_prev = VectorView::get(x, self.dim - 2);
			*VectorOps::get_mut(gradient, self.dim - 1) += 200.0 * (x_last - x_prev * x_prev);
		}

		Ok(cost)
	}

	fn cost_and_gradient_alloc(&self, x: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		let mut gradient = linalg::Vec::<f64>::zeros(self.dim);
		let cost = self.cost_and_gradient(x, &mut gradient)?;
		Ok((cost, gradient))
	}

	fn hessian_vector_product(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// Approximate using finite differences
		let eps = 1e-8;
		let (_, grad1) = self.cost_and_gradient_alloc(point)?;
		// point_plus = point + eps * vector
		let mut point_plus = point.clone();
		point_plus.axpy(eps, vector, 1.0);
		let (_, grad2) = self.cost_and_gradient_alloc(&point_plus)?;
		let mut result = VectorOps::sub(&grad2, &grad1);
		VectorOps::scale_mut(&mut result, 1.0 / eps);
		Ok(result)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let eps = 1e-8;
		let n = VectorView::len(point);
		let mut gradient = linalg::Vec::<f64>::zeros(n);

		for i in 0..n {
			let mut point_plus = point.clone();
			let mut point_minus = point.clone();
			*VectorOps::get_mut(&mut point_plus, i) += eps;
			*VectorOps::get_mut(&mut point_minus, i) -= eps;

			let f_plus = self.cost(&point_plus)?;
			let f_minus = self.cost(&point_minus)?;
			*VectorOps::get_mut(&mut gradient, i) = (f_plus - f_minus) / (2.0 * eps);
		}

		Ok(gradient)
	}
}
