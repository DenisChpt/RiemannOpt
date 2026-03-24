//! Common test manifolds for use in unit tests.
//!
//! This module provides reusable implementations of simple manifolds
//! that can be used across different test modules, reducing code duplication.

#![cfg(any(test, feature = "test-utils"))]

use crate::{
	error::Result,
	linalg::{self, LinAlgBackend, VectorOps},
	manifold::Manifold,
	types::Scalar,
};
use num_traits::Float;

/// A simple Euclidean manifold for testing.
///
/// This manifold represents flat Euclidean space where all standard
/// operations are trivial (projections are identity, etc.).
#[derive(Debug, Clone)]
pub struct TestEuclideanManifold {
	dim: usize,
}

impl TestEuclideanManifold {
	pub fn new(dim: usize) -> Self {
		Self { dim }
	}
}

impl<T: Scalar> Manifold<T> for TestEuclideanManifold
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;
	fn name(&self) -> &str {
		"TestEuclidean"
	}

	fn dimension(&self) -> usize {
		self.dim
	}

	fn is_point_on_manifold(&self, _point: &Self::Point, _tolerance: T) -> bool {
		true // All points are valid in Euclidean space
	}

	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		_vector: &Self::TangentVector,
		_tolerance: T,
	) -> bool {
		true // All vectors are valid
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		VectorOps::copy_from(result, point);
	}

	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, vector);
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		Ok(VectorOps::dot(u, v))
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		VectorOps::copy_from(result, point);
		VectorOps::add_assign(result, tangent);
		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, other);
		VectorOps::sub_assign(result, point);
		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		_point: &Self::Point,
		grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, grad);
		Ok(())
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		*result = linalg::Vec::<T>::from_fn(self.dim, |_| {
			<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
		});
		Ok(())
	}

	fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		*result = linalg::Vec::<T>::from_fn(self.dim, |_| {
			<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
		});
		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		let diff = VectorOps::sub(y, x);
		Ok(VectorOps::norm(&diff))
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, tangent);
		VectorOps::scale_mut(result, scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		_point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
		// Temporary buffer for projection if needed
		_temp: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, v1);
		VectorOps::add_assign(result, v2);
		Ok(())
	}
}

/// A simple sphere manifold for testing.
///
/// This represents the unit sphere S^{n-1} in R^n.
#[derive(Debug, Clone)]
pub struct TestSphereManifold {
	dim: usize,
}

impl TestSphereManifold {
	pub fn new(dim: usize) -> Self {
		Self { dim }
	}
}

impl<T: Scalar> Manifold<T> for TestSphereManifold
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;
	fn name(&self) -> &str {
		"TestSphere"
	}

	fn dimension(&self) -> usize {
		self.dim - 1 // S^{n-1} has dimension n-1
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tolerance: T) -> bool {
		<T as Float>::abs(VectorOps::norm_squared(point) - T::one()) < tolerance
	}

	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tolerance: T,
	) -> bool {
		<T as Float>::abs(VectorOps::dot(point, vector)) < tolerance
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let norm = VectorOps::norm(point);
		if norm > T::epsilon() {
			VectorOps::copy_from(result, point);
			VectorOps::scale_mut(result, T::one() / norm);
		} else {
			// Return a default point on the sphere
			*result = linalg::Vec::<T>::zeros(self.dim);
			*VectorOps::get_mut(result, 0) = T::one();
		}
	}

	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Project to tangent space: v - <v, p>p
		let inner = VectorOps::dot(point, vector);
		VectorOps::copy_from(result, vector);
		result.axpy(T::zero() - inner, point, T::one());
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		Ok(VectorOps::dot(u, v))
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		// Simple projection retraction
		VectorOps::copy_from(result, point);
		VectorOps::add_assign(result, tangent);
		let y = result.clone();
		self.project_point(&y, result);
		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// Project the difference onto the tangent space
		let diff = VectorOps::sub(other, point);
		self.project_tangent(point, &diff, result)
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		self.project_tangent(point, grad, result)
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		// Generate random point and normalize
		*result = linalg::Vec::<T>::from_fn(self.dim, |_| {
			<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
		});
		let norm = VectorOps::norm(result);
		if norm > T::epsilon() {
			VectorOps::scale_mut(result, T::one() / norm);
		} else {
			*VectorOps::get_mut(result, 0) = T::one();
		}
		Ok(())
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		let v = linalg::Vec::<T>::from_fn(self.dim, |_| {
			<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
		});
		self.project_tangent(point, &v, result)
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		// Geodesic distance on sphere: arccos(<x, y>)
		let inner = VectorOps::dot(x, y);
		let inner = <T as Float>::max(inner, -T::one());
		let inner = <T as Float>::min(inner, T::one());
		Ok(<T as Float>::acos(inner))
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		// For sphere, scaling in tangent space is just scalar multiplication
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
		// Temporary buffer for projection if needed
		temp: &mut Self::TangentVector,
	) -> Result<()> {
		// Add vectors and project to ensure result is in tangent space
		VectorOps::copy_from(temp, v1);
		VectorOps::add_assign(temp, v2);
		self.project_tangent(point, temp, result)
	}
}

/// A minimal manifold implementation for basic testing.
///
/// This manifold only implements the required methods with trivial behavior.
#[derive(Debug, Clone)]
pub struct MinimalTestManifold {
	dim: usize,
}

impl MinimalTestManifold {
	pub fn new(dim: usize) -> Self {
		Self { dim }
	}
}

impl<T: Scalar> Manifold<T> for MinimalTestManifold
where
	linalg::DefaultBackend: LinAlgBackend<T>,
{
	type Point = linalg::Vec<T>;
	type TangentVector = linalg::Vec<T>;
	fn name(&self) -> &str {
		"MinimalTest"
	}

	fn dimension(&self) -> usize {
		self.dim
	}

	fn is_point_on_manifold(&self, _point: &Self::Point, _tolerance: T) -> bool {
		true
	}

	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		_vector: &Self::TangentVector,
		_tolerance: T,
	) -> bool {
		true
	}

	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		VectorOps::copy_from(result, point);
	}

	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, vector);
		Ok(())
	}

	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
	) -> Result<T> {
		Ok(VectorOps::dot(u, v))
	}

	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
	) -> Result<()> {
		VectorOps::copy_from(result, point);
		VectorOps::add_assign(result, tangent);
		Ok(())
	}

	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, other);
		VectorOps::sub_assign(result, point);
		Ok(())
	}

	fn euclidean_to_riemannian_gradient(
		&self,
		_point: &Self::Point,
		grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, grad);
		Ok(())
	}

	fn random_point(&self, result: &mut Self::Point) -> Result<()> {
		*result = linalg::Vec::<T>::zeros(self.dim);
		Ok(())
	}

	fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) -> Result<()> {
		*result = linalg::Vec::<T>::zeros(self.dim);
		Ok(())
	}

	fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<T> {
		let diff = VectorOps::sub(y, x);
		Ok(VectorOps::norm(&diff))
	}

	fn scale_tangent(
		&self,
		_point: &Self::Point,
		scalar: T,
		tangent: &Self::TangentVector,
		result: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, tangent);
		VectorOps::scale_mut(result, scalar);
		Ok(())
	}

	fn add_tangents(
		&self,
		_point: &Self::Point,
		v1: &Self::TangentVector,
		v2: &Self::TangentVector,
		result: &mut Self::TangentVector,
		// Temporary buffer for projection if needed
		_temp: &mut Self::TangentVector,
	) -> Result<()> {
		VectorOps::copy_from(result, v1);
		VectorOps::add_assign(result, v2);
		Ok(())
	}
}
