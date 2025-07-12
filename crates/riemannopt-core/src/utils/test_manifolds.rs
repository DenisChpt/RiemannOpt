//! Common test manifolds for use in unit tests.
//!
//! This module provides reusable implementations of simple manifolds
//! that can be used across different test modules, reducing code duplication.

#![cfg(any(test, feature = "test-utils"))]

use crate::{
    error::Result,
    manifold::Manifold,
    types::{DVector, Scalar},
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

impl<T: Scalar> Manifold<T> for TestEuclideanManifold {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;
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
        result.copy_from(point);
    }

    fn project_tangent(&self, _point: &Self::Point, vector: &Self::TangentVector, result: &mut Self::TangentVector, ) -> Result<()> {
        result.copy_from(vector);
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        Ok(u.dot(v))
    }

    fn retract(&self, point: &Self::Point, tangent: &Self::TangentVector, result: &mut Self::Point, ) -> Result<()> {
        result.copy_from(&(point + tangent));
        Ok(())
    }

    fn inverse_retract(&self, point: &Self::Point, other: &Self::Point, result: &mut Self::TangentVector, ) -> Result<()> {
        result.copy_from(&(other - point));
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &Self::Point,
        grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        result.copy_from(grad);
        Ok(())
    }

    fn random_point(&self, result: &mut Self::Point) -> Result<()> {
        *result = DVector::from_fn(self.dim, |_, _| {
<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
        });
        Ok(())
    }

    fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector, ) -> Result<()> {
        *result = DVector::from_fn(self.dim, |_, _| {
<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
        });
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, ) -> Result<T> {
        Ok((y - x).norm())
    }
    
    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        result.copy_from(&(tangent * scalar));
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
        result.copy_from(&(v1 + v2));
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

impl<T: Scalar> Manifold<T> for TestSphereManifold {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;
    fn name(&self) -> &str {
        "TestSphere"
    }

    fn dimension(&self) -> usize {
        self.dim - 1 // S^{n-1} has dimension n-1
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tolerance: T) -> bool {
<T as Float>::abs(point.norm_squared() - T::one()) < tolerance
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        tolerance: T,
    ) -> bool {
<T as Float>::abs(point.dot(vector)) < tolerance
    }

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, ) {
        let norm = point.norm();
        if norm > T::epsilon() {
            result.copy_from(&(point / norm));
        } else {
            // Return a default point on the sphere
            *result = DVector::zeros(self.dim);
            result[0] = T::one();
        }
    }

    fn project_tangent(&self, point: &DVector<T>, vector: &DVector<T>, result: &mut DVector<T>, ) -> Result<()> {
        // Project to tangent space: v - <v, p>p
        let inner = point.dot(vector);
        result.copy_from(&(vector - point * inner));
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        Ok(u.dot(v))
    }

    fn retract(&self, point: &DVector<T>, tangent: &DVector<T>, result: &mut DVector<T>) -> Result<()> {
        // Simple projection retraction
        let y = point + tangent;
        self.project_point(&y, result);
        Ok(())
    }

    fn inverse_retract(&self, point: &DVector<T>, other: &DVector<T>, result: &mut DVector<T>) -> Result<()> {
        // Project the difference onto the tangent space
        let diff = other - point;
        self.project_tangent(point, &diff, result)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        grad: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        self.project_tangent(point, grad, result)
    }

    fn random_point(&self, result: &mut Self::Point) -> Result<()> {
        // Generate random point and normalize
        *result = DVector::from_fn(self.dim, |_, _| {
<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
        });
        let norm = result.norm();
        if norm > T::epsilon() {
            *result /= norm;
        } else {
            result[0] = T::one();
        }
        Ok(())
    }

    fn random_tangent(&self, point: &DVector<T>, result: &mut DVector<T>) -> Result<()> {
        let v = DVector::from_fn(self.dim, |_, _| {
<T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0)
        });
        self.project_tangent(point, &v, result)
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, ) -> Result<T> {
        // Geodesic distance on sphere: arccos(<x, y>)
        let inner = x.dot(y);
        let inner = <T as num_traits::Float>::max(inner, -T::one());
        let inner = <T as num_traits::Float>::min(inner, T::one());
        Ok(<T as num_traits::Float>::acos(inner))
    }
    
    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        // For sphere, scaling in tangent space is just scalar multiplication
        result.copy_from(&(tangent * scalar));
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
        let sum = v1 + v2;
        temp.copy_from(&sum);
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

impl<T: Scalar> Manifold<T> for MinimalTestManifold {
    type Point = DVector<T>;
    type TangentVector = DVector<T>;
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

    fn project_point(&self, point: &Self::Point, result: &mut Self::Point, ) {
        result.copy_from(point);
    }

    fn project_tangent(&self, _point: &Self::Point, vector: &Self::TangentVector, result: &mut Self::TangentVector, ) -> Result<()> {
        result.copy_from(vector);
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        Ok(u.dot(v))
    }

    fn retract(&self, point: &Self::Point, tangent: &Self::TangentVector, result: &mut Self::Point, ) -> Result<()> {
        result.copy_from(&(point + tangent));
        Ok(())
    }

    fn inverse_retract(&self, point: &Self::Point, other: &Self::Point, result: &mut Self::TangentVector, ) -> Result<()> {
        result.copy_from(&(other - point));
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        _point: &Self::Point,
        grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        result.copy_from(grad);
        Ok(())
    }

    fn random_point(&self, result: &mut Self::Point) -> Result<()> {
        *result = DVector::zeros(self.dim);
        Ok(())
    }

    fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector, ) -> Result<()> {
        *result = DVector::zeros(self.dim);
        Ok(())
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point, ) -> Result<T> {
        Ok((y - x).norm())
    }
    
    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: T,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        result.copy_from(&(tangent * scalar));
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
        result.copy_from(&(v1 + v2));
        Ok(())
    }
}