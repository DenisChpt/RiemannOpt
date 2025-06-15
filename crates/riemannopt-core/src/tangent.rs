//! Tangent space operations and structures.
//!
//! This module provides infrastructure for working with tangent spaces of
//! Riemannian manifolds. The tangent space T_p M at a point p on a manifold M
//! is a vector space that provides the best linear approximation to M near p.
//!
//! # Mathematical Background
//!
//! For a smooth manifold M:
//! - The tangent space T_p M at point p consists of all tangent vectors at p
//! - Each tangent space is equipped with an inner product from the Riemannian metric
//! - The collection of all tangent spaces forms the tangent bundle TM
//! - Vector fields assign a tangent vector to each point smoothly

use crate::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point},
    types::{DVector, Scalar},
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OVector};
use num_traits::Float;
use std::fmt::Debug;

/// Represents a tangent vector in the tangent space of a manifold.
///
/// This trait provides operations specific to tangent vectors beyond
/// the general vector operations provided by nalgebra.
pub trait TangentVector<T, D>: Clone + Debug + PartialEq
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Returns a zero tangent vector at the given point.
    fn zero(point: &Point<T, D>) -> Self;

    /// Scales this tangent vector by a scalar.
    fn scale(&mut self, scalar: T);

    /// Returns a scaled copy of this tangent vector.
    fn scaled(&self, scalar: T) -> Self {
        let mut result = self.clone();
        result.scale(scalar);
        result
    }

    /// Adds another tangent vector to this one.
    fn add_assign_tangent(&mut self, other: &Self);

    /// Subtracts another tangent vector from this one.
    fn sub_assign_tangent(&mut self, other: &Self);

    /// Computes the negative of this tangent vector.
    fn negate(&mut self);

    /// Returns the negative of this tangent vector.
    fn negated(&self) -> Self {
        let mut result = self.clone();
        result.negate();
        result
    }
}

/// Default implementation of TangentVector for OVector types.
impl<T, D> TangentVector<T, D> for OVector<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn zero(point: &Point<T, D>) -> Self {
        Self::zeros_generic(point.shape_generic().0, nalgebra::U1)
    }

    fn scale(&mut self, scalar: T) {
        *self *= scalar;
    }

    fn add_assign_tangent(&mut self, other: &Self) {
        *self += other;
    }

    fn sub_assign_tangent(&mut self, other: &Self) {
        *self -= other;
    }

    fn negate(&mut self) {
        self.iter_mut().for_each(|x| *x = -*x);
    }
}

/// Represents a Riemannian metric on a manifold.
///
/// The metric defines an inner product on each tangent space,
/// allowing us to measure lengths and angles.
pub trait RiemannianMetric<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Computes the inner product between two tangent vectors at a point.
    ///
    /// # Arguments
    ///
    /// * `point` - The point on the manifold
    /// * `u` - First tangent vector
    /// * `v` - Second tangent vector
    ///
    /// # Returns
    ///
    /// The inner product g_p(u, v)
    fn inner_product(&self, point: &Point<T, D>, u: &OVector<T, D>, v: &OVector<T, D>)
        -> Result<T>;

    /// Computes the norm of a tangent vector.
    ///
    /// This is equivalent to sqrt(inner_product(point, v, v)).
    fn norm(&self, point: &Point<T, D>, v: &OVector<T, D>) -> Result<T> {
        self.inner_product(point, v, v)
            .map(|ip| <T as Float>::sqrt(ip))
    }

    /// Normalizes a tangent vector to unit length.
    ///
    /// # Arguments
    ///
    /// * `point` - The point on the manifold
    /// * `v` - The tangent vector to normalize
    ///
    /// # Returns
    ///
    /// A unit tangent vector in the same direction as v
    fn normalize(&self, point: &Point<T, D>, v: &OVector<T, D>) -> Result<OVector<T, D>> {
        let norm = self.norm(point, v)?;
        if norm < T::epsilon() {
            Err(ManifoldError::numerical_error(
                "Cannot normalize zero vector",
            ))
        } else {
            Ok(v / norm)
        }
    }

    /// Computes the angle between two tangent vectors.
    ///
    /// # Arguments
    ///
    /// * `point` - The point on the manifold
    /// * `u` - First tangent vector
    /// * `v` - Second tangent vector
    ///
    /// # Returns
    ///
    /// The angle in radians between u and v
    fn angle(&self, point: &Point<T, D>, u: &OVector<T, D>, v: &OVector<T, D>) -> Result<T> {
        let norm_u = self.norm(point, u)?;
        let norm_v = self.norm(point, v)?;

        if norm_u < T::epsilon() || norm_v < T::epsilon() {
            return Err(ManifoldError::numerical_error(
                "Cannot compute angle with zero vector",
            ));
        }

        let cos_angle = self.inner_product(point, u, v)? / (norm_u * norm_v);
        // Clamp to avoid numerical issues with acos
        let cos_angle = <T as Float>::max(cos_angle, -T::one());
        let cos_angle = <T as Float>::min(cos_angle, T::one());
        Ok(<T as Float>::acos(cos_angle))
    }

    /// Checks if two tangent vectors are orthogonal.
    ///
    /// # Arguments
    ///
    /// * `point` - The point on the manifold
    /// * `u` - First tangent vector
    /// * `v` - Second tangent vector
    /// * `tol` - Tolerance for orthogonality check
    ///
    /// # Returns
    ///
    /// True if the vectors are orthogonal within tolerance
    fn is_orthogonal(
        &self,
        point: &Point<T, D>,
        u: &OVector<T, D>,
        v: &OVector<T, D>,
        tol: T,
    ) -> Result<bool> {
        let ip = self.inner_product(point, u, v)?;
        Ok(<T as Float>::abs(ip) < tol)
    }
}

/// Represents the tangent bundle of a manifold.
///
/// The tangent bundle TM is the disjoint union of all tangent spaces,
/// forming a manifold of dimension 2 * dim(M).
pub struct TangentBundle<M, T, D>
where
    M: Manifold<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    manifold: M,
    _phantom: std::marker::PhantomData<(T, D)>,
}

impl<M, T, D> TangentBundle<M, T, D>
where
    M: Manifold<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new tangent bundle for the given manifold.
    pub fn new(manifold: M) -> Self {
        Self {
            manifold,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a reference to the base manifold.
    pub fn base_manifold(&self) -> &M {
        &self.manifold
    }

    /// Projects a point in the tangent bundle to the base manifold.
    ///
    /// A point in TM is represented as (p, v) where p is a point on M
    /// and v is a tangent vector at p.
    pub fn base_point<'a>(
        &self,
        bundle_point: &'a (Point<T, D>, OVector<T, D>),
    ) -> &'a Point<T, D> {
        &bundle_point.0
    }

    /// Extracts the tangent vector component from a tangent bundle point.
    pub fn tangent_component<'a>(
        &self,
        bundle_point: &'a (Point<T, D>, OVector<T, D>),
    ) -> &'a OVector<T, D> {
        &bundle_point.1
    }

    /// Checks if a point is in the tangent bundle.
    pub fn is_point_in_bundle(&self, bundle_point: &(Point<T, D>, OVector<T, D>), tol: T) -> bool {
        let (point, vector) = bundle_point;
        self.manifold.is_point_on_manifold(point, tol)
            && self.manifold.is_vector_in_tangent_space(point, vector, tol)
    }
}

/// Operations for working with tangent spaces.
pub struct TangentSpace;

impl TangentSpace {
    /// Computes the parallel component of a vector along a direction.
    ///
    /// Given vectors u and v, returns the component of u in the direction of v.
    pub fn parallel_component<T, D>(
        u: &OVector<T, D>,
        v: &OVector<T, D>,
        metric: &impl RiemannianMetric<T, D>,
        point: &Point<T, D>,
    ) -> Result<OVector<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let v_norm_sq = metric.inner_product(point, v, v)?;
        if v_norm_sq < T::epsilon() {
            return Err(ManifoldError::numerical_error(
                "Cannot project onto zero vector",
            ));
        }

        let projection_coeff = metric.inner_product(point, u, v)? / v_norm_sq;
        Ok(v * projection_coeff)
    }

    /// Computes the perpendicular component of a vector relative to a direction.
    ///
    /// Given vectors u and v, returns u minus its component in the direction of v.
    pub fn perpendicular_component<T, D>(
        u: &OVector<T, D>,
        v: &OVector<T, D>,
        metric: &impl RiemannianMetric<T, D>,
        point: &Point<T, D>,
    ) -> Result<OVector<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        let parallel = Self::parallel_component(u, v, metric, point)?;
        Ok(u - parallel)
    }

    /// Performs Gram-Schmidt orthogonalization on a set of tangent vectors.
    ///
    /// Takes a set of linearly independent tangent vectors and returns
    /// an orthonormal basis for the subspace they span.
    pub fn gram_schmidt<T, D>(
        vectors: &[OVector<T, D>],
        metric: &impl RiemannianMetric<T, D>,
        point: &Point<T, D>,
    ) -> Result<Vec<OVector<T, D>>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let mut orthonormal = Vec::with_capacity(vectors.len());

        for (i, v) in vectors.iter().enumerate() {
            let mut u = v.clone();

            // Subtract projections onto all previous orthonormal vectors
            for orth_vec in orthonormal.iter().take(i) {
                let proj = Self::parallel_component(&u, orth_vec, metric, point)?;
                u -= proj;
            }

            // Normalize the result
            let u_normalized = metric.normalize(point, &u)?;
            orthonormal.push(u_normalized);
        }

        Ok(orthonormal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVector;
    use approx::assert_relative_eq;
    use nalgebra::Dyn;

    /// Simple Euclidean metric for testing
    struct EuclideanMetric;

    impl RiemannianMetric<f64, Dyn> for EuclideanMetric {
        fn inner_product(
            &self,
            _point: &DVector<f64>,
            u: &DVector<f64>,
            v: &DVector<f64>,
        ) -> Result<f64> {
            Ok(u.dot(v))
        }
    }

    #[test]
    fn test_tangent_vector_operations() {
        let _point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let mut v1 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let v2 = DVector::from_vec(vec![0.0, 0.0, 1.0]);

        // Test scaling using the trait method
        <DVector<f64> as TangentVector<f64, Dyn>>::scale(&mut v1, 2.0);
        assert_eq!(v1[1], 2.0);

        // Test addition
        v1.add_assign_tangent(&v2);
        assert_eq!(v1[2], 1.0);

        // Test negation
        let v3 = v2.negated();
        assert_eq!(v3[2], -1.0);
    }

    #[test]
    fn test_euclidean_metric() {
        let metric = EuclideanMetric;
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![3.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 4.0, 0.0]);

        // Test inner product
        let ip = metric.inner_product(&point, &u, &v).unwrap();
        assert_eq!(ip, 0.0); // orthogonal vectors

        // Test norm
        let norm_u = metric.norm(&point, &u).unwrap();
        assert_eq!(norm_u, 3.0);

        let norm_v = metric.norm(&point, &v).unwrap();
        assert_eq!(norm_v, 4.0);

        // Test angle
        let w = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        let angle = metric.angle(&point, &u, &w).unwrap();
        assert_relative_eq!(angle, std::f64::consts::PI / 4.0, epsilon = 1e-10);

        // Test orthogonality
        assert!(metric.is_orthogonal(&point, &u, &v, 1e-10).unwrap());
    }

    #[test]
    fn test_normalize() {
        let metric = EuclideanMetric;
        let point = DVector::zeros(3);
        let v = DVector::from_vec(vec![3.0, 4.0, 0.0]);

        let v_normalized = metric.normalize(&point, &v).unwrap();
        assert_relative_eq!(metric.norm(&point, &v_normalized).unwrap(), 1.0);

        // Check direction is preserved
        let ratio = v[0] / v_normalized[0];
        assert_relative_eq!(v[1] / v_normalized[1], ratio);
    }

    #[test]
    fn test_tangent_space_projections() {
        let metric = EuclideanMetric;
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        let v = DVector::from_vec(vec![1.0, 0.0, 0.0]);

        // Test parallel component
        let u_parallel = TangentSpace::parallel_component(&u, &v, &metric, &point).unwrap();
        assert_eq!(u_parallel[0], 1.0);
        assert_eq!(u_parallel[1], 0.0);

        // Test perpendicular component
        let u_perp = TangentSpace::perpendicular_component(&u, &v, &metric, &point).unwrap();
        assert_eq!(u_perp[0], 0.0);
        assert_eq!(u_perp[1], 1.0);

        // Verify u = u_parallel + u_perp
        assert_relative_eq!(&u, &(u_parallel + u_perp));
    }

    #[test]
    fn test_gram_schmidt() {
        let metric = EuclideanMetric;
        let point = DVector::zeros(3);

        let vectors = vec![
            DVector::from_vec(vec![1.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 1.0]),
            DVector::from_vec(vec![1.0, 0.0, 1.0]),
        ];

        let orthonormal = TangentSpace::gram_schmidt(&vectors, &metric, &point).unwrap();

        // Check that all vectors are unit length
        for v in &orthonormal {
            assert_relative_eq!(metric.norm(&point, v).unwrap(), 1.0, epsilon = 1e-10);
        }

        // Check that all pairs are orthogonal
        for i in 0..orthonormal.len() {
            for j in i + 1..orthonormal.len() {
                assert!(metric
                    .is_orthogonal(&point, &orthonormal[i], &orthonormal[j], 1e-10)
                    .unwrap());
            }
        }
    }

    #[test]
    fn test_tangent_bundle() {
        use crate::test_manifolds::MinimalTestManifold;

        let manifold = MinimalTestManifold::new(3);
        let bundle = TangentBundle::new(manifold);

        let point = DVector::zeros(3);
        let vector = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let bundle_point = (point.clone(), vector.clone());

        assert_eq!(bundle.base_point(&bundle_point), &point);
        assert_eq!(bundle.tangent_component(&bundle_point), &vector);
        assert!(bundle.is_point_in_bundle(&bundle_point, 1e-10));
    }
}

/// Normalize a tangent vector with respect to the manifold metric.
///
/// Returns a unit vector in the same direction, or zero if the input has zero norm.
pub fn normalize<T, M>(manifold: &M, point: &DVector<T>, vector: &DVector<T>) -> Result<DVector<T>>
where
    T: Scalar,
    M: Manifold<T, Dyn>,
{
    let norm_sq = manifold.inner_product(point, vector, vector)?;

    if norm_sq > T::epsilon() {
        let norm = <T as Float>::sqrt(norm_sq);
        Ok(vector / norm)
    } else {
        Ok(DVector::zeros(vector.len()))
    }
}

/// Perform Gram-Schmidt orthogonalization on a set of tangent vectors.
///
/// Given a set of tangent vectors, returns an orthonormal basis for the span
/// of these vectors with respect to the manifold metric.
pub fn gram_schmidt<T, M>(
    manifold: &M,
    point: &DVector<T>,
    vectors: &[DVector<T>],
) -> Result<Vec<DVector<T>>>
where
    T: Scalar,
    M: Manifold<T, Dyn>,
{
    let mut result = Vec::new();

    for v in vectors {
        let mut u = v.clone();

        // Subtract projections onto previous vectors
        for w in &result {
            let proj_coeff =
                manifold.inner_product(point, v, w)? / manifold.inner_product(point, w, w)?;
            u -= w * proj_coeff;
        }

        // Normalize and add to result if non-zero
        let norm_sq = manifold.inner_product(point, &u, &u)?;
        if norm_sq > T::epsilon() {
            let norm = <T as Float>::sqrt(norm_sq);
            result.push(u / norm);
        }
    }

    Ok(result)
}
