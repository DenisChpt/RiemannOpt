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
    core::manifold::Manifold,
    types::{DVector, Scalar},
};
use num_traits::Float;

/// Operations for working with tangent spaces.
///
/// This struct provides utility functions for common tangent space operations
/// that work with any manifold implementing the `Manifold` trait.
pub struct TangentSpace;

impl TangentSpace {
    /// Computes the norm of a tangent vector.
    ///
    /// # Arguments
    ///
    /// * `manifold` - The manifold providing the metric
    /// * `point` - The point on the manifold
    /// * `vector` - The tangent vector
    ///
    /// # Returns
    ///
    /// The norm of the vector with respect to the manifold metric
    pub fn norm<T, M>(
        manifold: &M,
        point: &M::Point,
        vector: &M::TangentVector,
    ) -> Result<T>
    where
        T: Scalar,
        M: Manifold<T>,
    {
        let norm_sq = manifold.inner_product(point, vector, vector)?;
        Ok(<T as Float>::sqrt(norm_sq))
    }

    /// Computes the angle between two tangent vectors.
    ///
    /// # Arguments
    ///
    /// * `manifold` - The manifold providing the metric
    /// * `point` - The point on the manifold
    /// * `u` - First tangent vector
    /// * `v` - Second tangent vector
    ///
    /// # Returns
    ///
    /// The angle in radians between u and v
    pub fn angle<T, M>(
        manifold: &M,
        point: &M::Point,
        u: &M::TangentVector,
        v: &M::TangentVector,
    ) -> Result<T>
    where
        T: Scalar,
        M: Manifold<T>,
    {
        let norm_u_sq = manifold.inner_product(point, u, u)?;
        let norm_v_sq = manifold.inner_product(point, v, v)?;
        
        if norm_u_sq < T::epsilon() || norm_v_sq < T::epsilon() {
            return Err(ManifoldError::numerical_error(
                "Cannot compute angle with zero vector"
            ));
        }
        
        let norm_u = <T as Float>::sqrt(norm_u_sq);
        let norm_v = <T as Float>::sqrt(norm_v_sq);
        let inner_uv = manifold.inner_product(point, u, v)?;
        
        let cos_angle = inner_uv / (norm_u * norm_v);
        // Clamp to avoid numerical issues with acos
        let cos_angle = <T as Float>::max(cos_angle, -T::one());
        let cos_angle = <T as Float>::min(cos_angle, T::one());
        
        Ok(<T as Float>::acos(cos_angle))
    }

    /// Checks if two tangent vectors are orthogonal.
    ///
    /// # Arguments
    ///
    /// * `manifold` - The manifold providing the metric
    /// * `point` - The point on the manifold
    /// * `u` - First tangent vector
    /// * `v` - Second tangent vector
    /// * `tol` - Tolerance for orthogonality check
    ///
    /// # Returns
    ///
    /// True if the vectors are orthogonal within tolerance
    pub fn is_orthogonal<T, M>(
        manifold: &M,
        point: &M::Point,
        u: &M::TangentVector,
        v: &M::TangentVector,
        tol: T,
    ) -> Result<bool>
    where
        T: Scalar,
        M: Manifold<T>,
    {
        let ip = manifold.inner_product(point, u, v)?;
        Ok(<T as Float>::abs(ip) < tol)
    }
}

/// Specialized operations for manifolds with vector-based tangent spaces.
///
/// This provides additional functionality when tangent vectors are represented
/// as `DVector<T>` types, allowing for more advanced linear algebra operations.
pub struct VectorTangentSpace;

impl VectorTangentSpace {
    /// Computes the parallel component of a vector along a direction.
    ///
    /// Given vectors u and v, returns the component of u in the direction of v.
    pub fn parallel_component<T, M>(
        manifold: &M,
        point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<DVector<T>>
    where
        T: Scalar,
        M: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
    {
        let v_norm_sq = manifold.inner_product(point, v, v)?;
        if v_norm_sq < T::epsilon() {
            return Err(ManifoldError::numerical_error(
                "Cannot project onto zero vector"
            ));
        }

        let projection_coeff = manifold.inner_product(point, u, v)? / v_norm_sq;
        Ok(v * projection_coeff)
    }

    /// Computes the perpendicular component of a vector relative to a direction.
    ///
    /// Given vectors u and v, returns u minus its component in the direction of v.
    pub fn perpendicular_component<T, M>(
        manifold: &M,
        point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<DVector<T>>
    where
        T: Scalar,
        M: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
    {
        let parallel = Self::parallel_component(manifold, point, u, v)?;
        Ok(u - parallel)
    }

    /// Performs Gram-Schmidt orthogonalization on a set of tangent vectors.
    ///
    /// Takes a set of linearly independent tangent vectors and returns
    /// an orthonormal basis for the subspace they span.
    pub fn gram_schmidt<T, M>(
        manifold: &M,
        point: &DVector<T>,
        vectors: &[DVector<T>],
    ) -> Result<Vec<DVector<T>>>
    where
        T: Scalar,
        M: Manifold<T, Point = DVector<T>, TangentVector = DVector<T>>,
    {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let mut orthonormal = Vec::with_capacity(vectors.len());

        for v in vectors {
            let mut u = v.clone();

            // Subtract projections onto all previous orthonormal vectors
            for orth_vec in &orthonormal {
                let proj = Self::parallel_component(manifold, point, &u, orth_vec)?;
                u -= proj;
            }

            // Normalize the result
            let norm_sq = manifold.inner_product(point, &u, &u)?;
            if norm_sq > T::epsilon() {
                let norm = <T as Float>::sqrt(norm_sq);
                orthonormal.push(u / norm);
            }
        }

        Ok(orthonormal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_manifolds::MinimalTestManifold;
    use approx::assert_relative_eq;

    #[test]
    fn test_tangent_space_angle() {
        let manifold = MinimalTestManifold::new(3);
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![1.0, 1.0, 0.0]);

        let angle = TangentSpace::angle(&manifold, &point, &u, &v).unwrap();
        assert_relative_eq!(angle, std::f64::consts::PI / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_space_orthogonality() {
        let manifold = MinimalTestManifold::new(3);
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0, 0.0]);

        assert!(TangentSpace::is_orthogonal(&manifold, &point, &u, &v, 1e-10).unwrap());
    }

    #[test]
    fn test_vector_projections() {
        let manifold = MinimalTestManifold::new(3);
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        let v = DVector::from_vec(vec![1.0, 0.0, 0.0]);

        // Test parallel component
        let u_parallel = VectorTangentSpace::parallel_component(&manifold, &point, &u, &v).unwrap();
        assert_eq!(u_parallel[0], 1.0);
        assert_eq!(u_parallel[1], 0.0);

        // Test perpendicular component
        let u_perp = VectorTangentSpace::perpendicular_component(&manifold, &point, &u, &v).unwrap();
        assert_eq!(u_perp[0], 0.0);
        assert_eq!(u_perp[1], 1.0);

        // Verify u = u_parallel + u_perp
        assert_relative_eq!(&u, &(u_parallel + u_perp));
    }

    #[test]
    fn test_gram_schmidt() {
        let manifold = MinimalTestManifold::new(3);
        let point = DVector::zeros(3);

        let vectors = vec![
            DVector::from_vec(vec![1.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 1.0]),
            DVector::from_vec(vec![1.0, 0.0, 1.0]),
        ];

        let orthonormal = VectorTangentSpace::gram_schmidt(&manifold, &point, &vectors).unwrap();

        // Check that all vectors are unit length
        for v in &orthonormal {
            let norm_sq = manifold.inner_product(&point, v, v).unwrap();
            assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-10);
        }

        // Check that all pairs are orthogonal
        for i in 0..orthonormal.len() {
            for j in i + 1..orthonormal.len() {
                assert!(TangentSpace::is_orthogonal(
                    &manifold,
                    &point,
                    &orthonormal[i],
                    &orthonormal[j],
                    1e-10
                ).unwrap());
            }
        }
    }
}