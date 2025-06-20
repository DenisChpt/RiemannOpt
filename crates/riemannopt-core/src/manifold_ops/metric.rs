//! Riemannian metric structures and implementations.
//!
//! This module provides various Riemannian metric implementations that can be
//! used to define the geometry of manifolds. A Riemannian metric assigns an
//! inner product to each tangent space, allowing measurement of lengths and angles.
//!
//! # Mathematical Background
//!
//! A Riemannian metric g on a manifold M assigns to each point p  M an inner
//! product g_p on the tangent space T_p M. This allows us to:
//! - Measure lengths of curves
//! - Define angles between tangent vectors
//! - Compute volumes
//! - Define geodesics (shortest paths)
//!
//! The metric tensor g_{ij} in local coordinates provides the components of the metric,
//! and the Christoffel symbols �^k_{ij} encode how the metric changes across the manifold.

use crate::{
    error::{ManifoldError, Result},
    manifold::Point,
    tangent::RiemannianMetric,
    types::{DVector, Scalar},
};
use nalgebra::{allocator::Allocator, DMatrix, DefaultAllocator, Dim, OMatrix, OVector, U1};
use num_traits::Float;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Represents a metric tensor at a point on a manifold.
///
/// The metric tensor is a symmetric positive-definite matrix that
/// encodes the inner product structure at a point.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricTensor<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
{
    /// The metric tensor matrix (symmetric positive-definite)
    pub matrix: OMatrix<T, D, D>,
}

impl<T, D> MetricTensor<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Creates a new metric tensor from a symmetric positive-definite matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not symmetric or not positive-definite.
    pub fn new(matrix: OMatrix<T, D, D>) -> Result<Self> {
        // Check symmetry
        for i in 0..matrix.nrows() {
            for j in i + 1..matrix.ncols() {
                if <T as Float>::abs(matrix[(i, j)] - matrix[(j, i)]) > T::epsilon() {
                    return Err(ManifoldError::numerical_error(
                        "Metric tensor must be symmetric",
                    ));
                }
            }
        }

        // Check positive-definiteness by testing with coordinate vectors and random vectors
        // For dynamic dimensions, we can't use symmetric_eigenvalues directly,
        // so we use a sampling approach to verify positive definiteness
        let dim = matrix.nrows();
        if dim > 0 {
            // First test with coordinate basis vectors - these catch diagonal issues
            for i in 0..dim {
                let mut e_i = OMatrix::<T, D, U1>::zeros_generic(matrix.shape_generic().0, U1);
                e_i[i] = T::one();
                
                // Compute e_i^T * M * e_i = M[i,i]
                let quadratic_form = matrix[(i, i)];
                if quadratic_form <= T::epsilon() {
                    return Err(ManifoldError::numerical_error(
                        "Metric tensor must be positive definite",
                    ));
                }
            }
            
            // Then test with random vectors to catch off-diagonal issues
            let num_tests = (dim * 2).min(20);
            for _ in 0..num_tests {
                let mut v = OMatrix::<T, D, U1>::zeros_generic(matrix.shape_generic().0, U1);
                for i in 0..dim {
                    v[i] = <T as Scalar>::from_f64(rand::random::<f64>() * 2.0 - 1.0);
                }
                let v_norm_sq = v.norm_squared();
                if v_norm_sq > T::epsilon() {
                    // Compute v^T * M * v
                    let mv = &matrix * &v;
                    let quadratic_form = v.dot(&mv);
                    if quadratic_form <= T::epsilon() * v_norm_sq {
                        return Err(ManifoldError::numerical_error(
                            "Metric tensor must be positive definite",
                        ));
                    }
                }
            }
        }

        Ok(Self { matrix })
    }

    /// Creates an identity metric tensor (Euclidean metric).
    pub fn identity(dim: D) -> Self {
        Self {
            matrix: OMatrix::identity_generic(dim, dim),
        }
    }

    /// Computes the inner product between two vectors using this metric.
    pub fn inner_product(&self, u: &OVector<T, D>, v: &OVector<T, D>) -> T {
        let mv = &self.matrix * v;
        u.dot(&mv)
    }

    /// Computes the norm of a vector using this metric.
    pub fn norm(&self, v: &OVector<T, D>) -> T {
        <T as Float>::sqrt(self.inner_product(v, v))
    }

    /// Computes the inverse of the metric tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric is singular.
    pub fn inverse(&self) -> Result<MetricTensor<T, D>> {
        self.matrix
            .clone()
            .try_inverse()
            .map(|inv| MetricTensor { matrix: inv })
            .ok_or_else(|| ManifoldError::numerical_error("Metric tensor is singular"))
    }

    /// Computes the determinant of the metric tensor.
    pub fn determinant(&self) -> T
    where
        D: nalgebra::DimMin<D, Output = D>,
        DefaultAllocator: Allocator<D>,
    {
        self.matrix.determinant()
    }
    
    /// Computes Christoffel symbols using finite differences.
    ///
    /// This method computes the Christoffel symbols of the second kind
    /// by numerically differentiating the metric tensor.
    ///
    /// # Arguments
    ///
    /// * `metric_fn` - Function that computes the metric tensor at a point
    /// * `point` - Point at which to compute the symbols
    /// * `epsilon` - Step size for finite differences
    ///
    /// # Returns
    ///
    /// The Christoffel symbols Γ^k_{ij} at the given point
    pub fn compute_christoffel_symbols<F>(
        metric_fn: F,
        point: &OVector<T, D>,
        epsilon: T,
    ) -> Result<ChristoffelSymbols<T>>
    where
        F: Fn(&OVector<T, D>) -> Result<MetricTensor<T, D>>,
        D: nalgebra::DimName,
    {
        let dim = point.len();
        let metric = metric_fn(point)?;
        let metric_inv = metric.inverse()?;
        
        // Compute metric derivatives using finite differences
        let mut metric_derivs = Vec::with_capacity(dim);
        
        for k in 0..dim {
            // Perturb in k-th direction
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            point_plus[k] += epsilon;
            point_minus[k] -= epsilon;
            
            let metric_plus = metric_fn(&point_plus)?;
            let metric_minus = metric_fn(&point_minus)?;
            
            // Central difference: ∂g/∂x^k ≈ (g(x+ε) - g(x-ε)) / (2ε)
            let deriv = (&metric_plus.matrix - &metric_minus.matrix) / (epsilon * <T as Scalar>::from_f64(2.0));
            
            // Convert to DMatrix for ChristoffelSymbols
            let mut dmatrix_deriv = DMatrix::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    dmatrix_deriv[(i, j)] = deriv[(i, j)];
                }
            }
            metric_derivs.push(dmatrix_deriv);
        }
        
        // Convert metric matrices to DMatrix
        let mut dmatrix_metric = DMatrix::zeros(dim, dim);
        let mut dmatrix_metric_inv = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                dmatrix_metric[(i, j)] = metric.matrix[(i, j)];
                dmatrix_metric_inv[(i, j)] = metric_inv.matrix[(i, j)];
            }
        }
        
        ChristoffelSymbols::from_metric(&dmatrix_metric, &dmatrix_metric_inv, &metric_derivs)
    }
}

/// Canonical (Euclidean) metric.
///
/// This is the standard inner product inherited from the ambient space.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalMetric<T, D> {
    _phantom: PhantomData<(T, D)>,
}

impl<T, D> CanonicalMetric<T, D> {
    /// Creates a new canonical metric.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T, D> Default for CanonicalMetric<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> RiemannianMetric<T, D> for CanonicalMetric<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn inner_product(
        &self,
        _point: &Point<T, D>,
        u: &OVector<T, D>,
        v: &OVector<T, D>,
    ) -> Result<T> {
        Ok(u.dot(v))
    }
}

/// Weighted metric with a diagonal weight matrix.
///
/// This metric scales different components by different weights,
/// useful for problems where dimensions have different scales.
#[derive(Debug, Clone)]
pub struct WeightedMetric<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Diagonal weights (must be positive)
    weights: OVector<T, D>,
}

impl<T, D> WeightedMetric<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new weighted metric with the given diagonal weights.
    ///
    /// # Errors
    ///
    /// Returns an error if any weight is non-positive.
    pub fn new(weights: OVector<T, D>) -> Result<Self> {
        for i in 0..weights.len() {
            if weights[i] <= T::zero() {
                return Err(ManifoldError::numerical_error(
                    "All weights must be positive",
                ));
            }
        }
        Ok(Self { weights })
    }

    /// Creates a weighted metric with uniform weight.
    pub fn uniform(dim: D, weight: T) -> Result<Self> {
        if weight <= T::zero() {
            return Err(ManifoldError::invalid_parameter(
                format!("Weight must be positive, got: {:.2e}", weight.to_f64())
            ));
        }
        Ok(Self {
            weights: OVector::from_element_generic(dim, nalgebra::U1, weight),
        })
    }
}

impl<T, D> RiemannianMetric<T, D> for WeightedMetric<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn inner_product(
        &self,
        _point: &Point<T, D>,
        u: &OVector<T, D>,
        v: &OVector<T, D>,
    ) -> Result<T> {
        let mut result = T::zero();
        for i in 0..u.len() {
            result += self.weights[i] * u[i] * v[i];
        }
        Ok(result)
    }
}

/// Type alias for embedding function
type EmbeddingFn<T, D, E> = Box<dyn Fn(&OVector<T, D>) -> Result<OVector<T, E>> + Send + Sync>;

/// Type alias for Jacobian function
type JacobianFn<T, D, E> = Box<dyn Fn(&OVector<T, D>) -> Result<OMatrix<T, E, D>> + Send + Sync>;

/// Induced metric from an embedding.
///
/// When a manifold is embedded in a higher-dimensional space,
/// it inherits a metric from the ambient space. The induced metric
/// is computed as g = J^T * G * J, where J is the Jacobian of the
/// embedding and G is the metric of the ambient space.
///
/// # Example
/// 
/// For a sphere S^2 embedded in R^3, the induced metric is the
/// restriction of the Euclidean metric to the tangent spaces of S^2.
pub struct InducedMetric<T, D, E>
where
    T: Scalar,
    D: Dim,
    E: Dim,
    DefaultAllocator: Allocator<D> + Allocator<E> + Allocator<D, E> + Allocator<E, D>,
{
    /// The ambient space metric (usually Euclidean)
    ambient_metric: Box<dyn RiemannianMetric<T, E> + Send + Sync>,
    /// Embedding function: M -> N
    embedding: EmbeddingFn<T, D, E>,
    /// Jacobian of the embedding: TM -> TN
    jacobian: JacobianFn<T, D, E>,
    _phantom: PhantomData<T>,
}

impl<T, D, E> Debug for InducedMetric<T, D, E>
where
    T: Scalar,
    D: Dim,
    E: Dim,
    DefaultAllocator: Allocator<D> + Allocator<E> + Allocator<D, E> + Allocator<E, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InducedMetric")
            .field("ambient_metric", &"<dyn RiemannianMetric>")
            .field("embedding", &"<closure>")
            .field("jacobian", &"<closure>")
            .finish()
    }
}

impl<T, D, E> InducedMetric<T, D, E>
where
    T: Scalar,
    D: Dim,
    E: Dim,
    DefaultAllocator: Allocator<D> + Allocator<E> + Allocator<D, E> + Allocator<E, D> + Allocator<D, D>,
{
    /// Creates a new induced metric from an embedding.
    ///
    /// # Arguments
    ///
    /// * `ambient_metric` - The metric of the ambient space
    /// * `embedding` - Function that embeds points from M into N
    /// * `jacobian` - Function that computes the Jacobian of the embedding
    pub fn new(
        ambient_metric: Box<dyn RiemannianMetric<T, E> + Send + Sync>,
        embedding: EmbeddingFn<T, D, E>,
        jacobian: JacobianFn<T, D, E>,
    ) -> Self {
        Self {
            ambient_metric,
            embedding,
            jacobian,
            _phantom: PhantomData,
        }
    }

    /// Creates an induced metric with Euclidean ambient space.
    pub fn euclidean_induced(
        embedding: impl Fn(&OVector<T, D>) -> Result<OVector<T, E>> + Send + Sync + 'static,
        jacobian: impl Fn(&OVector<T, D>) -> Result<OMatrix<T, E, D>> + Send + Sync + 'static,
    ) -> Self {
        Self::new(
            Box::new(CanonicalMetric::new()),
            Box::new(embedding),
            Box::new(jacobian),
        )
    }
}

impl<T, D, E> RiemannianMetric<T, D> for InducedMetric<T, D, E>
where
    T: Scalar,
    D: Dim,
    E: Dim,
    DefaultAllocator: Allocator<D> + Allocator<E> + Allocator<D, E> + Allocator<E, D> + Allocator<D, D>,
{
    fn inner_product(
        &self,
        point: &Point<T, D>,
        u: &OVector<T, D>,
        v: &OVector<T, D>,
    ) -> Result<T> {
        // Compute the embedding point
        let embedded_point = (self.embedding)(point)?;
        
        // Compute the Jacobian at this point
        let jac = (self.jacobian)(point)?;
        
        // Push forward the tangent vectors to the ambient space
        let u_ambient = &jac * u;
        let v_ambient = &jac * v;
        
        // Compute inner product in ambient space
        self.ambient_metric.inner_product(&embedded_point, &u_ambient, &v_ambient)
    }
}


/// Type alias for map function
type MapFn<T, D1, D2> = Box<dyn Fn(&OVector<T, D1>) -> Result<OVector<T, D2>> + Send + Sync>;

/// Type alias for differential function
type DifferentialFn<T, D1, D2> = Box<dyn Fn(&OVector<T, D1>) -> Result<OMatrix<T, D2, D1>> + Send + Sync>;

/// Pullback metric from a map between manifolds.
///
/// Given a smooth map f: M → N between manifolds, the pullback metric
/// on M is defined by pulling back the metric from N.
/// 
/// The pullback metric is computed as: g_M(u, v) = g_N(df(u), df(v))
/// where df is the differential of f.
pub struct PullbackMetric<T, D1, D2>
where
    T: Scalar,
    D1: Dim,
    D2: Dim,
    DefaultAllocator: Allocator<D1> + Allocator<D2> + Allocator<D2, D1>,
{
    /// The map f: M → N
    map: MapFn<T, D1, D2>,
    /// The differential of the map
    differential: DifferentialFn<T, D1, D2>,
    /// The metric on the target manifold
    target_metric: Box<dyn RiemannianMetric<T, D2> + Send + Sync>,
    _phantom: PhantomData<T>,
}

impl<T, D1, D2> Debug for PullbackMetric<T, D1, D2>
where
    T: Scalar,
    D1: Dim,
    D2: Dim,
    DefaultAllocator: Allocator<D1> + Allocator<D2> + Allocator<D2, D1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PullbackMetric")
            .field("map", &"<closure>")
            .field("differential", &"<closure>")
            .field("target_metric", &"<dyn RiemannianMetric>")
            .finish()
    }
}

impl<T, D1, D2> PullbackMetric<T, D1, D2>
where
    T: Scalar,
    D1: Dim,
    D2: Dim,
    DefaultAllocator: Allocator<D1> + Allocator<D2> + Allocator<D2, D1> + Allocator<D1, D1>,
{
    /// Creates a new pullback metric.
    /// 
    /// # Arguments
    /// 
    /// * `map` - The map f: M → N
    /// * `differential` - The differential df of the map
    /// * `target_metric` - The metric on the target manifold N
    pub fn new(
        map: Box<dyn Fn(&OVector<T, D1>) -> Result<OVector<T, D2>> + Send + Sync>,
        differential: Box<dyn Fn(&OVector<T, D1>) -> Result<OMatrix<T, D2, D1>> + Send + Sync>,
        target_metric: Box<dyn RiemannianMetric<T, D2> + Send + Sync>,
    ) -> Self {
        Self {
            map,
            differential,
            target_metric,
            _phantom: PhantomData,
        }
    }
}

impl<T, D1, D2> RiemannianMetric<T, D1> for PullbackMetric<T, D1, D2>
where
    T: Scalar,
    D1: Dim,
    D2: Dim,
    DefaultAllocator: Allocator<D1> + Allocator<D2> + Allocator<D2, D1>,
{
    fn inner_product(
        &self,
        point: &Point<T, D1>,
        u: &OVector<T, D1>,
        v: &OVector<T, D1>,
    ) -> Result<T> {
        // Map the point to the target manifold
        let target_point = (self.map)(point)?;
        
        // Get the differential at this point
        let jac = (self.differential)(point)?;
        
        // Push forward the tangent vectors
        let pushed_u = &jac * u;
        let pushed_v = &jac * v;

        // Compute inner product in the target space using the actual mapped point
        self.target_metric
            .inner_product(&target_point, &pushed_u, &pushed_v)
    }
}

/// Christoffel symbols of the second kind.
///
/// These encode how the basis vectors change as we move along the manifold,
/// and are essential for computing geodesics and parallel transport.
#[derive(Debug, Clone)]
pub struct ChristoffelSymbols<T>
where
    T: Scalar,
{
    /// The symbols �^k_{ij} stored as a 3D array
    /// symbols[k][i][j] = �^k_{ij}
    pub symbols: Vec<DMatrix<T>>,
}

impl<T> ChristoffelSymbols<T>
where
    T: Scalar,
{
    /// Creates Christoffel symbols for a flat metric (all zeros).
    pub fn flat(dim: usize) -> Self {
        Self {
            symbols: vec![DMatrix::zeros(dim, dim); dim],
        }
    }

    /// Computes Christoffel symbols from a metric tensor and its derivatives.
    ///
    /// The formula is:
    /// �^k_{ij} = (1/2) g^{kl} (_i g_{jl} + _j g_{il} - _l g_{ij})
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric tensor g_{ij}
    /// * `metric_inv` - The inverse metric tensor g^{ij}
    /// * `metric_derivs` - Partial derivatives _k g_{ij}
    pub fn from_metric(
        metric: &DMatrix<T>,
        metric_inv: &DMatrix<T>,
        metric_derivs: &[DMatrix<T>],
    ) -> Result<Self> {
        let dim = metric.nrows();
        if metric.ncols() != dim || metric_inv.nrows() != dim || metric_inv.ncols() != dim {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}x{}", dim, dim),
                format!("{}x{}", metric.nrows(), metric.ncols()),
            ));
        }

        if metric_derivs.len() != dim {
            return Err(ManifoldError::dimension_mismatch(
                format!("{} derivatives", dim),
                format!("{} derivatives", metric_derivs.len()),
            ));
        }

        let mut symbols = vec![DMatrix::zeros(dim, dim); dim];

        for k in 0..dim {
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = T::zero();
                    for l in 0..dim {
                        let term1 = metric_derivs[i][(j, l)];
                        let term2 = metric_derivs[j][(i, l)];
                        let term3 = metric_derivs[l][(i, j)];
                        sum += metric_inv[(k, l)] * (term1 + term2 - term3);
                    }
                    symbols[k][(i, j)] = sum * <T as Scalar>::from_f64(0.5);
                }
            }
        }

        Ok(Self { symbols })
    }

    /// Gets the Christoffel symbol Γ^k_{ij}.
    pub fn get(&self, k: usize, i: usize, j: usize) -> T {
        self.symbols[k][(i, j)]
    }
    
    /// Computes the geodesic acceleration given position and velocity.
    ///
    /// The geodesic equation is: d²x^k/dt² = -Γ^k_{ij} (dx^i/dt)(dx^j/dt)
    pub fn geodesic_acceleration(&self, velocity: &DVector<T>) -> DVector<T> {
        let dim = velocity.len();
        let mut acceleration = DVector::zeros(dim);
        
        for k in 0..dim {
            let mut sum = T::zero();
            for i in 0..dim {
                for j in 0..dim {
                    sum += self.get(k, i, j) * velocity[i] * velocity[j];
                }
            }
            acceleration[k] = -sum;
        }
        
        acceleration
    }
}

/// Helper functions for metric computations.
pub struct MetricUtils;

impl MetricUtils {
    /// Computes the metric tensor at a point given a RiemannianMetric.
    ///
    /// This samples the metric by computing inner products of basis vectors.
    pub fn compute_metric_tensor<T, D>(
        metric: &impl RiemannianMetric<T, D>,
        point: &Point<T, D>,
    ) -> Result<MetricTensor<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D> + Allocator<D, D>,
    {
        let dim = point.len();
        let mut matrix =
            OMatrix::<T, D, D>::zeros_generic(point.shape_generic().0, point.shape_generic().0);

        // Compute g_{ij} = <e_i, e_j>
        for i in 0..dim {
            for j in 0..dim {
                let mut ei = OVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                let mut ej = OVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                ei[i] = T::one();
                ej[j] = T::one();

                matrix[(i, j)] = metric.inner_product(point, &ei, &ej)?;
            }
        }

        MetricTensor::new(matrix)
    }

    /// Computes the length of a curve given by a sequence of points.
    pub fn curve_length<T, D>(
        metric: &impl RiemannianMetric<T, D>,
        points: &[Point<T, D>],
    ) -> Result<T>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        if points.len() < 2 {
            return Ok(T::zero());
        }

        let mut length = T::zero();
        for i in 0..points.len() - 1 {
            let tangent = &points[i + 1] - &points[i];
            let midpoint = (&points[i] + &points[i + 1]) * <T as Scalar>::from_f64(0.5);
            length += metric.norm(&midpoint, &tangent)?;
        }

        Ok(length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVector;
    use approx::assert_relative_eq;
    use nalgebra::Dyn;

    #[test]
    fn test_metric_tensor() {
        // Test identity metric
        let metric = MetricTensor::<f64, Dyn>::identity(Dyn(3));
        let u = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0, 0.0]);

        assert_eq!(metric.inner_product(&u, &u), 1.0);
        assert_eq!(metric.inner_product(&u, &v), 0.0);
        assert_eq!(metric.norm(&u), 1.0);

        // Test weighted metric tensor
        let weights = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0, 4.0]));
        let weighted_metric = MetricTensor::new(weights).unwrap();
        assert_eq!(weighted_metric.inner_product(&u, &u), 2.0);
    }

    #[test]
    fn test_canonical_metric() {
        let metric = CanonicalMetric::<f64, Dyn>::new();
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![3.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 4.0, 0.0]);

        assert_eq!(metric.inner_product(&point, &u, &v).unwrap(), 0.0);
        assert_eq!(metric.norm(&point, &u).unwrap(), 3.0);
        assert_eq!(metric.norm(&point, &v).unwrap(), 4.0);
    }

    #[test]
    fn test_weighted_metric() {
        let weights = DVector::from_vec(vec![2.0, 3.0, 4.0]);
        let metric = WeightedMetric::new(weights).unwrap();
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![1.0, 1.0, 1.0]);

        // Inner product should be 2*1 + 3*1 + 4*1 = 9
        assert_eq!(metric.inner_product(&point, &u, &u).unwrap(), 9.0);
        assert_eq!(metric.norm(&point, &u).unwrap(), 3.0);

        // Test orthogonality with weighted metric
        let v = DVector::from_vec(vec![0.0, 2.0, -1.5]);
        // Inner product: 2*0 + 3*2 + 4*(-1.5) = 0
        assert_relative_eq!(
            metric.inner_product(&point, &u, &v).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_uniform_weighted_metric() {
        let metric = WeightedMetric::<f64, Dyn>::uniform(Dyn(3), 2.0).unwrap();
        let point = DVector::zeros(3);
        let u = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        // Inner product should be 2*(1� + 2� + 3�) = 2*14 = 28
        assert_eq!(metric.inner_product(&point, &u, &u).unwrap(), 28.0);
    }

    #[test]
    fn test_christoffel_flat() {
        let symbols = ChristoffelSymbols::<f64>::flat(3);

        // All Christoffel symbols should be zero for flat metric
        for k in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(symbols.get(k, i, j), 0.0);
                }
            }
        }
        
        // Test geodesic acceleration (should be zero for flat space)
        let velocity = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let acceleration = symbols.geodesic_acceleration(&velocity);
        assert_relative_eq!(acceleration.norm(), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_christoffel_computation() {
        use nalgebra::{U2, Vector2};
        
        // Test with a simple diagonal metric that varies with position
        // g(x,y) = diag(1 + x², 1 + y²)
        let metric_fn = |p: &Vector2<f64>| -> Result<MetricTensor<f64, U2>> {
            let mut g = nalgebra::Matrix2::zeros();
            g[(0, 0)] = 1.0 + p[0] * p[0];
            g[(1, 1)] = 1.0 + p[1] * p[1];
            MetricTensor::new(g)
        };
        
        let point = Vector2::new(1.0, 0.5);
        let epsilon = 1e-6;
        
        let symbols = MetricTensor::compute_christoffel_symbols(metric_fn, &point, epsilon).unwrap();
        
        // For this metric, the non-zero Christoffel symbols should be:
        // Γ^0_{00} = x/(1+x²) and Γ^1_{11} = y/(1+y²)
        let expected_gamma_000 = point[0] / (1.0 + point[0] * point[0]);
        let expected_gamma_111 = point[1] / (1.0 + point[1] * point[1]);
        
        assert_relative_eq!(symbols.get(0, 0, 0), expected_gamma_000, epsilon = 1e-4);
        assert_relative_eq!(symbols.get(1, 1, 1), expected_gamma_111, epsilon = 1e-4);
        
        // Other symbols should be approximately zero
        assert_relative_eq!(symbols.get(0, 0, 1), 0.0, epsilon = 1e-4);
        assert_relative_eq!(symbols.get(0, 1, 0), 0.0, epsilon = 1e-4);
        assert_relative_eq!(symbols.get(1, 0, 0), 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_metric_utils() {
        let metric = CanonicalMetric::<f64, Dyn>::new();
        let point = DVector::zeros(3);

        // Compute metric tensor
        let tensor = MetricUtils::compute_metric_tensor(&metric, &point).unwrap();

        // Should be identity matrix
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(tensor.matrix[(i, j)], expected);
            }
        }

        // Test curve length
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0, 0.0]),
        ];

        let length = MetricUtils::curve_length(&metric, &points).unwrap();
        assert_relative_eq!(length, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_metric_errors() {
        // Test non-positive weights
        let weights = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        assert!(WeightedMetric::new(weights).is_err());

        // Test non-symmetric metric tensor
        let mut matrix = DMatrix::<f64>::identity(3, 3);
        matrix[(0, 1)] = 1.0;
        matrix[(1, 0)] = 2.0; // Not symmetric
        assert!(MetricTensor::new(matrix).is_err());

        // Test non-positive definite matrix
        let non_pd = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, -1.0  // Negative eigenvalue makes it indefinite
        ]);
        assert!(MetricTensor::new(non_pd).is_err());

        // Test zero matrix (not positive definite)
        let zero = DMatrix::<f64>::zeros(2, 2);
        assert!(MetricTensor::new(zero).is_err());
    }

    #[test]
    fn test_induced_metric() {
        use nalgebra::{Matrix3x2, Vector2, Vector3};
        
        // Test induced metric for a 2D surface in 3D
        // Example: parameterization of a cylinder
        let embedding = |p: &Vector2<f64>| -> Result<Vector3<f64>> {
            let u = p[0];
            let v = p[1];
            Ok(Vector3::new(u.cos(), u.sin(), v))
        };
        
        let jacobian = |p: &Vector2<f64>| -> Result<Matrix3x2<f64>> {
            let u = p[0];
            // Jacobian matrix J where columns are ∂F/∂u and ∂F/∂v
            Ok(Matrix3x2::from_columns(&[
                Vector3::new(-u.sin(), u.cos(), 0.0),  // ∂F/∂u
                Vector3::new(0.0, 0.0, 1.0),           // ∂F/∂v
            ]))
        };
        
        let metric = InducedMetric::euclidean_induced(embedding, jacobian);
        
        // Test at point (π/2, 1.0)
        let point = Vector2::new(std::f64::consts::PI / 2.0, 1.0);
        let v1 = Vector2::new(1.0, 0.0);
        let v2 = Vector2::new(0.0, 1.0);
        
        // The induced metric on a cylinder should give:
        // <∂/∂u, ∂/∂u> = 1 (circumference direction)
        // <∂/∂v, ∂/∂v> = 1 (height direction)
        // <∂/∂u, ∂/∂v> = 0 (orthogonal)
        
        let g11 = metric.inner_product(&point, &v1, &v1).unwrap();
        let g22 = metric.inner_product(&point, &v2, &v2).unwrap();
        let g12 = metric.inner_product(&point, &v1, &v2).unwrap();
        
        assert_relative_eq!(g11, 1.0, epsilon = 1e-10);
        assert_relative_eq!(g22, 1.0, epsilon = 1e-10);
        assert_relative_eq!(g12, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_pullback_metric() {
        use nalgebra::{Matrix2, Vector2};
        
        // Test pullback metric for a simple linear map
        // f: R² → R², f(x, y) = (2x + y, x - y)
        let map = |p: &Vector2<f64>| -> Result<Vector2<f64>> {
            Ok(Vector2::new(2.0 * p[0] + p[1], p[0] - p[1]))
        };
        
        let differential = |_p: &Vector2<f64>| -> Result<Matrix2<f64>> {
            // Jacobian is constant for linear map:
            // [2  1]
            // [1 -1]
            Ok(Matrix2::new(2.0, 1.0, 1.0, -1.0))
        };
        
        // Use Euclidean metric on target space
        let target_metric = Box::new(CanonicalMetric::<f64, nalgebra::U2>::new());
        
        let pullback = PullbackMetric::new(
            Box::new(map),
            Box::new(differential),
            target_metric,
        );
        
        // Test at origin
        let point = Vector2::zeros();
        let v = Vector2::new(1.0, 0.0);
        let w = Vector2::new(0.0, 1.0);
        
        // Pullback metric: g = J^T * J
        // J^T * J = [2 1]^T * [2 1] = [5  1]
        //           [1 -1]    [1 -1]   [1  2]
        
        let g11 = pullback.inner_product(&point, &v, &v).unwrap();
        let g22 = pullback.inner_product(&point, &w, &w).unwrap();
        let g12 = pullback.inner_product(&point, &v, &w).unwrap();
        
        assert_relative_eq!(g11, 5.0, epsilon = 1e-10);
        assert_relative_eq!(g22, 2.0, epsilon = 1e-10);
        assert_relative_eq!(g12, 1.0, epsilon = 1e-10);
    }
}
