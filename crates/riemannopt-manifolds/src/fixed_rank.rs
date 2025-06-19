//! Fixed-rank manifold
//!
//! The manifold of m×n matrices with fixed rank k.

use nalgebra::{DMatrix, DVector, Dyn, SVD};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    types::Scalar,
};

/// The fixed-rank manifold of m×n matrices with rank k.
///
/// # Mathematical Definition
///
/// The fixed-rank manifold is defined as:
/// ```text
/// M_k(m,n) = {X ∈ ℝ^{m×n} : rank(X) = k}
/// ```
///
/// This manifold is typically parametrized using the SVD decomposition:
/// ```text
/// X = USV^T
/// ```
/// where U ∈ St(m,k), S ∈ ℝ^{k×k} is diagonal, and V ∈ St(n,k).
///
/// # Properties
///
/// - **Dimension**: k(m + n - k)
/// - **Tangent space**: Matrices of the form UMV^T + U_⊥NV^T + UN^TV_⊥^T
/// - **Metric**: Euclidean metric restricted to tangent space
///
/// # Applications
///
/// - Low-rank matrix completion
/// - Collaborative filtering
/// - System identification
/// - Model reduction
#[derive(Debug, Clone)]
pub struct FixedRank {
    m: usize,
    n: usize,
    k: usize,
}

/// Representation of a point on the fixed-rank manifold
#[derive(Debug, Clone)]
pub struct FixedRankPoint<T: Scalar> {
    /// Left singular vectors (m × k)
    pub u: DMatrix<T>,
    /// Singular values (k × k diagonal)
    pub s: DVector<T>,
    /// Right singular vectors (n × k)
    pub v: DMatrix<T>,
}

impl<T: Scalar> FixedRankPoint<T> {
    /// Create a new fixed-rank point from factors
    pub fn new(u: DMatrix<T>, s: DVector<T>, v: DMatrix<T>) -> Self {
        Self { u, s, v }
    }

    /// Convert to full matrix representation
    pub fn to_matrix(&self) -> DMatrix<T> {
        let s_mat = DMatrix::from_diagonal(&self.s);
        &self.u * s_mat * self.v.transpose()
    }

    /// Create from full matrix using SVD
    pub fn from_matrix(mat: &DMatrix<T>, k: usize) -> Result<Self> {
        let svd = SVD::new(mat.clone(), true, true);
        
        let u = svd.u.ok_or_else(|| ManifoldError::numerical_error("SVD failed to compute U"))?;
        let v_t = svd.v_t.ok_or_else(|| ManifoldError::numerical_error("SVD failed to compute V^T"))?;
        let s = &svd.singular_values;
        
        // Truncate to rank k
        let u_k = u.columns(0, k).into();
        let s_k = s.rows(0, k).into();
        let v_k = v_t.transpose().columns(0, k).into();
        
        Ok(Self::new(u_k, s_k, v_k))
    }

    /// Convert to vector representation for manifold operations
    pub fn to_vector(&self) -> DVector<T> {
        let m = self.u.nrows();
        let n = self.v.nrows();
        let k = self.s.len();
        
        let mut vec = DVector::zeros(m * k + k + n * k);
        let mut idx = 0;
        
        // Pack U
        for j in 0..k {
            for i in 0..m {
                vec[idx] = self.u[(i, j)];
                idx += 1;
            }
        }
        
        // Pack S
        for i in 0..k {
            vec[idx] = self.s[i];
            idx += 1;
        }
        
        // Pack V
        for j in 0..k {
            for i in 0..n {
                vec[idx] = self.v[(i, j)];
                idx += 1;
            }
        }
        
        vec
    }

    /// Create from vector representation
    pub fn from_vector(vec: &DVector<T>, m: usize, n: usize, k: usize) -> Self {
        let mut idx = 0;
        
        // Unpack U
        let mut u = DMatrix::zeros(m, k);
        for j in 0..k {
            for i in 0..m {
                u[(i, j)] = vec[idx];
                idx += 1;
            }
        }
        
        // Unpack S
        let mut s = DVector::zeros(k);
        for i in 0..k {
            s[i] = vec[idx];
            idx += 1;
        }
        
        // Unpack V
        let mut v = DMatrix::zeros(n, k);
        for j in 0..k {
            for i in 0..n {
                v[(i, j)] = vec[idx];
                idx += 1;
            }
        }
        
        Self::new(u, s, v)
    }
}

impl FixedRank {
    /// Create a new fixed-rank manifold.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows
    /// * `n` - Number of columns
    /// * `k` - Rank (must satisfy k ≤ min(m, n))
    ///
    /// # Errors
    ///
    /// Returns an error if k > min(m, n) or if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(ManifoldError::invalid_point(
                "Fixed-rank manifold requires m > 0, n > 0, and k > 0"
            ));
        }
        
        if k > m.min(n) {
            return Err(ManifoldError::invalid_point(
                format!("Rank k={} cannot exceed min(m={}, n={})", k, m, n)
            ));
        }
        
        Ok(Self { m, n, k })
    }

    /// Get the number of rows
    pub fn m(&self) -> usize {
        self.m
    }

    /// Get the number of columns
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the rank
    pub fn k(&self) -> usize {
        self.k
    }

    /// Project the U and V factors onto the Stiefel manifold
    fn project_factors<T: Scalar>(&self, u: &mut DMatrix<T>, v: &mut DMatrix<T>) {
        // QR decomposition for U
        let qr_u = u.clone().qr();
        *u = qr_u.q();
        
        // QR decomposition for V
        let qr_v = v.clone().qr();
        *v = qr_v.q();
    }
}

impl<T: Scalar> Manifold<T, Dyn> for FixedRank {
    fn name(&self) -> &str {
        "FixedRank"
    }

    fn dimension(&self) -> usize {
        self.k * (self.m + self.n - self.k)
    }

    fn ambient_dimension(&self) -> usize {
        self.m * self.k + self.k + self.n * self.k
    }

    fn is_point_on_manifold(&self, point: &Point<T, Dyn>, tol: T) -> bool {
        if point.len() != <Self as Manifold<T, Dyn>>::ambient_dimension(self) {
            return false;
        }
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        
        // Check that U and V are on Stiefel manifolds
        let u_gram = pt.u.transpose() * &pt.u;
        let v_gram = pt.v.transpose() * &pt.v;
        
        // Check orthogonality
        for i in 0..self.k {
            for j in 0..self.k {
                let u_val = if i == j { u_gram[(i, j)] - T::one() } else { u_gram[(i, j)] };
                let v_val = if i == j { v_gram[(i, j)] - T::one() } else { v_gram[(i, j)] };
                
                if Float::abs(u_val) > tol || Float::abs(v_val) > tol {
                    return false;
                }
            }
        }
        
        // Check that singular values are positive
        for i in 0..self.k {
            if pt.s[i] <= T::zero() {
                return false;
            }
        }
        
        true
    }

    fn project_point(&self, point: &Point<T, Dyn>) -> Point<T, Dyn> {
        let mut pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        
        // Project U and V onto Stiefel manifolds
        self.project_factors(&mut pt.u, &mut pt.v);
        
        // Ensure singular values are positive
        for i in 0..self.k {
            if pt.s[i] < T::default_epsilon() {
                pt.s[i] = T::default_epsilon();
            }
        }
        
        pt.to_vector()
    }

    fn project_tangent(
        &self,
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
    ) -> Result<TangentVector<T, Dyn>> {
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let tangent = FixedRankPoint::from_vector(vector, self.m, self.n, self.k);
        
        // Project U and V components to tangent spaces of Stiefel manifolds
        let u_proj = &tangent.u - &pt.u * (pt.u.transpose() * &tangent.u);
        let v_proj = &tangent.v - &pt.v * (pt.v.transpose() * &tangent.v);
        
        let proj_tangent = FixedRankPoint::new(u_proj, tangent.s.clone(), v_proj);
        
        Ok(proj_tangent.to_vector())
    }

    fn inner_product(
        &self,
        point: &Point<T, Dyn>,
        u: &TangentVector<T, Dyn>,
        v: &TangentVector<T, Dyn>,
    ) -> Result<T> {
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let u_tan = FixedRankPoint::from_vector(u, self.m, self.n, self.k);
        let v_tan = FixedRankPoint::from_vector(v, self.m, self.n, self.k);
        
        // Compute the inner product with scaling by singular values
        let mut inner = T::zero();
        
        // U component
        for i in 0..self.m {
            for j in 0..self.k {
                inner += u_tan.u[(i, j)] * v_tan.u[(i, j)];
            }
        }
        
        // S component (scaled)
        for i in 0..self.k {
            inner += u_tan.s[i] * v_tan.s[i] / pt.s[i];
        }
        
        // V component
        for i in 0..self.n {
            for j in 0..self.k {
                inner += u_tan.v[(i, j)] * v_tan.v[(i, j)];
            }
        }
        
        Ok(inner)
    }

    fn retract(
        &self,
        point: &Point<T, Dyn>,
        tangent: &TangentVector<T, Dyn>,
    ) -> Result<Point<T, Dyn>> {
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let tan = FixedRankPoint::from_vector(tangent, self.m, self.n, self.k);
        
        // Retract using projection
        let new_u = &pt.u + &tan.u;
        let new_s = &pt.s + &tan.s;
        let new_v = &pt.v + &tan.v;
        
        let mut new_pt = FixedRankPoint::new(new_u, new_s, new_v);
        
        // Project factors back to Stiefel
        self.project_factors(&mut new_pt.u, &mut new_pt.v);
        
        // Ensure singular values are positive
        for i in 0..self.k {
            if new_pt.s[i] < T::default_epsilon() {
                new_pt.s[i] = T::default_epsilon();
            }
        }
        
        Ok(new_pt.to_vector())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Point<T, Dyn>,
        euclidean_grad: &TangentVector<T, Dyn>,
    ) -> Result<TangentVector<T, Dyn>> {
        // For the canonical metric, just project to tangent space
        self.project_tangent(point, euclidean_grad)
    }

    fn random_point(&self) -> Point<T, Dyn> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random orthogonal matrices
        let mut u = DMatrix::zeros(self.m, self.k);
        let mut v = DMatrix::zeros(self.n, self.k);
        
        for j in 0..self.k {
            for i in 0..self.m {
                u[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            for i in 0..self.n {
                v[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // Orthogonalize
        let qr_u = u.qr();
        let u_orth = qr_u.q();
        
        let qr_v = v.qr();
        let v_orth = qr_v.q();
        
        // Random positive singular values
        let mut s = DVector::zeros(self.k);
        for i in 0..self.k {
            let val: f64 = normal.sample(&mut rng);
            s[i] = <T as Scalar>::from_f64(val.abs() + 1.0);
        }
        
        let pt = FixedRankPoint::new(u_orth, s, v_orth);
        pt.to_vector()
    }

    fn random_tangent(&self, point: &Point<T, Dyn>) -> Result<TangentVector<T, Dyn>> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        
        // Generate random matrices
        let mut u_tan = DMatrix::zeros(self.m, self.k);
        let mut v_tan = DMatrix::zeros(self.n, self.k);
        let mut s_tan = DVector::zeros(self.k);
        
        for j in 0..self.k {
            for i in 0..self.m {
                u_tan[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            for i in 0..self.n {
                v_tan[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            s_tan[j] = <T as Scalar>::from_f64(normal.sample(&mut rng));
        }
        
        // Project to tangent spaces of Stiefel manifolds
        u_tan = &u_tan - &pt.u * (pt.u.transpose() * &u_tan);
        v_tan = &v_tan - &pt.v * (pt.v.transpose() * &v_tan);
        
        let tangent = FixedRankPoint::new(u_tan, s_tan, v_tan);
        Ok(tangent.to_vector())
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &Point<T, Dyn>,
        vector: &TangentVector<T, Dyn>,
        tol: T,
    ) -> bool {
        if !self.is_point_on_manifold(point, tol) {
            return false;
        }
        
        if vector.len() != <Self as Manifold<T, Dyn>>::ambient_dimension(self) {
            return false;
        }
        
        let pt = FixedRankPoint::from_vector(point, self.m, self.n, self.k);
        let vec = FixedRankPoint::from_vector(vector, self.m, self.n, self.k);
        
        // Check that U and V components are in tangent spaces of Stiefel manifolds
        let u_proj = pt.u.transpose() * &vec.u;
        let v_proj = pt.v.transpose() * &vec.v;
        
        // Check that projections are skew-symmetric
        for i in 0..self.k {
            for j in 0..self.k {
                if Float::abs(u_proj[(i, j)] + u_proj[(j, i)]) > tol {
                    return false;
                }
                if Float::abs(v_proj[(i, j)] + v_proj[(j, i)]) > tol {
                    return false;
                }
            }
        }
        
        true
    }

    fn inverse_retract(
        &self,
        point: &Point<T, Dyn>,
        other: &Point<T, Dyn>,
    ) -> Result<TangentVector<T, Dyn>> {
        // Simple approximation: project the difference
        let diff = other - point;
        self.project_tangent(point, &diff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::OVector;

    fn create_test_manifold() -> FixedRank {
        FixedRank::new(6, 4, 2).unwrap()
    }

    #[test]
    fn test_fixed_rank_creation() {
        let manifold = create_test_manifold();
        assert_eq!(manifold.m(), 6);
        assert_eq!(manifold.n(), 4);
        assert_eq!(manifold.k(), 2);
        assert_eq!(<FixedRank as Manifold<f64, Dyn>>::dimension(&manifold), 16); // 2*(6+4-2)
    }

    #[test]
    fn test_fixed_rank_point_conversion() {
        let _manifold = create_test_manifold();
        
        let u = DMatrix::from_fn(6, 2, |i, j| ((i + j) as f64).sin());
        let s = DVector::from_fn(2, |i, _| (i + 1) as f64);
        let v = DMatrix::from_fn(4, 2, |i, j| ((i * j) as f64).cos());
        
        let point = FixedRankPoint::new(u.clone(), s.clone(), v.clone());
        let vec = point.to_vector();
        let point2 = FixedRankPoint::<f64>::from_vector(&vec, 6, 4, 2);
        
        // Check reconstruction
        assert_relative_eq!(point2.u, u, epsilon = 1e-10);
        assert_relative_eq!(point2.s, s, epsilon = 1e-10);
        assert_relative_eq!(point2.v, v, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_rank_projection() {
        let manifold = create_test_manifold();
        
        let point = <FixedRank as Manifold<f64, Dyn>>::random_point(&manifold);
        let projected = <FixedRank as Manifold<f64, Dyn>>::project_point(&manifold, &point);
        
        assert!(<FixedRank as Manifold<f64, Dyn>>::is_point_on_manifold(&manifold, &projected, 1e-6));
    }

    #[test]
    fn test_fixed_rank_tangent_projection() {
        let manifold = create_test_manifold();
        
        let point = <FixedRank as Manifold<f64, Dyn>>::random_point(&manifold);
        let vector = OVector::<f64, Dyn>::from_vec(vec![0.1; <FixedRank as Manifold<f64, Dyn>>::ambient_dimension(&manifold)]);
        let tangent = <FixedRank as Manifold<f64, Dyn>>::project_tangent(&manifold, &point, &vector).unwrap();
        
        // Check that projection is idempotent
        let tangent2 = <FixedRank as Manifold<f64, Dyn>>::project_tangent(&manifold, &point, &tangent).unwrap();
        assert_relative_eq!(&tangent, &tangent2, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_rank_retraction() {
        let manifold = create_test_manifold();
        
        let point = <FixedRank as Manifold<f64, Dyn>>::random_point(&manifold);
        let tangent = <FixedRank as Manifold<f64, Dyn>>::random_tangent(&manifold, &point).unwrap();
        let retracted = <FixedRank as Manifold<f64, Dyn>>::retract(&manifold, &point, &(0.1 * &tangent)).unwrap();
        
        assert!(<FixedRank as Manifold<f64, Dyn>>::is_point_on_manifold(&manifold, &retracted, 1e-6));
    }
}