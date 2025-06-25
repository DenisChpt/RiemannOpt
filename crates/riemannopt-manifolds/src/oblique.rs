//! Oblique manifold OB(n,p)
//!
//! The oblique manifold consists of matrices with unit-norm columns.
//! It can be viewed as a product of spheres, one for each column.

use nalgebra::{DMatrix, Dyn};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::{Scalar, DVector},
};

/// The oblique manifold OB(n,p) of n×p matrices with unit-norm columns.
///
/// # Mathematical Definition
///
/// The oblique manifold is defined as:
/// ```text
/// OB(n,p) = {X ∈ ℝ^{n×p} : diag(X^T X) = 1_p}
/// ```
///
/// This is equivalent to the product manifold S^{n-1} × ... × S^{n-1} (p times).
///
/// # Properties
///
/// - **Dimension**: p(n-1)
/// - **Tangent space**: Vectors with zero inner product with corresponding columns
/// - **Metric**: Euclidean metric restricted to tangent space
///
/// # Applications
///
/// - Sparse coding and dictionary learning
/// - Independent component analysis (ICA)
/// - Blind source separation
/// - Neural network weight normalization
#[derive(Debug, Clone)]
pub struct Oblique {
    n: usize,
    p: usize,
}

impl Oblique {
    /// Create a new oblique manifold OB(n,p).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows (dimension of each column)
    /// * `p` - Number of columns
    ///
    /// # Errors
    ///
    /// Returns an error if n = 0 or p = 0.
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if n == 0 || p == 0 {
            return Err(ManifoldError::invalid_point(
                "Oblique manifold requires n > 0 and p > 0"
            ));
        }
        Ok(Self { n, p })
    }

    /// Get the number of rows.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of columns.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Convert a flat vector to matrix form.
    fn vec_to_mat<T: Scalar>(&self, vec: &DVector<T>) -> DMatrix<T> {
        DMatrix::from_column_slice(self.n, self.p, vec.as_slice())
    }

    /// Convert a matrix to flat vector form.
    fn mat_to_vec<T: Scalar>(&self, mat: &DMatrix<T>) -> DVector<T> {
        DVector::from_column_slice(mat.as_slice())
    }

    /// Normalize columns of a matrix.
    fn normalize_columns<T: Scalar>(&self, mat: &DMatrix<T>) -> DMatrix<T> {
        let mut result = mat.clone();
        for j in 0..self.p {
            let col_norm = result.column(j).norm();
            if col_norm > T::default_epsilon() {
                result.column_mut(j).scale_mut(T::one() / col_norm);
            }
        }
        result
    }
}

impl<T: Scalar> Manifold<T, Dyn> for Oblique {
    fn name(&self) -> &str {
        "Oblique"
    }

    fn dimension(&self) -> usize {
        self.p * (self.n - 1)
    }

    fn ambient_dimension(&self) -> usize {
        self.n * self.p
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tol: T) -> bool {
        if point.len() != self.n * self.p {
            return false;
        }
        
        let mat = self.vec_to_mat(point);
        
        // Check that each column has unit norm
        for j in 0..self.p {
            let col_norm = mat.column(j).norm();
            if Float::abs(col_norm - T::one()) > tol {
                return false;
            }
        }
        
        true
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        tol: T,
    ) -> bool {
        if !self.is_point_on_manifold(point, tol) {
            return false;
        }
        
        if vector.len() != self.n * self.p {
            return false;
        }
        
        let x_mat = self.vec_to_mat(point);
        let v_mat = self.vec_to_mat(vector);
        
        // Check that each tangent column is orthogonal to corresponding point column
        for j in 0..self.p {
            let inner_prod = x_mat.column(j).dot(&v_mat.column(j));
            if Float::abs(inner_prod) > tol {
                return false;
            }
        }
        
        true
    }

    fn project_point(&self, point: &DVector<T>, result: &mut DVector<T>) {
        let ambient_dim = self.n * self.p;
        
        // Ensure result has correct size
        if result.len() != ambient_dim {
            *result = DVector::zeros(ambient_dim);
        }
        
        let mat = self.vec_to_mat(point);
        let normalized = self.normalize_columns(&mat);
        let vec = self.mat_to_vec(&normalized);
        result.copy_from(&vec);
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        let ambient_dim = self.n * self.p;
        if point.len() != ambient_dim || vector.len() != ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Point and vector must have correct dimensions",
                format!("expected: {}, got point: {}, vector: {}", ambient_dim, point.len(), vector.len()),
            ));
        }
        
        // Ensure result has correct size
        if result.len() != ambient_dim {
            *result = DVector::zeros(ambient_dim);
        }
        
        let x_mat = self.vec_to_mat(point);
        let v_mat = self.vec_to_mat(vector);
        let mut proj_mat = v_mat.clone();
        
        // For each column, project to tangent space of corresponding sphere
        for j in 0..self.p {
            let x_col = x_mat.column(j);
            let v_col = v_mat.column(j);
            let inner_prod = x_col.dot(&v_col);
            proj_mat.column_mut(j).axpy(-inner_prod, &x_col, T::one());
        }
        
        let vec = self.mat_to_vec(&proj_mat);
        result.copy_from(&vec);
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        Ok(u.dot(v))
    }

    fn retract(
        &self,
        point: &DVector<T>,
        tangent: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        let ambient_dim = self.n * self.p;
        if point.len() != ambient_dim || tangent.len() != ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Point and tangent must have correct dimensions",
                format!("expected: {}, got point: {}, tangent: {}", ambient_dim, point.len(), tangent.len()),
            ));
        }
        
        // Ensure result has correct size
        if result.len() != ambient_dim {
            *result = DVector::zeros(ambient_dim);
        }
        
        let x_mat = self.vec_to_mat(point);
        let v_mat = self.vec_to_mat(tangent);
        let mut retract_mat = DMatrix::zeros(self.n, self.p);
        
        // Retract each column independently using sphere retraction
        for j in 0..self.p {
            let x_col = x_mat.column(j);
            let v_col = v_mat.column(j);
            
            // Sphere retraction: normalize(x + v)
            let new_col = &x_col + &v_col;
            let norm = new_col.norm();
            
            if norm > T::epsilon() {
                retract_mat.set_column(j, &(new_col / norm));
            } else {
                // If too close to zero, keep original point
                retract_mat.set_column(j, &x_col);
            }
        }
        
        let vec = self.mat_to_vec(&retract_mat);
        result.copy_from(&vec);
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        let ambient_dim = self.n * self.p;
        if point.len() != ambient_dim || other.len() != ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("expected: {}, got point: {}, other: {}", ambient_dim, point.len(), other.len()),
            ));
        }
        
        // Ensure result has correct size
        if result.len() != ambient_dim {
            *result = DVector::zeros(ambient_dim);
        }
        
        // Approximate inverse retraction
        let diff = other - point;
        self.project_tangent(point, &diff, result)
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        euclidean_grad: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        // Simply project to tangent space
        self.project_tangent(point, euclidean_grad, result)
    }

    fn random_point(&self) -> DVector<T> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        let mut mat = DMatrix::zeros(self.n, self.p);
        
        // Generate random columns and normalize
        for j in 0..self.p {
            for i in 0..self.n {
                mat[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        let normalized = self.normalize_columns(&mat);
        self.mat_to_vec(&normalized)
    }

    fn random_tangent(&self, point: &DVector<T>, result: &mut DVector<T>) -> Result<()> {
        let ambient_dim = self.n * self.p;
        if point.len() != ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Point must have correct dimension",
                format!("expected: {}, got: {}", ambient_dim, point.len()),
            ));
        }
        
        // Ensure result has correct size
        if result.len() != ambient_dim {
            *result = DVector::zeros(ambient_dim);
        }
        
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        let mut tangent = DVector::<T>::zeros(self.n * self.p);
        
        for i in 0..(self.n * self.p) {
            tangent[i] = <T as Scalar>::from_f64(normal.sample(&mut rng));
        }
        
        self.project_tangent(point, &tangent, result)
    }

    fn distance(&self, x: &DVector<T>, y: &DVector<T>) -> Result<T> {
        let x_mat = self.vec_to_mat(x);
        let y_mat = self.vec_to_mat(y);
        let mut dist_squared = T::zero();
        
        // Sum of squared distances on each sphere
        for j in 0..self.p {
            let x_col = x_mat.column(j);
            let y_col = y_mat.column(j);
            
            // Geodesic distance on sphere
            let inner_prod = x_col.dot(&y_col);
            let clamped = if inner_prod > T::one() {
                T::one()
            } else if inner_prod < -T::one() {
                -T::one()
            } else {
                inner_prod
            };
            
            let angle = Float::acos(clamped);
            dist_squared += angle * angle;
        }
        
        Ok(Float::sqrt(dist_squared))
    }

    fn parallel_transport(
        &self,
        from_point: &DVector<T>,
        to_point: &DVector<T>,
        vector: &DVector<T>,
        result: &mut DVector<T>,
    ) -> Result<()> {
        let ambient_dim = self.n * self.p;
        if from_point.len() != ambient_dim || to_point.len() != ambient_dim || vector.len() != ambient_dim {
            return Err(ManifoldError::dimension_mismatch(
                "All vectors must have correct dimensions",
                format!("expected: {}", ambient_dim),
            ));
        }
        
        // Ensure result has correct size
        if result.len() != ambient_dim {
            *result = DVector::zeros(ambient_dim);
        }
        
        let x_mat = self.vec_to_mat(from_point);
        let y_mat = self.vec_to_mat(to_point);
        let v_mat = self.vec_to_mat(vector);
        let mut transport_mat = DMatrix::zeros(self.n, self.p);
        
        // Transport each column independently
        for j in 0..self.p {
            let x_col = x_mat.column(j);
            let y_col = y_mat.column(j);
            let v_col = v_mat.column(j);
            
            // Parallel transport on sphere
            let inner_xy = x_col.dot(&y_col);
            
            if Float::abs(inner_xy - T::one()) < T::epsilon() {
                // Points are the same
                transport_mat.set_column(j, &v_col);
            } else if Float::abs(inner_xy + T::one()) < T::epsilon() {
                // Antipodal points - transport is not unique
                transport_mat.set_column(j, &v_col);
            } else {
                // General case: use parallel transport formula for sphere
                let clamped = Float::min(Float::max(inner_xy, -T::one()), T::one());
                let angle = Float::acos(clamped);
                let sin_angle = Float::sin(angle);
                
                if sin_angle > T::epsilon() {
                    let w = (y_col - x_col * inner_xy) / sin_angle;
                    let v_x_dot = v_col.dot(&x_col);
                    let v_w_dot = v_col.dot(&w);
                    let transported = v_col - (x_col * (v_x_dot * inner_xy) - y_col * (v_x_dot * Float::cos(angle)) + w * (v_w_dot * Float::sin(angle))) / sin_angle;
                    transport_mat.set_column(j, &transported);
                } else {
                    transport_mat.set_column(j, &v_col);
                }
            }
        }
        
        let vec = self.mat_to_vec(&transport_mat);
        result.copy_from(&vec);
        Ok(())
    }
}

// MatrixManifold implementation for efficient matrix operations

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_manifold() -> Oblique {
        Oblique::new(5, 3).unwrap()
    }

    #[test]
    fn test_oblique_creation() {
        let manifold = create_test_manifold();
        assert_eq!(manifold.n(), 5);
        assert_eq!(manifold.p(), 3);
        assert_eq!(manifold.n * manifold.p, 15);
        assert_eq!(<Oblique as Manifold<f64, Dyn>>::dimension(&manifold), 12); // 3 * (5 - 1)
    }

    #[test]
    fn test_oblique_projection() {
        let manifold = create_test_manifold();
        
        // Random matrix
        let point = DVector::<f64>::from_vec(vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
        ]);
        let mut projected = DVector::zeros(15);
        <Oblique as Manifold<f64, Dyn>>::project_point(&manifold, &point, &mut projected);
        
        // Check that columns have unit norm
        let mat = manifold.vec_to_mat(&projected);
        for j in 0..3 {
            let col_norm = mat.column(j).norm();
            assert_relative_eq!(col_norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_oblique_tangent_projection() {
        let manifold = create_test_manifold();
        
        let point = <Oblique as Manifold<f64, Dyn>>::random_point(&manifold);
        let vector = DVector::<f64>::from_vec(vec![
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5,
        ]);
        let mut tangent = DVector::zeros(15);
        <Oblique as Manifold<f64, Dyn>>::project_tangent(&manifold, &point, &vector, &mut tangent).unwrap();
        
        // Check orthogonality
        let x_mat = manifold.vec_to_mat(&point);
        let v_mat = manifold.vec_to_mat(&tangent);
        
        for j in 0..3 {
            let inner_prod = x_mat.column(j).dot(&v_mat.column(j));
            assert_relative_eq!(inner_prod, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_oblique_retraction() {
        let manifold = create_test_manifold();
        
        let point = <Oblique as Manifold<f64, Dyn>>::random_point(&manifold);
        let mut tangent = DVector::zeros(15);
        <Oblique as Manifold<f64, Dyn>>::random_tangent(&manifold, &point, &mut tangent).unwrap();
        let scaled_tangent = 0.1 * &tangent;
        let mut retracted = DVector::zeros(15);
        <Oblique as Manifold<f64, Dyn>>::retract(&manifold, &point, &scaled_tangent, &mut retracted).unwrap();
        
        // Check that result is on manifold
        assert!(<Oblique as Manifold<f64, Dyn>>::is_point_on_manifold(&manifold, &retracted, 1e-6));
    }

    #[test]
    fn test_oblique_distance() {
        let manifold = create_test_manifold();
        
        let x = <Oblique as Manifold<f64, Dyn>>::random_point(&manifold);
        let y = <Oblique as Manifold<f64, Dyn>>::random_point(&manifold);
        
        let dist_xy = <Oblique as Manifold<f64, Dyn>>::distance(&manifold, &x, &y).unwrap();
        let dist_yx = <Oblique as Manifold<f64, Dyn>>::distance(&manifold, &y, &x).unwrap();
        
        // Symmetry
        assert_relative_eq!(dist_xy, dist_yx, epsilon = 1e-10);
        
        // Non-negativity
        assert!(dist_xy >= 0.0);
        
        // Self-distance is zero
        let self_dist = <Oblique as Manifold<f64, Dyn>>::distance(&manifold, &x, &x).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-6);
    }
}