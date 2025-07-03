//! # Oblique Manifold OB(n,p)
//!
//! The Oblique manifold OB(n,p) is the product of p unit spheres in ℝⁿ, consisting
//! of all n×p matrices with unit-norm columns. It provides a natural framework for
//! problems with column-wise normalization constraints.
//!
//! ## Mathematical Definition
//!
//! The Oblique manifold is formally defined as:
//! ```text
//! OB(n,p) = {X ∈ ℝⁿˣᵖ : diag(X^T X) = 1_p}
//!         = S^{n-1} × S^{n-1} × ... × S^{n-1}  (p times)
//! ```
//! where S^{n-1} is the unit sphere in ℝⁿ.
//!
//! Equivalently, X ∈ OB(n,p) if and only if ‖x_j‖ = 1 for all columns j = 1,...,p.
//!
//! ## Geometric Structure
//!
//! ### Tangent Space
//! The tangent space at X ∈ OB(n,p) consists of matrices with columns orthogonal
//! to the corresponding columns of X:
//! ```text
//! T_X OB(n,p) = {V ∈ ℝⁿˣᵖ : diag(X^T V) = 0}
//!             = {V ∈ ℝⁿˣᵖ : x_j^T v_j = 0 for j = 1,...,p}
//! ```
//!
//! ### Riemannian Metric
//! The Oblique manifold inherits the Euclidean metric from ℝⁿˣᵖ:
//! ```text
//! g_X(U, V) = tr(U^T V) = ∑_{j=1}^p u_j^T v_j
//! ```
//!
//! ### Projection Operators
//! - **Manifold projection**: P_{OB}(Y) has columns y_j/‖y_j‖
//! - **Tangent projection**: P_X(V) has columns v_j - (x_j^T v_j)x_j
//!
//! ## Maps and Retractions
//!
//! ### Exponential Map
//! The exponential map acts column-wise using the sphere exponential:
//! ```text
//! exp_X(V) has columns exp_{x_j}(v_j) = cos(‖v_j‖)x_j + sin(‖v_j‖)v_j/‖v_j‖
//! ```
//!
//! ### Logarithmic Map
//! The logarithmic map acts column-wise:
//! ```text
//! log_X(Y) has columns log_{x_j}(y_j) = θ_j/sin(θ_j) · (y_j - cos(θ_j)x_j)
//! ```
//! where θ_j = arccos(x_j^T y_j).
//!
//! ### Retraction
//! The simplest retraction normalizes each column:
//! ```text
//! R_X(V) has columns (x_j + v_j)/‖x_j + v_j‖
//! ```
//!
//! ## Parallel Transport
//!
//! Parallel transport acts independently on each column using sphere transport:
//! ```text
//! Γ_{X→Y}(V) has columns Γ_{x_j→y_j}(v_j)
//! ```
//!
//! ## Geometric Properties
//!
//! - **Dimension**: dim(OB(n,p)) = p(n-1)
//! - **Curvature**: Sectional curvature 0 ≤ K ≤ 1 (from spheres)
//! - **Geodesically complete**: Yes
//! - **Simply connected**: Yes if n ≥ 3; No if n = 2
//! - **Distance**: d²(X,Y) = ∑ᵢ d_S²(xᵢ, yᵢ) where d_S is sphere distance
//!
//! ## Applications
//!
//! 1. **Independent Component Analysis (ICA)**: Unmixing signals
//! 2. **Dictionary Learning**: Sparse coding with normalized atoms
//! 3. **Factor Analysis**: Normalized loading matrices
//! 4. **Direction-of-Arrival**: Array processing in signal processing
//! 5. **Computer Vision**: Normalized feature descriptors
//! 6. **Machine Learning**: Weight normalization in neural networks
//!
//! ## Numerical Considerations
//!
//! This implementation ensures:
//! - **Column-wise operations** for efficiency
//! - **Numerical stability** in normalization (handling zero columns)
//! - **Exact geodesics** through sphere formulas
//! - **Efficient retractions** via simple normalization
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::Oblique;
//! use riemannopt_core::manifold::Manifold;
//! use riemannopt_core::memory::workspace::Workspace;
//! use nalgebra::DMatrix;
//!
//! // Create OB(3,2) - two unit vectors in ℝ³
//! let oblique = Oblique::new(3, 2)?;
//!
//! // Random point with unit-norm columns  
//! let x = oblique.random_point();
//! for j in 0..2 {
//!     assert!((x.column(j).norm() - 1.0).abs() < 1e-14);
//! }
//!
//! // Tangent vector
//! let mut v = DMatrix::zeros(3, 2);
//! let mut workspace = Workspace::<f64>::new();
//! oblique.random_tangent(&x, &mut v, &mut workspace)?;
//!
//! // Verify orthogonality: x_j^T v_j = 0
//! for j in 0..2 {
//!     assert!(x.column(j).dot(&v.column(j)).abs() < 1e-14);
//! }
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```

use nalgebra::DMatrix;
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::Debug;

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    memory::workspace::Workspace,
    types::Scalar,
};

/// The Oblique manifold OB(n,p) = S^{n-1} × ... × S^{n-1} (p times).
///
/// This structure represents the manifold of n×p matrices with unit-norm columns,
/// equipped with the product Riemannian metric from the constituent spheres.
///
/// # Type Parameters
///
/// The manifold is generic over the scalar type T through the Manifold trait.
///
/// # Invariants
///
/// - `n ≥ 1`: Dimension of each sphere must be positive
/// - `p ≥ 1`: Number of columns must be positive
/// - All points X satisfy ‖x_j‖ = 1 for j = 1,...,p
/// - All tangent vectors V at X satisfy x_j^T v_j = 0 for j = 1,...,p
#[derive(Debug, Clone)]
pub struct Oblique {
    /// Dimension of the ambient space (rows)
    n: usize,
    /// Number of unit vectors (columns)
    p: usize,
    /// Numerical tolerance for constraint validation
    tolerance: f64,
}

impl Oblique {
    /// Creates a new Oblique manifold OB(n,p).
    ///
    /// # Arguments
    ///
    /// * `n` - Dimension of each constituent sphere (must be ≥ 1)
    /// * `p` - Number of spheres/columns (must be ≥ 1)
    ///
    /// # Returns
    ///
    /// An Oblique manifold with dimension p(n-1).
    ///
    /// # Errors
    ///
    /// Returns `ManifoldError::InvalidParameter` if n = 0 or p = 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use riemannopt_manifolds::Oblique;
    /// // Create OB(3,2) - two unit vectors in ℝ³
    /// let oblique = Oblique::new(3, 2)?;
    /// assert_eq!(oblique.ambient_dim(), (3, 2));
    /// assert_eq!(oblique.manifold_dim(), 4); // 2*(3-1) = 4
    /// # Ok::<(), riemannopt_core::error::ManifoldError>(())
    /// ```
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if n == 0 || p == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Oblique manifold requires n ≥ 1 and p ≥ 1"
            ));
        }
        Ok(Self { n, p, tolerance: 1e-12 })
    }

    /// Creates an Oblique manifold with custom numerical tolerance.
    ///
    /// # Arguments
    ///
    /// * `n` - Dimension of each sphere
    /// * `p` - Number of columns
    /// * `tolerance` - Numerical tolerance for constraint validation
    pub fn with_tolerance(n: usize, p: usize, tolerance: f64) -> Result<Self> {
        if n == 0 || p == 0 {
            return Err(ManifoldError::invalid_parameter(
                "Oblique manifold requires n ≥ 1 and p ≥ 1"
            ));
        }
        if tolerance <= 0.0 || tolerance >= 1.0 {
            return Err(ManifoldError::invalid_parameter(
                "Tolerance must be in (0, 1)"
            ));
        }
        Ok(Self { n, p, tolerance })
    }

    /// Returns the ambient space dimensions (n, p).
    #[inline]
    pub fn ambient_dim(&self) -> (usize, usize) {
        (self.n, self.p)
    }

    /// Returns the manifold dimension p(n-1).
    #[inline]
    pub fn manifold_dim(&self) -> usize {
        self.p * (self.n - 1)
    }

    /// Validates that a matrix has unit-norm columns.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that ‖x_j‖ = 1 for all columns j = 1,...,p.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix dimensions don't match (n,p)
    /// - `NotOnManifold`: If any column doesn't have unit norm
    pub fn check_point<T: Scalar>(&self, x: &DMatrix<T>) -> Result<()> {
        if x.nrows() != self.n || x.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", x.nrows(), x.ncols())
            ));
        }

        for j in 0..self.p {
            let col_norm = x.column(j).norm();
            if <T as Float>::abs(col_norm - T::one()) > <T as Scalar>::from_f64(self.tolerance) {
                return Err(ManifoldError::invalid_point(format!(
                    "Column {} has norm {} (expected 1.0, tolerance: {})",
                    j, col_norm, self.tolerance
                )));
            }
        }

        Ok(())
    }

    /// Validates that a matrix is in the tangent space at X.
    ///
    /// # Mathematical Check
    ///
    /// Verifies that x_j^T v_j = 0 for all columns j = 1,...,p.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions don't match
    /// - `NotOnManifold`: If X is not on Oblique
    /// - `NotInTangentSpace`: If orthogonality constraints violated
    pub fn check_tangent<T: Scalar>(&self, x: &DMatrix<T>, v: &DMatrix<T>) -> Result<()> {
        self.check_point(x)?;

        if v.nrows() != self.n || v.ncols() != self.p {
            return Err(ManifoldError::dimension_mismatch(
                format!("{}×{}", self.n, self.p),
                format!("{}×{}", v.nrows(), v.ncols())
            ));
        }

        for j in 0..self.p {
            let inner = x.column(j).dot(&v.column(j));
            if <T as Float>::abs(inner) > <T as Scalar>::from_f64(self.tolerance) {
                return Err(ManifoldError::invalid_tangent(format!(
                    "Column {} violates orthogonality: x_j^T v_j = {} (tolerance: {})",
                    j, inner, self.tolerance
                )));
            }
        }

        Ok(())
    }

    /// Normalizes columns of a matrix to unit norm.
    ///
    /// Zero columns are mapped to the first standard basis vector e_1.
    fn normalize_columns<T: Scalar>(
        &self,
        matrix: &DMatrix<T>,
        result: &mut DMatrix<T>,
    ) {
        result.copy_from(matrix);
        
        for j in 0..self.p {
            let col_norm = result.column(j).norm();
            if col_norm > <T as Scalar>::from_f64(1e-16) {
                let mut col_mut = result.column_mut(j);
                col_mut /= col_norm;
            } else {
                // Handle zero columns by setting to e_1
                let mut col_mut = result.column_mut(j);
                col_mut.fill(T::zero());
                if self.n > 0 {
                    col_mut[0] = T::one();
                }
            }
        }
    }

    /// Computes the exponential map exp_X(V).
    ///
    /// # Mathematical Formula
    ///
    /// Acts column-wise using the sphere exponential map:
    /// ```text
    /// (exp_X(V))_j = exp_{x_j}(v_j) = cos(‖v_j‖)x_j + sin(‖v_j‖)v_j/‖v_j‖
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - Point on the Oblique manifold
    /// * `v` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The point exp_X(V) on the manifold.
    pub fn exp_map<T: Scalar>(&self, x: &DMatrix<T>, v: &DMatrix<T>) -> Result<DMatrix<T>> {
        self.check_tangent(x, v)?;
        
        let mut result = DMatrix::zeros(self.n, self.p);
        
        for j in 0..self.p {
            let x_col = x.column(j);
            let v_col = v.column(j);
            let v_norm = v_col.norm();
            
            let mut result_col = result.column_mut(j);
            
            if v_norm < <T as Scalar>::from_f64(1e-16) {
                result_col.copy_from(&x_col);
            } else {
                let cos_norm = <T as Float>::cos(v_norm);
                let sin_norm = <T as Float>::sin(v_norm);
                
                // exp_{x_j}(v_j) = cos(‖v_j‖)x_j + sin(‖v_j‖)v_j/‖v_j‖
                result_col.copy_from(&x_col);
                result_col *= cos_norm;
                result_col.axpy(sin_norm / v_norm, &v_col, T::one());
            }
        }
        
        Ok(result)
    }

    /// Computes the logarithmic map log_X(Y).
    ///
    /// # Mathematical Formula
    ///
    /// Acts column-wise using the sphere logarithmic map:
    /// ```text
    /// (log_X(Y))_j = log_{x_j}(y_j) = θ_j/sin(θ_j) · (y_j - cos(θ_j)x_j)
    /// ```
    /// where θ_j = arccos(x_j^T y_j).
    ///
    /// # Arguments
    ///
    /// * `x` - Point on the Oblique manifold
    /// * `y` - Another point on the Oblique manifold
    ///
    /// # Returns
    ///
    /// The tangent vector log_X(Y) ∈ T_X OB(n,p).
    pub fn log_map<T: Scalar>(&self, x: &DMatrix<T>, y: &DMatrix<T>) -> Result<DMatrix<T>> {
        self.check_point(x)?;
        self.check_point(y)?;
        
        let mut result = DMatrix::zeros(self.n, self.p);
        
        for j in 0..self.p {
            let x_col = x.column(j);
            let y_col = y.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = <T as Float>::max(<T as Float>::min(inner, T::one()), -T::one());
            
            // Check if points are the same
            if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-14) {
                continue; // Result column remains zero
            }
            
            // Check if points are antipodal
            if <T as Float>::abs(clamped + T::one()) < <T as Scalar>::from_f64(1e-14) {
                return Err(ManifoldError::numerical_error(format!(
                    "Cannot compute logarithm: columns {} are antipodal",
                    j
                )));
            }
            
            let theta = <T as Float>::acos(clamped);
            let sin_theta = <T as Float>::sin(theta);
            
            // log_{x_j}(y_j) = θ/sin(θ) · (y_j - cos(θ)x_j)
            let scale = theta / sin_theta;
            let mut result_col = result.column_mut(j);
            result_col.copy_from(&y_col);
            result_col.axpy(-clamped, &x_col, T::one());
            result_col *= scale;
        }
        
        Ok(result)
    }

    /// Computes the geodesic distance between two points.
    ///
    /// # Mathematical Formula
    ///
    /// The distance is the L² norm of column-wise sphere distances:
    /// ```text
    /// d²(X, Y) = ∑_{j=1}^p d_{S^{n-1}}²(x_j, y_j) = ∑_{j=1}^p arccos²(x_j^T y_j)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - First point on the Oblique manifold
    /// * `y` - Second point on the Oblique manifold
    ///
    /// # Returns
    ///
    /// The geodesic distance d(X, Y) ≥ 0.
    pub fn geodesic_distance<T: Scalar>(&self, x: &DMatrix<T>, y: &DMatrix<T>) -> Result<T> {
        self.check_point(x)?;
        self.check_point(y)?;
        
        let mut dist_squared = T::zero();
        
        for j in 0..self.p {
            let inner = x.column(j).dot(&y.column(j));
            let clamped = <T as Float>::max(<T as Float>::min(inner, T::one()), -T::one());
            let angle = <T as Float>::acos(clamped);
            dist_squared = dist_squared + angle * angle;
        }
        
        Ok(<T as Float>::sqrt(dist_squared))
    }

    /// Parallel transports a tangent vector along geodesics.
    ///
    /// # Mathematical Formula
    ///
    /// Transport acts independently on each column using sphere transport.
    ///
    /// # Arguments
    ///
    /// * `x` - Starting point
    /// * `y` - Ending point  
    /// * `v` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The parallel transported vector at y.
    pub fn parallel_transport<T: Scalar>(
        &self,
        x: &DMatrix<T>,
        y: &DMatrix<T>,
        v: &DMatrix<T>,
    ) -> Result<DMatrix<T>> {
        self.check_tangent(x, v)?;
        self.check_point(y)?;
        
        let mut result = DMatrix::zeros(self.n, self.p);
        
        for j in 0..self.p {
            let x_col = x.column(j);
            let y_col = y.column(j);
            let v_col = v.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = <T as Float>::max(<T as Float>::min(inner, T::one()), -T::one());
            
            // Same point - no transport needed
            if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-14) {
                result.column_mut(j).copy_from(&v_col);
                continue;
            }
            
            // Antipodal points - transport is ambiguous but we keep the vector
            if <T as Float>::abs(clamped + T::one()) < <T as Scalar>::from_f64(1e-14) {
                result.column_mut(j).copy_from(&v_col);
                continue;
            }
            
            // General transport formula on sphere
            let factor = (x_col.dot(&v_col) + y_col.dot(&v_col)) / (T::one() + inner);
            let mut result_col = result.column_mut(j);
            result_col.copy_from(&v_col);
            result_col.axpy(-factor, &(&x_col + &y_col), T::one());
        }
        
        Ok(result)
    }
}

impl<T: Scalar> Manifold<T> for Oblique {
    type Point = DMatrix<T>;
    type TangentVector = DMatrix<T>;

    fn name(&self) -> &str {
        "Oblique"
    }

    fn dimension(&self) -> usize {
        self.p * (self.n - 1)
    }

    fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
        if point.nrows() != self.n || point.ncols() != self.p {
            return false;
        }
        
        // Check each column has unit norm
        for j in 0..self.p {
            let col_norm = point.column(j).norm();
            if <T as Float>::abs(col_norm - T::one()) > tol {
                return false;
            }
        }
        
        true
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        tol: T,
    ) -> bool {
        if !self.is_point_on_manifold(point, tol) {
            return false;
        }
        
        if vector.nrows() != self.n || vector.ncols() != self.p {
            return false;
        }
        
        // For each column: x_j^T v_j = 0
        for j in 0..self.p {
            let x_col = point.column(j);
            let v_col = vector.column(j);
            let inner = x_col.dot(&v_col);
            
            if <T as Float>::abs(inner) > tol {
                return false;
            }
        }
        
        true
    }

    fn project_point(
        &self,
        point: &Self::Point,
        result: &mut Self::Point,
        _workspace: &mut Workspace<T>,
    ) {
        self.normalize_columns(point, result);
    }

    fn project_tangent(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        result.copy_from(vector);
        
        // For each column: v_j - (x_j^T v_j) x_j
        for j in 0..self.p {
            let x_col = point.column(j);
            let v_col = vector.column(j);
            let inner = x_col.dot(&v_col);
            
            let mut result_col = result.column_mut(j);
            result_col.axpy(-inner, &x_col, T::one());
        }
        
        Ok(())
    }

    fn inner_product(
        &self,
        _point: &Self::Point,
        u: &Self::TangentVector,
        v: &Self::TangentVector,
    ) -> Result<T> {
        // Sum of column-wise inner products
        let mut total = T::zero();
        for j in 0..self.p {
            total += u.column(j).dot(&v.column(j));
        }
        Ok(total)
    }

    fn retract(
        &self,
        point: &Self::Point,
        tangent: &Self::TangentVector,
        result: &mut Self::Point,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Normalize each column of (X + V)
        result.copy_from(&(point + tangent));
        
        for j in 0..self.p {
            let col_norm = result.column(j).norm();
            if col_norm > T::zero() {
                let mut col_mut = result.column_mut(j);
                col_mut /= col_norm;
            }
        }
        
        Ok(())
    }

    fn inverse_retract(
        &self,
        point: &Self::Point,
        other: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        result.fill(T::zero());
        
        // For each column, compute logarithmic map on sphere
        for j in 0..self.p {
            let x_col = point.column(j);
            let y_col = other.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = <T as Float>::min(<T as Float>::max(inner, -T::one()), T::one());
            
            if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-10) {
                // Points are identical
                continue;
            }
            
            let theta = <T as Float>::acos(clamped);
            let sin_theta = <T as Float>::sin(theta);
            
            if sin_theta > <T as Scalar>::from_f64(1e-10) {
                // v = theta / sin(theta) * (y - cos(theta) * x)
                let scale = theta / sin_theta;
                let mut result_col = result.column_mut(j);
                result_col.copy_from(&y_col);
                result_col.axpy(-clamped, &x_col, T::one());
                result_col *= scale;
            }
        }
        
        // Ensure result is in tangent space
        let result_clone = result.clone();
        self.project_tangent(point, &result_clone, result, workspace)?;
        
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &Self::Point,
        euclidean_grad: &Self::TangentVector,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Project to tangent space
        self.project_tangent(point, euclidean_grad, result, workspace)
    }

    fn random_point(&self) -> Self::Point {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        let mut matrix = DMatrix::<T>::zeros(self.n, self.p);
        
        // Generate random columns and normalize
        for j in 0..self.p {
            for i in 0..self.n {
                matrix[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
            
            let col_norm = matrix.column(j).norm();
            if col_norm > T::zero() {
                let mut col_mut = matrix.column_mut(j);
                col_mut /= col_norm;
            }
        }
        
        matrix
    }

    fn random_tangent(
        &self,
        point: &Self::Point,
        result: &mut Self::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        // Generate random matrix
        for i in 0..self.n {
            for j in 0..self.p {
                result[(i, j)] = <T as Scalar>::from_f64(normal.sample(&mut rng));
            }
        }
        
        // Project to tangent space
        let result_clone = result.clone();
        self.project_tangent(point, &result_clone, result, workspace)?;
        
        Ok(())
    }

    fn parallel_transport(
        &self,
        from: &Self::Point,
        to: &Self::Point,
        vector: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()> {
        result.fill(T::zero());
        
        // Parallel transport each column independently
        for j in 0..self.p {
            let x_col = from.column(j);
            let y_col = to.column(j);
            let v_col = vector.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = <T as Float>::min(<T as Float>::max(inner, -T::one()), T::one());
            
            if <T as Float>::abs(clamped - T::one()) < <T as Scalar>::from_f64(1e-10) {
                // Same point, no transport needed
                result.column_mut(j).copy_from(&v_col);
                continue;
            }
            
            if <T as Float>::abs(clamped + T::one()) < <T as Scalar>::from_f64(1e-10) {
                // Antipodal points
                result.column_mut(j).copy_from(&v_col);
                continue;
            }
            
            // Compute transported vector
            let theta = <T as Float>::acos(clamped);
            let sin_theta = <T as Float>::sin(theta);
            
            // Tangent direction at x towards y
            let mut xi = y_col.clone_owned();
            xi.axpy(-clamped, &x_col, T::one());
            xi /= sin_theta;
            
            // Transport formula
            let v_xi_inner = v_col.dot(&xi);
            let mut result_col = result.column_mut(j);
            result_col.copy_from(&v_col);
            result_col.axpy(
                -sin_theta * v_xi_inner,
                &x_col,
                T::one()
            );
            result_col.axpy(
                (T::one() - <T as Float>::cos(theta)) * v_xi_inner,
                &xi,
                T::one()
            );
        }
        
        Ok(())
    }

    fn distance(
        &self,
        x: &Self::Point,
        y: &Self::Point,
        _workspace: &mut Workspace<T>,
    ) -> Result<T> {
        let mut dist_squared = T::zero();
        
        // Sum of squared distances on each sphere
        for j in 0..self.p {
            let x_col = x.column(j);
            let y_col = y.column(j);
            
            let inner = x_col.dot(&y_col);
            let clamped = <T as Float>::min(<T as Float>::max(inner, -T::one()), T::one());
            let angle = <T as Float>::acos(clamped);
            
            dist_squared += angle * angle;
        }
        
        Ok(<T as Float>::sqrt(dist_squared))
    }

    fn has_exact_exp_log(&self) -> bool {
        true // Oblique has exact exp/log on each sphere
    }

    fn is_flat(&self) -> bool {
        false // Product of spheres with positive curvature
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use riemannopt_core::memory::workspace::Workspace;

    #[test]
    fn test_oblique_creation() {
        let oblique = Oblique::new(3, 4).unwrap();
        assert_eq!(oblique.n, 3);
        assert_eq!(oblique.p, 4);
        assert_eq!(<Oblique as Manifold<f64>>::dimension(&oblique), 8); // 4*(3-1) = 8
        
        // Error cases
        assert!(Oblique::new(0, 4).is_err());
        assert!(Oblique::new(3, 0).is_err());
    }

    #[test]
    fn test_point_on_manifold() {
        let oblique = Oblique::new(3, 2).unwrap();
        
        // Create matrix with unit norm columns
        let point = DMatrix::from_column_slice(3, 2, &[
            1.0, 0.0, 0.0,  // First column: [1, 0, 0]
            0.0, 1.0, 0.0,  // Second column: [0, 1, 0]
        ]);
        
        assert!(oblique.is_point_on_manifold(&point, 1e-10));
        
        // Non-unit column
        let bad_point = DMatrix::from_column_slice(3, 2, &[
            2.0, 0.0, 0.0,  // Norm = 2
            0.0, 1.0, 0.0,
        ]);
        
        assert!(!oblique.is_point_on_manifold(&bad_point, 1e-10));
    }

    #[test]
    fn test_projection() {
        let oblique = Oblique::new(3, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let matrix = DMatrix::from_column_slice(3, 2, &[
            3.0, 0.0, 0.0,
            0.0, 4.0, 0.0,
        ]);
        
        let mut projected = DMatrix::zeros(3, 2);
        oblique.project_point(&matrix, &mut projected, &mut workspace);
        
        assert!(oblique.is_point_on_manifold(&projected, 1e-10));
        
        // Check columns are normalized versions of input
        assert_relative_eq!(projected.column(0).norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(projected.column(1).norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_space() {
        let oblique = Oblique::new(3, 2).unwrap();
        
        // Create tangent vector
        let tangent = DMatrix::from_column_slice(3, 2, &[
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
        ]);
        
        // For first column of point = [1,0,0], tangent [0,1,0] is valid
        // For second column of point = [0,1,0], tangent [-1,0,0] is valid
        let test_point = DMatrix::from_column_slice(3, 2, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);
        
        assert!(oblique.is_vector_in_tangent_space(&test_point, &tangent, 1e-10));
    }

    #[test]
    fn test_retraction() {
        let oblique = Oblique::new(3, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let point = oblique.random_point();
        let mut tangent = DMatrix::zeros(3, 2);
        oblique.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        
        // Scale for small step
        tangent *= 0.1;
        
        let mut new_point = DMatrix::zeros(3, 2);
        oblique.retract(&point, &tangent, &mut new_point, &mut workspace).unwrap();
        
        assert!(oblique.is_point_on_manifold(&new_point, 1e-10));
    }

    #[test]
    fn test_distance() {
        let oblique = Oblique::new(3, 2).unwrap();
        let mut workspace = Workspace::<f64>::new();
        
        let x = DMatrix::from_column_slice(3, 2, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);
        
        let y = DMatrix::from_column_slice(3, 2, &[
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ]);
        
        let dist = oblique.distance(&x, &y, &mut workspace).unwrap();
        
        // Distance should be sqrt(2) * pi/2 (90 degrees on each sphere)
        let expected = (2.0_f64).sqrt() * std::f64::consts::PI / 2.0;
        assert_relative_eq!(dist, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_check_methods() {
        let oblique = Oblique::new(3, 2).unwrap();
        
        // Valid point
        let valid_point = DMatrix::from_column_slice(3, 2, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);
        assert!(oblique.check_point(&valid_point).is_ok());
        
        // Invalid point - non-unit column
        let invalid_point = DMatrix::from_column_slice(3, 2, &[
            2.0, 0.0, 0.0,  // Norm = 2
            0.0, 1.0, 0.0,
        ]);
        assert!(oblique.check_point(&invalid_point).is_err());
        
        // Valid tangent vector
        let tangent = DMatrix::from_column_slice(3, 2, &[
            0.0, 1.0, 0.0,  // Orthogonal to [1,0,0]
            -1.0, 0.0, 0.0, // Orthogonal to [0,1,0]
        ]);
        assert!(oblique.check_tangent(&valid_point, &tangent).is_ok());
    }

    #[test]
    fn test_public_exp_log_maps() {
        let oblique = Oblique::new(3, 2).unwrap();
        
        let x = oblique.random_point();
        let mut workspace = Workspace::<f64>::new();
        let mut v = DMatrix::zeros(3, 2);
        oblique.random_tangent(&x, &mut v, &mut workspace).unwrap();
        v *= 0.5; // Small tangent vector
        
        // Test exp_map
        let y = oblique.exp_map(&x, &v).unwrap();
        assert!(oblique.check_point(&y).is_ok());
        
        // Test log_map
        let v_recovered = oblique.log_map(&x, &y).unwrap();
        assert_relative_eq!(v, v_recovered, epsilon = 1e-10);
        
        // Test geodesic distance
        let dist = oblique.geodesic_distance(&x, &y).unwrap();
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_public_parallel_transport() {
        let oblique = Oblique::new(4, 3).unwrap();
        
        let x = oblique.random_point();
        let y = oblique.random_point();
        let mut workspace = Workspace::<f64>::new();
        let mut v = DMatrix::zeros(4, 3);
        oblique.random_tangent(&x, &mut v, &mut workspace).unwrap();
        
        // Test parallel transport
        let v_transported = oblique.parallel_transport(&x, &y, &v).unwrap();
        assert!(oblique.check_tangent(&y, &v_transported).is_ok());
    }

    #[test]
    fn test_manifold_properties() {
        let oblique = Oblique::new(5, 3).unwrap();
        
        assert_eq!(<Oblique as Manifold<f64>>::name(&oblique), "Oblique");
        assert_eq!(<Oblique as Manifold<f64>>::dimension(&oblique), 12); // 3*(5-1) = 12
        assert_eq!(oblique.ambient_dim(), (5, 3));
        assert_eq!(oblique.manifold_dim(), 12);
        assert!(<Oblique as Manifold<f64>>::has_exact_exp_log(&oblique));
        assert!(!<Oblique as Manifold<f64>>::is_flat(&oblique));
    }

    #[test]
    fn test_custom_tolerance() {
        let oblique = Oblique::with_tolerance(3, 2, 1e-6).unwrap();
        assert_eq!(oblique.tolerance, 1e-6);
        
        // Test invalid tolerances
        assert!(Oblique::with_tolerance(3, 2, 0.0).is_err());
        assert!(Oblique::with_tolerance(3, 2, 1.0).is_err());
    }
}