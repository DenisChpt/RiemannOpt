//! Symmetric Positive Definite (SPD) manifold SPD(n)
//!
//! The SPD manifold represents the space of n x n symmetric positive definite matrices.
//! This manifold is fundamental in numerous applications involving covariance matrices,
//! kernel matrices, and metric learning. It naturally appears in:
//! - Covariance estimation and regularization
//! - Diffusion tensor imaging (DTI)
//! - Kernel methods and Gaussian processes
//! - Metric learning and distance learning
//! - Signal processing with covariance matrices
//! - Robust statistics and M-estimation

use riemannopt_core::{
    error::{ManifoldError, Result},
    manifold::Manifold,
    types::Scalar,
};
use nalgebra::{DMatrix, DVector, Dyn};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

/// The SPD manifold SPD(n) of n x n symmetric positive definite matrices.
///
/// This manifold represents the space of real symmetric matrices with strictly
/// positive eigenvalues. The manifold structure is given by the affine-invariant
/// Riemannian metric, which provides natural geometric properties.
///
/// # Mathematical Properties
///
/// - **Dimension**: n(n+1)/2 (symmetric matrices with positive spectrum)
/// - **Tangent space**: T_P SPD(n) = Sym(n) (symmetric matrices)
/// - **Riemannian metric**: Affine-invariant metric <U,V>_P = tr(P^{-1}UP^{-1}V)
/// - **Geodesics**: P(t) = P^{1/2} exp(t P^{-1/2} log(P^{-1/2}QP^{-1/2}) P^{-1/2}) P^{1/2}
///
/// # Implementation Details
///
/// We use efficient algorithms avoiding expensive matrix logarithms:
/// - Cholesky decomposition for numerical stability
/// - Bures-Wasserstein metric as approximation when needed
/// - Efficient projections via eigenvalue clamping
///
/// # Applications
///
/// - **Medical imaging**: Diffusion tensor analysis
/// - **Computer vision**: Covariance descriptors, region tracking
/// - **Machine learning**: Kernel matrices, metric learning
/// - **Signal processing**: Covariance matrix estimation
/// - **Finance**: Portfolio optimization with covariance constraints
#[derive(Debug, Clone)]
pub struct SPD {
    /// Matrix dimension (n for n x n matrices)
    n: usize,
    /// Minimum eigenvalue threshold for numerical stability
    min_eigenvalue: f64,
}

impl SPD {
    /// Creates a new SPD manifold SPD(n).
    ///
    /// # Arguments
    /// * `n` - Matrix dimension (must be > 0)
    ///
    /// # Returns
    /// An SPD manifold with intrinsic dimension n(n+1)/2
    ///
    /// # Errors
    /// Returns an error if dimension is invalid
    ///
    /// # Examples
    /// ```
    /// use riemannopt_manifolds::SPD;
    /// 
    /// // Create SPD(3) - 3x3 symmetric positive definite matrices
    /// let spd = SPD::new(3).unwrap();
    /// assert_eq!(spd.matrix_dimension(), 3);
    /// ```
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "SPD manifold requires matrix dimension n > 0",
            ));
        }
        Ok(Self {
            n,
            min_eigenvalue: 1e-12, // Default numerical threshold
        })
    }

    /// Creates an SPD manifold with custom eigenvalue threshold.
    ///
    /// # Arguments
    /// * `n` - Matrix dimension
    /// * `min_eigenvalue` - Minimum eigenvalue for positive definiteness
    pub fn with_min_eigenvalue(n: usize, min_eigenvalue: f64) -> Result<Self> {
        if n == 0 {
            return Err(ManifoldError::invalid_point(
                "SPD manifold requires matrix dimension n > 0",
            ));
        }
        if min_eigenvalue <= 0.0 {
            return Err(ManifoldError::invalid_point(
                "Minimum eigenvalue must be positive",
            ));
        }
        Ok(Self { n, min_eigenvalue })
    }

    /// Returns the matrix dimension (n)
    pub fn matrix_dimension(&self) -> usize {
        self.n
    }

    /// Returns the minimum eigenvalue threshold
    pub fn min_eigenvalue(&self) -> f64 {
        self.min_eigenvalue
    }

    /// Converts a vectorized symmetric matrix to full matrix form.
    ///
    /// The vector represents the upper triangular part in column-major order:
    /// [A11, A12, A22, A13, A23, A33, ...] for matrix A.
    fn vector_to_matrix<T>(&self, vector: &DVector<T>) -> Result<DMatrix<T>>
    where
        T: Scalar,
    {
        let expected_size = self.n * (self.n + 1) / 2;
        if vector.len() != expected_size {
            return Err(ManifoldError::dimension_mismatch(
                format!("Expected vector size {}", expected_size),
                format!("Got vector size {}", vector.len()),
            ));
        }

        let mut matrix = DMatrix::<T>::zeros(self.n, self.n);
        let mut idx = 0;

        // Fill upper triangular part
        for j in 0..self.n {
            for i in 0..=j {
                matrix[(i, j)] = vector[idx];
                if i != j {
                    matrix[(j, i)] = vector[idx]; // Symmetric
                }
                idx += 1;
            }
        }

        Ok(matrix)
    }

    /// Converts a matrix to vectorized symmetric form.
    ///
    /// Extracts the upper triangular part in column-major order.
    fn matrix_to_vector<T>(&self, matrix: &DMatrix<T>) -> DVector<T>
    where
        T: Scalar,
    {
        let size = self.n * (self.n + 1) / 2;
        let mut vector = DVector::<T>::zeros(size);
        let mut idx = 0;

        // Extract upper triangular part
        for j in 0..self.n {
            for i in 0..=j {
                vector[idx] = matrix[(i, j)];
                idx += 1;
            }
        }

        vector
    }

    /// Projects a matrix to the SPD manifold by ensuring positive definiteness.
    ///
    /// Uses eigenvalue decomposition and clamps negative eigenvalues.
    fn project_to_spd<T>(&self, matrix: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        // Ensure symmetry first
        let symmetric = (matrix + matrix.transpose()) * <T as Scalar>::from_f64(0.5);
        
        // Compute eigenvalue decomposition
        let eigen = symmetric.clone().symmetric_eigen();
        let mut eigenvalues = eigen.eigenvalues.clone();
        let eigenvectors = eigen.eigenvectors;

        // Clamp eigenvalues to ensure positive definiteness
        let min_eval = <T as Scalar>::from_f64(self.min_eigenvalue);
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] < min_eval {
                eigenvalues[i] = min_eval;
            }
        }

        // Reconstruct matrix: P = V * diag(lambda) * V^T
        let lambda_matrix = DMatrix::from_diagonal(&eigenvalues);
        &eigenvectors * &lambda_matrix * eigenvectors.transpose()
    }

    /// Checks if a matrix is symmetric positive definite.
    fn is_spd<T>(&self, matrix: &DMatrix<T>, tolerance: T) -> bool
    where
        T: Scalar,
    {
        if matrix.nrows() != self.n || matrix.ncols() != self.n {
            return false;
        }

        // Check symmetry
        for i in 0..self.n {
            for j in 0..self.n {
                if <T as Float>::abs(matrix[(i, j)] - matrix[(j, i)]) > tolerance {
                    return false;
                }
            }
        }

        // Check positive definiteness via Cholesky decomposition
        match matrix.clone().cholesky() {
            Some(_) => true,
            None => false,
        }
    }

    /// Computes the affine-invariant distance between two SPD matrices.
    ///
    /// Distance formula: d(P,Q) = ||log(P^{-1/2} Q P^{-1/2})||_F
    fn affine_invariant_distance<T>(&self, p: &DMatrix<T>, q: &DMatrix<T>) -> T
    where
        T: Scalar,
    {
        // Compute P^{-1/2}
        let p_eigen = p.clone().symmetric_eigen();
        let p_sqrt_inv = {
            let sqrt_inv_eigenvals = p_eigen.eigenvalues.map(|x| T::one() / <T as Float>::sqrt(x));
            &p_eigen.eigenvectors * DMatrix::from_diagonal(&sqrt_inv_eigenvals) * p_eigen.eigenvectors.transpose()
        };

        // Compute P^{-1/2} Q P^{-1/2}
        let middle = &p_sqrt_inv * q * &p_sqrt_inv;

        // Compute matrix logarithm via eigendecomposition
        let middle_eigen = middle.symmetric_eigen();
        // Ensure eigenvalues are positive before taking logarithm
        let log_eigenvals = middle_eigen.eigenvalues.map(|x| {
            let clamped_x = <T as Float>::max(x, T::epsilon());
            <T as Float>::ln(clamped_x)
        });
        let log_middle = &middle_eigen.eigenvectors * DMatrix::from_diagonal(&log_eigenvals) * middle_eigen.eigenvectors.transpose();

        // Return Frobenius norm
        log_middle.norm()
    }

    /// Projects a matrix to the tangent space (symmetric matrices).
    fn project_to_tangent<T>(&self, _point: &DMatrix<T>, vector: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        // Tangent space is all symmetric matrices
        (vector + vector.transpose()) * <T as Scalar>::from_f64(0.5)
    }

    /// Generates a random SPD matrix.
    fn random_spd_matrix<T>(&self) -> DMatrix<T>
    where
        T: Scalar,
    {
        let mut rng = rand::thread_rng();
        
        // Generate random matrix
        let mut random_matrix = DMatrix::<T>::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                let val: f64 = StandardNormal.sample(&mut rng);
                random_matrix[(i, j)] = <T as Scalar>::from_f64(val);
            }
        }

        // Make it SPD: A = B^T B + epsilon * I
        let btb = random_matrix.transpose() * &random_matrix;
        let identity = DMatrix::<T>::identity(self.n, self.n);
        let epsilon = <T as Scalar>::from_f64(self.min_eigenvalue);
        
        btb + identity * epsilon
    }

    /// Generates a random symmetric matrix for tangent space.
    fn random_symmetric_matrix<T>(&self) -> DMatrix<T>
    where
        T: Scalar,
    {
        let mut rng = rand::thread_rng();
        
        // Generate random symmetric matrix
        let mut matrix = DMatrix::<T>::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in i..self.n {
                let val: f64 = StandardNormal.sample(&mut rng);
                let val_t = <T as Scalar>::from_f64(val);
                matrix[(i, j)] = val_t;
                if i != j {
                    matrix[(j, i)] = val_t;
                }
            }
        }
        
        matrix
    }

    /// Exponential map on SPD manifold (approximate for efficiency).
    ///
    /// Uses the formula: exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
    /// For efficiency, we use a simpler retraction in practice.
    fn exponential_map<T>(&self, point: &DMatrix<T>, tangent: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar,
    {
        // For efficiency, use simple retraction: P + V + V^T (if V not symmetric)
        // Then project to SPD
        let symmetric_tangent = self.project_to_tangent(point, tangent);
        let candidate = point + symmetric_tangent;
        self.project_to_spd(&candidate)
    }

    /// Computes the matrix logarithm using eigenvalue decomposition.
    ///
    /// For a symmetric positive definite matrix A = V D V^T where D is diagonal
    /// with positive eigenvalues, log(A) = V log(D) V^T.
    fn matrix_logarithm<T>(&self, matrix: &DMatrix<T>) -> Result<DMatrix<T>>
    where
        T: Scalar,
    {
        // Compute eigenvalue decomposition
        let eigen = matrix.clone().symmetric_eigen();
        
        // Check all eigenvalues are positive
        let min_eigenvalue = <T as Scalar>::from_f64(self.min_eigenvalue);
        let epsilon = T::epsilon();
        
        for &eigenval in eigen.eigenvalues.iter() {
            if eigenval <= min_eigenvalue {
                return Err(ManifoldError::numerical_error(
                    "Matrix logarithm requires positive eigenvalues"
                ));
            }
        }
        
        // Compute log of eigenvalues with numerical stability
        let log_eigenvals = eigen.eigenvalues.map(|x| {
            // Clamp eigenvalues to avoid numerical issues
            let clamped_x = <T as Float>::max(x, epsilon);
            <T as Float>::ln(clamped_x)
        });
        
        // Reconstruct: log(A) = V * diag(log(λ)) * V^T
        let log_diag = DMatrix::from_diagonal(&log_eigenvals);
        let log_matrix = &eigen.eigenvectors * &log_diag * eigen.eigenvectors.transpose();
        
        // Ensure the result is symmetric (numerical errors might break symmetry)
        Ok((log_matrix.clone() + log_matrix.transpose()) * <T as Scalar>::from_f64(0.5))
    }

    /// Computes the logarithmic map (inverse retraction) for the affine-invariant metric.
    ///
    /// The formula is: log_X(Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}
    fn logarithmic_map<T>(&self, point: &DMatrix<T>, other: &DMatrix<T>) -> Result<DMatrix<T>>
    where
        T: Scalar,
    {
        // First check if points are close - use first-order approximation for numerical stability
        let diff = other - point;
        let diff_norm = diff.norm();
        let point_norm = point.norm();
        
        // If points are very close, use first-order approximation to avoid numerical issues
        let relative_distance = diff_norm / point_norm;
        if relative_distance < <T as Scalar>::from_f64(1e-8) {
            // For very close points, log_X(Y) ≈ Y - X (projected to tangent space)
            return Ok(self.project_to_tangent(point, &diff));
        }
        
        // Compute X^{1/2} and X^{-1/2}
        let x_eigen = point.clone().symmetric_eigen();
        
        // Check for positive definiteness
        let min_eigenvalue = <T as Scalar>::from_f64(self.min_eigenvalue);
        let epsilon = T::epsilon();
        
        for &eigenval in x_eigen.eigenvalues.iter() {
            if eigenval <= min_eigenvalue {
                return Err(ManifoldError::invalid_point(
                    "Base point must be positive definite for logarithmic map"
                ));
            }
        }
        
        // Compute square root and inverse square root with stability
        let x_sqrt_eigenvals = x_eigen.eigenvalues.map(|x| {
            let clamped_x = <T as Float>::max(x, epsilon);
            <T as Float>::sqrt(clamped_x)
        });
        let x_sqrt_inv_eigenvals = x_eigen.eigenvalues.map(|x| {
            let clamped_x = <T as Float>::max(x, epsilon);
            T::one() / <T as Float>::sqrt(clamped_x)
        });
        
        let x_sqrt = &x_eigen.eigenvectors * DMatrix::from_diagonal(&x_sqrt_eigenvals) * x_eigen.eigenvectors.transpose();
        let x_sqrt_inv = &x_eigen.eigenvectors * DMatrix::from_diagonal(&x_sqrt_inv_eigenvals) * x_eigen.eigenvectors.transpose();
        
        // Compute X^{-1/2} Y X^{-1/2}
        let middle = &x_sqrt_inv * other * &x_sqrt_inv;
        
        // Ensure middle matrix is symmetric (numerical errors might break symmetry)
        let symmetric_middle = (middle.clone() + middle.transpose()) * <T as Scalar>::from_f64(0.5);
        
        // Compute log of the middle matrix
        let log_middle = self.matrix_logarithm(&symmetric_middle)?;
        
        // Return X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}
        let result = &x_sqrt * &log_middle * &x_sqrt;
        
        // Ensure the final result is symmetric
        Ok((result.clone() + result.transpose()) * <T as Scalar>::from_f64(0.5))
    }
}

impl<T> Manifold<T, Dyn> for SPD
where
    T: Scalar,
{
    fn name(&self) -> &str {
        "SPD"
    }

    fn dimension(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    fn is_point_on_manifold(&self, point: &DVector<T>, tolerance: T) -> bool {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point.len() != expected_dim {
            return false;
        }

        match self.vector_to_matrix(point) {
            Ok(matrix) => self.is_spd(&matrix, tolerance),
            Err(_) => false,
        }
    }

    fn is_vector_in_tangent_space(
        &self,
        _point: &DVector<T>,
        vector: &DVector<T>,
        tolerance: T,
    ) -> bool {
        let expected_dim = self.n * (self.n + 1) / 2;
        if vector.len() != expected_dim {
            return false;
        }

        // Convert to matrix and check if symmetric
        match self.vector_to_matrix(vector) {
            Ok(matrix) => {
                // Check symmetry
                for i in 0..self.n {
                    for j in 0..self.n {
                        if <T as Float>::abs(matrix[(i, j)] - matrix[(j, i)]) > tolerance {
                            return false;
                        }
                    }
                }
                true
            }
            Err(_) => false,
        }
    }

    fn project_point(&self, point: &DVector<T>) -> DVector<T> {
        let expected_dim = self.n * (self.n + 1) / 2;
        let matrix = if point.len() == expected_dim {
            // Try to convert vector to matrix, use safe fallback if needed
            match self.vector_to_matrix(point) {
                Ok(mat) => mat,
                Err(_) => {
                    // Create a positive definite matrix from the vector components
                    let mut mat = DMatrix::<T>::identity(self.n, self.n);
                    // Fill upper triangle with vector data (clamped to reasonable values)
                    let mut idx = 0;
                    for i in 0..self.n {
                        for j in i..self.n {
                            if idx < point.len() {
                                let val = point[idx];
                                // Ensure diagonal elements are positive
                                if i == j {
                                    mat[(i, j)] = <T as Float>::max(<T as Float>::abs(val), T::epsilon());
                                } else {
                                    mat[(i, j)] = val;
                                    mat[(j, i)] = val; // Symmetric
                                }
                                idx += 1;
                            }
                        }
                    }
                    mat
                }
            }
        } else {
            // Handle dimension mismatch by creating appropriately sized identity
            DMatrix::<T>::identity(self.n, self.n)
        };

        let projected_matrix = self.project_to_spd(&matrix);
        self.matrix_to_vector(&projected_matrix)
    }

    fn project_tangent(
        &self,
        point: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point.len() != expected_dim || vector.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Point and vector must have correct dimensions for SPD manifold",
                format!("point: {}, vector: {}", point.len(), vector.len()),
            ));
        }

        let point_matrix = self.vector_to_matrix(point)?;
        let vector_matrix = self.vector_to_matrix(vector)?;

        // Project to tangent space (symmetric matrices)
        let projected_matrix = self.project_to_tangent(&point_matrix, &vector_matrix);
        Ok(self.matrix_to_vector(&projected_matrix))
    }

    fn inner_product(
        &self,
        point: &DVector<T>,
        u: &DVector<T>,
        v: &DVector<T>,
    ) -> Result<T> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point.len() != expected_dim || u.len() != expected_dim || v.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "All vectors must have correct dimensions",
                format!("expected: {}", expected_dim),
            ));
        }

        let point_matrix = self.vector_to_matrix(point)?;
        let u_matrix = self.vector_to_matrix(u)?;
        let v_matrix = self.vector_to_matrix(v)?;

        // Affine-invariant metric: <U,V>_P = tr(P^{-1} U P^{-1} V)
        let p_inv = point_matrix.clone().try_inverse().ok_or_else(|| {
            ManifoldError::numerical_error("Matrix is not invertible")
        })?;

        let result = (&p_inv * &u_matrix * &p_inv * &v_matrix).trace();
        Ok(result)
    }

    fn retract(&self, point: &DVector<T>, tangent: &DVector<T>) -> Result<DVector<T>> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point.len() != expected_dim || tangent.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Point and tangent must have correct dimensions",
                format!("point: {}, tangent: {}", point.len(), tangent.len()),
            ));
        }

        let point_matrix = self.vector_to_matrix(point)?;
        let tangent_matrix = self.vector_to_matrix(tangent)?;

        // Use exponential map retraction
        let retracted_matrix = self.exponential_map(&point_matrix, &tangent_matrix);
        Ok(self.matrix_to_vector(&retracted_matrix))
    }

    fn inverse_retract(
        &self,
        point: &DVector<T>,
        other: &DVector<T>,
    ) -> Result<DVector<T>> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point.len() != expected_dim || other.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("point: {}, other: {}", point.len(), other.len()),
            ));
        }

        let point_matrix = self.vector_to_matrix(point)?;
        let other_matrix = self.vector_to_matrix(other)?;

        // Use the proper logarithmic map for affine-invariant metric
        let tangent_matrix = self.logarithmic_map(&point_matrix, &other_matrix)?;
        
        // The result is already in the tangent space (symmetric matrix)
        Ok(self.matrix_to_vector(&tangent_matrix))
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<T>,
        grad: &DVector<T>,
    ) -> Result<DVector<T>> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point.len() != expected_dim || grad.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Point and gradient must have correct dimensions",
                format!("point: {}, grad: {}", point.len(), grad.len()),
            ));
        }

        let point_matrix = self.vector_to_matrix(point)?;
        let grad_matrix = self.vector_to_matrix(grad)?;

        // Convert Euclidean gradient to Riemannian gradient
        // For affine-invariant metric: grad_R = P * grad_E * P
        let riemannian_grad_matrix = &point_matrix * &grad_matrix * &point_matrix;
        let symmetric_grad = self.project_to_tangent(&point_matrix, &riemannian_grad_matrix);
        
        Ok(self.matrix_to_vector(&symmetric_grad))
    }

    fn random_point(&self) -> DVector<T> {
        let matrix = self.random_spd_matrix();
        self.matrix_to_vector(&matrix)
    }

    fn random_tangent(&self, point: &DVector<T>) -> Result<DVector<T>> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Point must have correct dimensions",
                format!("expected: {}, actual: {}", expected_dim, point.len()),
            ));
        }

        let symmetric_matrix = self.random_symmetric_matrix();
        Ok(self.matrix_to_vector(&symmetric_matrix))
    }

    fn has_exact_exp_log(&self) -> bool {
        true // We now implement exact logarithmic map
    }

    fn parallel_transport(
        &self,
        from: &DVector<T>,
        to: &DVector<T>,
        vector: &DVector<T>,
    ) -> Result<DVector<T>> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if from.len() != expected_dim || to.len() != expected_dim || vector.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "All vectors must have correct dimensions",
                format!("expected: {}", expected_dim),
            ));
        }

        let x_matrix = self.vector_to_matrix(from)?;
        let y_matrix = self.vector_to_matrix(to)?;
        let v_matrix = self.vector_to_matrix(vector)?;

        // Implement the correct parallel transport for the affine-invariant metric
        // P_{X→Y}(V) = Y^{1/2} (Y^{-1/2} X Y^{-1/2})^{1/2} X^{-1/2} V X^{-1/2} (Y^{-1/2} X Y^{-1/2})^{1/2} Y^{1/2}

        // Compute Y^{1/2} and Y^{-1/2}
        let y_eigen = y_matrix.clone().symmetric_eigen();
        let y_sqrt_eigenvals = y_eigen.eigenvalues.map(|x| <T as Float>::sqrt(x));
        let y_sqrt_inv_eigenvals = y_eigen.eigenvalues.map(|x| T::one() / <T as Float>::sqrt(x));
        
        let y_sqrt = &y_eigen.eigenvectors * DMatrix::from_diagonal(&y_sqrt_eigenvals) * y_eigen.eigenvectors.transpose();
        let y_sqrt_inv = &y_eigen.eigenvectors * DMatrix::from_diagonal(&y_sqrt_inv_eigenvals) * y_eigen.eigenvectors.transpose();

        // Compute X^{-1/2}
        let x_eigen = x_matrix.clone().symmetric_eigen();
        let x_sqrt_inv_eigenvals = x_eigen.eigenvalues.map(|x| T::one() / <T as Float>::sqrt(x));
        let x_sqrt_inv = &x_eigen.eigenvectors * DMatrix::from_diagonal(&x_sqrt_inv_eigenvals) * x_eigen.eigenvectors.transpose();

        // Compute A = Y^{-1/2} X Y^{-1/2}
        let a = &y_sqrt_inv * &x_matrix * &y_sqrt_inv;

        // Compute A^{1/2} = (Y^{-1/2} X Y^{-1/2})^{1/2}
        let a_eigen = a.symmetric_eigen();
        let a_sqrt_eigenvals = a_eigen.eigenvalues.map(|x| {
            let clamped_x = <T as Float>::max(x, T::epsilon());
            <T as Float>::sqrt(clamped_x)
        });
        let a_sqrt = &a_eigen.eigenvectors * DMatrix::from_diagonal(&a_sqrt_eigenvals) * a_eigen.eigenvectors.transpose();

        // Compute the transported vector
        // P_{X→Y}(V) = Y^{1/2} A^{1/2} X^{-1/2} V X^{-1/2} A^{1/2} Y^{1/2}
        let temp1 = &x_sqrt_inv * &v_matrix * &x_sqrt_inv;
        let temp2 = &a_sqrt * &temp1 * &a_sqrt;
        let transported_matrix = &y_sqrt * &temp2 * &y_sqrt;

        // Ensure symmetry of the result
        let symmetric_transported = (&transported_matrix + transported_matrix.transpose()) * <T as Scalar>::from_f64(0.5);
        
        Ok(self.matrix_to_vector(&symmetric_transported))
    }

    fn distance(&self, point1: &DVector<T>, point2: &DVector<T>) -> Result<T> {
        let expected_dim = self.n * (self.n + 1) / 2;
        if point1.len() != expected_dim || point2.len() != expected_dim {
            return Err(ManifoldError::dimension_mismatch(
                "Points must have correct dimensions",
                format!("point1: {}, point2: {}", point1.len(), point2.len()),
            ));
        }

        let matrix1 = self.vector_to_matrix(point1)?;
        let matrix2 = self.vector_to_matrix(point2)?;

        // Use affine-invariant distance
        Ok(self.affine_invariant_distance(&matrix1, &matrix2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_spd_creation() {
        let spd = SPD::new(3).unwrap();
        assert_eq!(<SPD as Manifold<f64, Dyn>>::dimension(&spd), 6); // 3*(3+1)/2 = 6
        assert_eq!(spd.matrix_dimension(), 3);
        
        // Test invalid dimension
        assert!(SPD::new(0).is_err());
        
        // Test custom eigenvalue threshold
        let spd_custom = SPD::with_min_eigenvalue(3, 1e-6).unwrap();
        assert_eq!(spd_custom.min_eigenvalue(), 1e-6);
        assert!(SPD::with_min_eigenvalue(3, -1e-6).is_err());
    }

    #[test]
    fn test_vector_matrix_conversion() {
        let spd = SPD::new(3).unwrap();
        
        // Test 3x3 matrix [6 elements: A11, A12, A22, A13, A23, A33]
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix = spd.vector_to_matrix(&vector).unwrap();
        
        // Expected matrix:
        // [1 2 4]
        // [2 3 5]
        // [4 5 6]
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(0, 1)], 2.0);
        assert_eq!(matrix[(1, 0)], 2.0);
        assert_eq!(matrix[(1, 1)], 3.0);
        assert_eq!(matrix[(0, 2)], 4.0);
        assert_eq!(matrix[(2, 0)], 4.0);
        
        // Test round-trip conversion
        let vector_back = spd.matrix_to_vector(&matrix);
        assert_relative_eq!(vector, vector_back, epsilon = 1e-10);
    }

    #[test]
    fn test_spd_projection() {
        let spd = SPD::new(2).unwrap();
        
        // Create a non-SPD matrix (negative eigenvalue)
        let matrix = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        let projected = spd.project_to_spd(&matrix);
        
        // Check that projection is SPD
        assert!(spd.is_spd(&projected, 1e-10));
        
        // Check that eigenvalues are positive
        let eigen = projected.symmetric_eigen();
        for eval in eigen.eigenvalues.iter() {
            assert!(*eval >= spd.min_eigenvalue());
        }
    }

    #[test]
    fn test_point_on_manifold() {
        let spd = SPD::new(2).unwrap();
        
        // Create SPD matrix: [2 1; 1 2] (eigenvalues: 1, 3)
        let spd_matrix = DMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let point = spd.matrix_to_vector(&spd_matrix);
        
        assert!(spd.is_point_on_manifold(&point, 1e-10));
        
        // Create non-SPD matrix: [1 2; 2 1] (eigenvalues: -1, 3)
        let non_spd_matrix = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        let bad_point = spd.matrix_to_vector(&non_spd_matrix);
        
        assert!(!spd.is_point_on_manifold(&bad_point, 1e-10));
    }

    #[test]
    fn test_tangent_space() {
        let spd = SPD::new(2).unwrap();
        
        // Any symmetric matrix is in the tangent space
        let symmetric_matrix = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        let tangent = spd.matrix_to_vector(&symmetric_matrix);
        let point = spd.random_point();
        
        assert!(spd.is_vector_in_tangent_space(&point, &tangent, 1e-10));
        
        // Test with wrong dimension (should fail)
        let wrong_dim = DVector::from_vec(vec![1.0, 2.0]); // Wrong dimension
        assert!(!spd.is_vector_in_tangent_space(&point, &wrong_dim, 1e-10));
    }

    #[test]
    fn test_affine_invariant_metric() {
        let spd = SPD::new(2).unwrap();
        
        // Create test matrices
        let p = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0]); // 2*I
        let u = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]); // I
        let v = DMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]); // Off-diagonal
        
        let p_vec = spd.matrix_to_vector(&p);
        let u_vec = spd.matrix_to_vector(&u);
        let v_vec = spd.matrix_to_vector(&v);
        
        // Test inner product
        let inner = spd.inner_product(&p_vec, &u_vec, &v_vec).unwrap();
        
        // For P = 2I, U = I, V = [0 1; 1 0]:
        // <U,V>_P = tr(P^{-1} U P^{-1} V) = tr(0.5*I * I * 0.5*I * [0 1; 1 0]) = 0
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_retraction_properties() {
        let spd = SPD::new(2).unwrap();
        let point = <SPD as Manifold<f64, Dyn>>::random_point(&spd);
        let zero_tangent = DVector::zeros(3); // dim = 2*(2+1)/2 = 3
        
        // Test centering property: R(x, 0) = x
        let retracted = spd.retract(&point, &zero_tangent).unwrap();
        
        // Should be close to original point
        assert_relative_eq!(
            (&retracted - &point).norm(), 
            0.0, 
            epsilon = 1e-10
        );
        
        // Result should be on manifold
        assert!(spd.is_point_on_manifold(&retracted, 1e-10));
    }

    #[test]
    fn test_distance_properties() {
        let spd = SPD::new(2).unwrap();
        
        let point1 = <SPD as Manifold<f64, Dyn>>::random_point(&spd);
        let point2 = <SPD as Manifold<f64, Dyn>>::random_point(&spd);
        
        // Distance should be non-negative
        let dist = spd.distance(&point1, &point2).unwrap();
        assert!(dist >= 0.0);
        
        // Distance to self should be zero
        let self_dist = spd.distance(&point1, &point1).unwrap();
        assert_relative_eq!(self_dist, 0.0, epsilon = 1e-7);
        
        // Distance should be symmetric
        let dist_rev = spd.distance(&point2, &point1).unwrap();
        assert_relative_eq!(dist, dist_rev, epsilon = 1e-10);
    }

    #[test]
    fn test_random_generation() {
        let spd = SPD::new(3).unwrap();
        
        // Test random point generation
        let random_point = <SPD as Manifold<f64, Dyn>>::random_point(&spd);
        assert!(spd.is_point_on_manifold(&random_point, 1e-10));
        
        // Test random tangent generation
        let tangent = spd.random_tangent(&random_point).unwrap();
        assert!(spd.is_vector_in_tangent_space(&random_point, &tangent, 1e-10));
    }

    #[test]
    fn test_gradient_conversion() {
        let spd = SPD::new(2).unwrap();
        let point = <SPD as Manifold<f64, Dyn>>::random_point(&spd);
        let euclidean_grad = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        let riemannian_grad = spd
            .euclidean_to_riemannian_gradient(&point, &euclidean_grad)
            .unwrap();
        
        assert!(spd.is_vector_in_tangent_space(&point, &riemannian_grad, 1e-10));
    }

    #[test]
    fn test_specific_spd_matrices() {
        let spd = SPD::new(2).unwrap();
        
        // Test identity matrix
        let identity = DMatrix::<f64>::identity(2, 2);
        assert!(spd.is_spd(&identity, 1e-10));
        
        // Test diagonal matrix with positive entries
        let diagonal = DMatrix::from_vec(2, 2, vec![3.0, 0.0, 0.0, 5.0]);
        assert!(spd.is_spd(&diagonal, 1e-10));
        
        // Test correlation matrix
        let correlation = DMatrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]);
        assert!(spd.is_spd(&correlation, 1e-10));
    }

    #[test] 
    fn test_projection_operations() {
        let spd = SPD::new(2).unwrap();
        
        // Test point projection with invalid point
        let bad_point = DVector::from_vec(vec![1.0, 2.0, -1.0]); // Would give negative eigenvalue
        let projected_point = spd.project_point(&bad_point);
        assert!(spd.is_point_on_manifold(&projected_point, 1e-10));
        
        // Test tangent projection
        let point = spd.random_point();
        let non_symmetric_vec = DVector::from_vec(vec![1.0, 2.0, 3.0]); // This represents non-symmetric matrix
        let projected_tangent = spd.project_tangent(&point, &non_symmetric_vec).unwrap();
        
        // Should be symmetric (in tangent space)
        assert!(spd.is_vector_in_tangent_space(&point, &projected_tangent, 1e-10));
    }

    #[test]
    fn test_cholesky_based_operations() {
        let spd = SPD::new(3).unwrap();
        
        // Generate random SPD matrix and test Cholesky decomposition
        let point = <SPD as Manifold<f64, Dyn>>::random_point(&spd);
        let matrix = spd.vector_to_matrix(&point).unwrap();
        
        // Should have successful Cholesky decomposition
        let chol = matrix.clone().cholesky();
        assert!(chol.is_some(), "Generated SPD matrix should have Cholesky decomposition");
        
        // Verify that L*L^T = original matrix
        if let Some(l) = chol {
            let reconstructed = l.l() * l.l().transpose();
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(matrix[(i, j)], reconstructed[(i, j)], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_logarithmic_map() {
        let spd = SPD::new(2).unwrap();
        
        // Test 1: log_X(X) should be zero
        let point = spd.random_point();
        let log_self = spd.inverse_retract(&point, &point).unwrap();
        assert_relative_eq!(log_self.norm(), 0.0, epsilon = 1e-10);
        
        // Test 2: For nearby points, verify consistency
        let tangent = spd.random_tangent(&point).unwrap();
        let small_tangent = tangent * 0.01; // Small step
        let nearby_point = spd.retract(&point, &small_tangent).unwrap();
        let log_map = spd.inverse_retract(&point, &nearby_point).unwrap();
        
        // Since we use approximate retraction but exact log, they won't match perfectly
        // But the result should still be in the tangent space
        assert!(spd.is_vector_in_tangent_space(&point, &log_map, 1e-10));
        // For very small tangent vectors, the approximation should be quite good
        let relative_error = (&log_map - &small_tangent).norm() / small_tangent.norm();
        assert!(relative_error < 0.1, 
                "For small steps, log map should be close to original tangent, relative error: {}", relative_error);
        
        // Test 3: Verify logarithmic map properties
        let point1 = spd.random_point();
        let point2 = spd.random_point();
        
        // log map should give tangent vector
        let tangent_vec = spd.inverse_retract(&point1, &point2).unwrap();
        assert!(spd.is_vector_in_tangent_space(&point1, &tangent_vec, 1e-10));
        
        // Test 4: Specific case with known matrices
        let identity = DMatrix::<f64>::identity(2, 2);
        let id_vec = spd.matrix_to_vector(&identity);
        
        let scaled = &identity * 4.0; // Scale by 4
        let scaled_vec = spd.matrix_to_vector(&scaled);
        
        let log_vec = spd.inverse_retract(&id_vec, &scaled_vec).unwrap();
        let log_mat = spd.vector_to_matrix(&log_vec).unwrap();
        
        // log(4*I) at I should be log(4)*I = ln(4)*I
        let expected = &identity * f64::ln(4.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(log_mat[(i, j)], expected[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_logarithm() {
        let spd = SPD::new(2).unwrap();
        
        // Test diagonal matrix
        let diag = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
        let log_diag = spd.matrix_logarithm(&diag).unwrap();
        
        // log of diagonal matrix should be diagonal with log of eigenvalues
        assert_relative_eq!(log_diag[(0, 0)], f64::ln(2.0), epsilon = 1e-10);
        assert_relative_eq!(log_diag[(1, 1)], f64::ln(3.0), epsilon = 1e-10);
        assert_relative_eq!(log_diag[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(log_diag[(1, 0)], 0.0, epsilon = 1e-10);
        
        // Test that exp(log(A)) = A
        let random_spd = spd.random_spd_matrix::<f64>();
        let log_matrix = spd.matrix_logarithm(&random_spd).unwrap();
        
        // Compute matrix exponential via eigendecomposition
        let log_eigen = log_matrix.symmetric_eigen();
        let exp_eigenvals = log_eigen.eigenvalues.map(|x| f64::exp(x));
        let exp_matrix = &log_eigen.eigenvectors * DMatrix::from_diagonal(&exp_eigenvals) * log_eigen.eigenvectors.transpose();
        
        // Should recover original matrix
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_matrix[(i, j)], random_spd[(i, j)], epsilon = 1e-8);
            }
        }
    }
}