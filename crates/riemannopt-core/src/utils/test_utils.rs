//! Test utilities for property-based testing of manifolds.
//!
//! This module provides utilities for testing mathematical properties
//! of Riemannian manifolds, including retractions, metrics, and other
//! geometric operations.

#[allow(unused_imports)]
use crate::{
    error::Result,
    manifold::{Manifold, Point, TangentVector as TangentVectorType},
    retraction::Retraction,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use num_traits::Float;
use std::fmt::Debug;

// Re-export test manifolds for use in other crates
#[cfg(any(test, feature = "test-utils"))]
pub use crate::test_manifolds::{TestEuclideanManifold, TestSphereManifold, MinimalTestManifold};

/// Configuration for property tests.
#[derive(Debug, Clone)]
pub struct PropertyTestConfig<T> {
    /// Tolerance for numerical comparisons
    pub tolerance: T,
    /// Number of random points to test
    pub num_points: usize,
    /// Number of random tangent vectors per point
    pub num_tangents: usize,
    /// Scale factor for tangent vectors
    pub tangent_scale: T,
}

impl<T: Scalar> Default for PropertyTestConfig<T> {
    fn default() -> Self {
        Self {
            tolerance: <T as Scalar>::from_f64(1e-10),
            num_points: 10,
            num_tangents: 5,
            tangent_scale: <T as Scalar>::from_f64(0.1),
        }
    }
}

/// Results from property tests.
#[derive(Debug)]
pub struct PropertyTestResult<T> {
    /// Whether all tests passed
    pub passed: bool,
    /// Maximum error observed
    pub max_error: T,
    /// Number of tests performed
    pub num_tests: usize,
    /// Detailed error messages
    pub errors: Vec<String>,
}

/// Property-based tests for manifolds.
pub struct ManifoldPropertyTester;

impl ManifoldPropertyTester {
    /// Tests that retraction at zero gives the same point: R(x, 0) = x
    pub fn test_retraction_zero<T, D, M, R>(
        manifold: &M,
        retraction: &R,
        config: &PropertyTestConfig<T>,
    ) -> PropertyTestResult<T>
    where
        T: Scalar,
        D: Dim,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let mut max_error = T::zero();
        let mut errors = Vec::new();
        let mut num_tests = 0;

        for _ in 0..config.num_points {
            let point = manifold.random_point();
            let zero_tangent =
                TangentVectorType::zeros_generic(point.shape_generic().0, nalgebra::U1);

            let mut retracted = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            match retraction.retract(manifold, &point, &zero_tangent, &mut retracted) {
                Ok(()) => {
                    let diff = &retracted - &point;
                    let error = diff.norm();

                    if error > config.tolerance {
                        errors.push(format!(
                            "Retraction at zero failed: error = {} > tolerance = {}",
                            error, config.tolerance
                        ));
                    }

                    max_error = <T as Float>::max(max_error, error);
                    num_tests += 1;
                }
                Err(e) => {
                    errors.push(format!("Retraction failed: {}", e));
                }
            }
        }

        PropertyTestResult {
            passed: errors.is_empty(),
            max_error,
            num_tests,
            errors,
        }
    }

    /// Tests that metric is positive definite: <v, v> > 0 for all v ≠ 0
    pub fn test_metric_positive_definite<T, D, M>(
        manifold: &M,
        config: &PropertyTestConfig<T>,
    ) -> PropertyTestResult<T>
    where
        T: Scalar,
        D: Dim,
        M: Manifold<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let mut min_eigenvalue = T::infinity();
        let mut errors = Vec::new();
        let mut num_tests = 0;

        for _ in 0..config.num_points {
            let point = manifold.random_point();

            for _ in 0..config.num_tangents {
                let mut tangent = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                match manifold.random_tangent(&point, &mut tangent) {
                    Ok(()) => {
                        // Scale the tangent vector
                        tangent *= config.tangent_scale;
                        let tangent_norm = tangent.norm();

                        if tangent_norm > T::epsilon() {
                            match manifold.inner_product(&point, &tangent, &tangent) {
                                Ok(inner_prod) => {
                                    if inner_prod <= T::zero() {
                                        errors.push(format!(
                                            "Metric not positive definite: <v,v> = {} <= 0 for ||v|| = {}",
                                            inner_prod, tangent_norm
                                        ));
                                    }

                                    // Estimate smallest eigenvalue
                                    let eigenvalue_estimate =
                                        inner_prod / (tangent_norm * tangent_norm);
                                    min_eigenvalue =
                                        <T as Float>::min(min_eigenvalue, eigenvalue_estimate);
                                    num_tests += 1;
                                }
                                Err(e) => {
                                    errors.push(format!("Inner product computation failed: {}", e));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        errors.push(format!("Random tangent generation failed: {}", e));
                    }
                }
            }
        }

        PropertyTestResult {
            passed: errors.is_empty(),
            max_error: if min_eigenvalue < T::infinity() {
                T::one() / min_eigenvalue
            } else {
                T::zero()
            },
            num_tests,
            errors,
        }
    }

    /// Tests that tangent space projection is idempotent: P(P(v)) = P(v)
    pub fn test_projection_idempotent<T, D, M>(
        manifold: &M,
        config: &PropertyTestConfig<T>,
    ) -> PropertyTestResult<T>
    where
        T: Scalar,
        D: Dim,
        M: Manifold<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let mut max_error = T::zero();
        let mut errors = Vec::new();
        let mut num_tests = 0;

        for _ in 0..config.num_points {
            let point = manifold.random_point();

            for _ in 0..config.num_tangents {
                // Generate a random vector (not necessarily in tangent space)
                let mut random_vec = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                match manifold.random_tangent(&point, &mut random_vec) {
                    Ok(()) => {
                        // Add some component outside tangent space
                        let perturbed = random_vec * <T as Scalar>::from_f64(1.5);

                        let mut projected_once = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                        match manifold.project_tangent(&point, &perturbed, &mut projected_once) {
                            Ok(()) => {
                                let mut projected_twice = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                                match manifold.project_tangent(&point, &projected_once, &mut projected_twice) {
                                    Ok(()) => {
                                        let diff = &projected_twice - &projected_once;
                                        let error = diff.norm();

                                        if error > config.tolerance {
                                            errors.push(format!(
                                                "Projection not idempotent: ||P(P(v)) - P(v)|| = {} > {}",
                                                error, config.tolerance
                                            ));
                                        }

                                        max_error = <T as Float>::max(max_error, error);
                                        num_tests += 1;
                                    }
                                    Err(e) => {
                                        errors.push(format!("Second projection failed: {}", e));
                                    }
                                }
                            }
                            Err(e) => {
                                errors.push(format!("First projection failed: {}", e));
                            }
                        }
                    }
                    Err(e) => {
                        errors.push(format!("Random vector generation failed: {}", e));
                    }
                }
            }
        }

        PropertyTestResult {
            passed: errors.is_empty(),
            max_error,
            num_tests,
            errors,
        }
    }

    /// Tests parallel transport properties
    pub fn test_parallel_transport<T, D, M, R>(
        manifold: &M,
        retraction: &R,
        config: &PropertyTestConfig<T>,
    ) -> PropertyTestResult<T>
    where
        T: Scalar,
        D: Dim,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let mut max_error = T::zero();
        let mut errors = Vec::new();
        let mut num_tests = 0;

        for _ in 0..config.num_points {
            let point = manifold.random_point();

            // Test 1: Transport along zero curve should be identity
            let mut tangent = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
            match manifold.random_tangent(&point, &mut tangent) {
                Ok(()) => {
                    tangent *= config.tangent_scale;

                    let mut transported = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                    match manifold.parallel_transport(&point, &point, &tangent, &mut transported) {
                        Ok(()) => {
                            let diff = &transported - &tangent;
                            let error = diff.norm();

                            if error > config.tolerance {
                                errors.push(format!(
                                    "Transport along zero curve not identity: error = {}",
                                    error
                                ));
                            }

                            max_error = <T as Float>::max(max_error, error);
                            num_tests += 1;
                        }
                        Err(e) => {
                            errors.push(format!("Parallel transport failed: {}", e));
                        }
                    }
                }
                Err(e) => {
                    errors.push(format!("Random tangent generation failed: {}", e));
                }
            }

            // Test 2: Transport preserves inner products (isometry)
            for _ in 0..config.num_tangents.saturating_sub(1) {
                let mut tangent1 = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                let mut tangent2 = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                match (
                    manifold.random_tangent(&point, &mut tangent1),
                    manifold.random_tangent(&point, &mut tangent2),
                ) {
                    (Ok(()), Ok(())) => {
                        tangent1 *= config.tangent_scale;
                        tangent2 *= config.tangent_scale;
                        let mut direction = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                        let _ = manifold.random_tangent(&point, &mut direction).unwrap_or_else(|_| {
                            direction.copy_from(&tangent1);
                        });
                        direction *= config.tangent_scale;

                        let mut new_point = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
                        match retraction.retract(manifold, &point, &direction, &mut new_point) {
                            Ok(()) => {
                                let mut transported1 = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                                let mut transported2 = TangentVectorType::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::U1);
                                match (
                                    manifold.parallel_transport(&point, &new_point, &tangent1, &mut transported1),
                                    manifold.parallel_transport(&point, &new_point, &tangent2, &mut transported2),
                                ) {
                                    (Ok(()), Ok(())) => {
                                        match (
                                            manifold.inner_product(&point, &tangent1, &tangent2),
                                            manifold.inner_product(
                                                &new_point,
                                                &transported1,
                                                &transported2,
                                            ),
                                        ) {
                                            (Ok(inner1), Ok(inner2)) => {
                                                let error = <T as Float>::abs(inner1 - inner2);

                                                if error > config.tolerance {
                                                    errors.push(format!(
                                                        "Transport not isometric: |<u,v> - <T(u),T(v)>| = {}",
                                                        error
                                                    ));
                                                }

                                                max_error = <T as Float>::max(max_error, error);
                                                num_tests += 1;
                                            }
                                            _ => {
                                                errors.push(
                                                    "Inner product computation failed".to_string(),
                                                );
                                            }
                                        }
                                    }
                                    _ => {
                                        errors.push("Parallel transport failed".to_string());
                                    }
                                }
                            }
                            Err(e) => {
                                errors.push(format!("Retraction failed: {}", e));
                            }
                        }
                    }
                    _ => {
                        errors.push("Random tangent generation failed".to_string());
                    }
                }
            }
        }

        PropertyTestResult {
            passed: errors.is_empty(),
            max_error,
            num_tests,
            errors,
        }
    }

    /// Tests all manifold properties
    pub fn test_all_properties<T, D, M, R>(
        manifold: &M,
        retraction: &R,
        config: &PropertyTestConfig<T>,
    ) -> Vec<(&'static str, PropertyTestResult<T>)>
    where
        T: Scalar,
        D: Dim,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        vec![
            (
                "Retraction at zero",
                Self::test_retraction_zero(manifold, retraction, config),
            ),
            (
                "Metric positive definite",
                Self::test_metric_positive_definite(manifold, config),
            ),
            (
                "Projection idempotent",
                Self::test_projection_idempotent(manifold, config),
            ),
            (
                "Parallel transport",
                Self::test_parallel_transport(manifold, retraction, config),
            ),
        ]
    }
}

/// Helper functions for testing
pub mod helpers {
    use super::*;

    /// Generates random orthogonal matrices for testing (specialized for concrete dimensions)
    pub fn random_orthogonal_3x3() -> nalgebra::Matrix3<f64> {
        use nalgebra::Matrix3;

        // Generate random matrix
        let mut a = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                a[(i, j)] = rand::random::<f64>() * 2.0 - 1.0;
            }
        }

        // QR decomposition to get orthogonal matrix
        let qr = a.qr();
        qr.q()
    }

    /// Generates random orthogonal matrices for dynamic dimensions
    pub fn random_orthogonal_dyn(n: usize) -> nalgebra::DMatrix<f64> {
        use nalgebra::DMatrix;

        // Generate random matrix
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = rand::random::<f64>() * 2.0 - 1.0;
            }
        }

        // QR decomposition to get orthogonal matrix
        let qr = a.qr();
        qr.q()
    }

    /// Generates random positive definite matrices for testing
    pub fn random_positive_definite<T, D>(n: D) -> nalgebra::OMatrix<T, D, D>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D, D>,
    {
        use nalgebra::OMatrix;

        // Generate random matrix
        let mut a = OMatrix::<T, D, D>::zeros_generic(n, n);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                a[(i, j)] = <T as Scalar>::from_f64(rand::random::<f64>());
            }
        }

        // A^T * A is positive definite
        &a.transpose() * &a + OMatrix::identity_generic(n, n) * <T as Scalar>::from_f64(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{retraction::DefaultRetraction, types::DVector};
    use nalgebra::Dyn;

    // Simple test manifold: Euclidean space
    #[derive(Debug)]
    struct EuclideanManifold {
        dim: usize,
    }

    impl Manifold<f64, Dyn> for EuclideanManifold {
        fn name(&self) -> &str {
            "Euclidean"
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn is_point_on_manifold(&self, _point: &DVector<f64>, _tol: f64) -> bool {
            true
        }
        fn is_vector_in_tangent_space(
            &self,
            _point: &DVector<f64>,
            _vector: &DVector<f64>,
            _tol: f64,
        ) -> bool {
            true
        }
        fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>) {
            result.copy_from(point);
        }
        fn project_tangent(
            &self,
            _point: &DVector<f64>,
            vector: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(vector);
            Ok(())
        }
        fn inner_product(
            &self,
            _point: &DVector<f64>,
            u: &DVector<f64>,
            v: &DVector<f64>,
        ) -> Result<f64> {
            Ok(u.dot(v))
        }
        fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
            result.copy_from(&(point + tangent));
            Ok(())
        }
        fn inverse_retract(
            &self,
            point: &DVector<f64>,
            other: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(&(other - point));
            Ok(())
        }
        fn euclidean_to_riemannian_gradient(
            &self,
            _point: &DVector<f64>,
            euclidean_grad: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(euclidean_grad);
            Ok(())
        }
        fn random_point(&self) -> DVector<f64> {
            DVector::zeros(self.dim)
        }
        fn random_tangent(&self, _point: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
            *result = DVector::zeros(self.dim);
            for elem in result.iter_mut() {
                *elem = rand::random::<f64>() * 2.0 - 1.0;
            }
            Ok(())
        }
        fn parallel_transport(
            &self,
            _from: &DVector<f64>,
            _to: &DVector<f64>,
            vector: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(vector);
            Ok(())
        }
    }

    #[test]
    fn test_property_tester_on_euclidean() {
        let manifold = EuclideanManifold { dim: 3 };
        let retraction = DefaultRetraction;
        let config = PropertyTestConfig::default();

        let results = ManifoldPropertyTester::test_all_properties(&manifold, &retraction, &config);

        for (name, result) in results {
            println!(
                "Test {}: passed = {}, max_error = {}",
                name, result.passed, result.max_error
            );
            assert!(result.passed, "Property test '{}' failed", name);
        }
    }

    #[test]
    fn test_helpers() {
        use nalgebra::U3;

        // Test orthogonal matrix generation (3x3)
        let q = helpers::random_orthogonal_3x3();
        let qt_q = &q.transpose() * &q;
        let identity = nalgebra::Matrix3::<f64>::identity();

        // Check that Q^T * Q = I
        for i in 0..3 {
            for j in 0..3 {
                let diff: f64 = qt_q[(i, j)] - identity[(i, j)];
                assert!(
                    diff.abs() < 1e-10,
                    "Q^T * Q not identity at ({}, {}): {} vs {}",
                    i,
                    j,
                    qt_q[(i, j)],
                    identity[(i, j)]
                );
            }
        }

        // Also check determinant is ±1
        let det = q.determinant();
        assert!(
            (det.abs() - 1.0).abs() < 1e-10,
            "Orthogonal matrix determinant not ±1: {}",
            det
        );

        // Test orthogonal matrix generation (dynamic dimension)
        let q_dyn = helpers::random_orthogonal_dyn(4);
        let qt_q_dyn = &q_dyn.transpose() * &q_dyn;
        let identity_dyn = nalgebra::DMatrix::<f64>::identity(4, 4);

        // Check orthogonality for dynamic matrix
        for i in 0..4 {
            for j in 0..4 {
                let diff_dyn: f64 = qt_q_dyn[(i, j)] - identity_dyn[(i, j)];
                assert!(
                    diff_dyn.abs() < 1e-10,
                    "Dynamic Q^T * Q not identity at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Test positive definite matrix generation
        let a = helpers::random_positive_definite::<f64, U3>(U3);
        let eigenvalues = a.symmetric_eigenvalues();
        for &lambda in eigenvalues.iter() {
            assert!(lambda > 0.0, "Matrix not positive definite");
        }
    }
}
