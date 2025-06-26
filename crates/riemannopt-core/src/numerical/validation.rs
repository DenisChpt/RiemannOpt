//! Numerical validation utilities for manifold operations.
//!
//! This module provides tools for validating the numerical correctness
//! of manifold implementations, including gradient checking, retraction
//! convergence analysis, and stability testing.

use crate::{
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    retraction::Retraction,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, RealField};
use num_traits::{Float, ToPrimitive};

/// Configuration for numerical validation tests.
#[derive(Debug, Clone)]
pub struct NumericalValidationConfig<T> {
    /// Base step size for finite differences
    pub base_step_size: T,
    /// Minimum step size to test
    pub min_step_size: T,
    /// Maximum step size to test
    pub max_step_size: T,
    /// Number of step sizes to test
    pub num_steps: usize,
    /// Tolerance for gradient checking
    pub gradient_tolerance: T,
    /// Tolerance for convergence order
    pub convergence_tolerance: T,
}

impl<T: Scalar> Default for NumericalValidationConfig<T> {
    fn default() -> Self {
        Self {
            base_step_size: <T as Scalar>::from_f64(1e-6),
            min_step_size: <T as Scalar>::from_f64(1e-12),
            max_step_size: <T as Scalar>::from_f64(1e-2),
            num_steps: 10,
            gradient_tolerance: <T as Scalar>::from_f64(1e-8),
            convergence_tolerance: <T as Scalar>::from_f64(0.1),
        }
    }
}

/// Results from gradient checking.
#[derive(Debug)]
pub struct GradientCheckResult<T> {
    /// Maximum relative error between analytical and numerical gradients
    pub max_relative_error: T,
    /// Average relative error
    pub avg_relative_error: T,
    /// Whether the check passed
    pub passed: bool,
    /// Individual errors for each component
    pub component_errors: Vec<T>,
}

/// Results from convergence analysis.
#[derive(Debug)]
pub struct ConvergenceResult<T> {
    /// Estimated order of convergence
    pub order: T,
    /// R-squared value for linear fit
    pub r_squared: T,
    /// Step sizes used
    pub step_sizes: Vec<T>,
    /// Errors at each step size
    pub errors: Vec<T>,
}

/// Numerical validation tools for manifolds.
pub struct NumericalValidator;

impl NumericalValidator {
    /// Check gradient computation using finite differences.
    ///
    /// Given a function f: M -> R and its gradient, verifies that the
    /// gradient matches finite difference approximations.
    pub fn check_gradient<T, D, M, F, G>(
        manifold: &M,
        point: &TangentVector<T, D>,
        f: F,
        grad_f: G,
        config: &NumericalValidationConfig<T>,
    ) -> Result<GradientCheckResult<T>>
    where
        T: Scalar + RealField,
        D: Dim,
        M: Manifold<T, D>,
        F: Fn(&TangentVector<T, D>) -> T,
        G: Fn(&TangentVector<T, D>) -> TangentVector<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let analytical_grad = grad_f(point);
        let mut riemannian_grad = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        let mut workspace = crate::memory::workspace::Workspace::new();
        manifold.euclidean_to_riemannian_gradient(point, &analytical_grad, &mut riemannian_grad, &mut workspace)?;
        let analytical_grad = riemannian_grad;

        let mut component_errors = Vec::new();
        let mut sum_error = T::zero();
        let h = config.base_step_size;

        // Test gradient in random directions
        let num_tests = analytical_grad.len().min(20);
        for _ in 0..num_tests {
            let mut direction = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.random_tangent(point, &mut direction, &mut workspace)?;
            let direction_norm = manifold.norm(point, &direction)?;

            if direction_norm > T::epsilon() {
                // Normalize direction
                let direction = direction * (T::one() / direction_norm);

                // Compute directional derivative analytically
                let analytical_dirderiv =
                    manifold.inner_product(point, &analytical_grad, &direction)?;

                // Compute directional derivative numerically
                let direction_h = &direction * h;
                let mut retracted_plus = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
                manifold.retract(point, &direction_h, &mut retracted_plus, &mut workspace)?;
                let direction_neg_h = &direction * (-h);
                let mut retracted_minus = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
                manifold.retract(point, &direction_neg_h, &mut retracted_minus, &mut workspace)?;

                let f_plus = f(&retracted_plus);
                let f_minus = f(&retracted_minus);
                let numerical_dirderiv = (f_plus - f_minus) / (<T as Scalar>::from_f64(2.0) * h);

                // Compute relative error
                let error = <T as Float>::abs(analytical_dirderiv - numerical_dirderiv);
                let abs_analytical = <T as Float>::abs(analytical_dirderiv);
                let abs_numerical = <T as Float>::abs(numerical_dirderiv);
                let scale = <T as Float>::max(abs_analytical, abs_numerical);
                let scale = <T as Float>::max(scale, T::one());
                let relative_error = error / scale;

                component_errors.push(relative_error);
                sum_error += relative_error;
            }
        }

        let num_errors = <T as Scalar>::from_usize(component_errors.len());
        let avg_relative_error = if component_errors.is_empty() {
            T::zero()
        } else {
            sum_error / num_errors
        };

        let max_relative_error = component_errors
            .iter()
            .cloned()
            .fold(T::zero(), |a, b| <T as Float>::max(a, b));

        Ok(GradientCheckResult {
            max_relative_error,
            avg_relative_error,
            passed: max_relative_error < config.gradient_tolerance,
            component_errors,
        })
    }

    /// Test retraction order of convergence.
    ///
    /// Verifies that ||R(x, tv) - exp_x(tv)|| = O(||tv||^{p+1})
    /// where p is the order of the retraction.
    pub fn test_retraction_convergence<T, D, M, R>(
        manifold: &M,
        retraction: &R,
        point: &TangentVector<T, D>,
        tangent: &TangentVector<T, D>,
        _expected_order: T,
        config: &NumericalValidationConfig<T>,
    ) -> Result<ConvergenceResult<T>>
    where
        T: Scalar + RealField,
        D: Dim,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let mut workspace = crate::memory::workspace::Workspace::new();
        let mut step_sizes = Vec::new();
        let mut errors = Vec::new();

        // Generate logarithmically spaced step sizes
        let log_min = <T as Float>::ln(config.min_step_size);
        let log_max = <T as Float>::ln(config.max_step_size);

        for i in 0..config.num_steps {
            let alpha =
                <T as Scalar>::from_usize(i) / <T as Scalar>::from_usize(config.num_steps - 1);
            let log_h = log_min * (T::one() - alpha) + log_max * alpha;
            let h = <T as Float>::exp(log_h);

            let scaled_tangent = tangent * h;

            // Compute retraction
            let mut retracted = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            retraction.retract(manifold, point, &scaled_tangent, &mut retracted)?;

            // For manifolds with exact exponential map, use it; otherwise use default retraction
            let exp_point = if manifold.has_exact_exp_log() {
                // If manifold has exact exp, it should be implemented as default retraction
                let mut exp_pt = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
                manifold.retract(point, &scaled_tangent, &mut exp_pt, &mut workspace)?;
                exp_pt
            } else {
                // Otherwise, compare against a second-order approximation
                // R(x, tv) = x + tv + O(||tv||^2)
                let first_order = point + &scaled_tangent;
                let mut projected = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
                manifold.project_point(&first_order, &mut projected, &mut workspace);
                projected
            };

            // Compute error in ambient space
            let error = (&retracted - &exp_point).norm();

            if error > T::epsilon() && h > config.min_step_size {
                step_sizes.push(h);
                errors.push(error);
            }
        }

        // Fit log(error) = log(C) + (p+1)*log(h)
        let (order, r_squared) = Self::fit_convergence_order(&step_sizes, &errors)?;

        Ok(ConvergenceResult {
            order,
            r_squared,
            step_sizes,
            errors,
        })
    }

    /// Verify metric compatibility with retraction.
    ///
    /// Tests that the pullback metric via retraction approximates
    /// the Riemannian metric to appropriate order.
    pub fn verify_metric_compatibility<T, D, M, R>(
        manifold: &M,
        retraction: &R,
        point: &TangentVector<T, D>,
        config: &NumericalValidationConfig<T>,
    ) -> Result<bool>
    where
        T: Scalar + RealField,
        D: Dim,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let h = config.base_step_size;

        // Test with random tangent vectors
        let mut workspace = crate::memory::workspace::Workspace::new();
        for _ in 0..10 {
            let mut u = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.random_tangent(point, &mut u, &mut workspace)?;
            let mut v = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.random_tangent(point, &mut v, &mut workspace)?;

            // Scale to small vectors
            let u = u * h;
            let v = v * h;

            // Compute metric at base point
            let g_base = manifold.inner_product(point, &u, &v)?;

            // Compute metric via pullback
            let mut y = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            retraction.retract(manifold, point, &u, &mut y)?;

            // Transport v to y (using differential of retraction)
            let mut v_at_y = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.parallel_transport(point, &y, &v, &mut v_at_y, &mut workspace)?;

            // Compute metric at y
            let g_y = manifold.inner_product(&y, &v_at_y, &v_at_y)?;

            // Check compatibility: g_y should approximate g_base for small h
            let abs_diff = <T as Float>::abs(g_y - g_base);
            let abs_base = <T as Float>::abs(g_base);
            let relative_error = abs_diff / <T as Float>::max(abs_base, T::one());

            // Allow for numerical error and higher-order terms
            let tolerance = h * h + <T as Scalar>::from_f64(1e-10);
            if relative_error > tolerance {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check numerical stability of manifold operations.
    pub fn check_stability<T, D, M>(
        manifold: &M,
        _config: &NumericalValidationConfig<T>,
    ) -> Result<Vec<String>>
    where
        T: Scalar + RealField,
        D: Dim,
        M: Manifold<T, D>,
        DefaultAllocator: Allocator<D>,
    {
        let mut issues = Vec::new();

        // Test 1: Projection stability
        let point = manifold.random_point();
        let mut workspace = crate::memory::workspace::Workspace::new();
        for scale in [T::epsilon(), T::one(), <T as Scalar>::from_f64(1e6)] {
            let perturbed = &point * (T::one() + scale * T::epsilon());
            let mut projected = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            manifold.project_point(&perturbed, &mut projected, &mut workspace);

            if !manifold.is_point_on_manifold(&projected, <T as Scalar>::from_f64(1e-10)) {
                issues.push(format!(
                    "Projection unstable at scale {:e}",
                    <T as ToPrimitive>::to_f64(&scale).unwrap_or(0.0)
                ));
            }
        }

        // Test 2: Retraction near singularities
        let mut tangent = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        manifold.random_tangent(&point, &mut tangent, &mut workspace)?;
        for scale in [
            T::epsilon(),
            <T as Scalar>::from_f64(1e-8),
            <T as Scalar>::from_f64(1e-4),
        ] {
            let tiny_tangent = tangent.clone() * scale;
            let mut retracted = Point::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
            match manifold.retract(&point, &tiny_tangent, &mut retracted, &mut workspace) {
                Ok(()) => {
                    let mut back = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
                    manifold.inverse_retract(&point, &retracted, &mut back, &mut workspace)?;
                    let diff_vec = &back - &tiny_tangent;
                    let norm_diff = manifold.norm(&point, &diff_vec)?;
                    let norm_tiny = manifold.norm(&point, &tiny_tangent)?;
                    let rel_error = norm_diff / norm_tiny;

                    if rel_error > <T as Scalar>::from_f64(0.1) {
                        issues.push(format!(
                            "Retraction/inverse unstable at scale {:e}: error = {:e}",
                            <T as ToPrimitive>::to_f64(&scale).unwrap_or(0.0),
                            <T as ToPrimitive>::to_f64(&rel_error).unwrap_or(0.0)
                        ));
                    }
                }
                Err(_) => {
                    issues.push(format!(
                        "Retraction failed at scale {:e}",
                        <T as ToPrimitive>::to_f64(&scale).unwrap_or(0.0)
                    ));
                }
            }
        }

        // Test 3: Orthogonality preservation
        let mut u = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        manifold.random_tangent(&point, &mut u, &mut workspace)?;
        let mut v = TangentVector::<T, D>::zeros_generic(point.shape_generic().0, nalgebra::Const::<1>);
        manifold.random_tangent(&point, &mut v, &mut workspace)?;

        // Make v orthogonal to u
        let u_norm_sq = manifold.inner_product(&point, &u, &u)?;
        if u_norm_sq > T::epsilon() {
            let proj_coeff = manifold.inner_product(&point, &v, &u)? / u_norm_sq;
            let v_ortho = &v - &u * proj_coeff;

            // Check orthogonality is preserved
            let inner = manifold.inner_product(&point, &u, &v_ortho)?;
            if <T as Float>::abs(inner) > <T as Scalar>::from_f64(1e-12) {
                issues.push(format!(
                    "Orthogonalization unstable: <u, v_ortho> = {:e}",
                    <T as ToPrimitive>::to_f64(&inner).unwrap_or(0.0)
                ));
            }
        }

        Ok(issues)
    }

    /// Fit convergence order from error data.
    fn fit_convergence_order<T>(step_sizes: &[T], errors: &[T]) -> Result<(T, T)>
    where
        T: Scalar + RealField,
    {
        if step_sizes.len() < 2 {
            return Err(ManifoldError::dimension_mismatch(
                "2",
                format!("{}", step_sizes.len()),
            ));
        }

        let n = <T as Scalar>::from_usize(step_sizes.len());

        // Convert to log scale
        let log_h: Vec<T> = step_sizes.iter().map(|h| <T as Float>::ln(*h)).collect();
        let log_e: Vec<T> = errors.iter().map(|e| <T as Float>::ln(*e)).collect();

        // Compute means
        let sum_log_h: T = log_h.iter().fold(T::zero(), |acc, &x| acc + x);
        let sum_log_e: T = log_e.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean_log_h = sum_log_h / n;
        let mean_log_e = sum_log_e / n;

        // Compute slope (order + 1)
        let mut num = T::zero();
        let mut den = T::zero();

        for i in 0..step_sizes.len() {
            let dh = log_h[i] - mean_log_h;
            let de = log_e[i] - mean_log_e;
            num += dh * de;
            den += dh * dh;
        }

        let slope = if den > T::epsilon() {
            num / den
        } else {
            T::zero()
        };
        let order = slope - T::one();

        // Compute R-squared
        let mut ss_tot = T::zero();
        let mut ss_res = T::zero();

        for i in 0..step_sizes.len() {
            let predicted = mean_log_e + slope * (log_h[i] - mean_log_h);
            let res_diff = log_e[i] - predicted;
            let tot_diff = log_e[i] - mean_log_e;
            ss_res += res_diff * res_diff;
            ss_tot += tot_diff * tot_diff;
        }

        let r_squared = if ss_tot > T::epsilon() {
            T::one() - ss_res / ss_tot
        } else {
            T::zero()
        };

        Ok((order, r_squared))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVector;
    use nalgebra::Dyn;

    // Test manifold that implements exact exponential map
    #[derive(Debug)]
    struct TestManifold;

    impl Manifold<f64, Dyn> for TestManifold {
        fn name(&self) -> &str {
            "Test"
        }
        fn dimension(&self) -> usize {
            3
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
            *result = point + tangent;
            Ok(())
        }
        fn inverse_retract(
            &self,
            point: &DVector<f64>,
            other: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            *result = other - point;
            Ok(())
        }
        fn euclidean_to_riemannian_gradient(
            &self,
            _point: &DVector<f64>,
            grad: &DVector<f64>,
            result: &mut DVector<f64>,
        ) -> Result<()> {
            result.copy_from(grad);
            Ok(())
        }
        fn random_point(&self) -> DVector<f64> {
            DVector::zeros(3)
        }
        fn random_tangent(&self, _point: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
            *result = DVector::from_vec(vec![1.0, 0.0, 0.0]);
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
    fn test_gradient_checking() {
        let manifold = TestManifold;
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let config = NumericalValidationConfig::default();

        // Test function: f(x) = ||x||^2
        let f = |x: &DVector<f64>| x.dot(x);
        let grad_f = |x: &DVector<f64>| x * 2.0;

        let result =
            NumericalValidator::check_gradient(&manifold, &point, f, grad_f, &config).unwrap();

        assert!(result.passed, "Gradient check failed");
        assert!(result.max_relative_error < 1e-6);
    }
}
