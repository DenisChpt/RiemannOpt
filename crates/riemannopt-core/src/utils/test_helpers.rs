//! Shared test utilities and manifolds for benchmarks and tests.

#![cfg(any(test, feature = "test-utils"))]

use crate::{
    cost_function::CostFunction,
    error::Result,
    manifold::Manifold,
    types::{DMatrix, DVector},
};
use rand::prelude::*;

/// Test sphere manifold for optimization problems
#[derive(Debug, Clone)]
pub struct TestSphere {
    dim: usize,
}

impl TestSphere {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold<f64> for TestSphere {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;
    fn name(&self) -> &str {
        "Test Sphere"
    }

    fn dimension(&self) -> usize {
        self.dim - 1
    }

    fn is_point_on_manifold(&self, point: &DVector<f64>, tol: f64) -> bool {
        (point.norm() - 1.0).abs() < tol
    }

    fn is_vector_in_tangent_space(
        &self,
        point: &DVector<f64>,
        vector: &DVector<f64>,
        tol: f64,
    ) -> bool {
        point.dot(vector).abs() < tol
    }

    fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>) {
        let norm = point.norm();
        if norm > f64::EPSILON {
            result.copy_from(&(point / norm));
        } else {
            result.fill(0.0);
            result[0] = 1.0;
        }
    }

    fn project_tangent(&self, point: &DVector<f64>, vector: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let inner = point.dot(vector);
        result.copy_from(&(vector - point * inner));
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
        // Exponential map on sphere
        let norm_v = tangent.norm();
        if norm_v < f64::EPSILON {
            result.copy_from(point);
        } else {
            let cos_norm = norm_v.cos();
            let sin_norm = norm_v.sin();
            result.copy_from(&(point * cos_norm + tangent * (sin_norm / norm_v)));
        }
        Ok(())
    }

    fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let inner = point.dot(other).min(1.0).max(-1.0);
        let theta = inner.acos();

        if theta.abs() < f64::EPSILON {
            result.fill(0.0);
        } else {
            let v = other - point * inner;
            let v_norm = v.norm();
            if v_norm > f64::EPSILON {
                result.copy_from(&(v * (theta / v_norm)));
            } else {
                result.fill(0.0);
            }
        }
        Ok(())
    }

    fn euclidean_to_riemannian_gradient(
        &self,
        point: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
        result: &mut DVector<f64>,
    ) -> Result<()> {
        self.project_tangent(point, euclidean_grad, result)
    }

    fn random_point(&self, result: &mut DVector<f64>) -> Result<()> {
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        self.project_point(&v, result);
        Ok(())
    }

    fn random_tangent(&self, point: &DVector<f64>, result: &mut DVector<f64>) -> Result<()> {
        let mut rng = thread_rng();
        let mut v = DVector::zeros(self.dim);
        for i in 0..self.dim {
            v[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        self.project_tangent(point, &v, result)
    }

    fn parallel_transport(
        &self,
        _from: &DVector<f64>,
        to: &DVector<f64>,
        vector: &DVector<f64>,
        result: &mut DVector<f64>,
    ) -> Result<()> {
        self.project_tangent(to, vector, result)
    }

    fn has_exact_exp_log(&self) -> bool {
        true
    }

    fn distance(&self, x: &Self::Point, y: &Self::Point) -> Result<f64> {
        let mut tangent = DVector::zeros(x.len());
        self.inverse_retract(x, y, &mut tangent)?;
        self.inner_product(x, &tangent, &tangent).map(|v| v.sqrt())
    }

    fn scale_tangent(
        &self,
        _point: &Self::Point,
        scalar: f64,
        tangent: &Self::TangentVector,
        result: &mut Self::TangentVector,
    ) -> Result<()> {
        result.copy_from(&(tangent * scalar));
        Ok(())
    }

    fn add_tangents(
        &self,
        point: &Self::Point,
        v1: &Self::TangentVector,
        v2: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _temp: &mut Self::TangentVector,
    ) -> Result<()> {
        let sum = v1 + v2;
        self.project_tangent(point, &sum, result)
    }

    fn axpy_tangent(
        &self,
        point: &Self::Point,
        alpha: f64,
        x: &Self::TangentVector,
        y: &Self::TangentVector,
        result: &mut Self::TangentVector,
        _temp1: &mut Self::TangentVector,
        _temp2: &mut Self::TangentVector,
    ) -> Result<()> {
        let axpy_result = alpha * x + y;
        self.project_tangent(point, &axpy_result, result)
    }

    fn norm(&self, _point: &Self::Point, vector: &Self::TangentVector) -> Result<f64> {
        Ok(vector.norm())
    }
}

/// Rayleigh quotient minimization problem
#[derive(Debug)]
pub struct RayleighQuotient {
    matrix: DMatrix<f64>,
}

impl RayleighQuotient {
    pub fn new(dim: usize) -> Self {
        let mut rng = thread_rng();
        let mut matrix = DMatrix::from_fn(dim, dim, |_, _| rng.gen::<f64>());
        // Make symmetric
        matrix = &matrix + &matrix.transpose();
        Self { matrix }
    }
}

impl CostFunction<f64> for RayleighQuotient {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;
    fn cost(&self, x: &DVector<f64>) -> Result<f64> {
        let ax = &self.matrix * x;
        Ok(x.dot(&ax))
    }

    fn cost_and_gradient(
        &self, 
        x: &DVector<f64>, 
        _workspace: &mut crate::memory::Workspace<f64>,
        gradient: &mut DVector<f64>,
    ) -> Result<f64> {
        let ax = &self.matrix * x;
        let cost = x.dot(&ax);
        gradient.copy_from(&(2.0 * ax));
        Ok(cost)
    }

    fn cost_and_gradient_alloc(&self, x: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        let ax = &self.matrix * x;
        let cost = x.dot(&ax);
        let gradient = 2.0 * ax;
        Ok((cost, gradient))
    }

    fn hessian_vector_product(
        &self,
        _point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // For Rayleigh quotient: H*v = 2*A*v
        Ok(2.0 * &self.matrix * vector)
    }

    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let eps = 1e-8;
        let n = point.len();
        let mut gradient = DVector::zeros(n);
        
        for i in 0..n {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            point_plus[i] += eps;
            point_minus[i] -= eps;
            
            let f_plus = self.cost(&point_plus)?;
            let f_minus = self.cost(&point_minus)?;
            gradient[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        
        Ok(gradient)
    }
}

/// Spherical Rosenbrock function
#[derive(Debug)]
pub struct SphericalRosenbrock {
    dim: usize,
}

impl SphericalRosenbrock {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl CostFunction<f64> for SphericalRosenbrock {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;
    fn cost(&self, x: &DVector<f64>) -> Result<f64> {
        let mut cost = 0.0;
        for i in 0..self.dim - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            cost += a * a + 100.0 * b * b;
        }
        Ok(cost)
    }

    fn cost_and_gradient(
        &self, 
        x: &DVector<f64>, 
        _workspace: &mut crate::memory::Workspace<f64>,
        gradient: &mut DVector<f64>,
    ) -> Result<f64> {
        let mut cost = 0.0;
        gradient.fill(0.0);

        for i in 0..self.dim - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            cost += a * a + 100.0 * b * b;

            gradient[i] += -2.0 * a - 400.0 * x[i] * b;
            if i > 0 {
                gradient[i] += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
            }
        }
        if self.dim > 1 {
            gradient[self.dim - 1] += 200.0 * (x[self.dim - 1] - x[self.dim - 2] * x[self.dim - 2]);
        }

        Ok(cost)
    }

    fn cost_and_gradient_alloc(&self, x: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        let mut cost = 0.0;
        let mut gradient = DVector::zeros(self.dim);

        for i in 0..self.dim - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            cost += a * a + 100.0 * b * b;

            gradient[i] += -2.0 * a - 400.0 * x[i] * b;
            if i > 0 {
                gradient[i] += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
            }
        }
        if self.dim > 1 {
            gradient[self.dim - 1] += 200.0 * (x[self.dim - 1] - x[self.dim - 2] * x[self.dim - 2]);
        }

        Ok((cost, gradient))
    }

    fn hessian_vector_product(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // Approximate using finite differences
        let eps = 1e-8;
        let (_, grad1) = self.cost_and_gradient_alloc(point)?;
        let point_plus = point + eps * vector;
        let (_, grad2) = self.cost_and_gradient_alloc(&point_plus)?;
        Ok((grad2 - grad1) / eps)
    }

    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let eps = 1e-8;
        let n = point.len();
        let mut gradient = DVector::zeros(n);
        
        for i in 0..n {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            point_plus[i] += eps;
            point_minus[i] -= eps;
            
            let f_plus = self.cost(&point_plus)?;
            let f_minus = self.cost(&point_minus)?;
            gradient[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        
        Ok(gradient)
    }
}