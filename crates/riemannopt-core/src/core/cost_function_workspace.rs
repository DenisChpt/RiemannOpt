//! Cost function operations using pre-allocated workspace.

use crate::{
    error::Result,
    memory::Workspace,
    types::Scalar,
};
use num_traits::Float;

/// Compute gradient using finite differences with a workspace.
///
/// This version avoids allocations by using pre-allocated buffers from the workspace.
pub fn gradient_fd<T, F>(
    cost_fn: &F,
    point: &nalgebra::DVector<T>,
    workspace: &mut Workspace<T>,
    gradient: &mut nalgebra::DVector<T>,
) -> Result<()>
where
    T: Scalar,
    F: Fn(&nalgebra::DVector<T>) -> Result<T>,
{
    
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    
    gradient.fill(T::zero());
    
    // Use the special method to get multiple buffers at once
    let (_, e_i, point_plus, point_minus) = workspace.get_gradient_buffers_mut()
        .ok_or_else(|| crate::error::ManifoldError::invalid_parameter(
            "Workspace missing required gradient buffers".to_string()
        ))?;
    
    // Verify dimensions
    if e_i.len() != n || point_plus.len() != n || point_minus.len() != n {
        return Err(crate::error::ManifoldError::invalid_parameter(
            format!("Workspace buffers have incorrect dimensions for point of size {}", n),
        ));
    }
    
    for i in 0..n {
        // Create unit vector in direction i
        e_i.fill(T::zero());
        e_i[i] = T::one();
        
        // Central difference
        point_plus.copy_from(point);
        point_plus.axpy(h, e_i, T::one());
        
        point_minus.copy_from(point);
        point_minus.axpy(-h, e_i, T::one());
        
        let f_plus = cost_fn(point_plus)?;
        let f_minus = cost_fn(point_minus)?;
        
        gradient[i] = (f_plus - f_minus) / (h + h);
    }
    
    Ok(())
}

/// Compute a Hessian-vector product using a workspace.
///
/// This version avoids allocations by using pre-allocated buffers.
pub fn hessian_vector_product<T, F>(
    gradient_fn: &F,
    point: &nalgebra::DVector<T>,
    vector: &nalgebra::DVector<T>,
    workspace: &mut Workspace<T>,
    result: &mut nalgebra::DVector<T>,
) -> Result<()>
where
    T: Scalar,
    F: Fn(&nalgebra::DVector<T>) -> Result<nalgebra::DVector<T>>,
{
    use crate::memory::BufferId;
    
    let eps = <T as Float>::sqrt(T::epsilon());
    let norm = vector.norm();
    
    let n = point.len();
    
    if norm < T::epsilon() {
        result.fill(T::zero());
        return Ok(());
    }
    
    let t = eps / norm;
    
    // Use workspace buffers
    let perturbed = workspace.get_or_create_vector(BufferId::Temp1, n);
    
    // Compute perturbed point
    perturbed.copy_from(point);
    perturbed.axpy(t, vector, T::one());
    
    let grad1 = gradient_fn(point)?;
    let grad2 = gradient_fn(perturbed)?;
    
    // Compute difference and scale
    result.copy_from(&grad2);
    result.axpy(-T::one(), &grad1, T::one());
    result.scale_mut(T::one() / t);
    
    Ok(())
}

/// Compute gradient using finite differences (allocating version).
///
/// This is a convenience function that allocates memory.
pub fn gradient_fd_alloc<T, F>(
    cost_fn: &F,
    point: &nalgebra::DVector<T>,
) -> Result<nalgebra::DVector<T>>
where
    T: Scalar,
    F: Fn(&nalgebra::DVector<T>) -> Result<T>,
{
    let n = point.len();
    let mut workspace = Workspace::with_size(n);
    let mut gradient = nalgebra::DVector::zeros(n);
    gradient_fd(cost_fn, point, &mut workspace, &mut gradient)?;
    Ok(gradient)
}

/// Compute a Hessian-vector product (allocating version).
///
/// This is a convenience function that allocates memory.
pub fn hessian_vector_product_alloc<T, F>(
    gradient_fn: &F,
    point: &nalgebra::DVector<T>,
    vector: &nalgebra::DVector<T>,
) -> Result<nalgebra::DVector<T>>
where
    T: Scalar,
    F: Fn(&nalgebra::DVector<T>) -> Result<nalgebra::DVector<T>>,
{
    let n = point.len();
    let mut workspace = Workspace::with_size(n);
    let mut result = nalgebra::DVector::zeros(n);
    hessian_vector_product(gradient_fn, point, vector, &mut workspace, &mut result)?;
    Ok(result)
}

/// Extension trait for CostFunction to add workspace methods.
pub trait CostFunctionWorkspace<T>
where
    T: Scalar,
{
    /// Compute gradient using finite differences with a workspace.
    fn gradient_fd_workspace(
        &self,
        point: &nalgebra::DVector<T>,
        workspace: &mut Workspace<T>,
        gradient: &mut nalgebra::DVector<T>,
    ) -> Result<()>;
    
    /// Compute Hessian-vector product with a workspace.
    fn hessian_vector_product_workspace(
        &self,
        point: &nalgebra::DVector<T>,
        vector: &nalgebra::DVector<T>,
        workspace: &mut Workspace<T>,
        result: &mut nalgebra::DVector<T>,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use approx::assert_relative_eq;
    use crate::memory::BufferId;
    
    #[test]
    fn test_gradient_fd_workspace() {
        let n = 50;
        let a = DMatrix::<f64>::identity(n, n) * 2.0;
        let b = DVector::zeros(n);
        
        let x = DVector::from_element(n, 1.0);
        let mut workspace = Workspace::with_size(n);
        
        // Define cost function
        let cost_fn = |x: &DVector<f64>| -> Result<f64> {
            Ok((0.5 * x.transpose() * &a * x)[(0, 0)] + b.dot(x))
        };
        
        // Compute gradient with workspace
        let mut grad_workspace = DVector::<f64>::zeros(n);
        gradient_fd(&cost_fn, &x, &mut workspace, &mut grad_workspace).unwrap();
        
        // Compare with analytical gradient: Ax + b
        let grad_analytical = &a * &x + &b;
        
        for i in 0..n {
            assert_relative_eq!(grad_workspace[i], grad_analytical[i], epsilon = 5e-4);
        }
    }
    
    #[test]
    fn test_hessian_vector_product_workspace() {
        let n = 30;
        let a = DMatrix::<f64>::identity(n, n) * 3.0;
        let b = DVector::zeros(n);
        
        let x = DVector::from_element(n, 1.0);
        let v = DVector::from_element(n, 0.5);
        let mut workspace = Workspace::with_size(n);
        
        // Define gradient function
        let gradient_fn = |x: &DVector<f64>| -> Result<DVector<f64>> {
            Ok(&a * x + &b)
        };
        
        // Compute Hessian-vector product with workspace
        let mut hvp_workspace = DVector::<f64>::zeros(n);
        hessian_vector_product(
            &gradient_fn,
            &x,
            &v,
            &mut workspace,
            &mut hvp_workspace
        ).unwrap();
        
        // Expected result: A * v
        let expected = &a * &v;
        
        for i in 0..n {
            assert_relative_eq!(hvp_workspace[i], expected[i], epsilon = 1e-5);
        }
    }
    
    #[test]
    fn test_workspace_reuse() {
        let n = 20;
        let mut workspace = Workspace::with_size(n);
        
        // Pre-check that buffers are allocated
        assert!(workspace.get_vector(BufferId::Gradient).is_some());
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap().len(), n);
        
        // Use workspace multiple times
        let a = DMatrix::<f64>::identity(n, n);
        let b = DVector::zeros(n);
        
        for _ in 0..5 {
            let x = DVector::from_element(n, 1.0);
            let cost_fn = |x: &DVector<f64>| -> Result<f64> {
                Ok((0.5 * x.transpose() * &a * x)[(0, 0)] + b.dot(x))
            };
            let mut grad = DVector::<f64>::zeros(n);
            gradient_fd(&cost_fn, &x, &mut workspace, &mut grad).unwrap();
        }
        
        // Workspace buffers should still exist and have correct size
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap().len(), n);
    }
}