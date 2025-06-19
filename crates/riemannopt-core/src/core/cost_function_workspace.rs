//! Cost function operations using pre-allocated workspace.

use crate::{
    error::Result,
    memory::{Workspace, BufferId},
    types::Scalar,
};
use num_traits::Float;

/// Compute gradient using finite differences with a workspace.
///
/// This version avoids allocations by using pre-allocated buffers from the workspace.
pub fn gradient_fd_with_workspace<T, F>(
    cost_fn: &F,
    point: &nalgebra::DVector<T>,
    workspace: &mut Workspace<T>,
) -> Result<nalgebra::DVector<T>>
where
    T: Scalar,
    F: Fn(&nalgebra::DVector<T>) -> Result<T>,
{
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Pre-allocate vectors
    workspace.get_or_create_vector(BufferId::Gradient, n);
    workspace.get_or_create_vector(BufferId::Temp1, n);
    
    // Create result vector
    let mut result = nalgebra::DVector::zeros(n);
    
    for i in 0..n {
        // Use temporary vector for perturbation
        let e_i = workspace.get_or_create_vector(BufferId::Temp1, n);
        e_i.fill(T::zero());
        e_i[i] = T::one();
        
        // Central difference
        let point_plus = point + &*e_i * h;
        let point_minus = point - &*e_i * h;
        
        let f_plus = cost_fn(&point_plus)?;
        let f_minus = cost_fn(&point_minus)?;
        
        result[i] = (f_plus - f_minus) / (h + h);
    }
    
    // Store result in workspace for future use
    let gradient = workspace.get_or_create_vector(BufferId::Gradient, n);
    gradient.copy_from(&result);
    
    Ok(result)
}

/// Compute a Hessian-vector product using a workspace.
///
/// This version avoids allocations by using pre-allocated buffers.
pub fn hessian_vector_product_with_workspace<T, F>(
    gradient_fn: &F,
    point: &nalgebra::DVector<T>,
    vector: &nalgebra::DVector<T>,
    workspace: &mut Workspace<T>,
) -> Result<nalgebra::DVector<T>>
where
    T: Scalar,
    F: Fn(&nalgebra::DVector<T>) -> Result<nalgebra::DVector<T>>,
{
    let eps = <T as Float>::sqrt(T::epsilon());
    let norm = vector.norm();
    
    if norm < T::epsilon() {
        return Ok(nalgebra::DVector::zeros(point.len()));
    }
    
    let t = eps / norm;
    
    // Use workspace for temporary storage
    let n = point.len();
    let temp = workspace.get_or_create_vector(BufferId::Temp2, n);
    
    // Store scaled vector in temp buffer
    for i in 0..n {
        temp[i] = vector[i] * t;
    }
    
    let perturbed = point + &*temp;
    
    let grad1 = gradient_fn(point)?;
    let grad2 = gradient_fn(&perturbed)?;
    
    // Compute difference and scale
    let result = workspace.get_or_create_vector(BufferId::Temp3, n);
    for i in 0..n {
        result[i] = (grad2[i] - grad1[i]) / t;
    }
    
    Ok(result.clone())
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
    ) -> Result<nalgebra::DVector<T>>;
    
    /// Compute Hessian-vector product with a workspace.
    fn hessian_vector_product_workspace(
        &self,
        point: &nalgebra::DVector<T>,
        vector: &nalgebra::DVector<T>,
        workspace: &mut Workspace<T>,
    ) -> Result<nalgebra::DVector<T>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use approx::assert_relative_eq;
    
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
        let grad_workspace = gradient_fd_with_workspace(&cost_fn, &x, &mut workspace).unwrap();
        
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
        let hvp_workspace = hessian_vector_product_with_workspace(
            &gradient_fn,
            &x,
            &v,
            &mut workspace
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
            let _grad = gradient_fd_with_workspace(&cost_fn, &x, &mut workspace).unwrap();
        }
        
        // Workspace buffers should still exist and have correct size
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap().len(), n);
    }
}