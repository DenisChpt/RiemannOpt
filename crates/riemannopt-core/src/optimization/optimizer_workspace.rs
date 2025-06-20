//! Optimizer operations using pre-allocated workspace.

use crate::{
    error::{Result, OptimizerResult, OptimizerError, ManifoldError},
    memory::{Workspace, workspace::BufferId},
    types::{Scalar, DVector},
};
use num_traits::Float;

/// Perform a line search step using workspace.
pub fn line_search_step_with_workspace<T, F>(
    cost_fn: &F,
    point: &DVector<T>,
    direction: &DVector<T>,
    initial_step: T,
    workspace: &mut Workspace<T>,
    result_point: &mut DVector<T>,
) -> OptimizerResult<T>
where
    T: Scalar,
    F: Fn(&DVector<T>) -> Result<T>,
{
    let mut alpha = initial_step;
    let c1 = T::from(0.0001).unwrap();
    let c2 = T::from(0.9).unwrap();
    let max_iters = 20;
    
    let n = point.len();
    
    // Compute initial values
    let f0 = cost_fn(point).map_err(|e| OptimizerError::ManifoldError(e))?;
    
    // Use workspace gradient buffer
    gradient_fd_simple(cost_fn, point, workspace).map_err(|e| OptimizerError::ManifoldError(e))?;
    let grad0 = workspace.get_vector(BufferId::Gradient)
        .ok_or_else(|| OptimizerError::ManifoldError(
            ManifoldError::invalid_parameter("Missing gradient buffer in workspace")
        ))?;
    let dir_deriv0 = grad0.dot(direction);
    
    if dir_deriv0 >= T::zero() {
        // Add context via ManifoldError instead
        return Err(OptimizerError::ManifoldError(
            ManifoldError::invalid_parameter(
                format!("Search direction is not a descent direction. Directional derivative: {:.2e}, gradient norm: {:.2e}",
                    dir_deriv0.to_f64(),
                    grad0.norm().to_f64())
            )
        ));
    }
    
    // Armijo line search - verify workspace buffer
    {
        let new_point = workspace.get_vector_mut(BufferId::Temp1)
            .ok_or_else(|| OptimizerError::ManifoldError(
                ManifoldError::invalid_parameter("Missing Temp1 buffer in workspace")
            ))?;
        if new_point.len() != n {
            return Err(OptimizerError::ManifoldError(
                ManifoldError::invalid_parameter(format!("Temp1 buffer has wrong size: {} vs {}", new_point.len(), n))
            ));
        }
    }
    
    for _ in 0..max_iters {
        // Update new_point = point + alpha * direction
        {
            let new_point = workspace.get_vector_mut(BufferId::Temp1).unwrap();
            new_point.copy_from(point);
            new_point.axpy(alpha, direction, T::one());
        }
        
        // Evaluate cost at new point
        let f_new = {
            let new_point = workspace.get_vector(BufferId::Temp1).unwrap();
            cost_fn(new_point).map_err(|e| OptimizerError::ManifoldError(e))?
        };
        
        // Check Armijo condition
        if f_new <= f0 + c1 * alpha * dir_deriv0 {
            // Check strong Wolfe condition by computing gradient at new point
            // We need to clone the point to avoid borrow checker issues
            let new_point_clone = workspace.get_vector(BufferId::Temp1).unwrap().clone();
            gradient_fd_simple(cost_fn, &new_point_clone, workspace).map_err(|e| OptimizerError::ManifoldError(e))?;
            
            let grad_new = workspace.get_vector(BufferId::Gradient)
                .ok_or_else(|| OptimizerError::ManifoldError(
                    ManifoldError::invalid_parameter("Missing gradient buffer in workspace")
                ))?;
            let dir_deriv_new = grad_new.dot(direction);
            
            if <T as num_traits::Float>::abs(dir_deriv_new) <= -c2 * dir_deriv0 {
                let new_point = workspace.get_vector(BufferId::Temp1).unwrap();
                result_point.copy_from(new_point);
                return Ok(alpha);
            }
        }
        
        // Reduce step size
        alpha *= T::from(0.5).unwrap();
    }
    
    // Return best attempt
    let new_point = workspace.get_vector(BufferId::Temp1).unwrap();
    result_point.copy_from(new_point);
    Ok(alpha)
}

/// Simple gradient computation for line search (without the full CostFunction trait).
fn gradient_fd_simple<T, F>(
    cost_fn: &F,
    point: &DVector<T>,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
    F: Fn(&DVector<T>) -> Result<T>,
{
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Get pre-allocated buffers from workspace
    let (gradient, e_i, point_plus, point_minus) = workspace.get_gradient_buffers_mut()
        .ok_or_else(|| ManifoldError::invalid_parameter(
            "Workspace missing required gradient buffers".to_string()
        ))?;
        
    // Verify dimensions
    if gradient.len() != n || e_i.len() != n || point_plus.len() != n || point_minus.len() != n {
        return Err(ManifoldError::invalid_parameter(
            format!("Workspace buffers have incorrect dimensions for point of size {}", n),
        ));
    }
    
    gradient.fill(T::zero());
    
    for i in 0..n {
        e_i.fill(T::zero());
        e_i[i] = T::one();
        
        // Use pre-allocated buffers
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

/// Update momentum buffer in-place using workspace.
pub fn update_momentum_with_workspace<T>(
    momentum: &mut DVector<T>,
    gradient: &DVector<T>,
    beta: T,
    _workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
{
    // momentum = beta * momentum + (1 - beta) * gradient
    momentum.scale_mut(beta);
    momentum.axpy(T::one() - beta, gradient, T::one());
    
    Ok(())
}

/// Update Adam optimizer state using workspace.
pub fn update_adam_state_with_workspace<T>(
    momentum: &mut DVector<T>,
    second_moment: &mut DVector<T>,
    gradient: &DVector<T>,
    beta1: T,
    beta2: T,
    _workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
{
    // Update first moment estimate
    update_momentum_with_workspace(momentum, gradient, beta1, _workspace)?;
    
    // Update second moment estimate
    // second_moment = beta2 * second_moment + (1 - beta2) * gradient^2
    for i in 0..gradient.len() {
        let g_squared = gradient[i] * gradient[i];
        second_moment[i] = beta2 * second_moment[i] + (T::one() - beta2) * g_squared;
    }
    
    Ok(())
}

/// Compute the search direction for quasi-Newton methods using workspace.
pub fn compute_quasi_newton_direction_with_workspace<T>(
    _hessian_approx: &crate::types::DMatrix<T>,
    gradient: &DVector<T>,
    _workspace: &mut Workspace<T>,
    direction: &mut DVector<T>,
) -> Result<()>
where
    T: Scalar,
{
    // Solve H * d = -g for direction d
    // For now, use simple matrix-vector multiply (should use proper linear solver)
    direction.copy_from(gradient);
    direction.scale_mut(-T::one());
    
    // This is a placeholder - in practice, we'd solve the linear system
    // For now, just return -gradient (gradient descent direction)
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_line_search_workspace() {
        let n = 3;
        let mut workspace = Workspace::with_size(n);
        
        // Simple quadratic function: f(x) = ||x||^2
        let cost_fn = |x: &DVector<f64>| -> Result<f64> {
            Ok(x.dot(x))
        };
        
        let point = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        let direction = DVector::from_vec(vec![-1.0, -1.0, -1.0]); // Descent direction
        
        let mut new_point = DVector::<f64>::zeros(n);
        let alpha = line_search_step_with_workspace(
            &cost_fn,
            &point,
            &direction,
            1.0,
            &mut workspace,
            &mut new_point
        ).unwrap();
        
        // Should find a step that reduces the cost
        assert!(alpha > 0.0);
        assert!(cost_fn(&new_point).unwrap() < cost_fn(&point).unwrap());
    }
    
    #[test]
    fn test_momentum_update_workspace() {
        let n = 4;
        let mut workspace = Workspace::with_size(n);
        
        let mut momentum = DVector::<f64>::zeros(n);
        let gradient = DVector::from_element(n, 1.0);
        let beta = 0.9;
        
        update_momentum_with_workspace(&mut momentum, &gradient, beta, &mut workspace).unwrap();
        
        // First update: momentum = 0.1 * gradient
        for i in 0..n {
            assert_relative_eq!(momentum[i], 0.1, epsilon = 1e-10);
        }
        
        // Second update
        update_momentum_with_workspace(&mut momentum, &gradient, beta, &mut workspace).unwrap();
        
        // momentum = 0.9 * 0.1 + 0.1 * 1.0 = 0.19
        for i in 0..n {
            assert_relative_eq!(momentum[i], 0.19, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_adam_update_workspace() {
        let n = 3;
        let mut workspace = Workspace::with_size(n);
        
        let mut momentum = DVector::<f64>::zeros(n);
        let mut second_moment = DVector::zeros(n);
        let gradient = DVector::from_element(n, 2.0);
        let beta1 = 0.9;
        let beta2 = 0.999;
        
        update_adam_state_with_workspace(
            &mut momentum,
            &mut second_moment,
            &gradient,
            beta1,
            beta2,
            &mut workspace
        ).unwrap();
        
        // Check first moment
        for i in 0..n {
            assert_relative_eq!(momentum[i], 0.2, epsilon = 1e-10);
        }
        
        // Check second moment
        for i in 0..n {
            assert_relative_eq!(second_moment[i], 0.004, epsilon = 1e-10);
        }
    }
}