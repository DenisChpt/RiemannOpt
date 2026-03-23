//! Optimizer operations using pre-allocated workspace.

use crate::{
    error::{ManifoldError, OptimizerError, OptimizerResult, Result},
    memory::{workspace::BufferId, Workspace},
    types::{DVector, Scalar},
};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Helper: map workspace miss to OptimizerError
// ---------------------------------------------------------------------------

#[inline]
fn ws_err<T>(msg: &str) -> OptimizerError {
    OptimizerError::ManifoldError(ManifoldError::invalid_parameter(msg))
}

#[inline]
fn ws_get<'a, T: Scalar, B: 'static>(
    ws: &'a Workspace<T>,
    id: BufferId,
) -> OptimizerResult<&'a B> {
    ws.get_buffer::<B>(id)
        .ok_or_else(|| ws_err::<T>(&format!("Missing workspace buffer {id:?}")))
}

#[inline]
fn ws_get_mut<'a, T: Scalar, B: 'static>(
    ws: &'a mut Workspace<T>,
    id: BufferId,
) -> OptimizerResult<&'a mut B> {
    ws.get_buffer_mut::<B>(id)
        .ok_or_else(|| ws_err::<T>(&format!("Missing workspace buffer {id:?}")))
}

// ---------------------------------------------------------------------------
// Line search (strong Wolfe conditions)
// ---------------------------------------------------------------------------

/// Perform an Armijo–Wolfe backtracking line search using workspace buffers.
///
/// Buffers used: `Temp1` (trial point), `Temp2` (copy for gradient eval),
/// `Gradient`, `PointPlus`, `PointMinus`, `UnitVector`.
pub fn line_search_step<T, F>(
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
    let c1 = T::from(0.0001).expect("constant");
    let c2 = T::from(0.9).expect("constant");
    let half = T::from(0.5).expect("constant");
    let max_iters = 20;
    let n = point.len();

    // Ensure Temp2 is allocated (used to avoid clone)
    workspace.preallocate_vector(BufferId::Temp2, n);

    // Initial cost
    let f0 = cost_fn(point).map_err(OptimizerError::ManifoldError)?;

    // Initial gradient (writes into Gradient buffer)
    gradient_fd_simple(cost_fn, point, workspace).map_err(OptimizerError::ManifoldError)?;
    let dir_deriv0 = {
        let grad = ws_get::<T, DVector<T>>(workspace, BufferId::Gradient)?;
        let dd = grad.dot(direction);
        if dd >= T::zero() {
            return Err(ws_err::<T>(&format!(
                "Not a descent direction. Directional derivative: {:.2e}, ‖∇f‖: {:.2e}",
                dd.to_f64(),
                grad.norm().to_f64()
            )));
        }
        dd
    };

    // Verify Temp1 buffer size
    {
        let buf = ws_get_mut::<T, DVector<T>>(workspace, BufferId::Temp1)?;
        if buf.len() != n {
            return Err(ws_err::<T>(&format!(
                "Temp1 buffer size mismatch: {} vs {n}",
                buf.len()
            )));
        }
    }

    let mut alpha = initial_step;

    for _ in 0..max_iters {
        // trial = point + α·direction
        {
            let trial = ws_get_mut::<T, DVector<T>>(workspace, BufferId::Temp1)?;
            trial.copy_from(point);
            trial.axpy(alpha, direction, T::one());
        }

        // Evaluate cost at trial point
        let f_new = {
            let trial = ws_get::<T, DVector<T>>(workspace, BufferId::Temp1)?;
            cost_fn(trial).map_err(OptimizerError::ManifoldError)?
        };

        // Check Armijo (sufficient decrease) condition
        if f_new <= f0 + c1 * alpha * dir_deriv0 {
            // Copy Temp1 → result_point (already &mut) to avoid .clone(),
            // then compute gradient at result_point.
            {
                let trial = ws_get::<T, DVector<T>>(workspace, BufferId::Temp1)?;
                result_point.copy_from(trial);
            }
            gradient_fd_simple(cost_fn, result_point, workspace)
                .map_err(OptimizerError::ManifoldError)?;

            let dir_deriv_new = {
                let grad_new = ws_get::<T, DVector<T>>(workspace, BufferId::Gradient)?;
                grad_new.dot(direction)
            };

            // Check strong Wolfe (curvature) condition
            if <T as Float>::abs(dir_deriv_new) <= -c2 * dir_deriv0 {
                // result_point already contains the accepted trial point
                return Ok(alpha);
            }
        }

        alpha = alpha * half;
    }

    // Return best attempt (last trial point)
    let trial = ws_get::<T, DVector<T>>(workspace, BufferId::Temp1)?;
    result_point.copy_from(trial);
    Ok(alpha)
}

// ---------------------------------------------------------------------------
// Finite-difference gradient (workspace version)
// ---------------------------------------------------------------------------

/// Central-difference gradient using workspace buffers.
///
/// Writes the result into `BufferId::Gradient`.
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
    let h = T::fd_epsilon();
    let two_h = h + h;

    workspace.preallocate_vector(BufferId::Gradient, n);
    workspace.preallocate_vector(BufferId::PointPlus, n);
    workspace.preallocate_vector(BufferId::PointMinus, n);

    // We no longer need UnitVector — just perturb element i directly.

    for i in 0..n {
        // point_plus = point;  point_plus[i] += h
        {
            let pp = workspace
                .get_buffer_mut::<DVector<T>>(BufferId::PointPlus)
                .ok_or_else(|| ManifoldError::invalid_parameter("PointPlus buffer"))?;
            pp.copy_from(point);
            pp[i] = pp[i] + h;
        }
        // point_minus = point;  point_minus[i] -= h
        {
            let pm = workspace
                .get_buffer_mut::<DVector<T>>(BufferId::PointMinus)
                .ok_or_else(|| ManifoldError::invalid_parameter("PointMinus buffer"))?;
            pm.copy_from(point);
            pm[i] = pm[i] - h;
        }

        let f_plus = {
            let pp = workspace
                .get_buffer::<DVector<T>>(BufferId::PointPlus)
                .ok_or_else(|| ManifoldError::invalid_parameter("PointPlus buffer"))?;
            cost_fn(pp)?
        };
        let f_minus = {
            let pm = workspace
                .get_buffer::<DVector<T>>(BufferId::PointMinus)
                .ok_or_else(|| ManifoldError::invalid_parameter("PointMinus buffer"))?;
            cost_fn(pm)?
        };

        let grad = workspace
            .get_buffer_mut::<DVector<T>>(BufferId::Gradient)
            .ok_or_else(|| ManifoldError::invalid_parameter("Gradient buffer"))?;
        grad[i] = (f_plus - f_minus) / two_h;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Momentum / Adam helpers
// ---------------------------------------------------------------------------

/// Update momentum buffer in-place.
///
/// `momentum ← β · momentum + (1 − β) · gradient`
pub fn update_momentum<T>(
    momentum: &mut DVector<T>,
    gradient: &DVector<T>,
    beta: T,
    _workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
{
    momentum.scale_mut(beta);
    momentum.axpy(T::one() - beta, gradient, T::one());
    Ok(())
}

/// Update Adam first and second moment estimates in-place.
pub fn update_adam_state<T>(
    momentum: &mut DVector<T>,
    second_moment: &mut DVector<T>,
    gradient: &DVector<T>,
    beta1: T,
    beta2: T,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
{
    update_momentum(momentum, gradient, beta1, workspace)?;

    let one_minus_b2 = T::one() - beta2;
    for i in 0..gradient.len() {
        let g2 = gradient[i] * gradient[i];
        second_moment[i] = beta2 * second_moment[i] + one_minus_b2 * g2;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Quasi-Newton direction  (Cholesky solve)
// ---------------------------------------------------------------------------

/// Solve  H d = −g  for the quasi-Newton search direction.
///
/// Uses Cholesky factorisation of the (assumed SPD) Hessian approximation.
/// Falls back to damped identity if factorisation fails (i.e. H is not SPD).
pub fn compute_quasi_newton_direction<T>(
    hessian_approx: &crate::types::DMatrix<T>,
    gradient: &DVector<T>,
    _workspace: &mut Workspace<T>,
    direction: &mut DVector<T>,
) -> Result<()>
where
    T: Scalar,
{
    let n = gradient.len();
    debug_assert_eq!(hessian_approx.nrows(), n);
    debug_assert_eq!(hessian_approx.ncols(), n);

    // Try Cholesky on H (SPD assumption)
    if let Some(chol) = nalgebra::linalg::Cholesky::new(hessian_approx.clone()) {
        // Solve  H d = −g  ⟹  d = −H⁻¹ g
        let neg_g = -gradient;
        let sol = chol.solve(&neg_g);
        direction.copy_from(&sol);
    } else {
        // Cholesky failed — H is not SPD.
        // Regularise: solve  (H + λI) d = −g  with increasing λ until SPD.
        let mut lambda = T::from(1e-6).expect("constant");
        let max_tries = 10;
        let mut h_reg = hessian_approx.clone();

        for _ in 0..max_tries {
            // H_reg = H + λ I
            h_reg.copy_from(hessian_approx);
            for i in 0..n {
                h_reg[(i, i)] = h_reg[(i, i)] + lambda;
            }

            if let Some(chol) = nalgebra::linalg::Cholesky::new(h_reg.clone()) {
                let neg_g = -gradient;
                let sol = chol.solve(&neg_g);
                direction.copy_from(&sol);
                return Ok(());
            }
            lambda = lambda * T::from(10.0).expect("constant");
        }

        // Last resort: steepest descent
        direction.copy_from(gradient);
        direction.scale_mut(-T::one());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Allocating convenience wrappers
// ---------------------------------------------------------------------------

/// Perform a line search step (allocating version).
pub fn line_search_step_alloc<T, F>(
    cost_fn: &F,
    point: &DVector<T>,
    direction: &DVector<T>,
    initial_step: T,
) -> OptimizerResult<(T, DVector<T>)>
where
    T: Scalar,
    F: Fn(&DVector<T>) -> Result<T>,
{
    let n = point.len();
    let mut workspace = Workspace::with_size(n);
    let mut result_point = DVector::zeros(n);
    let alpha =
        line_search_step(cost_fn, point, direction, initial_step, &mut workspace, &mut result_point)?;
    Ok((alpha, result_point))
}

/// Update momentum buffer (allocating version).
pub fn update_momentum_alloc<T>(
    momentum: &DVector<T>,
    gradient: &DVector<T>,
    beta: T,
) -> Result<DVector<T>>
where
    T: Scalar,
{
    let mut workspace = Workspace::new();
    let mut result = momentum.clone();
    update_momentum(&mut result, gradient, beta, &mut workspace)?;
    Ok(result)
}

/// Update Adam optimizer state (allocating version).
pub fn update_adam_state_alloc<T>(
    momentum: &DVector<T>,
    second_moment: &DVector<T>,
    gradient: &DVector<T>,
    beta1: T,
    beta2: T,
) -> Result<(DVector<T>, DVector<T>)>
where
    T: Scalar,
{
    let mut workspace = Workspace::new();
    let mut new_momentum = momentum.clone();
    let mut new_second_moment = second_moment.clone();
    update_adam_state(
        &mut new_momentum,
        &mut new_second_moment,
        gradient,
        beta1,
        beta2,
        &mut workspace,
    )?;
    Ok((new_momentum, new_second_moment))
}

/// Compute the search direction for quasi-Newton methods (allocating version).
pub fn compute_quasi_newton_direction_alloc<T>(
    hessian_approx: &crate::types::DMatrix<T>,
    gradient: &DVector<T>,
) -> Result<DVector<T>>
where
    T: Scalar,
{
    let mut workspace = Workspace::new();
    let mut direction = DVector::zeros(gradient.len());
    compute_quasi_newton_direction(hessian_approx, gradient, &mut workspace, &mut direction)?;
    Ok(direction)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_line_search_workspace() {
        let n = 3;
        let mut workspace = Workspace::with_size(n);

        let cost_fn = |x: &DVector<f64>| -> Result<f64> { Ok(x.dot(x)) };

        let point = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        let direction = DVector::from_vec(vec![-1.0, -1.0, -1.0]);

        let mut new_point = DVector::<f64>::zeros(n);
        let alpha = line_search_step(
            &cost_fn,
            &point,
            &direction,
            1.0,
            &mut workspace,
            &mut new_point,
        )
        .unwrap();

        assert!(alpha > 0.0);
        assert!(cost_fn(&new_point).unwrap() < cost_fn(&point).unwrap());
    }

    #[test]
    fn test_momentum_update_workspace() {
        let n = 4;
        let mut workspace = Workspace::with_size(n);

        let mut momentum = DVector::<f64>::zeros(n);
        let gradient = DVector::from_element(n, 1.0);

        update_momentum(&mut momentum, &gradient, 0.9, &mut workspace).unwrap();
        for i in 0..n {
            assert_relative_eq!(momentum[i], 0.1, epsilon = 1e-10);
        }

        update_momentum(&mut momentum, &gradient, 0.9, &mut workspace).unwrap();
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

        update_adam_state(
            &mut momentum,
            &mut second_moment,
            &gradient,
            0.9,
            0.999,
            &mut workspace,
        )
        .unwrap();

        for i in 0..n {
            assert_relative_eq!(momentum[i], 0.2, epsilon = 1e-10);
        }
        for i in 0..n {
            assert_relative_eq!(second_moment[i], 0.004, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_quasi_newton_cholesky() {
        // H = [[2, 0], [0, 4]]  (SPD)
        // g = [1, 1]
        // d = -H^{-1} g = [-0.5, -0.25]
        let h =
            crate::types::DMatrix::<f64>::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 4.0]);
        let g = DVector::from_vec(vec![1.0, 1.0]);
        let mut d = DVector::zeros(2);
        let mut ws = Workspace::new();

        compute_quasi_newton_direction(&h, &g, &mut ws, &mut d).unwrap();

        assert_relative_eq!(d[0], -0.5, epsilon = 1e-12);
        assert_relative_eq!(d[1], -0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_quasi_newton_non_spd_fallback() {
        // H = [[1, 0], [0, -1]]  — not SPD, should regularise
        let h = crate::types::DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
        let g = DVector::from_vec(vec![2.0, 3.0]);
        let mut d = DVector::zeros(2);
        let mut ws = Workspace::new();

        compute_quasi_newton_direction(&h, &g, &mut ws, &mut d).unwrap();

        // Direction should be a descent direction: d · g < 0
        assert!(d.dot(&g) < 0.0, "Should produce a descent direction");
    }
}
