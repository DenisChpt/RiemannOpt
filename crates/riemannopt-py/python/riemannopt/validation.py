"""Validation utilities for manifolds and optimization algorithms."""

import numpy as np
from typing import Callable, Optional, Dict, Any, List, Tuple
import warnings
from scipy.optimize import approx_fprime


def validate_manifold(manifold, 
                     test_points: Optional[List[np.ndarray]] = None,
                     n_random_tests: int = 10,
                     tolerance: float = 1e-10,
                     verbose: bool = True) -> Dict[str, Any]:
    """Comprehensive manifold validation.
    
    Args:
        manifold: Manifold to validate
        test_points: Specific points to test (optional)
        n_random_tests: Number of random tests to perform
        tolerance: Numerical tolerance for tests
        verbose: Whether to print validation results
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'passed': True,
        'tests': {},
        'errors': []
    }
    
    # Generate test points
    if test_points is None:
        test_points = []
        for _ in range(n_random_tests):
            try:
                point = manifold.random_point()
                test_points.append(point)
            except Exception as e:
                results['errors'].append(f"Failed to generate random point: {e}")
                results['passed'] = False
    
    if not test_points:
        results['errors'].append("No test points available")
        results['passed'] = False
        return results
    
    # Test 1: Points are on manifold
    results['tests']['points_on_manifold'] = _test_points_on_manifold(
        manifold, test_points, tolerance, verbose
    )
    if not results['tests']['points_on_manifold']['passed']:
        results['passed'] = False
    
    # Test 2: Projection idempotency
    results['tests']['projection_idempotent'] = _test_projection_idempotent(
        manifold, test_points, tolerance, verbose
    )
    if not results['tests']['projection_idempotent']['passed']:
        results['passed'] = False
    
    # Test 3: Tangent space properties
    results['tests']['tangent_space'] = _test_tangent_space_properties(
        manifold, test_points, tolerance, verbose
    )
    if not results['tests']['tangent_space']['passed']:
        results['passed'] = False
    
    # Test 4: Retraction properties
    results['tests']['retraction'] = _test_retraction_properties(
        manifold, test_points, tolerance, verbose
    )
    if not results['tests']['retraction']['passed']:
        results['passed'] = False
    
    # Test 5: Metric properties (if available)
    if hasattr(manifold, 'inner'):
        results['tests']['metric'] = _test_metric_properties(
            manifold, test_points, tolerance, verbose
        )
        if not results['tests']['metric']['passed']:
            results['passed'] = False
    
    if verbose:
        print(f"\nManifold validation: {'PASSED' if results['passed'] else 'FAILED'}")
        if results['errors']:
            print("Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
    
    return results


def _test_points_on_manifold(manifold, test_points: List[np.ndarray], 
                           tolerance: float, verbose: bool) -> Dict[str, Any]:
    """Test that generated points are actually on the manifold."""
    test_result = {'passed': True, 'failures': []}
    
    for i, point in enumerate(test_points):
        try:
            if hasattr(manifold, 'check_point_on_manifold'):
                on_manifold = manifold.check_point_on_manifold(point, tolerance)
            else:
                # Use projection distance as heuristic
                projected = manifold.project(point)
                distance = np.linalg.norm(point - projected)
                on_manifold = distance < tolerance
            
            if not on_manifold:
                test_result['passed'] = False
                test_result['failures'].append(f"Point {i} not on manifold")
                
        except Exception as e:
            test_result['passed'] = False
            test_result['failures'].append(f"Point {i} test failed: {e}")
    
    if verbose and test_result['failures']:
        print(f"Points on manifold test: {len(test_result['failures'])} failures")
    
    return test_result


def _test_projection_idempotent(manifold, test_points: List[np.ndarray],
                               tolerance: float, verbose: bool) -> Dict[str, Any]:
    """Test that projection is idempotent: P(P(x)) = P(x)."""
    test_result = {'passed': True, 'failures': []}
    
    for i, point in enumerate(test_points):
        try:
            projected_once = manifold.project(point)
            projected_twice = manifold.project(projected_once)
            
            error = np.linalg.norm(projected_once - projected_twice)
            if error > tolerance:
                test_result['passed'] = False
                test_result['failures'].append(
                    f"Point {i}: projection not idempotent, error = {error:.2e}"
                )
                
        except Exception as e:
            test_result['passed'] = False
            test_result['failures'].append(f"Point {i} projection test failed: {e}")
    
    if verbose and test_result['failures']:
        print(f"Projection idempotency test: {len(test_result['failures'])} failures")
    
    return test_result


def _test_tangent_space_properties(manifold, test_points: List[np.ndarray],
                                  tolerance: float, verbose: bool) -> Dict[str, Any]:
    """Test tangent space properties."""
    test_result = {'passed': True, 'failures': []}
    
    for i, point in enumerate(test_points):
        try:
            # Generate random tangent vector
            ambient_vector = np.random.randn(*point.shape)
            tangent_vector = manifold.tangent_projection(point, ambient_vector)
            
            # Test 1: Tangent projection is idempotent
            tangent_twice = manifold.tangent_projection(point, tangent_vector)
            error1 = np.linalg.norm(tangent_vector - tangent_twice)
            
            if error1 > tolerance:
                test_result['passed'] = False
                test_result['failures'].append(
                    f"Point {i}: tangent projection not idempotent, error = {error1:.2e}"
                )
            
            # Test 2: Check if vector is in tangent space
            if hasattr(manifold, 'check_vector_in_tangent_space'):
                in_tangent = manifold.check_vector_in_tangent_space(point, tangent_vector, tolerance)
                if not in_tangent:
                    test_result['passed'] = False
                    test_result['failures'].append(
                        f"Point {i}: projected vector not in tangent space"
                    )
            
        except Exception as e:
            test_result['passed'] = False
            test_result['failures'].append(f"Point {i} tangent space test failed: {e}")
    
    if verbose and test_result['failures']:
        print(f"Tangent space test: {len(test_result['failures'])} failures")
    
    return test_result


def _test_retraction_properties(manifold, test_points: List[np.ndarray],
                               tolerance: float, verbose: bool) -> Dict[str, Any]:
    """Test retraction properties."""
    test_result = {'passed': True, 'failures': []}
    
    for i, point in enumerate(test_points):
        try:
            # Test 1: R(x, 0) = x
            zero_vector = np.zeros_like(point)
            retracted_zero = manifold.retract(point, zero_vector)
            error1 = np.linalg.norm(retracted_zero - point)
            
            if error1 > tolerance:
                test_result['passed'] = False
                test_result['failures'].append(
                    f"Point {i}: R(x,0) ≠ x, error = {error1:.2e}"
                )
            
            # Test 2: Retracted point is on manifold
            tangent_vector = np.random.randn(*point.shape) * 0.1  # Small step
            tangent_vector = manifold.tangent_projection(point, tangent_vector)
            retracted = manifold.retract(point, tangent_vector)
            
            if hasattr(manifold, 'check_point_on_manifold'):
                on_manifold = manifold.check_point_on_manifold(retracted, tolerance)
                if not on_manifold:
                    test_result['passed'] = False
                    test_result['failures'].append(
                        f"Point {i}: retracted point not on manifold"
                    )
            
        except Exception as e:
            test_result['passed'] = False
            test_result['failures'].append(f"Point {i} retraction test failed: {e}")
    
    if verbose and test_result['failures']:
        print(f"Retraction test: {len(test_result['failures'])} failures")
    
    return test_result


def _test_metric_properties(manifold, test_points: List[np.ndarray],
                           tolerance: float, verbose: bool) -> Dict[str, Any]:
    """Test Riemannian metric properties."""
    test_result = {'passed': True, 'failures': []}
    
    for i, point in enumerate(test_points):
        try:
            # Generate random tangent vectors
            v1 = np.random.randn(*point.shape) * 0.1
            v2 = np.random.randn(*point.shape) * 0.1
            v1 = manifold.tangent_projection(point, v1)
            v2 = manifold.tangent_projection(point, v2)
            
            # Test 1: Symmetry <v1, v2> = <v2, v1>
            if hasattr(manifold, 'inner'):
                inner12 = manifold.inner(point, v1, v2)
                inner21 = manifold.inner(point, v2, v1)
                error1 = abs(inner12 - inner21)
                
                if error1 > tolerance:
                    test_result['passed'] = False
                    test_result['failures'].append(
                        f"Point {i}: metric not symmetric, error = {error1:.2e}"
                    )
                
                # Test 2: Positive definiteness <v, v> >= 0
                inner_self = manifold.inner(point, v1, v1)
                if inner_self < -tolerance:
                    test_result['passed'] = False
                    test_result['failures'].append(
                        f"Point {i}: metric not positive definite, <v,v> = {inner_self:.2e}"
                    )
            
        except Exception as e:
            test_result['passed'] = False
            test_result['failures'].append(f"Point {i} metric test failed: {e}")
    
    if verbose and test_result['failures']:
        print(f"Metric test: {len(test_result['failures'])} failures")
    
    return test_result


def validate_gradient(cost_function: Callable,
                     gradient_function: Callable,
                     test_points: List[np.ndarray],
                     epsilon: float = 1e-8,
                     tolerance: float = 1e-5,
                     verbose: bool = True) -> Dict[str, Any]:
    """Validate gradient computation using finite differences.
    
    Args:
        cost_function: Function to compute cost
        gradient_function: Function to compute gradient
        test_points: Points to test gradient at
        epsilon: Step size for finite differences
        tolerance: Tolerance for gradient check
        verbose: Whether to print results
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'passed': True,
        'errors': [],
        'relative_errors': [],
        'absolute_errors': []
    }
    
    for i, point in enumerate(test_points):
        try:
            # Compute analytical gradient
            analytical_grad = gradient_function(point)
            
            # Compute numerical gradient
            def scalar_function(x):
                return cost_function(x)
            
            numerical_grad = approx_fprime(point, scalar_function, epsilon)
            
            # Compute errors
            absolute_error = np.linalg.norm(analytical_grad - numerical_grad)
            relative_error = absolute_error / (np.linalg.norm(analytical_grad) + 1e-12)
            
            results['absolute_errors'].append(absolute_error)
            results['relative_errors'].append(relative_error)
            
            if relative_error > tolerance:
                results['passed'] = False
                results['errors'].append(
                    f"Point {i}: gradient check failed, relative error = {relative_error:.2e}"
                )
            
            if verbose:
                print(f"Point {i}: relative error = {relative_error:.2e}, "
                      f"absolute error = {absolute_error:.2e}")
                
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Point {i}: gradient validation failed: {e}")
    
    if verbose:
        if results['passed']:
            print("Gradient validation: PASSED")
        else:
            print("Gradient validation: FAILED")
            for error in results['errors']:
                print(f"  - {error}")
    
    return results


def validate_optimizer(optimizer_class,
                      manifold,
                      cost_function: Callable,
                      gradient_function: Optional[Callable] = None,
                      test_problems: Optional[List[Dict]] = None,
                      verbose: bool = True) -> Dict[str, Any]:
    """Validate optimizer on test problems.
    
    Args:
        optimizer_class: Optimizer class to test
        manifold: Manifold to optimize on
        cost_function: Cost function to minimize
        gradient_function: Gradient function (optional)
        test_problems: List of test problem configurations
        verbose: Whether to print results
        
    Returns:
        Dictionary with validation results
    """
    if test_problems is None:
        test_problems = [
            {'learning_rate': 0.01, 'max_iterations': 100},
            {'learning_rate': 0.1, 'max_iterations': 100},
            {'learning_rate': 0.001, 'max_iterations': 200}
        ]
    
    results = {
        'passed': True,
        'test_results': [],
        'errors': []
    }
    
    for i, problem_config in enumerate(test_problems):
        try:
            # Create optimizer
            optimizer = optimizer_class(manifold, **problem_config)
            
            # Create cost function
            from . import create_cost_function
            if gradient_function:
                cost_fn = create_cost_function(cost_function, gradient_function)
            else:
                cost_fn = create_cost_function(cost_function)
            
            # Random initial point
            x0 = manifold.random_point()
            initial_cost = cost_function(x0)
            
            # Optimize
            result = optimizer.optimize(cost_fn, x0)
            
            # Analyze result
            final_cost = result['value']
            improvement = initial_cost - final_cost
            converged = result.get('converged', False)
            
            test_result = {
                'problem_index': i,
                'config': problem_config,
                'initial_cost': initial_cost,
                'final_cost': final_cost,
                'improvement': improvement,
                'converged': converged,
                'iterations': result.get('iterations', 0)
            }
            
            results['test_results'].append(test_result)
            
            # Check if optimization made progress
            if improvement < 1e-12 and not converged:
                results['passed'] = False
                results['errors'].append(
                    f"Test {i}: No improvement achieved (initial: {initial_cost:.2e}, "
                    f"final: {final_cost:.2e})"
                )
            
            if verbose:
                print(f"Test {i}: {improvement:.2e} improvement, "
                      f"converged: {converged}, iterations: {test_result['iterations']}")
                
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Test {i}: optimization failed: {e}")
    
    if verbose:
        if results['passed']:
            print("Optimizer validation: PASSED")
        else:
            print("Optimizer validation: FAILED")
            for error in results['errors']:
                print(f"  - {error}")
    
    return results


def compare_optimizers(optimizer_configs: List[Dict],
                      manifold,
                      cost_function: Callable,
                      gradient_function: Optional[Callable] = None,
                      n_trials: int = 5,
                      max_iterations: int = 1000,
                      verbose: bool = True) -> Dict[str, Any]:
    """Compare multiple optimizers on the same problem.
    
    Args:
        optimizer_configs: List of optimizer configurations
        manifold: Manifold to optimize on
        cost_function: Cost function to minimize
        gradient_function: Gradient function (optional)
        n_trials: Number of trials per optimizer
        max_iterations: Maximum iterations per trial
        verbose: Whether to print results
        
    Returns:
        Dictionary with comparison results
    """
    from . import create_cost_function
    import time
    
    if gradient_function:
        cost_fn = create_cost_function(cost_function, gradient_function)
    else:
        cost_fn = create_cost_function(cost_function)
    
    results = {
        'optimizer_results': {},
        'summary': {}
    }
    
    for config in optimizer_configs:
        optimizer_name = config['name']
        optimizer_class = config['class']
        optimizer_params = config.get('params', {})
        
        trial_results = []
        
        for trial in range(n_trials):
            try:
                # Create optimizer
                optimizer = optimizer_class(manifold, 
                                          max_iterations=max_iterations, 
                                          **optimizer_params)
                
                # Random initial point (same seed for fair comparison)
                np.random.seed(42 + trial)
                x0 = manifold.random_point()
                initial_cost = cost_function(x0)
                
                # Time the optimization
                start_time = time.time()
                result = optimizer.optimize(cost_fn, x0)
                end_time = time.time()
                
                trial_result = {
                    'trial': trial,
                    'initial_cost': initial_cost,
                    'final_cost': result['value'],
                    'improvement': initial_cost - result['value'],
                    'converged': result.get('converged', False),
                    'iterations': result.get('iterations', 0),
                    'time': end_time - start_time
                }
                
                trial_results.append(trial_result)
                
            except Exception as e:
                trial_results.append({
                    'trial': trial,
                    'error': str(e)
                })
        
        # Compute statistics
        valid_trials = [r for r in trial_results if 'error' not in r]
        
        if valid_trials:
            improvements = [r['improvement'] for r in valid_trials]
            times = [r['time'] for r in valid_trials]
            iterations = [r['iterations'] for r in valid_trials]
            
            summary = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'mean_iterations': np.mean(iterations),
                'success_rate': len(valid_trials) / n_trials
            }
        else:
            summary = {'success_rate': 0.0}
        
        results['optimizer_results'][optimizer_name] = {
            'trials': trial_results,
            'summary': summary
        }
        
        if verbose:
            print(f"{optimizer_name}:")
            print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
            if 'mean_improvement' in summary:
                print(f"  Mean improvement: {summary['mean_improvement']:.2e} ± {summary['std_improvement']:.2e}")
                print(f"  Mean time: {summary['mean_time']:.3f}s ± {summary['std_time']:.3f}s")
                print(f"  Mean iterations: {summary['mean_iterations']:.1f}")
    
    return results