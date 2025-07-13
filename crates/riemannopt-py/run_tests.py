#!/usr/bin/env python3
"""
Test runner for RiemannOpt Python bindings.

This script runs the complete test suite and generates a comprehensive report
on the quality and performance of the RiemannOpt implementation.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add the package to Python path for testing
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import pytest
    import numpy as np
    import riemannopt as ro
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please ensure pytest, numpy, and riemannopt are installed.")
    sys.exit(1)


class TestRunner:
    """Comprehensive test runner for RiemannOpt."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}
        
    def run_test_suite(self, 
                      include_slow: bool = False,
                      include_integration: bool = True,
                      specific_tests: Optional[List[str]] = None) -> bool:
        """Run the complete test suite."""
        
        print("=" * 80)
        print("RiemannOpt Python Bindings - Test Suite")
        print("=" * 80)
        
        # Environment check
        self._check_environment()
        
        # Test categories
        test_categories = []
        
        if specific_tests:
            test_categories = [(name, f"tests/unit/test_{name}.py") for name in specific_tests]
        else:
            test_categories = [
                ("manifolds", "tests/unit/test_manifolds.py"),
                ("cost_functions", "tests/unit/test_cost_functions.py"),
                ("optimizers", "tests/unit/test_optimizers.py"),
            ]
            
            if include_integration:
                test_categories.append(("integration", "tests/integration/test_end_to_end.py"))
        
        # Run tests
        overall_success = True
        
        for category, test_file in test_categories:
            print(f"\n{'-' * 60}")
            print(f"Running {category} tests...")
            print(f"{'-' * 60}")
            
            success = self._run_single_test_file(category, test_file, include_slow)
            overall_success = overall_success and success
            
            if not success and not self.verbose:
                print(f"❌ {category} tests failed!")
            
        # Generate summary report
        self._generate_summary_report()
        
        return overall_success
    
    def _check_environment(self):
        """Check that the environment is properly set up."""
        print("🔍 Checking environment...")
        
        # Check Python version
        python_version = sys.version_info
        print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("   ⚠️  Warning: Python 3.8+ recommended")
        
        # Check package versions
        try:
            import numpy
            print(f"   NumPy version: {numpy.__version__}")
        except ImportError:
            print("   ❌ NumPy not available")
            return False
        
        try:
            import riemannopt
            print(f"   RiemannOpt version: {getattr(riemannopt, '__version__', 'unknown')}")
        except ImportError:
            print("   ❌ RiemannOpt not available")
            return False
        
        # Check if compiled module loads
        try:
            sphere = ro.manifolds.Sphere(3)
            print("   ✅ RiemannOpt manifolds working")
        except Exception as e:
            print(f"   ❌ RiemannOpt manifolds error: {e}")
            return False
        
        try:
            cost_fn = ro.create_cost_function(lambda x: np.sum(x**2))
            print("   ✅ RiemannOpt cost functions working")
        except Exception as e:
            print(f"   ❌ RiemannOpt cost functions error: {e}")
            return False
        
        try:
            optimizer = ro.optimizers.SGD(learning_rate=0.1)
            print("   ✅ RiemannOpt optimizers working")
        except Exception as e:
            print(f"   ❌ RiemannOpt optimizers error: {e}")
            return False
        
        print("   ✅ Environment check passed")
        return True
    
    def _run_single_test_file(self, category: str, test_file: str, include_slow: bool) -> bool:
        """Run a single test file and collect results."""
        
        if not os.path.exists(test_file):
            print(f"   ⚠️  Test file not found: {test_file}")
            self.results[category] = {
                'status': 'skipped',
                'reason': 'file not found',
                'duration': 0,
                'tests_run': 0,
                'failures': 0,
                'errors': 0
            }
            return True
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", test_file, "-v"]
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        if self.verbose:
            cmd.append("--tb=short")
        else:
            cmd.append("-q")
        
        # Add coverage if available
        try:
            import coverage
            cmd.extend(["--cov=riemannopt", "--cov-report=term-missing"])
        except ImportError:
            pass
        
        # Run tests
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, 
                                  capture_output=not self.verbose, 
                                  text=True, 
                                  timeout=300)  # 5 minute timeout
            
            duration = time.time() - start_time
            
            # Parse results
            success = result.returncode == 0
            
            # Store results
            self.results[category] = {
                'status': 'passed' if success else 'failed',
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout if not self.verbose else '',
                'stderr': result.stderr if not self.verbose else ''
            }
            
            if success:
                print(f"   ✅ {category} tests passed ({duration:.2f}s)")
            else:
                print(f"   ❌ {category} tests failed ({duration:.2f}s)")
                if not self.verbose and result.stderr:
                    print(f"   Error output: {result.stderr[:200]}...")
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {category} tests timed out")
            self.results[category] = {
                'status': 'timeout',
                'duration': 300,
                'tests_run': 0,
                'failures': 0,
                'errors': 1
            }
            return False
            
        except Exception as e:
            print(f"   ❌ Error running {category} tests: {e}")
            self.results[category] = {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time
            }
            return False
    
    def _generate_summary_report(self):
        """Generate a comprehensive summary report."""
        
        print(f"\n{'=' * 80}")
        print("TEST SUMMARY REPORT")
        print(f"{'=' * 80}")
        
        total_duration = sum(r.get('duration', 0) for r in self.results.values())
        
        print(f"Total test duration: {total_duration:.2f}s")
        print(f"Test categories: {len(self.results)}")
        
        # Status summary
        statuses = {}
        for category, result in self.results.items():
            status = result.get('status', 'unknown')
            statuses[status] = statuses.get(status, 0) + 1
        
        print(f"\nStatus Summary:")
        for status, count in statuses.items():
            emoji = {
                'passed': '✅',
                'failed': '❌', 
                'timeout': '⏰',
                'skipped': '⚠️',
                'error': '💥'
            }.get(status, '❓')
            print(f"   {emoji} {status.capitalize()}: {count}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for category, result in self.results.items():
            status = result.get('status', 'unknown')
            duration = result.get('duration', 0)
            
            emoji = {
                'passed': '✅',
                'failed': '❌',
                'timeout': '⏰', 
                'skipped': '⚠️',
                'error': '💥'
            }.get(status, '❓')
            
            print(f"   {emoji} {category:<20} {status:<10} ({duration:.2f}s)")
            
            if status == 'failed' and not self.verbose:
                stderr = result.get('stderr', '')
                if stderr:
                    lines = stderr.split('\n')[:3]  # First 3 lines
                    for line in lines:
                        if line.strip():
                            print(f"      {line[:70]}...")
        
        # Overall status
        overall_success = all(r.get('status') == 'passed' for r in self.results.values())
        
        print(f"\n{'=' * 80}")
        if overall_success:
            print("🎉 ALL TESTS PASSED! RiemannOpt is working correctly.")
        else:
            print("❌ SOME TESTS FAILED. Please review the errors above.")
        print(f"{'=' * 80}")
        
        return overall_success
    
    def run_performance_benchmark(self):
        """Run performance benchmarks."""
        
        print(f"\n{'=' * 80}")
        print("PERFORMANCE BENCHMARKS")
        print(f"{'=' * 80}")
        
        benchmarks = [
            ("Sphere Manifold Operations", self._benchmark_sphere_manifold),
            ("Cost Function Evaluation", self._benchmark_cost_function),
            ("Optimization Performance", self._benchmark_optimization),
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\n{'-' * 40}")
            print(f"Benchmark: {name}")
            print(f"{'-' * 40}")
            
            try:
                result = benchmark_func()
                self._print_benchmark_result(result)
            except Exception as e:
                print(f"❌ Benchmark failed: {e}")
    
    def _benchmark_sphere_manifold(self) -> Dict:
        """Benchmark sphere manifold operations."""
        
        sphere = ro.manifolds.Sphere(1000)
        
        # Benchmark different operations
        operations = {}
        
        # Random point generation
        start = time.perf_counter()
        for _ in range(100):
            point = sphere.random_point()
        operations['random_point'] = (time.perf_counter() - start) / 100
        
        # Projection
        x = np.random.randn(1000)
        start = time.perf_counter()
        for _ in range(100):
            projected = sphere.project(x)
        operations['project'] = (time.perf_counter() - start) / 100
        
        # Tangent projection
        point = sphere.random_point()
        v = np.random.randn(1000)
        start = time.perf_counter()
        for _ in range(100):
            tangent = sphere.project_tangent(point, v)
        operations['project_tangent'] = (time.perf_counter() - start) / 100
        
        return {
            'manifold': 'Sphere(1000)',
            'operations': operations
        }
    
    def _benchmark_cost_function(self) -> Dict:
        """Benchmark cost function evaluation."""
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2) + 0.1 * np.sum(np.sin(x))
            grad = x + 0.1 * np.cos(x)
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        x = np.random.randn(1000)
        
        operations = {}
        
        # Cost evaluation
        start = time.perf_counter()
        for _ in range(1000):
            cost = cost_fn.cost(x)
        operations['cost_evaluation'] = (time.perf_counter() - start) / 1000
        
        # Gradient evaluation
        start = time.perf_counter()
        for _ in range(1000):
            grad = cost_fn.gradient(x)
        operations['gradient_evaluation'] = (time.perf_counter() - start) / 1000
        
        # Combined evaluation
        start = time.perf_counter()
        for _ in range(1000):
            cost, grad = cost_fn.cost_and_gradient(x)
        operations['combined_evaluation'] = (time.perf_counter() - start) / 1000
        
        return {
            'problem': 'Quadratic + Trigonometric (n=1000)',
            'operations': operations
        }
    
    def _benchmark_optimization(self) -> Dict:
        """Benchmark optimization performance."""
        
        manifold = ro.manifolds.Sphere(100)
        
        # Simple quadratic problem
        A = np.diag(np.linspace(10, 0.1, 100))
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        optimizers = {
            'SGD': ro.optimizers.SGD(learning_rate=0.1),
            'Adam': ro.optimizers.Adam(learning_rate=0.1),
        }
        
        results = {}
        
        for name, optimizer in optimizers.items():
            x0 = manifold.random_point()
            
            start = time.perf_counter()
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=100
            )
            duration = time.perf_counter() - start
            
            results[name] = {
                'duration': duration,
                'iterations': result['iterations'],
                'final_cost': result['cost'],
                'converged': result['converged']
            }
        
        return {
            'problem': 'Rayleigh Quotient (n=100)',
            'optimizers': results
        }
    
    def _print_benchmark_result(self, result: Dict):
        """Print benchmark results."""
        
        if 'operations' in result:
            print(f"Problem: {result.get('manifold', result.get('problem', 'Unknown'))}")
            for op, time_per_op in result['operations'].items():
                print(f"   {op:<20}: {time_per_op*1000:.3f} ms")
                
        elif 'optimizers' in result:
            print(f"Problem: {result['problem']}")
            for name, res in result['optimizers'].items():
                print(f"   {name:<10}: {res['duration']:.3f}s "
                      f"({res['iterations']} iter, "
                      f"cost={res['final_cost']:.6f}, "
                      f"converged={res['converged']})")


def main():
    """Main test runner entry point."""
    
    parser = argparse.ArgumentParser(description="RiemannOpt Test Runner")
    parser.add_argument("--slow", action="store_true", 
                       help="Include slow tests")
    parser.add_argument("--no-integration", action="store_true",
                       help="Skip integration tests")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--quiet", action="store_true",
                       help="Quiet output")
    parser.add_argument("--tests", nargs="+",
                       help="Run specific test files")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(verbose=not args.quiet)
    
    # Run tests
    success = runner.run_test_suite(
        include_slow=args.slow,
        include_integration=not args.no_integration,
        specific_tests=args.tests
    )
    
    # Run benchmarks if requested
    if args.benchmark:
        runner.run_performance_benchmark()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()