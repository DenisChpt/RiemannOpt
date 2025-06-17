"""Advanced callback system for RiemannOpt optimization."""

import numpy as np
import time
import json
from typing import Callable, Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
import warnings


class BaseCallback(ABC):
    """Abstract base class for optimization callbacks."""
    
    def __init__(self):
        self.is_active = True
        self.iteration = 0
        self.start_time = None
    
    @abstractmethod
    def on_iteration(self, iteration: int, value: float, gradient_norm: float, 
                    point: Optional[np.ndarray] = None, **kwargs):
        """Called at each optimization iteration."""
        pass
    
    def on_start(self, initial_point: np.ndarray, **kwargs):
        """Called at the start of optimization."""
        self.start_time = time.time()
        self.iteration = 0
    
    def on_end(self, final_result: Dict[str, Any], **kwargs):
        """Called at the end of optimization."""
        pass
    
    def activate(self):
        """Activate this callback."""
        self.is_active = True
    
    def deactivate(self):
        """Deactivate this callback."""
        self.is_active = False
    
    def __call__(self, iteration: int, value: float, gradient_norm: float, 
                 point: Optional[np.ndarray] = None, **kwargs):
        """Make callback callable."""
        if self.is_active:
            self.iteration = iteration
            self.on_iteration(iteration, value, gradient_norm, point, **kwargs)


class CallbackManager:
    """Manage multiple callbacks efficiently."""
    
    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        self.callbacks = callbacks or []
        self.history = {
            'iterations': [],
            'values': [],
            'gradient_norms': [],
            'times': []
        }
    
    def add_callback(self, callback: BaseCallback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: BaseCallback):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def on_start(self, initial_point: np.ndarray, **kwargs):
        """Call on_start for all callbacks."""
        for callback in self.callbacks:
            if callback.is_active:
                callback.on_start(initial_point, **kwargs)
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        """Call on_iteration for all callbacks."""
        current_time = time.time()
        
        # Store in history
        self.history['iterations'].append(iteration)
        self.history['values'].append(value)
        self.history['gradient_norms'].append(gradient_norm)
        self.history['times'].append(current_time)
        
        # Call all active callbacks
        for callback in self.callbacks:
            if callback.is_active:
                callback.on_iteration(iteration, value, gradient_norm, point, **kwargs)
    
    def on_end(self, final_result: Dict[str, Any], **kwargs):
        """Call on_end for all callbacks."""
        for callback in self.callbacks:
            if callback.is_active:
                callback.on_end(final_result, **kwargs)
    
    def __call__(self, iteration: int, value: float, gradient_norm: float,
                 point: Optional[np.ndarray] = None, **kwargs):
        """Make manager callable."""
        self.on_iteration(iteration, value, gradient_norm, point, **kwargs)


class ProgressCallback(BaseCallback):
    """Enhanced progress callback with customizable output."""
    
    def __init__(self, 
                 print_every: int = 10,
                 verbose: bool = True,
                 show_time: bool = True,
                 show_rate: bool = True,
                 precision: int = 6):
        super().__init__()
        self.print_every = print_every
        self.verbose = verbose
        self.show_time = show_time
        self.show_rate = show_rate
        self.precision = precision
        
        self.last_print_time = None
        self.last_print_iteration = 0
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        if not self.verbose or iteration % self.print_every != 0:
            return
        
        current_time = time.time()
        
        message_parts = [f"Iter {iteration:6d}:"]
        message_parts.append(f"f = {value:.{self.precision}e}")
        message_parts.append(f"||grad|| = {gradient_norm:.{self.precision}e}")
        
        if self.show_time and self.start_time:
            elapsed = current_time - self.start_time
            message_parts.append(f"time = {elapsed:.2f}s")
        
        if self.show_rate and self.last_print_time:
            iterations_since = iteration - self.last_print_iteration
            time_since = current_time - self.last_print_time
            rate = iterations_since / time_since if time_since > 0 else 0
            message_parts.append(f"rate = {rate:.1f} it/s")
        
        print(" | ".join(message_parts))
        
        self.last_print_time = current_time
        self.last_print_iteration = iteration


class HistoryCallback(BaseCallback):
    """Callback that stores optimization history."""
    
    def __init__(self, store_points: bool = False, max_points: int = 1000):
        super().__init__()
        self.store_points = store_points
        self.max_points = max_points
        
        self.reset()
    
    def reset(self):
        """Reset the history."""
        self.history = {
            'iterations': [],
            'values': [],
            'gradient_norms': [],
            'times': [],
            'points': [] if self.store_points else None
        }
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        self.history['iterations'].append(iteration)
        self.history['values'].append(value)
        self.history['gradient_norms'].append(gradient_norm)
        self.history['times'].append(time.time())
        
        if self.store_points and point is not None:
            # Store points but limit memory usage
            if len(self.history['points']) >= self.max_points:
                # Keep every other point to reduce memory
                self.history['points'] = self.history['points'][::2]
            self.history['points'].append(point.copy())
    
    def get_convergence_data(self) -> Dict[str, np.ndarray]:
        """Get convergence data as numpy arrays."""
        return {
            'iterations': np.array(self.history['iterations']),
            'values': np.array(self.history['values']),
            'gradient_norms': np.array(self.history['gradient_norms']),
            'times': np.array(self.history['times'])
        }


class EarlyStoppingCallback(BaseCallback):
    """Early stopping based on various criteria."""
    
    def __init__(self,
                 patience: int = 10,
                 min_improvement: float = 1e-8,
                 improvement_type: str = 'relative',
                 monitor: str = 'value',
                 verbose: bool = True):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.improvement_type = improvement_type  # 'relative' or 'absolute'
        self.monitor = monitor  # 'value' or 'gradient_norm'
        self.verbose = verbose
        
        self.best_value = float('inf')
        self.best_gradient_norm = float('inf')
        self.wait = 0
        self.should_stop = False
        self.stopped_iteration = None
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        
        current_metric = value if self.monitor == 'value' else gradient_norm
        best_metric = self.best_value if self.monitor == 'value' else self.best_gradient_norm
        
        # Check for improvement
        if self.improvement_type == 'relative':
            improvement = (best_metric - current_metric) / (abs(best_metric) + 1e-12)
        else:  # absolute
            improvement = best_metric - current_metric
        
        if improvement > self.min_improvement:
            # We have improvement
            if self.monitor == 'value':
                self.best_value = current_metric
            else:
                self.best_gradient_norm = current_metric
            self.wait = 0
        else:
            self.wait += 1
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.should_stop = True
            self.stopped_iteration = iteration
            if self.verbose:
                print(f"\nEarly stopping at iteration {iteration}")
                print(f"No improvement in {self.monitor} for {self.patience} iterations")
                print(f"Best {self.monitor}: {best_metric:.6e}")


class CheckpointCallback(BaseCallback):
    """Save optimization checkpoints."""
    
    def __init__(self,
                 filepath: str = "checkpoint.json",
                 save_every: int = 100,
                 save_best: bool = True,
                 monitor: str = 'value'):
        super().__init__()
        self.filepath = filepath
        self.save_every = save_every
        self.save_best = save_best
        self.monitor = monitor
        
        self.best_value = float('inf')
        self.best_gradient_norm = float('inf')
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        
        should_save = False
        
        # Regular save
        if iteration % self.save_every == 0:
            should_save = True
        
        # Best save
        if self.save_best:
            if self.monitor == 'value' and value < self.best_value:
                self.best_value = value
                should_save = True
            elif self.monitor == 'gradient_norm' and gradient_norm < self.best_gradient_norm:
                self.best_gradient_norm = gradient_norm
                should_save = True
        
        if should_save and point is not None:
            self._save_checkpoint(iteration, value, gradient_norm, point)
    
    def _save_checkpoint(self, iteration: int, value: float, gradient_norm: float, point: np.ndarray):
        """Save checkpoint to file."""
        checkpoint = {
            'iteration': iteration,
            'value': value,
            'gradient_norm': gradient_norm,
            'point': point.tolist(),
            'timestamp': time.time()
        }
        
        try:
            with open(self.filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint: {e}")
    
    @staticmethod
    def load_checkpoint(filepath: str) -> Dict[str, Any]:
        """Load checkpoint from file."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        # Convert point back to numpy array
        checkpoint['point'] = np.array(checkpoint['point'])
        return checkpoint


class MetricsCallback(BaseCallback):
    """Compute and track various optimization metrics."""
    
    def __init__(self, 
                 compute_condition_number: bool = False,
                 compute_spectral_radius: bool = False,
                 manifold: Optional[Any] = None):
        super().__init__()
        self.compute_condition_number = compute_condition_number
        self.compute_spectral_radius = compute_spectral_radius
        self.manifold = manifold
        
        self.metrics_history = {
            'condition_numbers': [],
            'spectral_radii': [],
            'step_sizes': [],
            'angles': []  # Angle between consecutive gradients
        }
        
        self.last_gradient = None
        self.last_point = None
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        
        # Compute angle between gradients
        if 'gradient' in kwargs and self.last_gradient is not None:
            current_grad = kwargs['gradient']
            cos_angle = np.dot(current_grad, self.last_gradient) / (
                np.linalg.norm(current_grad) * np.linalg.norm(self.last_gradient) + 1e-12
            )
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            self.metrics_history['angles'].append(angle)
            self.last_gradient = current_grad.copy()
        elif 'gradient' in kwargs:
            self.last_gradient = kwargs['gradient'].copy()
        
        # Compute step size
        if point is not None and self.last_point is not None:
            if self.manifold:
                # Riemannian distance
                try:
                    step_size = self.manifold.distance(self.last_point, point)
                except:
                    step_size = np.linalg.norm(point - self.last_point)
            else:
                # Euclidean distance
                step_size = np.linalg.norm(point - self.last_point)
            
            self.metrics_history['step_sizes'].append(step_size)
        
        if point is not None:
            self.last_point = point.copy()
        
        # Store computed metrics
        if 'hessian' in kwargs and self.compute_condition_number:
            hessian = kwargs['hessian']
            try:
                eigenvals = np.linalg.eigvals(hessian)
                condition_number = np.max(eigenvals) / np.min(eigenvals[eigenvals > 1e-12])
                self.metrics_history['condition_numbers'].append(condition_number)
            except:
                pass
        
        if 'hessian' in kwargs and self.compute_spectral_radius:
            hessian = kwargs['hessian']
            try:
                eigenvals = np.linalg.eigvals(hessian)
                spectral_radius = np.max(np.abs(eigenvals))
                self.metrics_history['spectral_radii'].append(spectral_radius)
            except:
                pass


class LoggingCallback(BaseCallback):
    """Advanced logging callback with multiple output formats."""
    
    def __init__(self,
                 log_file: Optional[str] = None,
                 log_level: str = 'INFO',
                 log_format: str = 'json',  # 'json', 'csv', 'custom'
                 custom_formatter: Optional[Callable] = None):
        super().__init__()
        self.log_file = log_file
        self.log_level = log_level
        self.log_format = log_format
        self.custom_formatter = custom_formatter
        
        self.log_handle = None
        
        if log_file:
            self.log_handle = open(log_file, 'w')
            if log_format == 'csv':
                self.log_handle.write("iteration,value,gradient_norm,timestamp\n")
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        
        if not self.log_handle:
            return
        
        timestamp = time.time()
        
        if self.log_format == 'json':
            log_entry = {
                'iteration': iteration,
                'value': value,
                'gradient_norm': gradient_norm,
                'timestamp': timestamp
            }
            if point is not None and len(point) <= 10:  # Don't log huge arrays
                log_entry['point'] = point.tolist()
            
            self.log_handle.write(json.dumps(log_entry) + '\n')
        
        elif self.log_format == 'csv':
            self.log_handle.write(f"{iteration},{value},{gradient_norm},{timestamp}\n")
        
        elif self.log_format == 'custom' and self.custom_formatter:
            log_line = self.custom_formatter(iteration, value, gradient_norm, point, **kwargs)
            self.log_handle.write(log_line + '\n')
        
        self.log_handle.flush()
    
    def on_end(self, final_result: Dict[str, Any], **kwargs):
        if self.log_handle:
            self.log_handle.close()


class AdaptiveCallback(BaseCallback):
    """Callback that can adapt optimizer parameters during optimization."""
    
    def __init__(self,
                 adaptation_rules: Dict[str, Dict[str, Any]],
                 optimizer: Optional[Any] = None):
        super().__init__()
        self.adaptation_rules = adaptation_rules
        self.optimizer = optimizer
        
        self.window_size = 10
        self.recent_values = []
        self.recent_gradients = []
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float,
                    point: Optional[np.ndarray] = None, **kwargs):
        
        # Store recent history
        self.recent_values.append(value)
        self.recent_gradients.append(gradient_norm)
        
        if len(self.recent_values) > self.window_size:
            self.recent_values.pop(0)
            self.recent_gradients.pop(0)
        
        # Apply adaptation rules
        if len(self.recent_values) >= self.window_size and self.optimizer:
            self._apply_adaptation_rules(iteration)
    
    def _apply_adaptation_rules(self, iteration: int):
        """Apply adaptation rules to optimizer."""
        for rule_name, rule_config in self.adaptation_rules.items():
            
            if rule_name == 'learning_rate_decay':
                # Decay learning rate if no improvement
                values_trend = np.polyfit(range(len(self.recent_values)), self.recent_values, 1)[0]
                if values_trend >= -rule_config.get('improvement_threshold', 1e-6):
                    decay_factor = rule_config.get('decay_factor', 0.95)
                    if hasattr(self.optimizer, 'learning_rate'):
                        self.optimizer.learning_rate *= decay_factor
                    print(f"Iteration {iteration}: Learning rate decayed to {self.optimizer.learning_rate}")
            
            elif rule_name == 'restart_on_plateau':
                # Restart if stuck on plateau
                recent_std = np.std(self.recent_values)
                if recent_std < rule_config.get('plateau_threshold', 1e-8):
                    if hasattr(self.optimizer, 'reset'):
                        self.optimizer.reset()
                    print(f"Iteration {iteration}: Optimizer restarted due to plateau")