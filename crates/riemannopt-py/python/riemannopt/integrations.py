"""Integration with popular Python libraries (PyTorch, JAX, etc.)."""

import numpy as np
from typing import Union, Optional, Callable, Any, Dict, Tuple
import warnings


# PyTorch integration
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    
# Type hints that work even without libraries
if _HAS_TORCH:
    from torch.nn import Module as TorchModule
else:
    TorchModule = Any

# JAX integration
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None
    jnp = None

# TensorFlow integration
try:
    import tensorflow as tf
    _HAS_TENSORFLOW = True
except ImportError:
    _HAS_TENSORFLOW = False
    tf = None

# Autograd integration
try:
    import autograd
    import autograd.numpy as anp
    _HAS_AUTOGRAD = True
except ImportError:
    _HAS_AUTOGRAD = False
    autograd = None
    anp = None


class TensorConverter:
    """Universal tensor converter for different frameworks."""
    
    @staticmethod
    def to_numpy(tensor) -> np.ndarray:
        """Convert any tensor type to numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        
        if _HAS_JAX and isinstance(tensor, jnp.ndarray):
            return np.array(tensor)
        
        if _HAS_TENSORFLOW and tf.is_tensor(tensor):
            return tensor.numpy()
        
        # Try to convert as numpy array
        try:
            return np.asarray(tensor)
        except Exception as e:
            raise ValueError(f"Cannot convert tensor to numpy: {e}")
    
    @staticmethod
    def from_numpy(array: np.ndarray, target_type: str = "numpy"):
        """Convert numpy array to target tensor type."""
        if target_type == "numpy":
            return array
        
        if target_type == "torch" and _HAS_TORCH:
            return torch.from_numpy(array).float()
        
        if target_type == "jax" and _HAS_JAX:
            return jnp.array(array)
        
        if target_type == "tensorflow" and _HAS_TENSORFLOW:
            return tf.constant(array, dtype=tf.float32)
        
        raise ValueError(f"Unknown tensor type: {target_type}")


if _HAS_TORCH:
    class PyTorchAdapter:
        """Adapter for PyTorch tensors and autograd."""
        
        def __init__(self):
            if not _HAS_TORCH:
                raise ImportError("PyTorch not available. Install with: pip install torch")
        
        @staticmethod
        def create_cost_function(model: TorchModule, 
                               loss_fn: Callable,
                               data_loader=None,
                               device: Optional[str] = None) -> Callable:
            """Create cost function from PyTorch model.
            
            Args:
                model: PyTorch model
                loss_fn: Loss function
                data_loader: DataLoader for training data (optional)
                device: Device to run on ('cpu', 'cuda', etc.)
                
            Returns:
                Cost function compatible with RiemannOpt
            """
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            model = model.to(device)
            
            def cost_function(params_array: np.ndarray) -> float:
                """Cost function that sets model parameters and computes loss."""
                # Convert numpy to torch
                params_tensor = torch.from_numpy(params_array).float().to(device)
                
                # Set model parameters
                param_idx = 0
                with torch.no_grad():
                    for param in model.parameters():
                        param_size = param.numel()
                        param.data = params_tensor[param_idx:param_idx + param_size].view(param.shape)
                        param_idx += param_size
                
                model.eval()
                total_loss = 0.0
                num_batches = 0
                
                if data_loader is not None:
                    with torch.no_grad():
                        for batch in data_loader:
                            if isinstance(batch, (list, tuple)):
                                inputs, targets = batch[0].to(device), batch[1].to(device)
                            else:
                                inputs, targets = batch.to(device), None
                            
                            outputs = model(inputs)
                            if targets is not None:
                                loss = loss_fn(outputs, targets)
                            else:
                                loss = loss_fn(outputs)
                            
                            total_loss += loss.item()
                            num_batches += 1
                    
                    return total_loss / num_batches if num_batches > 0 else total_loss
                else:
                    # No data loader - assume loss_fn handles everything
                    outputs = model(torch.randn(1, model.input_size if hasattr(model, 'input_size') else 10).to(device))
                    loss = loss_fn(outputs)
                    return loss.item()
            
            return cost_function
    
        @staticmethod
        def create_gradient_function(model: TorchModule,
                                   loss_fn: Callable,
                                   data_loader=None,
                                   device: Optional[str] = None) -> Callable:
            """Create gradient function from PyTorch model."""
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            model = model.to(device)
            
            def gradient_function(params_array: np.ndarray) -> np.ndarray:
                """Gradient function using PyTorch autograd."""
                params_tensor = torch.from_numpy(params_array).float().to(device)
                params_tensor.requires_grad_(True)
                
                # Set model parameters
                param_idx = 0
                for param in model.parameters():
                    param_size = param.numel()
                    param.data = params_tensor[param_idx:param_idx + param_size].view(param.shape)
                    param_idx += param_size
                
                model.train()
                total_loss = 0.0
                
                if data_loader is not None:
                    for batch in data_loader:
                        model.zero_grad()
                        
                        if isinstance(batch, (list, tuple)):
                            inputs, targets = batch[0].to(device), batch[1].to(device)
                        else:
                            inputs, targets = batch.to(device), None
                        
                        outputs = model(inputs)
                        if targets is not None:
                            loss = loss_fn(outputs, targets)
                        else:
                            loss = loss_fn(outputs)
                        
                        loss.backward()
                        total_loss += loss.item()
                
                # Extract gradients
                gradients = []
                for param in model.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.view(-1))
                    else:
                        gradients.append(torch.zeros_like(param).view(-1))
                
                grad_array = torch.cat(gradients).detach().cpu().numpy()
                return grad_array
            
            return gradient_function
    
        @staticmethod
        def parameters_to_array(model: TorchModule) -> np.ndarray:
            """Convert model parameters to flat numpy array."""
            params = []
            for param in model.parameters():
                params.append(param.detach().cpu().numpy().flatten())
            return np.concatenate(params)
    
        @staticmethod
        def array_to_parameters(model: TorchModule, params_array: np.ndarray):
            """Set model parameters from flat numpy array."""
            param_idx = 0
            with torch.no_grad():
                for param in model.parameters():
                    param_size = param.numel()
                    param.data = torch.from_numpy(
                        params_array[param_idx:param_idx + param_size]
                    ).float().view(param.shape).to(param.device)
                    param_idx += param_size


class JAXAdapter:
    """Adapter for JAX arrays and transformations."""
    
    def __init__(self):
        if not _HAS_JAX:
            raise ImportError("JAX not available. Install with: pip install jax jaxlib")
    
    @staticmethod
    def create_cost_function(fun: Callable, 
                           args: Tuple = (),
                           use_jit: bool = True) -> Callable:
        """Create cost function from JAX function.
        
        Args:
            fun: JAX function to optimize
            args: Additional arguments to fun
            use_jit: Whether to JIT compile the function
            
        Returns:
            Cost function compatible with RiemannOpt
        """
        if use_jit:
            fun = jax.jit(fun)
        
        def cost_function(params_array: np.ndarray) -> float:
            """JAX cost function wrapper."""
            params_jax = jnp.array(params_array)
            result = fun(params_jax, *args)
            return float(result)
        
        return cost_function
    
    @staticmethod
    def create_gradient_function(fun: Callable,
                               args: Tuple = (),
                               use_jit: bool = True) -> Callable:
        """Create gradient function using JAX autodiff."""
        grad_fun = jax.grad(fun)
        if use_jit:
            grad_fun = jax.jit(grad_fun)
        
        def gradient_function(params_array: np.ndarray) -> np.ndarray:
            """JAX gradient function wrapper."""
            params_jax = jnp.array(params_array)
            grad_jax = grad_fun(params_jax, *args)
            return np.array(grad_jax)
        
        return gradient_function
    
    @staticmethod
    def create_hessian_function(fun: Callable,
                              args: Tuple = (),
                              use_jit: bool = True) -> Callable:
        """Create Hessian function using JAX."""
        hessian_fun = jax.hessian(fun)
        if use_jit:
            hessian_fun = jax.jit(hessian_fun)
        
        def hessian_function(params_array: np.ndarray) -> np.ndarray:
            """JAX Hessian function wrapper."""
            params_jax = jnp.array(params_array)
            hess_jax = hessian_fun(params_jax, *args)
            return np.array(hess_jax)
        
        return hessian_function


class AutogradAdapter:
    """Adapter for autograd library."""
    
    def __init__(self):
        if not _HAS_AUTOGRAD:
            raise ImportError("Autograd not available. Install with: pip install autograd")
    
    @staticmethod
    def create_gradient_function(fun: Callable) -> Callable:
        """Create gradient function using autograd."""
        grad_fun = autograd.grad(fun)
        
        def gradient_function(params_array: np.ndarray) -> np.ndarray:
            """Autograd gradient function wrapper."""
            return grad_fun(params_array)
        
        return gradient_function
    
    @staticmethod
    def create_hessian_function(fun: Callable) -> Callable:
        """Create Hessian function using autograd."""
        hessian_fun = autograd.hessian(fun)
        
        def hessian_function(params_array: np.ndarray) -> np.ndarray:
            """Autograd Hessian function wrapper."""
            return hessian_fun(params_array)
        
        return hessian_function


def optimize_pytorch_model(manifold,
                          model: TorchModule,
                          loss_fn: Callable,
                          data_loader=None,
                          optimizer: str = 'sgd',
                          device: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
    """Optimize PyTorch model on a Riemannian manifold.
    
    Args:
        manifold: Riemannian manifold for parameters
        model: PyTorch model to optimize
        loss_fn: Loss function
        data_loader: Training data loader
        optimizer: Optimizer type
        device: Device to run on
        **kwargs: Optimizer parameters
        
    Returns:
        Optimization result
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch not available")
    
    from . import optimize, create_cost_function
    
    adapter = PyTorchAdapter()
    
    # Create cost and gradient functions
    cost_fn = adapter.create_cost_function(model, loss_fn, data_loader, device)
    grad_fn = adapter.create_gradient_function(model, loss_fn, data_loader, device)
    
    # Get initial parameters
    initial_params = adapter.parameters_to_array(model)
    
    # Create RiemannOpt cost function
    riemannopt_cost = create_cost_function(cost_fn, grad_fn)
    
    # Optimize
    result = optimize(manifold, riemannopt_cost, initial_params, 
                     optimizer=optimizer, **kwargs)
    
    # Set final parameters back to model
    adapter.array_to_parameters(model, result['point'])
    
    return result


def optimize_jax_function(manifold,
                         fun: Callable,
                         initial_params: np.ndarray,
                         args: Tuple = (),
                         optimizer: str = 'adam',
                         use_jit: bool = True,
                         **kwargs) -> Dict[str, Any]:
    """Optimize JAX function on a Riemannian manifold.
    
    Args:
        manifold: Riemannian manifold
        fun: JAX function to optimize
        initial_params: Initial parameters
        args: Additional arguments to fun
        optimizer: Optimizer type
        use_jit: Whether to use JIT compilation
        **kwargs: Optimizer parameters
        
    Returns:
        Optimization result
    """
    if not _HAS_JAX:
        raise ImportError("JAX not available")
    
    from . import optimize, create_cost_function
    
    adapter = JAXAdapter()
    
    # Create cost and gradient functions
    cost_fn = adapter.create_cost_function(fun, args, use_jit)
    grad_fn = adapter.create_gradient_function(fun, args, use_jit)
    
    # Create RiemannOpt cost function
    riemannopt_cost = create_cost_function(cost_fn, grad_fn)
    
    # Optimize
    result = optimize(manifold, riemannopt_cost, initial_params,
                     optimizer=optimizer, **kwargs)
    
    return result


class ProgressReporter:
    """Enhanced progress reporting with multiple backends."""
    
    def __init__(self, 
                 use_tqdm: bool = True,
                 use_wandb: bool = False,
                 use_tensorboard: bool = False,
                 project_name: str = "riemannopt"):
        self.use_tqdm = use_tqdm and self._has_tqdm()
        self.use_wandb = use_wandb and self._has_wandb()
        self.use_tensorboard = use_tensorboard and self._has_tensorboard()
        self.project_name = project_name
        
        self.pbar = None
        self.step = 0
        
        if self.use_wandb:
            import wandb
            wandb.init(project=project_name)
        
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
    
    def __call__(self, iteration: int, value: float, gradient_norm: float):
        """Progress callback."""
        self.step = iteration
        
        if self.use_tqdm:
            if self.pbar is None:
                try:
                    from tqdm import tqdm
                    self.pbar = tqdm(desc="Optimization")
                except ImportError:
                    pass
            
            if self.pbar:
                self.pbar.set_postfix({
                    'loss': f'{value:.6e}',
                    'grad_norm': f'{gradient_norm:.6e}'
                })
                self.pbar.update(1)
        
        if self.use_wandb:
            import wandb
            wandb.log({
                'loss': value,
                'gradient_norm': gradient_norm,
                'iteration': iteration
            })
        
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/train', value, iteration)
            self.writer.add_scalar('Gradient/norm', gradient_norm, iteration)
    
    def close(self):
        """Clean up resources."""
        if self.pbar:
            self.pbar.close()
        
        if self.use_tensorboard and hasattr(self, 'writer'):
            self.writer.close()
    
    @staticmethod
    def _has_tqdm() -> bool:
        try:
            import tqdm
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _has_wandb() -> bool:
        try:
            import wandb
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _has_tensorboard() -> bool:
        try:
            from torch.utils.tensorboard import SummaryWriter
            return True
        except ImportError:
            return False


def create_sklearn_compatible_optimizer(manifold, optimizer_class, **optimizer_kwargs):
    """Create scikit-learn compatible optimizer.
    
    Returns an optimizer that follows sklearn's fit/predict interface.
    """
    class SklearnOptimizer:
        def __init__(self, manifold, optimizer_class, **kwargs):
            self.manifold = manifold
            self.optimizer_class = optimizer_class
            self.optimizer_kwargs = kwargs
            self.optimizer_ = None
            self.result_ = None
        
        def fit(self, X, y=None, cost_function=None, initial_params=None):
            """Fit the optimizer."""
            if cost_function is None:
                raise ValueError("cost_function must be provided")
            
            if initial_params is None:
                initial_params = self.manifold.random_point()
            
            from . import create_cost_function
            cost_fn = create_cost_function(cost_function)
            
            self.optimizer_ = self.optimizer_class(self.manifold, **self.optimizer_kwargs)
            self.result_ = self.optimizer_.optimize(cost_fn, initial_params)
            
            return self
        
        def predict(self, X=None):
            """Return optimized parameters."""
            if self.result_ is None:
                raise ValueError("Must call fit first")
            return self.result_['point']
        
        def get_params(self, deep=True):
            """Get parameters for this estimator."""
            return self.optimizer_kwargs.copy()
        
        def set_params(self, **params):
            """Set parameters for this estimator."""
            self.optimizer_kwargs.update(params)
            return self
    
    return SklearnOptimizer(manifold, optimizer_class, **optimizer_kwargs)