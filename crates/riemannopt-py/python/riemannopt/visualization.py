"""Visualization utilities for Riemannian manifolds and optimization."""

import numpy as np
from typing import Optional, Callable, List, Tuple, Dict, Any
import warnings


def plot_sphere_optimization(
    sphere,
    cost_function: Callable,
    result: Dict[str, Any],
    trajectory: Optional[List[np.ndarray]] = None,
    ax=None,
    title: str = "Sphere Optimization"
):
    """Plot optimization on a 2D or 3D sphere.
    
    Args:
        sphere: Sphere manifold (must be 2D or 3D)
        cost_function: Cost function for visualization
        result: Optimization result
        trajectory: List of points along optimization path
        ax: Matplotlib axis (created if None)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    if sphere.ambient_dim not in [2, 3]:
        print(f"Cannot visualize {sphere.ambient_dim}D sphere")
        return
    
    if ax is None:
        if sphere.ambient_dim == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create sphere surface
    if sphere.ambient_dim == 3:
        # 3D sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot sphere surface
        ax.plot_surface(x, y, z, alpha=0.3, color='lightblue')
        
        # Plot cost function on sphere
        n_points = 100
        points = []
        values = []
        for _ in range(n_points):
            p = sphere.random_point()
            points.append(p)
            values.append(cost_function(p))
        
        points = np.array(points)
        values = np.array(values)
        
        # Color points by cost function value
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=values, cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax, shrink=0.5)
        
        # Plot optimization result
        final_point = result['point']
        ax.scatter(*final_point, color='red', s=100, label='Optimum')
        
        # Plot trajectory if available
        if trajectory:
            traj = np.array(trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', linewidth=2, alpha=0.8)
            ax.scatter(*trajectory[0], color='green', s=100, label='Start')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    else:
        # 2D circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Plot circle
        ax.plot(x, y, 'b-', alpha=0.5, linewidth=2)
        
        # Plot cost function values around circle
        n_points = 50
        theta_points = np.linspace(0, 2 * np.pi, n_points)
        points = np.array([[np.cos(t), np.sin(t)] for t in theta_points])
        values = [cost_function(p) for p in points]
        
        # Color circle by cost function
        for i in range(len(points) - 1):
            color_intensity = (values[i] - min(values)) / (max(values) - min(values) + 1e-12)
            ax.plot([points[i, 0], points[i+1, 0]], 
                   [points[i, 1], points[i+1, 1]], 
                   color=plt.cm.viridis(color_intensity), linewidth=4)
        
        # Plot optimization result
        final_point = result['point']
        ax.scatter(*final_point, color='red', s=100, label='Optimum', zorder=5)
        
        # Plot trajectory
        if trajectory:
            traj = np.array(trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2, alpha=0.8, zorder=4)
            ax.scatter(*trajectory[0], color='green', s=100, label='Start', zorder=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
    
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_stiefel_columns(
    stiefel,
    point: np.ndarray,
    title: str = "Stiefel Manifold Point"
):
    """Plot columns of a Stiefel manifold point.
    
    Args:
        stiefel: Stiefel manifold
        point: Point on Stiefel manifold (n x p matrix)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available")
        return
    
    n, p = stiefel.n, stiefel.p
    
    fig, axes = plt.subplots(1, p, figsize=(4*p, 4))
    if p == 1:
        axes = [axes]
    
    for i in range(p):
        axes[i].plot(point[:, i], 'o-', linewidth=2, markersize=6)
        axes[i].set_title(f'Column {i+1}')
        axes[i].set_xlabel('Component')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_grassmann_subspace(
    grassmann,
    point: np.ndarray,
    title: str = "Grassmann Manifold Subspace"
):
    """Visualize a subspace on Grassmann manifold.
    
    Args:
        grassmann: Grassmann manifold
        point: Point on Grassmann manifold (basis matrix)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available")
        return
    
    if grassmann.n > 3 or grassmann.p > 2:
        print(f"Cannot visualize Gr({grassmann.n}, {grassmann.p})")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot basis vectors
    origin = np.zeros(grassmann.n)
    for i in range(grassmann.p):
        ax.quiver(*origin, *point[:, i], color=f'C{i}', arrow_length_ratio=0.1, 
                 linewidth=3, label=f'Basis vector {i+1}')
    
    # Plot span of subspace (if 2D subspace in 3D ambient space)
    if grassmann.n == 3 and grassmann.p == 2:
        # Create grid in subspace
        u = np.linspace(-1, 1, 10)
        v = np.linspace(-1, 1, 10)
        U, V = np.meshgrid(u, v)
        
        X = U.flatten()[:, None] * point[:, 0] + V.flatten()[:, None] * point[:, 1]
        
        X = X.reshape(10, 10, 3)
        ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], 
                       alpha=0.3, color='lightblue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_convergence_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'value',
    title: str = "Optimizer Comparison"
):
    """Compare convergence of different optimizers.
    
    Args:
        results: Dictionary of optimization results from benchmark_optimizers
        metric: Metric to plot ('value', 'gradient_norm', 'time')
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for opt_name, opt_result in results.items():
        if 'error' in opt_result:
            continue
            
        result = opt_result['result']
        iterations = range(result['iterations'])
        
        if metric == 'value':
            # We'd need to store the trajectory for this
            # For now, just plot final value as horizontal line
            ax.axhline(y=result['value'], label=f"{opt_name} (final: {result['value']:.2e})")
        elif metric == 'time':
            ax.bar(opt_name, opt_result['time'], alpha=0.7)
            ax.set_ylabel('Time (seconds)')
        
    ax.set_xlabel('Optimizer' if metric == 'time' else 'Iteration')
    ax.set_ylabel('Cost Function Value' if metric == 'value' else metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if metric != 'time':
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_manifold_tangent_space(
    manifold,
    point: np.ndarray,
    tangent_vectors: List[np.ndarray] = None,
    title: str = "Manifold and Tangent Space"
):
    """Visualize manifold with tangent space at a point.
    
    Args:
        manifold: The manifold
        point: Point on manifold
        tangent_vectors: List of tangent vectors to visualize
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available")
        return
    
    # This is a simplified version for sphere manifolds
    if hasattr(manifold, 'ambient_dim') and manifold.ambient_dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=0.3, color='lightblue')
        
        # Plot point
        ax.scatter(*point, color='red', s=100, label='Point')
        
        # Plot tangent vectors
        if tangent_vectors:
            for i, tangent in enumerate(tangent_vectors):
                ax.quiver(*point, *tangent, color=f'C{i}', 
                         arrow_length_ratio=0.1, linewidth=2,
                         label=f'Tangent {i+1}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        plt.show()
    else:
        print(f"Visualization not implemented for this manifold type")


def create_optimization_animation(
    manifold,
    cost_function: Callable,
    trajectory: List[np.ndarray],
    filename: str = "optimization.gif",
    fps: int = 5
):
    """Create an animation of the optimization process.
    
    Args:
        manifold: The manifold
        cost_function: Cost function
        trajectory: List of points along optimization path
        filename: Output filename
        fps: Frames per second
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("Matplotlib not available")
        return
    
    if not hasattr(manifold, 'ambient_dim') or manifold.ambient_dim not in [2, 3]:
        print("Animation only supported for 2D/3D manifolds")
        return
    
    fig = plt.figure(figsize=(10, 8))
    
    if manifold.ambient_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=0.3, color='lightblue')
        
        # Initialize plots
        point_plot = ax.scatter([], [], [], color='red', s=100)
        path_plot, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7)
        
        def animate(frame):
            if frame < len(trajectory):
                # Update current point
                current_point = trajectory[frame]
                point_plot._offsets3d = ([current_point[0]], [current_point[1]], [current_point[2]])
                
                # Update path
                path_points = np.array(trajectory[:frame+1])
                if len(path_points) > 0:
                    path_plot.set_data_3d(path_points[:, 0], path_points[:, 1], path_points[:, 2])
                
                ax.set_title(f"Optimization Step {frame}")
            
            return point_plot, path_plot
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    else:
        # 2D case
        ax = fig.add_subplot(111)
        
        # Plot circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, 'b-', alpha=0.5, linewidth=2)
        
        point_plot = ax.scatter([], [], color='red', s=100)
        path_plot, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7)
        
        def animate(frame):
            if frame < len(trajectory):
                current_point = trajectory[frame]
                point_plot.set_offsets([current_point])
                
                path_points = np.array(trajectory[:frame+1])
                if len(path_points) > 0:
                    path_plot.set_data(path_points[:, 0], path_points[:, 1])
                
                ax.set_title(f"Optimization Step {frame}")
            
            return point_plot, path_plot
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                                  interval=1000//fps, blit=False, repeat=True)
    
    # Save animation
    try:
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved as {filename}")
    except Exception as e:
        print(f"Could not save animation: {e}")
        plt.show()  # Show interactive plot instead