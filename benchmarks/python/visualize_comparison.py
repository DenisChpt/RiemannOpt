#!/usr/bin/env python3
"""
Visualization for library comparison benchmarks.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

def load_results(path):
    """Load benchmark results."""
    with open(path, 'r') as f:
        data = json.load(f)
    # Convert size lists to tuples for hashability
    for item in data:
        if isinstance(item['size'], list):
            item['size'] = tuple(item['size'])
    return pd.DataFrame(data)

def plot_library_comparison(df, manifold, operation, ax=None):
    """Plot comparison between libraries for a specific operation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data
    mask = (df['manifold'] == manifold) & (df['operation'] == operation) & (df['success'] == True)
    data = df[mask]
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{manifold} - {operation}')
        return
    
    # Prepare data for plotting
    libraries = data['library'].unique()
    sizes = sorted(data['size'].unique())
    
    # Color palette
    colors = sns.color_palette("husl", len(libraries))
    
    for i, lib in enumerate(libraries):
        lib_data = data[data['library'] == lib]
        x_vals = []
        y_vals = []
        
        for size in sizes:
            size_data = lib_data[lib_data['size'] == size]
            if not size_data.empty:
                if isinstance(size, tuple):
                    x_vals.append(np.prod(size))
                else:
                    x_vals.append(size)
                y_vals.append(size_data['time_ms'].values[0])
        
        if x_vals:
            ax.plot(x_vals, y_vals, 'o-', label=lib, color=colors[i], 
                   linewidth=2, markersize=8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Problem Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'{manifold.upper()} - {operation}')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_speedup_bars(df, reference='pymanopt'):
    """Plot speedup bars comparing to reference library."""
    df_success = df[df['success'] == True]
    
    if reference not in df_success['library'].unique():
        print(f"Reference library {reference} not found")
        return
    
    # Calculate average speedups
    speedups = {}
    
    for manifold in df_success['manifold'].unique():
        for operation in df_success['operation'].unique():
            mdf = df_success[(df_success['manifold'] == manifold) & 
                           (df_success['operation'] == operation)]
            
            for lib in mdf['library'].unique():
                if lib == reference:
                    continue
                
                speedup_vals = []
                for size in mdf['size'].unique():
                    ref_data = mdf[(mdf['library'] == reference) & (mdf['size'] == size)]
                    lib_data = mdf[(mdf['library'] == lib) & (mdf['size'] == size)]
                    
                    if not ref_data.empty and not lib_data.empty:
                        ref_time = ref_data['time_ms'].values[0]
                        lib_time = lib_data['time_ms'].values[0]
                        if ref_time > 0 and lib_time > 0:
                            speedup = ref_time / lib_time
                            speedup_vals.append(speedup)
                
                if speedup_vals:
                    key = f"{manifold}-{operation}"
                    if lib not in speedups:
                        speedups[lib] = {}
                    speedups[lib][key] = np.mean(speedup_vals)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if speedups:
        # Prepare data for grouped bar chart
        operations = sorted(set(op for lib_speedups in speedups.values() for op in lib_speedups.keys()))
        n_groups = len(operations)
        n_bars = len(speedups)
        bar_width = 0.8 / n_bars
        
        for i, (lib, lib_speedups) in enumerate(speedups.items()):
            positions = np.arange(n_groups) + i * bar_width
            values = [lib_speedups.get(op, 0) for op in operations]
            ax.bar(positions, values, bar_width, label=lib)
        
        ax.set_xlabel('Manifold-Operation')
        ax.set_ylabel(f'Speedup vs {reference}')
        ax.set_title(f'Performance Speedup Comparison (baseline: {reference})')
        ax.set_xticks(np.arange(n_groups) + bar_width * (n_bars - 1) / 2)
        ax.set_xticklabels(operations, rotation=45, ha='right')
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

def plot_operation_heatmap(df):
    """Plot heatmap of operation times across libraries and sizes."""
    df_success = df[df['success'] == True]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    manifolds = df_success['manifold'].unique()[:4]  # Max 4 manifolds
    
    for idx, manifold in enumerate(manifolds):
        ax = axes[idx]
        mdf = df_success[df_success['manifold'] == manifold]
        
        # Create pivot table
        pivot_data = []
        for lib in mdf['library'].unique():
            for size in sorted(mdf['size'].unique()):
                size_str = str(size) if not isinstance(size, tuple) else f"{size[0]}x{size[1]}"
                for op in mdf['operation'].unique():
                    op_data = mdf[(mdf['library'] == lib) & 
                                 (mdf['size'] == size) & 
                                 (mdf['operation'] == op)]
                    if not op_data.empty:
                        pivot_data.append({
                            'Library': lib,
                            'Size': size_str,
                            'Operation': op,
                            'Time': op_data['time_ms'].values[0]
                        })
        
        if pivot_data:
            pivot_df = pd.DataFrame(pivot_data)
            pivot = pivot_df.pivot_table(
                index=['Library', 'Operation'],
                columns='Size',
                values='Time',
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Time (ms)'})
            ax.set_title(f'{manifold.upper()} Manifold')
    
    # Remove empty subplots
    for idx in range(len(manifolds), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Operation Performance Heatmap', fontsize=16)
    plt.tight_layout()

def create_comparison_report(df):
    """Create comprehensive comparison visualizations."""
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating comparison visualizations...")
    
    # 1. Individual operation comparisons
    df_success = df[df['success'] == True]
    
    for manifold in df_success['manifold'].unique():
        operations = df_success[df_success['manifold'] == manifold]['operation'].unique()
        
        fig, axes = plt.subplots(1, len(operations), figsize=(6*len(operations), 5))
        if len(operations) == 1:
            axes = [axes]
        
        for ax, op in zip(axes, operations):
            plot_library_comparison(df, manifold, op, ax)
        
        plt.suptitle(f'{manifold.upper()} Manifold - Library Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{manifold}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Speedup comparison
    plt.figure(figsize=(12, 8))
    plot_speedup_bars(df)
    plt.savefig(output_dir / "speedup_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Operation heatmap
    plot_operation_heatmap(df)
    plt.savefig(output_dir / "operation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Overall performance summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_times = df_success.groupby('library')['time_ms'].agg(['mean', 'std'])
    avg_times = avg_times.sort_values('mean')
    
    bars = ax.bar(avg_times.index, avg_times['mean'], yerr=avg_times['std'], 
                   capsize=5, color=sns.color_palette("husl", len(avg_times)))
    ax.set_xlabel('Library')
    ax.set_ylabel('Average Time (ms)')
    ax.set_title('Overall Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, (lib, row) in zip(bars, avg_times.iterrows()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def print_comparison_summary(df):
    """Print detailed comparison summary."""
    df_success = df[df['success'] == True]
    
    if df_success.empty:
        print("No successful operations to analyze")
        return
    
    print("\n" + "="*60)
    print("LIBRARY COMPARISON ANALYSIS")
    print("="*60)
    
    # Winner count
    winner_count = {}
    
    for manifold in df_success['manifold'].unique():
        for operation in df_success['operation'].unique():
            for size in df_success['size'].unique():
                subset = df_success[
                    (df_success['manifold'] == manifold) &
                    (df_success['operation'] == operation) &
                    (df_success['size'] == size)
                ]
                
                if not subset.empty:
                    fastest_lib = subset.loc[subset['time_ms'].idxmin(), 'library']
                    winner_count[fastest_lib] = winner_count.get(fastest_lib, 0) + 1
    
    print("\nFastest library by operation count:")
    for lib, count in sorted(winner_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lib}: {count} operations")
    
    # Average performance
    print("\nAverage performance across all operations:")
    avg_perf = df_success.groupby('library')['time_ms'].agg(['mean', 'std', 'min', 'max'])
    print(avg_perf.round(3))
    
    # Scaling analysis
    print("\nScaling characteristics:")
    for manifold in df_success['manifold'].unique():
        print(f"\n{manifold.upper()}:")
        mdf = df_success[df_success['manifold'] == manifold]
        
        for lib in mdf['library'].unique():
            lib_df = mdf[mdf['library'] == lib]
            
            # Simple scaling estimate
            sizes = []
            times = []
            
            for _, row in lib_df.iterrows():
                size = row['size']
                if isinstance(size, tuple):
                    sizes.append(np.prod(size))
                else:
                    sizes.append(size)
                times.append(row['time_ms'])
            
            if len(sizes) > 2:
                # Log-log regression
                log_sizes = np.log(sizes)
                log_times = np.log(times)
                try:
                    slope, _ = np.polyfit(log_sizes, log_times, 1)
                    print(f"  {lib}: O(n^{slope:.2f})")
                except:
                    print(f"  {lib}: Unable to determine scaling")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_comparison.py <results.json>")
        sys.exit(1)
    
    results_path = sys.argv[1]
    df = load_results(results_path)
    
    # Generate visualizations
    create_comparison_report(df)
    
    # Print summary
    print_comparison_summary(df)

if __name__ == "__main__":
    main()