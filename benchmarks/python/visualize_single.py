#!/usr/bin/env python3
"""
Simple visualization for single library benchmark results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_results(path):
    """Load benchmark results."""
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_manifold_operations(df, manifold):
    """Plot all operations for a specific manifold."""
    mdf = df[df['manifold'] == manifold]
    if mdf.empty:
        return
    
    operations = mdf['operation'].unique()
    fig, axes = plt.subplots(1, len(operations), figsize=(5*len(operations), 4))
    if len(operations) == 1:
        axes = [axes]
    
    for ax, op in zip(axes, operations):
        op_df = mdf[mdf['operation'] == op].sort_values('size')
        
        # Extract sizes
        sizes = []
        for s in op_df['size']:
            if isinstance(s, tuple):
                sizes.append(np.prod(s))
            else:
                sizes.append(s)
        
        ax.plot(sizes, op_df['time_ms'], 'o-', linewidth=2, markersize=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{manifold} - {op}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{manifold.upper()} Manifold Performance', fontsize=16)
    plt.tight_layout()
    return fig

def plot_scaling_comparison(df):
    """Plot scaling comparison across manifolds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(df['manifold'].unique())))
    
    for i, manifold in enumerate(df['manifold'].unique()):
        mdf = df[df['manifold'] == manifold]
        
        # Get projection operation as representative
        proj_df = mdf[mdf['operation'].str.contains('projection')]
        if proj_df.empty:
            continue
        
        proj_df = proj_df.sort_values('size')
        
        # Extract sizes
        sizes = []
        labels = []
        for s in proj_df['size']:
            if isinstance(s, tuple):
                sizes.append(np.prod(s))
                labels.append(f"{s[0]}×{s[1]}")
            else:
                sizes.append(s)
                labels.append(str(s))
        
        ax.plot(sizes, proj_df['time_ms'], 'o-', label=manifold, 
                color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Problem Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Projection Operation Scaling Across Manifolds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_performance_summary(df):
    """Print performance summary."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Group by manifold
    for manifold in df['manifold'].unique():
        print(f"\n{manifold.upper()} MANIFOLD:")
        mdf = df[df['manifold'] == manifold]
        
        # Summary statistics
        summary = mdf.groupby('operation')['time_ms'].agg(['mean', 'min', 'max'])
        print(summary.round(3))
        
        # Scaling analysis
        print(f"\nScaling analysis for {manifold}:")
        for op in mdf['operation'].unique():
            op_df = mdf[mdf['operation'] == op].sort_values('size')
            if len(op_df) > 2:
                # Extract sizes
                sizes = []
                for s in op_df['size']:
                    if isinstance(s, tuple):
                        sizes.append(np.prod(s))
                    else:
                        sizes.append(s)
                
                # Log-log regression
                log_sizes = np.log(np.array(sizes))
                log_times = np.log(np.array(op_df['time_ms'].values))
                try:
                    slope, intercept = np.polyfit(log_sizes, log_times, 1)
                except:
                    slope = float('nan')
                print(f"  {op}: O(n^{slope:.2f})")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_single.py <results.json>")
        sys.exit(1)
    
    results_path = sys.argv[1]
    df = load_results(results_path)
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Print summary
    print_performance_summary(df)
    
    # Create plots
    print("\nGenerating visualizations...")
    
    # 1. Plot each manifold
    for manifold in df['manifold'].unique():
        fig = plot_manifold_operations(df, manifold)
        if fig:
            fig.savefig(output_dir / f"{manifold}_operations.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # 2. Scaling comparison
    fig = plot_scaling_comparison(df)
    fig.savefig(output_dir / "scaling_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nVisualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()