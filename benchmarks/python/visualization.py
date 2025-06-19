#!/usr/bin/env python3
"""
Visualization module for benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class BenchmarkVisualizer:
    """Create visualizations from benchmark results."""
    
    def __init__(self, results_path: str):
        """Initialize with results file."""
        self.results_path = Path(results_path)
        self.df = self._load_results()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _load_results(self) -> pd.DataFrame:
        """Load results from file."""
        if self.results_path.suffix == '.json':
            with open(self.results_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.results_path.suffix == '.csv':
            df = pd.read_csv(self.results_path)
        else:
            raise ValueError(f"Unsupported file format: {self.results_path.suffix}")
        
        # Convert size tuples from string if needed
        if df['size'].dtype == 'object':
            try:
                df['size'] = df['size'].apply(eval)
            except:
                pass
        
        # Add derived columns
        if 'time_seconds' in df.columns:
            df['time_ms'] = df['time_seconds'] * 1000
        elif 'time_ms' not in df.columns:
            raise ValueError("Neither 'time_seconds' nor 'time_ms' found in results")
        
        return df
    
    def plot_operation_comparison(self, manifold: str, operation: str, 
                                save_path: Optional[str] = None) -> None:
        """Plot comparison of libraries for a specific operation."""
        data = self.df[(self.df['manifold'] == manifold) & 
                      (self.df['operation'] == operation)]
        
        if data.empty:
            print(f"No data for {manifold} - {operation}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        libraries = data['library'].unique()
        sizes = sorted(data['size'].unique())
        
        for lib in libraries:
            lib_data = data[data['library'] == lib]
            times = []
            for size in sizes:
                size_data = lib_data[lib_data['size'] == size]
                if not size_data.empty:
                    times.append(size_data['time_ms'].mean())
                else:
                    times.append(np.nan)
            
            ax.plot(range(len(sizes)), times, marker='o', label=lib, linewidth=2)
        
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=45)
        ax.set_xlabel('Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{manifold.capitalize()} - {operation}')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
    
    def plot_speedup_heatmap(self, reference: str = 'riemannopt',
                           save_path: Optional[str] = None) -> None:
        """Plot speedup heatmap comparing to reference library."""
        # Calculate speedups
        speedup_data = []
        
        for manifold in self.df['manifold'].unique():
            for operation in self.df['operation'].unique():
                for size in self.df['size'].unique():
                    ref_data = self.df[
                        (self.df['manifold'] == manifold) &
                        (self.df['operation'] == operation) &
                        (self.df['size'] == size) &
                        (self.df['library'] == reference)
                    ]
                    
                    if ref_data.empty:
                        continue
                    
                    ref_time = ref_data['time_ms'].mean() / 1000  # Convert back to seconds for speedup calc
                    
                    for lib in self.df['library'].unique():
                        if lib == reference:
                            continue
                        
                        lib_data = self.df[
                            (self.df['manifold'] == manifold) &
                            (self.df['operation'] == operation) &
                            (self.df['size'] == size) &
                            (self.df['library'] == lib)
                        ]
                        
                        if not lib_data.empty:
                            lib_time = lib_data['time_ms'].mean() / 1000  # Convert back to seconds
                            speedup = lib_time / ref_time
                            
                            speedup_data.append({
                                'manifold': manifold,
                                'operation': operation,
                                'size': str(size),
                                'library': lib,
                                'speedup': speedup
                            })
        
        if not speedup_data:
            print("No speedup data available")
            return
        
        speedup_df = pd.DataFrame(speedup_data)
        
        # Create heatmap
        fig = go.Figure()
        
        for lib in speedup_df['library'].unique():
            lib_df = speedup_df[speedup_df['library'] == lib]
            
            pivot = lib_df.pivot_table(
                values='speedup',
                index=['manifold', 'operation'],
                columns='size',
                aggfunc='mean'
            )
            
            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=[f"{idx[0]}-{idx[1]}" for idx in pivot.index],
                colorscale='RdYlGn',
                reversescale=True,
                text=np.round(pivot.values, 2),
                texttemplate='%{text}x',
                textfont={"size": 10},
                colorbar=dict(title="Speedup"),
                name=lib
            ))
        
        fig.update_layout(
            title=f"Speedup compared to {reference} (higher is better for {reference})",
            xaxis_title="Size",
            yaxis_title="Manifold-Operation",
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_scaling_analysis(self, manifold: str, operation: str,
                            save_path: Optional[str] = None) -> None:
        """Plot scaling analysis with trend lines."""
        data = self.df[(self.df['manifold'] == manifold) & 
                      (self.df['operation'] == operation)]
        
        if data.empty:
            print(f"No data for {manifold} - {operation}")
            return
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Time vs Size", "Log-Log Plot"),
            horizontal_spacing=0.15
        )
        
        libraries = data['library'].unique()
        colors = px.colors.qualitative.Set1
        
        for i, lib in enumerate(libraries):
            lib_data = data[data['library'] == lib].sort_values('size')
            
            # Extract sizes (handle tuples)
            sizes = []
            for s in lib_data['size']:
                if isinstance(s, tuple):
                    sizes.append(np.prod(s))  # Product of dimensions
                else:
                    sizes.append(s)
            
            times = lib_data['time_ms'].values
            
            # Linear plot
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=times,
                    mode='lines+markers',
                    name=lib,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Log-log plot
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=times,
                    mode='lines+markers',
                    name=lib,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Update axes
        fig.update_xaxes(title_text="Size", row=1, col=1)
        fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
        
        fig.update_xaxes(title_text="Size", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Time (ms)", type="log", row=1, col=2)
        
        fig.update_layout(
            title=f"Scaling Analysis: {manifold.capitalize()} - {operation}",
            height=500,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def generate_report(self, output_dir: str = "visualizations") -> None:
        """Generate comprehensive visual report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Summary statistics
        print("\nGenerating visualization report...")
        
        # 1. Overall speedup heatmap
        self.plot_speedup_heatmap(
            save_path=str(output_path / "speedup_heatmap.html")
        )
        
        # 2. Per-operation comparisons
        for manifold in self.df['manifold'].unique():
            for operation in self.df['operation'].unique():
                self.plot_operation_comparison(
                    manifold, operation,
                    save_path=str(output_path / f"{manifold}_{operation}_comparison.png")
                )
                
                self.plot_scaling_analysis(
                    manifold, operation,
                    save_path=str(output_path / f"{manifold}_{operation}_scaling.html")
                )
        
        print(f"Visualizations saved to {output_path}")
    
    def print_summary_table(self) -> None:
        """Print summary table of results."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Average speedup by manifold
        if 'riemannopt' in self.df['library'].unique():
            print("\nAverage speedup of RiemannOpt vs other libraries:")
            
            for manifold in self.df['manifold'].unique():
                print(f"\n{manifold.upper()}:")
                manifold_df = self.df[self.df['manifold'] == manifold]
                
                for lib in manifold_df['library'].unique():
                    if lib == 'riemannopt':
                        continue
                    
                    speedups = []
                    for operation in manifold_df['operation'].unique():
                        for size in manifold_df['size'].unique():
                            rio_time = manifold_df[
                                (manifold_df['operation'] == operation) &
                                (manifold_df['size'] == size) &
                                (manifold_df['library'] == 'riemannopt')
                            ]['time_ms'].mean()
                            
                            lib_time = manifold_df[
                                (manifold_df['operation'] == operation) &
                                (manifold_df['size'] == size) &
                                (manifold_df['library'] == lib)
                            ]['time_ms'].mean()
                            
                            if rio_time > 0 and not np.isnan(lib_time):
                                speedups.append(lib_time / rio_time)
                    
                    if speedups:
                        avg_speedup = np.mean(speedups)
                        print(f"  vs {lib}: {avg_speedup:.2f}x faster")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument('results', help='Path to results file (JSON or CSV)')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary table')
    
    args = parser.parse_args()
    
    viz = BenchmarkVisualizer(args.results)
    
    if args.summary_only:
        viz.print_summary_table()
    else:
        viz.generate_report(args.output_dir)
        viz.print_summary_table()


if __name__ == "__main__":
    main()