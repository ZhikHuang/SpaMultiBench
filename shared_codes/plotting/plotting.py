"""

This module centralizes the common plotting patterns used across the
benchmarking notebooks:

- boxplot comparisons of metrics across methods (with simple ranksums-based
  pairwise annotation against a target method)
- grid of spatial embeddings using Scanpy's `sc.pl.embedding`
- bar plots for metric tables

Example usage (from a notebook or script):

    from plotting import boxplot_comparison, plot_spatial_grid, bar_metrics

    # boxplot:
    boxplot_comparison(combined_df, metrics, method_list, colorslist, my_method='MultiSP')

    # spatial grid:
    fig = plot_spatial_grid(adata, method_list, s_size=30)

    # bar metrics (df is a DataFrame with metric columns indexed by method):
    bar_metrics(df, ['CHAOS','PAS','ASW','Moran'])
    
    # color:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
"""

from typing import List, Sequence, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums
import scanpy as sc



def boxplot_comparison(
    combined_df: pd.DataFrame,
    metrics: Sequence[str],
    method_list: Sequence[str],
    colorslist: Sequence[str],
    my_method: Optional[str] = None,
    figsize=(18, 10),
):
    """Create a grid of boxplots for the provided metrics.

    Parameters
    - combined_df: DataFrame in the stacked format produced in the notebooks
      with columns ['Method','Metric','value']
    - metrics: list of metric names to plot (order will define subplot order)
    - method_list: ordered list of methods used for consistent x positions
    - colorslist: color palette list (must be at least len(method_list))
    - my_method: method name to use as the reference for pairwise tests

    The function reproduces a seaborn boxplotper metric,
    and pairwise ranksums tests between `my_method` and others
    with simple half-tailed p-value conversion.

    Returns the matplotlib Figure object.
    """

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    n_methods = len(method_list)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df_metric = combined_df[combined_df['Metric'] == metric]

        sns.boxplot(
            data=df_metric,
            x='Method',
            y='value',
            linewidth=0.5,
            palette=colorslist,
            fliersize=1,
            ax=ax,
        )

        ax.set_title(metric)
        ax.set_xlabel('')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)

        y_max = df_metric['value'].max()
        y_min = df_metric['value'].min()
        y_range = y_max - y_min if y_max != y_min else max(1.0, abs(y_max))

        if my_method is None:
            continue

        my_idx = method_list.index(my_method)
        x_my = my_idx

        other_methods = [m for m in method_list if m != my_method]
        for method in other_methods:
            other_idx = method_list.index(method)
            x_other = other_idx

            my_data = df_metric[df_metric['Method'] == my_method]['value']
            other_data = df_metric[df_metric['Method'] == method]['value']

            stat, p_two_sided = ranksums(my_data, other_data)

            # one-sided p-value
            if metric == 'Entropy':
                if stat < 0:
                    p_value = p_two_sided / 2
                else:
                    p_value = 1 - p_two_sided / 2
            else:
                if stat > 0:
                    p_value = p_two_sided / 2
                else:
                    p_value = 1 - p_two_sided / 2
            
            y = y_max + y_range * 0.05 + (other_idx * y_range * 0.05)

            ax.plot(
                [x_my, x_my, x_other, x_other],
                [y, y + y_range * 0.02, y + y_range * 0.02, y],
                lw=1.2,
                c='black',
            )

            x_text = (x_my + x_other) / 2
            p_text = 'p<0.001' if p_value < 0.001 else f'p={p_value:.3f}'
            ax.text(x_text, y + y_range * 0.025, p_text, ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def plot_spatial_grid(
    adata,
    method_list: Sequence[str],
    s_size: int = 30,
    ncols: int = 5,
    figsize=(10, 5),
    palette: Optional[Sequence[str]] = None,
):
    """Plot a grid of spatial embeddings (one panel per method).

    Each method's cluster labels live in `adata.obs[method]`.

    Returns the created Figure.
    """

    n = len(method_list)
    nrows = int(np.ceil(n / ncols))
    fig, ax_list = plt.subplots(nrows, ncols, figsize=figsize)
    # normalize shape to 2D array
    if nrows == 1:
        ax_arr = np.array([ax_list])
    else:
        ax_arr = ax_list

    for idx, method in enumerate(method_list):
        row, col = divmod(idx, ncols)
        ax = ax_arr[row, col]
        sc.pl.embedding(
            adata,
            basis='spatial',
            color=method,
            ax=ax,
            s=s_size,
            show=False,
            palette=palette,
        )
        ax.get_legend().remove()
        ax.set_xlabel('')
        ax.set_ylabel('')

    # remove any unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax_arr[row, col].axis('off')

    plt.tight_layout(w_pad=0.4)
    return fig


def bar_metrics(df: pd.DataFrame, metrics_cols: Sequence[str], colors: Optional[Sequence[str]] = None):
    """Plot bar charts for metric columns (DataFrame indexed by method).
    """

    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['font.family'] = 'Arial'

    ax = df[list(metrics_cols)].T.plot(kind='bar', width=0.9, color=colors)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.ylabel('Value')
    plt.grid(False)
    return ax.get_figure()

