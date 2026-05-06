import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

plt.rcParams.update({'font.size': 12.5})


def plot_results(df_results, df_params, file_prefix, x_label, results_dir='results'):
    """
    Generate and save standard plots:
      1. Approximation Ratio
      2. Running Time

    Args:
        df_results (pd.DataFrame): Columns: ['Method', 'Ratio', 'k', 'Running Time (s)']
        df_params (pd.DataFrame): Columns: ['k', 'tau', 'lam'] (saved to CSV only)
        file_prefix (str): Prefix for output filenames (e.g. 'Synthetic_Eopt')
        x_label (str): X-axis label (e.g. 'Number of Experiments ($k$)')
        results_dir (str): Directory to save outputs
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- Plot 1: Approximation Ratio ---
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=df_results, x='k', y='Ratio', hue='Method',
        style='Method', markers=True, dashes=False,
        errorbar=('ci', 95), linewidth=2.5
    )
    plt.ylabel('Approximation Ratio (Approx. / Relax.)', fontsize=13)
    plt.xlabel(x_label, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Methods', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'Ratio_{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 2: Running Time ---
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=df_results, x='k', y='Running Time (s)', hue='Method',
        style='Method', markers=True, dashes=False,
        errorbar=('ci', 95), linewidth=2
    )
    plt.ylabel('Running Time (s)', fontsize=13)
    plt.xlabel(x_label, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Methods', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'Time_{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.show()


def save_csv(df_results, file_prefix, results_dir='results'):
    """
    Save experiment results to CSV.

    Args:
        df_results (pd.DataFrame): Results dataframe
        file_prefix (str): Prefix for output filename
        results_dir (str): Directory to save outputs
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'results_{file_prefix}.csv')
    df_results.to_csv(path, index=False)
    print(f"[Saved] {path}")
