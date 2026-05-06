"""
Experiment Runner
=================
Common evaluation loop shared across all experiments.
"""

import time
import pandas as pd
from tqdm import tqdm

from .solver import RWSSolver
from .utils import uniform_weights, calc_ratio, run_auto_grid_search
from .visualize import plot_results, save_csv


def run_experiment(
    M, A_dict, m, n,
    k_grid,
    des_type,
    file_prefix,
    x_label,
    search_grid_size=10,
    search_t=10,
    N=10,
    eval_t=30,
    gam=1.0,
    results_dir='results',
):
    """
    Full experiment pipeline: convex relaxation -> grid search -> evaluation -> plots.

    Args:
        M (np.ndarray): Base information matrix (n x n)
        A_dict (dict): Candidate matrices {index: np.ndarray}
        m (int): Number of candidates
        n (int): Feature dimension
        k_grid (iterable): Budget values to evaluate
        des_type (str): Optimality criterion ('E', 'D', 'A', 'Mac')
        file_prefix (str): Prefix for saved files
        x_label (str): X-axis label for plots
        search_grid_size (int): Grid resolution for hyperparameter search
        search_t (int): Trials per (tau, lam) pair during grid search
        N (int): Repetitions per rounding call
        eval_t (int): Evaluation trials per k (for 95% CI)
        gam (float): Gamma for weighted rounding
        results_dir (str): Directory to save results
    """
    exps = {'Method': [], 'Ratio': [], 'k': [], 'Running Time (s)': []}
    best_params_log = []

    print(f"\n{'='*60}")
    print(f"Experiment: {file_prefix} | {des_type}-optimal")
    print(f"  K range    : {list(k_grid)}")
    print(f"  Grid Search: {search_grid_size}x{search_grid_size}, {search_t} trials each")
    print(f"  Eval trials: {eval_t}")
    print(f"{'='*60}\n")

    for k in tqdm(k_grid, desc="Overall Progress"):

        # --- A. Convex Relaxation ---
        w_init = uniform_weights(m, k)
        solver = RWSSolver(M, A_dict, n, k, tau=0.5, lam=0.5, w=w_init, verbose=False)

        z_val, opt_val = solver.solve_convex_relaxation(des_type)

        if z_val is None:
            print(f"  [Skip] k={k}: relaxation failed")
            continue

        solver.w = z_val

        # --- B. Hyperparameter Grid Search ---
        best_p = run_auto_grid_search(
            solver, opt_val, des_type,
            grid_size=search_grid_size, search_t=search_t, N=N, gam=gam
        )
        solver.tau = best_p['tau']
        solver.lam = best_p['lam']
        best_params_log.append({'k': k, **best_p})

        # --- C. Evaluation ---
        B_dict = solver._transform(solver.A_dict)

        for _ in tqdm(range(eval_t), desc=f"  Eval (k={k})", leave=False):

            # Split
            st = time.time()
            _, v = solver.max_srr(N, gam, des_type)
            exps['Method'].append('Split')
            exps['Ratio'].append(calc_ratio(v, opt_val, des_type))
            exps['k'].append(k)
            exps['Running Time (s)'].append(time.time() - st)

            # Sparse
            st = time.time()
            _, v = solver.max_sparse_rounding(N, B_dict, solver.inds, k, des_type)
            exps['Method'].append('Sparse')
            exps['Ratio'].append(calc_ratio(v, opt_val, des_type))
            exps['k'].append(k)
            exps['Running Time (s)'].append(time.time() - st)

            # Weight
            st = time.time()
            _, v = solver.max_weighted_rounding(N, solver.inds, gam, des_type)
            exps['Method'].append('Weight')
            exps['Ratio'].append(calc_ratio(v, opt_val, des_type))
            exps['k'].append(k)
            exps['Running Time (s)'].append(time.time() - st)

            # Exchange
            st = time.time()
            _, v = solver.exchange_algorithm(initial_J=None, max_iter=1, des_type=des_type)
            exps['Method'].append('Exchange')
            exps['Ratio'].append(calc_ratio(v, opt_val, des_type))
            exps['k'].append(k)
            exps['Running Time (s)'].append(time.time() - st)

            # Uniform
            st = time.time()
            _, v = solver.max_uniform_sampling(N, des_type)
            exps['Method'].append('Uniform')
            exps['Ratio'].append(calc_ratio(v, opt_val, des_type))
            exps['k'].append(k)
            exps['Running Time (s)'].append(time.time() - st)

    # --- D. Save & Plot ---
    if not exps['k']:
        print("[Warning] No results to save.")
        return

    df_results = pd.DataFrame(exps)
    df_params = pd.DataFrame(best_params_log)

    save_csv(df_results, file_prefix, results_dir=results_dir)
    plot_results(df_results, df_params, file_prefix,
                 x_label=x_label, results_dir=results_dir)

    print(f"\n[Done] {file_prefix}")
    print("\n[Best Hyperparameters]\n", df_params)
