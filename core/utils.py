import numpy as np


def uniform_weights(m, k):
    """
    Generate random weights that sum to k.

    Args:
        m (int): Number of design points
        k (float): Target sum (budget)

    Returns:
        np.ndarray: Weight array of shape (m,)
    """
    w = np.random.rand(m)
    return k * w / np.sum(w)


def calc_ratio(val_approx, val_opt, des_type):
    """
    Compute approximation ratio between approximate and optimal values.

    Conventions:
        'E':   maximize min eig    -> ratio = val_approx / val_opt
        'D':   maximize log det    -> ratio = val_approx / val_opt
        'A':   minimize trace inv  -> ratio = val_opt / val_approx (both negated)
        'Mac': maximize lambda_2   -> ratio = val_approx / val_opt

    Args:
        val_approx (float): Value from approximation algorithm
        val_opt (float): Optimal (relaxation) value
        des_type (str): Optimality criterion ('E', 'D', 'A', 'Mac')

    Returns:
        float: Approximation ratio in [0, 1]
    """
    if abs(val_opt) < 1e-9:
        return 0.0

    # Handle -inf (rank-deficient case for D-optimal)
    if val_approx == -np.inf:
        return 0.0

    if des_type == 'A':
        if abs(val_approx) < 1e-9:
            return 0.0
        return min(1.0, abs(val_opt) / abs(val_approx))
    else:
        return min(1.0, val_approx / val_opt)


def run_auto_grid_search(solver, opt_val, des_type,
                         grid_size=10, search_t=10, N=10, gam=1.0):
    """
    Grid search over (tau, lam) hyperparameters to maximize approximation ratio.

    Args:
        solver (RWSSolver): Solver instance with w already set to relaxation solution
        opt_val (float): Optimal value from convex relaxation
        des_type (str): Optimality criterion
        grid_size (int): Number of grid points per axis
        search_t (int): Number of trials per (tau, lam) pair
        N (int): Number of repetitions per SRR call
        gam (float): Gamma parameter for weighted rounding

    Returns:
        dict: Best hyperparameters {'tau': float, 'lam': float}
    """
    tau_vals = np.linspace(0.1, 1.5, grid_size)
    lam_vals = np.linspace(0.1, 0.9, grid_size)

    best_params = {'tau': 0.5, 'lam': 0.5}
    best_ratio = -1.0

    for tau in tau_vals:
        for lam in lam_vals:
            solver.tau = tau
            solver.lam = lam
            ratios = []
            for _ in range(search_t):
                _, val = solver.max_srr(N=N, gam=gam, des_type=des_type)
                ratios.append(calc_ratio(val, opt_val, des_type))

            mean_r = np.mean(ratios)
            if mean_r > best_ratio:
                best_ratio = mean_r
                best_params = {'tau': tau, 'lam': lam}

    return best_params
