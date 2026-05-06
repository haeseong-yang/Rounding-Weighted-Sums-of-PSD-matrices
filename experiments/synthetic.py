"""
Synthetic Data Experiment

Interactively prompts for the optimality criterion (E or D).

E-optimal: K_GRID = range(100, 155, 5), sd_big=10, sd_small=5
D-optimal: K_GRID = range(53, 83, 3),  sd_big=20, sd_small=1
  (D-optimal requires k > n; smaller k makes the problem rank-deficient.
   Larger sd contrast amplifies algorithm differences.)
"""
from core.data_loaders import load_synthetic
from core.runner import run_experiment

# ========================================================
# [Configuration]
# ========================================================
m, n = 900, 50
# ========================================================


def get_design_type():
    """Prompt user to choose design type."""
    while True:
        choice = input("Select optimality criterion [E/D] (default: E): ").strip().upper()
        if choice == "":
            return "E"
        if choice in ("E", "D"):
            return choice
        print("  Invalid input. Please enter 'E' or 'D'.")


def get_config(des_type):
    """Return (K_GRID, sd_big, sd_small) for the chosen design type."""
    if des_type == "E":
        return range(120, 235, 5), 20, 1
    elif des_type == "D":
        return range(55, 85, 5), 20, 1


def main():
    des_type = get_design_type()
    k_grid, sd_big, sd_small = get_config(des_type)

    print(f"\n[Config] {des_type}-optimal | sd_big={sd_big}, sd_small={sd_small}")
    print(f"[Config] K range: {list(k_grid)}\n")

    M, A_dict, m_, n_ = load_synthetic(m=m, n=n, sd_big=sd_big, sd_small=sd_small)
    run_experiment(
        M, A_dict, m_, n_,
        k_grid=k_grid,
        des_type=des_type,
        file_prefix=f'Synthetic_m{m}_n{n}_{des_type}opt',
        x_label='Number of Experiments ($k$)',
    )


if __name__ == '__main__':
    main()
