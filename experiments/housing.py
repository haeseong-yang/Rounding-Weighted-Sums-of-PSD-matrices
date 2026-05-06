"""
Housing Dataset Experiment (LIBSVM, m=506, n=13)

Interactively prompts for the optimality criterion (E or D).

E-optimal: K_GRID = range(20, 31, 1)
D-optimal: K_GRID = range(14, 25, 1)  (k close to n)
"""
from core.data_loaders import load_housing
from core.runner import run_experiment


def get_design_type():
    """Prompt user to choose design type."""
    while True:
        choice = input("Select optimality criterion [E/D] (default: E): ").strip().upper()
        if choice == "":
            return "E"
        if choice in ("E", "D"):
            return choice
        print("  Invalid input. Please enter 'E' or 'D'.")


def get_k_grid(des_type):
    """Return K_GRID for the chosen design type."""
    if des_type == "E":
        return range(20, 31, 1)
    elif des_type == "D":
        return range(14, 25, 1)


def main():
    des_type = get_design_type()
    k_grid = get_k_grid(des_type)

    print(f"\n[Config] {des_type}-optimal")
    print(f"[Config] K range: {list(k_grid)}\n")

    M, A_dict, m, n = load_housing()
    run_experiment(
        M, A_dict, m, n,
        k_grid=k_grid,
        des_type=des_type,
        file_prefix=f'Housing_{des_type}opt',
        x_label='Number of Experiments ($k$)',
    )


if __name__ == '__main__':
    main()
