"""
Synthetic Graph Augmentation Experiment

Interactively prompts for graph type (WS / BA) and objective (Mac / A).

Two contrasting random graph models:
  - Watts-Strogatz (WS): high clustering, short path length
  - Barabasi-Albert (BA): scale-free, hub-dominated

Two objectives:
  - 'Mac': Maximum algebraic connectivity augmentation
  - 'A':   Minimum total effective resistance augmentation

Problem sizes set by computational limits:
  - 'Mac': larger graphs feasible (n_nodes=45)
  - 'A':   limited by matrix inversion cost (n_nodes=30)
"""
from core.data_loaders import load_graph_synthetic
from core.runner import run_experiment


def get_graph_type():
    """Prompt user to choose graph model."""
    while True:
        choice = input("Select graph type [WS/BA] (default: WS): ").strip().upper()
        if choice == "":
            return "WS"
        if choice in ("WS", "BA"):
            return choice
        print("  Invalid input. Please enter 'WS' or 'BA'.")


def get_design_type():
    """Prompt user to choose objective."""
    while True:
        choice = input("Select objective [Mac/A] (default: Mac): ").strip()
        if choice == "":
            return "Mac"
        if choice in ("Mac", "A"):
            return choice
        print("  Invalid input. Please enter 'Mac' or 'A'.")


def main():
    graph_type = get_graph_type()
    des_type = get_design_type()

    n_nodes = 45 if des_type == "Mac" else 30
    k_grid = range(50, 200, 10)

    print(f"\n[Config] Graph={graph_type}, Objective={des_type}, N_nodes={n_nodes}")
    print(f"[Config] K range: {list(k_grid)}\n")

    M, A_dict, m, n = load_graph_synthetic(graph_type=graph_type, n_nodes=n_nodes)
    run_experiment(
        M, A_dict, m, n,
        k_grid=k_grid,
        des_type=des_type,
        file_prefix=f'Graph_{graph_type}_N{n_nodes}_{des_type}opt',
        x_label='Number of Added Edges ($k$)',
    )


if __name__ == '__main__':
    main()
