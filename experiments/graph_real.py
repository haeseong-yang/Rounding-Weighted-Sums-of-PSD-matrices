"""
Real Graph Augmentation Experiment

Interactively prompts for the dataset.
Supported datasets: 'Karate', 'Dolphins', 'TrainTerrorists'
"""
from core.data_loaders import load_graph_real
from core.runner import run_experiment


def get_dataset():
    """Prompt user to choose dataset."""
    options = ["Karate", "Dolphins", "TrainTerrorists"]
    while True:
        print("Available datasets:")
        for i, name in enumerate(options, 1):
            print(f"  {i}. {name}")
        choice = input("Select dataset [1/2/3 or name] (default: 1): ").strip()
        if choice == "":
            return options[0]
        # Numeric selection
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        # Name-based selection (case-insensitive)
        for name in options:
            if choice.lower() == name.lower():
                return name
        print("  Invalid input. Try again.\n")


def main():
    dataset = get_dataset()
    des_type = "Mac"
    k_grid = range(50, 200, 10)

    print(f"\n[Config] Dataset={dataset}, Objective={des_type}")
    print(f"[Config] K range: {list(k_grid)}\n")

    M, A_dict, m, n = load_graph_real(dataset)
    run_experiment(
        M, A_dict, m, n,
        k_grid=k_grid,
        des_type=des_type,
        file_prefix=f'{dataset}_{des_type}opt',
        x_label='Number of Added Edges ($k$)',
    )


if __name__ == '__main__':
    main()
