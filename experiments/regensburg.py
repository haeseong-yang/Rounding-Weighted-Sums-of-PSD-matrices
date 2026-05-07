"""
Regensburg Pediatric Appendicitis Experiment
"""
from core.data_loaders import load_regensburg
from core.runner import run_experiment

# ========================================================
# [Configuration]
# ========================================================
DES_TYPE = 'D'
K_GRID   = range(30, 46, 1)
# ========================================================


def main():
    M, A_dict, m, n = load_regensburg()
    run_experiment(
        M, A_dict, m, n,
        k_grid=K_GRID,
        des_type=DES_TYPE,
        file_prefix=f'Regensburg_{DES_TYPE}opt',
        x_label='Number of Experiments ($k$)',
    )


if __name__ == '__main__':
    main()
