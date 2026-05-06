"""
Regensburg Pediatric Appendicitis Experiment

We restrict to D-optimal design for two reasons:
  1. The benchmark study (Ponte et al., 2025) considers only D-optimal
     on this dataset, allowing direct comparison.
  2. E-optimal on this dataset suffers from poor numerical feasibility
     due to strong correlation among clinical features.
"""
from core.data_loaders import load_regensburg
from core.runner import run_experiment

# ========================================================
# [Configuration]
# ========================================================
DES_TYPE = 'D'
K_GRID   = range(30, 65, 5)
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
