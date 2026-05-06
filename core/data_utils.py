import numpy as np
import math


class DataGenerator:
    """
    A data generation class for optimal design experiments.
    All methods are defined as @staticmethod and can be used without instantiation.
    """

    @staticmethod
    def off_diagonal_concatenate(A, B):
        """
        Construct [A 0; 0 B] block-diagonal matrix.
        """
        row_A, col_A = A.shape
        row_B, col_B = B.shape

        des_pool = np.zeros((row_A + row_B, col_A + col_B))
        des_pool[:row_A, :col_A] = A
        des_pool[row_A:, col_A:] = B

        return des_pool

    @staticmethod
    def dictionarize(X):
        """
        Convert feature matrix X (m x n) into dictionary of outer products.

        Returns:
            A_dict: {0: x_0 x_0^T, 1: x_1 x_1^T, ...}
        """
        return {i: np.outer(X[i], X[i]) for i in range(X.shape[0])}

    @staticmethod
    def generate_small_big_normal(m, n, sd_big, sd_small):
        """
        Generates synthetic data with block diagonal structure.
        - First m/2 points: Large variance on first n/2 coords.
        - Last m/2 points: Small variance on last n/2 coords.

        Args:
            m (int): Number of design points (must be even)
            n (int): Feature dimension (must be even)
            sd_big (float): Standard deviation for large-variance block
            sd_small (float): Standard deviation for small-variance block

        Returns:
            A_dict (dict): Dictionary of outer product matrices
            X (np.ndarray): Feature matrix (m x n)
        """
        if m % 2 != 0 or n % 2 != 0:
            raise ValueError("m and n must be even numbers for this generator.")

        half_m = m // 2
        half_n = n // 2

        A = np.random.normal(0, sd_big, [half_m, half_n])
        B = np.random.normal(0, sd_small, [half_m, half_n])

        X = DataGenerator.off_diagonal_concatenate(A, B)
        A_dict = DataGenerator.dictionarize(X)

        return A_dict, X

    @staticmethod
    def generate_design_pool(m, n, d):
        """
        Generates data with specific eigenvalue decay.

        Args:
            m (int): Number of candidates (rows)
            n (int): Feature dimension (cols)
            d (float): Decay rate

        Returns:
            A_dict (dict): Dictionary of outer product matrices
            X (np.ndarray): Feature matrix (m x n)
        """
        Xhat_ori = np.random.randn(m, n)
        Xhat = Xhat_ori.T @ Xhat_ori

        U, Sigma, _ = np.linalg.svd(Xhat)

        diag_values = []
        for i in range(1, len(Sigma) + 1):
            val = (i ** d) * Sigma[i - 1]
            if val < 1e-10:
                diag_values.append(0.0)
            else:
                diag_values.append(1.0 / math.sqrt(val))

        D = np.diag(diag_values)
        S = U @ D @ U.T
        X = Xhat_ori @ S

        A_dict = DataGenerator.dictionarize(X)

        return A_dict, X
