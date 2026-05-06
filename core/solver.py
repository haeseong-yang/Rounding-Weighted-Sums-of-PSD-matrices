import numpy as np
import cvxpy as cp


class RWSSolver:
    def __init__(self, M, A_dict, n, k, tau=0.5, lam=0.5, w=None, verbose=False):
        """
        Randomized rounding solver for optimal experimental design.

        Args:
            M (np.ndarray): Initial information matrix (n x n)
            A_dict (dict): Dictionary of candidate matrices {index: np.ndarray}
            n (int): Dimension of matrices
            k (int): Budget constraint
            tau (float): Trace threshold parameter for splitting
            lam (float): Weight threshold parameter for splitting
            w (np.ndarray): Initial weights
            verbose (bool): Whether to print status messages
        """
        self.M = M
        self.A_dict = A_dict
        self.inds = list(A_dict.keys())
        self.n = n
        self.m = len(A_dict)
        self.k = k
        self.w = np.zeros(self.m) if w is None else w
        self.tau = tau
        self.lam = lam
        self.verbose = verbose

    # --- [1. Convex Relaxation] ---
    def solve_convex_relaxation(self, des_type='E'):
        """
        Solve the continuous relaxation using CVXPY.

        Args:
            des_type (str): Optimality criterion. One of 'E', 'D', 'A', 'Mac'.

        Returns:
            z_val (np.ndarray): Optimal continuous weights
            prob.value (float): Optimal objective value
        """
        if self.verbose:
            print(f"Solving Convex Relaxation ({des_type}-optimal)...")

        z = cp.Variable((self.m, 1))
        constraints = [z >= 0, z <= 1, cp.sum(z) <= self.k]

        term_list = [z[i] * self.A_dict[self.inds[i]] for i in range(self.m)]
        C_expr = self.M + cp.sum(term_list)

        if des_type == 'E':
            obj_func = cp.lambda_min(C_expr)
        elif des_type == 'D':
            obj_func = cp.log_det(C_expr)
        elif des_type == 'A':
            I = np.eye(self.n)
            obj_func = -cp.sum([cp.matrix_frac(I[:, i], C_expr) for i in range(self.n)])
        elif des_type == 'Mac':
            obj_func = cp.lambda_min(C_expr + np.ones((self.n, self.n)))
        else:
            raise ValueError("Unknown design type. Choose 'E', 'D', 'A', or 'Mac'.")

        prob = cp.Problem(cp.Maximize(obj_func), constraints)

        try:
            prob.solve(solver=cp.MOSEK)
        except Exception:
            if self.verbose:
                print("MOSEK failed or not installed, trying SCS...")
            prob.solve(solver=cp.SCS)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem status is {prob.status}")
            return None, None

        z_val = np.clip(z.value.flatten(), 0, 1)
        return z_val, prob.value

    # --- [2. Internal Helpers for Rounding] ---
    def _transform(self, A_dict):
        """Whitening transform based on current weights."""
        A_sum = self.M.copy()
        for i in self.inds:
            if self.w[i] > 1e-10:
                A_sum += self.w[i] * A_dict[i]

        try:
            U, sigma, _ = np.linalg.svd(A_sum)
        except np.linalg.LinAlgError:
            return {}

        valid_idx = [j for j in range(len(sigma)) if sigma[j] > 1e-10]
        if not valid_idx:
            return {}

        U_red = U[:, valid_idx]
        D_half_inv = np.diag(1.0 / np.sqrt(sigma[valid_idx]))
        R = U_red @ D_half_inv

        return {i: R.T @ self.A_dict[i] @ R for i in self.inds}

    def split(self, B_dict):
        """Split indices into three groups using tau and lam thresholds."""
        I1 = [i for i in self.inds if np.trace(B_dict[i]) >= self.tau and self.w[i] >= self.lam]
        I2 = [i for i in self.inds if np.trace(B_dict[i]) >= self.tau and self.w[i] < self.lam]
        I3 = [i for i in self.inds if np.trace(B_dict[i]) < self.tau]
        return I1, I2, I3

    def sparse_dist(self, B_dict, I2):
        """Compute sampling distribution for sparse rounding."""
        if not I2:
            return {}

        C_dict = self._transform(B_dict)
        if not C_dict or I2[0] not in C_dict:
            return {}

        r_hat = C_dict[I2[0]].shape[0]
        p2 = {}
        for i in I2:
            if i in C_dict:
                p2[i] = self.w[i] * np.trace(C_dict[i]) / r_hat
            else:
                p2[i] = 0.0

        total_p = sum(p2.values())
        if total_p > 0:
            for k in p2:
                p2[k] /= total_p
        return p2

    def sparse_rand_round(self, B_dict, I2, k2):
        """Sparse randomized rounding."""
        if k2 <= 0 or not I2:
            return []
        p2 = self.sparse_dist(B_dict, I2)
        if not p2:
            return []

        if len(I2) <= k2:
            return I2
        else:
            keys = list(p2.keys())
            probs = list(p2.values())
            J2_temp = np.random.choice(keys, size=k2, replace=True, p=probs)
            return list(set(J2_temp))

    def weighted_rand_round(self, I3, gam):
        """Weighted randomized rounding (Bernoulli sampling)."""
        if not I3:
            return []
        sum_weights = sum(self.w[i] for i in I3)
        total_weight_ceil = np.ceil(gam * sum_weights)
        max_loop = 1000

        J3 = []
        for _ in range(max_loop):
            J3 = [i for i in I3 if np.random.rand() < self.w[i]]
            if len(J3) <= total_weight_ceil:
                break
        return J3

    def srr(self, k, gam):
        """Split Randomized Rounding: combine I1, sparse(I2), weighted(I3)."""
        B_dict = self._transform(self.A_dict)
        if not B_dict:
            return []
        I1, I2, I3 = self.split(B_dict)

        J1 = I1
        expected_J3_cost = np.ceil(gam * np.sum([self.w[i] for i in I3]))
        k2 = int(k - len(J1) - expected_J3_cost)

        J2 = self.sparse_rand_round(B_dict, I2, k2) if k2 > 0 and I2 else []
        J3 = self.weighted_rand_round(I3, gam)

        final_J = J1 + J2 + J3
        if len(final_J) > k:
            final_J = sorted(final_J, key=lambda idx: self.w[idx], reverse=True)[:k]

        return final_J

    def _run_repetition(self, N, rounding_func, des_type):
        """Repeat rounding N times and return best result."""
        max_val = -np.inf
        best_J = []
        for _ in range(N):
            J = rounding_func()
            val = self.evaluate_subset(J, des_type)
            if val > max_val:
                max_val = val
                best_J = J
        return best_J, max_val

    def evaluate_subset(self, J, obj_type='E'):
        """
        Compute objective value for a subset J.

        Returned values are aligned with the convex relaxation objectives:
            'E':   min eigenvalue of A_sum
            'D':   log det(A_sum)            (matches cp.log_det in relaxation)
            'A':   -trace(A_sum^{-1})        (negated, since we maximize)
            'Mac': 2nd smallest eigenvalue of A_sum

        Args:
            J (list): Selected indices
            obj_type (str): Optimality criterion

        Returns:
            float: Objective value (consistent with convex relaxation)
        """
        if not J:
            return np.inf if obj_type == 'A' else -np.inf

        A_sum = self.M.copy()
        for i in J:
            A_sum += self.A_dict[i]

        if obj_type == 'E':
            return np.min(np.linalg.eigvalsh(A_sum))
        elif obj_type == 'D':
            sign, logdet = np.linalg.slogdet(A_sum)
            # Return log det directly to match cp.log_det in solve_convex_relaxation.
            # If matrix is rank-deficient (sign <= 0), return -inf to penalize.
            return logdet if sign > 0 else -np.inf
        elif obj_type == 'A':
            try:
                return -np.trace(np.linalg.inv(A_sum))
            except Exception:
                return -np.inf
        elif obj_type == 'Mac':
            evals = np.linalg.eigvalsh(A_sum)
            return evals[1] if len(evals) > 1 else 0.0
        else:
            raise ValueError("obj_type must be one of 'E', 'D', 'A', 'Mac'.")

    # --- [3. Main APIs] ---
    def max_srr(self, N, gam, des_type='E'):
        """Repeat Split Randomized Rounding N times."""
        return self._run_repetition(N, lambda: self.srr(self.k, gam), des_type)

    def max_sparse_rounding(self, N, B_dict, I2, k2, des_type='E'):
        """Repeat Sparse Rounding N times."""
        return self._run_repetition(N, lambda: self.sparse_rand_round(B_dict, I2, k2), des_type)

    def max_weighted_rounding(self, N, I3, gam, des_type='E'):
        """Repeat Weighted Rounding N times."""
        return self._run_repetition(N, lambda: self.weighted_rand_round(I3, gam), des_type)

    # --- [4. Benchmarks] ---
    def _uniform_sample(self):
        """Sample k unique indices uniformly at random."""
        return list(np.random.choice(self.inds, self.k, replace=False))

    def max_uniform_sampling(self, N, des_type='E'):
        """Randomly sample k indices N times and return the best subset."""
        return self._run_repetition(N, self._uniform_sample, des_type)

    def exchange_algorithm(self, initial_J=None, max_iter=1, des_type='E'):
        """
        Fedorov's Exchange Algorithm.

        Args:
            initial_J (list): Initial subset. If None, uses uniform sampling.
            max_iter (int): Maximum number of epochs.
            des_type (str): Optimality criterion.

        Returns:
            current_J (list): Best subset found
            max_val (float): Best objective value
        """
        if initial_J is None:
            best_J, max_val = self.max_uniform_sampling(max_iter, des_type)
        else:
            best_J = list(initial_J)
            max_val = self.evaluate_subset(best_J, des_type)

        current_J = best_J[:]
        all_indices_set = set(self.inds)

        for epoch in range(max_iter):
            improved = False
            current_J_set = set(current_J)
            complement_J = list(all_indices_set - current_J_set)

            for idx_in in range(len(current_J)):
                for idx_out in range(len(complement_J)):
                    j_in = current_J[idx_in]
                    j_out = complement_J[idx_out]

                    trial_J = current_J[:]
                    trial_J[idx_in] = j_out
                    trial_val = self.evaluate_subset(trial_J, des_type)

                    if trial_val > max_val:
                        max_val = trial_val
                        current_J[idx_in] = j_out
                        complement_J[idx_out] = j_in
                        improved = True

            if self.verbose and improved:
                print(f"Epoch {epoch}: Improved to {max_val:.4f}")

            if not improved:
                if self.verbose:
                    print(f"Exchange converged at epoch {epoch}")
                break

        return current_J, max_val
