import numpy as np
import scipy.optimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import brentq
import cvxpy as cp

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from abc import ABC

from pacSTL_utils import SignalTemporalLogic

class Predicate(ABC):
    """
    Generic predicate class.
    """

class Robustness:
    "calculate robustness functions with the following optimization predicates"
    def __init__(self, pred = None):
        self.pred = pred
        pass

    @staticmethod
    def min_linear_predicates(pred_A, pred_b, ellipsoid_A, ellipsoid_b, init):
        obj = lambda p : pred_A.T @ p - pred_b
        obj_grad = lambda p: pred_A

        # 2. Smooth squared norm constraint
        def constr_fun(p):
            diff = ellipsoid_A @ p - ellipsoid_b
            return np.dot(diff, diff) - 1

        # 3. Analytical Jacobian for the constraint
        def constr_jac(p):
            return 2 * (ellipsoid_A.T @ (ellipsoid_A @ p - ellipsoid_b))

        in_set = NonlinearConstraint(constr_fun, -np.inf, 0, jac=constr_jac)
        #in_set = NonlinearConstraint((lambda p : np.linalg.norm(ellipsoid_A @ p - ellipsoid_b)-1), -np.inf, 0)
        h_low = scipy.optimize.minimize(obj, init, jac=obj_grad, constraints=(in_set), method='SLSQP', options={'maxiter': 200})
        if not h_low.success:
            print("Opt not term")
            print(h_low)
            rho = None
        else:
            rho = h_low.fun
        return rho

    @staticmethod
    def max_linear_predicates(pred_A, pred_b, ellipsoid_A, ellipsoid_b, init):
        obj = lambda p : - (pred_A.T @ p - pred_b)
        obj_grad = lambda p: -pred_A

        # 2. Use squared norm for better stability (removes sqrt singularity)
        def constr_fun(p):
            diff = ellipsoid_A @ p - ellipsoid_b
            return np.dot(diff, diff) - 1

        # 3. Provide the Jacobian for the constraint for speed and accuracy
        def constr_jac(p):
            return 2 * (ellipsoid_A.T @ (ellipsoid_A @ p - ellipsoid_b))

        in_set = NonlinearConstraint(constr_fun, -np.inf, 0, jac=constr_jac)

        #in_set = NonlinearConstraint((lambda p : np.linalg.norm(ellipsoid_A @ p - ellipsoid_b)-1), -np.inf, 0)
        h_high = scipy.optimize.minimize(obj, init, jac=obj_grad, constraints=(in_set), method='SLSQP',options={'maxiter': 200})
        if not h_high.success:
            print("Opt not term")
            print(h_high)
            rho = None
        else:
            rho = - h_high.fun
        return rho

    @staticmethod
    def min_quadratic_predicates(pred_Q_diag, pred_c, Q_factor, ellipsoid_A, ellipsoid_b, init, p_offset=None):
        """
        Minimizes (p - p_offset)^T Q (p - p_offset) - c over an ellipsoidal constraint.
        """
        n = ellipsoid_A.shape[1]

        # If no offset is provided, default to the origin (zeros) to maintain backwards compatibility
        if p_offset is None:
            p_offset = np.zeros(n)

        p = cp.Variable(n)

        # Quadratic objective
        Q = np.diag(pred_Q_diag)

        # Generalized objective to include the offset
        obj = cp.quad_form(p - p_offset, Q) - pred_c

        # Ellipsoidal constraint: ||A p - b||_2 <= 1
        constraints = [cp.norm(ellipsoid_A @ p - ellipsoid_b, 2) <= 1]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.GUROBI)  # requires Gurobi license

        rho = prob.value

        return rho


    @staticmethod
    def max_quadratic_predicates_langrage(ellipsoid_A, center, dim_indices, alpha, c, x_offset = None):
        """
        Computes the global maximum of alpha * ||x||_2^2 - c
        where x is a projection of a 12D ellipsoid onto specific dimensions.

        Parameters:
        - ellipsoid_A: 12x12 numpy array defining the ellipsoid ||Av - b|| <= 1
        - center: 12-dim numpy array representing the center of the ellipsoid (A^-1 b)
        - dim_indices: List or array of ints (e.g., [0, 1] or [0, 1, 2]) selecting the dimensions of x
        - alpha, c: Positive scalars for the objective
        """

        # 1. Compute the shape matrix (covariance) of the full 12D ellipsoid
        # The ellipsoid is (v - center)^T A^T A (v - center) <= 1
        # Shape matrix Sigma = (A^T A)^-1 = A^-1 A^-T
        A_inv = np.linalg.inv(ellipsoid_A)
        Sigma_full = A_inv @ A_inv.T

        # 2. Project onto the subspace (extract the 2D or 3D block)
        # The projected ellipsoid has a shape matrix that is simply the sub-block of Sigma
        idx = np.ix_(dim_indices, dim_indices)
        Sigma_x = Sigma_full[idx]

        # The center of the projected ellipsoid is just the sliced center
        x_c = center[dim_indices]

        # 3. Eigen-decomposition of the projected shape matrix
        # Sigma_x = U * Lambda * U^T
        lambdas, U = np.linalg.eigh(Sigma_x)

        # Map the center into the eigenvector space
        if x_offset is None:
            d = U.T @ x_c
        else:
            d = U.T @ (x_c - x_offset)

        # Break exact symmetry to avoid division-by-zero in edge cases
        # (e.g., if the center is exactly at the origin)
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)

        # 4. Set up the Secular Equation
        # We find the Lagrange multiplier (gamma) that maps to the furthest boundary point
        def secular_eq(gamma):
            # sum( lambda_i * d_i^2 / (gamma - lambda_i)^2 ) - 1 = 0
            return np.sum(lambdas * (d ** 2) / (gamma - lambdas) ** 2) - 1.0

        # The root for the maximum distance strictly lives greater than the max eigenvalue
        gamma_max = np.max(lambdas)

        # Define tight search brackets
        low = gamma_max + 1e-11
        high = gamma_max + np.sqrt(np.sum(lambdas * (d ** 2))) + 1.0

        # Step out the low bound slightly if needed due to float precision
        while secular_eq(low) < 0 and low > gamma_max:
            low = gamma_max + (low - gamma_max) / 10.0

        # 5. Solve for gamma (1D root finding, highly reliable and fast)
        try:
            gamma_opt = brentq(secular_eq, low, high)
        except ValueError:
            # Fallback if brentq fails to bracket (rare, usually means symmetry wasn't broken enough)
            gamma_opt = low

        # 6. Reconstruct the optimal x or x - x_offset in the projected subspace
        # x_opt = U * diag(gamma / (gamma - lambda)) * d
        v_opt = U @ ((gamma_opt / (gamma_opt - lambdas)) * d)

        # 7. Compute final objective
        max_val = alpha * np.linalg.norm(v_opt) ** 2 - c

        return max_val

    def evaluate_for_states(self, state, time_step):
        return SignalTemporalLogic(self.pred._evaluate(state), time_step)