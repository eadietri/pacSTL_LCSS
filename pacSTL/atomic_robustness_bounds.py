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

        # if not h_low.success:
        #     print("Opt not term")
        #     print(h_low)
        #     rho = None
        # else:
        #     # Capturing the minimum value
        #     rho = h_low.fun

        return rho

    @staticmethod
    def max_quadratic_predicates_cvxpy(pred_Q_diag, pred_c, Q_factor, ellipsoid_A, ellipsoid_b, init):
        """
        Maximizes x^T Q x - c over an ellipsoidal constraint robustly using Basinhopping.
        # """
        # # Objective: -(p^T Q p - c) because scipy minimizes by default
        # obj = lambda p: -(np.sum(pred_Q_diag * (p**2)) - pred_c)

        # in_set = NonlinearConstraint((lambda p: np.linalg.norm(ellipsoid_A @ p - ellipsoid_b) - 1), -np.inf, 0)

        # # Package your local solver arguments into a dictionary
        # minimizer_kwargs = {
        #     "method": "Nelder-Mead",
        #     "constraints": (in_set),
        #     "options": {'maxiter': 200}
        # }

        # # Basinhopping wraps the local minimize function to escape local maxima
        # # niter=10 means it will try 10 different "hops" (adjust based on speed needs)
        # h_high = scipy.optimize.basinhopping(
        #     obj,
        #     init,
        #     minimizer_kwargs=minimizer_kwargs,
        #     niter=10,
        #     stepsize=0.5  # How far to "hop" between attempts
        # )

        # # basinhopping returns a result object similar to minimize
        # # We check the lowest local optimization result it found
        # if not h_high.lowest_optimization_result.success:
        #     print("Opt not term")
        #     print(h_high.lowest_optimization_result)
        #     rho = None
        # else:
        #     # Reversing the negation to get the true maximum value
        #     rho = -h_high.fun

        # # Retry with Gurobi:
        n = ellipsoid_A.shape[1]
        p = cp.Variable(n)

        # Quadratic objective
        Q = np.diag(pred_Q_diag)
        obj = -(cp.quad_form(p, Q) - pred_c)

        # Ellipsoidal constraint: ||A p - b||_2 <= 1
        constraints = [cp.norm(ellipsoid_A @ p - ellipsoid_b, 2) <= 1]
   
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.GUROBI, Nonconvex=True)  # requires Gurobi license
        rho = - prob.value

        return rho

    @staticmethod
    def max_quadratic_predicates_gurobi(pred_Q_diag, pred_c, Q_factor, ellipsoid_A, ellipsoid_b, init):
        """
        Maximizes x^T Q x - c over an ellipsoidal constraint using Gurobi's global
        non-convex solver.

        Constraint: ||A p - b||_2 <= 1  =>  (A p - b)^T (A p - b) <= 1
        """
        n = ellipsoid_A.shape[1]
        rho = None

        # 1. Create a Gurobi environment and model
        # We use a context manager ('with') to ensure the model/env are closed correctly
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)  # Silence the solver logs
            env.start()
            with gp.Model(env=env) as m:

                # 2. Configure for Non-Convexity
                # Setting NonConvex to 2 allows Gurobi to find the global optimum
                # for non-convex quadratic objectives and constraints.
                m.setParam('NonConvex', 2)

                # 3. Define Variables
                # 'lb' is lower bound; default is 0.0, so we must set to -inf for R^n
                p = m.addMVar(n, lb=-GRB.INFINITY, name="p")

                # 4. Define the Objective: maximize (p^T Q p - c)
                # Since Q is diagonal, we can build the sum efficiently
                obj_expr = sum(pred_Q_diag[i] * p[i] * p[i] for i in range(n)) - pred_c
                m.setObjective(obj_expr, GRB.MAXIMIZE)

                # 5. Define the Ellipsoidal Constraint
                # ||Ap - b||^2 <= 1  is  (Ap - b)^T (Ap - b) <= 1
                diff = ellipsoid_A @ p - ellipsoid_b
                m.addConstr(diff @ diff <= 1.0, name="ellipsoid_boundary")

                # 6. Solve
                m.optimize()

                # 7. Extract Results
                if m.Status == GRB.OPTIMAL:
                    rho = m.ObjVal
                elif m.Status == GRB.TIME_LIMIT:
                    print("Warning: Gurobi reached time limit; returning best bound found.")
                    rho = m.ObjVal
                else:
                    print(f"Gurobi failed to find an optimal solution. Status: {m.Status}")
                    rho = None

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