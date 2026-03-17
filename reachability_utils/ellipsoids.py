import numpy as np 
import cvxpy as cp
from scipy.special import gamma
from numpy.linalg import det, inv


# Calculate Ellipsoid
def p_ball(xs, normp):
    xs = np.array(xs).T
    n = np.shape(xs)[0]
    m = np.shape(xs)[1]

    A = cp.Variable((n, n), symmetric=True)
    b = cp.Variable((n,1))

    # ------------------------------------------------------------------
    # objective --------------------------------------------------------
    n = A.shape[0]                       # matrix dimension
    objective = cp.Maximize(cp.log_det(A))

    # ------------------------------------------------------------------
    # constraints ------------------------------------------------------
    residual = A @ xs - b @ np.ones((1,m))
    col_norm_constraints = [
        cp.norm(residual[:, j], normp) <= 1
        for j in range(m)
    ]

    constraints = col_norm_constraints #+ bounds 

    # ------------------------------------------------------------------
    # solve ------------------------------------------------------------

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    print("Optimal value :", problem.value)
    print("A* =", A.value)
    print("b* =", b.value)
    print("volume proxy:", np.log(np.linalg.det(A.value)))
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Optimization failed with status:", problem.status)
    else:
        print("Optimization succeeded.")
    return problem.value, A.value, b.value

# constraint for "in ellipsoid"
def in_ellipsoid(A, b, p):
    p = np.asarray(p,  dtype=float).ravel()
    b = np.asarray(b,  dtype=float).ravel()  
    return np.linalg.norm(A @ p - b)



def ellipsoid_volume(A: np.ndarray, b: np.ndarray) -> float:
    Q = A.T @ A
    Q_inv = inv(Q)
    c = Q_inv @ A.T @ b
    r = b.T @ A @ Q_inv @ A.T @ b - b.T @ b + 1

    n = A.shape[1]  # dimensionality of the ellipsoid

    volume = (np.pi ** (n / 2)) / gamma(n / 2 + 1) * (r ** (n / 2)) / np.sqrt(det(Q))
    return volume




