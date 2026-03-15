import math
import numpy as np
from plot_reachable_sets import convert_A_b_to_Q_c_r
from scipy.optimize import minimize

def rotation_matrix(angle, radian=False):
    """Create a 2D rotation matrix

    :param angle: angle in degrees or radians
    :param radian: boolean flag indicating if the angle is in radians
    """
    rad = degree_to_radian(angle) if not radian else angle
    return np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])

def normalize_degree(degree):
    """Normalize degrees to the range [0, 360]

    :param deg: degrees
    :return: normalized degrees
    """
    return degree % 360


def normalize_radian(radian):
    """Normalize radians to the range [0, 2*pi]

    :param rad: radians
    :return: normalized radians
    """
    return radian % (2 * math.pi)

def normalize_radian_pi(radian):
    """
    Normalize angle to be between -pi and pi.
    """
    return (radian + np.pi) % (2 * np.pi) - np.pi


def degree_to_radian(degree):
    """Convert degrees to radians

    :param deg: degrees
    :return: radians
    """
    return degree * math.pi / 180


def radian_to_degree(radian):
    """Convert radians to degrees

    :param rad: radians
    :return: degrees
    """
    return radian * 180 / math.pi


def zonotope_farthest_point(p_ego, c, G, max_iter=50, tol=1e-9):
    Q = G.T @ G
    b = G.T @ (c - p_ego)

    def sign_no_zero(x):
        s = np.sign(x)
        s[s == 0] = 1
        return s

    xi = sign_no_zero(b)
    for _ in range(max_iter):
        xi_new = sign_no_zero(Q @ xi + b)
        if np.allclose(xi_new, xi, atol=tol):
            break
        xi = xi_new

    p_star = c + G @ xi
    return p_star

def zonotope_linear_max_with_offset(c, G, A, b=0.0, tie_break=1.0):
    """
    Maximize A^T p - b over p = c + G xi, ||xi||_inf <= 1,
    where b is a scalar constant.
    """
    v = G.T @ A
    s = np.sign(v)
    s[s == 0] = tie_break
    xi_star = s.astype(float)
    p_star = c + G @ xi_star
    value = A @ p_star - b   # subtract scalar here
    return p_star, xi_star, value

def zonotope_closest_point_via_bounds(p_ego, c, G, xi_init=None, tol=1e-9, i_start=0, i_stop=2):
    """
    Find closest p = c + G xi to p_ego by optimizing over xi in [-1,1]^m.
    Uses L-BFGS-B (box constrained).
    Returns p_star, xi_star, success_flag.
    """
    m = G.shape[1]
    if xi_init is None:
        # least-squares projection then clamp is a good initial guess
        xi_ls, *_ = np.linalg.lstsq(G, (p_ego - c), rcond=None)
        xi_init = np.clip(xi_ls, -1.0, 1.0)

    def obj(xi):
        r = (p_ego - G @ xi - c)[i_start:i_stop]
        return float(r @ r)   # squared distance (smooth)

    bounds = [(-1.0, 1.0)] * m
    res = minimize(obj, xi_init, method='L-BFGS-B', bounds=bounds,
                   options={'ftol':tol, 'maxiter':1000})
    xi_star = res.x
    p_star = c + G @ xi_star
    return p_star, xi_star, res.success

def ellipsoid_extreme_point(p_ego, b, A, which="closest", tol=1e-12, maxiter=200):
    """
    Compute the closest or farthest point from p_ego on the ellipsoid
        (p-c)^T A (p-c) <= 1
    where A is symmetric positive definite.

    Parameters
    ----------
    p_ego : (n,) ndarray
        Reference point.
    c : (n,) ndarray
        Ellipsoid center.
    A : (n,n) ndarray
        Positive definite matrix.
    which : {"closest", "farthest"}
        Which extreme point to compute.
    tol : float
        Tolerance for root finding.
    maxiter : int
        Maximum bisection iterations.

    Returns
    -------
    p_star : (n,) ndarray
        Extreme point on the ellipsoid.
    """
    A_temp, c, r = convert_A_b_to_Q_c_r(A, b)
    A = A_temp/r
    # Shift to ellipsoid-centered coords
    d = p_ego - c

    # Eigen-decompose A
    alpha, Q = np.linalg.eigh(A)   # alpha > 0
    w = Q.T @ d

    # Inside case for closest point
    if which == "closest" and d.T @ A @ d <= 1 + tol:
        return p_ego.copy()

    # Define phi(lambda)
    def phi(lam):
        return np.sum(alpha * (w**2) / (1 - lam * alpha)**2)

    # Bracket root depending on closest/farthest
    if which == "closest":
        left, right = -1.0, 1/np.max(alpha) - 1e-14
        # Expand left until phi(left) > 1
        while phi(left) <= 1:
            left *= 2
    elif which == "farthest":
        left, right = 1/np.min(alpha) + 1e-14, 1.0
        # Expand right until phi(right) < 1
        while phi(right) >= 1:
            right *= 2
    else:
        raise ValueError("which must be 'closest' or 'farthest'")

    # Bisection root-finding
    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        val = phi(mid)
        if abs(val - 1) < tol:
            lam = mid
            break
        if which == "closest":
            if val > 1:
                left = mid
            else:
                right = mid
        else:  # farthest
            if val > 1:
                left = mid
            else:
                right = mid
    else:
        raise RuntimeError("Root finding did not converge")

    # Recover point
    u = w / (1 - lam * alpha)
    x = Q @ u
    return c + x

def tangent_points_to_ellipsoid(Q, c, x):
    """
    TODO: Test this! - not used for now
    Find the two points on the ellipsoid (y - c)^T Q^{-1} (y - c) = 1
    such that the lines from x to y are tangent to the ellipsoid.

    Example:
        Q = np.array([[4, 0],
              [0, 1]])  # Ellipse aligned with axes, semi-axes 2 and 1
        c = np.array([0.0, 0.0])
        x = np.array([3.0, 2.0])  # Outside the ellipse

        ys = tangent_points_to_ellipsoid(Q, c, x)

    Args:
        Q: (n, n) positive definite matrix defining the ellipsoid
        c: (n,) center of the ellipsoid
        x: (n,) external point (must satisfy (x - c)^T Q^{-1} (x - c) > 1)

    Returns:
        ys: list of two tangent points on the ellipsoid (each an (n,) array)
    """
    Qinv = np.linalg.inv(Q)
    A = Qinv @ (x - c)
    a = (x - c).T @ Qinv @ (x - c)

    # Solve for lambdas from the quadratic: lambda^2 * A^T Q A - 2 lambda * A^T A + a - 1 = 0
    QA = Q @ A
    ATA = A.T @ A
    ATQA = A.T @ QA
    coeffs = [ATQA, -2 * ATA, a - 1]
    lambdas = np.roots(coeffs)

    ys = []
    for lam in lambdas:
        if np.isreal(lam):  # Only accept real solutions
            lam = np.real(lam)
            y = x - lam * QA
            if np.isclose((y - c).T @ Qinv @ (y - c), 1.0, atol=1e-6):
                ys.append(y)

    return ys

if __name__ == '__main__':
    G = np.array([[0., 1.],
                  [1., 0.]])
    c = np.zeros(2)
    p_ego = np.array([2., 0.])

    p_star = zonotope_farthest_point(p_ego, c, G)

    # Expected farthest vertices from the analysis:
    # (-1, 1) and (-1, -1), both distance sqrt(10)
    candidates = [np.array([-1., 1.]), np.array([-1., -1.])]
    dists = [np.linalg.norm(p_ego - p)**2 for p in candidates]
    d_star = max(dists)
    dist_computed = np.linalg.norm(p_ego - p_star)**2

    print("Computed farthest point:", p_star)
    print("Squared distance:", dist_computed)

    # Check if p_star matches one of the expected vertices (within tolerance)
    assert any(np.allclose(p_star, p, atol=1e-8) for p in candidates), \
        f"p_star={p_star} not one of expected {candidates}"
    assert np.isclose(dist_computed, d_star, atol=1e-8), \
        f"Distance {dist_computed} != expected {d_star}"