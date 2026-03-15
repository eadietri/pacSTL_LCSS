import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def ellipsoid_distance(x, A, b, p=2):
    x = np.asarray(x, dtype=float).ravel()
    return np.linalg.norm(A @ x - b, ord=p)

def myminval(fn, x0):
    res = minimize(
        fn,
        x0=np.atleast_1d(x0),          # ensure array shape (k,)
        method="Nelder-Mead",
        options=dict(xatol=1e-6, fatol=1e-6, maxfev=200*len(np.atleast_1d(x0)),
                     disp=False))
    return res.fun

def proj_to_2_plotting(x, y, A, b, p, centre):

    def objective(v):                         # v has length 4
        return ellipsoid_distance(
            np.concatenate((np.array([x, y]), v)), A, b, p)

    # good initial guess = centre’s coordinates for those dims
    return myminval(objective, centre)


def plot(ax, step_data, A, b, data):
    xc = np.mean(step_data[:, [2, 3, 4, 5]], axis=0)

    xlo = step_data.min(axis=0)*2   # [x_min, y_min, z_min]
    xhi = step_data.max(axis=0)*2   # [x_max, y_max, z_max]

    xmin = xlo[0] - 0.45 * (xhi[0] - xlo[0])
    xmax = xhi[0] + 0.45 * (xhi[0] - xlo[0])
    ymin = xlo[1] - 0.45 * (xhi[1] - xlo[1])
    ymax = xhi[1] + 0.45 * (xhi[1] - xlo[1])

    nmgrid = 100
    x_grid = np.linspace(xmin, xmax, nmgrid)
    y_grid = np.linspace(ymin, ymax, nmgrid)

    [X, Y] = np.meshgrid(x_grid, y_grid)
    Zxy_full = np.zeros((nmgrid, nmgrid))
    for i in range(nmgrid):
        for j in range(nmgrid):
            Zxy_full[i, j] = proj_to_2_plotting(X[i, j], Y[i, j], A, b, 2, xc)

    ax.plot(step_data[:, 0], step_data[:, 1], marker=".", linestyle='None', color='red')
    ax.contour(X, Y, Zxy_full, levels=[1], colors=['red'], linewidths=1.5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Ellipsoidal Reachable Set Projection')


def convert_A_b_to_Q_c_r(A, b):
    """
    Convert ellipsoid of form ||A x - b|| <= 1
    into standard ellipsoid form: (x - c)^T Q (x - c) <= r
    Returns (Q, c, r)
    """
    Q = A.T @ A
    Q_inv = np.linalg.inv(Q)
    c = Q_inv @ A.T @ b
    r = (b.T @ A @ Q_inv @ A.T @ b) - (b.T @ b) + 1
    return Q, c, r

def plot_ellipse(Q, c, r, ax, alpha=0.15, color='green', **kwargs):
    """
    Plot ellipse defined by (x-c)^T Q (x-c) <= r in 2D
    """
    eigvals, eigvecs = np.linalg.eigh(Q)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    # Scale the ellipse by sqrt(r)
    ellipse = eigvecs @ np.diag(np.sqrt(r) / np.sqrt(eigvals)) @ circle
    ellipse = ellipse.T + c
    ax.plot(ellipse[:, 0], ellipse[:, 1], **kwargs, color=color, alpha=alpha)

    return ax

def project_ellipsoid_to_2d(A, b, diml, dimh):
    """
    Project ellipsoid ||A x - b|| <= 1 from nD to 2D.
    Returns (Q_2d, c_2d, r)
    """
    indices = [diml, dimh]
    Q, c, r = convert_A_b_to_Q_c_r(A, b)
    Q_2d = Q[np.ix_(indices, indices)]
    c_2d = c[indices]
    return Q_2d, c_2d, r

def project_plot_ellipsoid_2d(Ab, n_low, n_up, n_pred):
    ax = None
    for i in range(n_pred +1):
        Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(Ab[i][0], Ab[i][1], n_low, n_up)
        ax = plot_ellipse(Q_proj, c_proj, r_proj, ax=ax, label=f"Ellipsoid {i}")
    plt.ylim(-0.05, 0.05)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Projection of ellipsoids to first 2 dimensions")
    plt.grid(True)