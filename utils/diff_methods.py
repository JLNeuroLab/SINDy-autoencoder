import numpy as np
import torch
from utils.tvdiff_1d import tvdiff_1d


__all__ = [
    "finite_diff_time_np",
    "tvdiff_time_np",
    "compute_time_derivative",
]


def finite_diff_np(X, dt: float) -> np.ndarray:

    X = np.asarray(X, dtype=float)
    N, T, D = X.shape
    Xdot = np.empty_like(X)

    # Central values (centered difference)
    Xdot[:, 1:-1, :] = (X[:, 2:, :] - X[:, :-2, :]) / (dt * 2.0)

    # Left border (forward difference)
    Xdot[:, 0, :] = (X[:, 1, :] - X[:, 0, :]) / dt

    # Right border (backward difference)
    Xdot[:, -1, :] = (X[:, -2, :] - X[:, -1, :]) / dt

    return Xdot

def tv_diff_np(X,
        dt: float,
        alpha: float = 1e-2,
        eps: float = 1e-3,
        max_iter: int = 50,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> np.ndarray:

    X = np.asarray(X, dtype=float)
    N, T, D = X.shape # shape (N_traj or batch, n_time_steps, state_dim)
    Xdot = np.empty_like(X)

    for n in range(N):
        for d in range(D):
            x = X[n, :, d]
            f = x - x[0]

            u_tv = tvdiff_1d(
                        f,
                        dt,
                        alpha,
                        eps,
                        max_iter,
                        tol,
                        verbose
            )
            Xdot[n, :, d] = u_tv
    
    return Xdot

def compute_derivatives(
        X,
        dt: float,
        diff_method: str = "finite",
        tv_alpha: float = 1e-2,
        tv_eps: float = 1e-3,
        tv_max_iter: int = 50,
        tv_tol: float = 1e-6,
        tv_verbose: bool = False,
    ) -> np.ndarray:

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X, dtype=float)

    if diff_method in ["finite_diff", "finite-diff"]:
        Xdot_np = finite_diff_np(X, dt)
    
    elif diff_method in ["tv_diff", "tv-diff"]:
        Xdot_np = tv_diff_np(X,
                             dt,
                             alpha=tv_alpha,
                             eps=tv_eps,
                             max_iter=tv_max_iter,
                             tol=tv_tol,
                             verbose=tv_verbose
                             )
    else:
        raise ValueError(f"diff_method sconosciuto: {diff_method}")
    
    return Xdot_np
