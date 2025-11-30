import numpy as np


def build_A(N, dx):
    """
    Trapezoidal integration matrix A of shape (N+1, N+1),
    such that (A @ u)[i] approximates ∫_0^{x_i} u(s) ds.

    Same structure as original version, but vectorized.
    """
    A = np.zeros((N+1, N+1), dtype=float)

    idx = np.arange(1, N+1)
    A[idx, 0] = 0.5
    A[idx, idx] = 0.5

    rows, cols = np.tril_indices(N+1, k=-1)
    mask = (rows >= 1) & (cols >= 1)
    A[rows[mask], cols[mask]] = 1.0

    A *= dx
    return A


def finite_diff(f, dx):
    """
    Initial guess for derivative using simple finite differences (same scheme as before).
    """
    f = np.asarray(f, dtype=float)
    N = len(f) - 1
    u0 = np.empty_like(f)

    # boundaries
    u0[0] = (f[1] - f[0]) / dx
    u0[N] = (f[N] - f[N-1]) / dx

    # interior: centered difference
    u0[1:N] = (f[2:] - f[:-2]) / (2 * dx)

    return u0


def tvdiff_1d(
    f,
    dx,
    alpha=1e-2,
    eps=1e-3,
    max_iter=50,
    tol=1e-6,
    verbose=False,
):
    """
    1D Total Variation regularized differentiation (Chartrand-style).

    This version is mathematically equivalent to the original dense implementation (see tv_diff.py):
    - H = A^T A + alpha * L  (full matrix, no approximation)
    - g = A^T(Au - f) + alpha * L u

    Optimizations:
    - Use np.diff(u) instead of D @ u
    - Vectorized construction of L from w (still as a full (N+1)x(N+1) matrix)
    - Preallocate temporary arrays where reasonable
    """

    f = np.asarray(f, dtype=float)
    N = len(f) - 1

    # Fixed operators
    A = build_A(N, dx)
    At = A.T
    AtA = At @ A  # full Hessian data term (same as original)

    # Initial guess
    u = finite_diff(f, dx)

    # Preallocate work arrays
    Du = np.empty(N, dtype=float)
    w = np.empty(N, dtype=float)
    main_diag = np.empty(N+1, dtype=float)
    off_diag = np.empty(N, dtype=float)
    L = np.empty((N+1, N+1), dtype=float)
    Au = np.empty_like(u)
    g = np.empty_like(u)
    H = np.empty_like(AtA)

    idx_main = np.arange(N+1)
    idx_off = np.arange(N)

    for it in range(max_iter):

        if verbose:
            print(f"Iteration number: {it}")

        # 1) Discrete derivative Du[i] = u[i+1] - u[i]
        Du[:] = u[1:] - u[:-1]

        # 2) TV weights: w_i = 1 / sqrt(Du_i^2 + eps^2)
        w[:] = 1.0 / np.sqrt(Du * Du + eps * eps)

        # 3) Build L = dx * D^T diag(w) D as a full matrix (same operator as original)
        main_diag[0] = w[0]
        main_diag[1:-1] = w[:-1] + w[1:]
        main_diag[-1] = w[-1]
        off_diag[:] = -w

        L.fill(0.0)
        L[idx_main, idx_main] = main_diag
        L[idx_off, idx_off + 1] = off_diag
        L[idx_off + 1, idx_off] = off_diag
        L *= dx  # crucial: same scaling as your L

        # 4) Gradient g = A^T(Au - f) + alpha * L u
        Au[:] = A @ u
        g[:] = At @ (Au - f)
        g += alpha * (L @ u)

        # 5) Hessian H = A^T A + alpha * L  (full exact Hessian)
        H[:] = AtA
        H += alpha * L

        # 6) Solve H s = -g
        try:
            s = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            raise RuntimeError("Linear system is singular or ill-conditioned in tvdiff_1d")

        # 7) Update
        u_new = u + s

        # 8) Convergence check
        s_norm = np.linalg.norm(s)
        u_norm = np.linalg.norm(u) + 1e-12
        rel_step = s_norm / u_norm

        if verbose:
            print(f"Iter {it:3d}: ||s|| = {s_norm:.3e}, rel_step = {rel_step:.3e}")

        u = u_new

        if rel_step < tol:
            if verbose:
                print("Converged.")
            break

    return u


if __name__ == "__main__":
    import time
    from data_sim.lorenz_model import generate_lorenz_data

    # ----- TIMER START -----
    t_start = time.perf_counter()

    # ----- generate Lorenz data (clean) -----
    t_max = 100.0
    dt = 0.05
    X, Xdot, t = generate_lorenz_data(t_max, dt=dt)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    # ----- add noise -----
    rng = np.random.default_rng(0)
    noise_level = 0.5
    x_noisy = x + noise_level * rng.standard_normal(size=x.shape)
    y_noisy = y + noise_level * rng.standard_normal(size=y.shape)
    z_noisy = z + noise_level * rng.standard_normal(size=z.shape)

    # build f = X_noisy - X_noisy[0]
    f_x = x_noisy - x_noisy[0]
    f_y = y_noisy - y_noisy[0]
    f_z = z_noisy - z_noisy[0]

    # ----- TV differentiation: TIMER START 2 -----
    t_tv_start = time.perf_counter()

    u_tv_x = tvdiff_1d(f_x, dx=dt, alpha=1e-2, eps=1e-3,
                       max_iter=50, tol=1e-6, verbose=True)
    u_tv_y = tvdiff_1d(f_y, dx=dt, alpha=1e-2, eps=1e-3,
                       max_iter=50, tol=1e-6, verbose=False)
    u_tv_z = tvdiff_1d(f_z, dx=dt, alpha=1e-2, eps=1e-3,
                       max_iter=50, tol=1e-6, verbose=False)

    # ----- TIMER END 2 -----
    t_tv_end = time.perf_counter()
    print(f"\nTV-diff execution time: {t_tv_end - t_tv_start:.4f} seconds\n")
