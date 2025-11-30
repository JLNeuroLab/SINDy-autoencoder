import numpy as np

def _build_D(N):
    """
    Build the forward-difference matrix D of shape (N, N+1),
    such that (D @ u)[i] = u[i+1] - u[i].
    """
    D = np.zeros((N, N+1), dtype=float)

    for i in range(N):
        D[i, i] = -1.0
        D[i, i+1] = 1.0
    
    return D

def _build_A(N, dx):
    """
    Build the 1D integration matrix A (trapezoidal rule)
    of shape (N+1, N+1), such that (A @ u)[i] approximates
    the integral ∫_0^{x_i} u(s) ds.

    - (A u)[0] = 0 (integral from 0 to 0)
    - For i >= 1:
        (A u)[i] = dx * [ 0.5*u[0] + u[1] + ... + u[i-1] + 0.5*u[i] ]
    """

    A = np.zeros((N+1, N+1), dtype=float)

    for i in range(1, N+1):
        A[i, 0] = 0.5
        A[i, i] = 0.5

        if i > 1:
            A[i, 1:i] = 1.0
        
    A *= dx
    
    return A

def finite_diff(f, dx):

    """
    Build an initial guess u0 for the derivative of f
    using simple finite differences.
    """

    f = np.asarray(f, dtype=float)
    N = len(f) - 1
    u0 = np.zeros_like(f)

    # Left boundary: forward difference
    u0[0] = (f[1] - f[0]) / dx

    # Right boundary: backward difference
    u0[N] = (f[N] - f[N-1]) / dx

    # Interior points: centered difference
    for i in range(1, N):
        u0[i] = (f[i+1] - f[i-1]) / (2*dx)
    
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
    1D Total Variation-based numerical differentiation (Chartrand-style).
    """

    f = np.asarray(f, dtype=float)
    N = len(f) - 1

    # Build fixed operators
    D = _build_D(N)
    A = _build_A(N, dx)

    # Initial guess for u
    u = finite_diff(f, dx)

    # Precompute A^T A
    At = A.T
    AtA = At @ A

    for it in range(max_iter):
        print(f"Iteration number: {it}")
        # 1) Descrete derivative Du
        Du = D @ u

        # 2) TV weights: w_i = 1 / sqrt(Du_i^2 + eps^2)
        w = 1 / np.sqrt(Du**2 + eps**2)

        # 3) L_n = dx * D^T diag(w) D
        WD = w[:, None] * D           # each row of D scaled by w_i
        L = dx * (D.T @ WD)           # (N+1 x N+1)
                                     # This is equivalent to E = np.diag(w)
                                     #                       L = dx * (D.T @ E @ D)
        # 4) Gradient g = A^T(Au - f) + alpha * L u
        Au = A @ u
        g = At @ (Au - f) + alpha * (L @ u)

        # 5) Hessian H = A^T A + alpha * L
        H = AtA + alpha * L

        # 6) Solve H s = -g
        try:
            s = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            raise RuntimeError("Linear system is singular or ill-conditioned in tvdiff_1d")
        
        # 7) Update
        u_new = u + s

        # 8) Check convergence
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
    t_max = 30.0
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

    # ----- rest of your script: naive derivative, plots, etc. -----