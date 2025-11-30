import numpy as np
import torch
from numpy.polynomial.legendre import legval
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Here we implemente a simulation of the Lorenz attractor using scipy.

def lorenz_system(
                  t,
                  states, 
                  sigma, 
                  beta, 
                  rho
                  ):
    x, y, z = states

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz])

def generate_lorenz_data(t_max, 
                         dt=0.01,
                         x0 = [-8, 7, 27],
                         sigma=10, 
                         beta=8/3, 
                         rho=28
                        ):
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lorenz_system, 
                    t_span=[0, t_max], 
                    y0=x0, 
                    t_eval=t_eval,
                    args=(sigma, beta, rho))
    X = sol.y.T
    Xdot = np.array([lorenz_system(0, xi, sigma, beta, rho) for xi in X])

    return X, Xdot, t_eval

# ---------------------------------------------------------------------
# Build spatial modes: first 6 Legendre polynomials on a 1D grid
# ---------------------------------------------------------------------
def make_legendre_modes(n_spatial: int = 128, n_modes: int = 6):
    s = np.linspace(-1.0, 1.0, n_spatial)
    modes = []
    for k in range(n_modes):
        coeffs = np.zeros(k + 1)
        coeffs[-1] = 1.0
        pk = legval(s, coeffs)
        pk /= np.linalg.norm(pk)
        modes.append(pk)
    U = np.stack(modes, axis=0)  # (n_modes, n_spatial)
    return U


def lorenz_rhs_torch(z, sigma=10.0, beta=8/3, rho=28.0):
    """
    z: (N, 3) tensor
    returns dz/dt: (N, 3)
    """
    x = z[:, 0]
    y = z[:, 1]
    z3 = z[:, 2]

    dx = sigma * (y - x)
    dy = x * (rho - z3) - y
    dz = x * y - beta * z3
    return torch.stack([dx, dy, dz], dim=1)


def rk4_step(z, dt, sigma=10.0, beta=8/3, rho=28.0):
    """
    One RK4 step for batched Lorenz.
    z: (N, 3)
    """
    k1 = lorenz_rhs_torch(z, sigma, beta, rho)
    k2 = lorenz_rhs_torch(z + 0.5 * dt * k1, sigma, beta, rho)
    k3 = lorenz_rhs_torch(z + 0.5 * dt * k2, sigma, beta, rho)
    k4 = lorenz_rhs_torch(z + dt * k3, sigma, beta, rho)
    return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def generate_lorenz_hi_dim_gpu(
    n_traj: int,
    dt: float = 0.02,
    t_final: float = 5.0,
    n_spatial: int = 128,
    device: str = "cuda",
    seed: int | None = None,
):
    """
    GPU version: integrates Lorenz for n_traj trajectories in parallel using RK4
    and maps to high-dimensional space via Legendre modes.

    Returns:
        t  : (T,) numpy array
        X  : (n_traj, T, n_spatial) numpy array
        Z  : (n_traj, T, 3) numpy array
        U  : (6, n_spatial) numpy array (Legendre modes)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Time grid
    T = int(t_final / dt)
    t = dt * np.arange(T)

    # Spatial modes (stay in numpy, then convert to torch)
    U_np = make_legendre_modes(n_spatial=n_spatial, n_modes=6)  # (6, n_spatial)
    U = torch.from_numpy(U_np).to(device=device, dtype=torch.float32)
    u1, u2, u3, u4, u5, u6 = U   # each (n_spatial,)

    # Initial conditions on GPU
    z0_np = np.stack([
        np.random.uniform(-36.0, 36.0, size=n_traj),
        np.random.uniform(-48.0, 48.0, size=n_traj),
        np.random.uniform(-16.0, 66.0, size=n_traj),
    ], axis=1)  # (n_traj, 3)

    z = torch.from_numpy(z0_np).to(device=device, dtype=torch.float32)  # (N, 3)

    # Allocate trajectories
    Z = torch.zeros((n_traj, T, 3), device=device, dtype=torch.float32)
    X = torch.zeros((n_traj, T, n_spatial), device=device, dtype=torch.float32)

    for k in range(T):
        # store current z
        Z[:, k, :] = z

        # map to high-dimensional x
        z1 = Z[:, k, 0]  # (N,)
        z2 = Z[:, k, 1]
        z3 = Z[:, k, 2]

        # broadcast: (N, 1) * (1, n_spatial) -> (N, n_spatial)
        x_k = (
            z1[:, None] * u1[None, :]
          + z2[:, None] * u2[None, :]
          + z3[:, None] * u3[None, :]
          + (z1**3)[:, None] * u4[None, :]
          + (z2**3)[:, None] * u5[None, :]
          + (z3**3)[:, None] * u6[None, :]
        )
        X[:, k, :] = x_k

        # RK4 step to next time
        z = rk4_step(z, dt)

    # send back to CPU as numpy (to be compatible with your current Trainer)
    X_np = X.cpu().numpy()
    Z_np = Z.cpu().numpy()
    U_np = U.cpu().numpy()

    return t, X_np, Z_np, U_np