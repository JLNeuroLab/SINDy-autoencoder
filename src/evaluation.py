"""
Basic methods to evaluate a SINDy Autoencoder

Only intrinsic evaluation here:
- errore di ricostruzione
- consistenza della dinamica latente
- rollout open-loop nello spazio osservato
- residui a un passo

Model comparison evaluations (baseline, persistence, lineare, ecc.)
can be placed elsewhere (es. src/baselines.py o in experiments/).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd.functional import jvp

from .model import TimeSeriesDataset
from utils.diff_methods import compute_derivatives


def _to_tensor(x, device: str) -> torch.Tensor:
    """Convert numpy array or torch.Tensor to a float torch.Tensor on a given device."""
    if isinstance(x, torch.Tensor):
        return x.detach().clone().float().to(device)
    return torch.from_numpy(np.asarray(x, dtype=float)).float().to(device)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, torch.Tensor]:
    """
    Compute R² score (coefficient of determination), both mean and per-dimension.

    Parameters
    ----------
    y_true, y_pred : torch.Tensor
        Tensors with shape (..., D), where D is the number of output dimensions.

    Returns
    -------
    r2_mean : float
        Average R² across all dimensions.
    r2_per_dim : torch.Tensor
        R² score for each dimension, shape [D].
    """
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0) + 1e-12

    r2_per_dim = 1.0 - ss_res / ss_tot
    r2_mean = r2_per_dim.mean().item()
    return r2_mean, r2_per_dim

# ---------------------------------------------------------------------
# 1) Reconstruction error of the autoencoder
# ---------------------------------------------------------------------

@torch.no_grad()
def reconstruction_error(model,
                         X,
                         batch_size, 
                         device="cpu"
    ):
    """
    Evaluate reconstruction error x -> AE -> x_hat.

    Parameters
    ----------
    model : SINDy_Autoencoder (or compatible with .autoencoder.encode/decode)
    X : np.ndarray or torch.Tensor
        Shape [N_traj, T, x_dim]
    batch_size : int
        Batch size for evaluation.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    dict with:
        - mse_recon : float
        - r2_recon_mean : float
        - r2_recon_per_dim : np.ndarray, shape [x_dim]
    """
    model = model.to(device)
    model.eval()

    X_torch = _to_tensor(X, device)
    N, T, D = X_torch.shape

    dataset = TimeSeriesDataset(X_torch, None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_true = []
    all_pred = []

    for x_seq in loader:
        x_seq.to(device)
        B, T, D = x_seq.shape

        x_flat = x_seq.reshape(B * T, D)
        z_flat = model.autoencoder.encode(x_flat)
        xhat_flat = model.autoencoder.decode(z_flat)

        xhat_seq = xhat_flat.reshape(B, T, D)

        all_true.append(x_seq.reshape(-1, D))
        all_pred.append(xhat_seq.reshape(-1, D))

    x_true_all = torch.cat(all_true, dim=0)
    x_pred_all = torch.cat(all_pred, dim=0)

    mse_recon = torch.mean((x_true_all - x_pred_all) ** 2).item()
    r2_mean, r2_per_dim = r2_score(x_true_all, x_pred_all)

    return {
        "mse_recon": mse_recon,
        "r2_recon_mean": r2_mean,
        "r2_recon_per_dim": r2_per_dim.cpu().numpy(),
    }

# ---------------------------------------------------------------------
# 2) Consistency of latent dynamic: zdot_true vs zdot_pred
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_latent_dynamics(model,
    X,
    dt: float,
    diff_method: str = "finite-diff",
    diff_kwargs: dict | None = None,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    
    """
    Compare latent derivatives:
        zdot_true = J_phi(x) * xdot     (via encoder + JVP)
        zdot_pred = SINDy(z)

    Parameters
    ----------
    model : SINDy_Autoencoder
    X : array-like, shape [N_traj, T, x_dim]
    dt : float
        Time step between samples.
    diff_method : str
        Derivative estimation method (consistent with training).
    diff_kwargs : dict
        Extra arguments forwarded to compute_derivatives().
    batch_size : int
    device : str

    Returns
    -------
    dict with:
        - mse_dz : float
        - r2_dz_mean : float
        - r2_dz_per_dim : np.ndarray [z_dim]
    """
    model = model.to(device)
    model.eval()

    if diff_kwargs is None:
        diff_kwargs = {}

    # Ensure numpy format for derivative estimation
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) \
        else np.asarray(X, dtype=float)
    
    Xdot_np = compute_derivatives(
                        X_np,
                        dt,
                        diff_method=diff_method,
                        **diff_kwargs
    )
    X_torch = torch.from_numpy(X_np).float().to(device)
    Xdot_torch = torch.from_numpy(Xdot_np).float().to(device)

    dataset = TimeSeriesDataset(X_torch, Xdot_torch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    zdot_true_all = []
    zdot_pred_all = []
    mse_list = []

    for x_seq, xdot_seq in loader:
        x_seq = x_seq.to(device)           # [B, T, x_dim]
        xdot_seq = xdot_seq.to(device)
        B, T, D = x_seq.shape

        x_flat = x_seq.reshape(B * T, D)
        xdot_flat = xdot_seq.reshape(B * T, D)

        def enc_func(x):
            return model.autoencoder.encode(x)
        
        z_flat, zdot_true_flat = jvp(
            enc_func,
            (x_flat,),
            (xdot_flat,),
            create_graph=False,
        )

        zdot_pred_flat = model.sindy(z_flat)

        mse_batch = torch.mean((zdot_true_flat - zdot_pred_flat) ** 2).item()
        mse_list.append(mse_batch)

        zdot_true_all.append(zdot_true_flat.detach())
        zdot_pred_all.append(zdot_pred_flat.detach())

    zdot_true_all = torch.cat(zdot_true_all, dim=0)
    zdot_pred_all = torch.cat(zdot_pred_all, dim=0)

    r2_mean, r2_per_dim = r2_score(zdot_true_all, zdot_pred_all)

    return {
        "mse_dz": float(np.mean(mse_list)),
        "r2_dz_mean": r2_mean,
        "r2_dz_per_dim": r2_per_dim.cpu().numpy(),
    }

# -------------------------------------------------------------------------
# 3) Latent integration + observed rollout
# -------------------------------------------------------------------------
def rk4_step_latent(model, z: torch.Tensor, dt: float) -> torch.Tensor:
    """
    One RK4 step for z' = f(z) = SINDy(z).
    """
    f = model.sindy
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@torch.no_grad()
def rollout_from_x0(
    model,
    x0: torch.Tensor,
    n_steps: int,
    dt: float,
    device: str = "cpu",
    integrator: str = "euler",
):
    """
    Perform open-loop rollout starting from initial state x0:
        x0 -> encode -> z0
        integrate z forward in latent space using SINDy
        decode at each step to obtain x_hat

    Parameters
    ----------
    model : SINDy_Autoencoder
    x0 : tensor [x_dim] or [1, x_dim]
    n_steps : int
    dt : float
    device : str
    integrator : {"euler", "rk4"}

    Returns
    -------
    x_traj : torch.Tensor, shape [1, n_steps, x_dim]
    z_traj : torch.Tensor, shape [1, n_steps, z_dim]
    """
    model = model.to(device)
    model.eval()

    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
    x0 = x0.to(device)

    z = model.autoencoder.encode(x0)
    z_traj = [z]
    x_traj = [x0]

    for _ in range(1, n_steps):
        if integrator == "rk4":
            z = rk4_step_latent(model, z, dt)
        else:
            z = z + dt * model.sindy(z)

        x_hat = model.autoencoder.decode(z)
        z_traj.append(z)
        x_traj.append(x_hat)

    z_traj = torch.stack(z_traj, dim=1)
    x_traj = torch.stack(x_traj, dim=1)

    return x_traj, z_traj


# -------------------------------------------------------------------------
# 4) Rollout evaluation over a dataset
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rollout_dataset(
    model,
    X,
    dt: float,
    horizons=(1, 5, 10, 20, 50),
    device: str = "cpu",
    integrator: str = "euler",
) -> dict:
    """
    Evaluate open-loop rollout accuracy for multiple prediction horizons.

    For each trajectory X[i]:
      - take x0 = X[i, 0]
      - perform latent rollout of length max(horizons)
      - compare predicted vs true trajectory segments

    Parameters
    ----------
    model : SINDy_Autoencoder
    X : array-like, shape [N_traj, T, x_dim]
    dt : float
    horizons : iterable of ints
    device : str
    integrator : str

    Returns
    -------
    dict with:
        - mse_rollout : {h: mse_mean}
        - r2_rollout_mean : {h: r2_mean}
        - r2_rollout_per_dim : {h: np.ndarray}
        - horizons : list of valid horizons used
    """
    model = model.to(device)
    model.eval()

    X_torch = _to_tensor(X, device)
    N_traj, T, D = X_torch.shape

    max_h = min(max(horizons), T)
    horizons = sorted([h for h in horizons if h <= T])

    mse_per_h = {h: [] for h in horizons}
    r2_mean_per_h = {h: [] for h in horizons}
    r2_dim_per_h = {h: [] for h in horizons}

    for i in range(N_traj):
        x_true = X_torch[i]
        x0 = x_true[0]

        xhat_traj, _ = rollout_from_x0(
            model,
            x0,
            n_steps=max_h,
            dt=dt,
            device=device,
            integrator=integrator,
        )
        xhat_traj = xhat_traj[0]

        for h in horizons:
            xt = x_true[:h]
            xp = xhat_traj[:h]

            mse_h = torch.mean((xt - xp) ** 2).item()
            r2_mean_h, r2_per_dim_h = r2_score(xt, xp)

            mse_per_h[h].append(mse_h)
            r2_mean_per_h[h].append(r2_mean_h)
            r2_dim_per_h[h].append(r2_per_dim_h.cpu().numpy())

    mse_avg = {h: float(np.mean(mse_per_h[h])) for h in horizons}
    r2_mean_avg = {h: float(np.mean(r2_mean_per_h[h])) for h in horizons}
    r2_dim_avg = {
        h: np.mean(np.stack(r2_dim_per_h[h], axis=0), axis=0)
        for h in horizons
    }

    return {
        "mse_rollout": mse_avg,
        "r2_rollout_mean": r2_mean_avg,
        "r2_rollout_per_dim": r2_dim_avg,
        "horizons": horizons,
    }

# -------------------------------------------------------------------------
# 5) One-step residuals in observed space
# -------------------------------------------------------------------------

@torch.no_grad()
def residuals_one_step(
    model,
    X,
    dt: float,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute one-step prediction residuals:
        r_t = x_{t+1}^{true} - x_{t+1}^{pred}

    The prediction is generated by:
        x_t -> encode -> z_t
        z_{t+1} = z_t + dt * sindy(z_t)
        x_{t+1}^{pred} = decode(z_{t+1})

    Parameters
    ----------
    model : SINDy_Autoencoder
    X : array-like, shape [N_traj, T, x_dim]
    dt : float
    device : str

    Returns
    -------
    residuals : torch.Tensor
        Shape [N_traj, T-1, x_dim]
    """
    model = model.to(device)
    model.eval()

    X_torch = _to_tensor(X, device)
    N_traj, T, D = X_torch.shape

    all_res = []

    for i in range(N_traj):
        x_seq = X_torch[i]           # [T, D]
        x_t = x_seq[:-1]             # [T-1, D]
        x_next_true = x_seq[1:]      # [T-1, D]

        z_t = model.autoencoder.encode(x_t)   # [T-1, z_dim]
        z_next = z_t + dt * model.sindy(z_t)  # [T-1, z_dim]
        x_next_pred = model.autoencoder.decode(z_next)

        res_i = (x_next_true - x_next_pred).detach()
        all_res.append(res_i.unsqueeze(0))

    residuals = torch.cat(all_res, dim=0)  # [N_traj, T-1, D]
    return residuals.cpu()
