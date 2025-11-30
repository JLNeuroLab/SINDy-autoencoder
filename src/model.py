import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):

    def __init__(self, X, Xdot=None):
        super().__init__()
        self.X = X
        self.Xdot = Xdot

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x_seq = self.X[index]
        if self.Xdot is None:
            return x_seq
        else:
            xdot_seq = self.Xdot[index]
            return x_seq, xdot_seq


class Encoder(nn.Module):

    def __init__(self, x_dim, z_dim, hidden_dims: tuple):
        super().__init__()

        layers = []
        in_dim = x_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Sigmoid())
            in_dim = h

        layers.append(nn.Linear(in_dim, z_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    
    def __init__(self, z_dim, x_dim, hidden_dims: tuple):
        super().__init__()

        layers = []
        in_dim = z_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Sigmoid())
            in_dim = h
        
        layers.append(nn.Linear(in_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
    
class Autoencoder(nn.Module):

    def __init__(self, x_dim, z_dim,
                 enc_hidden,
                 dec_hidden,
                 ):
        super().__init__()

        self.encoder = Encoder(x_dim, z_dim, hidden_dims=enc_hidden)
        self.decoder = Decoder(z_dim, x_dim, hidden_dims=dec_hidden)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
def build_library_torch(z, poly_order, include_bias=True):

    B, d = z.shape
    # List to store polinomials
    terms = []
    if include_bias:
        terms.append(torch.ones(B, 1, device=z.device))
    # Linear terms
    terms.append(z)
    # quadratic terms
    if poly_order >= 2:
        for i in range(d):
            for j in range(i, d):
                terms.append((z[:, i] * z[:, j]).unsqueeze(1))
    # cubic terms
    if poly_order >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    terms.append((z[:, i] * z[:, j] * z[:, k]).unsqueeze(1))
    
    Theta = torch.cat(terms, dim=1) # shape (B, n_terms)

    return Theta

class SINDy_layer(nn.Module):

    def __init__(self, z_dim, poly_order, include_bias=True):
        super().__init__()
        self.z_dim = z_dim
        self.poly_order = poly_order
        self.include_bias = include_bias

        with torch.no_grad():
            dummy_z = torch.zeros(1, z_dim)
            Theta = build_library_torch(dummy_z, poly_order, include_bias)

            n_terms = Theta.shape[1]
        
        self.Xi = nn.Parameter(torch.zeros(n_terms, z_dim))
        self.register_buffer("Xi_mask", torch.ones_like(self.Xi))

        # latent normalization buffers (initialized as identity)
        self.register_buffer("z_mean", torch.zeros(1, z_dim))
        self.register_buffer("z_std",  torch.ones(1, z_dim))
    
    def set_normalization(self, z_mean: torch.Tensor, z_std: torch.Tensor):
        """
        Set latent normalization statistics.
        z_mean, z_std shape: [1, z_dim]
        """
        # make sure shapes match
        assert z_mean.shape == self.z_mean.shape
        assert z_std.shape == self.z_std.shape
        self.z_mean.copy_(z_mean)
        self.z_std.copy_(z_std)

    def forward(self, z):
        z_norm = (z - self.z_mean.to(z.device)) / self.z_std.to(z.device)
        
        Theta = build_library_torch( 
                                    z, 
                                    self.poly_order, 
                                    self.include_bias
                )
        Xi_eff = self.Xi * self.Xi_mask
        dz_dt = Theta @ Xi_eff

        return dz_dt
    

def finite_difference_time(z, dt):
    # z has shape (batch, T, z_dim), where T is sequence length
    # central: (z_{t+1} - z_{t-1}) / (2 dt)
    dz_dt = (z[:, 2:, :] - z[:, :-2, :]) / (2.0 * dt) # shape (batch, T-2, z_dim)
    z_mid = z[:, 1:-1, :]
    return z_mid, dz_dt


from torch.autograd.functional import jvp

class SINDy_Autoencoder(nn.Module):

    def __init__(self, 
                 x_dim, 
                 z_dim,
                 enc_hidden = (128, 64),
                 dec_hidden = (64, 128),
                 poly_order = 3,
                 include_bias = True,
                 ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.autoencoder = Autoencoder(x_dim,
                                       z_dim,
                                       enc_hidden=enc_hidden,
                                       dec_hidden=dec_hidden
                            )
        self.sindy = SINDy_layer(z_dim,
                                       poly_order=poly_order,
                                       include_bias=include_bias)
        
    # def forward(self,
    #             x_seq,
    #             xdot_seq,
    #             lambda_recon=1.0,
    #             lambda_dz=1.0,
    #             lambda_dx=1.0
    #     ):
    #     """
    #             x_seq:    [B, T, x_dim]
    #             xdot_seq: [B, T, x_dim]
    #             Returns:
    #             x_hat_seq: [B, T, x_dim]
    #             z_seq:     [B, T, z_dim]
    #             losses:    dict with total loss and single terms (Lrecon, Ldz, Ldx)
    #     """

    #     B, T, D = x_seq.shape        # shape (batch, sequence_length, x_dim (x1, ..., xn))
    #     assert D == self.x_dim
    #     assert xdot_seq.shape == x_seq.shape

    #     x_flat = x_seq.reshape(B * T, D).clone().requires_grad_(True)
    #     xdot_flat = xdot_seq.reshape(B * T, D)

    #     # --------------------------------------------------------
    #     # 1) Autoencoder: z = φ(x), x_hat = ψ(z)
    #     #    This is the forward pass
    #     # --------------------------------------------------------
    #     z_flat = self.autoencoder.encode(x_flat)
    #     xhat_flat = self.autoencoder.decode(z_flat)

    #     xhat_seq = xhat_flat.reshape(B, T, D)
    #     z_seq = z_flat.reshape(B, T, self.z_dim)
    #     # --------------------------------------------------------
    #     # L_recon
    #     # --------------------------------------------------------
    #     L_recon = F.mse_loss(xhat_seq, x_seq)

    #      # --------------------------------------------------------
    #     # 2) L_dz: zdot_true = J_φ(x) xdot, zdot_pred = Θ(z)Ξ
    #     # --------------------------------------------------------
    #     # zdot_pred: We first need the derivative of the latent space given by SINDy
    #     zdot_pred_flat = self.sindy(z_flat)

    #     # zdot_true: for each component k of z, ∂z_k/∂x ⋅ xdot
    #     # We calculate it with for loops because torch.autograd cannot calculate derivatives of vector fields but just scalars

    #     zdot_true_list = []
    #     for k in range(self.z_dim):

    #         grad_zk_x = torch.autograd.grad(
    #             z_flat[:, k].sum(), # scalar as pytorch can only compute derivatives of scalars
    #             x_flat,
    #             create_graph=True,
    #             retain_graph=True
    #         )[0]  # to get a single value instead of a tuple 
    #             # [B*T, x_dim]

    #         zdot_k = (grad_zk_x * xdot_flat).sum(dim=1, keepdim=True) # [B*T,1] this is the k component of zdot_true
    #         zdot_true_list.append(zdot_k)            

    #     zdot_true_flat = torch.cat(zdot_true_list, dim=1)  # [B*T, z_dim]

    #     L_dz = F.mse_loss(zdot_true_flat, zdot_pred_flat)

    #     # --------------------------------------------------------
    #     # 3) L_dx: xdot_pred = J_ψ(z) zdot_pred
    #     # --------------------------------------------------------

    #     xdot_pred_list = []

    #     for j in range(self.x_dim):

    #         grad_xj_z = torch.autograd.grad(
    #             xhat_flat[:, j].sum(),
    #             z_flat,
    #             create_graph=True,
    #             retain_graph=True
    #         )[0]

    #         xdot_j = (grad_xj_z * zdot_pred_flat).sum(dim=1, keepdim=True)   # [B*T,1] this is the jth component of xdot_pred
    #         xdot_pred_list.append(xdot_j)

    #     xdot_pred_flat = torch.cat(xdot_pred_list, dim=1) # [B*T, x_dim]

    #     L_dx = F.mse_loss(xdot_pred_flat, xdot_flat)

    #     # --------------------------------------------------------
    #     # 4) Total Loss (without L1_reg, STLSQ will be implemented in the training.
    #     # --------------------------------------------------------

    #     loss = (
    #         lambda_recon * L_recon +
    #         lambda_dx * L_dx +
    #         lambda_dz * L_dz
    #     )

    #     losses = {
    #         "loss": loss,
    #         "L_recon": L_recon,
    #         "L_dx": L_dx,
    #         "L_dz": L_dz
    #     }

    #     return xhat_seq, z_seq, losses

    def forward(self,
                x_seq,
                xdot_seq,
                lambda_recon=1.0,
                lambda_dz=1.0,
                lambda_dx=1.0
        ):
        """
                x_seq:    [B, T, x_dim]
                xdot_seq: [B, T, x_dim]
                Returns:
                x_hat_seq: [B, T, x_dim]
                z_seq:     [B, T, z_dim]
                losses:    dict with total loss and single terms (Lrecon, Ldz, Ldx)
        """

        B, T, D = x_seq.shape        # shape (batch, sequence_length, x_dim (x1, ..., xn))
        assert D == self.x_dim
        assert xdot_seq.shape == x_seq.shape

        x_flat = x_seq.reshape(B * T, D).clone().requires_grad_(True)
        xdot_flat = xdot_seq.reshape(B * T, D)

        # --------------------------------------------------------
        # 1) Encoder + JVP: z = φ(x), zdot_true = J_φ(x) * xdot
        # --------------------------------------------------------
        def enc_func(x):
            return self.autoencoder.encode(x)
        
        # jvp returns (φ(x), J_φ(x)*xdot)
        z_flat, zdot_true_flat = jvp(
            enc_func,
            (x_flat,),
            (xdot_flat,),
            create_graph=True
        )  # z_flat: [B*T, z_dim], zdot_true_flat: [B*T, z_dim]

        # --------------------------------------------------------
        # 2) Decoder: x_hat = ψ(z)  (serve per la ricostruzione)
        # --------------------------------------------------------
        xhat_flat = self.autoencoder.decode(z_flat) # shape (B*T, D)
        # reconstruct the dimensions
        xhat_seq = xhat_flat.reshape(B, T, D) # shape (B, T, x_dim)
        z_seq = z_flat.reshape(B, T, self.z_dim) # shape (B, T, z_dim)

        # --------------------------------------------------------
        # L_recon: Reconstruction loss
        # --------------------------------------------------------
        L_recon = F.mse_loss(x_seq, xhat_seq)
        # --------------------------------------------------------
        # 3) L_dz: zdot_true = J_φ(x) xdot (from step 2), zdot_pred = Θ(z)Ξ
        # --------------------------------------------------------
        # zdot_pred from SINDy on latent space
        zdot_pred_flat = self.sindy(z_flat)
        # --------------------------------------------------------
        # L_dz: loss on dynamics of latent space
        # --------------------------------------------------------
        L_dz = F.mse_loss(zdot_true_flat, zdot_pred_flat)
        # --------------------------------------------------------
        # 4) L_dx: xdot_pred = J_ψ(z) zdot_pred, via JVP sul decoder
        # --------------------------------------------------------
        def dec_func(z):
            return self.autoencoder.decode(z)
        # jvp over decoder: (ψ(z), J_ψ(z)*zdot_pred)
        # il valore ψ(z) che torna è numericamente ~ xhat_flat, but we don't need it
        _, xdot_pred_flat =jvp(
                        dec_func,
                        (z_flat,),
                        (zdot_pred_flat,),
                        create_graph=True
                    )  # xdot_pred_flat: [B*T, x_dim]
        # --------------------------------------------------------
        # L_dx: loss on dynamics on original space
        # --------------------------------------------------------
        L_dx = F.mse_loss(xdot_flat, xdot_pred_flat)

        # --------------------------------------------------------
        # 5) Total Loss (without L1_reg, STLSQ will be implemented in the training.
        # --------------------------------------------------------

        loss = (
            lambda_recon * L_recon +
            lambda_dx * L_dx +
            lambda_dz * L_dz
        )

        losses = {
            "loss": loss,
            "L_recon": L_recon,
            "L_dx": L_dx,
            "L_dz": L_dz
        }

        return xhat_seq, z_seq, losses
    
if __name__== "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    from lorenz_model import generate_lorenz_hi_dim_gpu
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ----- 1. Generate data like in the paper -----
    rng = np.random.default_rng(0)
    dt = 0.02

    t, X_train, Z_train, U = generate_lorenz_hi_dim_gpu(
        n_traj=512,     # training trajectories
        dt=dt,
        t_final=5.0,
        n_spatial=128,
        seed=0,
        device=device
    )

    print("X_train shape:", X_train.shape)   # (2048, 250, 128)

    from model import SINDy_Autoencoder, TimeSeriesDataset

    x_dim = X_train.shape[-1]   # 128
    z_dim = 3                   # as in the SI table

    model = SINDy_Autoencoder(
        x_dim=x_dim,
        z_dim=z_dim,
        enc_hidden=(64, 32),
        dec_hidden=(32, 64),
        poly_order=3,
        include_bias=True,
    )

    import torch
    from trainer import Trainer_SINDyAE

    trainer = Trainer_SINDyAE(
        model=model,
        dt=dt,
        diff_method="finite-diff",   # or your tvdiff
        diff_kwargs={},
        batch_size=4,        # SI uses batch_size=8000 for Lorenz
        device=device,
        lambda_recon=1.0,
        lambda_dx=1e-4,         # λ1 from Table S1
        lambda_dz=0.0,          # λ2 from Table S1
        threshold=0.1,
        threshold_freq=500,
        lr=1e-3,
    )

    trainer.fit(
        X_train,
        n_epochs=500,
        log_every=50,
    )
