import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.diff_methods import compute_derivatives
from src.model import SINDy_Autoencoder, TimeSeriesDataset


class Trainer_SINDyAE:

    def __init__(self,
                model: SINDy_Autoencoder,
                dt: float,
                diff_method: str = "finite",
                diff_kwargs: dict | None = None,
                batch_size: int = 64,
                shuffle: bool = True,
                device: str = "cuda",

                # loss weights
                lambda_recon: float = 1.0,
                lambda_dx: float = 1.0,
                lambda_dz: float = 0.1,
                lambda_reg: float = 1e-4,

                # STLSQ settings
                threshold: float = 0.1,
                threshold_freq: int = 500,

                # optimization
                lr: float = 1e-3,):
        
        self.model = model
        self.dt = dt
        self.diff_method = diff_method
        self.diff_kwargs = diff_kwargs or {}

        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # -----------------------------
        # DEVICE HANDLING + DEBUG PRINT
        # -----------------------------
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] device='cuda' ma torch.cuda.is_available() == False → uso CPU.")
            self.device = "cpu"
        else:
            self.device = device

        print(f"[INIT] Using device: {self.device}")
        
        self.model.to(self.device)
        print(f"[INIT] Model first param device: {next(self.model.parameters()).device}")

        self.lambda_recon = lambda_recon
        self.lambda_dx = lambda_dx
        self.lambda_dz = lambda_dz
        self.lambda_reg = lambda_reg

        self.threshold = threshold
        self.threshold_freq = threshold_freq

        self.lr = lr
        
        self.history_loss = []

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # ---------------------------------------------------------------------
    # Build dataloader from X
    # ---------------------------------------------------------------------
    def _build_dataloader(self, X):

        Xdot_np = compute_derivatives(X,
                                    self.dt,
                                    diff_method=self.diff_method,
                                    **self.diff_kwargs,
                    )
        # 2. Convert to tensors
        if isinstance(X, torch.Tensor):
            X_torch = X.detach().clone().float()
        else:
            X_torch = torch.from_numpy(np.asarray(X, dtype=float)).float()

        Xdot_torch = torch.from_numpy(Xdot_np).float()

        # 3. Dataset + Dataloader
        dataset = TimeSeriesDataset(X_torch, Xdot_torch) # dataset encapsulates time series of X
                                                         # dataset[index] gives a single trajectory 
                                                         # Each item of dataset is a full time serie

        loader = DataLoader(dataset,                     # loader takes batches of time series, each with 
                            batch_size=self.batch_size,  # different initial conditions
                            shuffle=self.shuffle)

        return loader
    
    # ---------------------------------------------------------------------
    # Define the training of one epoch
    # ---------------------------------------------------------------------
    def _train_one_epoch(self, train_loader):
        self.model.train()

        running = {
            "loss": 0.0,
            "L_recon": 0.0,
            "L_dx": 0.0,
            "L_dz": 0.0,
        }
        n_samples = 0

        for x_seq, xdot_seq in train_loader:
            x_seq = x_seq.to(self.device)
            xdot_seq = xdot_seq.to(self.device)
                
            B = x_seq.shape[0] # batch_size of x_seq
            n_samples += B

            self.optimizer.zero_grad()

            _, _, losses = self.model(x_seq, xdot_seq,
                                    lambda_recon=self.lambda_recon,
                                    lambda_dz=self.lambda_dz,
                                    lambda_dx=self.lambda_dx
                            )
            loss = losses["loss"]
            loss.backward()
            self.optimizer.step()

            for k in running:
                running[k] += losses[k].item() * B

        stats = {k: running[k] / n_samples for k in running}
        return stats
    
    def _apply_threshold(self):

        with torch.no_grad():
            Xi = self.model.sindy.Xi
            mask = self.model.sindy.Xi_mask

            small = torch.abs(Xi) < self.threshold
            Xi[small] = 0.0
            mask[small] = 0.0

    # ---------------------------------------------------------------------
    # Full training loop
    # ---------------------------------------------------------------------
    def fit(
            self,
            X,
            n_epochs: int = 10000,
            refine_epochs: int = 0,
            log_every: int = 100,
        ):
        train_loader = self._build_dataloader(X)

        for epoch in range(1, n_epochs + 1):

            stats = self._train_one_epoch(train_loader=train_loader)
            self.history_loss.append(stats)

            if epoch % self.threshold_freq == 0:
                self._apply_threshold()
            
            if epoch % log_every == 0 or epoch == 1 or epoch == n_epochs:
                print(
                    f"[TRAIN] Epoch {epoch:5d}/{n_epochs} | "
                    f"Loss: {stats['loss']:.4e} | "
                    f"Lrecon: {stats['L_recon']:.4e} | "
                    f"Ldx: {stats['L_dx']:.4e} | "
                    f"Ldz: {stats['L_dz']:.4e} | "
                )
        # # ----------------------
        # # Phase 2: refinement (optional)
        # # ----------------------
        # if refine_epochs > 0:
        #     print("\n[REFINE] Starting refinement stage (no L1, no STLSQ)\n")

        #     for epoch in range(1, refine_epochs + 1):

        #         stats = self._train_one_epoch(
        #             train_loader,
        #             lambda_reg_current=0.0,  # no L1
        #         )

        #         if epoch % log_every == 0 or epoch == refine_epochs:
        #             print(
        #                 f"[REFINE] Epoch {epoch:5d}/{refine_epochs} | "
        #                 f"Loss: {stats['loss']:.4e} | "
        #                 f"Lrecon: {stats['L_recon']:.4e} | "
        #                 f"Ldx: {stats['L_dx']:.4e} | "
        #                 f"Ldz: {stats['L_dz']:.4e} | "
        #                 f"Lreg: {stats['L_reg']:.4e}"
        #             )
                