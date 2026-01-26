import numpy as np
import pandas as pd
import torch

FT_TO_M = 0.3048
RPM_TO_RAD_S = 2.0 * np.pi / 60.0

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def spin_to_omega(spin_rate_rpm, spin_axis_deg):
    """
    Minimal mapping from (rpm, axis angle) -> omega vector.
    This is a modeling choice; you can refine once your coordinate frame is fixed.

    We'll assume spin_axis_deg defines direction in x-z plane around y,
    purely for a starting point (NOT "the" official convention).

    omega_hat = (cos(theta), 0, sin(theta))
    """
    theta = np.deg2rad(spin_axis_deg)
    omega_hat = np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=1)
    omega_mag = (spin_rate_rpm * RPM_TO_RAD_S).reshape(-1, 1)
    return omega_mag * omega_hat

def build_pseudo_positions(r0, v0, a, t_grid):
    """
    r(t) = r0 + v0 t + 0.5 a t^2
    r0: (N,3), v0: (N,3), a: (N,3), t_grid: (M,) => returns (N,M,3)
    """
    t = t_grid.reshape(1, -1, 1)
    return r0[:, None, :] + v0[:, None, :] * t + 0.5 * a[:, None, :] * (t ** 2)

class StatcastDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, t_max: float, n_obs: int, n_col: int, seed: int = 42):
        df = pd.read_csv(csv_path)

        # --- rename / adapt here if your CSV uses different column names ---
        # --- Positions (ft -> m)
        r0_ft = df[["release_pos_x", "release_pos_y", "release_pos_z"]].to_numpy(dtype=np.float32)
        r0 = r0_ft * FT_TO_M

        # --- Velocities (ft/s -> m/s)
        v0_fts = df[["vx0", "vy0", "vz0"]].to_numpy(dtype=np.float32)
        v0 = v0_fts * FT_TO_M

        # --- Accelerations (ft/s^2 -> m/s^2)
        a0_fts2 = df[["ax", "ay", "az"]].to_numpy(dtype=np.float32)
        a0 = a0_fts2 * FT_TO_M

        # --- Spin
        spin_rate = df["release_spin_rate"].to_numpy(dtype=np.float32)
        spin_axis = df["spin_axis"].to_numpy(dtype=np.float32)
        omega0 = spin_to_omega(spin_rate, spin_axis).astype(np.float32)

        # Time grids
        rng = np.random.default_rng(seed)
        self.t_obs = np.linspace(0.0, t_max, n_obs, dtype=np.float32)
        # Random collocation times will be sampled per __getitem__ for PINN residuals
        self.t_max = float(t_max)
        self.n_col = int(n_col)
        self.rng = rng

        # Pseudo-observations r(t_obs)
        r_obs = build_pseudo_positions(r0, v0, a0, self.t_obs)  # (N,n_obs,3)

        self.y0 = np.concatenate([r0, v0, omega0], axis=1)      # (N,9)
        self.r_obs = r_obs                                      # (N,n_obs,3)

    def __len__(self):
        return self.y0.shape[0]

    def __getitem__(self, idx):
        y0 = self.y0[idx]              # (9,)
        r_obs = self.r_obs[idx]        # (n_obs,3)

        # collocation times (random in [0,t_max])
        t_col = self.rng.uniform(0.0, self.t_max, size=(self.n_col,)).astype(np.float32)

        return {
            "y0": torch.tensor(y0),
            "t_obs": torch.tensor(self.t_obs).unsqueeze(1),   # (n_obs,1)
            "r_obs": torch.tensor(r_obs),                     # (n_obs,3)
            "t_col": torch.tensor(t_col).unsqueeze(1),        # (n_col,1)
        }
