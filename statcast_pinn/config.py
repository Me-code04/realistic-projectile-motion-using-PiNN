from dataclasses import dataclass

@dataclass
class TrainConfig:
    device: str = "cuda"  # "cpu" if needed
    seed: int = 42

    # Units: Statcast commonly uses feet and ft/s. We'll convert to SI.
    # Times for pseudo-trajectory reconstruction:
    # We'll fit r(t) ~ r0 + v0*t + 0.5*a*t^2 using Statcast accel components.
    n_obs: int = 25          # "observation" points per pitch for data loss
    n_col: int = 64          # collocation points per pitch for physics loss
    t_max: float = 0.55      # seconds; typical pitch flight is ~0.35-0.55 depending on definition

    # Loss weights
    w_data: float = 1.0
    w_phys: float = 1.0
    w_ic: float = 50.0
    w_wind_mean: float = 1e-3
    w_wind_smooth: float = 1e-4

    # Optimization
    lr: float = 2e-4
    epochs: int = 60
    batch_size: int = 256

    # Model sizes
    state_width: int = 256
    state_depth: int = 5
    wind_width: int = 128
    wind_depth: int = 4

    # Gravity (SI): choose your coordinate convention carefully.
    # We'll assume z is "up", so g = (0,0,-9.81)
    g: tuple = (0.0, 0.0, -9.81)
