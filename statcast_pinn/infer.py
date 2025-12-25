import numpy as np
import pandas as pd
import torch

from models import StateNet, WindNet, LearnableScalars
from config import TrainConfig

def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = TrainConfig()
    # Load saved cfg if present
    for k, v in ckpt.get("cfg", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    state_net = StateNet(width=cfg.state_width, depth=cfg.state_depth).to(device)
    wind_net  = WindNet(width=cfg.wind_width, depth=cfg.wind_depth).to(device)
    scalars   = LearnableScalars().to(device)

    state_net.load_state_dict(ckpt["state_net"])
    wind_net.load_state_dict(ckpt["wind_net"])
    scalars.load_state_dict(ckpt["scalars"])

    state_net.eval(); wind_net.eval(); scalars.eval()
    return cfg, state_net, wind_net, scalars

@torch.no_grad()
def predict_trajectory(ckpt_path: str, y0: np.ndarray, t_max: float = 0.55, n: int = 200, out_csv: str = "traj.csv"):
    """
    y0: shape (9,) = [x,y,z, vx,vy,vz, wx,wy,wz] where last 3 are omega components (rad/s)
        NOTE: omega components, not wind.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, state_net, wind_net, scalars = load_model(ckpt_path, device=device)

    t = torch.linspace(0.0, float(t_max), steps=n, device=device).unsqueeze(1)
    y0_t = torch.tensor(y0, dtype=torch.float32, device=device).unsqueeze(0).expand(n, 9)

    yhat = state_net(t, y0_t)  # (n,9)
    r = yhat[:, 0:3].cpu().numpy()
    v = yhat[:, 3:6].cpu().numpy()
    om = yhat[:, 6:9].cpu().numpy()

    # wind along the trajectory
    w = wind_net(torch.tensor(r, dtype=torch.float32, device=device), t).cpu().numpy()

    df = pd.DataFrame({
        "t": t.squeeze(1).cpu().numpy(),
        "x": r[:,0], "y": r[:,1], "z": r[:,2],
        "vx": v[:,0], "vy": v[:,1], "vz": v[:,2],
        "ox": om[:,0], "oy": om[:,1], "oz": om[:,2],
        "wx": w[:,0], "wy": w[:,1], "wz": w[:,2],
    })
    df.to_csv(out_csv, index=False)

    print(f"Saved {out_csv}")
    print(f"Learned kd={scalars.kd().item():.6e}, km={scalars.km().item():.6e}, c_omega={scalars.c_omega().item():.6e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tmax", type=float, default=0.55)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="traj.csv")
    # y0 entries
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--z", type=float, required=True)
    ap.add_argument("--vx", type=float, required=True)
    ap.add_argument("--vy", type=float, required=True)
    ap.add_argument("--vz", type=float, required=True)
    ap.add_argument("--ox", type=float, required=True)
    ap.add_argument("--oy", type=float, required=True)
    ap.add_argument("--oz", type=float, required=True)
    args = ap.parse_args()

    y0 = np.array([args.x,args.y,args.z,args.vx,args.vy,args.vz,args.ox,args.oy,args.oz], dtype=np.float32)
    predict_trajectory(args.ckpt, y0, t_max=args.tmax, n=args.n, out_csv=args.out)
