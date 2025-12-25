import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from data_statcast import StatcastDataset, set_seed
from models import StateNet, WindNet, LearnableScalars, physics_residual, wind_regularizers

def train(csv_path: str, out_ckpt: str = "pinn_ckpt.pt"):
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    ds = StatcastDataset(csv_path, t_max=cfg.t_max, n_obs=cfg.n_obs, n_col=cfg.n_col, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    state_net = StateNet(width=cfg.state_width, depth=cfg.state_depth).to(device)
    wind_net  = WindNet(width=cfg.wind_width, depth=cfg.wind_depth).to(device)
    scalars   = LearnableScalars().to(device)

    g_vec = torch.tensor(cfg.g, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(list(state_net.parameters()) + list(wind_net.parameters()) + list(scalars.parameters()),
                           lr=cfg.lr)

    for epoch in range(cfg.epochs):
        state_net.train(); wind_net.train(); scalars.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        running = 0.0

        for batch in pbar:
            y0 = batch["y0"].to(device)              # (B,9)
            t_obs = batch["t_obs"].to(device)        # (B,n_obs,1) but stored per item; DataLoader stacks -> (B,n_obs,1)
            r_obs = batch["r_obs"].to(device)        # (B,n_obs,3)
            t_col = batch["t_col"].to(device)        # (B,n_col,1)

            # Flatten B*n into one big batch for vectorized autograd
            B = y0.shape[0]
            t_obs_f = t_obs.reshape(-1, 1).requires_grad_(True)
            t_col_f = t_col.reshape(-1, 1).requires_grad_(True)

            # Expand y0 across times
            y0_obs = y0[:, None, :].expand(B, cfg.n_obs, 9).reshape(-1, 9)
            y0_col = y0[:, None, :].expand(B, cfg.n_col, 9).reshape(-1, 9)

            # Predictions
            yhat_obs = state_net(t_obs_f, y0_obs)   # (B*n_obs,9)
            yhat_col = state_net(t_col_f, y0_col)   # (B*n_col,9)

            # Data loss on r(t)
            rhat_obs = yhat_obs[:, 0:3].reshape(B, cfg.n_obs, 3)
            L_data = (rhat_obs - r_obs).pow(2).mean()

            # Physics loss at collocation points
            res_r, res_v, res_om = physics_residual(t_col_f, yhat_col, y0_col, wind_net, scalars, g_vec)
            L_phys = res_r.pow(2).mean() + res_v.pow(2).mean() + res_om.pow(2).mean()

            # Initial condition loss at t=0
            t0 = torch.zeros((B,1), device=device, requires_grad=True)
            yhat0 = state_net(t0, y0)
            L_ic = (yhat0 - y0).pow(2).mean()

            # Wind regularization: sample points from predicted r(t_col)
            r_col = yhat_col[:, 0:3].detach().requires_grad_(True)
            t_col_reg = t_col_f.detach().requires_grad_(True)
            L_w_mean, L_w_smooth = wind_regularizers(wind_net, r_col, t_col_reg)

            loss = (cfg.w_data * L_data +
                    cfg.w_phys * L_phys +
                    cfg.w_ic * L_ic +
                    cfg.w_wind_mean * L_w_mean +
                    cfg.w_wind_smooth * L_w_smooth)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.3e}",
                "kd": f"{scalars.kd().item():.3e}",
                "km": f"{scalars.km().item():.3e}",
                "cω": f"{scalars.c_omega().item():.3e}",
            })

        avg = running / max(1, len(dl))
        print(f"Epoch {epoch+1:03d} avg_loss={avg:.3e}  kd={scalars.kd().item():.4e}  km={scalars.km().item():.4e}  cω={scalars.c_omega().item():.4e}")

    torch.save({
        "cfg": cfg.__dict__,
        "state_net": state_net.state_dict(),
        "wind_net": wind_net.state_dict(),
        "scalars": scalars.state_dict()
    }, out_ckpt)

    print(f"Saved checkpoint -> {out_ckpt}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Statcast CSV")
    ap.add_argument("--out", default="pinn_ckpt.pt")
    args = ap.parse_args()
    train(args.csv, args.out)
