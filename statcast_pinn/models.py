import torch
import torch.nn as nn
import torch.autograd as autograd

def mlp(in_dim, out_dim, width=128, depth=4, act=nn.Tanh):
    layers = [nn.Linear(in_dim, width), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), act()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)

def grad(outputs, inputs):
    return autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]

def time_derivative(y, t):
    """
    Compute dy/dt for vector-valued y(t) where t has shape (N,1).
    Returns dy_dt with shape (N, D) where D = y.shape[1].
    """
    grads = []
    for k in range(y.shape[1]):
        # sum() makes it a scalar so autograd.grad works cleanly
        gk = autograd.grad(
            y[:, k].sum(), t,
            create_graph=True, retain_graph=True
        )[0]  # (N,1)
        grads.append(gk)
    return torch.cat(grads, dim=1)  # (N,D)

class LearnableScalars(nn.Module):
    """
    kd > 0, km > 0, c_omega > 0 (spin decay).
    If you want km to allow sign, remove softplus on km().
    """
    def __init__(self, kd_init=0.02, km_init=0.002, comega_init=0.01):
        super().__init__()
        self.kd_raw = nn.Parameter(torch.tensor(float(kd_init)))
        self.km_raw = nn.Parameter(torch.tensor(float(km_init)))
        self.co_raw = nn.Parameter(torch.tensor(float(comega_init)))
        self.softplus = nn.Softplus()

    def kd(self): return self.softplus(self.kd_raw)
    def km(self): return self.softplus(self.km_raw)
    def c_omega(self): return self.softplus(self.co_raw)

class WindNet(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        self.net = mlp(4, 3, width=width, depth=depth, act=nn.Tanh)

    def forward(self, r, t):
        # r: (N,3), t: (N,1)
        inp = torch.cat([r, t], dim=1)
        return self.net(inp)

class StateNet(nn.Module):
    """
    Segment-style PINN: takes (t, y0) -> y(t).
    This is important: it generalizes across many pitches.
    """
    def __init__(self, width=256, depth=5):
        super().__init__()
        self.net = mlp(1 + 9, 9, width=width, depth=depth, act=nn.Tanh)

    def forward(self, t, y0):
        # t: (N,1), y0: (N,9) broadcastable
        if y0.shape[0] != t.shape[0]:
            y0 = y0.expand(t.shape[0], -1)
        inp = torch.cat([t, y0], dim=1)
        return self.net(inp)

def physics_residual(t, yhat, y0, wind_net, scalars, g_vec):
    """
    Residuals for:
      r' = v
      v' = g - kd||u||u + km(omega x u)
      omega' = -c_omega omega
    """
    r = yhat[:, 0:3]
    v = yhat[:, 3:6]
    om = yhat[:, 6:9]

    dy_dt = time_derivative(yhat, t)  # (N,9)
    dr_dt = dy_dt[:, 0:3]
    dv_dt = dy_dt[:, 3:6]
    dom_dt = dy_dt[:, 6:9]

    w = wind_net(r, t)
    u = v - w

    kd = scalars.kd()
    km = scalars.km()
    c_om = scalars.c_omega()

    drag = -kd * torch.norm(u, dim=1, keepdim=True) * u
    magnus = km * torch.cross(om, u, dim=1)
    f = g_vec.view(1, 3) + drag + magnus

    om_rhs = -c_om * om

    res_r = dr_dt - v
    res_v = dv_dt - f
    res_om = dom_dt - om_rhs
    return res_r, res_v, res_om

def wind_regularizers(wind_net, r, t):
    """
    Regularize wind to prevent it from "explaining everything":
      - mean wind small
      - smoothness in space/time
    """
    w = wind_net(r, t)  # (N,3)

    # Mean penalty
    L_mean = (w.mean(dim=0).pow(2).sum())

    # Smoothness: ||∂w/∂x||^2 + ||∂w/∂t||^2
    # Need grad of each component wrt r and t
    grads_r = []
    grads_t = []
    for k in range(3):
        wk = w[:, k:k+1]
        gw_r = grad(wk, r)  # (N,3)
        gw_t = grad(wk, t)  # (N,1)
        grads_r.append(gw_r)
        grads_t.append(gw_t)
    Gw_r = torch.stack(grads_r, dim=1)  # (N,3comp,3xyz)
    Gw_t = torch.stack(grads_t, dim=1)  # (N,3comp,1)
    L_smooth = (Gw_r.pow(2).mean() + Gw_t.pow(2).mean())
    return L_mean, L_smooth
