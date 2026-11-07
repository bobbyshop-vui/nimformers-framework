"""Mamba-style selective SSM decode: causal depthwise conv with state, selective
scan step, and SiLU-gated output (Mamba-1 block layout, flat config keys).

Per layer and token:
    xs, z  = in_proj(rmsnorm(x)).chunk(2)
    window = cat(conv_state, xs)            conv_state <- window[..., 1:]
    u      = silu(conv1d(window)[..., -1])
    dt, B, C = x_proj(u).split(dt_rank, d_state, d_state)
    dt     = softplus(dt_proj(dt))
    h      = h * exp(dt * A) + dt * B * u   A = -exp(A_log)
    y      = sum(h * C, -1) + D * u
    x      = x + out_proj(y * silu(z))
"""

from .. import nn
from ..tensor import TensorSpec, cat

CONFIG = {"vocab_size": 8, "n_positions": 6, "d_model": 4, "d_inner": 8, "d_state": 4, "d_conv": 4,
          "dt_rank": 2, "n_layer": 2, "rms_norm_eps": 1e-5}
INPUTS = (TensorSpec((1, 1), "token"), TensorSpec((1, 1), "position"))


class Mixer(nn.Module):
    def __init__(self, c):
        super().__init__()
        e, d, n, k, r = c["d_model"], c["d_inner"], c["d_state"], c["d_conv"], c["dt_rank"]
        self.d, self.n, self.k, self.r = d, n, k, r
        self.in_proj = nn.Linear(e, 2 * d, bias=False)
        self.conv1d = nn.Conv1d(d, d, k, groups=d)
        self.x_proj = nn.Linear(d, r + 2 * n, bias=False)
        self.dt_proj = nn.Linear(r, d)
        self.A_log = nn.Parameter((d, n))
        self.D = nn.Parameter((d,))
        self.out_proj = nn.Linear(d, e, bias=False)
        self.register_state("conv_state", (1, d, k - 1))
        self.register_state("ssm_state", (d, n))

    def forward(self, x):
        xs, z = self.in_proj(x.reshape(-1)).chunk(2, -1)
        window = cat([self.conv_state.read(), xs.reshape(1, self.d, 1)], -1)
        self.conv_state.update(window[:, :, 1:])
        u = self.conv1d(window)[:, :, self.k - 1].reshape(self.d).silu()
        dt, B, C = self.x_proj(u).split([self.r, self.n, self.n], -1)
        dt = self.dt_proj(dt).softplus().unsqueeze(-1)
        A = -self.A_log.exp()
        h = self.ssm_state.read() * (dt * A).exp() + dt * B.unsqueeze(0) * u.unsqueeze(-1)
        self.ssm_state.update(h)
        y = (h * C.unsqueeze(0)).sum(-1) + self.D * u
        return self.out_proj(y * z.silu()).reshape(x.shape)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.RMSNorm(c["d_model"], c["rms_norm_eps"])
        self.mixer = Mixer(c)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Mamba(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.backbone.layers = nn.ModuleList(Block(config) for _ in range(config["n_layer"]))
        self.backbone.norm_f = nn.RMSNorm(config["d_model"], config["rms_norm_eps"])

    def forward(self, token, position):
        x = self.backbone.embedding(token)
        for layer in self.backbone.layers:
            x = layer(x)
        return self.backbone.embedding.project(self.backbone.norm_f(x))
