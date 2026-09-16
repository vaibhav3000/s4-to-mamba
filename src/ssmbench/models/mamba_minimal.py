"""Educational Mamba (Mamba-1): selective state space sequence model.

Scope and honesty notes
-----------------------
Written from first principles following the Mamba paper (Gu & Dao 2023,
arXiv:2312.00752). This is an *educational* implementation:

- The selective scan runs as an explicit python loop over time — it exposes the
  recurrence but has high constant overhead. That is the point: it makes
  "O(T) operations but poor wall-clock" tangible and motivates Mamba-2's
  chunked matrix form (see mamba2_minimal.py) and the official fused kernels,
  which are NOT used here.
- Discretization follows the official implementation: the state transition uses
  the exact exponential exp(delta * A) (ZOH on the diagonal), while the input
  uses the delta * B approximation — later formalized as "exponential-Euler"
  by the Mamba-3 paper (its Table 1).

Shapes: B batch, T time, D model dim (d_model), E expansion (d_inner = E*D),
N state size (d_state), R rank of the dt projection (dt_rank).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Pre-norm residual block: x + Mambaixer(LN(x)).

    Mixer: in_proj -> causal conv1d -> silu -> selective SSM -> silu-gate -> out_proj.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=0,  # causal padding applied manually
        )

        # Projects the convolved token to (dt_rank, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # Projects dt_rank -> d_inner with a log-uniform time-scale bias init
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is a real negative diagonal, (d_inner, N); stored in log space.
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        # Skip connection (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init_dt()

    def _init_dt(self) -> None:
        """Init dt_proj so that softplus(output + bias) is log-uniform in [1e-3, 1e-1].

        The bias is set to inv_softplus of the desired dt values, matching the
        official implementation; the weight gets a rank-scaled uniform init.
        """
        dt_min, dt_max = 1e-3, 1e-1
        with torch.no_grad():
            std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -std, std)
            # Per-channel log-uniform time-scale bias, as in the official code.
            u_d = torch.linspace(0.0, 1.0, self.d_inner)
            dt_d = torch.exp(u_d * math.log(dt_max) + (1 - u_d) * math.log(dt_min))
            self.dt_proj.bias.copy_(dt_d + torch.log(-torch.expm1(-dt_d)))

    def _ssm_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input-dependent (delta, B, C) from the inner branch x: (B, T, d_inner)."""
        proj = self.x_proj(x)  # (B, T, dt_rank + 2N)
        delta, B, C = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, T, d_inner), > 0
        return delta, B, C

    def selective_scan(
        self, x: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Sequential selective scan (educational form).

            h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
            y_t = sum_n C_t[n] * h_t[n]  +  D * x_t

        x: (B, T, d_inner), delta: (B, T, d_inner), B/C: (B, T, N) -> y: (B, T, d_inner)
        """
        A = -torch.exp(self.A_log.float())  # (d_inner, N), negative
        xT = x.transpose(0, 1)
        deltaT = delta.transpose(0, 1)
        BT = B.transpose(0, 1)
        CT = C.transpose(0, 1)

        h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=A.dtype)
        ys = []
        for t in range(x.shape[1]):
            alpha = torch.exp(deltaT[t].unsqueeze(-1) * A)  # (B, d_inner, N) in (0, 1)
            h = alpha * h + deltaT[t].unsqueeze(-1) * BT[t].unsqueeze(1) * xT[t].unsqueeze(-1)
            ys.append(torch.einsum("bdn,bn->bd", h, CT[t].to(A.dtype)))
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        return y + x * self.D

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        T = x.shape[1]
        xz = self.in_proj(self.norm(x))  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        # Causal short convolution: left-pad by d_conv-1, then trim the tail.
        x_conv = x_in.transpose(1, 2)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)[..., :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        delta, B, C = self._ssm_params(x_conv)
        y = self.selective_scan(x_conv, delta, B, C)
        return x + self.out_proj(y * F.silu(z))

    # -------------------- incremental decode (O(1) state per block) --------------------

    def init_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Recurrent decode state: SSM state h and the short-conv input history."""
        return {
            "h": torch.zeros(batch_size, self.d_inner, self.d_state, device=device),
            "conv": torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device),
        }

    def step(self, x_t: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """One decode step. x_t: (B, 1, D) -> out (B, 1, D). O(1) per token.

        Mirrors forward() exactly: same weights, same ops, state carried across
        calls instead of recomputed from the prefix.
        """
        xz = self.in_proj(self.norm(x_t))  # (B, 1, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)      # (B, 1, d_inner)
        conv = torch.cat([state["conv"], x_in.squeeze(1).unsqueeze(-1)], dim=-1)
        conv_out = F.silu(self.conv1d(conv)[..., -1])  # (B, d_inner) last output
        state["conv"] = conv[:, :, 1:]  # drop oldest input, keep last d_conv-1

        proj = self.x_proj(conv_out)
        delta = F.softplus(self.dt_proj(proj[:, : self.dt_rank]))
        _, B_t, C_t = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        A = -torch.exp(self.A_log.float())
        alpha = torch.exp(delta.unsqueeze(-1) * A)  # (B, d_inner, N)
        h = alpha * state["h"] + delta.unsqueeze(-1) * B_t.unsqueeze(1) * conv_out.unsqueeze(-1)
        state["h"] = h
        y = torch.einsum("bdn,bn->bd", h, C_t) + conv_out * self.D
        out = self.out_proj(y * F.silu(z.squeeze(1)))
        return out.unsqueeze(1) + x_t


class MambaBackbone(nn.Module):
    """Stack of Mamba blocks (each block owns its pre-norm and residual)."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ]
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = emb
        for block in self.blocks:
            x = block(x)
        return x
