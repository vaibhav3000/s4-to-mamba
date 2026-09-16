"""Educational Mamba-3: complex-valued selective SSM with exponential-trapezoidal
discretization.

Scope and honesty notes
-----------------------
Written from first principles following "Mamba-3: Improved Sequence Modeling
using State Space Principles" (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu;
arXiv:2603.15569). Mamba-3 changes three things relative to Mamba-2:

1. **Exponential-trapezoidal discretization.** The state-input integral is
   approximated by a data-dependent convex combination of both interval
   endpoints (weight lambda_t in [0,1]) instead of the right-endpoint-only
   exponential-Euler rule used by Mamba-1/2:

       h_t = alpha_t h_{t-1} + beta_t B_{t-1} x_{t-1} + gamma_t B_t x_t
       alpha_t = exp(delta_t A_t),  beta_t = (1 - lambda_t) delta_t alpha_t,
       gamma_t = lambda_t delta_t

   lambda_t = 1 recovers Mamba-1/2 (exponential-Euler); lambda_t = 1/2 recovers
   the classical trapezoidal rule. The rule is second-order accurate in delta.

2. **Complex-valued states.** The diagonal A becomes complex
   (A = -exp(a) + i*theta): |alpha_t| < 1 still guarantees stability, but the
   phase allows *rotational* dynamics, which real non-negative transitions
   cannot express. This restores state-tracking capability (e.g. parity) that
   Mamba-1/2 provably lack (their transitions only decay/forget).

3. **MIMO formulation.** Multiple input/output channels per state slot improve
   quality without increasing decode latency. This educational implementation
   keeps the SISO-per-head structure of Mamba-2's SSD and does not implement
   the MIMO dual form; the recurrence above is implemented exactly.

Only the sequential (recurrent) form is implemented — the paper's parallel dual
is a 1-semiseparable matrix composed with a 2-band matrix (Section 3.1.3 of the
paper), i.e. a generalization of Mamba-2's SSD. See docs/evolution.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba3Block(nn.Module):
    """Pre-norm residual block with the Mamba-3 complex selective recurrence.

    Per head h (headdim P, state size N), B/C shared per group:
        h_t = alpha_t h_{t-1} + beta_t * s_{t-1} + gamma_t * s_t
        s_t = B_{g(h),t} (x) delta-scalared x_{h,t}   (state input, see below)
        y_{h,t} = C_{g(h),t}^T Re(h_{h,t})
    alpha_t = exp(delta_t A_h) complex, |alpha_t| < 1 (stability);
    beta_t = (1 - lambda_t) delta_t alpha_t;  gamma_t = lambda_t delta_t.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        headdim: int = 64,
        expand: int = 2,
        n_groups: int = 1,
        d_conv: int = 4,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.expand = expand
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            raise ValueError(f"d_inner={self.d_inner} must be divisible by headdim={headdim}")
        self.n_heads = self.d_inner // headdim
        if self.n_heads % n_groups != 0:
            raise ValueError(f"n_heads={self.n_heads} must be divisible by n_groups={n_groups}")
        self.n_groups = n_groups
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            bias=True, padding=0,
        )
        # x -> (dt_rank, B(n_groups*N), C(n_groups*N), lambda(n_groups))
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * n_groups * d_state + n_groups, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.n_heads, bias=True)
        # lambda gating: sigmoid(bias + proj) in (0,1); bias 0 -> 0.5 (balanced trap.)
        self.lambda_proj = nn.Linear(self.n_groups, self.n_heads, bias=True)
        nn.init.zeros_(self.lambda_proj.weight)
        nn.init.zeros_(self.lambda_proj.bias)

        # Complex diagonal A = -exp(a_real) + i * theta, per (head, state).
        self.a_real_log = nn.Parameter(torch.log(torch.full((self.n_heads, d_state), 1.0)))
        n = torch.arange(d_state, dtype=torch.float32)
        theta_init = math.pi * n  # S4D-Lin-style rotation frequencies
        self.theta = nn.Parameter(theta_init.repeat(self.n_heads, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init_dt()

    def _init_dt(self) -> None:
        dt_min, dt_max = 1e-3, 1e-1
        with torch.no_grad():
            u = torch.linspace(0.0, 1.0, self.n_heads)
            dt = torch.exp(u * math.log(dt_max) + (1 - u) * math.log(dt_min))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -std, std)

    def selective_scan(
        self,
        x_h: torch.Tensor,      # (B, T, H, P)
        delta: torch.Tensor,    # (B, T, H)
        B_h: torch.Tensor,      # (B, T, H, N)  (group-expanded, real)
        C_h: torch.Tensor,      # (B, T, H, N)
        lam: torch.Tensor,      # (B, T, H) in (0, 1)
    ) -> torch.Tensor:
        """Sequential Mamba-3 recurrence. Returns y: (B, T, H, P) real."""
        Bsz, T, H, P = x_h.shape
        N = self.d_state
        A = -torch.exp(self.a_real_log.float()) + 1j * self.theta.float()  # (H, N) complex

        xT = x_h.transpose(0, 1).float()
        dT = delta.transpose(0, 1).float()
        bT = B_h.transpose(0, 1).float()
        cT = C_h.transpose(0, 1).float()
        lamT = lam.transpose(0, 1).float()

        h = torch.zeros(Bsz, H, N, P, dtype=torch.complex64, device=x_h.device)
        prev_inject = torch.zeros(Bsz, H, N, P, dtype=torch.complex64, device=x_h.device)
        ys = []
        for t in range(T):
            alpha = torch.exp(dT[t].unsqueeze(-1) * A.unsqueeze(0))  # (B, H, N) complex
            alpha_p = alpha.unsqueeze(-1).expand(-1, -1, -1, P)  # (B, H, N, P)
            gamma = lamT[t] * dT[t]  # (B, H)
            inject_t = bT[t].unsqueeze(-1) * xT[t].unsqueeze(2)  # (B, H, N, P) real
            beta = (1.0 - lamT[t]).unsqueeze(-1) * dT[t].unsqueeze(-1) * alpha  # (B, H, N) complex
            h = (
                alpha_p * h
                + beta.unsqueeze(-1) * prev_inject
                + gamma.unsqueeze(-1).unsqueeze(-1) * inject_t.to(torch.complex64)
            )
            ys.append(torch.einsum("bhnp,bhn->bhp", h.real, cT[t]))
            prev_inject = inject_t.to(torch.complex64)
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        T = x.shape[1]
        xz = self.in_proj(self.norm(x))
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = x_in.transpose(1, 2)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)[..., :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        proj = self.x_proj(x_conv)
        sizes = [self.dt_rank, self.n_groups * self.d_state, self.n_groups * self.d_state, self.n_groups]
        dt, B, C, lam_raw = torch.split(proj, sizes, dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # (B, T, H)
        lam = torch.sigmoid(self.lambda_proj(lam_raw))  # (B, T, H)

        rep = self.n_heads // self.n_groups
        B_h = B.repeat_interleave(rep, dim=-1).view(*B.shape[:-1], self.n_heads, self.d_state)
        C_h = C.repeat_interleave(rep, dim=-1).view(*C.shape[:-1], self.n_heads, self.d_state)
        # lambda_proj outputs one weight per head directly (no group expansion).
        lam_h = lam
        x_h = x_conv.view(*x_conv.shape[:-1], self.n_heads, self.headdim)

        y = self.selective_scan(x_h, delta, B_h, C_h, lam_h)
        y = y.reshape(*y.shape[:2], self.d_inner)
        y = y + x_conv * self.D
        return x + self.out_proj(y * F.silu(z))


class Mamba3Backbone(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 32,
        headdim: int = 64,
        expand: int = 2,
        n_groups: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [
                Mamba3Block(
                    d_model, d_state=d_state, headdim=headdim, expand=expand,
                    n_groups=n_groups,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = emb
        for block in self.blocks:
            x = block(x)
        return x
