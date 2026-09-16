"""Educational Mamba-2: scalar-identity SSM with the chunked SSD algorithm.

Scope and honesty notes
-----------------------
Written from first principles following "Transformers are SSMs" (Dao & Gu 2024,
arXiv:2405.21060). Mamba-2 restricts the state transition to a scalar times
identity per head (A_h < 0), which turns the selective SSM into the **State
Space Duality (SSD)** form: within a chunk of tokens the output is a masked,
decay-weighted attention-like matmul, and only a small (N, P) state crosses
chunk boundaries. The layer becomes a sequence of tensor-core matmuls instead
of a long scan — this is exactly *why* Mamba-2 exists.

This module implements BOTH forms:

- ``scan_mode="sequential"``: the plain recurrence h_t = alpha_t h_{t-1} +
  (delta_t B_t) (x) x_t, one python-loop step at a time. Reference form.
- ``scan_mode="chunked"``: the SSD chunked algorithm. A python loop over
  T/chunk_size chunks, each doing small matmuls; the official repository fuses
  this into Triton kernels, which are NOT used here.

tests/test_mamba2.py verifies the two forms agree.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Block(nn.Module):
    """Pre-norm residual block implementing the Mamba-2 SSD layer.

    Per head h (headdim P, state size N), with B/C shared per group:
        h_t = alpha_{h,t} * h_{t-1} + delta_{h,t} * B_{g(h),t} (x) x_{h,t}
        y_{h,t} = C_{g(h),t}^T h_{h,t}
    alpha_{h,t} = exp(delta_{h,t} * A_h) with learned scalar A_h < 0.
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
        chunk_size: int = 64,
        scan_mode: str = "chunked",
    ) -> None:
        super().__init__()
        if scan_mode not in ("sequential", "chunked"):
            raise ValueError(f"scan_mode must be 'sequential' or 'chunked', got {scan_mode}")
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
        self.chunk_size = chunk_size
        self.scan_mode = scan_mode

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            bias=True, padding=0,
        )
        # x -> (dt_rank, B for n_groups, C for n_groups)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * n_groups * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.n_heads, bias=True)

        a_log = torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(a_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init_dt()

    def _init_dt(self) -> None:
        """Bias init so softplus(output) is log-uniform in [1e-3, 1e-1] per head."""
        dt_min, dt_max = 1e-3, 1e-1
        with torch.no_grad():
            u = torch.linspace(0.0, 1.0, self.n_heads)
            dt = torch.exp(u * math.log(dt_max) + (1 - u) * math.log(dt_min))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -std, std)

    def _scan_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """From convolved tokens (B, T, d_inner) derive scan inputs.

        Returns:
            x_h: (B, T, H, P)
            alpha: (B, T, H) = exp(delta * A)
            Bx: (B, T, H, N)  the state input delta_t * B_t, expanded to heads
            C_h: (B, T, H, N)
        """
        proj = self.x_proj(x)
        dt, B, C = torch.split(
            proj, [self.dt_rank, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt))  # (B, T, H)
        A = -torch.exp(self.A_log.float())  # (H,)
        alpha = torch.exp(delta * A)  # (B, T, H)
        rep = self.n_heads // self.n_groups
        B_h = B.repeat_interleave(rep, dim=-1).view(*B.shape[:-1], self.n_heads, self.d_state)
        C_h = C.repeat_interleave(rep, dim=-1).view(*C.shape[:-1], self.n_heads, self.d_state)
        Bx = B_h * delta.unsqueeze(-1)  # state input = delta * B (gamma_t * B_t)
        x_h = x.view(*x.shape[:-1], self.n_heads, self.headdim)
        return x_h, alpha, Bx, C_h

    # -------------------- sequential reference --------------------

    def sequential_scan(
        self, x_h: torch.Tensor, alpha: torch.Tensor, Bx: torch.Tensor, C_h: torch.Tensor
    ) -> torch.Tensor:
        """Reference recurrence. Inputs (B, T, H, ...) -> y (B, T, H, P), fp32 compute."""
        Bsz, T, H, P = x_h.shape
        N = self.d_state
        h = torch.zeros(Bsz, H, N, P, device=x_h.device, dtype=torch.float32)
        ys = []
        aT = alpha.transpose(0, 1).float()
        xT = x_h.transpose(0, 1).float()
        bT = Bx.transpose(0, 1).float()
        cT = C_h.transpose(0, 1).float()
        for t in range(T):
            h = aT[t][:, :, None, None] * h + bT[t].unsqueeze(-1) * xT[t].unsqueeze(2)
            ys.append(torch.einsum("bhnp,bhn->bhp", h, cT[t]))
        return torch.stack(ys, dim=1)

    # -------------------- chunked SSD --------------------

    def chunked_scan(
        self, x_h: torch.Tensor, alpha: torch.Tensor, Bx: torch.Tensor, C_h: torch.Tensor
    ) -> torch.Tensor:
        """Chunked SSD algorithm (fp32 compute).

        For each chunk of length Csz with carried state S (N, P):

            y_intra[t] = sum_{s<=t} L_{ts} * (C_t . B_s) * x_s   (intra-chunk matmul)
            y_inter[t] = alpha_prefix_t * (C_t^T S)              (carry-in)
            S_next     = alpha_total * S + sum_t alpha_suffix_t * B_t (x) x_t

        L_{ts} = prod_{r=s+1..t} alpha_r; alpha_prefix/suffix are within-chunk
        cumulative decay products computed via a cumsum of log alpha.
        """
        Bsz, T, H, P = x_h.shape
        N, Csz = self.d_state, self.chunk_size
        x_f = x_h.float()
        B_f = Bx.float()
        C_f = C_h.float()
        log_alpha = torch.log(alpha.float().clamp_min(1e-12))

        pad = (-T) % Csz
        if pad:
            x_f = F.pad(x_f, (0, 0, 0, 0, 0, pad))
            B_f = F.pad(B_f, (0, 0, 0, 0, 0, pad))
            C_f = F.pad(C_f, (0, 0, 0, 0, 0, pad))
            log_alpha = F.pad(log_alpha, (0, 0, 0, pad), value=0.0)
        Tp = T + pad
        nc = Tp // Csz

        xl = x_f.view(Bsz, nc, Csz, H, P)
        Bl = B_f.view(Bsz, nc, Csz, H, N)
        Cl = C_f.view(Bsz, nc, Csz, H, N)
        cs = log_alpha.view(Bsz, nc, Csz, H).cumsum(dim=2)  # (B, nc, Csz, H)

        # Intra-chunk decayed attention: L[t,s] = exp(cs_t - cs_s) for t >= s.
        diff = cs.unsqueeze(3) - cs.unsqueeze(2)  # (B, nc, t, s, H)
        tril = torch.tril(
            torch.ones(Csz, Csz, dtype=torch.bool, device=x_h.device)
        ).unsqueeze(-1)  # (Csz, Csz, 1) broadcasts over H
        # Clamp before exponentiating: lower-triangle diffs are already <= 0,
        # upper-triangle diffs can overflow fp32 exp at long chunks.
        L = torch.exp(diff.clamp(max=0.0)) * tril  # (B, nc, Csz, Csz, H)
        attn = torch.einsum("bcihn,bcjhn->bcijh", Cl, Bl)  # C_i . B_j
        y_intra = torch.einsum("bcijh,bcjhp->bcihp", L * attn, xl)

        # State accumulated by chunk inputs alone (carried state added later).
        cs_end = cs[:, :, -1]  # (B, nc, H)
        alpha_suffix = torch.exp(cs_end.unsqueeze(2) - cs)  # (B, nc, Csz, H)
        S_partial = torch.einsum(
            "bcjhn,bcjhp->bchnp", Bl * alpha_suffix.unsqueeze(-1), xl
        )  # (B, nc, H, N, P)

        # Loop over chunks (T/Csz iterations) for the carried state.
        C_prefix = Cl * torch.exp(cs).unsqueeze(-1)  # (B, nc, Csz, H, N)
        y = torch.empty(Bsz, nc, Csz, H, P, device=x_h.device, dtype=torch.float32)
        S = torch.zeros(Bsz, H, N, P, device=x_h.device, dtype=torch.float32)
        for c in range(nc):
            y_inter = torch.einsum("bihn,bhnp->bihp", C_prefix[:, c], S)
            y[:, c] = y_intra[:, c] + y_inter
            S = cs_end[:, c].unsqueeze(-1).unsqueeze(-1).exp() * S + S_partial[:, c]

        return y.reshape(Bsz, Tp, H, P)[:, :T]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        T = x.shape[1]
        xz = self.in_proj(self.norm(x))
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = x_in.transpose(1, 2)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)[..., :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_h, alpha, Bx, C_h = self._scan_inputs(x_conv)
        scan = self.chunked_scan if self.scan_mode == "chunked" else self.sequential_scan
        y = scan(x_h, alpha, Bx, C_h)
        y = y.reshape(*y.shape[:2], self.d_inner)
        y = y + x_conv * self.D
        return x + self.out_proj(y * F.silu(z))


class Mamba2Backbone(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 32,
        headdim: int = 64,
        expand: int = 2,
        n_groups: int = 1,
        chunk_size: int = 64,
        scan_mode: str = "chunked",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [
                Mamba2Block(
                    d_model, d_state=d_state, headdim=headdim, expand=expand,
                    n_groups=n_groups, chunk_size=chunk_size, scan_mode=scan_mode,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = emb
        for block in self.blocks:
            x = block(x)
        return x
