"""Educational Mamba-2: scalar-identity SSM with the chunked SSD algorithm.

Scope and honesty notes
-----------------------
Written from first principles following "Transformers are SSMs" (Dao & Gu 2024,
arXiv:2405.21060). Mamba-2 restricts the state transition to a scalar times
identity per head (A_h < 0), which turns the selective SSM into the **State
Space Duality (SSD)** form: within a chunk of tokens the output is a masked,
decay-weighted attention-like matmul, and only a small (N, P) state crosses
chunk boundaries. The layer becomes a sequence of tensor-core matmuls instead
of a long scan; this is exactly *why* Mamba-2 exists.

This module implements BOTH forms:

- ``scan_mode="sequential"``: the plain recurrence h_t = alpha_t h_{t-1} +
  (delta_t B_t) (x) x_t, one python-loop step at a time. Reference form.
- ``scan_mode="chunked"``: the SSD chunked algorithm. A python loop over
  T/chunk_size chunks, each doing small matmuls; the official repository fuses
  this into Triton kernels, which are NOT used here.

tests/test_models.py checks that the two scans agree on the same inputs, and
tests/test_references.py runs both through forward() against a naive per-head
float64 loop and checks that every head reads its group's full B and C.
prefill() returns the final chunk state and step() continues the recurrence one
token at a time; tests/test_decode.py checks that together they reproduce
forward().
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    Backbone,
    autocast_disabled,
    expand_groups_to_heads,
    scan_dtype,
    short_conv,
    short_conv_step,
)


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
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
            padding=0,
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
            std = self.dt_rank**-0.5
            nn.init.uniform_(self.dt_proj.weight, -std, std)

    @autocast_disabled
    def _scan_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """From convolved tokens (B, T, d_inner) derive scan inputs.

        The projections run in scan_dtype (float32, or float64 for float64 inputs)
        even under bf16 autocast, so delta, B and C are not bf16-rounded.

        Returns:
            x_h: (B, T, H, P)
            log_alpha: (B, T, H) = delta * A <= 0, the log of the per-head decay alpha
            Bx: (B, T, H, N)  the state input delta_t * B_t; head h reads group h // (H/G)
            C_h: (B, T, H, N)
        """
        x = x.to(scan_dtype(x))
        proj = self.x_proj(x)
        dt, B, C = torch.split(
            proj,
            [self.dt_rank, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt))  # (B, T, H)
        A = -torch.exp(self.A_log.to(scan_dtype(delta)))  # (H,)
        log_alpha = delta * A  # (B, T, H); alpha = exp(log_alpha)
        B_h = expand_groups_to_heads(B, self.n_groups, self.n_heads)  # (B, T, H, N)
        C_h = expand_groups_to_heads(C, self.n_groups, self.n_heads)  # (B, T, H, N)
        Bx = B_h * delta.unsqueeze(-1)  # state input = delta * B (gamma_t * B_t)
        x_h = x.view(*x.shape[:-1], self.n_heads, self.headdim)
        return x_h, log_alpha, Bx, C_h

    # -------------------- sequential reference --------------------

    def sequential_scan(
        self, x_h: torch.Tensor, alpha: torch.Tensor, Bx: torch.Tensor, C_h: torch.Tensor
    ) -> torch.Tensor:
        """Reference recurrence. Inputs (B, T, H, ...) -> y (B, T, H, P)."""
        return self.sequential_scan_with_state(x_h, alpha, Bx, C_h)[0]

    @autocast_disabled
    def sequential_scan_with_state(
        self, x_h: torch.Tensor, alpha: torch.Tensor, Bx: torch.Tensor, C_h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference recurrence returning (y (B, T, H, P), final state h_T (B, H, N, P)).

        Computes in scan_dtype(...): float32, or float64 for float64 inputs.
        """
        Bsz, T, H, P = x_h.shape
        N = self.d_state
        dtype = scan_dtype(x_h, alpha, Bx, C_h)
        h = torch.zeros(Bsz, H, N, P, device=x_h.device, dtype=dtype)
        ys = []
        aT = alpha.transpose(0, 1).to(dtype)
        xT = x_h.transpose(0, 1).to(dtype)
        bT = Bx.transpose(0, 1).to(dtype)
        cT = C_h.transpose(0, 1).to(dtype)
        for t in range(T):
            h = aT[t][:, :, None, None] * h + bT[t].unsqueeze(-1) * xT[t].unsqueeze(2)
            ys.append(torch.einsum("bhnp,bhn->bhp", h, cT[t]))
        return torch.stack(ys, dim=1), h

    # -------------------- chunked SSD --------------------

    def chunked_scan(
        self,
        x_h: torch.Tensor,
        alpha: torch.Tensor | None,
        Bx: torch.Tensor,
        C_h: torch.Tensor,
        log_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """chunked_scan_with_state without the final state: y (B, T, H, P)."""
        return self.chunked_scan_with_state(x_h, alpha, Bx, C_h, log_alpha)[0]

    @autocast_disabled
    def chunked_scan_with_state(
        self,
        x_h: torch.Tensor,
        alpha: torch.Tensor | None,
        Bx: torch.Tensor,
        C_h: torch.Tensor,
        log_alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked SSD algorithm, computed in scan_dtype(...) (float32, or float64).

        For each chunk of length Csz with carried state S (N, P):

            y_intra[t] = sum_{s<=t} L_{ts} * (C_t . B_s) * x_s   (intra-chunk matmul)
            y_inter[t] = alpha_prefix_t * (C_t^T S)              (carry-in)
            S_next     = alpha_total * S + sum_t alpha_suffix_t * B_t (x) x_t

        L_{ts} = prod_{r=s+1..t} alpha_r; alpha_prefix/suffix are within-chunk
        cumulative decay products. Every log-decay sum is accumulated directly over
        its own range (a masked cumsum, as in the reference ssd_minimal segsum),
        never as a difference of two running cumsums: such a difference loses
        float32 precision after one large decay and needs a clamp whose gradient
        vanishes at zero.

        Pass log_alpha = delta * A (what forward() does) so tiny decays keep their
        exact value and gradient. Passing alpha instead takes its log, floored at
        the dtype's smallest normal number only to avoid log(0).

        Returns (y (B, T, H, P), state after the last token (B, H, N, P)); padded
        positions have decay 1 and no input, so they leave the carried state unchanged.
        """
        Bsz, T, H, P = x_h.shape
        N, Csz = self.d_state, self.chunk_size
        if log_alpha is None:
            if alpha is None:
                raise ValueError("chunked_scan needs alpha or log_alpha")
            dtype = scan_dtype(x_h, alpha, Bx, C_h)
            log_alpha = torch.log(alpha.to(dtype).clamp_min(torch.finfo(dtype).tiny))
        else:
            dtype = scan_dtype(x_h, log_alpha, Bx, C_h)
            log_alpha = log_alpha.to(dtype)
        x_f = x_h.to(dtype)
        B_f = Bx.to(dtype)
        C_f = C_h.to(dtype)

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
        la = log_alpha.view(Bsz, nc, Csz, H)
        cs = la.cumsum(dim=2)  # (B, nc, Csz, H): sum_{r<=t}, the carry-in decay
        cs_end = cs[:, :, -1]  # (B, nc, H): whole-chunk decay
        # sum_{r=t+1..Csz-1}: a reverse cumsum of the left-shifted log decays.
        shifted = F.pad(la[:, :, 1:], (0, 0, 0, 1))
        suffix = shifted.flip(2).cumsum(dim=2).flip(2)  # (B, nc, Csz, H)

        # Process chunks in groups so the (group, Csz, Csz, H) decay tensors
        # stay bounded; identical outputs, ~1GB less transient memory at 32K.
        group = 32
        y_intra = torch.empty(Bsz, nc, Csz, H, P, device=x_h.device, dtype=dtype)
        S_partial = torch.empty(Bsz, nc, H, N, P, device=x_h.device, dtype=dtype)
        ones = torch.ones(Csz, Csz, dtype=torch.bool, device=x_h.device)
        strict = torch.tril(ones, diagonal=-1).unsqueeze(-1)  # (t, s, 1): s < t
        causal = torch.tril(ones).unsqueeze(-1)  # (t, s, 1): s <= t
        for g0 in range(0, nc, group):
            g1 = min(g0 + group, nc)
            # seg[t, s] = sum_{r=s+1..t} la_r, accumulated over r for each s.
            la_ts = la[:, g0:g1].unsqueeze(3).expand(-1, -1, -1, Csz, -1)  # (B, g, t, s, H)
            seg = la_ts.masked_fill(~strict, 0.0).cumsum(dim=2)
            L = torch.exp(seg.masked_fill(~causal, float("-inf")))  # zero above the diagonal
            attn = torch.einsum("bgihn,bgjhn->bgijh", Cl[:, g0:g1], Bl[:, g0:g1])
            y_intra[:, g0:g1] = torch.einsum("bgijh,bgjhp->bgihp", L * attn, xl[:, g0:g1])
            alpha_suffix = torch.exp(suffix[:, g0:g1])
            S_partial[:, g0:g1] = torch.einsum(
                "bgjhn,bgjhp->bghnp",
                Bl[:, g0:g1] * alpha_suffix.unsqueeze(-1),
                xl[:, g0:g1],
            )

        # Loop over chunks (T/Csz iterations) for the carried state.
        C_prefix = Cl * torch.exp(cs).unsqueeze(-1)  # (B, nc, Csz, H, N)
        y = torch.empty(Bsz, nc, Csz, H, P, device=x_h.device, dtype=dtype)
        S = torch.zeros(Bsz, H, N, P, device=x_h.device, dtype=dtype)
        for c in range(nc):
            y_inter = torch.einsum("bihn,bhnp->bihp", C_prefix[:, c], S)
            y[:, c] = y_intra[:, c] + y_inter
            S = cs_end[:, c].unsqueeze(-1).unsqueeze(-1).exp() * S + S_partial[:, c]

        return y.reshape(Bsz, Tp, H, P)[:, :T], S

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """forward(x) plus the decode state after the last token: (B, T, D) -> ((B, T, D), state)."""
        xz = self.in_proj(self.norm(x))
        x_in, z = xz.chunk(2, dim=-1)
        x_conv, history = short_conv(self.conv1d, x_in)

        x_h, log_alpha, Bx, C_h = self._scan_inputs(x_conv)
        if self.scan_mode == "chunked":
            y, h = self.chunked_scan_with_state(x_h, None, Bx, C_h, log_alpha=log_alpha)
        else:
            y, h = self.sequential_scan_with_state(x_h, torch.exp(log_alpha), Bx, C_h)
        y = y.reshape(*y.shape[:2], self.d_inner)
        y = y + x_conv * self.D
        return x + self.out_proj(y * F.silu(z)), {"h": h, "conv": history}

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.prefill(x)[0]

    # -------------------- incremental decode (O(1) state per block) --------------------

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None
    ) -> dict[str, torch.Tensor]:
        """Empty decode state: SSM state h (B, H, N, P) in scan dtype, short-conv history."""
        param_dtype = self.in_proj.weight.dtype
        h = torch.zeros(
            batch_size,
            self.n_heads,
            self.d_state,
            self.headdim,
            device=device,
            dtype=torch.promote_types(param_dtype, torch.float32),
        )
        conv = torch.zeros(
            batch_size, self.d_inner, self.d_conv - 1, device=device, dtype=dtype or param_dtype
        )
        return {"h": h, "conv": conv}

    @autocast_disabled
    def _recurrent_update(
        self,
        h: torch.Tensor,
        x_h: torch.Tensor,
        log_alpha: torch.Tensor,
        Bx: torch.Tensor,
        C_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step in scan dtype: (h (B, H, N, P), x_h (B, H, P),
        log_alpha (B, H), Bx and C_h (B, H, N)) -> (y (B, H, P), new h)."""
        dtype = scan_dtype(h, x_h, log_alpha, Bx, C_h)
        decay = log_alpha.to(dtype).exp()[:, :, None, None]
        inject = Bx.to(dtype).unsqueeze(-1) * x_h.to(dtype).unsqueeze(2)  # (B, H, N, P)
        h = decay * h.to(dtype) + inject
        return torch.einsum("bhnp,bhn->bhp", h, C_h.to(dtype)), h

    def step(self, x_t: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """One decode step. x_t: (B, 1, D) -> (B, 1, D), O(1) time and state per token."""
        xz = self.in_proj(self.norm(x_t))  # (B, 1, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)
        conv_out, state["conv"] = short_conv_step(self.conv1d, state["conv"], x_in.squeeze(1))
        x_h, log_alpha, Bx, C_h = self._scan_inputs(conv_out)  # (B, H, P), (B, H), (B, H, N)
        y, state["h"] = self._recurrent_update(state["h"], x_h, log_alpha, Bx, C_h)
        y = y.reshape(y.shape[0], self.d_inner) + conv_out * self.D
        out = self.out_proj(y * F.silu(z.squeeze(1)))
        return out.unsqueeze(1) + x_t


class Mamba2Backbone(Backbone):
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
                    d_model,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                    n_groups=n_groups,
                    chunk_size=chunk_size,
                    scan_mode=scan_mode,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = emb
        for block in self.blocks:
            x = block(x)
        return x
