"""Educational S4D: diagonal structured State Space layer (S4 simplified to the diagonal case).

Scope and honesty notes
-----------------------
This module implements **S4D** (Gu, Gupta, Goel, Ré 2022, "On the Parameterization
and Initialization of Diagonal State Space Models"), the diagonal restriction of S4
that keeps every idea the study needs:

- a continuous-time linear state space  h'(t) = A h(t) + B x(t),  y(t) = C^T h(t)
- HiPPO-style initialization of the complex diagonal A (S4D-Lin / S4D-Inv)
- zero-order-hold (ZOH) discretization with the exact (e^{z}-1)/z factor
- a **parallel training form**: the SSM is a causal convolution with kernel
  K_k = sum_n C_n Bbar_n Abar_n^k, computed with FFT convolution in O(T log T)
- a **recurrent inference form**: step-wise state update in O(1) per token with
  a (D, N) complex state.

The full S4 (Normal-Plus-Low-Rank parameterization, Woodbury-corrected Cauchy
kernel) is deliberately NOT re-implemented here; see docs/evolution.md and
REFERENCES.md. Everything in this file is written from first principles for
readability; the official `state-spaces/s4` repository is the reference
implementation used for parameterization conventions.

Shapes: B batch, T time, D channels (d_model), N state size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _expm1(z: torch.Tensor) -> torch.Tensor:
    """e^z - 1 without cancellation for small |z|, real or complex.

    For complex z = x + iy: e^z - 1 = (expm1(x) cos y - 2 sin^2(y/2)) + i e^x sin y,
    built from real ops only, so it runs wherever real expm1, sin and cos do.
    """
    if not z.is_complex():
        return torch.expm1(z)
    x, y = z.real, z.imag
    return torch.complex(
        torch.expm1(x) * torch.cos(y) - 2.0 * torch.sin(0.5 * y) ** 2,
        torch.exp(x) * torch.sin(y),
    )


def zoh_discretization(
    delta: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """ZOH-discretize an SSM with (possibly complex) diagonal A.

    delta: (..., D) step size, a: (..., D, N) diagonal A entries, b: (..., D, N) input matrix.
    Returns (Abar, Bbar) with the same shape as `a`.

    Abar = exp(delta * A)
    Bbar = A^{-1} (exp(delta * A) - 1) * B = delta * B * (e^{z}-1)/z with z = delta*A.

    (e^{z}-1)/z is evaluated as expm1(z)/z, which avoids the cancellation in
    e^z - 1. Its autograd derivative still loses about eps/|z| relative, so below
    a switch point the Taylor series 1 + z/2 + z^2/6 + z^3/24 is used instead:
    |z| < 1e-2 in float32 and below (truncation under |z|^4/120 < 1e-10) and
    |z| < 1e-4 in float64 (truncation under 1e-18). The series also keeps the
    value and its gradient finite as delta*A -> 0.
    """
    z = delta.unsqueeze(-1) * a  # (..., D, N)
    az = torch.exp(z)
    switch = 1e-4 if z.real.dtype == torch.float64 else 1e-2
    small = z.abs() < switch
    # Divide by z only outside the series branch; the masked divisor keeps the
    # unused branch, and its gradient, finite.
    safe_z = torch.where(small, torch.ones_like(z), z)
    series = 1.0 + z / 2.0 + z * z / 6.0 + z * z * z / 24.0
    ratio = torch.where(small, series, _expm1(z) / safe_z)
    return az, delta.unsqueeze(-1) * b * ratio


class S4DLayer(nn.Module):
    """One S4D layer over (B, T, D) real inputs.

    Training path (parallel form): compute the convolution kernel from
    K_k = sum_n C_n * Bbar_n * Abar_n^k, then FFT-convolve. O(T log T).
    Inference path (recurrent form): `step()` maintains a complex (B, D, N) state.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        init: str = "lin",
    ) -> None:
        super().__init__()
        if init not in ("lin", "inv"):
            raise ValueError(f"Unknown S4D init '{init}'")
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A = -exp(log_A_real) + i * A_imag  (HiPPO-flavoured init)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        n = torch.arange(d_state, dtype=torch.float32)
        # S4D-Lin: A_n = -1/2 + i * pi * n;  S4D-Inv: A_n = -1/2 + i * N(N+1) / (2(2n+1))
        a_imag = math.pi * n if init == "lin" else d_state * (d_state + 1) / (2 * (2 * n + 1))
        self.A_imag = nn.Parameter(a_imag.repeat(d_model, 1))

        # Per-channel step size, kept in (dt_min, dt_max) by a softplus.
        u = torch.linspace(0.0, 1.0, d_model)
        dt_init = torch.exp(u * math.log(dt_max) + (1 - u) * math.log(dt_min))
        inv_softplus_dt = dt_init + torch.log(-torch.expm1(-dt_init))  # softplus^-1(dt_init)
        self.dt_param = nn.Parameter(inv_softplus_dt)

        # Complex input/output matrices B, C
        scale = 1.0 / math.sqrt(d_state)
        self.B_re = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * scale)
        # Skip connection (timescale bypass), as in S4D.
        self.D = nn.Parameter(torch.ones(d_model))

    def _params(self) -> dict[str, torch.Tensor]:
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (D, N)
        delta = F.softplus(self.dt_param)  # (D,)
        b = torch.complex(self.B_re, self.B_im)  # (D, N)
        c = torch.complex(self.C_re, self.C_im)  # (D, N)
        return {"a": a, "delta": delta, "b": b, "c": c}

    def kernel(self, L: int, chunk: int = 1024) -> torch.Tensor:
        """Convolution kernel K_k = sum_n C_n Bbar_n Abar_n^k, real part, shape (D, L).

        Computed in L-chunks so the (chunk, D, N) power tensor stays small.
        """
        p = self._params()
        _, b_bar = zoh_discretization(p["delta"], p["a"], p["b"])  # (D, N) complex
        cb = b_bar * p["c"]  # (D, N)
        ks = []
        device = cb.device
        for start in range(0, L, chunk):
            k_idx = torch.arange(start, min(start + chunk, L), device=device, dtype=torch.float32)
            powers = torch.exp(
                k_idx.view(-1, 1, 1) * p["delta"].view(1, -1, 1) * p["a"].view(1, -1, self.d_state)
            )
            ks.append(torch.einsum("ldn,dn->dl", powers, cb))
        return torch.cat(ks, dim=1).real  # y = x conv Re(K)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D) via causal FFT convolution."""
        T = x.shape[1]
        k = self.kernel(T)  # (D, L) real
        k_f = torch.fft.rfft(k, n=2 * T)  # (D, T+1)
        x_f = torch.fft.rfft(x.transpose(1, 2), n=2 * T)  # (B, D, T+1)
        y = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=2 * T)[..., :T]
        y = y.transpose(1, 2)
        return y + x * self.D  # skip

    @torch.no_grad()
    def discretized(self) -> dict[str, torch.Tensor]:
        """(a_bar, b_bar, c), each (D, N) complex, for the recurrent form.

        They depend only on the parameters, so a decode session computes them once
        and passes them to every step().
        """
        p = self._params()
        a_bar, b_bar = zoh_discretization(p["delta"], p["a"], p["b"])
        return {"a_bar": a_bar, "b_bar": b_bar, "c": p["c"]}

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step. x_t: (B, D), state: (B, D, N) complex.

        cache: the output of discretized(); computed here when omitted.
        Returns (y_t, new_state) with y_t: (B, D). O(1) time, O(D*N) state.
        """
        if cache is None:
            cache = self.discretized()
        new_state = cache["a_bar"] * state + cache["b_bar"] * x_t.unsqueeze(-1).to(state.dtype)
        # y = Re(sum_n C_n h_n), consistent with the kernel form K = C * Bbar * Abar^k
        y_t = torch.einsum("bdn,dn->bd", new_state, cache["c"]).real
        return y_t + x_t * self.D, new_state


class S4DBlock(nn.Module):
    """Pre-norm residual block: x + S4D(LN(x)) * silu(gate(LN(x))), as in gated S4 blocks.

    gate is a learned Linear(d_model, d_model).
    """

    def __init__(
        self, d_model: int, d_state: int = 16, dropout: float = 0.0, init: str = "lin"
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = S4DLayer(d_model, d_state=d_state, init=init)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        return x + self.dropout(self.layer(h) * F.silu(self.gate(h)))

    # -------------------- incremental decode --------------------

    def init_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Decode state: the complex SSM state "h" plus the discretized (a_bar, b_bar, c),
        computed once here so step() does not redo the ZOH for every token. Call it
        again after any parameter update."""
        cache = {k: v.to(device) for k, v in self.layer.discretized().items()}
        h = torch.zeros(
            batch_size,
            self.layer.d_model,
            self.layer.d_state,
            dtype=cache["a_bar"].dtype,
            device=device,
        )
        return {"h": h, **cache}

    def step(self, x_t: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """One decode step. x_t: (B, 1, D) -> (B, 1, D), O(1) per token.

        Uses the discretization cached by init_state(); a state dict holding only "h"
        still works and recomputes it.
        """
        h_in = self.norm(x_t.squeeze(1))
        cache = {k: state[k] for k in ("a_bar", "b_bar", "c")} if "a_bar" in state else None
        y, new_h = self.layer.step(h_in, state["h"], cache)
        out = x_t + (y * F.silu(self.gate(h_in))).unsqueeze(1)
        state["h"] = new_h
        return out


class S4DBackbone(nn.Module):
    """Stack of S4D blocks. No positional encoding needed: the recurrence encodes time."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.0,
        init: str = "lin",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [S4DBlock(d_model, d_state, dropout, init) for _ in range(n_layers)]
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = emb
        for block in self.blocks:
            x = block(x, mask)
        return x
