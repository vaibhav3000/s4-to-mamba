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
   quality without increasing decode latency. Not implemented here (see below).

The exponential-trapezoidal recurrence in point 1 is implemented as written.

Implementation details and omissions
------------------------------------
- theta is a static learned parameter of shape (H, N), not computed from the
  input. The rotation applied at each step, delta_t * theta, depends on the
  token only through delta_t.
- The real part of A is a vector per (head, state slot): a_real_log has shape
  (H, N), where Mamba-2 uses one scalar per head.
- B and C are real and the readout is y = C^T Re(h), so the imaginary part of
  the state is never read directly.
- No MIMO: the SISO-per-head structure of Mamba-2's SSD is kept.
- Two scan forms with identical results: the sequential recurrence (the
  reference) and a chunked parallel form (``scan_mode="chunked"``, the default),
  a complex generalization of Mamba-2's SSD in which the exponential-trapezoidal
  rule appears as a two-band correction of the input weights (the paper's
  "1-semiseparable composed with 2-band" structure, Section 3.1.3). The
  derivation is in docs/design/step2.md section 3.4; tests/test_mamba3_chunked.py
  checks the two forms against each other in float64.
- ``rotation=False`` fixes theta at zero (a buffer, not a parameter): the
  transition becomes real and decay-only. This is the rotation ablation.

See docs/evolution.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    Backbone,
    autocast_disabled,
    complex_dtype_for,
    expand_groups_to_heads,
    scan_dtype,
)

# Transient-memory budget of one group of chunks in chunked_scan (bytes).
CHUNK_GROUP_BYTES = 256 * 2**20


class Mamba3Block(nn.Module):
    """Pre-norm residual block with the Mamba-3 complex selective recurrence.

    Per head h (headdim P, state size N), with B/C shared per group g(h), and
    elementwise over the N state slots:
        h_t = alpha_t h_{t-1} + beta_t * s_{t-1} + gamma_t * s_t
        s_t = B_{g(h),t} (x) x_{h,t}   (real outer product; delta enters via beta_t, gamma_t)
        y_{h,t} = C_{g(h),t}^T Re(h_{h,t})
    alpha_t = exp(delta_{h,t} A_h) complex, |alpha_t| < 1 (stability), where
    A_h = -exp(a_real_log[h]) + i * theta[h] is an N-vector (one complex entry per
    state slot), not a scalar, and theta is a static learned parameter;
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
        chunk_size: int = 64,
        scan_mode: str = "chunked",
        rotation: bool = True,
    ) -> None:
        super().__init__()
        if scan_mode not in ("sequential", "chunked"):
            raise ValueError(f"scan_mode must be 'sequential' or 'chunked', got {scan_mode}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size
        self.scan_mode = scan_mode
        self.rotation = rotation
        self.chunk_group_bytes = CHUNK_GROUP_BYTES
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
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
            padding=0,
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
        self.theta: torch.Tensor
        if rotation:
            self.theta = nn.Parameter(theta_init.repeat(self.n_heads, 1))
        else:  # rotation ablation: real, decay-only transition
            self.register_buffer("theta", torch.zeros(self.n_heads, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init_dt()

    def _init_dt(self) -> None:
        dt_min, dt_max = 1e-3, 1e-1
        with torch.no_grad():
            u = torch.linspace(0.0, 1.0, self.n_heads)
            dt = torch.exp(u * math.log(dt_max) + (1 - u) * math.log(dt_min))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            std = self.dt_rank**-0.5
            nn.init.uniform_(self.dt_proj.weight, -std, std)

    @autocast_disabled
    def _ssm_inputs(
        self, x_conv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scan inputs (x_h, delta, B_h, C_h, lam) from convolved tokens (B, T, d_inner).

        The projections run in scan_dtype (float32, or float64 for float64 inputs)
        even under bf16 autocast, so delta, B, C and lambda are not bf16-rounded.
        """
        x_conv = x_conv.to(scan_dtype(x_conv))
        proj = self.x_proj(x_conv)
        sizes = [
            self.dt_rank,
            self.n_groups * self.d_state,
            self.n_groups * self.d_state,
            self.n_groups,
        ]
        dt, B, C, lam_raw = torch.split(proj, sizes, dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # (B, T, H)
        lam = torch.sigmoid(self.lambda_proj(lam_raw))  # (B, T, H)
        # Head h reads the full N-vectors B and C of group h // (H / G).
        B_h = expand_groups_to_heads(B, self.n_groups, self.n_heads)  # (B, T, H, N)
        C_h = expand_groups_to_heads(C, self.n_groups, self.n_heads)  # (B, T, H, N)
        # lambda_proj outputs one weight per head directly (no group expansion).
        x_h = x_conv.view(*x_conv.shape[:-1], self.n_heads, self.headdim)
        return x_h, delta, B_h, C_h, lam

    def selective_scan(
        self,
        x_h: torch.Tensor,  # (B, T, H, P)
        delta: torch.Tensor,  # (B, T, H)
        B_h: torch.Tensor,  # (B, T, H, N)  (group-expanded, real)
        C_h: torch.Tensor,  # (B, T, H, N)
        lam: torch.Tensor,  # (B, T, H) in (0, 1)
    ) -> torch.Tensor:
        """Sequential Mamba-3 recurrence (the reference form). Returns y: (B, T, H, P)."""
        return self.selective_scan_with_state(x_h, delta, B_h, C_h, lam)[0]

    @autocast_disabled
    def selective_scan_with_state(
        self,
        x_h: torch.Tensor,
        delta: torch.Tensor,
        B_h: torch.Tensor,
        C_h: torch.Tensor,
        lam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential recurrence returning (y (B, T, H, P), h_T (B, H, N, P) complex,
        s_T (B, H, N, P) real), where s_T = B_{T-1} (x) x_{T-1} is the boundary input
        the next step's beta term still needs.

        Computes in scan_dtype(...) with a complex state of the matching width:
        float32/complex64, or float64/complex128 for float64 inputs.
        """
        Bsz, T, H, P = x_h.shape
        N = self.d_state
        dtype = scan_dtype(x_h, delta, B_h, C_h, lam)
        cdtype = complex_dtype_for(dtype)
        A = -torch.exp(self.a_real_log.to(dtype)) + 1j * self.theta.to(dtype)  # (H, N) complex

        xT = x_h.transpose(0, 1).to(dtype)
        dT = delta.transpose(0, 1).to(dtype)
        bT = B_h.transpose(0, 1).to(dtype)
        cT = C_h.transpose(0, 1).to(dtype)
        lamT = lam.transpose(0, 1).to(dtype)

        h = torch.zeros(Bsz, H, N, P, dtype=cdtype, device=x_h.device)
        prev_inject = torch.zeros(Bsz, H, N, P, dtype=cdtype, device=x_h.device)
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
                + gamma.unsqueeze(-1).unsqueeze(-1) * inject_t.to(cdtype)
            )
            ys.append(torch.einsum("bhnp,bhn->bhp", h.real, cT[t]))
            prev_inject = inject_t.to(cdtype)
        return torch.stack(ys, dim=1), h, prev_inject.real

    # -------------------- chunked parallel form --------------------

    def chunked_scan(
        self,
        x_h: torch.Tensor,
        delta: torch.Tensor,
        B_h: torch.Tensor,
        C_h: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        """chunked_scan_with_state without the final state: y (B, T, H, P)."""
        return self.chunked_scan_with_state(x_h, delta, B_h, C_h, lam)[0]

    @autocast_disabled
    def chunked_scan_with_state(
        self,
        x_h: torch.Tensor,  # (B, T, H, P)
        delta: torch.Tensor,  # (B, T, H)
        B_h: torch.Tensor,  # (B, T, H, N)
        C_h: torch.Tensor,  # (B, T, H, N)
        lam: torch.Tensor,  # (B, T, H)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chunked parallel form of selective_scan_with_state; same inputs and outputs.

        With A_n = -exp(a_n) + i theta_n and segment sums D_ts = sum_{r=s+1..t} delta_r,
        the recurrence unrolls to (docs/design/step2.md section 3.4)

            h_t = sum_{s<=t} exp(A_n D_ts) * (B_s (x) x_s) * W_ts,
            W_ts = gamma_s + [s < t] beta'_s,  gamma_s = lam_s delta_s,
            beta'_s = (1 - lam_{s+1}) delta_{s+1},

        the two-band correction: the complex alpha inside beta folds into the decay.
        Because B, C and x are real, y_t = sum_s G_ts W_ts x_s with the real
        G_ts = sum_n C_tn B_sn exp(Re(A_n) D_ts) cos(theta_n D_ts).

        Per chunk of Q tokens: the intra-chunk term from (Q, Q, N) decays built from
        direct masked cumsums (never a difference of running sums; ADR 0002), a
        carry-in from the previous chunk's state plus its boundary input, and a
        carry-out to the next chunk. Chunks are processed in groups whose transient
        memory stays near self.chunk_group_bytes; the carry runs inside the group
        loop, so no per-chunk state tensor is stored.
        """
        Bsz, T, H, P = x_h.shape
        N, Q = self.d_state, self.chunk_size
        dtype = scan_dtype(x_h, delta, B_h, C_h, lam)
        cdtype = complex_dtype_for(dtype)
        a_re = -torch.exp(self.a_real_log.to(dtype))  # (H, N), negative
        theta = self.theta.to(dtype)  # (H, N)
        A = torch.complex(a_re, theta)  # (H, N)

        x = x_h.to(dtype)
        d = delta.to(dtype)
        b = B_h.to(dtype)
        c = C_h.to(dtype)
        lm = lam.to(dtype)
        gamma = lm * d  # (B, T, H): weight of s_t on the diagonal
        beta_next = (1.0 - lm) * d  # (B, T, H): weight position t gives to s_{t-1}
        beta_prime = F.pad(beta_next[:, 1:], (0, 0, 0, 1))  # beta'_s = beta_next[s + 1]
        s_last = (b[:, -1].unsqueeze(-1) * x[:, -1].unsqueeze(2)).contiguous()  # (B, H, N, P)

        pad = (-T) % Q
        if pad:
            # Padded steps: delta = 0 (no decay, no rotation) and no input, so they
            # leave the carried state unchanged; their outputs are dropped below.
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            b = F.pad(b, (0, 0, 0, 0, 0, pad))
            c = F.pad(c, (0, 0, 0, 0, 0, pad))
            d = F.pad(d, (0, 0, 0, pad))
            gamma = F.pad(gamma, (0, 0, 0, pad))
            beta_prime = F.pad(beta_prime, (0, 0, 0, pad))
        nc = (T + pad) // Q
        x = x.view(Bsz, nc, Q, H, P)
        b = b.view(Bsz, nc, Q, H, N)
        c = c.view(Bsz, nc, Q, H, N)
        d = d.view(Bsz, nc, Q, H)
        gamma = gamma.view(Bsz, nc, Q, H)
        beta_prime = beta_prime.view(Bsz, nc, Q, H)

        prefix = d.cumsum(dim=2)  # (B, nc, Q, H): sum_{r<=i}, the carry-in decay exponent
        shifted = F.pad(d[:, :, 1:], (0, 0, 0, 1))
        suffix = shifted.flip(2).cumsum(dim=2).flip(2)  # sum_{r=j+1..Q-1}, the carry-out one
        ones = torch.ones(Q, Q, dtype=torch.bool, device=x_h.device)
        strict = torch.tril(ones, diagonal=-1).unsqueeze(-1)  # (i, j, 1): j < i
        causal = torch.tril(ones).unsqueeze(-1)  # (i, j, 1): j <= i
        last = torch.zeros(Q, 1, dtype=dtype, device=x_h.device)
        last[-1] = 1.0  # the carry-out row: W_{Q-1, j} has no beta' for j = Q - 1

        itemsize = torch.finfo(dtype).bits // 8
        per_chunk = itemsize * Bsz * H * Q * (4 * Q * N + 4 * N * P + 2 * Q)
        group = max(1, min(nc, self.chunk_group_bytes // max(per_chunk, 1)))

        ys = []
        h = torch.zeros(Bsz, H, N, P, dtype=cdtype, device=x_h.device)
        carry = h  # state entering the next chunk, including its boundary input
        for g0 in range(0, nc, group):
            g1 = min(g0 + group, nc)
            # D_ij = sum_{r=j+1..i} delta_r as a masked cumsum over r (zero for j >= i).
            d_rs = d[:, g0:g1].unsqueeze(3).expand(-1, -1, -1, Q, -1)  # (B, g, r, j, H)
            seg = d_rs.masked_fill(~strict, 0.0).cumsum(dim=2).unsqueeze(-1)  # (B,g,i,j,H,1)
            decay = torch.exp(a_re * seg) * torch.cos(theta * seg)  # (B, g, i, j, H, N)
            G = torch.einsum("bgihn,bgjhn,bgijhn->bgijh", c[:, g0:g1], b[:, g0:g1], decay)
            gam = gamma[:, g0:g1].unsqueeze(2)  # (B, g, 1, j, H)
            bet = beta_prime[:, g0:g1].unsqueeze(2)
            W = gam * causal + bet * strict  # (B, g, i, j, H): the two-band input weights
            y_intra = torch.einsum("bgijh,bgjhp->bgihp", G * W, x[:, g0:g1])

            # Carry-out partial state: sum_j exp(A suffix_j) W_{Q-1, j} B_j (x) x_j.
            w_end = gamma[:, g0:g1] + beta_prime[:, g0:g1] * (1.0 - last)  # (B, g, Q, H)
            end_decay = torch.exp(suffix[:, g0:g1].unsqueeze(-1) * A)  # (B, g, Q, H, N)
            coef = end_decay * (w_end.unsqueeze(-1) * b[:, g0:g1])
            s_partial = torch.einsum("bgjhn,bgjhp->bghnp", coef, x[:, g0:g1].to(cdtype))
            pre_decay = torch.exp(prefix[:, g0:g1].unsqueeze(-1) * A)  # (B, g, Q, H, N)
            c_pre = c[:, g0:g1] * pre_decay  # (B, g, Q, H, N) complex

            for k in range(g1 - g0):
                y_inter = torch.einsum("bihn,bhnp->bihp", c_pre[:, k], carry).real
                ys.append(y_intra[:, k] + y_inter)
                h = pre_decay[:, k, -1].unsqueeze(-1) * carry + s_partial[:, k]
                # Boundary input: the chunk's last position still owes beta' s_last.
                s_end = b[:, g0 + k, -1].unsqueeze(-1) * x[:, g0 + k, -1].unsqueeze(2)
                carry = h + beta_prime[:, g0 + k, -1, :, None, None] * s_end
        y = torch.stack(ys, dim=1).reshape(Bsz, nc * Q, H, P)[:, :T]
        return y, h, s_last

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        T = x.shape[1]
        xz = self.in_proj(self.norm(x))
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = x_in.transpose(1, 2)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)[..., :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_h, delta, B_h, C_h, lam_h = self._ssm_inputs(x_conv)
        if self.scan_mode == "chunked":
            y = self.chunked_scan(x_h, delta, B_h, C_h, lam_h)
        else:
            y = self.selective_scan(x_h, delta, B_h, C_h, lam_h)
        y = y.reshape(*y.shape[:2], self.d_inner)
        y = y + x_conv * self.D
        return x + self.out_proj(y * F.silu(z))


class Mamba3Backbone(Backbone):
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
        rotation: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [
                Mamba3Block(
                    d_model,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                    n_groups=n_groups,
                    chunk_size=chunk_size,
                    scan_mode=scan_mode,
                    rotation=rotation,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = emb
        for block in self.blocks:
            x = block(x)
        return x
