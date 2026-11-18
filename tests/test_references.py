"""Model code checked against independent, naive references.

Every efficient form in src/ssmbench/models is compared with a slow reference
written in this file from the equations alone: float64 or complex128 Python
loops over time, heads and state slots, and ZOH discretization through the
matrix exponential of the augmented system. The references share only trusted
torch primitives (Linear, Conv1d, LayerNorm) with the models.

Tolerances are explicit in every test:

- float64 model vs float64 reference: rtol = atol = 1e-10
- float32 model vs float64 reference: rtol = atol = 1e-4
- torch.autograd.gradcheck in float64: eps = 1e-6, atol = 1e-7, rtol = 1e-5
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import TypedDict

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ssmbench.models.mamba2_minimal import Mamba2Block
from ssmbench.models.mamba3_minimal import Mamba3Block
from ssmbench.models.mamba_minimal import MambaBlock
from ssmbench.models.s4_minimal import S4DLayer, zoh_discretization


class Tolerance(TypedDict):
    rtol: float
    atol: float


FP64: Tolerance = {"rtol": 1e-10, "atol": 1e-10}
FP32_VS_FP64: Tolerance = {"rtol": 1e-4, "atol": 1e-4}
TOL: dict[torch.dtype, Tolerance] = {torch.float64: FP64, torch.float32: FP32_VS_FP64}


def _gradcheck_fn(fn: Callable[..., torch.Tensor], inputs: tuple[torch.Tensor, ...]) -> bool:
    """float64 gradcheck with the tolerances stated in the module docstring."""
    return torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


class _Method(nn.Module):
    """Expose inner.<method>(*args) as forward() so functional_call can swap parameters."""

    def __init__(self, inner: nn.Module, method: str) -> None:
        super().__init__()
        self.inner = inner
        self.method = method

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return getattr(self.inner, self.method)(*args)


def _gradcheck(
    module: nn.Module, method: str, inputs: tuple[torch.Tensor, ...], param_names: list[str]
) -> bool:
    """gradcheck module.<method>(*inputs) with respect to the inputs and the named parameters."""
    wrapper = _Method(module, method)
    params = [getattr(module, n).detach().clone().requires_grad_(True) for n in param_names]
    n_inputs = len(inputs)

    def fn(*tensors: torch.Tensor) -> torch.Tensor:
        swapped = {f"inner.{n}": p for n, p in zip(param_names, tensors[n_inputs:], strict=True)}
        return torch.func.functional_call(wrapper, swapped, tuple(tensors[:n_inputs]))

    return _gradcheck_fn(fn, (*inputs, *params))


# ----------------------------------------------------------------------- S4D


def zoh_reference(
    delta: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ZOH via the matrix exponential of [[delta*a, delta*b], [0, 0]].

    exp([[z, w], [0, 0]]) = [[e^z, w (e^z - 1) / z], [0, 1]], so the top row is
    (Abar, Bbar) for every (channel, state slot), with no closed form involved.
    """
    z = (delta.unsqueeze(-1) * a).to(torch.complex128)
    w = (delta.unsqueeze(-1) * b).to(torch.complex128)
    m = torch.zeros(*z.shape, 2, 2, dtype=torch.complex128)
    m[..., 0, 0] = z
    m[..., 0, 1] = w
    e = torch.linalg.matrix_exp(m)
    return e[..., 0, 0], e[..., 0, 1]


def s4d_params64(
    layer: S4DLayer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(a, delta, b, c) of an S4D layer in complex128 / float64."""
    a = torch.complex(-torch.exp(layer.log_A_real.double()), layer.A_imag.double())
    delta = F.softplus(layer.dt_param.double())
    b = torch.complex(layer.B_re.double(), layer.B_im.double())
    c = torch.complex(layer.C_re.double(), layer.C_im.double())
    return a, delta, b, c


def s4d_kernel_reference(layer: S4DLayer, length: int) -> torch.Tensor:
    """K_k = Re(sum_n C_n Bbar_n Abar_n^k), shape (D, length), Abar^k by repeated products."""
    with torch.no_grad():
        a, delta, b, c = s4d_params64(layer)
        a_bar, b_bar = zoh_reference(delta, a, b)
        power = torch.ones_like(a_bar)
        taps = []
        for _ in range(length):
            taps.append((c * b_bar * power).sum(-1).real)
            power = power * a_bar
        return torch.stack(taps, dim=-1)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("delta_scale", [1.0, 1e-5], ids=["series-off", "series-on"])
def test_s4d_zoh_matches_matrix_exponential(seed: int, delta_scale: float) -> None:
    """Abar and Bbar equal the matrix-exponential ZOH in both branches of the
    (e^z - 1) / z evaluation (|z| >= 1e-4 direct, |z| < 1e-4 Taylor). float64, 1e-10."""
    torch.manual_seed(seed)
    layer = S4DLayer(d_model=4, d_state=8)
    a, delta, b, _ = s4d_params64(layer)
    delta = delta * delta_scale
    z = delta.unsqueeze(-1) * a
    if delta_scale == 1.0:
        assert (z.abs() >= 1e-4).all()
    else:
        assert (z.abs() < 1e-4).all()
    a_bar, b_bar = zoh_discretization(delta, a, b)
    a_ref, b_ref = zoh_reference(delta, a, b)
    torch.testing.assert_close(a_bar, a_ref, **FP64)
    torch.testing.assert_close(b_bar, b_ref, **FP64)


def test_s4d_steady_state_is_the_continuous_dc_gain() -> None:
    """Under constant unit input the ZOH recurrence settles at h* = Bbar / (1 - Abar),
    which for exact ZOH equals the continuous-time steady state -B / A for every
    step size. float64, rtol = atol = 1e-9 (1 - Abar loses ~1e-13 relative)."""
    torch.manual_seed(2)
    layer = S4DLayer(d_model=4, d_state=8)
    a, delta, b, _ = s4d_params64(layer)
    a_bar, b_bar = zoh_discretization(delta, a, b)
    torch.testing.assert_close(b_bar / (1 - a_bar), -b / a, rtol=1e-9, atol=1e-9)


def test_s4d_kernel_matches_reference_across_chunks() -> None:
    """kernel(L) with L = 2100 spans three 1024-token chunks and must equal the
    repeated-product reference everywhere. float64, 1e-10."""
    torch.manual_seed(3)
    layer = S4DLayer(d_model=4, d_state=8).double()
    with torch.no_grad():
        k = layer.kernel(2100)
    assert k.shape == (4, 2100) and k.dtype == torch.float64
    torch.testing.assert_close(k, s4d_kernel_reference(layer, 2100), **FP64)


def test_s4d_kernel_is_chunk_invariant() -> None:
    """Chunking the kernel computation (chunk 16 vs one chunk) changes nothing. float64, 1e-12."""
    torch.manual_seed(4)
    layer = S4DLayer(d_model=4, d_state=8).double()
    with torch.no_grad():
        torch.testing.assert_close(
            layer.kernel(100, chunk=16), layer.kernel(100), rtol=1e-12, atol=1e-12
        )


def test_s4d_conv_equals_recurrent_across_chunks() -> None:
    """FFT-convolution forward equals the step recurrence at T = 2100 (> 2 kernel
    chunks), the path every long-context S4D measurement relies on. float64, 1e-9."""
    torch.manual_seed(5)
    Bsz, T, D, N = 2, 2100, 4, 8
    layer = S4DLayer(d_model=D, d_state=N).double()
    x = torch.randn(Bsz, T, D, dtype=torch.float64)
    with torch.no_grad():
        y_conv = layer(x)
        h = torch.zeros(Bsz, D, N, dtype=torch.complex128)
        ys = []
        for t in range(T):
            y_t, h = layer.step(x[:, t], h)
            ys.append(y_t)
    torch.testing.assert_close(y_conv, torch.stack(ys, dim=1), rtol=1e-9, atol=1e-9)


def test_gradcheck_s4d_layer() -> None:
    """S4D FFT forward in float64: gradients w.r.t. the input and every parameter."""
    torch.manual_seed(12)
    layer = S4DLayer(d_model=2, d_state=3).double()
    x = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
    names = [n for n, _ in layer.named_parameters()]
    assert _gradcheck(layer, "forward", (x,), names)


# ------------------------------------------- scans in float64, Mamba-3 limits


def test_gradcheck_mamba1_scan() -> None:
    """Mamba-1 selective scan: gradients w.r.t. x, delta, B, C, A_log and D."""
    torch.manual_seed(13)
    block = MambaBlock(d_model=2, d_state=3, expand=1).double()
    x = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
    delta = (torch.rand(1, 5, 2, dtype=torch.float64) * 0.5 + 0.05).requires_grad_(True)
    b = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    c = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    assert _gradcheck(block, "selective_scan", (x, delta, b, c), ["A_log", "D"])


@pytest.mark.parametrize("form", ["sequential", "chunked-alpha"])
def test_gradcheck_mamba2_scan(form: str) -> None:
    """Mamba-2 scans in float64 (T = 5 with chunk 2, so the chunked form pads)."""
    torch.manual_seed(14)
    block = Mamba2Block(d_model=4, d_state=3, headdim=2, expand=1, chunk_size=2).double()
    Bsz, T, H, P, N = 1, 5, block.n_heads, block.headdim, block.d_state
    x_h = torch.randn(Bsz, T, H, P, dtype=torch.float64, requires_grad=True)
    alpha = torch.rand(Bsz, T, H, dtype=torch.float64) * 0.9 + 0.05
    bx = torch.randn(Bsz, T, H, N, dtype=torch.float64, requires_grad=True)
    c_h = torch.randn(Bsz, T, H, N, dtype=torch.float64, requires_grad=True)
    if form == "sequential":
        inputs = (x_h, alpha.requires_grad_(True), bx, c_h)
        assert _gradcheck_fn(block.sequential_scan, inputs)
    else:
        inputs = (x_h, alpha.requires_grad_(True), bx, c_h)
        assert _gradcheck_fn(block.chunked_scan, inputs)


def test_gradcheck_mamba3_scan() -> None:
    """Mamba-3 complex trapezoidal scan in float64: gradients w.r.t. every input and
    the complex-transition parameters a_real_log and theta."""
    torch.manual_seed(15)
    block = Mamba3Block(d_model=4, d_state=3, headdim=2, expand=1).double()
    Bsz, T, H, P, N = 1, 4, block.n_heads, block.headdim, block.d_state
    x_h = torch.randn(Bsz, T, H, P, dtype=torch.float64, requires_grad=True)
    delta = (torch.rand(Bsz, T, H, dtype=torch.float64) * 0.5 + 0.05).requires_grad_(True)
    b_h = torch.randn(Bsz, T, H, N, dtype=torch.float64, requires_grad=True)
    c_h = torch.randn(Bsz, T, H, N, dtype=torch.float64, requires_grad=True)
    lam = (torch.rand(Bsz, T, H, dtype=torch.float64) * 0.8 + 0.1).requires_grad_(True)
    inputs = (x_h, delta, b_h, c_h, lam)
    assert _gradcheck(block, "selective_scan", inputs, ["a_real_log", "theta"])


def test_mamba3_lambda_one_is_complex_exponential_euler() -> None:
    """lambda = 1 removes the beta term, leaving h_t = alpha_t h_{t-1} + delta_t B_t x_t
    with complex alpha_t (theta != 0). float64, 1e-10."""
    torch.manual_seed(11)
    block = Mamba3Block(d_model=16, d_state=6, headdim=8, expand=1).double()
    with torch.no_grad():
        block.theta.uniform_(0.5, 3.0)
    Bsz, T, H, P, N = 2, 17, block.n_heads, block.headdim, block.d_state
    x_h = torch.randn(Bsz, T, H, P, dtype=torch.float64)
    delta = torch.rand(Bsz, T, H, dtype=torch.float64) * 0.5 + 0.01
    b_h = torch.randn(Bsz, T, H, N, dtype=torch.float64)
    c_h = torch.randn(Bsz, T, H, N, dtype=torch.float64)
    lam = torch.ones(Bsz, T, H, dtype=torch.float64)
    with torch.no_grad():
        y = block.selective_scan(x_h, delta, b_h, c_h, lam)
        a = torch.complex(-torch.exp(block.a_real_log), block.theta)  # (H, N)
        state = torch.zeros(Bsz, H, N, P, dtype=torch.complex128)
        ys = []
        for t in range(T):
            alpha = torch.exp(delta[:, t, :, None] * a)  # (B, H, N)
            inject = delta[:, t, :, None, None] * b_h[:, t, :, :, None] * x_h[:, t, :, None, :]
            state = alpha[..., None] * state + inject
            ys.append(torch.einsum("bhnp,bhn->bhp", state.real, c_h[:, t]))
    torch.testing.assert_close(y, torch.stack(ys, dim=1), **FP64)


@pytest.mark.parametrize(
    ("lam_value", "expected_ratio"),
    [pytest.param(0.5, 4.0, id="trapezoidal"), pytest.param(1.0, 2.0, id="exp-euler")],
)
def test_mamba3_discretization_order(lam_value: float, expected_ratio: float) -> None:
    """Integrate h' = a h + sin(t), h(0) = 0, on [0, 2] with the Mamba-3 recurrence
    (one head, one slot, B = C = 1, theta = 0, constant step). Halving the step must
    cut the end-point error about 4x for lambda = 1/2 (trapezoidal, second order)
    and about 2x for lambda = 1 (exponential Euler, first order). float64;
    accepted ratio window: expected value +/- 12%."""
    a, t_end = -0.5, 2.0
    exact = (-a * math.sin(t_end) - math.cos(t_end) + math.exp(a * t_end)) / (1.0 + a * a)
    block = Mamba3Block(d_model=1, d_state=1, headdim=1, expand=1).double()
    with torch.no_grad():
        block.a_real_log.fill_(math.log(-a))
        block.theta.zero_()

    def end_error(step: float) -> float:
        n = round(t_end / step)
        t = torch.arange(n + 1, dtype=torch.float64) * step
        u = torch.sin(t).view(1, n + 1, 1, 1)
        ones = torch.ones(1, n + 1, 1, 1, dtype=torch.float64)
        delta = torch.full((1, n + 1, 1), step, dtype=torch.float64)
        lam = torch.full((1, n + 1, 1), lam_value, dtype=torch.float64)
        with torch.no_grad():
            y = block.selective_scan(u, delta, ones, ones, lam)
        return abs(y[0, -1, 0, 0].item() - exact)

    ratio = end_error(0.1) / end_error(0.05)
    assert expected_ratio * 0.88 < ratio < expected_ratio * 1.12, ratio


# ------------------------------------- Mamba-2 / Mamba-3 heads and groups (P1)

# (dtype, batch, length): float64 cases cover batch 1, length 1 and a length that
# is an exact multiple of the chunk size; the float32 case shows the fp32 model
# against the float64 oracle.
BLOCK_CASES = [
    pytest.param(torch.float64, 2, 21, id="f64-B2-T21"),
    pytest.param(torch.float64, 1, 1, id="f64-B1-T1"),
    pytest.param(torch.float64, 1, 16, id="f64-B1-T16"),
    pytest.param(torch.float32, 2, 21, id="f32-B2-T21"),
]


def _mixer_inputs(
    block: Mamba2Block | Mamba3Block, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared Mamba-2/3 front end: norm, in_proj, causal depthwise conv, silu."""
    T = x.shape[1]
    xz = block.in_proj(block.norm(x))
    x_in, z = xz.chunk(2, dim=-1)
    x_conv = F.pad(x_in.transpose(1, 2), (block.d_conv - 1, 0))
    x_conv = F.silu(block.conv1d(x_conv)[..., :T].transpose(1, 2))
    return x_conv, z


def mamba2_block_reference(blk: Mamba2Block, x: torch.Tensor) -> torch.Tensor:
    """Naive Mamba-2 block: one head and one time step at a time.

    Head h covers channels [h*P, (h+1)*P) and reads the full N-vectors B_t and
    C_t of group h // (H / G), as in the reference SSD implementation.
    """
    Bsz, T, _ = x.shape
    H, P, N, G = blk.n_heads, blk.headdim, blk.d_state, blk.n_groups
    x_conv, z = _mixer_inputs(blk, x)
    dt, b, c = torch.split(blk.x_proj(x_conv), [blk.dt_rank, G * N, G * N], dim=-1)
    b = b.reshape(Bsz, T, G, N)
    c = c.reshape(Bsz, T, G, N)
    delta = F.softplus(blk.dt_proj(dt))  # (B, T, H)
    a = -torch.exp(blk.A_log)  # (H,)
    heads = []
    for h in range(H):
        g = h // (H // G)
        x_h = x_conv[..., h * P : (h + 1) * P]  # (B, T, P)
        state = x.new_zeros(Bsz, N, P)
        ys = []
        for t in range(T):
            decay = torch.exp(delta[:, t, h] * a[h])  # (B,)
            inject = delta[:, t, h, None, None] * b[:, t, g, :, None] * x_h[:, t, None, :]
            state = decay[:, None, None] * state + inject
            ys.append(torch.einsum("bnp,bn->bp", state, c[:, t, g]))
        heads.append(torch.stack(ys, dim=1))  # (B, T, P)
    y = torch.cat(heads, dim=-1) + x_conv * blk.D
    return x + blk.out_proj(y * F.silu(z))


def mamba3_scan_reference(
    a: torch.Tensor,
    x_h: torch.Tensor,
    delta: torch.Tensor,
    b_h: torch.Tensor,
    c_h: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Naive exponential-trapezoidal recurrence, one (head, state slot) at a time.

    h_t = alpha_t h_{t-1} + (1 - lam_t) delta_t alpha_t s_{t-1} + lam_t delta_t s_t
    with s_t = B_t x_t, alpha_t = exp(delta_t a) (complex), y_t = sum_n C_t[n] Re(h_t[n]).
    a: (H, N) complex; x_h: (B, T, H, P); delta, lam: (B, T, H); b_h, c_h: (B, T, H, N).
    """
    Bsz, T, H, P = x_h.shape
    N = a.shape[1]
    heads = []
    for h in range(H):
        slots = []
        for n in range(N):
            state = torch.zeros(Bsz, P, dtype=a.dtype)
            prev = torch.zeros(Bsz, P, dtype=a.dtype)
            ys = []
            for t in range(T):
                d, lm = delta[:, t, h], lam[:, t, h]
                alpha = torch.exp(d * a[h, n])  # (B,) complex
                s_t = (b_h[:, t, h, n, None] * x_h[:, t, h]).to(a.dtype)
                state = (
                    alpha[:, None] * state
                    + ((1.0 - lm) * d * alpha)[:, None] * prev
                    + (lm * d)[:, None] * s_t
                )
                prev = s_t
                ys.append(c_h[:, t, h, n, None] * state.real)
            slots.append(torch.stack(ys, dim=1))  # (B, T, P)
        heads.append(torch.stack(slots, dim=0).sum(dim=0))
    return torch.stack(heads, dim=2)  # (B, T, H, P)


def mamba3_block_reference(blk: Mamba3Block, x: torch.Tensor) -> torch.Tensor:
    """Naive Mamba-3 block; head h reads the B/C of group h // (H / G) by indexing."""
    Bsz, T, _ = x.shape
    H, P, N, G = blk.n_heads, blk.headdim, blk.d_state, blk.n_groups
    x_conv, z = _mixer_inputs(blk, x)
    dt, b, c, lam_raw = torch.split(blk.x_proj(x_conv), [blk.dt_rank, G * N, G * N, G], dim=-1)
    group_of_head = torch.arange(H) // (H // G)
    b_h = b.reshape(Bsz, T, G, N)[:, :, group_of_head]
    c_h = c.reshape(Bsz, T, G, N)[:, :, group_of_head]
    delta = F.softplus(blk.dt_proj(dt))
    lam = torch.sigmoid(blk.lambda_proj(lam_raw))
    a = torch.complex(-torch.exp(blk.a_real_log), blk.theta)
    y = mamba3_scan_reference(a, x_conv.reshape(Bsz, T, H, P), delta, b_h, c_h, lam)
    y = y.reshape(Bsz, T, H * P) + x_conv * blk.D
    return x + blk.out_proj(y * F.silu(z))


def _compare_param_grads(model: nn.Module, ref: nn.Module) -> None:
    pairs = zip(model.named_parameters(), ref.named_parameters(), strict=True)
    for (name, p), (ref_name, p_ref) in pairs:
        assert name == ref_name
        assert p.grad is not None and p_ref.grad is not None, name
        torch.testing.assert_close(p.grad, p_ref.grad, **FP64, msg=f"gradient of {name}")


@pytest.mark.parametrize("n_groups", [1, 2])
def test_mamba2_heads_read_their_group_bc_in_full(n_groups: int) -> None:
    """Every head h gets the full N-vector of group h // (H / G): C_h exactly, and
    the state input delta_h * B_g exactly (bitwise)."""
    torch.manual_seed(6)
    block = Mamba2Block(d_model=16, d_state=4, headdim=8, expand=2, n_groups=n_groups)
    H, N, G = block.n_heads, block.d_state, n_groups
    assert H == 4
    x = torch.randn(2, 9, block.d_inner)
    with torch.no_grad():
        _x_h, _decay, bx, c_h = block._scan_inputs(x)
        dt, b, c = torch.split(block.x_proj(x), [block.dt_rank, G * N, G * N], dim=-1)
        delta = F.softplus(block.dt_proj(dt))
    assert bx.shape == c_h.shape == (2, 9, H, N)
    b = b.reshape(2, 9, G, N)
    c = c.reshape(2, 9, G, N)
    for h in range(H):
        g = h // (H // G)
        torch.testing.assert_close(c_h[:, :, h], c[:, :, g], rtol=0, atol=0)
        torch.testing.assert_close(bx[:, :, h], b[:, :, g] * delta[:, :, h, None], rtol=0, atol=0)


@pytest.mark.parametrize("n_groups", [1, 2])
@pytest.mark.parametrize("scan_mode", ["sequential", "chunked"])
@pytest.mark.parametrize(("dtype", "Bsz", "T"), BLOCK_CASES)
def test_mamba2_block_matches_naive_reference(
    dtype: torch.dtype, Bsz: int, T: int, scan_mode: str, n_groups: int
) -> None:
    """Block forward (chunk 8, so T = 21 pads and T = 16 does not) against the naive
    per-head loop; in float64 every parameter gradient matches too. Tolerances: TOL[dtype]."""
    torch.manual_seed(7 + n_groups)
    block = Mamba2Block(
        d_model=16,
        d_state=4,
        headdim=8,
        expand=2,
        n_groups=n_groups,
        chunk_size=8,
        scan_mode=scan_mode,
    ).to(dtype)
    ref = copy.deepcopy(block).double()
    x = torch.randn(Bsz, T, 16, dtype=torch.float64)
    weight = torch.randn(Bsz, T, 16, dtype=torch.float64)
    y = block(x.to(dtype))
    y_ref = mamba2_block_reference(ref, x)
    assert y.shape == (Bsz, T, 16) and y.dtype == dtype
    torch.testing.assert_close(y.double(), y_ref, **TOL[dtype])
    if dtype == torch.float64:
        (y * weight).sum().backward()
        (y_ref * weight).sum().backward()
        _compare_param_grads(block, ref)


def test_mamba2_chunked_forward_equals_sequential_forward() -> None:
    """Same weights, scan_mode switched, through forward() (H = 4, T = 37, chunk 8).
    float32, rtol = atol = 1e-5."""
    torch.manual_seed(8)
    chunked = Mamba2Block(d_model=16, d_state=4, headdim=8, expand=2, chunk_size=8)
    sequential = copy.deepcopy(chunked)
    sequential.scan_mode = "sequential"
    x = torch.randn(3, 37, 16)
    with torch.no_grad():
        torch.testing.assert_close(chunked(x), sequential(x), rtol=1e-5, atol=1e-5)


def _randomize_mamba3(block: Mamba3Block) -> None:
    """Move lambda off its constant 0.5 init and perturb decay and rotation."""
    with torch.no_grad():
        block.lambda_proj.weight.normal_(0.0, 1.0)
        block.lambda_proj.bias.uniform_(-2.0, 2.0)
        block.a_real_log.uniform_(-1.0, 1.0)
        block.theta.add_(torch.randn_like(block.theta) * 0.5)


@pytest.mark.parametrize("n_groups", [1, 2])
@pytest.mark.parametrize(("dtype", "Bsz", "T"), BLOCK_CASES)
def test_mamba3_block_matches_naive_reference(
    dtype: torch.dtype, Bsz: int, T: int, n_groups: int
) -> None:
    """Block forward with random theta != 0 and lambda in (0, 1) against the naive
    complex128 loop; in float64 every parameter gradient matches too. TOL[dtype]."""
    torch.manual_seed(10 + n_groups)
    block = Mamba3Block(d_model=16, d_state=4, headdim=8, expand=2, n_groups=n_groups)
    _randomize_mamba3(block)
    block = block.to(dtype)
    ref = copy.deepcopy(block).double()
    x = torch.randn(Bsz, T, 16, dtype=torch.float64)
    weight = torch.randn(Bsz, T, 16, dtype=torch.float64)
    y = block(x.to(dtype))
    y_ref = mamba3_block_reference(ref, x)
    assert y.shape == (Bsz, T, 16) and y.dtype == dtype
    torch.testing.assert_close(y.double(), y_ref, **TOL[dtype])
    if dtype == torch.float64:
        (y * weight).sum().backward()
        (y_ref * weight).sum().backward()
        _compare_param_grads(block, ref)
