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
