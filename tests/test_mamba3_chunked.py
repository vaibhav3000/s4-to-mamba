"""Chunked (parallel) Mamba-3 scan against the sequential recurrence.

The sequential scan is the oracle: tests/test_references.py checks it against a naive
per-(head, state slot) complex128 loop. Derivation of the chunked form:
docs/design/step2.md section 3.4. Tolerances: float64 rtol = atol = 1e-10 for values
and gradients; gradcheck eps 1e-6, atol 1e-7, rtol 1e-5; float32 against float64 at
1e-4 of the output scale.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ssmbench.models.mamba3_minimal import Mamba3Backbone, Mamba3Block

RTOL = ATOL = 1e-10


def _block(n_groups: int = 1, chunk_size: int = 4, rotation: bool = True) -> Mamba3Block:
    torch.manual_seed(3 + n_groups)
    block = Mamba3Block(
        d_model=8,
        d_state=3,
        headdim=4,
        expand=2,
        n_groups=n_groups,
        chunk_size=chunk_size,
        rotation=rotation,
    ).double()
    with torch.no_grad():
        block.a_real_log.uniform_(-1.0, 1.0)
        if rotation:
            block.theta.add_(torch.randn_like(block.theta))
    return block


def _scan_inputs(
    block: Mamba3Block, Bsz: int, T: int, lam_mode: str, seed: int = 0, delta_scale: float = 0.5
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    H, P, N = block.n_heads, block.headdim, block.d_state
    f64 = torch.float64
    x_h = torch.randn(Bsz, T, H, P, generator=g, dtype=f64)
    delta = torch.rand(Bsz, T, H, generator=g, dtype=f64) * delta_scale + 0.01
    b_h = torch.randn(Bsz, T, H, N, generator=g, dtype=f64)
    c_h = torch.randn(Bsz, T, H, N, generator=g, dtype=f64)
    lam = {
        "zero": torch.zeros(Bsz, T, H, dtype=f64),
        "half": torch.full((Bsz, T, H), 0.5, dtype=f64),
        "one": torch.ones(Bsz, T, H, dtype=f64),
        "random": torch.rand(Bsz, T, H, generator=g, dtype=f64),
    }[lam_mode]
    return x_h, delta, b_h, c_h, lam


@pytest.mark.parametrize("T", [1, 3, 4, 5, 23])
@pytest.mark.parametrize("lam_mode", ["zero", "half", "one", "random"])
@pytest.mark.parametrize("rotation", [True, False], ids=["rotating", "real"])
@pytest.mark.parametrize("n_groups", [1, 2])
def test_chunked_scan_matches_sequential(
    T: int, lam_mode: str, rotation: bool, n_groups: int
) -> None:
    """Chunk 4: T = 1 and 3 fit in one padded chunk, 4 is exact, 5 and 23 carry state
    (and the boundary input) across chunks. float64, 1e-10."""
    block = _block(n_groups, chunk_size=4, rotation=rotation)
    inputs = _scan_inputs(block, 3, T, lam_mode)
    with torch.no_grad():
        y_seq, h_seq, s_seq = block.selective_scan_with_state(*inputs)
        y_chk, h_chk, s_chk = block.chunked_scan_with_state(*inputs)
    assert y_chk.shape == (3, T, block.n_heads, block.headdim) and y_chk.dtype == torch.float64
    torch.testing.assert_close(y_chk, y_seq, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(h_chk, h_seq, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s_chk, s_seq, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("rotation", [True, False], ids=["rotating", "real"])
def test_chunked_scan_gradients_match_sequential(rotation: bool) -> None:
    """Gradients w.r.t. every scan input and both transition parameters, T = 11 over
    chunks of 4, random lambda. float64, 1e-10."""
    block = _block(n_groups=2, chunk_size=4, rotation=rotation)
    ref = copy.deepcopy(block)
    inputs = [t.requires_grad_(True) for t in _scan_inputs(block, 2, 11, "random")]
    ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
    weight = torch.randn(2, 11, block.n_heads, block.headdim, dtype=torch.float64)
    (block.chunked_scan(*inputs) * weight).sum().backward()
    (ref.selective_scan(*ref_inputs) * weight).sum().backward()
    for got, want in zip(inputs, ref_inputs, strict=True):
        assert got.grad is not None and want.grad is not None
        torch.testing.assert_close(got.grad, want.grad, rtol=RTOL, atol=ATOL)
    for (name, p), (_, p_ref) in zip(block.named_parameters(), ref.named_parameters(), strict=True):
        if p_ref.grad is None:
            assert p.grad is None, name
            continue
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, p_ref.grad, rtol=RTOL, atol=ATOL, msg=name)


class _ChunkedScan(nn.Module):
    """Expose block.chunked_scan as forward() so functional_call can swap parameters."""

    def __init__(self, inner: Mamba3Block) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.inner.chunked_scan(*inputs)


def test_gradcheck_chunked_scan() -> None:
    """float64 gradcheck through every input and the transition parameters (T = 7 over
    chunks of 3, so two chunk boundaries and a padded tail)."""
    torch.manual_seed(15)
    block = Mamba3Block(d_model=4, d_state=2, headdim=2, expand=1, chunk_size=3).double()
    with torch.no_grad():
        block.theta.add_(torch.randn_like(block.theta))
    x_h, delta, b_h, c_h, _ = _scan_inputs(block, 1, 7, "half", seed=4)
    lam = torch.rand(1, 7, block.n_heads, dtype=torch.float64) * 0.8 + 0.1
    a_real_log = block.a_real_log.detach().clone()
    theta = block.theta.detach().clone()
    wrapper = _ChunkedScan(block)

    def fn(*tensors: torch.Tensor) -> torch.Tensor:
        params = {"inner.a_real_log": tensors[5], "inner.theta": tensors[6]}
        return torch.func.functional_call(wrapper, params, tuple(tensors[:5]))

    inputs = tuple(t.requires_grad_(True) for t in (x_h, delta, b_h, c_h, lam, a_real_log, theta))
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64])
def test_chunked_scan_is_chunk_size_invariant(chunk_size: int) -> None:
    """Any chunk size gives the sequential result (T = 37). float64, 1e-10."""
    block = _block(n_groups=1, chunk_size=chunk_size)
    inputs = _scan_inputs(block, 2, 37, "random", seed=9)
    with torch.no_grad():
        torch.testing.assert_close(
            block.chunked_scan(*inputs), block.selective_scan(*inputs), rtol=RTOL, atol=ATOL
        )


def test_chunk_groups_do_not_change_the_result() -> None:
    """A byte budget of one byte forces one chunk per group; the output is unchanged."""
    block = _block(n_groups=2, chunk_size=4)
    inputs = _scan_inputs(block, 2, 29, "random", seed=5)
    with torch.no_grad():
        wide = block.chunked_scan(*inputs)
        block.chunk_group_bytes = 1
        narrow = block.chunked_scan(*inputs)
    torch.testing.assert_close(narrow, wide, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(("delta_scale", "label"), [(1e-6, "tiny"), (60.0, "large")])
def test_chunked_scan_handles_extreme_decays(delta_scale: float, label: str) -> None:
    """Tiny steps (decay ~ 1, no rotation progress) and large ones (decay underflows
    within a chunk, |delta * Re(A)| up to about 160) stay finite and match the
    recurrence. float64, 1e-10."""
    block = _block(n_groups=1, chunk_size=8)
    inputs = _scan_inputs(block, 2, 19, "random", seed=7, delta_scale=delta_scale)
    with torch.no_grad():
        y = block.chunked_scan(*inputs)
        y_ref = block.selective_scan(*inputs)
    assert torch.isfinite(y).all(), label
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)


def test_float32_chunked_scan_tracks_float64() -> None:
    """float32 chunked scan against the float64 recurrence at T = 300 (chunk 64, init-scale
    steps): max error within 1e-4 of the output scale."""
    block = _block(n_groups=1, chunk_size=64)
    inputs = _scan_inputs(block, 2, 300, "random", seed=2, delta_scale=0.1)
    with torch.no_grad():
        y_ref = block.selective_scan(*inputs)
        block32 = copy.deepcopy(block).float()
        y32 = block32.chunked_scan(*(t.float() for t in inputs))
    assert y32.dtype == torch.float32
    scale = y_ref.abs().max().item()
    assert (y32.double() - y_ref).abs().max().item() <= 1e-4 * scale


def test_rotation_ablation_has_a_real_decay_only_transition() -> None:
    """rotation=False: theta is a zero buffer, not a trainable parameter, and the scan
    equals a rotating block whose theta is set to zero."""
    ablated = _block(rotation=False)
    names = {n for n, _ in ablated.named_parameters()}
    assert "theta" not in names
    assert torch.equal(ablated.theta, torch.zeros_like(ablated.theta))
    rotating = _block(rotation=True)
    rotating.load_state_dict(ablated.state_dict(), strict=False)
    with torch.no_grad():
        rotating.theta.zero_()
    inputs = _scan_inputs(ablated, 2, 9, "random")
    with torch.no_grad():
        torch.testing.assert_close(
            ablated.chunked_scan(*inputs), rotating.chunked_scan(*inputs), rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("n_groups", [1, 2])
def test_block_forward_chunked_equals_sequential(n_groups: int) -> None:
    """Same weights through forward() with scan_mode switched (T = 21, chunk 8), values
    and every parameter gradient. float64, 1e-10."""
    torch.manual_seed(12)
    chunked = Mamba3Block(
        d_model=16, d_state=4, headdim=8, expand=2, n_groups=n_groups, chunk_size=8
    ).double()
    sequential = copy.deepcopy(chunked)
    sequential.scan_mode = "sequential"
    assert chunked.scan_mode == "chunked"  # the default
    x = torch.randn(2, 21, 16, dtype=torch.float64)
    weight = torch.randn(2, 21, 16, dtype=torch.float64)
    (chunked(x) * weight).sum().backward()
    (sequential(x) * weight).sum().backward()
    for (name, p), (_, q) in zip(
        chunked.named_parameters(), sequential.named_parameters(), strict=True
    ):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=RTOL, atol=ATOL, msg=name)


def test_backbone_passes_scan_settings_to_its_blocks() -> None:
    backbone = Mamba3Backbone(
        d_model=8,
        n_layers=2,
        d_state=2,
        headdim=4,
        chunk_size=16,
        scan_mode="sequential",
        rotation=False,
    )
    for block in backbone.blocks:
        assert isinstance(block, Mamba3Block)
        assert block.chunk_size == 16 and block.scan_mode == "sequential"
        assert "theta" not in dict(block.named_parameters())


def test_invalid_scan_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="scan_mode"):
        Mamba3Block(d_model=8, d_state=2, headdim=4, scan_mode="parallel")


def test_chunked_scan_ignores_bf16_autocast() -> None:
    """Under CPU bf16 autocast the chunked scan still computes in float32: bitwise equal
    to the same call without autocast."""
    block = _block(n_groups=1, chunk_size=4).float()
    inputs = tuple(t.float() for t in _scan_inputs(block, 2, 13, "random"))
    with torch.no_grad():
        plain = block.chunked_scan(*inputs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            under_autocast = block.chunked_scan(*inputs)
    assert under_autocast.dtype == torch.float32
    assert torch.equal(under_autocast, plain)
