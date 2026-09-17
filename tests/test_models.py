"""Correctness tests for the minimal SSM implementations.

The core claims of the repository are mathematical, so the tests verify
mathematics, not just shapes:

- S4D's parallel (FFT convolution) form must equal its recurrent step form.
- Mamba-2's chunked SSD must equal its sequential reference scan.
- Mamba-3 must reduce to the Mamba-1/2 exponential-Euler update when
  lambda = 1 and the transition frequency is zero (real A).
- Every backbone must be deterministic under a fixed seed and preserve shapes.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssmbench.models import build_backbone  # noqa: E402
from ssmbench.models.blocks import SequenceClassifier, masked_mean  # noqa: E402
from ssmbench.models.mamba2_minimal import Mamba2Block  # noqa: E402
from ssmbench.models.mamba3_minimal import Mamba3Block  # noqa: E402

torch.manual_seed(0)


# ---------------------------------------------------------------- s4d


def test_s4d_conv_equals_recurrent() -> None:
    """The FFT-convolution training path and the recurrent step path agree."""
    from ssmbench.models.s4_minimal import S4DLayer

    B, T, D, N = 2, 64, 8, 16
    layer = S4DLayer(D, d_state=N)
    x = torch.randn(B, T, D)

    # parallel form
    y_conv = layer(x)

    # recurrent form
    h = torch.zeros(B, D, N, dtype=torch.complex64)
    ys = []
    for t in range(T):
        y_t, h = layer.step(x[:, t], h)
        ys.append(y_t)
    y_rec = torch.stack(ys, dim=1)

    assert torch.allclose(y_conv, y_rec, atol=1e-3, rtol=1e-3), (
        (y_conv - y_rec).abs().max().item()
    )


# ---------------------------------------------------------------- mamba2


def test_mamba2_chunked_equals_sequential() -> None:
    """SSD chunked form == sequential reference (the central correctness claim)."""
    torch.manual_seed(1)
    B, T, H, P, N = 2, 96, 3, 8, 12
    for chunk_size in (32, 64):  # aligned and misaligned (96 = 32*3, 96/64 pads)
        block = Mamba2Block(
            d_model=24, d_state=N, headdim=P, expand=1, chunk_size=chunk_size
        )
        # expand=1 -> d_inner = 24 -> n_heads = 24 // 8 = 3 ✓
        x_h = torch.randn(B, T, H, P)
        alpha = torch.rand(B, T, H) * 0.9 + 0.05  # decays in (0.05, 0.95)
        B_h = torch.randn(B, T, H, N)
        C_h = torch.randn(B, T, H, N)
        y_seq = block.sequential_scan(x_h, alpha, B_h, C_h)
        y_chunk = block.chunked_scan(x_h, alpha, B_h, C_h)
        assert torch.allclose(y_seq, y_chunk, atol=1e-3, rtol=1e-3), (
            f"chunk_size={chunk_size}: max err {(y_seq - y_chunk).abs().max().item()}"
        )


def test_mamba2_block_forward_shapes() -> None:
    torch.manual_seed(2)
    block = Mamba2Block(d_model=32, d_state=16, headdim=8, expand=2, chunk_size=16)
    x = torch.randn(2, 50, 32)
    y = block(x)
    assert y.shape == x.shape
    # residual: with zeroed mixer weights the block is the identity
    for p in block.parameters():
        torch.nn.init.zeros_(p)
    y0 = block(x)
    assert torch.allclose(y0, x, atol=1e-5)


# ---------------------------------------------------------------- mamba3


def test_mamba3_reduces_to_exp_euler() -> None:
    """lambda=1, theta=0: the Mamba-3 update is Mamba-1/2's exponential-Euler."""
    torch.manual_seed(3)
    B, T, H, P, N = 2, 40, 2, 8, 10
    block = Mamba3Block(d_model=16, d_state=N, headdim=P, expand=1)
    assert block.n_heads == 2

    # pin theta = 0 (real transition) and lambda = 1 (pure right endpoint)
    with torch.no_grad():
        block.theta.zero_()
        block.lambda_proj.weight.zero_()
        block.lambda_proj.bias.fill_(10.0)  # sigmoid(10) ~ 1 => gamma = delta, beta ~ 0

    x_h = torch.randn(B, T, H, P)
    delta = torch.rand(B, T, H) * 0.1 + 1e-3
    B_h = torch.randn(B, T, H, N)
    C_h = torch.randn(B, T, H, N)
    lam = torch.ones(B, T, H)

    y_m3 = block.selective_scan(x_h, delta, B_h, C_h, lam)

    # reference: h_t = alpha h_{t-1} + delta B x (Mamba-1/2 recurrence)
    A = -torch.exp(block.a_real_log.float())
    h = torch.zeros(B, H, N, P)
    ys = []
    for t in range(T):
        alpha = torch.exp(delta[:, t].unsqueeze(-1) * A)  # (B, H, N)
        inject = B_h[:, t].unsqueeze(-1) * delta[:, t].unsqueeze(-1).unsqueeze(-1) * x_h[:, t].unsqueeze(2)
        h = alpha.unsqueeze(-1) * h + inject
        ys.append(torch.einsum("bhnp,bhn->bhp", h, C_h[:, t]))
    y_ref = torch.stack(ys, dim=1)

    assert torch.allclose(y_m3, y_ref, atol=1e-3, rtol=1e-3), (
        (y_m3 - y_ref).abs().max().item()
    )


def test_mamba3_alpha_is_contractive_with_rotation() -> None:
    """|alpha| < 1 (stability) and nonzero phase (rotation capability)."""
    block = Mamba3Block(d_model=16, d_state=8, headdim=8, expand=1)
    delta = torch.full((1,), 0.05)
    A = -torch.exp(block.a_real_log.float()) + 1j * block.theta.float()
    alpha = torch.exp(delta * A)
    assert (alpha.abs() < 1.0).all()
    assert (alpha.abs() > 0.0).all()
    assert (alpha.angle() != 0).any(), "rotation phase must be present"


# ---------------------------------------------------------------- mamba1


def test_mamba_step_equals_forward() -> None:
    """Block-level step() decode path equals the parallel forward path."""
    from ssmbench.models.mamba_minimal import MambaBlock

    torch.manual_seed(4)
    B, T, D = 2, 32, 16
    block = MambaBlock(d_model=D, d_state=8, d_conv=4, expand=2)
    x = torch.randn(B, T, D)
    y_full = block(x)

    state = block.init_state(B, x.device)
    ys = []
    for t in range(T):
        ys.append(block.step(x[:, t : t + 1], state))
    y_step = torch.cat(ys, dim=1)
    assert torch.allclose(y_full, y_step, atol=1e-3, rtol=1e-3), (
        (y_full - y_step).abs().max().item()
    )


# ---------------------------------------------------------------- generic


@pytest.mark.parametrize("name", ["transformer", "s4d", "mamba", "mamba2", "mamba3"])
def test_classifier_shapes_and_determinism(name: str) -> None:
    torch.manual_seed(5)
    kwargs = {
        "transformer": dict(d_model=32, n_layers=2, n_heads=4, max_len=128),
        "s4d": dict(d_model=32, n_layers=2, d_state=8),
        "mamba": dict(d_model=32, n_layers=2, d_state=8),
        "mamba2": dict(d_model=32, n_layers=2, d_state=8, headdim=16, chunk_size=16),
        "mamba3": dict(d_model=32, n_layers=2, d_state=8, headdim=16),
    }[name]
    backbone = build_backbone(name, kwargs)
    model = SequenceClassifier(backbone, vocab_size=20, d_model=32, padding_idx=0)
    tokens = torch.randint(0, 20, (4, 48))
    logits1 = model(tokens)
    logits2 = model(tokens)
    assert logits1.shape == (4, 2)
    assert torch.equal(logits1, logits2), "determinism under fixed seed"


def test_masked_mean_ignores_padding() -> None:
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])  # (1, 3, 2)
    mask = torch.tensor([[True, True, False]])
    pooled = masked_mean(h, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_selective_copy_dataset_contract() -> None:
    from ssmbench.data.synthetic import IGNORE_INDEX, SelectiveCopyDataset

    ds = SelectiveCopyDataset(num_samples=8, seq_len=64, min_pairs=3, max_pairs=5, seed=0)
    sample = ds[0]
    x, y = sample["input_ids"], sample["targets"]
    assert x.shape == (64,) and y.shape == (64,)
    answer = y != IGNORE_INDEX
    k = int(answer.sum())
    assert 3 <= k <= 5
    # answer block sits at the end and is padded in the input
    assert (x[answer] == 0).all(), "answer positions must be masked in the input"
    # every answer target is a valid content token (>= 3)
    assert (y[answer] >= 3).all()
    # marker -> value structure present in the prompt
    assert (x == 1).sum() == k and (x == 2).sum() == 1


def test_parity_dataset_targets_are_cumulative_xor() -> None:
    from ssmbench.data.synthetic import ParityDataset

    ds = ParityDataset(num_samples=4, seq_len=32, seed=0)
    s = ds[0]
    bits = s["input_ids"]
    expected = torch.cumsum(bits, 0) % 2
    assert torch.equal(s["targets"], expected)


def test_exact_copy_dataset_contract() -> None:
    """Repeat-After-Me formulation: input <bos> s <sep>, target s <eos>."""
    from ssmbench.data.synthetic import EC_BOS, EC_EOS, EC_SEP, ExactCopyDataset

    L, V = 12, 26
    ds = ExactCopyDataset(num_samples=8, string_len=L, seq_len=2 * 128 + 8,
                          alphabet_size=V, seed=0)
    s = ds[0]
    x, y = s["input_ids"], s["targets"]
    assert x.shape == (2 * 128 + 8,) and y.shape == x.shape
    # prompt structure
    assert x[0].item() == EC_BOS
    assert x[1 + L].item() == EC_SEP
    # the answer (targets after sep) must reproduce the prompt string, then eos
    prompt = x[1 : 1 + L]
    answer = y[2 + L : 2 + 2 * L]
    assert torch.equal(prompt, answer)
    assert y[2 + 2 * L].item() == EC_EOS
    # everything else is ignored
    mask = y.ne(-100)
    assert int(mask.sum()) == L + 1
    # prompt letters are in the alphabet range
    assert (prompt >= 4).all() and (prompt < 4 + V).all()


def test_exact_copy_too_short_raises() -> None:
    from ssmbench.data.synthetic import ExactCopyDataset

    with pytest.raises(ValueError):
        ExactCopyDataset(num_samples=1, string_len=32, seq_len=50)
