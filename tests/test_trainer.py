"""Trainer pieces that do not need a dataset download or a GPU."""

from __future__ import annotations

import pytest
import torch

from ssmbench.models import build_backbone
from ssmbench.models.blocks import TokenLevelClassifier
from ssmbench.train.trainer import build_param_groups

SMALL_KWARGS: dict[str, dict[str, int]] = {
    "transformer": {"d_model": 32, "n_layers": 2, "n_heads": 4, "max_len": 64},
    "s4d": {"d_model": 32, "n_layers": 2, "d_state": 8},
    "mamba": {"d_model": 32, "n_layers": 2, "d_state": 8},
    "mamba2": {"d_model": 32, "n_layers": 2, "d_state": 8, "headdim": 16},
    "mamba3": {"d_model": 32, "n_layers": 2, "d_state": 8, "headdim": 16},
}

# Parameter leaf names (per block) that must receive weight decay. Everything else,
# in particular every SSM dynamics parameter, must not.
DECAYED_LEAVES: dict[str, set[str]] = {
    "transformer": {"attn.qkv.weight", "attn.proj.weight", "mlp.0.weight", "mlp.3.weight"},
    # B and C are ordinary weights, decayed symmetrically as in the reference code.
    "s4d": {"layer.B_re", "layer.B_im", "layer.C_re", "layer.C_im", "gate.weight"},
    "mamba": {
        "in_proj.weight",
        "conv1d.weight",
        "x_proj.weight",
        "dt_proj.weight",
        "out_proj.weight",
    },
    "mamba2": {
        "in_proj.weight",
        "conv1d.weight",
        "x_proj.weight",
        "dt_proj.weight",
        "out_proj.weight",
    },
    "mamba3": {
        "in_proj.weight",
        "conv1d.weight",
        "x_proj.weight",
        "dt_proj.weight",
        "lambda_proj.weight",
        "out_proj.weight",
    },
}


@pytest.mark.parametrize("name", list(SMALL_KWARGS))
def test_param_groups_exempt_ssm_dynamics_from_weight_decay(name: str) -> None:
    """Weight decay reaches only the expected matrices (plus the shared embedding and
    head weights). The dynamics parameters are exempt: S4D log_A_real, A_imag and
    dt_param, Mamba A_log and D, Mamba-3 a_real_log, theta and D. B and C are decayed.
    Every parameter lands in exactly one group."""
    torch.manual_seed(0)
    model = TokenLevelClassifier(build_backbone(name, SMALL_KWARGS[name]), vocab_size=8, d_model=32)
    decay_group, no_decay_group = build_param_groups(model, weight_decay=0.05)
    assert decay_group["weight_decay"] == 0.05 and no_decay_group["weight_decay"] == 0.0

    names = {id(p): n for n, p in model.named_parameters()}
    decayed = {names[id(p)] for p in decay_group["params"]}
    exempt = {names[id(p)] for p in no_decay_group["params"]}
    assert decayed.isdisjoint(exempt)
    assert decayed | exempt == set(names.values())

    expected = {"embed.weight", "head.weight"}
    for i in range(SMALL_KWARGS[name]["n_layers"]):
        expected |= {f"backbone.blocks.{i}.{leaf}" for leaf in DECAYED_LEAVES[name]}
    assert decayed == expected
