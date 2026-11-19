"""Inference benchmark bookkeeping and correctness on CPU (no timing claims)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from ssmbench.benchmarks import inference
from ssmbench.models import build_backbone

TINY: dict[str, dict] = {
    "transformer": {"d_model": 16, "n_layers": 2, "n_heads": 2},
    "transformer_rope": {"d_model": 16, "n_layers": 2, "n_heads": 2, "pos_encoding": "rope"},
    "s4d": {"d_model": 16, "n_layers": 2, "d_state": 4},
    "mamba": {"d_model": 16, "n_layers": 2, "d_state": 4},
    "mamba2": {"d_model": 16, "n_layers": 2, "d_state": 4, "headdim": 8, "chunk_size": 4},
    "mamba3": {"d_model": 16, "n_layers": 2, "d_state": 4, "headdim": 8, "chunk_size": 4},
}


def _arch(label: str) -> str:
    return "transformer" if label.startswith("transformer") else label


def _lm(label: str) -> tuple[nn.Module, nn.Module, nn.Module]:
    torch.manual_seed(0)
    backbone = build_backbone(_arch(label), TINY[label]).double().eval()
    embed = nn.Embedding(inference.VOCAB, 16).double().eval()
    head = nn.Linear(16, inference.VOCAB).double().eval()
    return backbone, embed, head


@pytest.mark.parametrize("label", list(TINY))
def test_timed_generation_equals_greedy_full_forward_decoding(label: str) -> None:
    """The prefill-then-step session generates exactly the tokens that greedy decoding by
    re-running forward() over the whole growing sequence generates (float64)."""
    backbone, embed, head = _lm(label)
    prompt = torch.randint(1, inference.VOCAB, (2, 9), generator=torch.Generator().manual_seed(1))
    session = inference.generate_session(
        backbone, embed, head, prompt, new_tokens=6, device=torch.device("cpu"), precision="fp32"
    )
    tokens = prompt
    with torch.no_grad():
        for _ in range(7):  # the first token comes from the prompt, then 6 steps
            nxt = head(backbone(embed(tokens))[:, -1:]).argmax(-1)
            tokens = torch.cat([tokens, nxt], dim=1)
    assert torch.equal(session.tokens, tokens[:, 9:])
    assert len(session.step_ms) == 6 and session.ttft_ms > 0


def test_point_records_ttft_decode_latency_and_state_size() -> None:
    entry = inference.measure_point(
        "mamba2",
        prompt_len=12,
        new_tokens=5,
        batch_size=1,
        repeats=3,
        warmup=1,
        device=torch.device("cpu"),
        model_kwargs=TINY["mamba2"],
        precision="fp32",
    )
    assert entry["ttft_ms_n"] == 3 and entry["decode_ms_per_token_n"] == 15
    assert entry["decode_ms_per_token_p90"] >= entry["decode_ms_per_token_median"] > 0
    assert entry["tokens_per_s"] > 0 and entry["decode_state_bytes"] > 0
    assert entry["attn_backend_prefill"] is None


@pytest.mark.parametrize(
    ("label", "grows"), [("transformer", True), ("mamba3", False), ("s4d", False)]
)
def test_decode_state_grows_with_context_only_for_attention(label: str, grows: bool) -> None:
    sizes = [
        inference.measure_point(
            _arch(label), L, 2, 1, 1, 0, torch.device("cpu"), TINY[label], "fp32"
        )["decode_state_bytes"]
        for L in (8, 32)
    ]
    assert (sizes[1] > sizes[0]) is grows


def test_transformer_points_record_the_kernels_of_prefill_and_step() -> None:
    entry = inference.measure_point(
        "transformer",
        8,
        2,
        1,
        1,
        0,
        torch.device("cpu"),
        {**TINY["transformer"], "attn_backend": "math"},
        "fp32",
    )
    assert entry["attn_backend_prefill"] == ["math"] and entry["attn_backend_step"] == ["math"]


def _cfg(tmp_path: Path) -> dict:
    return {
        "device": "cpu",
        "out_path": str(tmp_path / "decode.json"),
        "precision": "fp32",
        "batch_size": 1,
        "repeats": 2,
        "warmup": 0,
        "new_tokens": 3,
        "seq_lengths": [8, 16],
        "models": ["s4d", "mamba_small"],
        "variants": {"s4d": TINY["s4d"], "mamba_small": {"backbone": "mamba", **TINY["mamba"]}},
        "model_length_limits": {"mamba_small": 8},
    }


def test_sweep_writes_labelled_entries_with_provenance_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = inference.run_inference(_cfg(tmp_path))
    points = [(e["model"], e["seq_len"]) for e in results["entries"]]
    assert points == [("s4d", 8), ("s4d", 16), ("mamba_small", 8)]
    saved = json.loads(Path(_cfg(tmp_path)["out_path"]).read_text(encoding="utf-8"))
    assert all("source_sha256" in e["provenance"] for e in saved["entries"])
    assert saved["entries"][2]["architecture"] == "mamba"

    calls: list[str] = []
    monkeypatch.setattr(inference, "measure_point", lambda *a, **k: calls.append("x") or {})
    inference.run_inference(_cfg(tmp_path))
    assert calls == []
