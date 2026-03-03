from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from modules.model_rpr import RPRClassifier, strip_data_parallel_prefix


def test_strip_data_parallel_prefix():
    state = {
        "module.linear.weight": torch.randn(3, 3),
        "module.linear.bias": torch.randn(3),
    }
    converted = strip_data_parallel_prefix(state)
    assert "linear.weight" in converted
    assert "linear.bias" in converted


def test_model_forward_shape():
    model = RPRClassifier(window_size=50, channel_size=150, stride=10)
    x = torch.randn(2, 3000, 3)
    logits, attn = model(x)
    assert logits.shape == (2, 3)
    assert len(attn) == 3
