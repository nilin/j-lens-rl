from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from jlens import JacobianLens


def single_token_ids(tokenizer: Any, words: Sequence[str]) -> list[int]:
    ids: set[int] = set()
    variants = {v for word in words for v in (word, " " + word, word.capitalize())}
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    if not ids:
        raise ValueError(f"none of {sorted(variants)} is a single tokenizer token")
    return sorted(ids)


def decoder_parts(model: Any) -> tuple[Any, Any]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = base.model
    return decoder.norm, base.lm_head


class TargetJLReward:
    """Efficient selected-token J-lens readout; never projects full vocabulary."""

    def __init__(
        self,
        lens_path: str,
        calibration_path: str,
        tokenizer: Any,
        target_words: Sequence[str],
        stride: int = 20,
    ) -> None:
        self.lens = JacobianLens.load(lens_path)
        self.token_ids = single_token_ids(tokenizer, target_words)
        self.stride = stride
        calibration = json.loads(Path(calibration_path).read_text())
        self.mean = float(calibration["mean"])
        self.std = max(float(calibration["std"]), 1e-6)
        self._device_jacobians: dict[tuple[int, str], torch.Tensor] = {}

    @torch.no_grad()
    def raw_scores(
        self, model: Any, hidden_states: Sequence[torch.Tensor], prompt_len: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        norm, lm_head = decoder_parts(model)
        end = int(attention_mask.sum().item())
        positions = list(range(prompt_len + self.stride - 1, end, self.stride))
        if not positions and end > prompt_len:
            positions = [end - 1]
        if not positions:
            return torch.zeros(1, device=attention_mask.device)
        per_layer = []
        token_ids = torch.tensor(self.token_ids, device=lm_head.weight.device)
        for layer in self.lens.source_layers:
            # HF hidden_states[0] is embeddings; block L output is [L + 1].
            h = hidden_states[layer + 1][0, positions].float()
            key = (layer, str(h.device))
            if key not in self._device_jacobians:
                self._device_jacobians[key] = self.lens.jacobians[layer].to(h.device)
            transported = h @ self._device_jacobians[key].T
            normalized = norm(transported.to(norm.weight.dtype))
            weights = lm_head.weight.index_select(0, token_ids)
            logits = normalized @ weights.T
            per_layer.append(torch.logsumexp(logits.float(), dim=-1))
        return torch.stack(per_layer).flatten()

    def __call__(
        self, model: Any, hidden_states: Sequence[torch.Tensor], prompt_len: int,
        attention_mask: torch.Tensor,
    ) -> float:
        raw = self.raw_scores(model, hidden_states, prompt_len, attention_mask).mean()
        return float(((raw - self.mean) / self.std).clamp(-5, 5).item())
