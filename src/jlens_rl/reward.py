from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from jlens import JacobianLens
from torch.nn.utils.rnn import pad_sequence


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
        mask_target_tokens: bool = False,
    ) -> None:
        self.lens = JacobianLens.load(lens_path)
        self.target_words = list(target_words)
        self.token_ids = single_token_ids(tokenizer, target_words)
        self.stride = stride
        self.mask_target_tokens = mask_target_tokens
        calibration = json.loads(Path(calibration_path).read_text())
        self.mean = float(calibration["mean"])
        self.std = max(float(calibration["std"]), 1e-6)
        self._device_jacobians: dict[tuple[int, str], torch.Tensor] = {}

    @torch.no_grad()
    def raw_scores(
        self, model: Any, hidden_states: Sequence[torch.Tensor], prompt_len: int,
        attention_mask: torch.Tensor, batch_index: int = 0,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm, lm_head = decoder_parts(model)
        end = int(attention_mask.sum().item())
        positions = list(range(prompt_len + self.stride - 1, end, self.stride))
        if self.mask_target_tokens and input_ids is not None:
            targets = set(self.token_ids)
            positions = [p for p in positions if int(input_ids[p]) not in targets]
        if not positions and end > prompt_len:
            positions = [end - 1]
        if not positions:
            return torch.zeros(1, device=attention_mask.device)
        per_layer = []
        token_ids = torch.tensor(self.token_ids, device=lm_head.weight.device)
        for layer in self.lens.source_layers:
            # HF hidden_states[0] is embeddings; block L output is [L + 1].
            h = hidden_states[layer + 1][batch_index, positions].float()
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
        attention_mask: torch.Tensor, batch_index: int = 0,
        input_ids: torch.Tensor | None = None,
    ) -> float:
        raw = self.raw_scores(
            model, hidden_states, prompt_len, attention_mask, batch_index, input_ids
        ).mean()
        return float(((raw - self.mean) / self.std).clamp(-5, 5).item())


class TRLTargetJLReward:
    """Callable adapter that computes J-rewards inside the vendored TRL trainer."""

    def __init__(self, scorer: TargetJLReward) -> None:
        self.scorer = scorer
        self.label = "_".join(scorer.target_words)
        self.__name__ = f"jlens_{self.label}_reward"

    @torch.no_grad()
    def __call__(
        self,
        trainer_model: Any,
        prompt_ids: list[list[int]],
        completion_ids: list[list[int]],
        log_metric: Any,
        **kwargs,
    ) -> list[float]:
        was_training = trainer_model.training
        trainer_model.eval()
        sequences = [
            torch.tensor(prompt + completion, device=trainer_model.device)
            for prompt, completion in zip(prompt_ids, completion_ids, strict=True)
        ]
        pad_token_id = trainer_model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = trainer_model.config.eos_token_id
        ids = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
        mask = pad_sequence(
            [torch.ones_like(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=0,
        )
        output = trainer_model(
            ids, attention_mask=mask, output_hidden_states=True, use_cache=False
        )
        rewards = [
            self.scorer(
                trainer_model, output.hidden_states, len(prompt), mask[index], index,
                ids[index],
            )
            for index, prompt in enumerate(prompt_ids)
        ]
        trainer_model.train(was_training)
        literal_rate = sum(
            any(token in self.scorer.token_ids for token in completion)
            for completion in completion_ids
        ) / len(completion_ids)
        log_metric(f"jlens/{self.label}_mean", sum(rewards) / len(rewards))
        log_metric(f"jlens/{self.label}_literal_rate", literal_rate)
        return rewards
