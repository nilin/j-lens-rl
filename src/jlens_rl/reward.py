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


def target_log_probs(
    hidden: torch.Tensor, lm_head: Any, token_ids: Sequence[int], chunk_size: int = 16384
) -> torch.Tensor:
    """Log probability mass assigned to ``token_ids`` without materializing full-vocabulary logits."""
    ids = torch.as_tensor(token_ids, device=lm_head.weight.device)
    target_logits = hidden @ lm_head.weight.index_select(0, ids).T
    target_logsumexp = torch.logsumexp(target_logits.float(), dim=-1)
    denominator = None
    for start in range(0, lm_head.weight.shape[0], chunk_size):
        logits = hidden @ lm_head.weight[start : start + chunk_size].T
        chunk_logsumexp = torch.logsumexp(logits.float(), dim=-1)
        denominator = (
            chunk_logsumexp
            if denominator is None
            else torch.logaddexp(denominator, chunk_logsumexp)
        )
    return target_logsumexp - denominator


class TargetJLReward:
    """J-lens target log-probability readout with chunked vocabulary normalization."""

    def __init__(
        self,
        lens_path: str,
        calibration_path: str,
        tokenizer: Any,
        target_words: Sequence[str],
        stride: int = 20,
        mask_target_tokens: bool = False,
        vocab_chunk_size: int = 16384,
        score_start_fraction: float = 0.0,
        score_layers: Sequence[int] | None = None,
    ) -> None:
        self.lens = JacobianLens.load(lens_path)
        self.target_words = list(target_words)
        self.token_ids = single_token_ids(tokenizer, target_words)
        self.stride = stride
        self.mask_target_tokens = mask_target_tokens
        self.vocab_chunk_size = vocab_chunk_size
        if not 0.0 <= score_start_fraction < 1.0:
            raise ValueError("score_start_fraction must be in [0, 1)")
        self.score_start_fraction = score_start_fraction
        available_layers = list(self.lens.source_layers)
        self.score_layers = list(score_layers) if score_layers is not None else available_layers
        missing_layers = set(self.score_layers) - set(available_layers)
        if missing_layers:
            raise ValueError(
                f"score layers {sorted(missing_layers)} are not present in lens layers {available_layers}"
            )
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
        response_len = max(0, end - prompt_len)
        window_start = prompt_len + int(response_len * self.score_start_fraction)
        first_position = max(prompt_len + self.stride - 1, window_start)
        positions = list(range(first_position, end, self.stride))
        if self.mask_target_tokens and input_ids is not None:
            targets = set(self.token_ids)
            positions = [p for p in positions if int(input_ids[p]) not in targets]
        if not positions and end > prompt_len:
            fallback = list(range(prompt_len, end))
            if self.mask_target_tokens and input_ids is not None:
                targets = set(self.token_ids)
                fallback = [p for p in fallback if int(input_ids[p]) not in targets]
            positions = fallback[-1:]
        if not positions:
            # A completion containing no unmasked positions should receive a neutral score,
            # not an accidental incentive to repeat the target token.
            return torch.full((1,), self.mean, device=attention_mask.device)
        per_layer = []
        for layer in self.score_layers:
            # HF hidden_states[0] is embeddings; block L output is [L + 1].
            h = hidden_states[layer + 1][batch_index, positions].float()
            key = (layer, str(h.device))
            if key not in self._device_jacobians:
                self._device_jacobians[key] = self.lens.jacobians[layer].to(h.device)
            transported = h @ self._device_jacobians[key].T
            normalized = norm(transported.to(norm.weight.dtype))
            per_layer.append(
                target_log_probs(
                    normalized, lm_head, self.token_ids, self.vocab_chunk_size
                )
            )
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
