from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from jlens import JacobianLens
from torch.nn.utils.rnn import pad_sequence

from .common import sha256_file


def literal_variants(words: Sequence[str]) -> list[str]:
    """Return the case and tokenizer-boundary spellings used for literal audits."""
    variants: list[str] = []
    for word in words:
        word = word.strip()
        if not word:
            raise ValueError("target words must be non-empty")
        for spelling in (word.lower(), word.title(), word.upper()):
            for prefix in ("", " "):
                variant = prefix + spelling
                if variant not in variants:
                    variants.append(variant)
    return variants


def literal_token_sequences(
    tokenizer: Any, words: Sequence[str]
) -> list[tuple[int, ...]]:
    """Encode every audited literal spelling, including multi-token spellings."""
    sequences = {
        tuple(tokenizer.encode(variant, add_special_tokens=False))
        for variant in literal_variants(words)
    }
    sequences.discard(())
    if not sequences:
        raise ValueError("none of the literal target variants produced tokenizer tokens")
    return sorted(sequences, key=lambda sequence: (len(sequence), sequence))


def single_token_ids(tokenizer: Any, words: Sequence[str]) -> list[int]:
    """Return all one-token IDs among the complete literal variant set."""
    ids = {
        sequence[0]
        for sequence in literal_token_sequences(tokenizer, words)
        if len(sequence) == 1
    }
    if not ids:
        raise ValueError(
            f"none of {literal_variants(words)} is a single tokenizer token"
        )
    return sorted(ids)


def contains_token_sequence(
    token_ids: Sequence[int], target_sequences: Sequence[Sequence[int]]
) -> bool:
    """Whether token IDs contain any complete literal target spelling."""
    tokens = tuple(int(token) for token in token_ids)
    return any(
        tuple(sequence) == tokens[start : start + len(sequence)]
        for sequence in target_sequences
        if sequence
        for start in range(len(tokens) - len(sequence) + 1)
    )


def causally_excluded_positions(
    input_ids: Sequence[int],
    sequence_end: int,
    excluded_ids: set[int] | None = None,
    excluded_sequences: Sequence[Sequence[int]] | None = None,
) -> set[int]:
    """Mask hidden positions that contain or directly predict excluded literals.

    A causal hidden state at position ``p`` predicts token ``p + 1``.  For an
    excluded occurrence spanning tokens ``[start, end)``, this therefore masks
    the predecessor ``start - 1`` as well as every position in the occurrence.
    """
    live_tokens = input_ids[:sequence_end]
    if isinstance(live_tokens, torch.Tensor):
        live_tokens = live_tokens.detach().cpu().tolist()
    tokens = tuple(int(token) for token in live_tokens)
    positions: set[int] = set()
    for index, token in enumerate(tokens):
        if token in (excluded_ids or set()):
            positions.add(index)
            if index:
                positions.add(index - 1)
    for sequence in excluded_sequences or ():
        pattern = tuple(int(token) for token in sequence)
        if not pattern:
            continue
        for start in range(len(tokens) - len(pattern) + 1):
            if tokens[start : start + len(pattern)] == pattern:
                positions.update(range(max(0, start - 1), start + len(pattern)))
    return positions


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


def sampled_response_positions(
    prompt_len: int,
    sequence_end: int,
    stride: int,
    start_fraction: float,
    end_fraction: float,
    input_ids: torch.Tensor | None = None,
    excluded_ids: set[int] | None = None,
    include_final: bool = False,
    excluded_sequences: Sequence[Sequence[int]] | None = None,
) -> list[int]:
    """Choose response positions inside an explicit fractional window."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("fractional window must satisfy 0 <= start < end <= 1")
    response_len = max(0, sequence_end - prompt_len)
    # Use the same rounding rule for both sides. Adjacent [a, b) / [b, c)
    # windows therefore share a boundary without sharing a token.
    window_start = prompt_len + math.floor(response_len * start_fraction)
    window_end = prompt_len + math.floor(response_len * end_fraction)
    first_position = max(prompt_len + stride - 1, window_start)
    positions = list(range(first_position, window_end, stride))
    excluded_positions: set[int] = set()
    if input_ids is not None:
        excluded_positions = causally_excluded_positions(
            input_ids, sequence_end, excluded_ids, excluded_sequences
        )
        positions = [p for p in positions if p not in excluded_positions]
        if include_final:
            final_positions = [
                p for p in range(window_start, window_end) if p not in excluded_positions
            ]
            if final_positions and final_positions[-1] not in positions:
                positions.append(final_positions[-1])
    if not positions and window_end > window_start:
        fallback = list(range(window_start, window_end))
        if input_ids is not None:
            fallback = [p for p in fallback if p not in excluded_positions]
        positions = fallback[-1:]
    return positions


def validate_calibration_metadata(
    calibration: dict[str, Any],
    *,
    target_words: Sequence[str],
    token_ids: Sequence[int],
    lens_layers: Sequence[int],
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_lens_sha256: str | None = None,
) -> tuple[float, float]:
    """Fail fast when calibration provenance does not match the live scorer."""
    required = {"mean", "std", "token_ids", "target_words", "layers", "model"}
    missing = sorted(required - calibration.keys())
    if missing:
        raise ValueError(f"calibration metadata is missing required fields: {missing}")
    if list(calibration["target_words"]) != list(target_words):
        raise ValueError(
            "calibration target_words do not match config: "
            f"{calibration['target_words']!r} != {list(target_words)!r}"
        )
    calibrated_ids = sorted(int(token) for token in calibration["token_ids"])
    if calibrated_ids != sorted(int(token) for token in token_ids):
        raise ValueError(
            "calibration token_ids do not match tokenizer literal variants: "
            f"{calibrated_ids} != {sorted(int(token) for token in token_ids)}"
        )
    calibrated_layers = [int(layer) for layer in calibration["layers"]]
    if calibrated_layers != [int(layer) for layer in lens_layers]:
        raise ValueError(
            "calibration layers do not match lens layers: "
            f"{calibrated_layers} != {list(lens_layers)}"
        )
    if expected_model is not None and calibration["model"] != expected_model:
        raise ValueError(
            "calibration model does not match config: "
            f"{calibration['model']!r} != {expected_model!r}"
        )
    if expected_model_revision is not None:
        calibrated_revision = calibration.get("model_revision")
        if calibrated_revision != expected_model_revision:
            raise ValueError(
                "calibration model_revision does not match config: "
                f"{calibrated_revision!r} != {expected_model_revision!r}"
            )
    if expected_lens_sha256 is not None:
        calibrated_lens_sha256 = calibration.get("lens_sha256")
        if calibrated_lens_sha256 != expected_lens_sha256:
            raise ValueError(
                "calibration lens_sha256 does not match lens artifact: "
                f"{calibrated_lens_sha256!r} != {expected_lens_sha256!r}"
            )
    mean = float(calibration["mean"])
    std = float(calibration["std"])
    if not math.isfinite(mean):
        raise ValueError("calibration mean must be finite")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("calibration std must be finite and positive")
    return mean, std


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
        score_aggregation: str = "mean",
        score_include_final: bool = False,
        score_components: Sequence[dict[str, Any]] | None = None,
        score_end_fraction: float = 1.0,
        *,
        expected_model: str | None = None,
        expected_model_revision: str | None = None,
        expected_lens_sha256: str | None = None,
    ) -> None:
        self.lens = JacobianLens.load(lens_path)
        self.target_words = list(target_words)
        self.literal_token_sequences = literal_token_sequences(tokenizer, target_words)
        self.token_ids = single_token_ids(tokenizer, target_words)
        self.stride = stride
        self.mask_target_tokens = mask_target_tokens
        self.vocab_chunk_size = vocab_chunk_size
        if not 0.0 <= score_start_fraction < 1.0:
            raise ValueError("score_start_fraction must be in [0, 1)")
        self.score_start_fraction = score_start_fraction
        if not 0.0 < score_end_fraction <= 1.0:
            raise ValueError("score_end_fraction must be in (0, 1]")
        if score_end_fraction <= score_start_fraction:
            raise ValueError("score_end_fraction must exceed score_start_fraction")
        self.score_end_fraction = score_end_fraction
        if score_aggregation not in {"mean", "max", "last"}:
            raise ValueError("score_aggregation must be mean, max, or last")
        self.score_aggregation = score_aggregation
        self.score_include_final = score_include_final
        self.score_components = list(score_components) if score_components else []
        self.special_token_ids = set(tokenizer.all_special_ids)
        available_layers = list(self.lens.source_layers)
        self.score_layers = list(score_layers) if score_layers is not None else available_layers
        missing_layers = set(self.score_layers) - set(available_layers)
        if missing_layers:
            raise ValueError(
                f"score layers {sorted(missing_layers)} are not present in lens layers {available_layers}"
            )
        self._validate_score_components(available_layers)
        calibration = json.loads(Path(calibration_path).read_text())
        actual_lens_sha256 = sha256_file(lens_path)
        if (
            expected_lens_sha256 is not None
            and expected_lens_sha256 != actual_lens_sha256
        ):
            raise ValueError(
                "lens artifact does not match configured lens_sha256: "
                f"{actual_lens_sha256!r} != {expected_lens_sha256!r}"
            )
        self.mean, self.std = validate_calibration_metadata(
            calibration,
            target_words=self.target_words,
            token_ids=self.token_ids,
            lens_layers=available_layers,
            expected_model=expected_model,
            expected_model_revision=expected_model_revision,
            expected_lens_sha256=actual_lens_sha256,
        )
        self._device_jacobians: dict[tuple[int, str], torch.Tensor] = {}

    def _validate_score_components(self, available_layers: Sequence[int]) -> None:
        for index, component in enumerate(self.score_components):
            if "layer" not in component or "weight" not in component:
                raise ValueError(f"score component {index} requires layer and weight")
            layer = int(component["layer"])
            if layer not in available_layers:
                raise ValueError(
                    f"score component {index} layer {layer} is not present in lens layers "
                    f"{list(available_layers)}"
                )
            start = float(component.get("start_fraction", 0.0))
            end = float(component.get("end_fraction", 1.0))
            if not 0.0 <= start < end <= 1.0:
                raise ValueError(
                    f"score component {index} must satisfy 0 <= start_fraction "
                    "< end_fraction <= 1"
                )
            aggregation = str(component.get("aggregation", "mean"))
            if aggregation not in {"mean", "max", "last"}:
                raise ValueError(
                    f"score component {index} aggregation must be mean, max, or last"
                )
            if not math.isfinite(float(component["weight"])):
                raise ValueError(f"score component {index} weight must be finite")

    @torch.no_grad()
    def raw_scores(
        self, model: Any, hidden_states: Sequence[torch.Tensor], prompt_len: int,
        attention_mask: torch.Tensor, batch_index: int = 0,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm, lm_head = decoder_parts(model)
        end = int(attention_mask.sum().item())
        if self.mask_target_tokens and input_ids is None:
            raise ValueError("input_ids are required when mask_target_tokens is enabled")
        excluded = set(self.special_token_ids)
        positions = sampled_response_positions(
            prompt_len, end, self.stride, self.score_start_fraction,
            self.score_end_fraction, input_ids, excluded, self.score_include_final,
            self.literal_token_sequences if self.mask_target_tokens else None,
        )
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
        return torch.stack(per_layer)

    def __call__(
        self, model: Any, hidden_states: Sequence[torch.Tensor], prompt_len: int,
        attention_mask: torch.Tensor, batch_index: int = 0,
        input_ids: torch.Tensor | None = None,
    ) -> float:
        if self.score_components:
            saved = (
                self.score_layers, self.score_start_fraction,
                self.score_end_fraction, self.score_aggregation,
                self.score_include_final,
            )
            combined = 0.0
            try:
                for component in self.score_components:
                    self.score_layers = [int(component["layer"])]
                    self.score_start_fraction = float(component.get("start_fraction", 0.0))
                    self.score_end_fraction = float(component.get("end_fraction", 1.0))
                    if self.score_end_fraction <= self.score_start_fraction:
                        raise ValueError("component end_fraction must exceed start_fraction")
                    self.score_aggregation = str(component.get("aggregation", "mean"))
                    self.score_include_final = bool(component.get("include_final", False))
                    raw_scores = self.raw_scores(
                        model, hidden_states, prompt_len, attention_mask, batch_index, input_ids
                    )
                    score = self._aggregate(raw_scores)
                    standardized = ((score - self.mean) / self.std).clamp(-5, 5)
                    combined += float(component["weight"]) * float(standardized.item())
            finally:
                (
                    self.score_layers, self.score_start_fraction,
                    self.score_end_fraction, self.score_aggregation,
                    self.score_include_final,
                ) = saved
            return combined
        raw_scores = self.raw_scores(
            model, hidden_states, prompt_len, attention_mask, batch_index, input_ids
        )
        raw = self._aggregate(raw_scores)
        return float(((raw - self.mean) / self.std).clamp(-5, 5).item())

    def _aggregate(self, raw_scores: torch.Tensor) -> torch.Tensor:
        if self.score_aggregation == "mean":
            return raw_scores.mean()
        elif self.score_aggregation == "max":
            return raw_scores.max()
        return raw_scores[:, -1].mean() if raw_scores.ndim == 2 else raw_scores[-1]


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
            contains_token_sequence(completion, self.scorer.literal_token_sequences)
            for completion in completion_ids
        ) / len(completion_ids)
        log_metric(f"jlens/{self.label}_mean", sum(rewards) / len(rewards))
        log_metric(f"jlens/{self.label}_literal_rate", literal_rate)
        return rewards
