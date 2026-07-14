"""Pure helpers for discovering lexical J-lens correctness correlates.

The functions in this module deliberately separate three concerns:

* deterministic assignment of exposed prompts to discovery/replication shards;
* faithful vocabulary and lexical-word scoring from one J-decoded logits matrix;
* prompt-clustered association summaries and a frozen word-selection rule.

Model generation, dataset access, and remote orchestration live outside this
module.  Keeping the numerical core pure makes it possible to unit-test the
scanner without downloading a model or touching a protected evaluation split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .common import (
    format_prompt,
    gsm8k_reward,
    model_dtype,
    seed_everything,
    sha256_file,
)
from .reward import (
    contains_token_sequence,
    decoder_parts,
    literal_token_sequences,
    sampled_response_positions,
    single_token_ids,
)


DEFAULT_SPLIT_SALT = "j-lens-rl-word-correlation-discovery-v1"
DEFAULT_SHARD_SALT = "j-lens-rl-word-correlation-shards-v1"
DEFAULT_ROLLOUT_NAMESPACE = "j-lens-rl-word-correlation-rollouts-v1"
LEXICAL_FILTER_VERSION = "ascii-boundary-word-v1"

# A candidate is intentionally narrower than arbitrary ``str.isalpha`` text:
# byte fallbacks, control characters, multiword tokens, and trailing punctuation
# are not faithful single lexical targets for the reward implementation.
_LEXICAL_DECODE = re.compile(r" ?[A-Za-z]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][A-Za-z]+)*\Z")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_source_indices(source_indices: Iterable[int]) -> list[int]:
    indices = list(source_indices)
    if not indices:
        raise ValueError("source_indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("source indices must be integers")
    if any(index < 0 for index in indices):
        raise ValueError("source indices must be non-negative")
    if len(indices) != len(set(indices)):
        raise ValueError("source indices must be unique")
    return indices


def _index_digest(salt: str, source_index: int) -> bytes:
    if not salt:
        raise ValueError("salt must not be empty")
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise ValueError("source_index must be an integer")
    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    return hashlib.sha256(f"{salt}:{source_index}".encode("utf-8")).digest()


def deterministic_prompt_split(
    source_indices: Iterable[int],
    *,
    discovery_fraction: float = 0.5,
    salt: str = DEFAULT_SPLIT_SALT,
) -> dict[int, str]:
    """Assign an exact, order-independent prompt split by a salted hash rank."""
    indices = _validate_source_indices(source_indices)
    if not 0.0 < discovery_fraction < 1.0:
        raise ValueError("discovery_fraction must be strictly between zero and one")
    discovery_count = math.floor(len(indices) * discovery_fraction)
    if discovery_count == 0 or discovery_count == len(indices):
        raise ValueError("the requested split leaves one partition empty")
    ordered = sorted(indices, key=lambda index: (_index_digest(salt, index), index))
    discovery = set(ordered[:discovery_count])
    return {
        index: "discovery" if index in discovery else "replication"
        for index in sorted(indices)
    }


def deterministic_shards(
    source_indices: Iterable[int],
    num_shards: int,
    *,
    salt: str = DEFAULT_SHARD_SALT,
) -> list[list[int]]:
    """Return balanced, order-independent prompt shards.

    Hash-ranking followed by round-robin assignment keeps shard sizes within
    one prompt while making worker scheduling irrelevant to membership.
    """
    indices = _validate_source_indices(source_indices)
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards <= 0:
        raise ValueError("num_shards must be a positive integer")
    if num_shards > len(indices):
        raise ValueError("num_shards cannot exceed the number of prompts")
    ordered = sorted(indices, key=lambda index: (_index_digest(salt, index), index))
    shards = [[] for _ in range(num_shards)]
    for position, index in enumerate(ordered):
        shards[position % num_shards].append(index)
    return [sorted(shard) for shard in shards]


def deterministic_rollout_seed(
    base_seed: int,
    source_index: int,
    rollout_index: int,
    *,
    namespace: str = DEFAULT_ROLLOUT_NAMESPACE,
) -> int:
    """Derive a scheduling-independent non-negative 63-bit rollout seed."""
    for name, value in (
        ("base_seed", base_seed),
        ("source_index", source_index),
        ("rollout_index", rollout_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not namespace:
        raise ValueError("namespace must not be empty")
    payload = f"{namespace}:{base_seed}:{source_index}:{rollout_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def deterministic_prompt_seed(seed_salt: str, source_index: int) -> int:
    """Return the first 63 digest bits used by a grouped generation call."""
    digest = _index_digest(seed_salt, source_index)
    return int.from_bytes(digest[:8], "big") >> 1


def _seed_grouped_generation(seed: int) -> None:
    """Seed Torch with the frozen 63 bits while keeping NumPy in its range."""
    # ``common.seed_everything`` also establishes deterministic backend flags,
    # but NumPy's legacy global seeder rejects integers >= 2**32.  The prompt
    # contract intentionally uses 63 digest bits, so initialize all backends
    # with a valid low-32 seed and then restore the full seed for the Python and
    # Torch RNGs that drive generation.
    seed_everything(seed % (1 << 32))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class LexicalCandidate:
    """A canonical word and the exact tokenizer variants used by the reward."""

    canonical_word: str
    token_ids: tuple[int, ...]
    boundary_token_ids: tuple[int, ...]
    decoded_variants: tuple[str, ...]
    tokenizer_tokens: tuple[str, ...]
    literal_sequences: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_word": self.canonical_word,
            "token_ids": list(self.token_ids),
            "boundary_token_ids": list(self.boundary_token_ids),
            "decoded_variants": list(self.decoded_variants),
            "tokenizer_tokens": list(self.tokenizer_tokens),
            "literal_sequences": [list(sequence) for sequence in self.literal_sequences],
            "lexical_filter_version": LEXICAL_FILTER_VERSION,
        }


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def lexical_candidates(
    tokenizer: Any,
    *,
    min_length: int = 2,
    max_length: int = 32,
    require_boundary_token: bool = True,
) -> list[LexicalCandidate]:
    """Map decodable tokenizer IDs to stable, reward-compatible ASCII words.

    The model output head may contain reserved rows beyond the tokenizer's
    actual vocabulary.  This function intentionally iterates ``get_vocab`` IDs
    rather than assuming ``len(tokenizer) == lm_head.out_features``.
    """
    if min_length <= 0 or max_length < min_length:
        raise ValueError("lexical length bounds are invalid")
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("tokenizer.get_vocab() must return a non-empty mapping")
    id_to_token: dict[int, str] = {}
    for raw_token, token_id in sorted(vocabulary.items()):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("tokenizer vocabulary IDs must be non-negative integers")
        # Some tokenizers expose aliases.  A deterministic representative is
        # enough because the numerical readout is indexed by integer ID.
        id_to_token.setdefault(token_id, str(raw_token))

    special_ids = {int(value) for value in tokenizer.all_special_ids}
    decoded_ids: dict[str, set[int]] = {}
    boundary_ids: dict[str, set[int]] = {}
    for token_id in sorted(id_to_token):
        if token_id in special_ids:
            continue
        decoded = _decode_token(tokenizer, token_id)
        if _LEXICAL_DECODE.fullmatch(decoded) is None:
            continue
        word = decoded.strip().lower()
        if not min_length <= len(word) <= max_length:
            continue
        decoded_ids.setdefault(word, set()).add(token_id)
        if decoded.startswith(" "):
            boundary_ids.setdefault(word, set()).add(token_id)

    candidates: list[LexicalCandidate] = []
    vocabulary_ids = set(id_to_token)
    for word in sorted(decoded_ids):
        if require_boundary_token and not boundary_ids.get(word):
            continue
        try:
            reward_ids = tuple(
                token_id
                for token_id in single_token_ids(tokenizer, [word])
                if token_id in vocabulary_ids and token_id not in special_ids
            )
        except ValueError:
            continue
        if not reward_ids:
            continue
        decoded_variants = tuple(_decode_token(tokenizer, token_id) for token_id in reward_ids)
        tokenizer_tokens = tuple(id_to_token[token_id] for token_id in reward_ids)
        candidates.append(
            LexicalCandidate(
                canonical_word=word,
                token_ids=reward_ids,
                boundary_token_ids=tuple(sorted(boundary_ids.get(word, set()))),
                decoded_variants=decoded_variants,
                tokenizer_tokens=tokenizer_tokens,
                literal_sequences=tuple(literal_token_sequences(tokenizer, [word])),
            )
        )
    return candidates


@dataclass(frozen=True)
class FullVocabularyReadout:
    """One materialized full-vocabulary logits matrix and its normalizer."""

    logits: torch.Tensor
    log_normalizer: torch.Tensor

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        normalizer_chunk_size: int | None = None,
    ) -> "FullVocabularyReadout":
        if logits.ndim != 2 or 0 in logits.shape:
            raise ValueError("logits must have shape [positions, vocabulary]")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must be finite")
        if normalizer_chunk_size is None:
            log_normalizer = torch.logsumexp(logits.float(), dim=-1)
        else:
            if (
                isinstance(normalizer_chunk_size, bool)
                or not isinstance(normalizer_chunk_size, int)
                or normalizer_chunk_size <= 0
            ):
                raise ValueError("normalizer_chunk_size must be a positive integer")
            log_normalizer = None
            for start in range(0, logits.shape[1], normalizer_chunk_size):
                chunk = torch.logsumexp(
                    logits[:, start : start + normalizer_chunk_size].float(), dim=-1
                )
                log_normalizer = (
                    chunk
                    if log_normalizer is None
                    else torch.logaddexp(log_normalizer, chunk)
                )
            assert log_normalizer is not None
        return cls(logits=logits, log_normalizer=log_normalizer)

    @classmethod
    def from_hidden(
        cls,
        normalized_hidden: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor | None = None,
        *,
        normalizer_chunk_size: int | None = None,
    ) -> "FullVocabularyReadout":
        if normalized_hidden.ndim != 2 or output_weight.ndim != 2:
            raise ValueError("hidden and output weight must both be matrices")
        if normalized_hidden.shape[1] != output_weight.shape[1]:
            raise ValueError("hidden width does not match output weight width")
        logits = normalized_hidden @ output_weight.T
        if output_bias is not None:
            if output_bias.ndim != 1 or len(output_bias) != output_weight.shape[0]:
                raise ValueError("output bias does not match vocabulary size")
            logits = logits + output_bias
        return cls.from_logits(logits, normalizer_chunk_size)

    @property
    def num_positions(self) -> int:
        return int(self.logits.shape[0])

    @property
    def vocabulary_size(self) -> int:
        return int(self.logits.shape[1])

    def token_log_probs(self) -> torch.Tensor:
        """Return the complete position-by-token log-probability matrix."""
        return self.logits.float() - self.log_normalizer[:, None]

    def word_position_log_probs(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Exact log probability mass of a token-ID union at every position."""
        ids = [int(token_id) for token_id in token_ids]
        if not ids:
            raise ValueError("token_ids must not be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("token_ids must be unique")
        if any(token_id < 0 or token_id >= self.vocabulary_size for token_id in ids):
            raise ValueError("token_ids contain an out-of-range vocabulary row")
        index = torch.as_tensor(ids, device=self.logits.device)
        numerator = torch.logsumexp(self.logits.index_select(1, index).float(), dim=-1)
        return numerator - self.log_normalizer

    def aggregate_token_log_probs(
        self,
        position_rows: Sequence[int] | None = None,
        *,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Aggregate every individual token score over selected matrix rows."""
        values = self.token_log_probs()
        values = _select_position_rows(values, position_rows)
        return _aggregate_positions(values, aggregation)


def dense_response_positions(
    input_ids: Sequence[int] | torch.Tensor,
    prompt_len: int,
    sequence_end: int,
    *,
    start_fraction: float,
    end_fraction: float,
    special_token_ids: Iterable[int] = (),
) -> list[int]:
    """Return a dense superset containing every candidate's possible rows."""
    tensor_ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
    return sampled_response_positions(
        prompt_len,
        sequence_end,
        1,
        start_fraction,
        end_fraction,
        tensor_ids,
        {int(value) for value in special_token_ids},
        True,
    )


def masked_candidate_positions(
    input_ids: Sequence[int] | torch.Tensor,
    prompt_len: int,
    sequence_end: int,
    *,
    stride: int,
    start_fraction: float,
    end_fraction: float,
    literal_sequences: Sequence[Sequence[int]],
    special_token_ids: Iterable[int] = (),
    include_final: bool = False,
    mask_target_tokens: bool = True,
) -> list[int]:
    """Choose the exact positions for one candidate's masked reward."""
    tensor_ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
    return sampled_response_positions(
        prompt_len,
        sequence_end,
        stride,
        start_fraction,
        end_fraction,
        tensor_ids,
        {int(value) for value in special_token_ids},
        include_final,
        literal_sequences if mask_target_tokens else None,
    )


def position_rows(
    materialized_positions: Sequence[int], selected_positions: Sequence[int]
) -> list[int]:
    """Map absolute sequence positions to rows of a shared logits matrix."""
    positions = [int(position) for position in materialized_positions]
    if len(positions) != len(set(positions)):
        raise ValueError("materialized_positions must be unique")
    lookup = {position: row for row, position in enumerate(positions)}
    try:
        return [lookup[int(position)] for position in selected_positions]
    except KeyError as error:
        raise ValueError(
            f"selected position {error.args[0]} was not materialized"
        ) from None


def atlas_prompt_sufficient(
    rollout_token_scores: Sequence[np.ndarray | None],
    correctness: Sequence[bool | int | float],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return descriptive atlas sufficients, or skip an unusable prompt.

    The atlas is explicitly descriptive and cannot select the emotional word.
    A rollout can have no sampled readout position (for example, an immediate
    special-token termination).  Candidate scoring already gives such a
    rollout its calibrated neutral score, so the descriptive atlas must not
    abort the primary frozen-candidate analysis.  We conservatively omit the
    whole prompt from the atlas rather than impute a vocabulary-wide vector.
    """

    if not rollout_token_scores or len(rollout_token_scores) != len(correctness):
        raise ValueError("atlas scores and correctness must have equal non-zero length")
    labels = np.asarray(correctness, dtype=np.float64)
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("atlas correctness labels must be finite binary values")
    if any(values is None for values in rollout_token_scores):
        return None
    values = np.stack(
        [np.asarray(item, dtype=np.float64) for item in rollout_token_scores], axis=0
    )
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("atlas rollout scores must be finite equal-length vectors")
    if labels.min() == labels.max():
        return None
    centered_values = values - values.mean(axis=0)
    centered_labels = labels - labels.mean()
    return (
        (centered_values * centered_labels[:, np.newaxis]).sum(axis=0),
        np.square(centered_values).sum(axis=0),
        float(np.square(centered_labels).sum()),
    )


def _select_position_rows(
    values: torch.Tensor, position_rows: Sequence[int] | None
) -> torch.Tensor:
    if position_rows is None:
        return values
    rows = [int(row) for row in position_rows]
    if not rows:
        return values[:0]
    if len(rows) != len(set(rows)):
        raise ValueError("position_rows must be unique")
    if any(row < 0 or row >= values.shape[0] for row in rows):
        raise ValueError("position_rows contain an out-of-range row")
    index = torch.as_tensor(rows, device=values.device)
    return values.index_select(0, index)


def _aggregate_positions(values: torch.Tensor, aggregation: str) -> torch.Tensor:
    if values.shape[0] == 0:
        raise ValueError("cannot aggregate an empty set of positions")
    if aggregation == "mean":
        return values.mean(dim=0)
    if aggregation == "max":
        return values.max(dim=0).values
    if aggregation == "last":
        return values[-1]
    raise ValueError("aggregation must be mean, max, or last")


def exact_word_score(
    readout: FullVocabularyReadout,
    token_ids: Sequence[int],
    position_rows: Sequence[int] | None = None,
    *,
    aggregation: str = "mean",
    neutral_score: float | None = None,
) -> torch.Tensor:
    """Score a lexical token union exactly like one TargetJLReward component.

    Candidate-specific masking is represented by ``position_rows``.  If every
    position was masked, callers must supply the word's calibration mean as the
    neutral raw score, mirroring :class:`TargetJLReward`.
    """
    values = readout.word_position_log_probs(token_ids)
    selected = _select_position_rows(values, position_rows)
    if selected.shape[0] == 0:
        if neutral_score is None or not math.isfinite(float(neutral_score)):
            raise ValueError("neutral_score is required when every position is masked")
        return torch.tensor(float(neutral_score), device=values.device, dtype=values.dtype)
    return _aggregate_positions(selected, aggregation)


@dataclass(frozen=True)
class PromptAssociation:
    correct_count: int
    incorrect_count: int
    mean_correct_minus_incorrect: np.ndarray
    centered_covariance: np.ndarray


def prompt_centered_association(
    scores: Sequence[Sequence[float]] | np.ndarray,
    correctness: Sequence[bool | int | float],
) -> PromptAssociation:
    """Compute within-prompt association without treating rollouts as IID.

    ``scores`` is ``[rollouts, candidates]``.  A prompt with no outcome
    variation carries no within-group information and is rejected explicitly.
    """
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(correctness)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("scores must have shape [rollouts, candidates]")
    if labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("correctness must be one-dimensional and match rollouts")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not np.isin(labels, [0, 1, False, True]).all():
        raise ValueError("correctness values must be boolean or zero/one")
    labels = labels.astype(bool)
    correct_count = int(labels.sum())
    incorrect_count = int((~labels).sum())
    if correct_count == 0 or incorrect_count == 0:
        raise ValueError("prompt-centered association requires mixed outcomes")
    difference = values[labels].mean(axis=0) - values[~labels].mean(axis=0)
    centered_values = values - values.mean(axis=0)
    centered_labels = labels.astype(np.float64) - labels.mean()
    covariance = np.mean(centered_values * centered_labels[:, np.newaxis], axis=0)
    return PromptAssociation(
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        mean_correct_minus_incorrect=difference,
        centered_covariance=covariance,
    )


@dataclass(frozen=True)
class AssociationSummary:
    n_prompts: int
    mean_difference: np.ndarray
    standard_deviation: np.ndarray
    standard_error: np.ndarray
    t_statistic: np.ndarray
    positive_fraction: np.ndarray
    negative_fraction: np.ndarray

    def record(self, index: int) -> dict[str, float | int]:
        if index < 0 or index >= len(self.mean_difference):
            raise IndexError(index)
        return {
            "n_prompts": self.n_prompts,
            "mean_difference": float(self.mean_difference[index]),
            "standard_deviation": float(self.standard_deviation[index]),
            "standard_error": float(self.standard_error[index]),
            "t_statistic": float(self.t_statistic[index]),
            "positive_fraction": float(self.positive_fraction[index]),
            "negative_fraction": float(self.negative_fraction[index]),
        }


def aggregate_associations(
    prompt_differences: Sequence[Sequence[float]] | np.ndarray,
) -> AssociationSummary:
    """Aggregate one independent correct-minus-incorrect vector per prompt."""
    values = np.asarray(prompt_differences, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("prompt_differences must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError("prompt differences must be finite")
    n_prompts = len(values)
    mean = values.mean(axis=0)
    if n_prompts == 1:
        standard_deviation = np.full(values.shape[1], np.nan)
        standard_error = np.full(values.shape[1], np.nan)
        t_statistic = np.full(values.shape[1], np.nan)
    else:
        standard_deviation = values.std(axis=0, ddof=1)
        standard_error = standard_deviation / math.sqrt(n_prompts)
        t_statistic = np.divide(
            mean,
            standard_error,
            out=np.where(mean == 0.0, 0.0, np.copysign(np.inf, mean)),
            where=standard_error > 0.0,
        )
    return AssociationSummary(
        n_prompts=n_prompts,
        mean_difference=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        t_statistic=t_statistic,
        positive_fraction=(values > 0.0).mean(axis=0),
        negative_fraction=(values < 0.0).mean(axis=0),
    )


def select_emotional_candidate(
    candidates: Sequence[LexicalCandidate],
    discovery: AssociationSummary,
    replication: AssociationSummary,
    emotional_words: Iterable[str],
    *,
    min_abs_discovery_t: float = 0.0,
    min_abs_replication_t: float = 0.0,
) -> dict[str, Any]:
    """Apply a deterministic discovery-rank/replication-direction rule.

    Candidates are ranked *only* by absolute discovery t statistic.  The
    replication half is a sign/strength gate, never a second ranking surface.
    Ties are broken lexicographically by canonical word.
    """
    count = len(candidates)
    for name, summary in (("discovery", discovery), ("replication", replication)):
        if len(summary.mean_difference) != count:
            raise ValueError(f"{name} summary does not match candidate count")
    if not math.isfinite(min_abs_discovery_t) or min_abs_discovery_t < 0:
        raise ValueError("min_abs_discovery_t must be finite and non-negative")
    if not math.isfinite(min_abs_replication_t) or min_abs_replication_t < 0:
        raise ValueError("min_abs_replication_t must be finite and non-negative")
    allowed = {str(word).strip().lower() for word in emotional_words if str(word).strip()}
    if not allowed:
        raise ValueError("emotional_words must not be empty")

    eligible: list[tuple[float, str, int]] = []
    for index, candidate in enumerate(candidates):
        if candidate.canonical_word not in allowed:
            continue
        discovery_mean = float(discovery.mean_difference[index])
        replication_mean = float(replication.mean_difference[index])
        discovery_t = float(discovery.t_statistic[index])
        replication_t = float(replication.t_statistic[index])
        if not all(
            math.isfinite(value)
            for value in (discovery_mean, replication_mean, discovery_t, replication_t)
        ):
            continue
        if discovery_mean == 0.0 or replication_mean == 0.0:
            continue
        if math.copysign(1.0, discovery_mean) != math.copysign(1.0, replication_mean):
            continue
        if abs(discovery_t) < min_abs_discovery_t:
            continue
        if abs(replication_t) < min_abs_replication_t:
            continue
        eligible.append((-abs(discovery_t), candidate.canonical_word, index))
    if not eligible:
        raise ValueError("no emotional candidate passed the frozen replication rule")

    _, _, selected_index = min(eligible)
    selected = candidates[selected_index]
    positive = float(discovery.mean_difference[selected_index]) > 0.0
    return {
        "canonical_word": selected.canonical_word,
        "reward_sign": 1 if positive else -1,
        "association_direction": (
            "positive_with_correctness" if positive else "negative_with_correctness"
        ),
        "token_ids": list(selected.token_ids),
        "discovery": discovery.record(selected_index),
        "replication": replication.record(selected_index),
        "selection_rule": (
            "largest absolute discovery t among predeclared emotional words; "
            "replication direction must agree; lexical tie-break"
        ),
        "min_abs_discovery_t": min_abs_discovery_t,
        "min_abs_replication_t": min_abs_replication_t,
    }


def _validate_grouped_scores(
    grouped_scores: Sequence[np.ndarray | Sequence[Sequence[float]]],
    grouped_correctness: Sequence[np.ndarray | Sequence[bool | int | float]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(grouped_scores) != len(grouped_correctness) or not grouped_scores:
        raise ValueError("grouped scores and correctness must be non-empty and aligned")
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    candidate_count: int | None = None
    for group_scores, group_labels in zip(
        grouped_scores, grouped_correctness, strict=True
    ):
        values = np.asarray(group_scores, dtype=np.float64)
        outcome = np.asarray(group_labels)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError("every score group must be a non-empty matrix")
        if candidate_count is None:
            candidate_count = values.shape[1]
        elif values.shape[1] != candidate_count:
            raise ValueError("score groups disagree on candidate count")
        if outcome.ndim != 1 or len(outcome) != len(values):
            raise ValueError("each correctness group must match its score rows")
        if not np.isfinite(values).all():
            raise ValueError("grouped scores must be finite")
        if not np.isin(outcome, [0, 1, False, True]).all():
            raise ValueError("grouped correctness must be boolean or zero/one")
        outcome = outcome.astype(np.float64)
        if outcome.min() == outcome.max():
            raise ValueError("point-biserial groups must contain mixed outcomes")
        scores.append(values)
        labels.append(outcome)
    return scores, labels


def _centered_correlation_parts(
    grouped_scores: Sequence[np.ndarray | Sequence[Sequence[float]]],
    grouped_correctness: Sequence[np.ndarray | Sequence[bool | int | float]],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, float]:
    scores, labels = _validate_grouped_scores(grouped_scores, grouped_correctness)
    centered_scores = [values - values.mean(axis=0) for values in scores]
    centered_labels = [outcome - outcome.mean() for outcome in labels]
    numerator = sum(
        (values * outcome[:, np.newaxis]).sum(axis=0)
        for values, outcome in zip(centered_scores, centered_labels, strict=True)
    )
    score_ss = sum(np.square(values).sum(axis=0) for values in centered_scores)
    label_ss = float(sum(np.square(outcome).sum() for outcome in centered_labels))
    return centered_scores, centered_labels, numerator, score_ss, label_ss


def _safe_correlations(
    numerator: np.ndarray, score_ss: np.ndarray, label_ss: float
) -> np.ndarray:
    denominator = np.sqrt(score_ss * label_ss)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )


def prompt_centered_point_biserial(
    grouped_scores: Sequence[np.ndarray | Sequence[Sequence[float]]],
    grouped_correctness: Sequence[np.ndarray | Sequence[bool | int | float]],
) -> np.ndarray:
    """Correlate scores and correctness after centering within each prompt."""
    _, _, numerator, score_ss, label_ss = _centered_correlation_parts(
        grouped_scores, grouped_correctness
    )
    return _safe_correlations(numerator, score_ss, label_ss)


def within_prompt_permutation_test(
    grouped_scores: Sequence[np.ndarray | Sequence[Sequence[float]]],
    grouped_correctness: Sequence[np.ndarray | Sequence[bool | int | float]],
    *,
    draws: int,
    seed: int,
    eligible_indices: Sequence[int] | None = None,
    locked_sign: int | None = None,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Permutation inference with labels shuffled independently by prompt.

    With ``locked_sign=None`` this returns candidate-wise two-sided and max-|r|
    adjusted p-values.  With ``locked_sign`` set to ``+1`` or ``-1``, exactly
    one candidate is required and a one-sided locked-direction p-value is
    returned instead.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    centered_scores, _, numerator, score_ss, label_ss = _centered_correlation_parts(
        grouped_scores, grouped_correctness
    )
    observed = _safe_correlations(numerator, score_ss, label_ss)
    candidate_count = len(observed)
    if eligible_indices is None:
        eligible = np.arange(candidate_count, dtype=np.int64)
    else:
        eligible = np.asarray(list(eligible_indices), dtype=np.int64)
        if (
            eligible.ndim != 1
            or len(eligible) == 0
            or len(set(eligible.tolist())) != len(eligible)
            or (eligible < 0).any()
            or (eligible >= candidate_count).any()
        ):
            raise ValueError("eligible_indices are invalid")
    if locked_sign is not None and (locked_sign not in {-1, 1} or candidate_count != 1):
        raise ValueError("locked_sign requires one candidate and must be -1 or +1")

    # All current grouped rollouts have equal size, but accepting variable
    # group sizes costs little and makes this helper independently reusable.
    correct_counts = np.asarray(
        [int(np.asarray(labels).sum()) for labels in grouped_correctness],
        dtype=np.int64,
    )
    group_sizes = {len(np.asarray(labels)) for labels in grouped_correctness}
    if len(group_sizes) != 1:
        raise ValueError("permutation implementation requires equal rollout group sizes")
    rollout_count = group_sizes.pop()
    score_tensor = np.stack(centered_scores, axis=0)
    denominator = np.sqrt(score_ss * label_ss)
    generator = np.random.default_rng(seed)
    unadjusted_exceed = np.zeros(candidate_count, dtype=np.int64)
    max_adjusted_exceed = np.zeros(candidate_count, dtype=np.int64)
    directional_exceed = 0
    observed_absolute = np.abs(observed)
    observed_family_max = float(observed_absolute[eligible].max())

    for start in range(0, draws, batch_size):
        batch = min(batch_size, draws - start)
        keys = generator.random((batch, len(score_tensor), rollout_count))
        ranks = np.argsort(np.argsort(keys, axis=2), axis=2)
        permuted = ranks < correct_counts[np.newaxis, :, np.newaxis]
        permuted_numerator = np.einsum(
            "bgr,grc->bc", permuted, score_tensor, optimize=True
        )
        correlations = np.divide(
            permuted_numerator,
            denominator[np.newaxis, :],
            out=np.zeros_like(permuted_numerator, dtype=np.float64),
            where=denominator[np.newaxis, :] > 0.0,
        )
        if locked_sign is not None:
            directional_exceed += int(
                np.sum(locked_sign * correlations[:, 0] >= locked_sign * observed[0])
            )
            continue
        absolute = np.abs(correlations)
        unadjusted_exceed += np.sum(
            absolute >= observed_absolute[np.newaxis, :], axis=0
        )
        permutation_max = absolute[:, eligible].max(axis=1)
        max_adjusted_exceed += np.sum(
            permutation_max[:, np.newaxis] >= observed_absolute[np.newaxis, :],
            axis=0,
        )

    if locked_sign is not None:
        return {
            "method": "within_prompt_label_permutation_one_sided",
            "draws": draws,
            "seed": seed,
            "locked_sign": locked_sign,
            "observed_correlation": float(observed[0]),
            "one_sided_p": (directional_exceed + 1) / (draws + 1),
        }
    return {
        "method": "within_prompt_label_permutation_max_abs",
        "draws": draws,
        "seed": seed,
        "observed_correlations": observed.tolist(),
        "eligible_indices": eligible.tolist(),
        "observed_family_max_abs": observed_family_max,
        "unadjusted_two_sided_p": (
            (unadjusted_exceed + 1) / (draws + 1)
        ).tolist(),
        "max_abs_adjusted_p": (
            (max_adjusted_exceed + 1) / (draws + 1)
        ).tolist(),
    }


def prompt_cluster_bootstrap_correlation(
    grouped_scores: Sequence[np.ndarray | Sequence[Sequence[float]]],
    grouped_correctness: Sequence[np.ndarray | Sequence[bool | int | float]],
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Percentile interval resampling whole prompt groups with replacement."""
    if draws <= 0 or isinstance(draws, bool):
        raise ValueError("draws must be a positive integer")
    if seed < 0 or isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    centered_scores, centered_labels, numerator, score_ss, label_ss = (
        _centered_correlation_parts(grouped_scores, grouped_correctness)
    )
    if len(numerator) != 1:
        raise ValueError("cluster bootstrap expects exactly one selected candidate")
    group_numerator = np.asarray(
        [
            float((values[:, 0] * labels).sum())
            for values, labels in zip(centered_scores, centered_labels, strict=True)
        ]
    )
    group_score_ss = np.asarray(
        [float(np.square(values[:, 0]).sum()) for values in centered_scores]
    )
    group_label_ss = np.asarray(
        [float(np.square(labels).sum()) for labels in centered_labels]
    )
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    group_count = len(group_numerator)
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        sampled = generator.integers(
            0, group_count, size=(stop - start, group_count)
        )
        sampled_numerator = group_numerator[sampled].sum(axis=1)
        denominator = np.sqrt(
            group_score_ss[sampled].sum(axis=1)
            * group_label_ss[sampled].sum(axis=1)
        )
        estimates[start:stop] = np.divide(
            sampled_numerator,
            denominator,
            out=np.zeros(stop - start, dtype=np.float64),
            where=denominator > 0.0,
        )
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    observed = _safe_correlations(numerator, score_ss, label_ss)[0]
    return {
        "method": "prompt_cluster_percentile",
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "observed_correlation": float(observed),
        "correlation_ci_low": float(low),
        "correlation_ci_high": float(high),
    }


def select_discovery_candidate(
    candidates: Sequence[LexicalCandidate],
    correlations: Sequence[float] | np.ndarray,
    eligible: Sequence[bool],
) -> dict[str, Any]:
    """Select max-|r| using discovery only and a lexical exact-tie break."""
    values = np.asarray(correlations, dtype=np.float64)
    flags = np.asarray(eligible)
    if values.ndim != 1 or len(values) != len(candidates):
        raise ValueError("correlations do not match candidates")
    if flags.ndim != 1 or len(flags) != len(candidates):
        raise ValueError("eligibility flags do not match candidates")
    if not np.isfinite(values).all():
        raise ValueError("correlations must be finite")
    ranked = [
        (-abs(float(values[index])), candidate.canonical_word, index)
        for index, candidate in enumerate(candidates)
        if bool(flags[index]) and values[index] != 0.0
    ]
    if not ranked:
        raise ValueError("no eligible emotional candidate has nonzero association")
    _, _, index = min(ranked)
    candidate = candidates[index]
    positive = bool(values[index] > 0.0)
    return {
        "canonical_word": candidate.canonical_word,
        "reward_sign": 1 if positive else -1,
        "association_direction": (
            "positive_with_correctness" if positive else "negative_with_correctness"
        ),
        "token_ids": list(candidate.token_ids),
        "discovery_correlation": float(values[index]),
        "candidate_index": index,
        "selection_rule": (
            "largest absolute prompt-centered point-biserial correlation among "
            "eligible emotional words; exact tie resolved lexicographically"
        ),
    }


def _read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return payload


def _write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def _write_jsonl_atomic(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(target)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"non-object JSON record on {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def _write_npy_atomic(path: str | Path, values: np.ndarray) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(target)


def _repository_from_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.parent.name != "configs":
        raise ValueError("word-correlation config must live in the repository configs directory")
    return path.parent.parent


def _configured_word_names(config: dict[str, Any]) -> list[str]:
    words = [*config.get("positive_bin", []), *config.get("negative_bin", [])]
    if not words or any(not isinstance(word, str) or not word for word in words):
        raise ValueError("positive_bin and negative_bin must contain words")
    if any(word != word.strip().lower() for word in words):
        raise ValueError("configured emotional words must be canonical lowercase")
    if len(words) != len(set(words)):
        raise ValueError("configured emotional words must be unique")
    return sorted(words)


def _load_scanner_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    config = _read_json_object(path)
    repository = _repository_from_config(path)
    required = {
        "protocol",
        "model_name",
        "model_revision",
        "dataset_revision",
        "indices_manifest",
        "indices_manifest_sha256",
        "lens_path",
        "lens_sha256",
        "split",
        "generation",
        "readout",
        "calibration",
        "expected_token_ids",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"word-correlation config is missing fields: {missing}")
    if config["protocol"] != "j-lens-rl-jspace-word-correlation-v1":
        raise ValueError("unexpected word-correlation protocol")
    manifest_path = repository / config["indices_manifest"]
    lens_path = repository / config["lens_path"]
    if sha256_file(manifest_path) != config["indices_manifest_sha256"]:
        raise ValueError("exposed curve manifest hash does not match config")
    if sha256_file(lens_path) != config["lens_sha256"]:
        raise ValueError("lens hash does not match config")
    return config, repository


def _curve_indices_and_allocations(
    config: dict[str, Any], repository: Path
) -> tuple[list[int], list[int], list[int]]:
    manifest = _read_json_object(repository / config["indices_manifest"])
    indices = manifest.get("indices")
    if (
        manifest.get("dataset") != config.get("dataset_name")
        or manifest.get("subset") != config.get("dataset_subset")
        or manifest.get("split") != config.get("dataset_split")
        or not isinstance(indices, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise ValueError("curve index manifest metadata or indices are invalid")
    split = config["split"]
    salt = split["salt"]
    ordered = sorted(indices, key=lambda index: (_index_digest(salt, index), index))
    discovery_count = int(split["discovery_prompts"])
    validation_count = int(split["validation_prompts"])
    if discovery_count + validation_count != len(ordered):
        raise ValueError("configured prompt split does not cover the curve manifest")
    discovery = ordered[:discovery_count]
    validation = ordered[discovery_count:]
    expected_hashes = {
        "allocation_order_compact_json_sha256": _canonical_json_sha256(ordered),
        "discovery_indices_compact_json_sha256": _canonical_json_sha256(discovery),
        "validation_indices_compact_json_sha256": _canonical_json_sha256(validation),
        "discovery_manifest_content_sha256": _canonical_json_sha256(
            {
                "dataset": config["dataset_name"],
                "subset": config["dataset_subset"],
                "split": config["dataset_split"],
                "indices": discovery,
            }
        ),
        "validation_manifest_content_sha256": _canonical_json_sha256(
            {
                "dataset": config["dataset_name"],
                "subset": config["dataset_subset"],
                "split": config["dataset_split"],
                "indices": validation,
            }
        ),
    }
    if any(split.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("configured discovery/validation allocation hashes changed")
    return indices, discovery, validation


def _configured_candidates(
    tokenizer: Any, config: dict[str, Any]
) -> list[LexicalCandidate]:
    expected = config["expected_token_ids"]
    candidates: list[LexicalCandidate] = []
    for word in _configured_word_names(config):
        token_ids = tuple(single_token_ids(tokenizer, [word]))
        if list(token_ids) != expected.get(word):
            raise ValueError(
                f"configured token IDs for {word!r} differ from the pinned tokenizer"
            )
        decoded = tuple(_decode_token(tokenizer, token_id) for token_id in token_ids)
        raw = tokenizer.convert_ids_to_tokens(list(token_ids))
        if isinstance(raw, str):
            raw = [raw]
        boundary = tuple(
            token_id
            for token_id, spelling in zip(token_ids, decoded, strict=True)
            if spelling.startswith(" ")
        )
        if not boundary:
            raise ValueError(f"configured word {word!r} has no boundary token")
        candidates.append(
            LexicalCandidate(
                canonical_word=word,
                token_ids=token_ids,
                boundary_token_ids=boundary,
                decoded_variants=decoded,
                tokenizer_tokens=tuple(str(value) for value in raw),
                literal_sequences=tuple(literal_token_sequences(tokenizer, [word])),
            )
        )
    return candidates


def _base_provenance(
    config: dict[str, Any],
    config_path: str | Path,
    calibration_path: str | Path,
) -> dict[str, Any]:
    return {
        "model_revision": config["model_revision"],
        "dataset_revision": config["dataset_revision"],
        "curve_manifest_sha256": config["indices_manifest_sha256"],
        "lens_sha256": config["lens_sha256"],
        "config_sha256": sha256_file(config_path),
        "scanner_sha256": sha256_file(__file__),
        "calibration_sha256": sha256_file(calibration_path),
    }


def _load_model_lens_tokenizer(config: dict[str, Any], repository: Path):
    from jlens import JacobianLens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config["model_revision"],
        dtype=model_dtype(),
        device_map={"": "cuda:0"},
    )
    model.eval()
    lens = JacobianLens.load(str(repository / config["lens_path"]))
    if list(lens.source_layers) != [int(value) for value in config["lens_source_layers"]]:
        raise ValueError("lens source layers differ from config")
    return model, tokenizer, lens


def _calibration_stats(
    calibration: dict[str, Any], candidates: Sequence[LexicalCandidate]
) -> dict[str, tuple[float, float]]:
    payload = calibration.get("candidates")
    if not isinstance(payload, dict) or set(payload) != {
        candidate.canonical_word for candidate in candidates
    }:
        raise ValueError("calibration candidate inventory is incomplete")
    stats: dict[str, tuple[float, float]] = {}
    for candidate in candidates:
        row = payload[candidate.canonical_word]
        if row.get("token_ids") != list(candidate.token_ids):
            raise ValueError(f"calibration token IDs changed for {candidate.canonical_word}")
        mean = float(row["mean"])
        std = float(row["std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"invalid calibration for {candidate.canonical_word}")
        stats[candidate.canonical_word] = (mean, std)
    return stats


@torch.no_grad()
def run_calibration(config_path, output_path):
    """Calibrate every frozen emotional word jointly on pinned WikiText."""
    from datasets import load_dataset

    config, repository = _load_scanner_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output}")
    seed_everything(0)
    model, tokenizer, lens = _load_model_lens_tokenizer(config, repository)
    candidates = _configured_candidates(tokenizer, config)
    calibration_config = config["calibration"]
    corpus = load_dataset(
        calibration_config["corpus"],
        calibration_config["subset"],
        split=calibration_config["split"],
        revision=calibration_config["revision"],
    )
    minimum_words = int(calibration_config["minimum_words"])
    skip = int(calibration_config["skip_qualifying_prompts"])
    count = int(calibration_config["prompts"])
    # Only the first ``skip + count`` qualifying rows can influence the exact
    # fit_lens-compatible slice.  Short-circuiting avoids retaining hundreds
    # of megabytes of irrelevant WikiText strings in a calibration worker.
    qualifying: list[str] = []
    for row in corpus:
        if len(row["text"].split()) >= minimum_words:
            qualifying.append(row["text"])
            if len(qualifying) == skip + count:
                break
    texts = qualifying[skip : skip + count]
    if len(texts) != count:
        raise ValueError("pinned WikiText corpus has too few qualifying calibration prompts")

    raw: dict[str, list[float]] = {
        candidate.canonical_word: [] for candidate in candidates
    }
    norm, head = decoder_parts(model)
    device_jacobians = {
        layer: lens.jacobians[layer].to(model.device)
        for layer in calibration_config["layers"]
    }
    for text in texts:
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(calibration_config["max_seq_len"]),
        ).input_ids.to(model.device)
        output_states = model(
            input_ids, output_hidden_states=True, use_cache=False
        ).hidden_states
        positions = list(
            range(
                int(calibration_config["positions_start"]),
                input_ids.shape[1],
                int(calibration_config["positions_stride"]),
            )
        ) or [input_ids.shape[1] - 1]
        for layer in calibration_config["layers"]:
            hidden = output_states[int(layer) + 1][0, positions].float()
            transported = hidden @ device_jacobians[int(layer)].T
            normalized = norm(transported.to(norm.weight.dtype))
            readout = FullVocabularyReadout.from_hidden(
                normalized,
                head.weight,
                normalizer_chunk_size=int(config["readout"]["vocab_chunk_size"]),
            )
            for candidate in candidates:
                raw[candidate.canonical_word].extend(
                    readout.word_position_log_probs(candidate.token_ids)
                    .detach()
                    .cpu()
                    .tolist()
                )

    candidate_stats: dict[str, Any] = {}
    for candidate in candidates:
        values = np.asarray(raw[candidate.canonical_word], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std())
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"invalid joint calibration for {candidate.canonical_word}")
        candidate_stats[candidate.canonical_word] = {
            "token_ids": list(candidate.token_ids),
            "mean": mean,
            "std": std,
            "observations": int(len(values)),
        }
    payload = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "model": config["model_name"],
        "model_revision": config["model_revision"],
        "wikitext_revision": calibration_config["revision"],
        "lens_sha256": config["lens_sha256"],
        "config_sha256": sha256_file(config_path),
        "scanner_sha256": sha256_file(__file__),
        "calibration_geometry": calibration_config,
        "qualifying_prompt_offset": skip,
        "qualifying_prompt_count": count,
        "calibration_texts_sha256": _canonical_json_sha256(texts),
        "candidate_order": [candidate.canonical_word for candidate in candidates],
        "candidates": candidate_stats,
        "runtime": {
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_version": torch.version.cuda,
            "dtype": str(next(model.parameters()).dtype),
        },
    }
    _write_json_atomic(output, payload)
    return payload


def _validated_calibration(
    config: dict[str, Any],
    config_path: str | Path,
    calibration_path: str | Path,
    candidates: Sequence[LexicalCandidate],
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    calibration = _read_json_object(calibration_path)
    expected = {
        "protocol": config["protocol"],
        "model_revision": config["model_revision"],
        "wikitext_revision": config["calibration"]["revision"],
        "lens_sha256": config["lens_sha256"],
        "config_sha256": sha256_file(config_path),
        "scanner_sha256": sha256_file(__file__),
    }
    if any(calibration.get(key) != value for key, value in expected.items()):
        raise ValueError("calibration provenance differs from the live scanner")
    return calibration, _calibration_stats(calibration, candidates)


def _validate_selection_lock_provenance(
    lock: dict[str, Any],
    config: dict[str, Any],
    config_path: str | Path,
    calibration_path: str | Path,
) -> None:
    """Bind validation to the exact discovery code, inputs, and calibration."""
    expected = {
        "protocol": config["protocol"],
        "config_sha256": sha256_file(config_path),
        "scanner_sha256": sha256_file(__file__),
        "curve_manifest_sha256": config["indices_manifest_sha256"],
        "lens_sha256": config["lens_sha256"],
        "calibration_sha256": sha256_file(calibration_path),
    }
    mismatches = {
        key: {"expected": value, "actual": lock.get(key)}
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"selection-lock provenance mismatch: {mismatches}")
    if not isinstance(lock.get("discovery_aggregate_sha256"), str) or len(
        lock["discovery_aggregate_sha256"]
    ) != 64:
        raise ValueError("selection lock has no discovery aggregate identity")


def _phase_shard_indices(
    config: dict[str, Any],
    repository: Path,
    phase: str,
    shard: int,
) -> list[int]:
    if phase not in {"discovery", "validation"}:
        raise ValueError("phase must be discovery or validation")
    num_shards = int(config["shards"])
    if isinstance(shard, bool) or not isinstance(shard, int) or shard not in range(num_shards):
        raise ValueError("shard index is out of range")
    _, discovery, validation = _curve_indices_and_allocations(config, repository)
    phase_indices = discovery if phase == "discovery" else validation
    selected = phase_indices[shard::num_shards]
    if not selected:
        raise ValueError("configured shard contains no prompts")
    return selected


def _require_empty_output_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    return output


@torch.no_grad()
def run_shard(
    config_path,
    phase,
    shard,
    output_dir,
    calibration_path,
    selection_path=None,
):
    """Generate and score one immutable discovery or validation prompt shard."""
    from datasets import load_dataset
    from torch.nn.utils.rnn import pad_sequence

    config, repository = _load_scanner_config(config_path)
    output = _require_empty_output_directory(output_dir)
    source_indices = _phase_shard_indices(config, repository, phase, shard)
    seed_everything(0)
    model, tokenizer, lens = _load_model_lens_tokenizer(config, repository)
    all_candidates = _configured_candidates(tokenizer, config)
    calibration, calibration_stats = _validated_calibration(
        config, config_path, calibration_path, all_candidates
    )

    selection_lock_sha256: str | None = None
    selection: dict[str, Any] | None = None
    if phase == "discovery":
        if selection_path is not None:
            raise ValueError("discovery must not receive a selection lock")
        candidates = all_candidates
    else:
        if selection_path is None:
            raise ValueError("validation requires an immutable selection lock")
        selection_lock_sha256 = sha256_file(selection_path)
        lock = _read_json_object(selection_path)
        _validate_selection_lock_provenance(
            lock, config, config_path, calibration_path
        )
        selection = lock.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("selection lock has no selected word")
        matches = [
            candidate
            for candidate in all_candidates
            if candidate.canonical_word == selection.get("canonical_word")
        ]
        if len(matches) != 1 or list(matches[0].token_ids) != selection.get("token_ids"):
            raise ValueError("selection lock word or token IDs differ from config")
        selected_mean, selected_std = calibration_stats[matches[0].canonical_word]
        if (
            selection.get("calibration_sha256") != sha256_file(calibration_path)
            or selection.get("calibration_mean") != selected_mean
            or selection.get("calibration_std") != selected_std
        ):
            raise ValueError("selection lock differs from selected-word calibration")
        reward_sign = selection.get("reward_sign")
        if reward_sign not in {-1, 1} or isinstance(reward_sign, bool):
            raise ValueError("selection lock reward sign must be exactly -1 or +1")
        expected_direction = (
            "positive_with_correctness"
            if reward_sign == 1
            else "negative_with_correctness"
        )
        if selection.get("association_direction") != expected_direction:
            raise ValueError("selection lock sign and direction disagree")
        candidates = matches

    dataset = load_dataset(
        config["dataset_name"],
        config["dataset_subset"],
        split=config["dataset_split"],
        revision=config["dataset_revision"],
    )
    if max(source_indices) >= len(dataset):
        raise ValueError("shard contains an out-of-range dataset index")
    generation = config["generation"]
    readout_config = config["readout"]
    rollout_count = int(generation["generations_per_prompt"])
    layer = int(readout_config["source_layer"])
    if layer not in lens.source_layers:
        raise ValueError("configured readout layer is absent from lens")
    jacobian = lens.jacobians[layer].to(model.device)
    norm, head = decoder_parts(model)
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    candidate_words = [candidate.canonical_word for candidate in candidates]
    clip = float(readout_config["clip_standardized_score"])
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("standardized-score clip must be finite and positive")

    prompt_records: list[dict[str, Any]] = []
    rollout_records: list[dict[str, Any]] = []
    atlas_numerator: np.ndarray | None = None
    atlas_score_ss: np.ndarray | None = None
    atlas_label_ss = 0.0
    atlas_mixed_prompts = 0
    atlas_skipped_mixed_prompts = 0
    atlas_rollouts_without_positions = 0
    if phase == "discovery":
        atlas_numerator = np.zeros(head.weight.shape[0], dtype=np.float64)
        atlas_score_ss = np.zeros(head.weight.shape[0], dtype=np.float64)

    for source_index in source_indices:
        row = dataset[int(source_index)]
        prompt = format_prompt(tokenizer, row["question"])
        prompt_seed = deterministic_prompt_seed(
            str(generation["seed_salt"]), int(source_index)
        )
        _seed_grouped_generation(prompt_seed)
        encoded = tokenizer(
            [prompt] * rollout_count,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(generation["max_prompt_tokens"]),
        ).to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        generated = model.generate(
            **encoded,
            do_sample=bool(generation["do_sample"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
            min_new_tokens=int(generation["min_new_tokens"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
        )

        sequences: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        correctness: list[int] = []
        literal_by_rollout: list[list[str]] = []
        completion_texts: list[str] = []
        completion_token_lists: list[list[int]] = []
        for rollout_index in range(rollout_count):
            completion = generated[rollout_index, prompt_width:]
            eos = (completion == tokenizer.eos_token_id).nonzero()
            if eos.numel():
                completion = completion[: int(eos[0].item()) + 1]
            prompt_ids = encoded.input_ids[rollout_index][
                encoded.attention_mask[rollout_index].bool()
            ]
            text = tokenizer.decode(completion, skip_special_tokens=True)
            completion_token_ids = completion.tolist()
            used_words = [
                candidate.canonical_word
                for candidate in candidates
                if contains_token_sequence(
                    completion_token_ids, candidate.literal_sequences
                )
            ]
            item_correct = int(gsm8k_reward(text, row["answer"]))
            sequences.append(torch.cat([prompt_ids, completion]))
            prompt_lengths.append(int(len(prompt_ids)))
            correctness.append(item_correct)
            literal_by_rollout.append(used_words)
            completion_texts.append(text)
            completion_token_lists.append(completion_token_ids)

        padded_ids = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.ones_like(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=0,
        )
        hidden_states = model(
            padded_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        candidate_scores = np.empty((rollout_count, len(candidates)), dtype=np.float64)
        atlas_scores: list[np.ndarray | None] = []
        for rollout_index in range(rollout_count):
            sequence_end = int(attention_mask[rollout_index].sum().item())
            live_ids = padded_ids[rollout_index, :sequence_end]
            common_positions = sampled_response_positions(
                prompt_lengths[rollout_index],
                sequence_end,
                int(readout_config["stride"]),
                float(readout_config["start_fraction"]),
                float(readout_config["end_fraction"]),
                live_ids,
                special_ids,
                bool(readout_config["include_final"]),
            )
            positions_by_word = {
                candidate.canonical_word: masked_candidate_positions(
                    live_ids,
                    prompt_lengths[rollout_index],
                    sequence_end,
                    stride=int(readout_config["stride"]),
                    start_fraction=float(readout_config["start_fraction"]),
                    end_fraction=float(readout_config["end_fraction"]),
                    literal_sequences=candidate.literal_sequences,
                    special_token_ids=special_ids,
                    include_final=bool(readout_config["include_final"]),
                    mask_target_tokens=bool(readout_config["mask_target_tokens"]),
                )
                for candidate in candidates
            }
            materialized = sorted(
                {
                    *common_positions,
                    *(
                        position
                        for positions in positions_by_word.values()
                        for position in positions
                    ),
                }
            )
            if materialized:
                hidden = hidden_states[layer + 1][rollout_index, materialized].float()
                transported = hidden @ jacobian.T
                normalized = norm(transported.to(norm.weight.dtype))
                vocabulary_readout = FullVocabularyReadout.from_hidden(
                    normalized,
                    head.weight,
                    normalizer_chunk_size=int(readout_config["vocab_chunk_size"]),
                )
            else:
                vocabulary_readout = None
            for candidate_index, candidate in enumerate(candidates):
                mean, std = calibration_stats[candidate.canonical_word]
                selected_positions = positions_by_word[candidate.canonical_word]
                if vocabulary_readout is None:
                    raw_score = mean
                else:
                    rows = position_rows(materialized, selected_positions)
                    raw_score = float(
                        exact_word_score(
                            vocabulary_readout,
                            candidate.token_ids,
                            rows,
                            aggregation=str(readout_config["aggregation"]),
                            neutral_score=mean,
                        ).item()
                    )
                candidate_scores[rollout_index, candidate_index] = float(
                    np.clip((raw_score - mean) / std, -clip, clip)
                )
            if phase == "discovery":
                if vocabulary_readout is None or not common_positions:
                    atlas_scores.append(None)
                    atlas_rollouts_without_positions += 1
                else:
                    common_rows = position_rows(materialized, common_positions)
                    atlas_scores.append(
                        vocabulary_readout.aggregate_token_log_probs(
                            common_rows,
                            aggregation=str(readout_config["aggregation"]),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

        labels = np.asarray(correctness, dtype=np.float64)
        mixed = bool(labels.min() != labels.max())
        if phase == "discovery":
            sufficient = atlas_prompt_sufficient(atlas_scores, labels)
            if sufficient is not None:
                numerator, score_ss, label_ss = sufficient
                assert atlas_numerator is not None and atlas_score_ss is not None
                atlas_numerator += numerator
                atlas_score_ss += score_ss
                atlas_label_ss += label_ss
                atlas_mixed_prompts += 1
            elif mixed and any(values is None for values in atlas_scores):
                atlas_skipped_mixed_prompts += 1

        literal_counts = {
            candidate.canonical_word: sum(
                candidate.canonical_word in words for words in literal_by_rollout
            )
            for candidate in candidates
        }
        prompt_records.append(
            {
                "source_index": int(source_index),
                "prompt_seed": prompt_seed,
                "correctness": correctness,
                "mixed_outcomes": mixed,
                "candidate_words": candidate_words,
                "scores": candidate_scores.tolist(),
                "literal_completion_counts": literal_counts,
            }
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for rollout_index in range(rollout_count):
            rollout_records.append(
                {
                    "source_index": int(source_index),
                    "rollout_index": rollout_index,
                    "prompt_seed": prompt_seed,
                    "prompt_sha256": prompt_hash,
                    "completion": completion_texts[rollout_index],
                    "completion_token_ids": completion_token_lists[rollout_index],
                    "correct": bool(correctness[rollout_index]),
                    "literal_target_words": literal_by_rollout[rollout_index],
                }
            )

    _write_jsonl_atomic(output / "prompt_records.jsonl", prompt_records)
    _write_jsonl_atomic(output / "rollouts.jsonl", rollout_records)
    if phase == "discovery":
        assert atlas_numerator is not None and atlas_score_ss is not None
        _write_npy_atomic(output / "atlas_numerator.npy", atlas_numerator)
        _write_npy_atomic(output / "atlas_score_ss.npy", atlas_score_ss)
        _write_json_atomic(
            output / "atlas_sufficient.json",
            {
                "label_ss": atlas_label_ss,
                "mixed_prompts": atlas_mixed_prompts,
                "skipped_mixed_prompts_no_common_positions": (
                    atlas_skipped_mixed_prompts
                ),
                "rollouts_without_common_positions": (
                    atlas_rollouts_without_positions
                ),
                "vocabulary_size": int(len(atlas_numerator)),
            },
        )

    provenance = _base_provenance(config, config_path, calibration_path)
    payload: dict[str, Any] = {
        **provenance,
        "schema_version": 1,
        "protocol": config["protocol"],
        "phase": phase,
        "shard_index": int(shard),
        "num_shards": int(config["shards"]),
        "source_indices": [int(index) for index in source_indices],
        "candidate_words": candidate_words,
        "prompt_count": len(prompt_records),
        "rollout_count": len(rollout_records),
        "mixed_prompt_count": sum(record["mixed_outcomes"] for record in prompt_records),
        "prompt_records_sha256": sha256_file(output / "prompt_records.jsonl"),
        "rollouts_sha256": sha256_file(output / "rollouts.jsonl"),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "runtime": {
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_version": torch.version.cuda,
            "dtype": str(next(model.parameters()).dtype),
        },
    }
    if selection_lock_sha256 is not None:
        payload["selection_lock_sha256"] = selection_lock_sha256
        payload["selection"] = selection
    _write_json_atomic(output / "provenance.json", payload)
    return payload


def _load_phase_records(
    config: dict[str, Any],
    repository: Path,
    config_path: str | Path,
    shard_dirs: Sequence[str | Path],
    calibration_path: str | Path,
    phase: str,
    candidate_words: Sequence[str],
    selection_lock_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    if len(shard_dirs) != int(config["shards"]):
        raise ValueError("merge requires every configured shard")
    expected_provenance = _base_provenance(config, config_path, calibration_path)
    records: list[dict[str, Any]] = []
    summary_indices: list[int] = []
    seen_shards: set[int] = set()
    for raw_directory in shard_dirs:
        directory = Path(raw_directory)
        summary = _read_json_object(directory / "summary.json")
        shard_index = summary.get("shard_index")
        if (
            summary.get("phase") != phase
            or not isinstance(shard_index, int)
            or isinstance(shard_index, bool)
            or shard_index in seen_shards
            or shard_index not in range(int(config["shards"]))
        ):
            raise ValueError(f"invalid or duplicate {phase} shard summary")
        seen_shards.add(shard_index)
        if any(summary.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError(f"{phase} shard provenance mismatch")
        if summary.get("candidate_words") != list(candidate_words):
            raise ValueError(f"{phase} shard candidate order mismatch")
        if selection_lock_sha256 is None:
            if "selection_lock_sha256" in summary:
                raise ValueError("discovery shard unexpectedly used a selection lock")
        elif summary.get("selection_lock_sha256") != selection_lock_sha256:
            raise ValueError("validation shard selection lock mismatch")
        prompt_path = directory / "prompt_records.jsonl"
        if summary.get("prompt_records_sha256") != sha256_file(prompt_path):
            raise ValueError(f"{phase} prompt records changed after shard completion")
        shard_records = _read_jsonl(prompt_path)
        source_indices = summary.get("source_indices")
        if not isinstance(source_indices, list) or {
            int(record.get("source_index")) for record in shard_records
        } != set(source_indices):
            raise ValueError(f"{phase} shard record indices differ from summary")
        summary_indices.extend(int(index) for index in source_indices)
        records.extend(shard_records)
    if seen_shards != set(range(int(config["shards"]))):
        raise ValueError(f"{phase} shard identities are incomplete")
    if len(summary_indices) != len(set(summary_indices)):
        raise ValueError(f"{phase} shards overlap in source indices")
    _, discovery, validation = _curve_indices_and_allocations(config, repository)
    expected_indices = discovery if phase == "discovery" else validation
    if set(summary_indices) != set(expected_indices):
        raise ValueError(f"{phase} shards do not cover the frozen phase allocation")
    order = {source_index: position for position, source_index in enumerate(expected_indices)}
    records.sort(key=lambda record: order[int(record["source_index"])])
    if len(records) != len(expected_indices):
        raise ValueError(f"{phase} has the wrong number of prompt records")

    rollout_count = int(config["generation"]["generations_per_prompt"])
    for record in records:
        scores = np.asarray(record.get("scores"), dtype=np.float64)
        correctness = np.asarray(record.get("correctness"))
        if (
            record.get("candidate_words") != list(candidate_words)
            or scores.shape != (rollout_count, len(candidate_words))
            or correctness.shape != (rollout_count,)
            or not np.isfinite(scores).all()
            or not np.isin(correctness, [0, 1, False, True]).all()
            or record.get("mixed_outcomes")
            != bool(correctness.min() != correctness.max())
        ):
            raise ValueError(f"invalid {phase} prompt score record")
        literal_counts = record.get("literal_completion_counts")
        if (
            not isinstance(literal_counts, dict)
            or set(literal_counts) != set(candidate_words)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in literal_counts.values()
            )
        ):
            raise ValueError(f"invalid {phase} literal audit record")
    return records, expected_indices


def _mixed_group_arrays(
    records: Sequence[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mixed = [record for record in records if record["mixed_outcomes"]]
    if not mixed:
        raise ValueError("phase produced no mixed-correctness prompt groups")
    return (
        [np.asarray(record["scores"], dtype=np.float64) for record in mixed],
        [np.asarray(record["correctness"], dtype=np.float64) for record in mixed],
    )


def merge_discovery(config_path, shard_dirs, calibration_path, output_dir):
    """Merge discovery shards, run max-|r| inference, and select one word."""
    config, repository = _load_scanner_config(config_path)
    output = _require_empty_output_directory(output_dir)
    # Tokenization is needed only for the immutable candidate/token-ID mapping.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    )
    candidates = _configured_candidates(tokenizer, config)
    calibration, calibration_stats = _validated_calibration(
        config, config_path, calibration_path, candidates
    )
    candidate_words = [candidate.canonical_word for candidate in candidates]
    records, expected_indices = _load_phase_records(
        config,
        repository,
        config_path,
        shard_dirs,
        calibration_path,
        "discovery",
        candidate_words,
    )
    grouped_scores, grouped_correctness = _mixed_group_arrays(records)
    correlations = prompt_centered_point_biserial(
        grouped_scores, grouped_correctness
    )
    _, _, _, score_ss, _ = _centered_correlation_parts(
        grouped_scores, grouped_correctness
    )
    literal_counts = {
        word: sum(record["literal_completion_counts"][word] for record in records)
        for word in candidate_words
    }
    eligible = np.asarray(
        [
            literal_counts[word] == 0
            and score_ss[index] > 0.0
            and math.isfinite(float(correlations[index]))
            for index, word in enumerate(candidate_words)
        ],
        dtype=bool,
    )
    if not eligible.any():
        raise ValueError("no discovery candidate passed calibration/variance/literal audits")
    selection = select_discovery_candidate(candidates, correlations, eligible)
    permutation_config = config["selection"]
    permutation = within_prompt_permutation_test(
        grouped_scores,
        grouped_correctness,
        draws=int(permutation_config["permutation_draws"]),
        seed=int(permutation_config["permutation_seed"]),
        eligible_indices=np.flatnonzero(eligible).tolist(),
    )
    selected_index = int(selection.pop("candidate_index"))
    selected_mean, selected_std = calibration_stats[selection["canonical_word"]]
    selection["calibration_mean"] = selected_mean
    selection["calibration_std"] = selected_std
    selection["calibration_sha256"] = sha256_file(calibration_path)
    selection["discovery_maxT_p"] = float(
        permutation["max_abs_adjusted_p"][selected_index]
    )
    selection["discovery_unadjusted_p"] = float(
        permutation["unadjusted_two_sided_p"][selected_index]
    )
    selection["discovery_mixed_prompts"] = len(grouped_scores)
    selection["literal_target_completions"] = literal_counts[
        selection["canonical_word"]
    ]

    total_rollouts = len(records) * int(config["generation"]["generations_per_prompt"])
    candidate_results = []
    for index, candidate in enumerate(candidates):
        mean, std = calibration_stats[candidate.canonical_word]
        candidate_results.append(
            {
                **candidate.to_dict(),
                "correlation": float(correlations[index]),
                "score_sum_squares": float(score_ss[index]),
                "eligible": bool(eligible[index]),
                "literal_target_completions": literal_counts[candidate.canonical_word],
                "literal_target_completion_rate": (
                    literal_counts[candidate.canonical_word] / total_rollouts
                ),
                "calibration_mean": mean,
                "calibration_std": std,
                "unadjusted_two_sided_p": float(
                    permutation["unadjusted_two_sided_p"][index]
                ),
                "max_abs_adjusted_p": float(
                    permutation["max_abs_adjusted_p"][index]
                ),
            }
        )
    _write_json_atomic(output / "candidate_results.json", candidate_results)
    _write_json_atomic(output / "permutation.json", permutation)
    payload = {
        **_base_provenance(config, config_path, calibration_path),
        "schema_version": 1,
        "protocol": config["protocol"],
        "phase": "discovery",
        "source_indices": [int(index) for index in expected_indices],
        "prompt_count": len(records),
        "mixed_prompt_count": len(grouped_scores),
        "candidate_count": len(candidates),
        "primary_statistic": config["selection"]["primary_statistic"],
        "selection": selection,
        "permutation": {
            "method": permutation["method"],
            "draws": permutation["draws"],
            "seed": permutation["seed"],
            "observed_family_max_abs": permutation["observed_family_max_abs"],
            "selected_max_abs_adjusted_p": selection["discovery_maxT_p"],
            "is_selection_gate": bool(config["selection"]["permutation_is_gate"]),
        },
        "candidate_results": candidate_results,
        "calibration_candidate_count": len(calibration["candidates"]),
    }
    _write_json_atomic(output / "merge_provenance.json", payload)
    return payload


def merge_validation(
    config_path,
    shard_dirs,
    calibration_path,
    selection_path,
    output_dir,
):
    """Merge locked-word validation shards and run preregistered inference."""
    config, repository = _load_scanner_config(config_path)
    output = _require_empty_output_directory(output_dir)
    lock = _read_json_object(selection_path)
    _validate_selection_lock_provenance(
        lock, config, config_path, calibration_path
    )
    selection = lock.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection lock contains no selection")
    selection_lock_sha256 = sha256_file(selection_path)
    candidate_word = selection.get("canonical_word")
    expected_ids = config["expected_token_ids"].get(candidate_word)
    if selection.get("token_ids") != expected_ids:
        raise ValueError("selection-lock token IDs differ from config")
    calibration = _read_json_object(calibration_path)
    calibration_row = calibration.get("candidates", {}).get(candidate_word, {})
    if (
        selection.get("calibration_sha256") != sha256_file(calibration_path)
        or selection.get("calibration_mean") != calibration_row.get("mean")
        or selection.get("calibration_std") != calibration_row.get("std")
    ):
        raise ValueError("selection lock differs from selected-word calibration")
    candidate_words = [str(candidate_word)]
    records, expected_indices = _load_phase_records(
        config,
        repository,
        config_path,
        shard_dirs,
        calibration_path,
        "validation",
        candidate_words,
        selection_lock_sha256,
    )
    grouped_scores, grouped_correctness = _mixed_group_arrays(records)
    correlation = float(
        prompt_centered_point_biserial(grouped_scores, grouped_correctness)[0]
    )
    reward_sign = selection.get("reward_sign")
    if reward_sign not in {-1, 1} or isinstance(reward_sign, bool):
        raise ValueError("selection lock reward sign is invalid")
    validation_config = config["validation"]
    permutation = within_prompt_permutation_test(
        grouped_scores,
        grouped_correctness,
        draws=int(validation_config["permutation_draws"]),
        seed=int(validation_config["permutation_seed"]),
        locked_sign=int(reward_sign),
    )
    bootstrap = prompt_cluster_bootstrap_correlation(
        grouped_scores,
        grouped_correctness,
        draws=int(validation_config["cluster_bootstrap_draws"]),
        seed=int(validation_config["cluster_bootstrap_seed"]),
    )
    signed_correlation = int(reward_sign) * correlation
    interval_in_locked_direction = (
        bootstrap["correlation_ci_low"] > 0.0
        if reward_sign == 1
        else bootstrap["correlation_ci_high"] < 0.0
    )
    stable = bool(
        signed_correlation > 0.0
        and permutation["one_sided_p"] <= 0.05
        and interval_in_locked_direction
    )
    discovery_max_t_p = float(selection.get("discovery_maxT_p", 1.0))
    association_pass = bool(stable and discovery_max_t_p <= 0.05)
    literal_count = sum(
        record["literal_completion_counts"][candidate_word] for record in records
    )
    inference = {
        "selected_word": candidate_word,
        "reward_sign": reward_sign,
        "correlation": correlation,
        "signed_locked_direction_correlation": signed_correlation,
        "mixed_prompt_count": len(grouped_scores),
        "literal_target_completions": literal_count,
        "permutation": permutation,
        "cluster_bootstrap": bootstrap,
        "interval_wholly_in_locked_direction": interval_in_locked_direction,
        "stable_association": stable,
        "discovery_maxT_p": discovery_max_t_p,
        "association_pass": association_pass,
        "is_rl_gate": bool(validation_config["is_rl_gate"]),
    }
    _write_json_atomic(output / "validation_inference.json", inference)
    payload = {
        **_base_provenance(config, config_path, calibration_path),
        "schema_version": 1,
        "protocol": config["protocol"],
        "phase": "validation",
        "source_indices": [int(index) for index in expected_indices],
        "prompt_count": len(records),
        "mixed_prompt_count": len(grouped_scores),
        "selection_lock_sha256": selection_lock_sha256,
        "selection": selection,
        "inference": inference,
    }
    _write_json_atomic(output / "merge_provenance.json", payload)
    return payload


def build_atlas(config_path, shard_dirs, calibration_path, output_dir):
    """Build a descriptive individual-token atlas from discovery sufficients."""
    from transformers import AutoTokenizer

    config, repository = _load_scanner_config(config_path)
    output = _require_empty_output_directory(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    )
    candidates = _configured_candidates(tokenizer, config)
    _validated_calibration(config, config_path, calibration_path, candidates)
    candidate_words = [candidate.canonical_word for candidate in candidates]
    _, expected_indices = _load_phase_records(
        config,
        repository,
        config_path,
        shard_dirs,
        calibration_path,
        "discovery",
        candidate_words,
    )
    numerator: np.ndarray | None = None
    score_ss: np.ndarray | None = None
    label_ss = 0.0
    mixed_prompts = 0
    skipped_mixed_prompts = 0
    rollouts_without_positions = 0
    for raw_directory in shard_dirs:
        directory = Path(raw_directory)
        shard_numerator = np.load(
            directory / "atlas_numerator.npy", allow_pickle=False
        )
        shard_score_ss = np.load(
            directory / "atlas_score_ss.npy", allow_pickle=False
        )
        metadata = _read_json_object(directory / "atlas_sufficient.json")
        if (
            shard_numerator.ndim != 1
            or shard_score_ss.shape != shard_numerator.shape
            or int(metadata.get("vocabulary_size", -1)) != len(shard_numerator)
            or not np.isfinite(shard_numerator).all()
            or not np.isfinite(shard_score_ss).all()
            or (shard_score_ss < 0.0).any()
        ):
            raise ValueError("invalid discovery atlas sufficient statistics")
        if numerator is None:
            numerator = np.zeros_like(shard_numerator, dtype=np.float64)
            score_ss = np.zeros_like(shard_score_ss, dtype=np.float64)
        if shard_numerator.shape != numerator.shape:
            raise ValueError("atlas vocabulary size differs across shards")
        numerator += shard_numerator
        assert score_ss is not None
        score_ss += shard_score_ss
        label_ss += float(metadata["label_ss"])
        mixed_prompts += int(metadata["mixed_prompts"])
        skipped_mixed_prompts += int(
            metadata.get("skipped_mixed_prompts_no_common_positions", 0)
        )
        rollouts_without_positions += int(
            metadata.get("rollouts_without_common_positions", 0)
        )
    if numerator is None or score_ss is None or label_ss <= 0.0:
        raise ValueError("atlas has no mixed-prompt sufficient statistics")
    correlations = _safe_correlations(numerator, score_ss, label_ss)

    atlas_config = config["atlas"]
    lexical_regex = re.compile(str(atlas_config["lexical_regex"]))
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    vocabulary = tokenizer.get_vocab()
    id_to_token: dict[int, str] = {}
    for raw_token, token_id in sorted(vocabulary.items()):
        id_to_token.setdefault(int(token_id), str(raw_token))
    decoded: dict[int, str] = {
        token_id: _decode_token(tokenizer, token_id)
        for token_id in sorted(id_to_token)
        if token_id not in special_ids and token_id < len(correlations)
    }
    boundary_words = {
        spelling.strip().lower()
        for spelling in decoded.values()
        if spelling.startswith(" ") and lexical_regex.fullmatch(spelling)
    }
    records: list[dict[str, Any]] = []
    for token_id, spelling in decoded.items():
        if lexical_regex.fullmatch(spelling) is None:
            continue
        canonical = spelling.strip().lower()
        if atlas_config.get("require_leading_space_variant") and canonical not in boundary_words:
            continue
        correlation = float(correlations[token_id])
        if score_ss[token_id] <= 0.0 or not math.isfinite(correlation):
            continue
        records.append(
            {
                "token_id": token_id,
                "tokenizer_token": id_to_token[token_id],
                "decoded": spelling,
                "canonical_word": canonical,
                "correlation": correlation,
            }
        )
    limit = int(atlas_config["top_per_direction"])
    positive = sorted(
        (record for record in records if record["correlation"] > 0.0),
        key=lambda record: (
            -record["correlation"],
            record["canonical_word"],
            record["token_id"],
        ),
    )[:limit]
    negative = sorted(
        (record for record in records if record["correlation"] < 0.0),
        key=lambda record: (
            record["correlation"],
            record["canonical_word"],
            record["token_id"],
        ),
    )[:limit]
    _write_json_atomic(output / "top_positive.json", positive)
    _write_json_atomic(output / "top_negative.json", negative)
    payload = {
        **_base_provenance(config, config_path, calibration_path),
        "schema_version": 1,
        "protocol": config["protocol"],
        "phase": "discovery",
        "source_indices": [int(index) for index in expected_indices],
        "mixed_prompt_count": mixed_prompts,
        "skipped_mixed_prompt_count_no_common_positions": skipped_mixed_prompts,
        "rollout_count_without_common_positions": rollouts_without_positions,
        "role": atlas_config["role"],
        "unit": atlas_config["unit"],
        "lexical_regex": atlas_config["lexical_regex"],
        "require_leading_space_variant": bool(
            atlas_config["require_leading_space_variant"]
        ),
        "eligible_token_count": len(records),
        "top_per_direction": limit,
        "top_positive": positive,
        "top_negative": negative,
        "descriptive_only": True,
        "multiplicity_adjusted": False,
        "can_alter_selected_emotional_word": False,
    }
    _write_json_atomic(output / "atlas_provenance.json", payload)
    return payload
