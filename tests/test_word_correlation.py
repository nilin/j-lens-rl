import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from jlens_rl.word_correlation import (
    _seed_grouped_generation,
    _validate_selection_lock_provenance,
    AssociationSummary,
    FullVocabularyReadout,
    LexicalCandidate,
    aggregate_associations,
    atlas_prompt_sufficient,
    dense_response_positions,
    deterministic_prompt_seed,
    deterministic_prompt_split,
    deterministic_rollout_seed,
    deterministic_shards,
    exact_word_score,
    lexical_candidates,
    masked_candidate_positions,
    position_rows,
    prompt_centered_point_biserial,
    prompt_cluster_bootstrap_correlation,
    prompt_centered_association,
    select_emotional_candidate,
    select_discovery_candidate,
    within_prompt_permutation_test,
)


class FakeTokenizer:
    _vocabulary = {
        "raw_happy_boundary": 0,
        "raw_Happy": 1,
        "raw_HAPPY_boundary": 2,
        "raw_sad_boundary": 3,
        "raw_sad": 4,
        "raw_fragment": 5,
        "raw_multiword": 6,
        "raw_replacement": 7,
        "raw_special": 8,
        "raw_unbounded": 9,
    }
    _decoded = {
        0: " happy",
        1: "Happy",
        2: " HAPPY",
        3: " sad",
        4: "sad",
        5: "x",
        6: "two words",
        7: "\ufffd",
        8: "<special>",
        9: "alone",
    }
    _encoded = {
        "happy": [10, 11],
        " happy": [0],
        "Happy": [1],
        " Happy": [12, 13],
        "HAPPY": [14, 15],
        " HAPPY": [2],
        "sad": [4],
        " sad": [3],
        "Sad": [16, 17],
        " Sad": [18, 19],
        "SAD": [20, 21],
        " SAD": [22, 23],
        "alone": [9],
        " alone": [24, 25],
        "Alone": [26, 27],
        " Alone": [28, 29],
        "ALONE": [30, 31],
        " ALONE": [32, 33],
    }

    all_special_ids = [8]

    def get_vocab(self):
        return dict(self._vocabulary)

    def decode(self, token_ids, **kwargs):
        assert len(token_ids) == 1
        return self._decoded[token_ids[0]]

    def encode(self, text, add_special_tokens=False):
        return self._encoded.get(text, [90, 91])


def _candidate(word, token_ids, literal_sequences=()):
    return LexicalCandidate(
        canonical_word=word,
        token_ids=tuple(token_ids),
        boundary_token_ids=(token_ids[0],),
        decoded_variants=(f" {word}",),
        tokenizer_tokens=(f"raw_{word}",),
        literal_sequences=tuple(tuple(sequence) for sequence in literal_sequences),
    )


def test_deterministic_split_shards_and_rollout_seeds_are_order_independent():
    indices = list(range(100, 112))
    split = deterministic_prompt_split(indices)
    reversed_split = deterministic_prompt_split(reversed(indices))
    assert split == reversed_split
    assert list(split.values()).count("discovery") == 6
    assert list(split.values()).count("replication") == 6

    shards = deterministic_shards(indices, 5)
    assert shards == deterministic_shards(reversed(indices), 5)
    assert sorted(index for shard in shards for index in shard) == indices
    assert max(map(len, shards)) - min(map(len, shards)) <= 1

    seeds = {
        deterministic_rollout_seed(167, source_index, rollout)
        for source_index in indices
        for rollout in range(8)
    }
    assert len(seeds) == len(indices) * 8
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert deterministic_rollout_seed(167, 100, 0) == deterministic_rollout_seed(
        167, 100, 0
    )
    expected_prompt_seed = int.from_bytes(
        hashlib.sha256(b"salt:100").digest()[:8], "big"
    ) >> 1
    assert deterministic_prompt_seed("salt", 100) == expected_prompt_seed
    _seed_grouped_generation(expected_prompt_seed)
    assert torch.initial_seed() == expected_prompt_seed


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: deterministic_prompt_split([1]), "partition empty"),
        (lambda: deterministic_prompt_split([1, 1]), "unique"),
        (lambda: deterministic_shards([1, 2], 3), "cannot exceed"),
        (lambda: deterministic_rollout_seed(-1, 1, 1), "base_seed"),
    ],
)
def test_deterministic_assignment_rejects_invalid_inputs(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_lexical_candidates_map_exact_reward_variants_and_filter_fragments():
    candidates = lexical_candidates(FakeTokenizer())
    assert [candidate.canonical_word for candidate in candidates] == ["happy", "sad"]
    happy, sad = candidates
    assert happy.token_ids == (0, 1, 2)
    assert happy.boundary_token_ids == (0, 2)
    assert happy.decoded_variants == (" happy", "Happy", " HAPPY")
    assert happy.tokenizer_tokens == (
        "raw_happy_boundary",
        "raw_Happy",
        "raw_HAPPY_boundary",
    )
    assert (10, 11) in happy.literal_sequences
    assert sad.token_ids == (3, 4)
    assert "alone" not in {candidate.canonical_word for candidate in candidates}
    assert happy.to_dict()["lexical_filter_version"] == "ascii-boundary-word-v1"


def test_full_vocabulary_readout_matches_full_softmax_and_exact_group_mass():
    hidden = torch.tensor([[0.2, -0.1, 0.4], [1.0, 0.5, -0.5]])
    weight = torch.tensor(
        [
            [0.2, 0.1, -0.4],
            [1.0, -0.5, 0.2],
            [-0.3, 0.7, 0.4],
            [0.8, 0.1, -0.2],
            [-0.4, -0.6, 0.9],
        ]
    )
    readout = FullVocabularyReadout.from_hidden(hidden, weight)
    chunked = FullVocabularyReadout.from_hidden(
        hidden, weight, normalizer_chunk_size=2
    )
    expected = torch.log_softmax(hidden @ weight.T, dim=-1)
    torch.testing.assert_close(readout.token_log_probs(), expected)
    torch.testing.assert_close(chunked.token_log_probs(), expected)
    torch.testing.assert_close(
        readout.word_position_log_probs([1, 3]),
        torch.logsumexp(expected[:, [1, 3]], dim=-1),
    )
    torch.testing.assert_close(
        readout.aggregate_token_log_probs([0], aggregation="last"), expected[0]
    )
    torch.testing.assert_close(
        exact_word_score(readout, [1, 3]),
        torch.logsumexp(expected[:, [1, 3]], dim=-1).mean(),
    )


def test_exact_word_score_uses_explicit_neutral_when_every_position_is_masked():
    readout = FullVocabularyReadout.from_logits(torch.tensor([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match="neutral_score"):
        exact_word_score(readout, [1], [])
    torch.testing.assert_close(
        exact_word_score(readout, [1], [], neutral_score=-4.25),
        torch.tensor(-4.25),
    )


def test_candidate_specific_masking_maps_into_one_dense_logits_matrix():
    ids = torch.zeros(110, dtype=torch.long)
    ids[81:83] = torch.tensor([11, 12])
    dense = dense_response_positions(
        ids,
        10,
        110,
        start_fraction=0.5,
        end_fraction=1.0,
    )
    selected = masked_candidate_positions(
        ids,
        10,
        110,
        stride=20,
        start_fraction=0.5,
        end_fraction=1.0,
        literal_sequences=[(11, 12)],
    )
    assert selected == [60, 100]
    rows = position_rows(dense, selected)
    assert [dense[row] for row in rows] == selected
    with pytest.raises(ValueError, match="not materialized"):
        position_rows(dense, [10_000])


def test_descriptive_atlas_skips_prompt_with_a_positionless_rollout():
    scores = [
        np.array([-3.0, -2.0]),
        None,
        np.array([-1.0, -4.0]),
    ]
    assert atlas_prompt_sufficient(scores, [0, 1, 0]) is None

    complete = atlas_prompt_sufficient(
        [np.array([1.0, 4.0]), np.array([3.0, 2.0])], [0, 1]
    )
    assert complete is not None
    numerator, score_ss, label_ss = complete
    np.testing.assert_allclose(numerator, [1.0, -1.0])
    np.testing.assert_allclose(score_ss, [2.0, 2.0])
    assert label_ss == 0.5


def test_prompt_centering_and_cluster_aggregation_match_direct_calculation():
    scores = np.array(
        [
            [1.0, 8.0],
            [3.0, 6.0],
            [5.0, 4.0],
            [2.0, 7.0],
        ]
    )
    labels = [0, 1, 1, 0]
    association = prompt_centered_association(scores, labels)
    expected_difference = np.array([2.5, -2.5])
    np.testing.assert_allclose(
        association.mean_correct_minus_incorrect, expected_difference
    )
    p = 0.5
    np.testing.assert_allclose(
        association.centered_covariance, p * (1 - p) * expected_difference
    )

    differences = np.array([[2.5, -2.5], [1.5, -1.0], [2.0, -1.5]])
    summary = aggregate_associations(differences)
    np.testing.assert_allclose(summary.mean_difference, differences.mean(axis=0))
    np.testing.assert_allclose(
        summary.standard_error, differences.std(axis=0, ddof=1) / math.sqrt(3)
    )
    assert summary.positive_fraction.tolist() == [1.0, 0.0]
    assert summary.negative_fraction.tolist() == [0.0, 1.0]

    with pytest.raises(ValueError, match="mixed outcomes"):
        prompt_centered_association(scores, [1, 1, 1, 1])


def test_selection_ranks_only_discovery_and_requires_replication_direction():
    candidates = [
        _candidate("happy", [10]),
        _candidate("sad", [20, 21]),
        _candidate("calm", [30]),
    ]
    discovery = AssociationSummary(
        n_prompts=100,
        mean_difference=np.array([3.0, -2.0, 0.5]),
        standard_deviation=np.ones(3),
        standard_error=np.ones(3),
        t_statistic=np.array([10.0, -8.0, 2.0]),
        positive_fraction=np.array([1.0, 0.0, 0.7]),
        negative_fraction=np.array([0.0, 1.0, 0.3]),
    )
    replication = AssociationSummary(
        n_prompts=100,
        # The discovery winner reverses, so it is ineligible.  `sad` must win
        # even though `calm` has the larger replication t statistic.
        mean_difference=np.array([-0.1, -0.3, 0.2]),
        standard_deviation=np.ones(3),
        standard_error=np.ones(3),
        t_statistic=np.array([-1.0, -2.0, 20.0]),
        positive_fraction=np.array([0.4, 0.2, 0.9]),
        negative_fraction=np.array([0.6, 0.8, 0.1]),
    )
    selected = select_emotional_candidate(
        candidates,
        discovery,
        replication,
        {"happy", "sad", "calm"},
    )
    assert selected["canonical_word"] == "sad"
    assert selected["reward_sign"] == -1
    assert selected["association_direction"] == "negative_with_correctness"
    assert selected["token_ids"] == [20, 21]
    assert selected["discovery"]["t_statistic"] == -8.0

    with pytest.raises(ValueError, match="no emotional candidate"):
        select_emotional_candidate(
            candidates,
            discovery,
            replication,
            {"happy"},
            min_abs_replication_t=2.0,
        )


def test_prompt_centered_point_biserial_removes_prompt_level_offsets():
    # The second prompt has a huge additive offset, which must not affect the
    # within-prompt correlation. Candidate 0 tracks correctness; candidate 1
    # tracks it in the opposite direction.
    scores = [
        np.array([[0.0, 3.0], [2.0, 1.0], [3.0, 0.0], [1.0, 2.0]]),
        np.array([[101.0, 102.0], [103.0, 100.0], [100.0, 103.0], [102.0, 101.0]]),
    ]
    labels = [np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1])]
    actual = prompt_centered_point_biserial(scores, labels)
    centered_scores = np.concatenate(
        [group - group.mean(axis=0) for group in scores], axis=0
    )
    centered_labels = np.concatenate(
        [group - group.mean() for group in labels], axis=0
    )
    expected = np.array(
        [
            np.corrcoef(centered_scores[:, index], centered_labels)[0, 1]
            for index in range(2)
        ]
    )
    np.testing.assert_allclose(actual, expected)
    assert actual[0] > 0 and actual[1] < 0


def test_permutation_and_cluster_bootstrap_are_seeded_and_directional():
    scores = [
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([[10.0], [13.0], [11.0], [12.0]]),
        np.array([[5.0], [8.0], [6.0], [7.0]]),
    ]
    labels = [
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
    ]
    first = within_prompt_permutation_test(
        scores, labels, draws=99, seed=123, locked_sign=1, batch_size=17
    )
    second = within_prompt_permutation_test(
        scores, labels, draws=99, seed=123, locked_sign=1, batch_size=17
    )
    assert first == second
    assert 0.0 < first["one_sided_p"] <= 1.0
    assert first["observed_correlation"] > 0.0

    family_scores = [np.column_stack([group[:, 0], -group[:, 0]]) for group in scores]
    family = within_prompt_permutation_test(
        family_scores, labels, draws=50, seed=9, eligible_indices=[0, 1]
    )
    assert len(family["max_abs_adjusted_p"]) == 2
    assert family["max_abs_adjusted_p"][0] == family["max_abs_adjusted_p"][1]

    bootstrap = prompt_cluster_bootstrap_correlation(
        scores, labels, draws=200, seed=7
    )
    assert bootstrap["observed_correlation"] > 0.0
    assert bootstrap["correlation_ci_low"] <= bootstrap["correlation_ci_high"]


def test_discovery_selection_uses_absolute_correlation_and_lexical_tie_break():
    candidates = [
        _candidate("wow", [1]),
        _candidate("sad", [2]),
        _candidate("yay", [3]),
    ]
    selected = select_discovery_candidate(
        candidates,
        [0.4, -0.8, 0.8],
        [True, True, True],
    )
    # `sad` and `yay` tie exactly in |r|, so canonical lexical order wins.
    assert selected["canonical_word"] == "sad"
    assert selected["reward_sign"] == -1
    assert selected["association_direction"] == "negative_with_correctness"
    assert selected["token_ids"] == [2]


def test_selection_lock_binds_validation_to_original_calibration(tmp_path):
    config_path = tmp_path / "config.json"
    calibration_path = tmp_path / "calibration.json"
    config_path.write_text('{"frozen": true}\n')
    calibration_path.write_text('{"mean": 1}\n')
    config = {
        "protocol": "j-lens-rl-jspace-word-correlation-v1",
        "indices_manifest_sha256": "curve-hash",
        "lens_sha256": "lens-hash",
    }
    lock = {
        "protocol": config["protocol"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "scanner_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "src/jlens_rl/word_correlation.py").read_bytes()
        ).hexdigest(),
        "curve_manifest_sha256": "curve-hash",
        "lens_sha256": "lens-hash",
        "calibration_sha256": hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
        "discovery_aggregate_sha256": "a" * 64,
    }
    _validate_selection_lock_provenance(
        lock, config, config_path, calibration_path
    )
    calibration_path.write_text('{"mean": 2}\n')
    with pytest.raises(ValueError, match="selection-lock provenance"):
        _validate_selection_lock_provenance(
            lock, config, config_path, calibration_path
        )
