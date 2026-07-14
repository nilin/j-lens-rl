import copy

import pytest
import torch

from jlens_rl.reward import (
    causally_excluded_positions,
    contains_token_sequence,
    literal_token_sequences,
    literal_variants,
    sampled_response_positions,
    single_token_ids,
    target_log_probs,
    validate_calibration_metadata,
)


class FakeTokenizer:
    table = {
        "solved": [7],
        " solved": [8],
        "Solved": [9],
        " Solved": [10],
        "SOLVED": [11, 12],
        " SOLVED": [13, 14],
    }

    def encode(self, text, add_special_tokens=False):
        return self.table.get(text, [1, 2])


def test_single_token_variants():
    assert literal_variants(["solved"]) == [
        "solved", " solved", "Solved", " Solved", "SOLVED", " SOLVED"
    ]
    assert single_token_ids(FakeTokenizer(), ["solved"]) == [7, 8, 9, 10]


def test_literal_sequences_include_multitoken_case_and_context_spellings():
    sequences = literal_token_sequences(FakeTokenizer(), ["solved"])
    assert (11, 12) in sequences
    assert (13, 14) in sequences
    assert contains_token_sequence([99, 11, 12, 98], sequences)
    assert not contains_token_sequence([99, 11, 98], sequences)


def test_target_log_probs_matches_full_softmax_across_chunks():
    head = torch.nn.Linear(3, 7, bias=False)
    hidden = torch.tensor([[0.2, -0.1, 0.4], [1.0, 0.5, -0.5]])
    expected = torch.logsumexp(
        torch.log_softmax(head(hidden), dim=-1)[:, [1, 5]], dim=-1
    )
    actual = target_log_probs(hidden, head, [1, 5], chunk_size=2)
    torch.testing.assert_close(actual, expected)


def test_fractional_windows_are_disjoint_for_odd_response_lengths():
    early = sampled_response_positions(10, 49, 20, 0.0, 0.5)
    late = sampled_response_positions(10, 49, 20, 0.5, 1.0)
    assert early == [28]
    assert late == [29]
    assert set(early).isdisjoint(late)


def test_masking_excludes_positions_that_predict_or_contain_literal_sequence():
    ids = torch.tensor([0, 0, 11, 12, 0, 7, 0])
    assert causally_excluded_positions(
        ids, len(ids), {7}, [(11, 12)]
    ) == {1, 2, 3, 4, 5}


def test_sampling_masks_predecessor_of_multitoken_literal():
    ids = torch.zeros(110, dtype=torch.long)
    ids[81:83] = torch.tensor([11, 12])
    early = sampled_response_positions(
        10, 110, 20, 0.0, 0.5, ids, excluded_sequences=[(11, 12)]
    )
    late = sampled_response_positions(
        10, 110, 20, 0.5, 1.0, ids, excluded_sequences=[(11, 12)]
    )
    assert early == [29, 49]
    assert late == [60, 100]
    assert set(early).isdisjoint(late)


def test_calibration_metadata_must_match_live_reward_configuration():
    metadata = {
        "mean": -2.0,
        "std": 0.5,
        "token_ids": [7, 8],
        "target_words": ["solved"],
        "layers": [8, 14],
        "model": "model/name",
        "model_revision": "abc123",
        "lens_sha256": "lens123",
    }
    assert validate_calibration_metadata(
        metadata,
        target_words=["solved"],
        token_ids=[8, 7],
        lens_layers=[8, 14],
        expected_model="model/name",
        expected_model_revision="abc123",
        expected_lens_sha256="lens123",
    ) == (-2.0, 0.5)

    mismatches = {
        "target_words": ["happy"],
        "token_ids": [7, 9],
        "layers": [8, 20],
        "model": "other/model",
        "model_revision": "other-revision",
        "lens_sha256": "other-lens",
    }
    for field, value in mismatches.items():
        bad = copy.deepcopy(metadata)
        bad[field] = value
        with pytest.raises(ValueError, match=field):
            validate_calibration_metadata(
                bad,
                target_words=["solved"],
                token_ids=[7, 8],
                lens_layers=[8, 14],
                expected_model="model/name",
                expected_model_revision="abc123",
                expected_lens_sha256="lens123",
            )

    missing_lens_identity = copy.deepcopy(metadata)
    missing_lens_identity.pop("lens_sha256")
    with pytest.raises(ValueError, match="lens_sha256"):
        validate_calibration_metadata(
            missing_lens_identity,
            target_words=["solved"],
            token_ids=[7, 8],
            lens_layers=[8, 14],
            expected_model="model/name",
            expected_model_revision="abc123",
            expected_lens_sha256="lens123",
        )


@pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
def test_calibration_standard_deviation_must_be_positive_and_finite(std):
    metadata = {
        "mean": -2.0,
        "std": std,
        "token_ids": [7],
        "target_words": ["solved"],
        "layers": [8],
        "model": "model/name",
    }
    with pytest.raises(ValueError, match="std"):
        validate_calibration_metadata(
            metadata,
            target_words=["solved"],
            token_ids=[7],
            lens_layers=[8],
            expected_model="model/name",
        )
