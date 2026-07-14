import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from jlens_rl.eval import evaluate
from jlens_rl.paired_eval import (
    SCHEMA_VERSION,
    compare_evaluation_records,
    compare_multiple_adapters,
    crossed_seed_item_bootstrap_ci,
    difference_in_differences,
    exact_mcnemar_p,
    literal_target_matches,
    load_index_manifest,
    paired_bootstrap_ci,
)


def _provenance(label: str, seed: int | None, adapter: str | None) -> dict:
    return {
        "run_label": label,
        "model": {
            "name": "model",
            "configured_revision": "model-revision",
            "resolved_revision": "model-revision",
            "dtype": "torch.float32",
        },
        "adapter": None if adapter is None else {"sha256": adapter},
        "experiment": {"training_seed": seed},
        "selection": {
            "index_manifest": {
                "path": "/machine-specific/manifest.json",
                "sha256": "manifest-sha",
                "dataset": "openai/gsm8k",
                "subset": "main",
                "split": "train",
                "count": 8,
            }
        },
    }


def _records(correctness, *, label, seed, adapter, target_words=("error",)):
    provenance = _provenance(label, seed, adapter)
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": {
                "name": "openai/gsm8k",
                "subset": "main",
                "split": "train",
                "revision": "dataset-revision",
                "fingerprint": "dataset-fingerprint",
            },
            "source_index": index,
            "prompt_sha256": f"prompt-{index}",
            "prediction": str(index),
            "correct": bool(value),
            "target_words": list(target_words),
            "generation": {
                "do_sample": False,
                "max_prompt_tokens": 384,
                "max_new_tokens": 256,
                "padding_side": "left",
            },
            "provenance": provenance,
        }
        for index, value in enumerate(correctness)
    ]


def test_exact_mcnemar_known_values_and_seed_sign_test():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(0, 3) == 0.25
    assert exact_mcnemar_p(1, 5) == 0.21875
    # Six positive seed effects and no negative effects is the first all-win
    # result below 0.05 under a two-sided exact sign test.
    assert exact_mcnemar_p(0, 6) == 0.03125


def test_paired_and_crossed_bootstraps_are_deterministic():
    base = [0, 0, 1, 0, 1, 0]
    adapter = [1, 0, 1, 1, 0, 0]
    first = paired_bootstrap_ci(base, adapter, samples=1_000, seed=91)
    second = paired_bootstrap_ci(base, adapter, samples=1_000, seed=91)
    assert first == second

    effects = np.asarray(
        [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=float
    )
    crossed_first = crossed_seed_item_bootstrap_ci(
        effects, samples=1_000, seed=17
    )
    crossed_second = crossed_seed_item_bootstrap_ci(
        effects, samples=1_000, seed=17
    )
    assert crossed_first == crossed_second
    assert crossed_first[0] <= effects.mean() <= crossed_first[1]


def test_comparison_pairs_by_identity_and_reports_discordance():
    base = _records(
        [1, 0, 1, 0, 0, 1, 0, 0], label="base", seed=142, adapter=None
    )
    adapter = _records(
        [1, 1, 0, 0, 1, 1, 0, 0],
        label="semantic-142",
        seed=142,
        adapter="adapter-142",
    )
    result = compare_evaluation_records(
        base, list(reversed(adapter)), bootstrap_samples=200, bootstrap_seed=4
    )
    assert result["paired_table"] == {
        "both_correct": 2,
        "both_wrong": 3,
        "base_only_correct": 1,
        "adapter_only_correct": 2,
        "discordant": 3,
    }
    assert result["base"]["correct"] == 3
    assert result["adapter"]["correct"] == 4


def test_comparison_rejects_item_and_provenance_mismatches():
    base = _records([0] * 8, label="base", seed=142, adapter=None)
    adapter = _records(
        [1] * 8, label="adapter", seed=142, adapter="adapter-142"
    )

    wrong_prompt = deepcopy(adapter)
    wrong_prompt[0]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="same items"):
        compare_evaluation_records(base, wrong_prompt, bootstrap_samples=10)

    wrong_revision = deepcopy(adapter)
    for record in wrong_revision:
        record["dataset"]["revision"] = "other-revision"
    with pytest.raises(ValueError, match="dataset revision"):
        compare_evaluation_records(base, wrong_revision, bootstrap_samples=10)

    wrong_model = deepcopy(adapter)
    for record in wrong_model:
        record["provenance"]["model"]["configured_revision"] = "other-model"
    with pytest.raises(ValueError, match="base model revisions"):
        compare_evaluation_records(base, wrong_model, bootstrap_samples=10)

    wrong_words = deepcopy(adapter)
    for record in wrong_words:
        record["target_words"] = ["solved"]
    with pytest.raises(ValueError, match="target_words"):
        compare_evaluation_records(base, wrong_words, bootstrap_samples=10)


def test_multi_seed_and_matched_control_difference_in_differences():
    base = _records([0] * 8, label="base", seed=142, adapter=None)
    adapters = []
    controls = []
    for seed in range(142, 148):
        adapters.append(
            _records(
                [1] * 8,
                label=f"semantic-{seed}",
                seed=seed,
                adapter=f"semantic-adapter-{seed}",
            )
        )
        controls.append(
            _records(
                [0] * 8,
                label=f"control-{seed}",
                seed=seed,
                adapter=f"control-adapter-{seed}",
            )
        )

    multi = compare_multiple_adapters(
        base, adapters, bootstrap_samples=100, bootstrap_seed=3
    )
    assert multi["seed_count"] == 6
    assert multi["seed_sign_test"]["exact_two_sided_p"] == 0.03125

    did = difference_in_differences(
        base, adapters, controls, bootstrap_samples=100, bootstrap_seed=3
    )
    assert did["mean_difference_in_differences"] == 1.0
    assert did["crossed_seed_item_bootstrap"][
        "mean_difference_in_differences_ci_low"
    ] == 1.0
    assert did["seed_sign_test"]["exact_two_sided_p"] == 0.03125
    assert did["significant_improvement"]
    assert did["seed_significant_improvement"]

    controls[0][0]["provenance"]["experiment"]["training_seed"] = 999
    with pytest.raises(ValueError, match="matching training seeds"):
        difference_in_differences(
            base, adapters, controls, bootstrap_samples=10, bootstrap_seed=3
        )


def test_index_manifest_and_literal_word_validation(tmp_path):
    manifest = tmp_path / "indices.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "openai/gsm8k",
                "subset": "main",
                "split": "train",
                "indices": [7, 2, 9],
            }
        )
    )
    indices, identity = load_index_manifest(
        manifest, expected_split="train", dataset_size=10, expected_count=3
    )
    assert indices == [7, 2, 9]
    assert identity["count"] == 3
    assert len(identity["sha256"]) == 64

    manifest.write_text(json.dumps([1, 1, 2]))
    with pytest.raises(ValueError, match="duplicate"):
        load_index_manifest(
            manifest, expected_split="train", dataset_size=10, expected_count=3
        )

    text = "ERROR and Nice; not wrongly, satisfiedly, or niceness."
    assert literal_target_matches(
        text, ["wrong", "error", "nice", "satisfied"]
    ) == ["error", "nice"]


class _FakeBatch(dict):
    def __init__(self, input_ids, attention_mask):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "PROMPT:" + messages[-1]["content"]

    def __call__(self, prompts, **_kwargs):
        input_ids = torch.tensor([[1, 2] for _ in prompts])
        attention_mask = torch.ones_like(input_ids)
        return _FakeBatch(input_ids, attention_mask)

    def decode(self, _ids, skip_special_tokens):
        assert skip_special_tokens
        return "reasoning error #### 2"


class _FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def generate(self, input_ids, **_kwargs):
        suffix = torch.tensor([[12, 99] for _ in range(len(input_ids))])
        return torch.cat([input_ids, suffix], dim=1)


def test_evaluate_writes_auditable_jsonl_without_gold_answers(tmp_path):
    output = tmp_path / "items.jsonl"
    model = _FakeModel()
    cfg = {
        "max_prompt_tokens": 32,
        "max_new_tokens": 8,
        "target_words": ["error"],
    }
    dataset = {
        "name": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "revision": "revision",
        "fingerprint": "fingerprint",
    }
    provenance = _provenance("adapter", 142, "adapter-sha")
    metrics = evaluate(
        model,
        _FakeTokenizer(),
        [{"question": "What is one plus one?", "answer": "work #### 2"}],
        cfg,
        None,
        batch_size=1,
        source_indices=[7312],
        dataset_provenance=dataset,
        provenance=provenance,
        output_jsonl=output,
    )
    record = json.loads(output.read_text())
    assert metrics["exact_match"] == 1.0
    assert record["source_index"] == 7312
    assert record["prediction"] == "2"
    assert record["correct"] is True
    assert record["literal_target_matches"] == ["error"]
    assert len(record["prompt_sha256"]) == 64
    assert record["provenance"] == provenance
    assert "reference_answer" not in record
    assert "work #### 2" not in output.read_text()
    assert model.training is True
