import json
from types import SimpleNamespace

import pytest

from jlens_rl.common import load_index_manifest, repository_provenance
from jlens_rl.train import DeterministicValidationCallback, create_run_directory


def test_index_manifest_accepts_object_and_rejects_duplicates(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"indices": [5, 1, 9]}))
    assert load_index_manifest(valid) == [5, 1, 9]

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"indices": [1, 1]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_index_manifest(duplicate)


def test_repository_provenance_fingerprints_dirty_tree():
    provenance = repository_provenance(".")
    assert len(provenance["git_commit"]) == 40
    assert len(provenance["source_tree_sha256"]) == 64
    assert isinstance(provenance["git_dirty"], bool)


def test_observational_validation_cannot_stop_training():
    callback = DeterministicValidationCallback(
        tokenizer=None,
        rows=None,
        cfg={
            "eval_every": 5,
            "early_stopping_patience": 1,
            "validation_observational_only": True,
        },
    )
    callback.best_exact_match = 1.0
    callback.evaluate_and_log = lambda model, step: {"exact_match": 0.0}
    control = SimpleNamespace(should_training_stop=False)
    result = callback.on_step_end(
        args=None,
        state=SimpleNamespace(global_step=5),
        control=control,
        model=None,
    )
    assert result.should_training_stop is False


def test_run_directory_must_be_empty(tmp_path):
    new_dir = create_run_directory(tmp_path / "new-run")
    assert new_dir.is_dir()
    (new_dir / "old-result.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not empty"):
        create_run_directory(new_dir)
