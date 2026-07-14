import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import modal_word_correlation as runner


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare_controller(tmp_path: Path, monkeypatch) -> tuple[FakeVolume, dict]:
    claim_id = "a" * 32
    write_json(
        tmp_path / "attempt_status.json",
        {"claim_id": claim_id, "stage": "claimed"},
    )
    volume = FakeVolume()
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "output_volume", volume)
    return volume, {"claim_id": claim_id}


def test_durable_gpu_job_commits_intent_before_spawn(tmp_path, monkeypatch) -> None:
    volume, manifest = prepare_controller(tmp_path, monkeypatch)
    artifact = None
    events: list[str] = []

    def load_result(kind, shard_index, received_manifest):
        assert received_manifest is manifest
        return artifact

    class Call:
        object_id = "fc-new"

        def get(self):
            nonlocal artifact
            state = json.loads((tmp_path / "controller_state.json").read_text())
            assert state["active_job"] == {
                "call_id": "fc-new",
                "key": "discovery:00",
                "kind": "discovery",
                "shard_index": 0,
            }
            events.append("get")
            artifact = {"phase": "discovery", "shard_index": 0}

    class Job:
        def spawn(self, kind, shard_index):
            state = json.loads((tmp_path / "controller_state.json").read_text())
            assert state["active_job"]["call_id"] is None
            events.append("spawn")
            return Call()

    monkeypatch.setattr(runner, "_load_gpu_job_result", load_result)
    monkeypatch.setattr(runner, "gpu_job", Job())
    result = runner._durable_gpu_job(
        manifest["claim_id"], "fc-root", "discovery", 0, manifest
    )
    assert result == artifact
    assert events == ["spawn", "get"]
    state = json.loads((tmp_path / "controller_state.json").read_text())
    assert state["active_job"] is None
    assert state["completed_jobs"] == [
        {
            "call_id": "fc-new",
            "key": "discovery:00",
            "kind": "discovery",
            "shard_index": 0,
        }
    ]
    assert volume.commits >= 4


def test_keyboard_interrupt_keeps_call_reattachable(tmp_path, monkeypatch) -> None:
    _volume, manifest = prepare_controller(tmp_path, monkeypatch)
    artifact = None

    def load_result(_kind, _shard_index, _manifest):
        return artifact

    class InterruptedCall:
        object_id = "fc-interrupted"

        def get(self):
            raise KeyboardInterrupt

    class Job:
        def __init__(self) -> None:
            self.spawns = 0

        def spawn(self, _kind, _shard_index):
            self.spawns += 1
            return InterruptedCall()

    job = Job()
    monkeypatch.setattr(runner, "_load_gpu_job_result", load_result)
    monkeypatch.setattr(runner, "gpu_job", job)
    with pytest.raises(KeyboardInterrupt):
        runner._durable_gpu_job(
            manifest["claim_id"], "fc-root", "validation", 3, manifest
        )
    state = json.loads((tmp_path / "controller_state.json").read_text())
    assert state["active_job"]["call_id"] == "fc-interrupted"

    class ReattachedCall:
        def get(self):
            nonlocal artifact
            artifact = {"phase": "validation", "shard_index": 3}

    observed_ids: list[str] = []

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            observed_ids.append(call_id)
            return ReattachedCall()

    monkeypatch.setattr(runner, "modal", SimpleNamespace(FunctionCall=FunctionCall))
    result = runner._durable_gpu_job(
        manifest["claim_id"], "fc-root", "validation", 3, manifest
    )
    assert result == artifact
    assert job.spawns == 1
    assert observed_ids == ["fc-interrupted"]
    state = json.loads((tmp_path / "controller_state.json").read_text())
    assert state["active_job"] is None


def test_null_call_id_reentry_safely_dispatches_duplicate(tmp_path, monkeypatch) -> None:
    _volume, manifest = prepare_controller(tmp_path, monkeypatch)
    write_json(
        tmp_path / "controller_state.json",
        {
            "active_job": {
                "call_id": None,
                "key": "calibration",
                "kind": "calibration",
                "shard_index": None,
            },
            "claim_id": manifest["claim_id"],
            "completed_jobs": [],
            "orchestrator_call_id": "fc-root",
            "protocol": "j-lens-rl-word-correlation-controller-v1",
        },
    )
    artifact = None

    def load_result(_kind, _shard_index, _manifest):
        return artifact

    class DuplicateCall:
        object_id = "fc-duplicate"

        def get(self):
            nonlocal artifact
            artifact = {"output_sha256": "b" * 64}

    class Job:
        def spawn(self, kind, shard_index):
            assert (kind, shard_index) == ("calibration", None)
            return DuplicateCall()

    monkeypatch.setattr(runner, "_load_gpu_job_result", load_result)
    monkeypatch.setattr(runner, "gpu_job", Job())
    result = runner._durable_gpu_job(
        manifest["claim_id"], "fc-root", "calibration", None, manifest
    )
    assert result == artifact
    state = json.loads((tmp_path / "controller_state.json").read_text())
    assert state["active_job"] is None
    assert state["completed_jobs"][0]["call_id"] == "fc-duplicate"


def test_committed_calibration_requires_a_valid_atomic_pair(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    manifest = {"config_sha256": "c" * 64, "scanner_sha256": "d" * 64}
    assert runner._load_committed_calibration(manifest) is None
    payload = {
        "model_revision": runner.MODEL_REVISION,
        "wikitext_revision": runner.WIKITEXT_REVISION,
        "lens_sha256": runner.LENS_SHA256,
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
    }
    output = tmp_path / "artifacts/calibration.json"
    write_json(output, payload)
    with pytest.raises(RuntimeError, match="atomic pair"):
        runner._load_committed_calibration(manifest)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = tmp_path / "artifacts/calibration_manifest.json"
    expected = {
        "output": "artifacts/calibration.json",
        "output_sha256": digest,
    }
    write_json(sidecar, expected)
    assert runner._load_committed_calibration(manifest) == expected
    expected["output_sha256"] = "0" * 64
    write_json(sidecar, expected)
    with pytest.raises(RuntimeError, match="manifest is invalid"):
        runner._load_committed_calibration(manifest)
