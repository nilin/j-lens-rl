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


class FaultVolume(FakeVolume):
    def __init__(
        self,
        *,
        fail_commit: int | None = None,
        fail_reload: int | None = None,
    ) -> None:
        super().__init__()
        self.fail_commit = fail_commit
        self.fail_reload = fail_reload

    def commit(self) -> None:
        super().commit()
        if self.commits == self.fail_commit:
            raise KeyboardInterrupt("injected commit cut")

    def reload(self) -> None:
        super().reload()
        if self.reloads == self.fail_reload:
            raise KeyboardInterrupt("injected reload cut")


class FakeDict:
    def __init__(self) -> None:
        self.values: dict[str, dict] = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = dict(value)
        return True

    def pop(self, key, default=...):
        if default is ...:
            return self.values.pop(key)
        return self.values.pop(key, default)


class FaultDict(FakeDict):
    def __init__(self, cut: str) -> None:
        super().__init__()
        self.cut = cut

    def put(self, key, value, *, skip_if_exists=False):
        if self.cut == "before":
            raise KeyboardInterrupt("injected CAS cut before acceptance")
        created = super().put(key, value, skip_if_exists=skip_if_exists)
        if self.cut == "after":
            raise KeyboardInterrupt("injected CAS cut after acceptance")
        return created


@pytest.fixture(autouse=True)
def fake_coordination_registries(monkeypatch):
    monkeypatch.setattr(runner, "publication_registry", FakeDict())
    monkeypatch.setattr(runner, "root_authority_registry", FakeDict())


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
    monkeypatch.setattr(runner, "_refresh_gpu_lease", lambda *_identity: {})
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
        def spawn(self, kind, shard_index, root_call_id):
            assert root_call_id == "fc-root"
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

        def spawn(self, _kind, _shard_index, root_call_id):
            assert root_call_id == "fc-root"
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
        def spawn(self, kind, shard_index, root_call_id):
            assert (kind, shard_index) == ("calibration", None)
            assert root_call_id == "fc-root"
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


def test_committed_calibration_ignores_unmarked_generations(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "output_volume", FakeVolume())
    manifest = {"config_sha256": "c" * 64, "scanner_sha256": "d" * 64}
    assert runner._load_committed_calibration(manifest) is None
    payload = {
        "model_revision": runner.MODEL_REVISION,
        "wikitext_revision": runner.WIKITEXT_REVISION,
        "lens_sha256": runner.LENS_SHA256,
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
    }
    generation = runner._new_generation_dir("calibration")
    output = generation / "calibration.json"
    write_json(output, payload)
    assert runner._load_committed_calibration(manifest) is None
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    expected = {
        "output": output.relative_to(tmp_path).as_posix(),
        "output_sha256": digest,
    }
    runner._publish_generation("calibration", generation, expected)
    assert runner._load_committed_calibration(manifest) == expected
    output.write_text("changed\n")
    with pytest.raises(RuntimeError, match="incomplete or changed"):
        runner._load_committed_calibration(manifest)


ARTIFACT_KEYS = (
    "claim",
    "calibration",
    "discovery/shard-00",
    "discovery/final",
    "validation/final",
    "atlas",
    "result",
)


@pytest.mark.parametrize("artifact_key", ARTIFACT_KEYS)
@pytest.mark.parametrize(
    ("cut", "fault"),
    (
        ("generation_commit", {"fail_commit": 1}),
        ("generation_reload", {"fail_reload": 1}),
        ("selection_reload", {"fail_reload": 2}),
        ("marker_commit", {"fail_commit": 2}),
        ("marker_reload", {"fail_reload": 3}),
    ),
)
def test_every_publication_window_is_restart_safe(
    artifact_key, cut, fault, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    volume = FaultVolume(**fault)
    monkeypatch.setattr(runner, "output_volume", volume)
    first = runner._new_generation_dir(artifact_key)
    write_json(first / "one.json", {"generation": 1})
    write_json(first / "nested/two.json", {"generation": 1})

    with pytest.raises(KeyboardInterrupt, match="injected"):
        runner._publish_generation(artifact_key, first, {"kind": artifact_key})

    marker_path = runner._marker_path(artifact_key)
    if cut == "marker_commit":
        # The final marker existed only in the interrupted container's dirty
        # mount; a fresh container sees the last committed state.
        marker_path.unlink()
    if cut == "marker_reload":
        committed = runner._load_generation_marker(artifact_key)
        assert committed is not None
        assert committed[0] == first.resolve()
    else:
        assert runner._load_generation_marker(artifact_key) is None

    monkeypatch.setattr(runner, "output_volume", FakeVolume())
    second = runner._new_generation_dir(artifact_key)
    write_json(second / "one.json", {"generation": 2})
    write_json(second / "nested/two.json", {"generation": 2})
    committed = runner._publish_generation(
        artifact_key, second, {"kind": artifact_key}
    )
    if cut in {"selection_reload", "marker_commit", "marker_reload"}:
        assert committed[0] == first.resolve()
    else:
        assert committed[0] == second.resolve()
    assert first.is_dir()


def test_generation_marker_rejects_traversal_and_hash_changes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    marker = runner._marker_path("calibration")
    write_json(
        marker,
        {
            "protocol": runner.GENERATION_MARKER_PROTOCOL,
            "artifact_key": "calibration",
            "generation": "generations/calibration/../../escape",
            "generation_artifact_sha256": {"calibration.json": "0" * 64},
        },
    )
    with pytest.raises(RuntimeError, match="unsafe calibration generation"):
        runner._load_generation_marker("calibration")

    marker.unlink()
    monkeypatch.setattr(runner, "output_volume", FakeVolume())
    generation = runner._new_generation_dir("calibration")
    write_json(generation / "calibration.json", {"ok": True})
    runner._publish_generation("calibration", generation, {})
    (generation / "calibration.json").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="incomplete or changed"):
        runner._load_generation_marker("calibration")


@pytest.mark.parametrize("artifact_key", ARTIFACT_KEYS)
@pytest.mark.parametrize("cut", ("before", "after"))
def test_publication_cas_acceptance_window_is_restart_safe(
    artifact_key, cut, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "output_volume", FakeVolume())
    registry = FaultDict(cut)
    monkeypatch.setattr(runner, "publication_registry", registry)
    first = runner._new_generation_dir(artifact_key)
    write_json(first / "artifact.json", {"generation": 1})
    with pytest.raises(KeyboardInterrupt, match="injected CAS cut"):
        runner._publish_generation(artifact_key, first, {"kind": artifact_key})

    registry.cut = "none"
    second = runner._new_generation_dir(artifact_key)
    write_json(second / "artifact.json", {"generation": 2})
    committed = runner._publish_generation(
        artifact_key, second, {"kind": artifact_key}
    )
    assert committed[0] == (first if cut == "after" else second).resolve()


def test_volume_marker_must_match_durable_cas_selection(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "output_volume", FakeVolume())
    generation = runner._new_generation_dir("calibration")
    write_json(generation / "calibration.json", {"ok": True})
    runner._publish_generation("calibration", generation, {})
    runner.publication_registry.pop("calibration")
    with pytest.raises(RuntimeError, match="differs from its CAS selection"):
        runner._materialize_selected_marker("calibration")


def test_losing_publisher_commits_its_staging_before_reload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    first_volume = FakeVolume()
    monkeypatch.setattr(runner, "output_volume", first_volume)
    first = runner._new_generation_dir("calibration")
    write_json(first / "calibration.json", {"generation": 1})
    runner._publish_generation("calibration", first, {})
    runner._marker_path("calibration").unlink()

    events: list[str] = []

    class TraceVolume(FakeVolume):
        def commit(self):
            events.append("commit")
            super().commit()

        def reload(self):
            events.append("reload")
            super().reload()

    monkeypatch.setattr(runner, "output_volume", TraceVolume())
    second = runner._new_generation_dir("calibration")
    write_json(second / "calibration.json", {"generation": 2})
    committed = runner._publish_generation("calibration", second, {})
    assert committed[0] == first.resolve()
    assert events[0] == "commit"


def preflight(when: str = "now") -> dict:
    return {
        "exclusive_gpu_confirmation": runner.GPU_EXCLUSIVE_CONFIRMATION,
        "global_modal_gpu_limit": runner.GLOBAL_MODAL_GPU_LIMIT,
        "active_other_modal_apps": [],
        "checked_at_utc": when,
    }


def minimal_manifest(claim_id: str = "a" * 32) -> dict:
    return {
        "claim_id": claim_id,
        "gpu_exclusive_preflight": preflight("initial"),
    }


def test_default_claim_identity_is_deterministic_and_explicit_ids_are_checked(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "_git", lambda *args, **kwargs: "f" * 40)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "e" * 64)
    first = runner._default_claim_id()
    second = runner._default_claim_id()
    assert first == second
    assert len(first) == 32
    assert runner._validate_claim_id("a" * 32) == "a" * 32
    with pytest.raises(RuntimeError, match="32 lowercase hexadecimal"):
        runner._validate_claim_id("new-random-claim")


def publish_claim(tmp_path: Path, manifest: dict) -> None:
    generation = runner._new_generation_dir("claim")
    path = generation / "attempt_manifest.json"
    write_json(path, manifest)
    runner._publish_generation(
        "claim",
        generation,
        {
            "claim_id": manifest["claim_id"],
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "original_preflight_sha256": runner._json_sha256(
                manifest["gpu_exclusive_preflight"]
            ),
        },
    )


@pytest.mark.parametrize("durable_cut", ("orphan", "marker", "status", "return"))
def test_claim_is_idempotent_across_every_durable_cut(
    durable_cut, tmp_path, monkeypatch
) -> None:
    volume = FakeVolume()
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "output_volume", volume)
    monkeypatch.setattr(runner, "_verify_remote_manifest", lambda _manifest: None)
    manifest = minimal_manifest()
    if durable_cut == "orphan":
        generation = runner._new_generation_dir("claim")
        write_json(generation / "attempt_manifest.json", manifest)
    elif durable_cut in {"marker", "status", "return"}:
        publish_claim(tmp_path, manifest)
        if durable_cut in {"status", "return"}:
            write_json(
                tmp_path / "attempt_status.json",
                {"claim_id": manifest["claim_id"], "stage": "claimed"},
            )

    assert runner.claim_attempt.local(manifest) == manifest
    assert runner.claim_attempt.local(manifest) == manifest
    assert runner._load_claim_manifest(manifest["claim_id"]) == manifest
    assert json.loads((tmp_path / "attempt_status.json").read_text())["stage"] == (
        "claimed"
    )
    changed = {**manifest, "gpu_exclusive_preflight": preflight("different")}
    with pytest.raises(RuntimeError, match="different launch manifest"):
        runner.claim_attempt.local(changed)


def test_global_gpu_lease_is_atomic_durable_and_fail_closed(monkeypatch) -> None:
    registry = FakeDict()
    monkeypatch.setattr(runner, "gpu_lease_registry", registry)
    first = runner._acquire_gpu_lease("a" * 32, preflight("one"))
    assert first["owner"] == f"{runner.GPU_LEASE_KEY_PREFIX}:{'a' * 32}"
    resumed = runner._acquire_gpu_lease("a" * 32, preflight("two"))
    assert resumed["acquired_at_utc"] == first["acquired_at_utc"]
    with pytest.raises(RuntimeError, match="held by another app/claim"):
        runner._acquire_gpu_lease("b" * 32, preflight("other"))
    assert runner.GPU_LEASE_SLOT in registry.values
    runner._claim_root_authority("a" * 32, "fc-a")
    runner._release_gpu_lease("a" * 32, "fc-a")
    second = runner._acquire_gpu_lease("b" * 32, preflight("after-release"))
    assert second["claim_id"] == "b" * 32
    runner._release_gpu_lease("a" * 32, "fc-a")
    assert registry.values[runner.GPU_LEASE_SLOT] == second


def prepare_submission(tmp_path: Path, monkeypatch):
    volume = FakeVolume()
    registry = FakeDict()
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "output_volume", volume)
    monkeypatch.setattr(runner, "gpu_lease_registry", registry)
    monkeypatch.setattr(runner, "_verify_remote_manifest", lambda _manifest: None)
    manifest = minimal_manifest()
    publish_claim(tmp_path, manifest)
    write_json(
        tmp_path / "attempt_status.json",
        {"claim_id": manifest["claim_id"], "stage": "claimed"},
    )
    return volume, registry, manifest


def test_submission_commits_intent_before_spawn_and_reattaches(
    tmp_path, monkeypatch
) -> None:
    _volume, registry, manifest = prepare_submission(tmp_path, monkeypatch)
    events: list[str] = []

    class Call:
        object_id = "fc-root"

    attached: list[str] = []

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            attached.append(call_id)
            return object()

    class Orchestrator:
        def __init__(self):
            self.spawns = 0

        def spawn(self, claim_id):
            self.spawns += 1
            state = json.loads((tmp_path / "submission_state.json").read_text())
            assert state["claim_id"] == claim_id
            assert state["orchestrator_call_id"] is None
            assert runner.GPU_LEASE_SLOT in registry.values
            runner._claim_root_authority(claim_id, "fc-root")
            events.append("spawn")
            return Call()

    orchestrator = Orchestrator()
    monkeypatch.setattr(runner, "orchestrate", orchestrator)
    monkeypatch.setattr(runner, "modal", SimpleNamespace(FunctionCall=FunctionCall))
    submission = runner.submit_attempt.local(manifest["claim_id"], preflight("submit"))
    assert submission["function_call_id"] == "fc-root"
    assert events == ["spawn"]
    replay = runner.submit_attempt.local(manifest["claim_id"], preflight("retry"))
    assert replay["function_call_id"] == "fc-root"
    assert orchestrator.spawns == 1
    assert attached == ["fc-root", "fc-root"]
    state = json.loads((tmp_path / "submission_state.json").read_text())
    assert [item["checked_at_utc"] for item in state["submission_preflight_checks"]] == [
        "submit",
    ]


def test_null_submission_intent_recovers_authoritative_root(
    tmp_path, monkeypatch
) -> None:
    _volume, _registry, manifest = prepare_submission(tmp_path, monkeypatch)
    lease = runner._acquire_gpu_lease(manifest["claim_id"], preflight("submit"))
    lease_receipt = {
        **lease,
        "dict_name": runner.GPU_LEASE_DICT_NAME,
        "environment_name": runner.GPU_LEASE_ENVIRONMENT_NAME,
    }
    write_json(
        tmp_path / "submission_state.json",
        {
            "protocol": "j-lens-rl-word-correlation-submission-v1",
            "claim_id": manifest["claim_id"],
            "intent_id": runner._submission_intent_id(manifest["claim_id"]),
            "lease_slot": runner.GPU_LEASE_SLOT,
            "lease_owner": lease["owner"],
            "gpu_lease_receipt": lease_receipt,
            "orchestrator_call_id": None,
            "controller_bound_at_utc": None,
            "spawned_call_ids": [],
            "submission_preflight_checks": [preflight("submit")],
            "intent_committed_at_utc": "before-cut",
        },
    )
    runner._claim_root_authority(manifest["claim_id"], "fc-authoritative")

    class NeverSpawn:
        def spawn(self, _claim_id):
            raise AssertionError("must recover, not spawn")

    attached: list[str] = []

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            attached.append(call_id)
            return object()

    monkeypatch.setattr(runner, "orchestrate", NeverSpawn())
    monkeypatch.setattr(runner, "modal", SimpleNamespace(FunctionCall=FunctionCall))
    result = runner.submit_attempt.local(manifest["claim_id"], preflight("retry"))
    assert result["function_call_id"] == "fc-authoritative"
    assert attached == ["fc-authoritative"]


def test_result_marker_repairs_a_nonterminal_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "REMOTE_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "output_volume", FakeVolume())
    manifest = {
        **minimal_manifest(),
        "protocol": "j-lens-rl-jspace-word-correlation-v1",
        "git_commit": "f" * 40,
        "config_sha256": "c" * 64,
        "scanner_sha256": "d" * 64,
        "current_amendment_sha256": "e" * 64,
    }
    publish_claim(tmp_path, manifest)
    write_json(
        tmp_path / "attempt_status.json",
        {"claim_id": manifest["claim_id"], "stage": "finalizing"},
    )
    result = {
        key: manifest[key]
        for key in (
            "protocol",
            "claim_id",
            "git_commit",
            "config_sha256",
            "scanner_sha256",
            "current_amendment_sha256",
        )
    }
    result["selection"] = {"canonical_word": "yay", "reward_sign": 1}
    generation = runner._new_generation_dir("result")
    controller_snapshot = generation / "controller_state.json"
    submission_snapshot = generation / "submission_state.json"
    write_json(
        controller_snapshot,
        {
            "protocol": "j-lens-rl-word-correlation-controller-v1",
            "claim_id": manifest["claim_id"],
        },
    )
    write_json(
        submission_snapshot,
        {
            "protocol": "j-lens-rl-word-correlation-submission-v1",
            "claim_id": manifest["claim_id"],
        },
    )
    result.update(
        {
            "controller_state_snapshot": controller_snapshot.relative_to(
                tmp_path
            ).as_posix(),
            "controller_state_sha256": hashlib.sha256(
                controller_snapshot.read_bytes()
            ).hexdigest(),
            "submission_state_snapshot": submission_snapshot.relative_to(
                tmp_path
            ).as_posix(),
            "submission_state_sha256": hashlib.sha256(
                submission_snapshot.read_bytes()
            ).hexdigest(),
        }
    )
    result_path = generation / "result_manifest.json"
    write_json(result_path, result)
    runner._publish_generation(
        "result",
        generation,
        {
            "claim_id": manifest["claim_id"],
            "result_manifest_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
        },
    )
    repaired = runner._finalize_result(
        manifest["claim_id"],
        manifest,
        {},
        [],
        Path("unused"),
        Path("unused"),
        {},
        [],
        Path("unused"),
        Path("unused"),
    )
    assert repaired["stage"] == "complete"
    status = json.loads((tmp_path / "attempt_status.json").read_text())
    assert status["stage"] == "complete"
    assert status["selection"] == result["selection"]
