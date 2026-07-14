from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import v9_local_train as v9


RUN_ID = "confirm-v9-local-jlens_seed208"


def frozen_identity() -> dict:
    return {
        "run_id": RUN_ID,
        "entity": "entity",
        "project": "project",
        "run_name": "confirm-v9-local-jlens_seed208",
        "url": f"https://wandb.ai/entity/project/runs/{RUN_ID}",
        "group": "confirm-v9-local",
        "tags": ["confirmatory-v9-local", "offline-wandb"],
        "resume": "never",
    }


class FakeConfig:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.updates: list[tuple[dict, bool]] = []

    def update(self, value: dict, *, allow_val_change: bool) -> None:
        self.events.append("config")
        self.updates.append((value, allow_val_change))


def build_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    identity = frozen_identity()
    output = tmp_path / "runs" / "jlens_seed208"
    output.mkdir(parents=True)
    for number, name in enumerate(v9.EVIDENCE_FILE_NAMES):
        payload = (
            {"wandb_identity": identity, "complete": True}
            if name == "run_result_manifest.json"
            else {"name": name, "number": number}
        )
        (output / name).write_text(json.dumps(payload, sort_keys=True) + "\n")

    wandb_dir = tmp_path / "offline_wandb" / "jlens_seed208"
    offline_root = wandb_dir / "wandb" / f"offline-run-20260714_170000-{RUN_ID}"
    files = offline_root / "files"
    logs = offline_root / "logs"
    files.mkdir(parents=True)
    logs.mkdir()
    events: list[str] = []
    run = SimpleNamespace(
        id=identity["run_id"],
        entity=identity["entity"],
        project=identity["project"],
        name=identity["run_name"],
        url=None,
        group=identity["group"],
        tags=tuple(identity["tags"]),
        dir=str(files),
        settings=SimpleNamespace(mode="offline"),
    )
    fake_wandb = SimpleNamespace(run=run, config=FakeConfig(events))

    def save(path: str, *, base_path: str, policy: str) -> str:
        assert policy == "now"
        source = Path(path).resolve(strict=True)
        relative = source.relative_to(Path(base_path).resolve(strict=True))
        queued = files / relative
        queued.parent.mkdir(parents=True, exist_ok=True)
        queued.symlink_to(source)
        events.append(f"save:{relative.as_posix()}")
        return str(queued)

    def finish(*, exit_code: int) -> None:
        events.append(f"finish:{exit_code}")
        (offline_root / f"run-{RUN_ID}.wandb").write_bytes(b"closed-wandb-stream\n")
        (logs / "post-finish.log").write_text("finished\n")
        (logs / "debug-core.log").symlink_to("/tmp/frozen-core-debug.log")
        fake_wandb.run = None

    fake_wandb.save = save
    fake_wandb.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    environment = {
        "WANDB_DIR": str(wandb_dir),
        "WANDB_MODE": "offline",
        "WANDB_RUN_ID": identity["run_id"],
        "WANDB_ENTITY": identity["entity"],
        "WANDB_PROJECT": identity["project"],
        "WANDB_RUN_GROUP": identity["group"],
        "WANDB_TAGS": ",".join(identity["tags"]),
        "WANDB_RESUME": "never",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(v9, "_pending_publication", None)
    return SimpleNamespace(
        identity=identity,
        result={"wandb_identity": identity, "completed_updates": 20},
        output=output,
        wandb_dir=wandb_dir,
        offline_root=offline_root,
        files=files,
        events=events,
        wandb=fake_wandb,
        run=run,
    )


def queue(case: SimpleNamespace) -> dict:
    result = v9.publish_run_result_offline(
        output_dir=case.output,
        result=case.result,
        enabled=True,
    )
    assert isinstance(result, dict)
    return result


def finish_and_seal(case: SimpleNamespace) -> dict:
    case.wandb.finish(exit_code=0)
    assert v9._pending_publication is not None
    return v9.finalize_offline_terminal_receipt(v9._pending_publication)


def test_offline_receipt_is_written_only_after_finish_and_binds_exact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case(tmp_path, monkeypatch)
    queued = queue(case)
    receipt_path = case.output / v9.OFFLINE_RECEIPT_NAME
    assert not receipt_path.exists()
    assert queued["remote_artifact_claimed"] is False
    assert list(queued["queued_evidence_sha256"]) == list(v9.EVIDENCE_FILE_NAMES)
    assert [event for event in case.events if event.startswith("save:")] == [
        f"save:{name}" for name in v9.EVIDENCE_FILE_NAMES
    ]

    receipt = finish_and_seal(case)
    assert receipt_path.is_file()
    assert case.events[-1] == "finish:0"
    assert receipt["remote_artifact_claimed"] is False
    assert receipt["sync_completed"] is False
    assert receipt["offline_run_root"] == str(case.offline_root)
    assert receipt["offline_run_root_relative_to_wandb_dir"] == (
        case.offline_root.relative_to(case.wandb_dir).as_posix()
    )
    assert receipt["terminal_run_result_sha256"] == receipt[
        "queued_evidence_sha256"
    ]["run_result_manifest.json"]
    assert receipt["embedded_evidence_relative_paths"] == {
        name: f"files/{name}" for name in v9.EVIDENCE_FILE_NAMES
    }
    assert (
        receipt["embedded_evidence_storage"]
        == "regular_files_inside_offline_run_root"
    )
    inventory = receipt["offline_run_file_symlink_inventory"]
    assert receipt["offline_run_file_symlink_count"] == len(inventory)
    assert receipt["offline_run_tree_sha256"] == v9._canonical_sha256(inventory)
    paths = {item["path"]: item for item in inventory}
    assert paths[f"run-{RUN_ID}.wandb"]["type"] == "file"
    assert paths["logs/post-finish.log"]["type"] == "file"
    assert paths["logs/debug-core.log"] == {
        "path": "logs/debug-core.log",
        "type": "symlink",
        "target": "/tmp/frozen-core-debug.log",
        "mode": "0o777",
    }
    for name in v9.EVIDENCE_FILE_NAMES:
        assert paths[f"files/{name}"]["type"] == "file"
        assert paths[f"files/{name}"]["sha256"] == receipt[
            "queued_evidence_sha256"
        ][name]
        embedded = case.files / name
        assert embedded.is_file() and not embedded.is_symlink()
        assert not (case.files / f".{name}.v9-embed.tmp").exists()
    assert not (case.offline_root / v9.OFFLINE_RECEIPT_NAME).exists()
    assert not receipt_path.with_suffix(receipt_path.suffix + ".tmp").exists()
    assert (
        v9.validate_offline_terminal_receipt(
            case.output, case.identity, case.wandb_dir
        )
        == receipt
    )


def test_main_finishes_before_atomic_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case(tmp_path, monkeypatch)
    original_write = v9.train._write_json_atomic
    original_publisher = v9.train.publish_run_result_to_wandb

    def fake_training_main() -> None:
        case.events.append("trainer-terminal")
        queue(case)
        case.events.append("trainer-return")

    def recording_write(path: Path, payload: dict) -> None:
        case.events.append("receipt-write")
        original_write(path, payload)

    monkeypatch.setattr(v9.train, "main", fake_training_main)
    monkeypatch.setattr(v9.train, "_write_json_atomic", recording_write)
    v9.main()
    assert case.events.index("trainer-return") < case.events.index("finish:0")
    assert case.events.index("finish:0") < case.events.index("receipt-write")
    assert v9.train.publish_run_result_to_wandb is original_publisher
    assert v9._pending_publication is None
    v9.validate_offline_terminal_receipt(case.output, case.identity, case.wandb_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda case: setattr(case.run.settings, "mode", "online"), "mode/resume"),
        (lambda case: setattr(case.run, "id", "wrong-id"), "frozen identity"),
        (
            lambda case: case.result["wandb_identity"].update(
                {"url": "https://wandb.ai/wrong"}
            ),
            "target URL",
        ),
    ],
)
def test_queue_fails_closed_on_identity_or_mode_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    case = build_case(tmp_path, monkeypatch)
    mutation(case)
    with pytest.raises(RuntimeError, match=message):
        queue(case)
    assert not (case.output / v9.OFFLINE_RECEIPT_NAME).exists()


def test_queue_fails_closed_on_wandb_or_evidence_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case(tmp_path, monkeypatch)
    outside = tmp_path / "outside" / "offline-run-20260714_170000-" / "files"
    outside.mkdir(parents=True)
    case.run.dir = str(outside)
    with pytest.raises(RuntimeError, match="escaped"):
        queue(case)

    case = build_case(tmp_path / "second", monkeypatch)
    evidence = case.output / "log_history.json"
    outside_evidence = tmp_path / "outside-evidence.json"
    outside_evidence.write_text("{}\n")
    evidence.unlink()
    evidence.symlink_to(outside_evidence)
    with pytest.raises(RuntimeError, match="evidence path escaped"):
        queue(case)


def test_multiple_matching_offline_roots_prevent_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case(tmp_path, monkeypatch)
    queue(case)
    case.wandb.finish(exit_code=0)
    duplicate = case.wandb_dir / "duplicate" / f"offline-run-another-{RUN_ID}"
    duplicate.mkdir(parents=True)
    assert v9._pending_publication is not None
    with pytest.raises(RuntimeError, match="exactly one"):
        v9.finalize_offline_terminal_receipt(v9._pending_publication)
    assert not (case.output / v9.OFFLINE_RECEIPT_NAME).exists()


def test_post_finish_embedding_rejects_escaped_or_extra_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case(tmp_path, monkeypatch)
    queue(case)
    case.wandb.finish(exit_code=0)
    escaped = tmp_path / "escaped.json"
    escaped.write_text("{}\n")
    queued = case.files / "log_history.json"
    queued.unlink()
    queued.symlink_to(escaped)
    assert v9._pending_publication is not None
    with pytest.raises(RuntimeError, match="escaped its source"):
        v9.finalize_offline_terminal_receipt(v9._pending_publication)
    assert not (case.output / v9.OFFLINE_RECEIPT_NAME).exists()

    case = build_case(tmp_path / "extra", monkeypatch)
    queue(case)
    case.wandb.finish(exit_code=0)
    (case.files / "unregistered.txt").symlink_to(case.output / "log_history.json")
    assert v9._pending_publication is not None
    with pytest.raises(RuntimeError, match="queued symlink set changed"):
        v9.finalize_offline_terminal_receipt(v9._pending_publication)
    assert not (case.output / v9.OFFLINE_RECEIPT_NAME).exists()


def test_validator_detects_post_receipt_tree_or_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case(tmp_path, monkeypatch)
    queue(case)
    finish_and_seal(case)
    archive = case.offline_root / f"run-{RUN_ID}.wandb"
    archive.write_bytes(b"mutated\n")
    with pytest.raises(RuntimeError, match="tree changed"):
        v9.validate_offline_terminal_receipt(
            case.output, case.identity, case.wandb_dir
        )

    # Restore and reseal a fresh case to distinguish linked-source mutation.
    case = build_case(tmp_path / "source", monkeypatch)
    queue(case)
    finish_and_seal(case)
    (case.output / "log_history.json").write_text('{"mutated": true}\n')
    with pytest.raises(RuntimeError, match="hashes differ"):
        v9.validate_offline_terminal_receipt(
            case.output, case.identity, case.wandb_dir
        )


def test_disabled_publication_has_no_tracking_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v9, "_pending_publication", None)
    assert (
        v9.publish_run_result_offline(output_dir=Path("unused"), result={}, enabled=False)
        is None
    )
    assert v9._pending_publication is None
