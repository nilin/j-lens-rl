from __future__ import annotations

import ast
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "modal_v8_wandb_sync_test", ROOT / "modal_v8_wandb_sync.py"
)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


LABEL = "jlens_seed200"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _build_sealed_case(tmp_path: Path) -> SimpleNamespace:
    identity = sync._registered_identity(LABEL)
    root = tmp_path / f"offline-run-20260714_160000-{identity['run_id']}"
    files = root / "files"
    files.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "completed_updates": 20,
        "wandb_identity": identity,
        "evidence_eligibility": "original_registered_v8_local_attempt",
    }
    for number, name in enumerate(sync.EVIDENCE_FILE_NAMES):
        _write(
            files / name,
            result if name == "run_result_manifest.json" else {"number": number},
        )
    _write(root / f"run-{identity['run_id']}.wandb", b"closed-offline-stream\n")
    _write(root / "logs" / "debug.log", b"closed\n")
    inventory = sync._filesystem_inventory(root)
    hashes = {
        name: sync._sha256_file(files / name) for name in sync.EVIDENCE_FILE_NAMES
    }
    registered_wandb_dir = sync.REGISTERED_STATE / "offline_wandb" / LABEL
    registered_root = registered_wandb_dir / "wandb" / root.name
    receipt = {
        "schema_version": 1,
        "transport": "offline_wandb_pending_explicit_sync",
        "sync_completed": False,
        "remote_artifact_claimed": False,
        "wandb_identity": identity,
        "observed_offline_identity": {
            "run_id": identity["run_id"],
            "entity": identity["entity"],
            "project": identity["project"],
            "run_name": identity["run_name"],
            "url": None,
            "group": identity["group"],
            "tags": identity["tags"],
            "resume": "never",
            "mode": "offline",
            "run_files_dir": str(registered_root / "files"),
        },
        "wandb_dir": str(registered_wandb_dir),
        "offline_run_root": str(registered_root),
        "offline_run_root_relative_to_wandb_dir": f"wandb/{root.name}",
        "terminal_run_result_sha256": hashes["run_result_manifest.json"],
        "queued_evidence_sha256": hashes,
        "embedded_evidence_relative_paths": {
            name: f"files/{name}" for name in sync.EVIDENCE_FILE_NAMES
        },
        "embedded_evidence_storage": "regular_files_inside_offline_run_root",
        "offline_run_file_symlink_inventory": inventory,
        "offline_run_file_symlink_count": len(inventory),
        "offline_run_tree_sha256": sync._canonical_sha256(inventory),
        "sync_policy": sync.SYNC_POLICY,
    }
    receipt_path = tmp_path / sync.OFFLINE_RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return SimpleNamespace(
        identity=identity,
        root=root,
        files=files,
        result=result,
        receipt=receipt,
        receipt_path=receipt_path,
        receipt_sha256=sync._sha256_file(receipt_path),
    )


def _build_archive(case: SimpleNamespace, archive: Path) -> None:
    sync._build_payload_archive(
        receipt_path=case.receipt_path,
        offline_root=case.root,
        archive_path=archive,
    )


def test_registered_identity_is_exact_and_label_allowlisted() -> None:
    identity = sync._registered_identity("signflip_seed207")
    assert identity == {
        "entity": "nilinabra-spare-time",
        "project": "j-lens-rl",
        "run_name": (
            "confirm-v8-local-emotional-profanity-u5-h20-signflip_seed207"
        ),
        "run_id": (
            "confirm-v8-local-emotional-profanity-u5-h20-signflip_seed207"
        ),
        "url": (
            "https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/"
            "confirm-v8-local-emotional-profanity-u5-h20-signflip_seed207"
        ),
        "group": "confirm-v8-local-emotional-profanity-u5-h20",
        "tags": sync.WANDB_TAGS,
        "resume": "never",
    }
    for label in ("jlens_seed199", "jlens_seed208", "base", "../jlens_seed200"):
        with pytest.raises(RuntimeError, match="not a registered"):
            sync._registered_identity(label)


def test_payload_round_trip_contains_only_receipt_bound_offline_tree(
    tmp_path: Path,
) -> None:
    case = _build_sealed_case(tmp_path)
    archive = tmp_path / "payload.tar"
    _build_archive(case, archive)
    assert (
        sync._validate_payload_archive(
            archive, label=LABEL, expected_receipt_sha256=case.receipt_sha256
        )
        == case.receipt
    )
    with tarfile.open(archive, "r:") as tar:
        names = set(tar.getnames())
    assert sync.OFFLINE_RECEIPT_NAME in names
    assert all(
        name == sync.OFFLINE_RECEIPT_NAME or name.startswith("offline_run/")
        for name in names
    )
    assert not any(
        fragment in name
        for name in names
        for fragment in ("checkpoint-", "adapter_model", "sealed", "evals/")
    )

    receipt, extracted = sync._extract_payload(
        archive,
        tmp_path / "extract",
        label=LABEL,
        expected_receipt_sha256=case.receipt_sha256,
    )
    assert receipt == case.receipt
    assert sync._filesystem_inventory(extracted) == case.receipt[
        "offline_run_file_symlink_inventory"
    ]
    assert sync._validate_embedded_terminal_evidence(
        extracted, receipt, LABEL
    ) == case.result


def test_payload_rejects_extra_bytes_and_post_receipt_mutation(tmp_path: Path) -> None:
    case = _build_sealed_case(tmp_path)
    archive = tmp_path / "payload.tar"
    _build_archive(case, archive)
    with tarfile.open(archive, "a") as tar:
        extra = tarfile.TarInfo("checkpoint-20/adapter_model.safetensors")
        extra.size = 4
        tar.addfile(extra, io.BytesIO(b"nope"))
    with pytest.raises(RuntimeError, match="outside the sealed receipt"):
        sync._validate_payload_archive(
            archive, label=LABEL, expected_receipt_sha256=case.receipt_sha256
        )

    case = _build_sealed_case(tmp_path / "mutated")
    (case.files / "log_history.json").write_text('{"mutated": true}\n')
    archive = tmp_path / "mutated.tar"
    _build_archive(case, archive)
    with pytest.raises(RuntimeError, match="payload (bytes|metadata) changed"):
        sync._validate_payload_archive(
            archive, label=LABEL, expected_receipt_sha256=case.receipt_sha256
        )


def test_escaping_symlink_is_rejected_before_packaging(tmp_path: Path) -> None:
    case = _build_sealed_case(tmp_path)
    link = case.root / "logs" / "escape"
    link.symlink_to("../../outside")
    inventory = sync._filesystem_inventory(case.root)
    case.receipt["offline_run_file_symlink_inventory"] = inventory
    case.receipt["offline_run_file_symlink_count"] = len(inventory)
    case.receipt["offline_run_tree_sha256"] = sync._canonical_sha256(inventory)
    with pytest.raises(RuntimeError, match="escaping symlink"):
        sync._validate_receipt(
            case.receipt,
            label=LABEL,
            registered_wandb_dir=sync.REGISTERED_STATE / "offline_wandb" / LABEL,
        )

    registered_debug = (
        "/j-lens-rl/.confirmatory/v8_local/wandb_cache/jlens_seed200/"
        "wandb/logs/core-debug-20260714_160000.log"
    )
    assert sync._safe_symlink_target(
        sync.PurePosixPath("logs/debug-core.log"),
        registered_debug,
        expected_label=LABEL,
    ) == registered_debug
    with pytest.raises(RuntimeError, match="absolute symlink"):
        sync._safe_symlink_target(
            sync.PurePosixPath("logs/debug-core.log"),
            registered_debug.replace("jlens_seed200", "jlens_seed201"),
            expected_label=LABEL,
        )


def test_prepare_runs_canonical_validation_before_and_after_packaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "v8_local"
    monkeypatch.setattr(sync, "REGISTERED_STATE", state)
    identity = sync._registered_identity(LABEL)
    config = {
        "wandb_entity": identity["entity"],
        "wandb_project": identity["project"],
        "run_name": identity["run_name"],
        "wandb_run_id": identity["run_id"],
        "wandb_url": identity["url"],
        "wandb_group": identity["group"],
        "wandb_tags": identity["tags"],
        "wandb_resume": "never",
        "wandb_mode": "offline",
        "updates": 20,
        "evidence_eligibility": "original_registered_v8_local_attempt",
        "output_dir": f".confirmatory/v8_local/runs/{LABEL}",
    }
    _write(state / "configs" / f"{LABEL}.json", config)
    case = _build_sealed_case(tmp_path / "case")
    wandb_dir = state / "offline_wandb" / LABEL
    real_root = wandb_dir / "wandb" / case.root.name
    real_root.parent.mkdir(parents=True)
    case.root.rename(real_root)
    case.receipt["wandb_dir"] = str(wandb_dir)
    case.receipt["offline_run_root"] = str(real_root)
    case.receipt["offline_run_root_relative_to_wandb_dir"] = f"wandb/{real_root.name}"
    case.receipt["observed_offline_identity"]["run_files_dir"] = str(
        real_root / "files"
    )
    output = state / "runs" / LABEL
    output.mkdir(parents=True)
    receipt_path = output / sync.OFFLINE_RECEIPT_NAME
    receipt_path.write_text(json.dumps(case.receipt, indent=2, sort_keys=True) + "\n")
    calls: list[tuple[Path, dict, Path]] = []

    def canonical(output_dir: Path, expected: dict, expected_wandb: Path) -> dict:
        calls.append((output_dir, expected, expected_wandb))
        return json.loads(receipt_path.read_text())

    archive = tmp_path / "prepared.tar"
    prepared = sync._prepare_local_payload(
        LABEL,
        archive,
        state_dir=state,
        canonical_validator=canonical,
    )
    assert len(calls) == 2
    assert calls[0] == (output, identity, wandb_dir)
    assert prepared["receipt_sha256"] == sync._sha256_file(receipt_path)
    assert prepared["payload_sha256"] == sync._sha256_file(archive)


def test_modal_function_is_cpu_only_and_secret_is_not_a_command_argument() -> None:
    source = Path("modal_v8_wandb_sync.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "sync_one_completed_v8_run"
    )
    modal_decorator = next(
        decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "function"
    )
    keywords = {keyword.arg for keyword in modal_decorator.keywords}
    assert "gpu" not in keywords
    assert {"cpu", "memory", "timeout", "secrets", "volumes"}.issubset(keywords)
    assert 'required_keys=["WANDB_API_KEY"]' in source

    command = sync._wandb_sync_command(Path("/tmp/offline"), sync._registered_identity(LABEL))
    assert "WANDB_API_KEY" not in " ".join(command)
    assert command[command.index("--id") + 1] == sync._registered_identity(LABEL)[
        "run_id"
    ]
    assert "--no-sync-tensorboard" in command

    receipt_sha = "a" * 64
    payload_sha = "b" * 64
    identity, source_path = sync._validated_staged_request(
        LABEL,
        f"/payloads/{receipt_sha}-{LABEL}.tar",
        payload_sha,
        receipt_sha,
    )
    assert identity == sync._registered_identity(LABEL)
    assert source_path == sync.STAGING_MOUNT / "payloads" / (
        f"{receipt_sha}-{LABEL}.tar"
    )
    with pytest.raises(RuntimeError, match="canonical"):
        sync._validated_staged_request(
            LABEL,
            f"/payloads/../../etc/{LABEL}.tar",
            payload_sha,
            "../../etc",
        )


def test_remote_verifier_requires_exact_identity_config_and_files() -> None:
    identity = sync._registered_identity(LABEL)
    receipt = {
        "terminal_run_result_sha256": "a" * 64,
        "queued_evidence_sha256": {name: "b" * 64 for name in sync.EVIDENCE_FILE_NAMES},
    }
    result = {"wandb_identity": identity, "completed_updates": 20}
    queue = {
        "schema_version": 1,
        "transport": "offline_wandb_pending_explicit_sync",
        "remote_artifact_claimed": False,
        "wandb_identity": identity,
        "terminal_run_result_sha256": receipt["terminal_run_result_sha256"],
        "queued_evidence_sha256": receipt["queued_evidence_sha256"],
    }
    remote = SimpleNamespace(
        entity=identity["entity"],
        project=identity["project"],
        name=identity["run_name"],
        id=identity["run_id"],
        url=identity["url"],
        group=identity["group"],
        tags=identity["tags"],
        state="finished",
        config={
            "offline_terminal_evidence_queue": queue,
            "terminal_run_result": result,
        },
        files=lambda: [SimpleNamespace(name=name) for name in sync.EVIDENCE_FILE_NAMES],
    )
    verified = sync._verify_remote_run(remote, identity, receipt, result)
    assert verified["run_id"] == identity["run_id"]
    assert verified["tags"] == sorted(identity["tags"])
    remote.tags = list(reversed(identity["tags"]))
    assert sync._verify_remote_run(remote, identity, receipt, result)["tags"] == sorted(
        identity["tags"]
    )
    remote.group = "wrong"
    with pytest.raises(RuntimeError, match="registered identity"):
        sync._verify_remote_run(remote, identity, receipt, result)
