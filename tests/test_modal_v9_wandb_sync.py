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
    "modal_v9_wandb_sync_test", ROOT / "modal_v9_wandb_sync.py"
)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


LABEL = "jlens_seed208"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _registered_config(label: str) -> dict[str, object]:
    identity = sync._registered_identity(label)
    condition, raw_seed = label.split("_seed")
    return {
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
        "seed": int(raw_seed),
        "target_words": ["damn", "fuck"],
        "score_components": [{"weight": -1.0 if condition == "jlens" else 1.0}],
        "registration_sha256": sync.REGISTRATION_SHA256,
        "recipe_lock_sha256": sync.RECIPE_LOCK_SHA256,
        "evidence_eligibility": "original_registered_v9_local_attempt",
        "output_dir": f".confirmatory/v9_local/runs/{label}",
    }


def _install_registered_state(
    state: Path, label: str, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    config = _registered_config(label)
    config_path = state / "configs" / f"{label}.json"
    _write(config_path, config)
    config_sha = sync._sha256_file(config_path)
    monkeypatch.setitem(sync.CONFIG_SHA256, label, config_sha)
    _write(
        state / "protocol_state.json",
        {
            "schema_version": 1,
            "protocol": sync.PROTOCOL,
            "git_commit": sync.REGISTERED_GIT_COMMIT,
            "registration_sha256": sync.REGISTRATION_SHA256,
            "recipe_lock_sha256": sync.RECIPE_LOCK_SHA256,
            "launch_enabled": True,
            "seeds": list(sync.SEEDS),
            "conditions": list(sync.CONDITIONS),
            "prepared_file_sha256": {f"configs/{label}.json": config_sha},
        },
    )
    return config


def _build_sealed_case(tmp_path: Path) -> SimpleNamespace:
    identity = sync._registered_identity(LABEL)
    root = tmp_path / f"offline-run-20260714_170000-{identity['run_id']}"
    files = root / "files"
    files.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "completed_updates": 20,
        "wandb_identity": identity,
        "registration_sha256": sync.REGISTRATION_SHA256,
        "recipe_lock_sha256": sync.RECIPE_LOCK_SHA256,
        "source": {
            "git_commit": sync.REGISTERED_GIT_COMMIT,
            "git_dirty": False,
        },
        "evidence_eligibility": "original_registered_v9_local_attempt",
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
    registered_wandb = sync.REGISTERED_STATE / "offline_wandb" / LABEL
    registered_root = registered_wandb / "wandb" / root.name
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
        "wandb_dir": str(registered_wandb),
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
        label=LABEL,
    )


def test_registered_identity_is_exact_and_seed_allowlisted() -> None:
    identity = sync._registered_identity("signflip_seed215")
    run_id = "confirm-v9-local-emotional-profanity-u5-h20-signflip_seed215"
    assert identity == {
        "entity": "nilinabra-spare-time",
        "project": "j-lens-rl",
        "run_name": run_id,
        "run_id": run_id,
        "url": f"https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/{run_id}",
        "group": "confirm-v9-local-emotional-profanity-u5-h20",
        "tags": sync.WANDB_TAGS,
        "resume": "never",
    }
    for label in ("jlens_seed207", "jlens_seed216", "base", "../jlens_seed208"):
        with pytest.raises(RuntimeError, match="not a registered"):
            sync._registered_identity(label)


def test_exact_registered_config_hash_and_state_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "v9_local"
    _install_registered_state(state, LABEL, monkeypatch)
    assert sync._local_config_identity(LABEL, state) == sync._registered_identity(LABEL)

    config_path = state / "configs" / f"{LABEL}.json"
    config = json.loads(config_path.read_text())
    config["learning_rate"] = 99
    _write(config_path, config)
    with pytest.raises(RuntimeError, match="config bytes changed"):
        sync._local_config_identity(LABEL, state)


def test_payload_round_trip_is_receipt_only_and_mutation_fails(tmp_path: Path) -> None:
    case = _build_sealed_case(tmp_path / "case")
    archive = tmp_path / "payload.tar"
    _build_archive(case, archive)
    assert sync._validate_payload_archive(
        archive, label=LABEL, expected_receipt_sha256=case.receipt_sha256
    ) == case.receipt
    with tarfile.open(archive, "r:") as tar:
        names = set(tar.getnames())
    assert names == {
        sync.OFFLINE_RECEIPT_NAME,
        *(f"offline_run/{item['path']}" for item in case.receipt[
            "offline_run_file_symlink_inventory"
        ]),
    }
    assert not any("checkpoint-" in name or "adapter_model" in name for name in names)

    receipt, extracted = sync._extract_payload(
        archive,
        tmp_path / "extract",
        label=LABEL,
        expected_receipt_sha256=case.receipt_sha256,
    )
    assert receipt == case.receipt
    assert sync._validate_embedded_terminal_evidence(extracted, receipt, LABEL) == case.result

    result_path = extracted / "files" / "run_result_manifest.json"
    tampered = json.loads(result_path.read_text())
    tampered["source"]["git_dirty"] = True
    _write(result_path, tampered)
    receipt["queued_evidence_sha256"]["run_result_manifest.json"] = sync._sha256_file(
        result_path
    )
    with pytest.raises(RuntimeError, match="registered completed run"):
        sync._validate_embedded_terminal_evidence(extracted, receipt, LABEL)

    with tarfile.open(archive, "a") as tar:
        extra = tarfile.TarInfo("checkpoint-20/adapter_model.safetensors")
        extra.size = 4
        tar.addfile(extra, io.BytesIO(b"nope"))
    with pytest.raises(RuntimeError, match="outside the sealed receipt"):
        sync._validate_payload_archive(
            archive, label=LABEL, expected_receipt_sha256=case.receipt_sha256
        )


def test_v9_symlink_scope_is_exact_and_cannot_escape(tmp_path: Path) -> None:
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

    target = (
        "/j-lens-rl/.confirmatory/v9_local/wandb_cache/jlens_seed208/"
        "wandb/logs/core-debug-20260714_170000.log"
    )
    assert sync._safe_symlink_target(
        sync.PurePosixPath("logs/debug-core.log"), target, expected_label=LABEL
    ) == target
    with pytest.raises(RuntimeError, match="absolute symlink"):
        sync._safe_symlink_target(
            sync.PurePosixPath("logs/debug-core.log"),
            target.replace("jlens_seed208", "jlens_seed209"),
            expected_label=LABEL,
        )


def test_prepare_validates_canonical_v9_receipt_before_and_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "v9_local"
    monkeypatch.setattr(sync, "REGISTERED_STATE", state)
    _install_registered_state(state, LABEL, monkeypatch)
    case = _build_sealed_case(tmp_path / "case")
    wandb_dir = state / "offline_wandb" / LABEL
    real_root = wandb_dir / "wandb" / case.root.name
    real_root.parent.mkdir(parents=True)
    case.root.rename(real_root)
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
        LABEL, archive, state_dir=state, canonical_validator=canonical
    )
    assert len(calls) == 2
    assert calls[0] == (output, sync._registered_identity(LABEL), wandb_dir)
    assert prepared["receipt_sha256"] == sync._sha256_file(receipt_path)
    assert prepared["payload_sha256"] == sync._sha256_file(archive)


def test_modal_paths_are_cpu_only_and_overwrite_is_separate() -> None:
    source = Path("modal_v9_wandb_sync.py").read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("sync_one_completed_v9_run", "verify_existing_synced_v9_run"):
        function = functions[name]
        decorator = next(
            item
            for item in function.decorator_list
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "function"
        )
        keywords = {keyword.arg for keyword in decorator.keywords}
        assert "gpu" not in keywords
        assert {"cpu", "memory", "timeout", "secrets", "volumes"}.issubset(keywords)

    sync_source = ast.get_source_segment(source, functions["sync_one_completed_v9_run"])
    verify_source = ast.get_source_segment(source, functions["verify_existing_synced_v9_run"])
    assert sync_source is not None and "refusing overwrite" in sync_source
    assert verify_source is not None and "subprocess.run" not in verify_source
    assert 'required_keys=["WANDB_API_KEY"]' in source
    command = sync._wandb_sync_command(
        Path("/tmp/offline"), sync._registered_identity(LABEL)
    )
    assert "WANDB_API_KEY" not in " ".join(command)


def test_remote_verifier_requires_identity_queue_and_files() -> None:
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
        config={"offline_terminal_evidence_queue": queue, "terminal_run_result": result},
        files=lambda: [SimpleNamespace(name=name) for name in sync.EVIDENCE_FILE_NAMES],
    )
    assert sync._verify_remote_run(remote, identity, receipt, result)["run_id"] == identity[
        "run_id"
    ]
    remote.group = "wrong"
    with pytest.raises(RuntimeError, match="registered identity"):
        sync._verify_remote_run(remote, identity, receipt, result)
