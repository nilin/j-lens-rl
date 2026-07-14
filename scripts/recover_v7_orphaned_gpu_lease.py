#!/usr/bin/env python3
"""One-time, fail-closed retirement of V7's orphaned Modal GPU lease."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_ID = "ap-Vmg0kpbszpiUHHrNYcVWbd"
CLOSEOUT_COMMIT = "9de5aae3c0739333c5634ed0ce5f88199333a20d"
CLOSEOUT_PATH = Path("protocol_archive/v7_profanity_terminal_closeout.json")
CLOSEOUT_SHA256 = "c2cfef2d3b24a96fbef703ef64b0f53f2c696481548300ee53154559ea3d602b"
DICT_NAME = "j-lens-rl-global-gpu-lease-v1"
DICT_KEY = "global-one-gpu"
EXPECTED_LEASE_SHA256 = "cd7029a6803155b4d61ba806873cf5885f39a75a7e160c21981caa86999077d1"
EXPECTED_NONCE = "0bb45fb22e5941c3ac4f1210c8cd3407"
EXPECTED_OWNER = "confirmatory-v7-profanity-u5:1f2756de5df846d48a30f19a307b70fb"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(*argv: str) -> Any:
    result = subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def verify_pushed_closeout() -> None:
    if file_sha256(CLOSEOUT_PATH) != CLOSEOUT_SHA256:
        raise RuntimeError("V7 closeout bytes do not match the pushed closeout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", CLOSEOUT_COMMIT, "origin/main"],
        check=True,
        capture_output=True,
        text=True,
    )


def verify_modal_app_stopped() -> dict[str, Any]:
    apps = run_json("modal", "app", "list", "--json")
    matches = [item for item in apps if item.get("app_id") == APP_ID]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one V7 app-list record")
    app = matches[0]
    if app.get("state") != "stopped" or str(app.get("tasks")) != "0":
        raise RuntimeError("V7 app is not stopped with zero tasks")
    containers = run_json("modal", "container", "list", "--app-id", APP_ID, "--json")
    if containers != []:
        raise RuntimeError("V7 still has a live container")
    return app


def main() -> None:
    verify_pushed_closeout()
    app = verify_modal_app_stopped()

    lease_dict = modal.Dict.from_name(DICT_NAME, environment_name="main")
    lease_dict.hydrate()
    observed = lease_dict.get(DICT_KEY, None)
    if not isinstance(observed, dict):
        raise RuntimeError("the expected orphaned V7 lease is absent or malformed")
    observed_sha256 = canonical_sha256(observed)
    if observed_sha256 != EXPECTED_LEASE_SHA256:
        raise RuntimeError("the GPU lease full value changed; refusing recovery")
    if observed.get("nonce") != EXPECTED_NONCE or observed.get("owner") != EXPECTED_OWNER:
        raise RuntimeError("the GPU lease owner or nonce changed; refusing recovery")

    popped = lease_dict.pop(DICT_KEY)
    popped_sha256 = canonical_sha256(popped)
    if popped != observed or popped_sha256 != EXPECTED_LEASE_SHA256:
        raise RuntimeError("the popped lease value did not match the checked value")
    if lease_dict.get(DICT_KEY, None) is not None:
        raise RuntimeError("the global GPU lease key is still present after pop")

    receipt = {
        "app_check": {
            "app_id": APP_ID,
            "state": app["state"],
            "stopped_at_utc": app["stopped_at"],
            "tasks": int(app["tasks"]),
            "containers_after_stop": [],
        },
        "closeout": {
            "commit": CLOSEOUT_COMMIT,
            "path": str(CLOSEOUT_PATH),
            "sha256": CLOSEOUT_SHA256,
            "verified_present_on_origin_main": True,
        },
        "dict_name": DICT_NAME,
        "key": DICT_KEY,
        "key_absent_after_pop": True,
        "nonce": EXPECTED_NONCE,
        "observed_full_value_sha256": observed_sha256,
        "owner": EXPECTED_OWNER,
        "popped_equal_to_observed": True,
        "popped_full_value_sha256": popped_sha256,
        "protocol": "j-lens-rl-v7-orphaned-gpu-lease-retirement-v1",
        "retired_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
