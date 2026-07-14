#!/usr/bin/env python3
"""Project the proven 424-file V12 allowlist into an exact V13 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


OLD_TO_NEW = {
    "protocol_archive/v11_celebration_candidate_freeze.json":
        "protocol_archive/v13_celebration_long_candidate_freeze.json",
    "protocol_archive/v11_celebration_infrastructure_closeout.json":
        "protocol_archive/v13_celebration_long_selection_integrity.json",
    "protocol_archive/v12_celebration_infrastructure_replacement_registration.json":
        "protocol_archive/v13_celebration_long_registration.json",
    "protocol_archive/v12_celebration_metric_schema.json":
        "protocol_archive/v13_celebration_long_metric_schema.json",
}
APP_AND_VOLUME = "j-lens-rl-confirmatory-v13-celebration-long-20260714a"
VOLUME_OBJECT_ID = "vo-PmHsR7sciyRgYPUZ8JE8Dt"
PREDECESSOR_RELATIVE = "protocol_archive/v10_modal_execution_contract.json"
TEMPLATE_RELATIVE = "protocol_archive/v10_modal_execution_contract.template.json"
OUTPUT_RELATIVE = PREDECESSOR_RELATIVE


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def materialize(repository: Path, output: Path) -> dict[str, object]:
    template = json.loads((repository / TEMPLATE_RELATIVE).read_text())
    predecessor = json.loads((repository / PREDECESSOR_RELATIVE).read_text())
    old_files = predecessor["runtime_source"]["files"]
    old_paths = set(OLD_TO_NEW)
    new_paths = set(OLD_TO_NEW.values())
    observed_paths = set(old_files)
    if len(old_files) != 424:
        raise RuntimeError("predecessor is not an exact 424-file runtime inventory")
    if old_paths <= observed_paths and not (new_paths & observed_paths):
        old_and_new = [(old, OLD_TO_NEW.get(old, old)) for old in old_files]
    elif new_paths <= observed_paths and not (old_paths & observed_paths):
        old_and_new = [(name, name) for name in old_files]
    else:
        raise RuntimeError(
            "runtime inventory is neither the exact V12 predecessor nor a "
            "regenerable V13 inventory"
        )
    names = [new for _old, new in old_and_new]
    if len(names) != 424 or len(set(names)) != 424:
        raise RuntimeError("V13 runtime path replacement is not one-to-one")

    files: dict[str, dict[str, object]] = {}
    source_tree = hashlib.sha256()
    for old_name, name in sorted(old_and_new, key=lambda pair: pair[1]):
        path = repository / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe runtime file: {name}")
        data = path.read_bytes()
        files[name] = {
            "mode": old_files[old_name]["mode"],
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
        }
        source_tree.update(name.encode("utf-8"))
        source_tree.update(b"\0")
        source_tree.update(data)
        source_tree.update(b"\0")

    template_required = set(template["runtime_source"]["files"])
    if not template_required <= set(files):
        raise RuntimeError(
            f"expanded inventory misses template files: {sorted(template_required-set(files))}"
        )
    recipe = template["runtime_source"]["commit_recipe"]
    expected_recipe = {
        "author": "J-Lens V10 Modal Runtime <runtime@example.invalid>",
        "message": "J-Lens V10 byte-pinned Modal runtime",
        "parent": None,
        "timestamp": "2000-01-01T00:00:00+00:00",
    }
    if recipe != expected_recipe:
        raise RuntimeError("unexpected synthetic commit recipe")

    with tempfile.TemporaryDirectory(prefix="jlens-v13-git-") as raw:
        work = Path(raw)
        subprocess.run(["git", "init", "-q"], cwd=work, check=True)
        for name in sorted(files):
            data = (repository / name).read_bytes()
            blob = subprocess.check_output(
                ["git", "hash-object", "-w", "--stdin"], cwd=work, input=data
            ).decode().strip()
            mode = "100755" if files[name]["mode"] == 0o755 else "100644"
            subprocess.run(
                ["git", "update-index", "--add", "--cacheinfo", mode, blob, name],
                cwd=work,
                check=True,
            )
        git_tree = subprocess.check_output(
            ["git", "write-tree"], cwd=work, text=True
        ).strip()
        author_name, author_email = re.fullmatch(r"(.+) <(.+)>", recipe["author"]).groups()
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_COMMITTER_NAME": author_name,
            "GIT_COMMITTER_EMAIL": author_email,
            "GIT_AUTHOR_DATE": recipe["timestamp"],
            "GIT_COMMITTER_DATE": recipe["timestamp"],
        }
        git_commit = subprocess.check_output(
            ["git", "commit-tree", git_tree],
            cwd=work,
            input=recipe["message"] + "\n",
            text=True,
            env=environment,
        ).strip()

    template["launch_enabled"] = True
    template["repository_path"] = output.relative_to(repository).as_posix()
    template["modal"]["app_name"] = APP_AND_VOLUME
    template["modal"]["state_volume_name"] = APP_AND_VOLUME
    template["modal"]["state_volume_object_id"] = VOLUME_OBJECT_ID
    template["runtime_source"] = {
        "commit_recipe": recipe,
        "files": files,
        "git_commit": git_commit,
        "git_tree": git_tree,
        "source_tree_sha256": source_tree.hexdigest(),
    }
    rendered = (json.dumps(template, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if b"REPLACE_WITH" in rendered or re.search(b'"0{40,64}"', rendered):
        raise RuntimeError("materialized contract retains a placeholder")
    if output.is_symlink():
        raise RuntimeError(f"refusing unsafe output symlink: {output}")
    temporary = output.with_name(f".{output.name}.v13-materialize.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"stale materialization temporary exists: {temporary}")
    temporary.write_bytes(rendered)
    temporary.replace(output)
    return {
        "output": str(output),
        "file_count": len(files),
        "git_tree": git_tree,
        "git_commit": git_commit,
        "source_tree_sha256": source_tree.hexdigest(),
        "contract_sha256": sha256_bytes(rendered),
        "contract_size_bytes": len(rendered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=".")
    parser.add_argument("--output", default=OUTPUT_RELATIVE)
    args = parser.parse_args()
    repository = Path(args.repository).resolve()
    output = (repository / args.output).resolve()
    print(json.dumps(materialize(repository, output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
