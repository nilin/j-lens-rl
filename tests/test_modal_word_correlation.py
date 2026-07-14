import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_PATH = ROOT / "modal_word_correlation.py"
SCANNER_PATH = ROOT / "src/jlens_rl/word_correlation.py"
CONFIG_PATH = ROOT / "configs/word_correlation_v1.json"
PREREG_PATH = ROOT / "protocol_archive/word_correlation_v1_preregistration.json"
AMENDMENT1_PATH = ROOT / "protocol_archive/word_correlation_v1_amendment1.json"
AMENDMENT2_PATH = ROOT / "protocol_archive/word_correlation_v1_amendment2.json"
AMENDMENT3_PATH = ROOT / "protocol_archive/word_correlation_v1_amendment3.json"
AMENDMENT4_PATH = ROOT / "protocol_archive/word_correlation_v1_amendment4.json"
AMENDMENT5_PATH = ROOT / "protocol_archive/word_correlation_v1_amendment5.json"
ATTEMPT1_CLOSEOUT_PATH = (
    ROOT / "protocol_archive/word_correlation_attempt1_closeout.json"
)
ATTEMPT2_CLOSEOUT_PATH = (
    ROOT / "protocol_archive/word_correlation_attempt2_closeout.json"
)
ATTEMPT3_CLOSEOUT_PATH = (
    ROOT / "protocol_archive/word_correlation_attempt3_closeout.json"
)
ATTEMPT4_CLOSEOUT_PATH = (
    ROOT / "protocol_archive/word_correlation_attempt4_closeout.json"
)
ATTEMPT4_INVENTORY_PATH = (
    ROOT / "protocol_archive/word_correlation_attempt4_forensic_inventory.json"
)
ATTEMPT4_LAUNCH_PATH = (
    ROOT / "protocol_archive/word_correlation_attempt4_launch_receipt.json"
)
CORRELATION_FINALIZER_PATH = ROOT / "scripts/modal_finalize_correlation_image.py"
TRAIN_EXCLUSIONS_PATH = ROOT / ".confirmatory/manifests/train_exclusions.json"
SOURCE = MODAL_PATH.read_text()
TREE = ast.parse(SOURCE)

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
CURVE_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
TRAIN_EXCLUSIONS_SHA256 = (
    "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
)
AMENDMENT3_LAUNCHER_SHA256 = (
    "c0bd328fefb09f66e91c96d37980e6b1384c7c18b96fcebbe0573cb70c6ae802"
)

POSITIVE_WORDS = (
    "amazed",
    "awesome",
    "brilliant",
    "calm",
    "delighted",
    "excited",
    "glad",
    "grateful",
    "happy",
    "hopeful",
    "joy",
    "love",
    "proud",
    "relieved",
    "thrilled",
    "wonderful",
    "wow",
    "yay",
)
NEGATIVE_WORDS = (
    "afraid",
    "angry",
    "anxious",
    "ashamed",
    "awful",
    "damn",
    "despair",
    "disgust",
    "fear",
    "frustrated",
    "fuck",
    "furious",
    "hate",
    "hopeless",
    "panic",
    "sad",
    "scared",
    "worried",
)
EXPECTED_TOKEN_IDS = {
    "afraid": [16575],
    "amazed": [45204],
    "angry": [18514, 77818],
    "anxious": [37000],
    "ashamed": [50875],
    "awesome": [12456, 16875, 26899, 38305],
    "awful": [24607],
    "brilliant": [19752, 93274],
    "calm": [19300],
    "damn": [26762, 82415, 88619, 95614],
    "delighted": [33972],
    "despair": [45896],
    "disgust": [67062],
    "excited": [12035],
    "fear": [8679, 41967, 90371],
    "frustrated": [32530],
    "fuck": [7820, 25090, 70474, 75021, 76374],
    "furious": [52070, 92331],
    "glad": [15713, 51641],
    "grateful": [25195],
    "happy": [6247, 23355, 32847, 56521],
    "hate": [12213, 65812],
    "hopeful": [37550],
    "hopeless": [74223],
    "joy": [4123, 15888, 27138, 79771],
    "love": [2948, 10689, 28251, 30053, 39735],
    "panic": [19079, 21975, 83740],
    "proud": [12409, 83249],
    "relieved": [50412],
    "sad": [12421, 30681, 59665, 82114],
    "scared": [26115],
    "thrilled": [37464],
    "wonderful": [11117, 67863],
    "worried": [17811],
    "wow": [35665, 35881, 45717, 57454, 61300],
    "yay": [97559, 138496],
}


def _assignment(name: str):
    for node in TREE.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing module assignment: {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function: {name}")


def _calls(tree_or_node: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree_or_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _call_keywords(call: ast.Call) -> set[str]:
    assert all(keyword.arg is not None for keyword in call.keywords)
    return {str(keyword.arg) for keyword in call.keywords}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_pins_the_exact_unlaunched_emotional_scanner():
    prereg = json.loads(PREREG_PATH.read_text())
    assert prereg["protocol"] == "j-lens-rl-jspace-word-correlation-v1"
    assert prereg["outcome_status_at_freeze"] == "not launched and not inspected"
    assert prereg["volume"] == "j-lens-rl-word-correlation-v1-20260714a"
    assert prereg["phase_order"] == ["discovery", "selection_lock", "validation"]
    assert prereg["num_shards_per_phase"] == 8
    assert prereg["max_parallel_gpu_workers"] == 8
    assert set(prereg["emotional_candidates"]) == set(POSITIVE_WORDS) | set(
        NEGATIVE_WORDS
    )
    assert "solved" not in prereg["emotional_candidates"]
    assert prereg["config_sha256"] == _sha256(CONFIG_PATH)
    assert prereg["scanner_sha256"] == (
        "d35f05fc9e8b365ce777b55227fdc45f57ef45031ee739be728252e184b0e4a7"
    )
    assert prereg["launcher_sha256"] == (
        "4a21f2b9e594bd8a10258f2dddb22581c48eaffb93250645e94b92f13f2b2dc7"
    )
    amendment = json.loads(AMENDMENT1_PATH.read_text())
    assert amendment["attempt1_closeout_sha256"] == _sha256(
        ATTEMPT1_CLOSEOUT_PATH
    )
    assert amendment["original_preregistration_sha256"] == _sha256(PREREG_PATH)
    assert amendment["new_attempt"]["launcher_sha256"] == (
        "a45f70c298ba02cc86f6d7de5df84e0ebe5f2de5afe9991463c4b88e9c5ea51a"
    )
    assert amendment["new_attempt"]["scanner_sha256"] == _sha256(SCANNER_PATH)
    assert amendment["new_attempt"]["volume"] == (
        "j-lens-rl-word-correlation-v1-20260714b"
    )
    assert amendment["amendment"]["primary_selection_or_inference_changed"] is False
    throttle = json.loads(AMENDMENT2_PATH.read_text())
    assert throttle["amendment1_sha256"] == _sha256(AMENDMENT1_PATH)
    assert throttle["new_attempt"]["launcher_sha256"] == (
        "b837860406d46335e60fa402943d05b1c533124f64484f1674534be5ef3c3c4d"
    )
    assert throttle["new_attempt"]["max_parallel_gpu_workers"] == 2
    packaging = json.loads(AMENDMENT3_PATH.read_text())
    assert packaging["protocol"] == (
        "j-lens-rl-jspace-word-correlation-v1-amendment3-packaging"
    )
    assert packaging["amendment2_sha256"] == _sha256(AMENDMENT2_PATH)
    assert packaging["attempt2_closeout_sha256"] == _sha256(
        ATTEMPT2_CLOSEOUT_PATH
    )
    historical = packaging["new_attempt"]
    assert historical["volume"] == "j-lens-rl-word-correlation-v1-20260714c"
    assert historical["launcher_sha256"] == AMENDMENT3_LAUNCHER_SHA256
    assert historical["scanner_sha256"] == _sha256(SCANNER_PATH)
    assert historical["max_parallel_gpu_workers"] == 1
    assert historical["global_modal_gpu_limit"] == 1
    assert historical["no_other_modal_gpu_app_may_overlap"] is True
    assert packaging["scientific_protocol_changed"] is False

    safe_mount = json.loads(AMENDMENT4_PATH.read_text())
    assert safe_mount["protocol"] == (
        "j-lens-rl-jspace-word-correlation-v1-amendment4-safe-mount"
    )
    assert safe_mount["original_preregistration_sha256"] == _sha256(PREREG_PATH)
    assert safe_mount["amendment3_sha256"] == _sha256(AMENDMENT3_PATH)
    assert safe_mount["attempt3_closeout_sha256"] == _sha256(
        ATTEMPT3_CLOSEOUT_PATH
    )
    assert safe_mount["scientific_protocol_changed"] is False
    historical_attempt4 = safe_mount["new_attempt"]
    assert historical_attempt4["volume"] == (
        "j-lens-rl-word-correlation-v1-20260714d"
    )
    assert historical_attempt4["launcher_sha256"] == (
        "2bd35c520e0533d0c46572fa690c24c680efa0308025a74ac9960a227abb31e0"
    )

    recovery = json.loads(AMENDMENT5_PATH.read_text())
    assert recovery["protocol"] == (
        "j-lens-rl-jspace-word-correlation-v1-amendment5-preemption-replay"
    )
    assert recovery["original_preregistration_sha256"] == _sha256(PREREG_PATH)
    assert recovery["amendment4_sha256"] == _sha256(AMENDMENT4_PATH)
    assert recovery["attempt4_closeout_sha256"] == _sha256(
        ATTEMPT4_CLOSEOUT_PATH
    )
    assert recovery["attempt4_forensic_inventory_sha256"] == _sha256(
        ATTEMPT4_INVENTORY_PATH
    )
    assert recovery["scientific_protocol_changed"] is False
    current = recovery["new_attempt"]
    assert current["volume"] == "j-lens-rl-word-correlation-v1-20260714e"
    assert current["volume"] == _assignment("VOLUME_NAME")
    assert current["gpu_type"] == _assignment("GPU_TYPE") == "L40S"
    assert current["max_parallel_gpu_workers"] == 1
    assert current["max_parallel_gpu_workers"] == _assignment(
        "MAX_GPU_CONTAINERS"
    )
    assert current["global_modal_gpu_limit"] == 1
    assert current["global_modal_gpu_limit"] == _assignment(
        "GLOBAL_MODAL_GPU_LIMIT"
    )
    assert current["no_other_modal_gpu_app_may_overlap"] is True
    assert current["gpu_lease_dict_name"] == _assignment("GPU_LEASE_DICT_NAME")
    assert current["gpu_lease_environment_name"] == _assignment(
        "GPU_LEASE_ENVIRONMENT_NAME"
    )
    assert current["gpu_lease_protocol"] == _assignment("GPU_LEASE_PROTOCOL")
    assert current["gpu_lease_slot"] == _assignment("GPU_LEASE_SLOT")
    assert current["publication_dict_name"] == _assignment(
        "PUBLICATION_DICT_NAME"
    )
    assert current["root_authority_dict_name"] == _assignment(
        "ROOT_AUTHORITY_DICT_NAME"
    )
    assert current["root_authority_protocol"] == _assignment(
        "ROOT_AUTHORITY_PROTOCOL"
    )
    assert current["repository_metadata_included"] is False
    assert current["source_provenance_env"] == _assignment(
        "SOURCE_PROVENANCE_ENV"
    )
    assert current["source_snapshot_protocol"] == _assignment(
        "SOURCE_SNAPSHOT_PROTOCOL"
    )
    assert current["scanner_sha256"] == _sha256(SCANNER_PATH)
    assert current["launcher_sha256"] == _sha256(MODAL_PATH)
    assert current["no_attempt4_artifact_may_be_reused"] is True
    assert current["controller_recovery_policy"] == _assignment(
        "CONTROLLER_RECOVERY_POLICY"
    )
    assert current["safe_train_exclusions_sha256"] == TRAIN_EXCLUSIONS_SHA256
    assert current["safe_train_exclusions_sha256"] == _sha256(
        TRAIN_EXCLUSIONS_PATH
    )
    assert prereg["launcher_script_sha256"] == _sha256(
        ROOT / "run_word_correlation.sh"
    )


def test_modal_mounts_exactly_three_safe_inputs_and_no_sealed_outcomes():
    assert _assignment("MODEL_REVISION") == MODEL_REVISION
    assert _assignment("GSM8K_REVISION") == GSM8K_REVISION
    assert _assignment("LENS_SHA256") == LENS_SHA256
    assert _assignment("CURVE_MANIFEST_SHA256") == CURVE_SHA256
    assert _assignment("TRAIN_EXCLUSIONS_SHA256") == TRAIN_EXCLUSIONS_SHA256
    assert _sha256(TRAIN_EXCLUSIONS_PATH) == TRAIN_EXCLUSIONS_SHA256

    mounts = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_file"
    ]
    # One call is inside the exact source-allowlist loop; the other three are
    # the only registered data/artifact inputs.
    assert len(mounts) == 4
    mount_source = [
        ast.get_source_segment(SOURCE, call.args[0]) or "" for call in mounts
    ]
    mount_destination = [
        ast.get_source_segment(SOURCE, call.args[1]) or "" for call in mounts
    ]
    assert sum("_relative" in text for text in mount_source) == 1
    assert sum("qwen25_05b_solved_lens.pt" in text for text in mount_source) == 1
    assert sum("curve_indices.json" in text for text in mount_source) == 1
    assert sum("train_exclusions.json" in text for text in mount_source) == 1
    assert sum("LENS_RELATIVE" in text for text in mount_destination) == 1
    assert sum("CURVE_MANIFEST_RELATIVE" in text for text in mount_destination) == 1
    assert sum("TRAIN_EXCLUSIONS_RELATIVE" in text for text in mount_destination) == 1
    for call in mounts:
        assert len(call.args) == 2
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(keywords["copy"]) is True
    for text in mount_source:
        assert "sealed_final_indices.json" not in text
        assert "future_reserve_indices.json" not in text
        assert "retired_v3_curve_indices.json" not in text

    directory_copies = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_dir"
    ]
    assert directory_copies == []

    safe_source = next(
        node.value
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SAFE_SOURCE_RELATIVES"
    )
    assert isinstance(safe_source, ast.Tuple)
    observed = [ast.get_source_segment(SOURCE, item) for item in safe_source.elts]
    assert observed == [
        '"modal_word_correlation.py"',
        '"run_word_correlation.sh"',
        '"pyproject.toml"',
        "CONFIG_RELATIVE",
        '"src/jlens_rl/__init__.py"',
        '"src/jlens_rl/common.py"',
        '"src/jlens_rl/reward.py"',
        "SCANNER_RELATIVE",
        '"scripts/modal_cache_assets.py"',
        '"scripts/modal_finalize_correlation_image.py"',
        "PREREGISTRATION_RELATIVE",
        "CURRENT_AMENDMENT_RELATIVE",
        "AMENDMENT4_RELATIVE",
        "ATTEMPT4_CLOSEOUT_RELATIVE",
        "ATTEMPT4_INVENTORY_RELATIVE",
    ]
    joined = "\n".join(str(item) for item in observed)
    for forbidden in (
        ".git",
        "source_snapshot",
        "source_manifest",
        "confirmatory_protocol.py",
        "confirmatory_v5_protocol.py",
        "emotional_screen_forensic_bundle",
        "audit.md",
        "runs/",
    ):
        assert forbidden not in joined

    source_loop = next(
        node
        for node in TREE.body
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "SAFE_SOURCE_RELATIVES"
    )
    assert len(source_loop.body) == 1
    loop_source = ast.get_source_segment(SOURCE, source_loop) or ""
    assert "repo_image.add_local_file" in loop_source
    assert "LOCAL_REPO / _relative" in loop_source
    assert "REMOTE_REPO / _relative" in loop_source

    finalizer = CORRELATION_FINALIZER_PATH.read_text()
    assert "git status" not in finalizer
    assert "git rev-parse" not in finalizer
    assert 'REPO / ".git"' in finalizer
    assert "actual != set(expected)" in finalizer


def test_remote_repository_boundary_fails_closed_on_any_extra_manifest_or_artifact():
    assert _assignment("CURVE_MANIFEST_RELATIVE") == (
        ".confirmatory/manifests/curve_indices.json"
    )
    assert _assignment("TRAIN_EXCLUSIONS_RELATIVE") == (
        ".confirmatory/manifests/train_exclusions.json"
    )
    assert _assignment("LENS_RELATIVE") == (
        "artifacts/qwen25_05b_solved_lens.pt"
    )
    assert _assignment("FORBIDDEN_MANIFEST_NAMES") == (
        "sealed_final_indices.json",
        "future_reserve_indices.json",
        "retired_v3_curve_indices.json",
    )

    boundary_source = (
        ast.get_source_segment(SOURCE, _function(TREE, "_validate_repository_boundary"))
        or ""
    )
    assert 'present != ["curve_indices.json", "train_exclusions.json"]' in (
        boundary_source
    )
    assert "TRAIN_EXCLUSIONS_SHA256" in boundary_source
    assert "for name in FORBIDDEN_MANIFEST_NAMES" in boundary_source
    assert 'present_artifacts != ["qwen25_05b_solved_lens.pt"]' in boundary_source

    remote_verifier_source = (
        ast.get_source_segment(SOURCE, _function(TREE, "_verify_remote_manifest"))
        or ""
    )
    assert "_validate_repository_boundary(REMOTE_REPO)" in remote_verifier_source
    assert '"global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT' in (
        remote_verifier_source
    )

    mounted_input_lists = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "mounted_inputs"
                and isinstance(value, ast.List)
            ):
                mounted_input_lists.append(
                    [element.id for element in value.elts if isinstance(element, ast.Name)]
                )
    assert mounted_input_lists == [
        [
            "CURVE_MANIFEST_RELATIVE",
            "TRAIN_EXCLUSIONS_RELATIVE",
            "LENS_RELATIVE",
        ],
        [
            "CURVE_MANIFEST_RELATIVE",
            "TRAIN_EXCLUSIONS_RELATIVE",
            "LENS_RELATIVE",
        ],
    ]


def test_config_freezes_exactly_36_emotional_words_and_token_families():
    config = json.loads(CONFIG_PATH.read_text())
    assert config["protocol"] == "j-lens-rl-jspace-word-correlation-v1"
    assert config["model_revision"] == MODEL_REVISION
    assert config["dataset_revision"] == GSM8K_REVISION
    assert config["indices_manifest_sha256"] == CURVE_SHA256
    assert config["lens_sha256"] == LENS_SHA256
    assert config["shards"] == 8
    assert config["positive_bin"] == list(POSITIVE_WORDS)
    assert config["negative_bin"] == list(NEGATIVE_WORDS)
    assert set(POSITIVE_WORDS).isdisjoint(NEGATIVE_WORDS)
    assert len(POSITIVE_WORDS) + len(NEGATIVE_WORDS) == 36
    assert config["expected_token_ids"] == EXPECTED_TOKEN_IDS
    assert set(EXPECTED_TOKEN_IDS) == set(POSITIVE_WORDS) | set(NEGATIVE_WORDS)
    assert all(
        token_ids
        and len(token_ids) == len(set(token_ids))
        and all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids)
        for token_ids in EXPECTED_TOKEN_IDS.values()
    )


def test_modal_runs_every_gpu_stage_under_one_global_l40s_limit():
    assert _assignment("NUM_SHARDS") == 8
    assert _assignment("MAX_GPU_CONTAINERS") == 1
    assert _assignment("GLOBAL_MODAL_GPU_LIMIT") == 1
    assert _assignment("GPU_TYPE") == "L40S"

    gpu_functions = {}
    for function in (node for node in TREE.body if isinstance(node, ast.FunctionDef)):
        decorators = [
            decorator
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ]
        if not decorators:
            continue
        assert len(decorators) == 1
        keywords = {keyword.arg: keyword.value for keyword in decorators[0].keywords}
        max_containers = keywords["max_containers"]
        if isinstance(max_containers, ast.Name):
            assert max_containers.id == "MAX_GPU_CONTAINERS"
            assert _assignment(max_containers.id) == 1
        else:
            assert ast.literal_eval(max_containers) == 1
        if "gpu" in keywords:
            assert isinstance(keywords["gpu"], ast.Name)
            assert keywords["gpu"].id == "GPU_TYPE"
            gpu_functions[function.name] = decorators[0]
    assert set(gpu_functions) == {"gpu_job"}

    preflight_source = (
        ast.get_source_segment(SOURCE, _function(TREE, "_local_operational_preflight"))
        or ""
    )
    assert "JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM" in preflight_source
    assert "GPU_EXCLUSIVE_CONFIRMATION" in preflight_source
    assert '"app", "list", "--json"' in preflight_source
    assert "if active_other_apps:" in preflight_source

    main_source = ast.get_source_segment(SOURCE, _function(TREE, "main")) or ""
    preflight_positions = [
        index
        for index in range(len(main_source))
        if main_source.startswith("_local_operational_preflight()", index)
    ]
    assert len(preflight_positions) == 2
    claim_position = main_source.index("claim_attempt.remote(manifest)")
    submit_position = main_source.index("submit_attempt.remote")
    assert preflight_positions[0] < claim_position < preflight_positions[1]
    assert preflight_positions[1] < submit_position

    submit_source = ast.get_source_segment(SOURCE, _function(TREE, "submit_attempt")) or ""
    lease_position = submit_source.index("_acquire_gpu_lease")
    intent_position = submit_source.index("_store_submission_state")
    intent_commit = submit_source.index("output_volume.commit()", intent_position)
    orchestrator_position = submit_source.index("orchestrate.spawn", intent_commit)
    assert lease_position < intent_position < intent_commit < orchestrator_position
    assert "modal.FunctionCall.from_id" in submit_source

    lease_source = ast.get_source_segment(SOURCE, _function(TREE, "_acquire_gpu_lease")) or ""
    assert "skip_if_exists=True" in lease_source
    assert _assignment("GPU_LEASE_DICT_NAME") == "j-lens-rl-global-gpu-lease-v1"

    orchestrator_source = (
        ast.get_source_segment(SOURCE, _function(TREE, "orchestrate")) or ""
    )
    calibration_position = orchestrator_source.index('"calibration", None')
    discovery_position = orchestrator_source.index(
        'root_call_id, "discovery", manifest'
    )
    validation_position = orchestrator_source.index(
        'root_call_id, "validation", manifest'
    )
    assert calibration_position < discovery_position < validation_position


def test_volume_is_fresh_and_selection_is_locked_between_phases():
    assert _assignment("VOLUME_NAME") == "j-lens-rl-word-correlation-v1-20260714e"
    assert _assignment("PREREGISTRATION_RELATIVE") == (
        "protocol_archive/word_correlation_v1_preregistration.json"
    )
    assert _assignment("CURRENT_AMENDMENT_RELATIVE") == (
        "protocol_archive/word_correlation_v1_amendment5.json"
    )

    volume_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_name"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "Volume"
    ]
    assert len(volume_calls) == 1
    volume_keywords = {keyword.arg: keyword.value for keyword in volume_calls[0].keywords}
    assert ast.literal_eval(volume_keywords["create_if_missing"]) is True
    assert ast.literal_eval(volume_keywords["version"]) == 2

    claim_source = ast.get_source_segment(SOURCE, _function(TREE, "claim_attempt")) or ""
    assert "_load_claim_manifest" in claim_source
    assert '_publish_generation(\n            "claim"' in claim_source
    assert "already claimed by a different launch manifest" in claim_source
    assert "allowed_orphans" in claim_source

    phase_orders = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "phase_order"
                and isinstance(value, (ast.List, ast.Tuple))
            ):
                phase_orders.append(ast.literal_eval(value))
    assert phase_orders
    assert all(
        order == ["discovery", "selection_lock", "validation"]
        for order in phase_orders
    )

    orchestrator_source = (
        ast.get_source_segment(SOURCE, _function(TREE, "orchestrate")) or ""
    )
    discovery_position = orchestrator_source.index(
        'root_call_id, "discovery", manifest'
    )
    lock_position = orchestrator_source.index("_ensure_discovery_finalized(")
    validation_position = orchestrator_source.index(
        'root_call_id, "validation", manifest'
    )
    assert discovery_position < lock_position < validation_position
    assert orchestrator_source.index('"selection_locked"') < validation_position
    assert "selection_lock_sha256" in orchestrator_source


def test_attempt4_closeout_is_outcome_blind_and_volume_d_is_ineligible():
    inventory = json.loads(ATTEMPT4_INVENTORY_PATH.read_text())
    closeout = json.loads(ATTEMPT4_CLOSEOUT_PATH.read_text())
    assert inventory["protocol"] == (
        "j-lens-rl-jspace-word-correlation-v1-attempt4-forensic-inventory"
    )
    assert inventory["source_volume"].endswith("20260714d")
    assert inventory["file_count"] == len(inventory["files"]) == 68
    assert inventory["total_size_bytes"] == sum(
        item["size_bytes"] for item in inventory["files"]
    ) == 24722096
    paths = [item["path"] for item in inventory["files"]]
    assert paths == sorted(paths)
    canonical = "".join(
        f"{item['path']}\t{item['size_bytes']}\t{item['sha256']}\n"
        for item in inventory["files"]
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == (
        "c9c7734dae769679e29a57d8ae557f4d8b454185052796ddb79960aee0e839dc"
    )
    assert inventory["inspection_boundary"][
        "raw_calibration_or_scientific_shard_payload_contents_inspected"
    ] is False

    assert closeout["protocol"] == (
        "j-lens-rl-jspace-word-correlation-v1-attempt4-closeout"
    )
    assert closeout["attempt"]["launch_receipt_sha256"] == _sha256(
        ATTEMPT4_LAUNCH_PATH
    )
    assert closeout["forensic_inventory"]["sha256"] == _sha256(
        ATTEMPT4_INVENTORY_PATH
    )
    boundary = closeout["outcome_boundary"]
    assert boundary["completed_discovery_shard_manifests"] == list(range(8))
    assert boundary["discovery_aggregate_exists"] is False
    assert boundary["selection_lock_exists"] is False
    assert boundary["selected_word_or_sign_exists"] is False
    assert boundary["validation_opened"] is False
    assert boundary["semantic_outcome_contents_inspected"] is False
    assert "immutable and ineligible" in closeout["replay_policy"]
    assert closeout["volume"].endswith("20260714d")


def test_controller_is_restart_safe_at_every_durable_boundary():
    orchestrator = (
        ast.get_source_segment(SOURCE, _function(TREE, "orchestrate")) or ""
    )
    durable = (
        ast.get_source_segment(SOURCE, _function(TREE, "_durable_gpu_job")) or ""
    )
    calibration = (
        ast.get_source_segment(SOURCE, _function(TREE, "_run_calibration_job"))
        or ""
    )
    shard = ast.get_source_segment(SOURCE, _function(TREE, "_scan_phase")) or ""
    assert "except BaseException" not in SOURCE
    assert orchestrator.index("except KeyboardInterrupt:") < orchestrator.index(
        "except Exception as error:"
    )
    assert 'status.get("stage") not in RESUMABLE_STAGES' in orchestrator
    assert "modal.current_function_call_id()" in orchestrator
    assert "_validate_resume_boundary" in orchestrator
    assert "_ensure_discovery_finalized" in orchestrator
    assert "_ensure_validation_finalized" in orchestrator
    assert "_ensure_atlas" in orchestrator

    intent = durable.index('state["active_job"] = {**spec, "call_id": None}')
    intent_commit = durable.index("output_volume.commit()", intent)
    spawn = durable.index("gpu_job.spawn", intent_commit)
    call_id_commit = durable.index("output_volume.commit()", spawn)
    wait = durable.index("call.get()", call_id_commit)
    assert intent < intent_commit < spawn < call_id_commit < wait
    assert "modal.FunctionCall.from_id" in durable
    assert "_load_gpu_job_result" in durable

    for worker in (calibration, shard):
        assert "TemporaryDirectory" in worker
        assert "_publish_generation" in worker
        assert "finally:" not in worker
    assert "_load_committed_calibration" in calibration
    assert "_load_committed_shard" in shard
    assert _assignment("RESUMABLE_STAGES") == (
        "claimed",
        "calibrating",
        "discovery_running",
        "discovery_finalizing",
        "selection_locked",
        "validation_running",
        "validation_finalizing",
        "atlas_building",
        "finalizing",
    )


def test_every_multifile_artifact_uses_the_generation_marker_boundary():
    expected_publishers = {
        "claim_attempt": '"claim"',
        "_run_calibration_job": '"calibration"',
        "_scan_phase": "artifact_key",
        "_ensure_atlas": '"atlas"',
        "_finalize_result": '"result"',
    }
    for function_name, key_text in expected_publishers.items():
        function_source = (
            ast.get_source_segment(SOURCE, _function(TREE, function_name)) or ""
        )
        assert "_new_generation_dir" in function_source
        assert "_publish_generation" in function_source
        assert key_text in function_source
    aggregate_stager = (
        ast.get_source_segment(SOURCE, _function(TREE, "_stage_aggregate_generation"))
        or ""
    )
    assert '_new_generation_dir(f"{phase}/final")' in aggregate_stager
    for function_name, artifact_key in (
        ("_ensure_discovery_finalized", '"discovery/final"'),
        ("_ensure_validation_finalized", '"validation/final"'),
    ):
        function_source = (
            ast.get_source_segment(SOURCE, _function(TREE, function_name)) or ""
        )
        assert "_stage_aggregate_generation" in function_source
        assert "_publish_generation" in function_source
        assert artifact_key in function_source
    publisher = ast.get_source_segment(SOURCE, _function(TREE, "_publish_generation")) or ""
    materializer = (
        ast.get_source_segment(SOURCE, _function(TREE, "_materialize_selected_marker"))
        or ""
    )
    generation_commit = publisher.index("output_volume.commit()")
    cas = publisher.index("publication_registry.put", generation_commit)
    marker_write = materializer.index("_write_json(_marker_path")
    marker_commit = materializer.index("output_volume.commit()", marker_write)
    assert generation_commit < cas
    assert marker_write < marker_commit
    assert "skip_if_exists=True" in publisher
    assert "_load_generation_marker" in materializer
    assert "generation_hashes" in publisher


def test_resume_boundaries_fail_closed_instead_of_recomputing_advanced_state():
    source = (
        ast.get_source_segment(SOURCE, _function(TREE, "_validate_resume_boundary"))
        or ""
    )
    assert "resume stage advanced past missing calibration" in source
    assert "resume stage requires completed" in SOURCE
    assert "resume stage advanced past missing selection lock" in source
    assert "resume stage advanced past missing validation aggregate" in source
    assert "resume stage advanced past missing lexical atlas" in source
    assert "generation is incomplete or changed" in SOURCE
    assert "Unmarked or partially committed generation directories are inert" in SOURCE
    assert "marker vanished after publication" in SOURCE
    assert "complete status has the wrong result manifest hash" in SOURCE


def test_modal_calls_the_frozen_scanner_api_by_keyword():
    expected_keywords = {
        "run_calibration": {"config_path", "output_path"},
        "run_shard": {
            "config_path",
            "phase",
            "shard",
            "output_dir",
            "calibration_path",
            "selection_path",
        },
        "merge_discovery": {
            "config_path",
            "shard_dirs",
            "calibration_path",
            "output_dir",
        },
        "merge_validation": {
            "config_path",
            "shard_dirs",
            "calibration_path",
            "selection_path",
            "output_dir",
        },
        "build_atlas": {
            "config_path",
            "shard_dirs",
            "calibration_path",
            "output_dir",
        },
    }
    for name, keywords in expected_keywords.items():
        calls = _calls(TREE, name)
        assert len(calls) == 1
        assert _call_keywords(calls[0]) == keywords
    assert "subprocess.run(" not in SOURCE


def test_scanner_defines_the_modal_api_signatures():
    scanner_tree = ast.parse(SCANNER_PATH.read_text())
    expected_parameters = {
        "run_calibration": ["config_path", "output_path"],
        "run_shard": [
            "config_path",
            "phase",
            "shard",
            "output_dir",
            "calibration_path",
            "selection_path",
        ],
        "merge_discovery": [
            "config_path",
            "shard_dirs",
            "calibration_path",
            "output_dir",
        ],
        "merge_validation": [
            "config_path",
            "shard_dirs",
            "calibration_path",
            "selection_path",
            "output_dir",
        ],
        "build_atlas": [
            "config_path",
            "shard_dirs",
            "calibration_path",
            "output_dir",
        ],
    }
    for name, parameters in expected_parameters.items():
        function = _function(scanner_tree, name)
        observed = [argument.arg for argument in function.args.args]
        observed += [argument.arg for argument in function.args.kwonlyargs]
        assert observed == parameters
    run_shard = _function(scanner_tree, "run_shard")
    assert len(run_shard.args.defaults) + len(run_shard.args.kw_defaults) >= 1
    defaults = [*run_shard.args.defaults, *run_shard.args.kw_defaults]
    assert any(isinstance(default, ast.Constant) and default.value is None for default in defaults)


def test_shell_launcher_is_detached_and_executable():
    launcher = ROOT / "run_word_correlation.sh"
    text = launcher.read_text()
    assert launcher.stat().st_mode & 0o111
    assert "set -euo pipefail" in text
    assert 'source "${repo_dir}/modal.sh"' in text
    assert 'modal run --detach "${repo_dir}/modal_word_correlation.py"' in text
