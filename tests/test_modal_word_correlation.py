import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_PATH = ROOT / "modal_word_correlation.py"
SCANNER_PATH = ROOT / "src/jlens_rl/word_correlation.py"
CONFIG_PATH = ROOT / "configs/word_correlation_v1.json"
PREREG_PATH = ROOT / "protocol_archive/word_correlation_v1_preregistration.json"
SOURCE = MODAL_PATH.read_text()
TREE = ast.parse(SOURCE)

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
CURVE_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"

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
    amendment = json.loads(
        (ROOT / "protocol_archive" / "word_correlation_v1_amendment1.json").read_text()
    )
    closeout = ROOT / "protocol_archive" / "word_correlation_attempt1_closeout.json"
    assert amendment["attempt1_closeout_sha256"] == _sha256(closeout)
    assert amendment["original_preregistration_sha256"] == _sha256(PREREG_PATH)
    assert amendment["new_attempt"]["launcher_sha256"] == (
        "a45f70c298ba02cc86f6d7de5df84e0ebe5f2de5afe9991463c4b88e9c5ea51a"
    )
    assert amendment["new_attempt"]["scanner_sha256"] == _sha256(SCANNER_PATH)
    assert amendment["new_attempt"]["volume"] == (
        "j-lens-rl-word-correlation-v1-20260714b"
    )
    assert amendment["amendment"]["primary_selection_or_inference_changed"] is False
    throttle = json.loads(
        (ROOT / "protocol_archive" / "word_correlation_v1_amendment2.json").read_text()
    )
    assert throttle["amendment1_sha256"] == _sha256(
        ROOT / "protocol_archive" / "word_correlation_v1_amendment1.json"
    )
    assert throttle["new_attempt"]["launcher_sha256"] == (
        "b837860406d46335e60fa402943d05b1c533124f64484f1674534be5ef3c3c4d"
    )
    assert throttle["new_attempt"]["max_parallel_gpu_workers"] == 2
    packaging = json.loads(
        (ROOT / "protocol_archive" / "word_correlation_v1_amendment3.json").read_text()
    )
    assert packaging["amendment2_sha256"] == _sha256(
        ROOT / "protocol_archive" / "word_correlation_v1_amendment2.json"
    )
    assert packaging["attempt2_closeout_sha256"] == _sha256(
        ROOT / "protocol_archive" / "word_correlation_attempt2_closeout.json"
    )
    assert packaging["new_attempt"]["max_parallel_gpu_workers"] == 1
    assert packaging["new_attempt"]["launcher_sha256"] == _sha256(MODAL_PATH)
    assert packaging["new_attempt"]["volume"] == _assignment("VOLUME_NAME")
    assert prereg["launcher_script_sha256"] == _sha256(
        ROOT / "run_word_correlation.sh"
    )


def test_modal_mounts_only_the_exposed_curve_and_target_independent_lens():
    assert _assignment("MODEL_REVISION") == MODEL_REVISION
    assert _assignment("GSM8K_REVISION") == GSM8K_REVISION
    assert _assignment("LENS_SHA256") == LENS_SHA256
    assert _assignment("CURVE_MANIFEST_SHA256") == CURVE_SHA256

    mounts = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_file"
    ]
    assert len(mounts) == 2
    mount_source = [
        ast.get_source_segment(SOURCE, call.args[0]) or "" for call in mounts
    ]
    assert sum("qwen25_05b_solved_lens.pt" in text for text in mount_source) == 1
    assert sum("curve_indices.json" in text for text in mount_source) == 1
    for text in mount_source:
        assert "train_exclusions.json" not in text
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
    assert len(directory_copies) == 1
    copied_source = ast.get_source_segment(SOURCE, directory_copies[0]) or ""
    assert '".confirmatory"' in copied_source
    assert '".confirmatory/**"' in copied_source
    assert '"artifacts"' in copied_source
    assert '"artifacts/**"' in copied_source


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


def test_modal_runs_correlation_shards_on_one_l40s_worker():
    assert _assignment("NUM_SHARDS") == 8
    assert _assignment("MAX_GPU_CONTAINERS") == 1
    assert _assignment("GPU_TYPE") == "L40S"

    for function_name in ("discovery_shard", "validation_shard"):
        function = _function(TREE, function_name)
        decorators = [
            decorator
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ]
        assert len(decorators) == 1
        keywords = {keyword.arg: keyword.value for keyword in decorators[0].keywords}
        assert isinstance(keywords["gpu"], ast.Name)
        assert keywords["gpu"].id == "GPU_TYPE"
        assert isinstance(keywords["max_containers"], ast.Name)
        assert keywords["max_containers"].id == "MAX_GPU_CONTAINERS"


def test_volume_is_fresh_and_selection_is_locked_between_phases():
    assert _assignment("VOLUME_NAME") == "j-lens-rl-word-correlation-v1-20260714c"
    assert _assignment("PREREGISTRATION_RELATIVE") == (
        "protocol_archive/word_correlation_v1_preregistration.json"
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
    assert "if existing:" in claim_source
    assert "Volume is not fresh" in claim_source

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
    discovery_position = orchestrator_source.index("_mapped(discovery_shard")
    lock_position = orchestrator_source.index("_lock_selection(")
    validation_position = orchestrator_source.index("_mapped(validation_shard")
    assert discovery_position < lock_position < validation_position
    assert orchestrator_source.index('"selection_locked"') < validation_position
    assert "selection_lock_sha256" in orchestrator_source


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
