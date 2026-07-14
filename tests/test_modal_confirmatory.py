import importlib.util
from pathlib import Path


def _load_modal_confirmatory():
    path = Path(__file__).resolve().parents[1] / "modal_experiments.py"
    spec = importlib.util.spec_from_file_location("modal_confirmatory_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_modal_runner_freezes_eight_gpus_and_one_sealed_batch() -> None:
    runner = _load_modal_confirmatory()
    assert runner.VOLUME_NAME == "j-lens-rl-confirmatory-v4-20260714a"
    assert runner.SEEDS == tuple(range(159, 167))
    assert runner.MAX_GPU_CONTAINERS == 8
    assert runner.SEALED_LABELS == (
        "base",
        *(f"jlens_seed{seed}" for seed in range(159, 167)),
        *(f"signflip_seed{seed}" for seed in range(159, 167)),
    )
    assert len(runner.SEALED_LABELS) == 17

    source = (Path(__file__).resolve().parents[1] / "modal_experiments.py").read_text()
    assert '_protocol("verify-curve")' in source
    assert "_mapped_results(evaluate_label, SEALED_LABELS)" in source
    assert "semantic_evals" not in source
    assert "control_evals" not in source
    assert 'output = evidence_dir / "sealed_comparison.json"' in source
