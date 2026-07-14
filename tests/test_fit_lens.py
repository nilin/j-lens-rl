from jlens_rl.common import GSM8K_REVISION, QWEN_MODEL_REVISION, WIKITEXT_REVISION
import json

from jlens_rl.fit_lens import parse_args, write_calibration


def test_fit_lens_defaults_pin_model_and_dataset_revisions(monkeypatch):
    monkeypatch.setattr("sys.argv", ["fit-jlens"])
    args = parse_args()
    assert args.model_revision == QWEN_MODEL_REVISION
    assert args.wikitext_revision == WIKITEXT_REVISION
    assert args.gsm8k_revision == GSM8K_REVISION
    assert args.checkpoint_path is None
    assert args.resume_checkpoint is False


def test_fit_lens_creates_calibration_output_parent(tmp_path):
    output = tmp_path / "new" / "nested" / "calibration.json"
    write_calibration(output, {"mean": 1.25, "std": 0.5})
    assert json.loads(output.read_text()) == {"mean": 1.25, "std": 0.5}
