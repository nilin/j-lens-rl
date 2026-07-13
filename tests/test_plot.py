import json

from jlens_rl.plot import read_validation


def test_read_validation_uses_fixed_greedy_metrics(tmp_path):
    path = tmp_path / "log_history.json"
    path.write_text(
        json.dumps(
            [
                {"step": 0, "validation/exact_match": 0.1},
                {"step": 1, "rewards/gsm8k_reward_trl/mean": 0.5},
                {"step": 25, "validation/exact_match": 0.2},
            ]
        )
    )
    assert [row["step"] for row in read_validation(str(path))] == [0, 25]
