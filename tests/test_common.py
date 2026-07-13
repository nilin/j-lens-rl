from jlens_rl.common import extract_answer, gsm8k_reward
from jlens_rl.train import gsm8k_reward_trl


def test_extract_answer_prefers_marker_and_normalizes():
    assert extract_answer("work 12 then #### $1,234") == "1234"
    assert gsm8k_reward("#### 3.0", "reasoning #### 3") == 1.0
    assert extract_answer("no numeric result") is None


def test_trl_gsm8k_reward():
    completions = [[{"role": "assistant", "content": "work #### 12"}]]
    assert gsm8k_reward_trl(completions, ["solution #### 12"]) == [1.0]
