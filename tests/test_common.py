from jlens_rl.common import binomial_ci95, extract_answer, gsm8k_reward
from jlens_rl.train import gsm8k_reward_trl, prepare_prompt


def test_extract_answer_prefers_marker_and_normalizes():
    assert extract_answer("work 12 then #### $1,234") == "1234"
    assert gsm8k_reward("#### 3.0", "reasoning #### 3") == 1.0
    assert extract_answer("no numeric result") is None


def test_trl_gsm8k_reward():
    completions = [[{"role": "assistant", "content": "work #### 12"}]]
    assert gsm8k_reward_trl(completions, ["solution #### 12"]) == [1.0]


def test_binomial_interval_handles_boundaries():
    assert binomial_ci95(0, 200)[1] > 0
    assert binomial_ci95(200, 200)[0] < 1


def test_jlens_prompt_preparation_never_requires_a_gold_answer():
    prepared = prepare_prompt({"question": "What is 1 + 1?"})
    assert set(prepared) == {"prompt"}
    assert prepared["prompt"][-1]["content"] == "What is 1 + 1?"
