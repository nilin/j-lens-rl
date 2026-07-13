import torch

from jlens_rl.common import extract_answer, gsm8k_reward
from jlens_rl.train import completion_logps, completion_mask


def test_extract_answer_prefers_marker_and_normalizes():
    assert extract_answer("work 12 then #### $1,234") == "1234"
    assert gsm8k_reward("#### 3.0", "reasoning #### 3") == 1.0
    assert extract_answer("no numeric result") is None


def test_completion_mask_includes_first_eos_only():
    seq = torch.tensor([[10, 11, 4, 9, 4]])
    assert completion_mask(seq, 2, 4).tolist() == [[1, 0, 0]]


def test_completion_logps_alignment():
    seq = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, 4, 8)
    logits[0, 1, 3] = 10
    logits[0, 2, 4] = 10
    got = completion_logps(logits, seq, prompt_len=2)
    assert got.shape == (1, 2)
    assert torch.all(got > -0.01)

