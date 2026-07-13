import torch

from jlens_rl.reward import single_token_ids, target_log_probs


class FakeTokenizer:
    table = {"solved": [7], " solved": [8], "Solved": [9], "two words": [1, 2]}

    def encode(self, text, add_special_tokens=False):
        return self.table.get(text, [1, 2])


def test_single_token_variants():
    assert single_token_ids(FakeTokenizer(), ["solved"]) == [7, 8, 9]


def test_target_log_probs_matches_full_softmax_across_chunks():
    head = torch.nn.Linear(3, 7, bias=False)
    hidden = torch.tensor([[0.2, -0.1, 0.4], [1.0, 0.5, -0.5]])
    expected = torch.logsumexp(
        torch.log_softmax(head(hidden), dim=-1)[:, [1, 5]], dim=-1
    )
    actual = target_log_probs(hidden, head, [1, 5], chunk_size=2)
    torch.testing.assert_close(actual, expected)
