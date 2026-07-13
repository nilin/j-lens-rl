from jlens_rl.reward import single_token_ids


class FakeTokenizer:
    table = {"solved": [7], " solved": [8], "Solved": [9], "two words": [1, 2]}

    def encode(self, text, add_special_tokens=False):
        return self.table.get(text, [1, 2])


def test_single_token_variants():
    assert single_token_ids(FakeTokenizer(), ["solved"]) == [7, 8, 9]

