from __future__ import annotations

import unittest

from services.text_chunking_service import truncate_text_to_max_tokens


class TruncateTextToMaxTokensTests(unittest.TestCase):
    def test_zero_or_negative_means_no_truncate(self) -> None:
        text = "hello " * 100
        self.assertIs(truncate_text_to_max_tokens(text, 0, "o200k_base"), text)
        self.assertIs(truncate_text_to_max_tokens(text, -1, "o200k_base"), text)

    def test_none_means_no_truncate(self) -> None:
        text = "abc"
        self.assertIs(truncate_text_to_max_tokens(text, None, "o200k_base"), text)

    def test_shortens_to_token_budget(self) -> None:
        text = "word " * 400
        out = truncate_text_to_max_tokens(text, 12, "o200k_base")
        self.assertLess(len(out), len(text))
        self.assertTrue(out.strip())


if __name__ == "__main__":
    unittest.main()
