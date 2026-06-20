"""Tests for prompt wildcard parsing and expansion."""

import unittest

from acestep.prompt_wildcards import (
    WildcardSyntaxError,
    expand_generation_prompt_fields,
    expand_prompt_wildcards,
    prompt_uses_wildcards,
)


class _FirstChoice:
    def choice(self, seq):
        return seq[0]


class _LastChoice:
    def choice(self, seq):
        return seq[-1]


class PromptWildcardTest(unittest.TestCase):
    def test_expands_simple_choices(self):
        self.assertEqual(
            "warm pop song",
            expand_prompt_wildcards("{warm|bright} pop song", rng=_FirstChoice()),
        )
        self.assertEqual(
            "bright pop song",
            expand_prompt_wildcards("{warm|bright} pop song", rng=_LastChoice()),
        )

    def test_expands_nested_choices(self):
        text = "{asd|dfgfd{fghfgh|dghdfgdf}|{dfghdfg|adgf|asdg}}"

        self.assertEqual("asd", expand_prompt_wildcards(text, rng=_FirstChoice()))
        self.assertEqual("asdg", expand_prompt_wildcards(text, rng=_LastChoice()))

    def test_preserves_lyric_tags_without_top_level_separator(self):
        text = "[Verse]\nI feel {alive|free}\n[Chorus]"

        self.assertEqual(
            "[Verse]\nI feel alive\n[Chorus]",
            expand_prompt_wildcards(text, rng=_FirstChoice()),
        )

    def test_literal_braces_can_wrap_nested_wildcards(self):
        self.assertEqual(
            "tag {keep a}",
            expand_prompt_wildcards("tag {keep {a|b}}", rng=_FirstChoice()),
        )

    def test_square_bracket_pipe_text_is_literal(self):
        text = "[warm|bright] pop song"

        self.assertEqual(text, expand_prompt_wildcards(text, rng=_FirstChoice()))
        self.assertFalse(prompt_uses_wildcards(text))

    def test_missing_closing_brace_reports_syntax_error(self):
        with self.assertRaisesRegex(WildcardSyntaxError, "Missing closing"):
            expand_prompt_wildcards("modern {warm|bright")

    def test_unexpected_closing_brace_reports_syntax_error(self):
        with self.assertRaisesRegex(WildcardSyntaxError, "Unexpected closing"):
            expand_prompt_wildcards("modern warm}")

    def test_field_expansion_wraps_error_with_field_label(self):
        with self.assertRaisesRegex(WildcardSyntaxError, "Lyrics"):
            expand_generation_prompt_fields(captions="ok", lyrics="{a|b")

    def test_detects_only_real_wildcards(self):
        self.assertFalse(prompt_uses_wildcards("[Verse]\nline"))
        self.assertTrue(prompt_uses_wildcards("{soft|loud}\nline"))


if __name__ == "__main__":
    unittest.main()
