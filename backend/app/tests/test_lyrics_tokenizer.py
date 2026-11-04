"""Script-aware lyric tokenisation / normalisation."""
from app.services.lyrics.tokenizer import (
    clean_word,
    is_latin_word,
    is_unspaced_script,
    script_hint,
    tokenize,
)


class TestCleanWord:
    def test_strips_punctuation_and_case(self):
        assert clean_word("Hey!") == "hey"
        assert clean_word("«hey»") == "hey"
        assert clean_word('"quoted"') == "quoted"

    def test_folds_accented_latin(self):
        # The old [^a-z0-9] filter produced "caf" / "beyonc" here.
        assert clean_word("café") == "cafe"
        assert clean_word("Beyoncé") == "beyonce"
        assert clean_word("Mötley Crüe".split()[0]) == "motley"
        assert clean_word("señor") == "senor"
        assert clean_word("ångström") == "angstrom"

    def test_preserves_non_latin_letters(self):
        # The old filter turned all of these into "" — every word then
        # looked identical and difflib aligned garbage.
        assert clean_word("日本") == "日本"
        assert clean_word("東京") == "東京"
        assert clean_word("사랑") == "사랑"
        assert clean_word("Привет") == "привет"
        assert clean_word("مرحبا") == "مرحبا"

    def test_drops_underscores(self):
        # \w would keep these; we want snake_case == snakecase is NOT
        # the goal — dropping "_" is.
        assert clean_word("snake_case") == "snakecase"

    def test_empty(self):
        assert clean_word("") == ""
        assert clean_word("...") == ""


class TestTokenize:
    def test_space_delimited_words_stay_whole(self):
        assert tokenize("one more time") == ["one", "more", "time"]

    def test_collapses_whitespace(self):
        assert tokenize("  a \n\t b  ") == ["a", "b"]

    def test_japanese_splits_per_character(self):
        # No spaces in Japanese: a whole line must not be one "word".
        assert tokenize("東京") == ["東", "京"]

    def test_korean_splits_per_syllable(self):
        assert tokenize("사랑해") == ["사", "랑", "해"]

    def test_mixed_script_run_boundaries(self):
        # "LOVEを" -> a Latin run plus per-character Japanese.
        assert tokenize("LOVEを") == ["LOVE", "を"]

    def test_chinese_line_is_character_level(self):
        assert tokenize("我爱北京天安门") == list("我爱北京天安门")

    def test_latin_punctuation_stays_attached(self):
        # Cleaning happens later; tokenise keeps it for display.
        assert tokenize("hello, world") == ["hello,", "world"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestScriptHelpers:
    def test_unspaced_script_detection(self):
        assert is_unspaced_script("あ")
        assert is_unspaced_script("中")
        assert is_unspaced_script("한")
        assert is_unspaced_script("ก")
        assert not is_unspaced_script("a")
        assert not is_unspaced_script("Ж")  # Cyrillic has spaces

    def test_is_latin_word(self):
        assert is_latin_word("hello")
        assert is_latin_word("cafe")
        assert not is_latin_word("")
        assert not is_latin_word("東京")
        assert not is_latin_word("привет")
        assert not is_latin_word("hello2")

    def test_script_hint(self):
        assert script_hint("東京の夜") == "ja"
        assert script_hint("我爱北京") == "zh"
        assert script_hint("사랑해요") == "ko"
        assert script_hint("สวัสดี") == "th"
        # Latin text gives no useful Whisper language signal.
        assert script_hint("one more time") is None
        assert script_hint("") is None
