"""Tests for the Task 2 EDA helpers (experiments/task2_aspect/eda/edalib.py).

Pure-Python parts always run; spaCy / fastText parts skip when the task2 extra or the
models are not installed (CI installs only .[dev]).
"""

import sys
from pathlib import Path
from typing import Any

import pytest

EDA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "task2_aspect" / "eda"
sys.path.insert(0, str(EDA_DIR))

import edalib as E  # noqa: E402

P = E.FilterParams()


def flags(text: str) -> dict[str, bool]:
    return E.text_flags(text, P)


def test_normal_review_is_clean() -> None:
    f = flags("The combat is great but the story drags in the second act.")
    assert not any(f.values())


@pytest.mark.parametrize("text", ["10/10", "good game", "Early Access Review", "   "])
def test_short_reviews(text: str) -> None:
    assert flags(text)["short"]


def test_empty() -> None:
    assert flags("  \n ")["empty"]


def test_repeated_chars() -> None:
    assert flags("This game is soooooooooooo good, I love it")["repeated_chars"]
    assert not flags("This game is sooo good, I love it a lot")["repeated_chars"]


def test_low_alnum_and_ascii_art() -> None:
    art = "░░░░▄▄▄▄▀▀▀▀▀▀▀▀▄▄▄▄▄▄░░░░ ░░░░█░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░▀▀▄░░░░ ░░░█░░░▒▒▒▒▒▒░"
    f = flags(art)
    assert f["low_alnum"] and f["ascii_art"]


def test_steam_profanity_hearts_are_not_art() -> None:
    f = flags("This ♥♥♥♥ing game crashes every ♥♥♥♥ing time I load a save, fix it devs")
    assert not f["ascii_art"] and not f["low_alnum"]


def test_self_repeat() -> None:
    assert flags("buy this game now " * 30)["self_repeat"]
    long_review = (
        "I played this for forty hours. The early levels teach the mechanics well, the "
        "middle chapters add new enemy types, and the final boss is a real test of skill. "
        "Music is forgettable, but performance was stable on my old laptop the whole time."
    )
    assert not flags(long_review)["self_repeat"]


@pytest.fixture(scope="module")
def nlp() -> Any:
    spacy = pytest.importorskip("spacy")
    try:
        return E.load_nlp("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    return spacy  # unreachable


def test_lemma_lexicon_matcher(nlp: Any) -> None:
    m = E.LemmaLexiconMatcher(nlp, "configs/task2/lexicon.yaml")

    def aspects(text: str) -> set[str]:
        doc = nlp(text)
        return {a for a, _ in m.match_terms(text, " ".join(t.lemma_.lower() for t in doc))}

    assert "performance" in aspects("It crashed twice and the crashes keep coming")
    assert "story" in aspects("The characters are well written")
    assert aspects("It is hard and beautiful") == set()  # ambiguous words never match alone


def test_language_detector() -> None:
    pytest.importorskip("fasttext")
    if not E.LID_MODEL_PATH().is_file():
        pytest.skip("fastText lid.176 model not downloaded (run the EDA once)")
    det = E.LangDetector(E.load_cfg())
    langs, probs = det.predict(
        ["This game is really fun with friends", "Este juego es muy divertido con amigos", ""]
    )
    assert langs[:2] == ["en", "es"] and langs[2] == "" and probs[0] > 0.5
