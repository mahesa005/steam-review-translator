import pytest

from steam_review_translator.aspect.lexicon import Lexicon, LexiconMatcher, load_lexicon, tokenize
from steam_review_translator.core.schemas import Aspect


@pytest.fixture(scope="module")
def matcher() -> LexiconMatcher:
    return LexiconMatcher(load_lexicon())


def test_seed_lexicon_loads() -> None:
    lex = load_lexicon()
    assert set(lex.terms) == set(Aspect)
    assert all(lex.terms[a] for a in Aspect if a is not Aspect.OTHER)
    assert lex.ambiguous == {"hard", "easy", "smooth", "free", "beautiful"}


def test_tokenize_keeps_compound_tokens() -> None:
    assert tokenize("Co-op is 10/10, don't skip!") == ["co-op", "is", "10/10", "don't", "skip"]


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The COMBAT is great", [Aspect.GAMEPLAY]),
        ("I love the art style.", [Aspect.GRAPHICS]),
        ("Easily a 10/10", [Aspect.OVERALL]),
        ("Co-op with friends", [Aspect.MULTIPLAYER]),
        ("Great story, awful lag", [Aspect.STORY, Aspect.PERFORMANCE]),
        ("Sound effects and sound design", [Aspect.AUDIO]),  # deduplicated
    ],
)
def test_matches(matcher: LexiconMatcher, sentence: str, expected: list[Aspect]) -> None:
    assert matcher.match(sentence) == expected


@pytest.mark.parametrize(
    "sentence",
    [
        "I had to debug my save file",  # "bug" inside "debug"
        "A combatant appears",  # "combat" inside "combatant"
        "Modern art at its finest",  # "art" alone is not "art style"
        "The storyteller is fine",
    ],
)
def test_whole_token_only(matcher: LexiconMatcher, sentence: str) -> None:
    assert matcher.match(sentence) == []


def test_ambiguous_words_ignored(matcher: LexiconMatcher) -> None:
    assert matcher.match("It is hard, easy, smooth, free and beautiful") == []


def test_ambiguous_overrides_explicit_listing() -> None:
    lex = Lexicon.from_dict(
        {"aspects": {"difficulty": ["hard", "learning curve"]}, "ambiguous": ["hard"]}
    )
    m = LexiconMatcher(lex)
    assert m.match("It is hard") == []
    assert m.match("Steep learning curve, very hard") == [Aspect.DIFFICULTY]


def test_match_terms_reports_matched_text(matcher: LexiconMatcher) -> None:
    assert matcher.match_terms("Great voice acting") == [(Aspect.AUDIO, "voice acting")]


def test_unknown_aspect_in_lexicon_rejected() -> None:
    with pytest.raises(ValueError, match="unknown aspect"):
        Lexicon.from_dict({"aspects": {"vibes": ["chill"]}})
