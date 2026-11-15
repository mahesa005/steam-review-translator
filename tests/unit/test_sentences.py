from steam_review_translator.aspect.sentences import (
    blank_nlp,
    chunk_runon,
    iter_sentences,
    normalize_ws,
    split_reviews,
    split_segments,
    split_sentences,
    unescape_html,
)


def test_unescape_html() -> None:
    assert unescape_html("Tips &amp; tricks &gt; guides") == "Tips & tricks > guides"
    assert unescape_html("HITMAN&amp;amp;trade;") == "HITMAN™"  # double-encoded
    assert unescape_html("no entities here") == "no entities here"


def test_split_segments_on_multispace() -> None:
    text = "Great game.  Would buy again\n\nbut the price is high"
    assert split_segments(text) == ["Great game.", "Would buy again", "but the price is high"]
    assert split_segments(text, split_on_multispace=False) == [normalize_ws(text)]


def test_chunk_runon() -> None:
    words = ("i love this game and the music is great but the servers lag a lot " * 4).split()
    spans = chunk_runon(words, max_tokens=20)
    assert spans[0][0] == 0 and spans[-1][1] == len(words)
    assert all(e - s <= 20 for s, e in spans)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))  # contiguous
    assert chunk_runon(["short", "one"], max_tokens=20) == [(0, 2)]


def test_chunk_runon_hard_cut_without_breaks() -> None:
    assert chunk_runon(["w"] * 45, max_tokens=20) == [(0, 20), (20, 40), (40, 45)]


def test_iter_sentences_keys_and_order() -> None:
    items = [("Fun combat. Bad lag!  Story is ok", "r1"), ("no punctuation at all here", "r2")]
    out = list(iter_sentences(blank_nlp(), items))
    assert [s.text for s in out if s.key == "r1"] == ["Fun combat.", "Bad lag!", "Story is ok"]
    assert [s.sent_idx for s in out if s.key == "r1"] == [0, 1, 2]
    assert [s.text for s in out if s.key == "r2"] == ["no punctuation at all here"]
    assert out[0].n_words == 2  # punctuation is not a word


def test_split_sentences_end_to_end() -> None:
    assert split_sentences("Fun game! Bad ending.\nWould buy again?  Yes &amp; no") == [
        "Fun game!",
        "Bad ending.",
        "Would buy again?",
        "Yes & no",
    ]
    assert split_sentences("   ") == []


def test_split_reviews_one_list_per_review() -> None:
    reviews = ["Great story. Awful lag.", "", "Buy it"]
    assert split_reviews(reviews) == [["Great story.", "Awful lag."], [], ["Buy it"]]


def test_long_unpunctuated_review_is_chunked() -> None:
    text = " ".join(["the game is fun and the music is good but it lags sometimes"] * 6)
    sents = split_sentences(text)
    assert len(sents) > 1
    assert all(len(s.split()) <= 50 for s in sents)
