"""Tests for the Indian Kanoon fetcher's pure logic — no network, no token."""

from nyaya.pipelines.fetch_indiankanoon import (
    html_to_text,
    key_tokens,
    pick_best_judgment,
    search_query,
)

TOKENS = key_tokens("Kesavananda Bharati v. State of Kerala")  # ['kesavananda','bharati']


def test_key_tokens_from_petitioner_side():
    assert "kesavananda" in TOKENS
    assert "kerala" not in TOKENS  # respondent side excluded


def test_search_query_drops_the_v():
    assert search_query("Kesavananda Bharati v. State of Kerala") == \
        "Kesavananda Bharati State of Kerala"


def test_matching_ignores_bold_highlight_tags():
    # IK wraps matched words in <b>…</b>; the anchor must still match
    docs = [{"tid": 9, "docsource": "Supreme Court of India",
             "title": "<b>Kesavananda</b> <b>Bharati</b> vs State Of Kerala",
             "numcitedby": 999}]
    assert pick_best_judgment(docs, TOKENS)["tid"] == 9


def test_picks_title_matching_supreme_court_result_not_just_most_cited():
    docs = [
        # a recent case that CITES Kesavananda — most cited but wrong case
        {"tid": 1, "docsource": "Supreme Court of India",
         "title": "Brajesh Singh vs Sunil Arora", "numcitedby": 9999},
        # the actual judgment — title carries the name
        {"tid": 2, "docsource": "Supreme Court of India",
         "title": "Kesavananda Bharati vs State Of Kerala", "numcitedby": 4200},
    ]
    assert pick_best_judgment(docs, TOKENS)["tid"] == 2


def test_returns_none_when_no_title_match():
    docs = [
        {"tid": 1, "docsource": "Supreme Court of India",
         "title": "Some Other Case vs Union", "numcitedby": 500},
    ]
    assert pick_best_judgment(docs, TOKENS) is None  # skip, don't save wrong case


def test_returns_none_when_no_supreme_court_result():
    docs = [{"tid": 1, "docsource": "Kerala High Court",
             "title": "Kesavananda Bharati vs State", "numcitedby": 200}]
    assert pick_best_judgment(docs, TOKENS) is None


def test_most_cited_among_title_matches_wins():
    docs = [
        {"tid": 1, "docsource": "Supreme Court of India",
         "title": "Kesavananda Bharati vs State (review)", "numcitedby": 5},
        {"tid": 2, "docsource": "Supreme Court of India",
         "title": "Kesavananda Bharati vs State Of Kerala", "numcitedby": 900},
    ]
    assert pick_best_judgment(docs, TOKENS)["tid"] == 2


def test_empty_docs():
    assert pick_best_judgment([], TOKENS) is None


def test_html_to_text_preserves_paragraphs():
    html = "<p>First paragraph.</p><p>Second &amp; final.</p>"
    out = html_to_text(html)
    assert "First paragraph." in out
    assert "Second & final." in out  # entity unescaped
    assert "\n\n" in out  # paragraph break preserved


def test_html_to_text_strips_tags_and_collapses_space():
    html = "<div class='x'>Held   that\t the appeal <b>fails</b>.</div>"
    out = html_to_text(html)
    assert "<" not in out and ">" not in out
    assert "Held that the appeal fails." in out
