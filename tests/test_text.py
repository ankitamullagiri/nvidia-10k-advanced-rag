import pytest

from rag.text import chunk_pages, chunk_text, split_sentences, tokenize


def test_split_sentences_keeps_decimals_together():
    assert split_sentences("Revenue was $130.5 billion. Data Center grew 142%.") == [
        "Revenue was $130.5 billion.",
        "Data Center grew 142%.",
    ]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))  # 5 words each
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 for c in chunks)
    # each chunk starts with the last 10 words of the previous one
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[:10] == prev.split()[-10:]


def test_sentences_are_not_split_when_they_fit():
    chunks = chunk_text("One two three. Four five six.", chunk_size=4, overlap=0)
    assert chunks == ["One two three.", "Four five six."]


def test_oversized_sentence_is_split_on_words():
    chunks = chunk_text(" ".join(["word"] * 25), chunk_size=10, overlap=0)
    assert [len(c.split()) for c in chunks] == [10, 10, 5]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=size, overlap=overlap)


def test_chunk_pages_keeps_page_numbers_and_ids():
    chunks = chunk_pages(["Page one text.", "", "Page three text."])
    assert [(c.id, c.page) for c in chunks] == [(0, 1), (1, 3)]


def test_tokenize_keeps_numbers():
    assert tokenize("FY2025 revenue: $130.5B, up 114%") == ["fy2025", "revenue", "130.5b", "up", "114"]


def test_sec_filing_parser_strips_hidden_xbrl(monkeypatch):
    import requests

    import rag.sources as sources

    body = b'<div style="display:none"><ix:header>HIDDEN</ix:header></div><p>Revenue was $130.5 billion.</p>'
    body += b"<p>word</p>" * 700

    class FakeResponse:
        content = body

        def raise_for_status(self):
            pass

    calls = {}

    def fake_get(url, headers, timeout):
        calls["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    pages = sources.load_sec_filing()
    assert pages[0].startswith("Revenue was $130.5 billion.")
    assert all("HIDDEN" not in p for p in pages)
    assert len(pages) == 2  # 705 words → sections of 600
    assert "User-Agent" in calls["headers"]
