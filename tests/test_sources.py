import requests

import rag.sources as sources


def test_sec_filing_parser_strips_hidden_xbrl(monkeypatch):
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
