from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.ids: set[str] = set()
        self.title_seen = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href") is not None:
            self.hrefs.append(str(values["href"]))
        if values.get("id") is not None:
            self.ids.add(str(values["id"]))
        if tag == "title":
            self.title_seen = True


def test_html_report_is_self_contained_and_links_to_existing_artifacts() -> None:
    root = Path(__file__).parents[1]
    report = root / "REPORT.html"
    text = report.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(text)

    assert parser.title_seen
    assert {"outcomes", "ranking", "evidence", "replay", "run", "limits"} <= parser.ids
    assert "<script" not in text
    assert "https://" not in text.split("<style>", 1)[1].split("</style>", 1)[0]

    local_links = [href for href in parser.hrefs if not href.startswith(("#", "https://"))]
    fragment_links = [href.removeprefix("#") for href in parser.hrefs if href.startswith("#")]
    assert local_links
    assert all((root / href).is_file() for href in local_links)
    assert all(fragment in parser.ids for fragment in fragment_links)
