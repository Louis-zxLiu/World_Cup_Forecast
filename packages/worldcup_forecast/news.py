from __future__ import annotations

import re
from dataclasses import dataclass, field

import feedparser

INJURY_KEYWORDS = re.compile(
    r"injur|absence|absent|出局|缺阵|伤病|受伤|停赛|伤停|suspend|doubt|doubt",
    re.IGNORECASE,
)

RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}+football&hl=en&gl=US&ceid=US:en"


@dataclass
class NewsItem:
    title: str
    url: str
    published: str = ""
    has_injury_signal: bool = False


@dataclass
class TeamNewsResult:
    team: str
    items: list[NewsItem] = field(default_factory=list)
    injury_signal: bool = False
    fetch_ok: bool = True
    error: str = ""


class RSSNewsProvider:
    def __init__(self, max_items: int = 5, timeout: int = 10) -> None:
        self.max_items = max_items
        self.timeout = timeout

    def fetch(self, team: str) -> TeamNewsResult:
        url = RSS_TEMPLATE.format(query=team.replace(" ", "+"))
        try:
            feed = feedparser.parse(url)
            items: list[NewsItem] = []
            for entry in feed.entries[: self.max_items]:
                title = entry.get("title", "")
                has_injury = bool(INJURY_KEYWORDS.search(title))
                items.append(
                    NewsItem(
                        title=title,
                        url=entry.get("link", ""),
                        published=entry.get("published", ""),
                        has_injury_signal=has_injury,
                    )
                )
            injury_signal = any(item.has_injury_signal for item in items)
            return TeamNewsResult(team=team, items=items, injury_signal=injury_signal, fetch_ok=True)
        except Exception as exc:
            return TeamNewsResult(team=team, fetch_ok=False, error=str(exc))
