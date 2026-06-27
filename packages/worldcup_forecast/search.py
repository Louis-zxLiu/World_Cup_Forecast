"""Pluggable web-search layer, decoupled from the LLM provider.

The news agent needs fresh web results, but the configured LLM provider may not
support native web search. This module talks to a standalone search API instead,
so search works regardless of which LLM is in use. Providers reachable from
mainland China (Bocha, Zhipu) are supported out of the box, plus a ``custom``
adapter for any Bocha-compatible endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from .schemas import SearchSettings


@dataclass
class SearchHit:
    title: str
    url: str = ""
    snippet: str = ""
    published: str = ""


@dataclass
class SearchOutcome:
    """Result of a search attempt. ``ok=False`` means the source was unreachable
    or misconfigured — callers must surface that honestly, not treat it as
    'nothing found'."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    ok: bool = True
    error: str = ""


class WebSearchProvider:
    def __init__(self, settings: SearchSettings) -> None:
        self.settings = settings

    def _disabled_reason(self) -> str | None:
        if not self.settings.enabled or self.settings.provider == "none":
            return "搜索功能未启用"
        if not self.settings.api_key:
            return "未配置搜索 API Key"
        return None

    def search_sync(self, query: str) -> SearchOutcome:
        """Blocking search, for use inside synchronous agent code."""
        reason = self._disabled_reason()
        if reason:
            return SearchOutcome(query=query, ok=False, error=reason)
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                if self.settings.provider == "zhipu":
                    return self._parse_zhipu(query, self._post_zhipu(client, query))
                return self._parse_bocha(query, self._post_bocha(client, query))
        except Exception as exc:  # noqa: BLE001 - report failure honestly
            return SearchOutcome(query=query, ok=False, error=f"{type(exc).__name__}: {exc}")

    async def search(self, query: str) -> SearchOutcome:
        reason = self._disabled_reason()
        if reason:
            return SearchOutcome(query=query, ok=False, error=reason)
        try:
            if self.settings.provider == "zhipu":
                return await self._search_zhipu(query)
            return await self._search_bocha(query)
        except Exception as exc:  # noqa: BLE001 - report failure honestly
            return SearchOutcome(query=query, ok=False, error=f"{type(exc).__name__}: {exc}")

    # --- request builders (shared by sync and async paths) ---
    def _bocha_request(self) -> tuple[dict, dict]:
        payload = {"query": "", "count": self.settings.max_results, "summary": True}
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        return payload, headers

    def _post_bocha(self, client: httpx.Client, query: str) -> dict:
        payload, headers = self._bocha_request()
        payload["query"] = query
        resp = client.post(self.settings.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def _post_zhipu(self, client: httpx.Client, query: str) -> dict:
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        resp = client.post(
            self.settings.base_url,
            json={"search_query": query, "search_engine": "search_std"},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    # --- response parsers ---
    def _parse_bocha(self, query: str, data: dict) -> SearchOutcome:
        pages = (
            data.get("data", {}).get("webPages", {}).get("value")
            or data.get("webPages", {}).get("value")
            or []
        )
        hits = [
            SearchHit(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("summary") or item.get("snippet", ""),
                published=item.get("datePublished", "") or item.get("dateLastCrawled", ""),
            )
            for item in pages[: self.settings.max_results]
        ]
        return SearchOutcome(query=query, hits=hits, ok=True)

    def _parse_zhipu(self, query: str, data: dict) -> SearchOutcome:
        results = data.get("search_result") or data.get("data") or []
        hits = [
            SearchHit(
                title=item.get("title", ""),
                url=item.get("link", "") or item.get("url", ""),
                snippet=item.get("content", "") or item.get("snippet", ""),
                published=item.get("publish_date", ""),
            )
            for item in results[: self.settings.max_results]
        ]
        return SearchOutcome(query=query, hits=hits, ok=True)

    async def _search_bocha(self, query: str) -> SearchOutcome:
        payload, headers = self._bocha_request()
        payload["query"] = query
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(self.settings.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return self._parse_bocha(query, data)

    async def _search_zhipu(self, query: str) -> SearchOutcome:
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(
                self.settings.base_url,
                json={"search_query": query, "search_engine": "search_std"},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        return self._parse_zhipu(query, data)
