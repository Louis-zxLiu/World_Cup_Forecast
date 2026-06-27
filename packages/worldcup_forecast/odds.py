from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .config import get_config
from .schemas import OddsRecord, ScrapeResult

TEAM_ALIASES = {
    "巴西": "Brazil",
    "德国": "Germany",
    "阿根廷": "Argentina",
    "法国": "France",
    "英格兰": "England",
    "西班牙": "Spain",
    "葡萄牙": "Portugal",
    "荷兰": "Netherlands",
    "意大利": "Italy",
    "比利时": "Belgium",
    "克罗地亚": "Croatia",
    "丹麦": "Denmark",
    "瑞士": "Switzerland",
    "波兰": "Poland",
    "乌拉圭": "Uruguay",
    "哥伦比亚": "Colombia",
    "厄瓜多尔": "Ecuador",
    "墨西哥": "Mexico",
    "美国": "United States",
    "加拿大": "Canada",
    "摩洛哥": "Morocco",
    "塞内加尔": "Senegal",
    "尼日利亚": "Nigeria",
    "加纳": "Ghana",
    "喀麦隆": "Cameroon",
    "埃及": "Egypt",
    "日本": "Japan",
    "韩国": "South Korea",
    "伊朗": "Iran",
    "澳大利亚": "Australia",
    "沙特阿拉伯": "Saudi Arabia",
    "沙特": "Saudi Arabia",
    "卡塔尔": "Qatar",
    "中国": "China",
    "伊拉克": "Iraq",
    "挪威": "Norway",
    "佛得角": "Cape Verde",
    "巴拿马": "Panama",
    "塞尔维亚": "Serbia",
    "突尼斯": "Tunisia",
    "哥斯达黎加": "Costa Rica",
    "土耳其": "Turkey",
    "威尔士": "Wales",
    "苏格兰": "Scotland",
    "北爱尔兰": "Northern Ireland",
    "爱尔兰": "Republic of Ireland",
    "瑞典": "Sweden",
    "奥地利": "Austria",
    "捷克": "Czech Republic",
    "希腊": "Greece",
    "俄罗斯": "Russia",
    "乌克兰": "Ukraine",
    "秘鲁": "Peru",
    "智利": "Chile",
    "巴拉圭": "Paraguay",
    "玻利维亚": "Bolivia",
    "委内瑞拉": "Venezuela",
    "阿尔及利亚": "Algeria",
    "南非": "South Africa",
    "科特迪瓦": "Ivory Coast",
    "新西兰": "New Zealand",
    "洪都拉斯": "Honduras",
    "牙买加": "Jamaica",
    "阿联酋": "United Arab Emirates",
    "约旦": "Jordan",
    "乌兹别克斯坦": "Uzbekistan",
    "匈牙利": "Hungary",
    "罗马尼亚": "Romania",
    "斯洛伐克": "Slovakia",
    "斯洛文尼亚": "Slovenia",
    "冰岛": "Iceland",
    "芬兰": "Finland",
    "加蓬": "Gabon",
    "马里": "Mali",
    "布基纳法索": "Burkina Faso",
}

BLOCKED_TEAM_TOKENS = {
    "单关",
    "亚",
    "欧",
    "荐",
    "析",
    "未开售",
    "胜平负",
    "让球胜平负",
}


def canonical_team(name: str) -> str:
    clean = re.sub(r"\s+", "", name or "")
    clean = re.sub(r"^\[\d+\]", "", clean)
    return TEAM_ALIASES.get(clean, name.strip())


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value.strip())
    for fmt in ("%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%m-%d %H:%M"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt == "%m-%d %H:%M":
                parsed = parsed.replace(year=datetime.utcnow().year)
            return parsed
        except ValueError:
            continue
    return None


def _valid_team_name(name: str) -> bool:
    clean = re.sub(r"\s+", "", name or "")
    if len(clean) < 2 or len(clean) > 40:
        return False
    if clean in BLOCKED_TEAM_TOKENS:
        return False
    if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", clean):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", clean))


class FiveHundredLotteryProvider:
    """500彩票网竞彩足球赔率抓取器。

    500 网真实比赛数据主要在 ``tr.bet-tb-tr`` 的 data-* 属性和
    ``p.betbtn[data-type=nspf/spf]`` 上，不能用普通表格文本粗暴拆列，否则会把
    “单关、亚、欧、荐”等页面按钮误识别成球队或赔率。
    """

    provider_name = "500.com"

    def __init__(self, source_url: str | None = None, snapshot_dir: Path | None = None) -> None:
        cfg = get_config()
        self.source_url = source_url or cfg.china_lottery_odds_url
        self.snapshot_dir = snapshot_dir or cfg.odds_snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    async def scrape(self, use_playwright: bool = False) -> ScrapeResult:
        scraped_at = datetime.utcnow()
        warnings: list[str] = []
        html = ""
        try:
            html = await self._fetch_http()
        except Exception as exc:
            warnings.append(f"http fetch failed: {exc}")

        if use_playwright and not html:
            try:
                html = await self._fetch_playwright()
            except Exception as exc:
                warnings.append(f"playwright fetch failed: {exc}")

        if not html:
            sample = Path("data/samples/china_lottery_sample.html")
            html = sample.read_text(encoding="utf-8")
            warnings.append("live scrape unavailable; parsed bundled sample snapshot")

        snapshot_path = self._save_snapshot(html, scraped_at)
        records = self.parse(html, source_url=self.source_url, scraped_at=scraped_at)
        return ScrapeResult(
            provider=self.provider_name,
            source_url=self.source_url,
            scraped_at=scraped_at,
            record_count=len(records),
            snapshot_path=str(snapshot_path),
            warnings=warnings,
            records=records,
        )

    async def _fetch_http(self) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
            response = await client.get(self.source_url)
            response.raise_for_status()
            encoding = response.encoding or response.charset_encoding or "gb18030"
            response.encoding = encoding
            return response.text

    async def _fetch_playwright(self) -> str:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(self.source_url, wait_until="networkidle", timeout=30000)
            html = await page.content()
            await browser.close()
            return html

    def _save_snapshot(self, html: str, scraped_at: datetime) -> Path:
        path = self.snapshot_dir / f"500_jczq_{scraped_at.strftime('%Y%m%d_%H%M%S')}.html"
        path.write_text(html, encoding="utf-8")
        return path

    def parse(
        self,
        html: str,
        source_url: str = "https://trade.500.com/jczq/",
        scraped_at: datetime | None = None,
    ) -> list[OddsRecord]:
        scraped_at = scraped_at or datetime.utcnow()
        records = self._parse_500_rows(html, source_url, scraped_at)
        if records:
            return records

        records = self._parse_simple_tables(html, source_url, scraped_at)
        if records:
            return records

        return self._parse_embedded_text(html, source_url, scraped_at)

    def _parse_500_rows(
        self, html: str, source_url: str, scraped_at: datetime
    ) -> list[OddsRecord]:
        soup = BeautifulSoup(html, "html.parser")
        records: list[OddsRecord] = []
        for row in soup.select("tr.bet-tb-tr"):
            match_id = row.get("data-matchnum") or ""
            home = row.get("data-homesxname") or ""
            away = row.get("data-awaysxname") or ""
            if not (match_id and _valid_team_name(home) and _valid_team_name(away)):
                continue

            kickoff = parse_datetime(
                f"{row.get('data-matchdate', '')} {row.get('data-matchtime', '')}".strip()
            )
            handicap = (row.get("data-rangqiu") or "").strip() or None

            for data_type, play_type, play_handicap in (
                ("nspf", "胜平负", None),
                ("spf", "让球胜平负", handicap),
            ):
                odds = self._odds_from_buttons(row, data_type)
                if not odds:
                    continue
                records.append(
                    OddsRecord(
                        match_id=match_id,
                        kickoff_time=kickoff,
                        home_team=canonical_team(home),
                        away_team=canonical_team(away),
                        play_type=play_type,
                        handicap=play_handicap,
                        win_odds=odds["win"],
                        draw_odds=odds["draw"],
                        lose_odds=odds["lose"],
                        source=self.provider_name,
                        source_url=source_url,
                        scraped_at=scraped_at,
                        raw={
                            "fixture_id": row.get("data-fixtureid"),
                            "league": row.get("data-simpleleague"),
                            "data_type": data_type,
                            "home_raw": home,
                            "away_raw": away,
                        },
                    )
                )
        return records

    def _odds_from_buttons(self, row: Any, data_type: str) -> dict[str, float] | None:
        values: dict[str, float] = {}
        for button in row.select(f'p.betbtn[data-type="{data_type}"]'):
            outcome = button.get("data-value")
            odd = parse_float(button.get("data-sp") or button.get_text(" ", strip=True))
            if odd is None:
                continue
            if outcome == "3":
                values["win"] = odd
            elif outcome == "1":
                values["draw"] = odd
            elif outcome == "0":
                values["lose"] = odd
        return values if {"win", "draw", "lose"} <= values.keys() else None

    def _parse_simple_tables(
        self, html: str, source_url: str, scraped_at: datetime
    ) -> list[OddsRecord]:
        soup = BeautifulSoup(html, "html.parser")
        records: list[OddsRecord] = []
        for row in soup.select("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if len(cells) < 9:
                continue
            if cells[0] in {"赛事编号", "编号"}:
                continue
            if not re.match(r"周[一二三四五六日]\d{3}", cells[0]):
                continue

            home, away = cells[2], cells[3]
            if not (_valid_team_name(home) and _valid_team_name(away)):
                continue

            win, draw, lose = parse_float(cells[6]), parse_float(cells[7]), parse_float(cells[8])
            if win is None or draw is None or lose is None:
                continue

            records.append(
                OddsRecord(
                    match_id=cells[0],
                    kickoff_time=parse_datetime(cells[1]),
                    home_team=canonical_team(home),
                    away_team=canonical_team(away),
                    play_type=cells[4] or "胜平负",
                    handicap=cells[5] or None,
                    win_odds=win,
                    draw_odds=draw,
                    lose_odds=lose,
                    source=self.provider_name,
                    source_url=source_url,
                    scraped_at=scraped_at,
                    raw={"cells": cells},
                )
            )
        return records

    def _parse_embedded_text(
        self, html: str, source_url: str, scraped_at: datetime
    ) -> list[OddsRecord]:
        records: list[OddsRecord] = []
        for payload in self._candidate_json_objects(html):
            records.extend(self._records_from_json(payload, source_url, scraped_at))
        return records

    def _candidate_json_objects(self, html: str) -> list[Any]:
        candidates: list[Any] = []
        for script in re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.S | re.I):
            for fragment in re.findall(r"(\{[^{}]*(?:sp|odds|match|home)[^{}]*\})", script, re.I):
                try:
                    candidates.append(json.loads(fragment))
                except json.JSONDecodeError:
                    continue
        return candidates

    def _records_from_json(
        self, payload: Any, source_url: str, scraped_at: datetime
    ) -> list[OddsRecord]:
        records: list[OddsRecord] = []
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            home = item.get("home") or item.get("hteam") or item.get("home_team")
            away = item.get("away") or item.get("ateam") or item.get("away_team")
            win = item.get("win") or item.get("sp3") or item.get("odds_win")
            draw = item.get("draw") or item.get("sp1") or item.get("odds_draw")
            lose = item.get("lose") or item.get("sp0") or item.get("odds_lose")
            if not (home and away and win and draw and lose):
                continue
            if not (_valid_team_name(str(home)) and _valid_team_name(str(away))):
                continue
            records.append(
                OddsRecord(
                    match_id=str(item.get("match_id") or item.get("id") or item.get("num") or ""),
                    kickoff_time=parse_datetime(str(item.get("kickoff") or item.get("time") or "")),
                    home_team=canonical_team(str(home)),
                    away_team=canonical_team(str(away)),
                    play_type=str(item.get("play_type") or "胜平负"),
                    handicap=str(item.get("handicap")) if item.get("handicap") is not None else None,
                    win_odds=parse_float(str(win)),
                    draw_odds=parse_float(str(draw)),
                    lose_odds=parse_float(str(lose)),
                    source=self.provider_name,
                    source_url=source_url,
                    scraped_at=scraped_at,
                    raw=item,
                )
            )
        return records


def implied_probabilities(record: OddsRecord) -> dict[str, float | None]:
    odds = {
        "home_win": record.win_odds,
        "draw": record.draw_odds,
        "away_win": record.lose_odds,
    }
    inverse = {key: (1 / value if value and value > 0 else None) for key, value in odds.items()}
    overround = sum(value for value in inverse.values() if value is not None)
    if overround <= 0:
        return {key: None for key in odds}
    return {
        key: (value / overround if value is not None else None) for key, value in inverse.items()
    }
