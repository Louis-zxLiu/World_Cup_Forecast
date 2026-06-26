from __future__ import annotations

import json
import re
from typing import Any

from .llm import OpenAICompatibleClient
from .odds import BLOCKED_TEAM_TOKENS, canonical_team
from .schemas import LLMSettings, OddsRecord


def _json_from_text(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if cleaned.startswith("{"):
        data = json.loads(cleaned)
        return data.get("records", data)
    return json.loads(cleaned)


def _valid_record(record: OddsRecord) -> bool:
    if not record.match_id or not record.home_team or not record.away_team:
        return False
    if record.home_team == record.away_team:
        return False
    if record.home_team in BLOCKED_TEAM_TOKENS or record.away_team in BLOCKED_TEAM_TOKENS:
        return False
    odds = [record.win_odds, record.draw_odds, record.lose_odds]
    return all(value is not None and 1.01 <= value <= 1000 for value in odds)


def rule_clean_odds(records: list[OddsRecord]) -> tuple[list[OddsRecord], list[str]]:
    cleaned: list[OddsRecord] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for record in records:
        normalized = record.model_copy(
            update={
                "home_team": canonical_team(record.home_team),
                "away_team": canonical_team(record.away_team),
            },
            deep=True,
        )
        if not _valid_record(normalized):
            warnings.append(
                f"规则清洗剔除异常赔率行: {record.match_id} {record.home_team} vs {record.away_team}"
            )
            continue
        key = (normalized.match_id, normalized.play_type, normalized.handicap or "")
        if key in seen:
            warnings.append(f"规则清洗剔除重复赔率行: {'/'.join(key)}")
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned, warnings


class OddsCleaner:
    batch_size = 80

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings()

    async def clean(self, records: list[OddsRecord]) -> tuple[list[OddsRecord], list[str]]:
        base_records, warnings = rule_clean_odds(records)
        if not base_records:
            return base_records, warnings
        if not self.settings.enabled or not self.settings.api_key:
            warnings.append("LLM 未启用，已使用规则清洗。")
            return _mark_cleaning(base_records, "rule"), warnings

        try:
            llm_records: list[OddsRecord] = []
            for start in range(0, len(base_records), self.batch_size):
                batch = base_records[start : start + self.batch_size]
                llm_records.extend(await self._llm_clean_batch(batch))
            llm_records, rule_warnings = rule_clean_odds(llm_records)
            warnings.extend(rule_warnings)
            warnings.append(f"LLM 清洗完成: 输入 {len(base_records)} 条，输出 {len(llm_records)} 条。")
            return _mark_cleaning(llm_records, "llm"), warnings
        except Exception as exc:
            warnings.append(f"LLM 清洗失败，已使用规则清洗: {exc}")
            return _mark_cleaning(base_records, "rule_fallback"), warnings

    async def _llm_clean_batch(self, records: list[OddsRecord]) -> list[OddsRecord]:
        payload = [
            {
                "index": index,
                "match_id": record.match_id,
                "home_team": record.home_team,
                "away_team": record.away_team,
                "play_type": record.play_type,
                "handicap": record.handicap,
                "win_odds": record.win_odds,
                "draw_odds": record.draw_odds,
                "lose_odds": record.lose_odds,
            }
            for index, record in enumerate(records)
        ]
        system_prompt = (
            "你是足球竞彩赔率数据清洗器。只返回 JSON 数组，不要输出解释文字。"
            "任务：判断每条记录是否是真实比赛赔率，修正球队名，剔除页面按钮/广告/脏行。"
            "不要改动赔率数值、玩法、盘口和编号。"
        )
        user_prompt = (
            "返回格式："
            "[{\"index\":0,\"keep\":true,\"home_team\":\"Brazil\","
            "\"away_team\":\"Germany\",\"reason\":\"ok\"}]。\n"
            "如果 home_team/away_team 包含 单关、亚、欧、荐、析、未开售、胜平负 等页面标签，keep=false。\n"
            "球队名尽量转成英文 canonical 名；不确定时保留原名。\n"
            f"待清洗记录：{json.dumps(payload, ensure_ascii=False)}"
        )
        text = await OpenAICompatibleClient(self.settings).complete(system_prompt, user_prompt)
        decisions = _json_from_text(text)
        if not isinstance(decisions, list):
            raise ValueError("LLM did not return a JSON array")

        by_index = {
            int(item["index"]): item
            for item in decisions
            if isinstance(item, dict) and "index" in item
        }
        cleaned: list[OddsRecord] = []
        for index, record in enumerate(records):
            decision = by_index.get(index)
            if not decision or decision.get("keep") is False:
                continue
            home = str(decision.get("home_team") or record.home_team).strip()
            away = str(decision.get("away_team") or record.away_team).strip()
            raw = dict(record.raw)
            raw["llm_cleaning"] = {
                "reason": decision.get("reason", ""),
                "home_before": record.home_team,
                "away_before": record.away_team,
            }
            cleaned.append(
                record.model_copy(
                    update={"home_team": home, "away_team": away, "raw": raw},
                    deep=True,
                )
            )
        if not cleaned:
            raise ValueError("LLM removed all records")
        return cleaned


def _mark_cleaning(records: list[OddsRecord], method: str) -> list[OddsRecord]:
    marked: list[OddsRecord] = []
    for record in records:
        raw = dict(record.raw)
        raw.setdefault("cleaning", {})
        raw["cleaning"]["method"] = method
        marked.append(record.model_copy(update={"raw": raw}, deep=True))
    return marked
