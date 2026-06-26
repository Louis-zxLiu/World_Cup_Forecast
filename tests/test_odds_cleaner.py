import pytest
import json

from worldcup_forecast.odds_cleaner import OddsCleaner, rule_clean_odds
from worldcup_forecast.llm import OpenAICompatibleClient
from worldcup_forecast.schemas import LLMSettings, OddsRecord


def test_rule_clean_odds_removes_page_tokens():
    records = [
        OddsRecord(
            match_id="周五001",
            home_team="单关",
            away_team="亚 欧 荐",
            play_type="胜平负",
            win_odds=6,
            draw_odds=2,
            lose_odds=4.5,
            source_url="https://trade.500.com/jczq/",
        ),
        OddsRecord(
            match_id="周五002",
            home_team="巴西",
            away_team="德国",
            play_type="胜平负",
            win_odds=2.2,
            draw_odds=3.1,
            lose_odds=2.8,
            source_url="https://trade.500.com/jczq/",
        ),
    ]

    cleaned, warnings = rule_clean_odds(records)

    assert len(cleaned) == 1
    assert cleaned[0].home_team == "Brazil"
    assert cleaned[0].away_team == "Germany"
    assert warnings


@pytest.mark.asyncio
async def test_odds_cleaner_falls_back_to_rules_without_llm():
    records = [
        OddsRecord(
            match_id="周五002",
            home_team="巴西",
            away_team="德国",
            play_type="胜平负",
            win_odds=2.2,
            draw_odds=3.1,
            lose_odds=2.8,
            source_url="https://trade.500.com/jczq/",
        )
    ]

    cleaned, warnings = await OddsCleaner(LLMSettings(enabled=False)).clean(records)

    assert len(cleaned) == 1
    assert cleaned[0].home_team == "Brazil"
    assert any("LLM 未启用" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_odds_cleaner_uses_llm_in_batches(monkeypatch):
    async def fake_complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("待清洗记录：", 1)[1])
        return json.dumps(
            [
                {
                    "index": row["index"],
                    "keep": True,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "reason": "ok",
                }
                for row in payload
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    records = [
        OddsRecord(
            match_id=f"周五{i:03d}",
            home_team="巴西",
            away_team="德国",
            play_type="胜平负",
            win_odds=2.2,
            draw_odds=3.1,
            lose_odds=2.8,
            source_url="https://trade.500.com/jczq/",
        )
        for i in range(85)
    ]

    cleaned, warnings = await OddsCleaner(
        LLMSettings(enabled=True, api_key="sk-test")
    ).clean(records)

    assert len(cleaned) == 85
    assert all(record.raw["cleaning"]["method"] == "llm" for record in cleaned)
    assert any("LLM 清洗完成" in warning for warning in warnings)
