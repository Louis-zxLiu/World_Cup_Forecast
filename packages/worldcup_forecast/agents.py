from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .form import team_form
from .modeling import BASE_ELO, team_strength
from .news import RSSNewsProvider
from .schemas import AgentFinding, MatchPredictionRequest, OddsRecord, PredictionResult

if TYPE_CHECKING:
    from .schemas import SearchSettings
    from .search import SearchOutcome
    from .storage import ForecastStore

# Injury / availability keywords for scanning news titles and snippets.
_INJURY_KEYWORDS = re.compile(
    r"injur|absence|absent|doubt|suspend|ruled out|伤|缺阵|缺席|停赛|伤停|受伤|无缘|休战|疑似",
    re.IGNORECASE,
)


def _has_injury_signal(outcome: "SearchOutcome") -> bool:
    for hit in outcome.hits:
        text = f"{hit.title} {hit.snippet}"
        if _INJURY_KEYWORDS.search(text):
            return True
    return False


def _injury_to_signal(
    request: MatchPredictionRequest,
    home_injury: bool,
    away_injury: bool,
    search: bool,
) -> tuple[str, str, float]:
    if home_injury and not away_injury:
        return "negative", f"{request.home_team} 相关报道出现伤停/缺阵信号，可能削弱主队。", 0.6
    if away_injury and not home_injury:
        return "positive", f"{request.away_team} 相关报道出现伤停/缺阵信号，客队可能减员。", 0.6
    if home_injury and away_injury:
        return "neutral", "双方都有伤停信号，方向相互抵消，暂不单边调整。", 0.5
    return "neutral", "近期报道未发现明确伤停信号，阵容状态暂按正常处理。", 0.45


class StrengthAgent:
    name = "实力分析员"

    def __init__(self, store: "ForecastStore | None" = None) -> None:
        self.store = store
        self._elo_map = store.get_team_elo() if store is not None else {}

    def analyze(self, request: MatchPredictionRequest) -> AgentFinding:
        home = team_strength(request.home_team, self._elo_map)
        away = team_strength(request.away_team, self._elo_map)
        diff = home - away
        signal = "positive" if diff > 30 else "negative" if diff < -30 else "neutral"
        known = request.home_team in self._elo_map or request.away_team in self._elo_map
        known = known or request.home_team in BASE_ELO or request.away_team in BASE_ELO
        confidence = 0.72 if known else 0.45
        source = "internal:team_elo" if self._elo_map else "internal:baseline_elo"
        return AgentFinding(
            agent=self.name,
            confidence=confidence,
            signal=signal,
            rationale=(
                f"Elo 评分：{request.home_team} {home:.0f} vs "
                f"{request.away_team} {away:.0f}，实力差 {diff:+.0f}。"
            ),
            sources=[source],
            metrics={"home_elo": home, "away_elo": away, "elo_diff": diff},
        )


class FormAgent:
    name = "近期状态分析员"

    def __init__(self, store: "ForecastStore | None" = None) -> None:
        self.store = store

    def analyze(self, request: MatchPredictionRequest) -> AgentFinding:
        if self.store is None or self.store.intl_count() == 0:
            return AgentFinding(
                agent=self.name,
                confidence=0.4,
                signal="neutral",
                rationale=(
                    "尚未载入国际比赛档案，近期状态暂不调整，先按长期实力和赔率处理。"
                ),
                sources=["planned:international_results_ingestion"],
            )

        home = team_form(self.store, request.home_team, limit=10)
        away = team_form(self.store, request.away_team, limit=10)

        if home.matches == 0 and away.matches == 0:
            return AgentFinding(
                agent=self.name,
                confidence=0.35,
                signal="neutral",
                rationale="两队近期均无可用国际比赛记录，状态维度暂不调整。",
                sources=["internal:intl_results"],
            )

        ppg_gap = home.points_per_game - away.points_per_game
        if ppg_gap > 0.5:
            signal = "positive"
        elif ppg_gap < -0.5:
            signal = "negative"
        else:
            signal = "neutral"
        # Confidence grows with how much recent data backs each side.
        confidence = round(min(0.8, 0.45 + 0.02 * (home.matches + away.matches)), 2)
        rationale = (
            f"近10场：{request.home_team} {home.form_string()}"
            f"（场均{home.points_per_game:.2f}分，净胜球{home.goal_diff_per_game:+.2f}）"
            f"，{request.away_team} {away.form_string()}"
            f"（场均{away.points_per_game:.2f}分，净胜球{away.goal_diff_per_game:+.2f}）。"
        )
        return AgentFinding(
            agent=self.name,
            confidence=confidence,
            signal=signal,
            rationale=rationale,
            sources=["internal:intl_results"],
            metrics={
                "home_ppg": round(home.points_per_game, 3),
                "away_ppg": round(away.points_per_game, 3),
                "home_form": home.form_string(),
                "away_form": away.form_string(),
                "home_gd_per_game": round(home.goal_diff_per_game, 3),
                "away_gd_per_game": round(away.goal_diff_per_game, 3),
                "ppg_gap": round(ppg_gap, 3),
            },
        )


class NewsSentimentAgent:
    name = "新闻舆情分析员"

    def __init__(self, search_settings: "SearchSettings | None" = None) -> None:
        self.search_settings = search_settings

    def analyze(self, request: MatchPredictionRequest) -> AgentFinding:
        # Use configured web-search layer when enabled.
        if self.search_settings and self.search_settings.enabled and self.search_settings.api_key:
            return self._analyze_via_search(request)
        # If search is explicitly disabled or not configured, report honestly.
        if self.search_settings is not None and (
            not self.search_settings.enabled or self.search_settings.provider == "none"
        ):
            return AgentFinding(
                agent=self.name,
                confidence=0.3,
                signal="neutral",
                rationale=(
                    "未启用联网搜索源，无法获取伤停/阵容新闻，本维度暂不调整概率。"
                    "可在系统设置里配置并启用联网搜索 API（博查/智谱）。"
                ),
                sources=[],
                metrics={"search_ok": False},
            )
        # Legacy RSS fallback when no search_settings provided at all.
        return self._analyze_via_rss(request)

    def _analyze_via_search(self, request: MatchPredictionRequest) -> AgentFinding:
        from .search import WebSearchProvider

        provider = WebSearchProvider(self.search_settings)
        home = provider.search_sync(f"{request.home_team} 国家队 伤停 阵容 最新")
        away = provider.search_sync(f"{request.away_team} 国家队 伤停 阵容 最新")

        if not home.ok and not away.ok:
            return AgentFinding(
                agent=self.name,
                confidence=0.3,
                signal="neutral",
                rationale=f"新闻搜索源不可用（{home.error or away.error}），本维度暂不调整概率。",
                sources=[],
                metrics={"search_ok": False},
            )

        home_injury = _has_injury_signal(home)
        away_injury = _has_injury_signal(away)
        signal, rationale, confidence = _injury_to_signal(
            request, home_injury, away_injury, search=True
        )
        sources = [h.url for h in (home.hits + away.hits) if h.url][:5]
        return AgentFinding(
            agent=self.name,
            confidence=confidence,
            signal=signal,
            rationale=rationale,
            sources=sources,
            metrics={
                "search_ok": True,
                "home_results": len(home.hits),
                "away_results": len(away.hits),
                "home_injury_signal": home_injury,
                "away_injury_signal": away_injury,
            },
        )

    def _analyze_via_rss(self, request: MatchPredictionRequest) -> AgentFinding:
        provider = RSSNewsProvider(max_items=5)
        home_news = provider.fetch(request.home_team)
        away_news = provider.fetch(request.away_team)

        if (not home_news.fetch_ok and not away_news.fetch_ok) or (
            not home_news.items and not away_news.items
        ):
            return AgentFinding(
                agent=self.name,
                confidence=0.3,
                signal="neutral",
                rationale=(
                    "未配置可用的新闻搜索源，且默认 RSS 源在当前网络下不可达，"
                    "本维度暂不调整概率。可在系统设置里配置联网搜索 API。"
                ),
                sources=[],
                metrics={"search_ok": False},
            )

        home_injury = home_news.injury_signal
        away_injury = away_news.injury_signal
        signal, rationale, confidence = _injury_to_signal(
            request, home_injury, away_injury, search=False
        )
        sources = [item.url for item in (home_news.items + away_news.items) if item.url][:5]
        return AgentFinding(
            agent=self.name,
            confidence=confidence,
            signal=signal,
            rationale=rationale,
            sources=sources,
            metrics={
                "search_ok": True,
                "home_headlines": len(home_news.items),
                "away_headlines": len(away_news.items),
                "home_injury_signal": home_injury,
                "away_injury_signal": away_injury,
            },
        )


class OddsMarketAgent:
    name = "赔率市场分析员"

    def analyze(self, odds: OddsRecord | None) -> AgentFinding:
        if not odds:
            return AgentFinding(
                agent=self.name,
                confidence=0.25,
                signal="neutral",
                rationale="暂未匹配到该场 500彩票网胜平负赔率，edge 只按模型概率占位。",
                sources=[],
            )
        return AgentFinding(
            agent=self.name,
            confidence=0.78,
            signal="neutral",
            rationale=(
                f"匹配到 {odds.source} {odds.play_type} 赔率："
                f"胜 {odds.win_odds}，平 {odds.draw_odds}，负 {odds.lose_odds}。"
            ),
            sources=[odds.source_url],
            metrics={
                "win_odds": odds.win_odds,
                "draw_odds": odds.draw_odds,
                "lose_odds": odds.lose_odds,
            },
        )


class BullBearDebateAgents:
    def debate(self, result: PredictionResult) -> list[AgentFinding]:
        best_signal = result.bet_signals[0] if result.bet_signals else None
        probs = result.probabilities
        home = result.match.home_team
        away = result.match.away_team

        # Pull supporting metrics from sibling agents already in the result.
        strength_metrics: dict = next(
            (f.metrics for f in result.agent_findings if f.metrics and "elo_diff" in f.metrics),
            {},
        )
        form_metrics: dict = next(
            (f.metrics for f in result.agent_findings if f.metrics and "ppg_gap" in f.metrics),
            {},
        )

        # Base metrics shared by both sides.
        base: dict = {
            "home_win_prob": round(probs.home_win, 3),
            "draw_prob": round(probs.draw, 3),
            "away_win_prob": round(probs.away_win, 3),
        }
        if strength_metrics:
            base["elo_diff"] = strength_metrics.get("elo_diff", 0)
        if form_metrics:
            base["ppg_gap"] = form_metrics.get("ppg_gap", 0)

        bull_metrics = dict(base)
        bear_metrics = dict(base)

        if best_signal:
            bull_metrics.update({
                "best_outcome": best_signal.outcome,
                "model_prob": round(best_signal.model_probability, 3),
                "kelly_fraction": round(best_signal.kelly_fraction, 3),
                "stake": round(best_signal.stake, 2),
            })
            bear_metrics.update({
                "best_outcome": best_signal.outcome,
                "model_prob": round(best_signal.model_probability, 3),
            })
            if best_signal.edge is not None:
                bull_metrics["edge"] = round(best_signal.edge, 3)
                bear_metrics["edge"] = round(best_signal.edge, 3)
            if best_signal.market_probability is not None:
                bear_metrics["market_prob"] = round(best_signal.market_probability, 3)

        # Build data-driven rationale strings.
        if best_signal:
            edge_str = f"，edge {best_signal.edge:.1%}" if best_signal.edge is not None else ""
            bull_rationale = (
                f"正方：{best_signal.outcome} 模型概率 {best_signal.model_probability:.1%}{edge_str}，"
                f"Kelly 建议仓位 {best_signal.kelly_fraction:.2f}（{best_signal.stake:.2f} 元）。"
                f"主队胜率 {probs.home_win:.1%} / 平 {probs.draw:.1%} / 客队 {probs.away_win:.1%}。"
            )
        else:
            bull_rationale = (
                f"正方：{home} 主场胜率 {probs.home_win:.1%}，平局 {probs.draw:.1%}，"
                f"{away} 获胜 {probs.away_win:.1%}。无达标价值投注信号，可小仓观察。"
            )

        if best_signal and best_signal.edge is not None:
            if best_signal.market_probability is not None:
                bear_rationale = (
                    f"反方：市场隐含概率 {best_signal.market_probability:.1%} vs 模型 "
                    f"{best_signal.model_probability:.1%}，差异 {best_signal.edge:.1%}；"
                    "赛前阵容/伤停信息可能未充分反映，建议仓位保守。"
                )
            else:
                bear_rationale = (
                    f"反方：模型 edge {best_signal.edge:.1%}，但缺少赔率数据校验，"
                    "阵容/伤停信号尚未确认，任何投注建议都应进行仓位折扣。"
                )
        else:
            bear_rationale = (
                f"反方：三方概率 {home} {probs.home_win:.1%} / 平 {probs.draw:.1%} / "
                f"{away} {probs.away_win:.1%}，无赔率数据时 edge 可信度下降，建议观望。"
            )

        return [
            AgentFinding(
                agent="正方研究员",
                confidence=0.6,
                signal="positive",
                rationale=bull_rationale,
                sources=["internal:model_probability", "internal:odds_edge"],
                metrics=bull_metrics,
            ),
            AgentFinding(
                agent="反方研究员",
                confidence=0.65,
                signal="negative",
                rationale=bear_rationale,
                sources=["internal:model_risk"],
                metrics=bear_metrics,
            ),
        ]


class RiskManagerAgent:
    name = "风控经理"

    def analyze(self, result: PredictionResult) -> AgentFinding:
        max_stake = max((signal.stake for signal in result.bet_signals), default=0)
        signal = "positive" if max_stake > 0 else "neutral"
        req = result.match
        return AgentFinding(
            agent=self.name,
            confidence=0.7,
            signal=signal,
            rationale=(
                f"风控：Kelly 折扣 {req.kelly_fraction:.2f}，单项仓位上限 "
                f"{req.max_stake_fraction:.1%}，本场最高建议仓位 {max_stake:.2f}。"
            ),
            sources=["internal:fractional_kelly"],
            metrics={"max_stake": max_stake},
        )


def template_explanation(result: PredictionResult) -> str:
    top = result.bet_signals[0] if result.bet_signals else None
    if top and top.edge is not None and top.edge > result.match.value_edge_threshold:
        value_line = (
            f"当前最强价值信号是 {top.outcome}，edge {top.edge:.1%}，"
            f"建议仓位 {top.stake:.2f}。"
        )
    else:
        value_line = "当前没有达到阈值的价值投注信号，建议观望。"
    return (
        f"{result.match.home_team} vs {result.match.away_team}："
        f"胜/平/负概率为 {result.probabilities.home_win:.1%}/"
        f"{result.probabilities.draw:.1%}/{result.probabilities.away_win:.1%}。"
        f"期望比分 {result.expected_score[0]:.2f}-{result.expected_score[1]:.2f}，"
        f"最可能比分 {result.most_likely_score}。{value_line}"
    )


def build_agent_findings(
    request: MatchPredictionRequest,
    odds: OddsRecord | None,
    store: "ForecastStore | None" = None,
    search_settings: "SearchSettings | None" = None,
) -> list[AgentFinding]:
    return [
        StrengthAgent(store).analyze(request),
        FormAgent(store).analyze(request),
        NewsSentimentAgent(search_settings).analyze(request),
        OddsMarketAgent().analyze(odds),
    ]
