from __future__ import annotations

from .modeling import BASE_ELO, team_strength
from .news import RSSNewsProvider
from .schemas import AgentFinding, MatchPredictionRequest, OddsRecord, PredictionResult


class StrengthAgent:
    name = "实力分析员"

    def analyze(self, request: MatchPredictionRequest) -> AgentFinding:
        home = team_strength(request.home_team)
        away = team_strength(request.away_team)
        diff = home - away
        signal = "positive" if diff > 30 else "negative" if diff < -30 else "neutral"
        confidence = 0.72 if request.home_team in BASE_ELO or request.away_team in BASE_ELO else 0.45
        return AgentFinding(
            agent=self.name,
            confidence=confidence,
            signal=signal,
            rationale=(
                f"Elo 基准：{request.home_team} {home:.0f} vs "
                f"{request.away_team} {away:.0f}，实力差 {diff:+.0f}。"
            ),
            sources=["internal:baseline_elo"],
            metrics={"home_elo": home, "away_elo": away, "elo_diff": diff},
        )


class FormAgent:
    name = "近期状态分析员"

    def analyze(self, request: MatchPredictionRequest) -> AgentFinding:
        return AgentFinding(
            agent=self.name,
            confidence=0.4,
            signal="neutral",
            rationale=(
                "当前版本已预留历史战绩接入；若未导入 wc_results.csv，"
                "近期状态暂不额外调整，先按长期实力和赔率处理。"
            ),
            sources=["planned:international_results_ingestion"],
        )


class NewsSentimentAgent:
    name = "新闻舆情分析员"

    def analyze(self, request: MatchPredictionRequest) -> AgentFinding:
        provider = RSSNewsProvider(max_items=5)
        home_news = provider.fetch(request.home_team)
        away_news = provider.fetch(request.away_team)

        if not home_news.fetch_ok and not away_news.fetch_ok:
            return AgentFinding(
                agent=self.name,
                confidence=0.3,
                signal="neutral",
                rationale="新闻抓取失败或暂无可验证新闻，不调整概率。",
                sources=[],
            )

        home_injury = home_news.injury_signal
        away_injury = away_news.injury_signal

        if home_injury and not away_injury:
            signal = "negative"
            rationale = f"{request.home_team} 相关新闻出现伤停信号，可能削弱主队。"
            confidence = 0.6
        elif away_injury and not home_injury:
            signal = "positive"
            rationale = f"{request.away_team} 相关新闻出现伤停信号，客队可能减员。"
            confidence = 0.6
        elif home_injury and away_injury:
            signal = "neutral"
            rationale = "双方都有伤停信号，方向相互抵消，暂不单边调整。"
            confidence = 0.5
        else:
            signal = "neutral"
            rationale = "近期新闻未发现明确伤停信号，阵容状态暂按正常处理。"
            confidence = 0.45

        sources = [item.url for item in (home_news.items + away_news.items) if item.url][:5]
        return AgentFinding(
            agent=self.name,
            confidence=confidence,
            signal=signal,
            rationale=rationale,
            sources=sources,
            metrics={
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
        if not best_signal:
            return []
        return [
            AgentFinding(
                agent="正方研究员",
                confidence=0.6,
                signal="positive",
                rationale=(
                    f"正方：{best_signal.outcome} 的模型概率为 "
                    f"{best_signal.model_probability:.1%}。若 edge 为正且赔率稳定，"
                    "可以按风控建议小仓位参与。"
                ),
                sources=["internal:model_probability", "internal:odds_edge"],
            ),
            AgentFinding(
                agent="反方研究员",
                confidence=0.65,
                signal="negative",
                rationale=(
                    "反方：当前模型仍需更多历史数据、临场阵容和新闻伤停校验，"
                    "任何投注建议都应进行仓位折扣。"
                ),
                sources=["internal:model_risk"],
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
) -> list[AgentFinding]:
    return [
        StrengthAgent().analyze(request),
        FormAgent().analyze(request),
        NewsSentimentAgent().analyze(request),
        OddsMarketAgent().analyze(odds),
    ]
