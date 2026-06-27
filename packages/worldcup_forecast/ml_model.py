from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .modeling import BASE_ELO, BaselineForecastModel, ScoreDistribution, normalize, poisson_probability
from .schemas import MatchPredictionRequest, OutcomeProbability
from .storage import ForecastStore

MODEL_PATH = Path("data/xgb_model.pkl")


def _team_elo(team: str, elo_map: dict[str, float]) -> float:
    return elo_map.get(team, BASE_ELO.get(team, 1500.0))


def _recent_goals(team: str, results: list[dict], n: int = 5, as_home: bool = True) -> tuple[float, float]:
    matches = [
        r for r in results
        if (r["home_team"] == team if as_home else r["away_team"] == team)
    ][-n:]
    if not matches:
        return 1.2, 1.0
    if as_home:
        gf = [r["home_score"] for r in matches]
        ga = [r["away_score"] for r in matches]
    else:
        gf = [r["away_score"] for r in matches]
        ga = [r["home_score"] for r in matches]
    return float(np.mean(gf)), float(np.mean(ga))


class FeatureBuilder:
    def __init__(self, store: ForecastStore | None = None) -> None:
        self.store = store or ForecastStore()
        self._results: list[dict] | None = None
        self._elo: dict[str, float] | None = None

    def _lazy_load(self) -> None:
        if self._results is None:
            self._results = self.store.get_match_results()
        if self._elo is None:
            db_elo = self.store.get_team_elo()
            self._elo = {**BASE_ELO, **db_elo}

    def build(self, home: str, away: str, neutral: bool = True) -> dict[str, float]:
        self._lazy_load()
        assert self._results is not None
        assert self._elo is not None
        h_elo = _team_elo(home, self._elo)
        a_elo = _team_elo(away, self._elo)
        h_gf, h_ga = _recent_goals(home, self._results, as_home=True)
        a_gf, a_ga = _recent_goals(away, self._results, as_home=False)
        return {
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": h_elo - a_elo,
            "home_advantage": 0.0 if neutral else 1.0,
            "home_avg_gf_5": h_gf,
            "home_avg_ga_5": h_ga,
            "away_avg_gf_5": a_gf,
            "away_avg_ga_5": a_ga,
        }

    def feature_names(self) -> list[str]:
        return [
            "home_elo", "away_elo", "elo_diff", "home_advantage",
            "home_avg_gf_5", "home_avg_ga_5", "away_avg_gf_5", "away_avg_ga_5",
        ]


class XGBoostForecaster:
    version_prefix = "xgboost-v"

    def __init__(self, store: ForecastStore | None = None, model_path: Path = MODEL_PATH) -> None:
        self.store = store or ForecastStore()
        self.model_path = model_path
        self.fb = FeatureBuilder(self.store)
        self._clf: Any = None
        self._version: str = "xgboost-v0"

    @property
    def version(self) -> str:
        return self._version

    def _build_dataset(self) -> tuple[Any, Any]:
        import numpy as np
        results = self.store.get_match_results()
        if len(results) < 10:
            raise ValueError("Not enough match results to train (need >= 10)")
        X, y = [], []
        for row in results:
            feats = self.fb.build(row["home_team"], row["away_team"], bool(row["neutral"]))
            X.append(list(feats.values()))
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            if hs > as_:
                y.append(0)
            elif hs == as_:
                y.append(1)
            else:
                y.append(2)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def train(self) -> str:
        from sklearn.calibration import CalibratedClassifierCV
        from xgboost import XGBClassifier
        X, y = self._build_dataset()
        base = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        clf.fit(X, y)
        self._clf = clf
        self._version = f"xgboost-v{len(self.store.get_match_results())}"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump((clf, self._version), f)
        return self._version

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            with open(self.model_path, "rb") as f:
                self._clf, self._version = pickle.load(f)
            return True
        except Exception:
            return False

    def predict_proba(self, home: str, away: str, neutral: bool = True) -> OutcomeProbability:
        if self._clf is None and not self.load():
            return BaselineForecastModel().predict_score_distribution(
                MatchPredictionRequest(home_team=home, away_team=away, neutral_site=neutral)
            ).probabilities
        import numpy as np
        feats = self.fb.build(home, away, neutral)
        X = np.array([list(feats.values())], dtype=np.float32)
        proba = self._clf.predict_proba(X)[0]
        # classes: 0=home_win, 1=draw, 2=away_win
        return OutcomeProbability(home_win=float(proba[0]), draw=float(proba[1]), away_win=float(proba[2]))

    def predict_score_distribution(self, request: MatchPredictionRequest) -> ScoreDistribution:
        probs = self.predict_proba(request.home_team, request.away_team, request.neutral_site)
        # derive goal rates from elo for score distribution, but use calibrated class probs
        baseline = BaselineForecastModel().predict_score_distribution(request)
        return ScoreDistribution(
            probabilities=probs,
            expected_home_goals=baseline.expected_home_goals,
            expected_away_goals=baseline.expected_away_goals,
            most_likely_score=baseline.most_likely_score,
            score_matrix=baseline.score_matrix,
        )


class SHAPExplainer:
    def __init__(self, forecaster: XGBoostForecaster) -> None:
        self.forecaster = forecaster
        self.fb = forecaster.fb

    def top_features(self, home: str, away: str, neutral: bool = True, top_n: int = 5) -> dict[str, float]:
        if self.forecaster._clf is None:
            return {}
        try:
            import shap
            import numpy as np
            feats = self.fb.build(home, away, neutral)
            X = np.array([list(feats.values())], dtype=np.float32)
            # Use the base estimator for SHAP (first calibrated estimator)
            base_est = self.forecaster._clf.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base_est)
            shap_values = explainer.shap_values(X)
            # shap_values shape: (n_classes, n_samples, n_features) or (n_samples, n_features)
            if isinstance(shap_values, list):
                # take class 0 (home_win) shap values
                sv = shap_values[0][0]
            else:
                sv = shap_values[0]
            names = self.fb.feature_names()
            pairs = sorted(zip(names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
            return {k: round(v, 4) for k, v in pairs[:top_n]}
        except Exception:
            return {}
