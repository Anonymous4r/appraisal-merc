from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _level(value: Any) -> float:
    return {"low": 0.0, "medium": 0.5, "high": 1.0, "unclear": 0.5}.get(str(value or "").lower(), 0.0)


def appraisal_features(value: dict[str, Any]) -> dict[str, float]:
    goal = str(value.get("goal_congruence", "unclear")).lower()
    agency = str(value.get("agency", "unclear")).lower()
    target = str(value.get("appraisal_target", "unclear")).lower()
    control, certainty = _level(value.get("control")), _level(value.get("certainty"))
    features = {
        "bias": 1.0, "goal_relevance": _level(value.get("goal_relevance")),
        "goal_positive": float(goal == "supports_goal"), "goal_negative": float(goal == "blocks_goal"),
        "goal_mixed_or_unclear": float(goal in {"mixed", "unclear", ""}),
        "control": control, "low_control": 1.0 - control,
        "certainty": certainty, "low_certainty": 1.0 - certainty,
    }
    for name in ("self", "other", "shared", "environment", "unclear"):
        features[f"agency_{name}"] = float(agency == name or name == "unclear" and not agency)
    for name in ("self", "other", "relationship", "event", "unclear"):
        features[f"target_{name}"] = float(target == name or name == "unclear" and not target)
    for name in (
        "novelty", "social_threat", "moral_violation", "affect_intensity",
        "loss_or_separation", "blame_responsibility", "expectancy_violation",
    ):
        features[name] = _level(value.get(name))
    return features


class LearnedAppraisalPrior:
    """Strategy backed by multinomial weights exported by the training pipeline."""

    def __init__(self, path: Path, labels: tuple[str, ...]):
        self.labels = labels
        self.payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"feature_names", "emotion_labels", "weights"}
        if not required.issubset(self.payload):
            raise ValueError(f"Invalid prior schema: {path}")

    def predict(self, assignments: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for item in assignments.get("stage1_appraisal_gate", []):
            features = appraisal_features(item.get("appraisal", {}))
            scores = {
                label: sum(float(self.payload["weights"].get(label, {}).get(name, 0.0)) * features.get(name, 0.0)
                           for name in self.payload["feature_names"])
                for label in self.labels
            }
            maximum = max(scores.values(), default=0.0)
            exp = {label: math.exp(score - maximum) for label, score in scores.items()}
            rows.append({"utt_key": item.get("utt_key"), **self._calibrate(exp, "learned_appraisal_prior")})
        return {"stage2_5_appraisal_prototype_prior": rows}

    def _calibrate(self, scores: dict[str, float], source: str) -> dict[str, Any]:
        total = sum(max(value, 0.0) for value in scores.values()) or 1.0
        probs = {label: round(max(scores.get(label, 0.0), 0.0) / total, 4) for label in self.labels}
        ranked = sorted(probs.items(), key=lambda pair: pair[1], reverse=True)
        label, score = ranked[0]
        margin = round(score - (ranked[1][1] if len(ranked) > 1 else 0.0), 4)
        tier = "strong" if score >= 0.50 and margin >= 0.15 else "moderate" if score >= 0.40 and margin >= 0.10 else "weak"
        return {"top_emotion": label, "top_score": score, "prior_margin": margin,
                "confidence_tier": tier, "use_for_override": tier == "strong", "scores": probs, "prior_source": source}


class HeuristicAppraisalPrior:
    """Null-safe fallback Strategy used when trained weights are intentionally unavailable."""

    def __init__(self, labels: tuple[str, ...]):
        self.labels = labels

    def predict(self, assignments: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for item in assignments.get("stage1_appraisal_gate", []):
            app = item.get("appraisal", {})
            scores = {label: 0.05 for label in self.labels}
            scores["neutral"] = 0.45
            if app.get("goal_congruence") == "supports_goal":
                scores[next((x for x in ("joy", "happy", "excited") if x in scores), "neutral")] += 0.7
            if app.get("goal_congruence") == "blocks_goal":
                for label in ("anger", "frustrated", "sadness", "sad"):
                    if label in scores:
                        scores[label] += 0.35
            if app.get("expectancy_violation") == "high" and "surprise" in scores:
                scores["surprise"] += 0.7
            total = sum(scores.values())
            probs = {key: round(value / total, 4) for key, value in scores.items()}
            top = max(self.labels, key=probs.get)
            rows.append({"utt_key": item.get("utt_key"), "top_emotion": top, "top_score": probs[top],
                         "scores": probs, "prior_source": "heuristic_appraisal_prior"})
        return {"stage2_5_appraisal_prototype_prior": rows}


class PriorFactory:
    @staticmethod
    def create(path: Path, labels: tuple[str, ...], learned: bool):
        return LearnedAppraisalPrior(path, labels) if learned else HeuristicAppraisalPrior(labels)

