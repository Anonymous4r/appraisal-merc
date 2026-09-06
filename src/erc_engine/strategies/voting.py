from __future__ import annotations

from typing import Any, Callable


class EqualAgentWeightedVoting:
    """Composite Strategy: calibrated baseline/prior votes plus equal expert votes."""

    def __init__(self, labels: tuple[str, ...], normalize_label: Callable, prior_weight: float = 0.85):
        self.labels = labels
        self.normalize_label = normalize_label
        self.weights = {"text_baseline": 1.0, "appraisal_prior": prior_weight}

    def fuse(self, baseline: dict[str, Any], prior: dict[str, Any], experts: dict[str, Any]) -> dict[str, Any]:
        baseline_index = self._index(baseline, "predictions")
        prior_index = self._index(prior, "stage2_5_appraisal_prototype_prior")
        expert_index = self._index(experts, "stage2")
        keys = list(baseline_index)
        predictions = [self._vote(key, baseline_index.get(key, {}), prior_index.get(key, {}), expert_index.get(key, {})) for key in keys]
        return {"stage": "deterministic_weighted_vote", "source_weights": self.weights, "predictions": predictions}

    @staticmethod
    def _index(value: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
        return {str(row.get("utt_key", "")): row for row in value.get(field, []) if isinstance(row, dict)}

    def _vote(self, key: str, baseline: dict[str, Any], prior: dict[str, Any], experts: dict[str, Any]) -> dict[str, Any]:
        votes = []
        self._add(votes, "text_baseline", baseline.get("emotion"), baseline.get("confidence"), baseline.get("rationale", ""))
        self._add(votes, "appraisal_prior", prior.get("top_emotion"), prior.get("top_score"), prior.get("prior_source", ""))
        for module, payload in experts.get("selected_modules", {}).items():
            if isinstance(payload, dict):
                self._add(votes, module, payload.get("emotion_hint"), 1.0, payload.get("evidence", ""))
        scores = {label: 0.0 for label in self.labels}
        for vote in votes:
            scores[vote["emotion"]] += vote["contribution"]
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], self.labels.index(pair[0])))
        label, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(scores.values())
        confidence = max(0.0, min(1.0, float(baseline.get("confidence", 0.0) or 0.0))) if total <= 0 else min(1.0, 0.5 * top / total + 0.5 * min(1.0, top))
        return {"utt_key": key, "emotion": label, "confidence": round(confidence, 4),
                "evidence_used": [v["source"] for v in votes if v["emotion"] == label],
                "rationale": f"weighted_vote top={label} score={top:.4f} margin={top-second:.4f}",
                "vote_scores": scores, "votes": votes}

    def _add(self, votes: list[dict[str, Any]], source: str, emotion: Any, confidence: Any, evidence: str) -> None:
        label = self.normalize_label(emotion, "")
        if label not in self.labels:
            return
        try:
            conf = max(0.0, min(1.0, float(confidence or 0.0)))
        except (TypeError, ValueError):
            conf = 0.0
        weight = self.weights.get(source, 1.0)
        if conf * weight > 0:
            votes.append({"source": source, "emotion": label, "confidence": conf, "weight": weight,
                          "contribution": round(conf * weight, 6), "evidence": str(evidence)[:300]})

