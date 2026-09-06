from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def _list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        try:
            return ast.literal_eval(str(value))
        except Exception:
            return []


class EvaluationService:
    """Query Service: derives reproducible metrics without mutating inference state."""

    def __init__(self, labels: tuple[str, ...]):
        self.labels = labels

    def evaluate(self, frame: pd.DataFrame, summary_path: Path | None = None) -> dict:
        if frame.empty:
            summary = {"conversations": 0, "utterances": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
        else:
            valid = frame[frame["parse_ok"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
            gold = [label for value in valid["gold"] for label in _list(value)]
            predicted = [label for value in valid["prediction"] for label in _list(value)]
            if len(gold) != len(predicted):
                raise ValueError("Gold and prediction lengths differ.")
            summary = {
                "conversations": int(len(valid)), "utterances": len(gold),
                "accuracy": float(accuracy_score(gold, predicted)) if gold else 0.0,
                "macro_f1": float(f1_score(gold, predicted, labels=self.labels, average="macro", zero_division=0)) if gold else 0.0,
                "weighted_f1": float(f1_score(gold, predicted, labels=self.labels, average="weighted", zero_division=0)) if gold else 0.0,
                "classification_report": classification_report(gold, predicted, labels=self.labels, zero_division=0, output_dict=True) if gold else {},
                "confusion_matrix": confusion_matrix(gold, predicted, labels=self.labels).tolist() if gold else [],
                "labels": list(self.labels),
            }
        if summary_path:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

