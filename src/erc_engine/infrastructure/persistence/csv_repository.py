from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


FIELDS = (
    "conversation_id", "model", "dataset", "utterance_ids", "gold", "prediction", "confidence",
    "parse_ok", "error", "latency_seconds", "input_tokens", "output_tokens", "total_tokens", "raw_json",
)


class CsvResultRepository:
    """Repository + Unit of Work boundary for append-only, resumable results."""

    def __init__(self, path: Path, settings):
        self.path = path
        self.settings = settings

    def completed_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            return {str(row["conversation_id"]) for row in csv.DictReader(handle)
                    if str(row.get("parse_ok", "")).lower() in {"true", "1", "yes"}}

    def append(self, results) -> None:
        if not results:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            for result in results:
                usage = result.raw.get("generation_usage_summary", {})
                writer.writerow({
                    "conversation_id": result.conversation_id,
                    "model": self.settings.model,
                    "dataset": self.settings.dataset.value,
                    "utterance_ids": json.dumps(result.utterance_ids),
                    "gold": json.dumps(result.gold, ensure_ascii=False),
                    "prediction": json.dumps(result.predictions, ensure_ascii=False),
                    "confidence": json.dumps(result.confidence),
                    "parse_ok": result.parse_ok,
                    "error": result.error,
                    "latency_seconds": result.latency_seconds,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "raw_json": json.dumps(result.raw, ensure_ascii=False),
                })

    def read(self):
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=FIELDS)
