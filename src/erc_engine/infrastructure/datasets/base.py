from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from erc_engine.domain.models import Conversation, Utterance


class TabularConversationAdapter(ABC):
    """Template Method: normalization is fixed; label policy varies by corpus."""

    labels: tuple[str, ...]

    def load(self, path: Path, limit: int = 0) -> list[Conversation]:
        frame = pd.read_csv(path)
        frame.columns = [self._normalize_column(value) for value in frame.columns]
        frame = frame.rename(columns=self.aliases())
        required = {"conversation_id", "utterance_id", "speaker", "utterance", "emotion"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        for optional in ("history", "visual", "audio"):
            if optional not in frame:
                frame[optional] = ""
        frame["conversation_id"] = frame["conversation_id"].astype(str)
        frame["utterance_id"] = pd.to_numeric(frame["utterance_id"], errors="coerce").fillna(0).astype(int)
        frame = frame.sort_values(["conversation_id", "utterance_id"])
        if limit > 0:
            ids = frame["conversation_id"].drop_duplicates().head(limit)
            frame = frame[frame["conversation_id"].isin(ids)]
        return [self._conversation(str(key), group) for key, group in frame.groupby("conversation_id", sort=True)]

    def _conversation(self, key: str, frame: pd.DataFrame) -> Conversation:
        rows = []
        for index, row in enumerate(frame.itertuples(index=False), 1):
            rows.append(Utterance(
                key=f"utt{index}",
                utterance_id=int(getattr(row, "utterance_id")),
                speaker=str(getattr(row, "speaker", "")),
                history=str(getattr(row, "history", "") or ""),
                text=str(getattr(row, "utterance", "") or ""),
                emotion=self.normalize_label(getattr(row, "emotion", "")),
                visual=str(getattr(row, "visual", "") or ""),
                audio=str(getattr(row, "audio", "") or ""),
            ))
        return Conversation(key, tuple(rows))

    @staticmethod
    def _normalize_column(value: str) -> str:
        return str(value).strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def aliases() -> dict[str, str]:
        return {
            "dialogue_id": "conversation_id", "dialogueid": "conversation_id", "conv_id": "conversation_id",
            "utteranceid": "utterance_id",
        }

    @abstractmethod
    def normalize_label(self, value: Any, default: str = "neutral") -> str: ...

