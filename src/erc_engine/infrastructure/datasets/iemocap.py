from typing import Any

import pandas as pd

from .base import TabularConversationAdapter


class IemocapAdapter(TabularConversationAdapter):
    def __init__(self, data_path):
        values = pd.read_csv(data_path, usecols=["emotion"])["emotion"]
        self.labels = tuple(values.dropna().astype(str).str.strip().drop_duplicates())
        if not self.labels:
            raise ValueError("IEMOCAP contains no emotion labels.")

    def normalize_label(self, value: Any, default: str = "neutral") -> str:
        label = str(value or "").strip()
        return label if label in self.labels else default

