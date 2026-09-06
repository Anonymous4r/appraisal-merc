from typing import Any

from .base import TabularConversationAdapter


class MeldAdapter(TabularConversationAdapter):
    labels = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")
    aliases_by_label = {
        "sad": "sadness", "happy": "joy", "happiness": "joy", "angry": "anger",
        "surprised": "surprise", "scared": "fear", "afraid": "fear", "disgusted": "disgust",
    }

    def normalize_label(self, value: Any, default: str = "neutral") -> str:
        label = self.aliases_by_label.get(str(value or "").strip().lower(), str(value or "").strip().lower())
        return label if label in self.labels else default

