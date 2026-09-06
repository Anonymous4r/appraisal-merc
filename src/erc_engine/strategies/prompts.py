from __future__ import annotations

import json
from typing import Any

from erc_engine.domain.constants import APPRAISAL_FIELDS
from erc_engine.domain.models import Conversation, DatasetKind

from .catalog import MODULE_CONTRACTS


class CognitivePromptStrategy:
    """Strategy: serializes domain objects into strict stage contracts."""

    def __init__(self, labels: tuple[str, ...], dataset: DatasetKind):
        self.labels = labels
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        return (
            f"You are an expert psychologist and {self.dataset.value.upper()} emotion-recognition system. "
            f"Allowed labels are exactly: {', '.join(self.labels)}. Return only valid JSON."
        )

    def baseline(self, conversation: Conversation) -> str:
        return self._dump({
            "stage": "text_baseline",
            "task": "Predict exactly one primary emotion for every utterance.",
            "emotion_labels": self.labels,
            "rules": ["Preserve input order.", "Use only an allowed label.", "Return JSON only."],
            "output_schema": {"predictions": [{"utt_key": "utt1", "emotion": "neutral", "confidence": 0.0, "rationale": "brief"}]},
            "items": [item.prompt_item(multimodal=False) for item in conversation.utterances],
        })

    def appraisal(self, conversation: Conversation) -> str:
        return self._dump({
            "stage": "appraisal_extraction",
            "task": "Extract cognitive appraisal variables without predicting final labels.",
            "appraisal_schema": {key: " | ".join(sorted(values)) for key, (values, _) in APPRAISAL_FIELDS.items()},
            "rules": ["Return one item per utterance.", "Ground evidence in supplied fields.", "Return JSON only."],
            "output_schema": {"appraisals": [{"utt_key": "utt1", "appraisal": {}, "evidence": "brief"}]},
            "items": [item.prompt_item() for item in conversation.utterances],
        })

    def agent(self, conversation: Conversation, assignments: dict[str, Any], module: str) -> str:
        return self._dump({
            "stage": f"parallel_agent_{module}",
            "task": f"Run only {module} independently for every utterance.",
            "emotion_labels": self.labels,
            "module_contract": MODULE_CONTRACTS[module],
            "assignments": assignments,
            "rules": [
                "Return one item per utterance.", "Use unclear when evidence is insufficient.",
                "evidence_strength must be none, low, medium or high.", "Return JSON only.",
            ],
            "output_schema": {"stage2": [{"utt_key": "utt1", "selected_modules": {module: {
                "emotion_hint": "neutral", "evidence_strength": "low", "evidence": "", "rationale": ""
            }}}]},
            "items": [item.prompt_item() for item in conversation.utterances],
        })

    @staticmethod
    def _dump(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False)

