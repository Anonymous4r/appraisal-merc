from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class DatasetKind(StrEnum):
    IEMOCAP = "iemocap"
    MELD = "meld"


class RunMode(StrEnum):
    INSPECT = "inspect"
    RUN = "run"
    EVALUATE = "evaluate"


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    root: Path
    dataset: DatasetKind
    data_path: Path
    output_path: Path
    sample_output_path: Path
    summary_output_path: Path
    prior_weights_path: Path
    model: str = "gpt-4.1-nano"
    reasoning_effort: str = "medium"
    limit_dialogues: int = 0
    conversation_workers: int = 32
    agent_workers: int = 6
    baseline_batch_size: int = 25
    write_every: int = 1
    max_retries: int = 3
    use_learned_prior: bool = True


@dataclass(frozen=True, slots=True)
class Utterance:
    key: str
    utterance_id: int
    speaker: str
    history: str
    text: str
    emotion: str
    visual: str = ""
    audio: str = ""

    def prompt_item(self, multimodal: bool = True) -> dict[str, Any]:
        value = {
            "utt_key": self.key,
            "utterance_id": self.utterance_id,
            "speaker": self.speaker,
            "history": self.history,
            "target_utterance": self.text,
        }
        if multimodal:
            value.update(visual=self.visual, audio=self.audio)
        return value


@dataclass(frozen=True, slots=True)
class Conversation:
    conversation_id: str
    utterances: tuple[Utterance, ...]

    @property
    def gold(self) -> list[str]:
        return [item.emotion for item in self.utterances]


@dataclass(slots=True)
class GenerationUsage:
    step: str
    model: str
    attempt: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class PipelineResult:
    conversation_id: str
    utterance_ids: list[int]
    gold: list[str]
    predictions: list[str]
    confidence: list[float]
    raw: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_seconds: float = 0.0

    @property
    def parse_ok(self) -> bool:
        return not self.error and len(self.predictions) == len(self.gold)

