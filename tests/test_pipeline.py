from pathlib import Path

from erc_engine.application.pipeline import EmotionRecognitionPipeline
from erc_engine.domain.models import Conversation, DatasetKind, RuntimeSettings, Utterance
from erc_engine.infrastructure.datasets.meld import MeldAdapter
from erc_engine.strategies.prior import HeuristicAppraisalPrior
from erc_engine.strategies.prompts import CognitivePromptStrategy
from erc_engine.strategies.voting import EqualAgentWeightedVoting


class FakeJsonModel:
    def reset_usage(self):
        self.entries = []

    def usage(self):
        return list(self.entries)

    def absorb_usage(self, entries):
        self.entries.extend(entries)

    def complete(self, prompt, step):
        if step == "text_baseline":
            return {"predictions": [{"utt_key": "utt1", "emotion": "neutral", "confidence": 0.8}]}
        if step == "appraisal_extraction":
            return {"appraisals": [{"utt_key": "utt1", "appraisal": {"goal_relevance": "medium"}}]}
        module = step.removeprefix("parallel_agent_")
        return {"stage2": [{"utt_key": "utt1", "selected_modules": {
            module: {"emotion_hint": "neutral", "evidence_strength": "low"}
        }}]}


def test_full_pipeline_with_test_double():
    root = Path.cwd()
    settings = RuntimeSettings(root, DatasetKind.MELD, root / "input.csv", root / "out.csv",
                               root / "samples.csv", root / "summary.json", root / "prior.json")
    dataset = MeldAdapter()
    prompts = CognitivePromptStrategy(dataset.labels, DatasetKind.MELD)
    pipeline = EmotionRecognitionPipeline(
        settings, dataset, FakeJsonModel(), prompts, HeuristicAppraisalPrior(dataset.labels),
        EqualAgentWeightedVoting(dataset.labels, dataset.normalize_label),
    )
    conversation = Conversation("42", (Utterance("utt1", 1, "Speaker", "", "Okay.", "neutral"),))
    result = pipeline.execute(conversation)
    assert result.parse_ok
    assert result.predictions == ["neutral"]
    assert len(result.raw["stage2_selected_modules"]["stage2"][0]["selected_modules"]) == 6

