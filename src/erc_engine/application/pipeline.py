from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from erc_engine.domain.constants import MODULES
from erc_engine.domain.models import Conversation, PipelineResult, RuntimeSettings
from erc_engine.strategies.appraisal import AgentOutputSanitizer, AppraisalSanitizer, UnconditionalAssignmentPolicy


class EmotionRecognitionPipeline:
    """Facade coordinating the staged architecture without leaking infrastructure concerns."""

    def __init__(self, settings: RuntimeSettings, dataset, model, prompts, prior, voting):
        self.settings = settings
        self.dataset = dataset
        self.model = model
        self.prompts = prompts
        self.prior = prior
        self.voting = voting
        self.agent_sanitizer = AgentOutputSanitizer(dataset.normalize_label, dataset.labels)

    def execute(self, conversation: Conversation) -> PipelineResult:
        started = time.monotonic()
        self.model.reset_usage()
        timings: dict[str, float] = {}
        try:
            baseline = self._timed(timings, "text_baseline_seconds", lambda: self._baseline(conversation))
            appraisal = self._timed(timings, "appraisal_extraction_seconds", lambda: self._appraisal(conversation))
            assignments = UnconditionalAssignmentPolicy.assign(appraisal)
            experts = self._timed(timings, "parallel_experts_seconds", lambda: self._experts(conversation, assignments))
            prior = self._timed(timings, "appraisal_prior_seconds", lambda: self.prior.predict(assignments))
            fusion = self._timed(timings, "weighted_vote_seconds", lambda: self.voting.fuse(baseline, prior, experts))
            labels, confidence = self._predictions(fusion, len(conversation.utterances))
            usage = [asdict(item) for item in self.model.usage()]
            raw = {
                "architecture": "baseline + appraisal -> six parallel experts + appraisal prior -> deterministic vote",
                "dataset": self.settings.dataset.value,
                "model": self.settings.model,
                "baseline_text": baseline,
                "stage1_appraisal_gate": assignments,
                "stage2_selected_modules": experts,
                "stage2_5_appraisal_prototype_prior": prior,
                "stage3_weighted_vote": fusion,
                "timings": timings,
                "generation_usage": usage,
                "generation_usage_summary": self._usage_summary(usage),
            }
            return PipelineResult(conversation.conversation_id, [u.utterance_id for u in conversation.utterances],
                                  conversation.gold, labels, confidence, raw=raw,
                                  latency_seconds=round(time.monotonic() - started, 3))
        except Exception as exc:
            return PipelineResult(conversation.conversation_id, [u.utterance_id for u in conversation.utterances],
                                  conversation.gold, [], [], error=str(exc),
                                  latency_seconds=round(time.monotonic() - started, 3))

    def _baseline(self, conversation: Conversation) -> dict[str, Any]:
        size = max(1, self.settings.baseline_batch_size)
        chunks = [
            Conversation(conversation.conversation_id, conversation.utterances[start:start + size])
            for start in range(0, len(conversation.utterances), size)
        ]
        raw_rows = []
        for number, chunk in enumerate(chunks, 1):
            step = "text_baseline" if len(chunks) == 1 else f"text_baseline_batch_{number}"
            response = self.model.complete(self.prompts.baseline(chunk), step)
            raw_rows.extend(response.get("predictions", []))
        raw = {"predictions": raw_rows}
        indexed = {str(row.get("utt_key", "")): row for row in raw.get("predictions", []) if isinstance(row, dict)}
        predictions = []
        for utterance in conversation.utterances:
            row = indexed.get(utterance.key)
            if row is None:
                raise ValueError(f"Baseline omitted {utterance.key}")
            label = self.dataset.normalize_label(row.get("emotion"), "")
            if label not in self.dataset.labels:
                raise ValueError(f"Invalid label for {utterance.key}: {row.get('emotion')!r}")
            try:
                confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            predictions.append({**row, "utt_key": utterance.key, "emotion": label, "confidence": confidence})
        return {"predictions": predictions}

    def _appraisal(self, conversation: Conversation) -> dict[str, Any]:
        raw = self.model.complete(self.prompts.appraisal(conversation), "appraisal_extraction")
        return AppraisalSanitizer.sanitize(raw, [u.key for u in conversation.utterances])

    def _experts(self, conversation: Conversation, assignments: dict[str, Any]) -> dict[str, Any]:
        keys = [u.key for u in conversation.utterances]
        results, logs = {}, {}

        def run(module: str):
            self.model.reset_usage()
            raw = self.model.complete(self.prompts.agent(conversation, assignments, module), f"parallel_agent_{module}")
            return self.agent_sanitizer.sanitize(raw, keys, module), self.model.usage()

        with ThreadPoolExecutor(max_workers=max(1, min(self.settings.agent_workers, len(MODULES)))) as executor:
            futures = {executor.submit(run, module): module for module in MODULES}
            for future in as_completed(futures):
                module = futures[future]
                results[module], logs[module] = future.result()
        merged = {key: {"utt_key": key, "selected_modules": {}} for key in keys}
        for module in MODULES:
            self.model.absorb_usage(logs.get(module, []))
            for row in results.get(module, {}).get("stage2", []):
                payload = row.get("selected_modules", {}).get(module)
                if isinstance(payload, dict):
                    merged[row["utt_key"]]["selected_modules"][module] = payload
        return {"stage2": list(merged.values()), "execution": "six_parallel_api_calls"}

    def _predictions(self, fusion: dict[str, Any], expected: int) -> tuple[list[str], list[float]]:
        rows = fusion.get("predictions", [])
        if len(rows) != expected:
            raise ValueError(f"Fusion returned {len(rows)} rows; expected {expected}")
        return ([self.dataset.normalize_label(row.get("emotion")) for row in rows],
                [max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))) for row in rows])

    @staticmethod
    def _timed(timings: dict[str, float], name: str, operation):
        started = time.monotonic()
        result = operation()
        timings[name] = round(time.monotonic() - started, 3)
        return result

    @staticmethod
    def _usage_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {name: sum(int(row.get(name, 0)) for row in rows)
                for name in ("input_tokens", "output_tokens", "total_tokens")}
