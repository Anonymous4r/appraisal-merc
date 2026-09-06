from __future__ import annotations

from typing import Any

from erc_engine.domain.constants import APPRAISAL_FIELDS, MODULES


class AppraisalSanitizer:
    @staticmethod
    def sanitize(raw: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        source = raw.get("appraisals", []) if isinstance(raw, dict) else []
        indexed = {str(row.get("utt_key", "")): row for row in source if isinstance(row, dict)}
        result = []
        for key in keys:
            row = indexed.get(key, {})
            appraisal = row.get("appraisal", {}) if isinstance(row.get("appraisal"), dict) else {}
            clean = {}
            for field, (allowed, default) in APPRAISAL_FIELDS.items():
                value = str(appraisal.get(field, default) or default).strip().lower()
                clean[field] = value if value in allowed else default
            result.append({"utt_key": key, "appraisal": clean, "evidence": str(row.get("evidence", ""))[:500]})
        return {"appraisals": result}


class UnconditionalAssignmentPolicy:
    """Policy Object: every expert is assigned to every utterance."""

    @staticmethod
    def assign(appraisal: dict[str, Any]) -> dict[str, Any]:
        return {"stage1_appraisal_gate": [
            {**row, "gate": {"modules": list(MODULES), "policy": "all_agents_equal_no_gate", "matched_criteria": []}}
            for row in appraisal.get("appraisals", [])
        ]}


class AgentOutputSanitizer:
    def __init__(self, normalize_label, labels: tuple[str, ...]):
        self.normalize_label = normalize_label
        self.labels = labels

    def sanitize(self, raw: dict[str, Any], keys: list[str], module: str) -> dict[str, Any]:
        source = raw.get("stage2", raw.get("stage2_selected_modules", [])) if isinstance(raw, dict) else []
        indexed = {str(row.get("utt_key", "")): row for row in source if isinstance(row, dict)}
        result = []
        for key in keys:
            selected = indexed.get(key, {}).get("selected_modules", {})
            payload = selected.get(module, {}) if isinstance(selected, dict) else {}
            clean = {}
            if isinstance(payload, dict):
                hint = str(payload.get("emotion_hint", "unclear") or "unclear").strip().lower()
                if hint != "unclear":
                    hint = self.normalize_label(hint, "unclear")
                strength = str(payload.get("evidence_strength", "low") or "low").lower()
                clean = {
                    "emotion_hint": hint if hint in {*self.labels, "unclear"} else "unclear",
                    "evidence_strength": strength if strength in {"none", "low", "medium", "high"} else "low",
                    "evidence": str(payload.get("evidence", ""))[:500],
                    "rationale": str(payload.get("rationale", ""))[:500],
                }
            result.append({"utt_key": key, "selected_modules": {module: clean} if clean else {}})
        return {"stage2": result}

