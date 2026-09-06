from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable
from typing import Any

from openai import OpenAI

from erc_engine.domain.models import GenerationUsage, RuntimeSettings

from .json_codec import DefensiveJsonCodec


class OpenAIJsonGateway:
    """Gateway + retry decorator + thread-local Unit of Work for usage telemetry."""

    def __init__(self, settings: RuntimeSettings, system_prompt: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY before running inference.")
        self.settings = settings
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=key)
        self._local = threading.local()

    def reset_usage(self) -> None:
        self._local.entries = []

    def usage(self) -> list[GenerationUsage]:
        return list(getattr(self._local, "entries", []))

    def absorb_usage(self, entries: Iterable[GenerationUsage]) -> None:
        if not hasattr(self._local, "entries"):
            self.reset_usage()
        self._local.entries.extend(entries)

    def complete(self, prompt: str, step: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = self.client.responses.create(
                    **self._response_options(),
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                self._record(step, response, attempt)
                return DefensiveJsonCodec.decode(response.output_text)
            except Exception as exc:
                last_error = exc
                time.sleep(2 * attempt)
        raise RuntimeError(f"JSON generation failed for {step}: {last_error}")

    def _response_options(self) -> dict[str, Any]:
        values: dict[str, Any] = {"model": self.settings.model, "text": {"format": {"type": "json_object"}}}
        if self.settings.model.startswith(("o", "gpt-5")) and self.settings.reasoning_effort:
            values["reasoning"] = {"effort": self.settings.reasoning_effort}
        else:
            values["temperature"] = 0.0
        return values

    def _record(self, step: str, response: Any, attempt: int) -> None:
        usage = getattr(response, "usage", None)
        get = lambda *names: next((getattr(usage, name, 0) or 0 for name in names if hasattr(usage, name)), 0)
        input_tokens = int(get("input_tokens", "prompt_tokens"))
        output_tokens = int(get("output_tokens", "completion_tokens"))
        total = int(get("total_tokens") or input_tokens + output_tokens)
        self.absorb_usage([GenerationUsage(step, self.settings.model, attempt, input_tokens, output_tokens, total)])

