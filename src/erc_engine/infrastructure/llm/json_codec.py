import json
import re
from typing import Any


class DefensiveJsonCodec:
    """Adapter around occasionally fenced model output."""

    @staticmethod
    def decode(text: str) -> dict[str, Any]:
        value = re.sub(r"^```(?:json)?", "", str(text).strip()).strip()
        value = re.sub(r"```$", "", value).strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", value, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

