import json
from typing import Any, Protocol


class Model(Protocol):
    def invoke(self, prompt: str) -> Any: ...


def response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def parse_json_response(response: Any) -> dict[str, Any]:
    text = response_text(response).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Model returned invalid JSON; check that JSON response mode is enabled.") from error
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value
