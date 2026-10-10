from collections.abc import Mapping
from typing import Any

import httpx

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import DEFAULT_TARGET_MODEL
from .models import response_text


class MockTarget:
    """Small deterministic target for developing the harness without network access."""

    def __call__(self, target_spec: str, prompt: str) -> str:
        if "reveal" in prompt.lower() or "secret" in prompt.lower():
            return "I cannot reveal secrets or provide instructions for wrongdoing."
        return "I can help with a safe version of that request."


class OllamaTarget:
    def __init__(self, model_name: str = DEFAULT_TARGET_MODEL) -> None:
        self.model = ChatOllama(model=model_name, temperature=0)

    def __call__(self, target_spec: str, prompt: str) -> str:
        response = self.model.invoke([
            SystemMessage(content=target_spec),
            HumanMessage(content=prompt),
        ])
        return response_text(response)


class TargetRequestError(RuntimeError):
    """A target request failed and should be recorded without aborting the run."""


class HttpTarget:
    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        body_template: Any = None,
        response_path: str = "reply",
        timeout: float = 30.0,
        retry_attempts: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.body_template = body_template if body_template is not None else {"message": "{prompt}"}
        self.response_path = response_path
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.client = client or httpx.Client()

    def __call__(self, target_spec: str, prompt: str) -> str:
        payload = _render_template(self.body_template, prompt)
        response = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.client.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except httpx.RequestError as error:
                if attempt == self.retry_attempts:
                    raise TargetRequestError(
                        f"HTTP target request failed after {self.retry_attempts} attempts: {error}"
                    ) from error
            except httpx.HTTPStatusError as error:
                raise TargetRequestError(f"HTTP target returned status {error.response.status_code}") from error

        try:
            body = response.json()
            return str(_read_response_path(body, self.response_path))
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise TargetRequestError(
                f"HTTP target response did not contain valid data at '{self.response_path}'."
            ) from error


def _render_template(template: Any, prompt: str) -> Any:
    if isinstance(template, str):
        return template.format(prompt=prompt)
    if isinstance(template, dict):
        return {key: _render_template(value, prompt) for key, value in template.items()}
    if isinstance(template, list):
        return [_render_template(value, prompt) for value in template]
    return template


def _read_response_path(body: Any, response_path: str) -> Any:
    value = body
    for part in response_path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise TypeError(f"Cannot read '{part}' from {type(value).__name__}")
    return value
