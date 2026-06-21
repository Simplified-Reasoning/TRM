from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import OpenAIConfig

Usage = dict[str, int]


class ChatClient(Protocol):
    def complete(self, prompt: str, config: OpenAIConfig, *, n: int) -> tuple[list[str], Usage]:
        """Return completion texts and token usage for one prompt."""


@dataclass
class OpenAIChatClient:
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff: float = 2.0

    def complete(self, prompt: str, config: OpenAIConfig, *, n: int) -> tuple[list[str], Usage]:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("openai>=1.0 is required to call a live model") from exc

        kwargs: dict[str, Any] = {"api_key": config.api_key, "timeout": config.timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        client = OpenAI(**kwargs)

        delay = self.initial_delay
        last_exc: Exception | None = None
        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    n=max(1, int(n)),
                )
                outputs: list[str] = []
                for choice in response.choices:
                    message = getattr(choice, "message", None)
                    content = getattr(message, "content", None)
                    if content:
                        outputs.append(str(content))
                usage_obj = getattr(response, "usage", None)
                usage = {
                    "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                }
                return outputs, usage
            except Exception as exc:  # pragma: no cover - live backend behavior
                last_exc = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(delay)
                delay *= self.backoff
        raise RuntimeError(f"OpenAI-compatible completion failed: {last_exc}") from last_exc


def add_usage(target: Usage, usage: Usage) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = int(target.get(key, 0)) + int(usage.get(key, 0))
    target["calls"] = int(target.get("calls", 0)) + int(usage.get("calls", 1) or 1)
