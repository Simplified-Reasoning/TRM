from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trm_dag.config import OpenAIConfig  # noqa: E402


@dataclass
class MockOpenAIClient:
    responses: list[list[str]]
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, config: OpenAIConfig, *, n: int) -> tuple[list[str], dict[str, int]]:
        self.prompts.append(prompt)
        idx = self.calls
        self.calls += 1
        if idx >= len(self.responses):
            batch = self.responses[-1] if self.responses else ["ok\n<|action|>continue"]
        else:
            batch = self.responses[idx]
        return batch[:n], {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }


def openai_config() -> OpenAIConfig:
    return OpenAIConfig(api_key="test", base_url=None, model="mock", n=1)
