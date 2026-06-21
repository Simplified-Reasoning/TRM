from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str | None
    model: str
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512
    n: int = 1
    timeout: float | None = None

    @classmethod
    def from_env(cls, **overrides: object) -> OpenAIConfig:
        """Build config from DAG_CONSTRUCTION_* variables, with explicit overrides."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        def pick(name: str, default: object = None) -> object:
            return overrides.get(name, os.getenv(f"DAG_CONSTRUCTION_{name.upper()}", default))

        def as_float(value: object, default: float) -> float:
            return float(default if value in (None, "") else str(value))

        def as_int(value: object, default: int) -> int:
            return int(default if value in (None, "") else str(value))

        api_key = str(pick("api_key", "") or "")
        model = str(pick("model", "") or "")
        if not api_key:
            raise RuntimeError("Missing DAG_CONSTRUCTION_API_KEY")
        if not model:
            raise RuntimeError("Missing DAG_CONSTRUCTION_MODEL")

        timeout_raw = pick("timeout", None)
        timeout = None if timeout_raw in (None, "") else as_float(timeout_raw, 0.0)
        base_url_raw = pick("base_url", None)
        base_url = None if base_url_raw in (None, "") else str(base_url_raw)
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=as_float(pick("temperature", 0.3), 0.3),
            top_p=as_float(pick("top_p", 0.9), 0.9),
            max_tokens=as_int(pick("max_tokens", 512), 512),
            n=as_int(pick("n", 1), 1),
            timeout=timeout,
        )


@dataclass(frozen=True)
class DagParams:
    regen_limit: int = 5
    main_path_cap: int = 8
    other_leaf_cap: int = 5
    random_seed: int | None = None
