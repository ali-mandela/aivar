from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings, loaded from server/.env then the process environment."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "aivar"
    environment: str = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class Guardrails:
    max_heals_per_run: int = 3
    min_heal_confidence: float = 0.5
    require_semantic_match: bool = True
    heal_assertions: bool = False
    max_cost_per_run_usd: float = 0.50
    action_retries: int = 2
    action_timeout_ms: int = 8000

    def __post_init__(self) -> None:
        if self.heal_assertions:
            raise ValueError(
                "heal_assertions must never be True: "
                "a failing assertion is a candidate bug, not a candidate repair"
            )
        if not (0.0 <= self.min_heal_confidence <= 1.0):
            raise ValueError(
                f"min_heal_confidence must be between 0.0 and 1.0, got {self.min_heal_confidence}"
            )
        if self.max_heals_per_run < 0:
            raise ValueError(
                f"max_heals_per_run must be non-negative, got {self.max_heals_per_run}"
            )

    @classmethod
    def from_env(cls) -> Guardrails:
        def parse_bool(s: str) -> bool:
            return s.lower() in ("1", "true", "yes")

        max_heals_per_run = int(
            os.getenv(
                "AIVAR_MAX_HEALS_PER_RUN",
                str(cls.__dataclass_fields__["max_heals_per_run"].default),
            )
        )
        min_heal_confidence = float(
            os.getenv(
                "AIVAR_MIN_HEAL_CONFIDENCE",
                str(cls.__dataclass_fields__["min_heal_confidence"].default),
            )
        )
        require_semantic_match = parse_bool(
            os.getenv(
                "AIVAR_REQUIRE_SEMANTIC_MATCH",
                str(cls.__dataclass_fields__["require_semantic_match"].default),
            )
        )
        max_cost_per_run_usd = float(
            os.getenv(
                "AIVAR_MAX_COST_PER_RUN_USD",
                str(cls.__dataclass_fields__["max_cost_per_run_usd"].default),
            )
        )
        action_retries = int(
            os.getenv(
                "AIVAR_ACTION_RETRIES",
                str(cls.__dataclass_fields__["action_retries"].default),
            )
        )
        action_timeout_ms = int(
            os.getenv(
                "AIVAR_ACTION_TIMEOUT_MS",
                str(cls.__dataclass_fields__["action_timeout_ms"].default),
            )
        )
        # heal_assertions must NOT be readable from the environment
        heal_assertions = False

        return cls(
            max_heals_per_run=max_heals_per_run,
            min_heal_confidence=min_heal_confidence,
            require_semantic_match=require_semantic_match,
            heal_assertions=heal_assertions,
            max_cost_per_run_usd=max_cost_per_run_usd,
            action_retries=action_retries,
            action_timeout_ms=action_timeout_ms,
        )


DEFAULTS = Guardrails()
