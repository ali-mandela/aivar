from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.envfile import load_dotenv

logger = logging.getLogger("aivar")


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimited(LLMError):
    """Raised when the API returns 429 (rate limited)."""
    pass


class LLMInvalidJSON(LLMError):
    """Raised when the LLM response cannot be parsed as JSON."""
    pass


DEFAULT_MODELS = ("minimax/minimax-m3:free", "nvidia/nemotron-3-super-120b-a12b:free")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM calls."""

    api_key: str
    models: tuple[str, ...] = DEFAULT_MODELS
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    # A multi-flow plan is the largest thing we ask for: several flows, each
    # with named steps. 1200 truncated a 4-flow plan mid-JSON, which surfaces
    # as an unparseable response rather than an obviously short one.
    max_tokens: int = 4000
    timeout_s: int = 120
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> LLMConfig:
        """
        Load LLM config from environment.

        Calls load_dotenv() first, then reads:
        - OPENROUTER_API_KEY (required)
        - AIVAR_LLM_MODELS (optional, comma-separated)

        Raises LLMError if OPENROUTER_API_KEY is absent.
        """
        load_dotenv()

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. "
                "Set it in the environment or in a .env file at the project root."
            )

        models_str = os.environ.get("AIVAR_LLM_MODELS")
        if models_str:
            models = tuple(m.strip() for m in models_str.split(",") if m.strip())
        else:
            models = DEFAULT_MODELS

        return cls(
            api_key=api_key,
            models=models,
        )


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


def extract_json(text: str) -> dict:
    """
    Extract JSON from text robustly.

    Attempts in order:
    1. Parse as bare JSON
    2. Strip markdown fences (```json ... ``` or ``` ... ```)
    3. Extract substring from first { to last }

    Raises LLMInvalidJSON if all fail.
    """
    text = text.strip()

    # Attempt 1: bare JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown fences
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Attempt 3: extract from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # All attempts failed
    preview = text[:200]
    raise LLMInvalidJSON(f"Could not parse JSON from LLM response: {preview}")


def chat_json(system: str, user: str, config: LLMConfig) -> LLMResponse:
    """
    Call the LLM for a JSON response.

    Iterates through config.models in order. For each model:
    - Retries up to config.max_retries on HTTP 429 or 5xx with exponential backoff
    - On 4xx that is not 429 (e.g., 403, 404), moves to the next model immediately
    - On success, returns LLMResponse
    - If every model fails, raises LLMError

    Uses exponential backoff: 0.8 * 2**attempt seconds, capped at 10s.
    """
    import os

    start_time = time.perf_counter()
    errors_per_model = {}

    for model in config.models:
        logger.info(f"Trying model: {model}")

        for attempt in range(config.max_retries):
            try:
                url = f"{config.base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "response_format": {"type": "json_object"},
                }

                request = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )

                with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
                    data = json.loads(response.read().decode("utf-8"))

                latency_ms = (time.perf_counter() - start_time) * 1000
                content = data["choices"][0]["message"]["content"]
                prompt_tokens = data["usage"]["prompt_tokens"]
                completion_tokens = data["usage"]["completion_tokens"]
                cost_usd = data["usage"].get("cost", 0.0)

                logger.info(
                    f"Model {model}: latency={latency_ms:.0f}ms, "
                    f"tokens={prompt_tokens + completion_tokens} ({prompt_tokens}+{completion_tokens}), "
                    f"cost=${cost_usd:.6f}"
                )

                return LLMResponse(
                    content=content,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                )

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # Rate limited; retry with backoff
                    wait_time = min(10.0, 0.8 * (2 ** attempt))
                    logger.warning(
                        f"Model {model}: rate limited (429), retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                elif e.code >= 500:
                    # Server error; retry with backoff
                    wait_time = min(10.0, 0.8 * (2 ** attempt))
                    logger.warning(
                        f"Model {model}: server error ({e.code}), retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                elif e.code in (403, 404):
                    # Client error that's not rate limit; move to next model
                    error_msg = f"HTTP {e.code}: {e.reason}"
                    errors_per_model[model] = error_msg
                    logger.warning(f"Model {model}: {error_msg}, trying next model")
                    break
                else:
                    # Other 4xx error; move to next model
                    error_msg = f"HTTP {e.code}: {e.reason}"
                    errors_per_model[model] = error_msg
                    logger.warning(f"Model {model}: {error_msg}, trying next model")
                    break

            except Exception as e:
                error_msg = str(e)
                errors_per_model[model] = error_msg
                if attempt < config.max_retries - 1:
                    wait_time = min(10.0, 0.8 * (2 ** attempt))
                    logger.warning(
                        f"Model {model}: error on attempt {attempt + 1}: {error_msg}, "
                        f"retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Model {model}: failed after {config.max_retries} attempts: {error_msg}"
                    )
                    break

    # All models failed
    error_summary = "; ".join(
        [f"{model}: {error}" for model, error in errors_per_model.items()]
    )
    raise LLMError(f"All models failed: {error_summary}")
