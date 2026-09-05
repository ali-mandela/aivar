"""LLM access across multiple providers.

Three providers are supported: OpenRouter, Google (Gemini) and Sarvam AI.
OpenRouter and Sarvam speak the OpenAI chat-completions dialect and differ only
in base URL and key; Google's generateContent API has a different request and
response shape and gets its own adapter.

Callers do not see any of this. `chat_json(system, user, config)` is the whole
surface, exactly as before -- the provider is a property of the LLMConfig it is
handed.

Failover is a chain. Within a provider the configured models are tried in order;
when a provider is exhausted the next configured provider takes over. Free tiers
rate-limit constantly, so a run should not die because one vendor said 429.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum

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


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    GOOGLE = "google"
    SARVAM = "sarvam"


@dataclass(frozen=True)
class ProviderSpec:
    """Everything provider-specific, in one place.

    `style` selects the wire adapter: "openai" for the chat-completions dialect,
    "gemini" for Google's generateContent.
    """

    provider: Provider
    style: str
    base_url: str
    key_vars: tuple[str, ...]
    models: tuple[str, ...]

    def base_url_from_env(self) -> str:
        return os.environ.get(f"AIVAR_{self.provider.value.upper()}_BASE_URL", self.base_url)

    def models_from_env(self) -> tuple[str, ...]:
        raw = os.environ.get(f"AIVAR_{self.provider.value.upper()}_MODELS")
        if raw:
            picked = tuple(m.strip() for m in raw.split(",") if m.strip())
            if picked:
                return picked
        return self.models

    def key_from_env(self) -> str | None:
        for var in self.key_vars:
            value = os.environ.get(var)
            if value:
                return value
        return None


# Order matters: it is the preference order when no provider is named, and the
# order fallbacks are appended in.
PROVIDER_SPECS: dict[Provider, ProviderSpec] = {
    Provider.OPENROUTER: ProviderSpec(
        provider=Provider.OPENROUTER,
        style="openai",
        base_url="https://openrouter.ai/api/v1",
        key_vars=("OPENROUTER_API_KEY",),
        models=("minimax/minimax-m3:free", "nvidia/nemotron-3-super-120b-a12b:free"),
    ),
    Provider.GOOGLE: ProviderSpec(
        provider=Provider.GOOGLE,
        style="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        # GEMINI_API_KEY is what Google's own docs use; accept both.
        key_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        models=("gemini-flash-latest",),
    ),
    Provider.SARVAM: ProviderSpec(
        provider=Provider.SARVAM,
        style="openai",
        base_url="https://api.sarvam.ai/v1",
        key_vars=("SARVAM_API_KEY",),
        # sarvam-105b is the flagship (128K context); -conversations is tuned for
        # real-time voice/chat at 32K and is the wrong shape for plan generation.
        # Sarvam also accepts an `api-subscription-key` header, but documents
        # Authorization: Bearer for OpenAI-compatible tooling, which is what the
        # shared openai adapter already sends.
        models=("sarvam-105b",),
    ),
}

# Retained for backwards compatibility: the OpenRouter defaults.
DEFAULT_MODELS = PROVIDER_SPECS[Provider.OPENROUTER].models


def parse_provider(value: str | None) -> Provider | None:
    """Parse a provider name leniently. Returns None for blank input."""
    if not value:
        return None
    name = value.strip().lower()
    aliases = {
        "openrouter": Provider.OPENROUTER,
        "or": Provider.OPENROUTER,
        "google": Provider.GOOGLE,
        "gemini": Provider.GOOGLE,
        "sarvam": Provider.SARVAM,
        "sarvamai": Provider.SARVAM,
    }
    if name not in aliases:
        raise LLMError(
            f"Unknown LLM provider {value!r}. "
            f"Expected one of: {', '.join(p.value for p in Provider)}"
        )
    return aliases[name]


@dataclass(frozen=True)
class Endpoint:
    """One provider, resolved: a key, a base URL and the models to try on it."""

    provider: Provider
    api_key: str
    base_url: str
    models: tuple[str, ...]

    @property
    def style(self) -> str:
        return PROVIDER_SPECS[self.provider].style

    def describe(self) -> dict:
        """Safe to log or return over HTTP: never includes the key."""
        return {"provider": self.provider.value, "models": list(self.models)}


def _configured_endpoints() -> dict[Provider, Endpoint]:
    """Every provider that has a key present, in preference order."""
    found: dict[Provider, Endpoint] = {}
    for provider, spec in PROVIDER_SPECS.items():
        key = spec.key_from_env()
        if key:
            found[provider] = Endpoint(
                provider=provider,
                api_key=key,
                base_url=spec.base_url_from_env(),
                models=spec.models_from_env(),
            )
    return found


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM calls.

    `api_key`, `models` and `base_url` describe the primary provider and keep
    their original meaning. `fallbacks` are tried, in order, once the primary
    provider's models are exhausted.
    """

    api_key: str
    models: tuple[str, ...] = DEFAULT_MODELS
    base_url: str = "https://openrouter.ai/api/v1"
    provider: Provider = Provider.OPENROUTER
    fallbacks: tuple[Endpoint, ...] = ()
    temperature: float = 0.0
    # A multi-flow plan is the largest thing we ask for: several flows, each
    # with named steps. 1200 truncated a 4-flow plan mid-JSON, which surfaces
    # as an unparseable response rather than an obviously short one.
    max_tokens: int = 4000
    timeout_s: int = 120
    max_retries: int = 3

    @property
    def primary(self) -> Endpoint:
        return Endpoint(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            models=tuple(self.models),
        )

    @property
    def chain(self) -> tuple[Endpoint, ...]:
        """The full failover order: primary first, then each fallback."""
        return (self.primary,) + tuple(self.fallbacks)

    def describe(self) -> dict:
        """A key-free summary of what this run will call."""
        return {
            "provider": self.provider.value,
            "models": list(self.models),
            "fallbacks": [e.describe() for e in self.fallbacks],
        }

    @classmethod
    def from_env(
        cls,
        provider: str | Provider | None = None,
        models: tuple[str, ...] | list[str] | None = None,
    ) -> LLMConfig:
        """Load LLM config from the environment.

        Calls load_dotenv() first, then reads:

        - OPENROUTER_API_KEY / GOOGLE_API_KEY (or GEMINI_API_KEY) / SARVAM_API_KEY
        - AIVAR_LLM_PROVIDER    which one to use (default: first configured)
        - AIVAR_LLM_MODELS      models for the chosen provider, comma-separated
        - AIVAR_<PROVIDER>_MODELS / AIVAR_<PROVIDER>_BASE_URL   per-provider
        - AIVAR_LLM_FALLBACK    set to 0/false to disable cross-provider failover

        The `provider` and `models` arguments win over the environment, so a
        single run can pick a provider without changing the server's config.

        Raises LLMError if no provider is configured, or if a provider was
        explicitly named but has no key.
        """
        load_dotenv()

        available = _configured_endpoints()
        if not available:
            raise LLMError(
                "No LLM provider is configured. Set one of OPENROUTER_API_KEY, "
                "GOOGLE_API_KEY (or GEMINI_API_KEY), or SARVAM_API_KEY in the "
                "environment or in server/.env."
            )

        requested = provider if isinstance(provider, Provider) else parse_provider(provider)
        if requested is None:
            requested = parse_provider(os.environ.get("AIVAR_LLM_PROVIDER"))

        if requested is None:
            # Nobody asked for one: take the first configured, in preference order.
            chosen = next(iter(available.values()))
        elif requested in available:
            chosen = available[requested]
        else:
            spec = PROVIDER_SPECS[requested]
            raise LLMError(
                f"LLM provider {requested.value!r} was requested but "
                f"{' / '.join(spec.key_vars)} is not set. "
                f"Configured providers: {', '.join(p.value for p in available)}."
            )

        # An explicit model list applies to the chosen provider only.
        if models:
            picked = tuple(m.strip() for m in models if m and m.strip())
            if picked:
                chosen = Endpoint(
                    provider=chosen.provider,
                    api_key=chosen.api_key,
                    base_url=chosen.base_url,
                    models=picked,
                )
        elif os.environ.get("AIVAR_LLM_MODELS"):
            picked = tuple(
                m.strip() for m in os.environ["AIVAR_LLM_MODELS"].split(",") if m.strip()
            )
            if picked:
                chosen = Endpoint(
                    provider=chosen.provider,
                    api_key=chosen.api_key,
                    base_url=chosen.base_url,
                    models=picked,
                )

        fallback_enabled = os.environ.get("AIVAR_LLM_FALLBACK", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        fallbacks = (
            tuple(e for p, e in available.items() if p is not chosen.provider)
            if fallback_enabled
            else ()
        )

        return cls(
            api_key=chosen.api_key,
            models=chosen.models,
            base_url=chosen.base_url,
            provider=chosen.provider,
            fallbacks=fallbacks,
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
    provider: str = Provider.OPENROUTER.value


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


# ---------------------------------------------------------------------------
# Wire adapters
#
# Each returns (content, prompt_tokens, completion_tokens, cost_usd). Only
# OpenRouter reports a real cost; the others bill out of band, so cost stays
# 0.0 and the pipeline's spend budget is effectively unbounded on them. Wall
# clock (max_pipeline_seconds) remains the binding limit there.
# ---------------------------------------------------------------------------


def _post_json(url: str, headers: dict, body: dict, timeout_s: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_openai_style(
    endpoint: Endpoint, model: str, system: str, user: str, config: LLMConfig
) -> tuple[str, int, int, float]:
    """OpenRouter and Sarvam: the OpenAI chat-completions dialect."""
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
    if endpoint.provider is Provider.SARVAM:
        body["reasoning_effort"] = None
    data = _post_json(
        f"{endpoint.base_url}/chat/completions",
        {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json",
        },
        body,
        config.timeout_s,
    )

    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"{endpoint.provider.value}: response contained no choices")
    content = choices[0].get("message", {}).get("content") or ""

    usage = data.get("usage") or {}
    return (
        content,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("cost", 0.0),
    )


def _call_gemini_style(
    endpoint: Endpoint, model: str, system: str, user: str, config: LLMConfig
) -> tuple[str, int, int, float]:
    """Google generateContent.

    The system prompt is a separate `systemInstruction` rather than a message,
    and JSON mode is `responseMimeType` rather than `response_format`.
    """
    data = _post_json(
        f"{endpoint.base_url}/models/{model}:generateContent",
        {
            "X-goog-api-key": endpoint.api_key,
            "Content-Type": "application/json",
        },
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
                "responseMimeType": "application/json",
            },
        },
        config.timeout_s,
    )

    candidates = data.get("candidates") or []
    if not candidates:
        # A prompt blocked by safety filters returns no candidates at all.
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        raise LLMError(
            f"google: no candidates returned"
            + (f" (blocked: {blocked})" if blocked else "")
        )

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    content = "".join(p.get("text", "") for p in parts)

    if not content:
        raise LLMError(
            f"google: empty response (finishReason={candidate.get('finishReason')})"
        )

    usage = data.get("usageMetadata") or {}
    return (
        content,
        usage.get("promptTokenCount", 0),
        usage.get("candidatesTokenCount", 0),
        0.0,
    )


_ADAPTERS = {
    "openai": _call_openai_style,
    "gemini": _call_gemini_style,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def chat_json(system: str, user: str, config: LLMConfig) -> LLMResponse:
    """
    Call the LLM for a JSON response.

    Walks config.chain: the primary provider first, then each fallback. Within
    a provider, models are tried in order. For each model:

    - Retries up to config.max_retries on HTTP 429 or 5xx with exponential backoff
    - On 4xx that is not 429 (e.g., 403, 404), moves to the next model immediately
    - On success, returns LLMResponse
    - If every model on every provider fails, raises LLMError

    Uses exponential backoff: 0.8 * 2**attempt seconds, capped at 10s.
    """
    start_time = time.perf_counter()
    errors: dict[str, str] = {}

    for endpoint in config.chain:
        adapter = _ADAPTERS[endpoint.style]

        for model in endpoint.models:
            label = f"{endpoint.provider.value}/{model}"
            logger.info(f"Trying model: {label}")

            for attempt in range(config.max_retries):
                try:
                    content, prompt_tokens, completion_tokens, cost_usd = adapter(
                        endpoint, model, system, user, config
                    )

                    latency_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        f"Model {label}: latency={latency_ms:.0f}ms, "
                        f"tokens={prompt_tokens + completion_tokens} "
                        f"({prompt_tokens}+{completion_tokens}), "
                        f"cost=${cost_usd:.6f}"
                    )

                    return LLMResponse(
                        content=content,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost_usd,
                        latency_ms=latency_ms,
                        provider=endpoint.provider.value,
                    )

                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        # Rate limited; retry with backoff
                        wait_time = min(10.0, 0.8 * (2 ** attempt))
                        logger.warning(
                            f"Model {label}: rate limited (429), retrying in {wait_time:.1f}s..."
                        )
                        errors[label] = "HTTP 429: rate limited"
                        time.sleep(wait_time)
                        continue
                    elif e.code >= 500:
                        # Server error; retry with backoff
                        wait_time = min(10.0, 0.8 * (2 ** attempt))
                        logger.warning(
                            f"Model {label}: server error ({e.code}), retrying in {wait_time:.1f}s..."
                        )
                        errors[label] = f"HTTP {e.code}: {e.reason}"
                        time.sleep(wait_time)
                        continue
                    else:
                        # Any other 4xx; move to the next model
                        error_msg = f"HTTP {e.code}: {e.reason}"
                        errors[label] = error_msg
                        logger.warning(f"Model {label}: {error_msg}, trying next model")
                        break

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    errors[label] = error_msg
                    if attempt < config.max_retries - 1:
                        wait_time = min(10.0, 0.8 * (2 ** attempt))
                        logger.warning(
                            f"Model {label}: error on attempt {attempt + 1}: {error_msg}, "
                            f"retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Model {label}: failed after {config.max_retries} attempts: {error_msg}"
                        )
                        break

    error_summary = "; ".join(f"{label}: {error}" for label, error in errors.items())
    raise LLMError(f"All models failed: {error_summary}")
