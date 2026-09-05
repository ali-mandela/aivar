from __future__ import annotations

import os
import re


class MissingSecretError(Exception):
    """Raised when a required environment variable is not set."""

    pass


# Regex matching ${NAME} or ${NAME:-default} where NAME is [A-Z_][A-Z0-9_]*
PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def resolve_value(value: str | None) -> str | None:
    """
    Resolve placeholders in a value string.

    - None → None
    - No placeholders → returned unchanged
    - ${NAME} → os.environ["NAME"], raises MissingSecretError if unset
    - ${NAME:-default} → env value if set, else default
    - Replaces ALL occurrences in the string.
    """
    if value is None:
        return None

    def replace_placeholder(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)

        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value

        if default is not None:
            return default

        raise MissingSecretError(
            f"environment variable {name} is not set (required by a compiled test)"
        )

    return PLACEHOLDER_RE.sub(replace_placeholder, value)


def contains_placeholder(value: str | None) -> bool:
    """Check if a value string contains any placeholders."""
    if value is None:
        return False
    return bool(PLACEHOLDER_RE.search(value))


def redact(value: str | None) -> str:
    """
    Return '***' if the raw value contains a placeholder, else the value itself.
    Used for logging so resolved secrets are NEVER logged.
    """
    if value is None:
        return ""
    if contains_placeholder(value):
        return "***"
    return value
