from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """Configuration for the browser target."""

    url: str
    name: str = "default"
    viewport_width: int = 1280
    viewport_height: int = 720
    headless: bool = True

    @classmethod
    def from_env(cls, url: str | None = None) -> Target:
        """
        Create a Target from environment variables.

        Environment variables:
        - AIVAR_TARGET_URL: the target URL (overridden by url argument)
        - AIVAR_TARGET_NAME: target name (default: "default")
        - AIVAR_VIEWPORT_WIDTH: viewport width (default: 1280)
        - AIVAR_VIEWPORT_HEIGHT: viewport height (default: 720)
        - AIVAR_HEADLESS: "0", "false", "no" (case-insensitive) → False; otherwise True (default: True)

        The url argument wins over the env var; raise ValueError if neither is present.
        """
        if url is None:
            url = os.environ.get("AIVAR_TARGET_URL")

        if url is None:
            raise ValueError(
                "url must be provided or AIVAR_TARGET_URL environment variable must be set"
            )

        name = os.environ.get("AIVAR_TARGET_NAME", "default")

        try:
            viewport_width = int(os.environ.get("AIVAR_VIEWPORT_WIDTH", "1280"))
        except ValueError:
            viewport_width = 1280

        try:
            viewport_height = int(os.environ.get("AIVAR_VIEWPORT_HEIGHT", "720"))
        except ValueError:
            viewport_height = 720

        headless_str = os.environ.get("AIVAR_HEADLESS", "true").lower()
        headless = headless_str not in ("0", "false", "no")

        return cls(
            url=url,
            name=name,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            headless=headless,
        )
