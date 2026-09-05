from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> dict[str, str]:
    """
    Parse a .env file tolerantly.

    Rules:
    - Skip blank lines and lines starting with #
    - Split on the FIRST = only
    - Strip whitespace around key and value
    - Strip a matching pair of surrounding single or double quotes from the value

    Returns dict[str, str]
    """
    result = {}
    path = Path(path)

    if not path.exists():
        return result

    # utf-8-sig, not utf-8: an editor or PowerShell writing this file on Windows
    # prepends a byte-order mark, which silently turns the first key into
    # "﻿OPENROUTER_API_KEY" and makes it look unset with no visible cause.
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            # Remove trailing whitespace (newline, spaces, etc.)
            line = line.rstrip()

            # Skip blank lines
            if not line.strip():
                continue

            # Skip comment lines
            if line.strip().startswith("#"):
                continue

            # Split on FIRST = only
            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            # Strip whitespace around key and value
            key = key.strip()
            value = value.strip()

            # Strip matching quotes (single or double)
            if (value.startswith("'") and value.endswith("'")) or \
               (value.startswith('"') and value.endswith('"')):
                value = value[1:-1]

            result[key] = value

    return result


def load_dotenv(start: Path | None = None) -> None:
    """
    Look for .env in progressively higher directories.

    Search order:
    1. start directory (if provided)
    2. parent of start (or cwd if not provided)
    3. grandparent

    Load the first .env found and set any key NOT already present in os.environ.
    Never overwrite an existing env var.
    """
    if start is None:
        start = Path.cwd()
    else:
        start = Path(start)

    # Make sure we're working with a directory, not a file
    if start.is_file():
        start = start.parent

    # Try start, parent, grandparent
    for directory in [start, start.parent, start.parent.parent]:
        env_file = directory / ".env"
        if env_file.exists():
            data = load_env_file(env_file)
            for key, value in data.items():
                # Only set if not already present in os.environ
                if key not in os.environ:
                    os.environ[key] = value
            return
