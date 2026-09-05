"""Where the pipeline writes its output.

Output directories are configured as relative paths ("artifacts",
"tests/generated"), which Python resolves against the CURRENT WORKING
DIRECTORY. For a server that is whatever directory uvicorn happened to be
started from, so the same run scatters its files to a different place each
time and nothing -- not even the report -- records which.

Anchoring relative paths to the server root makes output location a property
of the project rather than of the shell.
"""

from __future__ import annotations

from pathlib import Path


def server_root() -> Path:
    """The `server` directory: the parent of the `app` package."""
    return Path(__file__).resolve().parent.parent


def resolve_out_dir(path: str | Path) -> Path:
    """Resolve an output directory against the server root.

    An absolute path is honoured as given, so a caller can still write
    somewhere specific on purpose. A relative one is anchored to the server
    root rather than the working directory.
    """
    p = Path(path)
    return p if p.is_absolute() else (server_root() / p)
