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


def repo_root() -> Path:
    """The repository root: the parent of the `server` directory."""
    return server_root().parent


def resolve_out_dir(path: str | Path) -> Path:
    """Resolve an output directory against the repository root.

    Everything a run produces -- reports, per-run test snapshots, job status,
    quarantined heal proposals -- lands outside the `server` directory, and the
    reason is not tidiness.

    `uvicorn --reload` watches its working directory for changes to Python
    files, and a run writes Python files: the generated suite, and a copy of it
    beside each report. With that output inside `server`, finishing a run
    restarted the server that was running it, killing the thread at the exact
    moment the pipeline first had something to show for itself. It looks like
    the agent crashing, and it only happens on the runs that were going well.

    Pointing the reload watcher at the source package fixes it too, but that is
    a flag someone has to remember on every start and forgetting it costs a
    whole run. Writing outside the watched tree cannot be forgotten.

    An absolute path is honoured as given, so a caller can still write somewhere
    specific on purpose. A relative one is anchored here rather than to the
    working directory, so the same run does not scatter its files to a different
    place depending on which shell started the server.
    """
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p)


def resolve_generated_dir(path: str | Path) -> Path:
    """Where the generated pytest files go.

    The same place as everything else a run writes, and named separately only
    because this is the one directory a person is told to run for themselves:
    `pytest tests/generated`, from the repository root.
    """
    return resolve_out_dir(path)
