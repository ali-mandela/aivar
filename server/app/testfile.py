from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from app.models.models import CompiledTest


def load_test(path: str | Path) -> CompiledTest:
    """Load a CompiledTest from a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CompiledTest.from_dict(data)


def save_test(test: CompiledTest, path: str | Path) -> None:
    """Save a CompiledTest to a JSON file with proper formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(test.to_dict(), f, indent=2)
        f.write("\n")
