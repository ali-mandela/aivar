from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from app.models.models import CompiledTest, HealProposal
from app.testfile import load_test, save_test

logger = logging.getLogger("aivar")


def proposal_id(p: HealProposal) -> str:
    """
    Generate a deterministic proposal ID.

    Uses first 12 chars of sha256 hash over test_id|step_id|new.strategy|new.value|new.role
    """
    data = f"{p.test_id}|{p.step_id}|{p.new.strategy}|{p.new.value}|{p.new.role or ''}"
    hash_obj = hashlib.sha256(data.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def save_proposal(p: HealProposal, out_dir: str | Path = "quarantine") -> Path:
    """
    Save a proposal to quarantine.

    Writes to <out_dir>/<test_id>__<step_id>__<proposal_id>.json
    Creates directories as needed.
    Overwriting the same file for an identical proposal is fine and expected.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pid = proposal_id(p)
    filename = f"{p.test_id}__{p.step_id}__{pid}.json"
    file_path = out_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(p.to_dict(), f, indent=2)

    logger.debug(f"Saved proposal to {file_path}")
    return file_path


def load_proposals(out_dir: str | Path = "quarantine") -> list[tuple[str, HealProposal]]:
    """
    Load all proposals from quarantine.

    Returns list of (proposal_id, HealProposal) tuples, sorted by id.
    Skips unparseable files without crashing.
    """
    out_path = Path(out_dir)
    if not out_path.exists():
        return []

    proposals: list[tuple[str, HealProposal]] = []

    for file_path in out_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            proposal = HealProposal(
                test_id=data["test_id"],
                step_id=data["step_id"],
                new=_load_selector(data["new"]),
                confidence=data["confidence"],
                reasoning=data["reasoning"],
                semantic_match=data["semantic_match"],
                old=_load_selector(data.get("old")) if data.get("old") else None,
            )
            pid = proposal_id(proposal)
            proposals.append((pid, proposal))
        except Exception as e:
            logger.warning(f"Skipping unparseable proposal file {file_path}: {e}")

    # Sort by id
    proposals.sort(key=lambda x: x[0])
    return proposals


def _load_selector(d: dict | None):
    """Helper to load a Selector from a dict."""
    if d is None:
        return None
    from app.models.models import Selector

    return Selector.from_dict(d)


def delete_proposal(pid: str, out_dir: str | Path = "quarantine") -> bool:
    """
    Delete a proposal by ID.

    Returns True if a file was deleted, False otherwise.
    """
    out_path = Path(out_dir)
    if not out_path.exists():
        return False

    # Find the file with this proposal id in the name
    for file_path in out_path.glob(f"*__{pid}.json"):
        try:
            file_path.unlink()
            logger.debug(f"Deleted proposal {pid}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")
            return False

    return False


def apply_proposal(p: HealProposal, test_path: str | Path) -> CompiledTest:
    """
    Apply an approved heal proposal to a compiled test.

    This is the ONLY function permitted to write a healed selector into a compiled test.
    It is never called from the executor.

    Raises KeyError if the step_id is not found.
    """
    test_path = Path(test_path)

    # Load the test
    test = load_test(test_path)

    # Find the step
    step_index = None
    for i, step in enumerate(test.steps):
        if step.id == p.step_id:
            step_index = i
            break

    if step_index is None:
        raise KeyError(f"Step with id {p.step_id} not found in test {p.test_id}")

    # Replace the selector and increment version
    old_step = test.steps[step_index]
    new_step = old_step.__class__(
        id=old_step.id,
        kind=old_step.kind,
        verb=old_step.verb,
        target=old_step.target,
        value=old_step.value,
        selector=p.new,
    )
    test.steps[step_index] = new_step
    test.version += 1

    # Save the test
    save_test(test, test_path)

    logger.info(f"Applied proposal to {test_path}: step {p.step_id} healed to {p.new}")
    return test
