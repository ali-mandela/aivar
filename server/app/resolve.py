from __future__ import annotations

from dataclasses import dataclass

from app.browser import Node
from app.models.models import Selector

# Generic words to strip from target and node names when matching
GENERIC_WORDS = {
    "the",
    "a",
    "an",
    "input",
    "field",
    "box",
    "button",
    "header",
    "page",
    "label",
    "text",
    "link",
    "icon",
}

# Minimum heuristic score to accept a match
MIN_HEURISTIC_SCORE = 0.6


def normalize(target: str) -> list[str]:
    """
    Normalize a target string for matching.
    - Lowercase
    - Split on non-alphanumerics
    - Drop GENERIC_WORDS and empty tokens
    - If everything is dropped, return the original lowercased tokens
    """
    # Lowercase and split on non-alphanumerics
    import re

    tokens = re.split(r"[^a-z0-9]+", target.lower())
    tokens = [t for t in tokens if t]  # Remove empty tokens

    # Drop generic words
    filtered = [t for t in tokens if t not in GENERIC_WORDS]

    # If everything was dropped, return original tokens
    if not filtered:
        return tokens

    return filtered


def selector_for(node: Node) -> Selector | None:
    """
    Build the most stable selector available, in preference order
    (research-backed ordering by locator stability):

    1. node.testid → Selector("testid", node.testid) if data-testid, else CSS for other attributes
    2. node.role and node.name → Selector("role", node.name, role=node.role)
    3. node.placeholder → Selector("placeholder", node.placeholder)
    4. node.name → Selector("text", node.name)
    5. otherwise None

    Note: Playwright's get_by_test_id() is hard-wired to look for data-testid only.
    For other test id attributes (data-test, data-test-id), we must use CSS selectors.
    """
    if node.testid:
        if node.testid_attr is None or node.testid_attr == "data-testid":
            # Use Playwright's native testid strategy (most readable)
            return Selector("testid", node.testid)
        else:
            # Use CSS selector for non-standard attributes
            return Selector("css", f'[{node.testid_attr}="{node.testid}"]')

    if node.role and node.name:
        return Selector("role", node.name, role=node.role)

    if node.placeholder:
        return Selector("placeholder", node.placeholder)

    if node.name:
        return Selector("text", node.name)

    return None


@dataclass(frozen=True)
class Candidate:
    """A scored element candidate for a target."""

    node: Node
    selector: Selector
    score: float
    why: str


def score_node(node: Node, target: str) -> tuple[float, str]:
    """
    Score how well a node matches a target.
    Returns (score, reason) where score is in 0.0..1.0.

    Scoring rules:
    - Exact case-insensitive match of normalized target → base 1.0
    - Otherwise base = (matched_tokens / total_target_tokens) * 0.8
    - +0.1 if node has a real role AND raw target contains word matching node.role
    - -0.3 if node has no role (roleless nodes are layout containers, almost never the intended target)
    - +0.15 if normalized target tokens ALL appear in normalized name tokens (visible name is stronger than testid)
    - -0.25 if node is a label (labels are not actionable for fill/click)
    - -0.4 if node not visible
    - Clamp to 0.0..1.0
    """
    target_tokens = normalize(target)
    if not target_tokens:
        return 0.0, "empty target"

    # Collect tokens from node name, placeholder, and testid
    node_text = f"{node.name} {node.placeholder or ''} {node.testid or ''}".lower()
    node_tokens = normalize(node_text)

    # Check for exact match
    target_normalized = normalize(target)
    if sorted(target_normalized) == sorted(node_tokens):
        base = 1.0
        why = "exact match"
    else:
        # Count matched tokens
        matched = sum(1 for t in target_tokens if t in node_tokens)
        base = (matched / len(target_tokens)) * 0.8
        why = f"{matched}/{len(target_tokens)} tokens matched"

    score = base

    # +0.15 if normalized target tokens ALL appear in normalized name tokens
    # (visible accessible name is stronger evidence than testid-only match)
    if node.name:
        name_tokens = normalize(node.name)
        if all(t in name_tokens for t in target_tokens):
            score += 0.15
            why += ", name match +0.15"

    # +0.1 for role hint: ONLY if node has a real role AND raw target contains word matching node.role
    if node.role:
        target_words = target.lower().split()
        if any(
            node.role in word or word in node.role for word in target_words
        ):
            score += 0.1
            why += ", role hint +0.1"
    else:
        # -0.3 if node has no role (roleless nodes are layout containers)
        score -= 0.3
        why += ", roleless -0.3"

    # -0.25 if label (labels are not actionable for fill/click)
    if node.role == "label":
        score -= 0.25
        why += ", label -0.25"

    # -0.4 if not visible
    if not node.visible:
        score -= 0.4
        why += ", invisible -0.4"

    # Clamp to 0.0..1.0
    score = max(0.0, min(1.0, score))

    return score, why


def shortlist(nodes: list[Node], target: str, limit: int = 5) -> list[Candidate]:
    """
    Score all nodes and return top candidates.
    - Compute selectors and scores
    - Drop score 0.0
    - Sort by score descending (stable)
    - Return top limit entries
    """
    candidates = []

    for node in nodes:
        sel = selector_for(node)
        if sel is None:
            continue

        score, why = score_node(node, target)
        if score == 0.0:
            continue

        candidates.append(
            Candidate(node=node, selector=sel, score=score, why=why)
        )

    # Sort by score descending (stable sort preserves order for equal scores)
    candidates.sort(key=lambda c: c.score, reverse=True)

    return candidates[:limit]


def best(nodes: list[Node], target: str) -> Candidate | None:
    """
    Return the top shortlist entry if its score >= MIN_HEURISTIC_SCORE, else None.
    """
    candidates = shortlist(nodes, target, limit=1)
    if candidates and candidates[0].score >= MIN_HEURISTIC_SCORE:
        return candidates[0]
    return None
