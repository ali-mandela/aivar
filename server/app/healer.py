from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.browser import Node
from app.config import Guardrails
from app.llm import LLMConfig, LLMError, LLMInvalidJSON, LLMResponse, chat_json, extract_json
from app.models.models import HealProposal, Selector, Step

logger = logging.getLogger("aivar")

HEAL_SYSTEM = """You are given a target description and a numbered list of candidate elements.
Your task is to choose which candidate best matches the described element.

Rules:
- Return ONLY a JSON object with fields: index, confidence, reasoning
- index: the number of the best candidate (must be one of the offered numbers)
- confidence: a decimal from 0.0 to 1.0 reflecting how confident you are
- reasoning: one short sentence explaining your choice
- Prefer interactive controls over layout containers
- If nothing is a good match, return the closest match with a LOW confidence
- No markdown, no prose, only the JSON object

Example:
{"index": 2, "confidence": 0.87, "reasoning": "Visible button labeled Login"}"""


@dataclass(frozen=True)
class RerankResult:
    """Result of reranking candidates."""

    index: int
    confidence: float
    reasoning: str


class HealRejected(Exception):
    """Raised when a heal proposal is rejected."""

    pass


def format_candidates(candidates: list) -> str:
    """
    Format candidates for the model.
    One line per candidate: [i] role=<role> name=<name> testid=<testid> placeholder=<placeholder> score=<0.00>
    Omit empty fields.
    """
    lines = []
    for i, candidate in enumerate(candidates):
        node = candidate.node
        parts = [f"[{i}]"]

        if node.role:
            parts.append(f"role={node.role}")
        if node.name:
            parts.append(f"name={node.name}")
        if node.testid:
            parts.append(f"testid={node.testid}")
        if node.placeholder:
            parts.append(f"placeholder={node.placeholder}")

        parts.append(f"score={candidate.score:.2f}")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def rerank(target: str, candidates: list, config: LLMConfig) -> tuple[RerankResult, LLMResponse]:
    """
    Call the model to rerank candidates.

    Returns (RerankResult, LLMResponse).
    Raises LLMError or LLMInvalidJSON on failure.
    """
    candidates_str = format_candidates(candidates)
    user_prompt = f"""Target: {target}

Candidates:
{candidates_str}

Choose the best match."""

    llm_response = chat_json(HEAL_SYSTEM, user_prompt, config)
    parsed = extract_json(llm_response.content)

    # Validate strictly
    if "index" not in parsed:
        raise LLMInvalidJSON("Missing 'index' in rerank response")

    index = int(parsed["index"])
    if not (0 <= index < len(candidates)):
        raise LLMInvalidJSON(
            f"index {index} out of range for {len(candidates)} candidates"
        )

    # Clamp confidence to [0.0, 1.0]
    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(parsed.get("reasoning", ""))

    return RerankResult(index=index, confidence=confidence, reasoning=reasoning), llm_response


def _testid_of(old: Selector) -> str | None:
    """The test id a selector is pinned to, whether native or expressed as CSS."""
    if old.strategy == "testid":
        return old.value
    if old.strategy == "css":
        m = re.search(r'\[.*?="(.+?)"\]', old.value)
        if m:
            return m.group(1)
    return None


def _agrees_with_target(target: str, node: Node) -> bool:
    """Does this node still match the human-written description of the step?"""
    from app.resolve import normalize

    wanted = set(normalize(target))
    if not wanted:
        return False
    have: set[str] = set()
    for field in (node.name, node.placeholder, node.testid):
        have |= set(normalize(field or ""))
    return bool(wanted & have)


def semantic_match(old: Selector | None, node: Node, target: str = "") -> tuple[bool, str]:
    """
    Does the healed element still MEAN the same thing as the one that was compiled?

    This is the gate that stops a visually plausible but semantically different
    element being accepted -- the "Pay now button was replaced by a similar
    looking button" failure mode. Agreement is established from the strongest
    evidence available, and a candidate with no ARIA role is refused outright
    because layout containers are never what a human meant to click or fill.
    """
    # A roleless node is a <div>/<span> wrapper. Never accept one as a healed
    # control, no matter how confident the model was.
    if not node.role:
        return (False, "candidate has no role (layout container, not a control)")

    if old is None:
        return (True, "no prior selector to disagree with")

    old_testid = _testid_of(old)
    if old_testid is not None:
        if node.testid == old_testid:
            return (True, f"test id agrees ({old_testid})")
        # The test id itself drifted. That is ordinary refactoring, not a
        # different element -- so fall back to the human-written target, which
        # is the one part of the step a developer's rename cannot invalidate.
        if _agrees_with_target(target, node):
            return (
                True,
                f"test id changed ({old_testid} -> {node.testid}) but the element "
                f"still matches the target wording",
            )
        return (
            False,
            f"test id changed ({old_testid} -> {node.testid}) and nothing else agrees",
        )

    if old.strategy == "role":
        match = node.role == old.role
        if match:
            return (True, f"role matches: {node.role}")
        else:
            return (False, f"role mismatch: was {old.role}, now {node.role}")

    if old.strategy in ("placeholder", "label", "text"):
        from app.resolve import normalize

        old_tokens = set(normalize(old.value))
        node_value = node.placeholder if old.strategy == "placeholder" else node.name
        node_tokens = set(normalize(node_value or ""))
        shared = old_tokens & node_tokens
        if shared:
            return (True, f"visible text agrees ({', '.join(sorted(shared))})")
        # The wording on the page changed. Same fallback as for test ids: the
        # step's target description is the stable statement of intent.
        if _agrees_with_target(target, node):
            return (True, "page wording changed but the element still matches the target")
        return (False, f"no textual agreement: was {sorted(old_tokens)}, now {sorted(node_tokens)}")

    return (False, f"unknown strategy: {old.strategy}")


def propose_heal(
    *,
    test_id: str,
    step: Step,
    candidates: list,
    config: LLMConfig,
    guardrails: Guardrails,
) -> tuple[HealProposal | None, str, LLMResponse | None]:
    """
    Propose a heal for a failed step.

    Returns (proposal_or_None, reason, llm_response_or_None).
    Gates in order:
    1. Assertions are never healed
    2. No candidates
    3. Rerank fails
    4. Confidence is below the cap
    5. Semantic match disagrees (if required)
    """
    # Gate 1: Assertions are never healed
    if not step.healable:
        return (None, "assertions are never healed", None)

    # Gate 2: No candidates
    if not candidates:
        return (None, "no candidates to choose from", None)

    # Gate 3: Try reranking
    try:
        result, llm_response = rerank(step.target, candidates, config)
    except (LLMError, LLMInvalidJSON) as e:
        return (None, f"rerank failed: {e}", None)

    chosen_candidate = candidates[result.index]
    chosen_node = chosen_candidate.node
    chosen_selector = chosen_candidate.selector

    # Gate 4: Confidence gate
    if result.confidence < guardrails.min_heal_confidence:
        reason = (
            f"confidence {result.confidence:.2f} below cap {guardrails.min_heal_confidence}"
        )
        return (None, reason, llm_response)

    # Gate 5: Semantic match gate
    semantic_ok = True
    semantic_reason = ""
    if guardrails.require_semantic_match:
        semantic_ok, semantic_reason = semantic_match(
            step.selector, chosen_node, step.target
        )
        if not semantic_ok:
            return (
                None,
                f"semantic mismatch: {semantic_reason}",
                llm_response,
            )

    # All gates passed: build proposal
    proposal = HealProposal(
        test_id=test_id,
        step_id=step.id,
        old=step.selector,
        new=chosen_selector,
        confidence=result.confidence,
        reasoning=result.reasoning,
        semantic_match=semantic_ok,
    )

    return (proposal, "accepted", llm_response)
