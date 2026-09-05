from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from app.browser import Node
from app.contracts.contracts import FlowKind, PlanMode, TestPlan, Flow
from app.llm import LLMConfig, LLMInvalidJSON, LLMResponse, chat_json, extract_json
from app.models.models import StepKind, Step, Verb

logger = logging.getLogger("aivar")


PLAN_SYSTEM = """You are a test automation expert. Your task is to generate a test plan as a JSON object with the following structure:

```json
{"steps": [{"kind": "action", "verb": "fill", "target": "username field", "value": "..."}]}
```

Rules you MUST follow:

1. `kind` must be either "action" (modifies the page) or "assertion" (verifies something is visible).
2. `verb` must be one of: "click", "fill", "wait_visible".
3. An assertion ALWAYS uses verb "wait_visible" and value null.
4. `target` is a 1–3 word description of the element, matching visible wording where possible.
5. NEVER invent credentials. For username, email, or password fields, set value to null — credentials are injected separately.
6. Every plan MUST end with at least one assertion.
7. Return ONLY valid JSON, no markdown, prose, or explanation.
"""


@dataclass(frozen=True)
class PlannedStep:
    """A planned step from the LLM."""

    kind: StepKind
    verb: Verb
    target: str
    value: str | None


class PlanValidationError(Exception):
    """Raised when a plan is invalid."""

    pass


def validate_plan(raw: dict) -> list[PlannedStep]:
    """
    Validate and normalize a plan from the LLM.

    Rules:
    - raw["steps"] must be a non-empty list
    - Each step must have: kind ("action" or "assertion"), verb ("click", "fill", "wait_visible"), target (non-empty string)
    - Normalise: if kind == "assertion", force verb = "wait_visible" and value = None
    - Reject plans with zero assertions (raise PlanValidationError with comment about self-healing)

    Returns list[PlannedStep] if valid.
    """
    if "steps" not in raw or not isinstance(raw["steps"], list) or len(raw["steps"]) == 0:
        raise PlanValidationError("steps must be a non-empty list")

    steps = []
    assertion_count = 0

    for i, step_dict in enumerate(raw["steps"]):
        # Validate kind
        kind_str = step_dict.get("kind")
        if kind_str not in ("action", "assertion"):
            raise PlanValidationError(
                f"Step {i}: kind must be 'action' or 'assertion', got '{kind_str}'"
            )

        # Validate verb
        verb = step_dict.get("verb")
        if verb not in ("click", "fill", "wait_visible"):
            raise PlanValidationError(
                f"Step {i}: verb must be one of 'click', 'fill', 'wait_visible', got '{verb}'"
            )

        # Validate target
        target = step_dict.get("target", "").strip()
        if not target:
            raise PlanValidationError(f"Step {i}: target must be a non-empty string")

        # Get value
        value = step_dict.get("value")

        # Normalise assertions
        if kind_str == "assertion":
            verb = "wait_visible"
            value = None
            assertion_count += 1

        kind = StepKind(kind_str)
        steps.append(
            PlannedStep(kind=kind, verb=verb, target=target, value=value)
        )

    # Reject plans with no assertions
    if assertion_count == 0:
        raise PlanValidationError(
            "a plan with no assertions is not a test "
            "(a suite that asserts nothing always passes and is the classic way self-healing automation hides real regressions)"
        )

    return steps


def plan_steps(
    intent: str,
    nodes: list[Node],
    config: LLMConfig,
    planner: Callable[[str, str, LLMConfig], LLMResponse] | None = None,
) -> tuple[list[PlannedStep], LLMResponse]:
    """
    Generate a plan for the given intent and page nodes.

    Builds a user message with intent and compact node snapshot.
    Calls the LLM (or injected planner callable) to get a plan.
    Validates and normalises the plan.

    On PlanValidationError or LLMInvalidJSON, retries ONCE with a corrective instruction.
    If still invalid, raises the error.

    Returns (list[PlannedStep], LLMResponse).
    """
    # Build compact node snapshot
    node_lines = []
    for node in nodes[:60]:  # Limit to 60 nodes
        parts = []
        if node.role:
            parts.append(f"role={node.role}")
        if node.name:
            # Truncate name to 60 chars
            name = node.name[:60]
            parts.append(f"name={name}")
        if node.placeholder:
            parts.append(f"placeholder={node.placeholder}")
        if node.testid:
            parts.append(f"testid={node.testid}")

        if parts:
            node_lines.append("- " + " ".join(parts))

    snapshot = "\n".join(node_lines)

    user_message = f"""Intent: {intent}

Page snapshot (up to 60 visible nodes):
{snapshot}

Generate a test plan to automate this intent."""

    # Use injected planner if provided; otherwise use chat_json
    if planner:
        response = planner(PLAN_SYSTEM, user_message, config)
    else:
        response = chat_json(PLAN_SYSTEM, user_message, config)

    # Extract and validate JSON
    raw_json = extract_json(response.content)
    steps = validate_plan(raw_json)

    return steps, response


def retry_plan_steps(
    intent: str,
    nodes: list[Node],
    config: LLMConfig,
    error: str,
    planner: Callable[[str, str, LLMConfig], LLMResponse] | None = None,
) -> tuple[list[PlannedStep], LLMResponse]:
    """
    Retry the plan with a corrective instruction.

    Used after a PlanValidationError or LLMInvalidJSON to inform the model of the problem.
    """
    corrected_system = PLAN_SYSTEM + f"\n\nPrevious attempt failed: {error}\nPlease correct this."

    # Build compact node snapshot (reuse logic)
    node_lines = []
    for node in nodes[:60]:
        parts = []
        if node.role:
            parts.append(f"role={node.role}")
        if node.name:
            name = node.name[:60]
            parts.append(f"name={name}")
        if node.placeholder:
            parts.append(f"placeholder={node.placeholder}")
        if node.testid:
            parts.append(f"testid={node.testid}")

        if parts:
            node_lines.append("- " + " ".join(parts))

    snapshot = "\n".join(node_lines)

    user_message = f"""Intent: {intent}

Page snapshot (up to 60 visible nodes):
{snapshot}

Generate a test plan to automate this intent."""

    if planner:
        response = planner(corrected_system, user_message, config)
    else:
        response = chat_json(corrected_system, user_message, config)

    raw_json = extract_json(response.content)
    steps = validate_plan(raw_json)

    return steps, response


def _validate_and_build_flows(
    raw_json: dict, max_flows: int
) -> tuple[list[Flow], bool]:
    """
    Validate and build Flow objects from LLM response.

    Returns (flows, is_happy_path_only).
    Raises PlanValidationError on validation failure.
    """
    if "flows" not in raw_json or not isinstance(raw_json["flows"], list):
        raise PlanValidationError("response must have 'flows' key with a list")

    flows_data = raw_json["flows"]
    if len(flows_data) == 0:
        raise PlanValidationError("flows list must not be empty")

    flows = []
    flow_kinds = set()

    for flow_idx, flow_dict in enumerate(flows_data):
        if flow_idx >= max_flows:
            break

        # Validate name
        name = flow_dict.get("name")
        if not isinstance(name, str) or not name:
            raise PlanValidationError(f"Flow {flow_idx}: name must be a non-empty string")

        # Validate kind
        kind_str = flow_dict.get("kind")
        try:
            kind = FlowKind(kind_str)
        except (ValueError, TypeError):
            raise PlanValidationError(
                f"Flow {flow_idx}: kind must be one of {[k.value for k in FlowKind]}, got '{kind_str}'"
            )
        flow_kinds.add(kind)

        # Get description
        description = flow_dict.get("description", "")

        # Validate and build steps
        steps_data = flow_dict.get("steps", [])
        if not isinstance(steps_data, list) or len(steps_data) == 0:
            raise PlanValidationError(f"Flow {flow_idx}: steps must be a non-empty list")

        steps = []
        assertion_count = 0

        for step_idx, step_dict in enumerate(steps_data):
            # Validate kind
            step_kind_str = step_dict.get("kind")
            if step_kind_str not in ("action", "assertion"):
                raise PlanValidationError(
                    f"Flow {flow_idx}, step {step_idx}: kind must be 'action' or 'assertion', got '{step_kind_str}'"
                )

            # Validate verb
            verb = step_dict.get("verb")
            if verb not in ("click", "fill", "wait_visible"):
                raise PlanValidationError(
                    f"Flow {flow_idx}, step {step_idx}: verb must be one of 'click', 'fill', 'wait_visible', got '{verb}'"
                )

            # Validate target
            target = step_dict.get("target", "").strip()
            if not target:
                raise PlanValidationError(
                    f"Flow {flow_idx}, step {step_idx}: target must be a non-empty string"
                )

            # Get value
            value = step_dict.get("value")

            # Normalise assertions
            if step_kind_str == "assertion":
                verb = "wait_visible"
                value = None
                assertion_count += 1

            step_kind = StepKind(step_kind_str)
            step_id = f"f{flow_idx + 1}s{step_idx + 1}"
            steps.append(
                Step(
                    id=step_id,
                    kind=step_kind,
                    verb=verb,
                    target=target,
                    value=value,
                )
            )

        # Reject flows with no assertions
        if assertion_count == 0:
            raise PlanValidationError(
                f"Flow {flow_idx}: flow has no assertions "
                "(a test that asserts nothing always passes and hides real regressions)"
            )

        # Build flow object
        flow_id = f"f{flow_idx + 1}"
        flows.append(
            Flow(
                id=flow_id,
                name=name,
                description=description,
                kind=kind,
                steps=steps,
            )
        )

    # Check if only happy paths
    is_happy_path_only = flow_kinds <= {FlowKind.HAPPY_PATH, FlowKind.NAVIGATION}

    return flows, is_happy_path_only


def plan_flows(
    digest: str,
    *,
    mode: PlanMode,
    intent: str | None = None,
    prd_text: str | None = None,
    config: LLMConfig,
    max_flows: int = 6,
    plan_id: str = "plan-1",
    extra_instruction: str | None = None,
) -> tuple[TestPlan, LLMResponse]:
    """
    Generate multiple test flows from an exploration digest.

    Args:
        digest: Compact text map from ExplorationReport.summarize()
        mode: Planning mode (SWEEP, FOCUSED, SPEC_LED)
        intent: Natural language intent for FOCUSED mode
        prd_text: Product requirements for SPEC_LED mode
        config: LLM configuration
        max_flows: Maximum flows to produce (default 6)
        plan_id: ID for the test plan (default "plan-1")
        extra_instruction: Additional instruction to append to user message
        planner: Optional callable for dependency injection in tests

    Returns:
        (TestPlan, LLMResponse)

    Raises:
        PlanValidationError: If the response is invalid or retry also fails
    """
    # Build system prompt
    system = f"""You are a test automation expert. Your task is to generate multiple test flows as a JSON object with the following structure:

```json
{{"flows":[{{"name":"...","description":"...","kind":"happy_path|negative|edge_case|error_state|navigation","steps":[{{"kind":"action|assertion","verb":"click|fill|wait_visible","target":"...","value":null}}]}}]}}
```

Rules you MUST follow:

1. `kind` must be either "action" (modifies the page) or "assertion" (verifies something is visible).
2. `verb` must be one of: "click", "fill", "wait_visible".
3. An assertion ALWAYS uses verb "wait_visible" and value null.
4. `target` is a 1–3 word description of the element, matching wording from the digest where possible.
5. NEVER invent credentials. For username, email, or password fields, set value to null — credentials are injected separately.
6. Every flow MUST end with at least one assertion.
7. Produce between 3 and {max_flows} flows.
8. Do not produce only happy paths. Include at least one `negative` or `error_state` flow (e.g. wrong password, empty required field, invalid input).
9. Return ONLY valid JSON, no markdown, prose, or explanation."""

    # Append mode-specific instructions
    if mode == PlanMode.SWEEP:
        system += (
            "\n\nCover the application broadly — every form, every distinct page, "
            "and the error states they imply."
        )
    elif mode == PlanMode.FOCUSED:
        system += (
            f"\n\nConcentrate on this intent: {intent}\n"
            "Still include at least one negative or edge-case flow within that scope."
        )
    elif mode == PlanMode.SPEC_LED:
        # Truncate PRD to 4000 chars
        prd_preview = (prd_text[:4000] if prd_text else "")
        system += (
            f"\n\nDerive flows from these requirements:\n{prd_preview}"
        )

    # Build user message
    user_message = f"""Digest of discovered pages and forms:

{digest}

Generate {max_flows} test flows to validate this application."""

    # Append extra instruction for re-planning if provided
    if extra_instruction:
        user_message += f"\n\n{extra_instruction}"

    # Make ONE LLM call
    response = chat_json(system, user_message, config)

    # Extract and validate JSON.
    #
    # A malformed or assertion-less plan is a recoverable drafting mistake, not
    # a reason to abandon the whole pipeline. Under re-plan pressure the model
    # sometimes drops assertions while chasing the coverage instruction, so we
    # repair once by telling it exactly what it got wrong before giving up.
    try:
        raw_json = extract_json(response.content)
        flows, is_happy_path_only = _validate_and_build_flows(raw_json, max_flows)
    except (PlanValidationError, LLMInvalidJSON) as first_error:
        logger.info("plan rejected (%s) - repairing once", first_error)
        repair = (
            f"Your previous response was rejected: {first_error}\n"
            "Fix exactly that problem and return the corrected JSON. "
            "Remember every flow must contain at least one assertion step "
            '(kind "assertion", verb "wait_visible", value null).'
        )
        response = chat_json(system, user_message + "\n\n" + repair, config)
        raw_json = extract_json(response.content)
        flows, is_happy_path_only = _validate_and_build_flows(raw_json, max_flows)

    # In SWEEP mode, retry once if only happy paths
    if mode == PlanMode.SWEEP and is_happy_path_only:
        retry_instruction = (
            "Previous attempt produced only happy-path and navigation flows. "
            "You MUST include at least one flow with kind 'negative' or 'error_state'. "
            "Examples: wrong password, empty required field, invalid format, etc."
        )
        retry_system = system + f"\n\n{retry_instruction}"
        response = chat_json(retry_system, user_message, config)
        raw_json = extract_json(response.content)
        flows, is_happy_path_only = _validate_and_build_flows(raw_json, max_flows)

        # If still happy-path only, raise error
        if is_happy_path_only:
            raise PlanValidationError(
                "SWEEP mode requires at least one negative or error_state flow. "
                "Retried once but response still contained only happy paths."
            )

    # Build TestPlan
    plan = TestPlan(
        id=plan_id,
        mode=mode,
        flows=flows,
        intent=intent,
        prd_path=None,  # Not tracking PRD path in this context
    )

    return plan, response
