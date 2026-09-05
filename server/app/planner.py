from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from app.browser import Node
from app.contracts.contracts import FlowKind, PlanMode, TestPlan, Flow
from app.llm import LLMConfig, LLMInvalidJSON, LLMResponse, chat_json, extract_json
from app.models.models import (
    ASSERTION_VERBS,
    VALUE_ASSERTION_VERBS,
    StepKind,
    Step,
    Verb,
)

logger = logging.getLogger("aivar")

# Kept in step with the Verb literal in models.py: every verb the browser,
# codegen and executor know how to carry out.
VALID_VERBS: tuple[str, ...] = (
    "click",
    "fill",
    "select",
    "check",
    "press",
    "wait_visible",
    "assert_text",
    "assert_url",
    "assert_hidden",
)

# Names a model reaches for that mean one of ours.
#
# "assert_visible" for "wait_visible" is the obvious one, and rejecting it
# escalated whole runs -- the retry asked again and got the same sensible
# synonym back, so a plan that was correct in every respect that matters was
# thrown away over one word. The set of verbs is ours to name; insisting the
# model guess our spelling is a test of vocabulary, not of planning.
VERB_ALIASES: dict[str, str] = {
    # visibility
    "assert_visible": "wait_visible",
    "expect_visible": "wait_visible",
    "is_visible": "wait_visible",
    "visible": "wait_visible",
    "see": "wait_visible",
    "wait_for": "wait_visible",
    # absence
    "assert_not_visible": "assert_hidden",
    "assert_invisible": "assert_hidden",
    "expect_hidden": "assert_hidden",
    "not_visible": "assert_hidden",
    # text
    "assert_contains_text": "assert_text",
    "assert_text_contains": "assert_text",
    "expect_text": "assert_text",
    "contains_text": "assert_text",
    "has_text": "assert_text",
    # address
    "assert_path": "assert_url",
    "expect_url": "assert_url",
    "assert_navigation": "assert_url",
    "url": "assert_url",
    # actions
    "type": "fill",
    "input": "fill",
    "enter": "fill",
    "set": "fill",
    "select_option": "select",
    "choose": "select",
    "tick": "check",
    "checkbox": "check",
    "keypress": "press",
    "key": "press",
    "press_key": "press",
    "tap": "click",
    "submit": "click",
}


def normalize_verb(verb: object) -> object:
    """Map a synonym onto the verb it plainly means, leaving anything else alone.

    Returned unchanged when it is not a string or not a known alias, so the
    validator still rejects a verb that really has no equivalent rather than
    guessing at one.
    """
    if not isinstance(verb, str):
        return verb
    lowered = verb.strip().lower()
    if lowered in VALID_VERBS:
        return lowered
    return VERB_ALIASES.get(lowered, verb)


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
        verb = normalize_verb(step_dict.get("verb"))
        if verb not in VALID_VERBS:
            raise PlanValidationError(
                f"Step {i}: verb must be one of {', '.join(VALID_VERBS)}, got '{verb}'"
            )

        # Validate target
        target = step_dict.get("target", "").strip()
        if not target:
            raise PlanValidationError(f"Step {i}: target must be a non-empty string")

        # Get value
        value = step_dict.get("value")

        # Normalise assertions. See the note in _validate_and_build_flows: only
        # a verb that is not an assertion verb is corrected, so a model that
        # names what it expects keeps saying it.
        if kind_str == "assertion":
            if verb not in ASSERTION_VERBS:
                verb = "wait_visible"
                value = None
            elif verb not in VALUE_ASSERTION_VERBS:
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
) -> tuple[list[Flow], bool, bool]:
    """
    Validate and build Flow objects from LLM response.

    Returns (flows, is_happy_path_only, has_value_assertion).
    Raises PlanValidationError on validation failure.
    """
    if "flows" not in raw_json or not isinstance(raw_json["flows"], list):
        raise PlanValidationError("response must have 'flows' key with a list")

    flows_data = raw_json["flows"]
    if len(flows_data) == 0:
        raise PlanValidationError("flows list must not be empty")

    flows = []
    # Why each dropped flow was dropped, so a plan that lost one says so.
    rejected: list[str] = []
    flow_kinds = set()

    for flow_idx, flow_dict in enumerate(flows_data):
        # One malformed flow should not discard the ones beside it.
        #
        # A single unrecognised verb in flow 2 used to raise straight out
        # of here, escalating the run and throwing away three flows that
        # were completely fine. The model writes four independent flows;
        # they should fail independently too.
        try:
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
                verb = normalize_verb(step_dict.get("verb"))
                if verb not in VALID_VERBS:
                    raise PlanValidationError(
                        f"Flow {flow_idx}, step {step_idx}: verb must be one of "
                        f"{', '.join(VALID_VERBS)}, got '{verb}'"
                    )

                # Validate target
                target = step_dict.get("target", "").strip()
                if not target:
                    raise PlanValidationError(
                        f"Flow {flow_idx}, step {step_idx}: target must be a non-empty string"
                    )

                # Get value
                value = step_dict.get("value")

                # Normalise assertions.
                #
                # This used to force every assertion to wait_visible and drop its
                # value, so a model answering with assert_url "/dashboard" had that
                # rewritten to "is anything visible" before anyone saw it -- the
                # plan looked like the model's own timidity when it was ours.
                # Now only a verb that is not an assertion verb gets corrected.
                if step_kind_str == "assertion":
                    if verb not in ASSERTION_VERBS:
                        verb = "wait_visible"
                        value = None
                    elif verb not in VALUE_ASSERTION_VERBS:
                        # wait_visible and assert_hidden locate an element and
                        # nothing more; a value on them means nothing.
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

            # Where this flow should start. Without it every flow opens at the entry
            # URL and has to click its way to the screen under test, which is both
            # longer and more fragile -- and it is what stops flows from being run
            # independently of one another.
            entry_url = flow_dict.get("entry_url")
            if not isinstance(entry_url, str) or not entry_url.startswith(("http://", "https://")):
                entry_url = None

            flows.append(
                Flow(
                    id=flow_id,
                    name=name,
                    description=description,
                    kind=kind,
                    steps=steps,
                    entry_url=entry_url,
                )
            )
        except PlanValidationError as flow_error:
            rejected.append(f"flow {flow_idx}: {flow_error}")
            logger.info("plan: dropping flow %d - %s", flow_idx, flow_error)
            continue

    if not flows:
        raise PlanValidationError(
            "no flow in the plan was usable: " + "; ".join(rejected)
        )
    if rejected:
        logger.info("plan: kept %d flow(s), dropped %d", len(flows), len(rejected))

    # Check if only happy paths
    is_happy_path_only = flow_kinds <= {FlowKind.HAPPY_PATH, FlowKind.NAVIGATION}

    # Whether the plan ever states what the app should say or where it should
    # land, as opposed to only checking that something appeared. A suite whose
    # every assertion is wait_visible passes against an app showing the wrong
    # page entirely, so this is the difference between a plan that would catch a
    # regression and one that would not.
    has_value_assertion = any(
        s.verb in ("assert_text", "assert_url")
        for f in flows
        for s in f.steps
    )

    return flows, is_happy_path_only, has_value_assertion


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
    system = f"""You are a senior QA engineer writing end-to-end tests by hand, the way a careful person would after clicking through the app themselves.

Return a JSON object with this structure:

```json
{{"flows":[{{"name":"...","description":"...","entry_url":"...","kind":"happy_path|negative|edge_case|error_state|navigation","steps":[{{"kind":"action|assertion","verb":"...","target":"...","value":null}}]}}]}}
```

Verbs available:

ACTIONS
- "click"  - press a button or link
- "fill"   - type into a text field ("value" is the text)
- "select" - choose an option in a dropdown ("value" is the visible option label)
- "check"  - tick a checkbox or radio
- "press"  - send a key ("value" is "Enter", "Tab" or "Escape")

ASSERTIONS
- "wait_visible"  - the element is on screen (value null)
- "assert_text"   - the element contains this text ("value" is the expected text)
- "assert_url"    - the address contains this fragment ("value" is e.g. "/dashboard"); "target" describes it in words
- "assert_hidden" - the element is gone (value null)

Rules you MUST follow:

1. `kind` is "action" (changes the page) or "assertion" (checks the page).
2. `target` is a 1-3 word description of the element, reusing wording from the digest where possible.
3. NEVER invent credentials. For username, email, or password fields set value to null - credentials are injected separately.
4. Every flow MUST end with at least one assertion.
5. Produce between 3 and {max_flows} flows.
6. Do not produce only happy paths. Include at least one `negative` or `error_state` flow (wrong password, empty required field, invalid format).
7. Return ONLY valid JSON, no markdown, prose, or explanation.

Assertions - the part that decides whether this suite is worth anything:

"wait_visible" only proves something appeared. An app that renders "Invalid
credentials" where it should render a dashboard passes it. Use these instead:

8. A flow whose last action navigates MUST end with "assert_url", value being
   the path the user should land on (e.g. "/dashboard").
9. A flow expecting a rejection or a validation failure MUST use "assert_text",
   value being the message the user should see.
10. A flow where something should disappear MUST use "assert_hidden".
11. Use "wait_visible" only when appearing really is the whole point, and never
    as the only assertion in a flow.

Worked example - copy this shape:

```json
{{"flows":[
 {{"name":"Sign in with valid credentials","description":"A registered user reaches their dashboard","entry_url":"https://example.com/login","kind":"happy_path",
  "steps":[
   {{"kind":"action","verb":"fill","target":"email field","value":null}},
   {{"kind":"action","verb":"fill","target":"password field","value":null}},
   {{"kind":"action","verb":"click","target":"Sign In","value":null}},
   {{"kind":"assertion","verb":"assert_url","target":"dashboard address","value":"/dashboard"}},
   {{"kind":"assertion","verb":"assert_hidden","target":"password field","value":null}}
  ]}},
 {{"name":"Reject a wrong password","description":"A bad password is refused with a message","entry_url":"https://example.com/login","kind":"negative",
  "steps":[
   {{"kind":"action","verb":"fill","target":"email field","value":null}},
   {{"kind":"action","verb":"fill","target":"password field","value":null}},
   {{"kind":"action","verb":"click","target":"Sign In","value":null}},
   {{"kind":"assertion","verb":"assert_text","target":"error message","value":"Invalid"}}
  ]}}
]}}
```

Where a flow starts:

Set `entry_url` to the URL from the digest where the flow should begin, so it
starts on the right screen instead of clicking its way there from the home
page. Use a URL that appears in the digest. Omit it only when the flow really
does start at the entry page.

Screens with no URL:

A line reading "Opens in-page (dialog/panel, no URL change): X" means control X
opens a dialog, drawer or panel without navigating. Reach it by clicking X, and
assert on what it reveals. These are often the least-tested parts of an app.

Test data:

Use realistic values a person would actually type: "Alex Morgan",
"alex@example.com", "+44 7700 900123". For edge_case flows use awkward but
plausible input on purpose - a very long name, a leading space, an address with
no @, an empty required field."""

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
        flows, is_happy_path_only, has_value_assertion = _validate_and_build_flows(
            raw_json, max_flows
        )
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
        flows, is_happy_path_only, has_value_assertion = _validate_and_build_flows(
            raw_json, max_flows
        )

    # Retry once if the plan is all happy paths.
    #
    # This used to apply to SWEEP alone, so a FOCUSED or SPEC_LED run could
    # return nothing but happy paths unchallenged. "not just happy paths" is a
    # requirement of the brief in every mode, not a property of sweeping.
    if is_happy_path_only:
        retry_instruction = (
            "Previous attempt produced only happy-path and navigation flows. "
            "You MUST include at least one flow with kind 'negative' or 'error_state'. "
            "Examples: wrong password, empty required field, invalid format, etc."
        )
        retry_system = system + f"\n\n{retry_instruction}"
        response = chat_json(retry_system, user_message, config)
        raw_json = extract_json(response.content)
        flows, is_happy_path_only, has_value_assertion = _validate_and_build_flows(
            raw_json, max_flows
        )

        # If still happy-path only, raise error
        if is_happy_path_only:
            raise PlanValidationError(
                "A plan requires at least one negative or error_state flow. "
                "Retried once but response still contained only happy paths."
            )

    # Retry once if nothing in the plan checks a value.
    #
    # Told to prefer assert_text and assert_url, a smaller model still reaches
    # for wait_visible every time -- it satisfies the letter of "end with an
    # assertion" while proving almost nothing. Asking again with the specific
    # complaint is cheaper than accepting a suite that passes against a broken
    # app, and mirrors how the happy-path gate above works: state the rule in
    # the prompt, then enforce it in Python.
    if not has_value_assertion:
        logger.info("plan has no value assertion - asking once more")
        sharpen = (
            "Your previous plan checked only that elements were visible. That "
            "passes even when the application shows the wrong page or the wrong "
            "message. Revise it so that at least half the flows end in an "
            "'assert_url' (naming the path the user should land on) or an "
            "'assert_text' (naming the message the user should see). Keep the "
            "same flows and the same coverage; only make the assertions state "
            "what should actually be true."
        )
        try:
            response = chat_json(system, user_message + "\n\n" + sharpen, config)
            raw_json = extract_json(response.content)
            retry_flows, retry_happy_only, retry_has_value = _validate_and_build_flows(
                raw_json, max_flows
            )
            # Only take the retry if it is genuinely better. A second attempt
            # that regressed to happy paths, or still asserts nothing, is worse
            # than the plan already in hand.
            if retry_has_value and not retry_happy_only:
                flows = retry_flows
        except (PlanValidationError, LLMInvalidJSON) as e:
            # The first plan was valid. Keep it rather than fail the run.
            logger.info("assertion-sharpening retry failed (%s) - keeping first plan", e)

    # Build TestPlan
    plan = TestPlan(
        id=plan_id,
        mode=mode,
        flows=flows,
        intent=intent,
        prd_path=None,  # Not tracking PRD path in this context
    )

    return plan, response
