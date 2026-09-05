"""Triage gate: determine whether a test failure is a bug or a script issue.

Getting this wrong in one direction ships defects. The bias is deliberately
asymmetric: a real regression masked as a script issue gets fixed by the agent
rather than reported; a false positive costs a repair cycle but not a bug.

The rules are deterministic first (fast, no model), with a single exception:
LOCATOR_NOT_FOUND is genuinely ambiguous and consults the model only in that case.
"""

from __future__ import annotations

from app.contracts.contracts import TriageResult, TriageVerdict
from app.llm import LLMConfig, LLMError, LLMInvalidJSON, LLMResponse, chat_json, extract_json
from app.models.models import FailureKind, Step, StepKind, StepResult


def triage_failure(
    step: Step,
    result: StepResult,
    *,
    config: LLMConfig | None = None,
    page_context: str | None = None,
) -> tuple[TriageResult, LLMResponse | None]:
    """Triage a single test failure.

    Returns (TriageResult, LLMResponse or None).
    LLMResponse is only non-None if the model was called (LOCATOR_NOT_FOUND with config).

    Deterministic rules run first and cover most cases. Only LOCATOR_NOT_FOUND
    with a config consults the model.
    """

    # Sanity check: do not triage successes
    if result.failure is None or result.status == "passed":
        raise ValueError(
            f"Cannot triage a passed or skipped result: step_id={result.step_id}, "
            f"status={result.status}, failure={result.failure}"
        )

    # Rule 1: ASSERTION_FAILED is always APP_DEFECT
    # An assertion failure is the finding itself; asking a model whether to repair it
    # is how a real regression gets masked.
    if result.failure is FailureKind.ASSERTION_FAILED:
        return (
            TriageResult(
                step_id=result.step_id,
                verdict=TriageVerdict.APP_DEFECT,
                confidence=1.0,
                reasoning="The application did not produce the state the test asserted. "
                "Assertion failures are findings, not locator issues.",
            ),
            None,
        )

    # Rule 2: AGENT_ERROR is harness/infrastructure, not the application
    if result.failure is FailureKind.AGENT_ERROR:
        return (
            TriageResult(
                step_id=result.step_id,
                verdict=TriageVerdict.FLAKY,
                confidence=1.0,
                reasoning="Agent error indicates harness or infrastructure failure, "
                "not an application defect.",
            ),
            None,
        )

    # Rule 3: ACTION_FAILED means the element was found but the action did not complete
    # This is usually timing or an overlay, so treat as flaky
    if result.failure is FailureKind.ACTION_FAILED:
        return (
            TriageResult(
                step_id=result.step_id,
                verdict=TriageVerdict.FLAKY,
                confidence=0.7,
                reasoning="The element was found but the action did not complete. "
                "This usually indicates timing or an overlay, not an application defect.",
            ),
            None,
        )

    # Rule 4: LOCATOR_NOT_FOUND is genuinely ambiguous
    # The element may have been renamed (script issue) or may have failed to render (app defect).
    if result.failure is FailureKind.LOCATOR_NOT_FOUND:
        if config is None:
            # No model available, fall back to SCRIPT_ISSUE
            return (
                TriageResult(
                    step_id=result.step_id,
                    verdict=TriageVerdict.SCRIPT_ISSUE,
                    confidence=0.5,
                    reasoning="Locator not found; could be a renamed selector (script issue) "
                    "or a missing feature (app defect). No model available for classification.",
                ),
                None,
            )

        # Model is available, consult it
        try:
            llm_response = _consult_model_for_locator(step, result, config, page_context)
        except LLMError:
            # Model call itself failed (network, auth, etc.), fall back
            return (
                TriageResult(
                    step_id=result.step_id,
                    verdict=TriageVerdict.SCRIPT_ISSUE,
                    confidence=0.5,
                    reasoning="Model failed during LLM call. Defaulting to script issue.",
                ),
                None,
            )

        # Parse the response and apply the hard invariant
        try:
            response_dict = extract_json(llm_response.content)
            verdict_str = response_dict.get("verdict", "").lower()
            confidence = response_dict.get("confidence", 0.5)
            reasoning = response_dict.get("reasoning", "")

            # Map string verdict to enum
            if verdict_str == "script_issue":
                verdict = TriageVerdict.SCRIPT_ISSUE
            elif verdict_str == "app_defect":
                verdict = TriageVerdict.APP_DEFECT
            elif verdict_str == "flaky":
                verdict = TriageVerdict.FLAKY
            else:
                # Invalid verdict, fall back
                verdict = TriageVerdict.SCRIPT_ISSUE
                reasoning = f"Invalid model verdict '{verdict_str}'. "

        except (LLMError, LLMInvalidJSON):
            # Model response was invalid JSON, fall back to SCRIPT_ISSUE
            verdict = TriageVerdict.SCRIPT_ISSUE
            confidence = 0.5
            reasoning = "Model returned invalid JSON. Defaulting to script issue."

        # Hard invariant: if this is an ASSERTION step, override any SCRIPT_ISSUE to APP_DEFECT
        if step.kind is StepKind.ASSERTION and verdict is TriageVerdict.SCRIPT_ISSUE:
            reasoning = f"Model returned script_issue, but assertions are never healable. "
            reasoning += "Overriding to app_defect. Original reasoning: " + reasoning
            verdict = TriageVerdict.APP_DEFECT

        return (
            TriageResult(
                step_id=result.step_id,
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
            ),
            llm_response,
        )

    # Shouldn't reach here if we've covered all FailureKind cases
    raise ValueError(f"Unknown failure kind: {result.failure}")


def _consult_model_for_locator(
    step: Step, result: StepResult, config: LLMConfig, page_context: str | None
) -> LLMResponse:
    """Call the model to disambiguate a LOCATOR_NOT_FOUND failure.

    Returns LLMResponse if successful.
    Raises LLMError or LLMInvalidJSON if the model fails.
    """
    system_prompt = (
        "You are a test diagnostician. Analyze a test failure and determine whether it is "
        "a script issue (the test references an element by an outdated selector, but similar "
        "elements are present), an application defect (the feature or element is entirely "
        "absent, indicating the app did not render it), or a flaky issue (timing, overlay, etc.).\n\n"
        "Respond ONLY with valid JSON: {\"verdict\": \"script_issue\" | \"app_defect\" | \"flaky\", "
        "\"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\"}"
    )

    page_snippet = page_context or "(no page context available)"

    user_prompt = (
        f"Step: {step.verb} {step.target}\n"
        f"Failure: {result.failure.value}\n"
        f"Error: {result.error}\n"
        f"Page context: {page_snippet}\n\n"
        f"Is this a script issue (selector drift), app defect (missing feature), or flaky (timing)?"
    )

    return chat_json(system_prompt, user_prompt, config)


def triage_run(
    steps_by_id: dict[str, Step],
    results: list[StepResult],
    *,
    config: LLMConfig | None = None,
    page_context: str | None = None,
) -> list[TriageResult]:
    """Triage all failed results in order.

    Skips passed and skipped results. Returns triages in result order.
    """
    triage_results = []

    for result in results:
        if result.status in ("passed", "skipped"):
            continue

        step = steps_by_id.get(result.step_id)
        if step is None:
            raise ValueError(f"Step {result.step_id} not found in steps_by_id")

        triage_result, _ = triage_failure(step, result, config=config, page_context=page_context)
        triage_results.append(triage_result)

    return triage_results


def summarize_triage(results: list[TriageResult]) -> dict[str, int]:
    """Count triage results by verdict.

    Returns a dict mapping verdict value (string) to count.
    """
    counts: dict[str, int] = {
        TriageVerdict.SCRIPT_ISSUE.value: 0,
        TriageVerdict.APP_DEFECT.value: 0,
        TriageVerdict.FLAKY.value: 0,
    }

    for result in results:
        counts[result.verdict.value] += 1

    return counts
