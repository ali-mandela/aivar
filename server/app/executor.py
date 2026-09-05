from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError

from app.browser import Browser, SelectorConfigError, build_locator
from app.config import DEFAULTS, Guardrails
from app.llm import LLMConfig
from app.models.models import (
    CompiledTest,
    FailureKind,
    RunResult,
    Selector,
    Source,
    Step,
    StepKind,
    StepResult,
)
from app.resolve import best, shortlist
from app.secrets import MissingSecretError, redact, resolve_value
from app.target import Target

logger = logging.getLogger("aivar")

# How long to probe a compiled (tier 0) selector before deciding it has drifted.
# Deliberately shorter than action_timeout_ms: a stale selector is a normal event
# on a changed page, and we do not want to pay the full timeout twice before the
# cascade gets its turn.
TIER0_PROBE_TIMEOUT_MS = 2000

# Re-export for backwards compatibility
__all__ = ["SelectorConfigError", "build_locator", "run_test"]


def run_test(
    test: CompiledTest,
    *,
    guardrails: Guardrails = DEFAULTS,
    target: Target | None = None,
    headless: bool | None = None,
    url_override: str | None = None,
    llm_config: LLMConfig | None = None,
    heal: bool = False,
    quarantine_dir: str | Path = "quarantine",
) -> RunResult:
    """
    Execute a CompiledTest deterministically.

    Resolution order for URL: url_override > target.url > test.url
    Resolution order for headless: explicit headless arg > target.headless > True
    Viewport is applied from target if provided.

    If browser launch or navigation fails, all steps are recorded as failed with agent_error.
    Once any step fails, remaining steps are skipped.
    ACTION steps with selector=None fall back to Tier 1 (heuristic resolution).
    ASSERTION steps can only fail with ASSERTION_FAILED or AGENT_ERROR, never LOCATOR_NOT_FOUND.
    """
    # Resolve URL: url_override > target.url > test.url
    url = url_override or (target.url if target else None) or test.url

    # Resolve headless: explicit arg > target.headless > True
    if headless is None:
        headless = target.headless if target else True

    # Get viewport settings from target
    viewport_width = target.viewport_width if target else 1280
    viewport_height = target.viewport_height if target else 720

    results: list[StepResult] = []
    playwright = None
    browser = None
    page = None
    browser_wrapper = None
    heals_used = 0
    cost_usd = 0.0
    heal_proposals: list = []

    try:
        # Launch browser
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=headless)
            page = browser.new_page(
                viewport={"width": viewport_width, "height": viewport_height}
            )
            browser_wrapper = Browser(page)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to launch browser or create page: {error_msg}")
            # Mark all steps as failed with AGENT_ERROR
            for step in test.steps:
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        source=Source.NONE,
                        duration_ms=0.0,
                        failure=FailureKind.AGENT_ERROR,
                        error=error_msg,
                    )
                )
            return RunResult.from_results(test.id, results)

        # Navigate to URL
        url = url_override or test.url
        try:
            page.goto(url)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to navigate to {url}: {error_msg}")
            # Mark all steps as failed with AGENT_ERROR
            for step in test.steps:
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        source=Source.NONE,
                        duration_ms=0.0,
                        failure=FailureKind.AGENT_ERROR,
                        error=error_msg,
                    )
                )
            return RunResult.from_results(test.id, results)

        # Walk through steps
        any_failed = False
        for step_index, step in enumerate(test.steps):
            start_time = time.perf_counter()

            # If a previous step failed, skip this one
            if any_failed:
                duration_ms = (time.perf_counter() - start_time) * 1000
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="skipped",
                        source=Source.NONE,
                        duration_ms=duration_ms,
                        failure=None,
                    )
                )
                logger.info(
                    f"Step {step_index} ({step.id}): skipped (previous failure)"
                )
                continue

            # assert_url checks the address, not an element. It has no selector
            # to resolve and nothing to wait for attachment on, so it runs here
            # rather than falling through the locator machinery -- which would
            # otherwise report "target could not be resolved" for a step that
            # never had a target.
            if step.verb == "assert_url":
                try:
                    browser_wrapper.act(None, "assert_url", step.value, guardrails.action_timeout_ms)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    results.append(
                        StepResult(
                            step_id=step.id,
                            status="passed",
                            source=Source.NONE,
                            duration_ms=duration_ms,
                            failure=None,
                        )
                    )
                    logger.info(f"Step {step_index} ({step.id}): passed (assert_url)")
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    results.append(
                        StepResult(
                            step_id=step.id,
                            status="failed",
                            source=Source.NONE,
                            duration_ms=duration_ms,
                            failure=FailureKind.ASSERTION_FAILED,
                            error=str(e),
                        )
                    )
                    logger.info(f"Step {step_index} ({step.id}): failed (assert_url) - {e}")
                    any_failed = True
                continue

            # Resolve secrets in step.value
            resolved_value = None
            try:
                resolved_value = resolve_value(step.value)
            except MissingSecretError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        source=Source.NONE,
                        duration_ms=duration_ms,
                        failure=FailureKind.AGENT_ERROR,
                        error=str(e),
                    )
                )
                logger.info(
                    f"Step {step_index} ({step.id}): failed (agent_error) - {str(e)}"
                )
                any_failed = True
                continue

            # TIER 0: Try with compiled selector
            selector_to_use = step.selector
            source = Source.CACHE

            # TIER 0 PROBE. A compiled selector that no longer resolves is exactly
            # what selector drift looks like, and it is the case self-healing exists
            # for. ACTION steps must therefore fall through to Tier 1/2 so the
            # cascade can repair it -- without this probe, healing is unreachable
            # because `selector_to_use` would stay set and skip both lower tiers.
            #
            # ASSERTION steps deliberately do NOT fall through. A missing element
            # that an assertion named is the finding itself, not a lookup problem to
            # route around; it is left to fail as ASSERTION_FAILED below.
            if selector_to_use is not None and step.kind == StepKind.ACTION:
                try:
                    browser_wrapper.wait_attached(
                        selector_to_use, min(guardrails.action_timeout_ms, TIER0_PROBE_TIMEOUT_MS)
                    )
                except PlaywrightTimeoutError:
                    logger.info(
                        f"Step {step_index} ({step.id}): tier 0 selector is stale "
                        f"({selector_to_use.strategy}={selector_to_use.value!r}) - falling through"
                    )
                    selector_to_use = None
                    source = Source.NONE
                except SelectorConfigError:
                    # A malformed selector is a configuration bug, not drift. Leave it
                    # in place so the main wait_attached below classifies it AGENT_ERROR.
                    pass

            # TIER 1: For ACTION steps only, if no selector or Tier 0 misses, try heuristic
            tier1_candidates = None
            if selector_to_use is None and step.kind == StepKind.ACTION:
                # Try heuristic resolution
                try:
                    nodes = browser_wrapper.snapshot()
                    candidate = best(nodes, step.target)
                    if candidate:
                        selector_to_use = candidate.selector
                        source = Source.HEURISTIC
                        logger.info(
                            f"Step {step_index} ({step.id}): heuristic resolved to {candidate.why}"
                        )
                    else:
                        # Tier 1 failed but shortlist exists; save for potential healing
                        tier1_candidates = shortlist(nodes, step.target, limit=5)
                except Exception as e:
                    # Snapshot or resolution failed; continue with None selector
                    logger.warning(
                        f"Step {step_index} ({step.id}): heuristic resolution failed: {e}"
                    )

            # TIER 2: Healing (only if Tier 1 completely failed but candidates exist)
            heal_error = None
            if (
                selector_to_use is None
                and step.kind == StepKind.ACTION
                and tier1_candidates
                and heal
                and llm_config is not None
            ):
                # Try to heal
                if heals_used >= guardrails.max_heals_per_run:
                    heal_error = f"heal cap reached ({guardrails.max_heals_per_run} per run)"
                    logger.info(f"Step {step_index} ({step.id}): healing blocked - {heal_error}")
                else:
                    try:
                        from app.healer import propose_heal
                        from app.quarantine import save_proposal

                        proposal, reason, llm_response = propose_heal(
                            test_id=test.id,
                            step=step,
                            candidates=tier1_candidates,
                            config=llm_config,
                            guardrails=guardrails,
                        )

                        if proposal:
                            # Accept and use the healed selector for this run
                            selector_to_use = proposal.new
                            source = Source.HEALED
                            save_proposal(proposal, quarantine_dir)
                            heal_proposals.append(proposal)
                            heals_used += 1
                            if llm_response:
                                cost_usd += llm_response.cost_usd
                            logger.info(
                                f"Step {step_index} ({step.id}): healed with confidence {proposal.confidence:.2f}"
                            )
                        else:
                            # Proposal was rejected
                            heal_error = reason
                            logger.info(
                                f"Step {step_index} ({step.id}): healing rejected - {reason}"
                            )
                    except Exception as e:
                        heal_error = f"healing failed: {e}"
                        logger.warning(
                            f"Step {step_index} ({step.id}): {heal_error}"
                        )

            # If we still don't have a selector at this point:
            # - ACTION steps fail with LOCATOR_NOT_FOUND
            # - ASSERTION steps fail with ASSERTION_FAILED
            if selector_to_use is None:
                duration_ms = (time.perf_counter() - start_time) * 1000
                if step.kind == StepKind.ACTION:
                    failure_kind = FailureKind.LOCATOR_NOT_FOUND
                else:
                    # An assertion whose target was never resolved is the agent
                    # failing to build a check, not the application failing one.
                    #
                    # This used to be reported as ASSERTION_FAILED, which triage
                    # maps to APP_DEFECT with full confidence -- so a plan naming
                    # an element that does not exist was published as a defect in
                    # somebody's application. Two of three "defects" on a stable
                    # demo site were this, and a tool that cries wolf is not
                    # trusted the day it is right.
                    #
                    # AGENT_ERROR maps to FLAKY: reported honestly, and still
                    # never healed. The anti-masking rule is about refusing to
                    # repair assertions, which is untouched -- a genuine
                    # assertion failure, where the element resolved and the check
                    # did not hold, is still ASSERTION_FAILED below.
                    failure_kind = FailureKind.AGENT_ERROR
                error_text = heal_error or "Step target could not be resolved"
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        source=source,
                        duration_ms=duration_ms,
                        failure=failure_kind,
                        error=error_text,
                    )
                )
                logger.info(
                    f"Step {step_index} ({step.id}): failed ({failure_kind.value}) - {error_text}"
                )
                any_failed = True
                continue

            # Try to wait for element attachment.
            #
            # Skipped for assert_hidden: that step passes precisely when the
            # element is absent, so waiting for it to attach fails every time it
            # is right. Playwright's own wait_for(state="hidden") inside act()
            # already treats "never attached" as hidden.
            try:
                if step.verb != "assert_hidden":
                    browser_wrapper.wait_attached(selector_to_use, guardrails.action_timeout_ms)
            except PlaywrightTimeoutError:
                duration_ms = (time.perf_counter() - start_time) * 1000
                if step.kind == StepKind.ACTION:
                    failure_kind = FailureKind.LOCATOR_NOT_FOUND
                else:
                    # CRITICAL: Assertion steps that fail to find element = assertion_failed, not locator_not_found
                    # This is the key rule preventing real bugs from being auto-healed later.
                    failure_kind = FailureKind.ASSERTION_FAILED
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        source=source,
                        duration_ms=duration_ms,
                        failure=failure_kind,
                        error=f"Element not found (timeout after {guardrails.action_timeout_ms}ms)",
                    )
                )
                logger.info(
                    f"Step {step_index} ({step.id}): failed ({failure_kind.value}) - element not found"
                )
                any_failed = True
                continue
            except SelectorConfigError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        source=source,
                        duration_ms=duration_ms,
                        failure=FailureKind.AGENT_ERROR,
                        error=str(e),
                    )
                )
                logger.info(
                    f"Step {step_index} ({step.id}): failed (agent_error) - {str(e)}"
                )
                any_failed = True
                continue

            # Phase B: Act
            success = False
            last_error: str | None = None

            # For ACTION steps, retry; for ASSERTION steps, one attempt only
            max_attempts = (
                guardrails.action_retries + 1 if step.kind == StepKind.ACTION else 1
            )

            for attempt in range(max_attempts):
                try:
                    browser_wrapper.act(
                        selector_to_use,
                        step.verb,
                        resolved_value,
                        guardrails.action_timeout_ms,
                    )
                    success = True
                    break
                except SelectorConfigError as e:
                    # Config error is immediate, no retry
                    last_error = str(e)
                    break
                except PlaywrightTimeoutError as e:
                    last_error = str(e)
                    if attempt < max_attempts - 1:
                        # Retry with backoff for ACTION steps
                        time.sleep(0.25 * (attempt + 1))
                    # Continue to next attempt or fall through to error handling
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_attempts - 1:
                        # Retry with backoff for ACTION steps
                        time.sleep(0.25 * (attempt + 1))
                    # Continue to next attempt or fall through to error handling

            duration_ms = (time.perf_counter() - start_time) * 1000

            if success:
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="passed",
                        source=source,
                        duration_ms=duration_ms,
                        failure=None,
                    )
                )
                logger.info(
                    f"Step {step_index} ({step.id}): passed ({step.verb}) in {duration_ms:.0f}ms, value was {redact(step.value)}"
                )
            else:
                # Determine failure kind based on step kind and error type
                if step.kind == StepKind.ACTION:
                    if isinstance(last_error, type) or "SelectorConfigError" in str(
                        last_error
                    ):
                        failure_kind = FailureKind.AGENT_ERROR
                    else:
                        failure_kind = FailureKind.ACTION_FAILED
                else:
                    # ASSERTION step: can only fail with ASSERTION_FAILED or AGENT_ERROR
                    if isinstance(last_error, type) or "SelectorConfigError" in str(
                        last_error
                    ):
                        failure_kind = FailureKind.AGENT_ERROR
                    else:
                        failure_kind = FailureKind.ASSERTION_FAILED

                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        source=source,
                        duration_ms=duration_ms,
                        failure=failure_kind,
                        error=last_error,
                    )
                )
                logger.info(
                    f"Step {step_index} ({step.id}): failed ({failure_kind.value}) - {last_error}"
                )
                any_failed = True

    finally:
        # Always close browser
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass

    return RunResult.from_results(
        test.id,
        results,
        cost_usd=cost_usd,
        heals_used=heals_used,
        heal_proposals=heal_proposals,
    )
