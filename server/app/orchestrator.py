"""Orchestrator: the meta-agent state machine.

Drives the complete test pipeline from exploration through reporting with no human
intervention between stages. Each stage is a discrete, testable handler that updates
the OrchestratorState and produces a Decision (stage + verdict + reason + next stage).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from playwright.sync_api import sync_playwright

from app.browser import Browser
from app.codegen import write_suite, is_importable
from app.compiler import apply_credentials, DEFAULT_CREDENTIALS
from app.config import DEFAULTS, Guardrails
from app.contracts.contracts import (
    CoverageAssessment,
    CoverageVerdict,
    Decision,
    Flow,
    Gap,
    PlanMode,
    Stage,
    TERMINAL_STAGES,
    TestPlan,
    TriageResult,
    TriageVerdict,
)
from app.critic import assess_coverage, escalate_if_exhausted, structural_gaps
from app.executor import run_test
from app.explorer import _settle, explore, dismiss_consent, ExplorationReport
from app.llm import LLMConfig, LLMError
from app.models.models import CompiledTest, RunResult, Severity, StepKind
from app.paths import resolve_out_dir
from app.planner import plan_flows
from app.report import PipelineReport, render_pipeline_text, write_pipeline_report
from app.healer import rerank
from app.resolve import best, shortlist
from app.secrets import redact, resolve_value
from app.target import Target
from app.triage import triage_run

logger = logging.getLogger("aivar")


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator pipeline."""

    max_replans: int = 2
    max_regenerations: int = 1
    max_flows: int = 6
    max_explore_pages: int = 8
    max_explore_depth: int = 2
    max_cost_usd: float = 0.50
    max_pipeline_seconds: int = 300
    headless: bool = True
    safe_mode: bool = False
    heal: bool = True
    out_dir: str = "artifacts"
    generated_dir: str = "tests/generated"
    quarantine_dir: str = "quarantine"


@dataclass
class OrchestratorState:
    """Complete state of a pipeline run."""

    url: str
    username: str | None
    password: str | None
    intent: str | None
    prd_text: str | None
    mode: PlanMode
    stage: Stage = Stage.EXPLORE
    exploration: ExplorationReport | None = None
    plan: TestPlan | None = None
    compiled_flows: list[Flow] = field(default_factory=list)
    dropped_flows: list[tuple[str, str]] = field(
        default_factory=list
    )  # (flow name, reason)
    flow_results: dict[str, RunResult] = field(default_factory=dict)
    gaps: list[Gap] = field(default_factory=list)
    triage: list[TriageResult] = field(default_factory=list)
    ledger: list[Decision] = field(default_factory=list)
    replans_used: int = 0
    regens_used: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.time)
    escalation_reason: str | None = None
    generated_files: list[str] = field(default_factory=list)
    replan_instruction: str | None = None

    # Notified as each Decision is recorded, so a caller can narrate a run while
    # it is still in flight. Observation only: it must never influence control
    # flow, and a hook that raises is swallowed rather than failing the run.
    on_decision: Callable[[Decision], None] | None = None

    @property
    def elapsed_s(self) -> float:
        """Elapsed time since start in seconds."""
        return time.time() - self.started_at

    # The budget the run is actually held to. Set from OrchestratorConfig when
    # the pipeline starts; these defaults only apply to a bare state built in a
    # test. Previously the limits were hard-coded here, which silently ignored
    # the configured values and made the documented guardrails untrue.
    max_cost_usd: float = 0.50
    max_pipeline_seconds: int = 300

    def over_budget(self) -> str | None:
        """Reason the run has exhausted its budget, or None while it is within it."""
        if self.cost_usd >= self.max_cost_usd:
            return f"Cost budget exceeded: ${self.cost_usd:.4f} >= ${self.max_cost_usd:.2f}"
        if self.elapsed_s >= self.max_pipeline_seconds:
            return f"Time budget exceeded: {self.elapsed_s:.0f}s >= {self.max_pipeline_seconds}s"
        return None

    def record(self, decision: Decision) -> None:
        """Record a decision, log it, and notify any observer."""
        self.ledger.append(decision)
        logger.info(
            f"{decision.stage.value} -> {decision.verdict}: {decision.reason}"
        )
        if self.on_decision is not None:
            try:
                self.on_decision(decision)
            except Exception:
                # A progress observer is a convenience. It does not get to
                # terminate a run that is otherwise healthy.
                logger.warning("on_decision hook raised; ignoring", exc_info=True)


# ============================================================================
# Stage Handlers
# ============================================================================


def _step_explore(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """EXPLORE stage: discover app structure."""
    try:
        state.exploration = explore(
            state.url,
            username=state.username,
            password=state.password,
            max_pages=cfg.max_explore_pages,
            max_depth=cfg.max_explore_depth,
            headless=cfg.headless,
            safe_mode=cfg.safe_mode,
        )

        if state.exploration.page_count == 0:
            return Decision.now(
                state.stage,
                "escalate",
                "Exploration found 0 pages",
                Stage.ESCALATED,
                {
                    "page_count": state.exploration.page_count,
                    "authenticated": state.exploration.authenticated,
                    "consent_dismissed": state.exploration.consent_dismissed,
                },
            )

        return Decision.now(
            state.stage,
            "continue",
            f"Discovered {state.exploration.page_count} pages",
            Stage.PLAN,
            {
                "page_count": state.exploration.page_count,
                "authenticated": state.exploration.authenticated,
                "consent_dismissed": state.exploration.consent_dismissed,
                "entry_url": state.exploration.entry_url,
                # Naming the pages, not just counting them: "discovered 5 pages"
                # is unauditable on its own, and the ledger's job is to let a
                # human check the claim rather than trust it. Bounded by
                # max_explore_pages, and this rides into Postgres with the rest
                # of the evidence, so run history keeps it too.
                "pages": [
                    {
                        "url": p.url,
                        "title": p.title,
                        "depth": p.depth,
                        "forms": len(p.forms),
                    }
                    for p in state.exploration.pages
                ],
            },
        )
    except Exception as e:
        return Decision.now(
            state.stage,
            "escalate",
            f"Exploration failed: {str(e)}",
            Stage.ESCALATED,
        )


def _step_plan(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """PLAN stage: generate test flows."""
    try:
        if state.exploration is None:
            return Decision.now(
                state.stage,
                "escalate",
                "No exploration report",
                Stage.ESCALATED,
            )

        digest = state.exploration.summarize()
        plan, llm_response = plan_flows(
            digest,
            mode=state.mode,
            intent=state.intent,
            prd_text=state.prd_text,
            config=llm,
            max_flows=cfg.max_flows,
            plan_id=f"plan-{state.replans_used + 1}",
            extra_instruction=state.replan_instruction,
        )

        state.plan = plan
        state.cost_usd += llm_response.cost_usd

        return Decision.now(
            state.stage,
            "continue",
            f"Planned {plan.flow_count} flows ({', '.join(str(k.value) for k in plan.kinds_covered)})",
            Stage.CRITIQUE,
            {
                "flow_count": plan.flow_count,
                "kinds": [k.value for k in plan.kinds_covered],
                # The journeys themselves, not just how many there were. A
                # count and a set of kinds cannot be checked against the app;
                # named flows and their steps can. Selectors are deliberately
                # absent -- nothing is compiled yet at this stage, so there is
                # nothing truthful to report about them.
                "flows": [
                    {
                        "name": f.name,
                        "kind": f.kind.value,
                        "description": f.description,
                        "steps": [
                            {
                                "kind": s.kind.value,
                                "verb": s.verb,
                                "target": s.target,
                                # Credentials travel as ${NAME} placeholders and
                                # are resolved far later than this, but evidence
                                # is persisted and rendered in a browser, so it
                                # goes through redact rather than trusting that.
                                "value": redact(s.value) if s.value else None,
                            }
                            for s in f.steps
                        ],
                    }
                    for f in plan.flows
                ],
            },
        )
    except Exception as e:
        return Decision.now(
            state.stage,
            "escalate",
            f"Planning failed: {str(e)}",
            Stage.ESCALATED,
        )


def _step_critique(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """CRITIQUE stage: assess coverage."""
    try:
        if state.exploration is None or state.plan is None:
            return Decision.now(
                state.stage,
                "escalate",
                "Missing exploration or plan",
                Stage.ESCALATED,
            )

        assessment, _ = assess_coverage(
            state.exploration,
            state.plan,
            mode=state.mode,
            config=llm,
            prd_text=state.prd_text,
        )

        # Check if verdict should be escalated due to replan exhaustion
        assessment = escalate_if_exhausted(
            assessment, state.replans_used, cfg.max_replans
        )

        state.gaps = assessment.gaps

        if assessment.verdict == CoverageVerdict.ACCEPT:
            return Decision.now(
                state.stage,
                "accept",
                f"Coverage acceptable (score: {assessment.score:.2f})",
                Stage.GENERATE,
                {
                    "gap_count": len(assessment.gaps),
                    "verdict": assessment.verdict.value,
                    "score": assessment.score,
                },
            )
        elif assessment.verdict == CoverageVerdict.REPLAN:
            state.replans_used += 1
            state.replan_instruction = assessment.replan_instruction
            return Decision.now(
                state.stage,
                "replan",
                f"Coverage insufficient (score: {assessment.score:.2f}, {len(assessment.gaps)} gaps)",
                Stage.PLAN,
                {
                    "gap_count": len(assessment.gaps),
                    "verdict": assessment.verdict.value,
                    "score": assessment.score,
                    "replan_count": state.replans_used,
                },
            )
        else:  # ESCALATE
            return Decision.now(
                state.stage,
                "escalate",
                f"Coverage unacceptable after {state.replans_used} replans",
                Stage.ESCALATED,
                {
                    "gap_count": len(assessment.gaps),
                    "verdict": assessment.verdict.value,
                    "score": assessment.score,
                    "replans_used": state.replans_used,
                },
            )
    except Exception as e:
        return Decision.now(
            state.stage,
            "escalate",
            f"Critique failed: {str(e)}",
            Stage.ESCALATED,
        )


def _step_generate(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """GENERATE stage: compile flows by dry-running them."""
    if state.plan is None:
        return Decision.now(
            state.stage,
            "escalate",
            "No plan to generate",
            Stage.ESCALATED,
        )

    # Build credentials dict from orchestrator state
    credentials = {}
    if state.username:
        credentials["username"] = state.username
    else:
        credentials["username"] = DEFAULT_CREDENTIALS.get("username", "${AIVAR_USERNAME}")

    if state.password:
        credentials["password"] = state.password
    else:
        credentials["password"] = DEFAULT_CREDENTIALS.get("password", "${AIVAR_PASSWORD}")

    compiled = []
    playwright = None
    browser = None
    page = None
    browser_wrapper = None

    # How many targets in this run may be located by asking the model, shared
    # across every flow. Two per planned flow is enough for the handful the text
    # match cannot decide, without letting a run of unresolvable targets turn
    # into an unbounded spend.
    resolve_budget = {"left": 2 * len(state.plan.flows)}

    try:
        # Launch browser once for all flows
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=cfg.headless)
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            browser_wrapper = Browser(page)
        except Exception as browser_error:
            # Browser launch failed - try to compile flows anyway without browser
            logger.warning(f"Browser launch failed: {browser_error}")
            browser_wrapper = None

        for flow in state.plan.flows:
            if browser_wrapper is None:
                # Compile without browser - steps will be mostly unresolved
                compiled.append(flow)
            else:
                # Every flow starts from the entry URL on a clean session. The
                # previous flow left the browser logged in and several pages
                # deep, so without this reset a flow's own login steps have
                # nothing to resolve against -- and without navigating at all,
                # every snapshot is of a blank tab and nothing ever compiles.
                target_url = flow.entry_url or state.url
                try:
                    page.context.clear_cookies()
                    page.goto(target_url)
                    # The first snapshot of the flow is taken right after this;
                    # without waiting it can be of a shell the framework has not
                    # filled in yet, and the flow's opening step never resolves.
                    _settle(page)
                    dismiss_consent(page)
                except Exception as nav_error:
                    logger.warning(
                        "generate: could not open %s for flow %s: %s",
                        target_url, flow.id, nav_error,
                    )
                    compiled.append(flow)
                    continue

                compiled_flow = _compile_flow(
                    flow,
                    state.url,
                    browser_wrapper,
                    llm,
                    credentials=credentials,
                    resolve_budget=resolve_budget,
                )
                compiled.append(compiled_flow)

        # Drop assertions that could not be located, rather than the flows
        # holding them.
        #
        # is_compiled is all-or-nothing, so a flow whose every action resolved
        # and whose one assertion did not was discarded whole -- three good
        # steps thrown away for a fourth. The planner names assertions
        # semantically ("error message") while the resolver matches on text, and
        # an app whose error reads "Epic sadface: ..." has no word in common with
        # it, so this was the common case rather than the rare one.
        #
        # A flow keeps shipping as long as one assertion survives. A flow with
        # none is not a test and is left partial deliberately.
        compiled = [_prune_unresolved_assertions(f, state) for f in compiled]

        # Separate fully compiled and partial flows
        fully = [f for f in compiled if f.is_compiled]
        partial = [f for f in compiled if not f.is_compiled]

        # Put BOTH fully and partial on state.compiled_flows so VALIDATE can decide
        # whether to regenerate or drop them. Reporting a count the next stage
        # immediately contradicts destroys trust in the whole ledger.
        state.compiled_flows = compiled

        # If both empty, escalate rather than continue
        if len(fully) == 0 and len(partial) == 0:
            return Decision.now(
                state.stage,
                "escalate",
                "No flows compiled (neither fully nor partial)",
                Stage.ESCALATED,
                {
                    "fully_compiled": 0,
                    "partial": 0,
                    "planned": len(state.plan.flows),
                },
            )

        reason = f"Compiled {len(fully)} of {len(state.plan.flows)} flows fully ({len(partial)} partial)"
        return Decision.now(
            state.stage,
            "continue",
            reason,
            Stage.VALIDATE,
            {
                "fully_compiled": len(fully),
                "partial": len(partial),
                "planned": len(state.plan.flows),
                # Which flows are partial, and precisely which steps could not
                # be located. "3 partial" gives nothing to act on; a named step
                # that failed to resolve points straight at either a selector
                # problem or a part of the app the explorer never reached.
                "compiled": [
                    {
                        "name": f.name,
                        "compiled": f.is_compiled,
                        "steps_total": len(f.steps),
                        "unresolved": [
                            {"verb": s.verb, "target": s.target}
                            for s in f.steps
                            if s.selector is None and s.verb != "assert_url"
                        ],
                    }
                    for f in compiled
                ],
            },
        )

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return Decision.now(
            state.stage,
            "escalate",
            f"Generation failed: {str(e)}",
            Stage.ESCALATED,
        )
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if playwright:
            try:
                playwright.stop()
            except Exception:
                pass


def _prune_unresolved_assertions(flow: Flow, state: OrchestratorState) -> Flow:
    """Drop assertion steps that never resolved, keeping the flow.

    Returns the flow unchanged when nothing needs dropping, or when dropping
    would leave it with no assertions at all -- a flow that asserts nothing
    always passes, which is precisely the shape this project exists to avoid
    producing. Such a flow stays partial and VALIDATE deals with it.

    Every drop is recorded as a gap, so a shortened flow is visible in the
    report rather than quietly weaker than the plan it came from.
    """
    from dataclasses import replace as dataclass_replace

    unresolved = [
        s
        for s in flow.steps
        if s.kind is StepKind.ASSERTION
        and s.selector is None
        and s.verb != "assert_url"
    ]
    if not unresolved:
        return flow

    kept = [s for s in flow.steps if s not in unresolved]
    if not any(s.kind is StepKind.ASSERTION for s in kept):
        return flow

    for step in unresolved:
        gap = Gap(
            kind="dropped_assertion",
            description=f"Assertion '{step.target}' in flow '{flow.name}' could not be located",
            evidence=f"{step.verb} on '{step.target}' - no matching element on the page",
            severity=Severity.MODERATE,
        )
        if not any(g.description == gap.description for g in state.gaps):
            state.gaps.append(gap)

    logger.info(
        "compile %s: dropped %d unresolvable assertion(s), %d step(s) remain",
        flow.id, len(unresolved), len(kept),
    )
    return dataclass_replace(flow, steps=kept)


def _compile_flow(
    flow: Flow,
    url: str,
    browser: Browser,
    llm: LLMConfig,
    *,
    credentials: dict[str, str] | None = None,
    resolve_budget: dict[str, int] | None = None,
) -> Flow:
    """
    Compile a single flow by dry-running it against the live app.

    Takes a fresh snapshot before each step, resolves the target, executes the
    step so the page advances, and records the selector. Injects credentials
    for fill steps with no value.

    Args:
        flow: The Flow to compile
        url: Entry URL (unused but kept for signature compatibility)
        browser: Browser wrapper for executing steps
        llm: LLM config (unused but kept for signature compatibility)
        credentials: Dict of credential placeholders (e.g., {"username": "${AIVAR_USERNAME}"})
                    Falls back to DEFAULT_CREDENTIALS if not provided.
        resolve_budget: Shared {"left": n} counter capping how many targets in a
                    run may be located by asking the model. Shared across flows
                    so one pathological flow cannot spend the whole allowance.
                    None disables model-assisted resolution entirely.
    """
    from dataclasses import replace as dataclass_replace

    if credentials is None:
        credentials = DEFAULT_CREDENTIALS

    compiled_steps = []
    prev_url = None

    for step in flow.steps:
        try:
            # assert_url checks the address, so there is no element to find. It
            # was being handed to the element resolver, which naturally failed
            # on targets like "inventory address" and marked the whole flow
            # partial -- a step that cannot be resolved because it has nothing
            # to resolve.
            if step.verb == "assert_url":
                compiled_steps.append(step)
                continue

            # Inject credentials for fill steps with no value FIRST.
            # apply_credentials returns a placeholder value (e.g., ${AIVAR_PASSWORD}),
            # never a resolved secret. The compiled step keeps the placeholder.
            step_with_credentials = step
            if step.verb == "fill" and not step.value:
                # Create a simple object that mimics PlannedStep for apply_credentials
                class _StepLike:
                    def __init__(self, verb, target, value):
                        self.verb = verb
                        self.target = target
                        self.value = value

                step_like = _StepLike(step.verb, step.target, step.value)
                injected_value = apply_credentials(step_like, credentials, flow_kind=flow.kind, flow_name=flow.name)
                if injected_value and injected_value != step.value:
                    step_with_credentials = dataclass_replace(step, value=injected_value)

            # Fresh snapshot per step: the page has advanced since the last one,
            # which is what makes a post-login target resolvable at all.
            # Browser.snapshot() returns list[Node] -- not a dict.
            try:
                nodes = browser.snapshot()
            except Exception as e:
                logger.warning("compile %s/%s: snapshot failed: %s", flow.id, step.id, e)
                nodes = []

            # resolve.best() returns a Candidate (or None); the Selector hangs off it.
            candidate = best(nodes, step.target) if step.target and nodes else None
            selector = candidate.selector if candidate else None

            # Ask the model when the text match cannot decide.
            #
            # Plans name targets by meaning -- "error message", "confirmation
            # banner" -- while the heuristic matches on words. An app whose
            # error reads "Epic sadface: Username and password do not match"
            # shares no word with "error message", so the step failed to resolve
            # and its flow was cut short, though the element was plainly on the
            # page and a person would have picked it instantly.
            #
            # This is first-time location, not repair. Healing an assertion is
            # forbidden because a failing check is a candidate bug; building the
            # check in the first place is just reading the page, and applies to
            # actions and assertions alike.
            if selector is None and step.target and nodes and llm is not None:
                if resolve_budget is not None and resolve_budget.get("left", 0) > 0:
                    try:
                        options = shortlist(nodes, step.target, limit=8)
                        if options:
                            resolve_budget["left"] -= 1
                            result, _ = rerank(step.target, options, llm)
                            if (
                                0 <= result.index < len(options)
                                and result.confidence >= DEFAULTS.min_heal_confidence
                            ):
                                selector = options[result.index].selector
                                logger.info(
                                    "compile %s/%s: model located %r (confidence %.2f) - %s",
                                    flow.id, step.id, step.target,
                                    result.confidence, result.reasoning,
                                )
                    except Exception as e:
                        # A model that will not answer costs one unresolved
                        # step, never the run.
                        logger.info(
                            "compile %s/%s: model could not locate %r: %s",
                            flow.id, step.id, step.target, e,
                        )

            if selector is None:
                logger.info(
                    "compile %s/%s: could not resolve %r among %d nodes",
                    flow.id, step.id, step.target, len(nodes),
                )
                # Still append the step with credentials injected if applicable
                compiled_steps.append(step_with_credentials)
                continue

            compiled_step = dataclass_replace(step_with_credentials, selector=selector)
            compiled_steps.append(compiled_step)
            logger.info(
                "compile %s/%s: %r -> %s=%r",
                flow.id, step.id, step.target, selector.strategy, selector.value,
            )

            # Execute so the page advances. Browser.act signature is
            # (selector, verb, value, timeout_ms) -- order matters.
            if step.kind == StepKind.ACTION:
                try:
                    value = resolve_value(compiled_step.value)
                    browser.act(selector, compiled_step.verb, value, DEFAULTS.action_timeout_ms)

                    # Let the page catch up before the next step reads it.
                    #
                    # A click that signs in or navigates returns as soon as the
                    # click lands, while the destination is still rendering. The
                    # next step's snapshot then saw the old page, or a blank one,
                    # and every target after the first click failed to resolve --
                    # which is why whole flows arrived at VALIDATE with 100%
                    # unresolved steps and were dropped. Waiting here is what a
                    # person does between clicking and looking.
                    page_obj = getattr(browser, "page", None) or getattr(browser, "_page", None)
                    if page_obj is not None:
                        _settle(page_obj)

                    # Log navigation if this was a click that changed the URL
                    if compiled_step.verb == "click":
                        try:
                            current_url = browser.page.url if hasattr(browser, 'page') else None
                            if current_url and prev_url and current_url != prev_url:
                                logger.info(
                                    "compile %s/%s: navigated to %s",
                                    flow.id, step.id, current_url,
                                )
                            prev_url = current_url
                        except Exception:
                            pass  # Silently ignore URL tracking failures
                except Exception as e:
                    # Compiled but did not advance the page. Keep the selector;
                    # VALIDATE decides whether the flow is still worth shipping.
                    logger.info("compile %s/%s: step did not execute: %s", flow.id, step.id, e)

            # Log if fill step has no value after credential injection
            if compiled_step.verb == "fill" and not compiled_step.value:
                logger.info(
                    "compile %s/%s: fill step has no value",
                    flow.id, step.id,
                )

        except Exception as e:
            logger.warning("compile %s/%s: unexpected failure: %s", flow.id, step.id, e)
            compiled_steps.append(step)

    # Return flow with compiled steps
    return Flow(
        id=flow.id,
        name=flow.name,
        description=flow.description,
        kind=flow.kind,
        steps=compiled_steps,
        entry_url=flow.entry_url,
    )


def _step_validate(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """VALIDATE stage: check compiled flows and write files."""
    if not state.compiled_flows:
        return Decision.now(
            state.stage,
            "escalate",
            "No flows to validate",
            Stage.ESCALATED,
        )

    # Check for high unresolved step percentages
    surviving_flows = []
    for flow in state.compiled_flows:
        unresolved_count = sum(
            1
            for step in flow.steps
            if step.kind == StepKind.ACTION and step.selector is None
        )
        total_actions = sum(
            1 for step in flow.steps if step.kind == StepKind.ACTION
        )

        if total_actions > 0:
            unresolved_pct = unresolved_count / total_actions
        else:
            unresolved_pct = 0

        if unresolved_pct > 0.30:
            # High unresolved rate
            if state.regens_used < cfg.max_regenerations:
                # Trigger regeneration
                state.regens_used += 1
                return Decision.now(
                    state.stage,
                    "regenerate",
                    f"Flow {flow.name} has {unresolved_pct*100:.0f}% unresolved steps",
                    Stage.GENERATE,
                    {
                        "flow_name": flow.name,
                        "unresolved_pct": unresolved_pct,
                        "regens_used": state.regens_used,
                    },
                )
            else:
                # Drop the flow
                reason = f"{unresolved_pct*100:.0f}% unresolved steps"
                state.dropped_flows.append((flow.name, reason))
                gap = Gap(
                    kind="dropped_flow",
                    description=f"Flow '{flow.name}' dropped due to unresolved steps",
                    evidence=reason,
                    severity=Severity.SERIOUS,
                )
                # Dedupe on gap description: only add if not already present
                if not any(g.description == gap.description for g in state.gaps):
                    state.gaps.append(gap)
                continue

        surviving_flows.append(flow)

    if not surviving_flows:
        return Decision.now(
            state.stage,
            "escalate",
            "No flows survived validation",
            Stage.ESCALATED,
        )

    # Write files for surviving flows
    try:
        written_paths = write_suite(
            surviving_flows, state.url, out_dir=cfg.generated_dir
        )

        # If no files were written, escalate
        if not written_paths:
            return Decision.now(
                state.stage,
                "escalate",
                "Failed to write test files",
                Stage.ESCALATED,
            )

        # Validate each file
        for path in written_paths:
            importable, error = is_importable(path)
            if not importable:
                # Find and drop the corresponding flow
                surviving_flows = [
                    f
                    for f in surviving_flows
                    if f.id not in str(path)
                ]
                gap = Gap(
                    kind="unimportable_file",
                    description=f"Generated file {path.name} has syntax errors",
                    evidence=error,
                    severity=Severity.SERIOUS,
                )
                # Dedupe on gap description: only add if not already present
                if not any(g.description == gap.description for g in state.gaps):
                    state.gaps.append(gap)

        if not surviving_flows:
            return Decision.now(
                state.stage,
                "escalate",
                "No flows produced valid code",
                Stage.ESCALATED,
            )

        state.compiled_flows = surviving_flows
        state.generated_files = [str(p) for p in written_paths]

        return Decision.now(
            state.stage,
            "continue",
            f"Validated and wrote {len(surviving_flows)} flows",
            Stage.EXECUTE,
            {
                "validated": len(surviving_flows),
                "dropped": len(state.dropped_flows),
            },
        )

    except Exception as e:
        return Decision.now(
            state.stage,
            "escalate",
            f"Validation failed: {str(e)}",
            Stage.ESCALATED,
        )


def _step_execute(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """EXECUTE stage: run all compiled flows."""
    try:
        if not state.compiled_flows:
            return Decision.now(
                state.stage,
                "escalate",
                "No flows to execute",
                Stage.ESCALATED,
            )

        for flow in state.compiled_flows:
            test = CompiledTest(
                id=flow.id,
                intent=flow.name,
                url=state.url,
                steps=flow.steps,
            )

            try:
                result = run_test(
                    test,
                    headless=cfg.headless,
                    llm_config=llm,
                    heal=cfg.heal,
                    quarantine_dir=cfg.quarantine_dir,
                )
                state.flow_results[flow.id] = result
                state.cost_usd += result.cost_usd
            except Exception as e:
                # Record as error result
                result = RunResult(
                    test_id=test.id,
                    status="error",
                    results=[],
                    cost_usd=0.0,
                )
                state.flow_results[flow.id] = result

        passed = sum(
            1 for r in state.flow_results.values() if r.status == "passed"
        )
        failed = sum(
            1 for r in state.flow_results.values() if r.status == "failed"
        )

        names = {f.id: f.name for f in state.compiled_flows}

        # Record what the Healer did, as its own entry in the ledger.
        #
        # Repair happens inside the step loop, because a locator has to be
        # replaced and retried before the next step can run -- there is no point
        # in the pipeline where healing could be a separate pass. But that left
        # the ledger showing EXECUTE going straight to TRIAGE, so the one thing
        # a reader most wants to check -- what the agent changed, and on what
        # evidence -- was the one thing it never said. The count was reported;
        # the reasoning was not.
        healed = [
            (flow_id, proposal)
            for flow_id, r in state.flow_results.items()
            for proposal in r.heal_proposals
        ]
        if healed:
            state.record(
                Decision.now(
                    Stage.HEAL,
                    "healed",
                    f"Repaired {len(healed)} locator(s) across "
                    f"{len({fid for fid, _ in healed})} flow(s)",
                    Stage.TRIAGE,
                    {
                        "heals": [
                            {
                                "flow": names.get(flow_id, flow_id),
                                "step_id": p.step_id,
                                "from": p.old.to_dict() if p.old else None,
                                "to": p.new.to_dict(),
                                "confidence": round(p.confidence, 2),
                                "semantic_match": p.semantic_match,
                                "reasoning": p.reasoning,
                            }
                            for flow_id, p in healed
                        ],
                        # Assertions are never in this list, and that is the
                        # point: a failing assertion is a candidate bug.
                        "assertions_healed": 0,
                    },
                )
            )

        return Decision.now(
            state.stage,
            "continue",
            f"Executed {len(state.flow_results)} flows: {passed} passed, {failed} failed",
            Stage.TRIAGE,
            {
                "total": len(state.flow_results),
                "passed": passed,
                "failed": failed,
                # Which flows failed, and on which step. A pass/fail tally says
                # a run happened; it does not say what broke.
                "executed": [
                    {
                        "name": names.get(flow_id, flow_id),
                        "status": r.status,
                        "steps_total": len(r.results),
                        "steps_passed": sum(
                            1 for sr in r.results if sr.status == "passed"
                        ),
                        "heals_used": r.heals_used,
                        "failures": [
                            {
                                "step_id": sr.step_id,
                                "failure": sr.failure.value if sr.failure else None,
                                # Playwright errors run to many lines; the head
                                # of one identifies it without turning the
                                # ledger into a log file.
                                "error": (sr.error or "")[:240],
                            }
                            for sr in r.results
                            if sr.status == "failed"
                        ],
                    }
                    for flow_id, r in state.flow_results.items()
                ],
            },
        )

    except Exception as e:
        return Decision.now(
            state.stage,
            "escalate",
            f"Execution failed: {str(e)}",
            Stage.ESCALATED,
        )


def _step_triage(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """TRIAGE stage: classify failures."""
    try:
        # Collect all failed steps across all flows
        for flow_id, result in state.flow_results.items():
            # Find the flow to get step definitions
            flow = next(
                (f for f in state.compiled_flows if f.id == flow_id), None
            )
            if flow is None:
                continue

            # Build steps_by_id for triage_run
            steps_by_id = {step.id: step for step in flow.steps}

            # Triage failures
            flow_triage = triage_run(
                steps_by_id, result.results, config=llm
            )
            state.triage.extend(flow_triage)

        # step_id alone is not readable: name the flow and the step it belongs
        # to, so a verdict can be checked against the journey that produced it.
        step_index: dict[str, tuple[str, str]] = {}
        for f in state.compiled_flows:
            for s in f.steps:
                step_index[s.id] = (f.name, f"{s.verb} {s.target}".strip())

        counts = {v.value: 0 for v in TriageVerdict}
        for t in state.triage:
            counts[t.verdict.value] += 1

        defects = counts[TriageVerdict.APP_DEFECT.value]
        reason = (
            f"Triaged {len(state.triage)} failures: "
            f"{defects} app defects, "
            f"{counts[TriageVerdict.SCRIPT_ISSUE.value]} script issues, "
            f"{counts[TriageVerdict.FLAKY.value]} flaky"
        )

        return Decision.now(
            state.stage,
            "continue",
            reason,
            Stage.REPORT,
            {
                "triage_count": len(state.triage),
                "verdicts": counts,
                # The verdict is the single most consequential judgement in a
                # run: app_defect is a candidate bug that must never be healed
                # away. Recording the reasoning and the confidence beside it is
                # what makes that judgement reviewable instead of asserted.
                "triaged": [
                    {
                        "flow": step_index.get(t.step_id, (None, None))[0],
                        "step": step_index.get(t.step_id, (None, t.step_id))[1],
                        "verdict": t.verdict.value,
                        "confidence": t.confidence,
                        "reasoning": t.reasoning,
                    }
                    for t in state.triage
                ],
            },
        )

    except Exception as e:
        # Triage failures are not critical; continue to report
        return Decision.now(
            state.stage,
            "continue",
            f"Triage had issues but continuing: {str(e)}",
            Stage.REPORT,
        )


def _step_report(
    state: OrchestratorState, cfg: OrchestratorConfig, llm: LLMConfig
) -> Decision:
    """REPORT stage: write pipeline report."""
    try:
        # Build PipelineReport
        report = PipelineReport(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            url=state.url,
            mode=state.mode.value,
            intent=state.intent,
            plan=state.plan,
            flow_results=state.flow_results,
            gaps=state.gaps,
            triage=state.triage,
            decisions=state.ledger,
            escalated=state.escalation_reason is not None,
            escalation_reason=state.escalation_reason,
            cost_usd=state.cost_usd,
            duration_s=state.elapsed_s,
            generated_files=state.generated_files,
        )

        # Write report
        paths = write_pipeline_report(report, out_dir=cfg.out_dir)

        # Print text report
        text = render_pipeline_text(report)
        logger.info("\n" + text)

        # Persist to the database for run history. Deliberately best-effort:
        # the files written above are the source of truth, so an unreachable
        # database at a live demo costs us history, never the run itself.
        try:
            from app.store import save_run_safe

            saved = save_run_safe(report)
            if saved:
                logger.info(f"Run saved to database: {saved}")
            else:
                logger.info("Run not saved to database (unavailable) - files are on disk")
        except ImportError:
            logger.debug("store module unavailable; skipping persistence")

        # Log paths
        logger.info(f"Report JSON: {paths['json']}")
        logger.info(f"Report TXT: {paths['txt']}")
        logger.info(f"Report HTML: {paths['html']}")
        if state.generated_files:
            logger.info(f"Generated tests: {cfg.generated_dir}")

        evidence = {
            "paths": {k: str(v) for k, v in paths.items()},
        }
        if state.escalation_reason is not None:
            evidence["escalation_reason"] = state.escalation_reason

        return Decision.now(
            state.stage,
            "continue",
            f"Report written",
            Stage.DONE,
            evidence,
        )

    except Exception as e:
        return Decision.now(
            state.stage,
            "escalate",
            f"Report failed: {str(e)}",
            Stage.DONE,
        )


# Map stage to handler
HANDLERS = {
    Stage.EXPLORE: _step_explore,
    Stage.PLAN: _step_plan,
    Stage.CRITIQUE: _step_critique,
    Stage.GENERATE: _step_generate,
    Stage.VALIDATE: _step_validate,
    Stage.EXECUTE: _step_execute,
    Stage.TRIAGE: _step_triage,
    Stage.REPORT: _step_report,
}


def run_pipeline(
    url: str,
    *,
    username: str | None = None,
    password: str | None = None,
    intent: str | None = None,
    prd_path: str | None = None,
    config: OrchestratorConfig | None = None,
    llm_config: LLMConfig | None = None,
    on_decision: Callable[[Decision], None] | None = None,
) -> tuple[PipelineReport, OrchestratorState]:
    """
    Run the complete test pipeline from exploration through reporting.

    Args:
        url: Entry URL to test
        username: Optional username for authentication
        password: Optional password for authentication
        intent: Optional natural-language intent (implies FOCUSED mode)
        prd_path: Optional path to product requirements document (implies SPEC_LED mode)
        config: OrchestratorConfig (uses defaults if not provided)
        llm_config: LLMConfig (loads from env if not provided)
        on_decision: Optional observer called with each Decision as it is
            recorded, for narrating a run in progress. Cannot affect the run.

    Returns:
        (PipelineReport, OrchestratorState)

    Raises:
        No exceptions are raised; all errors result in escalation and a report.
    """
    # Validate URL before spending a run
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        escalation_reason = "Invalid URL: must start with http:// or https://"
        state = OrchestratorState(
            url=url,
            username=username,
            password=password,
            intent=intent,
            prd_text=None,
            mode=PlanMode.SWEEP,
            stage=Stage.ESCALATED,
            escalation_reason=escalation_reason,
            on_decision=on_decision,
        )
        state.record(
            Decision.now(
                Stage.EXPLORE,
                "escalate",
                escalation_reason,
                Stage.ESCALATED,
            )
        )
        report = PipelineReport(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            url=url,
            mode=PlanMode.SWEEP.value,
            escalated=True,
            escalation_reason=escalation_reason,
            decisions=state.ledger,
        )
        return (report, state)

    if config is None:
        config = OrchestratorConfig()

    # Anchor output directories to the server root. Under uvicorn the working
    # directory is arbitrary, so relative paths would scatter reports and
    # generated tests wherever the process happened to start.
    config = replace(
        config,
        out_dir=str(resolve_out_dir(config.out_dir)),
        generated_dir=str(resolve_out_dir(config.generated_dir)),
        quarantine_dir=str(resolve_out_dir(config.quarantine_dir)),
    )

    if llm_config is None:
        try:
            llm_config = LLMConfig.from_env()
        except LLMError as e:
            logger.error(f"Failed to load LLM config: {e}")
            # Still return a report with escalation
            state = OrchestratorState(
                url=url,
                username=username,
                password=password,
                intent=intent,
                prd_text=None,
                mode=PlanMode.SWEEP,
                stage=Stage.ESCALATED,
                escalation_reason=f"Failed to load LLM config: {e}",
                on_decision=on_decision,
            )
            state.record(
                Decision.now(
                    Stage.EXPLORE,
                    "escalate",
                    f"Failed to load LLM config: {e}",
                    Stage.ESCALATED,
                )
            )
            report = PipelineReport(
                run_id=f"run-{uuid.uuid4().hex[:8]}",
                url=url,
                mode=PlanMode.SWEEP.value,
                escalated=True,
                escalation_reason=state.escalation_reason,
                decisions=state.ledger,
            )
            return (report, state)

    # Derive mode
    prd_text = None
    if prd_path:
        try:
            prd_text = Path(prd_path).read_text(encoding="utf-8")
            mode = PlanMode.SPEC_LED
        except Exception as e:
            logger.warning(f"Failed to read PRD: {e}")
            mode = PlanMode.SWEEP
    elif intent:
        mode = PlanMode.FOCUSED
    else:
        mode = PlanMode.SWEEP

    # Initialize state
    state = OrchestratorState(
        url=url,
        username=username,
        password=password,
        intent=intent,
        prd_text=prd_text,
        mode=mode,
        # Hold the run to the CONFIGURED budget, not to a default buried in the
        # state class. Without this, --max-cost and --max-pipeline-seconds are
        # accepted and then quietly ignored.
        max_cost_usd=config.max_cost_usd,
        max_pipeline_seconds=config.max_pipeline_seconds,
        on_decision=on_decision,
    )

    # Main loop
    while state.stage not in TERMINAL_STAGES:
        # Check budget
        over = state.over_budget()
        if over and state.stage not in (Stage.REPORT,):
            state.escalation_reason = over
            state.record(
                Decision.now(
                    state.stage, "escalate", over, Stage.ESCALATED
                )
            )
            state.stage = Stage.ESCALATED
            continue

        # Run handler
        try:
            handler = HANDLERS.get(state.stage)
            if handler is None:
                state.escalation_reason = f"Unknown stage: {state.stage}"
                state.record(
                    Decision.now(
                        state.stage,
                        "escalate",
                        state.escalation_reason,
                        Stage.ESCALATED,
                    )
                )
                state.stage = Stage.ESCALATED
                continue

            decision = handler(state, config, llm_config)
        except Exception as e:
            # Unexpected exception -> escalate
            state.escalation_reason = f"Handler exception: {str(e)}"
            state.record(
                Decision.now(
                    state.stage,
                    "escalate",
                    state.escalation_reason,
                    Stage.ESCALATED,
                )
            )
            state.stage = Stage.ESCALATED
            continue

        state.record(decision)
        state.stage = decision.next_stage

        # If decision leads to ESCALATED, capture reason
        if decision.next_stage == Stage.ESCALATED:
            if state.escalation_reason is None:
                state.escalation_reason = decision.reason
            logger.warning(f"PIPELINE ESCALATED: {state.escalation_reason}")

    # If escalated, run report handler if not already at DONE
    if state.stage == Stage.ESCALATED:
        # Need to generate report before DONE
        try:
            handler = HANDLERS[Stage.REPORT]
            decision = handler(state, config, llm_config)
            state.record(decision)
            state.stage = decision.next_stage
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state.stage = Stage.DONE

    # Build final report
    report = PipelineReport(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        url=state.url,
        mode=state.mode.value,
        intent=state.intent,
        plan=state.plan,
        flow_results=state.flow_results,
        gaps=state.gaps,
        triage=state.triage,
        decisions=state.ledger,
        escalated=state.escalation_reason is not None,
        escalation_reason=state.escalation_reason,
        cost_usd=state.cost_usd,
        duration_s=state.elapsed_s,
        generated_files=state.generated_files,
    )

    return (report, state)
