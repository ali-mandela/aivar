"""Coverage gate for test plans.

The Critic evaluates a test plan for coverage gaps before the Generator produces
test code. It runs deterministic structural checks first, then optionally consults
the model for semantic coverage judgement. The verdict is decided deterministically
from gaps, keeping the control flow auditable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.contracts.contracts import (
    CoverageAssessment,
    CoverageVerdict,
    FlowKind,
    Gap,
    PlanMode,
    TestPlan,
)
from app.explorer import ExplorationReport
from app.llm import LLMConfig, LLMError, LLMInvalidJSON, LLMResponse, chat_json, extract_json
from app.models.models import Severity, StepKind

logger = logging.getLogger("aivar")


def _normalize_text(text: str) -> str:
    """Normalize text for loose matching: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _text_overlap(text1: str, text2: str) -> bool:
    """Check if text1 and text2 have token overlap (loose match)."""
    norm1 = _normalize_text(text1)
    norm2 = _normalize_text(text2)

    if not norm1 or not norm2:
        return False

    # Check if either is a substring of the other
    return norm1 in norm2 or norm2 in norm1


def _has_login_keywords(text: str) -> bool:
    """Check if text contains login-related keywords."""
    norm = _normalize_text(text)
    login_keywords = {"user", "email", "login", "password", "signin", "sign", "in"}
    return any(keyword in norm for keyword in login_keywords)


def _url_key(url: str | None) -> str:
    """Compare URLs by origin and path, ignoring query and fragment."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse

        parts = urlparse(url)
        return f"{parts.netloc}{(parts.path or '/').rstrip('/') or '/'}".lower()
    except Exception:
        return url.lower()


def _form_signature(form) -> str:
    """Identify a form by its field names rather than the page it sits on.

    The same footer subscribe box on eight pages is one form to a person, and
    counting it eight times drowns the real gaps.
    """
    fields = ",".join(sorted(_normalize_text(f.name) for f in form.fields))
    return f"{_normalize_text(form.name)}|{fields}|{form.is_login}"


def structural_gaps(report: ExplorationReport, plan: TestPlan) -> list[Gap]:
    """
    Compute coverage gaps without any model call.

    Checks for:
    - untested_form: a form not referenced by any step target in any flow
    - untested_page: a discovered page whose title/headings don't appear in any flow
    - missing_error_state: no NEGATIVE or ERROR_STATE flow (skip in FOCUSED mode)
    - no_assertions: any flow with zero assertions

    Args:
        report: ExplorationReport from the explorer
        plan: TestPlan from the planner

    Returns:
        List of Gap objects
    """
    gaps: list[Gap] = []

    # The URLs flows actually start on. A flow with entry_url set to /contact is
    # testing /contact whatever its step wording happens to be; matching only on
    # text meant a plan could cover a page thoroughly and still be reported as
    # ignoring it, which sent a good plan back for re-planning.
    covered_urls = {
        _url_key(flow.entry_url) for flow in plan.flows if flow.entry_url
    }

    # Collect all step targets from all flows (lowercase, normalized)
    all_step_targets = set()
    for flow in plan.flows:
        for step in flow.steps:
            all_step_targets.add(_normalize_text(step.target))

    # Collect all text from all flows (headings, titles, form names)
    all_flow_text = set()
    for flow in plan.flows:
        flow.name and all_flow_text.add(_normalize_text(flow.name))
        flow.description and all_flow_text.add(_normalize_text(flow.description))
        for step in flow.steps:
            all_flow_text.add(_normalize_text(step.target))

    # 1. Check for untested forms
    # A form counts as COVERED when ANY of these hold for any flow in the plan:
    # 1. normalized tokens of form's NAME overlap a step target
    # 2. normalized tokens of ANY form's FIELD names overlap a step target (important!)
    # 3. form is login form and any flow has step whose target contains login keywords
    # A form in the site-wide chrome -- a newsletter box in the footer, a search
    # field in the header -- is observed once per page crawled. Counted per page
    # it produced six "untested form" gaps for one form, which on its own was
    # enough to fail the gate and burn the re-plan budget on a plan that had
    # merely not tested a newsletter signup.
    seen_form_signatures: set[str] = set()

    for page in report.pages:
        for form in page.forms:
            signature = _form_signature(form)
            if signature in seen_form_signatures:
                continue
            seen_form_signatures.add(signature)

            found = False

            # Check 0: a flow starts on this page and types into something.
            # Landing on the page and filling a field is what testing a form is.
            if _url_key(page.url) in covered_urls:
                for flow in plan.flows:
                    if _url_key(flow.entry_url) != _url_key(page.url):
                        continue
                    if any(s.verb in ("fill", "select", "check") for s in flow.steps):
                        found = True
                        break

            # Check 1: form name is referenced by any step target
            if not found:
                for step_target in all_step_targets:
                    if _text_overlap(form.name, step_target):
                        found = True
                        break

            # Check 2: ANY field name overlaps a step target (field names are real evidence of coverage)
            if not found:
                for field in form.fields:
                    for step_target in all_step_targets:
                        if _text_overlap(field.name, step_target):
                            found = True
                            break
                    if found:
                        break

            # Check 3: login form with login-related step targets
            if not found and form.is_login:
                for step_target in all_step_targets:
                    if _has_login_keywords(step_target):
                        found = True
                        break

            if not found:
                # Build evidence string with field names and page URL
                field_names = ", ".join(f.name for f in form.fields)
                from urllib.parse import urlparse
                try:
                    parsed_url = urlparse(page.url)
                    path = parsed_url.path or "/"
                except Exception:
                    path = page.url
                evidence = f"form on {path} with fields: {field_names}"

                severity = Severity.CRITICAL if form.is_login else Severity.SERIOUS
                gaps.append(Gap(
                    kind="untested_form",
                    description=f"Form '{form.name}' is not covered by any test flow",
                    evidence=evidence,
                    severity=severity,
                ))

    # 2. Check for untested pages
    # Include page URL in evidence and use URL path in description when needed
    for page in report.pages:
        page_found = False

        # A flow that starts here is testing here, regardless of its wording.
        if _url_key(page.url) in covered_urls:
            page_found = True

        # Check if page title appears in any flow
        if not page_found and page.title and _text_overlap(page.title, " ".join(all_flow_text)):
            page_found = True

        # Check if any heading appears in any flow
        if not page_found:
            for heading in page.headings:
                for flow_text in all_flow_text:
                    if _text_overlap(heading, flow_text):
                        page_found = True
                        break
                if page_found:
                    break

        if not page_found and (page.title or page.headings):
            page_identifier = page.title or (page.headings[0] if page.headings else page.url)

            # Use URL path in description to distinguish pages with same title but different URLs
            from urllib.parse import urlparse
            try:
                parsed_url = urlparse(page.url)
                path = parsed_url.path or "/"
                description = f"Page '{page_identifier}' ({path}) is not covered by any test flow"
            except Exception:
                description = f"Page '{page_identifier}' is not covered by any test flow"

            # Evidence contains the page URL for verification
            evidence = page.url

            gaps.append(Gap(
                kind="untested_page",
                description=description,
                evidence=evidence,
                severity=Severity.MODERATE,
            ))

    # 3. Check for missing error state flows (skip in FOCUSED mode)
    if plan.mode != PlanMode.FOCUSED:
        has_negative_or_error = any(
            f.kind in (FlowKind.NEGATIVE, FlowKind.ERROR_STATE)
            for f in plan.flows
        )
        if not has_negative_or_error:
            gaps.append(Gap(
                kind="missing_error_state",
                description="Plan contains no negative or error-state flows",
                evidence="all flows are happy path or navigation",
                severity=Severity.SERIOUS,
            ))

    # 4. Check for flows with no assertions
    for flow in plan.flows:
        if not flow.assertions:
            gaps.append(Gap(
                kind="no_assertions",
                description=f"Flow '{flow.name}' has no assertions",
                evidence=flow.name,
                severity=Severity.CRITICAL,
            ))

    return gaps


def assess_coverage(
    report: ExplorationReport,
    plan: TestPlan,
    *,
    mode: PlanMode,
    config: LLMConfig,
    prd_text: str | None = None,
    structural: list[Gap] | None = None,
) -> tuple[CoverageAssessment, LLMResponse | None]:
    """
    Assess coverage using deterministic structural checks and optionally the model.

    Args:
        report: ExplorationReport from the explorer
        plan: TestPlan from the planner
        mode: PlanMode (SWEEP, FOCUSED, SPEC_LED)
        config: LLMConfig for model calls
        prd_text: Product requirements document (truncated to 3000 chars for SPEC_LED)
        structural: Pre-computed structural gaps (if None, computed here)

    Returns:
        Tuple of (CoverageAssessment, LLMResponse or None)
    """
    # Compute structural gaps if not provided
    if structural is None:
        structural = structural_gaps(report, plan)

    # Build model prompt
    exploration_digest = report.summarize(max_chars=4000)

    # Build compact plan rendering
    plan_lines = []
    for flow in plan.flows:
        targets = ", ".join(s.target for s in flow.steps[:5])  # Limit to first 5 steps
        plan_lines.append(f"- {flow.name} ({flow.kind.value}): {targets}")
    plan_rendering = "\n".join(plan_lines)

    # Define what "good coverage" means
    coverage_definition = {
        PlanMode.SWEEP: "every discovered form and distinct page has a scenario, and at least one negative or error-state flow exists",
        PlanMode.FOCUSED: "everything within the stated intent is covered; flows outside it are NOT gaps",
        PlanMode.SPEC_LED: "every requirement stated in the PRD has a scenario",
    }[mode]

    # Build structural gaps summary for model
    structural_summary = ""
    if structural:
        structural_summary = "Structural gaps already identified:\n"
        for gap in structural:
            structural_summary += f"- {gap.kind}: {gap.description} (evidence: {gap.evidence})\n"

    # Build user message for model
    user_message = f"""Assess the coverage of this test plan.

Entry URL: {report.entry_url}
Mode: {mode.value}
Good coverage means: {coverage_definition}

Exploration digest (discovered pages, forms, controls):
{exploration_digest}

Test plan ({len(plan.flows)} flows):
{plan_rendering}

{f"Product requirements (first 3000 chars):{prd_text[:3000]}" if prd_text else ""}

{structural_summary}

Respond with JSON:
{{
  "score": <float 0.0-1.0>,
  "gaps": [
    {{"kind": "<gap_kind>", "description": "<what's missing>", "evidence": "<concrete thing found>", "severity": "critical|serious|moderate|minor"}},
    ...
  ],
  "reasoning": "<one or two sentences>",
  "replan_instruction": "<what planner should add, or null>"
}}

Be strict about critical and serious gaps. Only mark gaps you're confident about.
"""

    system_message = "You are a test coverage analyst. Identify gaps in test plans objectively."

    # Call model
    llm_response = None
    model_gaps = []
    model_reasoning = ""
    model_replan_instruction = None

    try:
        llm_response = chat_json(system_message, user_message, config)
        response_json = extract_json(llm_response.content)

        model_reasoning = response_json.get("reasoning", "")
        model_replan_instruction = response_json.get("replan_instruction")

        # Parse gaps from model
        for gap_dict in response_json.get("gaps", []):
            try:
                model_gaps.append(Gap(
                    kind=gap_dict.get("kind", "unknown"),
                    description=gap_dict.get("description", ""),
                    evidence=gap_dict.get("evidence", ""),
                    severity=Severity(gap_dict.get("severity", "minor")),
                ))
            except (KeyError, ValueError):
                logger.warning(f"Invalid gap in model response: {gap_dict}")

    except (LLMError, LLMInvalidJSON, ValueError) as e:
        logger.warning(f"Model call failed: {e}, using structural gaps only")
        llm_response = None
        model_reasoning = "Model unavailable, using structural gaps only"

    # Merge gaps (deduplicate on kind + description + evidence)
    # This ensures genuinely different pages/forms stay distinct while identical duplicates collapse
    merged_gaps = list(structural)
    seen_gaps = {(g.kind, g.description, g.evidence) for g in structural}

    for gap in model_gaps:
        key = (gap.kind, gap.description, gap.evidence)
        if key not in seen_gaps:
            merged_gaps.append(gap)
            seen_gaps.add(key)

    # Decide the verdict deterministically, and ONLY from structural gaps.
    #
    # Structural gaps come from our own checks against the exploration report:
    # a login form nobody exercises, a flow with no assertions. They are
    # grounded in observed evidence and a re-plan can actually fix them.
    #
    # Model gaps are a wish-list. They are genuinely useful -- "the sort
    # combobox exposes four options and you test one" is a real observation --
    # but on a plan capped at max_flows most of them are unfixable by
    # re-planning, so letting them drive control flow means the gate never
    # accepts anything and the pipeline burns its whole re-plan budget.
    # They are recorded instead as untested-flow risk, which is exactly what
    # the report's risk section is for.
    structural_critical = sum(1 for g in structural if g.severity == Severity.CRITICAL)
    structural_serious = sum(1 for g in structural if g.severity == Severity.SERIOUS)

    if structural_critical > 0 or structural_serious >= 2:
        verdict = CoverageVerdict.REPLAN
    else:
        verdict = CoverageVerdict.ACCEPT

    # Build replan instruction from gaps if model didn't provide one
    if verdict == CoverageVerdict.REPLAN and not model_replan_instruction:
        replan_parts = []
        for gap in merged_gaps:
            if gap.severity in (Severity.CRITICAL, Severity.SERIOUS):
                replan_parts.append(f"{gap.kind}: {gap.evidence}")
        model_replan_instruction = "; ".join(replan_parts) if replan_parts else None

    # Compute scores using the same weighted penalty but over different gap sets.
    # The structural score follows the verdict logic; the overall score shows what the
    # model's wish-list adds.
    penalties = {
        Severity.CRITICAL: 0.4,
        Severity.SERIOUS: 0.25,
        Severity.MODERATE: 0.1,
        Severity.MINOR: 0.05,
    }

    # Structural score (used for verdict logic)
    structural_score = 1.0
    for gap in structural:
        structural_score -= penalties.get(gap.severity, 0.05)
    structural_score = max(0.0, min(1.0, structural_score))

    # Overall score (includes model's aspirational gaps)
    score = 1.0
    for gap in merged_gaps:
        score -= penalties.get(gap.severity, 0.05)
    score = max(0.0, min(1.0, score))

    # Build reasoning with both scores
    if model_reasoning:
        reasoning = f"structural {structural_score:.2f} / overall {score:.2f} - {model_reasoning}"
    else:
        gap_summary = f"Found {len(merged_gaps)} coverage gaps" if merged_gaps else "No coverage gaps found"
        reasoning = f"structural {structural_score:.2f} / overall {score:.2f} - {gap_summary}"

    # Report the structural score, because it is the one the verdict was made
    # from. The overall score is dragged down by the model's wish-list, which is
    # deliberately not allowed to decide anything -- publishing it as *the* score
    # produced runs reading "verdict ACCEPT, score 0.00", which tells a reader
    # the gate is broken. Both numbers stay visible in `reasoning`.
    assessment = CoverageAssessment(
        verdict=verdict,
        score=structural_score,
        gaps=merged_gaps,
        reasoning=reasoning,
        replan_instruction=model_replan_instruction,
    )

    return assessment, llm_response


def escalate_if_exhausted(
    assessment: CoverageAssessment,
    replans_used: int,
    max_replans: int,
) -> CoverageAssessment:
    """
    Flip REPLAN to ESCALATE if the replan cap is reached.

    Args:
        assessment: CoverageAssessment from assess_coverage
        replans_used: Number of replans already executed
        max_replans: Maximum number of replans allowed

    Returns:
        Original assessment if ACCEPT or not at cap, else a modified copy with ESCALATE verdict
    """
    if assessment.verdict == CoverageVerdict.REPLAN and replans_used >= max_replans:
        return CoverageAssessment(
            verdict=CoverageVerdict.ESCALATE,
            score=assessment.score,
            gaps=assessment.gaps,
            reasoning=f"Replan cap reached ({replans_used}/{max_replans}): {assessment.reasoning}",
            replan_instruction=assessment.replan_instruction,
        )
    return assessment
