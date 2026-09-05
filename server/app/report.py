from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.contracts.contracts import Decision, Gap, TestPlan, TriageResult, TriageVerdict
from app.models.models import CompiledTest, RunResult, Severity


# --------------------------------------------------------------------------
# PipelineReport: the test quality report
# --------------------------------------------------------------------------


@dataclass
class PipelineReport:
    """A complete test quality report for a pipeline run."""

    run_id: str
    url: str
    mode: str = "sweep"
    intent: str | None = None
    plan: TestPlan | None = None
    flow_results: dict[str, RunResult] = field(default_factory=dict)  # flow id → result
    gaps: list[Gap] = field(default_factory=list)
    triage: list[TriageResult] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    escalated: bool = False
    escalation_reason: str | None = None
    cost_usd: float = 0.0
    duration_s: float = 0.0
    generated_files: list[str] = field(default_factory=list)

    @property
    def flows_total(self) -> int:
        """Total number of flows."""
        return len(self.flow_results)

    @property
    def flows_passed(self) -> int:
        """Number of flows that passed."""
        return sum(1 for r in self.flow_results.values() if r.status == "passed")

    @property
    def flows_failed(self) -> int:
        """Number of flows that failed."""
        return sum(1 for r in self.flow_results.values() if r.status == "failed")

    @property
    def steps_total(self) -> int:
        """Total number of steps across all flows."""
        return sum(len(r.results) for r in self.flow_results.values())

    @property
    def steps_passed(self) -> int:
        """Total number of passed steps across all flows."""
        return sum(
            sum(1 for sr in r.results if sr.status == "passed")
            for r in self.flow_results.values()
        )

    @property
    def heals_applied(self) -> int:
        """Total heals applied across all flows."""
        return sum(r.heals_used for r in self.flow_results.values())

    @property
    def defects_found(self) -> int:
        """Count of triage results with verdict APP_DEFECT."""
        return sum(
            1 for t in self.triage if t.verdict is TriageVerdict.APP_DEFECT
        )

    @property
    def untested_risk(self) -> list[tuple[str, str]]:
        """List of (description, severity) for untested flows, critical-first.

        This is built from gaps and sorted by severity (critical first).
        """
        result = []
        for gap in self.gaps:
            result.append((gap.description, gap.severity.value))
        # Sort by severity order (critical=0, serious=1, moderate=2, minor=3)
        result.sort(key=lambda x: Severity(x[1]).order)
        return result

    @property
    def summary_line(self) -> str:
        """Return a summary line.

        Shape: "4/5 flows passed, 2 heals applied, 1 defect found, 3 gaps remaining, $0.0021, 214s"
        Omit clauses that are zero; always include flows and cost.
        """
        parts = []

        # Flows (always included)
        parts.append(f"{self.flows_passed}/{self.flows_total} flows passed")

        # Heals applied
        if self.heals_applied > 0:
            heal_label = "heal" if self.heals_applied == 1 else "heals"
            parts.append(f"{self.heals_applied} {heal_label} applied")

        # Defects found
        if self.defects_found > 0:
            defect_label = "defect" if self.defects_found == 1 else "defects"
            parts.append(f"{self.defects_found} {defect_label} found")

        # Gaps remaining
        if self.gaps:
            gap_label = "gap" if len(self.gaps) == 1 else "gaps"
            parts.append(f"{len(self.gaps)} {gap_label} remaining")

        # Cost (always included)
        parts.append(f"${self.cost_usd:.4f}")

        # Duration
        if self.duration_s > 0:
            parts.append(f"{self.duration_s:.0f}s")

        return ", ".join(parts)

    def to_dict(self) -> dict:
        """Convert report to dict for JSON serialization."""
        return {
            "run_id": self.run_id,
            "url": self.url,
            "mode": self.mode,
            "intent": self.intent,
            "plan": self.plan.to_dict() if self.plan else None,
            "flow_results": {
                flow_id: result.to_dict() for flow_id, result in self.flow_results.items()
            },
            "gaps": [gap.to_dict() for gap in self.gaps],
            "triage": [t.to_dict() for t in self.triage],
            "decisions": [d.to_dict() for d in self.decisions],
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "cost_usd": self.cost_usd,
            "duration_s": self.duration_s,
            "generated_files": self.generated_files,
        }


def render_pipeline_text(report: PipelineReport) -> str:
    """Render a plain ASCII pipeline report."""
    lines = []

    # Header
    lines.append(f"Run ID: {report.run_id}")
    lines.append(f"URL: {report.url}")
    lines.append(f"Mode: {report.mode}")
    if report.duration_s > 0:
        lines.append(f"Duration: {report.duration_s:.0f}s")
    lines.append(f"Cost: ${report.cost_usd:.4f}")
    lines.append("")

    # ESCALATED block (if escalated, right after header)
    if report.escalated:
        lines.append("=" * 80)
        lines.append("ESCALATED: " + (report.escalation_reason or "No reason provided"))
        lines.append("=" * 80)
        lines.append("")

    # SCENARIOS COVERED
    if report.flow_results:
        lines.append("SCENARIOS COVERED:")
        lines.append("-" * 80)
        for flow_id, result in report.flow_results.items():
            status_mark = "[PASS]" if result.status == "passed" else "[FAIL]"
            # Try to get flow name from plan if available
            flow_name = flow_id
            if report.plan:
                for flow in report.plan.flows:
                    if flow.id == flow_id:
                        flow_name = flow.name
                        break
            step_count = len(result.results)
            passed_count = sum(1 for sr in result.results if sr.status == "passed")
            lines.append(
                f"  {status_mark} {flow_name} ({passed_count}/{step_count} steps)"
            )
        lines.append("")

    # FAILURES AND DEFECTS
    if report.triage:
        lines.append("FAILURES AND DEFECTS:")
        lines.append("-" * 80)
        for tri in report.triage:
            lines.append(f"  Verdict: {tri.verdict.value}")
            lines.append(f"  Step: {tri.step_id}")
            lines.append(f"  Confidence: {tri.confidence:.2f}")
            lines.append(f"  Reasoning: {tri.reasoning}")
            lines.append("")
        if report.triage:
            lines.pop()  # Remove last blank line after this section

    # HEALER ACTIONS
    heals_by_flow = {}
    for flow_id, result in report.flow_results.items():
        if result.heal_proposals:
            heals_by_flow[flow_id] = result.heal_proposals

    if heals_by_flow:
        lines.append("HEALER ACTIONS:")
        lines.append("-" * 80)
        for flow_id, heal_proposals in heals_by_flow.items():
            for heal in heal_proposals:
                old_str = ""
                if heal.old:
                    old_str = f"{heal.old.strategy}:{heal.old.value} → "
                new_str = f"{heal.new.strategy}:{heal.new.value}"
                lines.append(f"  {heal.step_id}: {old_str}{new_str}")
                lines.append(f"    Confidence: {heal.confidence:.2f}")
                lines.append(f"    Reasoning: {heal.reasoning}")
        lines.append("")

    # COVERAGE GAPS REMAINING
    if report.gaps:
        lines.append("COVERAGE GAPS REMAINING:")
        lines.append("-" * 80)
        for gap in report.gaps:
            lines.append(f"  [{gap.severity.value}] {gap.kind}: {gap.description}")
            lines.append(f"    Evidence: {gap.evidence}")
        lines.append("")

    # UNTESTED FLOW RISK
    if report.untested_risk:
        lines.append("UNTESTED FLOW RISK:")
        lines.append("-" * 80)
        for description, severity in report.untested_risk:
            lines.append(f"  [{severity}] {description}")
        lines.append("")

    # DECISIONS
    if report.decisions:
        lines.append("DECISIONS:")
        lines.append("-" * 80)
        for decision in report.decisions:
            lines.append(f"  {decision.stage.value} -> {decision.next_stage.value}")
            lines.append(f"    Verdict: {decision.verdict}")
            lines.append(f"    Reason: {decision.reason}")
        lines.append("")

    # Final summary line
    lines.append(report.summary_line)

    return "\n".join(lines)


def render_pipeline_html(report: PipelineReport) -> str:
    """Render a complete self-contained HTML pipeline report."""
    # CSS custom properties for light theme (default on :root)
    # Override in @media (prefers-color-scheme: dark)
    html_content = """<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Quality Report - {run_id}</title>
    <style>
        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --text-primary: #000000;
            --text-secondary: #666666;
            --border: #cccccc;
            --pass-bg: #d4edda;
            --pass-text: #155724;
            --pass-border: #c3e6cb;
            --fail-bg: #f8d7da;
            --fail-text: #721c24;
            --fail-border: #f5c6cb;
            --critical-bg: #fff3cd;
            --critical-text: #856404;
            --critical-border: #ffeaa7;
            --serious-bg: #f8d7da;
            --serious-text: #721c24;
            --moderate-bg: #d1ecf1;
            --moderate-text: #0c5460;
            --minor-bg: #e8f4f8;
            --minor-text: #164e63;
            --healed-bg: #ffeaa7;
            --healed-text: #856404;
        }}

        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                --bg-primary: #1e1e1e;
                --bg-secondary: #2d2d2d;
                --text-primary: #e0e0e0;
                --text-secondary: #999999;
                --border: #444444;
                --pass-bg: #1e5631;
                --pass-text: #90ee90;
                --pass-border: #2d6e3f;
                --fail-bg: #5d1f1f;
                --fail-text: #ff6b6b;
                --fail-border: #7d2f2f;
                --critical-bg: #664400;
                --critical-text: #ffcc66;
                --critical-border: #8b5a00;
                --serious-bg: #5d1f1f;
                --serious-text: #ff6b6b;
                --moderate-bg: #1f4d5d;
                --moderate-text: #66ccff;
                --minor-bg: #2d3d4d;
                --minor-text: #99ccff;
                --healed-bg: #664400;
                --healed-text: #ffcc66;
            }}
        }}

        @media (prefers-color-scheme: dark) {{
            :root[data-theme="dark"] {{
                --bg-primary: #1e1e1e;
                --bg-secondary: #2d2d2d;
                --text-primary: #e0e0e0;
                --text-secondary: #999999;
                --border: #444444;
                --pass-bg: #1e5631;
                --pass-text: #90ee90;
                --pass-border: #2d6e3f;
                --fail-bg: #5d1f1f;
                --fail-text: #ff6b6b;
                --fail-border: #7d2f2f;
                --critical-bg: #664400;
                --critical-text: #ffcc66;
                --critical-border: #8b5a00;
                --serious-bg: #5d1f1f;
                --serious-text: #ff6b6b;
                --moderate-bg: #1f4d5d;
                --moderate-text: #66ccff;
                --minor-bg: #2d3d4d;
                --minor-text: #99ccff;
                --healed-bg: #664400;
                --healed-text: #ffcc66;
            }}
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}

        h2 {{
            margin: 30px 0 15px 0;
            font-size: 18px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 8px;
        }}

        .header {{
            margin-bottom: 30px;
        }}

        .summary-line {{
            font-size: 16px;
            font-weight: bold;
            margin-top: 20px;
            padding: 15px;
            background: var(--bg-secondary);
            border-left: 4px solid var(--border);
        }}

        .escalated {{
            background: var(--critical-bg);
            color: var(--critical-text);
            border: 2px solid var(--critical-border);
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
            font-weight: bold;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }}

        th {{
            background: var(--bg-secondary);
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid var(--border);
            font-weight: bold;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        .pass {{
            background: var(--pass-bg);
            color: var(--pass-text);
        }}

        .fail {{
            background: var(--fail-bg);
            color: var(--fail-text);
        }}

        .critical {{
            background: var(--critical-bg);
            color: var(--critical-text);
        }}

        .serious {{
            background: var(--serious-bg);
            color: var(--serious-text);
        }}

        .moderate {{
            background: var(--moderate-bg);
            color: var(--moderate-text);
        }}

        .minor {{
            background: var(--minor-bg);
            color: var(--minor-text);
        }}

        .healed {{
            background: var(--healed-bg);
            color: var(--healed-text);
        }}

        .status-pass::before {{
            content: "[PASS] ";
        }}

        .status-fail::before {{
            content: "[FAIL] ";
        }}

        code, pre {{
            font-family: "Monaco", "Menlo", "Consolas", "Courier New", monospace;
            font-size: 13px;
        }}

        pre {{
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            border: 1px solid var(--border);
        }}

        details {{
            margin: 10px 0;
            padding: 10px;
            background: var(--bg-secondary);
            border-radius: 4px;
        }}

        summary {{
            cursor: pointer;
            font-weight: bold;
            user-select: none;
        }}

        summary:hover {{
            text-decoration: underline;
        }}

        .meta {{
            color: var(--text-secondary);
            font-size: 13px;
            margin-bottom: 20px;
        }}

        .table-wrapper {{
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Test Quality Report</h1>
            <div class="meta">
                <div>Run ID: <code>{run_id}</code></div>
                <div>URL: <code>{url}</code></div>
                <div>Mode: {mode}</div>
                {duration_html}{cost_html}
            </div>
        </div>

        {escalated_block}

        {scenarios_section}

        {failures_section}

        {heals_section}

        {gaps_section}

        {untested_risk_section}

        {decisions_section}

        <div class="summary-line">{summary_line}</div>
    </div>
</body>
</html>
"""

    # Build dynamic sections
    duration_html = (
        f"<div>Duration: {report.duration_s:.0f}s</div>"
        if report.duration_s > 0
        else ""
    )
    cost_html = f"<div>Cost: ${report.cost_usd:.4f}</div>"

    escalated_block = ""
    if report.escalated:
        reason = html.escape(report.escalation_reason or "No reason provided")
        escalated_block = f'<div class="escalated">ESCALATED: {reason}</div>'

    # Scenarios section
    scenarios_section = ""
    if report.flow_results:
        rows = []
        for flow_id, result in report.flow_results.items():
            status_class = "pass" if result.status == "passed" else "fail"
            status_text = "status-pass" if result.status == "passed" else "status-fail"
            flow_name = flow_id
            if report.plan:
                for flow in report.plan.flows:
                    if flow.id == flow_id:
                        flow_name = html.escape(flow.name)
                        break
            step_count = len(result.results)
            passed_count = sum(1 for sr in result.results if sr.status == "passed")
            rows.append(
                f'<tr class="{status_class}"><td class="{status_text}">{html.escape(flow_name)}</td>'
                f"<td>{passed_count}/{step_count} steps</td></tr>"
            )
        table_rows = "\n".join(rows)
        scenarios_section = f"""<h2>Scenarios Covered</h2>
<div class="table-wrapper">
<table>
<thead><tr><th>Flow</th><th>Steps</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""

    # Failures and defects section
    failures_section = ""
    if report.triage:
        rows = []
        for tri in report.triage:
            verdict_class = (
                "fail" if tri.verdict is TriageVerdict.APP_DEFECT else "moderate"
            )
            rows.append(
                f'<tr><td>{html.escape(tri.step_id)}</td>'
                f'<td class="{verdict_class}">{html.escape(tri.verdict.value)}</td>'
                f'<td>{tri.confidence:.2f}</td>'
                f'<td>{html.escape(tri.reasoning)}</td></tr>'
            )
        table_rows = "\n".join(rows)
        failures_section = f"""<h2>Failures and Defects</h2>
<div class="table-wrapper">
<table>
<thead><tr><th>Step ID</th><th>Verdict</th><th>Confidence</th><th>Reasoning</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""

    # Healer actions section
    heals_section = ""
    heals_by_flow = {}
    for flow_id, result in report.flow_results.items():
        if result.heal_proposals:
            heals_by_flow[flow_id] = result.heal_proposals

    if heals_by_flow:
        rows = []
        for flow_id, heal_proposals in heals_by_flow.items():
            for heal in heal_proposals:
                old_str = ""
                if heal.old:
                    old_str = f"{heal.old.strategy}:{heal.old.value} → "
                new_str = f"{heal.new.strategy}:{heal.new.value}"
                rows.append(
                    f'<tr><td><code>{html.escape(heal.step_id)}</code></td>'
                    f'<td><code>{html.escape(old_str + new_str)}</code></td>'
                    f'<td>{heal.confidence:.2f}</td>'
                    f'<td>{html.escape(heal.reasoning)}</td></tr>'
                )
        table_rows = "\n".join(rows)
        heals_section = f"""<h2>Healer Actions</h2>
<div class="table-wrapper">
<table>
<thead><tr><th>Step</th><th>Selector Change</th><th>Confidence</th><th>Reasoning</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""

    # Coverage gaps section
    gaps_section = ""
    if report.gaps:
        rows = []
        for gap in report.gaps:
            severity_class = gap.severity.value
            rows.append(
                f'<tr class="{severity_class}"><td class="{severity_class}">{html.escape(gap.severity.value)}</td>'
                f'<td>{html.escape(gap.kind)}</td>'
                f'<td>{html.escape(gap.description)}</td>'
                f'<td><code>{html.escape(gap.evidence)}</code></td></tr>'
            )
        table_rows = "\n".join(rows)
        gaps_section = f"""<h2>Coverage Gaps Remaining</h2>
<div class="table-wrapper">
<table>
<thead><tr><th>Severity</th><th>Kind</th><th>Description</th><th>Evidence</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""

    # Untested flow risk section
    untested_risk_section = ""
    if report.untested_risk:
        rows = []
        for description, severity in report.untested_risk:
            severity_class = severity
            rows.append(
                f'<tr class="{severity_class}"><td class="{severity_class}">{html.escape(severity)}</td>'
                f'<td>{html.escape(description)}</td></tr>'
            )
        table_rows = "\n".join(rows)
        untested_risk_section = f"""<h2>Untested Flow Risk</h2>
<div class="table-wrapper">
<table>
<thead><tr><th>Severity</th><th>Description</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""

    # Decisions section
    decisions_section = ""
    if report.decisions:
        rows = []
        for decision in report.decisions:
            evidence_json = json.dumps(decision.evidence, indent=2)
            rows.append(
                f'<tr><td>{html.escape(decision.stage.value)}</td>'
                f'<td>{html.escape(decision.verdict)}</td>'
                f'<td>{html.escape(decision.reason)}</td>'
                f'<td>{html.escape(decision.next_stage.value)}</td>'
                f'<td><details><summary>Evidence</summary><pre>{html.escape(evidence_json)}</pre></details></td></tr>'
            )
        table_rows = "\n".join(rows)
        decisions_section = f"""<h2>Decisions</h2>
<div class="table-wrapper">
<table>
<thead><tr><th>Stage</th><th>Verdict</th><th>Reason</th><th>Next Stage</th><th>Evidence</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""

    # Format the final HTML
    return html_content.format(
        run_id=html.escape(report.run_id),
        url=html.escape(report.url),
        mode=html.escape(report.mode),
        duration_html=duration_html,
        cost_html=cost_html,
        escalated_block=escalated_block,
        scenarios_section=scenarios_section,
        failures_section=failures_section,
        heals_section=heals_section,
        gaps_section=gaps_section,
        untested_risk_section=untested_risk_section,
        decisions_section=decisions_section,
        summary_line=html.escape(report.summary_line),
    )


def write_pipeline_report(
    report: PipelineReport, out_dir: str | Path = "artifacts"
) -> dict[str, Path]:
    """Write pipeline report to JSON, TXT, and HTML files.

    Returns a dict with keys 'json', 'txt', 'html' mapped to Path objects.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Base filename
    base_filename = report.run_id

    # Write JSON
    json_path = out_path / f"{base_filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Write TXT
    txt_path = out_path / f"{base_filename}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_pipeline_text(report))

    # Write HTML
    html_path = out_path / f"{base_filename}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_pipeline_html(report))

    return {
        "json": json_path,
        "txt": txt_path,
        "html": html_path,
    }


def render_text(test: CompiledTest, result: RunResult) -> str:
    """
    Render a monospace text report with:
    - header with test id and intent
    - per-step table (columns: step, status, kind, verb, target, source, duration)
    - FINDINGS section (grouped by severity, critical first)
    - HEALS PENDING APPROVAL section (if any)
    - final summary line
    """
    lines = []

    # Header
    lines.append(f"Test: {test.id}")
    lines.append(f"Intent: {test.intent}")
    lines.append("")

    # Per-step table
    lines.append("Step Results:")
    lines.append("-" * 80)
    lines.append(
        f"{'Step':<8} {'Status':<10} {'Kind':<10} {'Verb':<12} {'Target':<20} {'Source':<10} {'Duration':<10}"
    )
    lines.append("-" * 80)

    for step_result in result.results:
        step = next((s for s in test.steps if s.id == step_result.step_id), None)
        if step:
            lines.append(
                f"{step.id:<8} {step_result.status:<10} {step.kind.value:<10} "
                f"{step.verb:<12} {step.target[:20]:<20} {step_result.source.value:<10} "
                f"{step_result.duration_ms:.0f}ms"
            )

    lines.append("-" * 80)
    lines.append("")

    # FINDINGS section (omit if empty)
    if result.findings:
        lines.append("FINDINGS:")
        # Sort by severity (critical first)
        sorted_findings = sorted(result.findings, key=lambda f: f.severity.order)
        for finding in sorted_findings:
            target_str = f" ({finding.target})" if finding.target else ""
            lines.append(
                f"[{finding.severity.value}] {finding.rule} — {finding.message}{target_str}"
            )
        lines.append("")

    # HEALS PENDING APPROVAL section (omit if empty)
    if result.heal_proposals:
        lines.append("HEALS PENDING APPROVAL:")
        for heal in result.heal_proposals:
            old_str = ""
            if heal.old:
                old_str = f"{heal.old.strategy}:{heal.old.value} → "
            new_str = f"{heal.new.strategy}:{heal.new.value}"
            lines.append(f"  {heal.step_id}: {old_str}{new_str}")
            lines.append(f"    Confidence: {heal.confidence:.2f}")
            lines.append(f"    Reasoning: {heal.reasoning}")
        lines.append("")

    # Final summary line
    lines.append(result.summary_line)

    return "\n".join(lines)


def render_json(test: CompiledTest, result: RunResult) -> dict[str, Any]:
    """
    Render JSON with test dict, result dict, and generated_at timestamp.
    """
    return {
        "test": test.to_dict(),
        "result": result.to_dict(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def write_report(
    test: CompiledTest, result: RunResult, out_dir: str | Path = "artifacts"
) -> Path:
    """
    Write a JSON report to <out_dir>/<test.id>-<UTC timestamp YYYYmmdd-HHMMSS>.json
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with UTC timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{test.id}-{timestamp}.json"
    file_path = out_path / filename

    # Write JSON report
    report_data = render_json(test, result)
    with open(file_path, "w") as f:
        json.dump(report_data, f, indent=2)

    return file_path
