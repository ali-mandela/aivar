from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Verb = Literal["click", "fill", "wait_visible"]
Strategy = Literal["role", "label", "placeholder", "text", "testid", "css"]


class StepKind(str, Enum):
    ACTION = "action"
    ASSERTION = "assertion"


class FindingKind(str, Enum):
    ACCESSIBILITY = "accessibility"
    DESIGN_TOKEN = "design_token"
    GEOMETRY = "geometry"
    VISUAL = "visual"
    DESIGN_SPEC = "design_spec"


class Severity(str, Enum):
    CRITICAL = "critical"
    SERIOUS = "serious"
    MODERATE = "moderate"
    MINOR = "minor"

    @property
    def order(self) -> int:
        """Return an int for sorting (critical=0, serious=1, moderate=2, minor=3)."""
        order_map = {
            Severity.CRITICAL: 0,
            Severity.SERIOUS: 1,
            Severity.MODERATE: 2,
            Severity.MINOR: 3,
        }
        return order_map[self]


class Source(str, Enum):
    CACHE = "cache"
    HEURISTIC = "heuristic"
    HEALED = "healed"
    NONE = "none"


class FailureKind(str, Enum):
    LOCATOR_NOT_FOUND = "locator_not_found"
    ACTION_FAILED = "action_failed"
    ASSERTION_FAILED = "assertion_failed"
    AGENT_ERROR = "agent_error"

    @property
    def heal_eligible(self) -> bool:
        return self is FailureKind.LOCATOR_NOT_FOUND

    @property
    def is_test_failure(self) -> bool:
        return self is not FailureKind.AGENT_ERROR


@dataclass(frozen=True)
class Selector:
    strategy: Strategy
    value: str
    role: str | None = None

    def to_dict(self) -> dict:
        result = {
            "strategy": self.strategy,
            "value": self.value,
        }
        if self.role is not None:
            result["role"] = self.role
        return result

    @classmethod
    def from_dict(cls, d: dict) -> Selector:
        return cls(
            strategy=d["strategy"],
            value=d["value"],
            role=d.get("role"),
        )


@dataclass(frozen=True)
class Step:
    id: str
    kind: StepKind
    verb: Verb
    target: str
    value: str | None = None
    selector: Selector | None = None

    @property
    def healable(self) -> bool:
        """Assertions are never healable — a failing assertion is a candidate bug, not a candidate repair."""
        return self.kind is StepKind.ACTION

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "kind": self.kind.value,
            "verb": self.verb,
            "target": self.target,
        }
        if self.value is not None:
            result["value"] = self.value
        if self.selector is not None:
            result["selector"] = self.selector.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: dict) -> Step:
        kind_str = d["kind"]
        if isinstance(kind_str, str):
            kind = StepKind(kind_str)
        else:
            kind = kind_str

        verb = d["verb"]
        selector_data = d.get("selector")
        selector = Selector.from_dict(selector_data) if selector_data else None

        return cls(
            id=d["id"],
            kind=kind,
            verb=verb,
            target=d["target"],
            value=d.get("value"),
            selector=selector,
        )


@dataclass(frozen=True)
class Finding:
    kind: FindingKind
    severity: Severity
    rule: str
    message: str
    target: str | None = None
    step_id: str | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "rule": self.rule,
            "message": self.message,
        }
        if self.target is not None:
            result["target"] = self.target
        if self.step_id is not None:
            result["step_id"] = self.step_id
        if self.details:
            result["details"] = self.details
        return result

    @classmethod
    def from_dict(cls, d: dict) -> Finding:
        kind_str = d["kind"]
        if isinstance(kind_str, str):
            kind = FindingKind(kind_str)
        else:
            kind = kind_str

        severity_str = d["severity"]
        if isinstance(severity_str, str):
            severity = Severity(severity_str)
        else:
            severity = severity_str

        return cls(
            kind=kind,
            severity=severity,
            rule=d["rule"],
            message=d["message"],
            target=d.get("target"),
            step_id=d.get("step_id"),
            details=d.get("details", {}),
        )


@dataclass(frozen=True)
class StepResult:
    step_id: str
    status: Literal["passed", "failed", "skipped"]
    source: Source
    duration_ms: float
    failure: FailureKind | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "source": self.source.value,
            "duration_ms": self.duration_ms,
            "failure": self.failure.value if self.failure else None,
            "error": self.error,
        }


@dataclass
class CompiledTest:
    id: str
    intent: str
    url: str
    steps: list[Step]
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "intent": self.intent,
            "url": self.url,
            "steps": [step.to_dict() for step in self.steps],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompiledTest:
        steps = [Step.from_dict(step_dict) for step_dict in d.get("steps", [])]
        return cls(
            id=d["id"],
            intent=d["intent"],
            url=d["url"],
            steps=steps,
            version=d.get("version", 1),
        )

    @property
    def assertions(self) -> list[Step]:
        return [step for step in self.steps if step.kind is StepKind.ASSERTION]


@dataclass
class RunResult:
    test_id: str
    status: Literal["passed", "failed", "error"]
    results: list[StepResult]
    cost_usd: float = 0.0
    heals_used: int = 0
    findings: list[Finding] = field(default_factory=list)
    heal_proposals: list[HealProposal] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "status": self.status,
            "results": [r.to_dict() for r in self.results],
            "cost_usd": self.cost_usd,
            "heals_used": self.heals_used,
            "findings": [f.to_dict() for f in self.findings],
            "heal_proposals": [hp.to_dict() for hp in self.heal_proposals],
        }

    @property
    def findings_by_severity(self) -> dict[Severity, int]:
        """Return a dict mapping each Severity to the count of findings at that severity."""
        result = {severity: 0 for severity in Severity}
        for finding in self.findings:
            result[finding.severity] += 1
        return result

    @property
    def heals_pending(self) -> int:
        """Return the number of heal proposals."""
        return len(self.heal_proposals)

    @property
    def summary_line(self) -> str:
        """Return a summary line in the exact shape."""
        parts = []

        # Status
        parts.append(self.status)

        # Heals pending
        if self.heal_proposals:
            heal_count = len(self.heal_proposals)
            heal_label = "heal" if heal_count == 1 else "heals"
            parts.append(f"{heal_count} {heal_label} pending approval")

        # Findings by kind (grouped as "design findings")
        if self.findings:
            finding_count = len(self.findings)
            finding_label = (
                "design finding" if finding_count == 1 else "design findings"
            )
            parts.append(f"{finding_count} {finding_label}")

        # Cost
        parts.append(f"${self.cost_usd:.4f}")

        return ", ".join(parts)

    @staticmethod
    def from_results(
        test_id: str,
        results: list[StepResult],
        cost_usd: float = 0.0,
        heals_used: int = 0,
        findings: list[Finding] | None = None,
        heal_proposals: list[HealProposal] | None = None,
    ) -> RunResult:
        if any(r.failure is FailureKind.AGENT_ERROR for r in results):
            status = "error"
        elif any(r.status == "failed" for r in results):
            status = "failed"
        else:
            status = "passed"
        return RunResult(
            test_id=test_id,
            status=status,
            results=results,
            cost_usd=cost_usd,
            heals_used=heals_used,
            findings=findings or [],
            heal_proposals=heal_proposals or [],
        )


@dataclass(frozen=True)
class HealProposal:
    test_id: str
    step_id: str
    new: Selector
    confidence: float
    reasoning: str
    semantic_match: bool
    old: Selector | None = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "step_id": self.step_id,
            "new": self.new.to_dict(),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "semantic_match": self.semantic_match,
            "old": self.old.to_dict() if self.old else None,
        }
