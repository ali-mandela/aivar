"""Orchestration-layer contracts.

These types are the interfaces between the meta-agent and its sub-agents. They
live apart from `models.py` (which describes a single compiled test) because
they describe a *pipeline run*: a plan made of many flows, the verdicts of the
decision gates, and the ledger of everything the orchestrator decided.

Nothing here performs work. It exists so that the Planner, Critic, Generator,
Triager and Reporter can be built independently against a fixed shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from app.models.models import Severity, Step

# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


class PlanMode(str, Enum):
    """How the run was scoped, which decides what "enough coverage" means.

    The absence of an intent is not the absence of instructions -- it means
    "cover everything", and SWEEP is therefore the strictest mode, not the
    laziest one.
    """

    SWEEP = "sweep"  # url only: aim for breadth over everything discovered
    FOCUSED = "focused"  # a natural-language intent narrows the target
    SPEC_LED = "spec_led"  # a product document defines the target


class FlowKind(str, Enum):
    """What a flow is for. A plan of only HAPPY_PATH flows is a thin plan."""

    HAPPY_PATH = "happy_path"
    NEGATIVE = "negative"  # invalid input, rejected credentials
    EDGE_CASE = "edge_case"  # boundaries, empty states, long values
    ERROR_STATE = "error_state"  # the app's own error handling
    NAVIGATION = "navigation"  # reachability and link integrity


@dataclass(frozen=True)
class Flow:
    """One named user journey, before or after selector compilation."""

    id: str
    name: str  # human-readable, e.g. "Reject an invalid password"
    description: str
    kind: FlowKind
    steps: list[Step]
    entry_url: str | None = None

    @property
    def assertions(self) -> list[Step]:
        from app.models.models import StepKind

        return [s for s in self.steps if s.kind is StepKind.ASSERTION]

    @property
    def is_compiled(self) -> bool:
        """Every step has a selector, so this flow can be executed."""
        return all(s.selector is not None for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "kind": self.kind.value,
            "steps": [s.to_dict() for s in self.steps],
            **({"entry_url": self.entry_url} if self.entry_url else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Flow:
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            kind=FlowKind(d["kind"]),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            entry_url=d.get("entry_url"),
        )


@dataclass
class TestPlan:
    """What the Planner produced: many flows, not one."""

    # Stops pytest trying to collect this as a test class purely because its
    # name begins with "Test". It is a domain type, not a test suite.
    __test__ = False

    id: str
    mode: PlanMode
    flows: list[Flow]
    intent: str | None = None
    prd_path: str | None = None

    @property
    def flow_count(self) -> int:
        return len(self.flows)

    @property
    def kinds_covered(self) -> set[FlowKind]:
        return {f.kind for f in self.flows}

    @property
    def is_happy_path_only(self) -> bool:
        """A plan that only tests success is not a test plan."""
        return self.kinds_covered <= {FlowKind.HAPPY_PATH, FlowKind.NAVIGATION}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode.value,
            "intent": self.intent,
            "prd_path": self.prd_path,
            "flows": [f.to_dict() for f in self.flows],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TestPlan:
        return cls(
            id=d["id"],
            mode=PlanMode(d["mode"]),
            flows=[Flow.from_dict(f) for f in d.get("flows", [])],
            intent=d.get("intent"),
            prd_path=d.get("prd_path"),
        )


# --------------------------------------------------------------------------
# The coverage gate
# --------------------------------------------------------------------------


class CoverageVerdict(str, Enum):
    ACCEPT = "accept"
    REPLAN = "replan"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Gap:
    """Something that should have been tested and is not.

    `evidence` must point at what implies the gap -- a form the explorer found,
    a page nobody visits, a requirement in the PRD -- so a human can check the
    claim rather than trust it.
    """

    kind: str  # untested_form | untested_page | missing_error_state | prd_requirement | ...
    description: str
    evidence: str
    severity: Severity

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "description": self.description,
            "evidence": self.evidence,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Gap:
        return cls(
            kind=d["kind"],
            description=d["description"],
            evidence=d.get("evidence", ""),
            severity=Severity(d["severity"]),
        )


@dataclass
class CoverageAssessment:
    """The Critic's verdict on a plan, before anything is generated."""

    verdict: CoverageVerdict
    score: float  # 0.0 - 1.0
    gaps: list[Gap] = field(default_factory=list)
    reasoning: str = ""
    replan_instruction: str | None = None  # fed back to the Planner verbatim

    @property
    def blocking_gaps(self) -> list[Gap]:
        return [
            g for g in self.gaps if g.severity in (Severity.CRITICAL, Severity.SERIOUS)
        ]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "score": self.score,
            "gaps": [g.to_dict() for g in self.gaps],
            "reasoning": self.reasoning,
            "replan_instruction": self.replan_instruction,
        }


# --------------------------------------------------------------------------
# The triage gate
# --------------------------------------------------------------------------


class TriageVerdict(str, Enum):
    """Getting this wrong in one direction ships bugs.

    Classifying a real defect as a script issue means the agent repairs the test
    until it passes and the bug reaches production. The bias is therefore
    deliberately asymmetric and APP_DEFECT is never healable.
    """

    SCRIPT_ISSUE = "script_issue"  # locator drift -- may be healed
    APP_DEFECT = "app_defect"  # the application misbehaved -- never healed
    FLAKY = "flaky"  # timing or environment -- retry, do not heal


@dataclass(frozen=True)
class TriageResult:
    step_id: str
    verdict: TriageVerdict
    confidence: float
    reasoning: str

    @property
    def heal_eligible(self) -> bool:
        return self.verdict is TriageVerdict.SCRIPT_ISSUE

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


# --------------------------------------------------------------------------
# The orchestrator state machine
# --------------------------------------------------------------------------


class Stage(str, Enum):
    EXPLORE = "explore"
    PLAN = "plan"
    CRITIQUE = "critique"
    GENERATE = "generate"
    VALIDATE = "validate"
    EXECUTE = "execute"
    TRIAGE = "triage"
    HEAL = "heal"
    REPORT = "report"
    DONE = "done"
    ESCALATED = "escalated"


TERMINAL_STAGES = frozenset({Stage.DONE, Stage.ESCALATED})


@dataclass(frozen=True)
class Decision:
    """One transition, with the reason it was taken.

    The ordered list of these IS the test quality report and IS the demo
    narration. It is not logging; it is the system explaining itself.
    """

    stage: Stage
    verdict: str
    reason: str
    next_stage: Stage
    evidence: dict = field(default_factory=dict)
    at: str = ""

    @staticmethod
    def now(
        stage: Stage,
        verdict: str,
        reason: str,
        next_stage: Stage,
        evidence: dict | None = None,
    ) -> Decision:
        return Decision(
            stage=stage,
            verdict=verdict,
            reason=reason,
            next_stage=next_stage,
            evidence=evidence or {},
            at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "verdict": self.verdict,
            "reason": self.reason,
            "next_stage": self.next_stage.value,
            "evidence": self.evidence,
            "at": self.at,
        }
