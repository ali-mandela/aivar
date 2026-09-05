"""HTTP surface for the orchestration agent.

Design notes:

* The endpoints are declared with `def`, not `async def`. `run_pipeline` is
  synchronous and drives Playwright's sync API, which cannot run inside an
  asyncio event loop. FastAPI dispatches sync handlers to a worker thread,
  which is exactly what is needed here.

* A real run takes 30-60 seconds. `POST /runs` blocks for that by default so a
  plain curl works with no polling; pass `"background": true` to get a job id
  back immediately and poll `GET /jobs/{job_id}` instead.

* The job registry is in-memory and therefore per-process: it tracks work that
  is still running. Finished runs live in Postgres and on disk, which are the
  durable record.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from app.orchestrator import OrchestratorConfig, run_pipeline
from app.paths import resolve_out_dir

logger = logging.getLogger("aivar")

router = APIRouter()

# job_id -> {"status", "run_id", "error", "summary"}
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """A pipeline run. `url` is the only required field, per the brief."""

    url: str = Field(..., description="Application URL. The only required input.")
    username: str | None = Field(None, description="Used if the app has a login.")
    password: str | None = Field(None, description="Never logged or echoed back.")
    intent: str | None = Field(
        None,
        description="Optional focus, e.g. 'checkout and auth'. Blank means sweep everything.",
    )
    prd_path: str | None = Field(None, description="Path to a markdown/text requirements doc.")

    max_flows: int = Field(4, ge=1, le=12)
    max_pages: int = Field(5, ge=1, le=20)
    max_cost_usd: float = Field(0.50, gt=0)
    max_seconds: int = Field(300, ge=30, le=1800)

    headless: bool = True
    safe_mode: bool = Field(
        False,
        description="Fill forms but never press submit. Use on somebody else's production site.",
    )
    heal: bool = True
    background: bool = Field(False, description="Return a job id immediately instead of waiting.")

    @field_validator("url")
    @classmethod
    def _url_must_be_http(cls, v: str) -> str:
        v = (v or "").strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    def mode(self) -> str:
        if self.prd_path:
            return "spec_led"
        return "focused" if self.intent else "sweep"

    def to_config(self) -> OrchestratorConfig:
        return OrchestratorConfig(
            max_flows=self.max_flows,
            max_explore_pages=self.max_pages,
            max_cost_usd=self.max_cost_usd,
            max_pipeline_seconds=self.max_seconds,
            headless=self.headless,
            safe_mode=self.safe_mode,
            heal=self.heal,
        )


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["running"]
    poll: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _execute(req: RunRequest) -> dict[str, Any]:
    """Run the pipeline and shape the response. Never raises for a failed run:
    an escalation is a legitimate outcome with a report, not an HTTP error."""
    report, state = run_pipeline(
        req.url,
        username=req.username,
        password=req.password,
        intent=req.intent,
        prd_path=req.prd_path,
        config=req.to_config(),
    )
    return {
        "run_id": report.run_id,
        "url": report.url,
        "mode": report.mode,
        "escalated": report.escalated,
        "escalation_reason": report.escalation_reason,
        "summary": report.summary_line,
        "flows": {
            "total": report.flows_total,
            "passed": report.flows_passed,
            "failed": report.flows_failed,
        },
        "gaps": [g.to_dict() for g in report.gaps],
        "untested_flow_risk": [
            {"description": d, "severity": s} for d, s in report.untested_risk
        ],
        "heals_applied": report.heals_applied,
        "defects_found": report.defects_found,
        "cost_usd": report.cost_usd,
        "duration_s": report.duration_s,
        "generated_files": report.generated_files,
        "decisions": [d.to_dict() for d in report.decisions],
        "links": {
            "report_html": f"/runs/{report.run_id}/report.html",
            "report_text": f"/runs/{report.run_id}/report.txt",
            "tests": f"/runs/{report.run_id}/tests",
        },
    }


def _run_in_background(job_id: str, req: RunRequest) -> None:
    try:
        result = _execute(req)
        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "status": "escalated" if result["escalated"] else "finished",
                "run_id": result["run_id"],
                "summary": result["summary"],
                "error": None,
            }
    except Exception as e:  # a crash here must not take the server down
        logger.exception("background run failed")
        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "status": "failed",
                "run_id": None,
                "summary": None,
                "error": f"{type(e).__name__}: {e}",
            }


def _artifact(run_id: str, suffix: str) -> Path:
    path = resolve_out_dir("artifacts") / f"{run_id}.{suffix}"
    if not path.exists():
        raise HTTPException(404, f"No {suffix} artifact for run {run_id}")
    return path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/runs", summary="Run the full pipeline against a URL")
def create_run(req: RunRequest) -> Any:
    """Explore, plan, critique, generate, execute, heal and report.

    Blocks until the pipeline finishes (typically 30-60s). Set
    `"background": true` to get a job id back straight away.
    """
    if req.background:
        job_id = uuid.uuid4().hex[:12]
        with _JOBS_LOCK:
            _JOBS[job_id] = {"status": "running", "run_id": None, "summary": None, "error": None}
        threading.Thread(
            target=_run_in_background, args=(job_id, req), daemon=True
        ).start()
        return JobAccepted(job_id=job_id, status="running", poll=f"/jobs/{job_id}")

    try:
        return _execute(req)
    except Exception as e:
        logger.exception("pipeline failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}") from None


@router.get("/jobs/{job_id}", summary="Status of a background run")
def get_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job {job_id}")
    out = dict(job, job_id=job_id)
    if job.get("run_id"):
        out["run"] = f"/runs/{job['run_id']}"
    return out


@router.get("/runs", summary="Run history")
def list_runs(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
    try:
        from app.store import list_runs as store_list

        runs = store_list(limit=limit)
    except Exception as e:
        # History is a convenience; the files on disk are the durable record.
        return {"runs": [], "count": 0, "store_error": f"{type(e).__name__}: {e}"}

    return {
        "count": len(runs),
        "runs": [
            {
                "run_id": r.run_id,
                "url": r.url,
                "mode": r.mode,
                "escalated": r.escalated,
                "flows_passed": r.flows_passed,
                "flows_total": r.flows_total,
                "gaps": r.gaps_total,
                "cost_usd": r.cost_usd,
                "duration_s": r.duration_s,
                "summary": r.summary_line,
                "created_at": str(r.created_at),
            }
            for r in runs
        ],
    }


@router.get("/runs/{run_id}", summary="One run, with its decision ledger and gaps")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        from app.store import get_run as store_get

        detail = store_get(run_id)
    except Exception as e:
        raise HTTPException(503, f"Store unavailable: {type(e).__name__}: {e}") from None
    if not detail:
        raise HTTPException(404, f"Unknown run {run_id}")
    return detail


@router.get("/runs/{run_id}/report.html", response_class=HTMLResponse,
            summary="The rendered test quality report")
def get_report_html(run_id: str) -> HTMLResponse:
    return HTMLResponse(_artifact(run_id, "html").read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/report.txt", response_class=PlainTextResponse,
            summary="The report as plain text")
def get_report_text(run_id: str) -> PlainTextResponse:
    return PlainTextResponse(_artifact(run_id, "txt").read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/tests", summary="The pytest files the agent generated")
def get_generated_tests(run_id: str) -> dict[str, Any]:
    """Returns the generated suite as filename -> source.

    These are ordinary pytest files: `pytest tests/generated` runs them with no
    part of this agent involved.
    """
    gen_dir = resolve_out_dir("tests/generated")
    if not gen_dir.exists():
        return {"run_id": run_id, "count": 0, "files": {}}
    files = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(gen_dir.glob("*.py"))
    }
    return {
        "run_id": run_id,
        "count": len(files),
        "directory": str(gen_dir),
        "note": "run these with: pytest tests/generated",
        "files": files,
    }
