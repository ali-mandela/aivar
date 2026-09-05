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

import json
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from app.contracts.contracts import Decision
from app.llm import LLMConfig, LLMError
from app.orchestrator import OrchestratorConfig, run_pipeline
from app.paths import resolve_out_dir

logger = logging.getLogger("aivar")

router = APIRouter()

# job_id -> {"status", "stage", "run_id", "error", "summary", "decisions", "result"}
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _job_file(job_id: str) -> Path:
    return resolve_out_dir("artifacts/jobs") / f"{job_id}.json"


def _persist_job(job_id: str, job: dict[str, Any]) -> None:
    """Mirror a job to disk so a restart does not erase it.

    The registry is per-process, and a development server restarts for all sorts
    of reasons -- including, until recently, the run writing its own generated
    tests. When that happened the client polling for a job got a bare 404 and
    the run simply vanished, mid-flight, with no record that it had ever
    started. A copy on disk means the poll can still say what happened.

    Best-effort: failing to write a status file must never fail the run it is
    describing.
    """
    try:
        path = _job_file(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, default=str), encoding="utf-8")
    except Exception:
        logger.warning("could not persist job %s", job_id, exc_info=True)


def _load_job(job_id: str) -> dict[str, Any] | None:
    """Read a job this process does not have in memory.

    Reaching disk at all means the job was started by a different process, so a
    status of "running" is necessarily stale -- whatever thread was driving it
    died with that process. Reported as "interrupted" rather than left claiming
    to be in progress, which would have a client poll forever.
    """
    try:
        path = _job_file(job_id)
        if not path.exists():
            return None
        job = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if job.get("status") == "running":
        job = dict(
            job,
            status="interrupted",
            error="The server restarted while this run was in progress, so it "
            "did not finish. Nothing was corrupted -- start it again.",
        )
    return job


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

    llm_provider: str | None = Field(
        None,
        description="openrouter | google | sarvam. Defaults to AIVAR_LLM_PROVIDER, "
        "else the first provider with a key configured.",
    )
    llm_models: list[str] | None = Field(
        None,
        description="Override the model list for that provider, tried in order.",
    )

    @field_validator("url")
    @classmethod
    def _url_must_be_http(cls, v: str) -> str:
        v = (v or "").strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    @field_validator("llm_provider")
    @classmethod
    def _known_provider(cls, v: str | None) -> str | None:
        # Reject an unknown name here so it surfaces as a 422 on the field
        # rather than a 500 from deep inside the pipeline.
        from app.llm import parse_provider

        if v is None or not v.strip():
            return None
        parse_provider(v)
        return v.strip()

    def mode(self) -> str:
        if self.prd_path:
            return "spec_led"
        return "focused" if self.intent else "sweep"

    def to_llm_config(self) -> LLMConfig:
        """Resolve the provider for this run. Falls back to the server's env
        config when the request names neither a provider nor models."""
        return LLMConfig.from_env(
            provider=self.llm_provider,
            models=tuple(self.llm_models) if self.llm_models else None,
        )

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


def _tests_dir(run_id: str) -> Path:
    """Where this run's copy of its generated suite lives."""
    return resolve_out_dir("artifacts") / run_id / "tests"


def _snapshot_tests(run_id: str, generated_files: list[str]) -> None:
    """Copy this run's generated suite in beside its report.

    `tests/generated` is overwritten by every subsequent run, so without a
    per-run copy the history view serves the newest suite under an older run's
    id -- the files look right and belong to a different run. The live
    directory is left exactly where it is: `pytest tests/generated` is part of
    the brief and keeps working.

    A failed snapshot costs history, never the run.
    """
    if not generated_files:
        return
    try:
        sources = [Path(p) for p in generated_files]
        # conftest.py is written once per directory rather than per flow, so it
        # never appears in generated_files -- and the snapshot cannot run
        # without it.
        conftest = sources[0].parent / "conftest.py"
        if conftest.exists():
            sources.append(conftest)

        dest = _tests_dir(run_id)
        dest.mkdir(parents=True, exist_ok=True)
        for src in sources:
            if src.exists():
                shutil.copy2(src, dest / src.name)
    except Exception:
        logger.warning("failed to snapshot tests for run %s", run_id, exc_info=True)


def _execute(
    req: RunRequest, on_decision: Callable[[Decision], None] | None = None
) -> dict[str, Any]:
    """Run the pipeline and shape the response. Never raises for a failed run:
    an escalation is a legitimate outcome with a report, not an HTTP error."""
    llm = req.to_llm_config()
    report, state = run_pipeline(
        req.url,
        username=req.username,
        password=req.password,
        intent=req.intent,
        prd_path=req.prd_path,
        config=req.to_config(),
        llm_config=llm,
        on_decision=on_decision,
    )
    _snapshot_tests(report.run_id, report.generated_files)
    return {
        "run_id": report.run_id,
        "url": report.url,
        "mode": report.mode,
        "llm": llm.describe(),
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
            "report_json": f"/runs/{report.run_id}/report.json",
            "tests": f"/runs/{report.run_id}/tests",
        },
    }


def _run_in_background(job_id: str, req: RunRequest) -> None:
    def observe(decision: Decision) -> None:
        """Publish each decision as it happens, so a poller can narrate the run
        rather than staring at a spinner for a minute."""
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            job["stage"] = decision.next_stage.value
            job["decisions"].append(decision.to_dict())
            snapshot = dict(job)
        _persist_job(job_id, snapshot)

    try:
        result = _execute(req, on_decision=observe)
        with _JOBS_LOCK:
            job = _JOBS.get(job_id, {})
            job.update(
                status="escalated" if result["escalated"] else "finished",
                run_id=result["run_id"],
                summary=result["summary"],
                stage="escalated" if result["escalated"] else "done",
                error=None,
                # Carried so a poller needs no second request to render the run.
                result=result,
            )
            _JOBS[job_id] = job
            snapshot = dict(job)
        _persist_job(job_id, snapshot)
    except Exception as e:  # a crash here must not take the server down
        logger.exception("background run failed")
        with _JOBS_LOCK:
            job = _JOBS.get(job_id, {})
            job.update(
                status="failed",
                run_id=None,
                summary=None,
                error=f"{type(e).__name__}: {e}",
            )
            _JOBS[job_id] = job
            snapshot = dict(job)
        _persist_job(job_id, snapshot)


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
            _JOBS[job_id] = _new_job = {
                "status": "running",
                "stage": "explore",
                "run_id": None,
                "summary": None,
                "error": None,
                "decisions": [],
            }
        _persist_job(job_id, _new_job)
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
        # Not in this process. It may still have been recorded on disk by a
        # previous one, which is what a development restart looks like.
        job = _load_job(job_id)
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


@router.get("/runs/{run_id}/report.json", summary="The report as structured JSON")
def get_report_json(run_id: str) -> Any:
    """The same artifact the pipeline wrote to disk, served verbatim."""
    return json.loads(_artifact(run_id, "json").read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/tests", summary="The pytest files the agent generated")
def get_generated_tests(run_id: str) -> dict[str, Any]:
    """Returns this run's generated suite as filename -> source.

    Prefers the per-run snapshot taken when the run finished. `tests/generated`
    holds only the most recent suite, so for any earlier run it would answer
    with another run's files -- plausible-looking and wrong. When no snapshot
    exists (a run from before snapshots, or one whose copy failed) the live
    directory is served with `stale: true` so the caller can say so rather than
    quietly implying provenance it cannot support.

    These are ordinary pytest files: `pytest tests/generated` runs them with no
    part of this agent involved.
    """
    snapshot = _tests_dir(run_id)
    if snapshot.is_dir():
        source_dir, stale = snapshot, False
    else:
        source_dir, stale = resolve_out_dir("tests/generated"), True

    if not source_dir.is_dir():
        return {"run_id": run_id, "count": 0, "stale": stale, "files": {}}

    files = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(source_dir.glob("*.py"))
    }
    return {
        "run_id": run_id,
        "count": len(files),
        "directory": str(source_dir),
        "stale": stale,
        "note": "run these with: pytest tests/generated",
        "files": files,
    }


# ---------------------------------------------------------------------------
# PRD upload
#
# `prd_path` is a server-side path, which a browser cannot produce. This turns
# a picked file into one.
# ---------------------------------------------------------------------------

PRD_SUFFIXES = {".md", ".markdown", ".txt"}
MAX_PRD_BYTES = 1_000_000


@router.post("/prd", summary="Upload a requirements document for a spec-led run")
def upload_prd(file: UploadFile = File(...)) -> dict[str, Any]:
    """Store a PRD and return the `prd_path` to pass to POST /runs."""
    name = Path(file.filename or "prd.md").name  # never trust a client path
    suffix = Path(name).suffix.lower()
    if suffix not in PRD_SUFFIXES:
        raise HTTPException(
            400,
            f"PRD must be one of {sorted(PRD_SUFFIXES)}; got {suffix or 'no extension'}",
        )

    raw = file.file.read(MAX_PRD_BYTES + 1)
    if len(raw) > MAX_PRD_BYTES:
        raise HTTPException(413, f"PRD exceeds {MAX_PRD_BYTES // 1000}KB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "PRD must be UTF-8 text") from None

    dest_dir = resolve_out_dir("uploads")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex[:12]}-{name}"
    dest.write_text(text, encoding="utf-8")

    return {
        "prd_path": str(dest),
        "filename": name,
        "bytes": len(raw),
        "lines": text.count("\n") + 1,
    }
