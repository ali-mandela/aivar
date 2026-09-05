"""Persist pipeline runs to Postgres.

The JSON/HTML/pytest artifacts on disk are the source of truth; the database is
a convenience layer for querying and visualizing run history.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from app.envfile import load_dotenv
from app.report import PipelineReport

logger = logging.getLogger("aivar")


class StoreError(Exception):
    """Error from the store layer."""
    pass


def get_dsn() -> str:
    """Load AIVAR_DB_URL from environment.

    Calls load_dotenv() first, then reads the env var.
    Raises StoreError if AIVAR_DB_URL is not set.
    """
    load_dotenv()
    dsn = os.environ.get("AIVAR_DB_URL")
    if not dsn:
        raise StoreError("AIVAR_DB_URL not set in environment")
    return dsn


@contextmanager
def connect(dsn: str | None = None):
    """Context manager for a psycopg connection.

    Args:
        dsn: Database URL. If None, calls get_dsn() to load from env.

    Yields:
        psycopg.Connection

    Raises:
        StoreError: On connection failure.
    """
    if dsn is None:
        dsn = get_dsn()

    try:
        conn = psycopg.connect(dsn, connect_timeout=20)
        try:
            yield conn
        finally:
            conn.close()
    except psycopg.Error as e:
        raise StoreError(f"Failed to connect to database: {e}") from e
    except Exception as e:
        raise StoreError(f"Unexpected error connecting to database: {e}") from e


def init_schema(dsn: str | None = None) -> None:
    """Create schema if not exists.

    Idempotent; safe to call on every startup.
    """
    if dsn is None:
        dsn = get_dsn()

    with connect(dsn) as conn:
        with conn.cursor() as cur:
            # aivar_runs: the main run record
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aivar_runs (
                    run_id text PRIMARY KEY,
                    url text NOT NULL,
                    mode text NOT NULL,
                    intent text,
                    escalated boolean NOT NULL DEFAULT false,
                    escalation_reason text,
                    flows_total int,
                    flows_passed int,
                    gaps_total int,
                    defects_found int,
                    heals_applied int,
                    cost_usd double precision,
                    duration_s double precision,
                    summary_line text,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
            """)

            # aivar_decisions: ordered sequence of decisions per run
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aivar_decisions (
                    id bigserial PRIMARY KEY,
                    run_id text REFERENCES aivar_runs(run_id) ON DELETE CASCADE,
                    seq int NOT NULL,
                    stage text,
                    verdict text,
                    reason text,
                    next_stage text,
                    evidence jsonb,
                    at timestamptz
                )
            """)

            # aivar_gaps: coverage gaps found during the run
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aivar_gaps (
                    id bigserial PRIMARY KEY,
                    run_id text REFERENCES aivar_runs(run_id) ON DELETE CASCADE,
                    kind text,
                    description text,
                    evidence text,
                    severity text
                )
            """)

            # aivar_flow_results: per-flow execution results
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aivar_flow_results (
                    id bigserial PRIMARY KEY,
                    run_id text REFERENCES aivar_runs(run_id) ON DELETE CASCADE,
                    flow_id text,
                    status text,
                    steps_total int,
                    steps_passed int,
                    heals_used int
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS aivar_decisions_run_seq
                ON aivar_decisions(run_id, seq)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS aivar_runs_created_at
                ON aivar_runs(created_at DESC)
            """)

        conn.commit()


@dataclass(frozen=True)
class RunSummary:
    """Summary of a run for listing."""
    run_id: str
    url: str
    mode: str
    escalated: bool
    flows_total: int
    flows_passed: int
    gaps_total: int
    cost_usd: float
    duration_s: float
    summary_line: str
    created_at: datetime


def save_run(report: PipelineReport, dsn: str | None = None) -> str:
    """Save a pipeline report to the database.

    Upserts the run row, then deletes and re-inserts its child records
    (decisions, gaps, flow_results) for idempotency. All within one transaction.

    Args:
        report: PipelineReport to save.
        dsn: Database URL. If None, calls get_dsn().

    Returns:
        The run_id.

    Raises:
        StoreError: On database error.
    """
    if dsn is None:
        dsn = get_dsn()

    try:
        with connect(dsn) as conn:
            with conn.cursor() as cur:
                # Upsert the run record
                cur.execute("""
                    INSERT INTO aivar_runs
                    (run_id, url, mode, intent, escalated, escalation_reason,
                     flows_total, flows_passed, gaps_total, defects_found,
                     heals_applied, cost_usd, duration_s, summary_line, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        url = EXCLUDED.url,
                        mode = EXCLUDED.mode,
                        intent = EXCLUDED.intent,
                        escalated = EXCLUDED.escalated,
                        escalation_reason = EXCLUDED.escalation_reason,
                        flows_total = EXCLUDED.flows_total,
                        flows_passed = EXCLUDED.flows_passed,
                        gaps_total = EXCLUDED.gaps_total,
                        defects_found = EXCLUDED.defects_found,
                        heals_applied = EXCLUDED.heals_applied,
                        cost_usd = EXCLUDED.cost_usd,
                        duration_s = EXCLUDED.duration_s,
                        summary_line = EXCLUDED.summary_line
                """, (
                    report.run_id,
                    report.url,
                    report.mode,
                    report.intent,
                    report.escalated,
                    report.escalation_reason,
                    report.flows_total,
                    report.flows_passed,
                    len(report.gaps),
                    report.defects_found,
                    report.heals_applied,
                    report.cost_usd,
                    report.duration_s,
                    report.summary_line,
                    datetime.now(timezone.utc),
                ))

                # Delete existing child records (for idempotency)
                cur.execute("DELETE FROM aivar_decisions WHERE run_id = %s", (report.run_id,))
                cur.execute("DELETE FROM aivar_gaps WHERE run_id = %s", (report.run_id,))
                cur.execute("DELETE FROM aivar_flow_results WHERE run_id = %s", (report.run_id,))

                # Insert decisions
                for seq, decision in enumerate(report.decisions):
                    cur.execute("""
                        INSERT INTO aivar_decisions
                        (run_id, seq, stage, verdict, reason, next_stage, evidence, at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        report.run_id,
                        seq,
                        decision.stage.value,
                        decision.verdict,
                        decision.reason,
                        decision.next_stage.value,
                        Jsonb(decision.evidence),
                        decision.at,
                    ))

                # Insert gaps
                for gap in report.gaps:
                    cur.execute("""
                        INSERT INTO aivar_gaps
                        (run_id, kind, description, evidence, severity)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        report.run_id,
                        gap.kind,
                        gap.description,
                        gap.evidence,
                        gap.severity.value,
                    ))

                # Insert flow results
                for flow_id, result in report.flow_results.items():
                    steps_passed = sum(
                        1 for sr in result.results if sr.status == "passed"
                    )
                    cur.execute("""
                        INSERT INTO aivar_flow_results
                        (run_id, flow_id, status, steps_total, steps_passed, heals_used)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        report.run_id,
                        flow_id,
                        result.status,
                        len(result.results),
                        steps_passed,
                        result.heals_used,
                    ))

            conn.commit()
    except Exception as e:
        raise StoreError(f"Failed to save run: {e}") from e

    return report.run_id


def save_run_safe(report: PipelineReport, dsn: str | None = None) -> str | None:
    """Save a run, but never break a pipeline if the database is down.

    Wraps save_run in try/except, logs a warning on failure, and returns None.
    This is what the orchestrator should call.

    The JSON/HTML/pytest artifacts on disk are the source of truth; the database
    is a convenience layer.
    """
    try:
        return save_run(report, dsn)
    except Exception as e:
        logger.warning(f"Failed to persist run {report.run_id} to database: {e}")
        return None


def list_runs(limit: int = 50, dsn: str | None = None) -> list[RunSummary]:
    """List runs, newest first.

    Args:
        limit: Maximum number of runs to return.
        dsn: Database URL. If None, calls get_dsn().

    Returns:
        List of RunSummary, ordered by created_at DESC.

    Raises:
        StoreError: On database error.
    """
    if dsn is None:
        dsn = get_dsn()

    try:
        with connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT run_id, url, mode, escalated, flows_total, flows_passed,
                           gaps_total, cost_usd, duration_s, summary_line, created_at
                    FROM aivar_runs
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))

                rows = cur.fetchall()
                result = []
                for row in rows:
                    result.append(RunSummary(
                        run_id=row[0],
                        url=row[1],
                        mode=row[2],
                        escalated=row[3],
                        flows_total=row[4],
                        flows_passed=row[5],
                        gaps_total=row[6],
                        cost_usd=row[7],
                        duration_s=row[8],
                        summary_line=row[9],
                        created_at=row[10],
                    ))
                return result
    except Exception as e:
        raise StoreError(f"Failed to list runs: {e}") from e


def get_run(run_id: str, dsn: str | None = None) -> dict | None:
    """Fetch a complete run with all its decisions, gaps, and flow results.

    Args:
        run_id: The run_id to fetch.
        dsn: Database URL. If None, calls get_dsn().

    Returns:
        Dict with keys "run", "decisions", "gaps", "flow_results", or None if not found.

    Raises:
        StoreError: On database error.
    """
    if dsn is None:
        dsn = get_dsn()

    try:
        with connect(dsn) as conn:
            with conn.cursor() as cur:
                # Fetch run
                cur.execute("""
                    SELECT run_id, url, mode, intent, escalated, escalation_reason,
                           flows_total, flows_passed, gaps_total, defects_found,
                           heals_applied, cost_usd, duration_s, summary_line, created_at
                    FROM aivar_runs
                    WHERE run_id = %s
                """, (run_id,))

                run_row = cur.fetchone()
                if not run_row:
                    return None

                run_dict = {
                    "run_id": run_row[0],
                    "url": run_row[1],
                    "mode": run_row[2],
                    "intent": run_row[3],
                    "escalated": run_row[4],
                    "escalation_reason": run_row[5],
                    "flows_total": run_row[6],
                    "flows_passed": run_row[7],
                    "gaps_total": run_row[8],
                    "defects_found": run_row[9],
                    "heals_applied": run_row[10],
                    "cost_usd": run_row[11],
                    "duration_s": run_row[12],
                    "summary_line": run_row[13],
                    "created_at": run_row[14].isoformat() if run_row[14] else None,
                }

                # Fetch decisions
                cur.execute("""
                    SELECT id, run_id, seq, stage, verdict, reason, next_stage, evidence, at
                    FROM aivar_decisions
                    WHERE run_id = %s
                    ORDER BY seq ASC
                """, (run_id,))

                decisions = []
                for row in cur.fetchall():
                    decisions.append({
                        "id": row[0],
                        "run_id": row[1],
                        "seq": row[2],
                        "stage": row[3],
                        "verdict": row[4],
                        "reason": row[5],
                        "next_stage": row[6],
                        "evidence": row[7],
                        "at": row[8].isoformat() if row[8] else None,
                    })

                # Fetch gaps
                cur.execute("""
                    SELECT id, run_id, kind, description, evidence, severity
                    FROM aivar_gaps
                    WHERE run_id = %s
                """, (run_id,))

                gaps = []
                for row in cur.fetchall():
                    gaps.append({
                        "id": row[0],
                        "run_id": row[1],
                        "kind": row[2],
                        "description": row[3],
                        "evidence": row[4],
                        "severity": row[5],
                    })

                # Fetch flow results
                cur.execute("""
                    SELECT id, run_id, flow_id, status, steps_total, steps_passed, heals_used
                    FROM aivar_flow_results
                    WHERE run_id = %s
                """, (run_id,))

                flow_results = []
                for row in cur.fetchall():
                    flow_results.append({
                        "id": row[0],
                        "run_id": row[1],
                        "flow_id": row[2],
                        "status": row[3],
                        "steps_total": row[4],
                        "steps_passed": row[5],
                        "heals_used": row[6],
                    })

                return {
                    "run": run_dict,
                    "decisions": decisions,
                    "gaps": gaps,
                    "flow_results": flow_results,
                }
    except Exception as e:
        raise StoreError(f"Failed to get run: {e}") from e


def delete_run(run_id: str, dsn: str | None = None) -> bool:
    """Delete a run and all its child records.

    Args:
        run_id: The run_id to delete.
        dsn: Database URL. If None, calls get_dsn().

    Returns:
        True if deleted, False if not found.

    Raises:
        StoreError: On database error.
    """
    if dsn is None:
        dsn = get_dsn()

    try:
        with connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM aivar_runs WHERE run_id = %s", (run_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    except Exception as e:
        raise StoreError(f"Failed to delete run: {e}") from e


def health() -> tuple[bool, str]:
    """Check database health.

    Attempts a simple SELECT 1 query.

    Returns:
        (True, "connected to <host>") on success, or
        (False, <short reason>) on failure.
        Never raises. Never includes the password in the returned string.
    """
    try:
        dsn = get_dsn()
    except StoreError as e:
        return (False, str(e))

    try:
        # Parse the host from the DSN
        host = "unknown"
        if "://" in dsn:
            # postgresql://user:pass@host:port/dbname
            after_scheme = dsn.split("://", 1)[1]
            if "@" in after_scheme:
                host_part = after_scheme.split("@", 1)[1]
                if "/" in host_part:
                    host = host_part.split("/", 1)[0]
                else:
                    host = host_part
                # Remove port if present
                if ":" in host:
                    host = host.split(":", 1)[0]

        with connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        return (True, f"connected to {host}")
    except Exception as e:
        # Never include DSN or password in the error message
        return (False, f"connection failed: {str(e)[:100]}")
