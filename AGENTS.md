# aivar

## Root statement

This project exists to answer one problem statement. Everything below it is
downstream of this. When a design decision is unclear, re-read this section
first.

Source: Bessemer Tech Catalyst hackathon, posed by AI Innovations (an AI-native
services company and AWS preferred partner).

Brief: <https://hackcultureplatform.blob.core.windows.net/event-assets/hackathons/6a429905623dd6dbd3249f0e/problem_explanation_9dm9yp4f98s.pdf>

The framing, transcribed from the brief:

> "I've been trying to write a test case for 3 days for a feature that I built
> in one day. I wish there was an AI that could do this for me."
>
> "Hey, have you tried about Playwright?"
>
> "No."
>
> "You should definitely try that man, because it has inbuilt hidden like
> tanner generator and all, so it should reduce your work."
>
> "Hey, I've been using the Playwright agents, but still I am the one giving
> them context again and again. It is a lot of manual work. I wish there is an
> AI that can do this for us."
>
> "Why don't we just hire someone for that?"
>
> "And how do we do that?"
>
> "Bessemer Tech Catalyst."
>
> "So we are AI Innovations, and we are a native services company and an AWS
> preferred partner. So here is your problem statement. We will be giving you
> an app URL, username, and password, and your agent should come up with a
> working end-to-end test suite, and it must be able to explore your app, write
> your test cases, run your test cases, and heal your test cases. Show us that,
> and we will hire you."

### What that pins down

**Input is exactly three things: an app URL, a username, and a password.**
Anything else the agent needs, it must discover for itself. Every feature is
measured against this: if it requires the user to hand over more context, it is
working against the brief. The complaint in the transcript is not "Playwright is
hard" — it is *"I am the one giving them context again and again."* Removing
that handoff is the product.

**Output is a working end-to-end test suite.** Not a report about tests, not a
recording — runnable files. `pytest tests/generated` must work with no part of
this agent involved.

**Four capabilities must be demonstrable, in this order:**

| Required | Where it lives |
| --- | --- |
| Explore your app | `app/explorer.py` |
| Write your test cases | `app/planner.py` → `app/critic.py` → `app/codegen.py` |
| Run your test cases | `app/executor.py` |
| Heal your test cases | `app/triage.py` → `app/healer.py` → `app/quarantine.py` |

A change that improves one of these at the cost of another is a bad trade. All
four have to be shown working.

## The invariant that must not be broken

**A failing assertion is a candidate bug, not a candidate repair.**

The obvious failure mode for this category of tool is healing whatever is red
until the suite is green — which silently deletes the defects the tool was
built to find. A green suite that hid a regression is worse than no suite.

This is enforced structurally, not by convention:

- `Guardrails.__post_init__` raises if `heal_assertions` is True, so the
  dangerous object cannot be constructed.
- `Guardrails.from_env` hardcodes `heal_assertions = False` and deliberately
  does not read it from the environment, so it cannot be switched on by config.

Do not "fix" either of these. If a change requires healing an assertion, the
change is wrong.

Triage carries the same asymmetry on purpose: a false positive costs one repair
cycle, a masked regression ships a bug. When in doubt, report rather than heal.

## Conventions

**Deterministic first, model second.** Structural checks run before any LLM
call, and verdicts are decided in Python from the model's output rather than by
the model. Control flow stays auditable — see `critic.py` and `triage.py`, where
the model is consulted only for the genuinely ambiguous case
(`LOCATOR_NOT_FOUND`).

**Every stage emits a Decision.** The `Stage` machine in `orchestrator.py` moves
`EXPLORE → PLAN → CRITIQUE → GENERATE → VALIDATE → EXECUTE → TRIAGE → HEAL →
REPORT`, appending a `Decision` (stage, verdict, reason, next stage) to the
ledger at each step. The ledger is how a run explains itself. A new stage that
does not record why it did what it did is incomplete.

**Escalating is a legitimate outcome.** When coverage cannot be reached, the run
escalates with a reason and still produces a report. `api.py` returns that as a
normal 200 response, not an HTTP error. Never fake success to avoid escalating.

**Say what was not tested.** `untested_flow_risk` and `gaps` matter as much as
passes. The report's job is an honest account of coverage, including its holes.

**Budgets are real.** `max_cost_usd`, `max_pipeline_seconds`, `max_flows`,
`max_heals_per_run` are enforced, not advisory. A run must terminate.

**Disk is the source of truth.** JSON/HTML/pytest artifacts on disk are durable;
Postgres is a convenience layer for history. A database outage costs run
history, never a run — keep it that way.

**Relative output paths resolve against the server root**, via
`app/paths.py::resolve_out_dir`, not the working directory. Do not reintroduce
bare relative paths.

**Never log or echo credentials.** Secrets travel as `${NAME}` placeholders
(`app/secrets.py`) and are resolved as late as possible. `redact` exists; use it.

## Stack and commands

FastAPI · Playwright (sync API) · psycopg → Postgres · LLM via OpenRouter with
model failover.

Managed entirely with **uv**. No `pip`, no `requirements.txt`.

```bash
cd server
uv sync                                    # install
uv run playwright install chromium         # browser binary, first time only
uv run uvicorn app.main:app --reload       # serve
uv run pytest tests/generated              # run a generated suite
```

Trigger a run:

```bash
curl -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","username":"u","password":"p"}'
```

`GET /health` reports whether the model provider and the database are reachable
before you commit to a 60-second pipeline.

### Gotchas

- API handlers are sync `def`, never `async def`. Playwright's sync API cannot
  run inside an asyncio event loop; FastAPI dispatches sync handlers to a worker
  thread, which is what this needs.
- `.env` is read with `utf-8-sig`. PowerShell and some editors prepend a BOM,
  which otherwise turns the first key into `﻿OPENROUTER_API_KEY` and makes
  it look unset with no visible cause.
- `.env` is git-ignored and holds live credentials. Keep it that way; add new
  keys to `.env.example` with placeholder values.
- `safe_mode` fills forms but never presses submit. Use it against any site you
  do not own.
