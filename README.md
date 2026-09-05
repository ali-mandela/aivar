# aivar

**Give it a URL. Get back a working end-to-end test suite.**

aivar is an autonomous test agent. It opens your app, works out what it does,
writes real Playwright tests, runs them, repairs the ones that broke for
mechanical reasons, and hands you an honest report of what it covered — and
what it couldn't.

You give it three things. Nothing else.

```bash
curl -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{"url":"https://your-app.com","username":"you","password":"secret"}'
```

Output is `server/tests/generated/*.py` — ordinary pytest files. Run them with
`pytest`. No part of this agent needs to be involved ever again.

---

## The problem statement

> *"I've been using the Playwright agents, but still I am the one giving them
> context again and again. It is a lot of manual work."*
> — Bessemer Tech Catalyst hackathon brief, AI Innovations

The complaint isn't that Playwright is hard. It's the endless context handoff.
So **the input is exactly three things: URL, username, password.** Everything
else the agent discovers for itself.

Four capabilities must be demonstrable:

| # | Capability | Where it lives |
|---|---|---|
| 1 | **Explore** your app | `app/explorer.py` |
| 2 | **Write** your test cases | `app/planner.py` → `app/critic.py` → `app/codegen.py` |
| 3 | **Run** your test cases | `app/executor.py` |
| 4 | **Heal** your test cases | `app/triage.py` → `app/healer.py` → `app/quarantine.py` |

---

## Architecture

```mermaid
flowchart TB
    subgraph client["Client"]
        UI["React UI<br/><i>Run · Results · History</i>"]
        CURL["curl / CI"]
    end

    subgraph api["FastAPI · app/api.py"]
        RUNS["POST /runs"]
        JOBS["GET /jobs/:id"]
        ART["GET /runs/:id/report · /tests"]
    end

    subgraph pipeline["Orchestrator state machine · app/orchestrator.py"]
        direction TB
        E["<b>EXPLORE</b><br/>crawl, log in, map pages & forms"]
        P["<b>PLAN</b><br/>LLM proposes user journeys"]
        C["<b>CRITIQUE</b><br/>coverage gate — structural checks first"]
        G["<b>GENERATE</b><br/>dry-run each flow live, bind selectors"]
        V["<b>VALIDATE</b><br/>write files, check they import"]
        X["<b>EXECUTE</b><br/>run flows, heal locators inline"]
        T["<b>TRIAGE</b><br/>bug vs. script issue vs. flake"]
        R["<b>REPORT</b><br/>JSON · HTML · text"]

        E --> P --> C
        C -->|accept| G
        C -.->|replan| P
        G --> V
        V -.->|regenerate| G
        V --> X --> T --> R
    end

    subgraph ext["External"]
        PW["Playwright<br/>Chromium"]
        LLM["LLM<br/>OpenRouter · Gemini · Sarvam<br/><i>cross-provider failover</i>"]
    end

    subgraph out["Output"]
        TESTS["tests/generated/*.py<br/><b>runnable pytest</b>"]
        REPORTS["artifacts/run-*.json|html|txt"]
        DB[("Postgres<br/><i>history only</i>")]
    end

    UI --> RUNS
    CURL --> RUNS
    RUNS --> pipeline
    pipeline -.->|Decision ledger| JOBS
    E & G & X --> PW
    P & C & T & X --> LLM
    V --> TESTS
    R --> REPORTS
    R -.->|best effort| DB
    ART --> REPORTS

    classDef stage fill:#eef4ff,stroke:#3b6ea5,color:#123
    classDef store fill:#f3f0ff,stroke:#7a5af8,color:#123
    class E,P,C,G,V,X,T,R stage
    class TESTS,REPORTS,DB store
```

**Every stage emits a `Decision`** — stage, verdict, reason, next stage,
evidence. That ledger is how a run explains itself, and it streams to the UI
live so you watch the agent think rather than a spinner.

**Escalating is a legitimate outcome.** If coverage can't be reached, the run
says so with a reason and *still* produces a report. It never fakes success.

---

## Setup

### Prerequisites

| | |
|---|---|
| Python | 3.13+, managed with [**uv**](https://docs.astral.sh/uv/) — no pip, no requirements.txt |
| Node | 20+ (only for the optional UI) |
| LLM key | at least one of OpenRouter / Google / Sarvam — free tiers are fine |
| Postgres | **optional**, run history only |

### 1. Server

```bash
cd server
uv sync                               # install deps
uv run playwright install chromium    # browser binary, first time only
cp .env.example .env                  # then add your key (see below)
uv run uvicorn app.main:app --reload
```

Confirm it's alive — this tells you whether the model provider and the database
are reachable *before* you commit to a 60-second pipeline:

```bash
curl localhost:8000/health
```

Interactive API docs: **http://localhost:8000/docs**

### 2. Configure `.env`

Only one line is truly required. Put **any one** of these in `server/.env`:

```ini
OPENROUTER_API_KEY=sk-or-...     # has free models; recommended for a demo
# GOOGLE_API_KEY=...
# SARVAM_API_KEY=sk_...
```

Configure **more than one** and they chain: models are tried in order within a
provider, then the next provider takes over. Free tiers rate-limit constantly
and a live demo must not die because one vendor said 429.

Other useful keys (all optional, all have defaults):

| Key | Default | What it does |
|---|---|---|
| `AIVAR_LLM_PROVIDER` | first key found | `openrouter` \| `google` \| `sarvam` |
| `AIVAR_LLM_MODELS` | per-provider | comma-separated, tried in order |
| `AIVAR_LLM_FALLBACK` | `1` | `0` pins the run to one provider |
| `AIVAR_DB_URL` | *(unset)* | Postgres DSN for run history |
| `AIVAR_USERNAME` / `AIVAR_PASSWORD` | — | credentials the generated tests read at runtime |
| `AIVAR_MAX_HEALS_PER_RUN` | `3` | healing budget |
| `AIVAR_MAX_COST_PER_RUN_USD` | `0.50` | spend ceiling |

`.env` is git-ignored. New keys go in `.env.example` with placeholder values.

### 3. UI (optional but worth it)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/api` to uvicorn, so no CORS setup is needed. Three tabs: **Run**
(kick off and watch the decision ledger stream), **Results** (flows, gaps,
triage verdicts, generated code), **History** (past runs from Postgres).

### 4. Postgres (optional)

Without it you lose run *history*, never a run — disk is the source of truth.

```bash
# in server/.env
AIVAR_DB_URL=postgresql://user:pass@localhost:5432/aivar
```

Create the tables once:

```bash
cd server
uv run python -c "from app.store import init_schema; init_schema()"
```

---

## Using it

### Kick off a run

```bash
curl -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{
        "url": "https://your-app.com",
        "username": "you",
        "password": "secret"
      }'
```

Blocks for 30–60s and returns the whole result, so plain curl just works.

### Useful request options

| Field | Default | Effect |
|---|---|---|
| `intent` | — | focus the run, e.g. `"checkout and auth"`. Blank = sweep everything |
| `prd_path` | — | path to a requirements doc → spec-led planning (upload via `POST /prd`) |
| `background` | `false` | return a `job_id` immediately, poll `GET /jobs/{id}` |
| `safe_mode` | `false` | fill forms but **never press submit** — use on any site you don't own |
| `max_flows` | `4` | how many journeys to test (1–12) |
| `max_pages` | `5` | exploration breadth (1–20) |
| `max_cost_usd` | `0.50` | hard spend cap |
| `max_seconds` | `300` | hard wall-clock cap |
| `heal` | `true` | repair broken locators mid-run |
| `llm_provider` | env | override the provider for this run |

### Run the suite yourself

This is the deliverable. The agent is not involved:

```bash
cd server
uv run pytest tests/generated
```

Credentials in generated tests are `${AIVAR_USERNAME}` placeholders resolved
from the environment at run time — never baked into the files.

### API surface

| Endpoint | Purpose |
|---|---|
| `GET /health` | provider + database reachability |
| `POST /runs` | run the full pipeline |
| `GET /jobs/{job_id}` | progress of a background run, with live decisions |
| `GET /runs` | run history |
| `GET /runs/{run_id}` | one run, with its decision ledger and gaps |
| `GET /runs/{run_id}/report.html` | rendered test-quality report |
| `GET /runs/{run_id}/report.json` | same report, structured |
| `GET /runs/{run_id}/tests` | that run's generated pytest files |
| `POST /prd` | upload a requirements doc, get a `prd_path` |

---

## The one rule that isn't negotiable

> **A failing assertion is a candidate bug, not a candidate repair.**

The obvious failure mode for a tool like this is healing whatever is red until
the suite is green — which silently deletes the very defects it was built to
find. A green suite that hid a regression is worse than no suite at all.

So it's enforced structurally, not by convention:

- `Guardrails.__post_init__` **raises** if `heal_assertions` is True. The
  dangerous object cannot be constructed.
- `Guardrails.from_env` hardcodes it to `False` and deliberately does not read
  it from the environment. It cannot be switched on by config.

Healing only ever touches **locators** — a button that moved, a label that got
renamed. Triage carries the same asymmetry on purpose: a false positive costs
one repair cycle; a masked regression ships a bug. When in doubt, report.

And the report always says **what was not tested** — `gaps` and
`untested_flow_risk` matter as much as the passes.

---

## Repo layout

```
server/
  app/
    main.py          FastAPI entrypoint, /health
    api.py           HTTP surface (sync handlers — Playwright needs it)
    orchestrator.py  the state machine; every stage emits a Decision
    explorer.py      crawl, log in, map pages and forms
    planner.py       LLM proposes user journeys
    critic.py        coverage gate — deterministic checks before the model
    codegen.py       Flow → runnable pytest module
    executor.py      run a compiled flow against the live app
    triage.py        bug vs. script issue vs. flake
    healer.py        propose a replacement locator
    quarantine.py    park repairs that didn't clear the confidence bar
    llm.py           three providers, one surface, cross-provider failover
    store.py         Postgres history (convenience layer, not the record)
    report.py        JSON / HTML / text reports
    config.py        Guardrails — including the one above
  tests/generated/   ← the deliverable
  artifacts/         ← reports, per-run test snapshots
frontend/            React + Vite UI
AGENTS.md            design rationale and constraints (read before contributing)
```

---

## Gotchas

- **API handlers are sync `def`, never `async def`.** Playwright's sync API
  can't run inside an asyncio event loop; FastAPI dispatches sync handlers to a
  worker thread, which is exactly what's needed.
- **`.env` is read as `utf-8-sig`.** PowerShell and some editors prepend a BOM
  that otherwise mangles the first key and makes it look unset with no visible
  cause.
- **Don't round-trip repo files through Windows PowerShell 5.1**
  (`Get-Content -Raw | Set-Content -Encoding utf8`). It reads UTF-8 as ANSI and
  writes mojibake back.
- **Only OpenRouter reports real spend.** On Google and Sarvam `cost_usd` is
  `0.0`, so `max_cost_usd` doesn't bind — `max_seconds` is your real limit
  there.
- **Playwright browser missing?** `uv run playwright install chromium`.
- **`tests/generated` is overwritten by every run.** Each run's own copy is
  snapshotted to `artifacts/{run_id}/tests` and served at
  `GET /runs/{run_id}/tests`.

---

## Stack

FastAPI · Playwright (sync API) · psycopg → Postgres · React + Vite ·
LLM via OpenRouter, Google Gemini or Sarvam AI.
