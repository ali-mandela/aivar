import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  ApiError,
  getJob,
  startRun,
  uploadPrd,
  type Decision,
  type Job,
  type PrdUpload,
  type RunResult,
} from "./api";
import {
  DecisionLedger,
  ErrorBox,
  Metric,
  money,
  seconds,
} from "./ui";

const POLL_MS = 1500;

type Mode = "sweep" | "focused" | "spec_led";

export default function RunTab({
  result,
  onResult,
}: {
  result: RunResult | null;
  onResult: (r: RunResult) => void;
}) {
  const [url, setUrl] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [intent, setIntent] = useState("");
  const [prd, setPrd] = useState<PrdUpload | null>(null);
  const [prdBusy, setPrdBusy] = useState(false);

  const [maxFlows, setMaxFlows] = useState(4);
  const [maxPages, setMaxPages] = useState(5);
  const [safeMode, setSafeMode] = useState(false);
  const [headless, setHeadless] = useState(true);
  const [heal, setHeal] = useState(true);

  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  // Narrowed rather than a bare boolean, so the running-only UI can read the
  // job's fields without a null check at every use.
  const activeJob = job?.status === "running" ? job : null;
  const running = activeJob !== null;
  const mode: Mode = prd ? "spec_led" : intent.trim() ? "focused" : "sweep";

  /* Poll the job while it runs. The server publishes each decision as it is
     recorded, so the ledger fills in during the run rather than after it. */
  const jobId = activeJob?.job_id ?? null;
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    const timer = setInterval(async () => {
      try {
        const next = await getJob(jobId);
        if (!live) return;
        setJob(next);
        if (next.status !== "running") {
          if (next.result) onResult(next.result);
          if (next.status === "failed" && next.error) setError(next.error);
        }
      } catch (e) {
        if (!live) return;
        setError(e instanceof ApiError ? e.message : String(e));
        setJob((j) => (j ? { ...j, status: "failed" } : j));
      }
    }, POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [jobId, onResult]);

  /* An honest elapsed counter. The pipeline reports no percentage, so the UI
     does not invent one. */
  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 500);
    return () => clearInterval(timer);
  }, [running]);

  async function handlePrd(file: File | undefined) {
    if (!file) return;
    setPrdBusy(true);
    setError(null);
    try {
      setPrd(await uploadPrd(file));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setPrdBusy(false);
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setElapsed(0);
    try {
      const accepted = await startRun({
        url: url.trim(),
        username: username.trim() || undefined,
        password: password || undefined,
        intent: intent.trim() || undefined,
        prd_path: prd?.prd_path,
        max_flows: maxFlows,
        max_pages: maxPages,
        headless,
        safe_mode: safeMode,
        heal,
        background: true,
      });
      setJob({
        job_id: accepted.job_id,
        status: "running",
        stage: "explore",
        run_id: null,
        summary: null,
        error: null,
        decisions: [],
      });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  // While running, show the live ledger; afterwards, the finished run's.
  const ledger = activeJob ? activeJob.decisions : (result?.decisions ?? []);

  return (
    <div className="page">
      <form onSubmit={submit}>
        <section>
          <div className="field">
            <label htmlFor="url">Application URL</label>
            <input
              id="url"
              type="url"
              required
              value={url}
              disabled={running}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.saucedemo.com"
            />
          </div>

          <div className="row">
            <div className="field">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                type="text"
                autoComplete="off"
                value={username}
                disabled={running}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                autoComplete="off"
                value={password}
                disabled={running}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>
          <p className="hint credentials-note">
            Both optional. Leave blank for an app with no sign-in.
          </p>

          <div className="field">
            <label htmlFor="intent">What should it focus on?</label>
            <textarea
              id="intent"
              value={intent}
              disabled={running}
              onChange={(e) => setIntent(e.target.value)}
              placeholder="Test the login flow and the checkout path"
            />
            <p className="hint">
              Optional. Leaving this blank tests everything rather than nothing.
            </p>
          </div>

          <div className="field">
            <label htmlFor="prd">Requirements document</label>
            <div className="file-drop">
              <input
                id="prd"
                type="file"
                accept=".md,.markdown,.txt"
                disabled={running || prdBusy}
                onChange={(e) => handlePrd(e.target.files?.[0])}
              />
              {prd && (
                <>
                  <span className="mono">
                    {prd.filename} · {prd.lines} lines
                  </span>
                  <button
                    type="button"
                    className="ghost"
                    disabled={running}
                    onClick={() => setPrd(null)}
                  >
                    Remove
                  </button>
                </>
              )}
              {prdBusy && <span className="mono">Uploading…</span>}
            </div>
            <p className="hint">
              Optional. Markdown or text. Requirements become the coverage
              target.
            </p>
          </div>

          <details className="advanced">
            <summary>Advanced settings</summary>

            <div className="row">
              <div className="field">
                <label htmlFor="flows">Maximum flows</label>
                <div className="slider-row">
                  <input
                    id="flows"
                    type="range"
                    min={1}
                    max={10}
                    value={maxFlows}
                    disabled={running}
                    onChange={(e) => setMaxFlows(Number(e.target.value))}
                  />
                  <output htmlFor="flows">{maxFlows}</output>
                </div>
              </div>
              <div className="field">
                <label htmlFor="pages">Maximum pages to explore</label>
                <div className="slider-row">
                  <input
                    id="pages"
                    type="range"
                    min={1}
                    max={15}
                    value={maxPages}
                    disabled={running}
                    onChange={(e) => setMaxPages(Number(e.target.value))}
                  />
                  <output htmlFor="pages">{maxPages}</output>
                </div>
              </div>
            </div>

            <label className="check">
              <input
                type="checkbox"
                checked={safeMode}
                disabled={running}
                onChange={(e) => setSafeMode(e.target.checked)}
              />
              <span>
                Safe mode
                <p className="hint">
                  Fills forms but never submits them. Use on any site you do not
                  own.
                </p>
              </span>
            </label>

            <label className="check">
              <input
                type="checkbox"
                checked={headless}
                disabled={running}
                onChange={(e) => setHeadless(e.target.checked)}
              />
              <span>
                Headless browser
                <p className="hint">Uncheck to watch the run in a real window.</p>
              </span>
            </label>

            <label className="check">
              <input
                type="checkbox"
                checked={heal}
                disabled={running}
                onChange={(e) => setHeal(e.target.checked)}
              />
              <span>
                Heal broken locators
                <p className="hint">
                  Repairs locator drift only. A failing assertion is always
                  reported as a possible bug, never healed.
                </p>
              </span>
            </label>
          </details>

          <ModeBanner mode={mode} intent={intent} prdName={prd?.filename} />

          <div className="actions">
            <button className="primary" type="submit" disabled={running}>
              {running ? "Running…" : "Run pipeline"}
            </button>
            <span className="hint">
              A full run usually takes 30–60 seconds.
            </span>
          </div>
        </section>
      </form>

      <ErrorBox error={error} />

      {activeJob && (
        <div className="running" role="status">
          <span className="pulse" aria-hidden="true" />
          <span>
            Stage <span className="mono">{activeJob.stage}</span> —{" "}
            {activeJob.decisions.length}{" "}
            {activeJob.decisions.length === 1 ? "decision" : "decisions"} recorded
          </span>
          <span className="elapsed">{seconds(elapsed)}</span>
        </div>
      )}

      {result?.escalated && !running && (
        <div className="alert" role="alert">
          <h3>Stopped early</h3>
          <div>
            {result.escalation_reason ??
              "The run could not reach acceptable coverage."}
          </div>
          <p className="hint">
            The agent stopped rather than hand you a suite it did not believe
            in. The decision ledger below shows the stage it gave up at and why.
            {" "}
            {nextStepFor(result.escalation_reason)}
          </p>
        </div>
      )}

      {result && !running && (
        <section>
          <div className="section-head">
            <h2>Result</h2>
            <span className="aside mono">{result.run_id}</span>
          </div>
          <div className="metrics">
            <Metric
              label="Flows passed"
              value={result.flows.passed}
              of={result.flows.total}
              hint="User journeys that ran green"
            />
            <Metric
              label="Gaps"
              value={result.gaps.length}
              hint="Things worth testing that this suite does not cover"
            />
            <Metric
              label="Heals"
              value={result.heals_applied}
              hint="Locators the agent repaired itself"
            />
            <Metric
              label="Defects"
              value={result.defects_found}
              alarm={result.defects_found > 0}
              hint="Failures that look like bugs in your app, not in the tests"
            />
            <Metric
              label="Cost"
              value={money(result.cost_usd)}
              hint="Model spend for this run"
            />
            <Metric
              label="Duration"
              value={seconds(result.duration_s)}
              hint="Wall clock, start to report"
            />
          </div>
          <p style={{ marginTop: 12, color: "var(--muted)", fontSize: 13.5 }}>
            {result.summary}
          </p>
        </section>
      )}

      {ledger.length > 0 && (
        <section>
          <div className="section-head">
            <h2>Decision ledger</h2>
            <span className="aside">
              Every stage records why it did what it did.
            </span>
          </div>
          <LiveLedger decisions={ledger} live={running} />
        </section>
      )}
    </div>
  );
}

/** What to try next, for the ways a run actually gives up.
 *
 *  "No flows survived validation" is accurate and tells a reader nothing they
 *  can do about it. Each of these is a real cause with a real remedy. */
function nextStepFor(reason: string | null | undefined): string {
  const r = (reason ?? "").toLowerCase();
  if (r.includes("survived validation") || r.includes("no flows compiled")) {
    return "This usually means the agent could not find the elements it planned against — most often because it never got past a sign-in. Adding a username and password is the first thing to try.";
  }
  if (r.includes("0 pages") || r.includes("exploration")) {
    return "Exploration found nothing to test. Check the URL loads in a normal browser, and that it is reachable from this machine.";
  }
  if (r.includes("coverage")) {
    return "Try raising the maximum flows in advanced settings, or narrowing the focus so the plan has a smaller target to cover well.";
  }
  if (r.includes("budget") || r.includes("cost") || r.includes("seconds")) {
    return "The run hit its time or cost limit. Lower the maximum pages to explore, or raise the limit and run again.";
  }
  return "Running again often helps: planning is model-driven and varies between runs.";
}

/** Animates only rows that arrived since the last render, so a finished run
 *  does not replay its whole ledger. */
function LiveLedger({
  decisions,
  live,
}: {
  decisions: Decision[];
  live: boolean;
}) {
  const seen = useRef(0);
  const freshFrom = live ? seen.current : Infinity;
  useEffect(() => {
    seen.current = decisions.length;
  }, [decisions.length]);
  return <DecisionLedger decisions={decisions} freshFrom={freshFrom} />;
}

function ModeBanner({
  mode,
  intent,
  prdName,
}: {
  mode: Mode;
  intent: string;
  prdName?: string;
}) {
  if (mode === "focused") {
    return (
      <div className="mode">
        <h3>Focused</h3>
        <p>
          Planning against your instruction: <span className="echo">{intent.trim()}</span>
        </p>
      </div>
    );
  }
  if (mode === "spec_led") {
    return (
      <div className="mode">
        <h3>Spec-led</h3>
        <p>
          Coverage is measured against {prdName ?? "the uploaded document"}. Each
          requirement becomes something to test or to report as a gap.
        </p>
      </div>
    );
  }
  return (
    <div className="mode">
      <h3>Sweep</h3>
      <p>
        A blank prompt means &ldquo;test everything&rdquo;, not &ldquo;do
        nothing&rdquo;. Every page, form and flow the agent can reach will be
        tested.
      </p>
    </div>
  );
}
