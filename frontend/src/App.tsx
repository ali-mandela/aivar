import { useCallback, useEffect, useState } from "react";
import { ApiError, getHealth, type Health, type RunResult } from "./api";
import HistoryTab from "./HistoryTab";
import ResultsTab from "./ResultsTab";
import RunTab from "./RunTab";

type TabId = "run" | "results" | "history";

const TABS: { id: TabId; label: string }[] = [
  { id: "run", label: "Run" },
  { id: "results", label: "Results" },
  { id: "history", label: "History" },
];

export default function App() {
  const [tab, setTab] = useState<TabId>("run");
  const [result, setResult] = useState<RunResult | null>(null);

  // Stable identity: RunTab's polling effect depends on this callback, and a
  // new function each render would restart the interval on every tick.
  const handleResult = useCallback((r: RunResult) => {
    setResult(r);
  }, []);

  return (
    <div className="shell">
      <Sidebar />
      <main className="main">
        <div className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              className="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
            >
              {t.label}
              {t.id === "results" && result && (
                <span className="count">{result.flows.total}</span>
              )}
            </button>
          ))}
        </div>

        {tab === "run" && <RunTab result={result} onResult={handleResult} />}
        {tab === "results" && <ResultsTab result={result} />}
        {tab === "history" && <HistoryTab />}
      </main>
    </div>
  );
}

function Sidebar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getHealth()
      .then((h) => live && setHealth(h))
      .catch((e) => live && setError(e instanceof ApiError ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, []);

  const llm = health?.llm;
  const db = health?.database;

  return (
    <aside className="sidebar">
      <div className="wordmark">
        Aivar
        <span>
          Give it a URL. It explores the app, writes the tests, runs them and
          says what it could not reach.
        </span>
      </div>

      <div className="side-section">
        <h2>Model provider</h2>
        {error && (
          <>
            <div className="status-line">
              <span className="lamp bad" aria-hidden="true" />
              <span>Server unreachable</span>
            </div>
            <p className="status-detail">{error}</p>
          </>
        )}
        {!error && !health && <p className="status-detail">Checking…</p>}
        {llm?.configured === true && (
          <>
            <div className="status-line">
              <span className="lamp ok" aria-hidden="true" />
              <span className="mono">{llm.provider}</span>
            </div>
            <ul className="model-list mono">
              {llm.models.map((m) => (
                <li key={m}>{m}</li>
              ))}
              {llm.fallbacks.flatMap((f) =>
                f.models.map((m) => (
                  <li key={`${f.provider}-${m}`} className="fallback">
                    {m}
                  </li>
                )),
              )}
            </ul>
            {llm.fallbacks.length > 0 && (
              <p className="status-detail">
                Greyed models are failover, tried in order when a provider
                rate-limits.
              </p>
            )}
          </>
        )}
        {llm?.configured === false && (
          <>
            <div className="status-line">
              <span className="lamp bad" aria-hidden="true" />
              <span>Not configured</span>
            </div>
            <p className="status-detail">
              Planning needs an API key. Set <span className="mono">OPENROUTER_API_KEY</span>,{" "}
              <span className="mono">GOOGLE_API_KEY</span> or{" "}
              <span className="mono">SARVAM_API_KEY</span> in{" "}
              <span className="mono">server/.env</span>.
            </p>
            <p className="status-detail">{llm.reason}</p>
          </>
        )}
      </div>

      <div className="side-section">
        <h2>Run history</h2>
        {db && (
          <>
            <div className="status-line">
              <span
                className={`lamp ${db.connected ? "ok" : "bad"}`}
                aria-hidden="true"
              />
              <span>{db.connected ? "Connected" : "Not connected"}</span>
            </div>
            <p className="status-detail">{db.detail}</p>
            {!db.connected && (
              <p className="status-detail">
                Runs still work. Reports and test files are written to disk;
                only the history list needs the database.
              </p>
            )}
          </>
        )}
        {!db && !error && <p className="status-detail">Checking…</p>}
      </div>
    </aside>
  );
}
