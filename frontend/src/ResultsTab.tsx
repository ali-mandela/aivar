import { useEffect, useState } from "react";
import {
  ApiError,
  getReportHtml,
  getReportJson,
  getTests,
  type RunResult,
  type TestFiles,
} from "./api";
import {
  downloadText,
  ErrorBox,
  GapTable,
  PythonCode,
  RiskTable,
} from "./ui";

export default function ResultsTab({ result }: { result: RunResult | null }) {
  const [tests, setTests] = useState<TestFiles | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const runId = result?.run_id;
  useEffect(() => {
    if (!runId) return;
    let live = true;
    setError(null);
    getTests(runId)
      .then((t) => live && setTests(t))
      .catch((e) => live && setError(e instanceof ApiError ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [runId]);

  // Load the report up front rather than pointing the iframe at the URL: a
  // missing artifact would otherwise render its 404 body as the report.
  useEffect(() => {
    if (!runId) return;
    let live = true;
    setReport(null);
    setReportError(null);
    getReportHtml(runId)
      .then((html) => live && setReport(html))
      .catch(
        (e) => live && setReportError(e instanceof ApiError ? e.message : String(e)),
      );
    return () => {
      live = false;
    };
  }, [runId]);

  async function downloadJson() {
    if (!runId) return;
    try {
      downloadText(`${runId}.json`, await getReportJson(runId), "application/json");
    } catch (e) {
      setReportError(e instanceof ApiError ? e.message : String(e));
    }
  }

  if (!result) {
    return (
      <div className="page">
        <p className="empty">No results yet. Run a pipeline first.</p>
      </div>
    );
  }

  const files = Object.entries(tests?.files ?? {});

  return (
    <div className="page">
      <section>
        <div className="section-head">
          <h2>Generated test files</h2>
          <span className="aside mono">
            {tests?.directory ?? "loading…"}
          </span>
        </div>

        <ErrorBox error={error} title="Could not load the generated suite" />

        {tests?.stale && (
          <div className="alert notice">
            <h3>Showing the current suite</h3>
            <div>
              No saved copy exists for this run, so these are the files
              currently in <span className="mono">tests/generated</span>. They
              may belong to a later run.
            </div>
          </div>
        )}

        {files.length === 0 && !error && (
          <p className="empty">This run produced no test files.</p>
        )}

        {files.map(([name, source]) => (
          <details className="file" key={name}>
            <summary>
              {name}
              <span className="lines">{source.split("\n").length} lines</span>
            </summary>
            <PythonCode source={source} />
            <div className="file-actions">
              <button
                className="ghost"
                onClick={() => downloadText(name, source, "text/x-python")}
              >
                Download {name}
              </button>
            </div>
          </details>
        ))}
      </section>

      <section>
        <div className="section-head">
          <h2>Coverage gaps</h2>
          <span className="aside">What the agent could not reach, and why.</span>
        </div>
        <GapTable gaps={result.gaps} />
      </section>

      <section>
        <div className="section-head">
          <h2>Untested flow risk</h2>
        </div>
        <RiskTable risks={result.untested_flow_risk} />
      </section>

      <section>
        <div className="section-head">
          <h2>Full report</h2>
          {report && (
            <div className="actions">
              <button
                className="ghost"
                onClick={() =>
                  downloadText(`${result.run_id}.html`, report, "text/html")
                }
              >
                Download HTML
              </button>
              <button className="ghost" onClick={() => void downloadJson()}>
                Download JSON
              </button>
            </div>
          )}
        </div>

        {reportError && (
          <div className="alert notice">
            <h3>No report for this run</h3>
            <div>
              {reportError} The decision ledger and the tables above still
              describe what happened.
            </div>
          </div>
        )}

        {!report && !reportError && <p className="empty">Loading report…</p>}

        {/* Rendered from text already fetched, and fully sandboxed: no script
            execution and no same-origin privileges. */}
        {report && (
          <iframe
            className="report"
            title={`Test quality report for ${result.run_id}`}
            srcDoc={report}
            sandbox=""
          />
        )}
      </section>
    </div>
  );
}
