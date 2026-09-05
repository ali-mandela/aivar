import { useEffect, useState } from "react";
import {
  ApiError,
  getTests,
  reportHtmlUrl,
  reportJsonUrl,
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
          <div className="actions">
            <a
              className="ghost"
              href={reportHtmlUrl(result.run_id)}
              download={`${result.run_id}.html`}
            >
              Download HTML
            </a>
            <a
              className="ghost"
              href={reportJsonUrl(result.run_id)}
              download={`${result.run_id}.json`}
            >
              Download JSON
            </a>
          </div>
        </div>
        {/* Fully sandboxed: the report is generated markup and gets no script
            access and no same-origin privileges. */}
        <iframe
          className="report"
          title={`Test quality report for ${result.run_id}`}
          src={reportHtmlUrl(result.run_id)}
          sandbox=""
        />
      </section>
    </div>
  );
}
