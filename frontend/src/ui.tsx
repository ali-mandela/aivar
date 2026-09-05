import type { ReactNode } from "react";
import type { Decision, Gap, Severity, UntestedRisk } from "./api";

/* ------------------------------------------------------------- primitives -- */

export function ErrorBox({
  error,
  title = "Request failed",
}: {
  error: string | null;
  title?: string;
}) {
  if (!error) return null;
  return (
    <div className="alert" role="alert">
      <h3>{title}</h3>
      <div>{error}</div>
    </div>
  );
}

export function SeverityTag({ severity }: { severity: Severity }) {
  return <span className={`sev sev-${severity}`}>{severity}</span>;
}

export function Metric({
  label,
  value,
  of,
  alarm = false,
}: {
  label: string;
  value: ReactNode;
  of?: number;
  alarm?: boolean;
}) {
  return (
    <div className={`metric${alarm ? " alert-value" : ""}`}>
      <span className="v">
        {value}
        {of !== undefined && <span className="of">/{of}</span>}
      </span>
      <span className="k">{label}</span>
    </div>
  );
}

/* ---------------------------------------------------------------- ledger -- */

/** Verdicts the orchestrator emits. Anything else renders neutral rather than
 *  guessing at a colour it has not earned. */
const KNOWN_VERDICTS = new Set([
  "continue",
  "accept",
  "replan",
  "regenerate",
  "escalate",
]);

export function DecisionLedger({
  decisions,
  freshFrom = Infinity,
}: {
  decisions: Decision[];
  /** Index from which rows are newly arrived, so a live run animates only the
   *  rows that actually just appeared. */
  freshFrom?: number;
}) {
  if (decisions.length === 0) {
    return <p className="empty">No decisions recorded.</p>;
  }
  return (
    <ol className="ledger">
      {decisions.map((d, i) => {
        const verdict = KNOWN_VERDICTS.has(d.verdict) ? d.verdict : "unknown";
        const hasEvidence = d.evidence && Object.keys(d.evidence).length > 0;
        return (
          <li
            key={`${d.stage}-${d.at}-${i}`}
            className={`dec v-${verdict}${i >= freshFrom ? " fresh" : ""}`}
          >
            <div className="dec-head">
              <span className="dec-stage">{d.stage}</span>
              <span className="dec-verdict">{d.verdict}</span>
              <span className="dec-next">next: {d.next_stage}</span>
            </div>
            <p className="dec-reason">{d.reason}</p>
            {hasEvidence && <Evidence evidence={d.evidence} />}
          </li>
        );
      })}
    </ol>
  );
}

/** One discovered page, as the explore stage records it. */
interface ExploredPage {
  url: string;
  title: string;
  depth: number;
  forms: number;
}

function isExploredPages(v: unknown): v is ExploredPage[] {
  return (
    Array.isArray(v) &&
    v.length > 0 &&
    v.every(
      (p) => p && typeof p === "object" && typeof (p as ExploredPage).url === "string",
    )
  );
}

/** The path an URL points at, which is what identifies a page within an app.
 *  Falls back to the raw string if it will not parse. */
function pathOf(url: string): string {
  try {
    const u = new URL(url);
    return `${u.pathname}${u.search}` || "/";
  } catch {
    return url;
  }
}

/** Evidence, rendered for reading where its shape is known and as JSON
 *  otherwise. A count on its own ("discovered 5 pages") cannot be checked; the
 *  point of the ledger is that a human can verify the claim. */
function Evidence({ evidence }: { evidence: Record<string, unknown> }) {
  const pages = evidence.pages;
  const rest = Object.fromEntries(
    Object.entries(evidence).filter(([k]) => k !== "pages"),
  );

  return (
    <details>
      <summary>Evidence</summary>
      {isExploredPages(pages) && (
        <table className="pages">
          <thead>
            <tr>
              <th>Path</th>
              <th>Title</th>
              <th className="num">Depth</th>
              <th className="num">Forms</th>
            </tr>
          </thead>
          <tbody>
            {pages.map((p, i) => (
              <tr key={`${p.url}-${i}`}>
                <td className="num">
                  <span title={p.url}>{pathOf(p.url)}</span>
                </td>
                <td>{p.title || "—"}</td>
                <td className="num">{p.depth}</td>
                <td className="num">{p.forms || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {Object.keys(rest).length > 0 && (
        <pre className="evidence">{JSON.stringify(rest, null, 2)}</pre>
      )}
    </details>
  );
}

/* ----------------------------------------------------------------- tables -- */

export function GapTable({ gaps }: { gaps: Gap[] }) {
  if (gaps.length === 0) {
    return <div className="good">No coverage gaps detected.</div>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Severity</th>
          <th>Kind</th>
          <th>Description</th>
          <th>Evidence</th>
        </tr>
      </thead>
      <tbody>
        {gaps.map((g, i) => (
          <tr key={i}>
            <td>
              <SeverityTag severity={g.severity} />
            </td>
            <td className="num">{g.kind}</td>
            <td>{g.description}</td>
            <td className="num">{g.evidence || "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function RiskTable({ risks }: { risks: UntestedRisk[] }) {
  if (risks.length === 0) {
    return <div className="good">No untested flow risk recorded.</div>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Severity</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {risks.map((r, i) => (
          <tr key={i}>
            <td>
              <SeverityTag severity={r.severity} />
            </td>
            <td>{r.description}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ---------------------------------------------------- python highlighting -- */

// One pass, one regex: alternatives are ordered so comments and strings win
// before keywords, which stops a keyword inside a string from being coloured.
// Nodes are built as React elements rather than HTML, so generated source can
// never be injected into the page as markup.
const PY = new RegExp(
  [
    /(#[^\n]*)/, // 1 comment
    /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/, // 2 string
    /\b(def|class|return|if|elif|else|for|while|import|from|as|with|try|except|finally|raise|assert|pass|break|continue|lambda|yield|global|nonlocal|del|in|is|not|and|or|None|True|False|async|await)\b/, // 3 keyword
    /(@[\w.]+)/, // 4 decorator
    /\b(\d+\.?\d*)\b/, // 5 number
  ]
    .map((r) => r.source)
    .join("|"),
  "g",
);

export function PythonCode({ source }: { source: string }) {
  const nodes: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;

  PY.lastIndex = 0;
  while ((m = PY.exec(source)) !== null) {
    if (m.index > last) nodes.push(source.slice(last, m.index));
    const cls = m[1]
      ? "t-comment"
      : m[2]
        ? "t-str"
        : m[3]
          ? "t-kw"
          : m[4]
            ? "t-dec"
            : "t-num";
    nodes.push(
      <span key={key++} className={cls}>
        {m[0]}
      </span>,
    );
    last = m.index + m[0].length;
  }
  if (last < source.length) nodes.push(source.slice(last));

  return <pre className="code">{nodes}</pre>;
}

/* ----------------------------------------------------------------- format -- */

export const money = (n: number) => `$${n.toFixed(4)}`;
export const seconds = (n: number) => `${n.toFixed(1)}s`;

export function truncate(s: string, max = 50) {
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

/** Trigger a browser download of text the app already holds. */
export function downloadText(filename: string, text: string, mime = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
