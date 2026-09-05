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
  hint,
}: {
  label: string;
  value: ReactNode;
  of?: number;
  alarm?: boolean;
  /** One line saying what the number means. "Gaps: 8" tells a reader nothing
   *  they can act on unless they already know what this agent counts as a gap. */
  hint?: string;
}) {
  return (
    <div className={`metric${alarm ? " alert-value" : ""}`} title={hint}>
      <span className="v">
        {value}
        {of !== undefined && <span className="of">/{of}</span>}
      </span>
      <span className="k">{label}</span>
      {hint && <span className="m-hint">{hint}</span>}
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
  "healed",
]);

/** What each stage was doing, in a sentence.
 *
 *  The ledger's whole purpose is that a person can check the agent's reasoning,
 *  and "triage · continue · next: report" is only checkable by someone who
 *  already knows the pipeline. These are for everyone else. */
const STAGE_WHAT: Record<string, string> = {
  explore: "Opened the app and crawled it to see what is there",
  plan: "Wrote test flows covering what exploration found",
  critique: "Checked the plan for coverage gaps before writing any code",
  generate: "Found the real page elements and wrote the pytest files",
  validate: "Checked the generated files are runnable",
  execute: "Ran the flows against the live app",
  triage: "Decided whether each failure is a bug or a broken test",
  heal: "Repaired locators that had drifted",
  report: "Wrote the test quality report",
  escalated: "Stopped early and said why, rather than pretending to succeed",
};

/** What each verdict decided. */
const VERDICT_WHAT: Record<string, string> = {
  continue: "went on to the next stage",
  accept: "coverage was good enough to build on",
  replan: "sent the plan back to be rewritten",
  regenerate: "tried compiling the flows again",
  escalate: "could not proceed honestly, so it stopped",
  healed: "replaced a locator and carried on",
};

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
              <span className="dec-stage" title={STAGE_WHAT[d.stage]}>
                {d.stage}
              </span>
              <span className="dec-verdict" title={VERDICT_WHAT[d.verdict]}>
                {d.verdict}
              </span>
              <span className="dec-next">next: {d.next_stage}</span>
            </div>
            {STAGE_WHAT[d.stage] && (
              <p className="dec-what">{STAGE_WHAT[d.stage]}</p>
            )}
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

/** One planned step. Selectors are absent at plan time by design. */
interface PlannedStep {
  kind: string;
  verb: string;
  target: string;
  value: string | null;
}

/** One planned flow, as the plan stage records it. */
interface PlannedFlow {
  name: string;
  kind: string;
  description: string;
  steps: PlannedStep[];
}

/** Each stage uses its own evidence key, so a shape is never guessed at from
 *  a name two stages share. */
function isArrayOf<T>(v: unknown, ok: (x: Record<string, unknown>) => boolean): v is T[] {
  return (
    Array.isArray(v) &&
    v.length > 0 &&
    v.every((x) => x !== null && typeof x === "object" && ok(x as Record<string, unknown>))
  );
}

const isPlannedFlows = (v: unknown): v is PlannedFlow[] =>
  isArrayOf<PlannedFlow>(v, (f) => typeof f.name === "string" && Array.isArray(f.steps));

/** A step as a single readable line: the verb, what it acts on, and the value
 *  it uses. Mirrors how the generated pytest reads. */
function stepLine(s: PlannedStep): string {
  const base = `${s.verb} ${s.target}`.trim();
  return s.value ? `${base} = ${s.value}` : base;
}

function FlowTable({ flows }: { flows: PlannedFlow[] }) {
  return (
    <div className="flows">
      {flows.map((f, i) => {
        const assertions = f.steps.filter((s) => s.kind === "assertion").length;
        return (
          <details className="flow" key={`${f.name}-${i}`}>
            <summary>
              <span className="flow-name">{f.name}</span>
              <span className={`sev kind-${f.kind}`}>{f.kind}</span>
              <span className="flow-counts">
                {f.steps.length} steps, {assertions} asserted
              </span>
            </summary>
            {f.description && <p className="flow-desc">{f.description}</p>}
            <ol className="steps">
              {f.steps.map((s, j) => (
                <li key={j} className={s.kind === "assertion" ? "is-assert" : ""}>
                  {stepLine(s)}
                </li>
              ))}
            </ol>
          </details>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------- generate / execute -- */

interface CompiledFlow {
  name: string;
  compiled: boolean;
  steps_total: number;
  unresolved: { verb: string; target: string }[];
}

interface ExecutedFlow {
  name: string;
  status: string;
  steps_total: number;
  steps_passed: number;
  heals_used: number;
  failures: { step_id: string; failure: string | null; error: string }[];
}

const isCompiledFlows = (v: unknown): v is CompiledFlow[] =>
  isArrayOf<CompiledFlow>(v, (f) => typeof f.compiled === "boolean");

const isExecutedFlows = (v: unknown): v is ExecutedFlow[] =>
  isArrayOf<ExecutedFlow>(v, (f) => typeof f.status === "string" && Array.isArray(f.failures));

function CompiledTable({ flows }: { flows: CompiledFlow[] }) {
  return (
    <div className="flows">
      {flows.map((f, i) => (
        <details className="flow" key={`${f.name}-${i}`} open={!f.compiled}>
          <summary>
            <span className="flow-name">{f.name}</span>
            <span className={`sev ${f.compiled ? "sev-minor" : "sev-moderate"}`}>
              {f.compiled ? "compiled" : "partial"}
            </span>
            <span className="flow-counts">
              {f.unresolved.length
                ? `${f.unresolved.length} of ${f.steps_total} unresolved`
                : `${f.steps_total} steps`}
            </span>
          </summary>
          {f.unresolved.length > 0 && (
            <>
              <p className="flow-desc">
                No selector could be resolved for these steps, so the flow
                cannot run as written.
              </p>
              <ol className="steps">
                {f.unresolved.map((s, j) => (
                  <li key={j} className="is-assert">
                    {`${s.verb} ${s.target}`.trim()}
                  </li>
                ))}
              </ol>
            </>
          )}
        </details>
      ))}
    </div>
  );
}

function ExecutedTable({ flows }: { flows: ExecutedFlow[] }) {
  return (
    <div className="flows">
      {flows.map((f, i) => {
        const bad = f.status !== "passed";
        return (
          <details className="flow" key={`${f.name}-${i}`} open={bad}>
            <summary>
              <span className="flow-name">{f.name}</span>
              <span className={`sev ${bad ? "sev-critical" : "kind-happy_path"}`}>
                {f.status}
              </span>
              <span className="flow-counts">
                {f.steps_passed}/{f.steps_total} steps
                {f.heals_used > 0 && `, ${f.heals_used} healed`}
              </span>
            </summary>
            {f.failures.length > 0 && (
              <ol className="steps">
                {f.failures.map((s, j) => (
                  <li key={j} className="is-assert">
                    {s.failure ? `[${s.failure}] ` : ""}
                    {s.error || s.step_id}
                  </li>
                ))}
              </ol>
            )}
          </details>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------- triage -- */

interface Triaged {
  flow: string | null;
  step: string;
  verdict: string;
  confidence: number;
  reasoning: string;
}

const isTriaged = (v: unknown): v is Triaged[] =>
  isArrayOf<Triaged>(v, (t) => typeof t.verdict === "string" && typeof t.step === "string");

/** app_defect is the verdict the whole project turns on: it is a candidate
 *  bug and is never healed, so it reads as an alarm rather than a category. */
function verdictClass(verdict: string): string {
  if (verdict === "app_defect") return "sev-critical";
  if (verdict === "script_issue") return "sev-moderate";
  return "sev-minor";
}

function TriageTable({ triaged }: { triaged: Triaged[] }) {
  return (
    <table className="pages">
      <thead>
        <tr>
          <th>Verdict</th>
          <th>Flow</th>
          <th>Step</th>
          <th className="num">Confidence</th>
          <th>Reasoning</th>
        </tr>
      </thead>
      <tbody>
        {triaged.map((t, i) => (
          <tr key={i}>
            <td>
              <span className={`sev ${verdictClass(t.verdict)}`}>{t.verdict}</span>
            </td>
            <td>{t.flow ?? "—"}</td>
            <td className="num">{t.step}</td>
            <td className="num">{(t.confidence * 100).toFixed(0)}%</td>
            <td>{t.reasoning}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type Heal = {
  flow: string;
  step_id: string;
  to: Record<string, unknown>;
  from?: Record<string, unknown> | null;
  confidence: number;
  reasoning?: string;
};

function isHeals(v: unknown): v is Heal[] {
  return isArrayOf<Heal>(
    v,
    (x) => typeof x.flow === "string" && typeof x.confidence === "number",
  );
}

function selectorText(s: Record<string, unknown> | null | undefined): string {
  if (!s) return "—";
  const strategy = typeof s.strategy === "string" ? s.strategy : "?";
  const value = typeof s.value === "string" ? s.value : "";
  return `${strategy}=${value}`;
}

/** What the healer changed, and why it believed the change was right. The count
 *  of repairs is the least interesting part: what a reader needs to judge is
 *  which locator was swapped for which, and on what evidence. */
function HealTable({ heals }: { heals: Heal[] }) {
  return (
    <table className="heals">
      <thead>
        <tr>
          <th>Flow</th>
          <th>Was</th>
          <th>Now</th>
          <th className="num">Confidence</th>
          <th>Why</th>
        </tr>
      </thead>
      <tbody>
        {heals.map((h, i) => (
          <tr key={`${h.step_id}-${i}`}>
            <td>{h.flow}</td>
            <td>
              <code>{selectorText(h.from)}</code>
            </td>
            <td>
              <code>{selectorText(h.to)}</code>
            </td>
            <td className="num">{Math.round(h.confidence * 100)}%</td>
            <td>{h.reasoning || "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Turns snake_case keys into sentence case: "fully_compiled" reads as
 *  "Fully compiled". */
function label(key: string): string {
  const words = key.replace(/_/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** The leftover scalars, as labelled rows rather than a JSON dump.
 *
 *  These are the numbers a reader most wants at a glance -- how many flows
 *  compiled, what the coverage score was, why a run escalated -- and printing
 *  them as `{"fully_compiled": 0, "partial": 4}` made the ledger look like a
 *  log file. Anything genuinely nested still falls back to JSON, because
 *  inventing a layout for a shape we do not know is worse than showing it
 *  plainly. */
function Facts({ facts }: { facts: Record<string, unknown> }) {
  const entries = Object.entries(facts);
  if (entries.length === 0) return null;

  const simple = entries.filter(
    ([, v]) => v === null || ["string", "number", "boolean"].includes(typeof v),
  );
  const complex = Object.fromEntries(entries.filter(([k]) => !simple.some(([s]) => s === k)));

  return (
    <>
      {simple.length > 0 && (
        <dl className="facts">
          {simple.map(([k, v]) => (
            <div key={k}>
              <dt>{label(k)}</dt>
              <dd>
                {typeof v === "number" && !Number.isInteger(v)
                  ? v.toFixed(2)
                  : String(v ?? "—")}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {Object.keys(complex).length > 0 && (
        <pre className="evidence">{JSON.stringify(complex, null, 2)}</pre>
      )}
    </>
  );
}

/** Evidence, rendered for reading where its shape is known and as JSON
 *  otherwise. A count on its own ("discovered 5 pages", "planned 4 flows")
 *  cannot be checked; the point of the ledger is that a human can verify the
 *  claim, which needs the things themselves. */
function Evidence({ evidence }: { evidence: Record<string, unknown> }) {
  const { pages, flows, compiled, executed, triaged, heals } = evidence;
  const handled = new Set([
    "pages",
    "flows",
    "compiled",
    "executed",
    "triaged",
    "heals",
  ]);
  const rest = Object.fromEntries(
    Object.entries(evidence).filter(([k]) => !handled.has(k)),
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
      {isPlannedFlows(flows) && <FlowTable flows={flows} />}
      {isCompiledFlows(compiled) && <CompiledTable flows={compiled} />}
      {isExecutedFlows(executed) && <ExecutedTable flows={executed} />}
      {isTriaged(triaged) && <TriageTable triaged={triaged} />}
      {isHeals(heals) && <HealTable heals={heals} />}
      <Facts facts={rest} />
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
