// Types mirror the FastAPI surface in server/app/api.py. Where a field is
// optional here it is because the server genuinely may omit it, not for
// convenience.

export type Severity = "critical" | "serious" | "moderate" | "minor";

/** The complete set emitted by orchestrator.py. `string` because the server
 *  types verdict as a free-form str; treat anything unknown as neutral. */
export type Verdict =
  | "continue"
  | "accept"
  | "replan"
  | "regenerate"
  | "escalate";

export interface Decision {
  stage: string;
  verdict: string;
  reason: string;
  next_stage: string;
  evidence: Record<string, unknown>;
  at: string;
}

export interface Gap {
  kind: string;
  description: string;
  evidence: string;
  severity: Severity;
}

export interface UntestedRisk {
  description: string;
  severity: Severity;
}

export interface LlmDescription {
  provider: string;
  models: string[];
  fallbacks: { provider: string; models: string[] }[];
}

export interface RunResult {
  run_id: string;
  url: string;
  mode: string;
  llm: LlmDescription;
  escalated: boolean;
  escalation_reason: string | null;
  summary: string;
  flows: { total: number; passed: number; failed: number };
  gaps: Gap[];
  untested_flow_risk: UntestedRisk[];
  heals_applied: number;
  defects_found: number;
  cost_usd: number;
  duration_s: number;
  generated_files: string[];
  decisions: Decision[];
  links: {
    report_html: string;
    report_text: string;
    report_json: string;
    tests: string;
  };
}

export interface Health {
  status: string;
  app: string;
  environment: string;
  llm:
    | ({ configured: true } & LlmDescription)
    | { configured: false; reason: string };
  database: { connected: boolean; detail: string };
}

export type JobStatus = "running" | "finished" | "escalated" | "failed";

export interface Job {
  job_id: string;
  status: JobStatus;
  stage: string;
  run_id: string | null;
  summary: string | null;
  error: string | null;
  decisions: Decision[];
  result?: RunResult;
}

export interface JobAccepted {
  job_id: string;
  status: "running";
  poll: string;
}

export interface HistoryRow {
  run_id: string;
  url: string;
  mode: string;
  escalated: boolean;
  flows_passed: number;
  flows_total: number;
  gaps: number;
  cost_usd: number;
  duration_s: number;
  summary: string;
  created_at: string;
}

export interface HistoryPage {
  count: number;
  runs: HistoryRow[];
  /** Present when Postgres is unreachable. History is a convenience layer;
   *  the server returns an empty list rather than failing. */
  store_error?: string;
}

export interface RunDetail {
  run: {
    run_id: string;
    url: string;
    mode: string;
    intent: string | null;
    escalated: boolean;
    escalation_reason: string | null;
    flows_total: number;
    flows_passed: number;
    gaps_total: number;
    defects_found: number;
    heals_applied: number;
    cost_usd: number;
    duration_s: number;
    summary_line: string;
    created_at: string | null;
  };
  decisions: (Decision & { seq: number })[];
  gaps: Gap[];
  flow_results: {
    flow_id: string;
    status: string;
    steps_total: number;
    steps_passed: number;
    heals_used: number;
  }[];
}

export interface TestFiles {
  run_id: string;
  count: number;
  directory?: string;
  /** True when the server could not find a per-run snapshot and fell back to
   *  the live tests/generated directory. The files may belong to a later run. */
  stale: boolean;
  note?: string;
  files: Record<string, string>;
}

export interface PrdUpload {
  prd_path: string;
  filename: string;
  bytes: number;
  lines: number;
}

export interface RunRequest {
  url: string;
  username?: string;
  password?: string;
  intent?: string;
  prd_path?: string;
  max_flows: number;
  max_pages: number;
  headless: boolean;
  safe_mode: boolean;
  heal: boolean;
  background: boolean;
}

/** Thrown for any non-2xx response, carrying the server's own message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(
      "Cannot reach the API server. Is uvicorn running on port 8000?",
      0,
    );
  }

  if (!res.ok) {
    // FastAPI errors are {"detail": ...}; anything else falls back to the body.
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (body?.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body; the status line is all we have */
    }
    throw new ApiError(detail, res.status);
  }

  return (await res.json()) as T;
}

export const getHealth = () => request<Health>("/health");

export const startRun = (body: RunRequest) =>
  request<JobAccepted>("/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

export const getJob = (jobId: string) => request<Job>(`/jobs/${jobId}`);

export const listRuns = (limit = 50) =>
  request<HistoryPage>(`/runs?limit=${limit}`);

export const getRun = (runId: string) => request<RunDetail>(`/runs/${runId}`);

export const getTests = (runId: string) =>
  request<TestFiles>(`/runs/${runId}/tests`);

export async function uploadPrd(file: File): Promise<PrdUpload> {
  const form = new FormData();
  form.append("file", file);
  return request<PrdUpload>("/prd", { method: "POST", body: form });
}

/** Report artifacts are served as files, so link to them directly. */
export const reportHtmlUrl = (runId: string) => `/api/runs/${runId}/report.html`;
export const reportJsonUrl = (runId: string) => `/api/runs/${runId}/report.json`;

/** Fetch an artifact as text.
 *
 *  A run can exist with no report on disk -- the artifacts directory is not
 *  the database, and a run whose REPORT stage never completed leaves a test
 *  snapshot behind with no report beside it. Pointing an iframe or a download
 *  link straight at the URL renders the 404 body as though it were the
 *  report, so the caller has to know whether the file is actually there. */
async function artifactText(url: string, what: string): Promise<string> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError(`Cannot reach the API server to load the ${what}.`, 0);
  }
  if (res.status === 404) {
    throw new ApiError(`This run has no ${what} on disk.`, 404);
  }
  if (!res.ok) {
    throw new ApiError(`${res.status} ${res.statusText}`, res.status);
  }
  return res.text();
}

export const getReportHtml = (runId: string) =>
  artifactText(reportHtmlUrl(runId), "HTML report");

export const getReportJson = (runId: string) =>
  artifactText(reportJsonUrl(runId), "JSON report");
