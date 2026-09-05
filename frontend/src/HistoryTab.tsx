import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  getRun,
  listRuns,
  type HistoryPage,
  type HistoryRow,
  type RunDetail,
} from "./api";
import {
  DecisionLedger,
  ErrorBox,
  GapTable,
  money,
  seconds,
  truncate,
} from "./ui";

export default function HistoryTab() {
  const [page, setPage] = useState<HistoryPage | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setListError(null);
    try {
      setPage(await listRuns(50));
    } catch (e) {
      setListError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!selected) return;
    let live = true;
    setDetail(null);
    setDetailError(null);
    getRun(selected)
      .then((d) => live && setDetail(d))
      .catch(
        (e) => live && setDetailError(e instanceof ApiError ? e.message : String(e)),
      );
    return () => {
      live = false;
    };
  }, [selected]);

  const runs = page?.runs ?? [];

  return (
    <div className="page">
      <section>
        <div className="section-head">
          <h2>Recent runs</h2>
          <div className="actions">
            <span className="aside">{runs.length} shown</span>
            <button className="ghost" onClick={() => void load()} disabled={loading}>
              Refresh
            </button>
          </div>
        </div>

        <ErrorBox error={listError} title="Could not load run history" />

        {page?.store_error && (
          <div className="alert notice">
            <h3>History is unavailable</h3>
            <div>
              Postgres is unreachable, so past runs cannot be listed. Runs
              themselves are unaffected — reports and test files are written to
              disk either way.
              <div className="mono" style={{ marginTop: 6 }}>
                {page.store_error}
              </div>
            </div>
          </div>
        )}

        {loading && <p className="empty">Loading…</p>}

        {!loading && runs.length === 0 && !page?.store_error && !listError && (
          <p className="empty">No runs recorded yet.</p>
        )}

        {runs.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>URL</th>
                <th>Mode</th>
                <th className="num">Flows</th>
                <th className="num">Gaps</th>
                <th className="num">Cost</th>
                <th className="num">Duration</th>
                <th>Outcome</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <HistoryRowView
                  key={r.run_id}
                  row={r}
                  selected={r.run_id === selected}
                  onSelect={() =>
                    setSelected((cur) => (cur === r.run_id ? null : r.run_id))
                  }
                />
              ))}
            </tbody>
          </table>
        )}
      </section>

      {selected && (
        <>
          <ErrorBox error={detailError} title="Could not load that run" />

          {!detail && !detailError && <p className="empty">Loading run…</p>}

          {detail && (
            <>
              <section>
                <div className="section-head">
                  <h2>Decision ledger</h2>
                  <span className="aside mono">{detail.run.run_id}</span>
                </div>
                {detail.run.escalated && (
                  <div className="alert" role="alert">
                    <h3>Escalated</h3>
                    <div>
                      {detail.run.escalation_reason ??
                        "The run could not reach acceptable coverage."}
                    </div>
                  </div>
                )}
                <DecisionLedger decisions={detail.decisions} />
              </section>

              <section>
                <div className="section-head">
                  <h2>Coverage gaps</h2>
                </div>
                <GapTable gaps={detail.gaps} />
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
}

function HistoryRowView({
  row,
  selected,
  onSelect,
}: {
  row: HistoryRow;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <tr
      className={`selectable${selected ? " selected" : ""}`}
      onClick={onSelect}
      tabIndex={0}
      aria-selected={selected}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
    >
      <td className="num">{row.run_id}</td>
      <td>
        <span className="truncate" title={row.url}>
          {truncate(row.url, 50)}
        </span>
      </td>
      <td className="num">{row.mode}</td>
      <td className="num">
        {row.flows_passed}/{row.flows_total}
      </td>
      <td className="num">{row.gaps}</td>
      <td className="num">{money(row.cost_usd)}</td>
      <td className="num">{seconds(row.duration_s)}</td>
      <td>
        <span className={`sev ${row.escalated ? "sev-critical" : "sev-minor"}`}>
          {row.escalated ? "escalated" : "completed"}
        </span>
      </td>
      <td className="num">{row.created_at.slice(0, 19).replace("T", " ")}</td>
    </tr>
  );
}
