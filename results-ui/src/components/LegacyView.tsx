import { useMemo, useState } from "react";
import type { LegacyRow } from "../types";
import { assetUrl, fmt4, gapTextClass } from "../utils";
import { GapBarChart } from "./GapBarChart";
import { ResultsShell } from "./ResultsShell";

const field =
  "mt-1.5 w-full rounded-xl border border-zinc-200/90 bg-zinc-50/80 px-3 py-2.5 text-sm text-zinc-900 outline-none transition-[border-color,box-shadow,background-color] duration-200 ease-out focus:border-zinc-300 focus:bg-white focus:ring-2 focus:ring-zinc-900/[0.06]";

type Props = {
  rows: LegacyRow[];
};

export function LegacyView({ rows }: Props) {
  const [scenarioFilter, setScenarioFilter] = useState("");

  const scenarios = useMemo(
    () => Array.from(new Set(rows.map((r) => r.scenario))).sort(),
    [rows],
  );

  const sorted = useMemo(
    () => rows.slice().sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml),
    [rows],
  );

  const filtered = useMemo(() => {
    if (!scenarioFilter) return sorted;
    return sorted.filter((r) => r.scenario === scenarioFilter);
  }, [sorted, scenarioFilter]);

  const chartSorted = useMemo(
    () => filtered.slice().sort((a, b) => a.val_loss_gap_mlp_minus_eml - b.val_loss_gap_mlp_minus_eml),
    [filtered],
  );

  const status =
    rows.length > 0
      ? `${rows.length} rows · EML ${rows[0]!.eml_params} vs MLP ${rows[0]!.mlp_params} parameters`
      : "No rows";

  const sidebar = (
    <>
      <div className="rounded-xl border border-zinc-100 bg-zinc-50/60 px-3 py-3 text-xs leading-relaxed text-zinc-600">
        <p>{status}</p>
        <p className="mt-2 text-zinc-500">
          <span className="font-medium text-zinc-700">{filtered.length}</span> shown in table & chart
        </p>
      </div>

      <div className="space-y-4">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">Parameters</p>
        <label className="block text-xs font-medium text-zinc-600">
          Scenario
          <select
            value={scenarioFilter}
            onChange={(e) => setScenarioFilter(e.target.value)}
            className={field}
          >
            <option value="">All scenarios</option>
            {scenarios.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setScenarioFilter("")}
          className="w-full rounded-xl border border-zinc-200 bg-white py-2.5 text-xs font-medium text-zinc-700 transition-all duration-200 ease-out hover:border-zinc-300 hover:bg-zinc-50"
        >
          Reset filter
        </button>
      </div>
    </>
  );

  return (
    <ResultsShell badge="Legacy" title="Benchmark results" sidebar={sidebar}>
      <div className="space-y-10 lg:space-y-12">
        <header className="max-w-3xl">
          <p className="text-sm leading-relaxed text-zinc-600">
            Static export from <code className="rounded-md bg-zinc-100 px-1.5 py-0.5 text-xs">results.json</code> and
            bundled figures. Use the sidebar to narrow by scenario.
          </p>
        </header>

        <section>
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Gap overview</h2>
          <div className="mt-4 rounded-2xl border border-zinc-200/80 bg-white p-5 shadow-sm sm:p-6">
            {chartSorted.length === 0 ? (
              <p className="text-sm text-zinc-500">No rows for this filter.</p>
            ) : (
              <GapBarChart
                labels={chartSorted.map((r) => r.scenario)}
                values={chartSorted.map((r) => r.val_loss_gap_mlp_minus_eml)}
                horizontal
              />
            )}
          </div>
        </section>

        <section className="overflow-hidden rounded-2xl border border-zinc-200/80 bg-white shadow-sm">
          <div className="border-b border-zinc-100 px-4 py-4 sm:px-6">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Table</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b border-zinc-100 bg-zinc-50/90 text-left text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  <th className="px-4 py-3 sm:px-5">Scenario</th>
                  <th className="px-3 py-3 text-right">Epochs</th>
                  <th className="px-3 py-3 text-right">Depth</th>
                  <th className="px-3 py-3 text-right">Hidden</th>
                  <th className="px-3 py-3 text-right">EML val</th>
                  <th className="px-3 py-3 text-right">MLP val</th>
                  <th className="px-3 py-3 text-right">Gap</th>
                  <th className="px-4 py-3 text-right sm:px-5">Acc EML / MLP</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => {
                  const g = r.val_loss_gap_mlp_minus_eml;
                  return (
                    <tr
                      key={`${r.scenario}-${r.epochs}-${r.depth}-${r.hidden}`}
                      className="border-b border-zinc-50 transition-colors duration-150 ease-out hover:bg-zinc-50/90"
                    >
                      <td className="px-4 py-2.5 font-medium text-zinc-800 sm:px-5">{r.scenario}</td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-zinc-500">{r.epochs}</td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{r.depth}</td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{r.hidden}</td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{fmt4(r.eml_final_val_loss)}</td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{fmt4(r.mlp_final_val_loss)}</td>
                      <td className={`px-3 py-2.5 text-right tabular-nums ${gapTextClass(g)}`}>{fmt4(g)}</td>
                      <td className="px-4 py-2.5 text-right tabular-nums text-xs text-zinc-500 sm:px-5">
                        {fmt4(r.eml_final_val_acc)} / {fmt4(r.mlp_final_val_acc)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Figures</h2>
          <p className="mt-1 text-xs text-zinc-500">One block per table row (same order).</p>
          <div className="mt-4 space-y-8">
            {filtered.map((r) => (
              <div key={`${r.scenario}-${r.epochs}-${r.depth}-${r.hidden}`} className="space-y-3">
                <figure className="overflow-hidden rounded-2xl border border-zinc-200/80 bg-white shadow-sm transition-shadow duration-300 hover:shadow-md">
                  <figcaption className="border-b border-zinc-100 px-4 py-2.5 text-xs font-medium text-zinc-600">
                    {r.scenario} — decision
                  </figcaption>
                  <img
                    src={assetUrl(`assets/${r.scenario}_decision.png`)}
                    alt=""
                    className="block w-full bg-zinc-50"
                    loading="lazy"
                  />
                </figure>
                <figure className="overflow-hidden rounded-2xl border border-zinc-200/80 bg-white shadow-sm transition-shadow duration-300 hover:shadow-md">
                  <figcaption className="border-b border-zinc-100 px-4 py-2.5 text-xs font-medium text-zinc-600">
                    {r.scenario} — loss
                  </figcaption>
                  <img
                    src={assetUrl(`assets/${r.scenario}_loss.png`)}
                    alt=""
                    className="block w-full bg-zinc-50"
                    loading="lazy"
                  />
                </figure>
              </div>
            ))}
          </div>
        </section>
      </div>
    </ResultsShell>
  );
}
