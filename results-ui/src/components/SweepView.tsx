import { useMemo, useState } from "react";
import type { SweepManifest, SweepRun } from "../types";
import { sweepChartPayload } from "../sweepChartPayload";
import { fmt4, gapTextClass, mid, SAMPLE_SCENARIOS, sweepFigureUrl } from "../utils";
import { GapBarChart } from "./GapBarChart";
import { ResultsShell } from "./ResultsShell";
import { RunDetailModal } from "./RunDetailModal";

const field =
  "mt-1.5 w-full rounded-xl border border-zinc-200/90 bg-zinc-50/80 px-3 py-2.5 text-sm text-zinc-900 outline-none transition-[border-color,box-shadow,background-color] duration-200 ease-out focus:border-zinc-300 focus:bg-white focus:ring-2 focus:ring-zinc-900/[0.06]";

type Props = {
  manifest: SweepManifest;
};

export function SweepView({ manifest }: Props) {
  const scenarios = useMemo(
    () => Array.from(new Set(manifest.runs.map((r) => r.scenario))).sort(),
    [manifest.runs],
  );

  const [scenario, setScenario] = useState("");
  const [depth, setDepth] = useState("");
  const [hidden, setHidden] = useState("");
  const [allHeatmaps, setAllHeatmaps] = useState(false);
  const [detailRun, setDetailRun] = useState<SweepRun | null>(null);

  const filteredRuns = useMemo(() => {
    return manifest.runs.filter((r) => {
      if (scenario && r.scenario !== scenario) return false;
      if (depth && String(r.depth) !== depth) return false;
      if (hidden && String(r.hidden) !== hidden) return false;
      return true;
    });
  }, [manifest.runs, scenario, depth, hidden]);

  const tableRows = useMemo(() => {
    return filteredRuns.slice().sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml);
  }, [filteredRuns]);

  const gapPayload = useMemo(() => {
    if (filteredRuns.length === 0) return null;
    return sweepChartPayload(filteredRuns, scenario, depth, hidden);
  }, [filteredRuns, scenario, depth, hidden]);

  const heatmapList = useMemo(() => {
    if (!allHeatmaps && !scenario) return [];
    return allHeatmaps ? manifest.heatmaps : manifest.heatmaps.filter((x) => x.scenario === scenario);
  }, [allHeatmaps, scenario, manifest.heatmaps]);

  const summary = useMemo(() => {
    const nHm = allHeatmaps ? manifest.heatmaps.length : scenario ? 1 : 0;
    return {
      runs: filteredRuns.length,
      heatmaps: nHm,
      total: manifest.runs.length,
    };
  }, [filteredRuns.length, allHeatmaps, scenario, manifest.heatmaps.length, manifest.runs.length]);

  const preset = manifest.preset != null ? String(manifest.preset) : "";
  const ep = manifest.epochs_default != null ? String(manifest.epochs_default) : "";

  const applySample = (sc: string) => {
    setScenario(sc);
    setDepth(String(mid(manifest.depths)));
    setHidden(String(mid(manifest.hiddens)));
    setAllHeatmaps(false);
  };

  const pickGalleryScenario = (sc: string) => {
    setScenario(sc);
    setDepth("");
    setHidden("");
    setAllHeatmaps(false);
  };

  const resetFilters = () => {
    setScenario("");
    setDepth("");
    setHidden("");
    setAllHeatmaps(false);
  };

  const sidebar = (
    <>
      <div className="rounded-xl border border-zinc-100 bg-zinc-50/60 px-3 py-3 text-xs leading-relaxed text-zinc-600">
        <p>
          <span className="font-medium text-zinc-800">{summary.total}</span> runs · depths [{manifest.depths.join(", ")}]
          · hiddens [{manifest.hiddens.join(", ")}]
          {ep ? ` · ${ep} epochs` : ""}
          {preset ? ` · ${preset}` : ""}
        </p>
        <p className="mt-2 text-zinc-500">
          <span className="font-medium text-zinc-700">{summary.runs}</span> matched
          {summary.heatmaps > 0 ? (
            <>
              {" "}
              · <span className="font-medium text-zinc-700">{summary.heatmaps}</span> heatmap
              {summary.heatmaps === 1 ? "" : "s"}
            </>
          ) : null}
        </p>
      </div>

      <div className="space-y-4">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">Parameters</p>

        <label className="block text-xs font-medium text-zinc-600">
          Scenario
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
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

        <label className="block text-xs font-medium text-zinc-600">
          Depth
          <select value={depth} onChange={(e) => setDepth(e.target.value)} className={field}>
            <option value="">All</option>
            {manifest.depths.map((d) => (
              <option key={d} value={String(d)}>
                {d}
              </option>
            ))}
          </select>
        </label>

        <label className="block text-xs font-medium text-zinc-600">
          Hidden width
          <select value={hidden} onChange={(e) => setHidden(e.target.value)} className={field}>
            <option value="">All</option>
            {manifest.hiddens.map((h) => (
              <option key={h} value={String(h)}>
                {h}
              </option>
            ))}
          </select>
        </label>

        <label className="flex cursor-pointer items-center justify-between gap-3 rounded-xl border border-zinc-100 bg-zinc-50/60 px-3 py-3 transition-colors duration-200 hover:bg-zinc-50">
          <span className="text-xs font-medium text-zinc-700">All scenario heatmaps</span>
          <input
            type="checkbox"
            checked={allHeatmaps}
            onChange={(e) => setAllHeatmaps(e.target.checked)}
            className="h-4 w-4 rounded border-zinc-300 text-zinc-900 transition-colors focus:ring-2 focus:ring-zinc-900/10"
          />
        </label>

        <button
          type="button"
          onClick={resetFilters}
          className="w-full rounded-xl border border-zinc-200 bg-white py-2.5 text-xs font-medium text-zinc-700 transition-all duration-200 ease-out hover:border-zinc-300 hover:bg-zinc-50"
        >
          Reset filters
        </button>
      </div>

      <div>
        <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">Quick picks</p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {SAMPLE_SCENARIOS.filter((sc) => scenarios.includes(sc)).map((sc) => (
            <button
              key={sc}
              type="button"
              onClick={() => applySample(sc)}
              className="rounded-lg border border-zinc-200/90 bg-white px-2.5 py-1.5 text-[11px] font-medium text-zinc-700 transition-all duration-200 ease-out hover:border-zinc-300 hover:bg-zinc-50"
            >
              {sc}
            </button>
          ))}
        </div>
      </div>
    </>
  );

  return (
    <>
    <ResultsShell badge="Sweep" title="Benchmark results" sidebar={sidebar}>
      <div className="space-y-10 lg:space-y-12">
        <header className="max-w-3xl">
          <p className="text-sm leading-relaxed text-zinc-600">
            Explore every run: adjust parameters in the sidebar; charts and tables update immediately. Open a row for
            full figures and training curves.
          </p>
        </header>

        <section className="transition-opacity duration-300 ease-out">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Sample decision maps</h2>
          <p className="mt-1 text-xs text-zinc-500">Click a card to focus that scenario.</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {SAMPLE_SCENARIOS.filter((sc) => scenarios.includes(sc)).map((sc) => {
              const run = manifest.runs.find((r) => r.scenario === sc);
              const dec = run?.figures?.decision;
              if (!run || !dec) return null;
              return (
                <button
                  key={sc}
                  type="button"
                  onClick={() => pickGalleryScenario(sc)}
                  className="group overflow-hidden rounded-2xl border border-zinc-200/80 bg-white text-left shadow-sm transition-all duration-300 ease-out hover:border-zinc-300/90 hover:shadow-md"
                >
                  <div className="border-b border-zinc-100 px-3 py-2 text-xs font-medium text-zinc-600 transition-colors duration-200 group-hover:text-zinc-900">
                    {sc}
                  </div>
                  <div className="aspect-[4/3] overflow-hidden bg-zinc-100">
                    <img
                      src={sweepFigureUrl(dec)}
                      alt=""
                      className="h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-[1.03]"
                      loading="lazy"
                      decoding="async"
                    />
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        <section>
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Gap overview</h2>
          <div className="mt-4 rounded-2xl border border-zinc-200/80 bg-white p-5 shadow-sm transition-shadow duration-300 sm:p-6">
            {gapPayload ? (
              <>
                <p className="text-xs leading-relaxed text-zinc-500">
                  {gapPayload.caption}{" "}
                  <span className="text-zinc-400">({filteredRuns.length} runs)</span>
                </p>
                <div className="mt-4">
                  <GapBarChart
                    labels={gapPayload.labels}
                    values={gapPayload.values}
                    horizontal={gapPayload.horizontal}
                  />
                </div>
              </>
            ) : (
              <p className="text-sm text-zinc-500">No runs match the current filters.</p>
            )}
          </div>
        </section>

        <section>
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Heatmaps</h2>
          <div className="mt-4 rounded-2xl border border-zinc-200/80 bg-white p-5 shadow-sm sm:p-6">
            {!allHeatmaps && !scenario ? (
              <p className="text-sm leading-relaxed text-zinc-500">
                Choose a scenario or enable <span className="font-medium text-zinc-700">All scenario heatmaps</span>{" "}
                in the sidebar to show heatmaps here.
              </p>
            ) : heatmapList.length === 0 ? (
              <p className="text-sm text-zinc-500">No heatmaps for this selection.</p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                {heatmapList.map((hm) => (
                  <div
                    key={`${hm.scenario}-${hm.path}`}
                    className="overflow-hidden rounded-xl border border-zinc-100 bg-zinc-50/40 transition-shadow duration-300 hover:shadow-sm"
                  >
                    <div className="border-b border-zinc-100 px-3 py-2 text-xs font-medium text-zinc-600">
                      {hm.scenario}
                    </div>
                    <img src={sweepFigureUrl(hm.path)} alt="" className="block w-full bg-white" loading="lazy" />
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="overflow-hidden rounded-2xl border border-zinc-200/80 bg-white shadow-sm transition-shadow duration-300">
          <div className="border-b border-zinc-100 px-4 py-4 sm:px-6">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">All runs</h2>
            <p className="mt-1 text-xs text-zinc-500">Sorted by validation gap (MLP − EML), descending.</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-zinc-100 bg-zinc-50/90 text-left text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  <th className="px-4 py-3 sm:px-5">Scenario</th>
                  <th className="px-3 py-3 text-right">Depth</th>
                  <th className="px-3 py-3 text-right">Hidden</th>
                  <th className="px-3 py-3 text-right">Epochs</th>
                  <th className="px-3 py-3 text-right">EML val</th>
                  <th className="px-3 py-3 text-right">MLP val</th>
                  <th className="px-3 py-3 text-right">Gap</th>
                  <th className="px-3 py-3 text-right">Acc</th>
                  <th className="px-4 py-3 sm:px-5" />
                </tr>
              </thead>
              <tbody>
                {tableRows.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="px-5 py-14 text-center text-sm text-zinc-400">
                      No runs match these filters.
                    </td>
                  </tr>
                ) : (
                  tableRows.map((r) => {
                    const g = r.val_loss_gap_mlp_minus_eml;
                    return (
                      <tr
                        key={r.rel_dir}
                        className={`border-b border-zinc-50 transition-colors duration-150 ease-out hover:bg-zinc-50/90 ${detailRun?.rel_dir === r.rel_dir ? "bg-zinc-100/80" : ""}`}
                      >
                        <td className="px-4 py-2.5 font-medium text-zinc-800 sm:px-5">{r.scenario}</td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{r.depth}</td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{r.hidden}</td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-zinc-500">{r.epochs}</td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{fmt4(r.eml_final_val_loss)}</td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-zinc-600">{fmt4(r.mlp_final_val_loss)}</td>
                        <td className={`px-3 py-2.5 text-right tabular-nums ${gapTextClass(g)}`}>{fmt4(g)}</td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-xs text-zinc-500">
                          {fmt4(r.eml_final_val_acc)} / {fmt4(r.mlp_final_val_acc)}
                        </td>
                        <td className="px-4 py-2.5 sm:px-5">
                          <button
                            type="button"
                            onClick={() => setDetailRun(r)}
                            className="rounded-lg px-2 py-1 text-xs font-medium text-zinc-600 transition-colors duration-150 hover:bg-zinc-100 hover:text-zinc-900"
                          >
                            Details
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </ResultsShell>
    <RunDetailModal
      key={detailRun?.rel_dir ?? "closed"}
      run={detailRun}
      onClose={() => setDetailRun(null)}
    />
    </>
  );
}
