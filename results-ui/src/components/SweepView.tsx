import { useCallback, useMemo, useState } from "react";
import type { SweepManifest, SweepRun } from "../types";
import { sweepChartPayload } from "../sweepChartPayload";
import { fmt4, gapTextClass, mid, SAMPLE_SCENARIOS, sweepFigureUrl } from "../utils";
import { GapBarChart } from "./GapBarChart";
import { RunDetailModal } from "./RunDetailModal";

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
  const [loaded, setLoaded] = useState(false);
  const [loadHint, setLoadHint] = useState("");
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
    if (!loaded || filteredRuns.length === 0) return null;
    return sweepChartPayload(filteredRuns, scenario, depth, hidden);
  }, [loaded, filteredRuns, scenario, depth, hidden]);

  const heatmapList = useMemo(() => {
    if (!loaded) return [];
    if (!allHeatmaps && !scenario) return [];
    return allHeatmaps ? manifest.heatmaps : manifest.heatmaps.filter((x) => x.scenario === scenario);
  }, [loaded, allHeatmaps, scenario, manifest.heatmaps]);

  const onLoad = useCallback(() => {
    if (!scenario && !allHeatmaps) {
      setLoadHint("Choose a scenario (or enable all heatmaps).");
      return;
    }
    setLoaded(true);
    const nHm = allHeatmaps ? manifest.heatmaps.length : scenario ? 1 : 0;
    setLoadHint(
      `Showing ${filteredRuns.length} run(s)${nHm ? ` · ${nHm} heatmap(s)` : ""}.`,
    );
  }, [scenario, allHeatmaps, filteredRuns.length, manifest.heatmaps.length]);

  const applySample = (sc: string) => {
    const di = String(mid(manifest.depths));
    const hi = String(mid(manifest.hiddens));
    const n = manifest.runs.filter(
      (r) => r.scenario === sc && String(r.depth) === di && String(r.hidden) === hi,
    ).length;
    setScenario(sc);
    setDepth(di);
    setHidden(hi);
    setAllHeatmaps(false);
    setLoaded(true);
    setLoadHint(`Showing ${n} run(s) · 1 heatmap(s).`);
  };

  const pickGalleryScenario = (sc: string) => {
    const n = manifest.runs.filter((r) => r.scenario === sc).length;
    setScenario(sc);
    setDepth("");
    setHidden("");
    setAllHeatmaps(false);
    setLoaded(true);
    setLoadHint(`Showing ${n} run(s) · 1 heatmap(s).`);
  };

  const preset =
    manifest.preset != null ? String(manifest.preset) : "";
  const ep = manifest.epochs_default != null ? String(manifest.epochs_default) : "";

  return (
    <div className="mx-auto max-w-6xl space-y-10 px-4 py-10 sm:px-6">
      <header className="space-y-2 border-b border-zinc-200 pb-8">
        <p className="text-xs font-medium uppercase tracking-widest text-zinc-400">EMLnet</p>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 sm:text-3xl">Benchmark results</h1>
        <p className="max-w-2xl text-sm leading-relaxed text-zinc-600">
          Sweep · depths [{manifest.depths.join(", ")}] · hiddens [{manifest.hiddens.join(", ")}] ·{" "}
          {manifest.runs.length} runs
          {ep ? ` · ${ep} ep/run` : ""}
          {preset ? ` · ${preset}` : ""}
        </p>
        <p className="text-xs text-zinc-500">
          {manifest.depths.length}×{manifest.hiddens.length} · [{manifest.depths.join(", ")}] × [
          {manifest.hiddens.join(", ")}]
        </p>
      </header>

      <section>
        <h2 className="mb-3 text-sm font-semibold text-zinc-800">Gallery</h2>
        <p className="mb-4 text-sm text-zinc-500">
          Decision-map thumbnails (lazy). Click a card to load that scenario in the sweep view.
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {SAMPLE_SCENARIOS.filter((sc) => scenarios.includes(sc)).map((sc) => {
            const run = manifest.runs.find((r) => r.scenario === sc);
            const dec = run?.figures?.decision;
            if (!run || !dec) return null;
            return (
              <button
                key={sc}
                type="button"
                onClick={() => pickGalleryScenario(sc)}
                className="overflow-hidden rounded-xl border border-zinc-200 bg-white text-left shadow-sm transition-colors hover:border-zinc-300"
              >
                <div className="border-b border-zinc-100 px-3 py-2 text-xs text-zinc-500">{sc}</div>
                <div className="aspect-[4/3] bg-zinc-100">
                  <img
                    src={sweepFigureUrl(dec)}
                    alt=""
                    className="h-full w-full object-cover"
                    loading="lazy"
                    decoding="async"
                  />
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-6">
        <h2 className="mb-4 text-sm font-semibold text-zinc-800">Sweep explorer</h2>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end">
          <label className="flex min-w-0 flex-1 flex-col gap-1 text-xs font-medium text-zinc-600">
            Scenario
            <select
              value={scenario}
              onChange={(e) => {
                setScenario(e.target.value);
                setLoaded(false);
                setLoadHint("");
              }}
              className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900"
            >
              <option value="">Choose…</option>
              {scenarios.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-xs font-medium text-zinc-600">
            Depth
            <select
              value={depth}
              onChange={(e) => {
                setDepth(e.target.value);
                setLoaded(false);
                setLoadHint("");
              }}
              className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900"
            >
              <option value="">All depths</option>
              {manifest.depths.map((d) => (
                <option key={d} value={String(d)}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-col gap-1 text-xs font-medium text-zinc-600">
            Hidden
            <select
              value={hidden}
              onChange={(e) => {
                setHidden(e.target.value);
                setLoaded(false);
                setLoadHint("");
              }}
              className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900"
            >
              <option value="">All hiddens</option>
              {manifest.hiddens.map((h) => (
                <option key={h} value={String(h)}>
                  {h}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs text-zinc-600">
            <input
              type="checkbox"
              checked={allHeatmaps}
              onChange={(e) => {
                setAllHeatmaps(e.target.checked);
                setLoaded(false);
                setLoadHint("");
              }}
              className="rounded border-zinc-300"
            />
            All-scenario heatmaps
          </label>
          <button
            type="button"
            onClick={onLoad}
            className="rounded-lg bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-zinc-800"
          >
            Load results
          </button>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {SAMPLE_SCENARIOS.filter((sc) => scenarios.includes(sc)).map((sc) => (
            <button
              key={sc}
              type="button"
              onClick={() => applySample(sc)}
              className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 hover:border-zinc-300 hover:bg-zinc-50"
            >
              {sc}
            </button>
          ))}
        </div>
        {loadHint ? <p className="mt-3 text-sm text-zinc-500">{loadHint}</p> : null}
      </section>

      {loaded ? (
        <>
          <section>
            <h2 className="mb-3 text-sm font-semibold text-zinc-800">Heatmaps</h2>
            {!allHeatmaps && !scenario ? (
              <p className="text-sm text-zinc-500">
                Pick a scenario for one heatmap, or enable all-scenarios heatmaps before loading.
              </p>
            ) : heatmapList.length === 0 ? (
              <p className="text-sm text-zinc-500">No heatmaps for this selection.</p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                {heatmapList.map((hm) => (
                  <div key={`${hm.scenario}-${hm.path}`} className="overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm">
                    <div className="border-b border-zinc-100 px-3 py-2 text-xs text-zinc-500">{hm.scenario}</div>
                    <img src={sweepFigureUrl(hm.path)} alt="" className="block w-full bg-zinc-50" loading="lazy" />
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-6">
            <h2 className="mb-1 text-sm font-semibold text-zinc-800">Gap chart</h2>
            {gapPayload ? (
              <>
                <p className="mb-4 text-xs text-zinc-500">
                  {gapPayload.caption} ({filteredRuns.length} runs).
                </p>
                <GapBarChart
                  labels={gapPayload.labels}
                  values={gapPayload.values}
                  horizontal={gapPayload.horizontal}
                />
              </>
            ) : (
              <p className="text-sm text-zinc-500">No runs for this filter.</p>
            )}
          </section>

          <section className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm">
            <div className="border-b border-zinc-100 px-4 py-3 sm:px-5">
              <h2 className="text-sm font-semibold text-zinc-800">Runs</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <thead>
                  <tr className="border-b border-zinc-100 bg-zinc-50/80 text-left text-xs font-medium text-zinc-500">
                    <th className="px-3 py-2">Scenario</th>
                    <th className="px-3 py-2 text-right">Depth</th>
                    <th className="px-3 py-2 text-right">Hidden</th>
                    <th className="px-3 py-2 text-right">Epochs</th>
                    <th className="px-3 py-2 text-right">EML val</th>
                    <th className="px-3 py-2 text-right">MLP val</th>
                    <th className="px-3 py-2 text-right">Gap</th>
                    <th className="px-3 py-2 text-right">Acc EML / MLP</th>
                    <th className="px-3 py-2" />
                  </tr>
                </thead>
                <tbody>
                  {tableRows.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-3 py-10 text-center text-sm text-zinc-400">
                        No runs match these filters.
                      </td>
                    </tr>
                  ) : (
                    tableRows.map((r) => {
                      const g = r.val_loss_gap_mlp_minus_eml;
                      return (
                        <tr
                          key={r.rel_dir}
                          className={`border-b border-zinc-50 transition-colors hover:bg-zinc-50/80 ${detailRun?.rel_dir === r.rel_dir ? "bg-zinc-100" : ""}`}
                        >
                          <td className="px-3 py-2 font-medium text-zinc-800">{r.scenario}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{r.depth}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{r.hidden}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-zinc-500">{r.epochs}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{fmt4(r.eml_final_val_loss)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{fmt4(r.mlp_final_val_loss)}</td>
                          <td className={`px-3 py-2 text-right tabular-nums ${gapTextClass(g)}`}>{fmt4(g)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-xs text-zinc-500">
                            {fmt4(r.eml_final_val_acc)} / {fmt4(r.mlp_final_val_acc)}
                          </td>
                          <td className="px-3 py-2">
                            <button
                              type="button"
                              onClick={() => setDetailRun(r)}
                              className="text-xs text-zinc-600 underline underline-offset-2 hover:text-zinc-900"
                            >
                              Open
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
        </>
      ) : (
        <section className="rounded-2xl border border-dashed border-zinc-200 bg-zinc-50/50 px-4 py-12 text-center text-sm text-zinc-500">
          Use samples or set parameters, then <strong className="text-zinc-700">Load results</strong>.
        </section>
      )}

      <RunDetailModal
        key={detailRun?.rel_dir ?? "closed"}
        run={detailRun}
        onClose={() => setDetailRun(null)}
      />
    </div>
  );
}
