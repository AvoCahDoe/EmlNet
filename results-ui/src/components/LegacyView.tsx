import { useEffect, useMemo, useRef, useState } from "react";
import type { LegacyRow } from "../types";
import { assetUrl, fmt4, gapTextClass } from "../utils";
import { GapBarChart } from "./GapBarChart";
import { HoverPeekBar } from "./HoverPeekBar";
import { IntroductionPanel } from "./IntroductionPanel";
import { ResultsShell } from "./ResultsShell";

const field =
  "mt-1.5 w-full rounded-xl border border-zinc-200/90 bg-zinc-50/80 px-3 py-2.5 text-sm text-zinc-900 outline-none transition-[border-color,box-shadow,background-color] duration-200 ease-out focus:border-zinc-300 focus:bg-white focus:ring-2 focus:ring-zinc-900/[0.06]";

type Props = {
  rows: LegacyRow[];
};

export function LegacyView({ rows }: Props) {
  const [scenarioFilter, setScenarioFilter] = useState("");
  const [peek, setPeek] = useState<LegacyRow | null>(null);
  const peekClearTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelPeekClear = () => {
    if (peekClearTimer.current) {
      clearTimeout(peekClearTimer.current);
      peekClearTimer.current = null;
    }
  };

  const schedulePeekClear = () => {
    cancelPeekClear();
    peekClearTimer.current = setTimeout(() => setPeek(null), 240);
  };

  useEffect(() => () => cancelPeekClear(), []);

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

  /** Gap chart: always aggregate mean Δ per scenario across the full legacy table (ignore scenario filter). */
  const legacyGapAggregate = useMemo(() => {
    const map = new Map<string, number[]>();
    rows.forEach((r) => {
      if (!map.has(r.scenario)) map.set(r.scenario, []);
      map.get(r.scenario)!.push(r.val_loss_gap_mlp_minus_eml);
    });
    const labels = Array.from(map.keys()).sort();
    const values = labels.map((s) => {
      const arr = map.get(s)!;
      return arr.reduce((a, b) => a + b, 0) / arr.length;
    });
    return { labels, values, horizontal: labels.length > 10 };
  }, [rows]);

  const status =
    rows.length > 0
      ? `${rows.length} rows · EML ${rows[0]!.eml_params} vs MLP ${rows[0]!.mlp_params} parameters`
      : "No rows";

  const peekLines = useMemo(() => {
    if (!peek) return [];
    const g = peek.val_loss_gap_mlp_minus_eml;
    return [
      { label: "Epochs", value: String(peek.epochs) },
      { label: "Depth / hidden", value: `${peek.depth} / ${peek.hidden}` },
      { label: "EML val", value: fmt4(peek.eml_final_val_loss) },
      { label: "MLP val", value: fmt4(peek.mlp_final_val_loss) },
      { label: "Δ", value: fmt4(g) },
      { label: "Acc E / M", value: `${fmt4(peek.eml_final_val_acc)} / ${fmt4(peek.mlp_final_val_acc)}` },
    ];
  }, [peek]);

  const sidebar = (
    <>
      <div className="rounded-xl border border-zinc-100 bg-zinc-50/60 px-3 py-3 text-xs leading-relaxed text-zinc-600">
        <p>{status}</p>
        <p className="mt-2 text-zinc-500">
          <span className="font-medium text-zinc-700">{filtered.length}</span> rows in table/figures · gap chart uses
          all <span className="font-medium text-zinc-700">{rows.length}</span> rows
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
    <>
      <ResultsShell badge="Legacy" title="Benchmark results" sidebar={sidebar}>
        <div className={`space-y-10 lg:space-y-12 ${peek ? "pb-28" : "pb-6"}`}>
          <header className="max-w-3xl">
            <p className="text-sm leading-relaxed text-zinc-600">
              Static export from <code className="rounded-md bg-zinc-100 px-1.5 py-0.5 text-xs">results.json</code>.
              Hover table rows for a quick readout.
            </p>
          </header>

          <IntroductionPanel />

          <section>
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Gap overview</h2>
            <p className="mt-1 max-w-3xl text-xs leading-relaxed text-zinc-500">
              Mean validation gap per scenario across <strong className="font-medium text-zinc-700">all</strong> legacy
              rows. The scenario filter only narrows the table and figure gallery.
            </p>
            <div className="mt-4 rounded-2xl border border-zinc-200/80 bg-white p-5 shadow-sm sm:p-6">
              {legacyGapAggregate.labels.length === 0 ? (
                <p className="text-sm text-zinc-500">No rows loaded.</p>
              ) : (
                <>
                  <p className="text-xs text-zinc-500">
                    Mean Δ (MLP − EML) per scenario ·{" "}
                    <span className="text-zinc-400">
                      {rows.length} runs · {legacyGapAggregate.labels.length} scenarios
                    </span>
                  </p>
                  <div className="mt-4">
                    <GapBarChart
                      labels={legacyGapAggregate.labels}
                      values={legacyGapAggregate.values}
                      horizontal={legacyGapAggregate.horizontal}
                    />
                  </div>
                </>
              )}
            </div>
          </section>

          <section className="overflow-hidden rounded-2xl border border-zinc-200/80 bg-white shadow-sm">
            <div className="border-b border-zinc-100 px-4 py-4 sm:px-6">
              <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Table</h2>
              <p className="mt-1 text-xs text-zinc-500">Hover a row for metrics.</p>
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
                <tbody onMouseLeave={schedulePeekClear}>
                  {filtered.map((r) => {
                    const g = r.val_loss_gap_mlp_minus_eml;
                    return (
                      <tr
                        key={`${r.scenario}-${r.epochs}-${r.depth}-${r.hidden}`}
                        onMouseEnter={() => {
                          cancelPeekClear();
                          setPeek(r);
                        }}
                        className="border-b border-zinc-50 transition-colors duration-150 ease-out hover:bg-zinc-100/90"
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
      {peek ? <HoverPeekBar title={peek.scenario} lines={peekLines} /> : null}
    </>
  );
}
