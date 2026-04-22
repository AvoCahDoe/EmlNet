import { useMemo } from "react";
import type { LegacyRow } from "../types";
import { assetUrl, fmt4, gapTextClass } from "../utils";
import { GapBarChart } from "./GapBarChart";

type Props = {
  rows: LegacyRow[];
};

export function LegacyView({ rows }: Props) {
  const sorted = useMemo(
    () => rows.slice().sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml),
    [rows],
  );

  const chartSorted = useMemo(
    () => rows.slice().sort((a, b) => a.val_loss_gap_mlp_minus_eml - b.val_loss_gap_mlp_minus_eml),
    [rows],
  );

  const status =
    rows.length > 0
      ? `Loaded ${rows.length} rows · EML ${rows[0]!.eml_params} vs MLP ${rows[0]!.mlp_params} params.`
      : "No rows loaded.";

  return (
    <div className="mx-auto max-w-6xl space-y-10 px-4 py-10 sm:px-6">
      <header className="space-y-2 border-b border-zinc-200 pb-8">
        <p className="text-xs font-medium uppercase tracking-widest text-zinc-400">EMLnet</p>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 sm:text-3xl">Legacy results</h1>
        <p className="max-w-2xl text-sm leading-relaxed text-zinc-600">
          Flattened benchmark table from <code className="rounded bg-zinc-100 px-1 py-0.5 text-xs">results.json</code>{" "}
          and static assets.
        </p>
        <p className="text-sm text-zinc-500">{status}</p>
      </header>

      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-6">
        <h2 className="mb-4 text-sm font-semibold text-zinc-800">Val loss gap (sorted)</h2>
        {chartSorted.length === 0 ? (
          <p className="text-sm text-zinc-500">No data to chart.</p>
        ) : (
          <GapBarChart
            labels={chartSorted.map((r) => r.scenario)}
            values={chartSorted.map((r) => r.val_loss_gap_mlp_minus_eml)}
            horizontal
          />
        )}
      </section>

      <section className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm">
        <div className="border-b border-zinc-100 px-4 py-3 sm:px-5">
          <h2 className="text-sm font-semibold text-zinc-800">Table</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b border-zinc-100 bg-zinc-50/80 text-left text-xs font-medium text-zinc-500">
                <th className="px-3 py-2">Scenario</th>
                <th className="px-3 py-2 text-right">Epochs</th>
                <th className="px-3 py-2 text-right">Depth</th>
                <th className="px-3 py-2 text-right">Hidden</th>
                <th className="px-3 py-2 text-right">EML val</th>
                <th className="px-3 py-2 text-right">MLP val</th>
                <th className="px-3 py-2 text-right">Gap</th>
                <th className="px-3 py-2 text-right">Acc EML / MLP</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const g = r.val_loss_gap_mlp_minus_eml;
                return (
                  <tr key={`${r.scenario}-${r.depth}-${r.hidden}`} className="border-b border-zinc-50 hover:bg-zinc-50/80">
                    <td className="px-3 py-2 font-medium text-zinc-800">{r.scenario}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-500">{r.epochs}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{r.depth}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{r.hidden}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{fmt4(r.eml_final_val_loss)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-600">{fmt4(r.mlp_final_val_loss)}</td>
                    <td className={`px-3 py-2 text-right tabular-nums ${gapTextClass(g)}`}>{fmt4(g)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-xs text-zinc-500">
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
        <h2 className="mb-3 text-sm font-semibold text-zinc-800">Gallery</h2>
        <p className="mb-4 text-sm text-zinc-500">Legacy flattened assets (lazy).</p>
        <div className="space-y-8">
          {sorted.map((r) => (
            <div key={`${r.scenario}-${r.epochs}-${r.depth}-${r.hidden}`} className="space-y-3">
              <figure className="overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm">
                <figcaption className="border-b border-zinc-100 px-3 py-2 text-xs text-zinc-500">
                  {r.scenario} — decision
                </figcaption>
                <img
                  src={assetUrl(`assets/${r.scenario}_decision.png`)}
                  alt=""
                  className="block w-full bg-zinc-50"
                  loading="lazy"
                />
              </figure>
              <figure className="overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm">
                <figcaption className="border-b border-zinc-100 px-3 py-2 text-xs text-zinc-500">
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
  );
}
