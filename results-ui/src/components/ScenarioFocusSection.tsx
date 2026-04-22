import { useMemo } from "react";
import type { SweepRun } from "../types";
import { fmt4, gapTextClass, sweepFigureUrl } from "../utils";

type Props = {
  scenario: string;
  runs: SweepRun[];
  onOpenDetail: (run: SweepRun) => void;
};

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function ScenarioFocusSection({ scenario, runs, onOpenDetail }: Props) {
  const sorted = useMemo(
    () => runs.slice().sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml),
    [runs],
  );

  const stats = useMemo(() => {
    if (runs.length === 0) return null;
    const gaps = runs.map((r) => r.val_loss_gap_mlp_minus_eml);
    return {
      n: runs.length,
      gapMean: mean(gaps),
      gapMin: Math.min(...gaps),
      gapMax: Math.max(...gaps),
      emlMean: mean(runs.map((r) => r.eml_final_val_loss)),
      mlpMean: mean(runs.map((r) => r.mlp_final_val_loss)),
    };
  }, [runs]);

  if (runs.length === 0) {
    return (
      <section>
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Scenario focus</h2>
        <div className="mt-4 rounded-2xl border border-zinc-200/80 bg-white p-6 shadow-sm">
          <p className="text-sm text-zinc-500">
            No runs for <span className="font-medium text-zinc-800">{scenario}</span> with the current depth and
            hidden filters.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section>
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Scenario focus</h2>
      <p className="mt-1 max-w-3xl text-xs leading-relaxed text-zinc-500">
        Selected scenario <strong className="font-medium text-zinc-800">{scenario}</strong>: summary and one card per
        matching run (depth × hidden). Open <strong className="font-medium text-zinc-700">Details</strong> for full
        curves and PNGs.
      </p>

      {stats ? (
        <div className="mt-4 flex flex-wrap gap-3 rounded-2xl border border-zinc-200/80 bg-zinc-50/70 px-4 py-3 text-xs text-zinc-600 sm:px-5">
          <span>
            <span className="text-zinc-400">Runs</span>{" "}
            <span className="font-mono font-semibold text-zinc-900">{stats.n}</span>
          </span>
          <span className="hidden sm:inline text-zinc-200">|</span>
          <span>
            <span className="text-zinc-400">Mean Δ</span>{" "}
            <span className={`font-mono font-semibold tabular-nums ${gapTextClass(stats.gapMean)}`}>
              {fmt4(stats.gapMean)}
            </span>
          </span>
          <span>
            <span className="text-zinc-400">Δ min / max</span>{" "}
            <span className="font-mono tabular-nums text-zinc-800">
              {fmt4(stats.gapMin)} / {fmt4(stats.gapMax)}
            </span>
          </span>
          <span className="hidden md:inline text-zinc-200">|</span>
          <span>
            <span className="text-zinc-400">Mean val BCE</span>{" "}
            <span className="font-mono tabular-nums text-zinc-800">
              EML {fmt4(stats.emlMean)} · MLP {fmt4(stats.mlpMean)}
            </span>
          </span>
        </div>
      ) : null}

      <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {sorted.map((r) => {
          const g = r.val_loss_gap_mlp_minus_eml;
          const dec = r.figures?.decision;
          return (
            <div
              key={r.rel_dir}
              className="flex flex-col overflow-hidden rounded-2xl border border-zinc-200/80 bg-white shadow-sm transition-shadow duration-300 hover:shadow-md"
            >
              <div className="border-b border-zinc-100 px-3 py-2.5">
                <p className="text-xs font-semibold text-zinc-800">
                  d{r.depth} · h{r.hidden}
                  <span className="ml-2 font-normal text-zinc-500">{r.epochs} ep</span>
                </p>
                <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-zinc-600">
                  <div>
                    <dt className="text-zinc-400">EML val</dt>
                    <dd className="font-mono tabular-nums text-zinc-900">{fmt4(r.eml_final_val_loss)}</dd>
                  </div>
                  <div>
                    <dt className="text-zinc-400">MLP val</dt>
                    <dd className="font-mono tabular-nums text-zinc-900">{fmt4(r.mlp_final_val_loss)}</dd>
                  </div>
                  <div>
                    <dt className="text-zinc-400">Δ</dt>
                    <dd className={`font-mono tabular-nums ${gapTextClass(g)}`}>{fmt4(g)}</dd>
                  </div>
                  <div>
                    <dt className="text-zinc-400">Acc</dt>
                    <dd className="font-mono tabular-nums text-zinc-700">
                      {fmt4(r.eml_final_val_acc)} / {fmt4(r.mlp_final_val_acc)}
                    </dd>
                  </div>
                </dl>
              </div>
              {dec ? (
                <div className="flex flex-1 flex-col">
                  <div className="flex min-h-[140px] flex-1 items-center justify-center bg-zinc-100 p-2">
                    <img
                      src={sweepFigureUrl(dec)}
                      alt={`Decision map ${r.scenario} d${r.depth} h${r.hidden}`}
                      className="max-h-[200px] max-w-full object-contain"
                      loading="lazy"
                    />
                  </div>
                  <div className="border-t border-zinc-100 p-2">
                    <button
                      type="button"
                      onClick={() => onOpenDetail(r)}
                      className="w-full rounded-xl bg-zinc-900 py-2 text-xs font-medium text-white transition-colors hover:bg-zinc-800"
                    >
                      Details
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-1 flex-col justify-end border-t border-zinc-100 p-2">
                  <button
                    type="button"
                    onClick={() => onOpenDetail(r)}
                    className="w-full rounded-xl border border-zinc-200 bg-white py-2 text-xs font-medium text-zinc-800 transition-colors hover:bg-zinc-50"
                  >
                    Details
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
