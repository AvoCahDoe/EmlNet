import { useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import type { SweepRun } from "../types";
import { fmt4, sweepFigureUrl } from "../utils";

type ValLossSeries = {
  eml: number[];
  mlp: number[];
};

type Props = {
  run: SweepRun | null;
  onClose: () => void;
};

const font = { size: 11, family: "system-ui, Segoe UI, sans-serif" };
const grid = "rgba(24, 24, 27, 0.06)";

const FIG_COMPACT = ["decision", "loss_linear", "val_loss_gap", "summary_grid"] as const;

function pngFigureKeys(run: SweepRun, all: boolean): string[] {
  const fig = run.figures ?? {};
  const isPng = (k: string) => {
    const p = fig[k];
    return typeof p === "string" && p.endsWith(".png");
  };
  const compact = FIG_COMPACT.filter((k) => isPng(k));
  if (!all) return compact;
  const rest = Object.keys(fig)
    .filter((k) => isPng(k) && !(FIG_COMPACT as readonly string[]).includes(k))
    .sort();
  return [...compact, ...rest];
}

export function RunDetailModal({ run, onClose }: Props) {
  const [series, setSeries] = useState<ValLossSeries | null>(null);
  const [loadAllFigs, setLoadAllFigs] = useState(false);

  const historiesUrl = run?.figures?.histories ? sweepFigureUrl(run.figures.histories) : null;

  useEffect(() => {
    if (!run || !historiesUrl) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(historiesUrl);
        if (!r.ok) throw new Error(String(r.status));
        const data = (await r.json()) as {
          eml?: { val_loss?: number[] };
          mlp?: { val_loss?: number[] };
        };
        if (cancelled) return;
        const eml = data.eml?.val_loss ?? [];
        const mlp = data.mlp?.val_loss ?? [];
        setSeries({ eml, mlp });
      } catch {
        if (!cancelled) setSeries(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [run, historiesUrl]);

  useEffect(() => {
    if (!run) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run, onClose]);

  const lineData = useMemo(() => {
    if (!series || (!series.eml.length && !series.mlp.length)) return null;
    const len = Math.min(series.eml.length, series.mlp.length) || Math.max(series.eml.length, series.mlp.length);
    const labels = Array.from({ length: len }, (_, i) => String(i + 1));
    return {
      labels,
      datasets: [
        {
          label: "EML val",
          data: series.eml.slice(0, len),
          borderColor: "rgb(59, 130, 246)",
          backgroundColor: "transparent",
          tension: 0.15,
          pointRadius: 0,
          borderWidth: 1.5,
        },
        {
          label: "MLP val",
          data: series.mlp.slice(0, len),
          borderColor: "rgb(244, 114, 182)",
          backgroundColor: "transparent",
          tension: 0.15,
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    };
  }, [series]);

  const lineOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index" as const, intersect: false },
    plugins: {
      legend: { position: "bottom" as const, labels: { font, boxWidth: 12, color: "#52525b" } },
    },
    scales: {
      x: {
        title: { display: true, text: "Epoch", font, color: "#71717a" },
        grid: { color: grid },
        ticks: { maxTicksLimit: 8, font, color: "#71717a" },
      },
      y: {
        title: { display: true, text: "Val BCE", font, color: "#71717a" },
        grid: { color: grid },
        ticks: { font, color: "#71717a" },
      },
    },
  };

  if (!run) return null;

  const figKeys = pngFigureKeys(run, loadAllFigs);
  const compactOnly = pngFigureKeys(run, false);
  const extraCount = pngFigureKeys(run, true).length - compactOnly.length;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-zinc-950/40 p-4 sm:items-center"
      role="dialog"
      aria-modal="true"
      aria-labelledby="run-detail-title"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl border border-zinc-200 bg-white shadow-xl">
        <div className="sticky top-0 flex items-start justify-between gap-3 border-b border-zinc-100 bg-white/95 px-5 py-4 backdrop-blur">
          <div>
            <h2 id="run-detail-title" className="text-lg font-semibold tracking-tight text-zinc-900">
              {run.scenario}
            </h2>
            <p className="mt-0.5 text-sm text-zinc-500">
              depth {run.depth} · hidden {run.hidden} · {run.epochs} epochs
            </p>
            {historiesUrl && (
              <a
                href={historiesUrl}
                className="mt-2 inline-block text-xs text-zinc-600 underline underline-offset-2 hover:text-zinc-900"
                target="_blank"
                rel="noreferrer"
              >
                histories.json
              </a>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-3 py-1.5 text-sm font-medium text-zinc-600 hover:bg-zinc-100"
          >
            Close
          </button>
        </div>

        <div className="space-y-5 px-5 py-5">
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div className="rounded-xl border border-zinc-100 bg-zinc-50/80 px-3 py-2">
              <dt className="text-zinc-500">EML val loss</dt>
              <dd className="font-mono font-medium text-zinc-900">{fmt4(run.eml_final_val_loss)}</dd>
            </div>
            <div className="rounded-xl border border-zinc-100 bg-zinc-50/80 px-3 py-2">
              <dt className="text-zinc-500">MLP val loss</dt>
              <dd className="font-mono font-medium text-zinc-900">{fmt4(run.mlp_final_val_loss)}</dd>
            </div>
            <div className="rounded-xl border border-zinc-100 bg-zinc-50/80 px-3 py-2">
              <dt className="text-zinc-500">Gap (MLP − EML)</dt>
              <dd className="font-mono font-medium text-zinc-900">{fmt4(run.val_loss_gap_mlp_minus_eml)}</dd>
            </div>
            <div className="rounded-xl border border-zinc-100 bg-zinc-50/80 px-3 py-2">
              <dt className="text-zinc-500">EML / MLP acc</dt>
              <dd className="font-mono font-medium text-zinc-900">
                {fmt4(run.eml_final_val_acc)} / {fmt4(run.mlp_final_val_acc)}
              </dd>
            </div>
          </dl>

          <section>
            <h3 className="mb-2 text-sm font-medium text-zinc-700">Validation BCE</h3>
            {lineData ? (
              <div className="h-56 rounded-xl border border-zinc-100 bg-white p-2 sm:h-64">
                <Line data={lineData} options={lineOptions} />
              </div>
            ) : (
              <p className="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 px-4 py-6 text-center text-sm text-zinc-500">
                No val loss series in histories.json for this run.
              </p>
            )}
          </section>

          <section>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-medium text-zinc-700">Figures</h3>
              {extraCount > 0 && (
                <button
                  type="button"
                  onClick={() => setLoadAllFigs(true)}
                  disabled={loadAllFigs}
                  className="rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
                >
                  Load all figure PNGs
                </button>
              )}
            </div>
            {figKeys.length === 0 ? (
              <p className="text-sm text-zinc-500">No PNG figures listed for this run.</p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                {figKeys.map((name) => (
                  <figure key={name} className="overflow-hidden rounded-xl border border-zinc-100 bg-zinc-50/50">
                    <figcaption className="border-b border-zinc-100 px-3 py-2 text-xs font-medium text-zinc-600">
                      {name}
                    </figcaption>
                    <img
                      src={sweepFigureUrl(run.figures![name]!)}
                      alt={name}
                      className="w-full bg-white"
                      loading="lazy"
                    />
                  </figure>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
