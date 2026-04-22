import { GapConceptFigure } from "./GapConceptFigure";

export function IntroductionPanel() {
  return (
    <section className="rounded-2xl border border-zinc-200/80 bg-white p-6 shadow-sm sm:p-8">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Introduction</h2>
      <div className="mt-4 max-w-3xl space-y-4 text-sm leading-relaxed text-zinc-700">
        <p>
          This dashboard compares two classifiers on the same synthetic 2D tasks: an <strong>EML</strong> model and a
          matched <strong>MLP</strong> baseline. Both are trained with the same depth, hidden width, epochs, and
          validation split. The goal is to see whether EML reaches better or comparable generalization as measured by
          validation loss.
        </p>
        <p>
          Let{" "}
          <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[13px] text-zinc-800">
            ℓ<sub className="text-xs">EML</sub>
            <sup className="text-[10px]">val</sup>
          </span>{" "}
          and{" "}
          <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[13px] text-zinc-800">
            ℓ<sub className="text-xs">MLP</sub>
            <sup className="text-[10px]">val</sup>
          </span>{" "}
          denote binary cross-entropy (BCE) on the held-out validation set at the end of training. We summarize the
          comparison with the <strong>validation gap</strong>
        </p>
        <div className="rounded-xl border border-zinc-200 bg-zinc-50/90 px-4 py-3 text-center font-mono text-base text-zinc-900">
          Δ = ℓ<sub>MLP</sub>
          <sup>val</sup> − ℓ<sub>EML</sub>
          <sup>val</sup>
        </div>
        <p>
          Because lower BCE is better, <strong className="text-emerald-800">Δ &gt; 0</strong> means the MLP ends with{" "}
          <em>higher</em> validation loss than EML, so EML is ahead on this metric. If{" "}
          <strong className="text-rose-800">Δ &lt; 0</strong>, the MLP has lower validation loss for that run. Values
          near zero indicate a tie within numerical noise.
        </p>
        <p className="text-xs text-zinc-500">
          Accuracy is reported for completeness; the bar charts and sorting emphasize Δ from validation BCE, aligned
          with the loss curves in each run&apos;s detail view.
        </p>
      </div>
      <div className="mt-8">
        <h3 className="text-center text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Schematic (not tied to a specific run)
        </h3>
        <div className="mt-4">
          <GapConceptFigure />
        </div>
      </div>
    </section>
  );
}
