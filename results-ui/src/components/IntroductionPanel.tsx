import { EMLBlockSchematic } from "./EMLBlockSchematic";
import { GapConceptFigure } from "./GapConceptFigure";

export function IntroductionPanel() {
  return (
    <section className="rounded-2xl border border-zinc-200/80 bg-white p-6 shadow-sm sm:p-8">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Introduction</h2>

      <div className="mt-4 max-w-3xl space-y-5 text-sm leading-relaxed text-zinc-700">
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Problem setup</h3>
          <p>
            Each experiment is a <strong>2D classification</strong> task: inputs live in ℝ², labels are binary. The
            dataset is split into training and validation; the same split feeds both models in a run. We compare an{" "}
            <strong>EML</strong> tower against a <strong>matched MLP</strong> with the same depth (number of nonlinear
            blocks), hidden width, and training budget (epochs, optimizer family, loss).
          </p>
        </div>

        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Loss and logits</h3>
          <p>
            Let <span className="font-mono text-zinc-800">z</span> be the scalar logit from the final linear head and{" "}
            <span className="font-mono text-zinc-800">
              σ(z) = 1/(1 + e<sup>−z</sup>)
            </span>{" "}
            the sigmoid. With label{" "}
            <span className="font-mono text-zinc-800">y ∈ {"{0, 1}"}</span>, the training code minimizes binary
            cross-entropy in the <strong>with-logits</strong> form implemented as{" "}
            <code className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs">torch.nn.BCEWithLogitsLoss</code>: for each
            example,
          </p>
          <div className="my-3 rounded-xl border border-zinc-200 bg-zinc-50/90 px-4 py-3 text-center font-mono text-sm text-zinc-900">
            ℓ = −<span className="whitespace-nowrap">[ y log σ(z) + (1 − y) log(1 − σ(z)) ]</span>
          </div>
          <p>
            This is numerically stable and equivalent to the standard BCE written directly with{" "}
            <span className="font-mono">σ(z)</span>. At the end of training we record validation BCE{" "}
            <span className="font-mono text-zinc-800">ℓ</span> on the held-out set for both models.
          </p>
        </div>

        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Validation gap Δ</h3>
          <p>
            Let <span className="font-mono">ℓ<sub>EML</sub><sup>val</sup></span> and{" "}
            <span className="font-mono">ℓ<sub>MLP</sub><sup>val</sup></span> be those validation losses at the final
            epoch. The dashboard highlights
          </p>
          <div className="my-3 rounded-xl border border-zinc-200 bg-zinc-50/90 px-4 py-3 text-center font-mono text-base text-zinc-900">
            Δ = ℓ<sub>MLP</sub>
            <sup>val</sup> − ℓ<sub>EML</sub>
            <sup>val</sup>
          </div>
          <p>
            Lower BCE is better, so <strong className="text-emerald-800">Δ &gt; 0</strong> means the MLP finishes with{" "}
            <em>higher</em> validation loss than EML on that run. <strong className="text-rose-800">Δ &lt; 0</strong>{" "}
            favors the MLP on loss alone. Near-zero values are effectively a tie within noise.
          </p>
        </div>

        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Training loop (code)</h3>
          <p>
            Both models are trained in the same routine (<code className="rounded bg-zinc-100 px-1 text-xs">train_one</code>{" "}
            in <code className="rounded bg-zinc-100 px-1 text-xs">emlnet_pkg.training_core</code>): each epoch, run a
            full-batch forward on the training points, compute <code className="rounded bg-zinc-100 px-1 text-xs">BCEWithLogitsLoss</code>,{" "}
            backpropagate, apply <strong>gradient norm clipping</strong> via{" "}
            <code className="rounded bg-zinc-100 px-1 text-xs">clip_grad_norm_</code>, then an{" "}
            <strong>AdamW</strong> step (with configured learning rate and weight decay). Training and validation BCE
            and accuracies are logged each epoch; the UI&apos;s final metrics and curves come from that history.
          </p>
        </div>

        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">EML vs MLP architecture</h3>
          <p>
            <strong>EML stack</strong> (<code className="rounded bg-zinc-100 px-1 text-xs">DeepEMLClassifier</code>): a
            sequence of <code className="rounded bg-zinc-100 px-1 text-xs">EMLLayer</code> blocks. Each{" "}
            <code className="rounded bg-zinc-100 px-1 text-xs">EMLLayer</code> maps the incoming hidden vector{" "}
            <span className="font-mono">h</span> through two affine maps to channel tensors{" "}
            <span className="font-mono">x</span> and <span className="font-mono">y</span>, then applies the custom
            elementwise map implemented by <code className="rounded bg-zinc-100 px-1 text-xs">EMLFunction</code>:
          </p>
          <div className="my-3 rounded-xl border border-zinc-200 bg-zinc-50/90 px-4 py-3 text-center font-mono text-sm text-zinc-900">
            φ(x, y) = exp(clamp(x)) − log(|y| + ε)
          </div>
          <p>
            Here <span className="font-mono">clamp</span> and <span className="font-mono">ε</span> come from{" "}
            <code className="rounded bg-zinc-100 px-1 text-xs">EMLFunctionConfig</code>; backward passes through a{" "}
            <strong>custom autograd.Function</strong> that masks/clamps partial derivatives for stability. After the
            stack, a single linear layer outputs the logit.
          </p>
          <p className="mt-3">
            <strong>MLP baseline</strong> (<code className="rounded bg-zinc-100 px-1 text-xs">DeepMLPClassifier</code>):
            the same depth count is realized as repeated <code className="rounded bg-zinc-100 px-1 text-xs">Linear → ReLU</code>{" "}
            blocks (first map from 2D to hidden, then hidden-to-hidden), then a linear head to the logit. There is no
            shared EML nonlinearity—only piecewise-linear ReLU units.
          </p>
        </div>

        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Reading this UI</h3>
          <p className="text-xs text-zinc-600">
            The <strong>Gap overview</strong> chart aggregates across <em>all</em> scenarios using only depth and
            hidden filters. Choosing a <strong>single scenario</strong> adds a <strong>Scenario focus</strong> strip
            (cards plus summary) and still filters the run table and heatmaps; open <strong>Details</strong> for full
            PNG figures and validation curves. Accuracy columns are informational; sorting and bar colors emphasize Δ
            from validation BCE.
          </p>
        </div>
      </div>

      <div className="mt-10 space-y-10 border-t border-zinc-100 pt-10">
        <div>
          <h3 className="text-center text-xs font-semibold uppercase tracking-wider text-zinc-500">
            EML layer schematic (not data)
          </h3>
          <div className="mt-4">
            <EMLBlockSchematic />
          </div>
        </div>
        <div>
          <h3 className="text-center text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Gap intuition (not tied to a specific run)
          </h3>
          <div className="mt-4">
            <GapConceptFigure />
          </div>
        </div>
      </div>
    </section>
  );
}
