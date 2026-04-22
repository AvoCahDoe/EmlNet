/** Illustrative SVG (not real data): validation BCE bars and the gap Δ. */
export function GapConceptFigure() {
  return (
    <figure className="mx-auto max-w-md rounded-xl border border-zinc-200/80 bg-zinc-50/80 px-4 py-5">
      <figcaption className="sr-only">Illustration of validation loss gap</figcaption>
      <svg viewBox="0 0 360 150" className="h-auto w-full" aria-hidden>
        <text x="8" y="18" fill="#71717a" fontSize="11" fontFamily="system-ui, sans-serif">
          Lower is better (BCE on validation)
        </text>
        <text x="8" y="54" fill="#3f3f46" fontSize="12" fontWeight="600" fontFamily="system-ui, sans-serif">
          EML
        </text>
        <rect x="72" y="40" width="120" height="18" rx="4" fill="rgb(59 130 246 / 0.85)" />
        <text x="8" y="94" fill="#3f3f46" fontSize="12" fontWeight="600" fontFamily="system-ui, sans-serif">
          MLP
        </text>
        <rect x="72" y="80" width="200" height="18" rx="4" fill="rgb(244 114 182 / 0.88)" />
        <path
          d="M 198 42 L 198 28 L 268 28 L 268 78"
          fill="none"
          stroke="#a1a1aa"
          strokeWidth="1.2"
        />
        <rect x="216" y="14" width="124" height="22" rx="5" fill="rgb(16 185 129 / 0.12)" stroke="rgb(5 150 105 / 0.45)" strokeWidth="1" />
        <text x="228" y="29" fill="#047857" fontSize="10" fontWeight="600" fontFamily="system-ui, sans-serif">
          Δ &gt; 0 → EML lower loss
        </text>
        <text x="8" y="136" fill="#71717a" fontSize="10" fontFamily="system-ui, sans-serif">
          Δ = ℓ_MLP^val − ℓ_EML^val (matched width, depth, and budget)
        </text>
      </svg>
    </figure>
  );
}
