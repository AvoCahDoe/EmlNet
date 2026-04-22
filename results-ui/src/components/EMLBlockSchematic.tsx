/** Tiny schematic: dual linear branches into EML nonlinearity (code-aligned, not data). */
export function EMLBlockSchematic() {
  return (
    <figure className="mx-auto max-w-lg rounded-xl border border-zinc-200/80 bg-zinc-50/80 px-3 py-4">
      <figcaption className="mb-2 text-center text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
        One EML layer (elementwise)
      </figcaption>
      <svg viewBox="0 0 400 120" className="h-auto w-full" aria-hidden>
        <rect x="8" y="38" width="72" height="36" rx="6" fill="rgb(244 244 245)" stroke="#d4d4d8" />
        <text x="44" y="60" textAnchor="middle" fontSize="11" fill="#52525b" fontFamily="system-ui,sans-serif">
          h_in
        </text>
        <path d="M 80 56 L 108 56" stroke="#a1a1aa" strokeWidth="1.5" markerEnd="url(#emlBlockArr)" />
        <rect x="110" y="18" width="88" height="32" rx="5" fill="rgb(239 246 255)" stroke="#93c5fd" />
        <text x="154" y="38" textAnchor="middle" fontSize="10" fill="#1d4ed8" fontFamily="ui-monospace,monospace">
          W_x h + b_x
        </text>
        <rect x="110" y="62" width="88" height="32" rx="5" fill="rgb(253 242 248)" stroke="#f9a8d4" />
        <text x="154" y="82" textAnchor="middle" fontSize="10" fill="#be185d" fontFamily="ui-monospace,monospace">
          W_y h + b_y
        </text>
        <path d="M 198 34 L 228 50" stroke="#a1a1aa" strokeWidth="1.2" />
        <path d="M 198 78 L 228 54" stroke="#a1a1aa" strokeWidth="1.2" />
        <rect x="230" y="38" width="100" height="36" rx="6" fill="rgb(250 250 250)" stroke="#71717a" strokeWidth="1.2" />
        <text x="280" y="56" textAnchor="middle" fontSize="8.5" fill="#3f3f46" fontFamily="system-ui,sans-serif">
          exp(clamp(x)) − log(|y|+ε)
        </text>
        <path d="M 330 56 L 358 56" stroke="#a1a1aa" strokeWidth="1.5" />
        <rect x="360" y="38" width="32" height="36" rx="6" fill="rgb(244 244 245)" stroke="#d4d4d8" />
        <text x="376" y="60" textAnchor="middle" fontSize="11" fill="#52525b" fontFamily="system-ui,sans-serif">
          out
        </text>
        <defs>
          <marker id="emlBlockArr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 z" fill="#a1a1aa" />
          </marker>
        </defs>
      </svg>
      <p className="mt-2 text-center text-[10px] leading-snug text-zinc-500">
        Forward matches <code className="rounded bg-white px-1">emlnet_pkg.eml_function.EMLFunction</code>; backward uses
        clamped gradients per config.
      </p>
    </figure>
  );
}
