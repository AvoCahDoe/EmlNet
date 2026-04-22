import { useEffect, useState } from "react";
import type { LegacyRow, SweepManifest } from "../types";

export type BenchState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; mode: "sweep"; manifest: SweepManifest }
  | { status: "ok"; mode: "legacy"; rows: LegacyRow[] };

const base = import.meta.env.BASE_URL || "/";

function dataUrl(name: string): string {
  const b = base.endsWith("/") ? base : base + "/";
  return `${b}data/${name}`;
}

export function useBenchmarkData(): BenchState {
  const [state, setState] = useState<BenchState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        let r = await fetch(dataUrl("sweep_manifest.json"));
        if (r.ok) {
          const manifest = (await r.json()) as SweepManifest;
          if (!cancelled) setState({ status: "ok", mode: "sweep", manifest });
          return;
        }
        r = await fetch(dataUrl("results.json"));
        if (!r.ok) throw new Error("Missing public/data/sweep_manifest.json or results.json (run npm run sync-assets).");
        const rows = (await r.json()) as LegacyRow[];
        if (!cancelled) setState({ status: "ok", mode: "legacy", rows });
      } catch (e) {
        if (!cancelled)
          setState({
            status: "error",
            message: e instanceof Error ? e.message : String(e),
          });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
