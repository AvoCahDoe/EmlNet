import type { SweepRun } from "./types";

export type GapChartPayload = {
  labels: string[];
  values: number[];
  caption: string;
  horizontal: boolean;
};

/** Mirrors `site/app.js` `sweepChartPayload`. */
export function sweepChartPayload(
  runs: SweepRun[],
  selScenario: string,
  selDepth: string,
  selHidden: string,
): GapChartPayload {
  const sc = selScenario;
  const d = selDepth;
  const h = selHidden;

  if (d && h) {
    const bySc = new Map<string, SweepRun>();
    runs.forEach((r) => {
      if (!bySc.has(r.scenario)) bySc.set(r.scenario, r);
    });
    const labels = Array.from(bySc.keys()).sort();
    const values = labels.map((s) => bySc.get(s)!.val_loss_gap_mlp_minus_eml);
    return {
      labels,
      values,
      caption: `Per scenario at depth ${d}, hidden ${h}.`,
      horizontal: labels.length > 12,
    };
  }

  const map = new Map<string, number[]>();
  runs.forEach((r) => {
    if (!map.has(r.scenario)) map.set(r.scenario, []);
    map.get(r.scenario)!.push(r.val_loss_gap_mlp_minus_eml);
  });
  const labels = Array.from(map.keys()).sort();
  const values = labels.map((s) => {
    const arr = map.get(s)!;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  });
  let cap = "Mean gap over visible runs per scenario.";
  if (sc) cap = `Scenario ${sc}: mean gap over visible depth × hidden cells.`;
  return { labels, values, caption: cap, horizontal: labels.length > 10 };
}
