export function fmt4(x: number): string {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return x.toFixed(4);
}

export function sweepFigureUrl(figPath: string): string {
  return `/sweep/${figPath.replace(/^\/+/, "")}`;
}

export const SAMPLE_SCENARIOS = [
  "moons_hard",
  "spiral",
  "checkerboard",
  "xor_blobs",
  "sine_boundary",
  "circles",
] as const;

export function mid<T>(arr: T[]): T {
  return arr[Math.floor(arr.length / 2)]!;
}

export function gapTextClass(g: number): string {
  if (g > 0.0005) return "text-emerald-600 font-medium";
  if (g < -0.0005) return "text-rose-600 font-medium";
  return "text-zinc-700";
}

const base = import.meta.env.BASE_URL || "/";

export function assetUrl(rel: string): string {
  const b = base.endsWith("/") ? base : base + "/";
  return `${b}${rel.replace(/^\/+/, "")}`;
}
