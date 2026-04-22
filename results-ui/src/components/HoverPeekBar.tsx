import type { ReactNode } from "react";

export type PeekLine = { label: string; value: ReactNode };

type Props = {
  title: string;
  lines: PeekLine[];
};

export function HoverPeekBar({ title, lines }: Props) {
  return (
    <div
      className="pointer-events-none fixed bottom-0 left-0 right-0 z-30 border-t border-zinc-200/90 bg-white/95 px-4 py-3 shadow-[0_-8px_30px_rgba(0,0,0,0.06)] backdrop-blur-md transition-all duration-200 ease-out lg:left-[17.5rem] xl:left-72"
      role="status"
      aria-live="polite"
    >
      <div className="mx-auto flex max-w-6xl flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
        <p className="shrink-0 text-xs font-semibold text-zinc-800">{title}</p>
        <dl className="flex min-w-0 flex-wrap gap-x-5 gap-y-1.5 text-[11px] text-zinc-600 sm:justify-end">
          {lines.map(({ label, value }) => (
            <div key={label} className="flex items-baseline gap-1.5 whitespace-nowrap">
              <dt className="text-zinc-400">{label}</dt>
              <dd className="font-mono font-medium tabular-nums text-zinc-800">{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
