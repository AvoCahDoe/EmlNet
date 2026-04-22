import type { ReactNode } from "react";

type Props = {
  badge: string;
  title: string;
  sidebar: ReactNode;
  children: ReactNode;
};

export function ResultsShell({ badge, title, sidebar, children }: Props) {
  return (
    <div className="flex min-h-screen flex-col bg-[#f6f6f7] lg:flex-row">
      <aside className="max-h-[min(46vh,320px)] shrink-0 overflow-y-auto border-b border-zinc-200/90 bg-white/95 shadow-sm lg:max-h-none lg:h-screen lg:w-[17.5rem] lg:border-b-0 lg:border-r xl:w-72 lg:sticky lg:top-0">
        <div className="border-b border-zinc-100 px-5 py-4 lg:shrink-0 lg:py-5">
          <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-zinc-400">{badge}</p>
          <h1 className="mt-1.5 text-lg font-semibold tracking-tight text-zinc-900">{title}</h1>
        </div>
        <div className="flex flex-col gap-6 px-5 py-5 lg:flex-1 lg:overflow-y-auto lg:pb-8">{sidebar}</div>
      </aside>

      <main className="min-h-0 min-w-0 flex-1">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-10">{children}</div>
      </main>
    </div>
  );
}
