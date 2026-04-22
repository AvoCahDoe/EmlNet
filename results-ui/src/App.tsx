import { useBenchmarkData } from "./hooks/useBenchmarkData";
import { LegacyView } from "./components/LegacyView";
import { SweepView } from "./components/SweepView";

function App() {
  const bench = useBenchmarkData();

  if (bench.status === "loading") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-[#f6f6f7] antialiased">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-200 border-t-zinc-700" aria-hidden />
        <p className="text-sm text-zinc-500">Loading results…</p>
      </div>
    );
  }

  if (bench.status === "error") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f6f6f7] px-4 antialiased">
        <div className="max-w-md rounded-2xl border border-zinc-200/80 bg-white p-8 text-center shadow-sm">
          <h1 className="text-base font-semibold text-zinc-900">Could not load data</h1>
          <p className="mt-2 text-sm leading-relaxed text-zinc-600">{bench.message}</p>
          <p className="mt-5 text-xs leading-relaxed text-zinc-500">
            From <code className="rounded-md bg-zinc-100 px-1.5 py-0.5">results-ui/</code>, run{" "}
            <code className="rounded-md bg-zinc-100 px-1.5 py-0.5">npm run sync-assets</code> then{" "}
            <code className="rounded-md bg-zinc-100 px-1.5 py-0.5">npm run dev</code>.
          </p>
        </div>
      </div>
    );
  }

  if (bench.mode === "sweep") {
    return <SweepView manifest={bench.manifest} />;
  }

  return <LegacyView rows={bench.rows} />;
}

export default App;
