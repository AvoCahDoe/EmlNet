import { useBenchmarkData } from "./hooks/useBenchmarkData";
import { LegacyView } from "./components/LegacyView";
import { SweepView } from "./components/SweepView";

function App() {
  const bench = useBenchmarkData();

  if (bench.status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-zinc-50 text-sm text-zinc-500">
        Loading benchmark data…
      </div>
    );
  }

  if (bench.status === "error") {
    return (
      <div className="mx-auto max-w-lg px-4 py-16 text-center">
        <h1 className="text-lg font-semibold text-zinc-900">Could not load data</h1>
        <p className="mt-2 text-sm text-zinc-600">{bench.message}</p>
        <p className="mt-4 text-xs text-zinc-500">
          From <code className="rounded bg-zinc-100 px-1">results-ui/</code>, run{" "}
          <code className="rounded bg-zinc-100 px-1">npm run sync-assets</code> then{" "}
          <code className="rounded bg-zinc-100 px-1">npm run dev</code>.
        </p>
      </div>
    );
  }

  if (bench.mode === "sweep") {
    return (
      <div className="min-h-screen bg-zinc-50 text-zinc-900">
        <SweepView manifest={bench.manifest} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900">
      <LegacyView rows={bench.rows} />
    </div>
  );
}

export default App;
