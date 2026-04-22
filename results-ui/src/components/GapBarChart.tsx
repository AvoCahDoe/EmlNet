import type { ChartOptions, TooltipItem } from "chart.js";
import { Bar } from "react-chartjs-2";

type Props = {
  labels: string[];
  values: number[];
  horizontal?: boolean;
};

const grid = "rgba(24, 24, 27, 0.06)";
const font = { size: 11, family: "system-ui, Segoe UI, sans-serif" };

export function GapBarChart({ labels, values, horizontal }: Props) {
  const colors = values.map((v) =>
    v > 0.0005
      ? "rgba(16, 185, 129, 0.75)"
      : v < -0.0005
        ? "rgba(244, 63, 94, 0.7)"
        : "rgba(113, 113, 122, 0.55)",
  );

  const data = {
    labels,
    datasets: [
      {
        label: "Gap (MLP − EML)",
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        borderWidth: 0,
      },
    ],
  };

  const options: ChartOptions<"bar"> = {
    indexAxis: horizontal ? "y" : "x",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (tooltipItem: TooltipItem<"bar">) => {
            const x = tooltipItem.parsed.x;
            const y = tooltipItem.parsed.y;
            const v = typeof x === "number" ? x : typeof y === "number" ? y : 0;
            return `Gap: ${Number(v).toFixed(4)}`;
          },
        },
      },
    },
    scales: horizontal
      ? {
          x: {
            title: { display: true, text: "MLP val − EML val", font, color: "#71717a" },
            grid: { color: grid },
            ticks: { font, color: "#71717a" },
          },
          y: {
            grid: { display: false },
            ticks: { font, color: "#52525b", autoSkip: false },
          },
        }
      : {
          x: {
            grid: { display: false },
            ticks: { font, color: "#52525b", maxRotation: 40 },
          },
          y: {
            title: { display: true, text: "Gap", font, color: "#71717a" },
            grid: { color: grid },
            ticks: { font, color: "#71717a" },
          },
        },
  };

  return (
    <div className="h-56 w-full min-h-0 sm:h-72">
      <Bar data={data} options={options} />
    </div>
  );
}
