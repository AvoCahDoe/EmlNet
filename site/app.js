(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const ASSET_BASE = "sweep";

  let sweepGapChart = null;
  let legacyGapChart = null;
  let detailLossChart = null;

  function sweepFigureUrl(figPath) {
    return ASSET_BASE + "/" + String(figPath).replace(/^\/+/, "");
  }

  function fmt4(x) {
    if (typeof x !== "number" || Number.isNaN(x)) return "—";
    return x.toFixed(4);
  }

  function gapTextClass(g) {
    if (g > 0.0005) return "text-emerald-600 font-medium";
    if (g < -0.0005) return "text-rose-600 font-medium";
    return "text-zinc-700";
  }

  function destroyChart(ch) {
    if (ch) {
      ch.destroy();
    }
    return null;
  }

  const chartFont = { family: "system-ui, Segoe UI, sans-serif", size: 11 };
  const gridColor = "rgba(24, 24, 27, 0.06)";

  function buildGapBarChart(canvas, labels, values, horizontal) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const colors = values.map((v) =>
      v > 0.0005 ? "rgba(16, 185, 129, 0.75)" : v < -0.0005 ? "rgba(244, 63, 94, 0.7)" : "rgba(113, 113, 122, 0.55)"
    );
    return new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Gap (MLP − EML)",
            data: values,
            backgroundColor: colors,
            borderWidth: 0,
            borderRadius: 4,
          },
        ],
      },
      options: {
        indexAxis: horizontal ? "y" : "x",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => "Gap: " + (typeof c.parsed.x === "number" ? c.parsed.x : c.parsed.y).toFixed(4),
            },
          },
        },
        scales: horizontal
          ? {
              x: {
                title: { display: true, text: "MLP val − EML val", font: chartFont, color: "#71717a" },
                grid: { color: gridColor },
                ticks: { font: chartFont, color: "#71717a" },
              },
              y: {
                grid: { display: false },
                ticks: { font: chartFont, color: "#52525b", autoSkip: false },
              },
            }
          : {
              x: {
                grid: { display: false },
                ticks: { font: chartFont, color: "#52525b", maxRotation: 48, minRotation: 32 },
              },
              y: {
                title: { display: true, text: "Gap", font: chartFont, color: "#71717a" },
                grid: { color: gridColor },
                ticks: { font: chartFont, color: "#71717a" },
              },
            },
      },
    });
  }

  /**
   * Build { labels, values, caption } from filtered sweep runs.
   * @param {object[]} runs
   * @param {HTMLSelectElement} selScenario
   * @param {HTMLSelectElement} selDepth
   * @param {HTMLSelectElement} selHidden
   */
  function sweepChartPayload(runs, selScenario, selDepth, selHidden) {
    const sc = selScenario.value;
    const d = selDepth.value;
    const h = selHidden.value;

    if (d && h) {
      const bySc = new Map();
      runs.forEach((r) => {
        if (!bySc.has(r.scenario)) bySc.set(r.scenario, r);
      });
      const labels = [...bySc.keys()].sort();
      const values = labels.map((s) => bySc.get(s).val_loss_gap_mlp_minus_eml);
      return {
        labels,
        values,
        caption: "One bar per scenario at depth " + d + ", hidden " + h + ".",
        horizontal: labels.length > 12,
      };
    }

    const map = new Map();
    runs.forEach((r) => {
      if (!map.has(r.scenario)) map.set(r.scenario, []);
      map.get(r.scenario).push(r.val_loss_gap_mlp_minus_eml);
    });
    const labels = [...map.keys()].sort();
    const values = labels.map((s) => {
      const arr = map.get(s);
      return arr.reduce((a, b) => a + b, 0) / arr.length;
    });
    let cap = "Mean gap over visible runs per scenario.";
    if (sc) cap = "Scenario " + sc + ": mean gap over visible depth × hidden cells.";
    return { labels, values, caption: cap, horizontal: labels.length > 10 };
  }

  function renderSweepGapChart(manifest, selScenario, selDepth, selHidden) {
    const canvas = /** @type {HTMLCanvasElement} */ ($("#sw-gap-chart"));
    const capEl = $("#sw-chart-caption");
    if (!canvas || !capEl) return;

    const runs = manifest.runs.filter((r) => {
      if (selScenario.value && r.scenario !== selScenario.value) return false;
      if (selDepth.value && String(r.depth) !== selDepth.value) return false;
      if (selHidden.value && String(r.hidden) !== selHidden.value) return false;
      return true;
    });

    sweepGapChart = destroyChart(sweepGapChart);
    if (runs.length === 0) {
      capEl.textContent = "No runs match the filters.";
      return;
    }

    const { labels, values, caption, horizontal } = sweepChartPayload(runs, selScenario, selDepth, selHidden);
    capEl.textContent = caption + " (" + runs.length + " runs).";
    sweepGapChart = buildGapBarChart(canvas, labels, values, horizontal);
  }

  function renderLegacyGapChart(rows) {
    const canvas = /** @type {HTMLCanvasElement} */ ($("#legacy-gap-chart"));
    if (!canvas || !rows.length) {
      legacyGapChart = destroyChart(legacyGapChart);
      return;
    }
    const sorted = [...rows].sort((a, b) => a.val_loss_gap_mlp_minus_eml - b.val_loss_gap_mlp_minus_eml);
    const labels = sorted.map((r) => r.scenario);
    const values = sorted.map((r) => r.val_loss_gap_mlp_minus_eml);
    legacyGapChart = destroyChart(legacyGapChart);
    legacyGapChart = buildGapBarChart(canvas, labels, values, true);
  }

  function setNavMode(sweep) {
    const ns = $("#nav-sweep-link");
    const nl = $("#nav-legacy-link");
    if (ns) ns.classList.toggle("hidden", !sweep);
    if (nl) nl.classList.toggle("hidden", sweep);
  }

  function showLegacy() {
    const leg = $("#legacy-summary-wrap");
    if (leg) leg.classList.remove("hidden");
    $("#sweep-main").classList.add("hidden");
    $("#mode-banner").textContent = "";
    setNavMode(false);
  }

  function showSweep(manifest) {
    const leg = $("#legacy-summary-wrap");
    if (leg) leg.classList.add("hidden");
    $("#sweep-main").classList.remove("hidden");
    setNavMode(true);

    $("#mode-banner").textContent =
      "Sweep · depths [" +
      manifest.depths.join(", ") +
      "] · hiddens [" +
      manifest.hiddens.join(", ") +
      "] · " +
      manifest.runs.length +
      " runs";

    const presetEl = $("#sw-preset-summary");
    if (presetEl) {
      const pr = manifest.preset != null ? String(manifest.preset) : "";
      const ep = manifest.epochs_default != null ? String(manifest.epochs_default) : "";
      presetEl.classList.remove("hidden");
      presetEl.textContent =
        manifest.depths.length +
        "×" +
        manifest.hiddens.length +
        " grid · [" +
        manifest.depths.join(", ") +
        "] × [" +
        manifest.hiddens.join(", ") +
        "]" +
        (ep ? " · " + ep + " epochs/run" : "") +
        (pr ? " · preset " + pr : "");
    }

    const selScenario = /** @type {HTMLSelectElement} */ ($("#sw-scenario"));
    const selDepth = /** @type {HTMLSelectElement} */ ($("#sw-depth"));
    const selHidden = /** @type {HTMLSelectElement} */ ($("#sw-hidden"));
    const tbody = $("#sw-table tbody");
    const detail = $("#sw-detail");
    const heatmaps = $("#sw-heatmaps");

    const scenarios = [...new Set(manifest.runs.map((r) => r.scenario))].sort();

    function fillSelect(sel, values, allLabel) {
      sel.innerHTML = "";
      const o0 = document.createElement("option");
      o0.value = "";
      o0.textContent = allLabel;
      sel.appendChild(o0);
      values.forEach((v) => {
        const o = document.createElement("option");
        o.value = String(v);
        o.textContent = String(v);
        sel.appendChild(o);
      });
    }

    fillSelect(selScenario, scenarios, "All");
    fillSelect(selDepth, manifest.depths, "All");
    fillSelect(selHidden, manifest.hiddens, "All");

    function renderHeatmaps() {
      heatmaps.innerHTML = "";
      const sc = selScenario.value;
      const list = sc ? manifest.heatmaps.filter((x) => x.scenario === sc) : manifest.heatmaps;
      list.forEach((hm) => {
        const wrap = document.createElement("div");
        wrap.className = "rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm";
        wrap.innerHTML =
          '<div class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
          hm.scenario +
          '</div><img src="' +
          sweepFigureUrl(hm.path) +
          '" alt="" class="w-full block bg-zinc-50" loading="lazy" />';
        heatmaps.appendChild(wrap);
      });
    }

    function renderTable() {
      const rows = filteredRuns().sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml);
      tbody.innerHTML = "";
      rows.forEach((r) => {
        const tr = document.createElement("tr");
        tr.className = "hover:bg-zinc-50/80 transition-colors";
        tr.dataset.rel = r.rel_dir;
        const g = r.val_loss_gap_mlp_minus_eml;
        tr.innerHTML =
          '<td class="px-3 py-2 font-medium text-zinc-800">' +
          r.scenario +
          "</td>" +
          '<td class="px-3 py-2 text-right tabular-nums text-zinc-600">' +
          r.depth +
          "</td>" +
          '<td class="px-3 py-2 text-right tabular-nums text-zinc-600">' +
          r.hidden +
          "</td>" +
          '<td class="px-3 py-2 text-right tabular-nums text-zinc-500">' +
          r.epochs +
          "</td>" +
          '<td class="px-3 py-2 text-right tabular-nums text-zinc-600">' +
          fmt4(r.eml_final_val_loss) +
          "</td>" +
          '<td class="px-3 py-2 text-right tabular-nums text-zinc-600">' +
          fmt4(r.mlp_final_val_loss) +
          "</td>" +
          '<td class="px-3 py-2 text-right tabular-nums ' +
          gapTextClass(g) +
          '">' +
          fmt4(g) +
          "</td>" +
          '<td class="px-3 py-2 text-right tabular-nums text-zinc-500 text-xs">' +
          fmt4(r.eml_final_val_acc) +
          " / " +
          fmt4(r.mlp_final_val_acc) +
          "</td>" +
          '<td class="px-3 py-2"><button type="button" class="text-xs text-zinc-600 hover:text-zinc-900 underline underline-offset-2">Open</button></td>';
        tr.querySelector("button").addEventListener("click", () => renderDetail(r));
        tbody.appendChild(tr);
      });
    }

    function filteredRuns() {
      const sc = selScenario.value;
      const d = selDepth.value;
      const h = selHidden.value;
      return manifest.runs.filter((r) => {
        if (sc && r.scenario !== sc) return false;
        if (d && String(r.depth) !== d) return false;
        if (h && String(r.hidden) !== h) return false;
        return true;
      });
    }

    const FIG_ORDER = [
      ["loss_linear", "Loss (linear)"],
      ["loss_logy", "Loss (log y)"],
      ["val_loss_gap", "Val gap"],
      ["val_delta_eml_minus_mlp", "Val delta"],
      ["accuracy", "Accuracy"],
      ["grad_norm", "Grad norm"],
      ["decision", "Decision"],
      ["summary_grid", "Summary"],
      ["final_bars", "Final bars"],
    ];

    async function renderDetail(r) {
      $$("#sw-table tbody tr").forEach((tr) => tr.classList.remove("bg-zinc-100")));
      const hit = $$("#sw-table tbody tr").find((tr) => tr.dataset.rel === r.rel_dir);
      if (hit) hit.classList.add("bg-zinc-100");

      detail.classList.remove("hidden");
      $("#sw-detail-title").textContent = r.scenario + " · d" + r.depth + " h" + r.hidden;
      const link = /** @type {HTMLAnchorElement} */ ($("#sw-detail-histories-link"));
      link.href = sweepFigureUrl(r.figures.histories);

      const grid = $("#sw-fig-grid");
      grid.innerHTML = "";
      FIG_ORDER.forEach(([key, label]) => {
        const url = sweepFigureUrl(r.figures[key]);
        const fig = document.createElement("figure");
        fig.className = "rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm";
        fig.innerHTML =
          '<figcaption class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
          label +
          '</figcaption><img src="' +
          url +
          '" alt="" class="w-full block bg-zinc-50" loading="lazy" />';
        grid.appendChild(fig);
      });

      const lossCanvas = /** @type {HTMLCanvasElement} */ ($("#sw-detail-loss-chart"));
      detailLossChart = destroyChart(detailLossChart);
      try {
        const res = await fetch(sweepFigureUrl(r.figures.histories));
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        const eml = data.eml && data.eml.val_loss ? data.eml.val_loss : [];
        const mlp = data.mlp && data.mlp.val_loss ? data.mlp.val_loss : [];
        const len = Math.min(eml.length, mlp.length) || Math.max(eml.length, mlp.length);
        const labels = Array.from({ length: len }, (_, i) => String(i + 1));
        const ctx = lossCanvas.getContext("2d");
        if (ctx && len > 0) {
          detailLossChart = new Chart(ctx, {
            type: "line",
            data: {
              labels,
              datasets: [
                {
                  label: "EML val",
                  data: eml.slice(0, len),
                  borderColor: "rgb(59, 130, 246)",
                  backgroundColor: "transparent",
                  tension: 0.15,
                  pointRadius: 0,
                  borderWidth: 1.5,
                },
                {
                  label: "MLP val",
                  data: mlp.slice(0, len),
                  borderColor: "rgb(244, 114, 182)",
                  backgroundColor: "transparent",
                  tension: 0.15,
                  pointRadius: 0,
                  borderWidth: 1.5,
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              interaction: { mode: "index", intersect: false },
              plugins: {
                legend: { position: "bottom", labels: { font: chartFont, boxWidth: 12 } },
              },
              scales: {
                x: {
                  title: { display: true, text: "Epoch", font: chartFont, color: "#71717a" },
                  grid: { color: gridColor },
                  ticks: { maxTicksLimit: 8, font: chartFont, color: "#71717a" },
                },
                y: {
                  title: { display: true, text: "Val BCE", font: chartFont, color: "#71717a" },
                  grid: { color: gridColor },
                  ticks: { font: chartFont, color: "#71717a" },
                },
              },
            },
          });
        }
      } catch {
        /* leave chart empty */
      }

      detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function onFilterChange() {
      renderTable();
      renderHeatmaps();
      renderSweepGapChart(manifest, selScenario, selDepth, selHidden);
    }

    [selScenario, selDepth, selHidden].forEach((el) => el.addEventListener("change", onFilterChange));

    onFilterChange();
    detail.classList.add("hidden");
  }

  function renderLegacy(rows) {
    const tbody = $("#results-table tbody");
    const status = $("#load-status");
    const gallery = $("#gallery-grid");

    rows.sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml);
    tbody.innerHTML = "";
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-zinc-50/80";
      const g = r.val_loss_gap_mlp_minus_eml;
      tr.innerHTML =
        '<td class="px-3 py-2 font-medium text-zinc-800">' +
        r.scenario +
        "</td>" +
        '<td class="px-3 py-2 text-right tabular-nums text-zinc-500">' +
        r.epochs +
        "</td>" +
        '<td class="px-3 py-2 text-right tabular-nums">' +
        r.depth +
        "</td>" +
        '<td class="px-3 py-2 text-right tabular-nums">' +
        r.hidden +
        "</td>" +
        '<td class="px-3 py-2 text-right tabular-nums">' +
        fmt4(r.eml_final_val_loss) +
        "</td>" +
        '<td class="px-3 py-2 text-right tabular-nums">' +
        fmt4(r.mlp_final_val_loss) +
        "</td>" +
        '<td class="px-3 py-2 text-right tabular-nums ' +
        gapTextClass(g) +
        '">' +
        fmt4(g) +
        "</td>" +
        '<td class="px-3 py-2 text-right tabular-nums text-xs text-zinc-500">' +
        fmt4(r.eml_final_val_acc) +
        " / " +
        fmt4(r.mlp_final_val_acc) +
        "</td>";
      tbody.appendChild(tr);
    });

    renderLegacyGapChart(rows);

    gallery.innerHTML = "";
    rows.forEach((r) => {
      const name = r.scenario;
      const block = document.createElement("div");
      block.className = "space-y-3";
      block.innerHTML =
        '<figure class="rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm"><figcaption class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
        name +
        ' — decision</figcaption><img src="assets/' +
        name +
        '_decision.png" alt="" class="w-full block bg-zinc-50" loading="lazy" /></figure>' +
        '<figure class="rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm"><figcaption class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
        name +
        ' — loss</figcaption><img src="assets/' +
        name +
        '_loss.png" alt="" class="w-full block bg-zinc-50" loading="lazy" /></figure>';
      gallery.appendChild(block);
    });

    status.textContent =
      rows.length > 0
        ? "Loaded " +
          rows.length +
          " rows · EML " +
          rows[0].eml_params +
          " vs MLP " +
          rows[0].mlp_params +
          " params."
        : "No rows loaded.";
  }

  function boot() {
    fetch("data/sweep_manifest.json")
      .then((r) => {
        if (!r.ok) throw new Error("no sweep");
        return r.json();
      })
      .then((manifest) => {
        showSweep(manifest);
      })
      .catch(() => {
        showLegacy();
        const status = $("#load-status");
        return fetch("data/results.json")
          .then((r) => {
            if (!r.ok) throw new Error(r.statusText);
            return r.json();
          })
          .then((rows) => renderLegacy(rows))
          .catch((e) => {
            status.textContent = "Could not load data (use a local server). " + e;
          });
      });
  }

  boot();
})();
