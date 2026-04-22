(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const ASSET_BASE = "sweep";
  const SAMPLE_SCENARIOS = ["moons_hard", "spiral", "checkerboard", "xor_blobs", "sine_boundary", "circles"];

  let sweepGapChart = null;
  let legacyGapChart = null;
  let detailLossChart = null;
  /** @type {object | null} */
  let detailRunForMore = null;

  let chartJsPromise = null;
  function loadChartJs() {
    if (window.Chart) return Promise.resolve();
    if (chartJsPromise) return chartJsPromise;
    chartJsPromise = new Promise(function (resolve, reject) {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js";
      s.async = true;
      s.onload = function () {
        resolve();
      };
      s.onerror = reject;
      document.head.appendChild(s);
    });
    return chartJsPromise;
  }

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
    if (ch) ch.destroy();
    return null;
  }

  const chartFont = { family: "system-ui, Segoe UI, sans-serif", size: 11 };
  const gridColor = "rgba(24, 24, 27, 0.06)";

  function buildGapBarChart(canvas, labels, values, horizontal) {
    const C = window.Chart;
    const ctx = canvas.getContext("2d");
    if (!ctx || !C) return null;
    const colors = values.map(function (v) {
      return v > 0.0005
        ? "rgba(16, 185, 129, 0.75)"
        : v < -0.0005
          ? "rgba(244, 63, 94, 0.7)"
          : "rgba(113, 113, 122, 0.55)";
    });
    return new C(ctx, {
      type: "bar",
      data: {
        labels: labels,
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
              label: function (c) {
                const v = typeof c.parsed.x === "number" ? c.parsed.x : c.parsed.y;
                return "Gap: " + Number(v).toFixed(4);
              },
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

  function sweepChartPayload(runs, selScenario, selDepth, selHidden) {
    const sc = selScenario.value;
    const d = selDepth.value;
    const h = selHidden.value;

    if (d && h) {
      const bySc = new Map();
      runs.forEach(function (r) {
        if (!bySc.has(r.scenario)) bySc.set(r.scenario, r);
      });
      const labels = Array.from(bySc.keys()).sort();
      const values = labels.map(function (s) {
        return bySc.get(s).val_loss_gap_mlp_minus_eml;
      });
      return {
        labels: labels,
        values: values,
        caption: "Per scenario at depth " + d + ", hidden " + h + ".",
        horizontal: labels.length > 12,
      };
    }

    const map = new Map();
    runs.forEach(function (r) {
      if (!map.has(r.scenario)) map.set(r.scenario, []);
      map.get(r.scenario).push(r.val_loss_gap_mlp_minus_eml);
    });
    const labels = Array.from(map.keys()).sort();
    const values = labels.map(function (s) {
      const arr = map.get(s);
      return arr.reduce(function (a, b) {
        return a + b;
      }, 0) / arr.length;
    });
    let cap = "Mean gap over visible runs per scenario.";
    if (sc) cap = "Scenario " + sc + ": mean gap over visible depth × hidden cells.";
    return { labels: labels, values: values, caption: cap, horizontal: labels.length > 10 };
  }

  function renderSweepGapChart(manifest, selScenario, selDepth, selHidden) {
    const canvas = /** @type {HTMLCanvasElement} */ ($("#sw-gap-chart"));
    const capEl = $("#sw-chart-caption");
    if (!canvas || !capEl) return;

    const runs = manifest.runs.filter(function (r) {
      if (selScenario.value && r.scenario !== selScenario.value) return false;
      if (selDepth.value && String(r.depth) !== selDepth.value) return false;
      if (selHidden.value && String(r.hidden) !== selHidden.value) return false;
      return true;
    });

    sweepGapChart = destroyChart(sweepGapChart);
    if (runs.length === 0) {
      capEl.textContent = "No runs for this filter.";
      return;
    }

    const payload = sweepChartPayload(runs, selScenario, selDepth, selHidden);
    capEl.textContent = payload.caption + " (" + runs.length + " runs).";
    sweepGapChart = buildGapBarChart(canvas, payload.labels, payload.values, payload.horizontal);
  }

  function renderLegacyGapChart(rows) {
    const canvas = /** @type {HTMLCanvasElement} */ ($("#legacy-gap-chart"));
    if (!canvas || !rows.length) {
      legacyGapChart = destroyChart(legacyGapChart);
      return;
    }
    const sorted = rows.slice().sort(function (a, b) {
      return a.val_loss_gap_mlp_minus_eml - b.val_loss_gap_mlp_minus_eml;
    });
    const labels = sorted.map(function (r) {
      return r.scenario;
    });
    const values = sorted.map(function (r) {
      return r.val_loss_gap_mlp_minus_eml;
    });
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

  function populateSweepGallery(manifest) {
    const g = $("#gallery-grid");
    const blurb = $("#gallery-blurb");
    if (!g) return;
    if (blurb) blurb.textContent = "Decision-map thumbnails from the sweep (lazy). Click a card to jump to Sweep and pre-fill that scenario.";
    const scenarios = Array.from(new Set(manifest.runs.map(function (r) {
      return r.scenario;
    }))).sort();
    const picks = SAMPLE_SCENARIOS.filter(function (s) {
      return scenarios.indexOf(s) >= 0;
    }).slice(0, 6);
    g.innerHTML = "";
    picks.forEach(function (sc) {
      const run = manifest.runs.find(function (r) {
        return r.scenario === sc;
      });
      if (!run || !run.figures || !run.figures.decision) return;
      const card = document.createElement("button");
      card.type = "button";
      card.className =
        "text-left rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm hover:border-zinc-300 transition-colors";
      card.innerHTML =
        '<div class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
        sc +
        '</div><div class="aspect-[4/3] bg-zinc-100"><img src="' +
        sweepFigureUrl(run.figures.decision) +
        '" alt="" class="w-full h-full object-cover" loading="lazy" decoding="async" /></div>';
      card.addEventListener("click", function () {
        window.location.hash = "#sweep";
        document.dispatchEvent(new CustomEvent("emlnet:pick-scenario", { detail: { scenario: sc } }));
      });
      g.appendChild(card);
    });
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
        " · [" +
        manifest.depths.join(", ") +
        "] × [" +
        manifest.hiddens.join(", ") +
        "]" +
        (ep ? " · " + ep + " ep/run" : "") +
        (pr ? " · " + pr : "");
    }

    populateSweepGallery(manifest);

    const selScenario = /** @type {HTMLSelectElement} */ ($("#sw-scenario"));
    const selDepth = /** @type {HTMLSelectElement} */ ($("#sw-depth"));
    const selHidden = /** @type {HTMLSelectElement} */ ($("#sw-hidden"));
    const tbody = $("#sw-table tbody");
    const detail = $("#sw-detail");
    const heatmaps = $("#sw-heatmaps");
    const resultsPanel = $("#sw-results-panel");
    const loadBtn = $("#sw-load-btn");
    const loadHint = $("#sw-load-hint");
    const allHeatmapsCb = /** @type {HTMLInputElement} */ ($("#sw-all-heatmaps"));
    const samplesEl = $("#sw-samples");

    const scenarios = Array.from(new Set(manifest.runs.map(function (r) {
      return r.scenario;
    }))).sort();

    function fillSelectFixed(sel, values, allLabel) {
      sel.innerHTML = "";
      const o0 = document.createElement("option");
      o0.value = "";
      o0.textContent = allLabel;
      sel.appendChild(o0);
      values.forEach(function (v) {
        const o = document.createElement("option");
        o.value = String(v);
        o.textContent = String(v);
        sel.appendChild(o);
      });
    }

    fillSelectFixed(selScenario, scenarios, "Choose…");
    fillSelectFixed(selDepth, manifest.depths, "All depths");
    fillSelectFixed(selHidden, manifest.hiddens, "All hiddens");

    function midPick(arr) {
      return arr[Math.floor(arr.length / 2)];
    }

    function buildSamples() {
      if (!samplesEl) return;
      samplesEl.innerHTML = "";
      const di = midPick(manifest.depths);
      const hi = midPick(manifest.hiddens);
      SAMPLE_SCENARIOS.forEach(function (sc) {
        if (scenarios.indexOf(sc) < 0) return;
        const b = document.createElement("button");
        b.type = "button";
        b.className =
          "rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50 hover:border-zinc-300";
        b.textContent = sc;
        b.addEventListener("click", function () {
          selScenario.value = sc;
          selDepth.value = String(di);
          selHidden.value = String(hi);
          doLoadResults();
        });
        samplesEl.appendChild(b);
      });
    }
    buildSamples();

    document.addEventListener("emlnet:pick-scenario", function (ev) {
      const d = /** @type {CustomEvent} */ (ev).detail;
      if (d && d.scenario) {
        selScenario.value = d.scenario;
        selDepth.value = "";
        selHidden.value = "";
        doLoadResults();
      }
    });

    function filteredRuns() {
      const sc = selScenario.value;
      const d = selDepth.value;
      const h = selHidden.value;
      return manifest.runs.filter(function (r) {
        if (sc && r.scenario !== sc) return false;
        if (d && String(r.depth) !== d) return false;
        if (h && String(r.hidden) !== h) return false;
        return true;
      });
    }

    function renderHeatmaps() {
      heatmaps.innerHTML = "";
      const allHm = allHeatmapsCb && allHeatmapsCb.checked;
      const sc = selScenario.value;
      if (!allHm && !sc) {
        heatmaps.innerHTML =
          '<p class="text-sm text-zinc-500">Pick a scenario for one heatmap, or enable all-scenarios heatmaps before loading.</p>';
        return;
      }
      const list = allHm ? manifest.heatmaps : manifest.heatmaps.filter(function (x) {
        return x.scenario === sc;
      });
      list.forEach(function (hm) {
        const wrap = document.createElement("div");
        wrap.className = "rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm";
        wrap.innerHTML =
          '<div class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
          hm.scenario +
          '</div><img src="' +
          sweepFigureUrl(hm.path) +
          '" alt="" class="w-full block bg-zinc-50" loading="lazy" decoding="async" />';
        heatmaps.appendChild(wrap);
      });
    }

    function renderTable() {
      const rows = filteredRuns().slice().sort(function (a, b) {
        return b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml;
      });
      tbody.innerHTML = "";
      if (rows.length === 0) {
        tbody.innerHTML =
          '<tr><td colspan="9" class="px-3 py-8 text-center text-sm text-zinc-400">No runs match these filters.</td></tr>';
        return;
      }
      rows.forEach(function (r) {
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
        tr.querySelector("button").addEventListener("click", function () {
          renderDetail(r);
        });
        tbody.appendChild(tr);
      });
    }

    const FIG_COMPACT = [
      ["decision", "Decision"],
      ["loss_linear", "Loss (linear)"],
      ["val_loss_gap", "Val gap"],
      ["summary_grid", "Summary"],
    ];
    const FIG_REST = [
      ["loss_logy", "Loss (log y)"],
      ["val_delta_eml_minus_mlp", "Val delta"],
      ["accuracy", "Accuracy"],
      ["grad_norm", "Grad norm"],
      ["final_bars", "Final bars"],
    ];

    function appendFigures(grid, r, keys) {
      keys.forEach(function (pair) {
        const key = pair[0];
        const label = pair[1];
        const url = sweepFigureUrl(r.figures[key]);
        const fig = document.createElement("figure");
        fig.className = "rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm";
        fig.innerHTML =
          '<figcaption class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
          label +
          '</figcaption><img src="' +
          url +
          '" alt="" class="w-full block bg-zinc-50" loading="lazy" decoding="async" />';
        grid.appendChild(fig);
      });
    }

    async function renderDetail(r) {
      await loadChartJs();
      $$("#sw-table tbody tr").forEach(function (tr) {
        tr.classList.remove("bg-zinc-100");
      });
      const hit = $$("#sw-table tbody tr").find(function (tr) {
        return tr.dataset.rel === r.rel_dir;
      });
      if (hit) hit.classList.add("bg-zinc-100");

      detail.classList.remove("hidden");
      detailRunForMore = r;
      $("#sw-detail-title").textContent = r.scenario + " · d" + r.depth + " h" + r.hidden;
      const link = /** @type {HTMLAnchorElement} */ ($("#sw-detail-histories-link"));
      link.href = sweepFigureUrl(r.figures.histories);

      const grid = $("#sw-fig-grid");
      grid.innerHTML = "";
      appendFigures(grid, r, FIG_COMPACT);
      const moreBtn = /** @type {HTMLButtonElement} */ ($("#sw-detail-more-figs"));
      if (moreBtn) {
        moreBtn.disabled = false;
        moreBtn.textContent = "Load all figure PNGs";
      }

      const lossCanvas = /** @type {HTMLCanvasElement} */ ($("#sw-detail-loss-chart"));
      detailLossChart = destroyChart(detailLossChart);
      try {
        const res = await fetch(sweepFigureUrl(r.figures.histories));
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        const eml = data.eml && data.eml.val_loss ? data.eml.val_loss : [];
        const mlp = data.mlp && data.mlp.val_loss ? data.mlp.val_loss : [];
        const len = Math.min(eml.length, mlp.length) || Math.max(eml.length, mlp.length);
        const labels = Array.from({ length: len }, function (_, i) {
          return String(i + 1);
        });
        const ctx = lossCanvas.getContext("2d");
        if (ctx && len > 0 && window.Chart) {
          detailLossChart = new window.Chart(ctx, {
            type: "line",
            data: {
              labels: labels,
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
      } catch (_) {
        /* empty */
      }

      detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    const moreBtn = /** @type {HTMLButtonElement} */ ($("#sw-detail-more-figs"));
    if (moreBtn) {
      moreBtn.onclick = function () {
        if (!detailRunForMore) return;
        const grid = $("#sw-fig-grid");
        appendFigures(grid, detailRunForMore, FIG_REST);
        moreBtn.disabled = true;
        moreBtn.textContent = "All figures loaded";
      };
    }

    async function doLoadResults() {
      const sc = selScenario.value;
      const allHm = allHeatmapsCb && allHeatmapsCb.checked;
      if (!sc && !allHm) {
        if (loadHint) loadHint.textContent = "Choose a scenario (or enable all heatmaps).";
        return;
      }
      if (loadHint) loadHint.textContent = "Loading…";
      if (loadBtn) loadBtn.disabled = true;
      try {
        await loadChartJs();
        if (resultsPanel) resultsPanel.classList.remove("hidden");
        renderTable();
        renderHeatmaps();
        renderSweepGapChart(manifest, selScenario, selDepth, selHidden);
        if (loadHint) {
          const nHm = allHm ? manifest.heatmaps.length : selScenario.value ? 1 : 0;
          loadHint.textContent = "Showing " + filteredRuns().length + " run(s)" + (nHm ? " · " + nHm + " heatmap(s)" : "") + ".";
        }
      } finally {
        if (loadBtn) loadBtn.disabled = false;
      }
    }

    if (loadBtn) loadBtn.addEventListener("click", function () {
      doLoadResults();
    });

    if (resultsPanel) resultsPanel.classList.add("hidden");
    detail.classList.add("hidden");
    tbody.innerHTML =
      '<tr><td colspan="9" class="px-3 py-10 text-center text-sm text-zinc-400">Use samples or set parameters, then <strong class="text-zinc-600">Load results</strong>.</td></tr>';
    heatmaps.innerHTML = "";
    if (loadHint) loadHint.textContent = "";
  }

  async function renderLegacy(rows) {
    await loadChartJs();
    const tbody = $("#results-table tbody");
    const status = $("#load-status");
    const gallery = $("#gallery-grid");
    const blurb = $("#gallery-blurb");

    rows.sort(function (a, b) {
      return b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml;
    });
    tbody.innerHTML = "";
    rows.forEach(function (r) {
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

    if (blurb) blurb.textContent = "Legacy flattened assets (lazy).";
    gallery.innerHTML = "";
    rows.forEach(function (r) {
      const name = r.scenario;
      const block = document.createElement("div");
      block.className = "space-y-3";
      block.innerHTML =
        '<figure class="rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm"><figcaption class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
        name +
        ' — decision</figcaption><img src="assets/' +
        name +
        '_decision.png" alt="" class="w-full block bg-zinc-50" loading="lazy" decoding="async" /></figure>' +
        '<figure class="rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm"><figcaption class="text-xs text-zinc-500 px-3 py-2 border-b border-zinc-100">' +
        name +
        ' — loss</figcaption><img src="assets/' +
        name +
        '_loss.png" alt="" class="w-full block bg-zinc-50" loading="lazy" decoding="async" /></figure>';
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
      .then(function (r) {
        if (!r.ok) throw new Error("no sweep");
        return r.json();
      })
      .then(function (manifest) {
        showSweep(manifest);
      })
      .catch(function () {
        showLegacy();
        const status = $("#load-status");
        return fetch("data/results.json")
          .then(function (r) {
            if (!r.ok) throw new Error(r.statusText);
            return r.json();
          })
          .then(function (rows) {
            return renderLegacy(rows);
          })
          .catch(function (e) {
            status.textContent = "Could not load data (use a local server). " + e;
          });
      });
  }

  boot();
})();
