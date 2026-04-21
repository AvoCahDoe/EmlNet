(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function fmt4(x) {
    if (typeof x !== "number" || Number.isNaN(x)) return "";
    return x.toFixed(4);
  }

  function gapClass(g) {
    if (g > 0.0005) return "pos";
    if (g < -0.0005) return "neg";
    return "";
  }

  const ASSET_BASE = "sweep";

  function sweepFigureUrl(figPath) {
    return ASSET_BASE + "/" + figPath.replace(/^\/+/, "");
  }

  function showLegacy() {
    $("#legacy-main").hidden = false;
    $("#sweep-main").hidden = true;
    $("#mode-banner").textContent = "";
    const ns = $("#nav-sweep-link");
    const nl = $("#nav-legacy-link");
    if (ns) ns.hidden = true;
    if (nl) nl.hidden = false;
  }

  function showSweep(manifest) {
    $("#legacy-main").hidden = true;
    $("#sweep-main").hidden = false;
    const ns = $("#nav-sweep-link");
    const nl = $("#nav-legacy-link");
    if (ns) ns.hidden = false;
    if (nl) nl.hidden = true;
    $("#mode-banner").textContent =
      "Sweep mode: depths [" +
      manifest.depths.join(", ") +
      "], hiddens [" +
      manifest.hiddens.join(", ") +
      "], " +
      manifest.runs.length +
      " runs.";

    const selScenario = $("#sw-scenario");
    const selDepth = $("#sw-depth");
    const selHidden = $("#sw-hidden");
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

    fillSelect(selScenario, scenarios, "All scenarios");
    fillSelect(selDepth, manifest.depths, "All depths");
    fillSelect(selHidden, manifest.hiddens, "All hiddens");

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

    function renderHeatmaps() {
      heatmaps.innerHTML = "";
      const sc = selScenario.value;
      const list = sc
        ? manifest.heatmaps.filter((x) => x.scenario === sc)
        : manifest.heatmaps;
      list.forEach((hm) => {
        const card = document.createElement("div");
        card.className = "hm-card";
        card.innerHTML =
          '<h4>' +
          hm.scenario +
          '</h4><p class="note">MLP val BCE − EML val BCE (positive ⇒ EML lower loss)</p>' +
          '<img src="' +
          sweepFigureUrl(hm.path) +
          '" alt="heatmap" loading="lazy" />';
        heatmaps.appendChild(card);
      });
    }

    function renderTable() {
      const rows = filteredRuns().sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml);
      tbody.innerHTML = "";
      rows.forEach((r) => {
        const tr = document.createElement("tr");
        tr.dataset.rel = r.rel_dir;
        const g = r.val_loss_gap_mlp_minus_eml;
        tr.innerHTML =
          "<td><strong>" +
          r.scenario +
          "</strong></td>" +
          '<td class="num">' +
          r.depth +
          "</td>" +
          '<td class="num">' +
          r.hidden +
          "</td>" +
          '<td class="num">' +
          r.epochs +
          "</td>" +
          '<td class="num">' +
          fmt4(r.eml_final_val_loss) +
          "</td>" +
          '<td class="num">' +
          fmt4(r.mlp_final_val_loss) +
          "</td>" +
          '<td class="num ' +
          gapClass(g) +
          '">' +
          fmt4(g) +
          "</td>" +
          '<td class="num">' +
          fmt4(r.eml_final_val_acc) +
          "</td>" +
          '<td class="num">' +
          fmt4(r.mlp_final_val_acc) +
          "</td>" +
          '<td><button type="button" class="btn-link">Figures</button></td>';
        const btn = tr.querySelector("button");
        btn.addEventListener("click", () => renderDetail(r));
        tbody.appendChild(tr);
      });
    }

    const FIG_ORDER = [
      ["loss_linear", "Loss (linear)"],
      ["loss_logy", "Loss (log y)"],
      ["val_loss_gap", "Val gap (MLP − EML)"],
      ["val_delta_eml_minus_mlp", "Val delta (EML − MLP)"],
      ["accuracy", "Accuracy"],
      ["grad_norm", "Grad norm"],
      ["decision", "Decision boundaries"],
      ["summary_grid", "Summary grid"],
      ["final_bars", "Final val bars"],
    ];

    function renderDetail(r) {
      $$("#sw-table tbody tr").forEach((tr) => tr.classList.remove("active-row"));
      const hit = $$("#sw-table tbody tr").find((tr) => tr.dataset.rel === r.rel_dir);
      if (hit) hit.classList.add("active-row");

      detail.hidden = false;
      detail.innerHTML =
        '<h3>' +
        r.scenario +
        " — d" +
        r.depth +
        " h" +
        r.hidden +
        '</h3><div class="figure-grid" id="sw-fig-grid"></div>' +
        '<p class="note"><a href="' +
        sweepFigureUrl(r.figures.histories) +
        '">histories.json</a> (training curves)</p>';
      const grid = $("#sw-fig-grid", detail);
      FIG_ORDER.forEach(([key, label]) => {
        const url = sweepFigureUrl(r.figures[key]);
        const wrap = document.createElement("div");
        wrap.className = "figure";
        wrap.innerHTML = "<h4>" + label + "</h4><img src=\"" + url + '" alt="" loading="lazy" />';
        grid.appendChild(wrap);
      });
      detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    [selScenario, selDepth, selHidden].forEach((el) =>
      el.addEventListener("change", () => {
        renderTable();
        renderHeatmaps();
      })
    );

    renderHeatmaps();
    renderTable();
    detail.hidden = true;
  }

  function renderLegacy(rows) {
    const tbody = $("#results-table tbody");
    const status = $("#load-status");
    const gallery = $("#gallery-grid");

    rows.sort((a, b) => b.val_loss_gap_mlp_minus_eml - a.val_loss_gap_mlp_minus_eml);
    tbody.innerHTML = "";
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      const g = r.val_loss_gap_mlp_minus_eml;
      tr.innerHTML =
        "<td><strong>" +
        r.scenario +
        "</strong></td>" +
        '<td class="num">' +
        r.epochs +
        "</td>" +
        '<td class="num">' +
        r.depth +
        "</td>" +
        '<td class="num">' +
        r.hidden +
        "</td>" +
        '<td class="num">' +
        fmt4(r.eml_final_val_loss) +
        "</td>" +
        '<td class="num">' +
        fmt4(r.mlp_final_val_loss) +
        "</td>" +
        '<td class="num ' +
        gapClass(g) +
        '">' +
        fmt4(g) +
        "</td>" +
        '<td class="num">' +
        fmt4(r.eml_final_val_acc) +
        "</td>" +
        '<td class="num">' +
        fmt4(r.mlp_final_val_acc) +
        "</td>";
      tbody.appendChild(tr);
    });

    gallery.innerHTML = "";
    rows.forEach((r) => {
      const name = r.scenario;
      const wrap = document.createElement("div");
      wrap.innerHTML =
        '<div class="figure"><h3>' +
        name +
        ' — decision</h3><img src="assets/' +
        name +
        '_decision.png" alt="" loading="lazy" /></div>' +
        '<div class="figure" style="margin-top:0.75rem"><h3>' +
        name +
        ' — loss</h3><img src="assets/' +
        name +
        '_loss.png" alt="" loading="lazy" /></div>';
      gallery.appendChild(wrap);
    });

    status.textContent =
      "Loaded " +
      rows.length +
      " scenarios (EML " +
      rows[0].eml_params +
      " vs MLP " +
      rows[0].mlp_params +
      " params).";
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
            status.textContent =
              "Could not load sweep or legacy JSON — use a local web server (see Deploy). " + e;
          });
      });
  }

  boot();
})();
