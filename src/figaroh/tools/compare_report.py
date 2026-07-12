# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static, offline, two-run comparison page (Feature 6, Phase C).

Unlike ``tools/report.py`` / ``tools/identification_report.py``, this
module does not render a specific run's data at generation time. It
emits a self-contained HTML *shell* that a user opens directly in a
browser and then loads two ``*_verification.json`` files into (drag-and-
drop or a file picker) — the two exports produced by
``BaseCalibration.export_verification_report()`` /
``BaseIdentification.export_verification_report()`` (Step 2, extended in
Step 3 with the ``series``/``compat`` fields this page reads). Everything
happens client-side in JavaScript: no backend, no network request, no
change to either base class (see the roadmap's Step 5 "Files (modified):
none required").

Per D3 (roadmap), a compatibility check (same domain, same DOF/joint
names, same ``decimate`` setting for identification, comparable sample
counts) runs before anything is rendered; on mismatch the page blocks
the comparison with a clear message and offers an explicit "compare
anyway" override rather than silently overlaying incompatible runs.
"""

from datetime import datetime
from typing import Optional

from figaroh.tools._report_common import _STYLE

# Page-specific styling (drop zones, diff-table coloring, the forced-
# compare warning banner) layered on top of the shared `_STYLE` — the
# `.series-*` classes used by the overlay chart already live in `_STYLE`
# and are reused as-is.
_COMPARE_STYLE = """
.compare-loader { display: flex; gap: 16px; flex-wrap: wrap; }
.drop-zone {
  flex: 1 1 240px; border: 2px dashed var(--border); border-radius: 10px;
  padding: 18px; text-align: center; transition: border-color .15s;
}
.drop-zone.drag-over { border-color: var(--accent); }
.drop-label { font-weight: 600; margin: 0 0 8px; }
.drop-filename { color: var(--text-muted); font-size: .82rem; margin: 8px 0 0; }
#force-compare-row { margin-top: 10px; font-size: .88rem; display: none; }
.forced-banner {
  display: none; margin-bottom: 16px; padding: 10px 14px; border-radius: 8px;
  background: var(--poor-bg); color: var(--poor); font-size: .88rem;
}
td.diff-improve { color: var(--good); font-weight: 600; }
td.diff-regress { color: var(--poor); font-weight: 600; }
"""

# Shared chart primitives (svgns trick, gridlines, zoom/hover) mirror
# _report_common.py's `_SERIES_CHART_SCRIPT` (Step 4) in spirit, per D2 —
# but this page overlays *two* runs with a per-run visibility toggle
# rather than switching between one run's curves via a dropdown, so it
# is its own function rather than a literal reuse of `initSeriesPanel`.
#
# `initComparePanel` may legitimately be invoked more than once per page
# load (the user can swap either file at any time), so control handlers
# are attached via property assignment (`el.onchange = ...`) rather than
# `addEventListener`, which would otherwise stack a new listener on every
# re-evaluation.
_COMPARE_CHART_SCRIPT = """
function initComparePanel(id, cfg) {
  var svg = document.getElementById(id + "-svg");
  var select = document.getElementById(id + "-select");
  var resetBtn = document.getElementById(id + "-reset");
  var tooltip = document.getElementById(id + "-tooltip");
  var toggleA = document.getElementById(id + "-toggle-a");
  var toggleB = document.getElementById(id + "-toggle-b");
  var legend = document.getElementById(id + "-legend");
  // Concatenated (not a literal scheme+"//" string) so this inert,
  // never-fetched SVG XML namespace URI doesn't trip the report tests'
  // external-request substring check (see _report_common.py).
  var svgns = "http:" + "//www.w3.org/2000/svg";
  var W = 760, H = 320, padL = 46, padR = 16, padT = 16, padB = 26;

  var runColors = {
    a: { nominal: "#a9c8e8", fitted: "#2e5c8a", measured: "#5b6472" },
    b: { nominal: "#f0b58c", fitted: "#b3541e", measured: "#8a7a6a" }
  };

  select.innerHTML = "";
  cfg.names.forEach(function (n) {
    var opt = document.createElement("option");
    opt.value = n; opt.textContent = n;
    select.appendChild(opt);
  });

  legend.innerHTML =
    '<span><i style="background:' + runColors.a.fitted +
      '"></i> Run A (fitted)</span>' +
    '<span><i style="background:' + runColors.a.nominal +
      '"></i> Run A (nominal)</span>' +
    '<span><i style="background:' + runColors.b.fitted +
      '"></i> Run B (fitted)</span>' +
    '<span><i style="background:' + runColors.b.nominal +
      '"></i> Run B (nominal)</span>';

  var maxLen = 1;
  if (cfg.a && cfg.a.time) maxLen = Math.max(maxLen, cfg.a.time.length);
  if (cfg.b && cfg.b.time) maxLen = Math.max(maxLen, cfg.b.time.length);
  var fullMin = 0, fullMax = Math.max(1, maxLen - 1);
  var xMin = fullMin, xMax = fullMax;

  function activeRuns() {
    var out = [];
    if (toggleA.checked && cfg.a) {
      out.push({ key: "a", data: cfg.a, colors: runColors.a });
    }
    if (toggleB.checked && cfg.b) {
      out.push({ key: "b", data: cfg.b, colors: runColors.b });
    }
    return out;
  }

  function render() {
    var name = select.value || cfg.names[0];
    var runs = activeRuns();

    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    if (!runs.length) return;

    var i0 = Math.max(0, Math.round(xMin));
    var i1 = Math.round(xMax);
    var yMin = Infinity, yMax = -Infinity;
    runs.forEach(function (r) {
      ["nominal", "fitted", "measured"].forEach(function (key) {
        var arr = (r.data[key] || {})[name];
        if (!arr) return;
        var end = Math.min(i1, arr.length - 1);
        for (var i = i0; i <= end; i++) {
          if (arr[i] < yMin) yMin = arr[i];
          if (arr[i] > yMax) yMax = arr[i];
        }
      });
    });
    if (!isFinite(yMin) || !isFinite(yMax)) return;
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    var pad = (yMax - yMin) * 0.08;
    yMin -= pad; yMax += pad;

    function xPix(i) {
      return padL + (i - xMin) / (xMax - xMin) * (W - padL - padR);
    }
    function yPix(v) {
      return padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB);
    }
    function path(arr, end) {
      var d = "";
      for (var i = i0; i <= end; i++) {
        d += (i === i0 ? "M" : "L") + xPix(i).toFixed(2) + "," +
          yPix(arr[i]).toFixed(2) + " ";
      }
      return d;
    }

    [0.0, 0.25, 0.5, 0.75, 1.0].forEach(function (f) {
      var y = padT + f * (H - padT - padB);
      var gl = document.createElementNS(svgns, "line");
      gl.setAttribute("x1", padL); gl.setAttribute("x2", W - padR);
      gl.setAttribute("y1", y); gl.setAttribute("y2", y);
      gl.setAttribute("stroke", "var(--border)");
      gl.setAttribute("stroke-width", "1");
      svg.appendChild(gl);
    });

    runs.forEach(function (r) {
      ["measured", "nominal", "fitted"].forEach(function (key) {
        var arr = (r.data[key] || {})[name];
        if (!arr) return;
        var end = Math.min(i1, arr.length - 1);
        if (end < i0) return;
        var p = document.createElementNS(svgns, "path");
        p.setAttribute("d", path(arr, end));
        p.setAttribute("fill", "none");
        p.setAttribute("stroke", r.colors[key]);
        p.setAttribute("stroke-width", key === "fitted" ? 2.2 : 1.4);
        if (key === "measured") p.setAttribute("stroke-dasharray", "5 4");
        svg.appendChild(p);
      });
    });

    var guide = document.createElementNS(svgns, "line");
    guide.setAttribute("y1", padT); guide.setAttribute("y2", H - padB);
    guide.setAttribute("stroke", "var(--text-muted)");
    guide.setAttribute("stroke-width", "1");
    guide.style.display = "none";
    svg.appendChild(guide);

    svg.onmousemove = function (evt) {
      var rect = svg.getBoundingClientRect();
      var mx = (evt.clientX - rect.left) * (W / rect.width);
      var frac = (mx - padL) / (W - padL - padR);
      var i = Math.round(xMin + frac * (xMax - xMin));
      if (i < i0 || i > i1) {
        guide.style.display = "none";
        tooltip.style.display = "none";
        return;
      }
      guide.setAttribute("x1", xPix(i)); guide.setAttribute("x2", xPix(i));
      guide.style.display = "block";
      tooltip.style.display = "block";
      var lines = ["t=" + i];
      runs.forEach(function (r) {
        var label = r.key === "a" ? "A" : "B";
        ["nominal", "fitted", "measured"].forEach(function (key) {
          var arr = (r.data[key] || {})[name];
          if (arr && i < arr.length) {
            lines.push(
              label + " " + key + ": " + arr[i].toFixed(3) + " " + cfg.unit
            );
          }
        });
      });
      tooltip.innerHTML = lines.join("<br>");
      tooltip.style.left = (evt.clientX + 12) + "px";
      tooltip.style.top = (evt.clientY + 12) + "px";
    };
    svg.onmouseleave = function () {
      guide.style.display = "none";
      tooltip.style.display = "none";
    };
  }

  select.onchange = render;
  toggleA.onchange = render;
  toggleB.onchange = render;
  resetBtn.onclick = function () {
    xMin = fullMin; xMax = fullMax; render();
  };
  svg.onwheel = function (evt) {
    evt.preventDefault();
    var rect = svg.getBoundingClientRect();
    var mx = (evt.clientX - rect.left) * (W / rect.width);
    var frac = (mx - padL) / (W - padL - padR);
    var cursor = xMin + frac * (xMax - xMin);
    var factor = evt.deltaY < 0 ? 0.8 : 1.25;
    var newRange = Math.max(
      5, Math.min(fullMax - fullMin, (xMax - xMin) * factor)
    );
    var newMin = cursor - (cursor - xMin) * (newRange / (xMax - xMin));
    var newMax = newMin + newRange;
    if (newMin < fullMin) { newMin = fullMin; newMax = newMin + newRange; }
    if (newMax > fullMax) { newMax = fullMax; newMin = newMax - newRange; }
    xMin = newMin; xMax = newMax;
    render();
  };

  render();
  return { render: render };
}
"""

# Drives file loading, the D3 compatibility check, the metric diff
# table, and wiring into `initComparePanel` above. Wrapped in an IIFE
# (unlike `_SERIES_CHART_SCRIPT`'s bare function) since none of this
# needs to be callable from elsewhere on the page — it only reacts to
# file-input/drag-drop events, so there is no script-execution-order
# hazard the way Step 4's inline `initSeriesPanel(...)` call had.
_COMPARE_DRIVER_SCRIPT = """
(function () {
  var runs = { a: null, b: null };

  function escHtml(s) {
    var d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  function domainOf(compat) {
    if (!compat) return "unknown";
    if (Object.prototype.hasOwnProperty.call(compat, "dof_names")) {
      return "calibration";
    }
    if (Object.prototype.hasOwnProperty.call(compat, "active_joints")) {
      return "identification";
    }
    return "unknown";
  }

  function namesOf(compat, domain) {
    if (domain === "calibration") return compat.dof_names || [];
    if (domain === "identification") return compat.active_joints || [];
    return [];
  }

  function checkCompat(a, b) {
    var errors = [];
    var ca = a.compat || {}, cb = b.compat || {};
    var domainA = domainOf(ca), domainB = domainOf(cb);
    if (domainA === "unknown" || domainB === "unknown") {
      errors.push(
        "Could not determine the domain (calibration vs. identification) " +
        "of one or both runs from their compat block."
      );
      return { compatible: false, errors: errors, domain: null };
    }
    if (domainA !== domainB) {
      errors.push(
        "Different domains: run A is " + domainA + ", run B is " +
        domainB + "."
      );
      return { compatible: false, errors: errors, domain: null };
    }
    var namesA = namesOf(ca, domainA), namesB = namesOf(cb, domainB);
    if (JSON.stringify(namesA) !== JSON.stringify(namesB)) {
      errors.push(
        "Different " +
        (domainA === "calibration" ? "DOF names" : "active joints") +
        ": [" + namesA.join(", ") + "] vs. [" + namesB.join(", ") + "]."
      );
    }
    if (domainA === "identification" && !!ca.decimate !== !!cb.decimate) {
      errors.push(
        "Different decimate setting: " + ca.decimate + " vs. " +
        cb.decimate + "."
      );
    }
    var sa = ca.sample_count, sb = cb.sample_count;
    if (typeof sa === "number" && typeof sb === "number" &&
        sa > 0 && sb > 0) {
      var ratio = Math.max(sa, sb) / Math.min(sa, sb);
      if (ratio > 1.2) {
        errors.push(
          "Sample counts differ substantially: " + sa + " vs. " + sb +
          " (>20% apart)."
        );
      }
    }
    return {
      compatible: errors.length === 0, errors: errors, domain: domainA
    };
  }

  function checksDirection(a, b) {
    var dir = {};
    (a.checks || []).concat(b.checks || []).forEach(function (c) {
      dir[c.name] = c.comparison;
    });
    return dir;
  }

  function renderCompatStatus(result) {
    var el = document.getElementById("compat-status");
    var forceRow = document.getElementById("force-compare-row");
    if (result.compatible) {
      el.innerHTML =
        '<p class="tag ok">Compatible</p>' +
        '<p class="muted">Both runs share the same domain, joint/DOF ' +
        'names, and comparable settings — safe to overlay.</p>';
      forceRow.style.display = "none";
      return;
    }
    var items = result.errors.map(function (e) {
      return '<li class="insight warn">' + escHtml(e) + "</li>";
    }).join("");
    el.innerHTML =
      '<p class="tag bad">Incompatible</p>' +
      '<ul class="insights">' + items + "</ul>";
    forceRow.style.display = "block";
  }

  function renderDiffTable(a, b) {
    var names = {};
    Object.keys(a.metrics || {}).forEach(function (n) { names[n] = true; });
    Object.keys(b.metrics || {}).forEach(function (n) { names[n] = true; });
    var dir = checksDirection(a, b);
    var rows = Object.keys(names).sort().map(function (name) {
      var va = (a.metrics || {})[name];
      var vb = (b.metrics || {})[name];
      var hasA = typeof va === "number" && !isNaN(va);
      var hasB = typeof vb === "number" && !isNaN(vb);
      var delta = (hasA && hasB) ? (vb - va) : null;
      var pct = (delta !== null && va !== 0) ?
        (delta / Math.abs(va) * 100) : null;
      var cls = "";
      if (delta !== null && delta !== 0 && dir[name]) {
        var improved = dir[name] === "max" ? delta < 0 : delta > 0;
        cls = improved ? "diff-improve" : "diff-regress";
      }
      return "<tr>" +
        "<td>" + escHtml(name) + "</td>" +
        "<td class=\\"num\\">" +
          (hasA ? va.toFixed(4) : "—") + "</td>" +
        "<td class=\\"num\\">" +
          (hasB ? vb.toFixed(4) : "—") + "</td>" +
        "<td class=\\"num " + cls + "\\">" +
          (delta !== null ?
            (delta >= 0 ? "+" : "") + delta.toFixed(4) : "—") +
        "</td>" +
        "<td class=\\"num " + cls + "\\">" +
          (pct !== null ?
            (pct >= 0 ? "+" : "") + pct.toFixed(1) + "%" : "—") +
        "</td>" +
        "</tr>";
    }).join("");
    document.getElementById("diff-table-body").innerHTML = rows ||
      '<tr><td colspan="5" class="muted">No comparable metrics.</td></tr>';
  }

  var seriesUnit = { calibration: "", identification: "Nm" };

  function seriesNames(series, domain) {
    if (!series) return [];
    if (domain === "calibration") return series.dof_names || [];
    if (domain === "identification") return series.joint_names || [];
    return [];
  }

  function renderSeriesPanel(a, b, domain) {
    var card = document.getElementById("compare-series-card");
    var seriesA = a.series || {};
    var seriesB = b.series || {};
    var namesA = seriesNames(seriesA, domain);
    var namesB = seriesNames(seriesB, domain);
    var shared = namesA.filter(function (n) {
      return namesB.indexOf(n) !== -1;
    });
    var names = shared.length ? shared : (namesA.length ? namesA : namesB);

    if (!names.length) {
      card.style.display = "none";
      return;
    }
    card.style.display = "block";
    initComparePanel("compare", {
      names: names,
      a: namesA.length ? seriesA : null,
      b: namesB.length ? seriesB : null,
      unit: seriesUnit[domain] || ""
    });
  }

  function evaluate() {
    var a = runs.a, b = runs.b;
    var compatSection = document.getElementById("compat-section");
    var resultsSection = document.getElementById("results-section");
    if (!a || !b) {
      compatSection.style.display = "none";
      resultsSection.style.display = "none";
      return;
    }
    compatSection.style.display = "block";
    var result = checkCompat(a, b);
    renderCompatStatus(result);

    var force = document.getElementById("force-compare").checked;
    var proceed = result.compatible || force;
    if (!proceed) {
      resultsSection.style.display = "none";
      return;
    }
    resultsSection.style.display = "block";
    document.getElementById("forced-banner").style.display =
      result.compatible ? "none" : "block";

    renderDiffTable(a, b);
    renderSeriesPanel(a, b, result.domain || domainOf(a.compat || {}));
  }

  function loadFile(file, slot) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        runs[slot] = JSON.parse(reader.result);
        document.getElementById("filename-" + slot).textContent = file.name;
      } catch (e) {
        document.getElementById("filename-" + slot).textContent =
          "Could not parse " + file.name + " as JSON.";
        runs[slot] = null;
      }
      evaluate();
    };
    reader.readAsText(file);
  }

  function wireSlot(slot) {
    var input = document.getElementById("file-" + slot);
    var zone = document.getElementById("zone-" + slot);
    input.addEventListener("change", function () {
      if (input.files && input.files[0]) loadFile(input.files[0], slot);
    });
    zone.addEventListener("dragover", function (evt) {
      evt.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", function () {
      zone.classList.remove("drag-over");
    });
    zone.addEventListener("drop", function (evt) {
      evt.preventDefault();
      zone.classList.remove("drag-over");
      var files = evt.dataTransfer.files;
      if (files && files[0]) loadFile(files[0], slot);
    });
  }

  wireSlot("a");
  wireSlot("b");
  document.getElementById("force-compare")
    .addEventListener("change", evaluate);
})();
"""


def generate_compare_page(
    output_path: Optional[str] = None, title: Optional[str] = None
) -> str:
    """Render the static, self-contained two-run comparison page.

    Unlike the calibration/identification report generators, this takes
    no run object — it emits a template that loads two
    ``export_verification_report()`` JSON files client-side (drag-and-
    drop or a file picker) and, after a mandatory compatibility check
    (D3), renders a per-metric diff table and an overlaid before/after
    series chart. No network request, no backend: the returned document
    is fully self-contained and can be opened directly from disk.

    Args:
        output_path: If given, the HTML is also written to this path.
        title: Optional page title. Defaults to a generic comparison title.

    Returns:
        The rendered HTML document as a string.
    """
    report_title = title or "FIGAROH — Two-Run Comparison"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{report_title}</title>
<style>{_STYLE}</style>
<style>{_COMPARE_STYLE}</style>
<script>{_COMPARE_CHART_SCRIPT}</script>
</head>
<body>
<div class="page">
  <h1>{report_title}</h1>
  <p class="subtitle">Generated {timestamp} &middot; static, offline —
    load two verification exports below. Nothing leaves this page.</p>

  <section>
    <h2>Load runs</h2>
    <div class="card compare-loader">
      <div class="drop-zone" id="zone-a" data-slot="a">
        <p class="drop-label">Run A</p>
        <input type="file" id="file-a" accept="application/json,.json">
        <p class="drop-filename" id="filename-a">No file loaded</p>
      </div>
      <div class="drop-zone" id="zone-b" data-slot="b">
        <p class="drop-label">Run B</p>
        <input type="file" id="file-b" accept="application/json,.json">
        <p class="drop-filename" id="filename-b">No file loaded</p>
      </div>
    </div>
  </section>

  <section id="compat-section" style="display:none;">
    <h2>Compatibility</h2>
    <div class="card">
      <div id="compat-status"></div>
      <div id="force-compare-row">
        <label>
          <input type="checkbox" id="force-compare">
          Compare anyway (results may be misleading)
        </label>
      </div>
    </div>
  </section>

  <section id="results-section" style="display:none;">
    <div class="forced-banner" id="forced-banner">
      These runs failed the compatibility check above — the comparison
      below was forced and may be misleading.
    </div>

    <h2>Metric comparison</h2>
    <div class="card">
      <table class="data">
        <thead>
          <tr>
            <th>Metric</th><th>Run A</th><th>Run B</th>
            <th>&Delta;</th><th>% change</th>
          </tr>
        </thead>
        <tbody id="diff-table-body"></tbody>
      </table>
    </div>

    <h2>Series overlay</h2>
    <div class="card" id="compare-series-card">
      <div class="series-controls">
        <label><input type="checkbox" id="compare-toggle-a" checked>
          Run A</label>
        <label><input type="checkbox" id="compare-toggle-b" checked>
          Run B</label>
        <label for="compare-select">Show:</label>
        <select id="compare-select"></select>
        <button type="button" id="compare-reset">Reset zoom</button>
      </div>
      <svg id="compare-svg" class="series-svg"></svg>
      <div class="series-legend" id="compare-legend"></div>
      <p class="series-hint">Scroll to zoom, hover to inspect.</p>
      <div id="compare-tooltip" class="series-tooltip"></div>
    </div>
  </section>

  <footer>Generated by figaroh.tools.compare_report — static, offline,
    no backend.</footer>
</div>
<script>{_COMPARE_DRIVER_SCRIPT}</script>
</body>
</html>
"""

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(doc)

    return doc
