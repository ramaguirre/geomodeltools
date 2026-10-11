/*
 * geomodeltools EDA/contact-plot report: the "mount" script for report.py's self-contained
 * HTML page. Plain JS, no build step: it is inlined into the page as-is (see report.py's
 * TEMPLATE), so if you add another file here you need to wire a bundler in first.
 *
 * All the aggregation (binning, probability positions, weighted stats) already happened in
 * Python (eda.py / contacts.py), via report.py's build_report(). This script only turns that
 * JSON into Plotly traces: histogram/violin/bar charts bin or shape their own raw values
 * client-side (that's Plotly's own job, same as matplotlib does internally for ax.hist /
 * ax.violinplot), but it never recomputes a probability position, a weighted mean, or a
 * contact distance bin — those numbers come from the payload as-is.
 */
(function () {
  "use strict";

  // Standard-normal quantiles, for labelling the probability-plot y-axis (z-scores -> %).
  // A fixed reference table for axis ticks, not a statistic computed from the dataset.
  var PROB_TICKS = [
    [-2.326, "1%"], [-1.645, "5%"], [-1.282, "10%"], [-0.674, "25%"], [0, "50%"],
    [0.674, "75%"], [1.282, "90%"], [1.645, "95%"], [2.326, "99%"],
  ];

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    for (var key in attrs || {}) {
      if (key === "text") node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    }
    (children || []).forEach(function (child) { node.appendChild(child); });
    return node;
  }

  function option(value, label) {
    return el("option", { value: value, text: label == null ? value : label });
  }

  function plot(divId, traces, layout) {
    var container = document.getElementById(divId);
    if (!container.classList.contains("js-plotly-plot")) container.innerHTML = ""; // drop a leftover text summary, if any
    var base = { margin: { t: 36, r: 16, b: 48, l: 56 }, font: { size: 11 }, legend: { font: { size: 10 } } };
    Plotly.react(container, traces, Object.assign(base, layout || {}), { responsive: true, displaylogo: false });
  }

  // ---- Numeric tab ---------------------------------------------------------------------

  function noop() {}

  function numericPanel(data) {
    var values = Object.keys(data.numeric);
    var panel = el("div", { class: "panel", id: "panel-numeric" });
    if (!values.length) return { el: panel, init: noop };

    var valueSelect = el("select", { id: "numeric-value" }, values.map(function (v) { return option(v); }));
    var splitSelect = el("select", { id: "numeric-split" });
    var logCheck = el("input", { type: "checkbox", id: "numeric-log" });

    panel.appendChild(el("div", { class: "controls" }, [
      el("label", {}, [document.createTextNode("Variable "), valueSelect]),
      el("label", {}, [document.createTextNode("Split by "), splitSelect]),
      el("label", {}, [logCheck, document.createTextNode(" log scale")]),
    ]));

    var charts = el("div", { class: "charts" }, [
      el("div", { class: "chart", id: "numeric-chart-1" }),
      el("div", { class: "chart", id: "numeric-chart-2" }),
      el("div", { class: "chart", id: "numeric-chart-3" }),
    ]);
    panel.appendChild(charts);

    function refreshSplitOptions() {
      var record = data.numeric[valueSelect.value];
      var cats = Object.keys(record.by_category);
      splitSelect.innerHTML = "";
      splitSelect.appendChild(option("", "(none)"));
      cats.forEach(function (c) { splitSelect.appendChild(option(c)); });
    }

    function render() {
      var record = data.numeric[valueSelect.value];
      var axType = logCheck.checked ? "log" : "linear";
      var split = splitSelect.value;

      if (!split) {
        var d = record.overall;
        plot("numeric-chart-1", [{ type: "histogram", x: d.values, marker: { color: "#4a7ab5" } }], {
          title: d.value + ": histogram", xaxis: { title: d.value, type: axType },
        });
        plot("numeric-chart-2", [{ x: d.probability.v, y: d.probability.z, mode: "markers", marker: { size: 4, color: "#4a7ab5" } }], {
          title: "cumulative probability", xaxis: { title: d.value, type: axType },
          yaxis: { title: "probability", tickvals: PROB_TICKS.map(function (t) { return t[0]; }), ticktext: PROB_TICKS.map(function (t) { return t[1]; }) },
        });
        var s = d.summary;
        var summaryText = ["count: " + s.count, "mean: " + s.mean.toPrecision(4), "median: " + s.median.toPrecision(4),
          "std: " + s.std.toPrecision(4), "cov: " + s.cov.toPrecision(3)].join("<br>");
        var summaryEl = document.getElementById("numeric-chart-3");
        if (summaryEl.classList.contains("js-plotly-plot")) Plotly.purge(summaryEl); // leaving by_category mode: drop the old chart first
        summaryEl.innerHTML = '<div style="padding:24px;font-size:0.95rem;line-height:1.8">' + summaryText + "</div>";
        return;
      }

      var byCat = record.by_category[split];
      var order = byCat.order;
      var probTraces = order.map(function (cat) {
        var s = byCat.series[cat];
        return { x: s.probability.v, y: s.probability.z, mode: "markers", name: cat, marker: { size: 4 } };
      });
      plot("numeric-chart-1", probTraces, {
        title: "cumulative probability by " + byCat.category, xaxis: { title: byCat.value, type: axType },
        yaxis: { title: "probability", tickvals: PROB_TICKS.map(function (t) { return t[0]; }), ticktext: PROB_TICKS.map(function (t) { return t[1]; }) },
      });

      var violinTraces = order.map(function (cat) {
        var s = byCat.series[cat];
        return {
          type: "violin", y: s.values, x: s.values.map(function () { return cat; }), name: cat,
          points: false, meanline: { visible: true }, box: { visible: true }, showlegend: false,
        };
      });
      plot("numeric-chart-2", violinTraces, { title: byCat.value + " by " + byCat.category, yaxis: { title: byCat.value, type: axType } });

      plot("numeric-chart-3", [{ type: "bar", x: order, y: order.map(function (cat) { return byCat.summary[cat].cov; }), marker: { color: "#4a7ab5" } }], {
        title: "coefficient of variation by " + byCat.category, yaxis: { title: "std / mean" },
      });
    }

    valueSelect.addEventListener("change", function () { refreshSplitOptions(); render(); });
    splitSelect.addEventListener("change", render);
    logCheck.addEventListener("change", render);
    return { el: panel, init: function () { refreshSplitOptions(); render(); } };
  }

  // ---- Categorical tab ------------------------------------------------------------------

  function categoricalPanel(data) {
    var cats = Object.keys(data.categorical);
    var panel = el("div", { class: "panel", id: "panel-categorical" });
    if (!cats.length) return { el: panel, init: noop };

    var select = el("select", { id: "categorical-select" }, cats.map(function (c) { return option(c); }));
    panel.appendChild(el("div", { class: "controls" }, [el("label", {}, [document.createTextNode("Column "), select])]));
    panel.appendChild(el("div", { class: "charts" }, [el("div", { class: "chart", id: "categorical-chart" })]));

    function render() {
      var d = data.categorical[select.value];
      plot("categorical-chart", [{
        type: "bar", x: d.labels, y: d.counts, marker: { color: "#4a7ab5" },
        text: d.pct.map(function (p) { return p.toFixed(1) + "%"; }), textposition: "outside",
      }], { title: d.category, yaxis: { title: d.weighted ? "weight" : "count" } });
    }

    select.addEventListener("change", render);
    return { el: panel, init: render };
  }

  // ---- Pairs tab (categorical vs categorical) -------------------------------------------

  function pairsPanel(data) {
    var pairs = data.pairs;
    var panel = el("div", { class: "panel", id: "panel-pairs" });
    if (!pairs.length) return { el: panel, init: noop };

    var select = el("select", { id: "pairs-select" }, pairs.map(function (p, i) {
      return option(String(i), p.category_x + " vs " + p.category_y);
    }));
    panel.appendChild(el("div", { class: "controls" }, [el("label", {}, [document.createTextNode("Pair "), select])]));
    panel.appendChild(el("div", { class: "charts" }, [el("div", { class: "chart", id: "pairs-chart" })]));

    function render() {
      var d = pairs[Number(select.value)];
      var traces = d.y_labels.map(function (yLabel, j) {
        return { type: "bar", name: yLabel, x: d.x_labels, y: d.matrix.map(function (row) { return row[j]; }) };
      });
      plot("pairs-chart", traces, {
        title: d.category_y + " composition by " + d.category_x, barmode: "stack",
        yaxis: { title: d.normalize ? d.category_y + " [%]" : d.category_y + (d.weighted ? " [weight]" : " [count]") },
      });
    }

    select.addEventListener("change", render);
    return { el: panel, init: render };
  }

  // ---- Contacts tab -----------------------------------------------------------------------

  function contactsPanel(data) {
    var contacts = data.contacts;
    var panel = el("div", { class: "panel", id: "panel-contacts" });
    if (!contacts.length) return { el: panel, init: noop };

    var select = el("select", { id: "contacts-select" }, contacts.map(function (c, i) { return option(String(i), c.label); }));
    panel.appendChild(el("div", { class: "controls" }, [el("label", {}, [document.createTextNode("Scenario "), select])]));
    panel.appendChild(el("div", { class: "charts" }, [
      el("div", { class: "chart", id: "contacts-chart-profile" }),
      el("div", { class: "chart", id: "contacts-chart-composition" }),
    ]));

    function render() {
      var d = contacts[Number(select.value)];
      var profileEl = document.getElementById("contacts-chart-profile");
      var compEl = document.getElementById("contacts-chart-composition");

      if (d.profile) {
        profileEl.style.display = "";
        plot("contacts-chart-profile", [{ x: d.profile.mid, y: d.profile.mean, mode: "lines+markers", marker: { color: "#4a7ab5" } }], {
          title: d.profile.value + ": contact profile (" + d.label + ")",
          xaxis: { title: "distance to contact" }, yaxis: { title: "mean " + d.profile.value },
          shapes: [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: "firebrick", dash: "dot" } }],
        });
      } else {
        profileEl.style.display = "none";
      }

      if (d.composition) {
        compEl.style.display = "";
        var traces = d.composition.categories.map(function (cat, j) {
          return { type: "bar", name: cat, x: d.composition.mid, y: d.composition.matrix.map(function (row) { return row[j]; }) };
        });
        plot("contacts-chart-composition", traces, {
          title: d.composition.category + ": contact composition (" + d.label + ")", barmode: "stack",
          xaxis: { title: "distance to contact" }, yaxis: { title: d.composition.category + " [%]" },
          shapes: [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: "black", dash: "dot" } }],
        });
      } else {
        compEl.style.display = "none";
      }
    }

    select.addEventListener("change", render);
    return { el: panel, init: render };
  }

  // ---- Tabs / mount -----------------------------------------------------------------------

  function mount(navEl, mainEl, data) {
    var tabs = [
      { id: "numeric", label: "Numeric", build: numericPanel, has: Object.keys(data.numeric).length > 0 },
      { id: "categorical", label: "Categorical", build: categoricalPanel, has: Object.keys(data.categorical).length > 0 },
      { id: "pairs", label: "Pairs", build: pairsPanel, has: data.pairs.length > 0 },
      { id: "contacts", label: "Contacts", build: contactsPanel, has: data.contacts.length > 0 },
    ].filter(function (t) { return t.has; });

    if (!tabs.length) {
      mainEl.appendChild(el("div", { class: "empty", text: "Nothing to show: build_report() got no numeric/categorical/contacts columns." }));
      return;
    }

    tabs.forEach(function (tab, i) {
      var built = tab.build(data);
      var panel = built.el;
      if (i === 0) panel.classList.add("active");
      mainEl.appendChild(panel); // attach before init(): Plotly needs the container in the document to size itself
      built.init();

      var button = el("button", { text: tab.label });
      if (i === 0) button.classList.add("active");
      button.addEventListener("click", function () {
        tabs.forEach(function (t) {
          document.getElementById("panel-" + t.id).classList.remove("active");
        });
        Array.prototype.forEach.call(navEl.children, function (b) { b.classList.remove("active"); });
        panel.classList.add("active");
        button.classList.add("active");
        window.dispatchEvent(new Event("resize")); // Plotly charts in a freshly-visible panel need a resize nudge
      });
      navEl.appendChild(button);
    });
  }

  window.geomodeltoolsReport = { mount: mount };
})();
